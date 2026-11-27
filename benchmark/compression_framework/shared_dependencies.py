"""Codec-specific shared component inventories; conservative on unknown calls."""
import ast
import copy
from pathlib import Path


def _value(node, codec):
    if isinstance(node, ast.Constant): return node.value
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return tuple(_value(x, codec) for x in node.elts)
    if isinstance(node, ast.Name) and node.id in ('name', 'base', 'level'):
        return {'name': codec, 'base': codec.partition(':')[0], 'level': codec.partition(':')[2]}[node.id]
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'self' and node.attr == 'NAME': return codec
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        left, right = _value(node.left, codec), _value(node.comparators[0], codec)
        op=node.ops[0]
        if isinstance(op, ast.Eq): return left == right
        if isinstance(op, ast.NotEq): return left != right
        if isinstance(op, ast.In): return left in right
        if isinstance(op, ast.NotIn): return left not in right
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'startswith' and len(node.args)==1:
        return _value(node.func.value, codec).startswith(_value(node.args[0], codec))
    raise ValueError('Runtime-dependent condition')


class CatalogBranch(ast.NodeTransformer):
    def __init__(self, codec): self.codec=codec;self.enabled=False
    def visit_FunctionDef(self,node):
        previous=self.enabled
        self.enabled=node.name in ('_engine','supports') if not previous else previous
        self.generic_visit(node)
        if self.enabled:
            for i, child in enumerate(node.body):
                if isinstance(child,(ast.Return,ast.Raise)):
                    node.body=node.body[:i+1];break
        self.enabled=previous
        return node
    def visit_If(self,node):
        if self.enabled:
            try: branch=node.body if _value(node.test,self.codec) else node.orelse
            except (ValueError,TypeError,AttributeError): pass
            else:
                result=[]
                for child in branch:
                    value=self.visit(child)
                    result.extend(value if isinstance(value,list) else [value])
                return result
        return self.generic_visit(node)


def shared_sources(root, route, stage):
    from measurement_identity import semantic_tree_digest
    framework=Path(root)/'benchmark/compression_framework'
    result={};visited=set()
    def include(path, symbols=None, backend_id=None):
        path=Path(path)
        if not path.is_file():return
        key=(str(path),tuple(sorted(symbols or ())))
        if key in visited:return
        visited.add(key)
        tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        if path.name=='catalog.py':
            # Specialize only the known dispatcher bindings; fall back to the
            # full component when dispatch semantics change.
            cls=next((n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Codec'),None)
            engine=next((n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_engine'),None) if cls else None
            bindings=[n for n in ast.walk(engine) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('name','base','level') for target in n.targets for t in ast.walk(target))] if engine else []
            expected=[ast.parse('name=self.NAME').body[0],ast.parse("base, _, level = name.partition(':')").body[0]]
            constructor=next((n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__'),None) if cls else None
            names=[n for n in ast.walk(constructor) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='NAME' for t in n.targets)] if constructor else []
            name_expected=ast.parse("self.NAME=entry['name']").body[0]
            if ([ast.dump(n) for n in bindings]==[ast.dump(n) for n in expected]
                    and len(names)==1 and ast.dump(names[0])==ast.dump(name_expected)):
                tree=CatalogBranch(route.name).visit(tree)
        if backend_id:
            for node in tree.body:
                if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BACKENDS' for t in node.targets) and isinstance(node.value,ast.Dict):
                    pairs=[(k,v) for k,v in zip(node.value.keys,node.value.values) if isinstance(k,ast.Constant) and k.value==backend_id]
                    if pairs:node.value=ast.Dict(keys=[k for k,v in pairs],values=[v for k,v in pairs])
        definitions={}
        for node in tree.body:
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):definitions[node.name]=node
            elif isinstance(node,(ast.Assign,ast.AnnAssign)):
                for target in node.targets if isinstance(node,ast.Assign) else [node.target]:
                    for name in ast.walk(target):
                        if isinstance(name,ast.Name):definitions[name.id]=node
        if symbols and '*' not in symbols:
            pending=list(symbols);selected=set();nodes=[]
            while pending:
                name=pending.pop()
                if name in selected or name not in definitions:continue
                selected.add(name);node=definitions[name];nodes.append(node)
                pending += [n.id for n in ast.walk(node) if isinstance(n,ast.Name) and n.id in definitions]
            # An unknown dynamic export invalidates the whole module conservatively.
            if not set(symbols)<=definitions.keys(): nodes=tree.body
        else:nodes=tree.body
        selected_ids={id(n) for n in nodes}
        nodes=[n for n in tree.body if id(n) in selected_ids]
        imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
        selected_tree=ast.Module(body=imports+nodes,type_ignores=[])
        label=('framework/'+path.relative_to(framework).as_posix()) if path.is_relative_to(framework) else 'provider/'+path.parent.name+'/'+path.name
        result[label+('::'+','.join(sorted(symbols)) if symbols else '')]=semantic_tree_digest(copy.deepcopy(selected_tree),stage)
        follow(selected_tree)
    def follow(tree):
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom) and node.level==0 and node.module:
                p=framework.joinpath(*node.module.split('.')).with_suffix('.py')
                if p.is_file():
                    names={a.name for a in node.names}
                    if names:include(p,names)
            elif isinstance(node,ast.Import):
                for alias in node.names:
                    p=framework.joinpath(*alias.name.split('.')).with_suffix('.py')
                    if p.is_file():include(p)
    # Inspect selected local adapters to find the shared components they import.
    decoders=set()
    import json
    note=route.folder/'decode_backends.json'
    if note.exists():
        entry=json.loads(note.read_text(encoding='utf8')).get(route.name,{})
        decoders={x['file'] for x in entry.values() if isinstance(x,dict) and x.get('file')}
    for path in route.folder.rglob('*.py'):
        if '__pycache__' in path.parts:continue
        if stage=='encode' and path.relative_to(route.folder).as_posix() in decoders:continue
        tree=ast.parse(path.read_text(encoding='utf-8-sig'));semantic_tree_digest(tree,stage);follow(tree)
    include(framework/'benchmark_routes.py',{'Route','module_at'})
    include(framework/'adapters/benchmark_cpu.py',{'CPUBackend'})
    plugin=getattr(route,'plugin',None)
    policy_file=route.folder/'fingerprint_dependencies.json'
    policy=json.loads(policy_file.read_text(encoding='utf8')) if policy_file.exists() else {}
    if policy and (set(policy)-{'schema_version','plugin_api'} or policy.get('schema_version')!=1 or type(policy.get('plugin_api',True)) is not bool):raise ValueError('Invalid dependency policy')
    uses_plugin=policy.get('plugin_api',True)
    if uses_plugin and getattr(plugin,'spec',None):include(framework/'plugins.py',{'PackageCodec'})
    elif uses_plugin and plugin is not None and type(plugin).__name__=='Codec':include(framework/'adapters/catalog.py',{'Codec'})
    if stage=='encode':include(framework/'codec_timing.py',{'Encoder','gpu_call','TIMING_VERSION'})
    else:
        include(framework/'cpu_decode_registry.py',{'resolve_decoder'})
        if route.mode.endswith('-GPU'):include(framework/'batch_decode.py',{'BatchDecoder'})
    provider=getattr(route,'gpu',None)
    if provider is not None:
        include(framework/'gpu_codec.py',{'GpuCodec'})
        if provider.declaration.get('type')=='shared':
            identifier=provider.declaration['backend'];candidates=[]
            for manifest in (Path(root)/'benchmark/compressors').glob('*/frame_provider.json'):
                entry=json.loads(manifest.read_text());p=manifest.parent/entry['file']
                tree=ast.parse(p.read_text(encoding='utf8'))
                candidates.append((p,tree))
            matched=[x for x in candidates if any(isinstance(n,ast.Constant) and n.value==identifier for n in ast.walk(x[1]))]
            for p,tree in matched or candidates:
                include(p,{'BACKENDS','backend','list_backends','encode_cpu_payload','pack_frame','unpack_frame','decode_gpu_frame'},backend_id=identifier)
    return result
