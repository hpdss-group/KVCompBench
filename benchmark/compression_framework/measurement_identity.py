"""Versioned measurement dependencies, separate from installation provenance.

Runner orchestration is governed by PROTOCOL. Changes to measurement boundaries
in the runner must increment the corresponding protocol, not a report version.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path

PROTOCOL = dict(data='kv-pages-v4', encode='codec-encode-v6',
                decode='codec-decode-v5', batch_encode='joined-pages-v2')
SCHEMA = 'layered-measurement-v1'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


class SemanticSource(ast.NodeTransformer):
    """Ignore documentation, comments and standalone diagnostic print calls."""
    def _body(self, node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body.pop(0)
        if not node.body:
            node.body = [ast.Pass()]
        return node

    visit_Module = _body
    visit_ClassDef = _body
    visit_FunctionDef = _body
    visit_AsyncFunctionDef = _body

    def visit_Expr(self, node):
        # Print arguments can have side effects: retain calls such as print(f()).
        value = node.value
        if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                and value.func.id == 'print'
                and not any(isinstance(n, (ast.Call, ast.Await, ast.NamedExpr, ast.Yield))
                            for argument in [*value.args, *(k.value for k in value.keywords)]
                            for n in ast.walk(argument))):
            return None
        return self.generic_visit(node)


def semantic_digest(path, stage=None):
    """Split explicit encode/decode entry points; unknown helpers remain shared.

    Shared constructors, setup, imports and globals are deliberately conservative.
    A helper with no directional name invalidates both stages.
    """
    path = Path(path)
    if path.suffix != '.py':
        return hashlib.sha256(path.read_bytes()).hexdigest()
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    return semantic_tree_digest(tree, stage)


def semantic_tree_digest(tree, stage=None):
    opposite = ('decode', 'decompress', 'unpack', 'cpu_decoder') if stage == 'encode' else ('encode', 'compress', 'pack') if stage == 'decode' else ()

    def excluded(name):
        # Match word boundaries: decompress must never be classified as compress.
        words = name.strip('_').split('_')
        return any(name.strip('_') == x or x in words for x in opposite)

    def prune(node):
        if hasattr(node, 'body') and isinstance(node.body, list):
            node.body = [n for n in node.body if not
                         (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and excluded(n.name))]
            for child in node.body:
                prune(child)
    if stage:
        prune(tree)
    tree = SemanticSource().visit(tree)
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def portable_installation(report):
    """Keep paths in provenance, never treat relocation as algorithm changes."""
    ignored = {'path', 'location', 'prefix', 'python', 'python_binary', 'origin',
               'manifest_sha256', 'recipe_sha256', 'status', 'errors', 'version_evidence', 'build_tools'}

    def strip(value):
        if isinstance(value, dict):
            return {key: strip(item) for key, item in value.items() if key not in ignored}
        if isinstance(value, list):
            return [strip(item) for item in value]
        return value
    return strip(report)


def stage_sources(root, route, stage):
    root = Path(root)
    framework = root / 'benchmark/compression_framework'
    from shared_dependencies import shared_sources
    result = shared_sources(root, route, stage)
    batch_module = framework / 'batch_measure.py'
    if batch_module.is_file():
        tree = ast.parse(batch_module.read_text(encoding='utf-8'))
        names = ('encode_frame', 'save_frame') if stage == 'encode' else ('decode_frames',)
        selected = ast.Module(body=[n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)) or getattr(n, 'name', '') in names], type_ignores=[])
        result['framework/batch_measure.py'] = hashlib.sha256(ast.dump(SemanticSource().visit(selected), include_attributes=False).encode()).hexdigest()
    # Codec-local decoder files have explicit declarations and cannot invalidate
    # encoding. Everything else is shared unless its entry points are directional.
    decoder_files = set()
    note = route.folder / 'decode_backends.json'
    if note.is_file():
        entry = json.loads(note.read_text(encoding='utf-8')).get(route.name, {})
        decoder_files = {value['file'] for value in entry.values()
                         if isinstance(value, dict) and value.get('file')}
        if stage == 'decode':
            result['decoder_declaration'] = fingerprint(entry)
    for path in sorted(route.folder.rglob('*')):
        if '__pycache__' in path.parts or not path.is_file() or path.suffix not in ('.py', '.cpp', '.c', '.h', '.hpp', '.so', '.dll', '.pyd'):
            continue
        relative = path.relative_to(route.folder).as_posix()
        if stage == 'encode' and relative in decoder_files:
            continue
        result['codec/' + relative] = semantic_digest(path, stage)
    return result


def layered_spec(root, route, provenance, count, encode_batch=1, group_blocks=1):
    """Return stable data/encoder IDs and a decoder-ID factory specification."""
    root = Path(root)
    framework = root / 'benchmark/compression_framework'
    data = dict(schema=SCHEMA, protocol=PROTOCOL['data'],
                input_sha256=provenance['input_sha256'], shape=provenance['shape'],
                dtype=provenance['storage_dtype'], page_format=provenance['page_format'],
                layout=provenance['layout'], count=count, validation=provenance['validation'],
                implementation=semantic_digest(framework / 'kv_pages.py'))
    installation = portable_installation(provenance['installation'])
    encode_installation = portable_installation(provenance.get('encode_installation', provenance['installation']))
    # CPU timing must not change solely because an unrelated CUDA installation changed.
    environment = copy.deepcopy(provenance['environment'])
    if route.mode.startswith('CPU-'):
        environment.pop('device', None)
        environment.pop('cuda', None)
    declaration = json.loads((route.folder / 'codec.json').read_text(encoding='utf-8'))
    entry = next(e for e in declaration['codecs'] if e['name'] == route.name)
    encode_entry = {k: v for k, v in entry.items() if k not in ('decompress', 'decompress_kwargs', 'description', 'enabled', 'gpu_backend', 'gpu_decode_backend', 'gpu_decode_frame_version')}
    encode_options = {k: v for k, v in route.options.items() if k not in ('batch_kind', 'decode_options', 'decode_batch_blocks')}
    encode_options['execution_path'] = route.mode.split('-')[0]
    encode = dict(schema=SCHEMA, protocol=PROTOCOL['encode'],
                  batch_protocol=PROTOCOL['batch_encode'] if encode_batch > 1 else None,
                  data=fingerprint(data), codec=route.name, entry=encode_entry,
                  options=encode_options, installation=encode_installation, environment=environment,
                  workers=provenance['workers'] if route.mode.startswith('CPU-') else 1,
                  encode_batch_blocks=group_blocks, source=stage_sources(root, route, 'encode'))
    plugin = getattr(route, 'plugin', None)
    adapter_entry = getattr(plugin, 'entry', {})
    encode['adapter_entry'] = {key: value for key, value in adapter_entry.items()
                               if key not in ('description', 'provenance')}
    frame_provider = getattr(route, 'gpu', None)
    if frame_provider is not None:
        # Some shared-frame CPU encoders take parameters from gpu_backend metadata.
        encode['frame_encoder'] = frame_provider.declaration
    decode = dict(schema=SCHEMA, protocol=PROTOCOL['decode'],
                  encoder=fingerprint(encode), entry=entry, options=route.options,
                  installation=installation, environment=provenance['environment'],
                  source=stage_sources(root, route, 'decode'))
    return dict(data=data, encode=encode, decode=decode)


def decoder_job(layers, mode, batch_blocks):
    spec = copy.deepcopy(layers['decode'])
    spec['mode'] = mode
    spec['batch_blocks'] = batch_blocks if mode.endswith('-GPU') else 1
    return fingerprint(spec)
