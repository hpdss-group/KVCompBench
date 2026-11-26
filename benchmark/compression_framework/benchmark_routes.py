"""Declarative benchmark plugins; no codec-name dispatch in the runner."""
from dataclasses import dataclass
from pathlib import Path
import importlib.util, hashlib, json, sys
from core import load_plugins

def module_at(path):
    path=Path(path).resolve(); key='benchmark_backend_'+hashlib.sha256(str(path).encode()).hexdigest()[:16]
    if key not in sys.modules:
        spec=importlib.util.spec_from_file_location(key,path)
        module=importlib.util.module_from_spec(spec);sys.modules[key]=module
        spec.loader.exec_module(module)
    return sys.modules[key]

@dataclass
class Route:
    name:str
    plugin:object
    folder:Path
    options:dict
    gpu:object=None
    dtype:str="bf16"
    @property
    def mode(self):return self.options['execution_path']
    @property
    def backend_id(self):return self.options['frame_format']
    @property
    def _folder(self):return self.folder
    @_folder.setter
    def _folder(self,value):self.folder=Path(value)
    @property
    def declaration(self):return {'type':'shared'}
    def runtime(self):
        from adapters.benchmark_cpu import CPUBackend
        filename=self.options.get('file')
        if not filename:return CPUBackend(self)
        path=(self.folder/filename).resolve()
        if not path.is_relative_to(self.folder.resolve()):raise ValueError('Backend escapes codec directory')
        return module_at(path).Backend(self)
    def encode_cpu(self,raw,dtype='bf16'):
        return self.gpu.encode_cpu(raw,dtype) if self.gpu else self.plugin.compress(raw,{'dtype':dtype})
    def decode_gpu(self,frame,size):
        return self.gpu.decode_gpu(frame,size) if self.gpu else self.plugin.decompress(frame,size,{'dtype':self.dtype})

def package_tier(folder):
    metadata = Path(folder)/'package.json'
    if not metadata.exists():
        return 'optional'
    value = json.loads(metadata.read_text(encoding='utf-8'))
    tier = value.get('tier')
    if tier not in ('core', 'optional'):
        raise ValueError('Package tier must be core or optional: '+str(metadata))
    return tier


def discover_routes(directory, codec_set='all', execution_device='declared', dtype='bf16'):
    if codec_set not in ('core', 'all'):
        raise ValueError('codec_set must be core or all')
    if execution_device not in ('cpu', 'declared'):
        raise ValueError('execution_device must be cpu or declared')
    directory=Path(directory);plugins=load_plugins(directory)
    if execution_device=='declared':
                from gpu_codec import discover_gpu_codecs
                gpu=discover_gpu_codecs(directory)
    else:gpu={}
    folders={c['name']:p.parent for p in directory.glob('*/codec.json') for c in json.loads(p.read_text())['codecs']}
    routes={}
    for name,plugin in plugins.items():
        folder=folders[name]
        if codec_set=='core' and package_tier(folder)!='core':continue
        metadata_file=folder/'package.json'
        metadata=json.loads(metadata_file.read_text()) if metadata_file.exists() else {}
        if dtype not in metadata.get('benchmark_dtypes',['bf16']):continue
        if not plugin.supports({'dtype':dtype}):continue
        if execution_device=='cpu':
            if getattr(plugin,'entry',{}).get('adapter')=='gpu-api' or getattr(plugin,'spec',{}).get('device')=='gpu':continue
            options=dict(execution_path='CPU-CPU',frame_format='plugin:'+name,thread_safe=True,batch_kind='cpu-single')
        else:
            manifest=folder/'benchmark.json'
            options=json.loads(manifest.read_text()).get(name) if manifest.exists() else None
            if options is None:
                options=dict(execution_path='CPU-CPU',frame_format='plugin:'+name,thread_safe=True,batch_kind='cpu-single')
        options=dict(options)
        options.update(options.pop('by_dtype',{}).get(dtype,{}))
        if options['execution_path'] not in ('CPU-CPU','CPU-GPU','GPU-GPU'):raise ValueError('Invalid execution_path')
        if options['execution_path'].endswith('-GPU') and not options.get('file'):raise ValueError('GPU route requires a backend file')
        routes[name]=Route(name,plugin,folder,options,gpu.get(name) if options.get('use_gpu_frame') else None,dtype=dtype)
    return routes


def select_routes(directory, codec_set="core", execution_device="cpu", codecs=None,
                  dtype="bf16", skip_incompatible=False, omitted=None):
    if codecs is None or not codecs.strip():
        routes = discover_routes(directory, codec_set=codec_set, execution_device=execution_device, dtype=dtype)
        if omitted is not None and dtype != 'bf16':
            baseline = discover_routes(directory, codec_set=codec_set, execution_device=execution_device, dtype='bf16')
            omitted.update({name: 'Unsupported input dtype: '+dtype for name in baseline.keys()-routes.keys()})
        return routes
    names = [name.strip() for name in codecs.split(',')]
    if any(not name for name in names):
        raise ValueError('Empty codec name in codecs')
    if len(names) != len(set(names)):
        raise ValueError('Duplicate codec names in codecs')
    known = load_plugins(directory)
    unknown = set(names)-known.keys()
    if unknown:
        raise ValueError('Unknown codecs: '+', '.join(sorted(unknown)))
    routes = discover_routes(directory, codec_set='all', execution_device=execution_device, dtype=dtype)
    missing = set(names)-routes.keys()
    if missing and not skip_incompatible:
        raise ValueError('Incompatible codecs for '+dtype+': '+', '.join(sorted(missing)))
    if omitted is not None:
        omitted.update({name: 'Unsupported input dtype or execution device: '+dtype for name in sorted(missing)})
    return {name: routes[name] for name in names if name in routes}
