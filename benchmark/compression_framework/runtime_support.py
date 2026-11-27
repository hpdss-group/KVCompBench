"""Optional codec-local runtime requirements; never substitute algorithms."""
import importlib.metadata,importlib.util,json,os,sys
from pathlib import Path
from packaging.requirements import Requirement

def unavailable(route,compressors):
    from codec_manager import identity
    report=identity(route.name,gpu=route.mode.endswith('-GPU'),enforce=False,directory=compressors)
    if report['errors']:return '; '.join(report['errors'])
    base=Path(compressors).resolve()
    for manifest in base.glob('*/codec.json'):
        declarations=json.loads(manifest.read_text(encoding='utf-8'))['codecs']
        if route.name not in {d['name'] for d in declarations}:continue
        note=manifest.parent/'runtime_requirements.json'
        if not note.exists():return None
        if not note.resolve().is_relative_to(base):raise ValueError('Runtime declaration escapes compressors')
        entry=json.loads(note.read_text(encoding='utf-8')).get(route.name,{})
        for text in entry.get('packages',[]):
            requirement=Requirement(text)
            try:version=importlib.metadata.version(requirement.name)
            except importlib.metadata.PackageNotFoundError:return 'Missing package: '+text
            if requirement.specifier and version not in requirement.specifier:return 'Unsupported package version: '+text+'; installed '+version
        # PEP 440 !=base also excludes local +patch releases. Block exact builds separately.
        for package,versions in entry.get('blocked_versions',{}).items():
            try:installed=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:return 'Missing package: '+package
            from packaging.version import Version
            if any(Version(installed)==Version(blocked) for blocked in versions):
                return 'Blocked package build: '+package+'=='+installed
        for module in entry.get('imports',[]):
            try:spec=importlib.util.find_spec(module)
            except (ImportError,ModuleNotFoundError):spec=None
            if spec is None:return 'Missing import: '+module
        library=entry.get('native_library')
        if library:
            if Path(library).name!=library:raise ValueError('Invalid native library name')
            from codec_manager import artifact_path
            target=artifact_path(library,'native/'+library+'/'+('codec.dll' if os.name=='nt' else 'libcodec.so'))
            if not target.is_file():return 'Missing native library: '+str(target)
        return None
    return None
