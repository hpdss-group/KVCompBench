"""Export project-owned source, adapters and manifests without third-party trees."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'input', 'output', '.runtime', '__pycache__', '.triton_cache',
            '.repository_archive', '.updates', '.git', '.venvs', 'node_modules'}
SOURCE_SUFFIXES = {'.py', '.json', '.md', '.txt', '.toml', '.yaml', '.yml',
                   '.c', '.cpp', '.h', '.hpp', '.cu', '.cuh', '.java',
                   '.html', '.css', '.js', '.sh', '.bat', '.ps1'}


def export(output):
    for folder in (ROOT / 'benchmark/compressors').iterdir():
        if folder.is_dir():
            for name in ('upstream', 'dependencies', 'tmp'):
                if (folder / name).exists():
                    raise ValueError('Archive the legacy third-party tree before export: ' + str(folder / name))
    files = {}
    roots = [ROOT / name for name in ('benchmark', 'tests', 'scripts', 'docs')]
    candidates = [p for p in ROOT.iterdir() if p.is_file()]
    for folder in roots:
        if folder.is_dir():
            import os
            for here, directories, names in os.walk(folder, followlinks=False):
                directories[:] = [d for d in directories if d not in EXCLUDED]
                candidates.extend(Path(here) / name for name in names)
    for path in candidates:
        relative = path.relative_to(ROOT)
        name = relative.as_posix()
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError('Source path escapes project: ' + name)
        if (path.suffix not in SOURCE_SUFFIXES and path.name not in ('.gitignore', '.gitattributes', 'LICENSE')):
            continue
        if (path.name == 'settings.txt' or path.name.endswith('.local.txt')
                or '.bak' in path.name or '.before' in path.name):
            continue
        files[name] = path.read_bytes()
    for name in ('benchmark/input/.gitkeep', 'benchmark/output/.gitkeep'):
        files[name] = b''
    manifest = {name: hashlib.sha256(raw).hexdigest() for name, raw in sorted(files.items())}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents overwriting a previous release artifact.
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(files.items()):
            archive.writestr(name, raw)
        archive.writestr('.source-manifest.json', json.dumps(manifest, sort_keys=True, indent=2))
    with zipfile.ZipFile(output) as archive:
        for name, expected in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError('Export checksum mismatch: ' + name)
    print(json.dumps(dict(output=str(output), files=len(files), bytes=output.stat().st_size,
                          sha256=hashlib.sha256(output.read_bytes()).hexdigest())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    export(parser.parse_args().output)
