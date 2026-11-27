"""Manifest-driven installation, provenance and opt-in codec diagnostics."""
from __future__ import annotations

import argparse
import ast
import configparser
from contextlib import ExitStack
from functools import lru_cache
import ctypes
import csv
import io
import hashlib
import importlib.metadata as md
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import venv

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ROOT / 'benchmark/compressors'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


_HASH_CACHE = None


def _binary_digest(path, size, modified_ns):
    """Reuse verified contents only while file identity and timestamps match."""
    global _HASH_CACHE
    cache_path = Path(sys.prefix) / 'kvcompbench/hash-cache.json'
    if _HASH_CACHE is None:
        try:
            data = json.loads(cache_path.read_text(encoding='utf-8'))
            _HASH_CACHE = data['files'] if data.get('schema_version') == 1 else {}
        except (OSError, ValueError, KeyError):
            _HASH_CACHE = {}
    target = Path(path)
    before = target.stat()
    stamp = [before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns]
    cached = _HASH_CACHE.get(str(path), {})
    if (os.environ.get('KVCOMPBENCH_REHASH') != '1' and cached.get('stat') == stamp
            and re.fullmatch('[0-9a-f]{64}', str(cached.get('sha256', '')))):
        return cached['sha256']
    result = digest(target)
    after = target.stat()
    if stamp != [after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns]:
        raise ValueError('Binary changed while hashing: ' + str(path))
    _HASH_CACHE[str(path)] = dict(stat=stamp, sha256=result)
    if cache_path.parent.is_dir():
        try:
            # Concurrent writers may lose cache hits, but never fabricate hashes.
            temporary = cache_path.with_name(cache_path.name + '.' + str(os.getpid()) + '.tmp')
            temporary.write_text(json.dumps(dict(schema_version=1, files=_HASH_CACHE)), encoding='utf-8')
            temporary.replace(cache_path)
        except OSError:
            pass  # Read-only environments retain the in-process cache.
    return result


def native_files(distribution):
    files = {}
    # Python 3.12 Distribution.files checks existence of *every* wheel member.
    # Filter RECORD first; otherwise a large PyTorch wheel causes thousands of
    # expensive DrvFS stat calls before the first native library is inspected.
    record = distribution.read_text('RECORD')
    candidates = (row[0] for row in csv.reader(io.StringIO(record)) if row) if record is not None else distribution.files or []
    for relative in candidates:
        name = str(relative).lower()
        if not name.endswith(('.so', '.pyd', '.dll', '.dylib')) and '.so.' not in name:
            continue
        path = distribution.locate_file(relative).absolute()
        if not path.is_file():
            raise ValueError('Missing installed native dependency: ' + str(path))
        stat = path.stat()
        files[str(relative)] = dict(path=str(path), sha256=_binary_digest(str(path), stat.st_size, stat.st_mtime_ns))
    return files


def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError('Installation path escapes its root: ' + str(relative))
    return path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(path)


def recipe_fingerprint(manifest):
    """Executable installation recipe; mirror URLs/documentation are provenance."""
    def normalized(value):
        if isinstance(value, dict):
            return {k: normalized(v) for k, v in value.items()
                    if k not in ('url', 'repository', 'description', 'timeout_seconds')}
        if isinstance(value, list):
            return [normalized(v) for v in value]
        return value
    return hashlib.sha256(json.dumps(normalized(manifest), sort_keys=True,
                                      separators=(',', ':')).encode()).hexdigest()


def inventory(directory=None):
    result = {}
    for folder in sorted(Path(directory or PACKAGES).iterdir()):
        spec = folder / 'codec.json'
        if not spec.is_file():
            continue
        entries = json.loads(spec.read_text(encoding='utf-8'))['codecs']
        install = folder / 'install.json'
        if not install.is_file():
            raise ValueError('Missing installation manifest: ' + str(install))
        manifest = json.loads(install.read_text(encoding='utf-8'))
        if manifest.get('schema_version') != 1 or manifest.get('package') != folder.name:
            raise ValueError('Invalid installation manifest: ' + str(install))
        revision = manifest.get('revision', '')
        if not revision or Path(revision).name != revision or revision in ('.', '..'):
            raise ValueError('Invalid pinned revision')
        for requirement in manifest.get('requirements', []) + manifest.get('gpu_requirements', []) + manifest.get('installed_distributions', []):
            if '==' not in requirement or any(x in requirement for x in ('*', '://', ';')):
                raise ValueError('Dependencies must be exact distribution==version pins')
        tier_file = folder / 'package.json'
        tier = json.loads(tier_file.read_text(encoding='utf-8')).get('tier', 'optional') if tier_file.is_file() else 'optional'
        if tier not in ('core', 'optional'):
            raise ValueError('Invalid package tier: ' + str(folder))
        result[folder.name] = dict(folder=folder, manifest=manifest, entries=entries, tier=tier,
                                   manifest_sha256=digest(install))
    return result


def select(names=None):
    packages = inventory()
    codecs = {entry['name']: name for name, value in packages.items() for entry in value['entries']}
    requested = [x.strip() for x in (names or 'identity,gzip,zlib,bz2,lzma').split(',')]
    if not requested or any(not x for x in requested) or len(requested) != len(set(requested)):
        raise ValueError('Codec selection must contain unique, nonempty names')
    chosen = []
    for name in requested:
        if name in codecs:
            chosen.append(name)
        elif name in packages:
            chosen.extend(entry['name'] for entry in packages[name]['entries'])
        else:
            raise ValueError('Unknown codec or package: ' + name)
    return list(dict.fromkeys(chosen)), {codecs[name]: packages[codecs[name]] for name in chosen}


def package_for(codec, directory=None):
    for package, value in inventory(directory).items():
        if any(entry['name'] == codec for entry in value['entries']):
            return package, value
    raise ValueError('Unknown codec: ' + codec)


def external_lock():
    filename = os.environ.get('KVCOMPBENCH_EXTERNAL_LOCK')
    if not filename:
        return None
    path = Path(filename).resolve()
    value = json.loads(path.read_text(encoding='utf-8'))
    if value.get('schema_version') != 1 or Path(value['python']).absolute() != Path(sys.executable).absolute():
        raise ValueError('External lock does not describe this interpreter')
    return value


def native_root(package):
    lock = external_lock()
    if lock and package in lock.get('packages', {}):
        value = lock['packages'][package]
        if value.get('native_root'):
            return Path(value['native_root']).resolve()
    return Path(sys.prefix) / 'native'


def artifact_path(package, relative):
    lock = external_lock()
    record = lock.get('packages', {}).get(package, {}) if lock else {}
    if relative in record.get('artifact_paths', {}):
        return Path(record['artifact_paths'][relative]).resolve()
    if relative.startswith('native/'):
        return contained(native_root(package), relative[len('native/'):])
    return contained(Path(sys.prefix), relative)


def source_dir(package, directory=None):
    return _source_root(package, inventory(directory)[package])


def _source_root(package, info):
    lock = external_lock()
    if lock and package in lock.get('packages', {}):
        value = lock['packages'][package]
        if value.get('source_dir'):
            return Path(value['source_dir']).resolve()
    return Path(sys.prefix) / 'kvcompbench/sources' / package / info['manifest']['revision']


def _source_file(relative, entry, base, external):
    if external and relative in external.get('source_files', {}):
        return Path(external['source_files'][relative]).resolve()
    path = contained(base, relative)
    if external and not path.is_file():
        # Accept the original pinned repository layout as well as the managed subset layout.
        path = contained(base, str(Path(*Path(entry['member']).parts[1:])))
    return path


def source_file(package, relative):
    info = inventory()[package]
    lock = external_lock()
    external = lock.get('packages', {}).get(package) if lock else None
    entry = info['manifest']['sources']['files'][relative]
    return _source_file(relative, entry, _source_root(package, info), external)


def download(item, cache):
    url, expected = item['url'], item['sha256']
    if not url.startswith('https://') or re.fullmatch('[0-9a-f]{64}', expected) is None:
        raise ValueError('Downloads require HTTPS and a SHA-256 pin')
    path = Path(cache) / expected
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if digest(path) != expected:
            raise ValueError('Cached download checksum mismatch: ' + str(path))
        return path
    temp = path.with_suffix('.partial')
    if temp.exists():
        raise FileExistsError('Inspect the incomplete previous download before retrying: ' + str(temp))
    created = False
    timeout = int(item.get('timeout_seconds', 300))
    if not 1 <= timeout <= 1800:
        raise ValueError('Download timeout must be between 1 and 1800 seconds')
    deadline = time.monotonic() + timeout
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'KVCompBench/1'})
        with urllib.request.urlopen(request, timeout=min(30, timeout)) as response, temp.open('xb') as output:
            created = True
            size = 0
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError('Pinned download exceeded its time limit: ' + url)
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > 1024 * 1024 * 1024:
                    raise ValueError('Download exceeds the 1 GiB limit')
                output.write(chunk)
        if digest(temp) != expected:
            raise ValueError('Download checksum mismatch: ' + url)
        temp.replace(path)
    finally:
        if created and temp.exists():
            temp.unlink()
    return path


def write_verified(path, raw, expected):
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Source checksum mismatch: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if digest(path) != expected:
            raise ValueError('Existing managed source has changed: ' + str(path))
        return
    temp = path.with_name(path.name + '.tmp')
    created = False
    try:
        with temp.open('xb') as output:
            created = True
            output.write(raw)
        temp.replace(path)
    finally:
        if created and temp.exists():
            temp.unlink()


def fetch_sources(spec, destination, cache):
    missing = False
    for relative, item in spec.get('files', {}).items():
        target = contained(destination, relative)
        if not target.exists():
            missing = True
        elif not target.is_file() or digest(target) != item.get('installed_sha256', item['sha256']):
            raise ValueError('Existing managed source has changed: ' + str(target))
    if spec.get('files') and not missing:
        return destination
    archives = {name: download(item, cache) for name, item in spec.get('archives', {}).items()}
    with ExitStack() as stack:
        opened = {name: stack.enter_context(tarfile.open(path)) for name, path in archives.items()}
        entries = [(relative, entry, opened[entry['archive']].getmember(entry['member']))
                   for relative, entry in spec.get('files', {}).items()]
        for relative, entry, member in sorted(entries, key=lambda x: (x[1]['archive'], x[2].offset_data)):
            path = contained(destination, relative)
            if not member.isfile() or member.size > 64 * 1024 * 1024:
                raise ValueError('Invalid source archive member')
            raw = opened[entry['archive']].extractfile(member).read()
            if hashlib.sha256(raw).hexdigest() != entry['sha256']:
                raise ValueError('Pinned archive member changed: ' + relative)
            if entry.get('patches'):
                text = raw.decode('utf-8')
                for patch in entry['patches']:
                    if not patch['old'] or text.count(patch['old']) != 1:
                        raise ValueError('Source patch precondition failed: ' + relative)
                    text = text.replace(patch['old'], patch['new'], 1)
                raw = text.encode('utf-8')
            write_verified(path, raw, entry.get('installed_sha256', entry['sha256']))
    return destination


def extract_resource(archive_path, target, strip=1):
    """Reject links and traversal; create executable files without chmod."""
    with tarfile.open(archive_path) as archive:
        for member in archive:
            if member.issym() or member.islnk():
                continue
            if member.isdev():
                raise ValueError('Devices are not permitted in toolchain archives')
            parts = Path(member.name).parts[strip:]
            if not parts or member.isdir():
                continue
            if not member.isfile():
                raise ValueError('Unsupported archive entry')
            path = contained(target, str(Path(*parts)))
            raw = archive.extractfile(member).read()
            if path.exists():
                if path.read_bytes() != raw:
                    raise ValueError('Existing toolchain file differs: ' + str(path))
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, member.mode & 0o777)
            with os.fdopen(descriptor, 'wb') as output:
                output.write(raw)


def platform_id():
    machine = platform.machine().lower()
    return platform.system().lower() + '-' + {'amd64': 'x86_64'}.get(machine, machine)


def run(command, cwd, environment, log):
    log.write('+ ' + ' '.join(map(str, command)) + '\n')
    log.flush()
    subprocess.run(list(map(str, command)), cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                   stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)


def expand_command(command, variables):
    output = []
    for item in command:
        if item.startswith('@glob:'):
            pattern = item[6:]
            prefix, relative = pattern.split('}/', 1)
            base = Path(variables[prefix[1:]])
            matches = sorted(base.glob(relative))
            if not matches:
                raise ValueError('Empty build source glob: ' + item)
            if any(not p.resolve().is_relative_to(base.resolve()) for p in matches):
                raise ValueError('Build source escapes managed directory')
            output.extend(str(p) for p in matches)
        else:
            output.append(item.format_map(variables))
    return output


def installed_distributions(requirements):
    result = {}
    for requirement in requirements:
        name, expected = requirement.split('==', 1)
        distribution = md.distribution(name)
        actual = distribution.version
        if actual != expected:
            raise ValueError(f'{name}: expected {expected}, installed {actual}')
        dependencies = {}
        from packaging.requirements import Requirement
        for text in distribution.requires or []:
            dependency = Requirement(text)
            if dependency.marker and not dependency.marker.evaluate({'extra': ''}):
                continue
            dependency_distribution = md.distribution(dependency.name)
            version = dependency_distribution.version
            if dependency.specifier and version not in dependency.specifier:
                raise ValueError(f'{name}: dependency {text}, installed {version}')
            dependencies[dependency.name] = dict(version=version,
                                                  location=str(dependency_distribution.locate_file('').resolve()),
                                                  native_files=native_files(dependency_distribution))
        result[name] = dict(version=actual, location=str(distribution.locate_file('').resolve()),
                            requirements=distribution.requires or [], dependencies=dependencies,
                            native_files=native_files(distribution))
    return result


def receipt_path(package):
    return Path(sys.prefix) / 'kvcompbench/receipts' / (package + '.json')


def reusable_receipt(path, info, prefix):
    if not path.is_file() or not info['manifest'].get('outputs'):
        return None
    receipt = json.loads(path.read_text(encoding='utf-8'))
    same_recipe = (receipt.get('recipe_sha256') == recipe_fingerprint(info['manifest'])
                   if receipt.get('recipe_sha256') else receipt.get('manifest_sha256') == info['manifest_sha256'])
    if not receipt.get('complete') or not same_recipe or receipt.get('revision') != info['manifest']['revision']:
        return None
    artifacts = receipt.get('artifacts', {})
    if not set(info['manifest']['outputs']) <= artifacts.keys():
        return None
    for relative, expected in artifacts.items():
        path = contained(prefix, relative)
        if not path.is_file() or digest(path) != expected:
            return None
    return receipt


def identity(codec, gpu=False, enforce=True, directory=None):
    package, info = package_for(codec, directory)
    manifest = info['manifest']
    executable = Path(sys.executable).resolve()
    executable_stat = executable.stat()
    result = dict(package=package, revision=manifest['revision'], manifest_sha256=info['manifest_sha256'],
                  python=str(Path(sys.executable).absolute()), python_binary=str(executable),
                  python_sha256=_binary_digest(str(executable), executable_stat.st_size, executable_stat.st_mtime_ns),
                  python_version=platform.python_version(),
                  platform=platform_id(), operating_system=platform.platform(),
                  prefix=str(Path(sys.prefix).resolve()), distributions={}, imports={}, artifacts={})
    errors = []
    requirements = manifest.get('requirements', []) + manifest.get('installed_distributions', []) + (manifest.get('gpu_requirements', []) if gpu else [])
    for requirement in requirements:
        try:
            result['distributions'].update(installed_distributions([requirement]))
        except (ImportError, md.PackageNotFoundError, ValueError) as error:
            errors.append(str(error))
    if gpu:
        try:
            torch = md.distribution('torch')
            result['distributions']['torch'] = dict(version=torch.version,
                                                    location=str(torch.locate_file('').resolve()),
                                                    requirements=torch.requires or [])
            if torch.version.split('+', 1)[0] != '2.9.1':
                raise ValueError('GPU routes require the pinned PyTorch 2.9.1 base version; installed ' + torch.version)
            version_file = torch.locate_file('torch/version.py')
            cuda = None
            for node in ast.parse(version_file.read_text(encoding='utf-8')).body:
                targets = node.targets if isinstance(node, ast.Assign) else ([node.target] if isinstance(node, ast.AnnAssign) else [])
                if any(isinstance(target, ast.Name) and target.id == 'cuda' for target in targets):
                    cuda = ast.literal_eval(node.value)
            result['torch_cuda_build'] = cuda
            if not isinstance(cuda, str) or not cuda.startswith('13.'):
                raise ValueError('GPU profile requires a CUDA13 PyTorch build; installed build metadata: ' + str(cuda))
            if not errors:
                result['distributions'].update(installed_distributions(['torch==' + torch.version]))
        except (ImportError, md.PackageNotFoundError, OSError, ValueError) as error:
            errors.append(str(error))
    for name in manifest.get('imports', []) + (manifest.get('gpu_imports', []) if gpu else []):
        try:
            spec = importlib.util.find_spec(name)
            if spec is None:
                raise ImportError(name)
            origin = Path(spec.origin) if spec.origin and spec.origin not in ('built-in', 'frozen') else None
            result['imports'][name] = dict(path=spec.origin or list(spec.submodule_search_locations or []),
                                           sha256=digest(origin) if origin and origin.is_file() else None)
        except (ImportError, ValueError, ModuleNotFoundError) as error:
            errors.append('Missing interface: ' + name + ': ' + str(error))
    lock = external_lock()
    external = lock.get('packages', {}).get(package) if lock else None
    receipt = json.loads(receipt_path(package).read_text()) if receipt_path(package).is_file() else None
    evidence = external or receipt
    if manifest.get('platforms') and platform_id() not in manifest['platforms'] and not external:
        errors.append('Unsupported managed runtime platform: ' + platform_id())
    if evidence:
        if evidence.get('revision') != manifest['revision']:
            errors.append('Installed upstream revision does not match manifest')
        if not external and not evidence.get('complete'):
            errors.append('Incomplete installation receipt')
        same_recipe = (evidence.get('recipe_sha256') == recipe_fingerprint(manifest)
                       if evidence.get('recipe_sha256') else evidence.get('manifest_sha256') == info['manifest_sha256'])
        if not external and not same_recipe:
            errors.append('Installation manifest changed; reinstall selected codec')
    for relative, expected in manifest.get('build_inputs', {}).items():
        path = contained(info['folder'], relative)
        if not path.is_file() or digest(path) != expected:
            errors.append('Build adapter changed: ' + str(path))
    if manifest.get('outputs') or manifest.get('sources'):
        if not evidence:
            errors.append('No managed receipt or explicit external lock; run setup --codecs ' + codec)
        outputs = set(manifest.get('outputs', [])) | set((evidence or {}).get('artifacts', {}))
        for relative in outputs:
            path = artifact_path(package, relative)
            if not path.is_file():
                errors.append('Missing artifact: ' + str(path))
            else:
                actual = digest(path)
                result['artifacts'][relative] = dict(path=str(path), sha256=actual)
                if not evidence or evidence.get('artifacts', {}).get(relative) != actual:
                    errors.append('Unverified/changed artifact: ' + str(path))
        source_base = _source_root(package, info)
        result['sources'] = {}
        for relative, item in manifest.get('sources', {}).get('files', {}).items():
            path = _source_file(relative, item, source_base, external)
            if not path.is_file() or digest(path) != item.get('installed_sha256', item['sha256']):
                errors.append('Missing/changed pinned source: ' + str(path))
            elif path.is_file():
                result['sources'][relative] = dict(path=str(path), sha256=digest(path))
    result['recipe_sha256'] = recipe_fingerprint(manifest)
    result['build_configuration'] = {key: manifest[key] for key in ('commands', 'build_inputs') if key in manifest}
    result['compiler_versions'] = sorted((evidence or {}).get('tools', {}).values())
    result.update(origin='external' if external else ('managed' if receipt else 'interpreter'),
                  version_evidence='external declared revision + verified hashes' if external else 'manifest + installed metadata',
                  build_tools=(evidence or {}).get('tools', {}),
                  status='ready' if not errors else 'unavailable', errors=errors)
    if enforce and errors:
        raise RuntimeError(codec + ': ' + '; '.join(errors))
    return result


def diagnose(names, gpu=False, correctness=False):
    from core import load_plugins
    plugins = load_plugins(PACKAGES)
    selected, _ = select(names)
    reports = []
    for name in selected:
        package, info = package_for(name)
        entry = next(e for e in info['entries'] if e['name'] == name)
        gpu_only = entry.get('device') == 'gpu' or info['manifest'].get('gpu_only', False)
        report = identity(name, gpu=gpu or gpu_only, enforce=False)
        report.update(codec=name, tier=info.get('tier', 'optional'), cpu='not_supported' if gpu_only else 'not_tested', gpu='not_requested')
        routes_file = info['folder'] / 'benchmark.json'
        options = json.loads(routes_file.read_text()).get(name, {}) if routes_file.is_file() else {}
        mode = options.get('execution_path', 'GPU-GPU' if gpu_only else 'CPU-CPU')
        decoders_file = info['folder'] / 'decode_backends.json'
        decoders = json.loads(decoders_file.read_text()).get(name, {}) if decoders_file.is_file() else {}
        report['declared_interfaces'] = dict(cpu_encode=mode.startswith('CPU-'), gpu_encode=mode.startswith('GPU-'),
                                              cpu_decode=mode.endswith('-CPU') or bool(decoders.get('cpu')),
                                              gpu_decode=mode.endswith('-GPU'))
        if report['declared_interfaces']['gpu_encode'] or report['declared_interfaces']['gpu_decode']:
            probe = report if gpu or gpu_only else identity(name, gpu=True, enforce=False)
            report['gpu_prerequisites'] = {key: probe.get(key) for key in
                                           ('status', 'errors', 'distributions', 'imports', 'torch_cuda_build')}
        if report['status'] == 'ready' and correctness:
            if gpu_only and not gpu:
                report['gpu'] = 'deferred: use doctor --gpu with an assigned device'
            else:
                try:
                    if gpu and (report['declared_interfaces']['gpu_decode'] or report['declared_interfaces']['gpu_encode']) and not os.environ.get('CUDA_VISIBLE_DEVICES', '').strip():
                        raise RuntimeError('GPU checks require an explicitly assigned CUDA_VISIBLE_DEVICES')
                    dtype = next((d for d in ('bf16', 'fp8_e4m3fn', 'fp8_e5m2', 'fp32', 'fp64', 'uint8', 'bytes') if plugins[name].supports({'dtype': d})), None)
                    if dtype is None:
                        raise ValueError('No supported diagnostic dtype')
                    unit = {'bf16': struct.pack('<5H', 0, 0x8000, 0x3f80, 0x7fc1, 0x7f80),
                            'fp32': struct.pack('<5f', 0, -0.0, 1.25, -3.5, 123.25),
                            'fp64': struct.pack('<5d', 0, -0.0, 1.25, -3.5, 123.25)}.get(dtype, b'\0\1\xffabc')
                    if not gpu_only:
                        for count in (1, 7, 1025):
                            raw = unit * count
                            frame = plugins[name].compress(raw, {'dtype': dtype})
                            if plugins[name].decompress(frame, len(raw), {'dtype': dtype}) != raw:
                                raise ValueError('CPU byte-exact round trip failed')
                        report['cpu'] = 'verified'
                    if gpu:
                        if not report['declared_interfaces']['gpu_decode']:
                            report['gpu'] = 'not_supported'
                        else:
                            from gpu_diagnostics import check_gpu
                            checks = check_gpu(name, plugins[name], info['folder'], PACKAGES)
                            report['gpu_checks'] = checks
                            failed = [item for item in checks if item['status'] != 'verified']
                            if failed:
                                raise ValueError('; '.join(item['dtype'] + ' ' + str(item['options']) + ': ' + item['error'] for item in failed))
                            report['gpu'] = 'verified' if checks else 'not_supported'
                except Exception as error:
                    report['status'] = 'failed'
                    report['errors'].append(type(error).__name__ + ': ' + str(error))
        reports.append(report)
    return reports


def install(names, profile='cpu', plan=False, base=True, environment_name=None):
    selected, packages = select(names)
    env_name = environment_name or platform_id() + '-managed-' + profile
    if Path(env_name).name != env_name or env_name in ('.', '..'):
        raise ValueError('Invalid environment name')
    prefix = ROOT / '.venvs' / env_name
    python = prefix / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    requirements = sorted({r for info in packages.values() for r in
                           info['manifest'].get('requirements', []) +
                           (info['manifest'].get('gpu_requirements', []) if profile == 'cuda13' else [])})
    for name, info in packages.items():
        manifest = info['manifest']
        if manifest.get('platforms') and platform_id() not in manifest['platforms']:
            raise RuntimeError(name + ': supported build platforms: ' + ', '.join(manifest['platforms']))
        if manifest.get('gpu_only') and profile != 'cuda13':
            raise RuntimeError(name + ' requires --device cuda13')
    if profile == 'cuda13' and platform_id() != 'linux-x86_64':
        raise RuntimeError('Managed CUDA13 installation currently supports Linux x86-64')
    if plan:
        return dict(codecs=selected, packages=list(packages), prefix=str(prefix), profile=profile, requirements=requirements,
                    manifests={name: info['manifest'] for name, info in packages.items()})
    prefix.mkdir(parents=True, exist_ok=True)
    lock = prefix / 'kvcompbench/install.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('x'):
        pass
    try:
        scratch = prefix / 'tmp'
        scratch.mkdir(exist_ok=True)
        os.environ.update(TMPDIR=str(scratch), TEMP=str(scratch), TMP=str(scratch))
        tempfile.tempdir = str(scratch)
        if not python.is_file():
            subprocess.run([sys.executable, '-m', 'venv', str(prefix)],
                           check=True, timeout=300, stdin=subprocess.DEVNULL)
        environment = dict(os.environ, TMPDIR=str(scratch), TEMP=str(scratch), TMP=str(scratch),
                           PIP_CACHE_DIR=str(prefix / 'pip-cache'), PYTHONDONTWRITEBYTECODE='1', MAX_JOBS='1')
        cache = prefix / 'kvcompbench/downloads'
        log_path = prefix / 'kvcompbench/install.log'
        with log_path.open('a', encoding='utf-8', buffering=1) as log:
            if base:
                index = 'https://download.pytorch.org/whl/' + ('cu130' if profile == 'cuda13' else 'cpu')
                run([python, '-m', 'pip', 'install', '--no-compile', 'torch==2.9.1', '--index-url', index], ROOT, environment, log)
                run([python, '-m', 'pip', 'install', '--no-compile', 'numpy==2.2.6', 'packaging==25.0', 'wheel==0.45.1', 'setuptools==80.9.0', '--index-url', 'https://pypi.org/simple'], ROOT, environment, log)
            if requirements:
                run([python, '-m', 'pip', 'install', '--no-compile', *requirements, '--index-url', 'https://pypi.org/simple'], ROOT, environment, log)
            run([python, '-m', 'pip', 'check'], ROOT, environment, log)
            versions = subprocess.check_output([str(python), '-m', 'pip', 'freeze'], env=environment, text=True, timeout=120)
            for name, info in packages.items():
                manifest = info['manifest']
                for relative, expected in manifest.get('build_inputs', {}).items():
                    if digest(contained(info['folder'], relative)) != expected:
                        raise ValueError('Build adapter checksum mismatch: ' + relative)
                src = prefix / 'kvcompbench/sources' / name / manifest['revision']
                if manifest.get('sources'):
                    fetch_sources(manifest['sources'], src, cache)
                record_path = prefix / 'kvcompbench/receipts' / (name + '.json')
                previous = reusable_receipt(record_path, info, prefix)
                for resource in ([] if previous else manifest.get('resources', [])):
                    archive = download(resource, cache)
                    target = contained(prefix, resource['target'])
                    if resource.get('extract'):
                        extract_resource(archive, target, resource.get('strip', 1))
                    else:
                        write_verified(target, archive.read_bytes(), resource['sha256'])
                variables = dict(source=str(src), plugin=str(info['folder']), prefix=str(prefix),
                                 python=str(python), jdk=str(prefix / 'toolchains/jdk17'))
                for directory in manifest.get('directories', []):
                    contained(prefix, directory).mkdir(parents=True, exist_ok=True)
                for command in ([] if previous else manifest.get('commands', [])):
                    run(expand_command(command, variables), ROOT, environment, log)
                artifacts = {}
                outputs = set(manifest.get('outputs', []))
                for pattern in manifest.get('output_globs', []):
                    matches = list(prefix.glob(pattern))
                    if not matches:
                        raise ValueError('No built artifacts match ' + pattern)
                    outputs.update(p.relative_to(prefix).as_posix() for p in matches if p.is_file())
                for relative in sorted(outputs):
                    path = contained(prefix, relative)
                    if not path.is_file():
                        raise RuntimeError('Build did not produce ' + str(path))
                    artifacts[relative] = digest(path)
                tools = previous.get('tools', {}) if previous else {}
                for executable in ([] if previous else manifest.get('tools', [])):
                    tool = executable.format_map(variables)
                    tools[tool] = subprocess.check_output([tool, '--version'], stderr=subprocess.STDOUT, text=True, timeout=20).splitlines()[0]
                if manifest.get('installed_distributions'):
                    run([python, '-m', 'pip', 'check'], ROOT, environment, log)
                    versions = subprocess.check_output([str(python), '-m', 'pip', 'freeze'], env=environment, text=True, timeout=120)
                atomic_json(record_path, dict(package=name, complete=True, revision=manifest['revision'],
                                             manifest_sha256=info['manifest_sha256'], artifacts=artifacts,
                                             recipe_sha256=recipe_fingerprint(manifest),
                                             python=str(python), tools=tools, dependencies=versions.splitlines(),
                                             reused_build=bool(previous)))
            run([python, '-m', 'pip', 'check'], ROOT, environment, log)
        settings = ROOT / 'benchmark/settings.txt'
        c = configparser.ConfigParser(interpolation=None)
        c.read(settings if settings.exists() else ROOT / 'benchmark/settings.example.txt', encoding='utf-8')
        distribution = c.get('environment', 'wsl_distribution', fallback='').strip()
        c['environment'] = {'name': env_name}
        if distribution:
            c['environment']['wsl_distribution'] = distribution
            c['environment']['wsl_env_path'] = str(prefix)
        with settings.open('w', encoding='utf-8') as handle:
            c.write(handle)
        return dict(codecs=selected, prefix=str(prefix), python=str(python), log=str(log_path))
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['setup', 'status', 'doctor'])
    parser.add_argument('--codecs')
    parser.add_argument('--device', choices=['cpu', 'cuda13'], default='cpu')
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--env', help='Explicit project-local environment to install into')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    names = args.codecs
    if args.command == 'status' and names is None:
        names = ','.join(e['name'] for value in inventory().values() for e in value['entries'])
    if args.command == 'setup':
        result = install(args.codecs, args.device, args.plan, environment_name=args.env)
    else:
        result = diagnose(names, gpu=args.gpu, correctness=args.command == 'doctor')
    if args.report:
        atomic_json(args.report.resolve(), result)
    print(json.dumps(result, indent=2))
    if args.command == 'doctor' and any(item['status'] != 'ready' for item in result):
        raise SystemExit(1)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(status='error', error=type(error).__name__ + ': ' + str(error)), indent=2))
        raise SystemExit(1)
