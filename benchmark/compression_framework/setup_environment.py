"""Install isolated CPU/CUDA API environments without modifying system software."""
import argparse
import configparser
import errno
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import venv

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ROOT / 'benchmark/settings.txt'


class Builder(venv.EnvBuilder):
    def ensure_directories(self, env_dir):
        try:
            return super().ensure_directories(env_dir)
        except OSError as error:
            alias = Path(env_dir) / 'lib64'
            if (os.name != 'nt' and error.errno in (errno.EPERM,errno.EACCES,errno.ENOTSUP,errno.ENOSYS)
                    and Path(str(error.filename2)).absolute() == alias.absolute() and not alias.exists()):
                alias.mkdir()
                return super().ensure_directories(env_dir)
            raise


def select_profile(profile, machine, system, driver_output=None):
    if system not in ('linux','windows'):
        raise RuntimeError('Supported systems: Linux and Windows')
    if profile != 'rtx5090':
        return profile
    if system != 'linux' or machine not in ('x86_64','amd64'):
        raise RuntimeError('RTX 5090 server profile requires Linux x86_64')
    devices = [line for line in (driver_output or '').splitlines() if '5090' in line]
    if not devices:
        raise RuntimeError('No RTX 5090 is reported by the NVIDIA driver')
    drivers = [int(line.rsplit(',',1)[1].strip().split('.')[0]) for line in devices]
    if min(drivers) < 570:
        raise RuntimeError('Unsupported driver for this RTX 5090 setup; administrator review required')
    return 'cuda13' if min(drivers) >= 580 else 'cuda12'


def environment_name(system, machine, profile):
    machine = {'amd64':'x86_64','arm64':'aarch64'}.get(machine.lower(),machine.lower())
    name = f'{system}-{machine}-{profile}'
    if not re.fullmatch(r'[a-z0-9_-]+',name):
        raise ValueError('Invalid environment identity')
    return name


def activate_setting(name):
    text = SETTINGS.read_text(encoding='utf-8')
    section = re.search(r'(?ms)^\[environment\]\s*\n(.*?)(?=^\[|\Z)',text)
    if section:
        body = section.group(1)
        if re.search(r'(?m)^name\s*=',body):
            body = re.sub(r'(?m)^name\s*=.*$',f'name = {name}',body)
        else:
            body = f'name = {name}\n' + body
        text = text[:section.start(1)] + body + text[section.end(1):]
    else:
        text = f'[environment]\nname = {name}\n\n' + text
    SETTINGS.write_text(text,encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=('cpu','cuda12','cuda13','rtx5090'),default='cpu')
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--index-url',default='https://pypi.org/simple',help='Python package index URL (default: official PyPI)')
    args = parser.parse_args()
    if sys.version_info < (3,10) or sys.maxsize <= 2**32:
        raise RuntimeError('A 64-bit Python 3.10+ installation with venv support is required')
    system = 'windows' if os.name == 'nt' else sys.platform
    machine = platform.machine().lower()
    driver_output = None
    if args.profile == 'rtx5090':
        if system != 'linux' or machine not in ('x86_64','amd64'):
            raise RuntimeError('RTX 5090 server profile requires Linux x86_64')
        if not any(ROOT.resolve().is_relative_to(p) for p in (Path('/home/liuxinyang'),Path('/data/lxy'))):
            raise RuntimeError('Shared-server project must be within /home/liuxinyang or /data/lxy')
        # Driver information only: no CUDA context or GPU workload is created.
        driver_output = subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True)
    profile = select_profile(args.profile,machine,system,driver_output)
    name = environment_name(system,machine,profile)
    env_root = ROOT / '.venvs'
    env = env_root / name
    python = env / ('Scripts/python.exe' if system == 'windows' else 'bin/python')
    print(f'Environment: {name}; profile: {profile}; path: {env}',flush=True)
    if args.preflight_only:
        return
    scratch = env / 'tmp'
    scratch.mkdir(parents=True,exist_ok=True)
    os.environ.update(TMPDIR=str(scratch),TEMP=str(scratch),TMP=str(scratch))
    tempfile.tempdir = str(scratch)
    if not python.exists():
        Builder(with_pip=True,symlinks=False).create(env)
    identity = json.loads(subprocess.check_output([str(python),'-I','-c','import platform,sys,json;print(json.dumps([sys.platform,platform.machine().lower(),list(sys.version_info[:2])]))'],text=True))
    expected_machine = environment_name(system,machine,profile)
    if environment_name('windows' if identity[0]=='win32' else identity[0],identity[1],profile) != expected_machine:
        raise RuntimeError('Existing environment belongs to another platform; preserve it and create a separate environment')
    environment = os.environ.copy()
    environment.update(PIP_CACHE_DIR=str(env/'pip-cache'),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    index = ['--index-url',args.index_url] if args.index_url else []
    def run(arguments):
        subprocess.run([str(python),*arguments],check=True,cwd=ROOT,env=environment)
    run(['-m','pip','install','--upgrade','pip',*index])
    run(['-m','pip','install','--no-compile','-r',str(Path(__file__).with_name('requirements.txt')),*index])
    from plugins import dependency_files
    for dependency in dependency_files(ROOT/'benchmark/compressors'):
        run(['-m','pip','install','--no-compile','-r',str(dependency),*index])
    if profile != 'cpu':
        cuda = 'cu12' if profile == 'cuda12' else 'cu13'
        run(['-m','pip','install','--no-compile',f'nvidia-nvcomp-{cuda}',*index])
    run(['-m','pip','check'])
    run(['-c','import numpy,zstandard,lz4.frame,snappy,fpzip; print("CPU codec imports: OK")'])
    # This checks CPU APIs only; GPU allocation requires an administrator-assigned device.
    framework = ROOT/'benchmark/compression_framework'
    run([str(framework/'cli.py'),'check','--device','cpu','--metadata','{"dtype":"fp32"}'])
    frozen = subprocess.check_output([str(python),'-m','pip','freeze'],text=True,env=environment)
    (env/'requirements.lock.txt').write_text(frozen,encoding='utf-8')
    (env/'environment.json').write_text(json.dumps(dict(name=name,python=str(python),platform=platform.platform(),profile=profile,gpu_tested=False),indent=2),encoding='utf-8')
    activate_setting(name)
    print(f'Environment ready: {python}',flush=True)
    print('Selected in benchmark/settings.txt. Run python3 kvbench.py cli list to inspect codecs.')


if __name__ == '__main__':
    main()
