"""Unified lossless compression CLI with INI settings and explicit overrides."""
import argparse
import configparser
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from core import DEFAULT_PLUGINS, load_plugins, pack, unpack

DEFAULT_SETTINGS = Path(__file__).resolve().parents[1] / 'settings.txt'


def settings(path):
    import configparser,re
    from pathlib import Path
    path=Path(path).resolve()
    config=configparser.ConfigParser(interpolation=None)
    with path.open(encoding='utf-8') as handle:config.read_file(handle)
    allowed={
        'environment':{'name','python','wsl_distribution','wsl_env_path'},
        'compression':{'codec','block_bytes','input','output','metadata'},
        'decompression':{'input','output','verify_sha256'},
        'benchmark_datasets':None,
        'benchmark':{'encode_batch_blocks','max_batch_bytes','max_batch_memory_fraction','retain_frames','cpu_encode_threads','decode_batch_blocks','skip_unavailable_codecs','codec_set','execution_device','codecs'},
        'benchmark_gpu_backends':None,
    }
    for section in config.sections():
        if section.startswith('dataset:'):
            name=section.split(':',1)[1]
            if Path(name).name!=name or not name.endswith('.pt') or set(config[section])-{'layout','axes','page_tokens','kv_semantics','dtype'}:raise ValueError('Invalid dataset section: '+section)
            continue
        if section not in allowed or (allowed[section] is not None and set(config[section])-allowed[section]):raise ValueError('Unknown configuration section or key: '+section)
        if section=='benchmark_datasets':
            for name,block in config[section].items():
                if Path(name).name!=name or not name.endswith('.pt') or int(block)<1:raise ValueError('Invalid benchmark dataset: '+name)
        if section=='benchmark_gpu_backends':
            for name,backend in config[section].items():
                if not name.replace('_','').isalnum() or not re.fullmatch(r'[a-z0-9][a-z0-9-]*',backend):raise ValueError('Invalid benchmark GPU backend declaration: '+name)
    from batch_policy import parse_batch, parse_max_bytes
    for field in ('encode_batch_blocks','decode_batch_blocks'):
        parse_batch(config.get('benchmark',field,fallback='1'))
    parse_max_bytes(config.get('benchmark','max_batch_bytes',fallback='536870912'))
    if not 0<config.getfloat('benchmark','max_batch_memory_fraction',fallback=0.5)<=0.8:raise ValueError('Invalid max batch budget')
    if config.get('benchmark','codec_set',fallback='all') not in ('core','all'):raise ValueError('codec_set must be core or all')
    if config.get('benchmark','execution_device',fallback='declared') not in ('cpu','declared'):raise ValueError('execution_device must be cpu or declared')
    if not config.getboolean('decompression','verify_sha256',fallback=True):raise ValueError('SHA-256 verification must remain enabled')
    return config,path.parent



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', default=str(DEFAULT_SETTINGS))
    parser.add_argument('--plugins', default=str(DEFAULT_PLUGINS))
    commands = parser.add_subparsers(dest='command', required=True)
    listing = commands.add_parser('list')
    listing.add_argument('--device', choices=('cpu','gpu','all'), default='cpu')
    probe = commands.add_parser('check')
    probe.add_argument('--codec', help='Check only one discovered codec')
    probe.add_argument('--device', choices=('cpu','gpu','all'), default='cpu')
    probe.add_argument('--metadata', default='{"dtype":"bf16"}')
    dependencies = commands.add_parser('install-plugin-deps')
    dependencies.add_argument('--index-url', default='https://pypi.org/simple')
    for command in ('compress','decompress'):
        sub = commands.add_parser(command)
        sub.add_argument('source', nargs='?')
        sub.add_argument('target', nargs='?')
        if command == 'compress':
            sub.add_argument('--codec')
            sub.add_argument('--block-bytes', type=int)
            sub.add_argument('--metadata')
    args = parser.parse_args()
    if args.command == 'install-plugin-deps':
        from plugins import dependency_files
        if sys.prefix == sys.base_prefix:
            raise RuntimeError('Run through run.sh or a project virtual environment to install plugin dependencies')
        scratch = Path(sys.prefix)/'tmp'
        scratch.mkdir(parents=True,exist_ok=True)
        environment = os.environ.copy()
        environment.update(TMPDIR=str(scratch),TEMP=str(scratch),TMP=str(scratch),PIP_CACHE_DIR=str(Path(sys.prefix)/'pip-cache'))
        for path in dependency_files(args.plugins):
            subprocess.run([sys.executable,'-m','pip','install','--no-compile','--index-url',args.index_url,'-r',str(path)],check=True,env=environment)
        return
    plugins = load_plugins(args.plugins)
    if args.command in ('list','check') and args.device != 'all':
        plugins = {name:p for name,p in plugins.items() if (getattr(p,'entry',{}).get('adapter') == 'gpu-api') == (args.device == 'gpu')}
    if args.command == 'list':
        for name, plugin in plugins.items():
            entry = getattr(plugin, 'entry', {})
            print(f'{name}\t{"adapter" if entry.get("adapter", True) else "inventory-only"}\t{entry.get("provenance", "custom")}')
        return
    if args.command == 'check':
        metadata = json.loads(args.metadata)
        if args.codec:
            if args.codec not in plugins:
                parser.error(f'Unknown codec in selected device group: {args.codec}')
            plugins = {args.codec: plugins[args.codec]}
        failures = 0
        for name, plugin in plugins.items():
            try:
                if not plugin.supports(metadata):
                    print(f'{name}\tSKIPPED_DTYPE')
                    continue
                if hasattr(plugin, 'prepare'):
                    plugin.prepare(metadata)
                raw = bytes(range(128)) * 2
                packet = plugin.compress(raw, dict(metadata))
                if plugin.decompress(packet, len(raw), dict(metadata)) != raw:
                    raise ValueError('Roundtrip mismatch')
                print(f'{name}\tPASS')
            except Exception as error:
                failures += 1
                print(f'{name}\tUNAVAILABLE_OR_FAILED\t{type(error).__name__}: {error}')
        if failures:
            raise SystemExit(1)
        return
    config, base = settings(args.settings)
    section = 'compression' if args.command == 'compress' else 'decompression'
    def location(override, key):
        value = override or config.get(section,key,fallback='').strip()
        if not value:
            raise ValueError(f'Supply {key} in settings.txt or on the command line')
        path = Path(value)
        return path if override or path.is_absolute() else base / path
    source, target = location(args.source,'input'), location(args.target,'output')
    if args.command == 'compress':
        codec = args.codec or config.get(section,'codec',fallback='zlib:6')
        block = args.block_bytes if args.block_bytes is not None else config.getint(section,'block_bytes',fallback=65536)
        metadata = json.loads(args.metadata or config.get(section,'metadata',fallback='{}'))
        if not isinstance(metadata,dict):
            raise ValueError('metadata must be a JSON object')
        print(json.dumps(pack(source,target,codec,block,metadata,args.plugins)))
    else:
        unpack(source,target,args.plugins)


if __name__ == '__main__':
    main()
