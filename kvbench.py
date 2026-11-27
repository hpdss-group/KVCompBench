"""Unified benchmark entrypoint with manifest-managed codec installations."""
from pathlib import Path
import argparse,configparser,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parent
B=ROOT/'benchmark'
FRAMEWORK=B/'compression_framework'
DEFAULT=B/'settings.txt'
sys.path.insert(0,str(FRAMEWORK))

def basic_settings():
    if not DEFAULT.exists():DEFAULT.write_text((B/'settings.example.txt').read_text(encoding='utf-8'),encoding='utf-8')
    return DEFAULT

def config_path(value):
    p=Path(value) if value else DEFAULT
    return p if p.is_absolute() else ROOT/p

def configuration(settings):
    c=configparser.ConfigParser(interpolation=None)
    if not c.read(settings,encoding='utf-8'):raise SystemExit('Missing settings: '+str(settings))
    return c

def interpreter(settings):
    c=configuration(settings);explicit=c.get('environment','python',fallback='')
    if explicit:p=config_path(Path(explicit).expanduser()).absolute()
    else:
        name=c.get('environment','name')
        if Path(name).name!=name or name in ('.','..'):raise ValueError('Invalid environment name')
        p=ROOT/'.venvs'/name/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not p.is_file():raise SystemExit('Environment missing. Run: python kvbench.py setup --codecs identity,gzip,zlib,bz2,lzma')
    return p

def environment(settings):
    c=configuration(settings);env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    external=c.get('environment','external_lock',fallback='')
    if external:env['KVCOMPBENCH_EXTERNAL_LOCK']=str(config_path(external).resolve())
    else:env.pop('KVCOMPBENCH_EXTERNAL_LOCK',None)
    return env

def call(args,env=None):
    return subprocess.run(list(map(str,args)),cwd=ROOT,env=env,check=True)


def wsl_path(value):
    value = str(value)
    if len(value) >= 3 and value[1] == ':' and value[2] in ('/', '\\'):
        return '/mnt/' + value[0].lower() + value[2:].replace('\\', '/')
    return value


def wsl_command(args, settings, argv):
    """Honor the existing Windows launch configuration for a Linux environment."""
    if os.name != 'nt' or (args.command == 'codecs' and not args.status):
        return None
    if args.command == 'setup':
        if args.python and not str(args.python).startswith('/'):
            return None
        if args.external:
            lock = json.loads(args.external.read_text(encoding='utf-8'))
            if not str(lock.get('python', '')).startswith('/'):
                return None
    c = configuration(settings)
    distribution = c.get('environment', 'wsl_distribution', fallback='').strip()
    if not distribution:
        return None
    prefix = c.get('environment', 'wsl_env_path', fallback='').strip()
    python = 'python3' if args.command == 'setup' else prefix.rstrip('/') + '/bin/python' if prefix else wsl_path(ROOT / '.venvs' / c.get('environment', 'name')) + '/bin/python'
    command = ['wsl.exe', '--distribution', distribution, '--cd', wsl_path(ROOT), '--exec', 'env']
    for key in ('CUDA_VISIBLE_DEVICES', 'KVCOMPBENCH_EXTERNAL_LOCK', 'KVCOMPBENCH_REHASH'):
        if key in os.environ:
            command.append(key + '=' + (wsl_path(os.environ[key]) if key.endswith('_LOCK') else os.environ[key]))
    command += [python, '-B', wsl_path(ROOT / 'kvbench.py'), *map(wsl_path, argv)]
    return command

def main():
    from codec_manager import install,inventory,select
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    setup=sub.add_parser('setup',help='Install only selected codecs from pinned manifests')
    setup.add_argument('--codecs')
    setup.add_argument('--device',choices=['cpu','cuda13'],default='cpu')
    setup.add_argument('--plan',action='store_true',help='Show plan without downloads or installation')
    setup.add_argument('--env',help='Explicit project-local environment to install into')
    setup.add_argument('--optional',action='store_true',help='Compatibility alias: all packages with CUDA13')
    setup.add_argument('--external',type=Path,help='Validate an existing interpreter/native installation lock JSON')
    setup.add_argument('--python',type=Path,help='Explicit existing interpreter (native/source codecs also require --external)')
    for name in ['run','cli']:
        command=sub.add_parser(name);command.add_argument('--settings')
    codecs=sub.add_parser('codecs',help='List manifests and installed versions')
    codecs.add_argument('--status',action='store_true');codecs.add_argument('--codecs');codecs.add_argument('--settings')
    doctor=sub.add_parser('doctor',help='Check versions, interfaces and byte-exact round trips')
    doctor.add_argument('--codecs');doctor.add_argument('--settings')
    doctor.add_argument('--gpu',action='store_true',help='Opt in to GPU allocation and correctness checks')
    doctor.add_argument('--report',type=Path)
    args,extra=parser.parse_known_args()
    if args.command not in ('run','cli') and extra:parser.error('Unexpected arguments: '+' '.join(extra))
    settings = config_path(getattr(args, 'settings', None))
    if settings.exists():
        delegated = wsl_command(args, settings, sys.argv[1:])
        if delegated:
            call(delegated)
            return
    if args.command=='setup':
        names=args.codecs
        if args.optional:
            if names or args.external or args.python:parser.error('--optional cannot be combined with --codecs or external installation options')
            names=','.join(inventory());args.device='cuda13'
        if args.external or args.python:
            if args.env:parser.error('--env cannot be combined with an external installation')
            if args.plan:parser.error('--external validation does not use --plan')
            if args.external and args.python:parser.error('Use either --external or --python')
            path=args.external.resolve() if args.external else None
            lock=json.loads(path.read_text(encoding='utf-8')) if path else None
            if lock and lock.get('schema_version')!=1:raise ValueError('Unsupported external lock schema')
            py=Path(lock['python']).absolute() if lock else args.python.absolute()
            if not py.is_file():raise ValueError('External interpreter does not exist')
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
            if path:env['KVCOMPBENCH_EXTERNAL_LOCK']=str(path)
            else:env.pop('KVCOMPBENCH_EXTERNAL_LOCK',None)
            names=','.join(select(names)[0])
            call([py,'-B',FRAMEWORK/'codec_manager.py','doctor','--codecs',names],env)
            c=configuration(basic_settings())
            distribution=c.get('environment','wsl_distribution',fallback='')
            c['environment']={'name':'external','python':str(py)}
            if distribution:
                c['environment']['wsl_distribution']=distribution
                c['environment']['wsl_env_path']=str(py.parent.parent)
            if path:c['environment']['external_lock']=str(path)
            with DEFAULT.open('w',encoding='utf-8') as f:c.write(f)
            print('Validated external environment selected:',py)
        else:
            result=install(names,args.device,args.plan,environment_name=args.env);print(json.dumps(result,indent=2))
            if not args.plan:
                call([result['python'],'-B',FRAMEWORK/'codec_manager.py','doctor','--codecs',','.join(result['codecs']),
                      '--report',Path(result['prefix'])/'kvcompbench/doctor.json'])
    elif args.command=='codecs' and not args.status:
        _,packages=select(args.codecs) if args.codecs else (None,inventory())
        for name,info in packages.items():print(info['tier'],name,info['manifest']['revision'],', '.join(e['name'] for e in info['entries']))
    else:
        settings=config_path(args.settings) if args.settings else basic_settings()
        if args.command=='codecs':
            try:py=interpreter(settings)
            except SystemExit:
                print('Configured environment missing; inspecting the current interpreter.',file=sys.stderr)
                py=Path(sys.executable)
            command=[py,'-B',FRAMEWORK/'codec_manager.py','status']
            if args.codecs:command+=['--codecs',args.codecs]
        elif args.command=='doctor':
            py=interpreter(settings);c=configuration(settings)
            names=args.codecs or c.get('benchmark','codecs',fallback='').strip()
            if not names:
                from benchmark_routes import package_tier
                group=c.get('benchmark','codec_set',fallback='core')
                names=','.join(e['name'] for info in inventory().values()
                               if group=='all' or package_tier(info['folder'])=='core' for e in info['entries'])
            command=[py,'-B',FRAMEWORK/'codec_manager.py','doctor','--codecs',names]
            if args.gpu:command.append('--gpu')
            if args.report:command+=['--report',args.report.resolve()]
        else:
            py=interpreter(settings)
            target=B/'run_full_cpu_encode_gpu_decode.py' if args.command=='run' else FRAMEWORK/'cli.py'
            command=[py,'-u','-B',target,'--settings',settings,*extra]
        call(command,environment(settings))
if __name__=='__main__':
    try:main()
    except Exception as error:raise SystemExit('Error: '+str(error)+'\nFor build failures, inspect the selected environment\'s kvcompbench/install.log.')
