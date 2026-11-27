"""Compatibility entry point for Manifest-managed CUDA/native installation."""
import argparse
import json
from codec_manager import install, inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--codecs')
    parser.add_argument('--env')
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--native-only', action='store_true')
    parser.add_argument('--index-url', default='https://pypi.org/simple')
    args = parser.parse_args()
    if args.index_url != 'https://pypi.org/simple':
        parser.error('Custom indexes require a reviewed Manifest change')
    packages = inventory()
    names = args.codecs or ','.join(name for name, item in packages.items()
        if not args.native_only or item['manifest'].get('outputs'))
    result = install(names, 'cuda13', args.plan, environment_name=args.env)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
