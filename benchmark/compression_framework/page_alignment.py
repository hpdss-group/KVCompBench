"""Resolve exact byte-aligned KV pages independently for each stored format."""
import configparser
import math
import re
from pathlib import Path
from kv_pages import Pages, dataset_configs


def parse_lcm(value):
    text = str(value).strip()
    if not re.fullmatch(r'[0-9]+', text):
        raise ValueError('LCM must be a non-negative integer (0 disables alignment)')
    return int(text)


def lcm_multipliers(config):
    # Retain compatibility with saved settings from older benchmark runs.
    fallback = parse_lcm(config.get('benchmark', 'lcm', fallback='0'))
    return {kind: parse_lcm(config.get('benchmark', 'lcm_' + kind,
                                     fallback=str(fallback)))
            for kind in ('bf16', 'fp8')}


def aligned_dataset_configs(settings, input_dir):
    config = configparser.ConfigParser(interpolation=None)
    with Path(settings).open(encoding='utf-8') as handle:
        config.read_file(handle)
    layouts = dataset_configs(settings)
    multipliers = lcm_multipliers(config)
    if not any(multipliers.values()):
        return layouts
    groups = {}
    for name, layout in layouts.items():
        probe = Pages(Path(input_dir) / name, dict(layout, page_tokens=1))
        multiplier = multipliers['bf16' if probe.dtype == 'bf16' else 'fp8']
        if multiplier == 0:
            del probe
            continue
        if layout['layout'] != 'kv':
            raise ValueError('LCM requires KV layout: ' + name)
        group = (probe.dtype, probe.page_format)
        groups.setdefault(group, []).append((name, probe.block_bytes))
        del probe
    for group, members in groups.items():
        multiplier = multipliers['bf16' if group[0] == 'bf16' else 'fp8']
        page_bytes = multiplier * math.lcm(*(size for _, size in members))
        for name, token_bytes in members:
            layouts[name]['page_tokens'] = str(page_bytes // token_bytes)
            print('LCM_PAGE', name, 'format', group[0], 'token_bytes', token_bytes,
                  'page_tokens', page_bytes // token_bytes, 'page_bytes', page_bytes,
                  flush=True)
    return layouts
