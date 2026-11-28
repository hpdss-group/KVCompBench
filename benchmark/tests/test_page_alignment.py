"""CPU tests for independent, scale-aware alignment of stored KV formats."""
import configparser
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'compression_framework'))
from page_alignment import aligned_dataset_configs, lcm_multipliers, parse_lcm
from kv_pages import Pages, dataset_configs


class PageAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = self.root / 'settings.ini'
        self.config = configparser.ConfigParser()
        self.config['benchmark'] = {}
        self.config['benchmark_datasets'] = {}
        for kind, dtype in [('bf16', torch.bfloat16),
                            ('fp8_e4m3fn', torch.float8_e4m3fn),
                            ('fp8_e5m2', torch.float8_e5m2)]:
            for model in ('ds', 'qwen'):
                name = model + '_' + kind + '.pt'
                shape = (1, 7, 512) if model == 'ds' else (1, 2, 1, 4, 9, 256)
                value = {'data': torch.zeros(shape).to(dtype)}
                if kind != 'bf16':
                    value['scale'] = torch.ones(shape[:-1] + (1,), dtype=torch.float32)
                torch.save(value, self.root / name)
                self.config['benchmark_datasets'][name] = '4096'
                self.config['dataset:' + name] = dict(
                    layout='kv', page_tokens='3', dtype=kind,
                    axes='layer,token,latent' if model == 'ds' else 'layer,kv,batch,head,token,dim',
                    kv_semantics='shared' if model == 'ds' else 'separate')

    def resolve(self, bf16, fp8):
        self.config['benchmark']['lcm_bf16'] = str(bf16)
        self.config['benchmark']['lcm_fp8'] = str(fp8)
        with self.settings.open('w') as handle:
            self.config.write(handle)
        return aligned_dataset_configs(self.settings, self.root)

    def test_independent_groups_and_lossless_page_boundaries(self):
        for bf16, fp8 in [(0, 0), (32, 1), (1, 2), (0, 2), (2, 0), (3, 4)]:
            with self.subTest(bf16=bf16, fp8=fp8):
                layouts = self.resolve(bf16, fp8)
                manual = dataset_configs(self.settings)
                for name, layout in layouts.items():
                    pages = Pages(self.root / name, layout)
                    multiple = bf16 if pages.dtype == 'bf16' else fp8
                    if multiple == 0:
                        self.assertEqual(layout, manual[name])
                    else:
                        self.assertEqual(pages.block_bytes,
                                         multiple * (4096 if pages.dtype == 'bf16' else 268320))
                    for index in {0, len(pages) - 1}:
                        raw = pages[index]
                        descriptor = pages.descriptor(index)
                        self.assertEqual(len(raw), descriptor['raw_bytes'])
                        data, scale = pages.split_page(index, raw)
                        self.assertEqual(len(data), descriptor['data_bytes'])
                        self.assertEqual(len(scale), descriptor['scale_bytes'])

    def test_only_selected_datasets_determine_lcm(self):
        self.config['benchmark_datasets'] = {'ds_bf16.pt': '4096'}
        layout = self.resolve(1, 0)['ds_bf16.pt']
        self.assertEqual(layout['page_tokens'], '1')

    def test_integer_validation_and_legacy_precedence(self):
        for bad in ('true', 'false', '-1', '1.5', '1e2', '+2', ''):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_lcm(bad)
        self.assertEqual(parse_lcm('0'), 0)
        self.assertEqual(parse_lcm('32'), 32)
        self.config['benchmark'] = {'lcm': '3', 'lcm_bf16': '0'}
        self.assertEqual(lcm_multipliers(self.config), {'bf16': 0, 'fp8': 3})
        self.config['benchmark']['lcm_fp8'] = '2'
        self.assertEqual(lcm_multipliers(self.config), {'bf16': 0, 'fp8': 2})


if __name__ == '__main__':
    unittest.main()
