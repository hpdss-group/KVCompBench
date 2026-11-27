"""FP8 route declarations, diagnostic coverage and repository source policy."""
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'benchmark/compression_framework')]
from benchmark_routes import select_routes
from core import load_plugins
from gpu_diagnostics import check_gpu, diagnostic_cases
import codec_manager
import tempfile
import os


class FinalRequirements(unittest.TestCase):
    def test_native_inventory_filters_record_before_statting_python_files(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'benchmark/output') as d:
            root = Path(d)
            library = root / 'library.so'
            library.write_bytes(b'library')
            class Distribution:
                def read_text(self, name):
                    return 'missing.py,,\nlibrary.so,,\n'
                def locate_file(self, name):
                    if name == 'missing.py':
                        raise AssertionError('Python source must be filtered before lookup')
                    return root / name
                @property
                def files(self):
                    raise AssertionError('Do not enumerate all installed wheel members')
            self.assertEqual(set(codec_manager.native_files(Distribution())), {'library.so'})

    def test_binary_hash_cache_invalidation_and_forced_recheck(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'benchmark/output') as d:
            root = Path(d)
            (root / 'kvcompbench').mkdir()
            binary = root / 'library.so'
            binary.write_bytes(b'first content')
            with patch.object(codec_manager.sys, 'prefix', str(root)), patch.object(codec_manager, '_HASH_CACHE', None):
                first = codec_manager._binary_digest(str(binary), binary.stat().st_size, binary.stat().st_mtime_ns)
                with patch.object(codec_manager, 'digest', side_effect=AssertionError('unnecessary reread')):
                    self.assertEqual(first, codec_manager._binary_digest(str(binary), binary.stat().st_size, binary.stat().st_mtime_ns))
                binary.write_bytes(b'changed content')
                changed = codec_manager._binary_digest(str(binary), binary.stat().st_size, binary.stat().st_mtime_ns)
                self.assertNotEqual(first, changed)
                with patch.dict(os.environ, KVCOMPBENCH_REHASH='1'), patch.object(codec_manager, 'digest', wraps=codec_manager.digest) as digest:
                    self.assertEqual(changed, codec_manager._binary_digest(str(binary), binary.stat().st_size, binary.stat().st_mtime_ns))
                    digest.assert_called_once()

    def test_fp8_frame_declarations_and_diagnostic_matrix(self):
        folder = ROOT / 'benchmark/compressors/splitzip'
        plugin = load_plugins(folder.parent)['splitzip']
        cases = diagnostic_cases(folder, plugin)
        self.assertEqual([(c['dtype'], c.get('options', {}).get('code_bits')) for c in cases],
                         [('bf16', None), ('fp8_e4m3fn', 2), ('fp8_e4m3fn', 3),
                          ('fp8_e5m2', 2), ('fp8_e5m2', 3), ('fp8_e5m2', 4)])
        for dtype in ('bf16', 'fp8_e4m3fn', 'fp8_e5m2'):
            route = select_routes(folder.parent, execution_device='declared', codecs='splitzip', dtype=dtype)['splitzip']
            self.assertEqual(route.backend_id, 'splitzip-packet-v1' if dtype == 'bf16' else 'splitzip-fp8-packet-v1')
        with patch('gpu_diagnostics.verify_case', side_effect=lambda route, case: dict(status='verified', dtype=route.dtype, bits=route.options.get('code_bits'))) as verify:
            reports = check_gpu('splitzip', plugin, folder, folder.parent)
        self.assertEqual(verify.call_count, 6)
        self.assertEqual(len(reports), 6)

    @unittest.skipUnless(shutil.which('node'), 'Node.js is required for the DOM regression')
    def test_actual_per_cell_batch_labels(self):
        subprocess.run(['node', str(ROOT / 'tests/test_report_batches.js'),
                        str(ROOT / 'benchmark/templates/report.js')], check=True, timeout=30)

    def test_third_party_sources_are_external(self):
        for package in ('alp', 'chimp', 'elf', 'fse', 'splitzip', 'streamvbyte'):
            self.assertFalse((ROOT / 'benchmark/compressors' / package / 'upstream').exists())
        self.assertFalse((ROOT / 'benchmark/compressors/openzl/dependencies').exists())


if __name__ == '__main__': unittest.main()
