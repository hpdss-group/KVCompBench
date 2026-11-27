"""Installation trust boundaries and measurement reuse regression checks."""
import copy
import importlib.util
import json
import hashlib
from pathlib import Path
import sys
import tempfile
import subprocess
import tarfile
import io
import types
import unittest
from unittest.mock import patch
import zlib

ROOT = Path(__file__).resolve().parents[1]
(ROOT / 'benchmark/output').mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(ROOT / 'benchmark'), str(ROOT / 'benchmark/compression_framework')]
import measurement_identity as identity
import codec_manager
import batch_measure


class Fingerprints(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'benchmark/output')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        framework = self.root / 'benchmark/compression_framework'
        framework.mkdir(parents=True)
        (framework / 'kv_pages.py').write_text('def page(): return b"page"\n')
        folder = self.root / 'benchmark/compressors/demo'
        folder.mkdir(parents=True)
        (folder / 'codec.json').write_text(json.dumps({'codecs': [{'name': 'demo', 'kind': 'python', 'compress': 'encode', 'decompress': 'decode'}]}))
        self.backend = folder / 'api.py'
        self.source = 'def encode(x): return x\ndef decode(x): return x\n'
        self.backend.write_text(self.source)
        self.route = types.SimpleNamespace(folder=folder, name='demo', mode='GPU-GPU', options={'execution_path': 'GPU-GPU', 'frame_format': 'demo-v1'})
        self.provenance = dict(input_sha256='data', shape=[32], storage_dtype='bf16',
            page_format='tensor', layout={'layout': 'flat'}, validation=False, workers=1,
            environment={'python': 'test', 'device': 'GPU'},
            installation={'revision': 'pinned', 'prefix': '/old', 'imports': {'demo': {'path': '/old/demo', 'sha256': 'same'}}, 'artifacts': {'lib': {'path': '/old/lib', 'sha256': 'binary'}}})

    def layers(self):
        return identity.layered_spec(self.root, self.route, self.provenance, 4, 2, 2)

    def test_decoder_change_preserves_encoder(self):
        old = self.layers()
        self.backend.write_text(self.source.replace('decode(x): return x', 'decode(x): return bytes(x)'))
        new = self.layers()
        self.assertEqual(old['encode'], new['encode'])
        self.assertNotEqual(identity.decoder_job(old, 'GPU-GPU', 4), identity.decoder_job(new, 'GPU-GPU', 4))

    def test_encode_change_invalidates_both(self):
        old = self.layers()
        self.backend.write_text(self.source.replace('encode(x): return x', 'encode(x): return bytes(x)'))
        new = self.layers()
        self.assertNotEqual(old['encode'], new['encode'])
        self.assertNotEqual(identity.decoder_job(old, 'GPU-GPU', 4), identity.decoder_job(new, 'GPU-GPU', 4))

    def test_documentation_logging_and_relocation(self):
        old = self.layers()
        self.backend.write_text('"""New documentation."""\n# comment\n' + self.source.replace('return x', 'print("progress"); return x'))
        self.provenance['installation']['prefix'] = '/new'
        self.provenance['installation']['imports']['demo']['path'] = '/new/demo'
        self.provenance['installation']['artifacts']['lib']['path'] = '/new/lib'
        self.assertEqual(old, self.layers())
        self.provenance['installation']['artifacts']['lib']['sha256'] = 'different'
        self.assertNotEqual(old['encode'], self.layers()['encode'])

    def test_selected_max_frame_size_changes_encoder_identity(self):
        old = identity.layered_spec(self.root, self.route, self.provenance, 4, 2, 2)
        new = identity.layered_spec(self.root, self.route, self.provenance, 4, 3, 3)
        self.assertNotEqual(old['encode'], new['encode'])
        self.assertNotEqual(identity.decoder_job(old, 'GPU-GPU', 4), identity.decoder_job(new, 'GPU-GPU', 4))

    def test_decode_batch_and_protocol(self):
        layers = self.layers()
        self.assertNotEqual(identity.decoder_job(layers, 'GPU-GPU', 4), identity.decoder_job(layers, 'GPU-GPU', 8))
        self.assertEqual(identity.decoder_job(layers, 'GPU-CPU', 4), identity.decoder_job(layers, 'GPU-CPU', 8))
        with patch.dict(identity.PROTOCOL, decode='new-decode-timing'):
            new = self.layers()
        self.assertEqual(layers['encode'], new['encode'])
        self.assertNotEqual(layers['decode'], new['decode'])

    def test_data_and_unrelated_codec(self):
        old = self.layers()
        other = self.route.folder.parent / 'other'
        other.mkdir()
        (other / 'api.py').write_text('def encode(x): return b"different"')
        self.assertEqual(old, self.layers())
        self.provenance['input_sha256'] = 'changed-input'
        self.assertNotEqual(old['encode'], self.layers()['encode'])

    def test_decoder_declaration_does_not_change_encoder(self):
        old = self.layers()
        (self.route.folder / 'decode_backends.json').write_text(json.dumps({'demo': {'cpu': {'file': 'cpu.py', 'function': 'decode'}}}))
        (self.route.folder / 'cpu.py').write_text('def decode(x,n): return x\n')
        self.assertEqual(old['encode'], self.layers()['encode'])
        self.assertNotEqual(old['decode'], self.layers()['decode'])


class Installation(unittest.TestCase):
    def test_offline_managed_install_and_receipt_reuse(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'benchmark/output') as d:
            root = Path(d)
            folder = root / 'benchmark/compressors/demo'
            folder.mkdir(parents=True)
            (folder / 'codec.json').write_text(json.dumps({'codecs': [{'name': 'demo'}]}))
            (root / 'benchmark/settings.example.txt').write_text('[environment]\nname=unset\n[benchmark]\ncodecs=demo\n')
            source = b'pinned upstream bytes\n'
            archive = root / 'upstream.tar.gz'
            with tarfile.open(archive, 'w:gz') as target:
                member = tarfile.TarInfo('upstream/source.txt')
                member.size = len(source)
                target.addfile(member, io.BytesIO(source))
            pin = hashlib.sha256(archive.read_bytes()).hexdigest()
            manifest = dict(schema_version=1, package='demo', revision='fixed', requirements=[], imports=['builtins'],
                sources=dict(archives={'demo': dict(url='https://offline.invalid/source', sha256=pin)},
                             files={'source.txt': dict(archive='demo', member='upstream/source.txt', sha256=hashlib.sha256(source).hexdigest())}),
                directories=['native/demo'], outputs=['native/demo/artifact.bin'],
                commands=[['{python}', '-c', 'from pathlib import Path; Path(r"{prefix}/native/demo/artifact.bin").write_bytes(Path(r"{source}/source.txt").read_bytes())']])
            (folder / 'install.json').write_text(json.dumps(manifest))
            # Prepopulate the checksum cache: exercises fetch/extract/build without networking.
            prefix = root / '.venvs' / (codec_manager.platform_id() + '-managed-cpu')
            cache = prefix / 'kvcompbench/downloads'
            cache.mkdir(parents=True)
            (cache / pin).write_bytes(archive.read_bytes())
            with patch.object(codec_manager, 'ROOT', root), patch.object(codec_manager, 'PACKAGES', folder.parent), patch.dict(codec_manager.os.environ):
                previous_tempdir = tempfile.tempdir
                try:
                    result = codec_manager.install('demo', base=False)
                    artifact = prefix / 'native/demo/artifact.bin'
                    self.assertEqual(artifact.read_bytes(), source)
                    modified = artifact.stat().st_mtime_ns
                    codec_manager.install('demo', base=False)
                    self.assertEqual(modified, artifact.stat().st_mtime_ns)
                    receipt = json.loads((prefix / 'kvcompbench/receipts/demo.json').read_text())
                    self.assertTrue(receipt['complete'] and receipt['reused_build'])
                    probe = ('import sys,json; from pathlib import Path; sys.path.insert(0,' + repr(str(ROOT / 'benchmark/compression_framework')) + '); '
                             'import codec_manager; print(json.dumps(codec_manager.identity("demo",directory=Path(' + repr(str(folder.parent)) + '))))')
                    report = json.loads(subprocess.check_output([result['python'], '-B', '-c', probe], text=True, timeout=30))
                    self.assertEqual(report['status'], 'ready')
                    artifact.write_bytes(b'corrupt')
                    failed_probe = probe.replace('identity("demo",', 'identity("demo",enforce=False,')
                    report = json.loads(subprocess.check_output([result['python'], '-B', '-c', failed_probe], text=True, timeout=30))
                    self.assertEqual(report['status'], 'unavailable')
                finally:
                    tempfile.tempdir = previous_tempdir

    def test_mirror_change_is_not_build_change(self):
        manifest = {'revision': 'fixed', 'sources': {'archives': {'upstream': {'url': 'https://one', 'sha256': 'pin'}}}}
        old = codec_manager.recipe_fingerprint(manifest)
        manifest['sources']['archives']['upstream']['url'] = 'https://mirror'
        self.assertEqual(old, codec_manager.recipe_fingerprint(manifest))
        manifest['sources']['archives']['upstream']['sha256'] = 'different'
        self.assertNotEqual(old, codec_manager.recipe_fingerprint(manifest))

    def test_inventory_and_stdlib_doctor(self):
        self.assertGreaterEqual(len(codec_manager.inventory()), 20)
        reports = codec_manager.diagnose('identity,zlib,bz2,lzma', correctness=True)
        self.assertTrue(all(r['status'] == 'ready' and r['cpu'] == 'verified' for r in reports))

    def test_reject_unpinned_download_and_traversal(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'benchmark/output') as d:
            with self.assertRaises(ValueError):
                codec_manager.download({'url': 'http://bad', 'sha256': 'missing'}, d)
            with self.assertRaises(ValueError):
                codec_manager.contained(d, '../escape')


class Pages:
    block_bytes = 4
    dataset_metadata = {}
    dtype = 'bf16'
    page_format = 'test'
    quantization_origin = 'test'

    def __getitem__(self, index): return bytes([index]) * 4
    def descriptor(self, index): return dict(data_bytes=4, scale_bytes=0)


class Runtime:
    stream = types.SimpleNamespace(synchronize=lambda: None)
    def setup(self, sample): pass
    def cpu_decoder(self): return lambda frame, n: zlib.decompress(frame)


class GPU:
    page_seconds = .002
    def __init__(self, fail=False, bad=False): self.calls = 0; self.fail = fail; self.bad = bad
    def submit(self, frames, sizes):
        self.calls += 1
        if self.fail and self.calls == 2: raise RuntimeError('interrupted')
        output = [zlib.decompress(f) for f in frames]
        if self.bad: output[0] += b'corrupt'
        return types.SimpleNamespace(finish=lambda: (output, .001, 'mock-gpu'))


class Resume(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'benchmark/output')
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.encode = patch.object(batch_measure, 'encode_frame', side_effect=lambda runtime, raw: (zlib.compress(raw), .001, .002)).start()
        self.addCleanup(patch.stopall)
        self.route = types.SimpleNamespace(backend_id='mock')

    def run_codec(self, jobs=None, gpu=None, batch=4, retain=True):
        return batch_measure.run_batched_codec(self.out, self.route, Runtime(), Pages(), 7,
            'mock', 'test.pt', {'layout': 'flat'}, jobs or {'GPU-GPU': 'gpu', 'GPU-CPU': 'cpu'},
            'encoder', ['GPU-GPU', 'GPU-CPU'], gpu or GPU(), batch, 2,
            validation_only=True, retain_frames=retain)

    def test_interruption_resume_and_decoder_only(self):
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            self.run_codec(gpu=GPU(fail=True))
        self.encode.reset_mock()
        rows = self.run_codec()
        self.encode.assert_not_called()
        self.assertTrue(all(r['verified_blocks'] == 7 and r['raw_bytes'] == 28 and r['validation_only'] for r in rows))
        gpu = GPU()
        self.assertEqual(rows, self.run_codec(gpu=gpu))
        self.assertEqual(gpu.calls, 0)
        self.encode.assert_not_called()
        new = self.run_codec(jobs={'GPU-GPU': 'gpu-v2', 'GPU-CPU': 'cpu'}, batch=2)
        self.encode.assert_not_called()
        self.assertEqual(rows[0]['encode_mib_s'], new[0]['encode_mib_s'])

    def test_corrupt_frame_and_tail_recovery(self):
        self.run_codec()
        journal = self.out / 'batch-progress.jsonl'
        with journal.open('ab') as handle: handle.write(b'{"partial"')
        self.run_codec()
        self.assertTrue(list(self.out.glob('*.tail.*')))
        (self.out / '.frames/encoder/0.bin').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            self.run_codec(jobs={'GPU-GPU': 'new-gpu', 'GPU-CPU': 'cpu'})

    def test_bad_decode_is_not_checkpointed(self):
        with self.assertRaisesRegex(RuntimeError, 'Lossless verification failed'):
            self.run_codec(gpu=GPU(bad=True))
        entries = [json.loads(line) for line in (self.out / 'batch-progress.jsonl').read_text().splitlines()]
        self.assertTrue(all(entry['type'] == 'encode' for entry in entries))

    def test_discard_frames_and_recreate(self):
        rows = self.run_codec(retain=False)
        self.assertFalse(list((self.out / '.frames/encoder').glob('*.bin')))
        self.encode.reset_mock()
        new = self.run_codec(jobs={'GPU-GPU': 'new-gpu', 'GPU-CPU': 'cpu'})
        self.assertGreater(self.encode.call_count, 0)
        self.assertEqual(rows[0]['encode_mib_s'], new[0]['encode_mib_s'])


if __name__ == '__main__': unittest.main()
