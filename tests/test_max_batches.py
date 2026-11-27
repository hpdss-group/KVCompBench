"""Max batch planning without allocating GPU resources."""
from pathlib import Path
import sys,types,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmark/compression_framework'))
from batch_policy import parse_batch,parse_max_bytes,largest_passing,resolve_batches

class Pages:
    block_bytes=4
    def __getitem__(self,i):return bytes([i%256])*4

class Stream:
    def synchronize(self):pass

class Planner(unittest.TestCase):
    def test_parser(self):
        self.assertEqual(parse_batch(' MAX '),'max')
        self.assertEqual(parse_batch('4096'),4096)
        for value in ['0','4097','1.5','bad']:
            with self.assertRaises(ValueError):parse_batch(value)
    def test_byte_cap_parser(self):
        self.assertIsNone(parse_max_bytes(' MAX '))
        self.assertIsNone(parse_max_bytes(None))
        self.assertEqual(parse_max_bytes('536870912'),536870912)
        for value in ['0','-1','bad','1.5']:
            with self.assertRaises(ValueError):parse_max_bytes(value)
    def test_uncapped_gpu_budget(self):
        group,decode,plan=self.plan('GPU-GPU','max','max',max_bytes='max',encode_limit=1000000,decode_limit=1000000,count=20000)
        self.assertEqual(group,15625)
        self.assertEqual(plan['raw_budget_bytes'],62500)
        self.assertEqual(plan['max_raw_bytes'],'max')
        self.assertEqual(self.plan('GPU-GPU','max','max',max_bytes='max')[:2],(20,40))
        self.assertEqual(self.plan('GPU-GPU','max',1,max_bytes=40)[0],10)
    def test_numeric_encoding_is_not_clamped_to_64_mib(self):
        runtime=types.SimpleNamespace(route=types.SimpleNamespace(mode='GPU-GPU'))
        pages=types.SimpleNamespace(block_bytes=1024*1024)
        self.assertEqual(resolve_batches(runtime,pages,10000,1024,128)[:2],(1024,128))
        self.assertEqual(resolve_batches(runtime,pages,17,1024,128)[:2],(17,128))
        pages.block_bytes=268320
        self.assertEqual(resolve_batches(runtime,pages,1290,512,128)[:2],(512,128))

    def test_search(self):
        seen=[]
        self.assertEqual(largest_passing(100,lambda n:seen.append(n) or n<=37),37)
        with self.assertRaises(RuntimeError):largest_passing(10,lambda n:False)
    def test_cpu_and_numeric(self):
        runtime=types.SimpleNamespace(route=types.SimpleNamespace(mode='CPU-CPU'))
        self.assertEqual(resolve_batches(runtime,Pages(),100,'max','max')[:2],(1,100))
        runtime.route.mode='GPU-GPU'
        self.assertEqual(resolve_batches(runtime,Pages(),100,16,32)[:2],(16,32))
    def plan(self,mode,encode,decode,encode_limit=80,decode_limit=160,count=100,**kw):
        class OOM(RuntimeError):pass
        cuda=types.SimpleNamespace(mem_get_info=lambda:(1000000,2000000),empty_cache=lambda:None,OutOfMemoryError=OOM)
        torch=types.SimpleNamespace(cuda=cuda)
        runtime=types.SimpleNamespace(route=types.SimpleNamespace(mode=mode),stream=Stream(),setup=lambda raw:None)
        def encode_frame(runtime,raw):
            if len(raw)>encode_limit:raise OOM('CUDA out of memory')
            return raw,0.0,0.0
        class Decoder:
            def __init__(self,runtime):pass
            def submit(self,frames,sizes):
                if sum(sizes)>decode_limit:raise OOM('CUDA out of memory')
                self.value=(frames,0.0,'mock');return self
            def finish(self):return self.value
        class Encoder:
            def __init__(self,runtime,raw):pass
            def measure(self,raw):return (raw,0,0,0,0)
        with patch.dict(sys.modules,{'torch':torch}),patch('batch_measure.encode_frame',encode_frame),patch('batch_decode.BatchDecoder',Decoder),patch('codec_timing.Encoder',Encoder):
            return resolve_batches(runtime,Pages(),count,encode,decode,max_bytes=kw.pop('max_bytes',1000000),**kw)
    def test_max_oom_and_complete_frames(self):
        group,decode,plan=self.plan('GPU-GPU','max','max')
        self.assertEqual((group,decode),(20,40));self.assertTrue(plan['calibrated'])
    def test_decoder_only_and_encoder_independence(self):
        self.assertEqual(self.plan('GPU-GPU',16,'max')[:2],(16,32))
        self.assertEqual(self.plan('CPU-GPU','max','max')[:2],(1,40))
        self.assertEqual(self.plan('GPU-GPU','max',8)[0],self.plan('GPU-GPU','max',128)[0])
    def test_over_4096(self):
        self.assertEqual(self.plan('GPU-GPU','max','max',encode_limit=1000000,decode_limit=1000000,count=10000)[0],10000)
    def test_whole_final_frame(self):
        self.assertEqual(self.plan('GPU-GPU',16,'max',count=30)[:2],(16,32))
    def test_budget_and_non_oom(self):
        self.assertEqual(self.plan('GPU-GPU','max',1,max_bytes=40)[0],10)
        with self.assertRaises(RuntimeError):self.plan('GPU-GPU','max','max',max_bytes=1)
        def broken(*a):raise RuntimeError('Invalid codec format')
        with patch('batch_measure.encode_frame',broken):
            # Verify errors unrelated to memory are not classified as capacity.
            with self.assertRaisesRegex(RuntimeError,'format'):largest_passing(10,broken)

if __name__=='__main__':unittest.main()
