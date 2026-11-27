"""SplitZip large-frame envelope and corruption rejection, without CUDA."""
from pathlib import Path
import sys,unittest,types,struct,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmark/compression_framework'))
from benchmark_routes import module_at
from adapters.paper_support import packet,unpacket
api=module_at(ROOT/'benchmark/compressors/splitzip/api.py')
backend=module_at(ROOT/'benchmark/compressors/splitzip/benchmark_backend.py')

class LargePackets(unittest.TestCase):
    def bf16(self,n,esc=0):
        h=dict(format='splitzip-packet-v1',n=n,n_esc=esc,chunk_size=1024,codebook=list(range(16)))
        parts=[b'p'*((n+1)//2),b's'*n,b'c'*(4*esc),b'l'*(2*esc),b'e'*esc]
        return h,parts
    def fp8(self,n,p16=False,p8=False):
        h=dict(format='splitzip-fp8-packet-v1',fmt='e4m3',code_bits=3,n=n,n_esc=0,p16=p16,p8=p8,codebook=[0,1,2,3,4,5,6])
        nch=api.nch_for(n);lanes=nch*128;parts=[b'p'*(lanes*4)]
        if p16:parts.append(b'q'*(lanes*2))
        if p8:parts.append(b'r'*lanes)
        parts += [b'c'*(nch*2),b'',b't'*(n-nch*1024)]
        return h,parts
    def test_bf16_segment_over_64_mib(self):
        n=64*1024*1024+1;h,parts=self.bf16(n);data=packet(h,parts)
        with self.assertRaisesRegex(ValueError,'segment'):unpacket(data)
        header,restored=api.decode_packet(data,n*2)
        self.assertEqual(restored,parts);self.assertEqual(header['n'],n)
    def test_fp8_segment_over_64_mib(self):
        n=(64*1024*1024+4096)*2;h,parts=self.fp8(n);data=packet(h,parts)
        with self.assertRaisesRegex(ValueError,'segment'):unpacket(data)
        self.assertEqual(api.decode_packet(data,n)[1],parts)
    def test_small_formats_and_planes(self):
        h,parts=self.bf16(32,7);self.assertEqual(api.decode_packet(packet(h,parts),64)[1],parts)
        for planes in [(False,False),(True,False),(False,True),(True,True)]:
            h,parts=self.fp8(8193,*planes);self.assertEqual(api.decode_packet(packet(h,parts),8193)[1],parts)
    def test_reject_dimensions_and_lengths(self):
        h,parts=self.bf16(32)
        for size in [0,63,65,-1,True,64.0]:
            with self.assertRaises(ValueError):api.decode_packet(packet(h,parts),size)
        for update in [dict(n=32.0),dict(n_esc=True),dict(n_esc=33),dict(chunk_size=0),dict(format='other')]:
            with self.assertRaises(ValueError):api.decode_packet(packet(dict(h,**update),parts),64)
        with self.assertRaises(ValueError):api.decode_packet(packet(h,[parts[0],parts[1]+b'x',*parts[2:]]),64)
        h,parts=self.fp8(8192)
        for update in [dict(p16='yes'),dict(n=8193),dict(code_bits=True),dict(fmt='bad')]:
            with self.assertRaises(ValueError):api.decode_packet(packet(dict(h,**update),parts),8192)
    def test_reject_corruption(self):
        h,parts=self.bf16(32);data=packet(h,parts)
        for value in [b'',data[:-1],data+b'x',struct.pack('<I',1024*1024+1)+data[4:]]:
            with self.assertRaises(ValueError):api.decode_packet(value,64)
        header=dict(h,lengths=[-1,32,0,0,0]);raw=json.dumps(header).encode();bad=struct.pack('<I',len(raw))+raw+b''.join(parts)
        with self.assertRaises(ValueError):api.decode_packet(bad,64)
    def test_benchmark_paths_validate_before_gpu_work(self):
        obj=backend.Backend.__new__(backend.Backend);obj.api=api;obj.torch=types.SimpleNamespace()
        for fp8 in [False,True]:
            obj.fp8=fp8;h,parts=self.fp8(8192) if fp8 else self.bf16(32)
            with self.assertRaises(ValueError):obj.prepare_decode([packet(h,parts)],[1])
        with self.assertRaisesRegex(ValueError,'count'):obj.prepare_decode([b''],[])

if __name__=='__main__':unittest.main()
