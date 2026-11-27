"""Official SplitZip Triton codec with explicit, independently decodable packets."""
from functools import lru_cache
import importlib.util
import os
from pathlib import Path
import sys
from adapters.paper_support import packet,unpacket

FP8_FORMATS={'fp8_e4m3fn':'e4m3','fp8_e5m2':'e5m2'}
DEFAULT_CODE_BITS=3
FP8_CHUNK=1024
FP8_SUPER=8


@lru_cache(maxsize=1)
def upstream():
    from codec_manager import source_file
    path=source_file('splitzip','codec_gpu.py')
    spec=importlib.util.spec_from_file_location('kv_official_splitzip',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def upstream_fp8():
    from codec_manager import source_file
    path=source_file('splitzip','codec_fp8_gpu.py')
    spec=importlib.util.spec_from_file_location('kv_official_splitzip_fp8',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def runtime():
    if not os.environ.get('CUDA_VISIBLE_DEVICES','').strip():raise RuntimeError('An administrator-assigned CUDA_VISIBLE_DEVICES is required')
    cache=Path(sys.prefix)/'tmp'
    for key,folder in [('TRITON_CACHE_DIR','triton'),('CUDA_CACHE_PATH','cuda'),('TORCHINDUCTOR_CACHE_DIR','torchinductor')]:
        path=cache/folder;path.mkdir(parents=True,exist_ok=True);os.environ.setdefault(key,str(path))
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable in the selected environment')
    return torch


def tensor_bytes(value):return value.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes()


def code_bits(metadata):
    bits=int(metadata.get('code_bits',DEFAULT_CODE_BITS))
    if bits not in (2,3,4):raise ValueError('splitzip code_bits must be 2, 3 or 4')
    return bits


def fp8_codec(metadata):
    dtype=metadata.get('dtype')
    if dtype not in FP8_FORMATS:raise ValueError(f'SplitZip FP8 requires an fp8 dtype, got {dtype!r}')
    fmt=FP8_FORMATS[dtype];bits=code_bits(metadata)
    if fmt=='e4m3' and bits==4:raise ValueError('4-bit codes do not apply to the 4-bit E4M3 exponent')
    module=upstream_fp8()
    return module,module.SplitZipFP8GPU(fmt,bits,device='cuda:0'),fmt,bits


def restore_state(codec,members):
    """Rebuild the calibrated codec state that upstream calibrate() would have produced."""
    members=[int(v) for v in members]
    if len(members)!=codec.K or len(set(members))!=codec.K:raise ValueError('Invalid SplitZip FP8 codebook')
    codec.codebook=members
    codec.window=(members==list(range(members[0],members[0]+codec.K))) and not codec.force_lut
    codec.base=members[0]
    rep=lambda v:int.from_bytes(bytes([v])*8,'little')
    codec.base_vec=rep(codec.base);codec.hi_mask=rep(0xFF^(codec.K-1));codec.lo_mask=rep(codec.K-1)
    torch=runtime()
    enc=torch.full((32,),0x80,dtype=torch.uint8);dec=torch.zeros(256,dtype=torch.uint8)
    for code,val in enumerate(members):enc[val]=code;dec[code]=val
    codec.enc_lut=enc.to(codec.device);codec.dec_lut=dec.to(codec.device)


def nch_for(n):return (n//(FP8_CHUNK*FP8_SUPER))*FP8_SUPER


def compress(raw,metadata):
    dtype=metadata.get('dtype')
    if dtype in FP8_FORMATS:return compress_fp8(raw,metadata)
    if dtype!='bf16' or len(raw)%2:raise ValueError('SplitZip requires native BF16 bytes')
    torch=runtime();module=upstream();chunk=int(metadata.get('splitzip_chunk_size',1024))
    codec=module.ChunkLocalSplitZipGPU(device='cuda:0',chunk_size=chunk)
    tensor=torch.frombuffer(bytearray(raw),dtype=torch.bfloat16).to('cuda:0')
    # The calibration pass is part of compression and is explicitly labelled.
    coverage=codec.calibrate(tensor)
    encoded=codec.encode(tensor)
    book=codec.dec_lut.cpu().tolist()
    header=dict(format='splitzip-packet-v1',n=encoded.n,n_esc=encoded.n_esc,chunk_size=encoded.chunk_size,
                codebook=book,calibration='per-block official calibrate; included in encode',coverage=coverage)
    return packet(header,[tensor_bytes(getattr(encoded,k)) for k in ('pk','sm','chunk_id','local_pos','esc_val')])


def compress_fp8(raw,metadata):
    torch=runtime();module,codec,fmt,bits=fp8_codec(metadata)
    tensor=torch.frombuffer(bytearray(raw),dtype=torch.uint8).to('cuda:0')
    coverage=codec.calibrate(tensor)
    encoded=codec.encode(tensor)
    parts=[tensor_bytes(encoded.p32)]
    if encoded.p16 is not None:parts.append(tensor_bytes(encoded.p16))
    if encoded.p8 is not None:parts.append(tensor_bytes(encoded.p8))
    parts.append(tensor_bytes(encoded.counts.to(torch.int16)))
    parts.append(tensor_bytes(encoded.esc))
    parts.append(tensor_bytes(encoded.tail))
    header=dict(format='splitzip-fp8-packet-v1',fmt=fmt,code_bits=bits,n=encoded.n,n_esc=encoded.n_esc,
                codebook=list(codec.codebook),p16=encoded.p16 is not None,p8=encoded.p8 is not None,
                calibration='per-frame official calibrate; included in encode',coverage=coverage)
    return packet(header,parts)


def decode_packet(data, expected_bytes):
    """Parse a SplitZip frame against its caller-supplied output byte count.

    BF16 escape offsets can occupy up to twice the raw byte count. FP8 planes
    and escapes also fit this bound. Exact format dimensions are checked below;
    the shared parser still checks header limits, integer lengths and bounds.
    """
    if type(expected_bytes) is not int or expected_bytes < 0:
        raise ValueError('Invalid SplitZip expected output size')
    try:
        header, parts = unpacket(data, maximum=2 * expected_bytes)
    except (KeyError, TypeError) as error:
        raise ValueError('Invalid SplitZip packet header') from error
    fmt = header.get('format')
    n, esc = header.get('n'), header.get('n_esc')
    if type(n) is not int or type(esc) is not int or not 0 <= esc <= n:
        raise ValueError('Invalid SplitZip dimensions')
    if fmt == 'splitzip-packet-v1':
        chunk = header.get('chunk_size')
        if n * 2 != expected_bytes or type(chunk) is not int or chunk < 1:
            raise ValueError('Invalid SplitZip dimensions')
        lengths = [(n + 1) // 2, n, 4 * esc, 2 * esc, esc]
    elif fmt == 'splitzip-fp8-packet-v1':
        bits = header.get('code_bits')
        if n != expected_bytes or header.get('fmt') not in ('e4m3', 'e5m2') or type(bits) is not int or bits not in (2, 3, 4):
            raise ValueError('Invalid SplitZip FP8 dimensions')
        if type(header.get('p16')) is not bool or type(header.get('p8')) is not bool:
            raise ValueError('Invalid SplitZip FP8 planes')
        nch = nch_for(n)
        lanes = nch * (FP8_CHUNK // 8)
        lengths = [lanes * 4]
        if header['p16']:
            lengths.append(lanes * 2)
        if header['p8']:
            lengths.append(lanes)
        lengths += [nch * 2, esc * 2, n - nch * FP8_CHUNK]
    else:
        raise ValueError('Invalid SplitZip packet format')
    if [len(part) for part in parts] != lengths:
        raise ValueError('Invalid SplitZip buffer sizes')
    return header, parts


def decompress(data,expected_bytes,metadata):
    header,_=decode_packet(data,expected_bytes)
    if header.get('format')=='splitzip-fp8-packet-v1':return decompress_fp8(data,expected_bytes,metadata)
    torch=runtime();module=upstream();header,parts=decode_packet(data,expected_bytes)
    n,esc,chunk=header['n'],header['n_esc'],header['chunk_size']
    if header['format']!='splitzip-packet-v1' or type(n)is not int or n*2!=expected_bytes or type(esc)is not int or not 0<=esc<=n:
        raise ValueError('Invalid SplitZip dimensions')
    if len(parts)!=5 or [len(b) for b in parts]!=[(n+1)//2,n,4*esc,2*esc,esc]:raise ValueError('Invalid SplitZip buffer sizes')
    book=header['codebook']
    if len(book)!=16 or any(type(x)is not int or not 0<=x<=255 for x in book):raise ValueError('Invalid SplitZip codebook')
    codec=module.ChunkLocalSplitZipGPU(device='cuda:0',chunk_size=chunk)
    codec.dec_lut=torch.tensor(book,dtype=torch.uint8,device='cuda:0')
    # Decoder uses dec_lut only; initialize the remaining documented ready-state arrays.
    codec.enc_lut=torch.zeros(256,dtype=torch.uint8,device='cuda:0')
    codec.enc_lut_marked=torch.full((256,),16,dtype=torch.uint8,device='cuda:0')
    codec.common_lut=torch.zeros(256,dtype=torch.uint8,device='cuda:0')
    def restore(b,dtype):
        return torch.frombuffer(bytearray(b),dtype=dtype).to('cuda:0') if b else torch.empty(0,dtype=dtype,device='cuda:0')
    pk,sm,chunk_id,local_pos,esc_val=[restore(b,d) for b,d in zip(parts,[torch.uint8,torch.uint8,torch.int32,torch.uint16,torch.uint8])]
    encoded=module.ChunkLocalGPUEncoded(pk,sm,torch.empty(0,dtype=torch.int32,device='cuda:0'),torch.empty(0,dtype=torch.int64,device='cuda:0'),chunk_id,local_pos,esc_val,n,esc,chunk)
    return tensor_bytes(codec.decode(encoded))


def decompress_fp8(data,expected_bytes,metadata):
    torch=runtime();module=upstream_fp8();header,parts=decode_packet(data,expected_bytes)
    fmt,bits,n,nesc=header.get('fmt'),header.get('code_bits'),header.get('n'),header.get('n_esc')
    has16,has8=bool(header.get('p16')),bool(header.get('p8'))
    if header.get('format')!='splitzip-fp8-packet-v1' or fmt not in ('e4m3','e5m2') or bits not in (2,3,4):
        raise ValueError('Invalid SplitZip FP8 header')
    if type(n)is not int or n!=expected_bytes or type(nesc)is not int or not 0<=nesc<=n:
        raise ValueError('Invalid SplitZip FP8 dimensions')
    nch=nch_for(n);lanes=nch*(FP8_CHUNK//8);body=nch*FP8_CHUNK
    if len(parts)!=4+int(has16)+int(has8):raise ValueError('Invalid SplitZip FP8 buffer count')
    expect=[lanes*4]
    if has16:expect.append(lanes*2)
    if has8:expect.append(lanes)
    expect+= [nch*2,nesc*2,n-body]
    if [len(b) for b in parts]!=expect:raise ValueError('Invalid SplitZip FP8 buffer sizes')
    codec=module.SplitZipFP8GPU(fmt,bits,device='cuda:0')
    restore_state(codec,header['codebook'])
    def restore(b,dtype):
        return torch.frombuffer(bytearray(b),dtype=dtype).to('cuda:0') if b else torch.empty(0,dtype=dtype,device='cuda:0')
    i=0
    p32=restore(parts[i],torch.int32);i+=1
    p16=restore(parts[i],torch.int16) if has16 else None;i+=int(has16)
    p8=restore(parts[i],torch.uint8) if has8 else None;i+=int(has8)
    counts=restore(parts[i],torch.int16).to(torch.int32);i+=1
    esc=restore(parts[i],torch.int16);i+=1
    tail=restore(parts[i],torch.uint8)
    starts=(torch.cumsum(counts,0)-counts).to(torch.int32) if counts.numel() else torch.empty(0,dtype=torch.int32,device='cuda:0')
    encoded=module.FP8Encoded(p32,p16,p8,counts,starts,esc,tail,n,nesc)
    return tensor_bytes(codec.decode(encoded))
