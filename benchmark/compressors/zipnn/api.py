"""Official ZipNN byte API; retain defensive copies for native buffer mutation."""
from functools import lru_cache
import importlib.metadata

@lru_cache(maxsize=8)
def engine(dtype):
    version=importlib.metadata.version('zipnn')
    if version=='0.5.4':raise ImportError('Use the sentinel-fixed 0.5.4+sentinel1 build supplied by python3 kvbench.py setup --optional')
    from zipnn import ZipNN
    return ZipNN(method='HUFFMAN',input_format='byte',bytearray_dtype={'bf16':'bfloat16','fp16':'float16','fp32':'float32'}[dtype],threads=1,lossy_compressed_type=0)


def compress(raw,metadata):return bytes(engine(metadata['dtype']).compress(bytearray(raw)))


def decompress(data,expected_bytes,metadata):return bytes(engine(metadata['dtype']).decompress(bytearray(data)))
