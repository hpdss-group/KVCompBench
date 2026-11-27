# alp

Registered codecs: alp.

Source: https://github.com/cwida/ALP

Pinned source/format identity is recorded in codec.json and upstream.json where applicable. The API shim calls the upstream implementation; bridge framing is project-specific and is included in stored-byte accounting.

Install dependencies with `bash python3 kvbench.py setup --optional`. See `benchmark/compression_framework/PAPER_CODECS.md` for supported dtypes, framing, fallback behavior, GPU calibration, and validation status.
