"""Dtype-aware, byte-exact GPU diagnostics using production encode/decode paths."""
import json


def diagnostic_cases(folder, plugin):
    path = folder / 'diagnostics.json'
    if path.is_file():
        metadata = json.loads(path.read_text(encoding='utf-8'))
        if metadata.get('schema_version') != 1 or not isinstance(metadata.get('gpu_cases'), list):
            raise ValueError('Invalid GPU diagnostic manifest: ' + str(path))
        cases = metadata['gpu_cases']
        if any(not isinstance(c, dict) or c.get('dtype') not in ('bf16', 'fp8_e4m3fn', 'fp8_e5m2')
               or not isinstance(c.get('options', {}), dict) for c in cases):
            raise ValueError('Invalid GPU diagnostic case')
        if any(not plugin.supports({'dtype': c['dtype']}) for c in cases):
            raise ValueError('Diagnostic dtype is not supported by the plugin')
        return cases
    return [dict(dtype=dtype) for dtype in ('bf16', 'fp8_e4m3fn', 'fp8_e5m2')
            if plugin.supports({'dtype': dtype})]


def verify_case(route, case):
    from codec_timing import Encoder
    from batch_measure import encode_frame
    from batch_decode import BatchDecoder
    import struct
    dtype = case['dtype']
    unit = (struct.pack('<8H', 0, 0x8000, 0x3f80, 0x7fc1, 0x7f80, 0xff80, 1, 0x7fff)
            if dtype == 'bf16' else bytes(range(256)))
    # Exercise both complete chunks and tails, including FP8 super-blocks.
    pages = [unit * n for n in (17, 19, 23)]
    runtime = route.runtime()
    encoder = Encoder(runtime, pages[0])
    joined = b''.join(pages)
    measured = [encoder.measure(raw)[0] for raw in pages[:2]]
    if route.mode.startswith('GPU-'):
        encode_frame(runtime, joined)  # warm the production joined-page path
        frame, _, _ = encode_frame(runtime, joined)
    else:
        frame = encoder.measure(joined)[0]
    frames = [*measured, frame]
    expected = [*pages[:2], joined]
    decoded, _, kind = BatchDecoder(runtime).submit(frames, list(map(len, expected))).finish()
    if decoded != expected:
        raise ValueError('GPU byte-exact round trip failed for ' + dtype)
    decoder = runtime.cpu_decoder()
    if decoder and any(decoder(frame, len(raw)) != raw for frame, raw in zip(frames, expected)):
        raise ValueError('Compatible CPU frame round trip failed for ' + dtype)
    return dict(dtype=dtype, options=case.get('options', {}), status='verified',
                frame_format=route.backend_id, encode_batch_blocks=3,
                decoded_frames=len(frames), verified_bytes=sum(map(len, expected)),
                gpu_decode=kind, cpu_decode='verified' if decoder else 'not_supported')


def check_gpu(name, plugin, folder, directory):
    from benchmark_routes import select_routes
    checks = []
    explicit = (folder / 'diagnostics.json').is_file()
    for case in diagnostic_cases(folder, plugin):
        dtype = case['dtype']
        try:
            routes = select_routes(directory, execution_device='declared', codecs=name,
                                   dtype=dtype, skip_incompatible=True)
            route = routes.get(name)
            if route is None or not route.mode.endswith('-GPU'):
                if explicit:
                    raise ValueError('Declared diagnostic GPU route is unavailable: ' + dtype)
                continue
            route.options = dict(route.options, **case.get('options', {}))
            checks.append(verify_case(route, case))
        except Exception as error:
            checks.append(dict(dtype=dtype, options=case.get('options', {}), status='failed',
                               error=type(error).__name__ + ': ' + str(error)))
    return checks
