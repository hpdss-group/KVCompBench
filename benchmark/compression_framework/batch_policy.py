"""Resource-bounded GPU batch selection; calibration is excluded from timing."""
import gc

POLICY_VERSION = 'gpu-max-batch-v1'


def parse_batch(value):
    value = str(value).strip().lower()
    if value == 'max':
        return value
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError('Batch size must be an integer from 1 to 4096, or max') from None
    if not 1 <= number <= 4096:
        raise ValueError('Batch size must be an integer from 1 to 4096, or max')
    return number


def parse_max_bytes(value):
    """None represents no fixed byte cap; GPU memory budgeting still applies."""
    if value is None or str(value).strip().lower() == 'max':
        return None
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError('max_batch_bytes must be a positive integer or max') from None
    if number < 1:
        raise ValueError('max_batch_bytes must be a positive integer or max')
    return number


def largest_passing(limit, attempt):
    """Find largest passing integer for a monotonic resource constraint."""
    low, high = 0, limit
    if attempt(high):
        return high
    while low < high:
        middle = (low + high + 1) // 2
        if attempt(middle):
            low = middle
        else:
            high = middle - 1
    if low < 1:
        raise RuntimeError('Even one complete codec frame exceeds the max batch resource budget')
    return low


def resolve_batches(runtime, pages, count, encode_request, decode_request,
                    max_bytes=512 * 1024 * 1024, memory_fraction=0.5):
    """Resolve max within a host-byte cap and a conservative GPU memory budget.

    Reserve half of currently free VRAM by default, and estimate eight bytes of
    working memory per raw byte before actual calibration. Sampling includes the
    first batch and the final frame. Data-dependent future failures are reported
    rather than silently changing frame layout after checkpoint identities exist.
    """
    max_bytes = parse_max_bytes(max_bytes)
    if not 0 < memory_fraction <= 0.8:
        raise ValueError('Invalid max batch memory fraction')
    # CPU APIs remain single-page; keep their host scheduling window bounded.
    host_window = max_bytes if max_bytes is not None else 512 * 1024 * 1024
    mode = runtime.route.mode
    gpu_encode = mode == 'GPU-GPU'
    gpu_decode = mode.endswith('-GPU')
    auto_encode = encode_request == 'max' and gpu_encode
    auto_decode = decode_request == 'max' and gpu_decode
    group = max(1, min(encode_request, count)) if gpu_encode and encode_request != 'max' else 1
    decode = decode_request if decode_request != 'max' else (max(1, min(count, host_window // (8 * pages.block_bytes))) if not gpu_decode else 1)
    plan = dict(policy=POLICY_VERSION, encode_requested=encode_request,
                decode_requested=decode_request, encode_blocks=group,
                decode_blocks=decode, calibrated=False)
    if not (auto_encode or auto_decode):
        return group, decode, plan
    import torch
    from batch_measure import encode_frame
    from batch_decode import BatchDecoder
    from codec_timing import Encoder
    runtime.setup(pages[0])
    runtime.stream.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    budget = int(free * memory_fraction) // 8
    if max_bytes is not None:
        budget = min(max_bytes, budget)
    if budget < pages.block_bytes:
        raise RuntimeError('Insufficient max batch budget for one full logical page')
    page_limit = min(count, budget // pages.block_bytes)
    plan.update(free_vram_bytes=free, total_vram_bytes=total,
                memory_fraction=memory_fraction, working_memory_factor=8,
                max_raw_bytes=max_bytes if max_bytes is not None else 'max', raw_budget_bytes=budget)

    def attempt(function):
        try:
            function()
            runtime.stream.synchronize()
            return True
        except RuntimeError as error:
            if not isinstance(error, torch.cuda.OutOfMemoryError) and 'out of memory' not in str(error).lower():
                raise
            return False
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    if auto_encode:
        def trial_encode(number):
            raw = b''.join(pages[i] for i in range(number))
            frame, _, _ = encode_frame(runtime, raw)
            decoded, _, _ = BatchDecoder(runtime).submit([frame], [len(raw)]).finish()
            if decoded != [raw]:
                raise RuntimeError('Max encode calibration failed lossless verification')
        group = largest_passing(page_limit, lambda number: attempt(lambda: trial_encode(number)))
        # Also check the final encoded frame, which can have a different shape.
        tail_start = ((count - 1) // group) * group
        if tail_start:
            def trial_tail():
                raw = b''.join(pages[i] for i in range(tail_start, count))
                frame, _, _ = encode_frame(runtime, raw)
                decoded, _, _ = BatchDecoder(runtime).submit([frame], [len(raw)]).finish()
                if decoded != [raw]:
                    raise RuntimeError('Max tail calibration failed lossless verification')
            if not attempt(trial_tail):
                raise RuntimeError('Max tail calibration ran out of memory')
    if auto_decode:
        # Whole encoded frames are indivisible, even if a numeric decode request
        # is smaller than the encoder frame. Probe in units of complete frames.
        frame_limit = max(1, min((count + group - 1) // group, budget // (pages.block_bytes * group)))
        if group * pages.block_bytes > budget:
            raise RuntimeError('One encoder frame exceeds max decode budget; reduce encoding batch')
        cpu_encoder = Encoder(runtime, pages[0]) if not gpu_encode else None
        def trial_decode(number):
            raws, frames = [], []
            for start in range(0, min(number * group, count), group):
                raw = b''.join(pages[i] for i in range(start, min(start + group, count)))
                frame = encode_frame(runtime, raw)[0] if gpu_encode else cpu_encoder.measure(raw)[0]
                raws.append(raw)
                frames.append(frame)
            restored, _, _ = BatchDecoder(runtime).submit(frames, list(map(len, raws))).finish()
            if restored != raws:
                raise RuntimeError('Max decode calibration failed lossless verification')
        frames = largest_passing(frame_limit, lambda number: attempt(lambda: trial_decode(number)))
        decode = frames * group
        # Scheduling uses floor(decode/group); retain a complete final frame.
        if decode < group:
            decode = group
    plan.update(encode_blocks=group, decode_blocks=decode, calibrated=True)
    return group, decode, plan
