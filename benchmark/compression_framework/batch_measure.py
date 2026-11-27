"""GPU joined-page encoding with independent encode/decode checkpoints."""
import hashlib
import json
import os
from pathlib import Path
import time

from codec_timing import gpu_call, TIMING_VERSION


def encode_frame(runtime, raw):
    started = time.perf_counter()
    prepared = runtime.prepare_encode(raw)
    runtime.stream.synchronize()
    value, seconds = gpu_call(lambda: runtime.encode(prepared), runtime.stream)
    frame = runtime.finish_encode(value, len(raw))
    return frame, seconds, time.perf_counter() - started


def decode_frames(runtime, gpu, mode, frames, raws, warmed):
    if mode.endswith('-GPU'):
        restored, seconds, kind = gpu.submit(frames, list(map(len, raws))).finish()
        page_seconds = gpu.page_seconds
    else:
        decoder = runtime.cpu_decoder()
        if decoder is None:
            raise RuntimeError('No compatible CPU decoder')
        restored = []
        seconds = 0
        for frame, raw in zip(frames, raws):
            if len(raw) not in warmed:
                if decoder(frame, len(raw)) != raw:
                    raise RuntimeError('CPU warmup verification failed')
                warmed.add(len(raw))
            started = time.perf_counter()
            restored.append(decoder(frame, len(raw)))
            seconds += time.perf_counter() - started
        page_seconds = seconds
        kind = 'cpu-single'
    if len(restored) != len(raws) or any(actual != raw for actual, raw in zip(restored, raws)):
        raise RuntimeError('Lossless verification failed: ' + mode)
    return seconds, page_seconds, kind


def save_frame(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as output:
        output.write(frame)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def run_batched_codec(out, route, runtime, pages, count, name, filename, layout,
                      jobs, encjob, modes, gpu, decode_batch, encode_batch,
                      setup_seconds=0, validation_only=False, retain_frames=True, group_blocks=None):
    # Import from the runner instance rather than invoking its main entry point.
    from run_full_cpu_encode_gpu_decode import load_journal, append
    journal = out / 'batch-progress.jsonl'
    done, encodes = load_journal(journal)
    group = group_blocks if group_blocks is not None else max(1, min(encode_batch, count))
    starts = list(range(0, count, group))
    cache = out / '.frames' / encjob
    initialized = False
    warmed = set()
    cpu_warmed = set()

    def initialize():
        nonlocal initialized, setup_seconds
        if not initialized:
            started = time.perf_counter()
            runtime.setup(pages[0])
            if runtime.stream is not None:
                runtime.stream.synchronize()
            setup_seconds = time.perf_counter() - started
            initialized = True

    # Frame boundaries depend only on encoding configuration, never decode_batch.
    for batch in range(0, len(starts), max(1, decode_batch // group)):
        frame_starts = starts[batch:batch + max(1, decode_batch // group)]
        missing = {mode: [i for i in frame_starts if (jobs[mode], i) not in done] for mode in modes}
        need = [i for i in frame_starts if any(i in indices for indices in missing.values())]
        frames = {}
        raws = {}
        for index in need:
            raw = b''.join(pages[i] for i in range(index, min(index + group, count)))
            raws[index] = raw
            old = encodes.get((encjob, index))
            saved = cache / (str(index) + '.bin')
            if old and saved.is_file():
                frame = saved.read_bytes()
                if hashlib.sha256(frame).hexdigest() != old['frame_sha256']:
                    raise ValueError('Cached frame checksum mismatch: ' + str(saved))
            else:
                initialize()
                # Warm each input size, including the final partial group.
                if len(raw) not in warmed:
                    encode_frame(runtime, raw)
                    warmed.add(len(raw))
                frame, seconds, page_seconds = encode_frame(runtime, raw)
                record = dict(block=index, blocks=min(group, count-index),
                              raw_bytes=len(raw), compressed_bytes=len(frame),
                              encode_seconds=seconds, encode_page_seconds=page_seconds,
                              setup_seconds=setup_seconds,
                              frame_sha256=hashlib.sha256(frame).hexdigest())
                if old:
                    if old['frame_sha256'] != record['frame_sha256']:
                        raise ValueError('Recreated frame changed; use a fresh experiment')
                else:
                    save_frame(saved, frame)
                    append(journal, 'encode', encjob, [record])
                    encodes[(encjob, index)] = record
                if old:
                    save_frame(saved, frame)
            frames[index] = frame
        for mode in modes:
            pending = missing[mode]
            if not pending:
                continue
            initialize()
            seconds, page_seconds, kind = decode_frames(runtime, gpu, mode,
                [frames[i] for i in pending], [raws[i] for i in pending], warmed=cpu_warmed)
            total = sum(len(raws[i]) for i in pending)
            records = [dict(encodes[(encjob, i)], decode_seconds=seconds * len(raws[i]) / total,
                            decode_page_seconds=page_seconds * len(raws[i]) / total,
                            decode_batch_kind=kind, encode_job=encjob,
                            actual_batch_blocks=sum(encodes[(encjob, j)]['blocks'] for j in pending)) for i in pending]
            append(journal, 'decode', jobs[mode], records)
            for record in records:
                done[(jobs[mode], record['block'])] = record
            print('BATCH_CHECKPOINT', filename, name, mode,
                  min(pending[-1] + group, count), '/', count, flush=True)
        if not retain_frames:
            for index in need:
                if all((job, index) in done for job in jobs.values()):
                    (cache / (str(index) + '.bin')).unlink(missing_ok=True)
    rows = []
    for mode in modes:
        records = [done[(jobs[mode], i)] for i in starts]
        raw = sum(r['raw_bytes'] for r in records)
        packed = sum(r['compressed_bytes'] for r in records)
        rows.append(dict(dataset=Path(filename).stem, storage_dtype=pages.dtype,
            page_format=pages.page_format, quantization_origin=pages.quantization_origin,
            quantization_metrics=json.dumps(pages.dataset_metadata.get('quantization_metrics', {}), sort_keys=True),
            data_bytes=sum(pages.descriptor(i)['data_bytes'] for i in range(count)),
            scale_bytes=sum(pages.descriptor(i)['scale_bytes'] for i in range(count)),
            validation_only=validation_only, codec=name, execution_path=mode,
            job_fingerprint=jobs[mode], encode_job=encjob, frame_format=route.backend_id,
            layout=layout['layout'], page_tokens=layout.get('page_tokens', ''),
            block_bytes=pages.block_bytes, cpu_encode_threads=1,
            decode_batch_blocks=decode_batch if mode.endswith('-GPU') else 1,
            encode_batch_blocks=group, encode_batch_min_blocks=min(r['blocks'] for r in records),
            encode_batch_requested_blocks=encode_batch, decode_batch_kind=records[0]['decode_batch_kind'],
            blocks=count, verified_blocks=sum(r['blocks'] for r in records),
            raw_bytes=raw, compressed_bytes=packed, compression_ratio=raw/packed,
            encode_mib_s=raw/1048576/sum(r['encode_seconds'] for r in records),
            encode_page_mib_s=raw/1048576/sum(r['encode_page_seconds'] for r in records),
            decode_mib_s=raw/1048576/sum(r['decode_seconds'] for r in records),
            decode_page_mib_s=raw/1048576/sum(r['decode_page_seconds'] for r in records),
            setup_seconds=records[0]['setup_seconds'], timing_version=TIMING_VERSION,
            fingerprint_schema='layered-measurement-v1', frames_retained=retain_frames,
            encode_timing_scope='gpu_codec_call_cuda_events_batched',
            decode_timing_scope='gpu_codec_call_cuda_events' if mode.endswith('-GPU') else 'cpu_codec_call_perf_counter'))
    return rows
