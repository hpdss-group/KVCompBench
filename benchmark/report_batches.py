"""Describe actual codec batches from existing measurement journals."""
import json
from collections import Counter


def _summary(records, total, decode, cpu):
    ordered = sorted(records.items())
    if not ordered or sum(r[0] for _, r in ordered) != total:
        return None
    cursor = 0
    for index, (span, actual) in ordered:
        if index != cursor:
            return None
        cursor += span
    if decode and not cpu:
        values = []
        position = 0
        while position < len(ordered):
            target = ordered[position][1][1]
            if type(target) is not int or target < 1:
                return None
            pages = 0
            while pages < target and position < len(ordered):
                span, actual = ordered[position][1]
                if actual != target:
                    return None
                pages += span
                position += 1
            if pages != target:
                return None
            values.append(target)
    else:
        values = [span for _, (span, actual) in ordered]
    if any(type(v) is not int or v < 1 for v in values):
        return None
    regular = max(values)
    tail = values[-1] if values[-1] < regular else None
    irregular = any(v != regular for v in values[:-1])
    return dict(regular=regular, tail=tail, irregular=irregular, observed=sorted(set(values)))


def annotate_batches(output, rows):
    wanted = {r['job_fingerprint'] for r in rows if r.get('job_fingerprint')}
    encoded = {r['encode_job'] for r in rows if r.get('encode_job')}
    enc, dec = {}, {}
    for filename in ('progress.jsonl', 'batch-progress.jsonl'):
        path = output / filename
        if not path.exists():
            continue
        with path.open(encoding='utf8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    batch = json.loads(line)
                except json.JSONDecodeError:
                    if not line.endswith('\n'):
                        continue
                    raise
                job = batch.get('job')
                kind = batch.get('type')
                if kind == 'encode' and job in encoded:
                    target = enc.setdefault(job, {})
                    for r in batch['records']:
                        target[r['block']] = (r.get('blocks', 1), r.get('blocks', 1))
                elif kind == 'decode' and job in wanted:
                    target = dec.setdefault(job, {})
                    for r in batch['records']:
                        span = r.get('blocks', 1)
                        target[r['block']] = (span, r.get('actual_batch_blocks'))
    for row in rows:
        total = int(row['blocks'])
        row['encode_batch_display'] = _summary(enc.get(row.get('encode_job'), {}), total, False, False)
        row['decode_batch_display'] = _summary(dec.get(row.get('job_fingerprint'), {}), total, True, row['execution_path'].endswith('-CPU'))
    return rows
