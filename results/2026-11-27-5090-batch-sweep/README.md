# RTX 5090 batch-size experiments

Eight completed full runs contain 236 rows each (1,888 total): 40 paths per BF16
dataset and 39 paths per scaled-FP8 dataset. Every recorded page passed byte-exact
validation. Both Qwen and DeepSeek preserve their KV layouts. BF16 pages contain
131,072 bytes; FP8 pages contain 268,320 bytes including FP32 scales. Tail pages
retain their actual lengths. Losslessness refers to stored BF16/FP8 bytes, not
recovery of pre-quantization BF16 from FP8 inputs.

| Directory | Requested encoder pages/frame | Requested GPU decoder pages/batch | Measured encoder constraint |
| --- | ---: | ---: | --- |
| [all_codecs_128_128](all_codecs_128_128/heatmap.html) | 128 | 128 | Historical 64 MiB cap |
| [all_codecs_128_256](all_codecs_128_256/heatmap.html) | 128 | 256 | Historical 64 MiB cap |
| [all_codecs_128_512](all_codecs_128_512/heatmap.html) | 128 | 512 | Historical 64 MiB cap |
| [all_codecs_128_1024](all_codecs_128_1024/heatmap.html) | 128 | 1024 | Historical 64 MiB cap |
| [all_codecs_256_128](all_codecs_256_128/heatmap.html) | 256 | 128 | FP8 frames capped at 250 pages |
| [all_codecs_512_128](all_codecs_512_128/heatmap.html) | 512 | 128 | FP8 frames capped at 250 pages |
| [all_codecs_1024_1024](all_codecs_1024_1024/heatmap.html) | 1024 | 1024 | BF16 512 pages, FP8 250 pages |
| [all_codecs_max](all_codecs_max/heatmap.html) | max | max | Resource calibration; fixed frame cap removed |

Decode batches contain complete encoded frames. Requested values can differ from
actual sizes when a frame is indivisible, a byte cap applies, or the final batch is
partial. In the max run, each GPU-GPU codec encoded a whole dataset in one frame:
5,074 pages for DeepSeek BF16; 4,096 for Qwen BF16; 1,290 for DeepSeek FP8; and
1,024 for Qwen FP8. CPU-GPU encoders still received individual pages.

## Measured throughput and limitations

The following paired medians use 128/128 as 1.0. GPU encoding covers 60 GPU-GPU
rows. Each decode path covers 60 rows. These are medians of matched per-row ratios,
not ratios of aggregate byte counts and not statistical confidence estimates.

| Request | GPU encode call | CPU-GPU decode call | GPU-GPU decode call | CPU-GPU decode page | GPU-GPU decode page |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128/128 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 128/256 | 0.984 | 1.868 | 1.809 | 1.230 | 0.690 |
| 128/512 | 1.010 | 3.467 | 3.424 | 1.087 | 0.674 |
| 128/1024 | 1.002 | 5.925 | 4.650 | 1.096 | 0.662 |
| 256/128 | 1.641 | 0.997 | 1.779 | 1.070 | 0.830 |
| 512/128 | 1.678 | 0.999 | 1.818 | 0.960 | 0.594 |
| 1024/1024 | 1.622 | 6.273 | 3.916 | 1.102 | 0.493 |
| max | 2.125 | 8.683 | 6.080 | 1.439 | 0.844 |

Codec-call timing excludes setup, calibration, warm-up, transfers, framing and
verification. GPU calls use CUDA events on the execution stream. Page timing adds
preparation, transfers and output materialization; it is not full application
latency. CSV throughput fields use MiB/s; the HTML converts them to decimal GB/s.
Identity can return an existing bytes object and is a no-op baseline, not a memory
copy-bandwidth measurement.

Runs used different RTX 5090 devices and shared-server time windows, without
repeated statistical trials. Preserved outliers include the DeepSeek FP8 E5
bitcomp encoding in 1024/1024, where one call consumed about 230 ms. No raw timing
outliers were removed. Larger batches are not uniformly faster, and max selects
capacity rather than throughput. One-frame max results do not represent random
single-page access. The max run also includes the SplitZip large-frame parser fix;
small-frame historical runs predate that correction.

## Files and provenance

Each experiment contains the original CSV, HTML and PNG, route capabilities,
unavailable-codec declarations, a portable settings example, experiment metadata,
and byte checksums. `comparison.csv` combines all rows and `relative_summary.csv`
contains matched median ratios. Input data, installation environments, compressed
frames and large JSONL journals remain outside this publication.

`server_source_sha256.json` records the transferred current server source bytes.
`current_source_sha256.json` records the publication source tree after documentation
and line-ending normalization; neither is a claim that earlier runs measured this
exact source tree. Per-row layered fingerprints remain in the original CSV files.

For a repeat run, install the declared codecs with `kvbench.py setup`, retain the
resulting managed environment section, and apply the dataset and benchmark sections
from the selected experiment. Historical 64 MiB-capped runs require their recorded
actual frame sizes or the historical implementation; requesting the same number
with the current uncapped code does not reproduce that cap automatically.
