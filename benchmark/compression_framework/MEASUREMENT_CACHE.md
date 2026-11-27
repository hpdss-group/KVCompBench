# Installation provenance and layered measurement reuse

## Installation management

`kvbench.py setup --codecs ...` selects codec packages through their pinned
`install.json` manifests. Downloaded sources, dependencies, build artifacts,
checksums and receipts remain under the managed environment. Existing adapters
load those artifacts through `codec_manager`; no adapter rewrite or manual
third-party source checkout is required. See [INSTALLATION.md](INSTALLATION.md).

`kvbench.py codecs --status` inspects installation/version/interface metadata.
`kvbench.py doctor` additionally performs CPU lossless checks. GPU correctness
checks require `doctor --gpu` and an explicitly assigned `CUDA_VISIBLE_DEVICES`.
Status inspection does not allocate a GPU.

Installation receipts now include `recipe_sha256`. Changing a mirror URL or
installation documentation leaves this executable-recipe identity unchanged.
Changed commands, source pins or build inputs invalidate the receipt. Receipts
without this field continue to require an exact original Manifest match; an
unverifiable old receipt is not silently migrated.

## Measurement layers

The fingerprint schema is `layered-measurement-v1`:

| Layer | Dependencies | Invalidated measurements |
| --- | --- | --- |
| Data | Input SHA-256, shape, dtype, page layout, validation selection, page implementation | Affected dataset and its dependent stages |
| Encode | Data ID, codec entry/options, actual installation identity, encoder implementation, worker count, effective frame grouping, encode protocol | Affected encoder and dependent decoders |
| Decode | Encoder ID, actual installation identity, decoder implementation, execution mode, GPU decode batch size, decode protocol | Affected decoder jobs |
| Reports | CSV rows and report templates | Report rendering only |

Paths, receipt origin, full Manifest hashes and download URLs are provenance,
not measurement dependencies. Actual artifact/source hashes, Python identity,
package versions, relevant dependency identities, build commands and compiler
versions remain measurement dependencies. Job JSON files retain full paths and
the exact code hashes for auditing alongside `measurement_layers`.

Codec-local CPU decoder files declared in `decode_backends.json` do not enter
encoder identities. Python AST hashing ignores comments, docstrings and simple
standalone progress prints; encode/decode entry points are separated by their
names. Unknown helpers, constructors, imports and setup are shared dependencies
and conservatively invalidate both stages. A combined native binary or pinned
upstream implementation also remains shared: this does not claim function-level
dependency analysis for third-party binaries.

The orchestration runner is intentionally absent from measurement identities.
Changes to its measurement boundaries, correctness requirements, grouping or
scheduling semantics must increment the corresponding `PROTOCOL` entry in
`measurement_identity.py`. Changes to stage implementation files are detected
automatically. Logging, report generation and unrelated framework files need no
protocol bump. This explicit contract is required when modifying the runner.

## Checkpoints and compressed frames

Ordinary per-page measurements use `progress.jsonl`. Joined-page GPU encoding
uses `batch-progress.jsonl`, with separate encoding and validated decoding
records. Each JSONL append is flushed and fsynced. An interrupted final line is
preserved in a `.tail.<timestamp>` file before recovery. A failed lossless check
never writes a completed decoding record.

Frame boundaries depend only on the resolved `encode_batch_blocks`,
not on `decode_batch_blocks`. Decode batches group complete frames, so a frame
larger than the requested decode page count is processed as one frame. Changing
the GPU decode batch size reuses encoding measurements and frames; compatible
CPU decoding does not depend on the GPU batch size.

```ini
[benchmark]
encode_batch_blocks = 256
decode_batch_blocks = 256
retain_frames = true
```

With `retain_frames = true` (the default), `.frames/<encode_job>/<first_page>.bin`
is retained, including after successful completion. This consumes disk space
approximately equal to the compressed payloads of all retained encoder jobs.
Cached frames are checked against their recorded SHA-256 before reuse.

With `retain_frames = false`, frames used in the active run are removed after
all selected decoding paths verify them. This does not perform a bulk cleanup
of historical caches. Missing frames can be recreated on demand; their hashes
must match before old encoding measurements are reused. Nondeterministic frame
changes cause an explicit error rather than mixing historical measurements.

Changing only a decoder can therefore keep the original encoding timings.
Codec setup/calibration still runs when the decoder needs it. CUDA warmup and
measurement are not required when all selected decoding jobs already completed.

Old whole-runner fingerprints and pre-layered batched checkpoints are preserved
but are not automatically accepted as compatible. The first run under this
schema measures new jobs. Historical CSV/HTML files are not resume records.
Use a new output directory to preserve published CSV/HTML snapshots; reuse a
directory to regenerate its current report from the selected matching jobs.

## Regression checks

```bash
python3 -B -m unittest discover -s tests -v
```

Checks cover installation receipts and integrity, an offline pinned-source
installation/build, decoder-only reuse, frame corruption, interrupted JSONL,
completed-job reuse, frame recreation, validation mode and fingerprint scope.
GPU scheduling tests use a simulated backend and do not establish GPU performance.

## Windows and WSL

When settings contain `environment.wsl_distribution`, the Windows entry point delegates run, status, doctor and managed setup to that distribution. `environment.wsl_env_path` selects the existing interpreter. CUDA_VISIBLE_DEVICES is forwarded explicitly. No GPU is selected automatically by installation diagnostics. Managed setup retains WSL routing while updating the selected environment path.

Existing environments are not silently adopted or rebuilt by this code update. `setup --external` validates an external installation lock; `setup --python` validates an existing Python API environment. Native/source codecs require pinned source and artifact evidence. A missing receipt is reported as unavailable until setup or explicit external validation succeeds.

## Reported batch sizes

Each heatmap cell displays actual regular and final-call encoding/decoding page counts reconstructed from measurement journals. Requested policies remain available in the detail tooltip and CSV. Small datasets and resource calibration may reduce the selected size. Missing historical batch records are explicitly shown as unknown. CPU codec APIs record one logical page per call.

GPU compatibility output is kept in an independent diagnostic directory. Its timings are small-input regression evidence, not full-dataset performance results.

## Shared component dependency scope

Shared sources are selected from the codec adapter's framework imports and referenced symbols, including their static dependencies. Importing a helper does not depend on unrelated classes in the same module. Builtin catalog dispatch is specialized for the selected codec only when the known dispatch bindings match; changed dispatch semantics fall back to the full component. Shared frame providers are selected by backend ID, with conservative fallback for unknown provider formats.

`fingerprint_dependencies.json` can declare whether a custom backend uses the plugin API. The SplitZip backend declares `plugin_api: false` because it uses its own runtime instead of the generic plugin API. Unknown dependencies remain conservative; this is not a complete dynamic Python call graph.

The existing encode/decode naming classification, runner protocol versions and aggregation entry points are unchanged. Existing fingerprints are not automatically migrated to the narrowed dependency inventory; old records remain available.


## Automatic GPU batch sizing

`benchmark/settings.txt` accepts `max` independently for `encode_batch_blocks`
and `decode_batch_blocks`. Numeric values from 1 to 4096 specify logical pages per encoder frame,
without a fixed 64 MiB frame cap. CPU codecs continue to receive individual page
calls; `max` only bounds their host scheduling window.

```ini
[benchmark]
encode_batch_blocks = max
decode_batch_blocks = max
max_batch_bytes = 536870912
max_batch_memory_fraction = 0.5
```

For GPU `max`, the raw-byte candidate budget is the smaller of `max_batch_bytes`
and `free_VRAM * max_batch_memory_fraction / 8`. The eightfold allowance covers
inputs, outputs and codec workspace; the fraction defaults to 0.5 and accepts
values greater than zero through 0.8. These are conservative resource limits,
not a claim to discover the device's absolute physical capacity. `max` is not
restricted by the numeric 4096-page limit.

Each codec and dataset is calibrated before measurement. Complete-page encoder
frames and complete-frame decode groups are checked using actual codec calls
and byte-exact output verification. An out-of-memory failure shrinks the search
interval; unrelated errors propagate. Encoder calibration includes the initial
batch and final frame. Decoder calibration uses the initial candidate group.
The search assumes capacity is monotonic in batch size. Its largest passing
candidate is a sampled resource-bound result, not a throughput optimum or a
guarantee for every possible data distribution.

Calibration and its transfers are excluded from reported measurement timings.
A later data-dependent memory failure stops the task and retains checkpoints;
it does not silently change encoder frame boundaries after task IDs exist.
Run again with a smaller byte cap if this occurs. GPU selection and launch
permissions remain separate from batch sizing.

Selected encoding frame size and decoding group size enter the existing layered
fingerprints. Changing only decoding size preserves the encoding identity.
Resource observations, configured policies and selected sizes are preserved in
job metadata, CSV and `BATCH_PLAN` log entries. A new run recalibrates; changed
resource availability can select a new batch and therefore a new task identity.

HTML obtains actual regular and final batch sizes from both JSONL journals,
reconstructing decode calls across per-frame records. Units appear in column
headings; cells show regular sizes and smaller final calls. Missing legacy
records are explicitly marked as unknown, rather than treated as measurements.


### Removing the fixed raw-byte cap

`max_batch_bytes = max` removes the fixed raw-byte cap for automatic GPU batch
selection. The candidate budget becomes `free_VRAM * memory_fraction / 8`;
remaining page count, complete-frame scheduling and calibration still apply.
The memory fraction and out-of-memory search remain active. Numeric positive
byte caps are still supported, and omitting the key retains the 512 MiB default.
Metadata records the unlimited cap as `max`, alongside the effective byte budget.
CPU APIs remain single-page calls and keep a bounded host scheduling window;
this option does not turn CPU APIs into batched codecs or remove all memory
safeguards. Actual selected batch sizes retain the existing fingerprint rules.


### SplitZip large-frame parsing

SplitZip's decode paths derive their segment limit from the caller's expected
raw byte count, then validate BF16 or FP8 dimensions and every buffer length.
This permits frames selected by automatic batching whose internal buffers exceed
64 MiB. Header bounds, integer segment sizes, payload bounds and trailing-byte
rejection remain active. Other codec adapters retain the shared parser's default
limit. This parser correction does not change the SplitZip wire format.


### Explicit numeric encoder batches

The fixed 64 MiB encoder-frame truncation is removed from both planning and the
batched runner fallback. Numeric requests use the requested page count, bounded
only by remaining dataset pages. Memory or codec failures are reported instead
of silently reducing numeric requests. Automatic resource budgeting and capacity
search remain specific to `max`. Decode calls still operate on complete frames.
Changed actual encoder frame sizes change the encoder and dependent decoder
identities; historical records remain intact.
