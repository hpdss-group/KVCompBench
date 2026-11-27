# Codec installation and diagnostics

`kvbench.py` is the public entry point. Codec adapters retain their existing encode/decode APIs; installation is declared separately in each package's `install.json`. New packages require both `codec.json` and `install.json` and may declare `package.json` for core/optional classification.

## Selection and environments

```bash
python3 kvbench.py setup --codecs gzip,zstd:3,lz4
python3 kvbench.py setup --codecs alp,chimp,elf,fse --plan
python3 kvbench.py setup --codecs alp,chimp,elf,fse
python3 kvbench.py codecs --status
python3 kvbench.py doctor --codecs gzip,zstd:3,lz4
python3 kvbench.py doctor --codecs fse,huff0 --report benchmark/.runtime/doctor.json
```

Names may identify a codec or a package; package selection installs its implementation once. Installation does not rewrite the benchmark codec list. `settings.txt` still controls the names and execution routes used for measurements.

Managed environments live under `.venvs/<platform>-managed-cpu/` or `.venvs/<platform>-managed-cuda13/`. Python API packages support platforms for which their pinned distributions are available. Native recipes and the managed CUDA13 profile currently target Linux x86-64. A system GCC/G++ compiler is required for C/C++; a pinned JDK is downloaded for Java recipes. No sudo, driver installation or system package changes are performed.

Managed setup installs framework CPU/CUDA PyTorch and pinned base utilities, then only the requested package dependencies. Direct dependency versions are fixed; the complete resolved `pip freeze` inventory and compiler versions are recorded in receipts. Transitive resolution, wheel selection and system compilers can still differ across platforms, so compare the recorded environment rather than assuming hardware-independent reproducibility.

Downloads, sources and installation records stay inside the selected environment:

```text
kvcompbench/downloads/<sha256>       # checksum-verified cache
kvcompbench/sources/<package>/<revision>/
kvcompbench/receipts/<package>.json  # manifest, artifacts, tools, dependencies
kvcompbench/install.log
kvcompbench/doctor.json
native/<package>/                  # compatible existing native ABI layout
toolchains/jdk17/                  # downloaded Java toolchain
```

Checksums are verified before extraction. Source selection is an explicit archive-member map. Paths escaping managed roots and archive links are never materialized. Existing source changes or corrupt cache entries produce an error rather than being silently overwritten. A per-environment exclusive lock prevents concurrent setup. Failed/incomplete downloads and stale locks require inspection; unrelated caches and environments are not automatically cleaned.

## Installation Manifest v1

A minimal Python package manifest:

```json
{
  "schema_version": 1,
  "package": "my_codec",
  "revision": "upstream-1.2.3",
  "requirements": ["my-codec==1.2.3"],
  "imports": ["my_codec"]
}
```

Optional fields:

| Field | Purpose |
| --- | --- |
| `platforms` | Explicit supported build platforms, e.g. `linux-x86_64` |
| `gpu_only` | Require the CUDA profile for a GPU-only implementation |
| `gpu_requirements`, `gpu_imports` | Additional pinned dependencies/interfaces for GPU routes |
| `installed_distributions` | Exact expected versions installed by a source-build command |
| `sources.archives` | HTTPS archive URLs and SHA-256 pins |
| `sources.files` | Destination → archive/member/hash mapping; optional checked text patches |
| `resources` | Pinned toolchain/JAR downloads, managed target paths and optional extraction |
| `build_inputs` | SHA-256 pins for project-owned bridge files used in compilation |
| `directories` | Managed build/output directories to create |
| `commands` | Argument arrays, run without a shell and with finite timeouts |
| `outputs`, `output_globs` | Required artifacts whose hashes enter the receipt |
| `tools` | Compiler executables whose actual versions enter the receipt |

Commands may use `{source}`, `{plugin}`, `{prefix}`, `{python}` and `{jdk}`. An argument such as `@glob:{source}/java/**/*.java` expands a bounded managed-source glob. Recipes are trusted executable plugin metadata and should be reviewed like adapter code.

ALP, Chimp/ELF, FSE, StreamVByte and SplitZip are fetched at immutable commits. ELF's existing overflow guard and ZipNN's CPython method-table sentinel fix are represented as checked patches with expected input and output hashes. The ZipNN build is versioned `0.5.4+sentinel1`; the unpatched build is not substituted. Upstream license files are retained in downloaded source selections where provided.

## Status and doctor

`codecs --status` is non-installing inspection: it reports distribution versions and dependency constraints, import paths/hashes, native artifact hashes, declared CPU/GPU capabilities, and receipt/manifest agreement. `ready` means those prerequisites match; `cpu: not_tested` and `gpu: not_requested` are not correctness claims. Missing packages are listed, not installed or silently selected from another algorithm.

`doctor` performs byte-exact CPU round trips with several input lengths and an appropriate supported dtype. GPU-only implementations are explicitly deferred unless `--gpu` is supplied. GPU checks require an assigned device:

```bash
python3 kvbench.py setup --codecs gzip,zlib,nvcomp-lz4 --device cuda13
CUDA_VISIBLE_DEVICES='GPU-assigned-uuid' python3 kvbench.py doctor --codecs gzip,zlib,nvcomp-lz4 --gpu
```

The GPU probe checks actual declared batch routes and compatible CPU decoders using the same frames. It is not a throughput benchmark. Missing CUDA, dependency/version mismatches, unsupported routes and lossless failures are reported separately. No GPU is automatically chosen or allocated by ordinary status/setup.

## Explicit existing installations

For installed Python APIs, specify the existing interpreter explicitly; versions and imports are checked and CPU round trips must pass before it is selected:

```bash
python3 kvbench.py setup --codecs zstd:3 --python /opt/existing-env/bin/python
```

No package installation is performed in that external interpreter. Native or source-based implementations additionally require an external lock describing their declared revision and artifact provenance:

```bash
python3 kvbench.py setup --codecs fse --external benchmark/.runtime/external.json
```

```json
{
  "schema_version": 1,
  "python": "/opt/existing-env/bin/python",
  "packages": {
    "fse": {
      "revision": "9f30e0918f87bd835fa040d922a208d7b219e50b",
      "native_root": "/opt/existing-codecs/native",
      "source_dir": "/opt/existing-codecs/fse-sources",
      "artifacts": {"native/fse/libcodec.so": "<actual-sha256>"}
    }
  }
}
```

Replace placeholders with actual values. `source_dir` may use the managed subset layout or the original pinned repository layout; `native_root` contains the package subdirectory. A package may supply `artifact_paths` (standard artifact key → absolute installed path) and `source_files` (normalized source key → absolute source path) for other existing layouts, without changing adapters. Java bridge classes and JARs may be mapped the same way. The framework validates the lock against the selected interpreter, expected revision, pinned source hashes and actual artifact hashes, and checks the existing adapter interface. It does not patch or rebuild an external installation. External revision evidence is explicitly labeled as a provider declaration plus verified hashes; arbitrary binary provenance cannot be proven from a version string alone.

After validation, local settings contain `environment.python` and, where needed, `environment.external_lock`. Managed environments remain the default when no explicit external interpreter is selected. Benchmark job specifications and container metadata record actual installation identity outside timed codec calls, including interpreter/platform identity and installed native dependency paths/hashes, so changes to versions, paths, manifests or artifacts invalidate resume fingerprints. Keep an active benchmark environment unchanged until the run finishes. Old benchmark result snapshots remain historical data and are not relabeled as runs of this installer.

## Hardware validation

On 2026-11-17, a separate Linux x86-64 environment with Python 3.10.12, PyTorch 2.9.1+cu130, Triton 3.5.1, nvCOMP 6.0.0.12 and an RTX 5090 passed the public `doctor --gpu` command for all 36 registered configurations: 26 CPU round trips and 20 GPU round trips. All 21 package manifests produced complete installation receipts. ZipNN's sentinel-fixed extension and the C/C++ and Java codecs were built in that environment. Matching cached wheels and checksum-verified JDK/JAR downloads were reused after slow network downloads timed out; this does not establish that an uncached installation will complete on every network.

An additional mixed-size GPU batch check passed for SplitZip, OpenZL, all nine nvCOMP algorithms, three Zstd levels, LZ4, Snappy, Gzip and Zlib. Each batch contained 10-, 70- and 10,250-byte tails, a 512 KiB page and 1 MiB of deterministic random bytes. GPU-decoded bytes and compatible CPU decoding were checked against the original bytes. These are correctness checks, not throughput measurements or a rerun of the published BF16 dataset results.

The checks exposed tail-byte loss with Cascaded's default logical input type. Its configuration now explicitly requests `data_type = "|u1"` and declares the `nvcomp-native:cascaded:u8` frame format; both the direct codec API and benchmark backend use the configured options. The failing tail case and mixed-size batch then passed. An external installation with an incorrect declared revision was also rejected before codec execution.


## Existing project environments and source release

An explicitly named project-local environment can be updated without selecting a new prefix:

```bash
python3 kvbench.py setup --codecs splitzip --device cuda13 --env linux-x86_64-paper-cuda13
```

The environment name must be a single path component. Setup still verifies pinned dependencies and source contents, records installation receipts, and retains WSL routing settings. A complete managed-source cache is reused only when every installed source hash matches its Manifest.

Third-party source trees and historical source archives belong outside `benchmark/compressors`. Local historical copies are kept under `.repository_archive/`, which is excluded from version control. The legacy `setup_paper_codecs.py` entry point now delegates to the same Manifest installer.

```bash
python3 scripts/export_source.py --output benchmark/output/kvcompbench-source.zip
```

The exporter rejects remaining `upstream`, `dependencies` or `tmp` directories inside Codec packages. The release includes project adapters, bridges, Manifests and tests, plus a per-file SHA-256 inventory. Data, settings, environments, results, archives and compiled third-party artifacts are excluded.

## GPU diagnostic matrix

`doctor --gpu --codecs splitzip` runs BF16 plus FP8 E4M3FN at 2/3 bits and FP8 E5M2 at 2/3/4 bits. `diagnostics.json` declares these cases. Each case checks actual GPU encoding, three-page joined encoding, mixed-size batch decoding, tails and bit-exact restoration. Compatible CPU decoders are checked when declared. E4M3FN 4-bit encoding is unsupported by this adapter and is not reported as verified.


## Native-library integrity cache

Managed environments cache computed native-library SHA-256 values in `kvcompbench/hash-cache.json`. Cache hits require matching device, inode, size, modification time and change time. Changed metadata causes the contents to be hashed again; the returned measurement identity always contains the content SHA-256. Concurrent cache writers can lose cache hits without changing results.

Set `KVCOMPBENCH_REHASH=1` for a full content recheck. The Windows entry point forwards this flag to WSL. The cache is not a defense against an attacker who can edit both files and cache metadata.

Wheel inventories are filtered from `RECORD` before checking native files. This avoids Python 3.12 checking every Python source file in a large PyTorch wheel over a WSL mounted drive. Native-file existence and actual hashes are still verified.
