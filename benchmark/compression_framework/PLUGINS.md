# Drop-in Codec Packages

## Add a Python API Without Writing an Adapter

Create this layout inside `benchmark/compressors/`:

```text
my_codec/
├── codec.json
├── API.md                # Optional interface documentation
└── requirements.txt      # Optional Python dependencies
```

`codec.json`:

```json
{
  "schema_version": 1,
  "codecs": [
    {
      "name": "my-zlib",
      "kind": "python",
      "version": "1",
      "module": "zlib",
      "compress": "compress",
      "decompress": "decompress",
      "compress_kwargs": {"level": 6},
      "device": "cpu"
    }
  ]
}
```

The framework discovers the package on the next invocation. No registry edits or framework changes are required:

```bash
python3 kvbench.py cli list
python3 kvbench.py cli check --codec my-zlib
python3 kvbench.py cli compress kv.raw kv.kvc --codec my-zlib
python3 kvbench.py cli decompress kv.kvc restored.raw
```

Alternatively set `codec = my-zlib` in the `[compression]` section of `benchmark/settings.txt`. The decompressor reads the codec name and version from the stored container. The package must remain installed for decoding.

The `python_api` template uses the standard-library zlib module and requires no extra dependency. Copy `compression_framework/plugin_templates/python_api/` into a new `compressors/` subdirectory and give its codec a unique name.

## Package Format

Each `codec.json` has `schema_version: 1` and a `codecs` array. A folder can expose multiple named variants. Names must be globally unique and cannot replace built-in names. A codec entry may set `enabled: false` to disable it. Folders without a manifest, and folders beginning with `.` or `_`, are ignored.

Common fields for Python and command entries:

| Field | Meaning |
|---|---|
| `name` | Name used in settings and CLI |
| `kind` | `python` or `command` |
| `version` | Required format/API version string |
| `device` | `cpu` by default; `gpu` enables GPU selection checks |
| `dtypes` | Optional list of accepted metadata dtype strings |
| `alignment` | Optional positive input/output byte alignment, default 1 |
| `distribution` | Optional installed Python distribution whose version is included in the container identity |
| `enabled` | Optional boolean, default true |
| `description` | Optional package description |

Malformed manifests, duplicate names, missing entry files, and package paths escaping their directory are rejected. Discovery does not import Python codec APIs or execute command-line programs. Import and runtime dependency errors are reported when a codec is selected or checked. Installed packages are trusted code; this mechanism is not a sandbox.

## Python Interfaces

Declare exactly one of:

- `module`: an importable API module in the selected environment, such as `zstandard` or `vendor_codec.api`.
- `file`: an API file relative to the package, such as `vendor/api.py` or `vendor/__init__.py`. Package entry points named `__init__.py` can use relative imports.

`compress` and `decompress` identify callables in that module. Dotted attributes are supported. Optional `compress_kwargs` and `decompress_kwargs` pass static keyword arguments. This direct binding works for APIs accepting and returning byte buffers; APIs requiring object construction or a different call signature need a small API shim supplied in the codec package.

With `interface: "bytes"` (default), calls are:

```python
compress(raw, **compress_kwargs) -> bytes_like
decompress(packet, **decompress_kwargs) -> bytes_like
```

With `interface: "context"`, calls are:

```python
compress(raw, metadata, **compress_kwargs) -> bytes_like
decompress(packet, expected_bytes, metadata, **decompress_kwargs) -> bytes_like
```

Return values must be bytes, bytearray, or memoryview. Numeric conversion and tensor interpretation are the responsibility of the codec API. The framework verifies decoded lengths and bit-exact restoration.

## Command-line Interfaces

Use `kind: "command"` with argument arrays:

```json
{
  "schema_version": 1,
  "codecs": [
    {
      "name": "my-zstd-cli",
      "kind": "command",
      "version": "1",
      "compress": ["zstd", "-q", "-f", "{input}", "-o", "{output}"],
      "decompress": ["zstd", "-q", "-d", "-f", "{input}", "-o", "{output}"],
      "timeout_seconds": 60
    }
  ]
}
```

Supported placeholders: `{input}`, `{output}`, `{expected_bytes}`, `{plugin_dir}`, and `{python}` (the selected environment's interpreter). Each command must use both input and output placeholders and create the output file. Programs are executed without a shell. Do not put redirection, pipes, environment assignments, or shell expansion in the argument array. Literal braces must be doubled. Nonzero status, missing output, wrong decoded length, and timeout are errors.

An executable can be installed on PATH or supplied as `./bin/tool` relative to the package. Packaged programs must match the host architecture and OS. Command scratch files are kept under `compression_framework/.runtime/plugin-tmp/` and cleaned after each call. External tools must also be configured to keep their own caches and temporary output within allowed directories when used on the shared server.

The command template requires an actual `zstd` executable. Installing the `zstandard` Python package does not supply that executable. The framework does not compile arbitrary source trees or install system packages automatically.

## Dependencies

Put Python requirements in the package's `requirements.txt`. Install them into the environment selected by settings:

```bash
python3 kvbench.py cli install-plugin-deps
```

The normal environment setup scripts also install requirements from enabled packages. Requirements are not installed automatically during discovery or decompression. Native executables and library/system dependencies must be provided separately through the package or a compatible installed API.

## Compatibility and GPU Use

Blocks must decode independently. Codec version, manifest identity, optional distribution version, and the declared API file fingerprint are checked when reading. Update the version when changing auxiliary modules, native libraries, executables, models, or other resources affecting decoding. Such resources are not recursively hashed or embedded automatically. A version mismatch fails instead of silently decoding with another implementation.

GPU packages must declare `device: "gpu"`; checks require an administrator-assigned `CUDA_VISIBLE_DEVICES`. Discovering or listing a GPU package does not execute it. Runtime hardware checks still belong to the actual GPU API. No GPU validation is implied by passing CPU plugin tests.

Existing built-in folders use `kind: "builtin"` descriptors that reference the framework's built-in library adapters. New external codecs should use `python` or `command` packages.

## Unified GPU Decode Frames

The full BF16 benchmark can use a CPU or GPU encoder with one GPU decode entry point. A GPU-decodable block is a versioned `KVG1` frame containing its backend ID, decoded byte length, and compressed payload. The frame is produced and consumed by `compression_framework/gpu_decode.py`; do not reproduce its binary header in a plugin.

An external codec which emits a supported GPU-decodable frame declares its backend in `codec.json`:

```json
{
  "name": "my-zstd-gpu-decode",
  "kind": "python",
  "version": "1",
  "file": "api.py",
  "interface": "context",
  "compress": "compress",
  "decompress": "decompress",
  "gpu_backend": {
    "type": "shared",
    "backend": "nvcomp-zstd-raw",
    "frame_version": 1
  }
}
```

The initially registered IDs are `nvcomp-zstd-raw`, `nvcomp-snappy-raw`, `nvcomp-gzip-raw`, `nvcomp-lz4-block-raw`, and `nvcomp-deflate-raw`. LZ4 uses raw blocks, not `lz4.frame`; Deflate uses raw Deflate, not a zlib-wrapped stream. A compatible CPU encoder can call `compress_cpu_to_gpu_frame(raw, backend_id)`. The GPU side calls `decode_gpu_frame(frame, expected_bytes)` after `CUDA_VISIBLE_DEVICES` selects an assigned device.

The full runner discovers every enabled codec whose `codec.json` declares `gpu_backend`; no runner configuration is required. For a codec-local implementation, declare `type: "module"`, `file: "gpu_backend.py"`, `id`, `cpu_encode`, `gpu_decode`, and optionally `gpu_encode`. That module owns the algorithm-specific code; the framework owns framing, dispatch, verification, and benchmark reporting.

For a genuinely new bitstream, add a framework-owned decoder backend in `gpu_decode.py`: provide a unique lowercase backend ID, a format description, and exactly one decoder implementation (an nvCOMP algorithm or a custom GPU decoder callable). Then declare that ID in the codec manifest and add bit-exact tests for malformed frames, short final blocks, all-zero, random, and production data. A new backend does not require a new GPU encoder; CPU and GPU encoders may emit the same versioned frame.
