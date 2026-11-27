# Compression framework

The project entry point is `python3 kvbench.py`; see the [project README](../../ReadMe.md) for installation, named codec selection, CPU/GPU decoding and benchmark execution.

```bash
python3 kvbench.py setup
python3 kvbench.py codecs
python3 kvbench.py cli list
python3 kvbench.py cli compress kv.raw kv.kvc --codec gzip
python3 kvbench.py cli decompress kv.kvc restored.raw
```

The default installation uses CPU PyTorch and core dependencies. Optional codec dependencies and GPU execution require the corresponding environment and an assigned device. `benchmark/settings.txt` is local configuration and is not published.

The runner schedules pages, measures codec and page timing, verifies decoded bytes and maintains fingerprinted JSONL journals. Codec directories own their declarations and implementations. See [PLUGINS.md](PLUGINS.md) for plugin APIs and [PAPER_CODECS.md](PAPER_CODECS.md) for optional implementations.
