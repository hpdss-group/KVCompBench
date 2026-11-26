# RTX 5090 full-page benchmark snapshot

The server export contains 236 full-run rows across DeepSeek and Qwen BF16, FP8 E4M3FN, and FP8 E5M2. Every exported row reports equal total and verified page counts. Four SplitZip FP8 rows extend the prior published results.

The FP8 implementation used for these four rows feeds the complete stored page byte stream, including FP32 scales, into the upstream FP8 codec with three-bit exponent codes. It calibrates on the first page and merges kernel-coded page bodies for batch decode. The exported CSV uses the historical route label `splitzip-packet-v1` for these rows. The actual FP8 packet header is `splitzip-fp8-packet-v1`, which the current route reports explicitly.

The CSV and availability metadata were exported from the completed server result directory. The checked-in adapter uses the same pinned upstream source through the manifest-managed loader. The recorded throughput comes from the server implementation used for that run.
