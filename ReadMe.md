# KVCompBench

面向 BF16 和 scaled FP8 KV Cache 的无损压缩基准，比较压缩比、编码和解码吞吐量，并逐页校验解码结果。

仓库：[hpdss-group/KVCompBench](https://github.com/hpdss-group/KVCompBench)

## 统一入口

根目录只使用 `kvbench.py`，要求 64 位 Python 3.10+。Linux 使用 `python3`，Windows 可使用 `python` 或 `py -3`。

```bash
python3 kvbench.py --help
python3 kvbench.py codecs
python3 kvbench.py setup
python3 kvbench.py run --validate-only --output benchmark/output/core_validation
python3 kvbench.py run --output benchmark/output/core_full
python3 kvbench.py cli list
```

基础安装创建项目内 CPU 虚拟环境，仅安装 CPU PyTorch 和核心依赖，不安装可选编解码器。安装会写入 `benchmark/settings.txt`；已有环境可通过该配置的 `environment.python` 指定解释器绝对路径。

## 默认与可选 codec

默认仅运行五个经典 CPU 基线：identity、gzip、zlib、bz2、lzma。zlib 和 lzma 包另有参数变体；留空 `codecs` 并设置 `codec_set = core` 时选择核心组的全部九个名称。

LZ4、Zstd、Snappy、FSE/Huff0、OpenZL、ZipNN、SplitZip、nvCOMP、StreamVByte、delta-zstd 及其他扩展继续保存在 `benchmark/compressors/`，默认不选中。插件由自己的 `package.json` 声明 `core` 或 `optional`；没有分类声明的插件视为 optional。支持数据类型也参与筛选，选择 all 不代表所有插件都支持 BF16。

`benchmark/settings.example.txt` 为基础配置示例。默认运行读取被 Git 忽略的 `benchmark/settings.txt`，使用精确的 `codecs` 名单和 `execution_device = cpu`。首次运行会从示例创建配置。可选扩展配置不预置；需要时复制示例为 `benchmark/settings.optional.local.txt` 并按需编辑。

```ini
[benchmark]
codecs = identity, gzip, zlib, bz2, lzma
codec_set = core
execution_device = cpu
```

非空 `codecs` 优先于 `codec_set`，可直接包含 `lz4`、`zstd:3` 等 optional 名称，无需修改插件分类。用 `python3 kvbench.py codecs` 查询精确名称。重复、空名称或当前数据类型/设备不支持的名称会报错。命令行 `--codecs` 覆盖配置名单；依赖仍需安装，显式选择的算法不会因 `skip_unavailable_codecs` 而静默消失。

将 `execution_device` 改为 `declared` 可使用插件声明的 CPU/GPU 路径：基础名单中的 gzip、zlib 同时测量兼容 CPU 与 GPU 解码并共享编码；identity、bz2、lzma 保留 CPU 解码。不同帧格式仍为不同配置。GPU 解码需要相应 CUDA PyTorch、nvCOMP 依赖和已分配的 GPU；纯 CPU 安装不包含 GPU 依赖。

完整扩展安装目前面向 Linux x86-64/CUDA 13，按需执行：

```bash
python3 kvbench.py setup --optional
python3 kvbench.py run --settings benchmark/settings.optional.local.txt --output benchmark/output/optional_full
```

扩展运行前，填写所需 `codecs` 名单；如需整个扩展组，留空 `codecs` 并设置 `codec_set = all`。按需使用 `execution_device = declared`，并安装相应依赖。GPU 使用必须先获得分配，设置 `CUDA_VISIBLE_DEVICES` 为已分配的 UUID。缺失依赖默认报错；分组选择时仅显式启用 `skip_unavailable_codecs` 才跳过并记录原因。

## 数据与页面

数据位于 `benchmark/input/`，配置中的 dataset 节明确张量轴。KV 模式按 layer/batch 与 token 范围切片，保留完整 K/V、head 和特征维；末页使用真实长度。Qwen 的 256-token 页为 1 MiB，DeepSeek 共享 latent 的 128-token 页为 128 KiB。具体大小取决于实际张量形状。flat 模式保留通用字节块基线。

由量化数据转换得到的 BF16 应保留来源元数据，不能称为原始量化前 BF16。

## 测量与续跑

初始化、校准、预热、传输和校验排除在 codec 吞吐之外；页面吞吐单独计入准备、封装和传输。CPU 使用调用边界计时，GPU 使用执行流上的 CUDA Event。SplitZip 使用首个页面校准后冻结的码本，并合并批次解码；小页面结果不能直接与大块 GPU 驻留输入结果比较。

同一编码配置的测量和帧由兼容 CPU/GPU 解码路径共享；不同帧格式是不同配置。复用相同输出目录可续跑，依据仅为指纹匹配且验证通过的 `progress.jsonl` 记录，CSV 不用于判断续跑。数据、依赖、源码或硬件改变可能导致指纹变化。

## 输出与维护

输出目录包含 `result.csv`、`progress.jsonl`、任务元数据及完成后生成的 `heatmap.html`、`heatmap.png`。HTML 内嵌数据、CSS 和 JavaScript，可离线打开。模板源文件位于 `benchmark/templates/`。

```bash
python3 benchmark/html_report.py benchmark/output/core_full
```

运行数据、虚拟环境、断点和本机配置不随仓库发布；经验证的汇总结果与报告保存在 `results/`。根目录的安装与运行统一使用 `kvbench.py`，不再提供独立平台包装脚本。

新增 codec 在自己的目录声明 API、设备、帧格式、批量能力和依赖；兼容 CPU 解码声明使用 `decode_backends.json`，共享帧提供者使用 `frame_provider.json`。公共 runner 负责选择、调度、计时和正确性校验。


## BF16 与 scaled FP8 输入

`settings.txt` 可同时列出两份 BF16 和四份 E4M3FN/E5M2 数据。每个 dataset 节支持 `dtype = auto` 或明确的 `bf16`、`fp8_e4m3fn`、`fp8_e5m2`，声明值必须与存储张量一致。

FP8 输入采用 `data` 张量和 FP32 `scale`。支持顶层 `scale` 及 DeepSeek 的 `quantization.scale`，当前要求最后一维逐向量缩放。KV 页按原有 layer/batch/token 范围同步切片 data 与 scale。页面负载依次存放 data 字节和 scale 字节，区段偏移、字节数、形状和量化来源保存在任务清单及续跑日志中。

DeepSeek 的 128-token FP8 页为 64 KiB data 加 512 B scale，共 64.5 KiB。Qwen 的 256-token FP8 页为 512 KiB data 加 8 KiB scale，共 520 KiB。尾页使用真实 token 数。flat 基线按完整 data 字节流随后接 scale 字节流进行分块。

压缩率与吞吐分母包含 data 和 scale。逐字节校验覆盖两部分，量化误差元数据单独记录。当前四份 FP8 文件由 BF16 再量化得到，结果标记 `requantized_bf16`。DeepSeek 的原始 NoPE/RoPE 转换说明作为来源链保留。

包级 `package.json` 中的 `benchmark_dtypes` 声明完整页面负载支持的精度。字节流 Codec 可声明 BF16 和两种 FP8。SplitZip 提供独立的 BF16 与 FP8 GPU 实现，FP8 编码支持 2/3 位 E4M3 与 2/3/4 位 E5M2。兼容性按各插件的数据类型声明筛选，并在报告中记录不支持的组合。FP8 页协议同时覆盖 data 和 scale。

报告支持存储精度筛选，并展示 `storage_dtype`、`page_format`、`data_bytes`、`scale_bytes`、`quantization_origin`。格式和缩放信息参与续跑指纹。默认结果目录为 `benchmark/output/kv_pages_v4_fp8`。使用 `--validate-only` 可先检查每份输入的首个、第二个和首层尾页。


## 安装管理与分层缓存

通过 `python kvbench.py setup --codecs ...` 按插件 `install.json` 安装固定版本，使用 `codecs --status` 查看安装状态，使用 `doctor` 检查基本无损正确性。已有安装可通过 `setup --python` 或 `setup --external` 显式选择并验证。GPU 检查需显式传入 `doctor --gpu`。

配置了 `environment.wsl_distribution` 时，Windows 入口自动调用配置的 WSL 环境；安装后保留 WSL 路由配置。现有 CUDA13 环境不会因代码更新而自动重新安装。

数据、编码和解码采用分层任务指纹。仅修改报告或普通日志不触发重测；修改解码时可复用编码记录和压缩帧。默认 `retain_frames = true`，需要节省空间时可在 `[benchmark]` 中设为 `false`。旧任务指纹不会自动迁移。

详见 [INSTALLATION.md](benchmark/compression_framework/INSTALLATION.md) 和 [MEASUREMENT_CACHE.md](benchmark/compression_framework/MEASUREMENT_CACHE.md)。


## GPU batch configuration

`benchmark/settings.txt` controls encoding and decoding independently:

```ini
[benchmark]
encode_batch_blocks = max
decode_batch_blocks = max
max_batch_bytes = max
max_batch_memory_fraction = 0.5
```

Numeric batch sizes from 1 to 4096 request logical pages, without a fixed 64 MiB
encoder-frame cap. GPU encoding joins pages into independently decodable frames;
GPU decoding groups complete frames, so an indivisible frame can exceed a smaller
decode request. CPU APIs remain page-at-a-time, with optional CPU encoding threads.

`max` calibrates each codec and dataset within a conservative VRAM budget. With
`max_batch_bytes = max`, the raw-byte candidate budget is half the free VRAM
divided by an eightfold working-memory allowance. It finds a sampled capacity
limit, not the throughput optimum. Calibration, transfers and correctness checks
are excluded from codec timing. Actual selected sizes participate in layered
fingerprints and are recorded in logs, CSV and offline HTML. Heatmap cells show
regular and final-call sizes; units appear in codec headers.

A complete portable GPU configuration is available in
[settings-max.example.txt](benchmark/settings-max.example.txt). Install the selected
codecs through `setup`, then copy its dataset and batch sections into the managed
`settings.txt`; retain the environment selected by setup. Large SplitZip frames
validate each buffer against the expected output dimensions instead of the legacy
64 MiB segment cap.

## Verified RTX 5090 batch experiments

[Eight full batch experiments](results/2026-11-27-5090-batch-sweep/README.md)
contain 1,888 byte-verified results covering six BF16/FP8 datasets. Published CSV,
HTML, plots, portable configurations and checksums exclude input tensors, virtual
environments, compressed-frame caches and checkpoint journals. Historical numeric
experiments retain their measured frame limits; their directory names are requests,
not guaranteed actual batch sizes. Codec throughput and page throughput must be
interpreted separately.

## Automatic byte-aligned KV pages

The benchmark can choose equal full-page byte sizes across selected datasets of
one stored format, while retaining complete KV tokens and FP8 scales:

```ini
[benchmark]
LCM_BF16 = 32
LCM_FP8 = 1
```

Each value is a non-negative integer. Zero keeps the dataset's `page_tokens`;
a positive value multiplies the least common multiple of the participating
per-token byte counts. BF16, FP8 E4M3FN and FP8 E5M2 form separate alignment
groups; the two FP8 groups share the FP8 multiplier. Tail pages retain their
actual lengths. Enabled alignment requires KV layout. The saved legacy `LCM`
setting remains a fallback when a format-specific setting is absent.

For the supplied DeepSeek/Qwen data, BF16 token payloads are 1,024/4,096 bytes,
so multiplier 32 selects 128 KiB pages (128/32 tokens). FP8 token payloads,
including scales, are 516/2,080 bytes, so multiplier 1 selects 268,320-byte pages
(520/129 tokens). These values depend on the selected tensors: selecting only
one dataset can change its group's least common multiple. Alignment is a
comparability rule, not an automatic throughput optimization.

Resolved `page_tokens` participates in the existing layered data fingerprint;
CSV and HTML report the resolved page bytes and actual measured batches.
Changing settings does not change an already-running process's saved experiment.
