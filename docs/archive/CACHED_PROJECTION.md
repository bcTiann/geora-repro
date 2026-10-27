# KV cache 差异：历史 V 是在哪里变得不同的？

2026-10-11，L17/S16。接续 [逐层定位](CACHED_LAYERS.md) 和 S14 的 attention 干预。

**本轮证明：原来第 0 层的六处历史 V 差异，在投影计算时就已经产生。原生 KV cache 的存入、追加和读取保持了这些值。固定同一份输入与权重，只改变计算分块，就精确重现六个差异。**

这定位了一个早期机制，还没有把最终约 19.7% 的 token 概率差唯一归因于它。

## 1. 比较的模型和输入

只有一个冻结的原始 `Qwen/Qwen2.5-1.5B-Instruct`，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。没有装入 GeoRA/LoRA，没有 reward、backward 或 optimizer step。KV cache 和 `generate` 使用 Transformers 原生实现；没有 vLLM 或 PiSSA。

复用 S12 保存的 token IDs，检查原来差异最大的 `This`：row=1、completion step=9、token ID=1986（索引从 0 开始）。两条路径使用同一个 batch，8 行；prompt padded width=50，目标前文 padded width=59，fixed width=66。目标行有 49 个可见有效历史 token。

| 路径 | 第 0 层 V 投影的实际输入形状 |
|---|---|
| 原生 cache | prompt `[8,50,1536]`，随后九次 `[8,1,1536]` |
| 固定宽度无 cache | 一次 `[8,66,1536]` |

参数及 forward 使用 BF16，native normalization 等内部操作保留框架自己的精度设置。原始 BF16 输入、V 权重、bias、输出和 cache 内容保存为原 dtype；提升 FP64 只用于比较与六个点积的本地分析，没有执行 FP64 全模型 forward。

## 2. 检查的程序流程

```text
同一原模型，固定已保存的 token 序列
    ↓
记录第 0 层 v_proj 的实际输入和输出
    ↓
记录 attention 从原生 cache 读到的历史 V
    ↓
验证：投影输出拼接、reshape 后，是否等于读出的 V
    ↓
固定同一份输入、权重、bias，按两种形状复算 v_proj
    ↓
重复同形状计算；检查是否精确恢复原生结果
    ↓
仅对六个不同的坐标，用 CPU FP64 计算点积＋bias
```

首先精确恢复 S12 的两组原始 logits，所以本轮定位的是以前那次差异。实际 token/position IDs 和目标可见前文的检查通过；模型保持 eval、无 dropout、无参数更新。

## 3. 实测结论

| 检查 | 结果 |
|---|---|
| 全 batch 的历史投影输入 | 724,992 个元素逐元素相同 |
| 原始投影输出到 cache 读取 | 120,832 个 V 元素逐元素相同 |
| 同输入、同权重，复现原计算分块 | 精确恢复 cached 和 fixed 的投影输出，包括原六个不同坐标 |
| 同形状重复计算 | 逐元素相同 |
| 连续内存复制对照 | 输出相同；实际输入本来就是连续、offset=0、256 字节对齐 |
| 原目标行有效历史 V | 12,544 个元素中六个不同，最大差 `0.00048828125` |

**六个差异全部来自 prompt 的 prefill**，对应 padded columns 11、12、30、33、34、44。它们比较的是 prompt 宽度 50 与固定宽度 66 的投影，不是后续单 token decode 或 cache 追加把值改坏。

V 投影本质上为同一 hidden vector 计算同一矩阵乘法和 bias。实数运算下，改变计算分块不应改变这个结果；本轮的浮点实测结果会改变。检查记录了形状和 stride，并精确复现了两组原生输出，但没有查验具体 GPU kernel 或确定其内部求和顺序，因此不称它为 AMD kernel bug。

## 4. BF16 舍入分界的具体例子

V 的 channel 57，在三个换行 token 位置上得到相同的 FP64 参考点积＋bias：

```text
FP64 参考值：              -0.09301755059277639
两个相邻 BF16 数的中点：   -0.093017578125
两者距离：                  0.0000000275322236

两条原生路径得到的 BF16 值：
                            -0.0927734375
                            -0.09326171875
```

参考值非常接近两个相邻 BF16 数的舍入分界。这支持以下机制：形状相关的细小计算差异，使结果落到分界的不同侧，最终保存成不同的 BF16 数。没有捕获 GPU 内部未舍入累加器，因此这一解释不等于已经测到具体累加顺序。

六对原生结果中五对相邻；另一个近零值不是相邻对，其两值中点不是舍入分界。该位置有明显相消：`0.0444419187 + (-0.04443359375)`。不能用同一个“跨相邻舍入分界”的故事解释全部六个值。

下载原始 snapshot 后，本地 NumPy FP64 点积和 `math.fsum` 对六个坐标的复查，与 GPU 报告中使用 CPU FP64 得到的结果逐项一致。这里的 FP64 参考使用原始已保存的 BF16 数值，没有重新读取捨入前的权重。

## 5. FP32 对照能否消除差异？

同一份 BF16 输入、权重、bias 精确提升到 FP32，执行 `F.linear`，再转 BF16：原六处中五处的差异消失。但全部有效历史仍有七处更小的差异，最大差为 `7.62939453125e-6`。因此不能说“提升 FP32 再转 BF16 就完全一致”，也不能说每个局部值都比原 BF16 结果更接近 FP64。

这只是局部投影对照，没有据此修改训练精度或训练路径。

## 6. 与最终 19.7% 的关系

这轮精确恢复原始两组 logits，`This` 的 fixed/cache 概率比仍为 `1.1969750175109515`。19.7% 是一个 token 的概率相对差，发生在同一原模型的两种执行路径之间，不是 GeoRA 与原模型的准确率差。

S14 已经证明：固定 attention 概率，仅替换历史 V，可以精确恢复该位置的第 0 层局部 attention 输出差；但只替换这一局部输出后，最终仍有约 19.27% 的概率差。本轮解释了六处历史 V 的来源，没有再声称它们解释整个最终差异。

已经通过验收的训练方案仍是让自回归采样与固定序列的概率计算使用相同执行路径；缓存和无缓存两种路径各自可以对齐。此前两条路径之间的 logits 差异仍然存在。参见 [训练路径及五步验收](CACHED_TRAINING.md)。

## 7. 检查、用量与可复查文件

| 执行 | 检查结果 | 资源 |
|---|---|---|
| 本地 tiny Qwen | 13/13 | CPU 两线程，无 GPU |
| Setonix login tiny Qwen | 13/13 | CPU 两线程，无 Slurm CPU allocation |
| MI250X 原模型，job `50665075` | 17/17 | 一个逻辑 GPU；Python 14.819 秒；allocation **36 秒**；两分钟硬上限 |
| 下载 snapshot 后独立复查 | 六个坐标和 cache 保存关系通过 | 本地 CPU，无 GPU |

Slurm 为 `COMPLETED, 0:0`，结束后的用户队列为空。临时模型副本已删除；原 checkpoint、初始化和远端原始诊断 snapshot 保留。累计已记录 21 个作业 allocation 为 3307 秒，即 55 分 7 秒；不含 login CPU，也不等于 utilization 或最终计费金额。

- [GPU 原始 JSON](../../reports/cache_projection_origin/50665075.json)
- [本地 CPU](../../reports/cache_projection_origin/cpu_local.json)、[Setonix CPU](../../reports/cache_projection_origin/cpu_setonix.json)
- [下载后独立复查](../../reports/cache_projection_origin/snapshot_verification.json)
- [模型副本校验](../../reports/cache_projection_origin/staging_record.json)、[脚本来源与清理](../../reports/cache_projection_origin/source_provenance.json)
- [捕获和形状对照代码](../../scripts/cache_projection_origin.py)、[入口](../../scripts/check_cache_projection_origin.py)、[Slurm 作业](../../jobs/cache_projection_origin.sbatch)

原始快照 SHA256：`6ab0d5a062230dff6ba9086f1a3c28ca4bbaf1f57909afddcf25f34a14d7fc77`。远端路径：`/scratch/pawsey0807/btian/geora/runs/projection-origin-50665075/projection_origin.safetensors`。本地保存在忽略 Git 的 `outputs/cache_projection_origin/50665075/`，不把模型权重片段加入 Git。

GPU JSON 的 `source_git_commit=b5bee415...` 是当时基础 checkout 的版本，新诊断脚本是单独上传的文件。四份实际上传文件的 SHA256 已与远端文件核对，见来源记录；不能把基础 checkout commit 当成这些新文件的来源。
