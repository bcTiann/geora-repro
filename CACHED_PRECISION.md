# AMD：同一权重数值，BF16 与 FP32 的有无 cache 概率差

2026-10-11，Setonix AMD Instinct MI250X，job `50668984`。一个冻结的原始 Qwen2.5-1.5B-Instruct；没有 GeoRA／LoRA，没有新采样、奖励或参数更新。

## 1. 本轮回答什么问题？

本地 Metal 的对照发现：保持权重数值和前文不变，把模型从 BF16 计算改成 FP32，能大幅降低有无 KV cache 的 token 概率差。本轮在 AMD 上检验同一问题，并将旧 AMD 单位置对照扩展到全部 **103 个有效回答 token**。

比较对象始终是：同一个模型、同一段前文、同一个下一 token，在两种执行方式下得到的条件概率。

$$
\text{概率相对差}=\frac{p_{\mathrm{无\ cache}}(v\mid\text{前文})}
{p_{\mathrm{有\ cache}}(v\mid\text{前文})}-1.
$$

不是 GeoRA 与原模型之间的差，也不是把 BF16 的概率直接减去 FP32 的概率。BF16 和 FP32 各自在本精度内比较有无 cache。

## 2. 四次计算，共用同一个冻结模型

1. 从固定 revision 的 checkpoint 载入原模型，权重转 BF16。
2. BF16 有 cache：使用 Transformers 原生 `DynamicCache`，按旧 token 序列逐步前进。
3. BF16 无 cache：同一 batch 固定宽度 66，用因果 mask 计算；分别取得同一预测位置的 logits。
4. 将当前模型全部 BF16 权重值原样提升 FP32。逐参数验证 **1,543,714,304** 个值全部不变，没有重新读取更高精度权重。
5. 关闭 BF16 autocast，以 FP32 重复有 cache 和无 cache 两种计算。
6. 两精度、两种执行方式的真实 logits，统一移到 CPU 用 FP64 full-vocab log-softmax 计算概率。

固定 [旧 AMD 报告](reports/cached_logits/50659217.json) 的八段回答，两个问题、各四段。没有重新随机生成；强制沿旧 token 序列前进的操作在捕获真实 logits 之后执行。

103 个位置包含回答的第一个 EOS，排除 EOS 后的 padding。核查所有有效预测前文的 token IDs、position IDs 和传入模型的二维 attention mask 一致；本轮没有逐层捕获全部 materialized attention masks。

## 3. 真实例子一：预测 `This`

```text
问题：What is 7 + 5? Explain briefly.
回答已经写到：7 + 5 = 12\n\n
比较下一个 token：This
```

| 同一前文下 `This` 的概率 | 有 cache | 无 cache | 无 / 有 − 1 |
|---|---:|---:|---:|
| BF16 计算 | 12.62812628% | 15.11555167% | +19.69750175% |
| 同样权重数值，FP32 计算 | 13.75969281% | 13.76014314% | +0.00327280% |

差异约缩小 **6018.54 倍**。本次 BF16 两组完整 vocab logits 与原最差位置的旧 snapshot **逐元素完全相同**；FP32 的数值也重现此前 job `50660028` 的单位置对照。

同样权重值，不同计算精度，得到的概率本身可以变化；本轮关心的是每种精度内两条执行方式是否趋近。

## 4. 真实例子二：预测回答结束

```text
问题：A box has 3 red balls and 5 blue balls. How many balls are there in total?
回答已经写到：There are 8 balls in total.
比较下一个 token：<|im_end|>
```

| 同一前文下结束 token 的概率 | 有 cache | 无 cache | 无 / 有 − 1 |
|---|---:|---:|---:|
| BF16 计算 | 33.80432709% | 35.59935756% | +5.31006125% |
| 同样权重数值，FP32 计算 | 31.74235930% | 31.74332618% | +0.00304603% |

这里是 AMD 的实际结果；不能把本地 Metal 的 12.84% 直接套到 AMD 上。

## 5. 全部 103 个有效 token

| 指标 | BF16 | 同样权重值、FP32 |
|---|---:|---:|
| 最大绝对概率相对差 | 19.69750175% | 0.00327280% |
| 最大绝对 logp 差 | 0.1797975554 | 0.0000327275 |
| 平均绝对 logp 差 | 0.0113921317 | 0.0000029899 |
| 全部有效位置、整个 vocab 的最大 logits 绝对差 | 0.5 | 0.0001984835 |
| 超过旧 ratio 容差 $2\times10^{-5}$ 的 token 数 | 64 / 103 | 5 / 103 |

两种精度的最大相对差都发生在上面的 `This` 位置。本批次 103 个 token 的绝对相对差均减小。

因此，在这一批固定前文中，**提高计算精度确实在 AMD 上也大幅降低了两种执行方式的概率差**。它仍不是逐元素完全相等，也没有自动通过旧训练路径的所有概率一致性门槛。

这支持低精度浮点计算是本地及 AMD 已观察差异的主要因素。具体 GPU kernel 的内部累加顺序没有捕获；本轮也没有改变生产训练路径、检查 FP32 的训练成本或得到新的任务分数。

## 6. 用什么精度、保存什么？

| 阶段 | 精度 |
|---|---|
| BF16 模型计算 | 冻结参数 BF16、BF16 autocast；保留框架内部 FP32 归一化、attention softmax 等设置 |
| FP32 模型计算 | 当前 BF16 权重值精确提升 FP32；关闭 BF16 autocast |
| 保存真实 logits | CPU FP32；BF16 值提升 FP32 是精确的 |
| 计算 token 概率、logp 差与相对差 | 同一套 CPU FP64 full-vocab log-softmax |
| 独立复核 | 从保存的 logits 用 NumPy FP64 指数归一化复算，不运行模型 |

模型 revision：`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。PyTorch `2.7.1a0+gite2d141d`，HIP `6.3.42134-a9a80e791`，Transformers `5.19.0`。

## 7. 检查、资源和文件

- 完整 AMD 检查 **24/24**，核心 Python 用时 **9.925 秒**，peak allocated GPU tensor memory **5.911 GiB**。
- 本地、Setonix login 的两层小模型回放检查各 **18/18**，先通过再申请 GPU。
- 独立 NumPy 复核两份原始 logits 文件的哈希，并从完整词表计算两精度、两路径共 **412 个条件概率**，同时复核 206 个概率相对差。
- Slurm 状态 `COMPLETED`，exit `0:0`；上限 2 分钟，实际 allocation **36 秒**，`gres/gpu=1`。结束后用户队列为空。
- 临时模型副本在完成检查后删除；原 checkpoint、GeoRA 初始化和本次原始 logits 保留。

代码执行 commit：`f4ac79c`。报告还记录实际源文件的 SHA256，不只依赖 commit 名称。

- [GPU 完整结果](reports/cached_precision/50668984.json)：全部 103 个 token 的前文、真实概率和 softmax 分解。
- [独立 NumPy 复核](reports/cached_precision/numpy_verification.json)。
- [本地 CPU](reports/cached_precision/cpu_local.json)、[Setonix CPU](reports/cached_precision/cpu_setonix.json)。
- [用量和清理记录](reports/cached_precision/remote_final.json)。
- [GPU 入口](scripts/check_cached_precision_gpu.py)、[同 token 两路径比较](scripts/cached_precision_comparison.py)、[独立复核脚本](scripts/verify_cached_precision.py)、[两分钟单 GPU 作业](jobs/cached_precision.sbatch)。

两份完整 logits snapshots 保存在：

```text
/scratch/pawsey0807/btian/geora/runs/cached-precision-50668984/
    BF16_valid_logits.safetensors
    same_weight_values_FP32_valid_logits.safetensors
```

每份包含 103 个有效预测位置的两组 full-vocab logits。大张量未加入 Git；报告保留路径、字节数和 SHA256。无需重新分配 GPU，就可以在 login CPU 复查这些值。
