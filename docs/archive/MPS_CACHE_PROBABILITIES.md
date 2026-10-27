# 本地 Metal：同一前文、同一实际 token 的概率比较

2026-10-11。使用本地 MPS，没有提交 Setonix 作业。本页是数值检查，没有训练更新，也不是任务准确率比较。

## 1. 到底在比较什么？

一个冻结的原始 Qwen2.5-1.5B-Instruct，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。没有安装 GeoRA 或 LoRA。

使用 [Setonix 原报告](../../reports/cached_logits/50659217.json) 中实际采样出的八个回答：两种问题，各四个回答。**本轮没有在 Metal 重新采样。** 同一回答的前文、下一 token 和有效长度直接复用原报告。

- 有 cache：通过 Transformers 原生 `DynamicCache`，prompt 宽度 50，随后逐 token 计算。用原生 LogitsProcessor 强制沿已有 token 序列前进。`lm_head` hook 在强制选择之前保存真实 logits，因此保存的模型概率不是强制选择后的 1。
- 无 cache：同一批 token IDs 和有效 mask，固定宽度 66，一次计算 decoder；每个位置分别用相同形状的 `lm_head` 计算整个 vocab 的 logits。因果 mask 排除未来 token。
- 两组真实 logits 分别移到 CPU，用同一套 FP64 log-softmax 得到选中 token 的概率。概率分母使用整个 151,936-token vocab。
- 比较 103 个有效 token，包含第一个结束 token，排除结束后的 padding。模型参数不变，无梯度或 optimizer step。

## 2. 真实输入、回答与条件概率

**Q1:** `What is 7 + 5? Explain briefly.`

所选实际回答：`7 + 5 = 12\n\nThis is a simple addition problem.`

**Q2:** `A box has 3 red balls and 5 blue balls. How many balls are there in total?`

所选实际回答：`There are 8 balls in total.`

表中前文是 assistant 已写出的部分，另外还包含固定的 system、完整问题及聊天模板。`\n` 表示换行。token 中的前导空格保留，`<|im_end|>` 表示回答结束。

| 问题 | 已生成的回答前文 | 实际采样的下一个 token | Metal 有 cache | Metal 无 cache | 相对差：无/有 − 1 |
|---|---|---|---:|---:|---:|
| Q1 | `7 + 5 = ` | `1` | 99.999815% | 99.999826% | +0.000011% |
| Q1 | `7 + 5` | ` =` | 11.016237% | 9.938952% | -9.779065% |
| Q1 | `7 + 5 = 12\n\n` | `This` | 13.372805% | 13.866548% | +3.692137% |
| Q2 | `There are ` | `8` | 43.706188% | 43.701235% | -0.011333% |
| Q2 | `There are 8 balls in total.` | `<|im_end|>` | 34.102923% | 38.481564% | +12.839487% |

不是每个 token 都有大的概率差。数字 `1` 几乎不变，而该 Q2 回答的结束 token 从约 34.10% 变成 38.48%，绝对差约 4.38 个百分点，相对差约 12.84%。这也是本轮 103 个 token 中最大的绝对相对比例差。

## 3. 为什么本地可以做，但不能替代 AMD 的原记录？

同一个 Q1 前文 `7 + 5 = 12\n\n`，同一个下一 token `This`：

| 平台 | 有 cache | 无 cache | 相对差 |
|---|---:|---:|---:|
| Setonix MI250X 原记录 | 12.628126% | 15.115552% | +19.697502% |
| 本地 Metal 本轮 | 13.372805% | 13.866548% | +3.692137% |

两边的模型 revision、已有 token 序列和权重 BF16 数值一致，但底层算子及 PyTorch 版本不同。因此本轮证明 Metal 也能进行这种检查并出现差异，不能据此把差异唯一归因于硬件或声称 Metal 必然更精确。

此前 [投影定位](CACHED_PROJECTION.md) 已在 AMD 上确认：固定相同输入、权重和 bias，仅改变 V 投影的计算分块，就精确恢复六处历史 V 差异；cache 存储保持投影结果。原最差 token 在相同 BF16 权重数值提升 FP32 后，相对概率差从约 19.6975% 降到约 0.0032728%。这些是低精度数值计算参与差异的直接证据。

完整 19.7% 的逐层传播还没有唯一归因；具体 GPU kernel 的内部累加顺序也没有测出。局部 V 差异不能单独解释全部最终差异。这次八段回答的概率比较尚未进行 Metal 逐层定位或 FP32 全模型对照；同日后续已完成，详见 [本地逐层传播实验](LOCAL_ERROR_PROPAGATION.md)。

## 4. 精度、用量与检查

- 本地 PyTorch `2.14.1`，Transformers `5.19.0`，设备 MPS。
- 模型加载到 CPU FP32，再将冻结权重转 BF16 并移到 MPS。已保存的 BF16 权重提升 FP32不会恢复历史舍入掉的信息。
- forward 使用 BF16 autocast，内部归一化等精度保留框架原设置。
- logits 从 BF16 精确提升 FP32 后移到 CPU；最终比较在 CPU FP64 进行。没有在 MPS 执行 FP64 全模型。
- 报告的 `status=passed` 表示八项诊断检查通过，**不表示有无 cache 的概率相等，也不表示修复了原问题**。
- 可复用脚本执行时间约 5.68 秒；此前临时入口约 8.88 秒。两次都是本地运行，新增 Setonix allocation 为 0。
- 完成时 MPS tensor 内存约 2.91 GiB、driver 内存约 3.10 GiB。这是结束时占用，不能称作峰值；也不包含整个 Python 进程的 CPU 内存。

## 5. 复查

- [可运行脚本](../../scripts/check_cached_probabilities_mps.py)
- [全部 103 个实际 token 的前文与概率](../../reports/cache_probabilities_mps/20261011.json)
- 原最差位置的完整 vocab logits 在 Git 忽略的 `outputs/cache_probabilities_mps/20261011/original_AMD_worst_position_Metal_logits.safetensors`。

```bash
uv run python scripts/check_cached_probabilities_mps.py   --output-dir outputs/cache_probabilities_mps/20261011
```

需要本地 MPS、已下载的固定 revision checkpoint，以及仓库里的 Setonix 原报告。该脚本不提交 Slurm 作业。

## 6. 后续：本地逐层定位已完成

同日使用同一冻结模型和已有 token，进一步记录两个预测位置的 28 层状态，并做逐层有效前文输入对齐、第 0 层 MLP 同输入形状对照、相同权重值 FP32 对照。最终一轮本地 Metal 检查 40/40，通过用时 49.18 秒；没有 Setonix 作业或参数更新。

`This` 的早期持续差异可在第 0 层 MLP 的相同输入值上，仅改变计算形状而精确重现；第 1 层对已有输入差异的响应占较大部分，后续传播并非逐层单调增长。BF16 的 `This` / 结束 token 相对概率差为 +3.692137% / +12.839487%；保留同样权重数值、FP32 计算时为 -0.000918% / -0.000550%。

每一步的比较对象、输入替换的定义、真实 logits 与 softmax 分母的变化、精度和原始证据集中写在 [LOCAL_ERROR_PROPAGATION.md](LOCAL_ERROR_PROPAGATION.md)。这些本地结果没有替代 AMD 19.70% 的原记录，也没有修改训练方案。
