# GeoRA：精度调查与两平台实验

日期：2026-10-11。本文集中解释本轮比较对象、实际计算、结果和训练建议。完整回答另见 [十题输出对照](reports/precision_investigation/gsm8k_precision_full_answers.html)；引用与源码细节见文末。

## 1. 这次有两个实验，模型身份不同

### 实验 A：精度和 KV cache 会不会改变完整回答？

使用一个**冻结的原始 Qwen2.5-1.5B-Instruct**，没有装入 GeoRA，没有训练更新。

同样的 10 道 GSM8K 题，在 Metal 和 AMD 上各运行四次：

| 条件 | 冻结权重和主干计算 | 如何得到下一个 token |
|---|---|---|
| BF16、有 cache | BF16 混合精度 | Transformers 保存前文的 K/V，继续计算新位置 |
| BF16、无 cache | BF16 混合精度 | 每次重新计算题目与已生成的完整前文 |
| FP32、有 cache | 相同权重值提升 FP32 | 同一个 Transformers cache 实现 |
| FP32、无 cache | 相同权重值提升 FP32 | 同一个无 cache 实现 |

每次选概率最高的 token，直到第一个 EOS。没有随机采样；不输入标准答案。最大允许 1024 个新 token，若到上限还没 EOS 就报不完整，不能把截断冒充完整回答。

四种设置保持权重数值一致：先加载 BF16 checkpoint，再逐参数核对准确提升 FP32。提升不会恢复原 checkpoint 已丢失的位数。两个平台使用同一 checkpoint revision、同一准备好的 smoke 前十题及提示词，排除 final test。batch=1；eager attention；关闭 dropout、编译及生成惩罚。这里 cache 是 **Transformers DynamicCache，不是 vLLM，也不是自己实现 K/V 存储**。

无 cache 在本轮指原生 `generate(use_cache=False)` 的**逐渐增长的前文**；之前训练验收的 `fixed_width` 始终保持固定宽度。两者不可混称。最后的概率摘要使用 FP32 logsumexp；BF16 内部仍有框架自己的 FP32 RMSNorm、RoPE 和 attention softmax。

每次保存完整文字、实际 token ID、EOS、用时和下一 token 的前八名 logits/概率。两个回答第一次选出不同 token 时，它们此前的文字相同；从那以后前文已经不同，不能把后续不同概率全部视为“同一输入下的数值误差”。

本轮软件也有差别：Metal PyTorch 2.14.1，AMD PyTorch 2.7.1a0/HIP 6.3.42134，Transformers 均 5.19.0。因此这是**现有两套平台环境的比较**，不是只改变芯片的一项纯硬件实验。

### 实验 B：GeoRA 改用 FP32，训练开销多大？

使用未训练的 GeoRA 初始化，r=16、alpha=32、rho=0.2、196 个目标层，18,464,768 个可训练 A/B 参数。

沿用我们已经验证的计算方式：

$$W_{\mathrm{eff}}x=W_{\mathrm{pre}}x+c[B(Ax)-B_0(A_0x)],\qquad c=\alpha/r.$$

A/B、A0/B0、两条低秩分支及其相减、A/B 参数梯度、AdamW moment、log-softmax 和 loss 保持 FP32。比较只改变**冻结主干的存储和计算精度**。BF16 用 autocast；FP32 关闭 autocast。两组冻结权重数值一致。

取以前已保存的四段真实 GSM8K 回答，每段最多取前 64 个 token 作为计算负载，总输入形状 `[4,175]`、实际 248 个有效回答 token。优势值来自原完整回答，保持固定。这不是重新采样，也不是用这些截短回答测任务正确率。

每个精度各用独立进程，预热两次、正式测三次。每次恢复相同初始 A/B、清空梯度、归零 AdamW moment/step；执行实际 GRPO loss、反向传播、梯度裁剪和 AdamW 首次更新。前向前后和更新结束均同步 GPU。模型加载、恢复初值、生成回答、SVD 和文件保存不计入计时。

这里只有**一个 policy 常驻**，reference 项用同条件初始旧概率作为固定 anchor，没有另驻原始 reference 模型，也没有计它的前向。因此以下结果是 policy 更新成本，不是完整 GRPO 一轮的总成本。均值为零的优势使初始 loss=0，但实际梯度非零；0 loss 不等于没有更新。

## 2. AMD 实际训练开销

硬件：一个逻辑 AMD Instinct MI250X，约 64 GiB 显存。无 gradient checkpointing，fixed-width 无 cache，AdamW foreach=False。

| 测量，中位数 | BF16 混合精度 | FP32 主干 | FP32 / BF16 |
|---|---:|---:|---:|
| 前向及 loss | 0.2196 秒 | 0.3654 秒 | 1.664 |
| 反向、裁剪、AdamW | 0.1719 秒 | 0.2345 秒 | 1.365 |
| 合计 | **0.3914 秒** | **0.6001 秒** | **1.533** |
| 实测 GPU tensor 峰值 | **6.354 GiB** | **9.941 GiB** | **1.564** |

所以本负载下 FP32 慢约 53%，峰值多 3.587 GiB。只看冻结权重存储，多 2.875 GiB；额外的激活等令实测峰值差更大。再常驻一个 FP32 原始 reference 相比 BF16 reference，权重自身还要多 2.875 GiB。**总训练显存并不必然恰好翻倍。**

这些数不能直接外推到 8 个回答、256/4096 个 token、vLLM、可微 cache 或长训练。以前我们的可微 cache BF16 运行峰值已到 42.47 GiB；该条件改 FP32 是否能装下，本轮没有测量。

CPU 小 Qwen 协议 75/75 通过；对实际 AMD 两份报告独立核对 59 项，包括相同输入、冻结数值、初始化、优势、形状、源码和统计重算，全部通过。两份原始报告在各自执行进程内记录了六份源码哈希；同作业推理阶段又在 login CPU 做外部独立复核。远端原始报告与本地保留副本的字节 SHA256 完全一致，没有后补字段。

## 3. 十题完整输出

完整文字、数字结果、格式失败和 token 一致性统一呈现在 [HTML 对照](reports/precision_investigation/gsm8k_precision_full_answers.html) 和 [逐题概览](reports/precision_investigation/gsm8k_precision_summary.md)。全部 **80 个回答自然生成到 EOS，没有截断**。逐 token 比较包含 EOS；四条 FP32 路径在十题上的完整 token 序列全部相同。

| 比较 | 完整 token 序列相同的题数 |
|---|---:|
| Metal：BF16 有／无 cache | 1/10 |
| AMD：BF16 有／无 cache | 3/10 |
| Metal：FP32 有／无 cache | 10/10 |
| AMD：FP32 有／无 cache | 10/10 |
| Metal／AMD：BF16 有 cache | 3/10 |
| Metal／AMD：BF16 无 cache | 3/10 |
| Metal／AMD：FP32 有 cache | 10/10 |
| Metal／AMD：FP32 无 cache | 10/10 |

几个实际题目与最后给出的数字如下，完整推理正文保留在 HTML：

| 实际题目 | 标准数字 | Metal BF16 cache／无 cache | AMD BF16 cache／无 cache | 四条 FP32 路径 |
|---|---:|---|---|---|
| 5000 平方英尺占二楼的四分之一，一楼面积是二楼两倍；还剩多少储物面积？ | 55000 | 15000 / 55000 | 15000 / 55000 | 均为 55000 |
| Boudreaux 接到 12 条项链，Rhonda 是他的一半，Latch 是 Rhonda 的三倍少 4 条；Latch 有多少？ | 14 | 14 / 18 | 12 / 12 | 均为 14 |
| Bob 每小时加薪 0.50 美元，每周 40 小时，住房补贴每月减少 60 美元；每周净增多少？ | 5 | 15 / 15 | 15 / 15 | 均为 5 |
| 每 4 粒种子产一穗、每穗卖 0.1 美元、100 粒种子成本 0.5 美元；赚 40 美元要卖多少穗？ | 500 | 900 / 400 | 900 / 400 | 均为 800，仍然错误 |

上表的问题是简要概括；完整题目以 HTML 原文为准。FP32 路径的严格得分都是 6/10：一个数字错误，三个格式解析失败。BF16 四条路径分别是 2/10、3/10、1/10、2/10。这里展示差异，不用十个预选训练题推断总体正确率。

重要的区别：**完整 token 一致不等于 logits／概率逐数值一致；FP32 不保证所有题算对。** 本轮是独立贪心生成。分岔前才是相同前文；分岔后的概率无法继续按同一前文成对比较。原先强制同一旧回答的 103-token 数值实验仍需单独保留。

独立检查已验证 80 条 token 解码与保存正文完全相同，HTML 80 段文字逐字符保留，评分和汇总计数与独立重算一致。原始 JSON 的旧辅助比较字段曾把两个无法解析的结果算作“数字相同”（Metal 10 条、AMD 8 条）；原记录保留，最终报告完全从 token 和严格评分重算，未来脚本已修正该字段，不需要重新推理。

严格 reward 仍沿用我们的小实验规则：必须恰好一个 `####`，后面全部是一个数字。`#### 150 tires` 会失败，尽管读者可能理解它想回答 150。格式失败与已解析出的错误数字分开列；两个解析失败不能算“数字答案一致”。这不是作者未公开的 exact parser，也不是标准 GSM8K 大样本测评。十题预选 smoke 数据不足以判断某精度总体更好。

## 4. 论文和官方代码到底用什么精度？

| 工作 | 已明确的内容 | 不能据此推断的内容 |
|---|---|---|
| GeoRA v4，C.1 | AdamW、BF16 训练、80GB GPU | 未分开报告 SVD、A/B 保存、optimizer、softmax、生成端 dtype |
| PiSSA 原论文 §5、D | LoRA/PiSSA 的 base 与 adapter 用 FP32；FullFT 用 BF16；做过精度消融 | FP32 并不在每个模型上都提高分数 |
| PiSSA 当前官方代码 | BF16 数学训练范例；PEFT 的 SVD 和残差构造先 FP32 | 当前代码范例不等于原论文的 FP32 设置 |
| MiLoRA 官方实现 | FP16 base、FP32 SVD/因子；数学训练开启 BF16 | 不能简称所有环节 BF16 |
| The Path Not Taken | 专门分析 BF16 如何隐藏微小权重更新；优化器、梯度归约/累积 FP32 | “可见更新稀疏”不是与保存精度无关的事实 |
| RL's Razor | FP32 消融中可见稀疏消失，作者的任务表现未改善 | 不能泛化为所有 LLM/RL 精度都无关紧要 |
| Evaluating PEFT for RLVR / PeRL | 官方配方 BF16 混合精度，PiSSA/MiLoRA 初始化含 FP32 SVD | 当前仓库分派和配置与论文有差异，exact run 未确认 |
| Small Subnetworks | BF16 检查点分析；DPO BF16 混合精度 | 完整 PRIME optimizer/rollout dtype 表未提供 |
| GRPO / DeepSeekMath、R1 | 未找到明确的 RL 训练 dtype 表；Math 官方推理范例 BF16 | 推理代码不能证明训练全程 BF16 |

GeoRA 的 BF16 声明见 [C.1](https://arxiv.org/html/2601.09361v4#A3.SS1)；PiSSA 原始精度与消融见 [§5](https://arxiv.org/html/2404.02948v4#S5)、[Appendix D](https://arxiv.org/html/2404.02948v4#A4)。PEFT 官方文档也明确适配器参数通常保存 FP32：[adapter dtype](https://huggingface.co/docs/peft/developer_guides/troubleshooting#selecting-the-dtype-of-the-adapter)。

与当前问题最直接相关的补充工作是 [Defeating the Training-Inference Mismatch via FP16](https://arxiv.org/html/2510.26788v1)：比较 BF16、FP16、FP32，还含 LoRA RL 实验，报告其设置下 FP16 改善生成和训练概率一致性与稳定性。它不是“任何模型都应该 FP16”的结论；其官方补丁还增加了 gradient scaling。论文中的 FP32 **推理**速度不是我们的 AMD **训练**测量。

[MiniMax-M1 §3.2](https://arxiv.org/html/2506.13585#S3.SS2) 有只把 LM-head 提升 FP32 的成功案例；其模型/根因与我们不同，不能直接套用。[verl 官方文档](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html)明确讨论相同权重因精度、内核或实现不同产生的概率差。**大家先验上确实在意这个问题。**

### 两种精度问题要分开

1. **保存权重的舍入**：小更新存成 BF16 后可能看不见。The Path Not Taken 的 precision gate 主要讨论它。
2. **计算过程的舍入**：权重完全一样，cache 和无 cache 的矩阵计算形状不同，logits 也可能不同。我们此前 19.7% 概率差和本轮完整回答实验讨论它。

把检查点提升 FP32，不会补回已丢失的权重信息；但后续运算更精细，所以仍能显著减少第二类差异。softmax 改成 FP32，无法补回输入 logits 中此前已经发生的舍入。

## 5. 当前建议

**正式 GeoRA/LoRA 论文对照继续用已经验证的 BF16 混合精度，把 FP32 保留为独立的数值及短训练对照。** 初始化 SVD、A/B/A0/B0 保存、低秩差分支、A/B 参数梯度和 AdamW 保持 FP32。十题的输出差异足以说明要监测精度，却不足以根据几道题的得分改变主实验设置。

对于我们的小模型，FP32 的实测代价可以承受；有价值的下一项是同一问题集、数据顺序和更新预算的短训练对照，重新生成回答，验证改变精度是否改变 reward、KL、熵和梯度，而不是只看文本是否一样。生产长回答先验证容量，不能从 `[4,175]` 的 9.94 GiB 推断任意配置都能装下。

生成回答与训练计算已选 token 的概率，要共用已验收的执行路径：无 cache 两端都 fixed-width；或两端都用已验证的可微 cache。**不要把 BF16 cached 生成和另一条无 cache 路径的概率混用，也不要把提高精度当作路径一致性的替代品。** FP32 仍可能有小差异，已有 AMD 103-token 实验中 5 个位置仍超过原严格 ratio 门限。

FP16 是有文献支持的候选对照，不直接切换正式训练：它范围更窄，需 gradient scaling 和溢出检查。当前不需要全模型 FP64。

## 6. 保存、重新执行与资源

本轮 Setonix job `50670799`：COMPLETED、0:0，一个逻辑 GPU，实际 **430 秒（7分10秒）**，上限 20 分钟；完成后队列为空。训练和十题推理共用这次 allocation。数据压缩/传输、文献调查和 CPU 验证不申请 GPU。

资源累计：23 个已记录作业，62分53秒 allocation，不包含 login CPU，也不是 8-GPU node-hours。

本机命令：

```bash
uv run python scripts/compare_gsm8k_precision.py --device mps \
  --output-dir outputs/gsm8k_precision_comparison/new-run
```

AMD 作业入口：[jobs/gsm8k_precision_comparison.sbatch](jobs/gsm8k_precision_comparison.sbatch)。它顺序运行两种 GeoRA 训练成本测量和原模型十题推理，结束自动释放。

证据：原始双精度训练报告、独立汇总、CPU 协议在 `reports/precision_investigation/`；未训练初始化和原 checkpoint 保留，真实长训练未启动。完整来源调查：[GeoRA/初始化基线](GEORA_PRECISION_SOURCES.md)、[RLVR/机制及工程](RLVR_PRECISION_SOURCES.md)。
