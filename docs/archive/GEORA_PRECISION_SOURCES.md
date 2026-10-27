# GeoRA 与最直接初始化基线的精度调查

核对日期：2026-10-11。只使用论文、作者官方仓库、Hugging Face PEFT 官方代码和文档。没有运行 GPU，没有改动 geora-repro 仓库。

## 结论先读

1. GeoRA 最新版 v4 的 Appendix C.1 明确报告 BF16 训练，但没有给出逐算子、初始化、优化器状态、概率计算或生成端的完整 dtype 表。不能把它改述成“所有东西全程 BF16”。
2. 最直接基线 PiSSA 的原论文明确采用 **base 与 adapter 都 FP32** 的 LoRA/PiSSA 实验，并且专门比较过 BF16 与 FP32。作者没有发现 FP32 总能带来更高任务分数。
3. PiSSA 当前官方仓库的训练范例已经使用 BF16，与论文原实验声明不同，应分别引用。其依赖 PEFT 0.14.0 的初始化会先在 FP32 做 SVD 与残差减法，再把残差转回原 dtype。
4. MiLoRA 官方初始化代码是 FP16 加载 base → FP32 SVD/构造 A、B → FP16 存残差、FP32 存 A/B；其数学任务训练脚本开启 BF16、TF32。
5. LoRA 的官方仓库明确提醒浮点精度会影响复现结果，并含 FP16 混合精度与未开启 FP16 的不同实验范例。LoRA 方法本身不规定唯一精度。
6. 官方 PEFT 默认将适配器参数保存为 FP32，也专门讨论半精度适配器的精度、速度、内存取舍。因此“先验上大家不在意精度”不成立。不过这些资料不能代替 GRPO 中生成与训练时概率一致性的具体检查。

## 1. GeoRA：论文明确说什么，没说什么

最新 arXiv 版本：v4，2026-07-31，15 页。不要使用缺少 Appendix C 的 v1 推断最新设置。

- [摘要页与版本记录](https://arxiv.org/abs/2601.09361)
- [v4 Appendix C.1](https://arxiv.org/html/2601.09361v4#A3.SS1)
- [v4 PDF，第 14 页](https://arxiv.org/pdf/2601.09361v4#page=14)

精度原文，19 个词：

> We train all models with AdamW in bfloat16 precision on 80GB GPUs.

| 环节 | 最新论文能确认的内容 | 证据强度 |
|---|---|---|
| 训练总设置 | AdamW，BF16，80 GB GPU | 明确报告，C.1 |
| SVD、mask、分位数计算 | 给数学算法，未明确 dtype | 未报告 |
| A0/B0 初始化计算与保存 | 给公式，未明确 dtype | 未报告 |
| 冻结残差计算与保存 | 式 4，未明确 dtype | 未报告 |
| base/adapter 参数各自 dtype | 没有分开列 | 未报告 |
| optimizer moments/master weights | 没有分开列 | 未报告 |
| normalization、softmax、log-probability、loss reduction | 没有逐算子精度表 | 未报告 |
| rollout 引擎、推理权重与 KV cache dtype | 没有足够实现细节 | 未报告 |
| reference 重建的数值误差容限 | 宣称数学初始化保持函数，未报告有限精度误差门槛 | 未报告 |

C.2 确认小模型设置是 Qwen2.5-1.5B-Instruct / GSM8K；这里训练仍遵循 C.1 的统一设置。主实验使用 Qwen3-8B-Base / Llama-3.1-8B-Instruct / DeepMath-103K。不能把本文小模型实验说成主实验。

作者官方技术文章也讨论了残差与初始适配器重建参考模型，但没有补充具体数值精度表：[美团官方解析](https://tech.meituan.com/2026/08/27/ACL-Outstanding-Paper-GeoRA.html)。文中的代数“准确”不能自动理解为浮点输出逐位相同。

## 2. PiSSA：原论文就专门关心 BF16 与 FP32

- [原论文 v4 §5](https://arxiv.org/html/2404.02948v4#S5)
- [Appendix D / Table 7](https://arxiv.org/html/2404.02948v4#A4)

§5 明确区分：LoRA 与 PiSSA 的 base 和 adapter 都使用 Float32；FullFT 为节约资源采用 BF16；量化基线用 NF4 base、FP32 adapter。这是论文的实验声明，不能用现今 README 的 BF16 范例覆盖它。

Appendix D 额外比较四个模型的 **全参数 SFT**，数据为 MetaMathQA-395K，3 个 epoch。FP32 相对 BF16 的 GSM8K 结果并不统一：

| 模型 | BF16 GSM8K | FP32 GSM8K |
|---|---:|---:|
| LLaMA-2-7B | 63.15 | 68.31 |
| Mistral-7B | 73.09 | 65.88 |
| Gemma-7B | 75.21 | 75.97 |
| LLaMA-3-8B | 81.96 | 75.44 |

这是精度确实能改变实验结果的先验；不是 FP32 一定更好的证明，也不是 KV-cache 概率一致性实验。

### 当前官方代码：已经是另一套混合精度范例

原 `GraphPKU/PiSSA` 已重定向 `MuLabPKU/PiSSA`。读取时 HEAD 为 `a6e4c9c1d8bc1b73c0fd2f524be8e00e26043395`。

- [run_pissa.sh L16–20](https://github.com/MuLabPKU/PiSSA/blob/a6e4c9c1d8bc1b73c0fd2f524be8e00e26043395/scripts/metamath_llama2_7b/run_pissa.sh#L16-L20)：`--bf16`。
- [run_lora.sh L10](https://github.com/MuLabPKU/PiSSA/blob/a6e4c9c1d8bc1b73c0fd2f524be8e00e26043395/scripts/metamath_llama2_7b/run_lora.sh#L10)：相同 flag。
- [train.py L160–175](https://github.com/MuLabPKU/PiSSA/blob/a6e4c9c1d8bc1b73c0fd2f524be8e00e26043395/train.py#L160-L175)：按 bf16 flag 选择 BF16/FP32 加载模型。
- [train.py L208–210](https://github.com/MuLabPKU/PiSSA/blob/a6e4c9c1d8bc1b73c0fd2f524be8e00e26043395/train.py#L208-L210)：名称包含 norm/gate 的模块参数转 FP32。这不保证 AMP 内每次 GEMM 都以 FP32 执行。
- [requirements.txt](https://github.com/MuLabPKU/PiSSA/blob/a6e4c9c1d8bc1b73c0fd2f524be8e00e26043395/requirements.txt)：PEFT 0.14.0。
- [utils/init_pissa.py L24–40](https://github.com/MuLabPKU/PiSSA/blob/a6e4c9c1d8bc1b73c0fd2f524be8e00e26043395/utils/init_pissa.py#L24-L40)：默认 BF16 加载，可选择 FP16/FP32。
- [PEFT v0.14.0 pissa_init，L234–267](https://github.com/huggingface/peft/blob/v0.14.0/src/peft/tuners/lora/layer.py#L234-L267)：把 base weight 转 FP32，SVD/低秩 SVD、A/B 构造与残差减法在这个 FP32 分支里进行，残差最后 cast 回原 dtype。
- [PEFT v0.14.0 adapter upcast，L325–366](https://github.com/huggingface/peft/blob/v0.14.0/src/peft/tuners/tuners_utils.py#L325-L366)：适配器 dtype 自动提升 FP32 的实现。

这段实现恰好说明“BF16 加载/训练”与“初始化 SVD 在 FP32 算”可以同时成立。

## 3. MiLoRA：原论文与官方源码

- [NAACL 原论文](https://aclanthology.org/2025.naacl-long.248.pdf)
- [arXiv v3](https://arxiv.org/html/2406.09044v3)
- [作者官方仓库](https://github.com/sufenlp/MiLoRA)

论文 §4.1 报告数学实验复用 PiSSA 的实现，但正文和附录没有清晰的逐环节 dtype 声明。仅说复用实现，不能推断它与 PiSSA 原论文每项精度设置都相同。

核对官方代码 HEAD：`c3c94693b26c800a96dba84a1fe92d7384b7c28d`。

- [svd_init.py L124](https://github.com/sufenlp/MiLoRA/blob/c3c94693b26c800a96dba84a1fe92d7384b7c28d/svd_init.py#L124)：FP16 加载 base。
- [svd_init.py L144–149](https://github.com/sufenlp/MiLoRA/blob/c3c94693b26c800a96dba84a1fe92d7384b7c28d/svd_init.py#L144-L149)：`p.data.float()` 传给 SVD；A/B 从 FP32 结果赋值；base 原地减 delta，因此保留原 base 的 FP16 存储 dtype。
- [svd_init.py L16–54](https://github.com/sufenlp/MiLoRA/blob/c3c94693b26c800a96dba84a1fe92d7384b7c28d/svd_init.py#L16-L54)：全 SVD、尾部选择、平方根与矩阵乘法构造 A/B/delta。
- [svd_init.py L159–163](https://github.com/sufenlp/MiLoRA/blob/c3c94693b26c800a96dba84a1fe92d7384b7c28d/svd_init.py#L159-L163)：直接保存该状态，没有另一个显式 dtype 转换。
- [scripts/run_train.sh L86–87](https://github.com/sufenlp/MiLoRA/blob/c3c94693b26c800a96dba84a1fe92d7384b7c28d/scripts/run_train.sh#L86-L87)：`--bf16 True` 与 `--tf32 True`。
- [configs/stage2.conf](https://github.com/sufenlp/MiLoRA/blob/c3c94693b26c800a96dba84a1fe92d7384b7c28d/configs/stage2.conf)：BF16 enabled auto，DeepSpeed ZeRO stage 2。

上述是静态源码推断，没有运行官方版本打印每个 optimizer state/实际 operator dtype。初始化高精度与训练 BF16 混精度的区别已经可以确认；优化器状态细节仍不能冒充实测。

## 4. LoRA：不规定唯一精度，官方主动提醒复现差异

- [原论文](https://arxiv.org/html/2106.09685)
- [官方仓库](https://github.com/microsoft/LoRA)

官方仓库 HEAD `c4593f060e6a368d7bb5af5273b8e42810cdef90`。原论文没有可据此填写完整逐算子精度表的声明；§7.1 提及 FP16 文件大小只是存储估算，不应拿来当训练 dtype 证据。

- [NLU README](https://github.com/microsoft/LoRA/blob/c4593f060e6a368d7bb5af5273b8e42810cdef90/examples/NLU/README.md#L9)：明确提醒 GPU/driver/CUDA/浮点精度/random seed 都会造成结果差异。
- [DeBERTa MNLI script L19](https://github.com/microsoft/LoRA/blob/c4593f060e6a368d7bb5af5273b8e42810cdef90/examples/NLU/deberta_v2_xxlarge_mnli.sh#L19)：`--fp16`。
- [RoBERTa MNLI script](https://github.com/microsoft/LoRA/blob/c4593f060e6a368d7bb5af5273b8e42810cdef90/examples/NLU/roberta_base_mnli.sh)：范例未开启 fp16 flag。
- [GPT-2 gpt2_ft.py L337–338](https://github.com/microsoft/LoRA/blob/c4593f060e6a368d7bb5af5273b8e42810cdef90/examples/NLG/src/gpt2_ft.py#L337-L338)：如果传 fp16，使用 Apex AMP O1。这是混合精度，不是所有变量 FP16。

## 5. 官方 PEFT 文档给的先验

[Selecting the dtype of the adapter](https://huggingface.co/docs/peft/developer_guides/troubleshooting#selecting-the-dtype-of-the-adapter)：默认适配器参数 FP32，即使 base 是 FP16/BF16；自 v0.12 起会自动 upcast，需要显式设置才能关闭。文档讨论了半精度适配器更小/略快，以及 loss 与数值溢出的取舍。

应严格区分：参数保存 dtype、算子输入 dtype、算子累加 dtype、输出 dtype。AMP 会改变部分实际计算路径，所以“适配器参数 FP32”本身不足以说明适配器 forward 一直 FP32。

## 对当前复现的建议边界

- 复现 GeoRA 主要结果的主设置仍应保留其报告的 BF16 训练，并对 LoRA/GeoRA 采用一致精度与计算路径。
- FP32 可以成为小规模数值基准/独立训练对照，不应在未测性能成本之前直接替代整个主设置。
- 初始化 SVD、A/B 保存、优化器状态等可以明确记录 FP32；这不是与 BF16 forward 矛盾，而是常见混合精度分工。
- 单纯把最终 softmax/logp 转 FP32，无法消除已经进入 logits 的 BF16 形状差异。
- 判断是否值得全模型 FP32，应分别测：同前文 token 概率误差、GRPO 首次 ratio 偏离、显存、step throughput 与短训练稳定性。生成文本相同并不足以保证训练概率一致；生成文本不同也不自动说明某精度任务效果更差。
- 对 cache/recompute 不一致，首先保持同 dtype、同前文、同 position/mask、同 prompt/answer tokens；必要时用实际生成端 logp 与训练端 logp 校验或进行经过验证的校正。该工程问题不只 GeoRA 特有。

其他直接机制引用包括 The Path Not Taken (2511.08567)、Evaluating Parameter Efficient Methods for RLVR (2512.23165)、Reinforcement Learning Finetunes Small Subnetworks in Large Language Models (2505.11711)。本子任务未重复深入这些，因为主代理已分配相关调查。
