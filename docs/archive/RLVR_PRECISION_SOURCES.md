# GeoRA / RLVR 精度调查（2026-10-11，primary sources）

## 先区分两件事

1. **存储精度影响看得见的权重更新。** BF16 检查点可能把小更新舍掉，于是比较前后权重时看见稀疏更新。The Path Not Taken / RL's Razor 讨论此事。
2. **计算精度影响同权重前向输出。** rollout 与训练使用不同序列形状或 kernel，舍入不同会导致 logits / 条件概率不同。Precision-RL / MiniMax / verl rollout correction 讨论此事。用户目前 KV cache 实验属于第 2 件。

“BF16 training”不能推出原始参数、可训练 A/B、优化器状态、梯度归约、SVD、norm、softmax 都为 BF16。下文分清论文明示、官方默认以及未明示。

## 1. The Path Not Taken（GeoRA 最直接机制来源）

- 论文：<https://arxiv.org/html/2511.08567v1#S2.SS3>；精度理论 <https://arxiv.org/html/2511.08567v1#S3.SS3>；训练配置 Appendix C.1。
- §2.1 明确设计 BF16-aware 权重更新稀疏指标。§2.3 解释可见稀疏只是优化偏置与有限精度交互的表象。§3.3 把 Precision 作为整个 Gate III：小更新没有跨过 BF16 间隔就不会在存储的权重上显现。
- §2.3 特别澄清 verl 的优化器状态与梯度归约/累积 FP32；有限精度不是几何偏置本身的来源。
- C.1：VeRL + vLLM rollout、FSDPv2 default mixed precision，NVIDIA H200。论文版本号原文写 “v4.0” 和 “v8.5”；未找到作者 exact runtime commit，不擅自认定一定是 v0.4.0 / v0.8.5。
- 作者也引用 RL's Razor 的 FP32 消融，指出 FP32 存储时稀疏大多消失，任务表现没有改善。
- 可用极短原文定位：标题 “Gate III: Precision Acts as a Lens Revealing the Compass”。
- 边界：权重存储/可见微更新不等于 cached-vs-uncached 前向概率误差；本文没有给用户当前代码的精度最优解。

### 匹配时期的官方 verl tag（代码证据，不能冒充作者 exact commit）

- <https://github.com/verl-project/verl/blob/v0.4.0/verl/workers/fsdp_workers.py#L173-L181>：actor 默认 FP32 加载。代码理由是否则 optimizer 也会 BF16，不合适。
- 同文件 <https://github.com/verl-project/verl/blob/v0.4.0/verl/workers/fsdp_workers.py#L242-L251>：MixedPrecision 参数运算 BF16，reduce FP32，buffer FP32。
- FSDP2 policy <https://github.com/verl-project/verl/blob/v0.4.0/verl/workers/fsdp_workers.py#L282-L299>：param_dtype / reduce_dtype 并 cast_forward_inputs。
- `verl/workers/actor/dp_actor.py`：BF16 autocast；无 cache 计算完整回答 token 的条件 log-probabilities。
- `verl/trainer/config/ppo_trainer.yaml`：rollout dtype=bfloat16，注释要求与 FSDP 对齐。
- `verl/utils/torch_functional.py` <https://github.com/verl-project/verl/blob/v0.4.0/verl/utils/torch_functional.py#L51-L105>：logprob 分 FlashAttention cross entropy / fallback。fallback 对 FP32/64 用 selected logit−logsumexp，对 BF16 换 log_softmax，因为作者标明 BF16 logsumexp subtraction 不稳定。**不应该笼统声称所有版本/所有分支 logprob 都显式 FP32。**

## 2. RL's Razor（GeoRA 直接引用，并做精度消融）

- 论文 <https://arxiv.org/html/2509.04259v1#S6>；LLM 超参 Appendix B.1 / Table 2。
- Table 2 的 SFT/SimPO 与 RL 均明确 bfloat16=True，AdamW。
- §6：作者重做相同训练为 float32 后，性能相同、权重更新不再稀疏，更新矩阵仍 full rank。
- 边界：该段没有逐一列出消融模型，紧接 Table 1 总结的是 MNIST setting。因此可以说“作者报告其对照实验”，不要声称证明所有 LLM / 所有 RLVR FP32 下都如此。
- 这个工作说明 **观察稀疏率必须报告存储精度/阈值**，不能把 exact-zero 稀疏率直接当作与精度无关的机制事实。

## 3. GRPO 原论文 / DeepSeek-R1（明示与未知分开）

- DeepSeekMath：<https://arxiv.org/html/2402.03300v3#S4>。RL 设置列了 GRPO、数据、batch、lr 等；本次未找到明确 BF16 / FP16 / FP32 的 RL 训练说明。
- DeepSeekMath 官方 README 推理例子明确加载 BF16：<https://github.com/deepseek-ai/DeepSeek-Math/blob/main/README.md>（Base / Instruct examples 的 `torch_dtype=torch.bfloat16`）。**这只能证实推荐推理 dtype，不能替代 GRPO 原训练精度证据。**
- DeepSeek-R1 初版：<https://arxiv.org/html/2501.12948v1>，本次未找到浮点精度设置；也查阅新版 HTML，未发现 precision dtype 的训练细节。不能从 DeepSeek-V3 FP8 预训练推断 R1 的全部 RL 流程都 FP8/BF16。
- 所以此行应标“RL 训练细节未明确”，不要因行业常用 BF16 补全成作者设定。

## 4. Defeating the Training-Inference Mismatch via FP16（非常贴近当前问题）

- 论文 <https://arxiv.org/html/2510.26788v1>，2025-10-30，早于 GeoRA；未确认 GeoRA 引用它，所以应标“直接相关补充工作”。
- §3.5 用 DeepSeek-R1-Distill-Qwen-1.5B 分析 BF16/FP16/FP32，AMC/AIME sampling，比较 token probability 与 sequence log-probability ratio。报告 FP16 序列 mismatch 约 24 倍小；不同推理精度单独并不必然带来任务准确率改善。
- §4.4 独立组合训练/推理精度：BF16 训练+FP32 推理在其对照中稳定，但 FP32 推理约 3 倍慢；FP16 两边一致表现好，吞吐不受该 3 倍代价。**这是作者框架/实验的速度，不是 MI250X 的测量，不应用作用户 FP32 训练的倍率。**
- §5.2：Qwen2.5-Math-1.5B + MATH + GRPO-Token-TIS，LoRA all-layer r=32 alpha=64，lr=4e-5。BF16 run 约600steps collapse，FP16稳定；有 LoRA 场景证据。
- §3.2 / §6：FP16 的动态范围较窄，需要 loss scaling / overflow checks。作者不主张 FP16 是所有模型最优。
- 官方实现：<https://github.com/sail-sg/Precision-RL>；最小补丁 <https://github.com/sail-sg/Precision-RL/blob/main/verl_fp16.patch>。
- 补丁改 actor / rollout dtype=float16、autocast dtype 参数化、增 ShardedGradScaler，scale backward、unscale clipping、scaler.step/update；FSDP reduce/buffer 仍 FP32。
- 精度选择不只有 BF16 / 全 FP32；FP16 是值得单独验证的第三方案，但不要未验证就切生产训练。

## 5. MiniMax-M1（局部提高精度的明确成功案例）

- 论文 §3.2：<https://arxiv.org/html/2506.13585#S3.SS2>；Figure 3。
- 作者遇到 rollout 和训练概率不同且 reward 不增长；逐层分析把主要误差定位于 LM-head 大激活；**只将 LM output head 提到 FP32**，训练/推理 token probabilities correlation 约0.9x→0.99x，reward恢复增长。
- 这是局部升级而不是整模型 FP32。论文未给整套 optimizer/master/softmax 的完整 dtype 表。
- 作者明确说该问题没有在小 dense softmax attention 模型上出现。不能把它当成用户 Qwen 的根因或直接抄 LM-head-only 修复；用户已有 Metal 输入形状对照表明早期 MLP 差异传播，原因不同。

## 6. 官方训练框架已经把精度/内核差视为已知问题

- verl rollout correction 文档（2026-09-23版）：<https://verl.readthedocs.io/en/latest/algo/rollout_corr.html>。Overview 的 Policy Mismatch 包括不同 precision（FP8/FP16/BF16/FP32）、不同 backend、相同 weights 的不同实现。
- 明确推荐先记录 mismatch metrics，再按必要启用 importance sampling / rejection；这不表示一次概率相对误差存在就一定导致训练崩溃。
- vLLM：<https://docs.vllm.ai/en/stable/usage/reproducibility/>，默认不保证可复现；即便特殊设置，也只保证相同硬件/相同版本范围。
- vLLM batch invariance：<https://docs.vllm.ai/en/latest/features/batch_invariance/>，让 batch size/order 不影响输出，作为 RL/debugging 的需求。当前官方支持硬件列 NVIDIA >=8.0 和 Intel XPU，**没有 AMD MI250X**；不能给用户一条环境变量就宣称已解决AMD。

## 建议向用户说明的优先级（建议，不是已完成训练结论）

1. 精度是 RL 的实际工程变量和相关机制变量；不能默认完全忽略。
2. 维持 BF16 论文复现基线，同时把 FP32 小试验作为数值对照。相同初始化、输入/有效位置、same-prefix概率、loss/ratio路径必须一致。
3. 对当前1.5B先测同batch同length 的若干 forward/backward/optimizer step，实测 FP32 vs BF16 吞吐、峰值显存；不能用推理速度乘数代替训练测量，更不能由Metal直接估计AMD。
4. FP32 A/B 主参数、初始化SVD/保存、loss/logprob并不要求全模型FP32；BF16 frozen base + FP32 adapter/master/optimizer + FP32关键norm/softmax是不同的精度策略，需要分别记录。
5. BF16训练与FP32 rollout混搭可能依然不匹配，不能认为推理FP32自动修所有训练问题。需要记录实际rollout logprob与训练recompute logprob比值。
6. 如果BF16 pilot 的概率一致性或reward/KL/entropy异常，先做针对性的FP16或局部FP32对照；若小规模全FP32能接受，可用于验证，再决定正式复现是否改变论文精度。

## 极短可直接用于主答复的结论

GeoRA只写了AdamW+bfloat16训练，未公开所有算子/初始化/optimizer dtype。最直接机制来源The Path Not Taken把precision作为一个理论Gate，并明确FP32 optimizer/gradient reduction；RL's Razor报告FP32可改变可见稀疏性但未改善其对照任务表现。当前KV cache概率误差也有专门论文Precision-RL，比较BF16/FP16/FP32和LoRA训练；社区并非不在意。FP32用于诊断有价值，但正式训练是否全FP32需要AMD短吞吐/显存实测，不能由本地10道题自动决定。

## 补充 7. Evaluating Parameter Efficient Methods for RLVR（2512.23165）

- 论文 <https://arxiv.org/html/2512.23165v1#S2.SS4>：§2.4 给出 TRL、Accelerate、DeepSpeed ZeRO-2 和 co-located vLLM；本次正文未找到 BF16/FP16/FP32、optimizer-state 或 SVD dtype 的明示。
- **官方仓库确认链**：作者 Jinjin Gu 的 INSAIT 主页 <https://insait.ai/dr-jinjin-gu/>，该论文条目的 Code 直接指向 <https://github.com/MikaStars39/PeRL>。不能因搜索没有搜到就标“无代码”。
- 本次静态检查主分支 commit `6c9f85ad1c89f7a8e1b9260641aadccf909f25b6`（2026-05-20）；不是原论文确切训练 commit 的证明。
- **模型/训练**：`recipes/trl/openr1/dapo_pissa.sh:16` 等配方显式 `--config.model.dtype bfloat16`；`modules/trl/perl/train.py:87-92` 用 BF16 加载、FlashAttention2；`recipes/trl/accelerate/ds_zero2_4gpu.yaml:14` 明确 `mixed_precision: bf16`。
- **初始化**：`modules/trl/perl/lora/milora.py:111-118` 在 `base_weight.float()` 上做 SVD，随后因子、残差按 base dtype 舍入；`lora/adapter.py:45-59` 的 PiSSA 配置 `pissa_niter_4`。环境快照 `env/requirements_hard.txt` 锁 PEFT 0.18.0 / TRL 0.25.1 / DeepSpeed 0.18.2 / vLLM 0.10.2。
- **依赖代码证据**：PEFT 0.18.0 的 `pissa_init` 明确转 FP32 后做 SVD、重构残差再转原 dtype：<https://github.com/huggingface/peft/blob/v0.18.0/src/peft/tuners/lora/layer.py#L291-L322>。`PeftModel` 默认 `autocast_adapter_dtype=True`，通用 adapter 的 BF16/FP16 参数会提升 FP32：<https://github.com/huggingface/peft/blob/v0.18.0/src/peft/peft_model.py#L94-L122> 和 <https://github.com/huggingface/peft/blob/v0.18.0/src/peft/tuners/tuners_utils.py#L478-L490>。自写 MiLoRA 后续又覆盖参数 dtype，不应无条件套用默认。
- **优化器主参数**：配方使用 ZeRO-2；锁定 DeepSpeed 0.18.2 默认路径建立 FP32 master partition，并把它交给 optimizer：<https://github.com/deepspeedai/DeepSpeed/blob/v0.18.2/deepspeed/runtime/zero/stage_1_and_2.py#L390-L417>。这说明 BF16 mixed precision 不等于 optimizer master 都 BF16；仍是依赖版本静态证据。
- **rollout**：配方明确 vLLM colocate，但没有单独指定 rollout dtype。锁定 TRL 0.25.1 的 `LLM(...)` 不传 dtype，使用 vLLM 默认，不把它冒充作者额外验证的设定：<https://github.com/huggingface/trl/blob/v0.25.1/trl/trainer/grpo_trainer.py#L548-L565>。
- **logprob**：锁定 TRL 先无 cache 得到 logits，再调用 `selective_log_softmax`；有 FP32/64 和低精度分支，后者避免 BF16 的 logsumexp 相减；没有在该函数中统一显式 `.float()`。实际输出还受 Accelerate autocast 包装影响，不应声称所有 softmax/logprob 均 BF16或FP32：<https://github.com/huggingface/trl/blob/v0.25.1/trl/trainer/grpo_trainer.py#L824-L834>；<https://github.com/huggingface/trl/blob/v0.25.1/trl/trainer/utils.py#L1392-L1422>。
- **当前仓库边界**：当前 `train.py:98-99` 调的是 `apply_lora`，未调用 `apply_peft` 多方法分派；所以当前 PiSSA/MiLoRA 配方与具体执行路径须进一步核对，不能把模块里存在实现当作这次论文运行的保证。另外论文说 optimizer offload，但现有 ZeRO YAML 是 `offload_optimizer_device: none`；明确记录差异。
- 核查文件固定链接：<https://github.com/MikaStars39/PeRL/blob/6c9f85ad1c89f7a8e1b9260641aadccf909f25b6/modules/trl/perl/train.py>；<https://github.com/MikaStars39/PeRL/blob/6c9f85ad1c89f7a8e1b9260641aadccf909f25b6/modules/trl/perl/lora/milora.py>；<https://github.com/MikaStars39/PeRL/blob/6c9f85ad1c89f7a8e1b9260641aadccf909f25b6/recipes/trl/accelerate/ds_zero2_4gpu.yaml>。

## 补充 8. Reinforcement Learning Finetunes Small Subnetworks in LLMs（2505.11711）

- 论文 v2 <https://arxiv.org/html/2505.11711v2#S2.SS2>：§2.2 明确所有分析模型为 bfloat16，并以权重差的绝对值 `1e-5` 作默认相等阈值；Appendix A / Table 6 给不同阈值的结果。研究公开检查点的 dtype，不能证明这些模型生产时所有算子的训练 dtype。
- Appendix B 的 DPO gradient masking 实验明确 **bfloat16 mixed precision + DeepSpeed Stage 3**；PRIME 实验给训练超参，没有完整 optimizer/master/rollout/logprob dtype 表。§4 明确 DPO 用 Open-Instruct，PRIME 用 verl。
- v2 §6 脚注直接承认无限精度时更新可能密集、有限精度影响可见稀疏；未看到自行 BF16 vs 全 FP32 的完整消融。不要把脚注里的极小数例子当作精确 BF16 ULP 理论。
- 作者官方仓库 <https://github.com/SagnikMukherjee/sparsity_in_rl>（论文/作者一一对应，NeurIPS 版也链接）：当前 `src` 只有 `check_sparsity.py` 和 `update_rank.py`，为**检查点分析代码**；没有作者的 DPO/PRIME 完整训练/rollout 实现可供这里核查。
- `src/check_sparsity.py` 默认 `torch_dtype=bfloat16`，加载前后模型，再在该 dtype 相减；整体统计同时给 exact zero 和 `torch.isclose(..., atol=tol)`。<https://github.com/SagnikMukherjee/sparsity_in_rl/blob/main/src/check_sparsity.py#L9-L15>、<https://github.com/SagnikMukherjee/sparsity_in_rl/blob/main/src/check_sparsity.py#L30-L38>、<https://github.com/SagnikMukherjee/sparsity_in_rl/blob/main/src/check_sparsity.py#L85-L99>。
- 因此可报告：**BF16 检查点分析明确，DPO BF16 mixed precision 明确，初始化/SVD/优化器状态/PRIME rollout 具体 dtype 未公开到足以复原的粒度。**不能从当前分析脚本推断全部训练流程 BF16。
