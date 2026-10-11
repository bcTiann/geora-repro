# BF16 LoRA／GeoRA 分段试跑

## 1. 2026-10-12 的决定

先推进 BF16 下的两种初始化方法，不立即扩大为 BF16／FP32 × LoRA／GeoRA 四组。

- [Precision-RL §4.4](https://arxiv.org/html/2510.26788v1#S4.SS4)主要改变生成与训练引擎的精度搭配。其 LoRA 实验约600步失稳；不是本项目的冻结主干精度×初始化方法消融。
- [Beyond Precision §3–4](https://arxiv.org/html/2602.01826v1#S4)讨论异构引擎和优化动态，基线约300步失稳。FP16也未保证稳定；逐项精度配置没有完整列出。
- 我们已经验收了生成与训练概率计算使用相同路径的 BF16 五步及恢复。100步、一个seed的精度对照不足以判断上述长期现象。当前A/B与AdamW状态已是FP32，也不能用冻结主干精度对照检验A/B微小更新是否被存储舍入吞掉。

这使四组实验可以延后。先在两组试跑中持续记录概率一致性、梯度范数、奖励、回答长度、解析失败和截断。若出现数值异常，再从保存的检查点和回答做针对性的FP32对照。

本轮已完成新入口及CPU协议验证：本机和Setonix登录节点的训练分段／恢复16项、评估／隔离／坏检查点拒绝73项均通过；现有128题validation隔离另行通过。报告在 `reports/grpo_pilot_cpu/`，Setonix兼容性记录为 `setonix_cpu.json`。这些使用小网络与确定性生成夹具，不是完整Qwen GPU结果。本轮未申请GPU，100步及其validation结果仍待执行。

## 2. 精度与数据流

固定冻结主干、reference为BF16；A/B、GeoRA A0/B0、低秩差分支、log-softmax、loss、A/B梯度和AdamW状态为FP32。GeoRA沿用已经验收的差分表达式：

$$W_{\mathrm{pre}}x+c\big[B(Ax)-B_0(A_0x)\big],\qquad c=\alpha/r.$$

训练入口直接调用原五步验收器的 `one_step`；生成与训练计算都用 `fixed_width`。不会采用之前失败的 cached 生成＋无cache训练概率组合。

每次更新：

```text
固定train顺序取2题
  → 每题采样4个完整回答（最多256 token）
  → 数字检查得到reward，组内标准化得到advantage
  → 同路径计算old／reference／current的已采样token概率
  → 概率一致性检查、GRPO loss、backward、AdamW、scheduler
  → 保存回答记录，周期保存完整训练状态
```

采用固定1024题train；保留所有同奖励组，不筛选“更容易产生训练信号”的题。128题validation已从原始train中留出，test不参与这一阶段。

配置目标100个有效optimizer steps。首段每方法最多10步：`--stop-after-step 10` 是绝对步数，恢复后继续到20时写20，不是再加20。时限控制只决定何时暂停，不改变训练配置身份。

## 3. 新入口及输出

| 文件 | 职责 |
|---|---|
| `scripts/run_grpo_pilot.py` | 加载相同模型／初始化，开始或恢复分段训练 |
| `scripts/pilot_runtime.py` | 原子记录、训练边界、数据位置、恢复历史和时间控制 |
| `scripts/evaluate_gsm8k_pilot.py` | 固定validation上的greedy数字答案评估，无梯度更新 |
| `jobs/grpo_pilot.sbatch` | 单逻辑GPU，20分钟上限，Python工作时限1000秒 |
| `jobs/evaluate_gsm8k_pilot.sbatch` | 同资源规模下独立运行validation评估 |

训练保存A/B与A0/B0、AdamW、scheduler、Python/Torch/采样器RNG、数据位置及已完成记录的索引；不重复保存完整冻结主干。1000秒是边界软预算：反向传播开始后会完成更新、保存和审计，20分钟Slurm上限留额外余量。没有完成更新的采样不会进入有效训练步数。已消耗的采样与时间仍记录；中途未完成采样的token总数标为未知，不当作0。

冻结主干、reference和A0/B0用前后摘要核对，不额外常驻三份GPU权重。运行前后源文件哈希、模型revision、数据manifest和训练配置身份均记录。恢复时不会把不同方法、配置或数据的检查点接在一起。

初始化身份同时包含因子和rank／alpha，避免相同A/B被误用不同缩放。精确训练恢复要求相同GPU类型、PyTorch/HIP/Transformers版本；升级环境应作为新的验收。

训练报告区分有效步数与实际调用次数；失败后恢复旧检查点造成的重复工作保留为历史，不冒充新有效训练步。

## 4. 固定验证评估在比较什么

依次保存未训练Base、LoRA检查点、GeoRA检查点在**相同validation题目**上的完整回答和数字正确率。所有模型使用batch=1、greedy、原生Transformers KV cache、最多256 token、相同prompt和严格`####`解析器。

这里使用cache只用于独立评估答案，不把其概率送入GRPO。模型没有梯度更新，训练RNG也不读取或修改。达到长度上限而没有EOS的回答计0分；格式失败与已解析的错误数字分开报告。两组模型按相同评估步数和规则比较，最高validation正确率选检查点，平分选较早步。

这项缩小长度、样本量和预算的试跑只能检验持续训练、恢复、成本及早期验证表现；不是GeoRA Table 8分数的复现，也不能证明长期不会崩溃。

## 5. Setonix 提交顺序

先在login节点同步代码、检查准备好的数据和初始化。新入口可以直接读取现有的pinned原checkpoint，不必再创建3.55GB副本。若原目录读取慢而选择临时副本，先用现有CPU工具验证，并要求 `staging_record.json` 的 `validated` 状态。不要在GPU作业中重新初始化SVD或下载模型。

以下shell变量由你设置为实际存在的目录：

```bash
geora_init="$MYSCRATCH/geora/initializations/login-20261009T115357Z-2781659"
geora_data="$MYSCRATCH/geora/datasets/gsm8k"
geora_staged="$MYSCRATCH/geora/checkpoints/geora_base"
mkdir -p "$MYSCRATCH/geora/runs/logs"
```

先评估Base：

```bash
sbatch --account=pawsey0807-gpu \
  --output="$MYSCRATCH/geora/runs/logs/pilot-base-%j.log" \
  jobs/evaluate_gsm8k_pilot.sbatch base "$geora_data" "$geora_staged" \
  "$MYSCRATCH/geora/runs/pilot-validation-base"
```

再提交两方法的首段。每次作业只占一个逻辑GPU；先等第一方法完成，再提交第二方法，测得首段工作量后再定分段大小。

```bash
sbatch --account=pawsey0807-gpu \
  --output="$MYSCRATCH/geora/runs/logs/pilot-lora-%j.log" \
  jobs/grpo_pilot.sbatch lora "$geora_init" "$geora_data" "$geora_staged" \
  "$MYSCRATCH/geora/runs/pilot-lora" 10

sbatch --account=pawsey0807-gpu \
  --output="$MYSCRATCH/geora/runs/logs/pilot-geora-%j.log" \
  jobs/grpo_pilot.sbatch geora "$geora_init" "$geora_data" "$geora_staged" \
  "$MYSCRATCH/geora/runs/pilot-geora" 10
```

训练后评估检查点，例如LoRA第10步：

```bash
sbatch --account=pawsey0807-gpu \
  --output="$MYSCRATCH/geora/runs/logs/pilot-lora-validation-%j.log" \
  jobs/evaluate_gsm8k_pilot.sbatch lora "$geora_data" "$geora_staged" \
  "$MYSCRATCH/geora/runs/pilot-validation-lora-step10" \
  "$MYSCRATCH/geora/runs/pilot-lora/checkpoints/step-00010"
```

GeoRA在检查点参数后再传入 `"$geora_init"`。验证报告逐题保存；若显示 `paused_between_questions`，用原参数和原output目录继续，最后追加 `128 resume`（Base／LoRA没有adapter／init的参数位置用空字符串补齐），跳过已完成题。只比较全部128题完成后的正确率，不把不同题数的部分结果直接比较。

每个训练作业报告提供 `latest_checkpoint`。恢复时沿用同一个output目录并传该检查点，例如继续LoRA到20步：

```bash
sbatch --account=pawsey0807-gpu \
  --output="$MYSCRATCH/geora/runs/logs/pilot-lora-%j.log" \
  jobs/grpo_pilot.sbatch lora "$geora_init" "$geora_data" "$geora_staged" \
  "$MYSCRATCH/geora/runs/pilot-lora" 20 \
  "$MYSCRATCH/geora/runs/pilot-lora/checkpoints/step-00010"
```

只有首段确实完成10步时才使用示例中的step-00010；因时间提前暂停则使用报告列出的实际检查点。`completed`表示已到配置的100步，`paused_at_boundary`表示可继续，`failed`需要先查报告。

查看实际排队与日志：

```bash
squeue -u "$USER"
tail -f "$MYSCRATCH/geora/runs/logs/pilot-lora-实际JOBID.log"
```

不要把预留20分钟上限当成实际GPU用量；实际用量由结束后的Slurm记录汇总。


## 6. 已完成的Base基线（2026-10-12）

Setonix job `50674076` 完成固定128题原模型验证：32正确（25%）、26数字错误、28非截断格式失败、42截断；四类互斥。一次单逻辑GPU allocation793秒，COMPLETED 0:0，已释放；CPU独立审计660项通过。完整解释、全部回答及证据见 [BASE_VALIDATION.md](BASE_VALIDATION.md)。

Base是本项目尚未微调的Qwen2.5-1.5B-Instruct，BF16、greedy、batch=1、原生cache、最多256新token。25%是本协议基线，不能直接代替公开benchmark；42道截断和格式遵循会影响分数。下一步仍是LoRA／GeoRA各10步及相同验证；本轮未开始两种训练。
