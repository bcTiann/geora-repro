# LoRA/GeoRA长回答GRPO：首步对照

用户最新要求各10步后先停下来讲解。LoRA已完成10步及1452项独立CPU核对；GeoRA仍在完成第10步。原100步与训练后128题方案暂停接续，不自动提交。逐步说明见[十步训练的数据流](ten_step_walkthrough.md)。

## 实测首步

两边使用同一道train:01480问题，生成8份完全相同的token序列，1124token且全部EOS结束，逐token采样概率、奖励、优势均完全相同。严格规则reward3/8。

| 首步指标 | LoRA | GeoRA |
|---|---:|---:|
| 真实优化器更新 | 1 | 1 |
| 累积反向次数 | 8 | 8 |
| 改变的A/B张量 | 196/392 | 392/392 |
| 梯度范数（裁剪前） | 1.069740 | 1.855623 |
| 生成/重算log概率最大差 | 0 | 0 |
| 训练步秒数 | 514.72 | 646.99 |
| 生成8份回答秒数 | 69.33 | 88.27 |
| 重新计算概率秒数 | 204.29 | 253.54 |
| 反向及更新秒数 | 237.93 | 299.59 |

LoRA初始化B=0，因此首步A的梯度为0，只有B的196个张量改变；GeoRA初始A/B都非零，两者都可以在首步改变。张量数量和梯度范数不能说明哪种方法效果更好，后续要比较验证题表现。

冻结主干、GeoRA初始A0/B0、参考模型在更新前后保持相同。两边都保存了当前A/B、AdamW、scheduler、随机数状态、数据游标及步骤记录，checkpoint为step-00001。独立CPU审计：GeoRA204项通过，LoRA197项通过，均0失败；另外分别在Setonix登录节点核对大状态文件的实际SHA256和字节数。

## 精度与容量

- CPU初始化/mask/SVD为FP32。
- GPU冻结主干/参考权重BF16；A/B/A0/B0和低秩分支FP32。
- 修正转BF16加回主干；log_softmax、loss和AdamW状态FP32；adapter保存FP32。
- 原生DynamicCache生成及重新计算概率均batch1；16-token函数块重算，历史K/V未detach。
- GeoRA先通过最大1024输入＋4096回答的完整前向/反向：40.59GiB峰值、1668.68秒，无参数更新。该峰值包含合成容量检查，不能解释为短回答真实首步的单独峰值。

## 已结束作业用量

| 作业 | 工作 | 逻辑GPU数 | 实际分配秒数 |
|---|---|---:|---:|
| 50699081 | GeoRA最大容量＋首步＋保存/收尾 | 1 | 2381 |
| 50702975 | LoRA首步＋保存/收尾 | 1 | 562 |
| 合计 | 首步阶段 | | 2943（49分03秒） |

两作业均COMPLETED/0:0并释放。GeoRA50703316在gpu-dev运行；LoRA50703647因gpu-dev账号MaxJobs=1曾排队，已将未启动作业移到允许并发的gpu分区，仍请求1个逻辑GPU。两组目标10步，达到目标或时间边界自动保存并结束。普通gpu与gpu-dev的gres/GPU计费权重均为128，见[分区及账号实测限制](resource_policy.json)。是否已经分配以实时队列为准。

## 从哪里读证据

- [配对首步检查](paired_first_audit.json)。
- [GeoRA完整解释](geora_first/summary.md)；[全部8份回答](geora_first/answers.json)。
- [LoRA全部8份回答](lora_first/answers.json)。
- [GeoRA独立审计](geora_first/independent_training_audit.json)、[断点实际字节核对](geora_first/checkpoint_bytes_verified.json)。
- [LoRA独立审计](lora_first/independent_training_audit.json)、[断点实际字节核对](lora_first/checkpoint_bytes_verified.json)。
- [4096-token训练前128题基线](../base4096/comparison.md)：42/128，只有1题截断。

这是工程和首步更新验收；还不能宣称GeoRA优于LoRA，也不是论文完整复现。当前安排以[LONG_RUN_PLAN](../../docs/archive/LONG_RUN_PLAN.md)为准。
