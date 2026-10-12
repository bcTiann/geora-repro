# 128题判分、4096-token基线与长回答训练计划

2026-10-12。用户已授权连续执行；当前 goal active。历史256-token结果保留。

## 本轮完成标准

1. 使用已有128份回答比较旧规则和固定版本verl strict，交付全部判定变化。
2. 同一原模型、同一128题、同一prompt与BF16计算，max_prompt=1024、max_new_tokens=4096，完成新基线及CPU独立复核。
3. 完成长回答训练容量检查，并从未训练初始化开始LoRA与GeoRA各100个GRPO更新；先验收第一步及10步，再继续100步。每步固定一个问题、8个回答；两方法使用相同问题顺序和优化预算，不按奖励筛题。
4. 对两方法第100步checkpoint分别进行同一128题验证，比较Base/LoRA/GeoRA，交付训练曲线、回答、检查点和实际GPU用量。

本轮100步是独立指定的小规模训练，不等同于论文完整Table8。正式GSM8K训练量、MATH500等数学评估及域外能力评估列为后续阶段。论文未给出完整总步数等细节，不能声称本轮完全复现作者协议。

## 已固定的协议

| 部分 | 本轮设置 |
|---|---|
| 基底 | Qwen2.5-1.5B-Instruct，revision989aa7980e4cf806f80c7fef2b1adb7bc71aa306 |
| 数据 | 既有固定GSM8K来源及train/validation隔离；不重新划分128题，不使用test |
| 判分 | verl commit1d367a4f459a43768f76312c0d3430a99ec46b8c的strict；format_score=0 |
| 提取细节 | 检查最后300字符，取最后一个匹配的####数字；去逗号后字符串相等，9.00不等于9 |
| 截断 | 非EOS结束回答reward=0；单列截断、无法提取和规则不匹配 |
| 长度 | prompt1024、response4096；达到EOS立即停止，不强迫生成4096 |
| Base及训练后验证 | batch1、greedy、原生Transformers DynamicCache、eager attention |
| 训练采样 | 每个问题8个回答，依次以batch1生成；temperature1、top_p1、top_k0 |
| GRPO更新 | 先用全部8个reward计算组内优势；逐回答计算loss/8并累积梯度；最后统一clip和一次AdamW更新 |
| 初始化与计算精度 | SVD及初始化因子FP32；冻结主干BF16；A/B/A0/B0和低秩分支FP32；log_softmax/loss/optimizer状态FP32；adapter及训练状态按既有FP32checkpoint协议保存 |
| GeoRA计算 | 已通过的Wpre*x+c[B(Ax)-B0(A0x)]顺序 |
| 超参数 | r16、alpha32、rho0.2、lr1e-6、KL0.001；其余独立选择记录在配置 |
| 训练概率路径 | 生成和重算均使用batch1的同一原生cache顺序；历史K/V保持梯度；以16-token函数块重算节省显存 |
| 策略概率检查 | 保留原token logp/ratio/sequence门限，不放宽、不混用cache与无cache概率 |
| global batch | 每步一个问题，明确小于论文128；rollout8对齐论文；不是正式全量训练 |

数据manifest的旧answer_rule用于记录历史标签提取来源；新运行的answer_rule由独立配置明确选择。禁止旧检查点续到新协议，禁止覆盖旧报告。

## 执行与资源

- CPU完成判分、源码/模型来源、prompt和报告核对；不申请CPU Slurm。
- Setonix每个作业只申请一个逻辑GPU，gres/gpu=1；权重、数据和初始化复用已有副本。
- Base验证按完成的问题保存；需要时继续剩余问题，避免重新生成已完成问题。
- 训练先量测第一步的实际回答长度、峰值显存及耗时；梯度显存guard55GiB。
- 普通逐token cache反向传播的历史张量可能随长度平方增长，4096不能只改上限。函数块checkpoint必须先通过CPU概率/梯度对照，再进入GPU。
- 首段后按测得时间分配后续作业；每个完成更新处可保存和恢复。不得因用量上限临近而丢掉已有训练状态。
- 本轮先以累计24个单逻辑GPU小时为资源控制上限；预计超出时保留检查点并向用户说明需要的额外用量，不自动扩到多卡。
- 记录每个job实际ElapsedRaw、AllocTRES、token数、峰值显存、退出码和队列释放状态。申请时限不是实际消耗。

## 当前进度

- [x] 128份CPU重判：32→31题正确，9份提取变化、1份最终奖励变化；无GPU。
- [x] 固定判分源码、许可证及hash，历史规则默认行为保留。
- [x] 既有评估/恢复CPU检查73/73通过。
- [x] 4096-token Base验证42/128=32.8125%，截断1/128；1589项CPU独立复核通过，单GPU977秒。
- [x] 长回答GPU训练容量和两方法第一步：已通过，均保存step1。
- [ ] 两方法各10步，再各100步。
- [ ] 两个训练checkpoint的128题验证、结果解释与资源核对。

判分变化详见[完整报告](reports/regrade_128/comparison.html)。新配置为[grpo_long_pilot.json](configs/grpo_long_pilot.json)。

## 训练入口验收与下一次GPU作业

本地与Setonix容器CPU均通过缓存分段21项、梯度累积6项、反向超时7项；分段恢复19项、本地短容量fixture13项通过。首次GeoRA作业复用已加载的policy与reference，先用人工固定序列1024+4096做一次完整前向/反向容量检查（无optimizer更新），再做首个真实GRPO步骤。容量检查保存干净step0，55GiB guard和超时覆盖反向重算；完成后复用模型，不额外重载。LoRA首步随后独立执行。

基线四格结果：256/Decimal32、256/verl31、4096/Decimal43、4096/verl42。固定verl下提升11题由长度解除截断造成；没有训练。128旧token前缀与86份已完成旧回答全部不变。新报告见[base4096/comparison.html](reports/base4096/comparison.html)。


## 2026-10-12首个GPU结果

GeoRA作业50699081完成：合成1024+4096的完整前向反向通过，峰值40.59GiB；随后真实第1步用时646.99秒，8回答/1124token、reward3/8、392个A/B张量改变、生成与重算概率误差0、冻结状态不变，保存step-00001。实际单GPU分配2381秒。CPU独立审计204通过；状态文件实际SHA256另在登录节点核对通过。详见[首步报告](reports/long_training/geora_first/summary.md)。

LoRA首步50702975运行中；GeoRA恢复至10步作业50703316已排队，依赖LoRA首步成功。后续尚未完成。首步速度提示两组100步可能超出先前自定的24单GPU小时控制线，待两组10步测量后重新估算并说明资源安排；24不是用户明确给出的单GPU小时上限。


LoRA首步50702975完成：197项独立CPU审计通过、0失败，真实step514.72秒，196个B张量改变，梯度1.069740；全部8份回答及概率与GeoRA首步完全相同，断点实际字节核对通过。首步两个作业实际GPU分配合计2943秒。两组10步续跑各用1逻辑GPU并行：GeoRA50703316、LoRA50703647。两组100步及训练后验证仍待完成。汇总见[首步对照](reports/long_training/summary.md)。


资源安排补充：实时查询发现gpu-dev关联MaxJobs=1、普通gpu关联MaxJobs=64；两分区gres/GPU计费权重同为128。已将尚未运行的LoRA50703647移至gpu，gres/gpu=1和4小时时限不变；GeoRA50703316继续gpu-dev。不取消正在计算的步骤，也不增加单作业GPU数。详见[资源限制记录](reports/long_training/resource_policy.json)。
