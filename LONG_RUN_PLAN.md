# 128题判分、4096-token基线与长回答训练计划

2026-10-12。用户已授权连续执行；goal active。当前训练源代码固定，历史256-token结果保留。

## 完成标准

1. 对既有128份回答，用历史规则及固定版本verl strict重判，展示所有判定变化。
2. 同一原模型、128题、prompt及BF16设置，以1024输入/4096回答token上限完成基线及独立CPU复核。
3. 最大长度训练容量检查通过后，从未训练初始化开始，LoRA与GeoRA各100次GRPO更新。先验收首步及各10步；固定问题顺序、每步1题/8回答、不按奖励筛题。
4. 对实际第100步断点分别完成相同128题验证，交付Base/LoRA/GeoRA比较、训练曲线、完整回答、断点来源和实际GPU用量。

100步是我们指定的小规模实验，不等同于论文Table8完整复现。正式GSM8K训练量、MATH500及域外评估属于后续阶段；论文缺失的训练细节由本轮配置明确记录。

## 当前进度与证据

| 阶段 | 已确认状态 | 证据 |
|---|---|---|
| 128份CPU重判 | 旧规则32题正确→verl31题；9份提取变化、1份奖励变化；GPU0 | [判定变化](reports/regrade_128/comparison.html) |
| 4096-token Base128题 | 42/128=32.8125%，截断1题；1589项独立CPU核对通过，GPU977秒 | [基线报告](reports/base4096/comparison.html) |
| 长回答容量 | 1024＋4096完整前向及反向通过；峰值40.59GiB，1668.68秒；参数未更新 | [GeoRA容量与首步](reports/long_training/geora_first/summary.md) |
| LoRA/GeoRA真实首步 | 两边1次AdamW、8次反向、生成/重算概率差0、完整step1断点；冻结状态不变 | [首步对照](reports/long_training/summary.md) |
| 两边各10步 | 已从step1恢复，作业正在进行；尚待完整验收 | GeoRA50703316，LoRA50703647 |
| 两边各100步 | 待10步实测速度与用量后分段继续 | 配置optimizer_steps=100 |
| 第100步128题评估 | 尚未执行 | 见本文终局验证门槛 |

基线四格：256/Decimal32、256/verl31、4096/Decimal43、4096/verl42。固定verl下增加11题来自解除长度截断，没有训练。全部128旧token前缀和86份旧EOS回答保持相同。

首步独立CPU审计：GeoRA204项、LoRA197项通过，均0失败；12项配对检查通过。两边8份回答、逐token采样概率、奖励、优势完全相同。两份大状态文件另在Setonix登录节点按实际字节核对SHA256和大小，不下载权重。19个训练源文件的执行hash均核对一致。

## 固定协议


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

- 判分、文件/provenance/报告审计在本地或Setonix登录节点CPU进行；不申请CPU Slurm。
- 每个训练/评估作业只请求1个逻辑MI250X GPU，gres/gpu=1；模型、数据、初始化复用已有副本。
- 基线按完成问题保存；训练只在完成整组、AdamW及scheduler更新后保存完整断点，每步保留恢复状态。超时前未更新的组清空梯度并恢复随机数状态，计入实际消耗，不算成功更新。
- 原生cache生成与概率重算均batch1；16-token函数块保留历史K/V梯度。最大1024+4096已实测通过，保留55GiB guard及原概率门限。
- gpu-dev关联MaxJobs=1，普通gpu关联MaxJobs=64；两分区gres/GPU计费权重均128。两组十步作业各用1个逻辑GPU：GeoRA在gpu-dev，LoRA未启动时从gpu-dev移到gpu后已开始；不打断正在计算的步骤。
- gpu-dev时限最多4小时，普通gpu最多24小时。后续按实测速度分段；每段结束检查实际latest checkpoint及进度，再接续，避免重复完成步骤。
- 先前24单逻辑GPU小时是我们自定的保守预警线，不是用户给出的硬上限。第一步已提示100步×2可能超过该线；十步结束后报告更新估算，按用户此前允许的一节点24小时量级安排后续，预计超出授权量级时先说明并等待所需批准。
- 记录实际ElapsedRaw、AllocTRES、退出码、token数、峰值和队列释放；请求时限不是实际耗时，放置在一个节点不等于请求整个节点。

已结束本轮作业：失败基线50675134=27秒、成功基线50697776=977秒；GeoRA容量＋首步50699081=2381秒，LoRA首步50702975=562秒。合计3947单逻辑GPU秒；正在运行的十步作业尚未计入最终合计。

[账号及分区实测限制](reports/long_training/resource_policy.json)，[已记录作业账本](reports/setonix_jobs.json)。

## 十步验收后如何继续

1. 两报告须到逻辑step10，完成10个整组更新；核对每组8份回答、规则/优势、概率门限、梯度、冻结状态、记录和完整最新断点。
2. 不要求每组都产生非零policy优势。同分组不筛掉，记录其零优势及KL贡献；查看是否存在真实非零参数更新。
3. 用逐步耗时、回答长度和已消耗GPU时间估算剩余90步及验证成本；保证两方法使用相同总更新数和问题顺序。
4. 从各自实际latest checkpoint接续至100，源码与语义配置保持相同。源码、模型、数据或初始化不一致时停止该次恢复并调查；不伪造hash通过门槛。

## 终局验证门槛与入口

已有评估器可直接载入训练boundary checkpoint，无需merge或转换。评估器允许任意合法步数，提交前必须额外核对实际断点是第100步，不能凭step-00100目录名判断：

- 训练报告logical_optimizer_steps=100，最新manifest.status=complete。
- manifest.progress.completed_optimizer_steps=100、completed_rollouts=100、next_data_position=100；manifest与报告记录的实际latest路径一致。
- adapter实际字节hash及大小一致；六项provenance一致；GeoRA A0/B0与未训练初始化逐元素相同。

同一语义配置hash：d374c348e61404d5d09e2302d4662d2406eedecc29b774c422ba4f7c5e8424e5。

数据manifest hash：992aa5f708daf169900d30ac15d44f574cec5b25afc9f5842387fb7bb3109cc0。

入口jobs/evaluate_gsm8k_pilot.sbatch的位置参数依次为：method、data目录、原模型目录、评估output、实际adapter checkpoint目录、GeoRA未训练初始化目录（LoRA为空字符串）、128、fresh或resume、configs/grpo_long_pilot.json、runtime秒。请求单逻辑GPU/1小时，程序上限3300秒；需要时从已完成问题恢复。

终局两份validation.json必须status=completed、complete=true、128题、completed_optimizer_steps=100；题目ID与新Base128题逐项同序，模型版本、规则、语义配置、解码、精度及运行环境一致。保留全部回答/token/EOS/截断/提取值/奖励，独立重算总分；最后核对作业已释放、汇总实际用量，再决定goal complete。
