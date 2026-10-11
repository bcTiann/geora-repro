# 原模型的128题验证结果

2026-10-12。Setonix 作业 **50674076** 已完成并释放资源。

## 1. 这次在做什么

用下载的原始 **Qwen2.5-1.5B-Instruct**，逐题回答我们固定留出的128道 GSM8K validation 问题，保存每个回答，再与标准数字答案比较。这建立后续 LoRA／GeoRA 试跑的原模型基线。

```text
128道固定验证题（模型只读问题与格式提示）
    → 原模型逐题贪心生成一个回答
    → 记录完整文字、token IDs、EOS／截断
    → 按固定规则提取最终数字并判分
    → 保存结果，在CPU独立复核
```

没有装入 LoRA／GeoRA 适配器，没有 SVD、loss、反向传播或 optimizer step。`Instruct` 是已经做过指令后训练的公开模型；这里“原模型”指本项目尚未微调的起点。

## 2. 结果：32/128 = 25%

下表先把所有截断回答归到一类，再将自然结束的回答按正误／格式分类，各行互不重叠。

| 分类 | 题数 | 占128题比例 |
|---|---:|---:|
| 数字正确、格式合规、自然结束 | 32 | 25.00% |
| 格式合规，但数字错误 | 26 | 20.31% |
| 自然结束，但格式不合规 | 28 | 21.88% |
| 到256个新token仍没有EOS，截断 | 42 | 32.81% |
| 合计 | 128 | 100% |

得1分须同时满足：以 EOS 结束；完整回答只出现一个 `####`；其后除了空白只有支持的十进制数字；数字与标准答案精确相等。允许合法千位逗号和正负号，不接受单位、公式或额外文字。每题0或1分，平均就是32/128。

28道自然结束的格式失败中：20道没有 `####`，6道其后不是单独的合规数字，2道出现多个 `####`。这些回答可能数学正确，也可能错误；格式失败不等于数学一定正确。

原始JSON的 `parse_failure_count=70` 已经包含42道截断，**不能再把70与42相加**。本轮没有“数字解析正确但仍被截断”的回答。

## 3. 真实例子

### 正确：train:03352

问题：8支铅笔；蓝笔是铅笔的2倍，黑笔比蓝笔多10支，红笔比铅笔少2支；共多少支笔？

模型计算蓝笔16、黑笔26、红笔6，最后输出 `#### 48`，与标准答案48一致，234 token后自然结束，得1分。

### 数字错误：train:01193

问题：3个橙子每个4.50美元，7杯果汁每杯0.50美元，3罐蜂蜜每罐5美元，4盆植物每2盆18美元；共花多少钱？

正确计算是13.5+3.5+15+36=68。模型错误处理植物价格及加法，最后输出 `#### 26`。格式合规，数字不对，得0分。

### 数学正确但格式失败：train:06075

问题：2名保镖，每人每小时20美元，每天8小时，一周7天；共付多少钱？

模型正确算出20×8×7×2=2240，并写出 `the total amount Tim pays the bodyguards in a week is $2,240.`；但没有 `#### 2240`，因此本规则得0分。这是直接阅读完整输出确认的例子，未用宽松解析修改正式分数。

### 截断：train:02241

问题：郊狼以15英里/小时提前走了1小时；追赶者速度30英里/小时，多久追上？标准答案1小时。

模型在256 token处结束为 `Therefore, it will take Darrel 1 hour to`，没有EOS或完整最终答案，得0分。中间还把30−15误算成25；不能因结尾出现1就当作完整且正确的回答。

所有128题的原文与完整模型回答：[可筛选的完整报告](reports/pilot_base_validation/full_answers.html)。

## 4. 固定了哪些设置

| 部分 | 本轮设置 |
|---|---|
| 模型版本 | `Qwen/Qwen2.5-1.5B-Instruct`，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| 数据来源 | 固定 GSM8K commit `3101c7d5072418e28b9008a6636bde82a006892c` |
| 验证隔离 | 从原始train固定留出128题，与本项目7345题train及16题smoke互斥；本轮不使用官方test |
| 输入 | 相同chat模板和问题；要求解释后以 `#### <number>` 结束；不输入标准答案 |
| prompt上限 | 512 token，实际最长195；没有截断输入问题 |
| 生成 | batch=1，每题一个greedy回答，不随机采样；最多256个新token |
| cache／attention | 原生Transformers KV cache，eager attention；本轮不使用vLLM |
| EOS／pad | EOS `151645`；pad `151643`；结束规则随结果记录 |
| 精度 | 主干参数BF16、BF16 autocast，保留框架内部数值处理；不强制每个算子为BF16 |
| 参数更新 | 无适配器、无梯度更新；没有本轮初始化或训练状态保存精度问题 |
| 保存 | UTF-8 JSON保存文字、整数token IDs及判分，不保存新的模型权重 |
| 运行环境 | PyTorch `2.7.1a0+gite2d141d`，HIP `6.3.42134-a9a80e791`，Transformers `5.19.0` |
| 执行代码 | commit `9159479d00c759dbefd7e77b96d265856d26fe5a`；源码与配置哈希保留 |

## 5. GPU用量与核对

一次单逻辑GPU作业完成全部128题，没有恢复重跑或额外GPU检查。

| 指标 | 测量 |
|---|---:|
| Slurm allocation | 793秒，即13分13秒，0.2203 GPU小时 |
| Python评估总耗时（含加载／保存） | 775.302秒 |
| 逐题生成耗时之和 | 757.125秒 |
| 生成token总数（含EOS） | 26,772 |
| 每题平均新token数（含EOS） | 209.156 |
| PyTorch allocated张量峰值显存 | 2.967 GiB |
| Slurm状态／退出码 | COMPLETED／0:0 |

`--nodes=1` 是运行位置；`gres/gpu=1` 明确只分配一个逻辑GPU。20分钟是上限，实际用量793秒包含作业启动和清理；不是八卡用量，也不是最终金额或GPU利用率。结束后该作业已不在队列中。

模型运行前在login CPU核对128题及prompt长度；运行后在本地CPU完成660项独立检查，包括：从固定原数据重建验证顺序和训练隔离；逐题token解码与全文相同；遇首个EOS结束或正好到256上限；用独立数字语法与Decimal重新评分；所有汇总计数一致。下载字节的SHA256与远端相同，实际执行源码哈希与本地源文件一致。CPU复核没有申请GPU。

## 6. 怎么理解25%，接下来做什么

25%是**当前prompt、格式规则、BF16 greedy、256-token上限、这128题**的正确率。它同时受到数学推理、格式遵循和回答长度影响；不能解释为原模型只能做对25%的数学题。

42/128截断说明长度上限对当前试跑有明显影响。后续若分数改善，应同时查看数字错误、格式失败和截断是否减少，不能把所有改善都归为推理能力提升。128题中每多对1题，正确率变化0.78125个百分点，小幅差异还不足以证明某方法普遍更好。

下一步计划是 LoRA／GeoRA 各10个optimizer steps的完整模型试跑，再在相同128题、相同评估规则下与这个Base比较；保留同题逐项变化。本轮仅完成Base，没有启动这两个训练作业。更长训练、多个seed及论文指定的正式benchmark仍待做；这不是GeoRA Table 8分数。

## 7. 原始证据

- [完整逐题JSON](reports/pilot_base_validation/validation.json)
- [完整回答HTML](reports/pilot_base_validation/full_answers.html)
- [独立审计](reports/pilot_base_validation/independent_audit.json)及[审计源码](reports/pilot_base_validation/independent_audit.py)
- [资源记录](reports/pilot_base_validation/resource_record.json)、[作业日志](reports/pilot_base_validation/pilot-base-50674076.log)
- [分类／长度汇总](reports/pilot_base_validation/analysis_summary.json)、[源码核对](reports/pilot_base_validation/source_audit.json)
- [执行时配置快照](reports/pilot_base_validation/executed_protocol.json)：保留当时规划状态；实际进度以本文件为准
