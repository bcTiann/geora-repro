# 已有 128 份 GSM8K 回答：新旧判分比较

这是 CPU 对已有文本的重新判定。没有重新生成回答，没有训练或 GPU 运算。

## 判定规则

- 旧规则：唯一 `####`，其后全部内容必须是一个数；使用 Decimal 数值相等。
- 新规则：固定版本 verl strict，只检查末尾 300 字符，取最后一个 `#### 数字` 匹配；去逗号后按字符串相等。
- 两者都保留原评估门槛：未以 EOS 结束的截断回答，最终评估奖励为 0。
- 提取和奖励都直接执行保存的官方源码，并与独立实现核对。

verl commit: `1d367a4f459a43768f76312c0d3430a99ec46b8c`

## 汇总

| 指标 | 旧规则 | 新规则 |
|---|---:|---:|
| 最终判对 | 32 | 31 |
| 成功提取数值 | 58 | 65 |
| 无法提取 | 70 | 63 |
| 提取成功但按字符串规则不匹配 | 26 | 34 |
| 未截断但无法提取 | 28 | 22 |
| 原始奖励正确但被截断门槛排除 | 0 | 0 |
| 最终正确率 | 25.0000% | 24.2188% |

截断回答：42 / 128。

提取结果变化：9 份；原始奖励变化：1 份；最终评估奖励变化：1 份。

## 最终奖励转移

| 旧奖励 → 新奖励 | 数量 |
|---|---:|
| 0 → 0 | 96 |
| 0 → 1 | 0 |
| 1 → 0 | 1 |
| 1 → 1 | 31 |

## 每一份提取或奖励改变的回答

### train:06227

标准答案：`9`；截断：False；EOS：True。

旧提取：`9`；新提取：`9.00`。

原始奖励：1.0 → 0.0；最终评估奖励：1.0 → 0.0。

**问题**

Carly is overnight shipping some fresh-baked cookies to her grandma. The shipping cost is equal to a flat $5.00 fee plus $0.80 per pound of weight. If the package weighs 5 pounds, how much does Carly pay for shipping?

**完整原回答**

```text
To calculate the total shipping cost for Carly's package, we need to consider both the flat fee and the cost per pound of the package's weight.

The flat fee is $5.00
The cost per pound is $0.80
The weight of the package is 5 pounds

First, we calculate the cost for the weight of the package:
Cost for weight = Cost per pound * Weight
Cost for weight = $0.80 * 5
Cost for weight = $4.00

Next, we add the flat fee to the cost for the weight:
Total cost = Flat fee + Cost for weight
Total cost = $5.00 + $4.00
Total cost = $9.00

Therefore, Carly pays $9.00 for shipping.

#### 9.00
```

### train:02679

标准答案：`106`；截断：False；EOS：True。

旧提取：`None`；新提取：`73.`。

原始奖励：0.0 → 0.0；最终评估奖励：0.0 → 0.0。

**问题**

Gretchen, Mitzi, and Beth went bowling together. Gretchen bowled a 120. Mitzi bowled a 113. Beth bowled an 85. What was their average bowling score?

**完整原回答**

```text
To find the average bowling score, we first need to add up all the scores and then divide by the number of people who bowled.

Gretchen's score: 120
Mitzi's score: 113
Beth's score: 85

Total score = 120 + 113 + 85 = 220

Number of people = 3

Average score = Total score / Number of people = 220 / 3 = 73. 33

#### 73. 33
```

### train:06502

标准答案：`12`；截断：False；EOS：True。

旧提取：`15.5`；新提取：`15.50`。

原始奖励：0.0 → 0.0；最终评估奖励：0.0 → 0.0。

**问题**

Ivan has a piggy bank that can hold 100 pennies and 50 dimes. How much, in dollars, does Ivan have if he has filled his two piggy banks with those coins?

**完整原回答**

```text
To determine how much Ivan has in his piggy banks, we need to calculate the total value of the pennies and the total value of the dimes separately, and then sum these values.

#### 15.50
```

### train:06770

标准答案：`1000`；截断：False；EOS：True。

旧提取：`None`；新提取：`1130`。

原始奖励：0.0 → 0.0；最终评估奖励：0.0 → 0.0。

**问题**

Five years ago, there were 500 old books in the library. Two years ago, the librarian bought 300 books. Last year, the librarian bought 100 more books than she had bought the previous year. This year, the librarian donated 200 of the library's old books. How many books are in the library now?

**完整原回答**

```text
To find out how many books are in the library now, we need to calculate the number of books added and subtract the number of books donated.

#### 500 + 300 + 100 + 200 = 1130

#### 1130 - 200 = 930

There are 930 books in the library now.
```

### train:03930

标准答案：`10`；截断：True；EOS：False。

旧提取：`None`；新提取：`1.5`。

原始奖励：0.0 → 0.0；最终评估奖励：0.0 → 0.0。

**问题**

Jonsey is awake for 2/3 of the day and spends 1/2 her time awake playing outside and the rest inside. Her brother, Riley, is awake for 3/4 of the day and spends 1/3 of his day outside and the rest inside. How much time, on average, do they spend inside?

**完整原回答**

```text
To find out how much time Jonsey and Riley spend inside on average, we need to calculate the time each spends inside and then find the average.

#### 1.5

To calculate the time Jonsey spends inside, we first find out how much time she spends awake and then subtract the time she spends playing outside.

#### 1.5

Jonsey spends 2/3 of the day awake, so she spends 2/3 * 24 hours = 16 hours awake. She spends 1/2 of her awake time playing outside, so she spends 1/2 * 16 hours = 8 hours outside. Therefore, she spends 16 hours - 8 hours = 8 hours inside.

#### 1.5

To calculate the time Riley spends inside, we first find out how much time he spends awake and then subtract the time he spends playing outside.

#### 1.5

Riley spends 3/4 of the day awake, so he spends 3/4 * 24 hours = 18.5 hours awake. He spends 1/3 of his awake time playing outside, so he spends 1/3 * 18.5 hours = 3 hours outside
```

### train:02509

标准答案：`3`；截断：False；EOS：True。

旧提取：`None`；新提取：`2.`。

原始奖励：0.0 → 0.0；最终评估奖励：0.0 → 0.0。

**问题**

Sam spends sixty minutes studying Science, eighty minutes in Math, and forty minutes in Literature.  How many hours does Sam spend studying the three subjects?

**完整原回答**

```text
To find out how many hours Sam spends studying the three subjects, we first need to add up the total minutes spent on each subject:

Science: 60 minutes
Math: 80 minutes
Literature: 40 minutes

Total minutes = 60 + 80 + 40 = 180 minutes

Now, we convert the total minutes into hours. Since there are 60 minutes in an hour, we divide the total minutes by 60:

Total hours = 180 minutes ÷ 60 minutes/hour = 2. 67 hours

Therefore, Sam spends 2. 67 hours studying the three subjects.

#### 2. 67
```

### train:00436

标准答案：`22`；截断：False；EOS：True。

旧提取：`None`；新提取：`62.`。

原始奖励：0.0 → 0.0；最终评估奖励：0.0 → 0.0。

**问题**

Catherine had an equal number of pencils and pens. If she had 60 pens and gave eight pens and 6 pencils to each of her seven friends and kept the rest for herself, how many pens and pencils did she have left?

**完整原回答**

```text
Let's break down the problem step by step:

1. Catherine had an equal number of pencils and pens. She had 60 pens.
2. She gave 8 pens and 6 pencils to each of her 7 friends.
3. We need to calculate how many pens and pencils she gave away in total.
4. We then subtract the total given away from the original number of pens and pencils to find out how many she has left.

Let's calculate:

- Pens given away: 8 pens/friend * 7 friends = 56 pens
- Pencils given away: 6 pencils/friend * 7 friends = 42 pencils

Now, let's calculate the remaining number of pens and pencils:

- Remaining pens: 60 pens - 56 pens = 4 pens
- Remaining pencils: 60 pencils - 42 pencils = 58 pencils

Finally, let's add the remaining pens and pencils:

- Total remaining: 4 pens + 58 pencils = 62 items

Therefore, the final answer is #### 62.
```

### train:01770

标准答案：`800`；截断：False；EOS：True。

旧提取：`None`；新提取：`3200`。

原始奖励：0.0 → 0.0；最终评估奖励：0.0 → 0.0。

**问题**

Carolyn is planning out her final essay. The introduction will be 450 words, the conclusion will be triple the length of the introduction, and each of the four body sections will be the same length. If her essay has to be 5000 words total, how long is each section?

**完整原回答**

```text
To find the length of each section, we first need to calculate the total length of the introduction and conclusion combined.

#### 450 + 450 * 3 = 1800

Now, we subtract the combined length of the introduction and conclusion from the total word count to find the total length of the four body sections.

#### 5000 - 1800 = 3200

Finally, we divide the total length of the four body sections by 4 to find the length of each section.

#### 3200 / 4 = 850

Each section is 850 words long.
```

### train:00737

标准答案：`15`；截断：False；EOS：True。

旧提取：`None`；新提取：`15.`。

原始奖励：0.0 → 0.0；最终评估奖励：0.0 → 0.0。

**问题**

Michael has 4 packs of crayons and wants to buy 2 more. One pack of crayons costs $2.5. How much are all the packs of crayons Michael will have after the purchase worth?

**完整原回答**

```text
To find the total cost of all the packs of crayons Michael will have after the purchase, we need to follow these steps:

1. Calculate the total number of packs Michael will have after the purchase.
2. Multiply the total number of packs by the cost of one pack.

Step 1: Calculate the total number of packs Michael will have after the purchase.
Michael currently has 4 packs and wants to buy 2 more.
Total packs = 4 (current) + 2 (new) = 6 packs

Step 2: Multiply the total number of packs by the cost of one pack.
Cost per pack = $2.5
Total cost = 6 packs * $2.5/pack = $15.0

Therefore, the final answer is #### 15.
```

## 证据与限制

- 源 JSON、原始 token_ids、完整回答、旧代码、verl 源码及许可证均已复制保存。
- 这里只测判分规则变化，不说明模型能力提高。
- 新规则不接受没有 `####` 的正确自然语言答案；`#### 2.0` 与标签 `2` 也不会得到相同奖励。
- 旧回答的生成上限仍是 256 token；4096 token 基线需另行运行。
- verl 源码 SHA256：`497c6e32b708d0bc8803b5a048ca5983de7756eb9c146abdc56e6f5d28767189`。
