# 同一原模型：有無 KV cache 的 logits 比較

更新：2026-10-11，L13/S12。**已確認兩條路徑的差異在 softmax 之前就出現；把保存的 logits 用同一套 CPU FP64 softmax 計算，仍重現原來約 19.7% 的概率比例差異。**

## 1. 比較對象與資料流程

這次只載入**一個凍結的原始模型**：Qwen2.5-1.5B-Instruct，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。没有裝入 GeoRA，沒有另一個「計算概率的模型」，也沒有使用 vLLM。

```text
同一個原始模型、同一批問題
    ↓
有 cache：生成回答，保存 lm_head 輸出的原始 logits 與 token IDs
    ↓
無 cache：把剛才的相同 token IDs 送回同一個模型
          逐位置計算整個 vocab 的 logits
    ↓
先直接比較兩組 logits，再分別用同一套 softmax 計算概率
```

有 cache 的路徑使用 `Transformers.generate(use_cache=True)`：prompt 先算一次，後續復用每層的 KV。無 cache 的路徑使用既有固定寬度計算：整批長度 66，每個預測位置只能透過 causal／attention mask 看到相同的有效前文。它沒有重新抽樣另一份回答。前文 token IDs、左 padding、有效位置與 position IDs 對齊；兩條路徑的執行布局仍不同。

兩種 prompt 分別是 `What is 7 + 5? Explain briefly.` 和 `A box has 3 red balls and 5 blue balls. How many balls are there in total?`，各 4 列，prompt 長度 40／50；seed=29，最多 16 個新 tokens，共 103 個有效 completion tokens。temperature=1，無 top-k／top-p 截斷或 penalty。有效 mask 包含第一個 EOS，排除其後 padding。

## 2. logits 是在哪裡保存的？

我們在 `lm_head` 的輸出處掛臨時 hook，保存生成當下真正算出的 logits；它位於 softmax 和 generation processors **之前**。生成結束後移除 hook。沒有額外做一次 cached forward 來代替當時的數值。

| 項目 | 實際精度 |
|---|---|
| MI250X 原始模型權重與 forward | 權重 BF16，forward 使用 BF16 autocast；eager attention，部分內部運算沿用 FP32 |
| hook 捕獲的原始 logits | BF16，逐值提升成 FP32 後保存 |
| 生成與無 cache 路徑的 log_softmax | GPU FP32 |
| 診斷程式分析已捕獲的 logits | CPU FP64 |
| 下載後的獨立復查 | NumPy CPU FP64，直接計算 softmax |
| 訓練更新 | 0；沒有 reward、backward 或 optimizer step |

BF16 forward 不表示每個內部算子都使用 BF16；例如 RMSNorm 與 attention 內部的 softmax 有 FP32 計算。BF16 → FP32／FP64 是精確保存已有數值，不會補回 BF16 forward 已捨入掉的信息。這次沒有做 FP64 模型 forward。

兩個控制檢查：捕獲 logits 用相同 GPU FP32 `log_softmax` 算出的 sampled-token logp，與生成記錄的 logp **完全一致，最大差 0**；同一份 logits 在同一設備重算的結果也完全一致。

## 3. 實際結果：token `This`

原有最大概率比例差異發生在第二列回答的第 10 個 token（報告零起算 `row=1, step=9`，ID `1986`，文字 `This`）。此時兩條路徑看到的有效對話前文相同，包括：

```text
user: What is 7 + 5? Explain briefly.
assistant: 7 + 5 = 12\n\n
```

完整 system／chat template、token IDs、attention mask 都保存在 JSON。

| 相同位置、相同 token | 有 KV cache | 無 KV cache，固定寬度 |
|---|---:|---:|
| softmax 前的 logit | 23.25 | 23.50 |
| 用相同 NumPy FP64 softmax 算出的概率 | 12.6281263% | 15.1155517% |

相對差異是：

$$
\frac{0.15115551672434263}{0.12628126277745813}-1
=0.1969750175108693
\approx 19.6975\%.
$$

這是某一 token 的概率**相對比例差異**；概率絕對差約 2.4874 個百分點。不是模型準確率差異，也不是全部 token 的平均誤差。

### 為什麼 logit 差 0.25，概率不直接乘上 exp(0.25)？

softmax 的分母也由整個 vocab 的 logits 決定。先定義 $z_v$ 為 token $v$ 的 logit，$p_i$ 為我們關注 token 的概率：

$$
p_i=\frac{e^{z_i}}{\sum_v e^{z_v}},\qquad
\log p_i=z_i-\log\left(\sum_v e^{z_v}\right).
$$

兩邊的 log-softmax 分母項分別是 `25.31924361552085` 和 `25.389446060081053`。因此「無 cache 減有 cache」為：

$$
\Delta\log p_i
=(23.50-23.25)-(25.389446060081053-25.31924361552085)
=0.17979755544.
$$

取指數得到概率比約 `1.196975`。這個計算直接由保存的兩組 logits 重現原差異。

103 個有效 token 上，實際 GPU FP32 的 logp 差與 CPU FP64 重算的 logp 差，最大殘差只有 `3.6856058027723293e-7`；遠小於上述 `0.17979765`。因此這輪主要差異來自送進 softmax 的 logits，最後 softmax 的 FP32 舍入不足以解釋這個約 19.7% 的比例差異。

## 4. 已確認與尚未定位

- 103 個有效位置 × 151,936 vocab 的 logits，最大絕對差 `0.5`，平均絕對差 `0.0573847791`。
- 去掉每個位置 logits 差異的共同均值後，最大差仍有 `0.5778723329`；差異並非只有 softmax 會消除的共同平移。
- 增加一個無 cache 前文寬度對照：batch 保持 8，寬度改為當下前文 59，而非固定 66。其 logits 與 cached／固定 66 路徑最大差都為 `0.34375`。

本次確認的是**softmax 前的 logits 計算路徑產生不同數值**。cache 與固定寬度路徑同時涉及不同序列布局，尚不能將全部差異單獨歸因於 KV cache 本身、某一層、某個 kernel 或 BF16 捨入。寬度對照也沒有唯一定位具體算子。

原有訓練 sampler 與 likelihood 門限保持不變。這次診斷檢查 6/6 通過，表示直接比較與數值分解完成；不表示兩條路徑概率一致，不會把 S11 未通過的接入門限改成通過。概率對齊或校正仍待驗證。

## 5. 執行、保存與 Setonix 用量

| 檢查 | 結果 | GPU allocation |
|---|---|---:|
| 本機 tiny Qwen FP32／BF16 | 10/10；FP32 最大 logits 差約 6.15e-8，BF16 精確一致 | 0 |
| Setonix 登入節點 tiny Qwen FP32／BF16，2 CPU threads | 10/10；FP32 最大 logits 差約 8.94e-8，BF16 精確一致 | 0 |
| 完整原模型 MI250X BF16，job `50659217` | 6/6 診斷檢查；確認 softmax 前差異 | **34 秒，一個邏輯 GPU** |
| 下載原始 logits 後，本機 NumPy FP64 復查 | 概率比與報告相差約 8.22e-14 | 0 |

GPU job 硬上限兩分鐘，Python 本體 6.825 秒，峰值 allocated memory 3.040 GiB；allocation 包含啟動與清理。Slurm 狀態 `COMPLETED, 0:0`，本輪作業已從 queue 消失。CPU 校驗過的臨時模型副本已刪除，scratch 的原始 checkpoint 與 GeoRA 初始化保留。所有已記錄 16 個作業 allocation 合計 **1865 秒，即 31 分 5 秒**。

代碼：`scripts/cached_logits_diagnostic.py` 負責 hook、成對計算、數值分解和保存；`scripts/check_cached_logits_cpu.py` 做 tiny CPU 檢查；`scripts/check_cached_logits_gpu.py` 載入一個原模型；`jobs/cached_logits.sbatch` 申請單 GPU。實際 GPU 執行代碼 commit 是 `7db8af1`。

小型報告隨 Git 保存：

- [本機 CPU](reports/cached_logits/cpu_local.json)、[Setonix CPU](reports/cached_logits/cpu_setonix.json)。
- [完整 GPU 診斷](reports/cached_logits/50659217.json)、[下載後的 NumPy FP64 復查](reports/cached_logits/cpu_snapshot_verification.json)。
- [模型副本校驗](reports/cached_logits/staging_record.json)、[作業狀態與清理](reports/cached_logits/remote_final.json)、[總用量](reports/setonix_jobs.json)。

最差位置的兩個完整 vocab 向量保存在 scratch：

```text
/scratch/pawsey0807/btian/geora/runs/cached-logits-50659217/native_bf16_worst_logits.safetensors
```

本機副本：`outputs/cached_logits/50659217/native_bf16_worst_logits.safetensors`。大小 `1,215,656` bytes，SHA256 `f865de782dd4079f02683335c0a6cb69067cab6daa03dd11d5488d2b917a6f1b`。二進制原始向量由 Git 忽略。

CPU 小模型檢查命令：

```bash
uv run python scripts/check_cached_logits_cpu.py \
  --output-dir /private/tmp/geora-paired-logits-cpu
```

只重新分析已保存的向量、不申請 GPU：

```bash
uv run python - <<'PY'
import numpy as np
from safetensors.numpy import load_file

vectors = load_file('outputs/cached_logits/50659217/native_bf16_worst_logits.safetensors')
probabilities = {}
for name, logits in vectors.items():
    logits = logits.astype(np.float64)
    weights = np.exp(logits - logits.max())
    probabilities[name] = weights[1986] / weights.sum()
print(probabilities)
print('No-cache / cached probability:', probabilities['no_cache_logits'] / probabilities['cached_logits'])
PY
```
