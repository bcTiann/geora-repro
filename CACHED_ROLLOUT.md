# KV cache 生成與訓練概率檢查

這一步接在 LoRA／GeoRA 各五步 GRPO 與完整恢復驗收之後。目標是減少生成回答的重複運算，並確認生成時的概率與訓練重算的概率一致。

## 我們在比較什麼？

舊路徑每生成一個 token，都重新計算整個固定寬度的前文。候選路徑用 `Transformers.generate(use_cache=True)`：先計算 prompt，再保存每層的 key/value，後續主要計算新 token。

對候選路徑實際生成的同一批 token，記錄兩個數：

- `behavior_logps`：生成時真正用於抽樣的分布，對這個 token 給出的 log probability。
- `scored_logps`：訓練用的固定寬度、無 cache 路徑，重新計算相同 token 的 log probability。

比較 `scored_logps - behavior_logps`，再查看其指數所對應的概率比。若不一致，不能只把重算值叫作 old logp，就宣稱生成與訓練代表同一策略。

原模型也是對照：若原模型就有相同誤差，問題可能來自 cache／矩陣形狀的浮點計算；若只有 GeoRA 有，才進一步檢查適配器。

## 這次候選做法

- 新增獨立的 `scripts/cached_rollout.py`，目前只允許短生成，最多 64 tokens。
- 使用全新的顯式 generation config：temperature=1、top-k=0、top-p=1、無額外 penalty、dynamic cache；沒有繼承模型的隱藏採樣選項。
- 從實際 sampling scores 的 FP32 `log_softmax` 取得 token 概率。
- 保留原有固定寬度的資料格式；mask 包含第一個 EOS，排除之後的 padding。pad 與 EOS 相同時仍按位置判斷。
- 專用 generator 暫時借用全局 RNG，生成後還原全局狀態，保存專用狀態；可精確重放。僅支援單個 sampling process/thread 的 CPU 或 CUDA／ROCm。
- 生成用 `no_grad`，不用 `inference_mode`，因此產生的 IDs 可以進入可求導的訓練重算。
- 每批 cache 獨立，不跨參數更新使用。

**原來通過驗證的 `grpo_rollout.py` 與五步訓練入口沒有改動。** 這還不是正式訓練切換。

## 門限與精度

沿用既有門限，沒有放寬：最大逐 token logp 差 `2e-5`、最大概率比偏離 1 為 `2e-5`、最大整段 logp 差 `1e-3`。

完整模型保持原精度政策：凍結權重 BF16，A/B/A0/B0 與低秩分支及相減 FP32，校正轉 BF16 後加入原分支。採樣／打分的 log-softmax 是 FP32。

機械檢查通過與 likelihood gate 通過是不同欄位。報告中的 `decision` 明確表示候選能否進到下一輪；門限不過時保留舊 sampler。

## 本地已完成

本地 tiny Qwen：FP32／BF16 各測原模型、初始化 GeoRA、一次合成更新後的 GeoRA、pad=EOS 的立即結束情況，共 8 cases、34 項機械檢查。

- FP32 最大逐 token logp 差：`2.384185791015625e-7`。
- BF16 最大逐 token logp 差：`0`。
- 8 個 cases 的 likelihood gate 均通過。
- 同 seed 初始化 GeoRA 與原模型 cached IDs／logp 完全一致；私有 RNG 可精確重放；EOS／截斷／左 padding／backward 通過。

這是合成小模型檢查，沒有 GSM8K reward，沒有 GRPO 訓練或任務分數。報告見 `reports/cached_rollout/cpu_local.json`。

```bash
uv run python scripts/check_cached_rollout_cpu.py \
  --output-dir /private/tmp/geora-cached-rollout-cpu
```

## Setonix 短檢查

`jobs/cached_rollout.sbatch` 申請一個邏輯 GPU，兩分鐘硬上限。只測原模型與初始化 GeoRA，8 個回答、最多 16 個新 tokens；不計算 reward，不做 optimizer update。

使用既有初始化與 CPU 校驗過的模型副本。輸出 `cached_rollout_gpu.json`，包括實際 sampling/scoring 差異、精度、時間和顯存。初始化通過後，還要測已更新策略及完整訓練／恢復，才能切換訓練 sampler。

全 vocab scores 有額外顯存成本，這個實作目前用於短診斷。較長 pilot 前要避免保留每個位置的全部 vocab scores。
