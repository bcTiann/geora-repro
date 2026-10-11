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


## 2026-10-11：完整模型結果與決定

本機與 Setonix login CPU 均完成 34/34 機械檢查、8/8 likelihood gates。完整 Qwen2.5-1.5B-Instruct／MI250X 的結果不同：job 50658753 完成 10/10 機械檢查，但原模型與初始化 GeoRA 的兩個 likelihood gates 均未通過。

| 相同 103 個有效 tokens 的指標 | 原模型 | 初始化 GeoRA | 既有門限 |
|---|---:|---:|---:|
| 最大逐 token logp 差 | 0.17979765 | 0.17979765 | 0.00002 |
| 平均逐 token 絕對 logp 差 | 0.01139211 | 0.01139211 | 僅記錄 |
| scoring／sampling 概率比範圍 | 0.88708549–1.19697513 | 相同 | 偏離 1 最多 0.00002 |
| 最大整段絕對 logp 差 | 0.06114433 | 0.06114433 | 0.001 |

兩種模型在相同 seed 的 cached token IDs 和 sampling logp **完全一致**。這支持「初始化保持原模型的這條生成路徑」；兩者又有完全相同的 scoring 差異，說明不一致也存在於原模型，不能歸因於 GeoRA 初始化。這輪沒有逐算子定位，尚未證明具體是哪個 kernel 或中間張量造成差異。

為什麼阻止接入？即使尚未更新參數，兩條運算路徑對同一 token 給出的概率比已可能達到約 1.197；直接接入既有 GRPO scorer，會把這種路徑差異混入我們原本想測量的參數更新。19.7% 是最大概率比偏差，不是準確率下降或全部 token 的平均誤差。tiny CPU 通過不能代替完整模型 GPU 驗收。

報告中的 `status="passed"` 表示機械檢查完成；`likelihood_gate_passed=false`、`decision="keep_validated_fixed_width_training_sampler"` 才是本輪接入決定。**既有五步 GRPO 的 sampler、精度和門限均未改動。**

| 作業 | 結果 | 實際 allocation | GPU 更新 |
|---|---|---:|---:|
| 50658705 | tokenizer 回傳物件未轉成整數 IDs；在數值測量前失敗 | 34 秒 | 0 |
| 50658753 | 修正後完成診斷；10/10 機械檢查、0/2 likelihood gates | 38 秒 | 0 |

首個腳本錯誤已修正：先 `apply_chat_template(tokenize=False)`，再 `tokenizer.encode(..., add_special_tokens=False)`；本機與 Setonix CPU 均確認 8 列整數 IDs，長度為 40／50。兩次 allocation 均是一個邏輯 GPU、兩分鐘上限，合計 **72 秒**。成功診斷 Python 本體 9.741 秒、峰值 allocated memory 3.178 GiB；這不是整個 allocation 時間，也沒有同条件舊 sampler 的速度對照。

原始報告：

- [本機 CPU](reports/cached_rollout/cpu_local.json)、[Setonix CPU](reports/cached_rollout/cpu_setonix.json)。
- [首次失敗](reports/cached_rollout/50658705.json)、[完整模型診斷](reports/cached_rollout/50658753.json)。
- [本轮副本校驗](reports/cached_rollout/staging_record.json)、[資源與清理記錄](reports/cached_rollout/remote_final.json)。

## 2026-10-11：後續原始 logits 診斷已完成

[L13/S12](CACHED_LOGITS.md) 只用一個原模型，直接保存並比較兩條路徑 softmax 前的 logits。job 50659217 確認最大差 0.5；原有最差 token `This` 的 logit 是 23.25／23.50。下載後用相同 NumPy CPU FP64 softmax 得到概率 12.6281263%／15.1155517%，仍重現 19.6975% 的相對比例差異。差異已在 logits 中出現；具體上游算子與可用校正尚未定位。新增 allocation 34 秒，一個邏輯 GPU。S11 未通過的 likelihood 門限及原訓練 sampler 保持原樣。

## 下一個具體操作

先在 CPU 小模型建立「按 cache 生成的順序逐 token、可求導地重算 logp」的對照，核對 probability、padding/EOS 和 A/B 梯度；計算損失時不能把歷史 KV 一律 detach，否則會截斷參數對過去 token 的影響。這是候選方案，尚未實作或通過。另一條方案是有明確算法依據並經驗證的 rollout importance correction，不能只把兩條路徑的 old logp 改名就當作修正。

CPU 對照通過後，才安排完整模型的短 GPU 對照，包含更新後的 LoRA／GeoRA。然後驗收真實 GRPO 更新與存檔恢復，再量測長回答成本。當前暫緩擴大 cached pilot；原有固定寬度訓練路徑仍可使用。

## 後續定位（L14/S13，2026-10-11）

[逐層定位](CACHED_LAYERS.md) 已完成：固定回放原有 tokens，實際輸入／位置／可見 keys 一致；第 0 層歷史 V 和 attention 輸出出現差異。原始 Q 微差在 RoPE 後消失，不能據此單獨解釋端到端偏差。相同權重數值改用 FP32 forward，原最差 token 的概率相對差由 19.6975% 降至 0.0032728%，仍超原 ratio 門限，尚未驗收完整訓練修正。新增單 GPU allocation 36 秒；目前 17 作業合計 1901 秒。以上是後續證據，本頁 S12／S11 記錄保留各自執行時的範圍與用量。
