# 當前目標：選定並驗收 GRPO 的採樣與概率計算路徑

建立／完成：2026-10-11。狀態：**完成**。

## 結果與決定

受控實驗確認：歷史 V 的差異足以造成該位置的第 0 層 attention 輸出差異，但單層替換仍留下約 19.27% 的 token 概率差，不能把它說成完整誤差的唯一來源。

新 cache 路徑讓生成與可求導概率重算使用相同的 native Transformers 計算順序；LoRA／GeoRA 各五步真實 GSM8K GRPO及第 3 步恢復重放均 **58/58**。兩方法共 **19,991 個實際有效 completion tokens**（包括重放）的採樣／重算 logp 差為 0；原門限不變。BF16 原權重、FP32 因子／loss／optimizer 的精度策略保留。

**下一輪 pilot 預設使用既有 `fixed_width` 無 cache 路徑；新 cache 路徑保留為已驗收的選項。** 新 cache 生成較快，但本次整段訓練耗時和峰值顯存較高。兩輪實際回答長度不同，這是運行成本觀察，沒有正式同軌跡加速倍數結論。方法、命令、成本表見 [CACHED_TRAINING.md](CACHED_TRAINING.md)，原始報告與統一記錄見 [EXPERIMENT_RECORD.md](EXPERIMENT_RECORD.md)。

本 goal 新增三個單邏輯 GPU 作業，allocation **1370 秒（22 分 50 秒）**；均 `COMPLETED, 0:0`，臨時模型副本已清理。100 步 pilot 配置已整理為[計畫](configs/grpo_pilot_plan.json)，尚未實作通用 pilot runner 或啟動長程訓練。

## 要達成什麼

完成 KV cache 數值差異的受控歸因，選定下一輪 LoRA／GeoRA GRPO 可可靠使用的路徑，交付可執行命令、驗收報告和資源成本。

目標到短程流程驗收為止；100 步 pilot 和論文任務分數對照接續安排。整體進度見 [REPRODUCTION_PLAN.md](REPRODUCTION_PLAN.md)；已完成的初始化目標見 [GOAL_INITIALIZATION_COMPLETED.md](GOAL_INITIALIZATION_COMPLETED.md)。

## 出發點

- 已完成：GeoRA difference 初始化、單步更新、LoRA／GeoRA 各五步無 cache GRPO 及完整恢復。
- 已確認：同一原模型、同一有效前文，有／無 cache 的 logits 不同；原最差 token 的概率相對差 19.6975%。
- 已定位：第 0 層歷史 V 與 attention 輸出有差；原始 Q 微差在 RoPE 後消失，不能直接當作根因。
- 相同權重數值 FP32 forward 把該 token 差異縮小至 0.0032728%，仍超原 ratio 門限，沒有完成完整訓練驗收。

上述是本 goal 開始前的結果，詳見 [CACHED_LAYERS.md](CACHED_LAYERS.md) 和 [EXPERIMENT_RECORD.md](EXPERIMENT_RECORD.md)。本輪接續完成受控干預與短 cache 對齊（L15/S14）、256-token 的真實五步及恢復驗收（L16/S15）。

## 執行順序與完成條件

### 1. 分離差異來源

- [x] 使用原保存的前文，捕獲並保存實際 attention 權重 P 和歷史 V；原始 BF16 snapshot 已核對 SHA256。
- [x] 固定相同 P/V 數值，只改 P×V 的執行形狀。
- [x] 固定 P 和形狀，只替換 cached／重算的歷史 V。
- [x] 分別量化 attention 輸出影響，再用受控替換檢查後續 logits／token 概率的變化；區分局部來源與端到端放大。

### 2. 驗證可用方案

最多驗證兩類有明確依據的候選：計算路徑／精度對齊；或具有明確推導的 importance correction。每次改變記錄實際計算順序和 dtype。

- [x] 候選先通過 CPU 小模型與梯度預檢，再測完整模型。
- [x] 完整批次的所有有效 tokens 均有採樣概率、重算概率、差值及有限值檢查；不能只看原最差 token。
- [x] 覆蓋初始化與更新後策略，短／較長回答及不同有效長度。
- [x] 直接對齊方案維持原門限：最大逐 token logp 差 `2e-5`、ratio 偏離 1 不超過 `2e-5`、最大整段 logp 差 `1e-3`。
- 本輪直接對齊通過，未採用 importance correction；沒有以校正名義放寬原門限。

L15/S14 完整 GeoRA 初始化16及合成更新後64 tokens 均逐元素一致；L16/S15 另完成真實 GSM8K、256-token 容量和兩方法恢復。

不要求每個中間算子逐位一致。相同重算路徑 old/current ratio=1、小模型通過或初始化生成相同，都不足以單獨驗收完整流程。

### 3. 短程訓練與恢復

若選用新路徑：

- [x] LoRA／GeoRA 在相同資料與協定下，各完成五個真實 GSM8K GRPO 有效更新。
- [x] 驗證 loss／logp／梯度有限、僅 A/B 更新、凍結參數不變。
- [x] checkpoint 後下一步恢復包含 optimizer、scheduler、RNG、資料位置；本輪每批僅更新一次，於完成更新後存檔，沒有樣本重用中途存檔。
- [x] 比較恢復前後的下一次更新，採樣 token、reward、advantage、logp、更新後 A/B、AdamW、scheduler 和 RNG 精確一致。
- [x] 使用相同五步及重放協定量測端到端時間、實際 token 數及峰值顯存，含額外初始分支和 attention 重算；實際轨跡不同，不能當作配對速度基準。

若新方案不合格，選用既有無 cache 路徑時，復用已通過的五步與恢復證據；只有具體修改或剩餘風險才重跑。

### 4. 作出決定並交付

- [x] 新 cache 路徑通過直接對齊驗收；pilot 預設沿用無 cache 路徑，根據完整成本選擇。
- 候選皆不合格時的 fallback 條件未觸發；沒有追加無限精度實驗。
- [x] 更新執行命令、pilot 配置、復現計畫與統一實驗記錄；提交推送程式和小型報告。
- [x] 核對實際 Slurm allocation、確認本目標作業已釋放，清理臨時副本。

## 資源使用

CPU 完成預檢與資料分析。Setonix GPU 合併相關對照、復用已驗證初始化並減少重複載入；按工作量安排合理時限，不拘泥每次幾十秒。開始新一類 GPU 工作時通知用途，完成後記錄實際用量。權重與資料留在 scratch，不加入 Git。

## 完成的定義

有受控歸因證據、明確採用／不採用 cache 的決定、相應流程驗收、可重現命令與成本記錄，即完成本目標。僅讓某個誤差變小或診斷腳本通過，不代表目標完成。
