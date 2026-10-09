# 本輪目標（已完成）：驗證 GeoRA 等價重排與單步更新

建立：2026-10-10。實際數值與用量集中記錄在 [EXPERIMENT_RECORD.md](EXPERIMENT_RECORD.md)。

## 要回答的問題

把殘差形式 `F x + c B(Ax)` 改為 `W_pre x + c [B(Ax) - B0(A0x)]` 後，能否保留原始 BF16 模型的初始化輸出，且仍然可以正確更新與恢復 A/B？其中 `F=W_pre-cB0A0`、`c=alpha/r`。

這是本次復現的數值實作候選。實數代數等價，不代表作者採用了這個計算順序，也不保證浮點訓練軌跡和原 residual 路徑完全一致。

## 精度與資料

- 原始凍結模型與 attention 保留原有 BF16 路徑。
- A/B、A0/B0 保存 FP32；兩條低秩分支及相減使用 FP32，校正轉回原分支輸出 dtype 後相加。
- 初始 A0/B0 分支對輸入保留梯度，僅因子固定。
- 復用既有 FP32 初始化，不重新做完整模型 SVD。
- 保存 checkpoint 時記錄 `forward_mode=difference` 與實際 runtime precision；舊無標籤 checkpoint 的預設仍是 residual。

## 驗收條件與順序

1. 本機：更新後的兩層輸出、輸入梯度、A/B 梯度，與獨立 FP64 有效權重公式一致。
2. 本機：小型 Qwen 的 FP32/BF16 初始化 logits 精確一致；A/B 單步更新、凍結參數、模型與 optimizer 重載通過；舊 residual 模式仍可運作。
3. Setonix：完整 196 個投影保留原權重；三個固定短輸入的初始化 logits 全部精確一致，同時保留原有 2%／KL 門檻。
4. Setonix：一次短答案 cross-entropy 的 backward/AdamW；392 個 A/B 梯度與更新、凍結參數不變、實際 logits 改變、有效更新非零。
5. Setonix：初始化與更新後的因子／manifest／模型 logits／optimizer 保存重載一致。
6. 整理原始報告、實際精度、局限和 Slurm 用量，推送 GitHub，確認 allocation 已釋放。

任一前置門檻失敗時保留報告，先定位原因，再決定最小的修正或下一次測量；不放寬門檻宣稱成功。

## 資源與邊界

本機 CPU 完成小模型預檢。Setonix 使用一個邏輯 GPU，每次短作業上限 3 分鐘；依實際失敗證據才安排必要的後續作業。模型與 optimizer 檔案留在 scratch，Git 只保存程式、教程與小報告。

本目標只完成初始化和機械更新／重載驗證，不包含 GRPO、多步穩定性、GSM8K 分數或 LoRA／GeoRA 性能結論。

## 進度

- [x] 新模式實作、兩層 FP64 公式與梯度對照。
- [x] 本機小模型 FP32/BF16 的更新與重載。
- [x] Setonix 容器 CPU 預檢及完整 1.5B 驗收。
- [x] 統一記錄結果、用量及完成狀態。

完成：2026-10-10。完整模型 43/43 項通過，單個邏輯 GPU 實際 allocation 42 秒；初始化與重載誤差為 0。單步 CE 的 logits 變化較大，仍需下一階段評估步長／多步穩定性。原始報告及具體限制見 [EXPERIMENT_RECORD.md 的 L8/S7](EXPERIMENT_RECORD.md)。
