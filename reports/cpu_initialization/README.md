# 先前的 CPU 初始化測量

`checks.json` 與 `manifest.json` 原樣保存自 `~/RLVR/outputs/geora_full_model_check`，由當時的 CPU FP32 全模型 notebook 產生。

它們記錄全部 196 層的初始化、只有 A/B 可訓練，以及保存／重新載入結果。完整模型的 `optimizer_steps` 為 0；這些不是 GPU 或 GRPO 的結果。

新倉庫的 notebook 清空了舊執行輸出並改用倉庫相對路徑；這兩份 JSON 保留作為先前測量的紀錄。重新執行的結果寫入 Git 忽略的 `outputs/geora_full_model_check`，不自動覆蓋此報告。
