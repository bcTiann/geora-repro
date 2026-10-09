# GeoRA：自動檢查流程

本流程涵蓋接入 GRPO 前的初始化、混合精度、一次參數更新、reference 和儲存載入檢查。不是 GRPO 實驗，也不測試 GSM8K 分數。

## 一次提交，兩個作業

在 Setonix **登入節點的系統 shell** 中執行；不必手動進容器：

```bash
cd "$MYSOFTWARE/geora/code"
git pull --ff-only
/bin/bash jobs/submit_checks.sh
```

刻意使用 `/bin/bash`：載入 PyTorch module 後，裸寫 `bash` 會進入容器，而這一步要在主機提交 Slurm 作業。

腳本依 `$PAWSEY_PROJECT` 選擇 CPU account 和帶 `-gpu` 的 GPU account。本專案分別為 `pawsey0807` 和 `pawsey0807-gpu`；CPU `work` account/partition 配對尚待第一次提交確認。若 CPU 提交被拒絕，流程會停止，不提交 GPU。若 GPU 提交被拒絕，先前印出的 CPU job 可能仍在執行，請保留其 ID。

需要改 account 時可在提交前設定 `GEORA_CPU_ACCOUNT` 或 `GEORA_GPU_ACCOUNT`。

| 作業 | 工作 | 申請資源 | 時間上限 |
|---|---|---|---|
| `prepare_initialization.sbatch` | CPU FP32 SVD、196 層初始化與因子儲存 | `work`，8 CPU 核、20 GiB RAM，無 GPU | 30 分鐘 |
| `geora_training_check.sbatch` | BF16 forward、FP32 A/B backward／AdamW、儲存載入 | `gpu-dev`，1 個邏輯 GPU 及配套 CPU/RAM | 5 分鐘 |

GPU 作業使用 `afterok` 依賴：CPU 初始化成功後才有資格啟動，等待時不佔 GPU。CPU 作業失敗時，依賴無法滿足的 GPU 作業自動取消。上限不是固定消耗時間，作業完成或報錯便退出。[Slurm 作業依賴說明](https://slurm.schedmd.com/sbatch.html)

第一次完整 SVD 在 Setonix 的耗時尚未測量，30 分鐘只是 CPU 作業的初始上限。

## 檢查哪些事情

### CPU 初始化

1. 基底 checkpoint 的固定版本記錄正確。
2. 28 層 × Q/K/V/O/gate/up/down 共 196 個目標層，名稱精確匹配。
3. 可訓練 A/B 參數數為 18,464,768。
4. 每層 FP32 的 `F + (alpha/r) B0A0` 與原始權重的相對誤差不超過 `1e-6`。
5. A0/B0/A/B 都是 FP32 且有限；當前 A/B 等於初始 A0/B0。
6. 保存 FP32 因子及 manifest，GPU 載入時不再做 SVD。

### 完整模型檢查

- 只有預期的 A/B 可訓練；凍結參數為 BF16，A/B 和 A0/B0 為 FP32。
- 原始模型及 GeoRA 初始化模型的 forward 都有限，記錄 logits 形狀及初始誤差。
- 初始化 checkpoint 重載後，logits 逐 bit 一致。
- 一個短問答的 cross-entropy backward：每個 A/B 的梯度有限、非零且為 FP32，凍結參數沒有梯度。
- AdamW 更新一次後，每個 A/B 都實際改變，所有凍結權重、bias、初始 A0/B0 逐 bit 不變。
- AdamW 的兩個 moment tensors 是有限的 FP32，step 為 1。
- 更新後 logits 有限且確實改變，更新後 loss 有限。記錄 loss 變化，但不把單步 loss 下降設為必須條件。
- 使用獨立、凍結、沒有 GeoRA 的原始模型作 reference；更新 A/B 後 reference logits 不變。
- 更新後 adapter 經檔案保存／讀取，全部因子逐 bit 一致。
- 用新讀入的原始 FP32 模型及保存的 A0/B0 重建 F，再載入更新後 A/B；凍結權重和 logits 與保存前逐 bit 一致。
- AdamW state 保存／讀取後逐 bit 一致。

更新測試使用答案 `5` 加結束 token，prompt 的 label 設為 `-100`；只對答案計算下一 token cross-entropy。這是檢查反向傳播與更新的工具，不是 RLVR/GRPO 損失。learning rate 為 `1e-4`、weight decay 為 0、梯度裁剪上限為 1。

reference 檢查只驗證這個腳本的獨立原始模型。未接入的 GRPO 框架仍需另行驗證 reference 路徑；不能停用 adapter 後把殘差 F 當成 W_pre。

## 初始誤差的停止條件

BF16 舍入和兩分支運算會造成差異，初始化 logits 不要求逐 bit 等於原始模型。預設停止條件為：

- 全部 logits 的相對 L2 誤差 ≤ 0.02。
- 最後位置的 `KL(reference || initialized)` ≤ 0.02 nats。

這兩個閾值是本專案的工程檢查門檻，不是论文的理論界限。失敗時保存實際誤差，先定位原因再决定是否改門檻。最大絕對差亦會記錄。

相同環境內的初始／訓練後重載要求逐 bit 相同；跨硬體、軟體版本或精度的重載不沿用這個結論。

## 查看作業與結果

提交腳本會印出 CPU 和 GPU 的 JOBID、log 及報告路徑。查詢自己的作業：

```bash
squeue -u "$USER"
```

以下的 `CPU_JOBID`、`GPU_JOBID` 請替換成印出的數字：

```bash
cat "$MYSCRATCH/geora/runs/logs/initialization-CPU_JOBID.log"
cat "$MYSCRATCH/geora/runs/logs/geora-check-GPU_JOBID.log"
cat "$MYSCRATCH/geora/runs/geora-check-GPU_JOBID/gpu_checks.json"
```

CPU 成功標記：`GeoRA CPU initialization passed.`

GPU 成功標記：`All GeoRA GPU pre-GRPO checks passed.`

GPU 報告每完成一項就更新；檢查不通過會報錯、以非零狀態退出並保存已完成的檢查。作業被強制終止時，報告可能仍是 `running`，要結合 Slurm 狀態和 log 判斷。

資料位置：

```text
$MYSCRATCH/geora/initializations/CPU_JOBID/
  adapter.safetensors          初始 FP32 A0/B0/A/B
  manifest.json
  initialization_checks.json

$MYSCRATCH/geora/runs/geora-check-GPU_JOBID/
  initial_adapter.safetensors  初始化儲存／載入檢查用副本
  trained_adapter.safetensors 一次更新後的 FP32 因子
  manifest.json
  optimizer.pt                一次更新後的 AdamW state
  gpu_checks.json
```

本次 checkpoint 保存模型因子和 optimizer state；尚未包含真實 GRPO trainer 的 scheduler、資料進度或 RNG 等完整恢復狀態。

GPU 峰值記憶體包含 reference、凍結參數快照和重載副本，是檢查流程的峰值，不能直接作為正式訓練顯存需求。

提前取消這組流程時，取消兩個 ID：`scancel CPU_JOBID GPU_JOBID`。若 CPU 已完成、只想重新跑 GPU，可在登入節點執行：

```bash
export GEORA_INIT_DIR="$MYSCRATCH/geora/initializations/CPU_JOBID"
sbatch --account="${PAWSEY_PROJECT}-gpu" \
  --output="$MYSCRATCH/geora/runs/logs/geora-check-%j.log" \
  jobs/geora_training_check.sbatch
```

這會讀取已保存的初始化，不重做 SVD。

## 已驗證與待驗證

2026-10-09：使用者回傳 Setonix job `50574324` 的原始模型 GPU BF16 forward 成功輸出：logits `[1, 40, 151936]`、全部有限、PyTorch 峰值張量顯存 2.962 GiB、腳本內耗時 13.296 秒、optimizer steps 為 0。

新的檢查流程已在本機小型 Qwen 模型的 CPU FP32 與 CPU BF16 模式各通過 27 項檢查；故意把 A/B 錯轉 BF16 時會被拒絕。可用 `uv run python tests/check_validation_workflow.py` 重跑小型流程。這些驗證不等於 1.5B 全模型在 Setonix 的 GeoRA GPU 結果；後者尚待提交作業。

通過後才接下一階段：短 GRPO 的生成、答案檢查器、組內優勢、reference、更新及 trainer checkpoint 檢查，然後固定預算比較 LoRA／GeoRA。
