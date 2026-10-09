# GeoRA：自動檢查流程

本流程涵蓋接入 GRPO 前的初始化、混合精度、一次參數更新、reference 和儲存載入檢查。不是 GRPO 實驗，也不測試 GSM8K 分數。實際結果、精度對照與資源記錄集中在 [EXPERIMENT_RECORD.md](EXPERIMENT_RECORD.md)。

目前 1.5B 的 BF16 初始化 logits 門檻未通過；下列更新／重載步驟是在前置檢查通過後才執行的流程。

## 一次執行：登入節點初始化，再提交 GPU 作業

在 Setonix **登入節點的系統 shell** 中執行；不必手動進容器：

```bash
cd "$MYSOFTWARE/geora/code"
git pull --ff-only
/bin/bash jobs/submit_checks.sh
```

刻意使用 `/bin/bash`：載入 PyTorch module 後，裸寫 `bash` 會進入容器，而這一步要在主機提交 Slurm 作業。

依使用者確認，這個專案允許在登入節點執行這類 CPU 初始化。流程改為：

1. 登入節點：透過 PyTorch 容器直接執行 CPU FP32 SVD，預設 2 個計算執行緒，畫面和檔案同時保留進度。
2. 初始化命令成功退出後：提交 `gpu-dev` 的 1 個邏輯 GPU 作業，進行完整模型檢查。

不提交 CPU Slurm 作業，不使用 CPU account，也不在初始化期間申請 GPU。CPU 初始化失敗時，腳本停止，GPU 作業不會提交。GPU 作業使用 `${PAWSEY_PROJECT}-gpu`，本專案已確認為 `pawsey0807-gpu`；需要改時可設定 `GEORA_GPU_ACCOUNT`。

| 階段 | 工作 | 執行位置 | 時間安排 |
|---|---|---|---|
| `prepare_geora_initialization.py` | CPU FP32 SVD、196 層初始化與因子儲存 | 登入節點的容器 Python，預設 2 threads | 同步執行至完成 |
| `geora_training_check.sbatch` | BF16 forward、FP32 A/B backward／AdamW、儲存載入 | `gpu-dev`，1 個邏輯 GPU 及配套 CPU/RAM | 5 分鐘上限，完成自動釋放 |

腳本會自行載入 module 並呼叫 venv Python，不需手動啟動或激活容器。CPU 初始化在目前終端前景執行，保持連線，直到出現成功標記和 GPU JOBID。若有需要中止登入節點的初始化，按 Ctrl+C；此時 GPU 尚未提交。

每次初始化使用 `login-UTC時間-程序ID` 的新目錄，避免覆蓋先前結果。它不是 Slurm JOBID。初始化檔案保存後可以直接重跑 GPU，不必重新 SVD。

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

腳本先印出初始化目錄與 CPU log，初始化完成後才印出 GPU JOBID，並自動在目前終端顯示 GPU 日誌，直到作業離開 queue。Python 使用 `-u` 輸出；每項檢查的短測量值會同時寫入日誌及 JSON。CPU 階段不會出現在 `squeue`；查詢已提交的 GPU 作業：

```bash
squeue -u "$USER"
```

Ctrl+C 只停止觀看 GPU 日誌，不取消 GPU 作業；取消仍用 `scancel JOBID`。只想提交、不在終端等待，可用 `jobs/submit_gpu_check.sh INIT_DIR --no-follow`。重新接上已有作業的日誌：

```bash
/bin/bash jobs/watch_gpu_check.sh JOBID "$MYSCRATCH/geora/runs/logs/geora-check-JOBID.log"
```

作業離開 queue 不等於通過，仍以 log／JSON 判定；日誌觀看不消耗 GPU 計算。CPU 日誌路徑直接使用腳本印出的那一條；以下 `GPU_JOBID` 請替換成印出的 GPU 作業數字：

```bash
cat "$MYSCRATCH/geora/runs/logs/geora-check-GPU_JOBID.log"
cat "$MYSCRATCH/geora/runs/geora-check-GPU_JOBID/gpu_checks.json"
```

CPU 成功標記：`GeoRA CPU initialization passed.`

GPU 成功標記：`All GeoRA GPU pre-GRPO checks passed.`

GPU 報告每完成一項就更新；檢查不通過會報錯、以非零狀態退出並保存已完成的檢查。作業被強制終止時，報告可能仍是 `running`，要結合 Slurm 狀態和 log 判斷。

資料位置（GPU 檔案樹為全流程通過後的預期產物；初始化門檻失敗時不會產生更新檔案）：

```text
$MYSCRATCH/geora/initializations/login-UTC時間-程序ID/
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

全流程通過時，模型因子與 optimizer state 分別保存；尚未包含真實 GRPO trainer 的 scheduler、資料進度或 RNG 等完整恢復狀態。目前 1.5B 作業未執行 optimizer step，因此沒有訓練後 checkpoint。

GPU 峰值記憶體包含 reference、凍結參數快照和重載副本，是檢查流程的峰值，不能直接作為正式訓練顯存需求。

CPU 初始化期間可用 Ctrl+C 中止。GPU 提交後則用 `scancel GPU_JOBID` 提前釋放。若初始化已完成、只想重新跑 GPU，將下面初始化路徑換成先前印出的實際目錄，在登入節點執行：

```bash
/bin/bash jobs/submit_gpu_check.sh \
  "$MYSCRATCH/geora/initializations/login-UTC時間-程序ID"
```

這會先確認三個初始化檔案存在，再將目錄作為明確的 batch script 參數傳入 GPU 作業，不重做 SVD。GPU 腳本也會將收到的目錄印入 log；不再依賴 `GEORA_INIT_DIR` 環境變數。

## 精度診斷與局部定位

完整模型診斷復用保存的初始化，不重新 SVD、不更新參數：

```bash
/bin/bash jobs/submit_gpu_check.sh \
  "$MYSCRATCH/geora/initializations/login-20261009T115357Z-2781659" --diagnose
```

`diagnose_geora_precision.py` 比較逐位置 KL/TV、答案 token 機率、同輸入投影誤差和逐層傳播誤差；FP32 對照從原始權重與 A0/B0 重新構建 F。輸出 `precision_diagnostics.json` 與 `gpu_checks.json`。診斷完成只代表測量完成，不代表更新檢查通過。

第 0 層 attention 的小檢查只申請一個邏輯 GPU，1 分鐘上限：

```bash
sbatch --export=ALL --account="${PAWSEY_PROJECT}-gpu" \
  --output="$MYSCRATCH/geora/runs/logs/attention-probe-%j.log" \
  jobs/first_attention_probe.sbatch \
  "$MYSCRATCH/geora/initializations/login-20261009T115357Z-2781659"
```

此探針比較五種投影／attention 精度設定，測量第 0 層 O 輸出；沒有完整模型、SVD 或 optimizer。結果及模式定義見 [EXPERIMENT_RECORD.md 第 5 節](EXPERIMENT_RECORD.md)。

本機小模型可重跑流程與診斷測試：

```bash
uv run python tests/check_validation_workflow.py
uv run python tests/check_precision_diagnostic.py
```

小模型通過不代表 1.5B 通過。先驗證完整模型的候選精度修正，再執行一次 A/B 更新和更新後重載，之後接 GRPO。


## 完整模型候選精度診斷（2026-10-10 已測量，門檻未通過）

小模型預檢在本機：`uv run python tests/check_forward_precision.py`；Setonix 容器內用目前已啟用的 `python`。完整診斷只做 forward，三種模式、三個短輸入，共享保存因子，不重新 SVD／生成／更新。

需要重跑時，在 Setonix 登入節點：

```bash
cd "$MYSOFTWARE/geora/code"
git pull --ff-only
mkdir -p "$MYSCRATCH/geora/runs/logs"
geora_job_id=$(sbatch --parsable --export=ALL \
  --account="${PAWSEY_PROJECT}-gpu" \
  --output="$MYSCRATCH/geora/runs/logs/full-forward-%j.log" \
  jobs/full_forward_precision.sbatch \
  "$MYSCRATCH/geora/initializations/login-20261009T115357Z-2781659")
echo "Job: $geora_job_id"
echo "Log: $MYSCRATCH/geora/runs/logs/full-forward-$geora_job_id.log"
```

腳本一個邏輯 GPU、三分鐘上限；報告為 `$MYSCRATCH/geora/runs/full-forward-JOB_ID/full_forward_precision.json`。日誌檔建立後用 `tail -f` 可持續顯示；Ctrl-C 只停止追蹤，取消作業需 `scancel JOB_ID`。診斷報告的 `completed` 不表示所有模式通過；判斷應看各模式與各輸入的 gate 和逐位置分布指標。結果、精度及剩餘實驗集中在 [EXPERIMENT_RECORD.md 第 5 節](EXPERIMENT_RECORD.md)。
