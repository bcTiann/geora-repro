# GeoRA：自動檢查流程

本流程涵蓋接入 GRPO 前的初始化、混合精度、一次參數更新、reference 和儲存載入檢查。不是 GRPO 實驗，也不測試 GSM8K 分數。

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

資料位置：

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

本次 checkpoint 保存模型因子和 optimizer state；尚未包含真實 GRPO trainer 的 scheduler、資料進度或 RNG 等完整恢復狀態。

GPU 峰值記憶體包含 reference、凍結參數快照和重載副本，是檢查流程的峰值，不能直接作為正式訓練顯存需求。

CPU 初始化期間可用 Ctrl+C 中止。GPU 提交後則用 `scancel GPU_JOBID` 提前釋放。若初始化已完成、只想重新跑 GPU，將下面初始化路徑換成先前印出的實際目錄，在登入節點執行：

```bash
/bin/bash jobs/submit_gpu_check.sh \
  "$MYSCRATCH/geora/initializations/login-UTC時間-程序ID"
```

這會先確認三個初始化檔案存在，再將目錄作為明確的 batch script 參數傳入 GPU 作業，不重做 SVD。GPU 腳本也會將收到的目錄印入 log；不再依賴 `GEORA_INIT_DIR` 環境變數。

## 已驗證與待驗證

2026-10-09：原本的 CPU `work` 提交被 Slurm 拒絕，沒有開始 CPU 計算或提交 GPU。使用者確認可在登入節點執行 CPU 初始化後，採用上述登入節點 → GPU 的流程。

2026-10-09：使用者回傳 Setonix job `50574324` 的原始模型 GPU BF16 forward 成功輸出：logits `[1, 40, 151936]`、全部有限、PyTorch 峰值張量顯存 2.962 GiB、腳本內耗時 13.296 秒、optimizer steps 為 0。

新的檢查流程已在本機小型 Qwen 模型的 CPU FP32 與 CPU BF16 模式各通過 27 項檢查；故意把 A/B 錯轉 BF16 時會被拒絕。可用 `uv run python tests/check_validation_workflow.py` 重跑小型流程。這些驗證不等於 1.5B 全模型在 Setonix 的 GeoRA GPU 結果；後者的 CPU 初始化已完成，GPU 檢查尚待成功執行。

通過後才接下一階段：短 GRPO 的生成、答案檢查器、組內優勢、reference、更新及 trainer checkpoint 檢查，然後固定預算比較 LoRA／GeoRA。

### 2026-10-09：登入節點初始化成功，GPU 傳參待重跑

使用者回傳 196/196 層初始化及五項 CPU 檢查通過；最後一層的累計耗時為 914.9 秒。保存目錄為：

```text
/scratch/pawsey0807/btian/geora/initializations/login-20261009T115357Z-2781659
```

GPU job `50578377` 在 shell 階段因 `GEORA_INIT_DIR` 未設定而停止，未啟動 Python 模型檢查。日誌不能確定環境變數在哪個環節丟失。修正後透過明確的腳本參數傳遞目錄，可直接復用上述初始化；不重新執行 CPU SVD。

提交路徑已以本機模擬驗證：環境中不含 `GEORA_INIT_DIR` 時，GPU batch script 仍會把收到的目錄傳給 Python；缺少參數或初始化檔案會停止。這項驗證沒有執行真實 Slurm 或 GPU 計算。

### 2026-10-09：BF16 初始化 logits 檢查失敗

使用者回傳 GPU job `50578622`：目錄傳遞、392 個 FP32 A/B tensors、凍結 BF16 參數、原始 reference 及有限 forward 通過。初始化 logits 相對 L2 誤差 `0.17153804`，最大絕對差 `7.7265625`，超過原有 2% 門檻；最後位置 KL 為 `6.194578e-8` nats。只檢查最後位置不能證明所有位置的分布接近，尤其該輸入末尾包含 EOS。此作業沒有 optimizer step；不放寬門檻。

只做精度診斷、復用已保存初始化：

```bash
/bin/bash jobs/submit_gpu_check.sh \
  "$MYSCRATCH/geora/initializations/login-20261009T115357Z-2781659" --diagnose
```

診斷使用同一個短輸入，記錄所有 token 位置的 KL／total variation、去除每個位置共同偏移後的 logits 相對誤差、答案預測位置的正確 token 機率。每層另量測：相同 reference 輸入下的 local output 誤差、正常傳播的 output 誤差、BF16 F 加回 FP32 BA 後的有效權重誤差。最後从原始 checkpoint 和 A0/B0 **重新重建 FP32 F**，執行 FP32 全模型比較；不重新 SVD，也不更新參數。

輸出 `precision_diagnostics.json` 及 `gpu_checks.json`。診斷完成表示測量完成，**不表示原有初始化／更新門檻通過**。BF16 的兩分支運算及 F 舍入目前是待檢驗的原因，不能由此次失敗直接確定。診斷會額外載入一個 FP32 模型，資源配置仍是同一個邏輯 GPU。

### 2026-10-09：直接執行第 0 層 attention 小檢查

Codex 透過既有 SSH 連線直接核對 Setonix；CPU preflight 通過後，提交 `gpu-dev`、`gres/gpu=1`、1 分鐘上限的 job `50579802`。Slurm 回報 `COMPLETED`，實際 allocation 為 22 秒；Python 腳本內 7.202 秒，其中五組對照核心計算 1.298 秒，PyTorch peak allocated memory 為 0.130 GiB。Slurm AllocTRES 同時記錄 `cpu=16`、`mem=29440M`、`gres/gpu=1`；只使用一個邏輯 GPU，配套 CPU 數以此實際記錄為準。GPU 作業已離開 queue。

程式只讀取 42 個 token 的 embedding 行、第 0 層 input norm、Q/K/V/O 權重及初始因子，沒有建立完整 1.5B 模型、重新 SVD 或 optimizer step。手動重建的 native attention 與安裝的 Transformers eager implementation，在 reference 與 GeoRA 兩邊的輸出誤差都為 0；原始程式來源保存於 report。

| 第 0 層對照：原始模型／GeoRA 採相同設定 | O 投影輸出相對誤差 |
|---|---:|
| 原有 BF16 autocast 路徑 | 12.3341% |
| 只把 QK 分數／縮放／mask 計算改 FP32 | 3.5347% |
| QK、softmax 與 value 加權和用 FP32，輸出轉 BF16 | 3.5336% |
| F 保留 FP32、線性投影用 FP32，輸出轉 BF16；attention 仍用原路徑 | 0.0234853% |
| 此 attention 與輸入 norm／RoPE 全部 FP32 | 0.000636743% |

原有路徑的 attention 分數絕對值最大 22528；用完全相同的量化 Q/K 改在 FP32 重算，原生 BF16 分數的最大誤差為 104.5879。原始模型與 GeoRA 的最大分數差為 128。具體 head 1、query position 34：原始模型有多個相同的 17536 分數，attention 權重各約 1/21；GeoRA 一項變成 17664 後，權重集中到該項（1.0）。這支持「大數值 QK 分數的 BF16 舍入，會放大少量投影差異」的原因；softmax 轉 FP32 無法恢复前一步丟失的分數精度。

**範圍限制：**這是第 0 層、單一固定輸入的定位結果。0.0235% 是 matched-precision 對照，不是完整模型 logits 門檻結果；改變投影精度也同時保留了 FP32 F，尚未單獨拆分兩者的效果。既有訓練精度及 2%／KL 門檻未改。後續需用完整模型 forward 確認修正方案，才進入更新檢查。

原始測量：[reports/attention_probe/50579802.json](reports/attention_probe/50579802.json)。重跑時只申請一個邏輯 GPU，初始化目錄透過參數傳遞：

```bash
sbatch --export=ALL --account="${PAWSEY_PROJECT}-gpu" \
  --output="$MYSCRATCH/geora/runs/logs/attention-probe-%j.log" \
  jobs/first_attention_probe.sbatch \
  "$MYSCRATCH/geora/initializations/login-20261009T115357Z-2781659"
```
