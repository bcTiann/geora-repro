# GeoRA 復現：本機與 Setonix 實驗記錄

更新：2026-10-10；新增完整模型候選精度實驗，遠端報告與 Slurm 狀態核對至 2026-10-10。涵蓋原有 `~/RLVR` 學習專案與目前 `~/geora-repro` 的實際執行結果。

**目前的結論：FP32 的 GeoRA 全模型初始化與初始化 checkpoint 重載已通過；目前預設 BF16 路徑的全模型初始化輸出檢查未通過。** 第 0 層 attention 的對照實驗已定位到精度敏感環節；完整模型的兩個候選方案把 logits 相對誤差降至約 3.5%～3.8%，仍未通過既有 2% 門檻。1.5B 模型的參數更新、GRPO、GSM8K 分數及 LoRA／GeoRA 對照尚未執行。

本文件集中記錄「做過什麼、使用什麼精度、得到什麼結果」。[PRECISION.md](PRECISION.md) 說明目前程式的精度約定；[CHECKS.md](CHECKS.md) 與 [SETONIX_GUIDE.md](SETONIX_GUIDE.md) 保留操作方法。

## 1. 先看整個實驗順序

我們先分析別人公開的模型權重，再實作 GeoRA 初始化，最後測試完整模型與混合精度。這幾類實驗回答不同的問題。

| 編號 | 實驗與目的 | 執行位置 | 實際範圍 | 結果 | 參數更新 |
|---|---|---|---|---|---|
| L1 | 公開 checkpoint 的權重幾何 | Mac CPU | 一個 Q，後擴展到 28 層 Q/K/V | 完成測量；不是 GeoRA 對照 | 無 |
| L2 | GeoRA 公式與單步更新 | Mac CPU | 第 0 層 Q，FP64 數學參照 | 初始化保持原輸出；A/B 可更新 | 一次 SGD，含噪聲目標 MSE |
| L3 | 在完整模型中只替換一個 Q | Mac CPU | 1.5B 全模型 FP32 forward | 初始化與保存／重載通過 | 無 |
| L4 | 全 196 層 GeoRA 初始化 | Mac CPU | 1.5B 全模型 FP32 forward | 初始化與保存／重載通過 | 無 |
| L5 | 自動檢查流程的回歸測試 | Mac CPU | 隨機初始化的小型 Qwen，FP32/BF16 | 兩種模式各 27 項通過 | 各一次 AdamW |
| S1 | GPU 環境與原始模型 | Setonix MI250X | BF16 矩陣乘法、原始 1.5B forward | 有限值檢查通過 | 無 |
| S2 | 全 196 層 FP32 初始化 | Setonix 登入節點 CPU | SVD、因子、殘差與保存 | 五項檢查通過 | 無 |
| S3 | 預設混合精度全模型檢查 | Setonix MI250X | 原始模型 vs 初始化 GeoRA | 初始化 logits 誤差 17.1538%，未通過 | 在 backward 前停止 |
| S4 | 全模型 FP32/BF16 精度診斷 | Setonix MI250X | 同一初始化、同一輸入的對照 | FP32 誤差很小；BF16 分布已有變化 | 無 |
| L6/S5 | 第 0 層 attention 精度定位 | Mac CPU / Setonix CPU、GPU | 五種運算路徑的局部對照 | FP32 投影方案顯著降低局部誤差 | 無 |
| L7 | 候選 helper 預檢與 MPS 能力探針 | Mac CPU / MPS；Setonix CPU | 小 Qwen 14 投影；MPS 16×16 BF16 乘法 | dtype、恢復、梯度通路通過；MPS 小乘法有限 | 無 optimizer；單投影 backward |
| S6 | 完整模型候選精度比較 | Setonix MI250X | 三種模式 × 三個短輸入，196 投影 | 候選改善誤差，但均未通過 2% 門檻 | 無 |

本機 L1～L6 及 L7 的小模型預檢使用 **CPU**。L7 另外確認 MPS 的 16×16 BF16 乘法可執行；沒有在 MPS 上執行完整 1.5B 模型或精度對照。

## 2. 精度必須分成幾件事

FP64、FP32、BF16 分別是 64、32、16 bit 浮點格式。BF16 保留較大的數值範圍，但有效數字比 FP32 少。同一個實驗可以同時使用多種格式。

| 名稱 | 指的是什麼 | 本次例子 |
|---|---|---|
| 來源保存精度 | 下載的 safetensors 裡，數字以什麼格式存放 | Qwen2.5-1.5B-Instruct 原始權重為 BF16 |
| 初始化計算精度 | 建 mask、做 SVD、算 A0/B0/F 的格式 | 正式全層初始化用 CPU FP32 |
| 運行時參數保存精度 | 模型在 RAM/VRAM 裡的參數 dtype | 目前 GPU 檢查的 F 為 BF16，A/B 為 FP32 |
| forward 計算精度 | 矩陣乘法、attention 等實際走的路徑 | BF16 autocast 可臨時降低 FP32 A/B 的乘法精度 |
| 檔案保存精度 | 我們寫出的 checkpoint 裡保存什麼、用什麼格式 | 緊湊 adapter 保存 FP32 A0/B0/A/B |
| 誤差統計精度 | forward 之後，計算範數、KL 等的格式 | logits 範數 FP32；診斷 KL/TV 在 CPU FP64 |
| 更新精度 | 梯度、可訓練參數及 optimizer 狀態的格式 | 設計為 FP32 A/B、梯度與 AdamW moments |

**把 BF16 原始權重轉 FP32，會精確保留已保存的 BF16 數值，不會恢復下載前丟失的有效數字。** 同樣，把 BF16 forward 的 logits 轉 FP32/FP64 做統計，也不會把那次 forward 變成高精度計算。

### 2.1 使用的模型與來源精度

來源精度直接讀取 safetensors 檔案標頭核對，沒有僅依靠 config 的 `dtype` 字段推斷。完整記錄見 [checkpoint_headers.json](reports/checkpoint_headers.json)。

| 用途／名稱 | 模型 | 固定 revision | 檔案內 tensor 精度 |
|---|---|---|---|
| 幾何分析 `sft_base` | `Qwen/Qwen2.5-Math-1.5B` | `4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2` | 338 個 BF16 |
| 幾何分析 `before` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562` | 339 個 BF16 |
| 幾何分析 `after` | `agentica-org/DeepScaleR-1.5B-Preview` | `e3f524ce413a296b4d388e7560dd5c82c1c56725` | 兩片共 339 個 FP32 |
| GeoRA 實作 `geora_base` | `Qwen/Qwen2.5-1.5B-Instruct` | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | 338 個 BF16 |

`before` 一個權重檔、`after` 兩個權重分片加 index，是下載檔案的組織方式。它們的來源保存精度也不同，不能用檔案個數判斷模型能力。checkpoint 保存精度本身不能證明作者訓練時每一步的運算精度。

### 2.2 正式 GeoRA 初始化做了什麼

目前全層配置固定為 `r=16`、`alpha=32`、`rho=0.2`，因此 `c=alpha/r=2`。每個原始線性權重 `W_pre` 的形狀為 `[輸出維度, 輸入維度]`。

```text
原始 BF16 W_pre → CPU FP32
    ↓
對 W_pre 做 SVD → rank-r 重建 → 按元素幅值取 spectral mask
W_pre 的元素幅值 → Euclidean mask
    ↓
兩個 bool mask 取聯集 → 保留原權重對應元素 → W_geo
    ↓
W_geo 做 SVD → 取前 r 個成分 → FP32 A0、B0
    ↓
FP32 F = W_pre - c·B0A0，凍結 F
A = A0、B = B0，只有 A/B 可訓練
    ↓
保存 FP32 A0/B0/A/B 及 manifest
```

兩個 mask 都使用 `rho` 分位數和 `<=`；重複值可使實際保留比例略高於 `rho`。聯集保留比例通常大於單個 mask 的比例，不是最終只有 20% 元素。

若 `W_geo=U_geo Σ_geo V_geo^T`，則：

```text
B0 = U_geo[:, :r] @ diag(sqrt(sigma_geo[:r]))   # [輸出維度, r]
A0 = diag(sqrt(sigma_geo[:r])) @ Vh_geo[:r, :]  # [r, 輸入維度]
F  = W_pre - c * (B0 @ A0)
```

這份實作用 `B @ A` 的命名。初始實數等式是 `F+cB0A0=W_pre`；矩陣乘法的浮點舍入和拆成兩條 forward 分支會產生數值差異。訓練期間 F 保持不變；相對原始權重的更新是 `c·(BA-B0A0)`。

28 個 block，每個替換 Q/K/V/O/gate/up/down，共 **196 層**，排除 `lm_head`。共有 **392 個可訓練 A/B tensors、18,464,768 個可訓練參數**；A0/B0 是凍結 buffers，原有 bias 也凍結。

## 3. 本機實驗

### L1：公開模型權重幾何，沒有在本機訓練模型

原有專案 `/Users/tianbaochen/RLVR` 讀取公開 checkpoint：以 `before→after` 的權重差觀察 RLVR 更新；另外用 `sft_base→before` 作為 SFT/蒸餾段的探索性比較。這是已發布模型的分析，沒有自行跑 SFT 或 GRPO，也沒有安裝 GeoRA。

| 部分 | 單個 Q notebook | 全 28 層 Q/K/V 腳本 |
|---|---|---|
| 輸入 | 第 0 層 Q，`1536×1536` | 84 個矩陣；Q `1536×1536`，K/V `256×1536` |
| 讀取與差值 | 保存值轉 FP32，差值 FP32 | 保存值直接轉 FP64，差值 FP64 |
| SVD／子空間重合 | FP32 | FP64 |
| 方向更新能量 | 將差值及基底轉 FP64 後計算 | FP64 |
| 權重輸出 | 提取的 `.pt` 矩陣為 FP32 | 不重新保存模型權重 |
| 分析輸出 | notebook、圖、JSON | JSON、CSV、圖 |
| 設備 | CPU | CPU，4 threads |

分析的兩個問題是：更新作用在原權重的哪些輸入方向上，以及更新本身能否低秩壓縮。前者用 `||ΔW v_k||² / ||ΔW||_F²`，其中 `v_k` 來自 **原權重** 的 SVD；後者對 **ΔW 本身** 做 SVD，累加它的奇異值平方。兩個圖的橫軸不是同一組方向。

第 0 層 Q 的單矩陣結果：原權重前 16 個方向承受 20.1552% 更新能量，均勻參考為 1.0417%；更新本身的 rank-16 近似保留 47.0793% 能量，最佳相對 Frobenius 誤差為 0.727466。因此，此矩陣既呈現可壓縮更新，也不能被描述為完全避開原權重強方向。

84/84 矩陣的 FP64 分析完成，總腳本記錄 33.581 秒。跨層中位數如下：

| 指標 | Q | K | V |
|---|---:|---:|---:|
| `||ΔW||_F / ||W_before||_F` | 0.00119336 | 0.00121614 | 0.00132982 |
| 原權重 Top-16 輸入子空間最小重合奇異值 | 0.9999954 | 0.9999800 | 0.9999479 |
| 原權重前 16 個方向的更新能量 | 4.4480% | 3.2926% | 1.7118% |
| 更新自身 rank-16 保留能量 | 27.3448% | 43.6187% | 52.0549% |
| 更新自身 rank-128 保留能量 | 63.4858% | 88.7731% | 89.4367% |

K/V 的 compact SVD 只提供 256 個輸入方向，不能當作完整的 1536 維輸入基底。腳本另外計算其餘 1280 維輸入空間的更新能量。

這些結果來自不同公開 checkpoint，訓練資料、步數與優化器未受控制；它們支持個別權重結構的觀察，不能單獨驗證 GeoRA 機制或證明 SFT/RLVR 的普遍差異。把兩邊都轉 BF16 後元素相同，也不代表訓練時那些位置的梯度為零。

來源：[FP64 執行記錄](reports/background_geometry/run.json)、[84 矩陣 CSV](reports/background_geometry/metrics.csv)、[分析報告](reports/background_geometry/report.md)、[第 0 層 Q](reports/background_geometry/layer_00_q.json)。原程式為 `~/RLVR/analyze_attention_weights.py`。

### L2：單矩陣 FP64 參照與一次 SGD

`geora_initialization.ipynb` 取 Qwen2.5-1.5B-Instruct 第 0 層 Q。來源 BF16 權重與 bias 轉 FP64，mask、兩次 SVD、A0/B0、F、單層 forward、梯度及更新全部用 CPU FP64。

輸入是隨機 `[4,1536]`；目標設定為「原始層輸出 + 0.1 × 隨機噪聲」。做一次 SGD，learning rate 0.1，loss 為 MSE。這是驗證「F 不動、A/B 可以動」的數學測試，不是語言模型訓練或 GRPO。

| 測量 | 結果 |
|---|---:|
| mask 聯集保留元素 | 34.6968% |
| `W_geo` rank-16 近似保留能量 | 6.7742% |
| 初始有效權重相對誤差 | 1.4053e-17 |
| 初始單層輸出相對誤差 | 1.0758e-15 |
| 一次更新後 MSE | 0.00987366 → 0.00973757 |
| F／A／B | F 精確不變；A、B 均改變 |

沒有保存權重 checkpoint，只保存 notebook 的程式與執行輸出。`W_geo` 的最佳 rank-r 近似不保證保留大量能量，這個單矩陣結果就是例子。

保留的[原始 notebook 程式與文字輸出](reports/local_notebook_outputs.json)來自 `~/RLVR/geora_initialization.ipynb`；倉庫中的同名 notebook 是清空輸出的教程副本。

### L3：一個 Q 替換進完整模型

`geora_model_check.ipynb` 在 CPU 讀入完整 FP32 模型，使用 eager attention。這個早期版本先用 FP64 算初始化，再把 A/B 轉 FP32，用 FP32 因子計算 F。完整模型 forward 是 FP32，誤差範數使用 FP64；沒有 backward 或 optimizer。

有效 Q 權重相對誤差為 7.5915e-9；全模型 logits 相對誤差 **1.1914e-5（0.0011914%）**，最大絕對差 0.000497043。最後位置 argmax 不變；保存／重新載入後 logits 最大差為 0。

早期 `q_module.safetensors` 保存 **A、B、F、bias 四個 FP32 tensors**，共 9,640,248 bytes。此格式包含 F；不要與後來全層的緊湊 adapter 格式混為一談。

來源：[單 Q metadata](reports/local_single_q/metadata.json)，原始執行輸出與權重位於 `~/RLVR/geora_model_check.ipynb`、`~/RLVR/outputs/geora_single_q_check/`。

### L4：全 196 層 CPU FP32 初始化與重載

正式全層版本全部使用 CPU FP32：原權重載入、mask、兩次 SVD、因子、殘差、模型參數與完整 forward。沒有 autocast。固定輸入為 42-token 純 prompt，系統文字為 `Give a short answer.`，使用 `eval()` 和 eager attention；只測試初始化，optimizer steps 為 0。

| 測量 | 結果 |
|---|---:|
| 196 層初始化時間 | 130.0205 秒 |
| 最大逐層有效權重相對誤差 | 1.1859e-8 |
| 完整模型 logits 相對誤差 | **2.6065e-5（0.0026065%）** |
| 完整模型 logits 最大絕對差 | 0.002243996 |
| 最後位置 argmax | 17 → 17 |
| 初始化 checkpoint 重載後 logits 最大差 | **0，逐元素完全相同** |

`adapter.safetensors` 保存 **784 個 FP32 tensors：每層 A0/B0/A/B**，共 147,800,440 bytes（約 141 MiB）。**不保存 F、完整原始模型或 optimizer。** 重載時讀入固定原始模型，再用 A0/B0 在 CPU FP32 重建 F，載入當前 A/B；不重新 SVD。

來源：[checks.json](reports/cpu_initialization/checks.json)、[manifest.json](reports/cpu_initialization/manifest.json)。本機產物在 `~/geora-repro/outputs/geora_full_model_check/`，是從原有 RLVR 專案保留的已執行產物；建立新倉庫時沒有再跑一次完整模型。

### L5：小模型驗證檢查程式，不代表 1.5B 已更新成功

`tests/check_validation_workflow.py` 建立隨機小型 Qwen：2 layers、hidden size 32、intermediate size 48、vocab 64、`r=2, alpha=4`。CPU FP32 與 CPU BF16 模式各通過 27 項檢查。

兩種模式都以 CPU FP32 初始化；FP32 模式整個 forward 用 FP32。BF16 模式僅將凍結參數轉 BF16、保留 FP32 A/B，再做 BF16 autocast。loss 使用 FP32 logits；A/B 梯度與 AdamW moments 為 FP32，各更新一次。

檢查包括只有 A/B 可訓練、凍結參數不變、更新後輸出改變，以及 FP32 adapter 和 optimizer 保存／重載。故意將 A/B 錯轉 BF16 會被拒絕。測試產物在臨時目錄，完成後刪除。

另有 `tests/check_precision_diagnostic.py` 檢查誤差／KL 計算和無更新的診斷路徑。這些測試驗證程式邏輯，不提供真實模型的任務分數。

## 4. Setonix 實驗

### 4.1 環境與檔案位置

| 項目 | 本機 | Setonix |
|---|---|---|
| 程式 | `/Users/tianbaochen/geora-repro` | `$MYSOFTWARE/geora/code` |
| Python 環境 | 本機獨立 `uv`／`.venv` | 容器內 `venv --system-site-packages` |
| Python / PyTorch | 3.12.13 / 2.14.1 | 3.12.3 / `2.7.1a0+gite2d141d` |
| Transformers | 5.19.0 | 5.19.0 |
| GPU runtime | 這些實驗使用 CPU | HIP `6.3.42134-a9a80e791` |
| checkpoint | 連到 `~/RLVR/checkpoints/geora_base` | `$MYSCRATCH/geora/checkpoints/geora_base` |
| 執行結果 | `outputs/`，小型記錄在 `reports/` | `$MYSCRATCH/geora/runs/` |

Setonix module 是 `pytorch/2.7.1-rocm6.3.3`，容器提供 ROCm PyTorch；自己的 venv 補裝依賴，不另裝 Mac 版 PyTorch。環境在 `$MYSOFTWARE/manual/software/geora-environments/py312-rocm633`。

本次 `$MYSOFTWARE=/software/projects/pawsey0807/btian`、`$MYSCRATCH=/scratch/pawsey0807/btian`。登入主機的 `setonix-01` 等編號會改變，不是固定執行地址。

### S1：GPU 環境與原始模型 forward

Job `50574180` 確認容器可見一個 AMD Instinct MI250X 邏輯 GPU，63.98 GiB。隨機 `512×512` BF16 矩陣乘法輸出全部有限；不是模型或 GeoRA 檢查。互動 allocation 最後因兩分鐘時限結束，Slurm 記為 `TIMEOUT`，不影響已完成的矩陣測量。

Job `50574324` 執行 `scripts/check_gpu_base_model.py`：原始模型直接以 BF16 載入 GPU，在 inference mode 做 eager forward，沒有安裝 GeoRA、生成答案或更新參數。logits 為 BF16，形狀 `[1,40,151936]`，全部有限。BF16 路徑內部的 RMSNorm／softmax 仍有 FP32 計算，不能稱所有運算都為 BF16。

腳本時間 13.296 秒，PyTorch peak allocated memory 2.962 GiB，Slurm 整個 allocation 40 秒。來源：[50574324.json](reports/setonix_base_forward/50574324.json)。這項只確認模型能載入與產生有限 logits，不判定回答能力。

### S2：登入節點上的全 196 層 FP32 初始化

`jobs/submit_checks.sh` 先在登入節點的容器 Python 執行 `scripts/prepare_geora_initialization.py`，預設 2 CPU threads。依使用者已確認的權限，不申請專門 CPU allocation；此階段也沒有使用 GPU。

流程與本機正式初始化相同：原始 BF16→FP32，兩次 CPU FP32 SVD，FP32 A0/B0/F，逐層重建有效權重，再保存 FP32 A0/B0/A/B。此階段沒有完整模型 logits forward 或 optimizer。

五項檢查通過，最大有效權重相對誤差 **1.1725e-8**；完整腳本 915.292 秒，約 15 分 15 秒。終端最後一層的 `914.9s` 是當時的累計進度，不是完整腳本時間。

保存目錄：

```text
$MYSCRATCH/geora/initializations/login-20261009T115357Z-2781659/
  adapter.safetensors          FP32 A0/B0/A/B，784 tensors
  manifest.json
  initialization_checks.json
```

來源：[初始化檢查](reports/setonix_initialization/initialization_checks.json)。Setonix 因子是在該機 CPU 重新初始化得到，沒有把 Mac 因子直接當作 Setonix 初始化。跨平台 SVD 不要求因子逐元素完全相同。

### S3：全模型混合精度檢查在更新之前停止

第一次 GPU job `50578377` 因初始化目錄未傳到 batch script，在 Python 啟動前停止。後續改成明確傳入目錄參數，復用 S2 初始化，不重新 SVD。

Job `50578622` 執行 `scripts/check_geora_training.py`，採用以下精度：

1. 從原始模型和 FP32 A0/B0，在 **CPU FP32 重建 F**。
2. 將凍結參數（包括 F）轉 **BF16**；A/B、A0/B0 保持 **FP32**。
3. 模型搬到 GPU，使用 **BF16 autocast** 做 GeoRA forward。FP32 A/B 的乘法可臨時選 BF16；參數本身仍是 FP32。
4. 獨立載入凍結的原始模型作 reference，用相同輸入及 attention 設定比較。

這裡系統文字為 `Answer briefly.`；輸入是 40-token prompt 加 `5` 和 EOS，共 42 tokens。它與 L3/L4 的 42-token 純 prompt 不同，不能只按 token 數把跨實驗誤差當成同一輸入的硬體對照。先檢查初始化 logits，再決定是否允許 backward。

| 檢查 | 結果 |
|---|---|
| 只有預期的 FP32 A/B 可訓練；初始因子 FP32 | 通過 |
| 凍結權重 BF16；reference 為原始模型 | 通過 |
| 原始與 GeoRA forward 有限，`[1,42,151936]` | 通過 |
| 全部 logits 相對誤差 | **0.17153804（17.1538%），超過 2% 門檻** |
| 最大 logits 絕對差 | 7.7265625 |
| 舊檢查的最後位置 KL | 6.1946e-8 nats |

來源：[50578622.json](reports/setonix_bf16_gate/50578622.json)。**作業失敗後沒有 backward、AdamW step、更新後 adapter 或 optimizer checkpoint。** 程式寫好了這些檢查，不代表它們已在 1.5B 模型上執行。

最後位置在 EOS 之後，不是預測 `5` 的位置；很小的最後位置 KL 不能代表其他位置分布接近。因此保持原有門檻，另做逐位置診斷。

### S4：同一初始化的全模型 FP32/BF16 診斷

Job `50579441` 執行 `scripts/diagnose_geora_precision.py`。復用 S2 的 A0/B0，沒有新 SVD，也沒有更新參數。BF16 對照沿用 S3；FP32 對照則從原始權重與因子**重新重建 FP32 F**，使模型參數和 forward 全部用 FP32。不是把已舍入的 BF16 F 再轉回 FP32。

| 原始模型 vs GeoRA，相同精度比較 | BF16 路徑 | FP32 路徑 |
|---|---:|---:|
| 全 logits 相對誤差 | 17.1538% | **0.00087605%** |
| 最大絕對差 | 7.72656 | 0.000378609 |
| 去除每個位置共同偏移後的相對誤差 | 19.1363% | 0.00106140% |
| 42 個位置平均 KL，nats | 0.105074 | 3.1363e-10 |
| 最大位置 KL，nats | 0.879461 | 3.4016e-9 |
| 平均 total variation | 0.104253 | 6.0438e-6 |
| 各位置 top-1 token 一致比例 | 92.8571% | 100% |

logit position 39 預測答案 token `5`（token ID 20）：原始 BF16 模型給它 **0.316340** 機率，GeoRA BF16 給 **0.0696066**。這表明差異影響了機率分布，不能僅用 logits 的共同偏移解釋。

診斷也單獨比較原始模型的 BF16 與 FP32 路徑：此固定輸入的 logits 相對差為 30.8695%，平均 KL 0.409115。這是另一個比較基準，不能與「同精度原始模型 vs GeoRA」混在一起，也不代表任務分數下降 30.9%。

來源：[逐層／逐位置診斷](reports/setonix_precision_diagnostic/50579441.json)、[診斷作業記錄](reports/setonix_precision_diagnostic/50579441_checks.json)。Slurm `COMPLETED` 和診斷的 `completed` 表示測量成功完成；**不表示 S3 的初始化門檻通過**。

## 5. 本機與 Setonix 的 attention 精度定位

### L6/S5：固定同一輸入，只測第 0 層

`scripts/probe_first_attention.py` 只讀取 42 個 token 的 embedding 行、第 0 層 input norm、Q/K/V/O 權重和保存的因子；不建立完整 1.5B 模型，也不重新 SVD。依序在 Mac CPU、Setonix CPU preflight、Setonix GPU job `50579802` 執行。

每一行都比較**同一運算設定的原始 attention 與 GeoRA attention**，測量的是 O 投影輸出 `[1,42,1536]`，在 residual 相加之前。它不是全模型 logits、生成準確率或 GSM8K 分數。

探針的 Q/K/V/O 和 F/A/B **在內存中保留 FP32**；native 用 autocast 臨時選 BF16 投影，會產生權重舍入。這和 S3 的「F 直接存 BF16」在參數存儲上不同，不能說兩個程式的每個步驟完全相同。

令 P 為 attention softmax 機率，以下 `P×V` 的 V 是 attention 的 value tensor，與 SVD 的右奇異向量矩陣不同。

| 模式 | Q/K/V/O 投影 | QK 分數、縮放與 mask | softmax / P×V |
|---|---|---|---|
| `native` | BF16 autocast | BF16 路徑 | softmax FP32→BF16；P×V BF16 |
| `fp32_scores` | 同 native | FP32 | softmax FP32→BF16；P×V BF16 |
| `fp32_core` | 同 native | FP32 | softmax 與 P×V FP32；context 轉 BF16 |
| `fp32_projections` | F/A/B 及投影計算 FP32，各投影輸出轉 BF16 | 同 native | 同 native |
| `fp32_all` | FP32 | FP32 | FP32 |

前四種使用 BF16 input norm 輸出與 BF16 RoPE 返回值；最後一種使用 FP32。RMSNorm 內部的 variance/rsqrt、RoPE 頻率/trig 本來就有 FP32 步驟。**native 的 softmax 本來就是 FP32**，不是加精度模式才把它提高。

| O 投影輸出的相對誤差 | Mac CPU | Setonix MI250X |
|---|---:|---:|
| `native` | 12.4369% | 12.3341% |
| `fp32_scores` | 3.5089% | 3.5347% |
| `fp32_core` | 3.5056% | 3.5336% |
| `fp32_projections` | 0.0497751% | 0.0234853% |
| `fp32_all` | 0.00350389% | 0.000636743% |

Mac 和 Setonix 使用各自已保存的初始化因子，並非完全相同的 tensor；這個表提供一致趨勢，不把數值差全歸因於硬體。兩台手動重建的 native eager attention，都與各自安裝的 Transformers 實作完全一致，原始／GeoRA 兩邊的比較誤差都是 0。

Setonix native 的 QK 分數絕對值最大 22528。使用完全相同的量化 Q/K 改在 FP32 重算，native 分數的最大差是 104.5879；原始模型 vs GeoRA 的最大分數差為 128。head 1、query position 34 中，原始模型有 21 項相同的 17536 分數，attention 機率各約 1/21；GeoRA 一項成為 17664 後，機率集中到該位置（1.0）。這裡的機率是分配給前文位置的 attention 權重，不是輸出 vocabulary 的 token 機率。

這支持「大數值 QK 分數的 BF16 舍入，能放大少量投影差異」的解釋。softmax 使用 FP32 不能恢復 QK 乘法已丟失的數字。只提高分數精度仍有約 3.5% 局部誤差；保留 FP32 F 並提高投影精度，局部誤差更小。

**相對 S3 的預設完整模型路徑，`fp32_projections` 既保留 FP32 F，又採用 FP32 投影計算，尚未單獨拆分兩者效果。** 在本節探針內，native 和其他模式本來都保留 FP32 F，模式差異主要是運算路徑。 截至 S5，這個局部低誤差尚未驗證能延伸到完整模型；後續 S6 測量了完整模型，候選均未達到門檻。預設精度設定沒有被自動取代。

來源：[Mac CPU](reports/attention_probe/local_cpu.json)、[Setonix CPU preflight](reports/attention_probe/setonix_cpu_preflight.json)、[Setonix GPU 50579802](reports/attention_probe/50579802.json)。GPU 核心計算 1.298 秒，腳本 7.202 秒，allocation 22 秒；PyTorch peak allocated memory 0.130 GiB。

### L7：先在本機及 Setonix CPU 驗證候選 helper

`tests/check_forward_precision.py` 使用兩層、14 個投影的小型 Qwen。Mac CPU 與 Setonix 容器 CPU 都實際通過以下檢查：原始／GeoRA 投影的內部線性計算 FP32、輸出 BF16；候選 context 正常退出或拋出例外後恢復原 forward；拒絕已舍入的 BF16 F；FP32 QK／mask 與 BF16 P×V 確實執行；A/B 的 FP32 梯度通路保留。最後一項只做單投影 backward，沒有 optimizer step。這不是完整模型的更新測試。

本機另做 16×16 MPS 隨機矩陣乘法，BF16 autocast 的輸出為 BF16，所有數值有限；PyTorch 2.14.1 的 MPS built／available 都為 true。這只確認小乘法能力，沒有測量完整模型誤差或跨後端一致性。

浮點格式相同不代表算子、autocast、加總順序或舍入結果相同。[PyTorch 數值精度說明](https://docs.pytorch.org/docs/2.7/notes/numerical_accuracy.html) 明確指出跨 CPU／GPU／版本不保證 bitwise 一致；[MPS 文件](https://docs.pytorch.org/docs/2.14/notes/mps.html) 說明其使用 Metal 的執行路徑。現有 CPU 與 AMD 局部探針都出現誤差放大，支持問題並非只在 AMD 上發生；因兩邊因子並不完全相同，還不能量化硬體的獨立影響。完整驗收應在之後實際訓練的 Setonix AMD 路徑完成。

來源：[預檢執行摘要](reports/forward_precision_preflight.json)、[可重跑的小模型檢查](tests/check_forward_precision.py)。

### S6：完整 1.5B 模型的三種 forward 路徑

Job `50584822` 使用 `scripts/check_full_forward_precision.py`。復用 S2 的保存因子，從原始 CPU FP32 權重重建 F，沒有新 SVD。先測兩個候選，最後才把凍結參數轉 BF16 測 native，避免把已舍入的 F 上轉冒充原 FP32 F。每一種模式都對**原始模型與 GeoRA 同時使用相同設定**。

| 模式 | 196 個 Q/K/V/O/gate/up/down | attention 分數 | 其餘參數／激活 |
|---|---|---|---|
| `native` | 凍結 F／原始權重 BF16，A/B FP32；乘法 BF16 autocast | 原有 eager 路徑 | BF16 混合精度 |
| `fp32_projections` | F／原始權重、A/B 保留 FP32；投影禁用 autocast、計算 FP32；輸出轉 BF16 | 同 native | embedding、norm、lm_head BF16；原有 BF16 激活路徑 |
| `fp32_projections_scores` | 同上 | QK、縮放、mask FP32；softmax FP32→BF16；P×V 仍 BF16 | 同上 |

三個輸入分別為原有短算術 prompt＋`5`＋EOS（42 tokens）、同一 prompt 不接答案（40 tokens），以及牛頓第二定律的純 prompt（42 tokens）。沒有生成答案；所有 logits 均是固定輸入的 forward 結果。

| 原始 vs GeoRA logits 相對誤差 | 算術＋答案 | 算術純 prompt | 物理純 prompt |
|---|---:|---:|---:|
| `native` | 17.1538% | 17.5993% | 16.8106% |
| `fp32_projections` | 3.7423% | 3.8353% | 3.7052% |
| `fp32_projections_scores` | 3.4787% | 3.5755% | 3.5206% |

**九次比較均未通過原有 2% logits 相對誤差門檻，沒有放寬門檻。** 候選能改善完整模型，但單層 attention 的極小誤差不能直接外推到 28 層的端到端輸出。報告同時保存所有位置的 KL／TV 和 top-1；最後位置 KL 小仍不足以代表其他位置。

算術＋答案輸入的逐位置摘要：

| 模式 | 平均 KL，nats | 最大位置 KL，nats | 平均 TV | top-1 一致 |
|---|---:|---:|---:|---:|
| `native` | 0.105074 | 0.879461 | 0.104253 | 92.8571% |
| `fp32_projections` | 0.00505251 | 0.0442711 | 0.0246858 | 97.6190% |
| `fp32_projections_scores` | 0.00352558 | 0.0379122 | 0.0192750 | 100% |

position 39 預測答案 token `5`，各模式的 reference 也隨精度改變：

| 模式 | 原始模型 p(5) | GeoRA p(5) | 此位置 KL，nats |
|---|---:|---:|---:|
| `native` | 0.316340 | 0.0696066 | 0.277201 |
| `fp32_projections` | 0.428306 | 0.312465 | 0.0295903 |
| `fp32_projections_scores` | 0.468086 | 0.471574 | 0.000450326 |

不能把跨模式 p(5) 增大直接解讀為性能提高；這裡沒有評估任務分數。

**原始模型自身的控制組：** 以 native 原始模型為 reference，FP32 投影原始模型的 logits 相對差為 3.5185%，平均 KL 0.00443560；FP32 投影＋分數原始模型的差為 29.3145%，平均 KL 0.319377、最大 KL 5.03716。因此，候選改善的是「相同新精度下的 GeoRA 初始化等價性」，尚未證明保留原 native BF16 模型的數值行為。

另外，FP32 投影＋分數模式中，同一數學前綴在算術＋答案 position 39 與算術純 prompt 的最後位置，KL 分別為 0.000450326 和 0.0165406。運算形狀／序列長度會影響浮點執行結果；本報告沒有單獨定位是哪一個算子導致，不能將這個差異歸因為某一硬體缺陷。

保存因子前後逐元素相同，沒有參數梯度，`optimizer_steps=0`、`backward_calls=0`、`svd_calls=0`。候選的 PyTorch peak allocated memory 為 **10.8574 GiB**，包括同時保留原始模型和 GeoRA；native 為 6.0114 GiB，不能當作訓練顯存需求。整個 Python 腳本 62.358 秒，三種模式的比較段約 2.312／0.815／0.772 秒，其他時間包含載入與準備。

申請一個邏輯 GPU、時限 3 分鐘，實際 allocation **87 秒**；Slurm `COMPLETED`、退出碼 `0:0`，完成後 queue 不再有此作業。診斷成功完成並不表示初始化 gate 通過。

來源：[完整原始報告 50584822.json](reports/full_forward/50584822.json)、[三分鐘作業腳本](jobs/full_forward_precision.sbatch)。程式 commit 為 `340b403891ec588c491dc76517437bc5beea5e19`。本次沒有改變預設訓練精度。

## 6. 保存、重載與更新精度總表

| 實驗 | 初始化／殘差計算 | 運行參數 | forward | 我們保存的權重檔 | 實際更新 |
|---|---|---|---|---|---|
| L2 單 Q 參照 | FP64 | FP64 | FP64 | 無 | FP64 SGD 一步 |
| L3 單 Q 完整模型 | 初始化 FP64，A/B 轉 FP32 後算 F | 全 FP32 | FP32 | A/B/F/bias，FP32 | 無 |
| L4 本機全層 | CPU FP32 | 全 FP32 | FP32 | A0/B0/A/B，FP32 | 無 |
| L5 小模型 FP32 | CPU FP32 | 全 FP32 | FP32 | FP32 因子，另存 optimizer；臨時檔 | FP32 AdamW 一步 |
| L5 小模型 BF16 | CPU FP32 | 凍結 BF16；A/B、A0/B0 FP32 | BF16 autocast | 同上 | FP32 AdamW 一步 |
| S2 Setonix 初始化 | CPU FP32 | CPU 全 FP32 | 不測全模型 forward | A0/B0/A/B，FP32 | 無 |
| S3 Setonix 預設檢查 | CPU FP32 重建 F | 凍結 BF16；A/B、A0/B0 FP32 | BF16 autocast | 復用 S2；未寫更新檔 | 無，門檻失敗 |
| S4 FP32 診斷 | CPU FP32 重建 F | 全 FP32 | FP32 | JSON；不另存權重 | 無 |
| L6/S5 局部探針 | 用保存因子重建 FP32 F，無 SVD | 投影 F/A/B FP32；其他依模式 | 五種路徑見上表 | JSON；不另存權重 | 無 |
| S6 完整候選 | 用 S2 因子重建 FP32 F，無 SVD | 候選投影 FP32；其餘 BF16；native 最後轉換 | 三種路徑見上表 | JSON；不另存權重 | 無 |

AdamW 單步檢查的設計為 `lr=1e-4`、`weight_decay=0`、gradient norm 裁剪上限 1；loss 對答案 token 做 FP32 cross-entropy，A/B 梯度和 moment tensors 為 FP32。**這在小模型測過，在 Setonix 1.5B 上因前置門檻失敗而未執行。** FP32 更新參數不等於每次 forward 的乘法都是 FP32。

全層緊湊 checkpoint 保留初始 A0/B0，是為了重建凍結 F；當前 A/B 是要恢復的可訓練狀態。直接把當前 BA 加到原始 W_pre 會重複加入初始部分。不能把停用 adapter 得到的 F 當成原始 reference；本次 GPU 檢查使用獨立的原始模型。

## 7. 所有「誤差」到底比較什麼

同一個公式可以用在不同對象上，必須連同對象一起讀：

```text
relative_error = norm(actual - reference) / norm(reference)
```

| 指標 | actual / reference 是什麼 | 用途 |
|---|---|---|
| 有效權重誤差 | F+cB0A0 / W_pre | 初始化矩陣重建 |
| 局部投影誤差 | 用同一輸入送進 GeoRA / 原始投影 | 排除上游輸入變化 |
| attention O 輸出誤差 | 完整第 0 層 attention 的 GeoRA / 原始輸出 | 看 QK、softmax、P×V 如何放大差異 |
| 全模型 logits 誤差 | 同一固定輸入的 GeoRA / 原始 logits | 模型端到端初始化檢查 |
| reload 誤差 | 保存後重載 / 保存前，同設定 | checkpoint 恢復一致性 |

矩陣／tensor 範數指全部元素的 L2/Frobenius 範數；17% 是整體相對量，不是每個元素都差 17%，更不是準確率差 17%。零 reload 誤差只表示恢复同一運算狀態，不表示那個狀態與原始模型足夠接近。

全模型 logits 是每個位置對整個 vocab 的分數。診斷對每個位置做 softmax，計算 `KL(reference || GeoRA)` 和 total variation；後者是 `0.5·sum(abs(p-q))`。KL 使用 nats，不轉成百分比。KL/TV 在 CPU FP64 算，避免大詞表分布的減法／加總誤差；極小負 KL 可以是浮點舍入，不能解讀成負距離。

舊 gate 的最後位置 KL 用 FP32；新診斷用 FP64，所以 50578622 的 6.1946e-8 與 50579441 的 2.3912e-8 不是同精度的統計。兩者都接近零，且都不足以代表所有位置。

## 8. Setonix 已用資源與目前停在哪裡

2026-10-09 查詢前六個作業，2026-10-10 新增並查詢 S6 作業；`sacct -X` 的記錄如下。每個作業的 `AllocTRES` 都是 `gres/gpu=1,node=1,cpu=16,mem=29440M`；一個 node 是放置位置，這些作業沒有分配八個邏輯 GPU。`billing=128` 是 Slurm 的計費權重字段，不能直接解讀為 128 秒或八卡計費。

| Job | 工作 | 上限 | 實際 allocation | Slurm 狀態 |
|---|---|---|---|---|
| 50574180 | 互動 GPU 環境測試 | 2 分鐘 | 2 分 7 秒 | TIMEOUT |
| 50574324 | 原始模型 BF16 forward | 5 分鐘 | 40 秒 | COMPLETED |
| 50578377 | shell 傳參失敗 | 5 分鐘 | 8 秒 | FAILED |
| 50578622 | GeoRA BF16 初始化門檻 | 5 分鐘 | 1 分 5 秒 | FAILED |
| 50579441 | 全模型精度診斷 | 5 分鐘 | 1 分 27 秒 | COMPLETED |
| 50579802 | 第 0 層 attention 探針 | 1 分鐘 | 22 秒 | COMPLETED |
| 50584822 | 完整模型候選精度比較 | 3 分鐘 | 1 分 27 秒 | COMPLETED |

這七個作業的 allocation 時間合計 **7 分 16 秒，均為一個邏輯 GPU**；不包含登入節點 CPU 初始化。allocation 時間包含啟動和清理，與 Python 的核心計算時間不同，也不是 GPU utilization 或最終計費金額。來源：[Slurm 查詢記錄](reports/setonix_jobs.json)。

完整模型的 FP32 投影候選已測量，尚未通過。下一步建議先測試數值上等價的重排：

```text
原始：F x + c B(Ax)，F = W_pre - c B0A0
候選：W_pre x + c [B(Ax) - B0(A0x)]
```

實數代數上兩者相同；在初始 A=A0、B=B0 時，候選的低秩校正可直接相減為零，保留原始 BF16 分支。校正轉回原始分支 dtype 後再相加，以免意外改變後續 activation 精度。這是待驗證候選，尚未實作／執行，不能宣稱已通過。先在本機驗證初始化、A/B 梯度及保存重載，再用同一初始化與短輸入在 Setonix 完整驗收。初始化通過後才做 1.5B 的 A/B 單步更新與更新後重載、短 GRPO、固定預算的 LoRA／GeoRA 任務對照。

### 原始記錄的使用約定

初次整理只讀取已有檔案／遠端報告；2026-10-10 後續新增 L7／S6，實際申請了一次短 GPU 作業，預設訓練精度未改變。原始 RLVR 專案保留；重要的小型報告複製到本倉庫的 `reports/`，權重與環境不加入 Git。

舊 `extract_weight.ipynb` 的 summary 曾引用 kernel 遺留的角度變數，與當前重合度輸出不一致；本文件使用後續 FP64 腳本的重合奇異值，不沿用那些角度。舊全層 notebook 也殘留一次 NameError 和不同輪次的進度；全層數值以最終獨立 `checks.json`／manifest 為準，不能把 notebook 保存的所有輸出當作一輪乾淨的連續執行。
