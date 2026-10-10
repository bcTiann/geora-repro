# GeoRA 復現：本機與 Setonix 實驗記錄

更新：2026-10-10；新增 LoRA／GeoRA 共用五步、完整訓練恢復與成本記錄，遠端報告與 Slurm 狀態核對至 2026-10-10。涵蓋原有 `~/RLVR` 學習專案與目前 `~/geora-repro` 的實際執行結果。

**目前的結論：LoRA／GeoRA 各完成 5 步真實 GSM8K GRPO，並通過第 2 步存檔後重做第 3 步的精確恢復（GeoRA 52/52，LoRA 52/52）。** 原 residual BF16 失敗紀錄保留；任務效果與正式 benchmark 尚未驗證。

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
| L8/S7 | 等價重排、更新與保存重載 | Mac CPU；Setonix CPU／MI250X | 本機小模型＋完整 1.5B | 完整模型 43 項通過；初始化與重載誤差 0 | 各一次短 CE AdamW；不是 GRPO |
| L9/S8 | 短生成／cache／padding、步長比較 | Mac CPU；Setonix CPU／MI250X | 小模型＋完整 1.5B | 完整模型 64 項通過；相同生成路徑精確一致 | 三次獨立首步 CE；最後恢復初始化 |
| L10/S9 | GSM8K 與真實一次 GRPO 更新 | Mac CPU；Setonix CPU／MI250X | 完整 1.5B、2 題×4 回答 | CPU 預檢及完整模型 20/20 通過 | 一次真實 reward GRPO AdamW |
| L11/S10 | 共用 LoRA／GeoRA 五步與完整恢復 | Mac CPU；Setonix CPU／MI250X | 完整 1.5B、每方法五步＋一次重放 | 連續更新、精確恢復、首批一致通過 | 每方法五個真實 GRPO 有效步 |

本機 L1～L6、L7 的小模型預檢及 L8 使用 **CPU**。L7 另外確認 MPS 的 16×16 BF16 乘法可執行；沒有在 MPS 上執行完整 1.5B 模型或精度對照。

## 2. 精度必須分成幾件事

FP64、FP32、BF16 分別是 64、32、16 bit 浮點格式。BF16 保留較大的數值範圍，但有效數字比 FP32 少。同一個實驗可以同時使用多種格式。

| 名稱 | 指的是什麼 | 本次例子 |
|---|---|---|
| 來源保存精度 | 下載的 safetensors 裡，數字以什麼格式存放 | Qwen2.5-1.5B-Instruct 原始權重為 BF16 |
| 初始化計算精度 | 建 mask、做 SVD、算 A0/B0/F 的格式 | 正式全層初始化用 CPU FP32 |
| 運行時參數保存精度 | 模型在 RAM/VRAM 裡的參數 dtype | 預設 residual 的 F 為 BF16；S7 difference 的 W_pre 為 BF16；A/B 均 FP32 |
| forward 計算精度 | 矩陣乘法、attention 等實際走的路徑 | residual autocast 可降低 A/B 乘法精度；difference 的兩條低秩分支禁用 autocast，算 FP32 |
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

## 5. 本機與 Setonix：精度定位、生成與 GRPO 驗收

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

### L8/S7：等價重排保留原始 BF16 分支，完成機械更新與重載

本輪目標與驗收條件記錄於 [GOAL.md](GOAL.md)。新模式為：

```text
y = W_pre x + c [B(Ax) - B0(A0x)]，c = alpha/r
```

它與 `F x+c B(Ax)`、`F=W_pre-cB0A0` 在實數代數上等價，但浮點求值方式不同。實作 `forward_mode=difference` 直接在 fresh 原始模型上保留 W_pre，不先計算 F 再嘗試加回去。原始權重、bias、embedding、norm、lm_head 存 BF16；native attention 設定未變，仍是原有混合精度路徑。

兩條低秩分支共享同一份 FP32 輸入，A/B 和 A0/B0 均為 FP32，在 autocast 關閉的區域計算及相減。校正轉回原分支輸出 dtype 後相加。A0/B0 本身固定，**其分支仍保留對輸入的梯度**，不能用 `no_grad` 把前層梯度切掉。初始 A=A0、B=B0 時，兩條低秩輸出相同，校正直接為零。

**本機預檢：** 更新後的兩層線性網路，輸出與獨立的 FP64 有效權重公式相比，最大絕對誤差 1.073e-7；輸入及 A/B 梯度最大絕對誤差不超過 1.193e-7。兩層測試能檢查傳回前層的梯度，並非只看 A/B 是否有數字。小型 Qwen 的 FP32/BF16 模式各通過 29 項檢查，每種模式各做一次 AdamW。Setonix 容器 CPU 同樣通過這些檢查；舊 residual 模式的 FP32/BF16 小模型回歸各 28 項通過，誤轉 BF16 A/B 的負例仍被拒絕。

**完整模型：** GPU job `50585414` 復用 S2 保存因子，沒有完整模型 SVD。在與 S6 相同的三個固定短輸入上先檢查精確一致，通過後才做一次短答案 CE 更新和保存重載。

| 完整模型檢查 | 實際結果 |
|---|---|
| 196 個目標的凍結原始權重及 bias | 與獨立原始模型精確相同 |
| 算術＋答案、算術純 prompt、物理純 prompt | 初始化 logits 相對誤差／最大差均 0；KL／TV 均 0 |
| 原有 2% logits／0.02 nats 最後位置 KL 門檻 | 保留且通過；另加精確一致門檻 |
| 可訓練範圍 | 392 個 FP32 A/B tensors、18,464,768 個參數 |
| backward | 392 個梯度均 FP32、有限、非零；norm 範圍 0.49908～40.0952 |
| AdamW 後 A/B | 全部更新且有限；因子變化 norm 範圍 0.00638311～0.0378458 |
| 凍結原始參數／A0/B0／reference | 338 個凍結參數精確不變；初始因子不變；reference logits 不變 |
| 初始及更新後的模型重載 | logits 最大差／相對誤差均 0 |
| adapter、manifest、optimizer 重載 | 因子精確恢復；FP32 AdamW moments／step 精確恢復；mode 亦保存 |
| 有效矩陣更新 | 全 196 層有限、非零；Frobenius norm 範圍 0.0206274～0.120496 |

有效更新仍為 `c(BA-B0A0)`。統計透過 FP64 小 Gram 矩陣計算其 Frobenius norm，先寫成 `B(A-A0)+(B-B0)A0`，避免相近大矩陣能量相減，也不配置完整稠密 ΔW。小矩陣對照確認這個統計與直接形成 ΔW 的 FP64 範數一致。

全部 **43/43 項檢查通過**。這是本次復現的數值實作候選，不能因此聲稱作者的 BF16 實作也用了相同順序，或新模式與 residual 的浮點訓練軌跡完全相同。

#### 這次「更新正常」的具體邊界

`lr=1e-4`、`weight_decay=0`、gradient clipping norm 上限 1、AdamW 一步；對算術 prompt 後的 `5` 與 EOS 做 cross-entropy，沒有 rollout 或 GRPO。裁剪前梯度 norm 為 157.7638。

| 更新前後同一模型，同一輸入 | 測量 |
|---|---:|
| CE loss | 0.5805116 → 0.000190763 |
| logits 相對變化 | **46.7574%** |
| 最大 logits 絕對差 | 20.09375 |
| 所有位置平均／最大 KL，nats | **1.25955 / 11.7139** |
| 平均／最大 TV | 0.260667 / 0.974427 |
| top-1 token 一致比例 | 76.1905% |

這個大的輸出變化發生在**更新之後**，不同於之前初始化即產生偏差的問題。它證明 FP32 A/B 的更新可以影響 BF16 模型；不證明步長合適、策略改動受控、多步穩定或任務能力提高。單個已知答案的 loss 大幅下降不能當成評估分數。

梯度裁剪也不是 AdamW 參數／策略移動幅度的直接上限；第一次 Adam 的逐元素 moments 正規化會抵消大部分一致的梯度縮放。此處沒有 GRPO 的 ratio clipping 或 KL 項，因此不能把這次 CE 更新當成受信賴域約束的 RL 更新。

#### 保存內容與執行用量

Setonix scratch 實際產生：

```text
$MYSCRATCH/geora/runs/difference-check-50585414/
  initial_adapter.safetensors  FP32 A0/B0/A/B，約 141 MiB
  initial_manifest.json        optimizer_steps=0、forward_mode=difference
  trained_adapter.safetensors FP32 A0/B0/A/B，約 141 MiB
  manifest.json                optimizer_steps=1、forward_mode=difference
  optimizer.pt                 step／FP32 moments，約 142 MiB
  difference_checks.json
```

所有原始／更新後權重及 optimizer 檔案留在 scratch，Git 只保存報告與 manifest。兩份 manifest 同時記錄實際 `runtime_precision`；舊未標 mode 的因子仍預設使用 residual。載入有 mode 標記的 checkpoint 時，衝突的明確 mode 要在改動模型前拒絕。

一個邏輯 GPU、3 分鐘上限；Slurm `COMPLETED`、退出碼 `0:0`，實際 allocation **42 秒**，完成後 queue 為空。Python 腳本 22.744 秒，PyTorch peak allocated memory **12.3458 GiB**，包含 reference、快照和重載副本。`optimizer_steps=1`、`backward_calls=1`、`svd_calls=0`。完成的 optimizer step 立即寫入報告，避免後續重載失敗掩蓋已更新的事實。

來源：[Mac CPU](reports/difference/local_cpu.json)、[Setonix CPU](reports/difference/setonix_cpu.json)、[完整模型 50585414](reports/difference/50585414.json)、[初始 manifest](reports/difference/initial_manifest.json)、[單步後 manifest](reports/difference/trained_manifest.json)。執行程式 commit 為 `75037a65833bdd1cb1fb11e0b0453d92f45bc4e1`。

這輪目標的初始化／單步更新／序列化驗收已完成。後續 L9/S8 已補上獨立步長比較、短生成／KV cache 及 padding forward；多步策略變化、padding 訓練與 GRPO／外部 trainer 的整合仍需檢查。merge 必須用 `W_pre+c(BA-B0A0)`；不能按普通 LoRA 把 `cBA` 直接加回 W_pre。

### L9/S8：短生成、KV cache、padding 與独立首步的步長比較

這輪繼續使用 S7 的 `difference` 模式與原有因子，沒有 SVD、GRPO 或新的任務評估。原始凍結權重 BF16、A/B/A0/B0 FP32、兩條低秩分支及相減 FP32，校正轉回 BF16 後相加；loss 為 FP32 CE，更新與 AdamW moments 為 FP32。KL/TV 在 CPU FP64 計算，有效 ΔW 範數用 FP64 小 Gram 矩陣。沒有改變初始化或保存精度。

#### 生成與 padding 的比較對象

本機及 Setonix 容器 CPU 的兩層小型 Qwen，FP32/BF16 各 **21/21** 項通過：真實 Transformers `generate`、獨立 DynamicCache 預填及三步解碼、左側 padding，以及答案標籤屏蔽和 causal shift 的獨立 token 索引對照。

完整 1.5B 作業 `50596774` 的總檢查 **64/64 通過**。使用兩個長度為 40／42 token 的 prompt，帶左側 padding；greedy generation 最多 8 個新 token。未更新的 GeoRA 與原模型在**相同輸入布局、精度和 cache 路徑**下，valid logits、生成 token、原始 logits 和生成分數精確相同。每個模型建立自己的 cache，沒有共用可變 cache；預填 40 token 後，以固定 token 連續三步確認 cache 長度為 41、42、43。

同時測量原模型自身的不同計算路徑，避免錯把路徑差異歸因於 GeoRA：

| 原模型自身對照 | 本次 logits 相對差 |
|---|---:|
| 單條輸入 vs 左側 padding 批次，同有效 token | 4.0424%、3.8146% |
| cache 單 token vs 完整前文重新計算，三步 | 4.3867%、2.3018%、2.7660% |

這些路徑可能使用不同的矩陣乘法形狀和捨入。本次 cached／uncached 的短 greedy token 序列仍相同，但不能要求跨路徑 logits 精確相同，也不能把這些數字作為「誤差底限」從後面的更新變化中扣除。生成的精確一致驗收比較的是各路徑中的 GeoRA vs 原模型。padding 的 loss 標籤機制在小模型核對；完整模型做了 padding forward／generation，尚未做 padding batch 的訓練更新。

#### 每個步長都從初始化獨立出發

三次試驗各自原位恢復 A=A0、B=B0，清空梯度，新建空狀態的 AdamW；重算同一個答案 CE 梯度並裁剪 norm 上限 1。初始 loss 都是 0.5805116，裁剪前梯度 norm 都是 157.7638，裁剪後逐元素梯度精確相同。AdamW `betas=(0.9,0.999)`、`eps=1e-8`、`weight_decay=0`；每輪 optimizer 的 step 都是 1。

| LR | 全 196 層有效 ΔW 合併範數 | 更新前後整段 logits 變化 | 答案預測位置 KL，nats | 正確答案首 token 的機率 | 更新後 CE |
|---|---:|---:|---:|---:|---:|
| 1e-04 | 1.09068 | 46.7574% | 11.7139 | 0.31634 → 0.99997 | 0.0001907626 |
| 1e-05 | 0.109067 | 5.7308% | 1.4173 | 0.31634 → 0.93507 | 0.03704078 |
| 1e-06 | 0.0109067 | 4.3694% | 0.00664044 | 0.31634 → 0.36857 | 0.5047485 |

這裡 ΔW=`c(BA-B0A0)`，跨層合併範數為 `sqrt(sum(layer_norm**2))`，不是直接相加各層範數。答案預測位置為 `logits[:,39]`，比較整個詞表的分布；`logits[:,40]` 預測 EOS，`logits[:,41]` 已在 EOS 後，沒有參與該 CE 的監督。原始 JSON 同時保留全部位置、prompt 末位置和實際監督位置的指標。

LR 減少十倍時，有效權重變化約減少十倍，但 logits／概率變化不必線性縮放。`1e-6` 的 logits 仍改變 4.3694%，這是固定路徑下實際觀察到的更新後變化；本次未分離其數學更新與浮點放大的成分。答案預測 KL 0.00664、機率 0.31634→0.36857，表示本次首步改動較小，不能據此稱為已證明安全的 GRPO 學習率。

物理問題 prompt 的末位置 KL 依次為 0.0837636、0.0000507692、0.0000088030。三個輸入形式只有兩個不同問題：算術帶答案、算術純 prompt、物理純 prompt；前兩者的答案預測前文相同，不是獨立測試集。沒有 reward、GRPO ratio、KL regularizer 或任務分數。

各試驗均通過 FP32 梯度／參數有限、有效更新非零、所有凍結參數／A0/B0 不變及 reference 不變。最後 A/B 又恢復初始化，logits 精確恢復原始模型。報告的 `optimizer_steps=3` 和 `independent_optimizer_steps=3` 是三次獨立首步，`final_model_state=untrained_restored`；沒有連續三步训练，也沒有保存新的訓練 checkpoint。原 S7 的保存檔案未改動。

#### 載入超時與實際用量

第一次作業 `50596505` 在權重載入期間達到三分鐘時限，Slurm allocation 為 **197 秒**（包含結束清理）；只有初始化來源檢查完成，**backward=0、optimizer=0**。原始 JSON 因外部終止仍是 `running`，必須連同 Slurm `TIMEOUT` 判讀，不能宣稱檢查完成。step CPU time 約 18 秒、MaxRSS 約 4.41 GiB。

CPU faulthandler 定位到 safetensors mmap 的材料化／dtype 轉換；關閉 mmap、關閉異步載入後，直接讀原 scratch 檔仍在 120 秒達到 CPU timeout。這些觀察指向載入階段，沒有證據指向 GeoRA forward 或梯度錯誤，也沒有單獨證明底層儲存系統的原因。

為避免再次消耗 GPU 時間，以 CPU 串流複製到 software 的臨時 cache（**141.643 秒**），逐檔核對 SHA256，並確認載入的 **338 個 FP32 張量全部精確等於原 BF16 checkpoint 值**。臨時副本＋關閉 mmap／異步載入的 CPU 模型載入為 **7.061 秒**；驗證總時間 16.653 秒。聯合措施解決了本次載入問題，沒有分別測量每個措施的作用。原始模型仍在 scratch；臨時大型副本已清理，不提交 Git。

成功作業 `50596774` 一個邏輯 GPU、三分鐘上限，allocation **46 秒**，Python **26.969 秒**，peak allocated memory **9.4275 GiB**；Slurm `COMPLETED`、退出碼 `0:0`，完成後 queue 為空。本輪兩次 GPU allocation 合計 **243 秒（4 分 03 秒）**，包括超時，CPU 診斷／複製不佔 GPU allocation。

來源：[本機 CPU](reports/continuation/local_cpu.json)、[Setonix CPU](reports/continuation/setonix_cpu.json)、[載入診斷](reports/continuation/loading_diagnostic.json)、[超時原始報告](reports/continuation/50596505.json)、[完整模型 50596774](reports/continuation/50596774.json)。成功執行程式 commit 為 `ce3de48e7881a57fd4833f396876d0e231f355db`；運行時載入選項和完整副本校驗資訊在報告中。

本階段的短生成／cache／padding forward 及獨立首步測量完成。下一階段是 GRPO 的 reward、advantage、ratio、KL、reference 與多步更新整合；尚未證明多步穩定性、長序列生成或 GeoRA 任務效果。

### L10/S9：GSM8K 接入一次真實 GRPO 更新

目的：驗證「問題 → 採樣回答 → 數值 reward → 同題 advantage → clipped token loss＋reference k3 → A/B 更新」整條資料流。[操作與檔案分工](GRPO_SMOKE.md)、[逐步教學 notebook](notebooks/gsm8k_grpo_tutorial.ipynb)。

GSM8K 固定官方 commit `3101c7d5072418e28b9008a6636bde82a006892c`，7473 train／1319 test；從 train 分出 128 validation，剩餘 7345 train，16 題 smoke 只來自 train。SHA256、題目 ID 與來源見 [manifest](reports/grpo/data_manifest.json)。模型只讀問題和格式提示，不讀標準解答。唯一 `####` 後只有數值才能解析，Decimal 精確比較；截斷 training reward=0。

Mac／Setonix login CPU：資料／reward 11 項及 8792 參考答案、手算 GRPO 14 項、小模型 FP32/BF16 更新、固定寬度 rollout/scoring 22 項通過。小模型用人工 token／reward，與真實模型報告分開。Notebook 15 個 code cell 在本機 CPU 執行，沒有載入 1.5B。

GPU 不做 SVD。讀原始 BF16 checkpoint 的 FP32 值並載入未訓練 FP32 A0/B0；運行時 frozen base BF16、A/B/A0/B0 FP32、低秩分支及相減 FP32、校正轉回 BF16。softmax/logp/GRPO loss、AdamW moments 與保存的 adapter 使用 FP32，沿用 S7/S8 difference 模式。

採樣／打分固定整段 decoder shape、no cache、每個位置相同 `[8,1,hidden]` lm_head shape，未來 token 被 causal／attention mask 遮住。這是本次數值對齊選擇，吞吐較慢；沒有假設普通 cache 和全段 scoring 相同。預先固定 token max delta logp／ratio 各 2e-5，整段 max delta logp 1e-3。

Job **50601905**，程式 commit `294577e1d9d4bc651ca7d277edbf666ab3cdb063`，**20/20 通過**。第一批 2×4 回答共 1479 有效 token，reward `[[0,0,0,1],[0,0,0,0]]`，advantage `[-0.57735,-0.57735,-0.57735,1.73205,0,0,0,0]`。behavior/old/gradient-enabled current/reference 的有效 token 初始 logp 差異全為 0。train mode、dropout=0；完整左 padding batch backward 產生有限非零 FP32 梯度，AdamW lr=1e-6 更新 A/B。原權重、A0/B0/reference 不變；更新後 scoring 和短 stochastic sampling 通過。

**reward=0 不一定是算錯。** 第一題兩段雖寫出數字 4，一段缺 `####`，另一段寫 `#### 4 Scoops`；第三段數值錯且無標記。第二題 3/4 在 256 token 上限截斷，另一段給出 30000。真實文字見 [報告](reports/grpo/50601905.json) 和 notebook。不能把這組 reward 當作數學準確率；後續單獨報格式失敗、數值錯誤與截斷。

更新前 scalar loss 約 -1.49e-8（FP32 加總近零）、k3=0；更新後舊樣本 surrogate loss=0.0006171、k3=0.0009681，未證明單步改善。裁剪前梯度 norm=1.30023，有效 ΔW 合併 Frobenius norm=0.01092795。k3 對固定 token 直接求導，採用 GRPO surrogate 約定；更新後舊樣本均值不是精確 current-policy KL。

Slurm allocation **124 秒**，Python **104.02 秒**，峰值 allocated GPU memory **28.90 GiB**。CPU checkpoint staging 10.40 秒、校驗 10.86 秒，不佔 GPU。臨時 software copy 用後移除。更新一次的 FP32 factors／manifest／optimizer 保留於 `$MYSCRATCH/geora/runs/grpo-smoke-50601905/trained_adapter`，未覆寫 untrained 初始化；尚未驗收完整 RNG／資料位置 resume。

下一步：階段 4 的少量連續更新、LoRA 共用流程及完整訓練狀態恢復。固定順序保留同分組，各方法不能自行篩選 reward 更有訊號的題目。本次不是論文分數復現。

### L11/S10：LoRA／GeoRA 連續五步與完整訓練恢復

目的：驗證更新後的策略能持續採樣與更新，以及存檔後能否接著做同一次訓練。兩種方法共用 `scripts/check_grpo_continuation.py` 和 `configs/grpo_continuation.json`；流程與完整命令見 [GRPO_CONTINUATION.md](GRPO_CONTINUATION.md)。

本機／Setonix login CPU：LoRA 41/41，GeoRA／LoRA × FP32／BF16 的完整 boundary 恢復 4/4 通過。實際隨機抽樣、下一次 GRPO 更新、AdamW、scheduler 和 RNG 完全相同；另驗證來源不符、進行中的 rollout、overwrite 和 optimizer 參數綁定。這些小模型答案／reward 是人工機械檢查，與下列真實 GSM8K 結果分开。

完整模型沿用已保存的未訓練 GeoRA 初始化，沒有重做 SVD。LoRA 則使用私有 CPU generator 產生 Gaussian A、B=0；std=0.02 是我們明示的設定，沒有作者 baseline 程式可核對。兩者 rank=16、alpha=32，196 投影／18,464,768 可訓練參數。原始模型、data manifest、前 10 個 train IDs、prompt、reward、seed、每步 2 題×4 回答、最多 256 tokens、lr=1e-6、GRPO 損失及精度均共用。

精度：原始 frozen weights BF16；A/B、GeoRA A0/B0、低秩分支和相減 FP32，校正轉 BF16 後相加。log-softmax/loss、梯度、AdamW moments 與保存的 adapter 都是 FP32。LoRA 沒有 A0/B0 支路；reference 仍是固定原始模型。採樣／打分沿用固定寬度 no-cache 路徑，每步核對有效 token likelihood；此實作偏重可核對性，不能當作高吞吐框架的效率結果。

每一步存檔的內容是 current adapter、AdamW、constant scheduler、CPU/GPU/Python RNG、專用 rollout generator 和下一題的位置，另保存 base/init/data/config 來源及檔案 hash。checkpoint 位於 completed-rollout boundary，沒有進行中的 old batch。

第 2 步後存檔，正常做第 3 步，再恢復第 2 步重做第 3 步；題目、回答 token IDs、reward、advantage、behavior/old logp、更新後 A/B、AdamW、scheduler，以及所有記錄的 RNG 狀態逐項完全相同。恢復後繼續第 4、5 步。每方法 **5 個有效步，實際 6 次 rollout/backward/optimizer step**；額外的一次重放成本已計入 allocation。兩種方法的第一批回答和 behavior logp 也完全相同。

| 方法 | 作業／檢查 | 有效 completion tokens | Python 秒 | Allocation 秒 | 峰值顯存 GiB |
|---|---|---:|---:|---:|---:|
| GeoRA | 50622579，52/52 | 7537 | 461.654 | 480 | 29.008 |
| LoRA | 50622756，52/52 | 7593 | 412.549 | 434 | 28.874 |

本輪兩個作業依次使用一個邏輯 GPU，共 allocation **914 秒（15 分 14 秒）**。CPU 預檢／副本校驗不佔 GPU allocation。兩個作業 `COMPLETED`、queue 最後為空，臨時大型 model copy 已刪除；原始模型、資料、未訓練因子與每步 training checkpoints 保留在 scratch。來源 commit 為 `dc5487e7bb91f976b0ddf042b3b19cd2076ca120`。

這次不篩選混合 reward 題目；整組同分也保留。下表中 `reward sum` 是該步 8 個回答的 reward 總和；格式失敗和截斷同樣影響數值。它們是工程短程紀錄，不能據此判定 GeoRA 或 LoRA 的任務效果較好。

| 步 | GeoRA reward sum | LoRA reward sum | GeoRA 梯度範數 | LoRA 梯度範數 |
|---|---:|---:|---:|---:|
| 1 | 1 | 1 | 1.300229 | 0.774142 |
| 2 | 1 | 0 | 1.423936 | 0.000424 |
| 3 | 4 | 0 | 3.847309 | 0.000154 |
| 4 | 0 | 1 | 0.000093 | 0.459502 |
| 5 | 0 | 0 | 0.000144 | 0.000058 |

GeoRA：10 組中 6 組同分，40 回答中 21 次解析失敗、14 次截斷。LoRA：8 組同分，25 次解析失敗、13 次截斷；解析失敗與截斷可能重疊。更新後的短採樣／重新打分仍通過 likelihood 門檻；所有 frozen weights、GeoRA A0/B0 與 reference 不變。

GeoRA 的 34 個 reward=0 回答按互斥原因分為：14 個截斷、8 個已 EOS 但不能按格式解析、12 個已 EOS 且解析出的數值錯誤。21 個解析失敗與 14 個截斷有 13 個重疊，不能相加。實際回答中也有數值正確但格式不合的案例。reward 只查最終數值與格式，不驗證推理過程的每一句。

原始報告：[GeoRA](reports/grpo_continuation/50622579.json)、[LoRA](reports/grpo_continuation/50622756.json)、[共用設定與首批比較](reports/grpo_continuation/comparison.json)、[LoRA 本機 CPU](reports/grpo_continuation/lora_local_cpu.json)、[恢復本機 CPU](reports/grpo_continuation/resume_local_cpu.json)、[LoRA Setonix CPU](reports/grpo_continuation/lora_setonix_cpu.json)、[恢復 Setonix CPU](reports/grpo_continuation/resume_setonix_cpu.json)、[本輪副本校驗](reports/grpo_continuation/staging_record.json)。完整 answers/logp/precision/timing 與每步 checkpoint 路徑在原始 JSON 中。

下一步：接入較高吞吐的 rollout 路徑前，重新測量它和 scoring 的 likelihood 差異；再固定 pilot 的 reward/長度協定並做較長訓練及 validation。尚未做任務分數對照、長 response 或論文 Table 8 的 benchmark。

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
| L8/S7 difference | 復用 S2 初始因子，保留原 W_pre，無 SVD | 原始權重 BF16；A/B、A0/B0 FP32 | 原始 BF16 分支＋FP32 低秩差，輸出 BF16 | FP32 初始／更新因子，mode/runtime metadata，optimizer | FP32 AdamW 一步 |
| L9/S8 | 復用 S2 初始因子，無 SVD | 同 S7 | 同 S7，新增 generate／cache | JSON；無新訓練權重檔 | FP32 AdamW 三次獨立首步，最後還原 |
| L10/S9 | 復用 S2 初始因子，無 SVD | 同 S7 | 固定寬度 BF16 base＋FP32 低秩差；logp/loss FP32 | FP32 A/B/A0/B0、AdamW moments | 一次真實 GRPO |
| L11/S10 | GeoRA 復用 S2；LoRA Gaussian A／零 B，CPU FP32 | frozen BF16；A/B 和 GeoRA A0/B0 FP32 | 共用固定寬度 BF16 base＋FP32 adapter；logp/loss FP32 | adapter FP32、AdamW FP32 moments、scheduler、RNG、資料位置 | 每方法五個 GRPO 有效步＋一次重放 |

AdamW 單步檢查的設計為 `lr=1e-4`、`weight_decay=0`、gradient norm 裁剪上限 1；loss 對答案 token 做 FP32 cross-entropy，A/B 梯度和 moment tensors 為 FP32。**S3 residual 的 1.5B 前置門檻失敗，當時未執行；S7 difference 的 1.5B 已完成一次並通過重載。** FP32 更新參數不等於每次 forward 的乘法都是 FP32。

全層緊湊 checkpoint 保留初始 A0/B0，是為了在 residual 模式重建凍結 F、或在 difference 模式計算初始校正；當前 A/B 是要恢復的可訓練狀態。直接把當前 BA 加到原始 W_pre 會重複加入初始部分。residual 模式不能把停用 adapter 得到的 F 當成原始 reference；本次 GPU 檢查使用獨立的原始模型。

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

2026-10-09 查詢前六個作業，2026-10-10 新增並查詢 S6／S7／S8／S9／S10 作業；`sacct -X` 的記錄如下。每個作業的 `AllocTRES` 都是 `gres/gpu=1,node=1,cpu=16,mem=29440M`；一個 node 是放置位置，這些作業沒有分配八個邏輯 GPU。`billing=128` 是 Slurm 的計費權重字段，不能直接解讀為 128 秒或八卡計費。

| Job | 工作 | 上限 | 實際 allocation | Slurm 狀態 |
|---|---|---|---|---|
| 50574180 | 互動 GPU 環境測試 | 2 分鐘 | 2 分 7 秒 | TIMEOUT |
| 50574324 | 原始模型 BF16 forward | 5 分鐘 | 40 秒 | COMPLETED |
| 50578377 | shell 傳參失敗 | 5 分鐘 | 8 秒 | FAILED |
| 50578622 | GeoRA BF16 初始化門檻 | 5 分鐘 | 1 分 5 秒 | FAILED |
| 50579441 | 全模型精度診斷 | 5 分鐘 | 1 分 27 秒 | COMPLETED |
| 50579802 | 第 0 層 attention 探針 | 1 分鐘 | 22 秒 | COMPLETED |
| 50584822 | 完整模型候選精度比較 | 3 分鐘 | 1 分 27 秒 | COMPLETED |
| 50585414 | difference 初始化、單步更新與重載 | 3 分鐘 | 42 秒 | COMPLETED |
| 50596505 | 繼續測試，載入超時、無更新 | 3 分鐘 | 197 秒 | TIMEOUT |
| 50596774 | 生成／cache／padding、獨立 LR 首步 | 3 分鐘 | 46 秒 | COMPLETED |
| 50601905 | 一次真實 GSM8K GRPO 更新 | 10 分鐘 | 124 秒 | COMPLETED |
| 50622579 | geora 五步／完整恢復 | 10 分鐘 | 480 秒 | COMPLETED |
| 50622756 | lora 五步／完整恢復 | 10 分鐘 | 434 秒 | COMPLETED |

這十三個作業的 allocation 時間合計 **29 分 19 秒，均為一個邏輯 GPU**；不包含登入節點 CPU 初始化。allocation 時間包含啟動和清理，與 Python 的核心計算時間不同，也不是 GPU utilization 或最終計費金額。來源：[Slurm 查詢記錄](reports/setonix_jobs.json)。

等價重排、L9/S8 生成與步長比較、L10/S9 一次真實 GRPO 流程均完成。L11/S10 的共用五步與完整恢復亦完成；接著驗證高吞吐 rollout，再做 pilot 和任務評估。固定預算的 LoRA／GeoRA 任務對照要在這些流程通過後安排；S7 的短 CE 單步不是完整論文復現。

### 原始記錄的使用約定

初次整理只讀取已有檔案／遠端報告；2026-10-10 後續完成 L7／S6、L8／S7 及 L9／S8；S6／S7 各一次 GPU 作業，S8 一次載入超時、一次完成；S9 一次完整 GRPO smoke 完成；S10 兩方法各一次五步／恢復作業完成。各階段的運算精度與用量分別記錄在上文。原始 RLVR 專案保留；重要的小型報告複製到本倉庫的 `reports/`，權重與環境不加入 Git。

舊 `extract_weight.ipynb` 的 summary 曾引用 kernel 遺留的角度變數，與當前重合度輸出不一致；本文件使用後續 FP64 腳本的重合奇異值，不沿用那些角度。舊全層 notebook 也殘留一次 NameError 和不同輪次的進度；全層數值以最終獨立 `checks.json`／manifest 為準，不能把 notebook 保存的所有輸出當作一輪乾淨的連續執行。
