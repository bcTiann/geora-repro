# GeoRA：Setonix 逐步操作教程

更新：2026-10-10。本文件保留環境、儲存與提交作業的操作步驟。完整實驗流程、精度、結果與資源用量集中在 [EXPERIMENT_RECORD.md](EXPERIMENT_RECORD.md)。

本次目標：在 Setonix 的一個邏輯 GPU 上，完成 BF16 forward、FP32 A/B 單步更新及保存／載入檢查，之後接短 GRPO。

## 程式倉庫與環境的安排

本機程式倉庫為 `~/geora-repro`。原有 `~/RLVR` 保留作為學習與權重分析專案。

- 本機：在新倉庫使用自己的 `uv` 環境。
- Setonix 程式：放在 `$MYSOFTWARE/geora/code`，與本機使用同一份 Git 版本。
- Setonix 環境：使用 PyTorch 容器內的 Python 建立獨立 venv，放在 `$MYSOFTWARE/manual/software/geora-environments`。
- 模型、資料與執行產物：另外存放在 `$MYSCRATCH/geora`，由程式目錄的相應連結指向它們。

本機的 `pyproject.toml` 與 `uv.lock` 記錄已使用的 Mac 依賴配置。Setonix 的追加依賴已確認，版本見下方狀態表；不將本機 `.venv` 傳到 Setonix，也不在 Setonix 直接執行本機的 `uv sync`。

## 目前已知與待確認

使用者已確認可以使用 GPU 節點，且 module 查詢結果包含：

- `pytorch/2.7.1-rocm6.3.3`。
- 主機端 ROCm 有多個版本，預設 `rocm/6.4.1`。
- Singularity 有多個版本，預設 `singularity/4.1.0-slurm`。

容器 venv、依賴、模型及原始模型 GPU forward 已確認。使用者提供了早期執行輸出；後續也透過 SSH 直接核對報告並執行局部探針。原 residual BF16 初始化門檻未通過；difference 模式已通過完整 1.5B 初始化、單步更新／重載、短生成／cache／padding 與獨立步長比較。另已完成一次真實 GRPO 更新；連續更新和完整訓練恢復尚未驗證；當前結果見 [EXPERIMENT_RECORD.md](EXPERIMENT_RECORD.md)。

## 已收到的第 1 步結果

以下數值來自使用者在 2026-10-08 貼出的 Setonix 輸出，不是遠端自行執行的結果。

| 項目 | 實際值 |
|---|---|
| USER | `btian` |
| PAWSEY_PROJECT | `pawsey0807` |
| HOME | `/home/btian` |
| MYSOFTWARE | `/software/projects/pawsey0807/btian` |
| MYSCRATCH | `/scratch/pawsey0807/btian` |

`hostname` 本次為 `setonix-05`，但使用者確認每次連線編號可能不同。它只用來辨識當下的主機；不把這個名稱寫成固定的連線、目錄或作業目標。

`/software` 與 `/scratch` 的個人目錄都屬於 group `pawsey0807`，權限為 `drwxr-s---`，帶有 setgid，讓新建子項繼承該目錄的 group。

| 儲存區 | 使用者容量 | 專案容量 | 容量使用率 | 使用者檔案數／比例 |
|---|---:|---:|---:|---:|
| /scratch | 8.01 GiB | 54.34 TiB | 5.3%（專案） | 25／0.0% |
| /software | 1.26 GiB | 193.28 GiB | 75.5%（專案） | 24,635／9.9% |
| /home | 442.65 MiB | 不適用 | 43.2%（使用者） | 322／3.2% |

`/software` 的 75.5% 是整個專案的使用率；`/scratch` 與 `/software` 的檔案數比例則是個人的。這次輸出支持沿用程式／環境放 software、權重與執行資料放 scratch 的安排。

Compute 表只列出 `pawsey0807` 的 Allocation 1、Usage 0。這份表沒有顯示 GPU account 的實際額度；使用者已確認 GPU 節點可用，實際 GPU 作業已確認使用 `pawsey0807-gpu`，見第 5 步。

## 先認識三種工作位置

| 名稱 | 功能 | 我們的操作 |
|---|---|---|
| 登入節點 | 登入、查詢、編輯、提交 GPU 作業，以及本專案已確認允許的 CPU 初始化 | 預設 2 threads 執行 SVD 初始化 |
| GPU 計算節點 | 由 Slurm 分配的 GPU 執行環境 | 執行模型 forward、backward 與訓練 |
| 你的儲存目錄 | 程式與資料所在的共享檔案系統 | 計算節點與容器讀取這些檔案 |

登入成功不等於已取得 GPU。取得資源後仍要在作業中確認 PyTorch 能看見 GPU。[Pawsey PyTorch 手冊](https://pawsey.atlassian.net/wiki/spaces/US/pages/51931230/PyTorch)

## 第 1 步：查自己的目錄與配額

### 1.1 目錄的用途

`$名稱` 表示讀取一個 shell 環境變數，例如 `$MYSCRATCH` 會展開成你實際的 scratch 路徑。

| 環境變數／服務 | 我們放什麼 | 原因 |
|---|---|---|
| `$HOME` | 小型個人設定，例如 shell 與 SSH 設定 | 個人主目錄，容量與檔案數配額較小 |
| `$MYSOFTWARE` | GeoRA 程式、Python 環境、Slurm 腳本 | 使用者軟體的存放位置 |
| `$MYSCRATCH` | 模型權重、資料集、cache、log、訓練 checkpoint | 作業使用的工作資料位置 |
| Acacia | 長期保留的重要結果與 checkpoint 副本 | 長期物件儲存；之後單獨設定 |

這張表是本專案的配置，依據 [Setonix 軟體環境](https://pawsey.atlassian.net/wiki/spaces/US/pages/51929054/Setonix+Software+Environment) 與 [檔案系統用途](https://pawsey.atlassian.net/wiki/spaces/US/pages/51925876/Pawsey+Filesystems+and+their+Use)。實際路徑以登入後的環境變數為準。

Scratch 未存取達 21 天的檔案可能被清除，不能作為重要結果的唯一保存位置；判準是存取時間。[官方 Scratch 清理說明](https://pawsey.atlassian.net/wiki/spaces/US/pages/51926296/Files+in+Scratch+Were+Deleted)

### 1.2 現在執行的命令

在目前的 Setonix 登入終端執行：

```bash
hostname

printf 'USER=%s\nPAWSEY_PROJECT=%s\nHOME=%s\nMYSOFTWARE=%s\nMYSCRATCH=%s\n' \
  "$USER" "$PAWSEY_PROJECT" "$HOME" "$MYSOFTWARE" "$MYSCRATCH"

ls -ld "$HOME" "$MYSOFTWARE" "$MYSCRATCH"

pawseyAccountBalance -s
```

每個命令的意思：

1. `hostname`：顯示目前登入哪一台主機。
2. `printf`：顯示使用者、預設專案，以及三個實際目錄。
3. `ls -ld`：顯示目錄本身的權限、擁有者與 group；`-d` 不展開目錄內容。
4. `pawseyAccountBalance -s`：查儲存使用量與配額。容量和檔案數都要看；有剩餘 GB 仍可能用完檔案數配額。[官方檔案系統手冊](https://pawsey.atlassian.net/wiki/spaces/US/pages/51925876/Pawsey+Filesystems+and+their+Use)

**本步已完成，實際結果見上表。** 如果未來换預設專案或路徑，重新執行這段查詢。

`PAWSEY_PROJECT` 用於預設儲存專案。GPU 作業的 Slurm account 通常加 `-gpu`；這不表示要把儲存目錄的專案名也改成帶 `-gpu`。[GPU 作業手冊](https://pawsey.atlassian.net/wiki/pages/viewpage.action?pageId=1202094082)

## 第 2 步：確認 PyTorch 容器環境（已完成）

`module` 用於載入系統提供的軟體設定。這個 PyTorch module 是容器化安裝：它提供 `python3` 等包裝命令，替我們呼叫容器裡的程式，並載入所需 Singularity 依賴。

先沿用這個 module；它標示的 ROCm 6.3.3 是容器的建置版本，不必為了主機預設版本而把它替換掉。[Pawsey PyTorch 手冊](https://pawsey.atlassian.net/wiki/spaces/US/pages/51931230/PyTorch)

在目前的登入終端逐一執行：

```bash
module show pytorch/2.7.1-rocm6.3.3
module load pytorch/2.7.1-rocm6.3.3
module list
type -a python3 bash
```

- `show`：查看這個 module 的設定。
- `load`：將設定與包裝命令加入目前的 shell。
- `list`：確認載入了哪些 module 與依賴。
- `type -a`：查看 shell 實際找到的 `python3`、`bash`；後面建立環境會使用容器的入口。

### 2.1 已收到的 module 結果

使用者提供的輸出確認：

- `pytorch/2.7.1-rocm6.3.3` 已載入，其依賴為 `singularity/4.1.0-mpi`。
- `python3` 與 `bash` 的第一個命令來源位於 `/software/setonix/2025.08/containers/wrappers/quay.io/pawsey/pytorch/2.7.1-rocm6.3.3/bin/`。
- module 說明中，`python3` 包裝命令透過 `singularity exec` 執行容器內的 `/usr/bin/python3`；`bash` 同樣執行容器內的 `/bin/bash`。
- `venv` 包裝命令會使用容器 Python 執行 `-m venv --system-site-packages`。

因此目前的入口是：

```text
終端輸入 python3 → 包裝命令 → Singularity → 容器內 /usr/bin/python3
```

容器檔案位於系統的 `/software/setonix/2025.08/containers/sifs/`，完整檔案路徑由 module 的 `SINGULARITY_CONTAINER` 指定。容器內的既有軟體由系統提供；我們的追加套件環境會另外保存在個人的 software 目錄。

`module avail conda` 沒找到 module，只表示目前的 module 查詢沒有 Conda 項目。我們採用官方文件提供的容器 Python 加 venv 路線。

### 2.2 接著查容器內的實際版本

執行以下小型查詢。它會讀取 Python／PyTorch 版本，以及三個後續會用到的套件版本：

```bash
python3 -c '
import sys
import torch
from importlib.metadata import version, PackageNotFoundError

print("Python:", sys.version)
print("Executable:", sys.executable)
print("PyTorch:", torch.__version__)
print("Torch location:", torch.__file__)
print("HIP:", torch.version.hip)
print("GPU visible:", torch.cuda.is_available())

for package in ("transformers", "safetensors", "huggingface-hub"):
    try:
        print(package + ":", version(package))
    except PackageNotFoundError:
        print(package + ": not installed")
'
```

`HIP` 是 PyTorch 對應的 AMD ROCm/HIP 建置版本；不是 `None` 才符合這次的環境方向。

**module、包裝命令及容器 Python 已由使用者輸出確認。** 實際 Python 版本與 PyTorch 能否匯入，以執行結果為準。套件顯示 `not installed` 只表示該套件尚未安裝，之後在 venv 補齊。

登入節點上 GPU 不可見可以是正常結果；GPU 檢查會在第 5 步的計算節點進行。

## 第 3 步：建立程式與資料目錄（程式及模型目錄已建立）

採用以下目錄安排，datasets 與後續訓練產物尚待建立：

```text
$MYSOFTWARE/geora/
  code/                  Python 程式
  jobs/                  Slurm 腳本

$MYSOFTWARE/manual/software/
  geora-environments/    我們補裝套件的環境

$MYSCRATCH/geora/
  checkpoints/           原始模型及初始 A0/B0
  datasets/              GSM8K 等資料
  cache/                 套件下載與模型 cache
  runs/                  log、測量、訓練輸出
```

這些是本專案選定的子目錄；程式、venv、模型及快取已建立，訓練所需的其他目錄之後再補。

### 3.1 從 GitHub 取得這份程式

GitHub 首次推送完成後，在 Setonix 登入終端執行以下命令。目標目錄若已存在，先確認內容，避免把兩份程式混在一起。Private 倉庫要先配置 GitHub 認證；若 clone 要求登入或報錯，回傳訊息後再繼續。

```bash
mkdir -p "$MYSOFTWARE/geora"
git clone https://github.com/bcTiann/geora-repro.git "$MYSOFTWARE/geora/code"
cd "$MYSOFTWARE/geora/code"
git status --short --branch
git log -1 --oneline
```

`git clone` 會建立 `code` 目錄並下載倉庫中的程式。這一步不建立 Python 環境，也不下載模型。權重、資料與初始化 adapter 會在第 6 步放到 scratch，並讓程式讀取那裡的檔案。

往後本機推送新提交後，在 Setonix 的程式目錄更新：

```bash
git pull --ff-only
git log -1 --oneline
```

`--ff-only` 只接受直接向前更新；如果 Setonix 有自己的提交導致分歧，先回傳訊息再處理。

## 第 4 步：容器 Python 與 venv（已完成）

環境位於 `$MYSOFTWARE/manual/software/geora-environments/py312-rocm633`，使用容器 Python 的 `venv --system-site-packages`。本次容器內 uv 安裝遇到 TLS 憑證驗證失敗，因此採用 Python 自帶的 venv。

已確認：Python 3.12.3、PyTorch `2.7.1a0+gite2d141d`、HIP `6.3.42134-a9a80e791`，PyTorch 來源仍為容器的 `/usr/local/lib/python3.12/dist-packages/torch`。追加套件為 Transformers 5.19.0、Safetensors 0.8.0、Hugging Face Hub 1.33.0；安裝時用環境中的 `torch-constraint.txt` 約束保留既有 PyTorch。

每次手動进入容器後激活：

```bash
source "$MYSOFTWARE/manual/software/geora-environments/py312-rocm633/bin/activate"
```

本機仍使用 uv；Setonix 不直接同步包含 Mac PyTorch 版本的根目錄 lockfile。[Pawsey 容器 venv 指南](https://pawsey.atlassian.net/wiki/spaces/US/pages/51931230/PyTorch)

## 第 5 步：短 GPU 環境檢查（已完成）

使用者成功申請 `gpu-dev`、account `pawsey0807-gpu`、一個邏輯 GPU、`--time=00:02:00`；job ID 為 `50574180`。實際看到一個 AMD Instinct MI250X，顯存 63.98 GiB。512×512 的 BF16 GPU 矩陣乘法完成，輸出全為有限值。

這確認兩分鐘申請在此次被接受，以及容器的 GPU 基本運算可用。這一步只檢查基本矩陣乘法；後續原始模型 forward 的結果見第 6.2 步，difference 模式的 backward／更新與後續測試見 [實驗記錄 L8/S7、L9/S8](EXPERIMENT_RECORD.md)。

GPU 申請只指定節點和 GPU 數，系統配套提供 CPU 與記憶體；`--cpus-per-task` 放在後續的 `srun` 執行步驟。[官方 GPU 作業指南](https://pawsey.atlassian.net/wiki/spaces/US/pages/51929056/Example+Slurm+Batch+Scripts+for+Setonix+on+GPU+Compute+Nodes)

## 第 6 步：模型檔案與 CPU 初始化（已完成）

下載腳本已回報固定版本下載完成。模型存於 `$MYSCRATCH/geora/checkpoints/geora_base`，倉庫的 `checkpoints/geora_base` 以符號連結指向該位置。`check_local_setup.py` 已通過版本記錄及權重檔案標頭檢查：338 個 tensors。196 層 CPU 初始化已由使用者回傳成功，保存位置見第 7 步。

原始模型固定為 `Qwen/Qwen2.5-1.5B-Instruct`，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。

早期 residual 檢查從原始 checkpoint 與 A0/B0 重建 F。下一階段採已驗證的 difference 路徑：從 fresh 原始模型保留 W_pre，載入未訓練的 FP32 因子；凍結權重 BF16、A/B/A0/B0 FP32。兩種路徑的詳細约定見 [PRECISION.md](PRECISION.md)；確認來源與檔案完整後再執行模型檢查。

## 第 6.1 步：自動執行完整模型 BF16 forward（已完成，命令供重跑）

先退出任何仍存續的互動 GPU allocation，回到登入節點。在倉庫更新程式並提交五分鐘上限的短作業：

```bash
cd "$MYSOFTWARE/geora/code"
git pull --ff-only
mkdir -p "$MYSCRATCH/geora/runs/logs"
sbatch --account=pawsey0807-gpu \
  --output="$MYSCRATCH/geora/runs/logs/base-model-%j.log" \
  jobs/base_model_forward.sbatch
```

記下 `Submitted batch job` 後的 ID。以該 ID 取代 `<jobid>` 查詢與讀取輸出：

```bash
squeue -j <jobid>
cat "$MYSCRATCH/geora/runs/logs/base-model-<jobid>.log"
```

作業中 `pytorch-exec` 使用已建立的 venv Python，無須手動進容器或激活。腳本只從本地 checkpoint 載入 BF16 原始模型，使用短問題做一次前向計算，檢查 logits 是否有限，記錄形狀、精度、PyTorch 配置、GPU 峰值張量記憶體及執行時間。沒有安裝 GeoRA 或更新參數，也不以這次 forward 判定回答能力。

結果位於 `$MYSCRATCH/geora/runs/base-model-<jobid>/base_model_forward.json`。通過時 log 顯示 `Base-model GPU BF16 forward passed.`。作業完成或報錯即釋放資源，五分鐘是上限；若超時，先看 log 再決定是否增加時間。

## 第 6.2 步：原始模型 GPU forward 結果

作業 `50574324` 的原始模型 BF16 forward 已通過有限值檢查。精度、形狀、時間與原始 JSON 統一見 [EXPERIMENT_RECORD.md 的 S1](EXPERIMENT_RECORD.md)。

## 第 7 步：自動執行 GeoRA 初始化與單步檢查

詳見 [CHECKS.md](CHECKS.md)。在登入節點一次提交：

```bash
cd "$MYSOFTWARE/geora/code"
git pull --ff-only
/bin/bash jobs/submit_checks.sh
```

原本 CPU `work` 作業的提交被 Slurm 拒絕，未啟動計算或提交 GPU。使用者隨後確認這個專案可以在登入節點跑 CPU 初始化，流程改為直接在登入節點的容器 Python 做完整 FP32 SVD，預設 2 threads。初始化成功後才提交 GPU 作業（1 個邏輯 GPU、5 分鐘上限），完成自動釋放。CPU 階段前景執行，進度同時印到終端與 scratch 日誌；它不會出現在 Slurm 作業列表。

GPU 檢查：凍結參數 BF16、A/B FP32、初始化與原始模型的 logits 差異、每個 A/B 的梯度與更新、凍結參數及 A0/B0 不變、AdamW FP32 狀態，以及初始化／訓練後的 checkpoint 重載。獨立原始 reference 的 logits 在更新後必須不變。結果逐項寫入 JSON，失敗會停止。

小模型回歸測試與 Setonix 196 層 CPU 初始化已通過；早期 GPU 傳參失敗已修復，後續預設 BF16 路徑在初始化 logits 門檻停止。歷史作業、實際精度與結果統一見 [EXPERIMENT_RECORD.md](EXPERIMENT_RECORD.md)。

已保存的初始化可直接復用。在登入節點更新程式後，只重跑 GPU：

```bash
cd "$MYSOFTWARE/geora/code"
git pull --ff-only
/bin/bash jobs/submit_gpu_check.sh \
  "$MYSCRATCH/geora/initializations/login-20261009T115357Z-2781659"
```

提交腳本檢查初始化檔案，再將目錄作為明確參數傳入 GPU batch script。此次不執行 CPU SVD；提交後自動顯示 GPU 日誌，Ctrl+C 只停止觀看。job `50578622` 在初始化 logits 門檻停止，尚未更新參數。全模型精度診斷與局部 attention 探針後續已完成，結果見 [EXPERIMENT_RECORD.md](EXPERIMENT_RECORD.md)；重跑診斷命令見 [CHECKS.md](CHECKS.md)。

## 第 8 步：GSM8K／GRPO（已完成一次真實更新）

整體安排與驗收見 [REPRODUCTION_PLAN.md 階段 3](REPRODUCTION_PLAN.md)。先在本機/登入節點準備 GSM8K 與答案解析，核對可手算的 GRPO loss、advantage、ratio、KL 和 completion mask；完成 CPU 預檢後，再申請一個邏輯 GPU 做短 rollout 與一次真實 reward 更新。

實際入口為 `scripts/check_grpo_training.py`、`configs/grpo_smoke.json`、`jobs/grpo_smoke.sbatch`。CPU 到 GPU 提交／live log 命令見 [GRPO_SMOKE.md](GRPO_SMOKE.md)。job 50601905：20/20、一個邏輯 GPU allocation 124 秒；資料和訓練產物保留在 scratch，臨時 software copy 已清理。後續再驗證連續更新、完整恢復及 LoRA 共用流程。

## 當前進度

環境、CPU 初始化已通過；原 residual BF16 路徑的失敗紀錄保留。difference 等價重排已通過完整 1.5B 初始化、一次 A/B 更新和模型／optimizer 重載；本輪 [GOAL.md](GOAL.md) 完成。後續生成／cache／padding forward、獨立首步比較已通過 64 項；一次真實 GRPO 更新亦已完成；下一階段做連續更新、LoRA 共用流程及完整訓練恢復。所有逐項結果統一見 [EXPERIMENT_RECORD.md](EXPERIMENT_RECORD.md)，重跑入口見 [CHECKS.md 文末](CHECKS.md)。
