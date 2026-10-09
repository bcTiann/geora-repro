# GeoRA：Setonix 逐步操作教程

更新：2026-10-09。這份文件隨實際輸出逐步補齊；有「待確認」的步驟，先回傳輸出再繼續。

本次目標：在 Setonix 的一個邏輯 GPU 上，完成 BF16 forward、FP32 A/B 單步更新及保存／載入檢查，之後接短 GRPO。

## 程式倉庫與環境的安排

本機程式倉庫為 `~/geora-repro`。原有 `~/RLVR` 保留作為學習與權重分析專案。

- 本機：在新倉庫使用自己的 `uv` 環境。
- Setonix 程式：放在 `$MYSOFTWARE/geora/code`，與本機使用同一份 Git 版本。
- Setonix 環境：使用 PyTorch 容器內的 Python 建立獨立 venv，放在 `$MYSOFTWARE/manual/software/geora-environments`。
- 模型、資料與執行產物：另外存放在 `$MYSCRATCH/geora`，由程式目錄的相應連結指向它們。

本機的 `pyproject.toml` 與 `uv.lock` 記錄已使用的 Mac 依賴配置。Setonix 的追加依賴尚待實際版本確認；不將本機 `.venv` 傳到 Setonix，也不在 Setonix 直接執行本機的 `uv sync`。

## 目前已知與待確認

使用者已確認可以使用 GPU 節點，且 module 查詢結果包含：

- `pytorch/2.7.1-rocm6.3.3`。
- 主機端 ROCm 有多個版本，預設 `rocm/6.4.1`。
- Singularity 有多個版本，預設 `singularity/4.1.0-slurm`。

已收到第 1 步的實際路徑與配額，以及第 2 步的 module／包裝命令輸出。PyTorch module 已載入；接著確認容器內 Python、PyTorch 與已有套件。GPU 可見性及 GeoRA 訓練尚未實測。

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

Compute 表只列出 `pawsey0807` 的 Allocation 1、Usage 0。這份表沒有顯示 GPU account 的實際額度；使用者已確認 GPU 節點可用，GPU 作業的 account 會在後續步驟核對。

## 先認識三種工作位置

| 名稱 | 功能 | 我們的操作 |
|---|---|---|
| 登入節點 | 登入、查詢、編輯腳本、提交作業 | 先查路徑、配額、module |
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

## 第 2 步：確認 PyTorch 容器環境（現在執行）

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

**module 與包裝命令已確認；回傳這次 Python 查詢結果後再建立環境。** 實際 Python 版本與 PyTorch 能否匯入，以執行結果為準。套件顯示 `not installed` 只表示該套件尚未安裝，之後在 venv 補齊。

登入節點上 GPU 不可見可以是正常結果；GPU 檢查會在第 5 步的計算節點進行。

## 第 3 步：建立程式與資料目錄（路徑與配額已確認，待建立）

預計採用以下結構，之後再給建立命令：

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

這些是我們選的專案子目錄，不是系統已經建立好的路徑。建立後會確認目錄與 group，再放檔案。

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

## 第 4 步：只補裝需要的 Python 套件（待容器版本確認）

Pawsey 建議在容器內使用其 Python 建立 `venv --system-site-packages`，把環境保存在 `/software` 下。這樣環境能沿用容器既有的 PyTorch，再補裝缺少的套件。[官方 Python 環境安裝說明](https://pawsey.atlassian.net/wiki/spaces/US/pages/51931230/PyTorch)

目前本機 `pyproject.toml` 固定 `torch==2.14.1`；Setonix 則使用 Pawsey 的 ROCm／Cray PyTorch。GPU 環境需要獨立的依賴配置，直接沿用本機 `uv sync` 可能替換掉它。等實際版本確認後，再決定追加依賴的安裝指令；本機仍維持 `uv run`。

本步要核對：Python 與 PyTorch 來源、Transformers／Safetensors 等套件是否已有，以及補装前後 PyTorch 沒有被替換。

## 第 5 步：取得一個 GPU，先跑環境檢查（待專案確認）

Slurm 是分配計算資源的系統。我們先用一個邏輯 GPU，以小腳本確認：

- `torch.version.hip` 有值，使用 ROCm build。
- GPU 型號、數量與顯存。
- BF16 的矩陣乘法和 backward 能執行。
- A/B 為 FP32，AdamW 更新正常。

資源設定與 `srun` 執行設定會依 [Pawsey GPU 作業手冊](https://pawsey.atlassian.net/wiki/pages/viewpage.action?pageId=1202094082) 填寫。account 使用確認後的實際專案，不猜專案代碼。

## 第 6 步：放入模型與已保存的初始化

傳輸或下載原始模型到 scratch，並核對固定 revision、檔案大小及完整性；再放入已保存的初始因子與共用程式。

原始模型固定為 `Qwen/Qwen2.5-1.5B-Instruct`，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。

完成這一步後，GPU 上從原始 checkpoint 與 A0/B0 重建 F，依 [PRECISION.md](PRECISION.md) 將凍結部分轉 BF16、保留 A/B FP32。確認檔案完整後才執行模型檢查。

## 第 7 步：GeoRA 單步檢查，再接短 GRPO

依序量測同精度初始化 logits、A/B 梯度與更新、凍結權重不變，以及更新後保存／載入的一致性。

這些通過後，接 GSM8K 的短 GRPO，驗證生成、答案檢查器、reward、reference 及訓練更新。短程流程跑通後再設定 LoRA／GeoRA 的比較實驗。

## 狀態記錄

| 項目 | 狀態 |
|---|---|
| Setonix GPU 使用權限 | 使用者已確認 |
| PyTorch module 與容器入口 | 已載入，包裝命令來源已確認 |
| 個人路徑與配額 | 使用者已提供，見第 1 步結果 |
| 容器 Python 與追加依賴 | 待確認 |
| GPU 環境實测 | 尚未執行 |
| 模型檔案在 Setonix 的完整性 | 尚未確認 |
| GeoRA GPU 單步／GRPO | 尚未執行 |
