# GeoRA 獨立復現

這個倉庫分階段實作與驗證 [GeoRA: Geometry-Aware Low-Rank Adaptation for RLVR](https://arxiv.org/abs/2601.09361)。先讀 [復現計畫與目前進度](REPRODUCTION_PLAN.md)，再查 [本機與 Setonix 實驗記錄](EXPERIMENT_RECORD.md)：前者安排後續工作，後者集中說明已執行的流程、精度與結果。

LoRA／GeoRA 各完成 5 步真實 GSM8K GRPO，並通過第 2 步存檔後重做第 3 步的精確恢復（GeoRA 52/52，LoRA 52/52）。凍結權重與 reference 不變，兩種方法第一批採樣的八份回答完全相同。任務分數和正式 benchmark 待做。原 residual BF16 失敗紀錄保留。

## 從哪裡開始

| 檔案 | 用途 |
|---|---|
| [notebooks/gsm8k_grpo_tutorial.ipynb](notebooks/gsm8k_grpo_tutorial.ipynb) | 從 GSM8K、reward 到 GRPO 更新的 CPU 教程，附真實回答 |
| [GRPO_SMOKE.md](GRPO_SMOKE.md) | 資料／GRPO 檔案分工、精度、命令和結果 |
| [GRPO_CONTINUATION.md](GRPO_CONTINUATION.md) | 共用五步訓練、完整 boundary checkpoint、恢復與資源記錄 |
| [REPRODUCTION_PLAN.md](REPRODUCTION_PLAN.md) | 全局階段、目前位置、高吞吐 rollout 驗收與正式實驗安排 |
| [EXPERIMENT_RECORD.md](EXPERIMENT_RECORD.md) | 統一的實驗目的、流程、精度、結果、作業用量與目前進度 |
| [geora_layers.py](geora_layers.py) | GeoRA 線性層、mask 與 SVD 初始化、全層替換、adapter 儲存／載入 |
| [geora_initialization.ipynb](geora_initialization.ipynb) | 第 0 層 Q 矩陣的 FP64 參照初始化與單步更新 |
| [geora_model_check.ipynb](geora_model_check.ipynb) | 只替換一個 Q 層，檢查完整模型 forward 與儲存／載入 |
| [geora_full_model_check.ipynb](geora_full_model_check.ipynb) | 全部 196 個目標層的 CPU FP32 初始化與載入檢查 |
| [PRECISION.md](PRECISION.md) | 精度、目標層、殘差重建與 reference 策略約定 |
| [jobs/submit_gpu_check.sh](jobs/submit_gpu_check.sh) | 復用已保存的初始化，提交 GPU 檢查並即時顯示日誌；支援精度診斷 |
| [CHECKS.md](CHECKS.md) | 登入節點初始化後自動提交 GPU 完整檢查；包含結果解讀及失敗門檻 |
| [SETONIX_GUIDE.md](SETONIX_GUIDE.md) | Setonix 容器、儲存路徑與下一階段操作記錄 |
| [configs/base_model.json](configs/base_model.json) | 固定模型版本、rank、alpha、rho 與目標模組設定 |
| [reports/cpu_initialization/](reports/cpu_initialization/) | 原有 CPU 初始化的測量與 manifest，隨程式碼保存 |

建議先讀 `REPRODUCTION_PLAN.md` 查看路線與進度，再按需讀 `EXPERIMENT_RECORD.md`、`PRECISION.md` 和 notebook。Notebook 從倉庫根目錄開始，各自從上到下執行。

## 本機環境與模型

在本機先安裝 `uv`，再於倉庫根目錄建立依賴環境：

```bash
cd /Users/tianbaochen/geora-repro
uv sync --locked
uv run python -c 'import sys, torch, transformers; print(sys.executable); print(torch.__version__); print(transformers.__version__)'
```

在 VS Code 等 notebook 編輯器選擇本倉庫 `.venv/bin/python` 作為 kernel。`notebooks` 預設依賴組提供 `ipykernel`；一般 Python 命令使用 `uv run python`。首次同步可能需要下載依賴；環境由本倉庫的配置建立。

快速核對環境、模型與保存的初始化：

```bash
uv run python scripts/check_local_setup.py
```

基底模型固定為 `Qwen/Qwen2.5-1.5B-Instruct`，revision：

```text
989aa7980e4cf806f80c7fef2b1adb7bc71aa306
```

本機 `checkpoints/geora_base` 連到原有 `/Users/tianbaochen/RLVR/checkpoints/geora_base`，避免重複存放權重。在新的電腦下載同一 revision：

```bash
uv run python scripts/download_base_model.py
```

下載腳本將權重與 tokenizer 放到 `checkpoints/geora_base`。Notebook 只讀本機模型檔案；全模型 notebook 也核對下載時記錄的 revision。

原有全模型初始化產物已複製到 `outputs/geora_full_model_check/`：`adapter.safetensors`、`manifest.json`、`checks.json`。其中 adapter 保存初始 A0/B0 與當前 A/B；載入時用固定版本的原始模型重建殘差 F。

`checkpoints/`、`outputs/`、資料集、cache 與 Python 環境由 Git 忽略。複製或 clone 程式碼後，需要另外準備模型與執行產物。重新執行全模型 notebook 會重新做 SVD 並寫入該輸出目錄；重要結果需另存。

## 實驗結果

全部已執行結果集中在 [EXPERIMENT_RECORD.md](EXPERIMENT_RECORD.md)，包括早期 FP64 單 Q、全模型 FP32、Setonix BF16 失敗、FP32 對照、attention 五種精度路徑，以及各自的保存格式。`reports/` 保留小型原始報告；早期 notebook 的輸出已清空；新的 GSM8K/GRPO 教程保存本機 CPU 輸出並讀取 Setonix 真實報告。

## 本機倉庫建立檢查

2026-10-08：獨立 `.venv` 已建立，PyTorch／Transformers／ipykernel 匯入及模型／adapter 的檔案路徑與 header 檢查通過。原有程式未修改，複製的 adapter SHA256 與來源相同。完整記錄見 [bootstrap_setup.json](reports/bootstrap_setup.json)。這次沒有重跑完整模型 forward 或訓練。

本機使用 Git 管理程式版本。`bootstrap_setup.json` 保留倉庫建立時的狀態；後續提交與推送狀態以 Git 歷史為準。

## 在 Setonix 取得程式

本機的流程是修改程式、`git commit`、`git push`；Setonix 首次用 `git clone`，之後用 `git pull --ff-only` 取得已推送的版本。

首次取得程式的目錄為 `$MYSOFTWARE/geora/code`，具體命令見 [SETONIX_GUIDE.md 第 3 步](SETONIX_GUIDE.md)。Private 倉庫需先在 Setonix 配置 GitHub 認證。

Git 只傳程式、配置、教程與小型報告。模型權重、初始化 checkpoint 和 Python 環境另外準備；本機指向 `~/RLVR` 的模型連結也不會隨 clone 傳過去。

## Setonix 下一步

Setonix 使用 Pawsey 的 `pytorch/2.7.1-rocm6.3.3` 容器入口，另建容器內的 `venv --system-site-packages`，沿用容器已提供的 ROCm PyTorch，再補裝需要的套件。本機 `uv sync` 的 PyTorch 依賴配置不適用於這個容器環境。實際 Python、追加依賴與 GPU 可見性依 [SETONIX_GUIDE.md](SETONIX_GUIDE.md) 逐步確認。

本輪 [GOAL.md](GOAL.md) 已完成：`difference` 等價重排通過完整模型的初始化、一次 A/B 更新與模型／optimizer 保存重載。完整結果、數值邊界和 GPU 用量見 [EXPERIMENT_RECORD.md 的 L8/S7](EXPERIMENT_RECORD.md)。原有 residual 保留作對照；小 CE 單步不是 GRPO 或任務分數復現。

整體計畫見 [REPRODUCTION_PLAN.md](REPRODUCTION_PLAN.md)。目前階段 1（來源與初始化）、2（數值與機械更新）完成；階段 3（資料/答案解析、GRPO 小例子與一次真實獎勵更新）亦完成；階段 4 的共用五步／完整恢復已完成；接著驗證較高吞吐的 rollout，再做 pilot、正式評估與幾何分析。

## 重跑已完成的 difference 檢查

本機小模型：

```bash
uv run python tests/check_difference_forward.py
```

Setonix 使用已有初始化，不重新 SVD：

```bash
cd "$MYSOFTWARE/geora/code"
git pull --ff-only
mkdir -p "$MYSCRATCH/geora/runs/logs"
sbatch --export=ALL --account="${PAWSEY_PROJECT}-gpu" \
  --output="$MYSCRATCH/geora/runs/logs/difference-check-%j.log" \
  jobs/difference_training_check.sbatch \
  "$MYSCRATCH/geora/initializations/login-20261009T115357Z-2781659"
```

一個邏輯 GPU，三分鐘上限；log 和 JSON 保存到 scratch。此腳本先檢查三個初始化輸入，再做一次短 CE 更新／重載。`jobs/submit_checks.sh` 仍是原 residual 初始化與 GPU 檢查入口，供建立因子及舊路徑對照。操作與日誌查看見 [CHECKS.md](CHECKS.md)。
