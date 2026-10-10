# 第一次真實 GSM8K GRPO 更新

這一步要把已驗證的 GeoRA 模型接到真實題目、模型自行生成的回答、答案檢查器和 GRPO loss，做 **一次** A/B 更新。它是訓練流程的驗收，尚不是論文表格的效果復現。

教學 notebook：[從題目到一次更新](notebooks/gsm8k_grpo_tutorial.ipynb)。資料與評分詳見 [GSM8K_DATA.md](GSM8K_DATA.md)。

## 心智模型：輸入、計算、輸出

```text
原始 Qwen checkpoint + 已保存但尚未訓練的 A0/B0
                          ↓
差值計算：Wpre x + (alpha/r) × [B(Ax) − B0(A0x)]
                          ↓
固定 GSM8K train 題目 → 每題自行抽樣 4 份回答
                          ↓
檢查最終數字 → 每份回答 reward 0 或 1
                          ↓
同題 4 個 reward → 減組平均、除組標準差 → advantage
                          ↓
核對抽樣時概率與重新打分概率一致
                          ↓
GRPO loss → backward → AdamW 更新 A/B 一次
                          ↓
檢查凍結部分未變、更新有效，保存回答、報告和新 adapter
```

`Wpre` 是原始預訓練模型的凍結權重；`A,B` 是可訓練參數；`A0,B0` 是保存的固定初值。開始時 `A=A0,B=B0`，差值為零，所以模型仍然是原模型。reference 是另一份原模型，始終凍結。

本步讀取既有初始化，不重做 mask 或 SVD；GPU 上沒有 SVD。不使用先前 CE 檢查更新過的 adapter。

## 哪個檔案負責哪一步

| 檔案 | 職責 |
| --- | --- |
| `configs/grpo_smoke.json` | 固定本次題數、抽樣設定、learning rate、clip、KL 和概率一致性門檻 |
| `scripts/gsm8k_data.py` | 固定資料來源、train/validation/test 分割、prompt 和數字獎勵 |
| `scripts/grpo_rollout.py` | 生成回答、記錄抽樣概率、以一致路徑重新打分 |
| `scripts/grpo_math.py` | advantage、token mask、概率比、裁剪 loss、sampled-token KL 項 |
| `scripts/check_grpo_math.py` | 人工可計算的 loss 與梯度 CPU 檢查 |
| `scripts/check_grpo_tiny.py` | 隨機 tiny Qwen 的 A/B 反向傳播 CPU 檢查，不衡量解題能力 |
| `scripts/check_grpo_rollout_cpu.py` | tiny Qwen 的抽樣／打分一致性、padding、EOS、更新後生成檢查 |
| `scripts/check_grpo_training.py` | 串起真實 1.5B 模型與上述各步，只做一次 GRPO 更新 |
| `jobs/grpo_smoke.sbatch` | 申請一個 logical GPU，執行完整檢查 |

## 為什麼這次有專門的抽樣／打分路徑

前面的測試發現：即使原模型參數沒變，BF16 的 cache 生成與一次完整回答打分，也可能因矩陣形狀、計算順序不同得到不同概率。兩次重新打分得到 `current/old=1`，不能單獨证明它就是實際抽樣時的概率。

這次為驗收選定一條固定路徑：

1. decoder 每次都讀相同的 `[8 份回答, 固定總長度]`；未生成的位置填 padding 並遮罩，不使用 KV cache。
2. 每一步只對當前預測位置的 `[8,1,hidden_size]` 呼叫 `lm_head`，得到整個 vocab 的 logits。
3. 回答生成完後，重新打分沿用相同寬度 decoder 與逐位置 `lm_head` 形狀。
4. policy 處於 `train()`，dropout 為 0；溫度 1，不使用 top-k/top-p 篩選或其它概率修改器。

未來位置受 causal mask 遮罩，因此填好完整回答後不應改變先前位置的數學結果；固定算子形狀也減少浮點路径變化。**仍然要在 MI250X 上實測通過 gate 才允許 backward。**

目前固定門檻：所有有效 token 的最大 `|Δlogp|` ≤ `2e-5`、最大 `|exp(Δlogp)−1|` ≤ `2e-5`，每份回答的 `|ΣΔlogp|` ≤ `0.001`。也核對帶梯度的 current 打分、old 打分和初始 reference。這是小規模正確性路徑；之後更快的 cache 或外部 rollout engine 需要另做驗證。

## 精度：哪些數字在哪裡

| 內容 | 精度與用途 |
| --- | --- |
| 原始模型 checkpoint | 磁碟上的 BF16 原權重；來源與 revision 固定 |
| 既有 GeoRA 初始化 | CPU FP32 mask/SVD 建構，A0/B0 及初始化 state 以 FP32 保存 |
| GPU 凍結原模型、reference | BF16 權重和原生 BF16 計算路徑 |
| GPU A/B 和固定 A0/B0 | FP32 儲存；只有 A/B 可以更新 |
| 兩個低秩支路及它們的相減 | FP32，關閉這部分 autocast；差值再轉回 base output dtype 相加 |
| 抽樣 softmax、log-softmax、概率比與 loss | FP32 |
| A/B 梯度與 AdamW moments | FP32 |
| 更新後保存的 adapter | FP32 current A/B 和固定 A0/B0；另存 manifest、optimizer state |

為載入驗證，CPU 可以把原始 BF16 數值讀成 FP32；這不會恢復原始 BF16 保存之前已丟失的精度。報告中的 FP64 更新範數僅用於數值診斷，不是 FP64 訓練。

此差值運算順序是我們驗證後採用的實作，不能聲稱作者使用了完全相同的浮點順序。

## 先在本地做 CPU 檢查

在倉庫目錄執行：

```bash
uv run python scripts/gsm8k_data.py --data-dir datasets/gsm8k
uv run python tests/check_gsm8k_data.py --data-dir datasets/gsm8k --report reports/grpo/gsm8k_local_cpu.json
uv run python scripts/check_grpo_math.py --output-dir reports/grpo/local_preflight
uv run python scripts/check_grpo_tiny.py --output-dir reports/grpo/local_preflight
uv run python scripts/check_grpo_rollout_cpu.py --output-dir reports/grpo/local_preflight
```

前兩個檢查資料與 reward；後三個只用小矩陣或 tiny Qwen，沒有載入 1.5B 模型、沒有使用 GPU。

## Setonix：login 先驗證，再申請 GPU

在 login node，不需要 CPU Slurm allocation。使用既有 ROCm container 和它的環境：

```bash
cd "$MYSOFTWARE/geora/code"
module load pytorch/2.7.1-rocm6.3.3
geora_python="$MYSOFTWARE/manual/software/geora-environments/py312-rocm633/bin/python"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
geora_preflight="$MYSCRATCH/geora/runs/grpo-preflight"
pytorch-exec "$geora_python" tests/check_gsm8k_data.py --data-dir "$MYSCRATCH/geora/datasets/gsm8k" --report "$geora_preflight/gsm8k_cpu.json"
pytorch-exec "$geora_python" scripts/check_grpo_math.py --output-dir "$geora_preflight"
pytorch-exec "$geora_python" scripts/check_grpo_tiny.py --output-dir "$geora_preflight"
pytorch-exec "$geora_python" scripts/check_grpo_rollout_cpu.py --output-dir "$geora_preflight"
```

資料目錄可由本地傳過來；如果已有原始三檔，可用 `gsm8k_data.py --offline` 驗證和分割。容器環境與本地 uv 環境各自獨立。

GPU job 要求先在 login node 建立並驗證一份臨時 checkpoint copy，避免共享 scratch 讀取變慢消耗 GPU 時間。選一個新的空目錄，例如：

```bash
geora_copy="$MYSOFTWARE/manual/cache/geora-grpo-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
export HF_DEACTIVATE_ASYNC_LOAD=1
pytorch-exec "$geora_python" scripts/stage_checkpoint_copy.py --output-dir "$geora_copy"
pytorch-exec "$geora_python" scripts/check_staged_checkpoint.py --checkpoint-dir "$geora_copy"
```

copy 只複製原始 checkpoint，不改動 scratch 原檔。CPU 驗證其檔案雜湊及全部 338 個原權重；`staging_record.json` 必須是 `validated`，才能提交 GPU job。用完後只清理這次的臨時 copy。

CPU 檢查全部通過後，在 host login shell 提交，三個參數依次是 **未訓練初始化、資料、已驗證 copy**：

```bash
mkdir -p "$MYSCRATCH/geora/runs/logs"
geora_job=$(sbatch --parsable --account=pawsey0807-gpu \
  --output="$MYSCRATCH/geora/runs/logs/grpo-smoke-%j.log" \
  jobs/grpo_smoke.sbatch \
  "$MYSCRATCH/geora/initializations/login-20261009T115357Z-2781659" \
  "$MYSCRATCH/geora/datasets/gsm8k" \
  "$geora_copy")
/bin/bash jobs/watch_gpu_check.sh "$geora_job" "$MYSCRATCH/geora/runs/logs/grpo-smoke-$geora_job.log"
```

job 申請 `nodes=1, gres=gpu:1`，是一個 logical GPU，不是八個；Slurm 最多 10 分鐘，程式完成一次更新就退出，提前結束即提前釋放。程式另有有限候選池與運行時間檢查。實際時間與 GPU 用量以本次 Slurm 和 JSON 報告為準。

查看隊列：`squeue -u "$USER"`。提前取消：`scancel JOBID`。觀看 log 時按 Ctrl+C 只停止觀看，不取消 job。

## 報告怎麼讀

本次輸出在 `$MYSCRATCH/geora/runs/grpo-smoke-JOBID/`：

- `grpo_checks.json`：每題、每份回答、最終答案解析、原始 reward、截斷情況、實際訓練 reward、advantage、有效 token、behavior/old/reference/update 後 logp、各項 gate、時間與峰值顯存。
- `trained_adapter/adapter.safetensors`：更新一次後的 FP32 A/B 及固定 A0/B0。
- `trained_adapter/manifest.json`：difference 路徑、來源和本次配置。
- `trained_adapter/optimizer.pt`：這次 AdamW state；不代表完整 RNG／資料位置的訓練恢復已驗證。

需要至少一題同時有 reward 1 和 0，才能驗證正負 advantage 都產生真實信號。只在這次首次 smoke，允許依固定 16 題順序有限搜尋這種組；未來 pilot 或正式 LoRA/GeoRA 比較不能篩掉同分組，更不能各方法各自挑答錯題。

回答未到 EOS 就被長度截斷時，本次訓練 reward 固定為 0。解析 reward 與 training reward 都保存在報告，避免把截斷與答錯混在一起猜。

報告中的 `kl` 是在這批已抽樣回答上計算、直接求導的 sampled-token k3 surrogate。更新前 policy=reference 時它為零；更新後的值是 **同一批舊回答上的 KL surrogate 診斷**，不是重新從 current policy 抽樣得到的完整 current-policy KL，也不是整個 vocab 精確 KL。

此步通過表示：真實資料、獎勵、概率、梯度和保存串起來了。它不证明 GeoRA 比 LoRA 更好、不給論文表格分數；接著才安排少量連續更新與公平的 LoRA/GeoRA 對照。

## 本次已執行結果（2026-10-10）


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

