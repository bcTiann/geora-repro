# 連續 GRPO 與訓練恢復檢查

這一階段接在一次真實 GSM8K 更新之後。要驗證的是：**更新後的策略能否繼續採樣、計算損失、更新，以及存檔後能否接著做同一次訓練。** 尚未用任務分數判斷方法的效果。

## 1. 一次完整的訓練循環

```text
固定資料順序取 2 題
    ↓
目前策略對每題採樣 4 個回答
    ↓
最終數值比對 → reward → 每題的相對 advantage
    ↓
保存這批回答的 old logp；取得固定 reference 與可求導 current logp
    ↓
likelihood 檢查 → GRPO loss → backward → AdamW → scheduler
    ↓
清除梯度，保存完整訓練狀態；繼續下一批題目
```

一批回答只更新一次。下一批由**更新後的策略**重新採樣。reference 仍是固定的原始模型；old logp 每一批重新保存，兩者不要混淆。

每次都保留全部回答，包括整組 reward 相同的情況。此時 policy advantage 為零；KL 或 AdamW 的歷史動量仍可能影響更新。程式不要求每一步的 reward 上升，也不要求每一步梯度非零。

## 2. LoRA 與 GeoRA 共用什麼？

| 項目 | 共用設定 |
|---|---|
| 模型 | Qwen2.5-1.5B-Instruct，原始 pinned revision |
| 投影 | 28 blocks × Q/K/V/O/gate/up/down，196 個；排除 lm_head |
| 可訓練量 | r=16、alpha=32；392 個 A/B tensors，18,464,768 參數 |
| 資料 | 已固定的 smoke train 前 10 題，按順序使用 |
| 每步 | 2 題，每題 4 回答；最多 256 completion tokens |
| 採樣與 reward | seed=20261010、temperature=1、無 top-k/top-p 截斷；沿用嚴格 `#### <number>` 數值比對 |
| optimizer | AdamW，lr=1e-6；固定 learning rate；其餘見配置 |
| GRPO | group population std、clip=0.2、beta=0.001，一批一次更新 |
| 精度 | frozen weights BF16；A/B、GeoRA A0/B0、低秩分支 FP32；校正轉 BF16 後相加 |
| likelihood | 維持已驗證的固定寬度、無 KV cache 路徑，每步核對採樣與重算的 logp |

LoRA 用隨機 A、零 B，計算 $W_{pre}x+cB(Ax)$；GeoRA 從保存的未訓練因子開始，計算 $W_{pre}x+c[B(Ax)-B_0(A_0x)]$。LoRA 的 Gaussian A 標準差 0.02 是我們明示的設定，沒有作者程式可核對，不能稱為作者的精確 baseline 實作。

LoRA 初始化用獨立 CPU generator；兩種方法的 rollout 也各用獨立、同 seed 的 GPU generator。模型更新後分布不同，之後回答不同是正常的。

## 3. 為什麼單存 A/B 不夠？

例如第 2 步後存檔，接著要做第 3 步：

- **A/B** 決定目前策略。
- **AdamW 的一階、二階動量與 step counter** 決定下一次梯度如何轉成更新。
- **scheduler** 決定 learning rate 和已走的步數；這次用固定 learning rate，仍保存狀態。
- **隨機數狀態** 決定下一批採樣；包括專用 rollout generator、Torch CPU/GPU 與 Python RNG。
- **資料位置** 決定下一次取哪兩題。
- **來源記錄** 確認 base、初始化、資料和配置相同；來源不符時拒絕恢復。

每一步只在「這批已完成更新、已清除梯度、沒有進行中的 rollout」存檔。若以後重用一批 rollout 做多次更新，還需保存該批回答、old logp 和批內更新位置；目前不支援這種中途存檔。

## 4. 恢復怎麼驗證？

1. 完成第 2 步，保存 boundary checkpoint。
2. 正常採樣並更新第 3 步，保存答案、logp、A/B、optimizer 和 generator 的比較副本。
3. 恢復第 2 步 checkpoint，在相同 runtime 重做第 3 步。
4. 要求題目、token IDs、reward、advantage、logp、更新後 A/B、AdamW、scheduler、generator 狀態完全一致。
5. 用恢復後的路徑繼續第 4、5 步。

所以每種方法是 **5 個有效訓練步，實際跑 6 次 rollout/backward/optimizer step**：其中一次是重放驗證，其成本單獨記錄，不能算成第 6 個有效訓練步。

每一步另查 loss/梯度/參數有限、FP32 因子、mask 與 likelihood 一致；結尾查 frozen weights、GeoRA A0/B0 與 reference 完全不變。恢復時原位複製 A/B，保留 Parameter 物件，避免 optimizer 還指向舊物件。

## 5. 執行入口

| 檔案 | 在訓練流程中負責什麼 |
|---|---|
| `lora_layers.py`／`geora_layers.py` | 決定線性層的計算及可訓練 A/B；不是 GRPO 算法 |
| `scripts/gsm8k_data.py`／`scripts/grpo_math.py` | 讀取題目、解析答案，以及計算 advantage 和損失 |
| `scripts/grpo_rollout.py` | 讓模型採樣回答，並重算這些 token 的 log probabilities |
| `scripts/check_grpo_continuation.py` | 把採樣、打分、更新、存檔與驗證接成連續循環 |
| `scripts/grpo_training_state.py` | 保存／恢復 A/B、optimizer、scheduler、RNG 和資料位置 |
| `configs/grpo_continuation.json` | 固定這次五步檢查的所有選項 |
| `jobs/grpo_continuation.sbatch` | 向 Setonix 申請一個邏輯 GPU，啟動上述 Python 程式 |

本地 CPU 預檢：

```bash
uv run python scripts/check_lora_cpu.py --output-dir /private/tmp/geora-lora-cpu
uv run python scripts/check_grpo_resume_cpu.py --output-dir /private/tmp/geora-resume-cpu
```

Setonix 登入節點先準備臨時 checkpoint 副本。載入模組後，`pytorch-exec` 會在既有容器內使用我們已建立的環境。這些 CPU 命令不申請 GPU：

```bash
module load pytorch/2.7.1-rocm6.3.3
geora_python="$MYSOFTWARE/manual/software/geora-environments/py312-rocm633/bin/python"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export GEORA_VALIDATED_CHECKPOINT="$MYSOFTWARE/manual/cache/geora-continuation-$(date -u +%Y%m%dT%H%M%SZ)"
pytorch-exec "$geora_python" -u scripts/stage_checkpoint_copy.py \
  --output-dir "$GEORA_VALIDATED_CHECKPOINT"
pytorch-exec "$geora_python" -u scripts/check_staged_checkpoint.py \
  --checkpoint-dir "$GEORA_VALIDATED_CHECKPOINT"
mkdir -p "$MYSCRATCH/geora/runs/logs"
```

CPU 預檢及副本校驗通過後，從登入節點提交：

```bash
sbatch --account=pawsey0807-gpu \
  --output="$MYSCRATCH/geora/runs/logs/grpo-five-geora-%j.log" \
  jobs/grpo_continuation.sbatch geora \
  "$MYSCRATCH/geora/initializations/login-20261009T115357Z-2781659" \
  "$MYSCRATCH/geora/datasets/gsm8k" "$GEORA_VALIDATED_CHECKPOINT"
```

LoRA 使用同一個 job，只把方法參數 `geora` 改成 `lora`；本輪等 GeoRA 結束後才執行 LoRA。`GEORA_VALIDATED_CHECKPOINT` 是先經 CPU checksum 和載入值校驗的臨時副本路徑。一次分配一個邏輯 GPU，Slurm 硬上限 10 分鐘；程式的 540 秒預算在運算階段邊界檢查，不會立即中斷已開始的運算或存檔。每一步都保存 boundary，若失敗保留報告與最近 checkpoint。

`/bin/bash jobs/watch_gpu_check.sh <job-id> <log-path>` 可邊看邊印 log。Ctrl-C 只停止觀看；取消作業仍用 `scancel <job-id>`。

正式恢復入口是 `scripts/check_grpo_continuation.py --resume-dir <checkpoint-step-N>`，還需原始 model/data/init/output/method 參數。完整重新建立模型後，先驗證來源，再恢復訓練狀態。來源或 backend 不同會拒絕恢復。

## 6. 如何解讀結果？

報告 `continuation_checks.json` 的 `batches` 是 5 個有效步；`resume_replay` 是額外驗證。每個回答保留文字、tokens、數值解析與實際 training reward。`physical_*` 是這次 allocation 內實際做的工作。

平均 reward 同時受數學正確性、格式和長度上限影響。解析失敗、截斷、整組同分需分開看。5 步的 reward 差異不能判定 LoRA/GeoRA 優劣；這一階段的交付是連續流程、完整恢復和實測成本。

數值記錄仍採 GRPO 的 sampled-token k3 代理項。它不是每個位置完整 vocab 的精確 KL；固定舊回答更新後的診斷也不能當作更新後策略的精確 KL。

## 7. 本輪實際結果

LoRA／GeoRA 各完成 5 步真實 GSM8K GRPO，並通過第 2 步存檔後重做第 3 步的精確恢復（GeoRA 52/52，LoRA 52/52）。一個邏輯 GPU 依次執行，allocation 合計 914 秒。CPU LoRA 41/41、完整恢復 4/4，兩處 runtime 均通過；完整結果見 [EXPERIMENT_RECORD.md L11/S10](EXPERIMENT_RECORD.md) 和 [comparison.json](reports/grpo_continuation/comparison.json)。兩個原始 report、每步回答與 checkpoint 路徑均保留。

不用載入模型，也能重新核對兩份報告的來源、資料順序、mask、likelihood、恢復記錄和首批一致性：

```bash
uv run python scripts/check_grpo_continuation_reports.py \
  --geora reports/grpo_continuation/50622579.json \
  --lora reports/grpo_continuation/50622756.json \
  --config configs/grpo_continuation.json \
  --jobs reports/setonix_jobs.json \
  --output /private/tmp/geora-continuation-comparison.json
```

本輪 JSON 核對為 66/66；這與 GPU 程式各自的 52 項檢查分開。A/B、AdamW 等張量的精確一致由 GPU 執行器實際比較，JSON 記錄其結果；只讀報告不會重新計算那些張量。

## 8. 接續階段：KV cache 診斷（L12/S11）

2026-10-11 新增獨立的 cached rollout 候選，檢查生成時實際抽樣的 token logp，是否與本文件的固定寬度、無 cache 訓練 scorer 一致。**上述五步流程及 `grpo_rollout.py` 沒有修改。** 完整方法與門限見 [CACHED_ROLLOUT.md](CACHED_ROLLOUT.md)。

本機和 Setonix 登入節點的 tiny CPU 檢查均為 34/34 機械檢查、8/8 likelihood 門限通過；FP32 最大逐 token logp 差 `2.384185791015625e-7`，BF16 為 `0`。合成小模型有一次 AdamW 檢查，不是 GSM8K GRPO 訓練。

完整 MI250X BF16 job `50658753` 為 10/10 機械檢查通過，兩個 likelihood cases 均未通過（0/2）。初始化 GeoRA 與原模型的 cached tokens／logp 完全相同；但兩者各自與無 cache 訓練 scorer 比較時，在 103 個有效 token 上得到相同的差異指標：

| 測量 | 結果 |
|---|---|
| 最大逐 token logp 差 | `0.17979764938354492` |
| 平均絕對逐 token logp 差 | `0.011392114763262687` |
| scored／behavior 概率比範圍 | `[0.8870854910554354, 1.196975129959193]` |
| 最大絕對整段 logp 差 | `0.06114433162656496` |

這是短生成的數值診斷，沒有 reward 或 GPU optimizer update，也沒有比較任務分數。原模型亦有相同差異，因此目前不能把問題歸因於 GeoRA 初始化；具體的算子或運算順序原因仍未定位。

兩次作業均只申請一個邏輯 GPU、兩分鐘上限。job `50658705` 在 tokenizer API 處失敗，實際 allocation 34 秒；以 `tokenize=False` 取得文字再 `tokenizer.encode` 的修正經本機與登入節點 CPU 檢查後，job `50658753` 完成數值測量，Python 執行 9.741 秒、allocation 38 秒。本階段 allocation 合計 **72 秒**，另於上一階段的 914 秒記錄；失敗成本沒有省略。

**下一步保留已驗證的無 cache 訓練 sampler**，先以 CPU 診斷採樣／打分路徑差異，或驗證有算法依據的 importance correction。候選通過後，還要測完整模型已更新策略，以及 LoRA／GeoRA GRPO 和精確恢復。較長 pilot 前另需節省 scores 記錄的顯存；目前沒有足夠證據宣稱 KV cache 訓練已驗收或有多少加速。

## 9. 成對原始 logits 與 softmax 檢查（L13/S12）

2026-10-11 用一個凍結的原始 Qwen 模型，對同一批前文與已生成 token 比較兩條路徑的 logits：cached 生成時直接捕獲 `lm_head` 輸出，再用既有固定寬度、無 cache 路徑計算相同位置。沒有裝入 GeoRA、reward 或 value 模型，也沒有 optimizer update。方法與完整報告見 [CACHED_LOGITS.md](CACHED_LOGITS.md)。

本機與 Setonix 登入節點的 tiny CPU 檢查均為 10/10，涵蓋 FP32／BF16 兩個原模型 cases。完整 MI250X BF16 job `50659217` 的 6/6 檢查通過：捕獲 logits 經 GPU FP32 `log_softmax` 重算生成 logp 的誤差為 `0`；相同 logits 重複計算的誤差也為 `0`。兩條路徑在 103 個有效 token 上的原始 logits 最大差為 `0.5`。

原有約 19.7% 差異的位置是 `row=1`、`completion step=9`（均從 0 開始），token ID `1986`。診斷程式在 CPU 上用相同的 FP64 計算分析兩份已捕獲的 logits：

| 該位置的值 | cached 路徑 | 固定寬度、無 cache 路徑 |
|---|---|---|
| 選中 token 的原始 logit | `23.25` | `23.50` |
| 全 vocab 的 $\log(\sum_v e^{z_v})$ | `25.31924361552085` | `25.389446060081053` |

所以兩邊對這個 token 的 logp 差可分解為（第二項是 softmax 分母的對數差）：

$$
\Delta\log p
=(23.50-23.25)
 -(25.389446060081053-25.31924361552085)
\approx 0.17979755544.
$$

FP64 概率比為 `1.1969750175109515`，實際 GPU FP32 記錄為 `1.196975129959193`，均表示無 cache 重算概率相對 cached 概率高約 19.7%。**這個位置的主要差異已出現在 softmax 的輸入 logits 中；提高最後 softmax 的計算精度仍重現差異。** FP64 分析沒有恢復 BF16 forward 中已失去的精度。

兩份原始 logits 已下載到本機，SHA256 相同；獨立 NumPy FP64 softmax 得到該 token `This` 的概率 12.6281263%／15.1155517%，概率比 `1.1969750175108693`，與診斷報告相差約 8.22e-14。復查不使用 GPU。

另做八行 batch 的 prefix 寬度對照：59 個位置的無 cache 輸入，對 cached 與固定寬度 66 的原始 logits 最大差各為 `0.34375`。這支持繼續檢查計算路徑，但尚不能指定是哪個 decoder 算子、kernel 或 BF16 捨入造成差異，也沒有驗證出可用校正。

最差位置兩份 FP32 logits 保存為 `1,215,656` bytes 的 safetensors；原始檔保留於 scratch 與本機由 Git 忽略的 outputs，JSON 保存前文、token ID、值和 SHA256 `f865de782dd4079f02683335c0a6cb69067cab6daa03dd11d5488d2b917a6f1b`。job 只用一個邏輯 GPU、兩分鐘上限，實際 allocation **34 秒**；所有已記錄作業 allocation 累計由 1831 增至 **1865 秒**。

`grpo_rollout.py` 與已驗證的訓練 sampler 保持原樣。這次交付是概率差異的直接 logits 證據；概率對齊／校正、更新後策略、完整訓練與恢復仍待驗收，沒有任務分數或加速結論。

## 10. 同一 token 的逐層重放與 FP32 對照（L14/S13）

[逐層診斷](CACHED_LAYERS.md) 使用 S12 已保存的回答，重放同一原模型的 cached、prefix 無 cache、固定寬度無 cache 三條計算路徑，沒有重新採樣、GeoRA 或 optimizer update。實際有效 token IDs、position IDs 與目標 attention 可見 key 一致；BF16 重放精確恢復 S12 的兩份 logits snapshot。

固定寬度比較中，當前 token 最早在第 0 層 `q_proj` 出現微差，但 RoPE 後 Q 相同。歷史 V 有差，第 0 層 attention 概率相同，其加權 V 輸出仍不同。這定位了早期數值分岔，尚未證明單一 kernel 有 bug；BF16 autocast 路徑包含部分內部 FP32 計算，不能把所有運算概括為 BF16。

把同一組權重值轉為 FP32，關閉 BF16 autocast，該 token 的無 cache／cached 概率比由 `1.1969750175109515` 縮至 `1.0000327280290207`。它仍超過既有 `2e-5` 相對比例門限，且只測這個位置，不能當作全部 103 token、更新後策略或 GRPO 訓練驗收。**已驗證的訓練 sampler、配置與門限保持原樣。**

job `50660028` 的診斷檢查為 23/23，實際 allocation **36 秒**，一個邏輯 GPU、兩分鐘上限；17 個已記錄作業合計 **1901 秒（31 分 41 秒）**。這裡的「診斷通過」是證據和重放檢查通過，概率對齊／校正仍待完成。


## 11. 後續完成：原生 cache 五步與完整恢復（L16/S15）

本文件前述52/52屬於 `fixed_width` 路徑。新增可選 `native_cached_differentiable` backend：生成與可求導重算使用相同 native DynamicCache 順序，逐token取logp；backward只重算純attention，cache.update保持在checkpoint外。沒有detach歷史K/V，也沒有使用vLLM。原預設config／fixed路徑保留。

GeoRA `50660901`、LoRA `50661193` 均58/58：各五步真實GSM8K＋一次第3步恢復重放，所有有效tokens採樣/old和current/old概率精確一致；完整A/B、AdamW、scheduler、RNG及資料位置恢復通過。方法、精度、實際成本和cache重跑命令見 [CACHED_TRAINING.md](CACHED_TRAINING.md)，[原始報告及比較](reports/grpo_cached_continuation/comparison.json)。完整GPU梯度張量沒有保存供事後比較；runtime檢查梯度有限、更新後參數和optimizer精確恢復，CPU另驗證checkpoint梯度一致。

cache生成較快，但本次整段訓練耗時與顯存較高；實際token長度與兩次數值軌跡不同，不能當作配對速度測試。下一輪pilot預設共用fixed路徑；[pilot plan](configs/grpo_pilot_plan.json) 是規劃檔，通用訓練runner及validation評估仍待實作。
