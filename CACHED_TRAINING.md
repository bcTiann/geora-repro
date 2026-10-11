# 使用 KV cache 的 GRPO：模型、資料流與驗收

本階段驗證 **LoRA／GeoRA 各五步真實 GSM8K 訓練，以及存檔後下一步的精確恢復**。每批 2 題、每題 4 個回答，因此 batch=8；每個回答最多 256 個新 token。兩方法使用 Qwen2.5-1.5B-Instruct、rank=16、alpha=32；GeoRA 復用 rho=0.2 的初始化。模型 revision 見 [base_model.json](configs/base_model.json)，訓練配置為 [grpo_cached_continuation.json](configs/grpo_cached_continuation.json)。兩方法的 cache 五步與恢復均已通過；下一輪 pilot 預設沿用已驗收的無 cache 路徑，成本與選擇見第 7 節。

## 1. 到底有幾個模型？

一次作業有兩個模型實例：

| 實例 | 權重與用途 | 是否更新 |
|---|---|---|
| policy | 原始 Qwen2.5-1.5B-Instruct 加上 LoRA 或 GeoRA；生成回答，並計算這些回答各 token 的概率 | 只更新 A/B |
| reference | 同一個原始 checkpoint 的凍結副本；計算同一段前文下的參考概率，供 KL 代理項使用 | 否 |

「原模型」指加入適配器前的權重，也是 reference 的來源。沒有另一個專門計算概率的模型。`old_logp` 和 `current_logp` 都由 **同一個 policy** 計算：前者不記錄求導圖，後者記錄求導圖；在本批 optimizer 更新之前，它們的 A/B 相同。

GeoRA 沿用已驗收的計算順序：

$$
W_{\mathrm{eff}}=W_{\mathrm{pre}}+c(BA-B_0A_0),\qquad c=\alpha/r.
$$

初始化時 A=A0、B=B0，因此校正為零；後續 A/B 改變，A0/B0 保持不變。LoRA 則計算 $W_{\mathrm{pre}}x+cB(Ax)$，從隨機 A、零 B 開始。

## 2. 一批回答如何變成一次更新？

```text
取固定順序的 2 道 GSM8K 題目，每題複製 4 份 prompt
    ↓
policy：native generate → prefill → 逐 token decode，使用 DynamicCache
    ↓
保存實際抽到的 token IDs、每個 token 的 logp、首次 EOS 的位置
    ↓
解析 #### 後的數值，與資料中的最終答案比對 → reward → group advantage
    ↓
policy：從 prompt 重新 prefill／逐 token decode → old_logp（不求導）
reference：從相同 prompt 重新 prefill／逐 token decode → reference_logp（不求導）
policy：從相同 prompt 重新 prefill／逐 token decode → current_logp（可求導）
    ↓
檢查抽樣／重算概率一致 → GRPO loss → backward → AdamW → scheduler
    ↓
清除梯度，保存 A/B、optimizer、scheduler、RNG、下一批資料位置
```

**重新計算概率時不重新抽樣答案。** 例如保存的回答是 `[a,b,c]`，重算流程先用 prompt 的最後一個位置計算 a 的概率，再送入 a 計算 b 的概率，再送入 b 計算 c 的概率。每次先算 vocab logits，再用 FP32 `log_softmax` 取保存 token 的 logp。

每次重算都建立自己的新 cache。生成階段的 cache 不進入 backward；可求導重算中的歷史 K/V **不 detach**，因此後面 token 的 loss 仍能經由歷史狀態更新較早計算中的 A/B。首次 EOS 計入 loss，之後 padding 不計入；已結束的 row 保留在 batch 中，執行布局和 native generate 一致。

一批只更新一次。下一批由更新後的 policy 重新生成回答；reference 仍固定。整組 reward 相同的回答也保留，policy advantage 為零，不為了得到非零梯度而篩選資料。

## 3. 為什麼需要兩種省記憶體方法？

### 生成：只保存抽到的 token 的概率

若保存 256 步的全部 vocab scores，會保存形如 `[8,256,151936]` 的大量資料。`cached_training_rollout.py` 在每一步只暫存該步的 vocab logp；下一步拿到剛抽出的 token ID 後，取出它的 logp，再釋放前一步的 vocab 張量。最後保存的是 `[8,256]` 的 token logp 和 mask。生成仍有當步 logits 等臨時張量，但不累積整段的 vocab scores。

這個 recorder 是 native generate 的最後且唯一的 logits processor。它先施加配置中的 temperature，再記錄將送入 softmax／multinomial 的分布；生成的 RNG 用獨立 generator，存檔時也保存其狀態。

### 求梯度：backward 時重算純 attention 的中間量

可求導 cache 需要保留歷史梯度，但 eager attention 的中間量會隨序列增長。這裡使用 activation checkpointing：保留重算所需的輸入，把一部分 attention 中間量留到 backward 時重新計算，以運算換記憶體。[PyTorch 2.7 的 checkpoint 文件](https://docs.pytorch.org/docs/2.7/checkpoint.html) 說明這個機制，並建議明示 `use_reentrant=False`。

```text
投影／位置旋轉得到 Q、K、V
    ↓
cache.update：本次 K/V 加入歷史（在 checkpoint 外）
    ↓
checkpoint：純 attention(Q, 歷史 K, 歷史 V, mask)
    ↓
得到本次輸出，繼續算 current_logp 與 loss
    ↓
backward：必要時重算純 attention；不再次執行 cache.update
```

`cached_attention_checkpoint.py` 只包住原生 eager attention 的純計算，保持原函式、輸入、dtype 和 autocast。前向完成後即恢復全域函式；backward 使用每次前向綁定的原函式。設定為 `use_reentrant=False`、`preserve_rng_state=True`。不啟用 Hugging Face 的整層 gradient checkpointing，避免它改變 cache 行為。

這是 **native Transformers Qwen2 eager／DynamicCache** 路徑；本輪沒有 vLLM。

## 4. 精度與不能放寬的驗收條件

| 階段 | 實際選擇 |
|---|---|
| GeoRA 初始化來源 | 復用 mask 篩選後、以 CPU FP32 SVD 得到並保存的因子；本輪不重新 SVD |
| 凍結原始權重 | BF16 |
| A/B、GeoRA A0/B0、低秩校正分支 | FP32；校正轉 BF16 後與 base 輸出相加 |
| 生成及重新計算概率 | 同一原生 cache 執行順序，使用 BF16 autocast；保留原生算子自己的精度處理，例如正規化內部轉 FP32 |
| log_softmax、GRPO loss、A/B 梯度及 AdamW 動量 | FP32 |
| 訓練 checkpoint | A/B 與 optimizer 張量保持其 FP32 精度；另保存 RNG、scheduler、資料位置及來源 |

每批的全部有效 token 均須滿足：最大 logp 差不超過 `2e-5`，概率比偏離 1 不超過 `2e-5`，每個回答的總 logp 差不超過 `1e-3`。抽樣概率對 `old_logp`、`current_logp` 對 `old_logp` 都檢查。這些門檻沿用既有驗收，沒有因換 cache 而放寬。

同時檢查 196 個目標層只有 392 個 A/B 張量可訓練；loss、梯度及更新後參數有限；backward 確實重算純 attention，autocast 一致；凍結權重、A0/B0 和 reference 不變。第 2 步存檔後重放第 3 步，要求回答 token、reward、advantage、logp、更新後 A/B、AdamW、scheduler 和 RNG 完全一致。五個有效步包含一次額外重放，因此實際有六次 rollout／backward／optimizer 呼叫。

## 5. 已驗證到哪裡？

| 證據 | 已完成 | 尚未涵蓋 |
|---|---|---|
| S14，GPU job 50660628 | 26/26；完整 GeoRA 初始化 16 token、合成更新後 64 token；共 292 個有效 token 的抽樣／重算 logp 精確一致，完整 A/B backward 有限 | 真實 GSM8K 五步、完整 LoRA、256 token 的 GPU 容量與恢復 |
| 新 streaming sampler，Mac／Setonix CPU | 各76/76；包括 256 token、temperature、EOS／padding、native scores 與 RNG 對照、可求導重算 | 完整模型 GPU |
| 新 attention checkpoint，Mac／Setonix CPU | 各45/45；logp、梯度、歷史梯度、autocast／RNG、異常恢復、保存的張量儲存量 | 完整模型 GPU |

| 完整 GPU 驗收 | GeoRA `50660901` | LoRA `50661193` |
|---|---|---|
| 檢查 | 58/58 | 58/58 |
| 有效訓練步／實際執行更新（含重放） | 5／6 | 5／6 |
| 有效 completion tokens：不含／含重放 | 8003／9834 | 8129／10157 |
| 全部採樣／old、current／old 的最大 logp 差 | 0 | 0 |
| 初始 policy／reference、最終更新策略的 logp 差 | 0 | 0 |
| Python 本體／Slurm allocation | 695.426／715 秒 | 574.733／593 秒 |
| GPU peak allocated memory | 42.466 GiB | 42.340 GiB |
| 第 2 步後存檔、第 3 步重放 | A/B、AdamW、scheduler、RNG、回答及概率精確一致 | 同左 |

兩方法初始 batch 的八份回答完全一致。後續更新後軌跡不同。每步仍保留同分組；GeoRA 10 組中 8 組同分、26/40 解析失敗、19/40 截斷；LoRA 分別為 6/10、27/40、17/40。這些是流程診斷，不是兩方法的效果排序。當前 256-token 上限影響 reward，擴長前須另測容量與概率一致性。

原始報告：[GeoRA](reports/grpo_cached_continuation/50660901.json)、[LoRA](reports/grpo_cached_continuation/50661193.json)；統一比較：[comparison.json](reports/grpo_cached_continuation/comparison.json)。GPU 中沒有保存完整梯度張量供事後逐元素比較；原始執行檢查了梯度有限及更新後 A/B／optimizer 的精確恢復。獨立 JSON 復查還核對了 reference logp、loss 與 gradient norm；CPU checkpoint 對照另驗證梯度逐元素相同。

S14 的人工 reward 更新不算 GSM8K 訓練步。詳細歷史證據見 [EXPERIMENT_RECORD.md 的 L15/S14](EXPERIMENT_RECORD.md)；兩方法曾完成的五步無 cache 訓練是另一條已驗收路徑，不能替代本輪 cache 五步。

## 6. Setonix：從 CPU 預檢到單 GPU 作業

以下在登入節點執行。已有 ROCm 容器與 Python 環境；CPU 預檢使用 2 threads，不另申請 Slurm CPU allocation。

```bash
cd "$MYSOFTWARE/geora/code"
git pull --ff-only
module load pytorch/2.7.1-rocm6.3.3
geora_python="$MYSOFTWARE/manual/software/geora-environments/py312-rocm633/bin/python"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
geora_preflight="$MYSCRATCH/geora/runs/cache-preflight-$(date -u +%Y%m%dT%H%M%SZ)"
pytorch-exec "$geora_python" -u scripts/check_cached_training_rollout_cpu.py \
  --output-dir "$geora_preflight/streaming"
pytorch-exec "$geora_python" -u scripts/check_cached_attention_checkpoint_cpu.py \
  --output-dir "$geora_preflight/attention-checkpoint"
```

本次臨時模型副本 `geora-cache-goal-20261011` 已於所有作業結束後清理。重跑時先準備一個新的 CPU 校驗副本；以下只複製已下載的 pinned 模型，不重新下載。模型與初始化的原始檔仍在 scratch。

```bash
geora_initialization="$MYSCRATCH/geora/initializations/login-20261009T115357Z-2781659"
geora_data="$MYSCRATCH/geora/datasets/gsm8k"
geora_checkpoint="$MYSOFTWARE/manual/cache/geora-cache-five-$(date -u +%Y%m%dT%H%M%SZ)"
pytorch-exec "$geora_python" -u scripts/stage_checkpoint_copy.py \
  --output-dir "$geora_checkpoint"
pytorch-exec "$geora_python" -u scripts/check_staged_checkpoint.py \
  --checkpoint-dir "$geora_checkpoint"
```

CPU 預檢及模型副本校驗通過後，先提交 GeoRA；它結束並檢查結果後，再將方法改為 `lora` 提交同一個 job。模型、初始化和資料路徑用位置參數傳入，不依賴上一個互動 shell 的 `GEORA_INIT_DIR`。

```bash
mkdir -p "$MYSCRATCH/geora/runs/logs"
geora_method=geora
geora_job_id=$(sbatch --parsable --export=NONE --account=pawsey0807-gpu \
  --output="$MYSCRATCH/geora/runs/logs/grpo-cache-five-${geora_method}-%j.log" \
  jobs/grpo_cached_continuation.sbatch "$geora_method" \
  "$geora_initialization" "$geora_data" "$geora_checkpoint")
geora_log="$MYSCRATCH/geora/runs/logs/grpo-cache-five-${geora_method}-${geora_job_id}.log"
/bin/bash jobs/watch_gpu_check.sh "$geora_job_id" "$geora_log"
```

作業使用 `gpu-dev` 的 **一個邏輯 GPU**，Slurm 上限 20 分鐘；程式預算 900 秒，在運算階段邊界檢查 deadline。每個可求導 cache 步驟開始前檢查 allocated memory，超過 55 GiB 即中止並保存錯誤；這不是保證不會出現 OOM 的 GPU 峰值預測。作業日誌即時印出生成進度、概率重算階段與記憶體。

`policy_loss=0` 不能推出 reward 全相同或梯度為零：更新前 ratio=1，分組正負 advantage 可在損失值中抵消；對每個回答概率求導後，更新訊號仍可能非零。實際更新以 reward、gradient norm 和參數變化一起判斷。

報告位置為 `$MYSCRATCH/geora/runs/grpo-cached-five-<method>-<job-id>/continuation_checks.json`，每步另有 `checkpoint-step-N`。作業取消用 `scancel <job-id>`；觀看 log 時按 Ctrl-C 只停止觀看。

同一配置恢復訓練可加 `--resume-dir <checkpoint-step-N>`，保留原 `--config`、`--method`、模型、初始化及資料參數。配置來源包含 backend；不同路徑的 checkpoint 不混用。完整恢復入口與狀態格式見 [GRPO_CONTINUATION.md](GRPO_CONTINUATION.md)。

## 7. 這輪結果能支持什麼？

本輪結果支持「在指定 runtime、batch=8、最長 256 個新 token 下，cache 生成與梯度所用概率對齊，LoRA／GeoRA 能連續更新並精確恢復」。更長回答、更大 batch、多 GPU、其他 attention 後端尚未驗收。五步 reward 不用來判斷兩方法的任務效果；也不宣稱已完整復現論文或取得 vLLM 加速。


### 下一輪使用哪條路徑？

**預設 `fixed_width`：生成和概率重算皆使用相同的無 cache 路徑。** 新 `native_cached_differentiable` 保留為已驗收的可選 backend，不與 fixed scorer 混用。

| 同樣五個有效步＋一次恢復重放 | 舊 fixed GeoRA | 新 cache GeoRA | 舊 fixed LoRA | 新 cache LoRA |
|---|---:|---:|---:|---:|
| Slurm allocation（秒） | 480 | 715 | 434 | 593 |
| Python（秒） | 461.654 | 695.426 | 412.549 | 574.733 |
| peak allocated memory（GiB） | 29.008 | 42.466 | 28.874 | 42.340 |
| 實際有效 completion tokens（含重放） | 8655 | 9834 | 9377 | 10157 |

cache 生成省去歷史的重複計算，但梯度路徑需要逐 token 建立求導圖，並在 backward 重算 attention。GeoRA 新流程每步 rollout 約 20 秒，三路概率重算與 loss 約 53 秒，backward＋更新約 37 秒；舊流程概率重算和更新只需數秒。這說明生成較快並不能單獨證明整段訓練更快。

表中是兩次實際運行的成本：答案長度、decode 步數、數值軌跡、額外檢查及載入時間不同，沒有配對速度基準或硬體普遍結論。根據目前成本，先讓兩方法共用已通過的 fixed 路徑進行 pilot；更高效的 attention／訓練框架需另行驗收。

[grpo_pilot_plan.json](configs/grpo_pilot_plan.json) 規劃 100 步、1024 題 train／128 題 validation，維持已驗收的每批 2 題×4 回答、最多256 tokens。它是計畫檔；`check_grpo_continuation.py` 仍為五步驗收器。下一階段先實作通用 pilot runner 及固定 validation 評估、核對現有五步行為與恢復，再按實測成本分段提交。這輪未啟動 pilot 或論文分數對照。

本 goal 三個單邏輯 GPU 作業的 allocation 共 1370 秒（22 分 50 秒）。所有作業終態與副本清理見 [remote_final.json](reports/cache_intervention/remote_final.json)，含歷史失敗的全部成本見 [setonix_jobs.json](reports/setonix_jobs.json)。
