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

本地 CPU 預檢：

```bash
uv run python scripts/check_lora_cpu.py --output-dir /private/tmp/geora-lora-cpu
uv run python scripts/check_grpo_resume_cpu.py --output-dir /private/tmp/geora-resume-cpu
```

Setonix 在 CPU 預檢和 checkpoint 副本校驗通過後，從登入節點提交：

```bash
sbatch --account=pawsey0807-gpu \
  --output="$MYSCRATCH/geora/runs/logs/grpo-five-geora-%j.log" \
  jobs/grpo_continuation.sbatch geora \
  "$MYSCRATCH/geora/initializations/login-20261009T115357Z-2781659" \
  "$MYSCRATCH/geora/datasets/gsm8k" "$GEORA_VALIDATED_CHECKPOINT"
```

LoRA 使用同一個 job，只把方法參數 `geora` 改成 `lora`。`GEORA_VALIDATED_CHECKPOINT` 是先經 CPU checksum 和載入值校驗的臨時副本路徑。一次分配一個邏輯 GPU，10 分鐘上限，Python 540 秒上限；每一步都保存 boundary，若失敗保留報告與最近 checkpoint。

`/bin/bash jobs/watch_gpu_check.sh <job-id>` 可邊看邊印 log。Ctrl-C 只停止觀看；取消作業仍用 `scancel <job-id>`。

正式恢復入口是 `scripts/check_grpo_continuation.py --resume-dir <checkpoint-step-N>`，還需原始 model/data/init/output/method 參數。完整重新建立模型後，先驗證來源，再恢復訓練狀態。來源或 backend 不同會拒絕恢復。

## 6. 如何解讀結果？

報告 `continuation_checks.json` 的 `batches` 是 5 個有效步；`resume_replay` 是額外驗證。每個回答保留文字、tokens、數值解析與實際 training reward。`physical_*` 是這次 allocation 內實際做的工作。

平均 reward 同時受數學正確性、格式和長度上限影響。解析失敗、截斷、整組同分需分開看。5 步的 reward 差異不能判定 LoRA/GeoRA 優劣；這一階段的交付是連續流程、完整恢復和實測成本。

數值記錄仍採 GRPO 的 sampled-token k3 代理項。它不是每個位置完整 vocab 的精確 KL；固定舊回答更新後的診斷也不能當作更新後策略的精確 KL。

## 7. 本輪實際結果

待執行。完成後會填入 CPU/GPU 報告、Slurm allocation、精度、恢復比較和限制。
