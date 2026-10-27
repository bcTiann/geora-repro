# 有無 KV cache：逐層定位數值差異

更新：2026-10-11，L14/S13。接續 [原始 logits 檢查](CACHED_LOGITS.md)。

本文件記錄L14/S13；後續已完成受控干預和cache訓練驗收，見最後一節。

**已定位到第 0 層的投影與 attention 計算開始出現路徑差異。相同權重數值改用 FP32 forward，原最差 token 的概率相對差由 19.6975% 降到 0.0032728%。這是定位結果，尚未通過完整 cached 訓練的概率一致性驗收。**

## 1. 比較的是哪個模型、哪段文字？

只有一個凍結的原始 `Qwen/Qwen2.5-1.5B-Instruct`，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。沒有 GeoRA、vLLM、獎勵或參數更新。

固定使用 S12 保存的 token IDs；不重新抽一段不同回答。檢查的是原來概率偏差最大的 token：第 1 行、第 9 個 completion step，token ID `1986`，文字 `This`（索引從 0 開始）。batch 仍有 8 行。

同一模型執行三條路徑：

| 路徑 | 如何計算這個位置 | 該次輸入的序列寬度 |
|---|---|---:|
| cached | 先處理 prompt；之後每次只輸入一個新 token，使用歷史 K/V | prompt 50；decode 1 |
| prefix no-cache | 從頭重算截至該位置的全部前文 | 59 |
| fixed no-cache | 使用訓練原有固定寬度輸入；causal mask 遮住未來 | 66 |

59／66 包括左側 padding；這個位置能看到的有效前文有 **49 個 tokens**。比較的是這個相同前文位置的中間量和下一個 token 的 logits。

cached replay 用保存的 IDs 強制繼續，hook 在強制選 token 之前讀取原始 logits。完整 151936 個 vocab logits 的兩項驗證均精確通過：cached replay 重現 S12 的 cached snapshot；fixed no-cache 重現 S12 的無 cache snapshot。因而本輪確實定位原有差異。

## 2. 先排除輸入、位置和遮罩問題

實際 forward hook 與 attention hook 確認：

- 三條路徑在有效前文上的 token IDs、position IDs 相同。
- 該位置實際能讀到的 49 個 key 位置相同。
- embedding 和第 0 層 input normalization 的輸出逐元素相同。

cached 的注意力張量為 `[8,12,1,128]` query 與 `[8,2,59,128]` key；fixed 路徑 query/key 寬度為 66。選定位置可讀的 keys 相同，但整批乘法的形狀不同。這不是「只改 cache 開關、所有算子形狀都固定」的消融。

## 3. 第 0 層到底哪裡不同？

下表比較 **cached 與 fixed no-cache，在該目標位置**的 BF16 native forward：

| 計算邊界 | 最大絕對差 | 解讀 |
|---|---:|---|
| embedding | 0 | 相同輸入 |
| layer 0 input norm | 0 | 投影前相同 |
| 原始 Q 投影 | `1.9073486e-6` | 最早捕獲到的目標位置微差 |
| 原始 K／V 投影（目標位置） | 0 | 這個 token 的 K/V 相同 |
| RoPE 之後的目標 Q | 0 | 前面的 Q 微差在後續捨入中消失 |
| attention 使用的歷史 K | 0 | 49 個有效位置相同 |
| attention probabilities | 0 | 該 query 對歷史位置的權重相同 |
| attention 使用的歷史 V | `0.00048828125` | 12544 個元素中 6 個不同 |
| attention 輸出 | `0.00006103515625` | 加權組合歷史 V 後開始不同 |

**「目前這個 token 的 V 相同」不等於「整段歷史的 V 相同」。** cached 保存的是先前逐步算出的 V；無 cache 用這次的整批輸入重新計算歷史 V。本輪另外捕獲了完整有效歷史，沒有只比較當前 token。

attention probabilities 是該位置給每個歷史 token 的權重；attention 輸出是用這些權重加權歷史 V。這裡記作 $P V$，其中 $P$ 是 attention 權重，並非 vocab 的 softmax 概率。

這段結果不能簡化成「Q 的微差造成 19.7%」：Q 微差在進入 attention 前已經消失。也不能說 6 個 V 元素已被證明解釋全部偏差：兩條路徑的 $P V$ 乘法形狀也不同，尚未分離歷史 V 差異與乘法執行方式各自的影響。

prefix no-cache 對照在目標位置的原始 Q/K/V 都與 cached 精確相同，最早目標位置差異直接在 attention 輸出出現；歷史 V 有 3 個元素不同，最大差同為 `0.00048828125`。兩種無 cache 路徑都支持繼續檢查歷史投影和 attention 值的組合。

## 4. 後面如何傳到 logits？

以下仍是同一目標位置，cached 與 fixed 路徑比較：

| 邊界 | 最大絕對差 | 相對 L2 差 |
|---|---:|---:|
| 第 0 層 block 輸出 | `0.00390625` | `0.117849%` |
| 第 1 層 block 輸出 | `0.0625` | `1.459812%` |
| 第 15 層 block 輸出 | `0.25` | `1.267869%` |
| 第 27 層 block 輸出 | `1.5` | `1.967614%` |
| 最後 normalization | `1.0` | `1.618069%` |
| 全 vocab logits | `0.265625` | `1.748613%` |

差異沿後續層傳播，沒有逐層單調增大。表中的 L2 差衡量整個向量，不是單一 token 的概率差，也不是模型準確率。

`This` 自己的 logit 為 cached `23.25`、fixed `23.50`。兩個完整 vocab 向量用同一 FP64 log-softmax 計算，其概率比為 `1.1969750175109515`，即相對差 **19.6975%**。S12 跨全部 103 個位置的最大 logits 差是 `0.5`；本表的 `0.265625` 只屬於本輪定位的這一個位置。

## 5. 相同輸入的局部投影對照

取第 0 層 input norm 的同一份輸入數值，使用同一 Q/K/V 權重，分別以連續張量 `[8,1,1536]`、`[8,59,1536]`、`[8,66,1536]` 執行投影，然後只比較同一目標位置。

Q 的 one/full 對照重現 `1.9073486e-6`、一個元素不同的原始微差；full 對照精確重現完整 forward 的投影值。這支持 **局部 Linear 的數值結果會受整批形狀影響**，不需要先假設上游 hidden state 已不同。

目標 K/V 的這項 BF16 對照結果相同；它不代表所有歷史位置都相同。報告也保存 FP32 `F.linear` 對照，FP32 在某個局部算子上的兩條路徑差異不一定比 BF16 更小；不能把端到端改善推廣成所有中間量均改善。

本輪沒有查驗底層 kernel ID 或 kernel 原始碼，因此不把形狀敏感的浮點結果稱為 AMD kernel bug。

## 6. FP32 對照改了什麼？

先載入 BF16 原模型完成追蹤，再把**這個模型已有的 BF16 權重數值提升到 FP32**。BF16 數值可以精確表示在 FP32 中；沒有重新讀取捨入前的原始 FP32 checkpoint。報告另外逐元素驗證一個代表性 Q 權重的提升完全一致，並非逐一驗證全部權重檔。

固定同樣 token IDs、輸入、位置、三條路徑，改用 FP32 forward：

| forward 設定 | fixed/cached 的 `This` 概率比 | 相對概率差 |
|---|---:|---:|
| native BF16 autocast | `1.1969750175109515` | `19.69750175%` |
| 相同權重數值，FP32 forward | `1.0000327280290207` | `0.00327280%` |

BF16 forward 中 native normalization 和 attention softmax 包含 FP32 計算，不是每一個算子都 BF16。捕獲的中間量提升 FP32 保存到 CPU，差值／概率分析用 FP64；**沒有 FP64 模型 forward**。

FP32 對照支持低精度與不同執行形狀對端到端概率差有重要影響，但僅檢查原來最差的這一個 token。其 ratio 偏離 1 為 `3.2728e-5`，仍超過既有 `2e-5` 門限；沒有檢查全部 103 tokens、更新後策略、GRPO backward 或恢復。這不構成可用 FP32 訓練修正的驗收。

## 7. 檢查與用量

| 執行 | 結果 | 時間與资源 |
|---|---|---|
| Mac CPU 小模型 | 38/38，4 cases；FP32/BF16 均含 decode | Python 本體 0.119 秒 |
| Setonix login CPU 小模型 | 38/38，4 cases；同上 | 2 CPU threads，0.356 秒，無 GPU allocation |
| MI250X 原模型，job `50660028` | 23/23 定位檢查，BF16／相同權重 FP32 兩 cases | Python 本體 10.064 秒；allocated peak 5.894 GiB |
| 該 GPU 作業 | `COMPLETED, 0:0` | **一個邏輯 GPU，實際 allocation 36 秒，硬上限 2 分鐘** |

本轮把逐層追蹤、局部投影對照與 FP32 對照放在同一次模型載入／allocation。CPU 預先驗證臨時 checkpoint 的 338 個權重 tensors 與固定來源相同；GPU 驗證該來源記錄、revision 與舊 logits snapshot hash。

本輪作業 queue 已空，臨時模型副本已刪除；scratch 原 checkpoint／初始化保留。CPU staging manifest 在清理前未下載，原來源驗證已見 GPU gate／執行輸出；不把缺少的 manifest 補造為原始文件。所有 17 個已記錄作業 allocation 合計 **1901 秒，即 31 分 41 秒**。

## 8. 下一步怎樣區分原因？

若繼續定位，下一次 forward 先捕獲並保存實際 attention 權重與歷史 V，再做兩個受控比較。當前 JSON 保存的是差異指標，並未持久保存原始 P/V 張量：

1. **固定相同 $P,V$ 數值，只改 $P V$ 執行形狀**，測乘法本身的差異。
2. **固定相同形狀和 $P$，只替換 cached／重算的 V**，測歷史 V 差異造成多少输出變化。

以上是尚未執行的因果對照。取得候選修正後，仍需全部有效 tokens、更新後策略、LoRA／GeoRA GRPO 與恢復檢查。現有訓練 sampler、精度政策及概率一致性門限保持原設定。

## 9. 代碼和報告

- [逐層捕獲與局部對照](../../scripts/cache_layer_probe.py)、[CPU 檢查](../../scripts/check_cache_layers_cpu.py)、[GPU 入口](../../scripts/check_cache_layers_gpu.py)、[Slurm job](../../jobs/cache_layers.sbatch)。
- [Mac CPU](../../reports/cache_layers/cpu_local.json)、[Setonix CPU](../../reports/cache_layers/cpu_setonix.json)、[GPU 完整結果](../../reports/cache_layers/50660028.json)、[作業與清理](../../reports/cache_layers/remote_final.json)、[用量總表](../../reports/setonix_jobs.json)。
- 實際 GPU 執行代碼 commit：`d4ca571`。遠端報告：`/scratch/pawsey0807/btian/geora/runs/cache-layers-50660028/cache_layers_gpu.json`。


## 10. 後續受控干預與路徑驗收（L15/S14、L16/S15）

S14已捕獲原生BF16 attention P和歷史V。固定其數值只改query數或零概率future key造成的reduction寬度，這一位置的P@V差為0；固定P/形狀只換歷史V，attention輸出差為0.00006103515625，精確恢復原局部差異。單層端到端V替換改變部分vocab logits，但目標This logit仍23.50、概率比仍1.1927040976，故歷史V只是已證實的局部貢獻。原始snapshot、控制條件、限制與26/26結果見 [S14報告](../../reports/cache_intervention/50660628.json) 及 [EXPERIMENT_RECORD.md](EXPERIMENT_RECORD.md)。

S15改變下一步訓練所用的路徑：生成與可求導重算都使用相同native cache順序，並在backward重算純attention以控制顯存。LoRA/GeoRA各58/58，五步及完整恢復通過，全部有效tokens概率差0，沒有放寬門限。這不表示以前cached和無cache的logits變得一樣，也不是找到唯一kernel bug。方法、兩個真實訓練報告、成本及下一輪選擇見 [CACHED_TRAINING.md](CACHED_TRAINING.md)。


## 11. 历史 V 来源已进一步定位（L17/S16）

新实验完整重现六个历史 V 差异，并验证它们在投影时就已产生，原生 cache 存取逐元素保持输出值；六处均来自 prompt prefill50 对 fixed66。相同输入／权重值的不同投影形状及 BF16 舍入分界证据、FP32 对照限制、本地原始张量复查和 36 秒 GPU 用量见 [CACHED_PROJECTION.md](CACHED_PROJECTION.md)。这仍不是完整 19.7% 最终概率差的唯一归因。


## 后续：全批次的 AMD FP32 对照

同日 job `50668984` 将本页的单 `This` 位置对照扩展到 S12 八段旧回答的全部 103 个有效 token，并逐元素验证所有 1,543,714,304 个参数的 BF16→FP32 数值保持。最大概率相对差仍为 BF16 19.69750175%／FP32 0.00327280%，103 个位置的绝对相对差均减小；FP32 仍有 5 个位置超过旧 ratio 容差 2e-5。完整 GPU 检查 24/24，独立 NumPy 复核 412 个条件概率，新增一个逻辑 GPU allocation 36 秒；没有改训练配置。详细流程、真实概率、精度与用量见 [CACHED_PRECISION.md](CACHED_PRECISION.md)。
