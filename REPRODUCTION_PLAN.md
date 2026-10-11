# GeoRA 復現計畫與進度

更新：2026-10-12。已執行的結果以 [EXPERIMENT_RECORD.md](EXPERIMENT_RECORD.md) 和 `reports/` 為準；本文件安排尚待執行的工作。

**目前位置：階段 1–3 完成；階段 4 的 LoRA／GeoRA 五步、完整恢復及 cache 候選均已驗收，分段pilot入口和validation评估已实现；原模型256-token诊断已完成（32/128=25%，单逻辑GPU793秒），但42/128截断，不能继续作为效果实验基线。下一步先对齐1024／4096长度与最终数值提取规则，建立新基线并验证长回答容量，再开始LoRA／GeoRA训练。详见下文“长度与正式目标修正”及 [BASE_VALIDATION.md](BASE_VALIDATION.md)。** L12/S11 的混合計算路徑 likelihood 曾失敗；L13–S14 完成 logits、逐層及 P/V 受控干預。L16/S15 改為生成和梯度皆使用相同 native cache 順序，兩方法各58/58、共19,991實際有效 completion tokens 精確對齊，原門限不變。新 cache 完整成本較高，下一輪 pilot 預設沿用已驗收的 `fixed_width`。方法與成本見 [CACHED_TRAINING.md](CACHED_TRAINING.md)，本輪完成標準見 [GOAL.md](GOAL.md)。尚未比較論文任務分數。

## 1. 第一個復現目標

先完成論文的小模型設定：**Qwen2.5-1.5B-Instruct，在 GSM8K 上用 GRPO 訓練，比較原始模型、LoRA 與 GeoRA。** 正式結果對應論文 Table 8 的 1.5B LoRA／GeoRA 兩行；原始模型分數另外保留，作為改進與遺忘的參照。

論文附錄 C.2 明確列出這組模型和訓練資料。Table 8 的數學評估是 AIME24、AIME25、MATH500、OlymMATH，並有 HumanEval、GPQA、MMLU 的域外評估；**GSM8K 訓練 reward 或 test accuracy 不能直接代替該表的分數**。[GeoRA v4，附錄 C.2–C.3 與 Table 8](https://arxiv.org/html/2601.09361v4)

完成這一組後，再依預算加入 PiSSA／MiLoRA、FullFT／SparseFT、消融或更大模型。8B 主實驗、醫療及 32B 程式碼實驗列為擴展，分別標記完成狀態。

## 2. 整體路線

| 階段 | 要回答的問題 | 完成標準 | 狀態 |
|---|---|---|---|
| 1. 來源與初始化 | 模型、mask、SVD 和 A0/B0 是否符合選定設定？ | 固定來源版本；196 個目標層；FP32 因子與有效權重檢查 | 已完成 |
| 2. 數值與更新 | 裝入 GeoRA 後能否維持初始輸出，且正確更新和重載？ | 完整模型初始化、A/B 單步 CE、保存重載、短生成/cache/padding 通過 | 已完成 |
| **3. GRPO 流程** | 真實採樣、reward、advantage 和策略損失能否正確產生更新？ | CPU 小例子驗收，再完成 1.5B 的一次真實 GRPO 更新 | **已完成，S9 20/20** |
| 4. 短程對照 | 連續更新與恢復是否可靠？實際訓練要多少資源？ | 共用五步／完整恢復與成本量測 | 两条路径五步及完整恢复已完成；pilot入口／评估已实现；Base128题已完成，LoRA／GeoRA各10步待做 |
| 5. 正式任務實驗 | 相同訓練預算下，效果與能力保留有何差異？ | 固定協定訓練及評估；記錄 Base/LoRA/GeoRA 分數與用量 | 待做 |
| 6. 機制與擴展 | 結果是否支持論文提出的幾何解釋？ | 分析實際 GRPO 更新，加入必要消融及其他 baseline | 待做 |

## 3. 我們已完成什麼

- 固定基底 `Qwen/Qwen2.5-1.5B-Instruct`，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。
- `r=16`、`alpha=32`、`rho=0.2`；28 個 block 的 Q/K/V/O/gate/up/down，共 196 個投影，排除 lm_head。
- 392 個可訓練 A/B tensors，共 18,464,768 個參數；其餘權重和 A0/B0 固定。
- 完整模型 S7：43/43，初始化、一次 CE 更新、模型與 optimizer 保存重載通過。
- 完整模型 S8：64/64，未更新的短 greedy 生成、KV cache、左側 padding，以及三種學習率的獨立首步通過。
- 本機與 Setonix 容器 CPU 小模型，FP32/BF16 各 21/21 的生成與 padding 檢查通過。
- L12/S11：本機與 Setonix 登入節點的 cached rollout 小模型，各 34/34 機械檢查、8/8 likelihood 門限通過；FP32 最大 token logp 差 `2.384185791015625e-7`，BF16 為 `0`。
- S11 完整 MI250X BF16：10/10 機械檢查通過，初始化 GeoRA 與原模型 cached tokens／logp 完全相同；兩者對無 cache 訓練 scorer 的 likelihood 門限均未通過（0/2）。最大 token logp 差 `0.17979764938354492`，概率比範圍 `[0.8870854910554354, 1.196975129959193]`。兩次 GPU allocation 合計 72 秒，沒有 optimizer update；第一個 job 在 tokenizer API 處失敗，修正後才取得數值結果。
- L13/S12：成對 logits 的 tiny CPU 檢查在本機與 Setonix 登入節點均為 10/10，FP32／BF16 各一個原模型 case，沒有更新。完整 MI250X BF16 只載入一個凍結原模型，6/6 診斷檢查通過；103 個有效 token 的原始 logits 最大差 `0.5`，捕獲 logits 重算生成 logp 的誤差為 `0`。最差位置的 FP64 概率比 `1.1969750175109515` 重現 S11 的約 19.7% 差異，確認主要差異已在 softmax 前出現；具體 decoder／kernel／BF16 原因尚未定位。job `50659217` 新增 allocation 34 秒、沒有 GeoRA 或 optimizer update；累計已記錄 allocation 1865 秒。
- L14/S13：[逐層診斷](CACHED_LAYERS.md) 重放 S12 的同一原模型／同一 token，MI250X 檢查 23/23，BF16 兩條路徑精確恢復舊 logits snapshot；實際有效 token、position IDs 與目標 attention 可見 key 一致。固定寬度比較中，當前 token 最早在第 0 層 `q_proj` 出現微差，但 RoPE 後 Q 相同；歷史 V 不同，第 0 層 attention 概率相同而其加權 V 輸出不同。BF16 autocast 路徑包含部分內部 FP32 計算。同一權重值全 FP32 對照把該 token 概率比縮至 `1.0000327280290207`，仍超過原 `2e-5` 相對比例門限；僅此 token 的定位，不是全部 103 token 或訓練驗收，也未指定單一 kernel bug。job `50660028` 一個邏輯 GPU，新增 allocation 36 秒；17 個已記錄作業合計 1901 秒。

- L15/S14：實際 attention P/V snapshot 受控干預26/26；固定 P/V 改乘法形狀的該局部差為0，只換歷史V恢復第0層目標attention輸出差。單層端到端替換仍有約19.27%概率差，沒有唯一根因結論。短GeoRA初始化／合成更新的cache概率亦精確對齊。
- L16/S15：CPU streaming sampler和attention checkpoint在本機／Setonix各76/76、45/45；MI250X cache GeoRA與LoRA均58/58，五步及完整恢復通過。兩者分別9834／10157實際有效tokens採樣／重算差0；沒有importance correction或門限放寬。三個新單GPU作業共1370秒，全部終止并清理副本。詳細源版本、精度、報告與全部用量見 [EXPERIMENT_RECORD.md](EXPERIMENT_RECORD.md)。

S8 最後恢復 `untrained_restored`。三次 CE 試驗是各自從初始化出發的一步，沒有累積成三步模型；也沒有任務獎勵。S7 的單步測試檔案另行保留。**正式 GRPO 訓練必須從基底＋未訓練初始化開始。**

原 residual 的 BF16 初始化門檻曾失敗。目前通過驗收的是本次實作的 `difference` 計算順序：

```text
output = W_pre @ x + c * [B @ (A @ x) - B0 @ (A0 @ x)]
c = alpha / r
```

它與論文殘差形式在實數代數下等價，但浮點運算順序不同。原始分支 BF16；A/B/A0/B0、兩條低秩分支及相減 FP32，校正轉回 BF16 後相加。這個工程選擇要出現在最終復現報告中；計時也包含額外初始分支的成本。

先前 `/Users/tianbaochen/RLVR` 的公開 checkpoint 幾何分析作為背景探索保留。後續論文機制分析使用本次實際訓練出的 LoRA／GeoRA 更新。

## 4. 論文設定與我們的試跑設定

下列論文值已於本次重新核對；缺項需在正式實驗前處理。[GeoRA v4，附錄 C.1–C.2](https://arxiv.org/html/2601.09361v4)

| 項目 | 論文公開設定 | 本次安排 |
|---|---|---|
| 優化 | GRPO、AdamW、BF16 | 階段 3 起接入；維持已驗證的逐項精度 |
| 適配器 | all-linear；r=16、alpha=32；GeoRA rho=0.2 | 196 個投影；記錄排除 lm_head 的具體範圍 |
| 預設 learning rate | 1e-6 | 首次 GRPO 試跑採此值；CE 測量只作診斷 |
| Global batch、每題 rollout | 128、8 | 短檢查縮小；正式前明確 batch 指 prompt 還是回答，以及梯度累積方式 |
| KL coefficient | 啟用時 0.001 | 首次試跑採 0.001，reference 是凍結的原始模型 |
| 最大 prompt／response 長度 | 1024／4096 | 短檢查及目前 pilot 計畫均 512／256，標記縮減；擴長前另行驗收 |
| 數學訓練資源 | 一節點、8 個 80GB GPU | 先用一個 Setonix 邏輯 GPU；擴大前量測及核對分散式運算 |

尚待固定或確認：訓練總步數/epoch、prompt 原文、採樣 temperature/top-p/top-k、old policy 更新與樣本重用次數、clip epsilon、advantage 的標準差約定、KL 估計/求導方式、AdamW 的其他選項、scheduler、seed、截斷回答規則、checkpoint 選擇和每個 benchmark 的解碼/重複採樣協定。

目前讀到的附錄沒有逐項說清這些設定。後续配置需寫出每項來源：論文、外部算法實作或我們的選擇。若無法補齊，結果標記為公開設定下的獨立復現；保留所有差異，避免把數字接近當作相同實驗協定的證據。

## 5. 已完成：階段 3 的具體工作與驗收

### 3A. 本機/登入節點：資料與可手算的 GRPO 檢查

1. 從 [GSM8K 原始倉庫](https://github.com/openai/grade-school-math) 固定版本取得 `train.jsonl`、`test.jsonl`，保存來源、SHA256 和題目 ID。訓練從 train 中取題；從 train 分出調參用 validation；test 只作最終評估。
2. 整理成「問題、標準最終數值、題目 ID」。模型輸入只含問題與固定 prompt；標準解答不放入生成的前文。[GSM8K 資料格式](https://github.com/openai/grade-school-math#dataset-details)
3. 建立答案提取與數值比對：正確 reward=1，錯誤或無法解析 reward=0。用實際格式測負數、逗號、小數、多個中間數值及缺失最終答案；使用 Decimal/精確規則，避免用鬆散的數值容差接受錯答案。
4. 用可手算 reward 組與小模型，驗證分組均值/標準差、正負 advantage、ratio、正負分支的 clipping、KL 及梯度。整組同分時 advantage=0，記錄零 policy 訊號組；更新後仍可能有 KL 梯度。不製造 reward 差異。
5. 核對 causal shift 與 completion token mask：排除 prompt 和 padding，EOS 與截斷回答依明確規則處理；用獨立 token 索引確認 loss。

短程先沿用現有 PyTorch/Transformers 和自訂 GeoRALinear，建立可讀、可測的採樣/打分/更新流程。GRPO 的算法參照 [DeepSeekMath](https://arxiv.org/abs/2402.03300)。倉庫已提供可讀的短 GRPO 共用流程；正式擴大時再按吞吐評估訓練框架，框架/rollout engine 的 reference、dtype 和有效權重必須另行驗收。

### 3B. Setonix：第一次真實 GRPO 更新

建議檢查配置：一個邏輯 GPU；GSM8K train 固定 16 題的候選池；每次 2 題、每題 4 個回答；response 最長 256；temperature=1、top-p=1、top-k=0，關閉其他採樣修飾；lr=1e-6、beta=0.001。這是本次流程檢查配置。

未由論文逐項指定的 smoke 選項先明確採用：clip epsilon=0.2；組內 population 標準差（除以 G）加 1e-8；每批 rollout 一次更新；先對每個回答的有效 completion token 平均，再對回答平均。AdamW betas=(0.9,0.999)、eps=1e-8、weight_decay=0，gradient norm 上限 1；adapter dropout=0。這些已作為工程檢查的明示配置執行，後續正式設定依來源補齊情況另行固定。

```text
模型對問題採樣多個完整回答
    ↓
提取每個回答的最終數值 → reward
    ↓
同題回答的 reward → advantage
    ↓
保存 old log probabilities，取得 current/reference log probabilities
    ↓
計算 clipped GRPO loss + KL → backward → 只更新 A/B
    ↓
檢查更新後生成、凍結權重、reference、數值及日誌
```

old log probabilities 必須在該批次更新前保存並 detach，固定到該批次使用結束；reference 全程固定原始策略。先用相同 teacher-forcing scoring 路徑檢查未更新 old/current 的 ratio=1、初始化 policy/reference KL 接近零。這只能驗證打分與保存一致，不能證明 rollout 與 scoring 代表同一個策略。

另外測量「實際採樣概率」與「重算已保存 token 的概率」：全部有效 token 的 delta logp、ratio 和整段 logp 差。S9/S10 的 fixed 路徑和S15的新cached路徑皆通過；S11的cache生成＋fixed重算則失敗，該歷史結果保留。S15 用相同執行順序對齊兩端，未採用 importance correction；最大逐token logp差／ratio偏離1仍為2e-5，整段差1e-3。只有old/current相同不夠，實際sampled/old和更新後策略亦須核對，所有有效token的概率、ratio和KL須有限。

至少在一個**真實混合 reward 組**中完成非零策略梯度與 A/B 更新；如果整組同分，記錄並繼續有限候選題，不因 loss=0 就宣稱完成更新。檢查-only 的人工 reward 例子與真實答案 reward 報告分開保存。

尋找混合組僅限首次機械驗收。階段 4、pilot 和正式對照按固定資料順序保留全部組，包括同分組；各方法不能自行篩選更有訊號的題目或回答。

驗收：

- [x] CPU 資料、答案解析、reward、advantage、clipping、KL 和 token mask 檢查通過。
- [x] stochastic sampling 的回答/概率有限；更新前 old/current 的相同 scoring 路徑一致。
- [x] rollout/scoring 的 likelihood 差異符合事先固定的協定，必要的路徑對齊/校正已驗證。
- [x] 實際 train mode、dropout 設定、device 搬移與 dtype 符合配置；沒有把整個 adapter 轉成 BF16。
- [x] 首次真實 GRPO 的 loss、梯度、參數有限，A/B 有實際變化。
- [x] 凍結權重/A0/B0/reference 不變，reference 仍代表原始模型。
- [x] 更新後可生成；完整 padding batch 的 backward 通過。
- [x] 報告保存題目、回答、reward、advantage、token 數、ratio/KL、clip fraction、梯度、精度和用量。

本次 CPU 預檢／載入計時先完成；job 50601905 實際 allocation 124 秒，1479 有效 token 的 likelihood 差異 0，20/20 通過。格式和截斷也影響 reward，限制見 [GRPO_SMOKE.md](GRPO_SMOKE.md) 與 [報告](reports/grpo/50601905.json)；教學見 [notebook](notebooks/gsm8k_grpo_tutorial.ipynb)。

## 6. 階段 4：短程連續更新與 LoRA 對照

LoRA／GeoRA 各完成 5 步真實 GSM8K GRPO，並通過第 2 步存檔後重做第 3 步的精確恢復（GeoRA 52/52，LoRA 52/52）。一個邏輯 GPU 依次執行，allocation 合計 914 秒；實際結果見 [L11/S10](EXPERIMENT_RECORD.md) 與 [操作說明](GRPO_CONTINUATION.md)。

本輪已用同一短程配置各跑 LoRA/GeoRA 的 5 次連續更新，核對更新後採樣、batch token mask 和數值。全組 reward 相同的比例、解析失敗率、長度上限命中率都要記錄；reward 不要求每一步單調增加。

保存並恢復 adapter、optimizer、scheduler、RNG、資料位置；若 checkpoint 在樣本重用期間，還要恢復該批回答與 old log probabilities。比較恢復前後的相同 scoring 路徑與下一次固定批次更新；純模型/optimizer 重載不足以驗收完整訓練恢復。

上述無 cache 五步與恢復已通過。後續 S11–S14 釐清混用生成／概率重算路徑的差異；S15 以新 cache 路徑完成兩方法五步、所有token概率核對與完整恢復。受控歸因只建立第0層歷史V的局部貢獻，並未找到整個19.7%差異的唯一根因。[CACHED_TRAINING.md](CACHED_TRAINING.md) 保存完整資料流和成本；新cache可用，但本次整段耗時與顯存較高，pilot兩方法共用已有fixed路徑。

**下一步：驗收通用pilot入口的完整模型首段，再分段推進。** [PILOT_RUN.md](PILOT_RUN.md)說明新訓練／評估入口；[grpo_pilot_plan.json](configs/grpo_pilot_plan.json)保留100步協定。`check_grpo_continuation.py`仍只驗收五步，新入口直接沿用它的單步計算。規劃固定1024題train／128題validation；每批2題、每題4回答、最長256token，每種方法最多100個optimizer steps，同seed/order/reward/概率路徑。先保存未訓練Base的同協定validation結果；用validation選checkpoint，不查看test選參數。

維持已驗收的batch與長度；本輪cache的截斷／解析失敗比例較高，下一階段要繼續記錄。若改為8回答或512+tokens，先核對容量、全部token概率及恢復，不能視為現有驗收自動涵蓋。先在pilot第一個短段量測完整工作量，再訂Slurm時限及按checkpoint分段；本輪沒有提交pilot。

pilot 驗收是流程連續可用、恢復可靠及測得資源成本；100 步的分數用於決定正式配置，不當作論文 Table 8 的復現結果。若單次 GPU 預算不足，按 checkpoint 分段；不將短 CE 的耗時外推成 GRPO 時間。

## 7. 階段 5：正式 LoRA／GeoRA 實驗

訓練開始前保存一份固定配置，明確列出尚缺的論文選項、我們的補充選擇及差異。

- 相同 base revision、target modules、r/alpha、dtype、prompt、答案解析器、reward 和 GRPO 公式。
- 相同 train/validation 題目、資料順序、seed、採樣協定、reference 和 rollout/optimizer 預算。模型分歧後採樣的文字自然不同。
- 事先固定 checkpoint 選擇規則；用 validation 調參，test 與正式 benchmark 不參與調參。
- 保留 Base 的相同評估協定結果；每個回答及正誤判定可追溯。列出截斷和解析失敗。
- 首輪按共同 seed 比較；有差異後建議增加到 3 個 seed，報各輪分數與均值/波動。此重複是我們新增的可靠性檢查。

評估分兩層：

1. **流程與近端效果**：GSM8K validation/test，確認在這項訓練任務的效果；標記為我們新增的測量。
2. **論文對照**：先完成 MATH500，再擴到其餘 Table 8 的 ID/OOD benchmark。只有全部指定協定與分數完成後，才標記相應表格子集已復現。若先只做 MATH500，明確寫「Table 8 的 MATH500 LoRA/GeoRA 子集」。

GPU 秒、生成 token 數、optimizer steps、初始化時間、載入時間、rollout/打分/backward 分段耗時與峰值顯存分別報告。MI250X 與論文硬體不同；本次 difference 多一條固定分支，因此效率結果按實際實作解讀。

## 8. 階段 6：幾何、消融和更大模型

用正式 GRPO 的 `delta_W=W_trained-W_pre`，分析各層原權重方向上的更新、奇異值譜、主要子空間重合與更新低秩可壓縮性；依論文定義對齊指標，保存 trainable tensors、方法和 normalization。

按優先順序增加：僅 Euclidean／僅 spectral／union mask；Random-r／Tail-r；PiSSA/MiLoRA；必要時 FullFT/SparseFT。資料與訓練/評估預算固定，再分析差異。8B/32B 與跨域實驗另立配置和資源預算。

## 9. 執行與記錄約定

- 本機：編寫、CPU 小例子、報告整理；Setonix 登入節點：兩執行緒的資料與必要 CPU 初始化/預檢；Setonix GPU：完整模型生成與訓練。
- 程式/環境放 software，模型/資料/cache/checkpoint/結果放 scratch。scratch 載入變慢時先 CPU 診斷，必要的臨時副本先校驗、結束後清理。
- 逐階段保存 JSON、實際 dtype、source commit、dataset IDs、配置和 Slurm allocation。失敗/timeout 保留，計入用量。
- 現有 FP32 初始化直接復用；正常進度沿用已通過的檢查，遇到具體變更或風險才重跑相應項目。
- [GOAL.md](GOAL.md) 是已完成的採樣／概率計算路徑驗收目標；已完成的初始化目標保留於 [GOAL_INITIALIZATION_COMPLETED.md](GOAL_INITIALIZATION_COMPLETED.md)。本文件是整體復現進度入口。階段 3／4 的實際腳本入口與報告見 [GRPO_SMOKE.md](GRPO_SMOKE.md)、[GRPO_CONTINUATION.md](GRPO_CONTINUATION.md)、[CACHED_ROLLOUT.md](CACHED_ROLLOUT.md)、[CACHED_LOGITS.md](CACHED_LOGITS.md) 與 [CACHED_LAYERS.md](CACHED_LAYERS.md)。

**下一可交付结果调整为：完成1024／4096长度和明确最终数值提取规则的评估协议，重新建立Base基线，并量测长回答训练容量。完成后再做LoRA／GeoRA首段训练。已完成的256-token、25%结果保留为历史诊断，不与新协议分数直接比较。**


## 10. 2026-10-11 精度调查后的状态

Metal／AMD 各完成同一十题的四条件完整生成，另完成同一 GeoRA 短更新负载的 BF16／FP32 成本对照及相关文献调查，见 [PRECISION_INVESTIGATION.md](PRECISION_INVESTIGATION.md)。正式 LoRA／GeoRA 对照仍保留已验收的 BF16 混合精度和共同概率计算路径；FP32 独立作为数值／短训练对照。十题输出与单 policy 计时不等于论文成绩，也不代替完整 pilot 容量测量。下一交付仍是第 6 节的通用 pilot runner 和固定 validation；本轮未提交 pilot。

## 11. 2026-10-12 阅读后的精度决定与新入口

暂缓BF16／FP32×LoRA／GeoRA四组100步精度消融；先推进已验收精度和fixed_width概率路径下的两方法试跑。理由、精度表、训练／评估数据流、脚本和资源控制见 [PILOT_RUN.md](PILOT_RUN.md)。新入口支持完整边界恢复，固定validation评估不参与训练loss；尚未启动100步pilot，不作论文分数或长期稳定性结论。


## 12. 长度与正式目标修正（2026-10-12）

原模型128题诊断有42题（32.81%）达到256新token上限。将早期256-token流程检查直接延用到效果比较，会使长度和格式遵循过多影响结果。后续效果实验先调整协议，再开始训练；本轮未追加GPU作业。

重新直接核对 [GeoRA v4附录C.1–C.3、Table8](https://arxiv.org/html/2601.09361v4)：数学**训练**公开上限为prompt1024／response4096；1.5B扩展使用Qwen2.5-1.5B-Instruct、GSM8K。论文的数学评估为AIME24／AIME25／MATH500／OlymMATH，Table8没有GSM8K分数栏。数学评估的生成上限、temperature和重复采样次数没有逐项单列，不能宣称已确认其全部评估也使用4096。

先前128题是我们从GSM8K train留出的近端验证集，用于验证生成、判分及资源成本，并为同协议的小规模变化提供观察；它没有比较GeoRA，也没有复现Table8。25%属于当前严格格式和短长度条件。此前承接“LoRA／GeoRA各10步、256token”的安排被本节替代。

修正后的顺序：

1. 对齐公开长度：后续数学训练目标prompt1024／response4096；下一版Base验证也采用4096作为明确的独立复现选择。4096是上限，遇EOS立即结束。
2. 在运行前冻结并测试最终数字的提取规则。论文只写提取最终数值再exact match，没有声明必须唯一`####`。支持哪些明确最终答案格式需列出来源／补充选择，解析失败与数值错误分别报告；不能任取推理中的数字来凑正确。历史严格`####`结果保留，不覆盖、不与新规则混算。
3. 重建长回答Base基线，128题仍只作诊断／调参；正式任务评价先落实MATH500，并继续列出Table8其余数学及域外指标的复现范围。
4. 先量测4096回答长度下的生成、全部有效token概率、backward、显存及恢复容量。现有单逻辑GPU、batch与fixed_width的256验收不自动覆盖4096。优先用实际提前EOS的长度、分组微批和必要的梯度累积控制内存；容量结果决定训练实现。
5. 长回答训练验收后开始LoRA／GeoRA对照，两方法共用数据顺序、每题采样数、有效batch定义和优化预算。论文默认rollout=8、global batch=128；需明确我们实现的batch单位及梯度累积，不能将每批2题／4回答直接称为同设置。

论文未报告的总训练步数／epoch、prompt原文、最终数值提取代码、checkpoint选择及详细评估解码，需要记录为补充选择或向作者确认。扩大训练步数之前先补齐协议；少步数验收的目的仍是流程和容量，不能当方法效果结论。

`configs/grpo_pilot_plan.json` 的256-token数值保留用于历史流程检查和证据身份，不作为修正后效果训练配置。新协议需独立保存，避免旧检查点／旧基线与新长度或评分规则混用。

## 13. 已授权的连续执行目标（2026-10-12）

已有128份重判→4096-token Base基线→长回答容量→LoRA/GeoRA各100步→同一128题验证。具体协议、验收和单逻辑GPU资源控制见[LONG_RUN_PLAN.md](LONG_RUN_PLAN.md)。固定verl原版字符串判分，历史Decimal结果保留；本轮是独立小规模训练，不代表Table8完整复现。
