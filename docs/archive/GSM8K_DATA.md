# GSM8K：資料與獎勵的固定規則

這是我們第一次真實 GRPO 更新的資料入口。它只處理文字與數字，完全不需要 GPU。

## 1. 從哪裡下載

使用 [OpenAI 的 GSM8K 原始倉庫](https://github.com/openai/grade-school-math)，固定在 commit
`3101c7d5072418e28b9008a6636bde82a006892c`。程式只下载兩個 JSONL 和 MIT license，沒有额外的模型下載，也不需要 `datasets` 套件。

| 原始檔案 | 筆數 | SHA256 |
| --- | ---: | --- |
| `train.jsonl` | 7,473 | `17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465` |
| `test.jsonl` | 1,319 | `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14` |

每筆原始資料包含 `question` 和 `answer`。`answer` 包括人工解題文字，最後用 `####` 標示數字答案。

原始資料與分割後的資料都放在被 Git 忽略的 `datasets/`，Setonix 則放在 `$MYSCRATCH/geora/datasets/`。完整原始人工答案保存在 `raw/`；訓練入口只保留題目與獨立的數字標籤。

## 2. 我們如何分割

固定 seed `20261010`，對原始訓練集每個 ID（如 `train:01480`）計算 `SHA256(seed:ID)`，按它排序：

1. 前 128 題作為 validation。
2. 剩下 7,345 題作為 train。
3. train 的前 16 題作為這次 smoke 檢查的固定候選池。
4. 原始 test 的 1,319 題完整保留，這次檢查不使用。

因此 smoke 候選池只來自 train，不依照模型答對或答錯重新挑題。`manifest.json` 保存分割規則、來源網址、檔案雜湊、validation ID 與 16 個候選 ID。這是我們的復現規則，不是已確認的作者資料分割。

## 3. 模型看到什麼

```python
rows = load_prepared_rows(data_directory, "smoke")
row = rows[0]
prompt = build_prompt(row["question"])
```

`prompt` 包含固定的作答格式說明和題目，要求模型以 `#### <number>` 結尾。`row["target"]` 是檢查器用的答案，不拼進 prompt；原始人工解題文字也不拼進 prompt。

例如第一個固定候選是 `train:01480`：問兩個人的冰淇淋球數相差多少。檢查器的標籤是 `4`，模型輸入只包含題目與固定說明。

## 4. 一份回答如何變成 reward

```python
result = score_answer(completion, row["target"])
reward = result["reward"]
```

規則是：回答必須有且只有一個 `####`；它後面除了空白，只能是一個數字。用 Python `Decimal` 精確比較數值，相同給 `1.0`，不同或無法解析給 `0.0`。

| 模型回答 | 參考答案 | reward | 原因 |
| --- | --- | ---: | --- |
| `先算 3 和 7。\n#### 10` | `10` | 1 | 中間數字不影響最後答案 |
| `#### 1,234.500` | `1234.5` | 1 | 合法千位分隔與小數尾零等值 |
| `#### -0.50` | `-.5` | 1 | 負小數等值 |
| `#### 0.30000000000000004` | `0.3` | 0 | 精確數字不同，沒有鬆動容差 |
| `答案是 5` | `5` | 0 | 沒有明確最終答案標記 |
| `#### 5 cats` | `5` | 0 | 標記後有單位文字 |
| `#### 5\n#### 6` | `6` | 0 | 多個標記有歧義 |
| `#### 1/2` | `0.5` | 0 | 這次不解析分數或算式 |

也不接受科學記號、百分比、貨幣符號、NaN 或 Infinity。原始 8,792 筆參考答案全部能被這個數字規則解析。**這個嚴格格式與 prompt 是我們選定的實驗條件，不能宣稱是 GeoRA 作者的原始評分器。**

如果模型在 token 上限內沒有完成 final marker，這份回答是無法解析，reward 為 0；不從最後一個推理數字猜答案。

## 5. 如何執行

本地：

```bash
uv run python scripts/gsm8k_data.py --data-dir datasets/gsm8k
uv run python tests/check_gsm8k_data.py \
  --data-dir datasets/gsm8k \
  --report reports/grpo/gsm8k_local_cpu.json
```

Setonix 的 container 環境中：

```bash
python scripts/gsm8k_data.py --data-dir "$MYSCRATCH/geora/datasets/gsm8k"
python tests/check_gsm8k_data.py \
  --data-dir "$MYSCRATCH/geora/datasets/gsm8k" \
  --report "$MYSCRATCH/geora/runs/grpo-preflight/gsm8k_cpu.json"
```

若原始三個檔案已在資料目錄的 `raw/`，第一個命令可加 `--offline`，只驗證與分割。可以將本地準備好的小資料目錄傳去 Setonix，避免在 container 裡重複下載。

CPU 檢查包括數字解析、答錯與格式拒絕、prompt 不洩漏答案、分割互斥、smoke 只來自 train、固定來源雜湊、檔案改壞能被拒絕，以及離線重跑的 manifest 完全一致。它通過只表示資料和 reward 入口正確，尚不表示模型做過 GRPO 更新或得到論文的分數。
