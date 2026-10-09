# GeoRA 復現：精度與驗證約定

更新：2026-10-09。這份文件記錄本次復現採用的設定；後續 notebook 與訓練程式依此執行。

## 1. 採用的精度

| 部分 | 本機 CPU 正確性檢查 | GPU 訓練 |
|---|---|---|
| 讀入的原始 checkpoint | BF16 權重轉 FP32 | 保留固定版本作為來源 |
| mask、兩次 SVD、初始 A/B、殘差計算 | FP32 | 初始化計算仍用 FP32 |
| 凍結的模型權重與殘差 F | FP32 | BF16 |
| 可訓練的 A/B 參數 | FP32 | FP32 |
| A/B 的 AdamW 狀態 | 使用 AdamW 時為 FP32 | FP32 |
| forward | FP32 | BF16 autocast 混合精度 |

GPU 上的 FP32 A/B 指參數本身保留 FP32；autocast 可以在 forward 中選用 BF16 運算。A/B 的更新累積在 FP32 參數中，不在每一步將參數改存 BF16。

包裝成 GeoRA 後，不能直接對整個模型呼叫 `.to(torch.bfloat16)`，那也會轉換 A/B。GPU 階段只轉換凍結部分，並重新檢查所有參數的 dtype。

**必須在執行時檢查實際 dtype。** 初始化完成後、訓練開始前檢查凍結權重與 A/B；第一次 optimizer step 後檢查 AdamW 狀態。設定 `bf16=True` 或啟用 autocast，並不能單獨證明每個 tensor 都符合上表。

舊的 FP64 單矩陣 notebook 保留為數學與誤差參照。正式全模型初始化採 FP32；FP64 不是論文公開的訓練設定。

## 2. 官方資料交代到哪裡

[GeoRA 附錄 C.1](https://arxiv.org/html/2601.09361v4) 明確寫出使用 AdamW 與 BF16 訓練，但沒有交代 SVD、殘差計算、A/B 儲存及 optimizer 狀態各自的精度。上表中逐項的精度分配是本次復現選定的工程設定。

PiSSA 官方倉庫 `GraphPKU/PiSSA` 現在重定向到 [MuLabPKU/PiSSA](https://github.com/MuLabPKU/PiSSA)。它的 [MetaMath 訓練範例](https://github.com/MuLabPKU/PiSSA/blob/main/scripts/metamath_llama2_7b/run_pissa.sh) 使用 BF16；其依賴的 [PEFT 0.14.0 PiSSA 初始化實作](https://github.com/huggingface/peft/blob/v0.14.0/src/peft/tuners/lora/layer.py#L217-L251) 先把權重轉 FP32，再計算 SVD、因子與殘差，之後將基底權重轉回其原有 dtype。本次 FP32 初始化參照這條路徑。

**LoRA 與 GeoRA 使用相同的訓練精度設定。** 比較時也要固定模型版本、資料、訓練預算及評估方法。

## 3. 初始化的計算順序

對每個目標線性層，原始權重記為 W_pre，形狀為 `[輸出維度, 輸入維度]`。

採用 `r = 16`、`alpha = 32`、`rho = 0.2`，因此 `c = alpha / r = 2`。

1. 將原始 checkpoint 中的 BF16 權重精確轉成 FP32，得到 W_pre32。這一步不恢復 checkpoint 已經丟失的精度。
2. 對 W_pre32 做 SVD，取前 r 個成分重建矩陣，按其元素絕對值建立 spectral mask。
3. 按 W_pre32 的元素絕對值建立 Euclidean mask。兩個 mask 都取 `rho` 分位數作閾值，使用 `<=`；重複值可能讓實際保留比例略高於 rho。
4. 將兩個布林 mask 取聯集，保留原始 W_pre32 對應位置的值，得到 W_geo。
5. 對 W_geo 做第二次 SVD，取前 r 個成分，設定：

   ```text
   Sigma_sqrt = diag(sqrt(singular_values_geo[:r]))
   B0 = U_geo[:, :r] @ Sigma_sqrt
   A0 = Sigma_sqrt @ Vh_geo[:r, :]
   ```

   B0 的形狀為 `[輸出維度, r]`；A0 的形狀為 `[r, 輸入維度]`。B0 @ A0 是 W_geo 的 rank-r 近似。
6. 用 FP32 計算：

   ```text
   F32 = W_pre32 - c * (B0 @ A0)
   ```

7. 凍結 F；讓 A/B 從 A0/B0 開始訓練。本機保存 FP32 F；GPU 訓練前將 F 轉 BF16，A/B 仍保留 FP32。

該層的 forward 為：

```text
output = input @ F.T + c * ((input @ A.T) @ B.T) + 原有 bias
```

**F 只在初始化時計算一次。** 之後更新 A/B 時不能重新計算 F，否則會抵消學到的改動。

## 4. 舍入與初始化比較

實數算術下，初始 `F + c * (B0 @ A0) = W_pre`。FP32 計算及轉 BF16 會引入舍入；拆成兩條 forward 分支也可能改變運算順序。因此不要求初始化 logits 逐 bit 相等。

初始化比較使用相同 dtype、相同輸入與推理模式的原始模型。記錄 logits 的最大絕對差與相對誤差；若誤差異常，再定位到各層。FP32 本機檢查與 BF16 GPU 檢查分別記錄，不能將兩者的誤差直接當作同一項指標。

模型切到 `eval()` 並固定輸入，避免 dropout 或採樣隨機性干擾比較。訓練啟動後再確認只有 A/B 收到梯度及更新，凍結權重與 bias 不變。

## 5. 目標層與目前狀態

基底模型：`Qwen/Qwen2.5-1.5B-Instruct`。

固定 revision：`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。

本次 all-linear 範圍是 28 個 Transformer block，每個 block 的：

```text
self_attn.q_proj
self_attn.k_proj
self_attn.v_proj
self_attn.o_proj
mlp.gate_proj
mlp.up_proj
mlp.down_proj
```

合計 **196 個線性層**，排除 `lm_head`；原有 bias 凍結。程式需實際列出並斷言目標名稱與數量。

目前已完成：

- 第 0 層 Q 的 FP64 參照初始化與單步更新檢查。
- 單層替換的完整模型 forward / 儲存載入檢查。
- **全部 196 層的 CPU FP32 初始化、固定輸入的完整模型 logits 比較，以及初始化因子 checkpoint 的儲存／重新載入。**

全層實測記錄於 [geora_full_model_check.ipynb](geora_full_model_check.ipynb)，原始測量保存於 [checks.json](reports/cpu_initialization/checks.json)：

| 測量 | 本機結果 |
|---|---:|
| 可訓練 A/B 參數數量 | 18,464,768 |
| 196 層初始化耗時 | 130.0 秒 |
| 各層有效權重的最大相對誤差 | 1.19e-8 |
| 完整模型 logits 相對誤差 | 2.61e-5 |
| 完整模型 logits 最大絕對差 | 0.002244 |
| 保存／重新載入後 logits 最大絕對差 | 0 |

logits 比較使用固定的 42-token 輸入，不代表任務評估。全模型檢查只確認 requires_grad 清單中只有 A/B；本次沒有 optimizer step。GeoRA GPU BF16、全模型更新後重載及 GRPO 框架的 reference 行為仍待驗證。原始模型 GPU BF16 forward 已由使用者回傳成功輸出，詳見 `CHECKS.md`。

## 6. 下一步清單

1. 在 GPU 上重建已保存的初始化，僅將凍結部分轉 BF16；A/B 保留 FP32。
2. 用相同 BF16 forward 設定比較原始模型與初始化模型，重新量測誤差。
3. 做一次 A/B 更新，檢查梯度、AdamW 狀態 dtype、A/B 的實際變化，以及凍結權重與 bias 不變；再確認更新後 checkpoint 重載一致。
4. 接入 GRPO 訓練框架，驗證 reference 確實代表原始模型，跑 GSM8K 短程測試。
5. 固定同一組訓練及評估設定，再比較 LoRA 與 GeoRA。

## 7. 儲存、載入與 reference 模型

輕量 checkpoint 至少保存：

- 原始模型名稱與固定 revision。
- 初始化／運行 dtype、r、alpha、rho、目標模組名稱，以及必要的軟體版本。
- 每層 FP32 初始 A0/B0，以及當前 FP32 A/B。

載入時，先從固定版本的原始權重重建 FP32 殘差：

```text
F32 = W_pre32 - c * (B0 @ A0)
```

然後依運行模式保留 FP32 F 或轉成 BF16 F，並載入當前 A/B。不必重新跑 SVD；A0/B0 就是重建 F 的依據。在相同環境及 dtype 下，檢查儲存前後有效權重與 logits 是否一致。

本次 F 在 CPU FP32 重建，再搬到 GPU。不同裝置或軟體版本的矩陣乘法可能有不同舍入，跨環境載入要重新量測誤差，不能直接沿用本機逐 bit 相同的結論。

只把訓練後的 A/B 加到原始 W_pre，會重複加入初始那一部分。相對原始模型的實際更新是：

```text
delta_W = c * (B @ A - B0 @ A0)
```

GRPO 的 reference 必須代表指定的原始策略。**直接停用 GeoRA adapter 得到的是 F，而非 W_pre**，因此不能把 `disable_adapter()` 默認當成正確 reference。接入訓練框架時，需使用原始模型或明確重建原始權重的 reference 路徑，並驗證其輸出。

## 8. 下一階段的執行環境

2026-10-08：使用者確認 Setonix GPU 節點可直接使用。下一階段選擇 **Setonix GPU**，使用 ROCm 版 PyTorch。

先用一個邏輯 GPU（MI250X 的一個 GCD，64 GB 顯存）完成小批量、短輸入的 BF16 forward 與 FP32 A/B 單步更新檢查，再接短 GRPO。這是起步檢查的資源配置，不代表論文完整 batch 或多卡訓練配置。[Pawsey 官方規格](https://pawsey.atlassian.net/wiki/spaces/US/pages/51929028/Setonix+General+Information)

ROCm 版 PyTorch 沿用 `torch.cuda` 與 `device='cuda'` 的 API 名稱；這些名稱在 Setonix 上指向 AMD GPU。需使用對應 ROCm 的安裝或容器，不能直接複製 Mac 的 `.venv`。[PyTorch HIP 說明](https://docs.pytorch.org/docs/2.14/notes/hip.html)

使用者已回傳 MI250X 的 BF16 矩陣乘法和原始模型 forward 成功結果；實際 PyTorch 為 `2.7.1a0+gite2d141d`、HIP 為 `6.3.42134-a9a80e791`。GeoRA 的混合精度、梯度及更新後重載由 `CHECKS.md` 的作業驗證。A100 保留為具體框架相容問題的備選；本機繼續用於 notebook 與矩陣分析。
