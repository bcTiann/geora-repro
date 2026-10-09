# GeoRA 復現：精度與驗證約定

更新：2026-10-10。這份文件說明目前實作的精度約定；實際執行結果集中在 [EXPERIMENT_RECORD.md](EXPERIMENT_RECORD.md)。

**下表是目前 GPU 檢查的預設配置，尚未通過 1.5B 初始化 logits 門檻，不能當作已驗證的訓練方案。** FP32 投影與 FP32 投影＋attention 分數已做完整模型診斷，仍未通過 2% logits 門檻，尚未取代預設。詳見實驗記錄 S6。`difference` 模式已通過初始化／單步更新／重載，精度另見第 9 節；這不把其多步訓練視為已驗證。

## 1. 採用的精度

| 部分 | 本機 CPU 正確性檢查 | 目前 GPU 檢查預設 |
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

7. 凍結 F；讓 A/B 從 A0/B0 開始。本機運行內存中 F 為 FP32；目前 GPU 檢查將 F 轉 BF16，A/B 仍保留 FP32。緊湊 checkpoint 不保存 F，而保存重建它所需的 A0/B0，見第 7 節。

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

實際通過／失敗狀態及每項精度見 [EXPERIMENT_RECORD.md](EXPERIMENT_RECORD.md)。目標名稱、數量和 dtype 應由執行時斷言核對；預期配置不能代替實際結果。

## 6. 驗證順序

完整模型初始化輸出 → A/B 單步更新與重載 → GRPO reference 與訓練 → LoRA／GeoRA 任務對照。初始化門檻失敗時，在 backward 前停止，先用完整模型驗證精度修正。

## 7. 儲存、載入與 reference 模型

輕量 checkpoint 至少保存：

- 原始模型名稱與固定 revision。
- 初始化／運行 dtype、r、alpha、rho、目標模組名稱，以及必要的軟體版本。
- 每層 FP32 初始 A0/B0，以及當前 FP32 A/B。

`residual` 模式載入時，先從固定版本的原始權重重建 FP32 殘差：

```text
F32 = W_pre32 - c * (B0 @ A0)
```

然後依運行模式保留 FP32 F 或轉成 BF16 F，並載入當前 A/B。不必重新跑 SVD；A0/B0 就是重建 F 的依據。在相同環境及 dtype 下，檢查儲存前後有效權重與 logits 是否一致。

本次 F 在 CPU FP32 重建，再搬到 GPU。不同裝置或軟體版本的矩陣乘法可能有不同舍入，跨環境載入要重新量測誤差，不能直接沿用本機逐 bit 相同的結論。

只把訓練後的 A/B 加到原始 W_pre，會重複加入初始那一部分。相對原始模型的實際更新是：

```text
delta_W = c * (B @ A - B0 @ A0)
```

GRPO 的 reference 必須代表指定的原始策略。**residual 模式直接停用 GeoRA adapter 得到的是 F，而非 W_pre**，因此不能把 `disable_adapter()` 默認當成正確 reference。接入訓練框架時，需使用原始模型或明確重建原始權重的 reference 路徑，並驗證其輸出。

## 8. 執行環境與混合精度邊界

本機使用獨立 uv 環境，Setonix 使用 ROCm PyTorch 容器與容器內 venv；具體版本、路徑和已用資源見 [EXPERIMENT_RECORD.md](EXPERIMENT_RECORD.md)，操作方法見 [SETONIX_GUIDE.md](SETONIX_GUIDE.md)。

BF16 forward 是混合精度路徑，不代表每一步都是 BF16。RMSNorm 的統計、RoPE 頻率與三角函數、eager attention 的 softmax 可在 FP32 計算；但先前的 BF16 QK 分數精度不會因 FP32 softmax 恢復。運算精度與參數／checkpoint 保存精度應分別驗證。

誤差統計也獨立於模型精度：GPU logits 可先轉 FP32 做範數，再用 CPU FP64 算 KL/TV。這不會改變產生 logits 時的模型計算精度。


## 9. 已驗證的 difference 模式

`GeoRALinear`／`load_geora_state` 支援明確的 `forward_mode=difference`，計算 `W_pre x+c[B(Ax)-B0(A0x)]`。預設 residual 未改。原始 frozen W_pre 和 native attention 保留 BF16；A/B/A0/B0 存 FP32；兩條低秩分支與相減禁用 autocast、計算 FP32，校正轉回原始輸出的 BF16 後相加。

初始分支對 x 保留梯度。FP32 A/B gradients 和 AdamW moments 已在 1.5B 單步實際檢查。公式／梯度、三個初始化輸入、凍結參數及初始／更新後保存重載結果見 [EXPERIMENT_RECORD.md 的 L8/S7](EXPERIMENT_RECORD.md)。更新後的大策略變化仍待處理，不能從這輪推論 GRPO 穩定性。

載入 difference 模式時保留原始 W_pre，不重建 F；manifest 記錄 `forward_mode` 與 `runtime_precision`，`precision_policy` 保留初始化來源的約定。舊無 mode checkpoint 預設 residual；有標記的 checkpoint 不允許衝突覆蓋。merge 仍為 `W_pre+c(BA-B0A0)`，不是普通 LoRA merge。外部 trainer／rollout engine 的接入尚未驗證。
