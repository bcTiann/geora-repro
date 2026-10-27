# MATH-500 pilot results

We evaluated Qwen2.5-1.5B-Instruct and three adapter variants on all 500 questions of [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500/tree/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be), using the same prompt, greedy decoding, a 4,096-token response limit, and Math-Verify 0.8.0 for the primary score. LoRA and GeoRA checkpoints were taken after 25 training updates.

| Variant | Correct | Accuracy |
| --- | ---: | ---: |
| Base model | 270 / 500 | 54.0% |
| Untrained GeoRA | 279 / 500 | 55.8% |
| LoRA, 25 updates | 269 / 500 | 53.8% |
| GeoRA, 25 updates | 284 / 500 | 56.8% |

GeoRA scored 3.0 percentage points above LoRA in this pilot. The paired question-level 95% bootstrap interval for the difference was −0.2 to +6.2 points; the exact McNemar test gave p = 0.096. GeoRA after training was 1.0 point above untrained GeoRA. These observations use one training seed and do not establish a reliable training advantage or reproduce the paper's full experiment. The bootstrap interval does not measure variation across training seeds.

The later audit re-counted saved per-question results and checked the paired exact test; it did not regenerate responses or rerun Math-Verify. The bootstrap interval is from the original frozen analysis. Depending on the variant, 22–28 responses reached the length limit. Math-Verify is an automated evaluator and can make extraction or equivalence errors. The prompt, scoring and training budget are independent reproduction choices, so these scores are not directly comparable to the paper's table.
