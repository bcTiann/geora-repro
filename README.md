# GeoRA: A Reproduction and GRPO Training Study

An independent implementation and empirical study of [GeoRA](https://arxiv.org/abs/2601.09361), a parameter-efficient fine-tuning method. The project compares GeoRA with LoRA for mathematical reasoning and investigates reliable GPU training with TRL and vLLM.

## Pilot results

Qwen2.5-1.5B-Instruct was trained on GSM8K with GRPO. The table reports a **single-seed, 25-update pilot** and two initialization controls, evaluated on all 500 MATH-500 questions with greedy decoding and a 4,096-token generation limit. The primary score uses Math-Verify 0.8.0.

| Model | MATH-500 correct | Accuracy |
|---|---:|---:|
| Base | 270 / 500 | 54.0% |
| Untrained GeoRA | 279 / 500 | 55.8% |
| LoRA, 25 updates | 269 / 500 | 53.8% |
| GeoRA, 25 updates | 284 / 500 | 56.8% |

These are preliminary observations, not proof of a training advantage: untrained GeoRA already differs from the base model, and one training seed cannot establish robustness. The pilot was evaluated under a fixed protocol; it is not an exact reproduction of the paper's undisclosed training and evaluation settings.

See [pilot results and evaluation notes](RESULTS.md) for the comparison and its limitations.

## Engineering work

- Implemented GeoRA and a binary, exact-numeric GSM8K reward for GRPO.
- Integrated a pinned TRL development version with vLLM on AMD ROCm: one GPU GCD generates responses while another trains. Full policy synchronization includes GeoRA residual weights and buffers.
- Checked complete synchronized weights and compared token log-probabilities across the learner, merged Hugging Face model, and vLLM. The checks record numerical differences rather than assuming identical probabilities.
- Completed two real GPU updates per method with a process restart and checkpoint restore, checking model and training state. These are integration tests, not the 25-update pilot above.

## Code

- [Training entry point](vllm_modern/train_native.py) and [frozen training components](vllm_modern/frozen_training/)
- [Policy synchronization](vllm_modern/native_sync.py) and [three-way probability checks](vllm_modern/three_way_probability.py)
- [Proposed experiment configuration](vllm_modern/formal_config.json)

## Status

The planned 400-update LoRA/GeoRA comparison has **not started**. Final initialization, evaluation inputs, and compute budget still need to be bound before launch. The paper's full experimental results have not yet been reproduced.
