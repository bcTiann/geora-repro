"""One mechanical GRPO update on tiny random Qwen models, CPU only.

Artificial token IDs and reward labels test wiring, gradients and precision.
They are not real answers, GSM8K rewards or evidence of task improvement.
No pretrained model or dataset downloads are required.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIRECTORY))
sys.path.insert(0, str(PROJECT_DIRECTORY / "scripts"))

from geora_layers import GeoRALinear, export_adapter_state, install_geora, load_geora_state
from geora_check_utils import forward_context, move_with_precision
from grpo_math import causal_token_log_probs, completion_token_mask, group_advantages, grpo_loss


def run_tiny_update(dtype: torch.dtype) -> dict:
    torch.manual_seed(37)
    configuration = Qwen2Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
        eos_token_id=2,
        pad_token_id=0,
    )
    configuration._attn_implementation = "eager"
    reference = Qwen2ForCausalLM(configuration).float().eval()
    reference.requires_grad_(False)
    initialization_model = copy.deepcopy(reference)
    initialization_manifest = install_geora(initialization_model, rank=2, alpha=4, rho=0.2)
    initial_factors = export_adapter_state(initialization_model)
    policy = copy.deepcopy(reference)
    difference_manifest = dict(initialization_manifest, forward_mode="difference")
    load_geora_state(policy, initial_factors, difference_manifest)
    move_with_precision(reference, "cpu", dtype)
    move_with_precision(policy, "cpu", dtype)
    policy.train()

    # Two prompt lengths test left padding. Each question gets two artificial answers.
    input_ids = torch.tensor([[0, 5, 6, 7, 8, 2, 0], [0, 5, 6, 7, 9, 10, 2], [5, 6, 7, 8, 11, 2, 0], [5, 6, 7, 8, 12, 13, 2]])
    attention_mask = (input_ids != 0).long()
    completion_starts = torch.tensor([4, 4, 4, 4])
    token_mask = completion_token_mask(input_ids, attention_mask, completion_starts, [2])[:, 1:]
    rewards = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    advantages = group_advantages(rewards).flatten()
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "use_cache": False}

    trainable = {name: parameter for name, parameter in policy.named_parameters() if parameter.requires_grad}
    frozen_snapshots = {name: parameter.detach().clone() for name, parameter in policy.named_parameters() if not parameter.requires_grad}
    factor_buffer_snapshots = {name: value.detach().clone() for name, value in policy.named_buffers() if name.endswith((".A0", ".B0"))}
    trainable_snapshots = {name: parameter.detach().clone() for name, parameter in trainable.items()}
    assert len(trainable) == 14 and all(name.endswith((".A", ".B")) for name in trainable)
    assert all(module.forward_mode == "difference" for module in policy.modules() if isinstance(module, GeoRALinear))

    with torch.no_grad(), forward_context("cpu", dtype):
        old_logits = policy(**inputs).logits
        reference_logits = reference(**inputs).logits
    assert torch.equal(old_logits, reference_logits)
    old_log_probs = causal_token_log_probs(old_logits, input_ids).detach()
    reference_log_probs = causal_token_log_probs(reference_logits, input_ids).detach()
    with forward_context("cpu", dtype):
        current_logits = policy(**inputs).logits
        current_log_probs = causal_token_log_probs(current_logits, input_ids)
        terms = grpo_loss(current_log_probs, old_log_probs, reference_log_probs, advantages, token_mask)
    assert torch.equal(terms["ratios"][token_mask], torch.ones_like(terms["ratios"][token_mask]))
    assert terms["kl"].item() == 0.0

    optimizer = torch.optim.AdamW(trainable.values(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    terms["loss"].backward()
    gradients = [parameter.grad for parameter in trainable.values()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    gradient_norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), max_norm=1.0)
    assert gradient_norm.item() > 0.0 and torch.isfinite(gradient_norm)
    optimizer.step()

    changed_parameters = [name for name, parameter in trainable.items() if not torch.equal(parameter, trainable_snapshots[name])]
    assert any(name.endswith(".A") for name in changed_parameters)
    assert any(name.endswith(".B") for name in changed_parameters)
    assert all(torch.equal(parameter, frozen_snapshots[name]) for name, parameter in policy.named_parameters() if name in frozen_snapshots)
    assert all(torch.equal(value, factor_buffer_snapshots[name]) for name, value in policy.named_buffers() if name in factor_buffer_snapshots)
    assert all(parameter.grad is None for parameter in reference.parameters())
    with torch.no_grad(), forward_context("cpu", dtype):
        updated_logits = policy(**inputs).logits
        updated_terms = grpo_loss(causal_token_log_probs(updated_logits, input_ids), old_log_probs, reference_log_probs, advantages, token_mask)
    assert torch.isfinite(updated_logits).all() and torch.isfinite(updated_terms["loss"])
    return {
        "status": "passed",
        "frozen_dtype": str(dtype),
        "adapter_dtype": "torch.float32",
        "forward_mode": "difference",
        "reward_source": "artificial mechanical labels, not GSM8K",
        "rewards": rewards.tolist(),
        "advantages": advantages.tolist(),
        "completion_lengths": token_mask.sum(dim=1).tolist(),
        "initial_logits_exact": True,
        "initial_old_current_ratios_exactly_one": True,
        "initial_kl": terms["kl"].item(),
        "loss_before": terms["loss"].item(),
        "loss_after_same_answers": updated_terms["loss"].item(),
        "kl_after_same_answers": updated_terms["kl"].item(),
        "pre_clip_gradient_norm": gradient_norm.item(),
        "changed_trainable_tensor_count": len(changed_parameters),
        "frozen_weights_unchanged": True,
        "initial_factor_buffers_unchanged": True,
        "reference_has_no_gradients": True,
        "optimizer_steps": 1,
        "learning_rate": 1e-4,
        "learning_rate_scope": "CPU tiny mechanical test only; full-model smoke uses its own configuration",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    torch.set_num_threads(2)
    report = {"status": "running", "device": "cpu", "torch_version": torch.__version__, "models": {}}
    try:
        for dtype in (torch.float32, torch.bfloat16):
            report["models"][str(dtype)] = run_tiny_update(dtype)
            print(f"PASS: tiny difference GRPO mechanical update {dtype}", flush=True)
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        if arguments.output_dir is not None:
            arguments.output_dir.mkdir(parents=True, exist_ok=True)
            path = arguments.output_dir / "grpo_tiny_checks.json"
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(f"Report: {path}", flush=True)


if __name__ == "__main__":
    main()
