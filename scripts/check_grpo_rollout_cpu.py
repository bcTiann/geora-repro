"""Validate aligned rollout/scoring and padded backward on a tiny CPU Qwen.

Synthetic tiny-model token samples exercise EOS, masks and derivatives. They
are numerical checks, not GSM8K answers, rewards, or a real GRPO update.
"""

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIRECTORY))

from geora_layers import GeoRALinear, iter_target_linears
from geora_check_utils import CheckReport, forward_context, move_with_precision
from grpo_rollout import (
    likelihood_alignment_metrics,
    position_ids_from_mask,
    sample_completions,
    score_completion_logps,
)


def tiny_models(dtype):
    torch.manual_seed(18)
    base = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=37,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_dropout=0.0,
        eos_token_id=2,
        pad_token_id=0,
    ))
    base.config._attn_implementation = "eager"
    base.requires_grad_(False)
    reference = copy.deepcopy(base)
    for name, layer in list(iter_target_linears(base)):
        parent_name, leaf_name = name.rsplit(".", 1)
        initial_A = torch.randn(2, layer.in_features) * 0.02
        initial_B = torch.randn(layer.out_features, 2) * 0.02
        replacement = GeoRALinear(
            layer, initial_A, initial_B, 2.0, forward_mode="difference",
        )
        setattr(base.get_submodule(parent_name), leaf_name, replacement)
    move_with_precision(reference, "cpu", dtype)
    move_with_precision(base, "cpu", dtype)
    base.train()
    reference.eval()
    return base, reference


def run_dtype(dtype, report):
    prefix = "fp32" if dtype == torch.float32 else "bf16"
    model, reference = tiny_models(dtype)
    prompt_ids = [[3, 4, 5]] * 4 + [[6, 7, 8, 9, 10]] * 4
    # More EOS IDs in this synthetic tiny vocabulary reliably exercise variable
    # completion lengths. The real model uses its actual EOS configuration.
    eos_ids = [2, 3, 4, 5, 6]
    rollout = sample_completions(
        model, prompt_ids, 0, eos_ids, 8, "cpu", dtype,
        generator=torch.Generator().manual_seed(29),
    )
    report.require(prefix + "_fixed_batch_and_width", rollout["input_ids"].shape == (8, 13)
                   and rollout["completion_mask"].shape == (8, 8),
                   prompt_lengths=rollout["prompt_lengths"],
                   completion_lengths=rollout["completion_lengths"].tolist())
    report.require(prefix + "_eos_variable_lengths_and_truncation",
                   rollout["completion_lengths"].unique().numel() > 1
                   and rollout["ended_with_eos"].any().item()
                   and (~rollout["ended_with_eos"]).any().item())
    valid_mask = rollout["completion_mask"]
    valid_lengths = rollout["completion_lengths"]
    expected_mask = torch.arange(8)[None, :] < valid_lengths[:, None]
    eos_semantics = True
    for row in range(8):
        if rollout["ended_with_eos"][row]:
            last_token = rollout["completion_ids"][row, valid_lengths[row] - 1].item()
            eos_semantics &= last_token in eos_ids
    report.require(prefix + "_mask_includes_first_eos_and_excludes_later_padding",
                   torch.equal(valid_mask, expected_mask) and eos_semantics
                   and torch.equal(rollout["attention_mask"][:, 5:].bool(), valid_mask)
                   and (rollout["behavior_logps"][~valid_mask] == 0).all().item())
    old_logps = score_completion_logps(
        model, rollout, "cpu", dtype, gradient_enabled=False,
    )
    metrics = likelihood_alignment_metrics(rollout["behavior_logps"], old_logps, valid_mask)
    report.require(prefix + "_actual_sampling_and_old_scoring_aligned",
                   metrics["finite"] and metrics["max_absolute_delta_logp"] <= 2e-5
                   and metrics["max_absolute_ratio_minus_one"] <= 2e-5, **metrics)
    current_logps = score_completion_logps(
        model, rollout, "cpu", dtype, gradient_enabled=True,
    )
    report.require(prefix + "_unchanged_current_old_exact",
                   torch.equal(current_logps.detach(), old_logps)
                   and current_logps.requires_grad and not old_logps.requires_grad)
    reference_logps = score_completion_logps(
        reference, rollout, "cpu", dtype, gradient_enabled=False,
    )
    report.require(prefix + "_initial_policy_reference_exact",
                   torch.equal(old_logps, reference_logps))
    # An independently selected token/head position checks the causal shift.
    with torch.no_grad(), forward_context("cpu", dtype):
        hidden = model.model(
            input_ids=rollout["input_ids"],
            attention_mask=rollout["attention_mask"],
            position_ids=position_ids_from_mask(rollout["attention_mask"]),
            use_cache=False,
        ).last_hidden_state
        first_logits = model.lm_head(hidden[:, 4:5, :]).squeeze(1).float()
        first_selected = first_logits.log_softmax(-1).gather(
            1, rollout["completion_ids"][:, :1],
        ).squeeze(1)
    report.require(prefix + "_first_completion_uses_last_prompt_position",
                   torch.equal(first_selected, old_logps[:, 0]))
    frozen_snapshot = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters() if not parameter.requires_grad
    }
    initial_snapshot = {
        name: buffer.detach().clone()
        for name, buffer in model.named_buffers() if name.endswith((".A0", ".B0"))
    }
    reference_snapshot = {
        name: parameter.detach().clone() for name, parameter in reference.named_parameters()
    }
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    # This synthetic differentiability check supplies no task rewards and is
    # deliberately separate from the hand-tested GRPO loss and real GPU step.
    loss = -(current_logps * valid_mask).sum() / valid_mask.sum()
    loss.backward()
    report.require(prefix + "_full_eight_row_left_padded_backward",
                   all(parameter.grad is not None and parameter.grad.dtype == torch.float32
                       and torch.isfinite(parameter.grad).all().item() for parameter in trainable)
                   and any(parameter.grad.abs().sum().item() > 0 for parameter in trainable),
                   loss=loss.item())
    optimizer = torch.optim.AdamW(trainable, lr=1e-6, weight_decay=0.0)
    optimizer.step()
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    report.require(prefix + "_frozen_initial_and_reference_unchanged",
                   all(torch.equal(parameters[name], value) for name, value in frozen_snapshot.items())
                   and all(torch.equal(buffers[name], value) for name, value in initial_snapshot.items())
                   and all(torch.equal(dict(reference.named_parameters())[name], value)
                           for name, value in reference_snapshot.items())
                   and all(parameter.grad is None for parameter in reference.parameters()))
    updated_rollout = sample_completions(
        model, prompt_ids, 0, eos_ids, 8, "cpu", dtype,
        generator=torch.Generator().manual_seed(30),
    )
    updated_logps = score_completion_logps(
        model, updated_rollout, "cpu", dtype, gradient_enabled=False,
    )
    updated_metrics = likelihood_alignment_metrics(
        updated_rollout["behavior_logps"], updated_logps, updated_rollout["completion_mask"],
    )
    report.require(prefix + "_post_update_sampling_and_scoring_aligned",
                   updated_metrics["finite"] and updated_metrics["max_absolute_delta_logp"] <= 2e-5
                   and updated_metrics["max_absolute_ratio_minus_one"] <= 2e-5, **updated_metrics)
    report.require(prefix + "_train_and_reference_modes_preserved", model.training and not reference.training)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    torch.set_num_threads(2)
    report = CheckReport(arguments.output_dir, "rollout_preflight.json")
    started = time.perf_counter()
    report.data.update(
        stage="tiny_qwen_aligned_rollout_and_padded_backward",
        scope="Synthetic numerical checks; no task rewards or real GRPO training",
        torch_version=torch.__version__,
        likelihood_absolute_logp_limit=2e-5,
        likelihood_ratio_minus_one_limit=2e-5,
    )
    try:
        for dtype in (torch.float32, torch.bfloat16):
            run_dtype(dtype, report)
        report.data["elapsed_seconds"] = time.perf_counter() - started
        report.finish()
    except Exception as error:
        report.data.update(status="failed", error=str(error),
                           elapsed_seconds=time.perf_counter() - started)
        report.write()
        raise
    print("Tiny-Qwen aligned rollout and padded backward passed.")
    print("Report:", arguments.output_dir / "rollout_preflight.json")


if __name__ == "__main__":
    main()
