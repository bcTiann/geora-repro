"""Measure cached sampling versus the validated training scorer on tiny Qwen.

This is a numerical diagnostic. Passing mechanical checks does not mean the
likelihood gate passed, and synthetic updates are not GSM8K/GRPO training.
"""

import argparse
import json
from pathlib import Path
import time

import torch
import transformers

from cached_rollout import sample_completions_cached
from check_grpo_rollout_cpu import tiny_models
from geora_check_utils import CheckReport
from grpo_rollout import likelihood_alignment_metrics, score_completion_logps


LOGP_LIMIT = 2e-5
RATIO_LIMIT = 2e-5
SEQUENCE_LIMIT = 1e-3


def likelihood_measurement(model, rollout, device, dtype):
    started = time.perf_counter()
    scored = score_completion_logps(
        model, rollout, device, dtype, gradient_enabled=False,
    )
    metrics = likelihood_alignment_metrics(
        rollout["behavior_logps"], scored, rollout["completion_mask"],
    )
    metrics["scoring_seconds"] = time.perf_counter() - started
    metrics["gate_passed"] = (
        metrics["finite"] and metrics["max_absolute_delta_logp"] <= LOGP_LIMIT
        and metrics["max_absolute_ratio_minus_one"] <= RATIO_LIMIT
        and max(map(abs, metrics["sequence_delta_logp"])) <= SEQUENCE_LIMIT
    )
    return metrics


def measure_case(model, prompts, pad_id, eos_ids, new_tokens, device, dtype,
                 report, label, *, check_replay=False, check_backward=False):
    generator = torch.Generator(device=device).manual_seed(29)
    initial_private_state = generator.get_state().clone()
    global_cpu_state = torch.get_rng_state().clone()
    global_device_state = torch.cuda.get_rng_state().clone() if device == "cuda" else None
    started = time.perf_counter()
    rollout = sample_completions_cached(
        model, prompts, pad_id, eos_ids, new_tokens, device, dtype,
        generator=generator,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - started
    advanced_private_state = generator.get_state().clone()
    report.require(label + "_private_rng_isolated_and_advanced",
                   torch.equal(global_cpu_state, torch.get_rng_state())
                   and (global_device_state is None
                        or torch.equal(global_device_state, torch.cuda.get_rng_state()))
                   and not torch.equal(initial_private_state, advanced_private_state))
    mask = rollout["completion_mask"]
    lengths = rollout["completion_lengths"]
    # Derive expected lengths from token IDs independently of the stored mask.
    expected_lengths = []
    expected_ended = []
    for tokens in rollout["completion_ids"][:, :rollout["generated_steps"]].tolist():
        first_eos = next((index for index, token in enumerate(tokens) if token in eos_ids), None)
        expected_lengths.append(len(tokens) if first_eos is None else first_eos + 1)
        expected_ended.append(first_eos is not None)
    independent_lengths = torch.tensor(expected_lengths, device=device)
    independent_ended = torch.tensor(expected_ended, dtype=torch.bool, device=device)
    expected_mask = torch.arange(new_tokens, device=device)[None, :] < independent_lengths[:, None]
    report.require(label + "_completion_layout_and_first_eos",
                   torch.equal(mask, expected_mask)
                   and torch.equal(lengths, independent_lengths)
                   and torch.equal(rollout["ended_with_eos"], independent_ended)
                   and torch.equal(rollout["attention_mask"][:, rollout["prompt_width"]:].bool(), mask)
                   and (rollout["behavior_logps"][~mask] == 0).all().item()
                   and not rollout["input_ids"].is_inference(),
                   completion_lengths=lengths.tolist(), generated_steps=rollout["generated_steps"])
    if check_replay:
        generator.set_state(initial_private_state)
        replay = sample_completions_cached(
            model, prompts, pad_id, eos_ids, new_tokens, device, dtype,
            generator=generator,
        )
        report.require(label + "_private_rng_exact_replay",
                       all(torch.equal(rollout[key], replay[key]) for key in (
                           "input_ids", "completion_mask", "behavior_logps"))
                       and torch.equal(advanced_private_state, generator.get_state()))
    metrics = likelihood_measurement(model, rollout, device, dtype)
    report.require(label + "_sampling_and_scoring_finite", metrics["finite"],
                   max_absolute_delta_logp=metrics["max_absolute_delta_logp"],
                   max_absolute_ratio_minus_one=metrics["max_absolute_ratio_minus_one"],
                   likelihood_gate_passed=metrics["gate_passed"])
    record = dict(label=label, dtype=str(dtype), path=rollout["path"],
                  generation_seconds=generation_seconds, likelihood=metrics)
    report.data["cases"].append(record)
    report.write()
    if check_backward:
        current = score_completion_logps(model, rollout, device, dtype, gradient_enabled=True)
        loss = -(current * mask).sum() / mask.sum()
        loss.backward()
        parameters = [value for value in model.parameters() if value.requires_grad]
        report.require(label + "_padded_scoring_backward",
                       loss.requires_grad and torch.isfinite(loss).item()
                       and all(value.grad is not None and value.grad.dtype == torch.float32
                               and torch.isfinite(value.grad).all().item() for value in parameters)
                       and any(value.grad.abs().sum().item() > 0 for value in parameters))
        optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return rollout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    torch.set_num_threads(2)
    report = CheckReport(arguments.output_dir, "cached_rollout_cpu.json")
    report.data.update(
        stage="tiny_cached_rollout_likelihood_diagnostic", cases=[],
        scope="No real model, task reward, GRPO step or GPU allocation",
        torch_version=torch.__version__, transformers_version=transformers.__version__,
        gates=dict(max_absolute_delta_logp=LOGP_LIMIT,
                   max_absolute_ratio_minus_one=RATIO_LIMIT,
                   max_absolute_sequence_delta_logp=SEQUENCE_LIMIT),
        production_sampler_changed=False,
    )
    started = time.perf_counter()
    try:
        prompts = [[3, 4, 5]] * 4 + [[6, 7, 8, 9, 10]] * 4
        for dtype in (torch.float32, torch.bfloat16):
            prefix = "fp32" if dtype == torch.float32 else "bf16"
            model, reference = tiny_models(dtype)
            native = measure_case(reference, prompts, 0, [2, 3, 4, 5, 6], 8, "cpu", dtype,
                                  report, prefix + "_native", check_replay=True)
            initial = measure_case(model, prompts, 0, [2, 3, 4, 5, 6], 8, "cpu", dtype,
                                   report, prefix + "_geora_initial", check_backward=True)
            report.require(prefix + "_initial_cached_policy_matches_native",
                           torch.equal(native["input_ids"], initial["input_ids"])
                           and torch.equal(native["behavior_logps"], initial["behavior_logps"]))
            report.require(prefix + "_mixed_eos_and_truncated_rows",
                           initial["ended_with_eos"].any().item()
                           and (~initial["ended_with_eos"]).any().item())
            measure_case(model, prompts, 0, [2, 3, 4, 5, 6], 8, "cpu", dtype,
                         report, prefix + "_geora_after_synthetic_update")
            # Every vocabulary ID is EOS: stop immediately, including a sampled
            # pad ID. This checks position masks when pad is itself an EOS ID.
            immediate = measure_case(model, prompts, 2, list(range(37)), 8, "cpu", dtype,
                                     report, prefix + "_pad_equals_eos")
            report.require(prefix + "_immediate_eos_is_one_scored_token",
                           immediate["generated_steps"] == 1
                           and (immediate["completion_lengths"] == 1).all().item())
        report.data["likelihood_gate_passed"] = all(
            case["likelihood"]["gate_passed"] for case in report.data["cases"])
        report.data["decision"] = ("candidate_requires_full_model_validation"
                                   if report.data["likelihood_gate_passed"]
                                   else "keep_validated_fixed_width_training_sampler")
        report.data["mechanical_checks_passed"] = True
        report.data["elapsed_seconds"] = time.perf_counter() - started
        report.finish()
    except Exception as error:
        report.data.update(status="failed", error=str(error),
                           elapsed_seconds=time.perf_counter() - started)
        report.write()
        raise
    print("Mechanical checks passed; likelihood gate:", report.data["likelihood_gate_passed"])
    print("Production training sampler unchanged.")


if __name__ == "__main__":
    main()
