"""Independent CPU tests for immutable-boundary cached checkpointing.

The native uncheckpointed oracle is copied from the previously accepted scorer
with only its validation cap raised in this temporary test directory. Real
native sampling checks additionally use the accepted probability recorder.
"""

import argparse
import copy
import hashlib
import json
import os
import types
from pathlib import Path
import sys
import time

import torch

REPOSITORY = Path(os.environ.get("GEORA_REPOSITORY", Path.cwd())).resolve()
if not (REPOSITORY / "scripts/check_cached_score_cpu.py").exists():
    REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))
sys.path.insert(0, str(REPOSITORY))
from cached_block_score import score_completion_logps_block_checkpointed
from cached_training_rollout import sample_completions_cached_training
from check_cached_score_cpu import tiny_adapter
from grpo_rollout import likelihood_alignment_metrics


source_path = REPOSITORY / "scripts/cached_score.py"
source = source_path.read_text()
oracle = types.ModuleType("uncheckpointed_cached_oracle")
exec(compile(source.replace("capacity <= 256", "capacity <= 4096"),
             str(source_path), "exec"), oracle.__dict__)


def require(checks, name, condition, **values):
    checks.append(dict(name=name, passed=bool(condition), **values))
    print("PASS:" if condition else "FAIL:", name, values, flush=True)
    if not condition:
        raise RuntimeError(name)


def fixed_rollout(*, steps, dtype, unequal=False):
    prompt_width = 5
    capacity = steps + 3
    batch = 2 if unequal else 1
    generator = torch.Generator().manual_seed(908)
    ids = torch.randint(7, 37, (batch, prompt_width + capacity), generator=generator)
    attention = torch.ones_like(ids)
    if unequal:
        ids[0, :2] = 0
        attention[0, :2] = 0
    lengths = torch.tensor([steps, steps // 2] if unequal else [steps])
    completion_mask = torch.arange(capacity)[None, :] < lengths[:, None]
    ids[:, prompt_width:] = torch.where(completion_mask, ids[:, prompt_width:], 0)
    for row, length in enumerate(lengths.tolist()):
        ids[row, prompt_width + length - 1] = 2
    attention[:, prompt_width:] = completion_mask.long()
    return dict(input_ids=ids, attention_mask=attention, completion_mask=completion_mask,
                prompt_width=prompt_width, max_new_tokens=capacity, generated_steps=steps,
                temperature=1.0, use_cache=True)


def compare_case(checks, kind, dtype, steps, block_size, *, unequal=False, last_only=False):
    native = tiny_adapter(kind, dtype)
    # Exercise nonzero LoRA A gradients and nontrivial GeoRA corrections.
    with torch.no_grad():
        for name, parameter in native.named_parameters():
            if parameter.requires_grad:
                parameter.add_(torch.randn_like(parameter) * 0.003)
    blocked = copy.deepcopy(native)
    rollout = fixed_rollout(steps=steps, dtype=dtype, unequal=unequal)
    native_trace, blocked_trace = [], []
    expected = oracle.score_completion_logps_cached(
        native, rollout, "cpu", dtype, gradient_enabled=True, trace_callback=native_trace.append)
    statistics = {}
    actual = score_completion_logps_block_checkpointed(
        blocked, rollout, "cpu", dtype, gradient_enabled=True, block_size=block_size,
        trace_callback=blocked_trace.append, checkpoint_statistics=statistics)
    label = f"{kind}_{str(dtype).split('.')[-1]}_{steps}_b{block_size}"
    require(checks, label + "_native_forward_logps_exact", torch.equal(actual, expected))
    require(checks, label + "_native_inputs_positions_masks_cache_lengths_exact",
            native_trace == blocked_trace)
    mask = rollout["completion_mask"]
    if last_only:
        lengths = mask.sum(-1)
        rows = torch.arange(mask.shape[0])
        expected_loss = -expected[rows, lengths - 1].mean()
        actual_loss = -actual[rows, lengths - 1].mean()
    else:
        weights = torch.linspace(0.7, 1.3, steps + 3)[None, :]
        expected_loss = -(expected * mask * weights).sum() / mask.sum()
        actual_loss = -(actual * mask * weights).sum() / mask.sum()
    expected_loss.backward()
    actual_loss.backward()
    expected_params = {n: p for n, p in native.named_parameters() if p.requires_grad}
    actual_params = {n: p for n, p in blocked.named_parameters() if p.requires_grad}
    differences = []
    passed = True
    for name, parameter in expected_params.items():
        gradient = actual_params[name].grad
        if gradient is None or parameter.grad is None:
            passed = False
            continue
        difference = (gradient - parameter.grad).abs().max().item()
        differences.append(difference)
        passed &= bool(torch.allclose(gradient, parameter.grad, rtol=2e-5, atol=2e-6))
    require(checks, label + "_adapter_gradients_match_without_detached_history", passed,
            max_absolute_gradient_difference=max(differences), last_token_only=last_only)
    require(checks, label + "_every_checkpoint_recomputed_with_matching_autocast",
            statistics["recompute_blocks"] == statistics["checkpoint_blocks"]
            and statistics["autocast_states_match"], **statistics)
    require(checks, label + "_trace_observer_not_replayed_during_backward",
            len(blocked_trace) == steps)
    with torch.no_grad():
        inference = score_completion_logps_block_checkpointed(
            blocked, rollout, "cpu", dtype, gradient_enabled=False, block_size=block_size)
    require(checks, label + "_inference_values_exact_no_graph",
            torch.equal(inference, expected.detach()) and not inference.requires_grad)


def sampled_case(checks):
    model = tiny_adapter("geora", torch.bfloat16)
    rollout = sample_completions_cached_training(
        model, [[3, 4, 5]] * 2 + [[6, 7, 8, 9, 10]] * 2,
        0, [2, 3, 4, 5, 6], 32, "cpu", torch.bfloat16,
        generator=torch.Generator().manual_seed(29))
    actual = score_completion_logps_block_checkpointed(
        model, rollout, "cpu", torch.bfloat16, gradient_enabled=True, block_size=7)
    metrics = likelihood_alignment_metrics(rollout["behavior_logps"], actual, rollout["completion_mask"])
    require(checks, "real_sampled_BF16_native_logps_exact", metrics["exact"], **metrics)
    require(checks, "real_sampled_variable_EOS_lengths_present",
            len(set(rollout["completion_lengths"].tolist())) > 1,
            lengths=rollout["completion_lengths"].tolist())
    (-actual[rollout["completion_mask"]].mean()).backward()
    require(checks, "real_sampled_gradients_finite",
            all(p.grad is not None and torch.isfinite(p.grad).all().item()
                for p in model.parameters() if p.requires_grad))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("cpu_report.json"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    checks = []
    started = time.perf_counter()
    report = dict(stage="functional_cached_block_checkpoint_cpu", status="running", checks=checks,
                  accepted_oracle_source=str(source_path),
                  accepted_oracle_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  oracle_change="Only validation capacity cap256 raised4096; forward/backward arithmetic unchanged",
                  scope="Tiny CPU native Qwen models; not full-model GPU capacity or runtime")
    try:
        sampled_case(checks)
        compare_case(checks, "lora", torch.float32, 512, 64, last_only=True)
        compare_case(checks, "geora", torch.bfloat16, 300, 64, unequal=True)
        compare_case(checks, "geora", torch.float32, 65, 16, unequal=True, last_only=True)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
