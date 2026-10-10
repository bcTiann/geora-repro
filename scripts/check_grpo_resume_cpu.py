"""Prove complete boundary resume using tiny GeoRA/LoRA CPU models.

Both FP32 and BF16 frozen-model paths use FP32 A/B and AdamW. Rewards are
artificial labels for mechanics, never GSM8K task scores. Compare a two-update
uninterrupted run with save-after-one/rebuild/load/update, including the next
sampled token IDs, Python/global Torch/rollout RNG draws, and optimizer state.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import sys
import tempfile

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIRECTORY))

from geora_layers import GeoRALinear, iter_target_linears
from geora_check_utils import move_with_precision
from grpo_math import group_advantages, grpo_loss
from grpo_rollout import sample_completions, score_completion_logps
from grpo_training_state import (
    BOUNDARY, canonical_json_sha256, load_boundary_checkpoint,
    make_constant_scheduler, save_boundary_checkpoint,
)
from lora_layers import install_lora


def tiny_models(method: str, dtype: torch.dtype):
    torch.manual_seed(18)
    original = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=37, hidden_size=32, intermediate_size=48,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        attention_dropout=0.0, eos_token_id=2, pad_token_id=0,
    )).float()
    original.config._attn_implementation = "eager"
    original.requires_grad_(False)
    reference = copy.deepcopy(original)
    model = copy.deepcopy(original)
    if method == "geora":
        initialization_generator = torch.Generator().manual_seed(901)
        for name, layer in list(iter_target_linears(model)):
            parent_name, leaf_name = name.rsplit(".", 1)
            A0 = torch.randn(2, layer.in_features, generator=initialization_generator) * 0.02
            B0 = torch.randn(layer.out_features, 2, generator=initialization_generator) * 0.02
            replacement = GeoRALinear(layer, A0, B0, 2.0, forward_mode="difference")
            setattr(model.get_submodule(parent_name), leaf_name, replacement)
    else:
        install_lora(model, rank=2, alpha=4, seed=901, init_std=0.02)
    move_with_precision(model, "cpu", dtype)
    move_with_precision(reference, "cpu", dtype)
    model.train()
    reference.eval()
    return model, reference


def make_optimizer(model):
    return torch.optim.AdamW(
        [value for value in model.parameters() if value.requires_grad],
        lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
    )


def adapter_snapshot(model):
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters() if value.requires_grad}


def frozen_snapshot(model):
    state = {name: value.detach().clone() for name, value in model.named_parameters() if not value.requires_grad}
    state.update({name: value.detach().clone() for name, value in model.named_buffers() if name.endswith((".A0", ".B0"))})
    return state


def assert_tree_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor) and torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for name in expected:
            assert_tree_equal(actual[name], expected[name])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_tree_equal(left, right)
    else:
        assert actual == expected


def mechanical_update(model, reference, optimizer, scheduler, generator, dtype):
    prompts = [[3, 4, 5]] * 2 + [[6, 7, 8, 9, 10]] * 2
    rollout = sample_completions(model, prompts, 0, [2, 3, 4, 5], 4, "cpu", dtype, generator=generator)
    old_logp = score_completion_logps(model, rollout, "cpu", dtype, gradient_enabled=False)
    ref_logp = score_completion_logps(reference, rollout, "cpu", dtype, gradient_enabled=False)
    current_logp = score_completion_logps(model, rollout, "cpu", dtype, gradient_enabled=True)
    advantages = group_advantages(torch.tensor([[1.0, 0.0], [1.0, 0.0]])).flatten()
    terms = grpo_loss(current_logp, old_logp, ref_logp, advantages, rollout["completion_mask"])
    optimizer.zero_grad(set_to_none=True)
    terms["loss"].backward()
    norm = torch.nn.utils.clip_grad_norm_([value for value in model.parameters() if value.requires_grad], 1.0, error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return {"token_ids": rollout["completion_ids"].clone(), "mask": rollout["completion_mask"].clone(),
            "loss": terms["loss"].item(), "gradient_norm": norm.item()}


def next_rng_draws(generator):
    return {"torch_cpu": torch.randn(5), "python": [random.random() for _ in range(5)],
            "rollout_generator": torch.rand(5, generator=generator)}


def run_case(method: str, dtype: torch.dtype, directory: Path):
    model, reference = tiny_models(method, dtype)
    frozen_before = frozen_snapshot(model)
    optimizer = make_optimizer(model)
    scheduler = make_constant_scheduler(optimizer)
    generator = torch.Generator().manual_seed(412)
    random.seed(211)
    provenance = {
        "method": method, "model_repository": "synthetic_random_tiny_qwen", "model_revision": "seed18",
        "dataset_manifest_sha256": canonical_json_sha256({"data": "synthetic fixed CPU prompts"}),
        "training_config_sha256": canonical_json_sha256({"dtype": str(dtype), "optimizer": "AdamW", "lr": 1e-4}),
        "initialization_sha256": canonical_json_sha256({"method": method, "seed": 901, "rank": 2, "alpha": 4}),
        "reference": "original_random_tiny_qwen_seed18",
    }
    first = mechanical_update(model, reference, optimizer, scheduler, generator, dtype)
    progress = {"boundary": BOUNDARY, "rollout_in_flight": False, "completed_optimizer_steps": 1,
                "completed_rollouts": 1, "next_data_position": 2}
    saved_adapter = adapter_snapshot(model)
    checkpoint = directory / f"{method}-{str(dtype).split('.')[-1]}-step1"
    manifest = save_boundary_checkpoint(checkpoint, model=model, optimizer=optimizer, scheduler=scheduler,
                                        rollout_generator=generator, progress=progress, provenance=provenance, device="cpu")
    assert_tree_equal(adapter_snapshot(model), saved_adapter)

    expected_draws = next_rng_draws(generator)
    second = mechanical_update(model, reference, optimizer, scheduler, generator, dtype)
    expected_adapter = adapter_snapshot(model)
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    expected_scheduler = copy.deepcopy(scheduler.state_dict())
    expected_rng_after = {"torch_cpu": torch.get_rng_state().clone(), "python": random.getstate(),
                          "rollout": generator.get_state().clone()}

    resumed, resumed_reference = tiny_models(method, dtype)
    resumed_optimizer = make_optimizer(resumed)
    resumed_scheduler = make_constant_scheduler(resumed_optimizer)
    resumed_generator = torch.Generator().manual_seed(999)
    parameter_identities = {name: id(value) for name, value in resumed.named_parameters() if value.requires_grad}
    restored_progress = load_boundary_checkpoint(checkpoint, model=resumed, optimizer=resumed_optimizer,
                                                scheduler=resumed_scheduler, rollout_generator=resumed_generator,
                                                expected_provenance=provenance, device="cpu")
    assert restored_progress == progress
    assert parameter_identities == {name: id(value) for name, value in resumed.named_parameters() if value.requires_grad}
    assert_tree_equal(adapter_snapshot(resumed), saved_adapter)
    actual_draws = next_rng_draws(resumed_generator)
    assert_tree_equal(actual_draws, expected_draws)
    resumed_second = mechanical_update(resumed, resumed_reference, resumed_optimizer, resumed_scheduler, resumed_generator, dtype)
    assert_tree_equal(resumed_second, second)
    assert_tree_equal(adapter_snapshot(resumed), expected_adapter)
    assert_tree_equal(resumed_optimizer.state_dict(), expected_optimizer)
    assert_tree_equal(resumed_scheduler.state_dict(), expected_scheduler)
    assert_tree_equal({"torch_cpu": torch.get_rng_state(), "python": random.getstate(), "rollout": resumed_generator.get_state()}, expected_rng_after)
    assert_tree_equal(frozen_snapshot(model), frozen_before)
    assert_tree_equal(frozen_snapshot(resumed), frozen_before)

    # Reject mismatched provenance before copying A/B or changing any RNG state.
    state_before_rejection = adapter_snapshot(resumed)
    cpu_rng_before_rejection = torch.get_rng_state().clone()
    rollout_rng_before_rejection = resumed_generator.get_state().clone()
    python_rng_before_rejection = random.getstate()
    mismatched = dict(provenance, dataset_manifest_sha256="0" * 64)
    try:
        load_boundary_checkpoint(checkpoint, model=resumed, optimizer=resumed_optimizer, scheduler=resumed_scheduler,
                                 rollout_generator=resumed_generator, expected_provenance=mismatched, device="cpu")
    except ValueError as error:
        assert "provenance" in str(error)
    else:
        raise AssertionError("Mismatched dataset provenance was accepted.")
    assert_tree_equal(adapter_snapshot(resumed), state_before_rejection)
    assert torch.equal(torch.get_rng_state(), cpu_rng_before_rejection)
    assert torch.equal(resumed_generator.get_state(), rollout_rng_before_rejection)
    assert random.getstate() == python_rng_before_rejection

    # Save and reload on the existing model also reproduces the next update.
    load_boundary_checkpoint(checkpoint, model=model, optimizer=optimizer, scheduler=scheduler,
                             rollout_generator=generator, expected_provenance=provenance, device="cpu")
    assert_tree_equal(next_rng_draws(generator), expected_draws)
    same_model_second = mechanical_update(model, reference, optimizer, scheduler, generator, dtype)
    assert_tree_equal(same_model_second, second)
    assert_tree_equal(adapter_snapshot(model), expected_adapter)
    assert_tree_equal(optimizer.state_dict(), expected_optimizer)

    # Completed-boundary guards and immutable directories reject accidental overwrites.
    try:
        save_boundary_checkpoint(checkpoint, model=model, optimizer=optimizer, scheduler=scheduler,
                                 rollout_generator=generator, progress=progress, provenance=provenance, device="cpu")
    except FileExistsError:
        pass
    else:
        raise AssertionError("Checkpoint overwrite was allowed.")
    try:
        save_boundary_checkpoint(directory / "must-not-exist", model=model, optimizer=optimizer, scheduler=scheduler,
                                 rollout_generator=generator, progress=dict(progress, rollout_in_flight=True), provenance=provenance, device="cpu")
    except ValueError as error:
        assert "in-flight" in str(error)
    else:
        raise AssertionError("In-flight checkpoint save was allowed.")

    return {"status": "passed", "method": method, "frozen_dtype": str(dtype), "adapter_dtype": "torch.float32",
            "reward_scope": "artificial CPU mechanics, not GSM8K", "uninterrupted_optimizer_steps": 2,
            "saved_completed_optimizer_steps": 1, "saved_next_data_position": 2,
            "next_sampling_tokens_equal": True, "next_cpu_python_and_rollout_rng_draws_equal": True,
            "next_update_adapter_optimizer_scheduler_exact": True, "same_model_inplace_resume_exact": True,
            "parameter_objects_preserved": True, "frozen_weights_and_initial_buffers_unchanged": True,
            "provenance_mismatch_rejected_without_mutation": True, "in_flight_and_overwrite_rejected": True,
            "first_gradient_norm": first["gradient_norm"], "second_gradient_norm": second["gradient_norm"],
            "checkpoint_adapter_tensor_count": len(manifest["adapter_tensors"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    torch.set_num_threads(2)
    report = {"status": "running", "stage": "tiny_cpu_full_boundary_resume", "torch_version": str(torch.__version__), "cases": []}
    try:
        with tempfile.TemporaryDirectory(prefix="geora-grpo-resume-") as temporary:
            for method in ("geora", "lora"):
                for dtype in (torch.float32, torch.bfloat16):
                    result = run_case(method, dtype, Path(temporary))
                    report["cases"].append(result)
                    print(f"PASS: exact full boundary resume {method} {dtype}", flush=True)
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        if arguments.output_dir is not None:
            arguments.output_dir.mkdir(parents=True, exist_ok=True)
            path = arguments.output_dir / "resume_cpu_checks.json"
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(f"Report: {path}", flush=True)


if __name__ == "__main__":
    main()
