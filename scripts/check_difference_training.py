"""Verify difference-form initialization, one update, and round trips on ROCm.

Reuse saved initialization factors without SVD. The original model retains its
native BF16 path; only the two low-rank correction branches compute FP32.
The cross-entropy update is a mechanical check, not GRPO or a task evaluation.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIRECTORY))

from geora_layers import GeoRALinear, load_geora_state
from geora_check_utils import (
    CheckReport, fresh_fp32_model, inference_logits, logits_difference,
    move_with_precision, pinned_configuration, validate_manifest,
)
from check_geora_training import validate_model_and_update
from check_full_forward_precision import PROMPTS
from diagnose_geora_precision import position_metrics


RUNTIME_PRECISION = {
    "base_weight_storage": "bfloat16_original_W_pre",
    "base_forward": "native_bfloat16_autocast",
    "A_B_A0_B0_storage": "float32",
    "low_rank_branches_and_subtraction": "float32_autocast_disabled",
    "correction_before_base_addition": "cast_to_base_output_dtype",
    "attention": "unmodified_native_eager",
}


@torch.no_grad()
def effective_update_norm(module):
    """Compute ||c(BA-B0A0)||_F using small FP64 Gram matrices.

    BA-B0A0 = B(A-A0) + (B-B0)A0. This avoids subtracting large
    near-equal Gram energies and never allocates a full dense delta_W.
    """
    A = module.A.detach().double()
    B = module.B.detach().double()
    A0 = module.A0.double()
    B0 = module.B0.double()
    left = torch.cat((B, B-B0), dim=1)
    right = torch.cat((A-A0, A0), dim=0)
    left_gram = left.T @ left
    right_gram = right @ right.T
    squared_norm = (left_gram * right_gram.T).sum()
    if squared_norm.item() < -1e-12:
        raise RuntimeError("Negative FP64 effective-update squared norm")
    return module.scaling * squared_norm.clamp_min(0).sqrt().item()


def run(arguments, report):
    started = time.perf_counter()
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("This Setonix check requires a visible ROCm GPU")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    configuration, checkpoint = pinned_configuration()
    manifest_path = arguments.init_dir / "manifest.json"
    saved_manifest = json.loads(manifest_path.read_text())
    validate_manifest(saved_manifest, configuration)
    state = load_file(str(arguments.init_dir / "adapter.safetensors"))
    report.data.update(
        stage="geora_difference_pre_grpo_checks",
        optimizer_steps=0, backward_calls=0, svd_calls=0,
        model_repository=configuration["model_repository"],
        model_revision=configuration["model_revision"],
        source_git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        initialization_directory=str(arguments.init_dir),
        initialization_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        torch_version=torch.__version__, hip_version=torch.version.hip,
        gpu_name=torch.cuda.get_device_name(0), slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        forward_mode="difference", runtime_precision=RUNTIME_PRECISION,
        scope="One mechanical cross-entropy step only; no GRPO or task scores.",
    )
    report.write()
    report.require("checkpoint_is_untrained_initialization", all(
        torch.equal(state[f"{target['name']}.{factor}"], state[f"{target['name']}.{factor}0"])
        for target in saved_manifest["target_modules"] for factor in ("A", "B")
    ))
    # Record the intentional new interpretation before loading. This changes
    # runtime evaluation only, not the saved mask/SVD factors or scaling.
    manifest = dict(
        saved_manifest, forward_mode="difference", runtime_precision=RUNTIME_PRECISION,
        checkpoint_contents="initial A0/B0 and current A/B; retain original W_pre in difference mode",
    )
    device = "cuda"
    frozen_dtype = torch.bfloat16

    def base_model_factory():
        return fresh_fp32_model(checkpoint)

    reference = base_model_factory()
    move_with_precision(reference, device, frozen_dtype)
    model = base_model_factory()
    load_geora_state(model, state, manifest)
    move_with_precision(model, device, frozen_dtype)
    targets = [(name, module) for name, module in model.named_modules() if isinstance(module, GeoRALinear)]
    report.require("all_196_difference_targets", len(targets) == 196
                   and all(module.forward_mode == "difference" for _, module in targets),
                   target_count=len(targets))
    base_exact = all(
        torch.equal(module.base_layer.weight, reference.get_submodule(name).weight)
        and (module.base_layer.bias is None
             or torch.equal(module.base_layer.bias, reference.get_submodule(name).bias))
        for name, module in targets
    )
    report.require("all_original_projection_weights_and_biases_exact", base_exact)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    cases = []
    report.data["initial_forward_cases"] = {}
    for name, question, answer in PROMPTS:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = tokenizer.encode(text, add_special_tokens=False)
        ids = list(prompt_ids)
        if answer is not None:
            ids += tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
        input_ids = torch.tensor([ids], device=device)
        inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        labels = input_ids.clone()
        labels[:, :len(prompt_ids)] = -100
        expected = inference_logits(reference, inputs, device, frozen_dtype)
        actual = inference_logits(model, inputs, device, frozen_dtype)
        metrics = logits_difference(expected, actual)
        report.data["initial_forward_cases"][name] = {
            "question": question, "appended_answer": answer, "token_count": len(ids),
            "prompt_token_count": len(prompt_ids), "logits_metrics": metrics,
            "position_metrics": position_metrics(expected, actual, labels),
        }
        report.require(f"initial_forward_finite_{name}",
                       torch.isfinite(expected).all().item() and torch.isfinite(actual).all().item())
        report.require(f"initial_logits_exact_{name}", torch.equal(expected, actual), **metrics)
        # The former gate is retained; the additional exact-equality gate is stronger.
        report.require(f"existing_limits_retained_{name}",
                       metrics["relative_l2_error"] <= 0.02
                       and metrics["last_token_reference_to_actual_kl_nats"] <= 0.02,
                       relative_error_limit=0.02, kl_limit_nats=0.02)
        cases.append((name, inputs, labels))
    report.require("factors_unchanged_by_initial_forwards", all(
        torch.equal(getattr(module, factor).detach().cpu(), state[f"{name}.{factor}"])
        for name, module in targets for factor in ("A0", "B0", "A", "B")
    ))
    del state
    if not arguments.forward_only:
        _, inputs, labels = cases[0]
        validate_model_and_update(
            model, reference, manifest, base_model_factory, inputs, labels,
            arguments.output_dir, report, device, frozen_dtype,
        )
        updates = [{"name": name, "effective_update_frobenius_norm": effective_update_norm(module)}
                   for name, module in targets]
        report.require("all_effective_updates_finite_nonzero",
                       all(0 < row["effective_update_frobenius_norm"] < float("inf") for row in updates),
                       layers=updates)
        trained_manifest = json.loads((arguments.output_dir / "manifest.json").read_text())
        report.require("trained_manifest_persists_difference_mode",
                       trained_manifest["forward_mode"] == "difference"
                       and trained_manifest["runtime_precision"] == RUNTIME_PRECISION)
    torch.cuda.synchronize()
    report.data.update(
        elapsed_seconds=time.perf_counter()-started,
        peak_allocated_memory_gib=torch.cuda.max_memory_allocated()/2**30,
    )
    report.finish()
    print("Difference-form checks passed; optimizer steps:", report.data["optimizer_steps"], flush=True)
    print("Report:", report.path, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--forward-only", action="store_true")
    arguments = parser.parse_args()
    report = CheckReport(arguments.output_dir, "difference_checks.json")
    attempted_start = time.perf_counter()
    try:
        run(arguments, report)
    except Exception as error:
        report.data.update(
            status="failed", error=str(error),
            elapsed_seconds=time.perf_counter()-attempted_start,
        )
        if torch.cuda.is_available():
            report.data["peak_allocated_memory_gib"] = torch.cuda.max_memory_allocated()/2**30
        report.write()
        raise
