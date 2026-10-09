"""Compare untrained GeoRA and its original model under matched precision.

Reuse saved FP32 initialization. No SVD, generation, backward, or optimizer.
Candidate settings are temporary diagnostics, not a production policy change.
"""

import argparse
from contextlib import ExitStack
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

from forward_precision import (
    fp32_attention_scores, fp32_projection_forwards,
    prepare_fp32_projections, projection_modules,
)
from geora_check_utils import (
    fresh_fp32_model, inference_logits, logits_difference,
    move_with_precision, pinned_configuration, validate_manifest,
)
from diagnose_geora_precision import position_metrics
from geora_layers import GeoRALinear, load_geora_state


PROMPTS = (
    ("legacy_with_answer", "What is 2 + 3? Answer briefly.", "5"),
    ("legacy_prompt_only", "What is 2 + 3? Answer briefly.", None),
    ("physics_prompt_only", "What is Newton's second law? Answer in one short sentence.", None),
)


def parameter_dtypes(model):
    counts = {}
    for parameter in model.parameters():
        label = str(parameter.dtype)
        counts[label] = counts.get(label, 0) + parameter.numel()
    return counts


def forward_comparison(reference, model, cases, mode, device, report, persist):
    """Change both models together; measure all positions and answer positions."""
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    mode_report = {
        "reference_parameter_dtypes": parameter_dtypes(reference),
        "geora_parameter_dtypes": parameter_dtypes(model),
        "projection_parameter_dtype": str(projection_modules(model)[0][1].base_layer.weight.dtype),
        "cases": {},
    }
    report["modes"][mode] = mode_report
    references = {}
    with ExitStack() as contexts:
        if mode != "native":
            contexts.enter_context(fp32_projection_forwards(reference))
            contexts.enter_context(fp32_projection_forwards(model))
        if mode == "fp32_projections_scores":
            contexts.enter_context(fp32_attention_scores())
        for name, inputs, labels, prompt_length in cases:
            ref_logits = inference_logits(reference, inputs, device, torch.bfloat16)
            geo_logits = inference_logits(model, inputs, device, torch.bfloat16)
            if not torch.isfinite(ref_logits).all() or not torch.isfinite(geo_logits).all():
                raise RuntimeError(f"Non-finite logits: {mode}/{name}")
            metrics = logits_difference(ref_logits, geo_logits)
            passed = (
                metrics["relative_l2_error"] <= report["existing_limits"]["relative_l2_error"]
                and metrics["last_token_reference_to_actual_kl_nats"] <= report["existing_limits"]["last_position_kl_nats"]
            )
            row = {
                "token_count": inputs["input_ids"].shape[1],
                "prompt_token_count": prompt_length,
                "logits_shape": list(ref_logits.shape),
                "logits_metrics": metrics,
                "all_position_metrics": position_metrics(ref_logits, geo_logits, labels),
                "meets_existing_initialization_limits": passed,
            }
            mode_report["cases"][name] = row
            references[name] = ref_logits
            print("FORWARD", mode, name, json.dumps(metrics), flush=True)
            persist()
    if device == "cuda":
        torch.cuda.synchronize()
        mode_report["peak_allocated_memory_gib"] = torch.cuda.max_memory_allocated() / 2**30
    mode_report["elapsed_seconds"] = time.perf_counter() - started
    mode_report["all_cases_meet_existing_limits"] = all(
        row["meets_existing_initialization_limits"] for row in mode_report["cases"].values()
    )
    persist()
    return references


def run(arguments, report, persist):
    started = time.perf_counter()
    device = arguments.device
    if device == "cuda" and (torch.version.hip is None or not torch.cuda.is_available()):
        raise RuntimeError("A visible ROCm GPU is required for this Setonix experiment")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    configuration, checkpoint = pinned_configuration()
    manifest_path = arguments.init_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest, configuration)
    state = load_file(str(arguments.init_dir / "adapter.safetensors"))
    for target in manifest["target_modules"]:
        for factor in ("A", "B"):
            if not torch.equal(state[f"{target['name']}.{factor}"], state[f"{target['name']}.{factor}0"]):
                raise RuntimeError("The experiment requires an untrained initialization")

    report.update(
        stage="full_model_forward_precision_comparison",
        optimizer_steps=0, backward_calls=0, svd_calls=0,
        model_repository=configuration["model_repository"],
        model_revision=configuration["model_revision"],
        initialization_directory=str(arguments.init_dir),
        initialization_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        torch_version=torch.__version__, hip_version=torch.version.hip,
        device=device, slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        gpu_name=torch.cuda.get_device_name(0) if device == "cuda" else None,
        source_git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        modes={}, reference_controls={},
        existing_limits={"relative_l2_error": 0.02, "last_position_kl_nats": 0.02},
        scope="Measured diagnostics; completed does not mean every mode passed. Defaults unchanged.",
    )
    persist()
    print("Loading original CPU FP32 model and reconstructing GeoRA F from saved factors.", flush=True)
    reference = fresh_fp32_model(checkpoint)
    model = fresh_fp32_model(checkpoint)
    load_geora_state(model, state, manifest)
    target_names = prepare_fp32_projections(reference, device)
    geo_names = prepare_fp32_projections(model, device)
    if target_names != geo_names or len(target_names) != 196:
        raise RuntimeError("Expected all 196 matched decoder projections")
    report["target_module_count"] = len(target_names)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    cases = []
    for name, question, answer in PROMPTS:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = tokenizer.encode(text, add_special_tokens=False)
        ids = list(prompt_ids)
        if answer is not None:
            ids += tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
        input_ids = torch.tensor([ids], device=device)
        labels = input_ids.clone()
        labels[:, :len(prompt_ids)] = -100
        inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        cases.append((name, inputs, labels, len(prompt_ids)))
    report["prompt_definitions"] = [
        {"name": name, "question": question, "appended_answer": answer,
         "appended_eos": answer is not None}
        for name, question, answer in PROMPTS
    ]

    candidate_references = {}
    for mode in ("fp32_projections", "fp32_projections_scores"):
        candidate_references[mode] = forward_comparison(
            reference, model, cases, mode, device, report, persist,
        )

    # Run native last: this downcasts F, so it must not precede FP32 candidates.
    # A/B remain trainable FP32 parameters, although inference_mode prevents grads.
    move_with_precision(reference, device, torch.bfloat16)
    move_with_precision(model, device, torch.bfloat16)
    native_references = forward_comparison(
        reference, model, cases, "native", device, report, persist,
    )
    for mode, references in candidate_references.items():
        report["reference_controls"][mode] = {
            name: logits_difference(native_references[name], logits)
            for name, logits in references.items()
        }

    # No training is performed: verify the same initial factors remain installed.
    unchanged = True
    for name, module in model.named_modules():
        if isinstance(module, GeoRALinear):
            for factor in ("A0", "B0", "A", "B"):
                unchanged = unchanged and torch.equal(getattr(module, factor).detach().cpu(), state[f"{name}.{factor}"])
    report["factor_values_unchanged"] = unchanged
    report["no_parameter_gradients"] = all(p.grad is None for p in model.parameters())
    if not unchanged or not report["no_parameter_gradients"]:
        raise RuntimeError("Forward-only experiment changed factors or created gradients")
    report["peak_allocated_memory_gib"] = max(
        mode.get("peak_allocated_memory_gib", 0) for mode in report["modes"].values()
    )
    report["elapsed_seconds"] = time.perf_counter() - started
    report["status"] = "completed"
    persist()
    print("Full-model forward comparison completed; optimizer steps: 0.", flush=True)
    print("Report:", arguments.output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running"}

    def persist():
        arguments.output.write_text(json.dumps(report, indent=2) + "\n")

    try:
        run(arguments, report, persist)
    except Exception as error:
        report.update(status="failed", error=str(error))
        persist()
        raise


if __name__ == "__main__":
    main()
