"""Check generation/cache/padding and independent first-step LR sensitivity.

Reuse the recorded FP32 GeoRA factors. Each LR starts at A=A0, B=B0 with
a fresh AdamW optimizer. This is a CE diagnostic, not GRPO or task evaluation.
"""

import argparse
import faulthandler
import hashlib
import json
import math
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
from check_geora_training import causal_loss
from check_difference_training import RUNTIME_PRECISION, effective_update_norm
from check_full_forward_precision import PROMPTS
from diagnose_geora_precision import position_metrics
from geora_generation_checks import check_generation_paths


@torch.no_grad()
def reset_initial_factors(targets):
    """Restore the same initial point without replacing Parameter objects."""
    for _, module in targets:
        module.A.copy_(module.A0)
        module.B.copy_(module.B0)


def learning_rate_trials(model, reference, cases, targets, report, device, frozen_dtype):
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    frozen_snapshot = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    initial_snapshot = {
        name: buffer.detach().clone() for name, buffer in model.named_buffers()
        if name.endswith((".A0", ".B0"))
    }
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    expected_names = {name + "." + factor for name, _ in targets for factor in ("A", "B")}
    report.require("lr_expected_trainable_scope", trainable_names == expected_names and len(trainable) == 392
                   and sum(p.numel() for p in trainable) == 18464768)
    before_logits = {
        case["name"]: inference_logits(reference, case["inputs"], device, frozen_dtype)
        for case in cases
    }
    training_case = cases[0]
    report.data["learning_rate_trials"] = []
    first_loss = None
    first_gradient_norm = None
    first_gradients = None
    for learning_rate in (1e-4, 1e-5, 1e-6):
        reset_initial_factors(targets)
        model.zero_grad(set_to_none=True)
        torch.manual_seed(0)
        model.eval()  # Exclude dropout randomness from this controlled comparison.
        initial_logits = inference_logits(model, training_case["inputs"], device, frozen_dtype)
        report.require(f"lr_{learning_rate:g}_starts_at_exact_initial_logits",
                       torch.equal(initial_logits, before_logits[training_case["name"]]))
        optimizer = torch.optim.AdamW(
            trainable, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
        )
        report.require(f"lr_{learning_rate:g}_optimizer_starts_empty", len(optimizer.state) == 0)
        loss = causal_loss(model, training_case["inputs"], training_case["labels"], device, frozen_dtype)
        report.require(f"lr_{learning_rate:g}_initial_loss_finite", torch.isfinite(loss).item(),
                       loss_before=loss.item())
        report.data["backward_attempted"] = True
        report.write()
        loss.backward()
        report.data["backward_calls"] += 1
        report.write()
        gradient_valid = all(
            p.grad is not None and p.grad.dtype == torch.float32
            and torch.isfinite(p.grad).all().item() for p in trainable
        )
        report.require(f"lr_{learning_rate:g}_gradients_finite_fp32", gradient_valid)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable, max_norm=1.0, error_if_nonfinite=True,
        ).item()
        if first_loss is None:
            first_loss = loss.item()
            first_gradient_norm = gradient_norm
            first_gradients = [p.grad.detach().clone() for p in trainable]
        report.require(f"lr_{learning_rate:g}_same_initial_loss_and_clipped_gradients",
                       loss.item() == first_loss and gradient_norm == first_gradient_norm
                       and all(torch.equal(p.grad, g) for p, g in zip(trainable, first_gradients)),
                       gradient_norm_before_clipping=gradient_norm)
        optimizer.step()
        report.data["optimizer_steps"] += 1
        report.data["independent_optimizer_steps"] += 1
        report.write()
        report.require(f"lr_{learning_rate:g}_fresh_optimizer_first_step",
                       len(optimizer.state) == len(trainable)
                       and all(s["step"].item() == 1 for s in optimizer.state.values()))
        report.require(f"lr_{learning_rate:g}_updated_parameters_finite",
                       all(torch.isfinite(p).all().item() for p in trainable))
        factor_norms = [
            torch.linalg.vector_norm((getattr(module, factor)-getattr(module, factor+"0")).detach().double()).item()
            for _, module in targets for factor in ("A", "B")
        ]
        layer_updates = [
            {"name": name, "effective_update_frobenius_norm": effective_update_norm(module)}
            for name, module in targets
        ]
        report.require(f"lr_{learning_rate:g}_effective_updates_finite_nonzero",
                       all(math.isfinite(row["effective_update_frobenius_norm"])
                           and row["effective_update_frobenius_norm"] > 0 for row in layer_updates))
        parameters = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        report.require(f"lr_{learning_rate:g}_frozen_parameters_and_initial_factors_unchanged",
                       all(torch.equal(parameters[name], value) for name, value in frozen_snapshot.items())
                       and all(torch.equal(buffers[name], value) for name, value in initial_snapshot.items())
                       and all(p.grad is None for p in parameters.values() if not p.requires_grad))
        after_loss = causal_loss(model, training_case["inputs"], training_case["labels"], device, frozen_dtype)
        report.require(f"lr_{learning_rate:g}_updated_loss_finite", torch.isfinite(after_loss).item(),
                       loss_after=after_loss.item())
        measurements = {}
        for case in cases:
            actual = inference_logits(model, case["inputs"], device, frozen_dtype)
            expected = before_logits[case["name"]]
            report.require(f"lr_{learning_rate:g}_forward_finite_{case['name']}",
                           torch.isfinite(actual).all().item())
            positions = position_metrics(expected, actual, case["labels"])
            prompt_position = case["prompt_token_count"] - 1
            measurements[case["name"]] = {
                "all_positions": logits_difference(expected, actual),
                "prompt_next_token": positions[prompt_position],
                "supervised_prediction_positions": [p for p in positions if p["predicts_supervised_answer"]],
            }
        row = {
            "learning_rate": learning_rate, "independent_first_step": True,
            "loss_before": loss.item(), "loss_after": after_loss.item(),
            "gradient_norm_before_clipping": gradient_norm,
            "gradient_clip_limit": 1.0,
            "adapter_factor_change_combined_l2": math.sqrt(sum(n*n for n in factor_norms)),
            "effective_weight_change_combined_frobenius": math.sqrt(sum(
                u["effective_update_frobenius_norm"]**2 for u in layer_updates)),
            "effective_layer_updates": layer_updates,
            "outputs": measurements,
        }
        report.data["learning_rate_trials"].append(row)
        report.write()
        print("LR TRIAL:", json.dumps({
            "lr": learning_rate, "loss_after": row["loss_after"],
            "weight_change_norm": row["effective_weight_change_combined_frobenius"],
            "answer_prompt_kl_nats": measurements[training_case["name"]]["prompt_next_token"]["kl_nats"],
            "physics_prompt_kl_nats": measurements[cases[-1]["name"]]["prompt_next_token"]["kl_nats"],
        }), flush=True)
        del optimizer, loss, after_loss
    # No trained checkpoint is exported: these are independent diagnostic trials.
    reset_initial_factors(targets)
    model.zero_grad(set_to_none=True)
    report.require("lr_final_state_restored_to_untrained_factors", all(
        torch.equal(module.A, module.A0) and torch.equal(module.B, module.B0)
        for _, module in targets))
    report.require("lr_reference_logits_unchanged", all(
        torch.equal(inference_logits(reference, case["inputs"], device, frozen_dtype),
                    before_logits[case["name"]]) for case in cases))
    report.require("lr_restored_logits_exact", torch.equal(
        inference_logits(model, training_case["inputs"], device, frozen_dtype),
        before_logits[training_case["name"]]))
    report.data["final_model_state"] = "untrained_restored"
    report.write()


def run(arguments, report):
    started = time.perf_counter()
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("This check requires a visible ROCm GPU")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    configuration, checkpoint = pinned_configuration()
    if arguments.checkpoint_dir is not None:
        checkpoint = arguments.checkpoint_dir
        staging = json.loads((checkpoint / "staging_record.json").read_text())
        if staging["status"] != "validated" or staging["model_revision"] != configuration["model_revision"]:
            raise RuntimeError("Temporary checkpoint must pass CPU checksum/value validation first")
        report.data["checkpoint_staging"] = staging
    manifest_path = arguments.init_dir / "manifest.json"
    saved_manifest = json.loads(manifest_path.read_text())
    validate_manifest(saved_manifest, configuration)
    state = load_file(str(arguments.init_dir / "adapter.safetensors"))
    manifest = dict(saved_manifest, forward_mode="difference", runtime_precision=RUNTIME_PRECISION)
    report.data.update(
        stage="difference_generation_and_independent_lr_trials", optimizer_steps=0,
        backward_calls=0, independent_optimizer_steps=0, svd_calls=0,
        forward_mode="difference", runtime_precision=RUNTIME_PRECISION,
        model_repository=configuration["model_repository"], model_revision=configuration["model_revision"],
        source_git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        initialization_directory=str(arguments.init_dir),
        initialization_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        torch_version=torch.__version__, hip_version=torch.version.hip,
        gpu_name=torch.cuda.get_device_name(0), slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        scope="Untrained generation checks plus three independent CE first steps; no GRPO or task scores.",
        checkpoint_directory=str(checkpoint),
        model_loader_options={"disable_mmap": arguments.disable_mmap,
                              "HF_DEACTIVATE_ASYNC_LOAD": os.environ.get("HF_DEACTIVATE_ASYNC_LOAD")},
    )
    report.write()
    report.require("untrained_factor_source", all(
        torch.equal(state[f"{target['name']}.{factor}"], state[f"{target['name']}.{factor}0"])
        for target in manifest["target_modules"] for factor in ("A", "B")))
    report.data["model_loading_seconds"] = {}
    def load_model(name):
        report.data["current_stage"] = "loading_" + name
        report.write()
        print("Loading", name, "from", checkpoint, flush=True)
        load_started = time.perf_counter()
        loaded = fresh_fp32_model(checkpoint, disable_mmap=arguments.disable_mmap)
        report.data["model_loading_seconds"][name] = time.perf_counter() - load_started
        report.write()
        return loaded
    reference = load_model("reference")
    model = load_model("difference_model")
    load_geora_state(model, state, manifest)
    del state
    device, frozen_dtype = "cuda", torch.bfloat16
    move_with_precision(reference, device, frozen_dtype)
    move_with_precision(model, device, frozen_dtype)
    targets = [(name, module) for name, module in model.named_modules() if isinstance(module, GeoRALinear)]
    report.require("all_196_difference_targets", len(targets) == 196
                   and all(module.forward_mode == "difference" for _, module in targets))
    report.require("mixed_parameter_precision", all(
        p.dtype == (torch.float32 if p.requires_grad else frozen_dtype) for p in model.parameters()))
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    cases = []
    prompts = []
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
        cases.append({"name": name, "inputs": {
            "input_ids": input_ids, "attention_mask": torch.ones_like(input_ids),
        }, "labels": labels, "prompt_token_count": len(prompt_ids)})
        if answer is None:
            prompts.append(prompt_ids)
    report.require("CE_only_answer_and_EOS_labels", (cases[0]["labels"][:, 1:] != -100).sum().item() == 2,
                   supervised_token_count=2)
    generation = check_generation_paths(
        model, reference, prompts, tokenizer.pad_token_id, tokenizer.eos_token_id,
        device, frozen_dtype, report, max_new_tokens=8,
    )
    report.data["generation"] = generation
    report.data["current_stage"] = "independent_lr_trials"
    report.write()
    learning_rate_trials(model, reference, cases, targets, report, device, frozen_dtype)
    torch.cuda.synchronize()
    report.data.update(elapsed_seconds=time.perf_counter()-started,
                       peak_allocated_memory_gib=torch.cuda.max_memory_allocated()/2**30,
                       current_stage="finished")
    report.finish()
    print("Continuation checks passed:", report.path, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, help="Optional checksum-verified temporary checkpoint copy.")
    parser.add_argument("--disable-mmap", action="store_true", help="Read safetensors before materializing weights.")
    arguments = parser.parse_args()
    report = CheckReport(arguments.output_dir, "continuation_checks.json")
    attempted_start = time.perf_counter()
    faulthandler.dump_traceback_later(30, repeat=True)
    try:
        run(arguments, report)
    except Exception as error:
        report.data.update(status="failed", error=str(error), elapsed_seconds=time.perf_counter()-attempted_start)
        if torch.cuda.is_available():
            report.data["peak_allocated_memory_gib"] = torch.cuda.max_memory_allocated()/2**30
        report.write()
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
