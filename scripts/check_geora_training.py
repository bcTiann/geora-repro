"""Check mixed precision, one A/B update, reference, and checkpoint round trips.

The short cross-entropy update is a mechanical backward/optimizer check.
It is not a GRPO run or a task performance evaluation.
"""

import argparse
import json
import math
import os
from pathlib import Path
import time

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from geora_check_utils import (
    CheckReport,
    forward_context,
    fresh_fp32_model,
    inference_logits,
    logits_difference,
    move_with_precision,
    pinned_configuration,
    validate_manifest,
)
from geora_layers import GeoRALinear, export_adapter_state, load_geora_state


def restored_model(base_model_factory, state, manifest, device, frozen_dtype):
    # Reconstruct F on CPU from the original W and saved initial factors, not trained B@A.
    model = base_model_factory()
    load_geora_state(model, state, manifest)
    move_with_precision(model, device, frozen_dtype)
    return model


def checked_forward(model, inputs, device, frozen_dtype, report, name):
    logits = inference_logits(model, inputs, device, frozen_dtype)
    report.require(name, torch.isfinite(logits).all().item(), shape=list(logits.shape))
    return logits


def causal_loss(model, inputs, labels, device, frozen_dtype):
    with forward_context(device, frozen_dtype):
        logits = model(**inputs, use_cache=False).logits
        shifted_logits = logits[:, :-1].float().reshape(-1, logits.shape[-1])
        shifted_labels = labels[:, 1:].reshape(-1)
        return torch.nn.functional.cross_entropy(shifted_logits, shifted_labels)


def validate_model_and_update(
    model,
    reference,
    manifest,
    base_model_factory,
    inputs,
    labels,
    output_directory,
    report,
    device,
    frozen_dtype,
    max_initial_relative_error=0.02,
    max_initial_kl=0.02,
):
    """Run the same checks on the real GPU model and small local test models."""
    expected_trainable = {
        f"{target['name']}.{factor}"
        for target in manifest["target_modules"]
        for factor in ("A", "B")
    }
    actual_trainable = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    report.require("only_expected_A_B_trainable", set(actual_trainable) == expected_trainable,
                   tensor_count=len(actual_trainable),
                   parameter_count=sum(parameter.numel() for parameter in actual_trainable.values()))
    report.require("parameter_precision", all(
        parameter.dtype == (torch.float32 if parameter.requires_grad else frozen_dtype)
        and parameter.device.type == torch.device(device).type
        for parameter in model.parameters()
    ), frozen_dtype=str(frozen_dtype), adapter_dtype="torch.float32")
    initial_buffers = {
        name: buffer.detach().clone()
        for name, buffer in model.named_buffers()
        if name.endswith((".A0", ".B0"))
    }
    report.require("initial_factor_buffers_fp32", all(
        buffer.dtype == torch.float32 for buffer in initial_buffers.values()
    ) and len(initial_buffers) == len(expected_trainable))
    report.require("reference_is_original_frozen_model",
                   not any(isinstance(module, GeoRALinear) for module in reference.modules())
                   and not any(parameter.requires_grad for parameter in reference.parameters()))

    reference_logits = checked_forward(reference, inputs, device, frozen_dtype, report, "reference_forward_finite")
    initial_logits = checked_forward(model, inputs, device, frozen_dtype, report, "initialized_forward_finite")
    differences = logits_difference(reference_logits, initial_logits)
    report.require("initialized_logits_close_to_original",
                   differences["relative_l2_error"] <= max_initial_relative_error
                   and differences["last_token_reference_to_actual_kl_nats"] <= max_initial_kl,
                   **differences,
                   relative_error_limit=max_initial_relative_error,
                   kl_limit_nats=max_initial_kl)

    # Check the initialized checkpoint separately from the checkpoint after training.
    initial_state = export_adapter_state(model)
    save_file(initial_state, str(output_directory / "initial_adapter.safetensors"))
    disk_initial_state = load_file(str(output_directory / "initial_adapter.safetensors"))
    initial_reloaded = restored_model(base_model_factory, disk_initial_state, manifest, device, frozen_dtype)
    initial_reloaded_logits = checked_forward(
        initial_reloaded, inputs, device, frozen_dtype, report, "initial_reloaded_forward_finite"
    )
    report.require("initial_reload_logits_exact", torch.equal(initial_logits, initial_reloaded_logits),
                   **logits_difference(initial_logits, initial_reloaded_logits))
    del initial_reloaded, disk_initial_state, initial_state

    # Keep exact snapshots of all frozen parameters and initial factors on the device.
    frozen_snapshot = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    trainable_snapshot = {name: parameter.detach().clone() for name, parameter in actual_trainable.items()}
    optimizer = torch.optim.AdamW(list(actual_trainable.values()), lr=1e-4, weight_decay=0.0)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = causal_loss(model, inputs, labels, device, frozen_dtype)
    report.require("finite_training_loss", torch.isfinite(loss).item(), loss_before=loss.item())
    loss.backward()

    gradient_measurements = []
    for name, parameter in actual_trainable.items():
        gradient = parameter.grad
        norm = torch.linalg.vector_norm(gradient).item() if gradient is not None else 0.0
        valid = gradient is not None and gradient.dtype == torch.float32
        valid = valid and torch.isfinite(gradient).all().item() and norm > 0
        gradient_measurements.append({"name": name, "valid": bool(valid), "gradient_norm": norm})
    report.require("all_A_B_gradients_finite_fp32_nonzero",
                   all(item["valid"] for item in gradient_measurements),
                   gradients=gradient_measurements)
    report.require("frozen_parameters_receive_no_gradients", all(
        parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad
    ))
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        list(actual_trainable.values()), max_norm=1.0, error_if_nonfinite=True
    )
    optimizer.step()

    change_measurements = []
    for name, parameter in actual_trainable.items():
        change = torch.linalg.vector_norm(parameter.detach() - trainable_snapshot[name]).item()
        change_measurements.append({"name": name, "change_norm": change})
    report.require("all_A_B_parameters_changed", all(
        item["change_norm"] > 0 for item in change_measurements
    ), updates=change_measurements, gradient_norm_before_clipping=gradient_norm.item())
    report.require("updated_A_B_finite", all(
        torch.isfinite(parameter).all().item() for parameter in actual_trainable.values()
    ))
    parameters_after = dict(model.named_parameters())
    report.require("all_frozen_weights_and_biases_exactly_unchanged", all(
        torch.equal(value, parameters_after[name]) for name, value in frozen_snapshot.items()
    ), tensor_count=len(frozen_snapshot))
    buffers_after = dict(model.named_buffers())
    report.require("all_A0_B0_exactly_unchanged", all(
        torch.equal(value, buffers_after[name]) for name, value in initial_buffers.items()
    ))

    optimizer_valid = len(optimizer.state) == len(actual_trainable)
    for parameter in actual_trainable.values():
        state = optimizer.state[parameter]
        optimizer_valid = optimizer_valid and state["step"].item() == 1
        for name in ("exp_avg", "exp_avg_sq"):
            value = state[name]
            optimizer_valid = optimizer_valid and value.dtype == torch.float32
            optimizer_valid = optimizer_valid and torch.isfinite(value).all().item()
    report.require("adamw_moments_finite_fp32_after_one_step", optimizer_valid,
                   parameter_state_count=len(optimizer.state))

    updated_logits = checked_forward(model, inputs, device, frozen_dtype, report, "updated_forward_finite")
    report.require("update_changes_model_output", not torch.equal(initial_logits, updated_logits),
                   **logits_difference(initial_logits, updated_logits))
    loss_after = causal_loss(model, inputs, labels, device, frozen_dtype).detach().item()
    report.require("finite_loss_after_update", math.isfinite(loss_after), loss_after=loss_after)

    # The frozen reference stays the original policy, including after an adapter update.
    reference_after = checked_forward(reference, inputs, device, frozen_dtype, report, "reference_after_update_finite")
    report.require("original_reference_logits_exactly_unchanged", torch.equal(reference_logits, reference_after))

    trained_state = export_adapter_state(model)
    adapter_path = output_directory / "trained_adapter.safetensors"
    save_file(trained_state, str(adapter_path))
    manifest_path = output_directory / "manifest.json"
    trained_manifest = dict(manifest, optimizer_steps=1)
    manifest_path.write_text(json.dumps(trained_manifest, indent=2) + "\n")
    disk_manifest = json.loads(manifest_path.read_text())
    disk_state = load_file(str(adapter_path))
    report.require("trained_factors_exact_after_file_roundtrip", all(
        torch.equal(value, disk_state[name]) for name, value in trained_state.items()
    ))
    reloaded = restored_model(base_model_factory, disk_state, disk_manifest, device, frozen_dtype)
    reloaded_parameters = dict(reloaded.named_parameters())
    report.require("reloaded_frozen_weights_match_trained_model", all(
        torch.equal(value, reloaded_parameters[name])
        for name, value in frozen_snapshot.items()
    ))
    reloaded_logits = checked_forward(reloaded, inputs, device, frozen_dtype, report, "trained_reloaded_forward_finite")
    report.require("trained_reload_logits_exact", torch.equal(updated_logits, reloaded_logits),
                   **logits_difference(updated_logits, reloaded_logits))

    optimizer_path = output_directory / "optimizer.pt"
    torch.save(optimizer.state_dict(), optimizer_path)
    reloaded_optimizer = torch.optim.AdamW(
        [parameter for parameter in reloaded.parameters() if parameter.requires_grad],
        lr=1e-4,
        weight_decay=0.0,
    )
    disk_optimizer_state = torch.load(optimizer_path, map_location="cpu", weights_only=True)
    reloaded_optimizer.load_state_dict(disk_optimizer_state)
    before_state = optimizer.state_dict()
    after_state = reloaded_optimizer.state_dict()
    optimizer_equal = before_state["param_groups"] == after_state["param_groups"]
    optimizer_equal = optimizer_equal and before_state["state"].keys() == after_state["state"].keys()
    for key, state in before_state["state"].items():
        for name, value in state.items():
            optimizer_equal = optimizer_equal and torch.equal(value.cpu(), after_state["state"][key][name].cpu())
    report.require("optimizer_state_exact_after_reload", optimizer_equal)
    report.data.update(optimizer_steps=1, update_loss="short_answer_cross_entropy_mechanical_check")


def run(arguments, report):
    started = time.perf_counter()
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("This check requires a visible ROCm GPU inside a Slurm job.")
    configuration, checkpoint_directory = pinned_configuration()
    manifest = json.loads((arguments.init_dir / "manifest.json").read_text())
    validate_manifest(manifest, configuration)
    state = load_file(str(arguments.init_dir / "adapter.safetensors"))
    report.require("checkpoint_is_untrained_initialization", all(
        torch.equal(state[f"{target['name']}.{factor}"], state[f"{target['name']}.{factor}0"])
        for target in manifest["target_modules"] for factor in ("A", "B")
    ))
    torch.manual_seed(0)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    device = "cuda"
    frozen_dtype = torch.bfloat16
    torch.cuda.reset_peak_memory_stats()

    def base_model_factory():
        return fresh_fp32_model(checkpoint_directory)

    reference = base_model_factory()
    move_with_precision(reference, device, frozen_dtype)
    model = restored_model(base_model_factory, state, manifest, device, frozen_dtype)
    del state

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_directory, local_files_only=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2 + 3? Answer briefly."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode("5", add_special_tokens=False) + [tokenizer.eos_token_id]
    input_ids = torch.tensor([prompt_ids + answer_ids], device=device)
    inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
    labels = input_ids.clone()
    labels[:, :len(prompt_ids)] = -100
    report.data.update(
        stage="geora_gpu_pre_grpo_checks",
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        model_repository=configuration["model_repository"],
        model_revision=configuration["model_revision"],
        torch_version=torch.__version__,
        hip_version=torch.version.hip,
        gpu_name=torch.cuda.get_device_name(0),
        prompt_token_count=len(prompt_ids),
        supervised_answer_token_count=len(answer_ids),
    )
    if arguments.diagnose_only:
        from diagnose_geora_precision import diagnose_initialization
        report.data["stage"] = "geora_initialization_precision_diagnostic"
        diagnose_initialization(
            model, reference, manifest, base_model_factory, inputs, labels,
            arguments.output_dir, report, device,
        )
    else:
        validate_model_and_update(
            model, reference, manifest, base_model_factory, inputs, labels,
            arguments.output_dir, report, device, frozen_dtype,
            arguments.max_initial_relative_error, arguments.max_initial_kl,
        )
    torch.cuda.synchronize()
    report.data.update(
        elapsed_seconds=time.perf_counter() - started,
        peak_allocated_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
    )
    report.finish()
    if arguments.diagnose_only:
        print("GeoRA precision diagnostic completed; optimizer steps: 0.", flush=True)
    else:
        print("All GeoRA GPU pre-GRPO checks passed.", flush=True)
    print("Report:", report.path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnose-only", action="store_true",
                        help="Measure initialization precision without updating parameters.")
    parser.add_argument("--max-initial-relative-error", type=float, default=0.02)
    parser.add_argument("--max-initial-kl", type=float, default=0.02)
    arguments = parser.parse_args()
    for value in (arguments.max_initial_relative_error, arguments.max_initial_kl):
        if not math.isfinite(value) or value < 0:
            parser.error("Initialization error limits must be finite and nonnegative.")
    report = CheckReport(arguments.output_dir, "gpu_checks.json")
    try:
        run(arguments, report)
    except Exception as error:
        report.data.update(status="failed", error=str(error))
        report.write()
        raise


if __name__ == "__main__":
    main()
