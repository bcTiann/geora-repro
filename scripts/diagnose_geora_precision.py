"""Measure initialization differences without SVD, training, or tolerance changes."""

import json

import torch

from geora_check_utils import forward_context, inference_logits, logits_difference
from geora_layers import export_adapter_state, load_geora_state


def relative_error(reference, actual):
    reference = reference.float()
    actual = actual.float()
    denominator = torch.linalg.vector_norm(reference).clamp_min(1e-12)
    return (torch.linalg.vector_norm(actual - reference) / denominator).item()


def position_metrics(reference, actual, labels):
    """Logits at t predict the label at t+1, including the supervised answer."""
    reference_logp = reference.double().log_softmax(dim=-1)
    actual_logp = actual.double().log_softmax(dim=-1)
    reference_p = reference_logp.exp()
    actual_p = actual_logp.exp()
    kl = (reference_p * (reference_logp - actual_logp)).sum(dim=-1)
    tv = (reference_p - actual_p).abs().sum(dim=-1) / 2
    labels = labels.detach().cpu()
    rows = []
    for batch in range(reference.shape[0]):
        for position in range(reference.shape[1]):
            target = labels[batch, position + 1].item() if position + 1 < labels.shape[1] else -100
            row = {
                "batch": batch,
                "logit_position": position,
                "kl_nats": kl[batch, position].item(),
                "total_variation": tv[batch, position].item(),
                "reference_top_token": reference[batch, position].argmax().item(),
                "actual_top_token": actual[batch, position].argmax().item(),
                "predicts_supervised_answer": target != -100,
            }
            if target != -100:
                row.update(
                    target_token=target,
                    reference_target_probability=reference_p[batch, position, target].item(),
                    actual_target_probability=actual_p[batch, position, target].item(),
                )
            rows.append(row)
    return rows


def diagnose_initialization(
    model, reference, manifest, base_model_factory, inputs, labels,
    output_directory, report, device,
):
    # This function accepts the current BF16-frozen / FP32-adapter models.
    # It never changes factors or performs an optimizer step.
    diagnostic = {"optimizer_steps": 0, "layers": []}
    path = output_directory / "precision_diagnostics.json"
    report.data.update(optimizer_steps=0, diagnostic_path=str(path))

    def persist():
        path.write_text(json.dumps(diagnostic, indent=2) + "\n")

    reference_modules = dict(reference.named_modules())
    model_modules = dict(model.named_modules())
    recorded_reference = {}
    handles = []

    def capture_reference(name):
        def hook(module, arguments, output):
            # Keep the exact same input to distinguish local rounding from drift
            # inherited from preceding layers. The short prompt bounds this memory.
            recorded_reference[name] = (arguments[0].detach().clone(), output.detach().clone())
        return hook

    for target in manifest["target_modules"]:
        name = target["name"]
        handles.append(reference_modules[name].register_forward_hook(capture_reference(name)))
    try:
        reference_logits = inference_logits(reference, inputs, device, torch.bfloat16)
    finally:
        for handle in handles:
            handle.remove()
    report.require("diagnostic_bf16_reference_finite", torch.isfinite(reference_logits).all().item())

    def compare_layer(name):
        def hook(module, arguments, output):
            reference_input, reference_output = recorded_reference.pop(name)
            # Call forward directly to avoid recursively invoking this hook.
            with forward_context(device, torch.bfloat16):
                same_input_output = module.forward(reference_input)
            with torch.autocast(device_type=torch.device(device).type, enabled=False):
                original_weight = reference_modules[name].weight.float()
                # F has already been rounded to BF16. Reconstructing this effective
                # weight in FP32 reveals storage error, separate from forward error.
                effective_weight = module.base_layer.weight.float() + module.scaling * (module.B.float() @ module.A.float())
                weight_error = relative_error(original_weight, effective_weight)
            row = {
                "name": name,
                "same_input_output_relative_error": relative_error(reference_output, same_input_output),
                "propagated_output_relative_error": relative_error(reference_output, output),
                "bf16_residual_effective_weight_relative_error": weight_error,
            }
            diagnostic["layers"].append(row)
            print("LAYER:", json.dumps(row), flush=True)
            persist()
        return hook

    handles = []
    for target in manifest["target_modules"]:
        name = target["name"]
        handles.append(model_modules[name].register_forward_hook(compare_layer(name)))
    try:
        actual_logits = inference_logits(model, inputs, device, torch.bfloat16)
    finally:
        for handle in handles:
            handle.remove()
        recorded_reference.clear()
    report.require("diagnostic_bf16_initialized_finite", torch.isfinite(actual_logits).all().item())
    diagnostic["bf16_logits"] = logits_difference(reference_logits, actual_logits)
    diagnostic["bf16_positions"] = position_metrics(reference_logits, actual_logits, labels)
    print("BF16 LOGITS:", json.dumps(diagnostic["bf16_logits"]), flush=True)
    for row in diagnostic["bf16_positions"]:
        if row["predicts_supervised_answer"]:
            print("BF16 ANSWER POSITION:", json.dumps(row), flush=True)
    persist()

    # A BF16 F upcast to FP32 would retain its lost bits. Instead rebuild fresh F
    # from original weights and saved A0/B0, with no SVD, before the FP32 comparison.
    state = export_adapter_state(model)
    fp32_model = base_model_factory()
    load_geora_state(fp32_model, state, manifest)
    fp32_model.to(device=device)
    # The original checkpoint is BF16, so upcasting its original frozen model is exact.
    reference.to(dtype=torch.float32)
    fp32_reference_logits = inference_logits(reference, inputs, device, torch.float32)
    fp32_actual_logits = inference_logits(fp32_model, inputs, device, torch.float32)
    report.require("diagnostic_fp32_reference_finite", torch.isfinite(fp32_reference_logits).all().item())
    report.require("diagnostic_fp32_initialized_finite", torch.isfinite(fp32_actual_logits).all().item())
    diagnostic["fp32_logits"] = logits_difference(fp32_reference_logits, fp32_actual_logits)
    diagnostic["fp32_positions"] = position_metrics(fp32_reference_logits, fp32_actual_logits, labels)
    diagnostic["reference_bf16_vs_fp32_logits"] = logits_difference(fp32_reference_logits, reference_logits)
    print("FP32 LOGITS:", json.dumps(diagnostic["fp32_logits"]), flush=True)
    print("REFERENCE BF16 VS FP32:", json.dumps(diagnostic["reference_bf16_vs_fp32_logits"]), flush=True)
    diagnostic["status"] = "completed"
    persist()
    print("Precision diagnostic saved:", path, flush=True)
