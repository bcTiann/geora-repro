"""Shared file, precision, and report operations for reproduction checks."""

from contextlib import nullcontext
import json
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIRECTORY))

from geora_layers import GeoRALinear, PRECISION_POLICY


def pinned_configuration() -> tuple[dict, Path]:
    configuration = json.loads(
        (PROJECT_DIRECTORY / "configs/base_model.json").read_text()
    )
    checkpoint_directory = PROJECT_DIRECTORY / configuration["checkpoint_directory"]
    record_path = checkpoint_directory / ".cache/huggingface/download/model.safetensors.metadata"
    if record_path.read_text().splitlines()[0] != configuration["model_revision"]:
        raise RuntimeError("The base checkpoint revision does not match the configuration.")
    return configuration, checkpoint_directory


def fresh_fp32_model(checkpoint_directory: Path, *, disable_mmap: bool = False) -> torch.nn.Module:
    # Preserve the former default call; pass the new loader option only when requested.
    loading_options = {"disable_mmap": True} if disable_mmap else {}
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_directory,
        dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
        **loading_options,
    )
    model.requires_grad_(False)
    model.eval()
    return model


def validate_manifest(manifest: dict, configuration: dict) -> None:
    for key in ("model_repository", "model_revision"):
        if manifest[key] != configuration[key]:
            raise RuntimeError(f"Initialization {key} differs from the pinned base.")
    for key, value in configuration["initialization"].items():
        if manifest[key] != value:
            raise RuntimeError(f"Initialization {key} differs from the configuration.")
    if manifest["precision_policy"] != PRECISION_POLICY:
        raise RuntimeError("Initialization precision policy differs from this implementation.")
    expected_names = {
        f"model.layers.{layer}.{projection}"
        for layer in range(28)
        for projection in configuration["target_projections"]
    }
    actual_names = [target["name"] for target in manifest["target_modules"]]
    if len(actual_names) != 196 or set(actual_names) != expected_names:
        raise RuntimeError("Initialization does not contain exactly the expected 196 targets.")


def move_with_precision(model: torch.nn.Module, device: str, frozen_dtype: torch.dtype) -> None:
    # Cast only frozen parameters; A/B and initial A0/B0 retain FP32 storage.
    with torch.no_grad():
        for parameter in model.parameters():
            if not parameter.requires_grad:
                parameter.data = parameter.data.to(dtype=frozen_dtype)
    model.to(device=device)


def forward_context(device: str, frozen_dtype: torch.dtype):
    if frozen_dtype == torch.bfloat16:
        return torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16)
    return nullcontext()


def inference_logits(model, inputs, device, frozen_dtype) -> torch.Tensor:
    model.eval()
    with torch.inference_mode(), forward_context(device, frozen_dtype):
        return model(**inputs, use_cache=False).logits.detach().float().cpu()


def logits_difference(reference: torch.Tensor, actual: torch.Tensor) -> dict:
    difference = actual - reference
    denominator = torch.linalg.vector_norm(reference).clamp_min(1e-12)
    # Compute distribution metrics at every position. The final EOS position alone
    # can be almost deterministic and hide differences earlier in the sequence.
    reference_logp = reference.double().log_softmax(dim=-1)
    actual_logp = actual.double().log_softmax(dim=-1)
    reference_p = reference_logp.exp()
    actual_p = actual_logp.exp()
    kl = (reference_p * (reference_logp - actual_logp)).sum(dim=-1)
    tv = (reference_p - actual_p).abs().sum(dim=-1) / 2
    centered_reference = reference - reference.mean(dim=-1, keepdim=True)
    centered_actual = actual - actual.mean(dim=-1, keepdim=True)
    centered_denominator = torch.linalg.vector_norm(centered_reference).clamp_min(1e-12)
    return {
        "relative_l2_error": (torch.linalg.vector_norm(difference) / denominator).item(),
        "max_absolute_error": difference.abs().max().item(),
        "centered_relative_l2_error": (
            torch.linalg.vector_norm(centered_actual - centered_reference) / centered_denominator
        ).item(),
        "last_token_reference_to_actual_kl_nats": kl[:, -1].mean().item(),
        "all_positions_mean_kl_nats": kl.mean().item(),
        "all_positions_max_kl_nats": kl.max().item(),
        "all_positions_mean_total_variation": tv.mean().item(),
        "all_positions_max_total_variation": tv.max().item(),
        "top1_agreement_fraction": (
            reference.argmax(dim=-1) == actual.argmax(dim=-1)
        ).double().mean().item(),
    }



class CheckReport:
    """Persist every completed check so a failure leaves an inspectable report."""

    def __init__(self, directory: Path, filename: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / filename
        self.data = {"status": "running", "checks": []}

    def write(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2) + "\n")

    def require(self, name: str, condition: bool, **measurements) -> None:
        self.data["checks"].append({"name": name, "passed": bool(condition), **measurements})
        if not condition:
            self.data["status"] = "failed"
        self.write()
        print(f"{'PASS' if condition else 'FAIL'}: {name}", flush=True)
        # Keep long per-parameter arrays in JSON, but print small measurements live.
        printable = {
            key: value for key, value in measurements.items()
            if isinstance(value, (str, int, float, bool)) or value is None
            or (isinstance(value, list) and len(value) <= 8
                and all(isinstance(item, (str, int, float, bool)) for item in value))
        }
        if printable:
            print(json.dumps(printable, ensure_ascii=False), flush=True)
        if not condition:
            raise RuntimeError(f"Check failed: {name}; see {self.path}")

    def finish(self) -> None:
        self.data["status"] = "passed"
        self.write()
