"""Save and restore complete small-GRPO state at completed rollout boundaries.

The caller rebuilds the pinned base and the same initialized adapter first.
Restoration copies current FP32 A/B in place; it never reloads a full base or
replaces Parameter objects. Frozen GeoRA A0/B0 must already match the checkpoint.
Reference weights are reconstructed from the pinned reference provenance.

The checkpoint includes adapter tensors, AdamW, the constant LR scheduler,
progress, Python/global Torch RNG and the independent rollout generator RNG.
There are no in-flight rollout samples; saving during a rollout is rejected.
Exact RNG replay requires the same device backend and visible GPU layout.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile

from safetensors.torch import load_file, save_file
import torch


CHECKPOINT_FORMAT = 1
BOUNDARY = "after_rollout_and_optional_update"
PROVENANCE_FIELDS = (
    "method", "model_repository", "model_revision", "dataset_manifest_sha256",
    "training_config_sha256", "initialization_sha256",
)


def canonical_json_sha256(value: dict) -> str:
    """Hash JSON content independent of dictionary insertion order."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _constant_multiplier(step: int) -> float:
    return 1.0


def make_constant_scheduler(optimizer: torch.optim.Optimizer):
    """Create the only scheduler policy supported by these boundary checkpoints."""
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _constant_multiplier)
    scheduler.geora_schedule_kind = "constant_v1"
    return scheduler


def _validate_progress(progress: dict) -> None:
    if progress.get("boundary") != BOUNDARY or progress.get("rollout_in_flight") is not False:
        raise ValueError("Checkpoint must be after a completed rollout, with no in-flight samples.")
    for name in ("completed_optimizer_steps", "completed_rollouts", "next_data_position"):
        value = progress.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Progress {name} must be a nonnegative integer.")
    if progress["completed_optimizer_steps"] > progress["completed_rollouts"]:
        raise ValueError("This trainer supports at most one update per completed rollout.")
    # Confirm all additional progress fields are portable JSON values.
    canonical_json_sha256(progress)


def _validate_provenance(provenance: dict) -> None:
    for name in PROVENANCE_FIELDS:
        if not isinstance(provenance.get(name), str) or not provenance[name]:
            raise ValueError(f"Required nonempty provenance field: {name}.")
    for name in ("dataset_manifest_sha256", "training_config_sha256", "initialization_sha256"):
        value = provenance[name]
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"Provenance {name} must be a lowercase SHA256 digest.")
    canonical_json_sha256(provenance)


def _trainable_parameters(model) -> dict[str, torch.nn.Parameter]:
    parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
    if not parameters or any(not name.endswith((".A", ".B")) for name in parameters):
        raise ValueError("Only named adapter A/B parameters may be trainable.")
    if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in parameters.values()):
        raise ValueError("Trainable adapter factors must be finite FP32.")
    return parameters


def _adapter_tensors(model) -> dict[str, torch.Tensor]:
    tensors = {name: value.detach().cpu().contiguous().clone()
               for name, value in _trainable_parameters(model).items()}
    for name, value in model.named_buffers():
        if name.endswith((".A0", ".B0")):
            if value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError("Initial GeoRA factors must be finite FP32.")
            tensors[name] = value.detach().cpu().contiguous().clone()
    return tensors


def _optimizer_parameter_names(model, optimizer) -> list[list[str]]:
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("This checkpoint format requires AdamW.")
    parameters = _trainable_parameters(model)
    names_by_id = {id(value): name for name, value in parameters.items()}
    groups = []
    seen = set()
    for group in optimizer.param_groups:
        names = []
        for parameter in group["params"]:
            name = names_by_id.get(id(parameter))
            if name is None or name in seen:
                raise ValueError("Optimizer must cover each trainable A/B once, with no frozen parameters.")
            seen.add(name)
            names.append(name)
        groups.append(names)
    if seen != set(parameters):
        raise ValueError("Optimizer is missing trainable A/B parameters.")
    return groups


def _validate_scheduler(scheduler, optimizer, completed_steps: int) -> None:
    if not isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
        raise ValueError("Use make_constant_scheduler for the constant LR policy.")
    if scheduler.optimizer is not optimizer or getattr(scheduler, "geora_schedule_kind", None) != "constant_v1":
        raise ValueError("Scheduler does not have the verified constant policy.")
    if scheduler.last_epoch != completed_steps:
        raise ValueError("Scheduler step count must equal completed optimizer steps.")
    if len(scheduler.lr_lambdas) != len(optimizer.param_groups) or any(
        function is not _constant_multiplier for function in scheduler.lr_lambdas
    ):
        raise ValueError("Scheduler multiplier must be the fixed constant function.")


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _validate_optimizer_state(state: dict, parameter_groups: list[list[str]], tensors: dict,
                              completed_steps: int) -> None:
    """Verify saved AdamW moments and step counters before restoring parameters."""
    saved_groups = state.get("param_groups")
    if not isinstance(saved_groups, list) or len(saved_groups) != len(parameter_groups):
        raise ValueError("Saved optimizer parameter groups are invalid.")
    names_by_identifier = {}
    for saved, names in zip(saved_groups, parameter_groups):
        identifiers = saved.get("params", [])
        if len(identifiers) != len(names):
            raise ValueError("Saved optimizer group parameter count differs.")
        for identifier, name in zip(identifiers, names):
            if identifier in names_by_identifier:
                raise ValueError("Saved optimizer parameter identifiers repeat.")
            names_by_identifier[identifier] = name
    moment_states = state.get("state")
    if not isinstance(moment_states, dict) or any(identifier not in names_by_identifier for identifier in moment_states):
        raise ValueError("Saved optimizer moment names are invalid.")
    for identifier, moments in moment_states.items():
        if not isinstance(moments, dict) or not {"step", "exp_avg", "exp_avg_sq"}.issubset(moments):
            raise ValueError("Saved AdamW moments are incomplete.")
        step = moments["step"]
        if not isinstance(step, torch.Tensor) or step.numel() != 1 or not torch.isfinite(step).all():
            raise ValueError("Saved AdamW step must be a finite scalar tensor.")
        step_number = step.item()
        if step_number < 0 or int(step_number) != step_number or step_number > completed_steps:
            raise ValueError("Saved AdamW step is inconsistent with completed progress.")
        factor_shape = tensors[names_by_identifier[identifier]].shape
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if name not in moments:
                continue
            value = moments[name]
            if not isinstance(value, torch.Tensor) or value.shape != factor_shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError(f"Saved AdamW {name} must be a finite FP32 factor-shaped tensor.")


def _capture_device_rng(device: torch.device) -> dict:
    if device.type == "cpu":
        return {"backend": "cpu", "states": []}
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA/ROCm device RNG requested without visible GPUs.")
        return {"backend": "cuda", "states": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]}
    raise ValueError(f"Unsupported RNG device backend: {device.type}")


def save_boundary_checkpoint(
    directory: Path,
    *,
    model,
    optimizer,
    scheduler,
    rollout_generator: torch.Generator,
    progress: dict,
    provenance: dict,
    device: str | torch.device,
) -> dict:
    """Write a new immutable checkpoint directory; never overwrite an old one.

    Call optimizer.step(), scheduler.step(), and clear gradients before saving.
    progress includes boundary=BOUNDARY, rollout_in_flight=False,
    completed_optimizer_steps, completed_rollouts, and next_data_position.
    Provenance content must match exactly at reload, including extra fields.
    """
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"Checkpoint already exists: {directory}")
    _validate_progress(progress)
    _validate_provenance(provenance)
    _validate_scheduler(scheduler, optimizer, progress["completed_optimizer_steps"])
    parameter_groups = _optimizer_parameter_names(model, optimizer)
    if any(value.grad is not None for value in _trainable_parameters(model).values()):
        raise ValueError("Clear completed-step gradients before saving a boundary checkpoint.")
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Boundary replay currently supports CPU or CUDA/ROCm, not MPS.")
    if torch.device(rollout_generator.device).type != device.type:
        raise ValueError("Rollout generator backend must match the training device.")
    tensors = _adapter_tensors(model)
    optimizer_state = _cpu_tree(optimizer.state_dict())
    _validate_optimizer_state(optimizer_state, parameter_groups, tensors, progress["completed_optimizer_steps"])
    state = {
        "optimizer": optimizer_state,
        "scheduler": _cpu_tree(scheduler.state_dict()),
        "torch_cpu_rng": torch.get_rng_state().clone(),
        "torch_device_rng": _capture_device_rng(device),
        "python_random_rng": random.getstate(),
        "rollout_generator_rng": rollout_generator.get_state().cpu().clone(),
        "rollout_generator_device": str(rollout_generator.device),
    }
    manifest = {
        "training_checkpoint_format": CHECKPOINT_FORMAT,
        "status": "complete",
        "boundary": BOUNDARY,
        "provenance": provenance,
        "provenance_sha256": canonical_json_sha256(provenance),
        "progress": progress,
        "model_training": bool(model.training),
        "optimizer_type": "AdamW",
        "optimizer_parameter_names": parameter_groups,
        "scheduler_kind": "constant_v1",
        "adapter_tensors": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in tensors.items()},
        "runtime": {"torch_version": str(torch.__version__), "hip_version": torch.version.hip, "device_backend": device.type},
    }
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}.writing-", dir=directory.parent))
    try:
        save_file(tensors, str(temporary / "adapter.safetensors"))
        torch.save(state, temporary / "training_state.pt")
        manifest["files"] = {name: {"sha256": file_sha256(temporary / name), "bytes": (temporary / name).stat().st_size}
                             for name in ("adapter.safetensors", "training_state.pt")}
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.rename(temporary, directory)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def load_boundary_checkpoint(
    directory: Path,
    *,
    model,
    optimizer,
    scheduler,
    rollout_generator: torch.Generator,
    expected_provenance: dict,
    device: str | torch.device,
) -> dict:
    """Validate all recorded identity/shape/order fields, then restore in place.

    The caller's model already has the pinned original frozen weights and the
    same initial adapter. RNG restoration runs last, so construction/checking
    cannot consume the next training random draw. Return the saved progress.
    """
    directory = Path(directory)
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Boundary replay currently supports CPU or CUDA/ROCm, not MPS.")
    _validate_provenance(expected_provenance)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("training_checkpoint_format") != CHECKPOINT_FORMAT or manifest.get("status") != "complete":
        raise ValueError("Unsupported or incomplete training checkpoint.")
    if manifest.get("boundary") != BOUNDARY or manifest.get("scheduler_kind") != "constant_v1":
        raise ValueError("Checkpoint boundary/scheduler policy differs.")
    _validate_progress(manifest["progress"])
    if manifest["provenance"] != expected_provenance or manifest["provenance_sha256"] != canonical_json_sha256(expected_provenance):
        raise ValueError("Checkpoint provenance differs from the current run.")
    parameter_groups = _optimizer_parameter_names(model, optimizer)
    if manifest["optimizer_parameter_names"] != parameter_groups:
        raise ValueError("Optimizer parameter order differs from the checkpoint.")
    if getattr(scheduler, "geora_schedule_kind", None) != "constant_v1" or scheduler.optimizer is not optimizer:
        raise ValueError("Use make_constant_scheduler for the current optimizer before reload.")
    _validate_scheduler(scheduler, optimizer, scheduler.last_epoch)
    if manifest["runtime"]["device_backend"] != device.type:
        raise ValueError("Exact RNG replay requires the same device backend.")
    for name in ("adapter.safetensors", "training_state.pt"):
        record = manifest["files"].get(name)
        path = directory / name
        if record is None or path.stat().st_size != record["bytes"] or file_sha256(path) != record["sha256"]:
            raise ValueError(f"Checkpoint file size/hash mismatch: {name}")
    tensors = load_file(str(directory / "adapter.safetensors"), device="cpu")
    expected_tensors = _adapter_tensors(model)
    if set(tensors) != set(expected_tensors) or set(tensors) != set(manifest["adapter_tensors"]):
        raise ValueError("Checkpoint adapter tensor names differ.")
    for name, value in tensors.items():
        descriptor = manifest["adapter_tensors"][name]
        if value.dtype != torch.float32 or list(value.shape) != descriptor["shape"] or descriptor["dtype"] != "torch.float32":
            raise ValueError(f"Invalid adapter tensor shape or dtype: {name}")
        if value.shape != expected_tensors[name].shape or not torch.isfinite(value).all():
            raise ValueError(f"Adapter shape mismatch or non-finite factor: {name}")
        if name.endswith((".A0", ".B0")) and not torch.equal(value, expected_tensors[name]):
            raise ValueError(f"Frozen initialization differs: {name}")
    state = torch.load(directory / "training_state.pt", map_location="cpu", weights_only=True)
    _validate_optimizer_state(state["optimizer"], parameter_groups, tensors, manifest["progress"]["completed_optimizer_steps"])
    if state["scheduler"].get("geora_schedule_kind") != "constant_v1" or state["scheduler"]["last_epoch"] != manifest["progress"]["completed_optimizer_steps"]:
        raise ValueError("Saved scheduler policy or completed step count differs.")
    saved_groups = state["optimizer"]["param_groups"]
    if len(saved_groups) != len(optimizer.param_groups):
        raise ValueError("Optimizer group count differs.")
    for saved, current in zip(saved_groups, optimizer.param_groups):
        saved_settings = {key: value for key, value in saved.items() if key != "params"}
        current_settings = {key: value for key, value in current.items() if key != "params"}
        if saved_settings != current_settings or len(saved["params"]) != len(current["params"]):
            raise ValueError("Optimizer hyperparameters or parameter count differ.")
    if state["torch_device_rng"]["backend"] != device.type or torch.device(state["rollout_generator_device"]).type != device.type:
        raise ValueError("Recorded RNG backend differs.")
    if torch.device(rollout_generator.device).type != device.type:
        raise ValueError("Current rollout generator backend differs.")
    device_states = state["torch_device_rng"]["states"]
    expected_count = torch.cuda.device_count() if device.type == "cuda" else 0
    if len(device_states) != expected_count:
        raise ValueError("Visible device RNG layout differs.")
    # Validate RNG payloads with private generators before mutating real state.
    torch.Generator().set_state(state["torch_cpu_rng"])
    if device.type == "cuda":
        for index, rng_state in enumerate(device_states):
            torch.Generator(device=f"cuda:{index}").set_state(rng_state)
    torch.Generator(device=rollout_generator.device).set_state(state["rollout_generator_rng"])
    random.Random().setstate(state["python_random_rng"])

    with torch.no_grad():
        for name, parameter in _trainable_parameters(model).items():
            parameter.copy_(tensors[name].to(device=parameter.device))
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    optimizer.zero_grad(set_to_none=True)
    model.train(manifest["model_training"])
    torch.set_rng_state(state["torch_cpu_rng"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(device_states)
    random.setstate(state["python_random_rng"])
    rollout_generator.set_state(state["rollout_generator_rng"])
    return dict(manifest["progress"])
