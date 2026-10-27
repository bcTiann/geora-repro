"""Observe stock single-learner checkpoint restoration without replacing it.

Only checkpoints produced by this bounded diagnostic are loaded. Tensor values,
optimizer/scheduler state, learner RNG and Trainer's data-skip state are checked
after their real load hooks. Server RNG is deliberately outside this contract.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import random
import struct

import numpy as np
import torch


def _frame(value: bytes) -> bytes:
    return len(value).to_bytes(8, "big") + value


def _digest_value(value) -> bytes:
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise TypeError("Only dense strided checkpoint tensors are supported")
        tensor = value.detach().cpu().contiguous()
        header = json.dumps([str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return b"tensor" + _frame(header) + _frame(raw)
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object arrays are not checkpoint state")
        header = json.dumps([value.dtype.str, list(value.shape)], separators=(",", ":")).encode()
        return b"ndarray" + _frame(header) + _frame(np.ascontiguousarray(value).tobytes())
    if isinstance(value, np.generic):
        return b"numpy_scalar" + _digest_value(np.asarray(value))
    if value is None:
        return b"none"
    if isinstance(value, bool):
        return b"true" if value else b"false"
    if isinstance(value, int):
        return b"int" + _frame(str(value).encode())
    if isinstance(value, float):
        return b"float" + struct.pack(">d", value)
    if isinstance(value, str):
        return b"str" + _frame(value.encode("utf-8"))
    if isinstance(value, bytes):
        return b"bytes" + _frame(value)
    if isinstance(value, Mapping):
        pairs = sorted((_digest_value(key), _digest_value(item)) for key, item in value.items())
        return b"mapping" + b"".join(_frame(key) + _frame(item) for key, item in pairs)
    if isinstance(value, (list, tuple)):
        label = b"list" if isinstance(value, list) else b"tuple"
        return label + b"".join(_frame(_digest_value(item)) for item in value)
    raise TypeError(f"Unsupported checkpoint state type: {type(value).__qualname__}")


def state_digest(value) -> str:
    """Device-independent, typed, order-stable digest of nested checkpoint state."""
    return hashlib.sha256(_digest_value(value)).hexdigest()


def load_torch(path):
    """Use the pinned Trainer's NumPy allowlist, never unrestricted pickle."""
    from transformers.trainer import safe_globals

    with safe_globals():
        return torch.load(Path(path), map_location="cpu", weights_only=True)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_inventory(path, variant) -> dict:
    path = Path(path)
    if variant not in ("geora_mixed", "geora", "lora"):
        raise ValueError(f"Unknown checkpoint variant: {variant}")
    names = ["trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"]
    names += (["adapter_model.safetensors", "adapter_config.json"] if variant == "lora"
              else ["geora.safetensors", "geora_config.json"])
    for name in names:
        if not (path / name).is_file() or (path / name).stat().st_size == 0:
            raise ValueError(f"Incomplete checkpoint: {name}")
    state = json.loads((path / "trainer_state.json").read_text())
    if not isinstance(state.get("global_step"), int) or state["global_step"] < 0:
        raise ValueError("Invalid checkpoint global_step")
    return {"checkpoint": str(path.resolve()), "variant": variant,
            "global_step": state["global_step"],
            "files_sha256": {name: _file_digest(path / name) for name in names}}


def verify_loaded_policy(model, checkpoint, variant) -> dict:
    """Compare saved adapter/residual/bias values with the actual loaded model."""
    from safetensors.torch import load_file

    checkpoint = Path(checkpoint)
    if variant == "lora":
        from peft.utils.save_and_load import get_peft_model_state_dict

        expected = load_file(str(checkpoint / "adapter_model.safetensors"), device="cpu")
        actual = get_peft_model_state_dict(model, save_embedding_layers=False)
    elif variant in ("geora_mixed", "geora"):
        expected = load_file(str(checkpoint / "geora.safetensors"), device="cpu")
        actual = {}
        for name, layer in model.named_modules():
            if all(hasattr(layer, key) for key in ("W_res", "A", "B", "scaling")):
                prefix = name + "." if name else ""
                actual.update({prefix + key: value for key, value in layer.state_dict().items()})
    else:
        raise ValueError(f"Unknown checkpoint variant: {variant}")
    if not expected or set(expected) != set(actual):
        raise ValueError("Loaded policy tensor coverage differs from the saved checkpoint")
    for name, value in expected.items():
        loaded = actual[name].detach().cpu()
        if value.dtype != loaded.dtype or value.shape != loaded.shape or not torch.equal(value, loaded):
            raise ValueError(f"Loaded policy differs bitwise from the checkpoint: {name}")
    return {"bitwise_equal": True, "tensor_count": len(expected),
            "state_sha256": state_digest(actual)}


def _learner_rng(expected):
    actual = {"python": random.getstate(), "numpy": np.random.get_state(),
              "cpu": torch.random.get_rng_state()}
    if "cuda" in expected:
        if not torch.cuda.is_available():
            raise ValueError("CUDA/HIP RNG was saved but no CUDA/HIP device is available")
        actual["cuda"] = (torch.cuda.random.get_rng_state_all() if isinstance(expected["cuda"], list)
                          else torch.cuda.random.get_rng_state())
    if set(actual) != set(expected):
        raise ValueError(f"Unsupported learner RNG keys: {sorted(set(expected)-set(actual))}")
    return actual


class ResumeAuditMixin:
    """Put before the existing Trainer class in the diagnostic class MRO.

    Set ``resume_audit_dir`` and ``resume_audit_variant`` before ``train``.
    Each override delegates to stock behavior first and then verifies what was
    really restored. It does not assign any replacement state to the Trainer.
    """

    def _resume_record(self, kind, checkpoint, **values):
        record = {"kind": kind, "checkpoint": str(Path(checkpoint).resolve()), "passed": True, **values}
        if not hasattr(self, "resume_audit_records"):
            self.resume_audit_records = []
        self.resume_audit_records.append(record)
        destination = Path(self.resume_audit_dir)
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / "resume_audit.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    def _resume_model(self, model=None):
        current = self.model if model is None else model
        return self.accelerator.unwrap_model(current)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        result = super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        policy = verify_loaded_policy(self._resume_model(model), resume_from_checkpoint, self.resume_audit_variant)
        self._resume_record("policy", resume_from_checkpoint, **policy)
        return result

    def _load_optimizer_and_scheduler(self, checkpoint):
        result = super()._load_optimizer_and_scheduler(checkpoint)
        if checkpoint is None:
            return result
        for kind, name, value in (("optimizer", "optimizer.pt", self.optimizer),
                                  ("scheduler", "scheduler.pt", self.lr_scheduler)):
            expected = load_torch(Path(checkpoint) / name)
            digest = state_digest(value.state_dict())
            if digest != state_digest(expected):
                raise ValueError(f"Restored {kind} differs from the checkpoint")
            self._resume_record(kind, checkpoint, state_sha256=digest)
        return result

    def _load_rng_state(self, checkpoint):
        expected = load_torch(Path(checkpoint) / "rng_state.pth") if checkpoint is not None else None
        result = super()._load_rng_state(checkpoint)
        if expected is None:
            return result
        actual = _learner_rng(expected)
        if state_digest(actual) != state_digest(expected):
            raise ValueError("Restored learner RNG differs from the checkpoint")
        self._resume_record("rng", checkpoint, state_sha256=state_digest(actual),
                            keys=sorted(actual), server_rng_restored=False)
        return result

    def _init_training_state(self, max_steps, num_update_steps_per_epoch, num_train_epochs,
                             resume_from_checkpoint, trial):
        result = super()._init_training_state(max_steps, num_update_steps_per_epoch, num_train_epochs,
                                               resume_from_checkpoint, trial)
        if resume_from_checkpoint is None:
            return result
        saved = json.loads((Path(resume_from_checkpoint) / "trainer_state.json").read_text())
        step = saved["global_step"]
        if self.args.ignore_data_skip:
            raise ValueError("This acceptance requires Trainer's actual data skip on resume")
        expected = (step // num_update_steps_per_epoch,
                    (step % num_update_steps_per_epoch) * self.args.gradient_accumulation_steps)
        if self.state.global_step != step or tuple(result) != expected:
            raise ValueError("Trainer global step or resumed data skip differs from the saved boundary")
        every = self.args.steps_per_generation * self.num_iterations
        if (step * self.args.gradient_accumulation_steps) % every:
            raise ValueError("This checkpoint boundary would require unsaved GRPO buffered completions")
        self._resume_record("training_state", resume_from_checkpoint, global_step=step,
                            epochs_trained=result[0], microbatches_skipped=result[1],
                            grpo_buffer_boundary_aligned=True,
                            data_order_verified_here=False)
        return result

    def resume_audit_summary(self, require_complete=True):
        records = getattr(self, "resume_audit_records", [])
        required = {"policy", "optimizer", "scheduler", "rng", "training_state"}
        missing = sorted(required - {row["kind"] for row in records})
        checkpoints = sorted({row["checkpoint"] for row in records})
        if require_complete and (missing or len(checkpoints) != 1):
            raise ValueError(f"Incomplete resume audit: missing={missing}, checkpoints={checkpoints}")
        return {"passed": not missing and len(checkpoints) == 1, "missing": missing,
                "checkpoints": checkpoints, "records": records,
                "scope": "Policy, optimizer, scheduler, learner RNG and stock Trainer skip; server sampling RNG not restored"}
