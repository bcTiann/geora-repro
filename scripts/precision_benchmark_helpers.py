"""Draft runtime-only helpers for GeoRA precision benchmarking.

Not executed. Model loading and fixed-batch construction deliberately remain
outside this helper. Each precision should run in a separate process.
"""

from __future__ import annotations

import statistics
import time

import torch


def synchronize(device: str) -> None:
    backend = torch.device(device).type
    if backend == "cuda":
        # PyTorch exposes AMD HIP through the same cuda namespace.
        torch.cuda.synchronize(device)
    elif backend == "mps":
        torch.mps.synchronize()


def memory_snapshot(device: str) -> dict:
    backend = torch.device(device).type
    if backend == "cuda":
        return {
            "allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
            "reserved_gib": torch.cuda.memory_reserved(device) / 2**30,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        }
    if backend == "mps":
        # These are snapshots, not a peak estimate.
        return {
            "current_allocated_gib": torch.mps.current_allocated_memory() / 2**30,
            "driver_allocated_gib": torch.mps.driver_allocated_memory() / 2**30,
            "peak_measurement_available": False,
        }
    return {}


@torch.no_grad()
def restore_initial_step(model, optimizer, initial_parameters: dict) -> None:
    """Restore the same first-step parameter and moment values in every trial.

    Warmup creates moment storage. Subsequent trials reset moments in place,
    avoiding a fresh allocation while retaining the real AdamW operation.
    The restoration itself is excluded from timed work.
    """
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            parameter.copy_(initial_parameters[name])
    optimizer.zero_grad(set_to_none=True)
    for state in optimizer.state.values():
        state["exp_avg"].zero_()
        state["exp_avg_sq"].zero_()
        state["step"].zero_()


def train_trial(
    model, optimizer, compute_logps, construct_loss, rollout, device,
    frozen_dtype, old_logp, reference_logp, advantages, *, max_grad_norm=1.0,
) -> dict:
    """One fixed-data forward, GRPO loss, backward, clipping and optimizer step.

    compute_logps is score_completion_logps or an explicit production scorer.
    All probabilities use that same scorer and dtype for this condition.
    old_logp/reference_logp are immutable no-grad tensors outside this timing.
    """
    synchronize(device)
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_memory = memory_snapshot(device)
    started = time.perf_counter()
    current_logp = compute_logps(
        model, rollout, device, frozen_dtype, gradient_enabled=True,
    )
    terms = construct_loss(
        current_logp, old_logp, reference_logp, advantages,
        rollout["completion_mask"], clip_epsilon=0.2, kl_coefficient=0.001,
    )
    synchronize(device)
    forward_finished = time.perf_counter()
    terms["loss"].backward()
    trainable = [p for p in model.parameters() if p.requires_grad]
    grad_norm = torch.nn.utils.clip_grad_norm_(
        trainable, max_grad_norm, error_if_nonfinite=True,
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    update_finished = time.perf_counter()
    result = {
        "forward_and_loss_seconds": forward_finished - started,
        "backward_clip_step_seconds": update_finished - forward_finished,
        "total_seconds": update_finished - started,
        "loss": terms["loss"].item(),
        "gradient_norm_before_clipping": grad_norm.item(),
        "start_memory": start_memory,
        "memory": memory_snapshot(device),
    }
    del terms, current_logp, grad_norm
    return result


def benchmark(
    model, optimizer, compute_logps, construct_loss, rollout, device,
    frozen_dtype, old_logp, reference_logp, advantages, *, warmup=2, repeats=3,
) -> dict:
    """Warm the same shape and repeat exactly the same first update."""
    if warmup < 1 or repeats < 3:
        raise ValueError("Use at least one warmup and three measured trials")
    initial_parameters = {
        name: p.detach().cpu().clone()
        for name, p in model.named_parameters() if p.requires_grad
    }
    measurements = []
    for index in range(warmup + repeats):
        restore_initial_step(model, optimizer, initial_parameters)
        synchronize(device)
        record = train_trial(
            model, optimizer, compute_logps, construct_loss, rollout, device,
            frozen_dtype, old_logp, reference_logp, advantages,
        )
        record["warmup"] = index < warmup
        print({"trial": index, **record}, flush=True)
        if not record["warmup"]:
            measurements.append(record)
    summary = {}
    for key in (
        "forward_and_loss_seconds", "backward_clip_step_seconds", "total_seconds",
    ):
        values = [r[key] for r in measurements]
        summary[key] = {
            "median": statistics.median(values),
            "minimum": min(values),
            "maximum": max(values),
        }
    return {
        "warmup_count": warmup,
        "measurement_count": repeats,
        "optimizer_steps_are_independent_first_step_trials": True,
        "valid_completion_tokens": rollout["completion_mask"].sum().item(),
        "input_shape": list(rollout["input_ids"].shape),
        "fixed_width": rollout["fixed_width"],
        "frozen_dtype": str(frozen_dtype),
        "summary": summary,
        "measurements": measurements,
    }
