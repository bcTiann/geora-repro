"""GeoRA initialization and compact adapter checkpoints for Qwen projections.

Initialization runs on CPU in FP32. The local correctness run also uses FP32
for the frozen model. For GPU mixed precision, cast only frozen base layers to
BF16 and keep A, B, A0, and B0 in FP32. Apply BF16 autocast in the training loop.
"""

from collections.abc import Callable, Iterator, Mapping
import math
import re

import torch
from torch import nn
from torch.nn import functional


TARGET_PATTERN = re.compile(
    r"(?:^|\.)layers\.\d+\."
    r"(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)$"
)

PRECISION_POLICY = {
    "initialization_device": "cpu",
    "mask_dtype": "float32",
    "svd_dtype": "float32",
    "initial_factor_dtype": "float32",
    "residual_calculation_dtype": "float32",
    "local_frozen_model_dtype": "float32",
    "training_frozen_model_dtype": "bfloat16",
    "trainable_adapter_dtype": "float32",
    "optimizer_state_dtype": "float32",
    "training_forward": "bfloat16_autocast",
}


def _validate_rank(rank: int, shape: tuple[int, int]) -> None:
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise TypeError("rank must be an integer")
    if rank < 1 or rank > min(shape):
        raise ValueError(f"rank={rank} is invalid for weight shape {shape}")


def _validate_ratio(sparsity_ratio: float) -> None:
    if not math.isfinite(sparsity_ratio) or not 0 <= sparsity_ratio <= 1:
        raise ValueError("sparsity_ratio must be finite and between 0 and 1")


def _validate_forward_mode(forward_mode: str) -> None:
    if forward_mode not in ("residual", "difference"):
        raise ValueError("forward_mode must be 'residual' or 'difference'")


@torch.no_grad()
def initialize_geo_factors(
    weight: torch.Tensor,
    rank: int,
    sparsity_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Return A0 [r, input], B0 [output, r], and initialization statistics.

    sparsity_ratio is the quantile used independently by each mask. It does
    not specify the final fraction after their union; ties can increase it.
    Both SVDs are exact, reduced SVDs in FP32, with descending singular values.
    """
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("weight must be a real floating-point matrix")
    _validate_rank(rank, tuple(weight.shape))
    _validate_ratio(sparsity_ratio)
    W_pre = weight.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(W_pre).all():
        raise ValueError("weight contains non-finite values")

    # First SVD: reconstruct the leading rank-r part of the original weight.
    U_pre, S_pre, Vh_pre = torch.linalg.svd(W_pre, full_matrices=False)
    W_pre_rank_r = (U_pre[:, :rank] * S_pre[:rank]) @ Vh_pre[:rank, :]
    spec_magnitudes = W_pre_rank_r.abs()
    threshold_spec = torch.quantile(
        spec_magnitudes.reshape(-1),
        sparsity_ratio,
        interpolation="linear",
    )
    M_spec = spec_magnitudes <= threshold_spec
    del U_pre, S_pre, Vh_pre, W_pre_rank_r, spec_magnitudes

    # Select small original entries separately, then retain their union.
    euc_magnitudes = W_pre.abs()
    threshold_euc = torch.quantile(
        euc_magnitudes.reshape(-1),
        sparsity_ratio,
        interpolation="linear",
    )
    M_euc = euc_magnitudes <= threshold_euc
    M_geo = M_spec | M_euc
    W_geo = W_pre * M_geo
    stats = {
        "spectral_threshold": threshold_spec.item(),
        "euclidean_threshold": threshold_euc.item(),
        "spectral_keep_fraction": M_spec.float().mean().item(),
        "euclidean_keep_fraction": M_euc.float().mean().item(),
        "union_keep_fraction": M_geo.float().mean().item(),
    }
    del euc_magnitudes, M_spec, M_euc, M_geo

    # Second SVD: split the rank-r approximation of W_geo into B0 @ A0.
    U_geo, S_geo, Vh_geo = torch.linalg.svd(W_geo, full_matrices=False)
    sqrt_S_geo_r = S_geo[:rank].sqrt()
    B0 = (U_geo[:, :rank] * sqrt_S_geo_r).contiguous()
    A0 = (sqrt_S_geo_r[:, None] * Vh_geo[:rank, :]).contiguous()
    total_geo_energy = S_geo.square().sum().item()
    retained_geo_energy = S_geo[:rank].square().sum().item()
    stats["w_geo_retained_energy"] = (
        retained_geo_energy / total_geo_energy if total_geo_energy else 0.0
    )
    # This tests factor reconstruction; the frozen residual is checked below.
    W_geo_rank_r = B0 @ A0
    approximation_error = torch.linalg.vector_norm(W_geo - W_geo_rank_r)
    stats["w_geo_relative_approximation_error"] = (
        approximation_error.item() / math.sqrt(total_geo_energy)
        if total_geo_energy else 0.0
    )
    return A0, B0, stats


class GeoRALinear(nn.Module):
    """Frozen base projection and trainable low-rank factors.

    For x [..., input], A/A0 [rank, input], and B/B0 [output, rank],
    the default residual mode stores F = W_pre - scaling * B0 @ A0
    in base_layer.weight and computes F x + scaling * B(Ax).

    Difference mode keeps W_pre in base_layer.weight and computes the native
    base output plus scaling * (B(Ax) - B0(A0x)). The two low-rank branches
    and their difference use FP32 with autocast disabled; the correction is
    cast to the base output dtype before addition. A0/B0 are frozen FP32
    buffers, but their branch remains differentiable with respect to x.
    Both modes retain FP32 A/B parameters and use the same compact factors.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        A0: torch.Tensor,
        B0: torch.Tensor,
        scaling: float,
        *,
        forward_mode: str = "residual",
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("base_layer must be torch.nn.Linear")
        _validate_forward_mode(forward_mode)
        if not math.isfinite(scaling) or scaling <= 0:
            raise ValueError("scaling must be finite and positive")
        if A0.ndim != 2 or B0.ndim != 2:
            raise ValueError("A0 and B0 must be matrices")
        if not A0.is_floating_point() or not B0.is_floating_point():
            raise ValueError("A0 and B0 must be real floating-point matrices")
        rank = A0.shape[0]
        expected_A_shape = (rank, base_layer.in_features)
        expected_B_shape = (base_layer.out_features, rank)
        shapes_match = (
            tuple(A0.shape) == expected_A_shape
            and tuple(B0.shape) == expected_B_shape
        )
        if not shapes_match:
            raise ValueError(
                f"Expected A0 {expected_A_shape} and B0 {expected_B_shape}, "
                f"got {tuple(A0.shape)} and {tuple(B0.shape)}"
            )
        _validate_rank(rank, tuple(base_layer.weight.shape))
        if not torch.isfinite(A0).all() or not torch.isfinite(B0).all():
            raise ValueError("initial factors contain non-finite values")

        self.base_layer = base_layer.to(dtype=torch.float32)
        self.base_layer.requires_grad_(False)
        self.scaling = float(scaling)
        self.forward_mode = forward_mode
        factor_device = self.base_layer.weight.device
        self.register_buffer(
            "A0", A0.detach().to(device=factor_device, dtype=torch.float32).clone()
        )
        self.register_buffer(
            "B0", B0.detach().to(device=factor_device, dtype=torch.float32).clone()
        )
        self.A = nn.Parameter(self.A0.clone())
        self.B = nn.Parameter(self.B0.clone())
        if self.forward_mode == "residual":
            with torch.no_grad():
                initial_adapter_weight = self.B0 @ self.A0
                residual_weight = self.base_layer.weight - self.scaling * initial_adapter_weight
                self.base_layer.weight.copy_(residual_weight)
        self.train(base_layer.training)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frozen_output = self.base_layer(x)
        if self.forward_mode == "difference":
            with torch.autocast(device_type=x.device.type, enabled=False):
                input_fp32 = x.to(dtype=torch.float32)
                compressed_input = functional.linear(input_fp32, self.A)
                adapter_output = functional.linear(compressed_input, self.B)
                initial_compressed_input = functional.linear(input_fp32, self.A0)
                initial_adapter_output = functional.linear(initial_compressed_input, self.B0)
                correction_fp32 = self.scaling * (adapter_output - initial_adapter_output)
            correction = correction_fp32.to(dtype=frozen_output.dtype)
            return frozen_output + correction
        compressed_input = functional.linear(x, self.A)
        adapter_output = functional.linear(compressed_input, self.B)
        return frozen_output + self.scaling * adapter_output


def iter_target_linears(model: nn.Module) -> Iterator[tuple[str, nn.Linear]]:
    """Yield only seven attention/MLP projections within decoder layers."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and TARGET_PATTERN.search(name):
            yield name, module


@torch.no_grad()
def install_geora(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 32,
    rho: float = 0.2,
    progress_callback: Callable[[int, int, str, dict], None] | None = None,
) -> dict:
    """Initialize each target in turn; return its checkpoint manifest.

    progress_callback receives (completed_count, total_count, module_name,
    layer_stats). Only one layer's SVD workspaces are needed at a time.
    """
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    _validate_ratio(rho)
    targets = list(iter_target_linears(model))
    if not targets:
        raise ValueError("No supported decoder projection layers were found")
    if any(isinstance(module, GeoRALinear) for module in model.modules()):
        raise ValueError("The model already contains GeoRA layers")
    if any(module.weight.device.type != "cpu" for _, module in targets):
        raise ValueError("Move the model to CPU before installing GeoRA")
    for _, module in targets:
        _validate_rank(rank, tuple(module.weight.shape))

    model.requires_grad_(False)
    target_metadata = []
    for index, (name, module) in enumerate(targets, start=1):
        A0, B0, stats = initialize_geo_factors(module.weight, rank, rho)
        original_weight = module.weight.detach().to(dtype=torch.float32).clone()
        geo_layer = GeoRALinear(module, A0, B0, scaling=alpha / rank)
        effective_weight = (
            geo_layer.base_layer.weight
            + geo_layer.scaling * (geo_layer.B @ geo_layer.A)
        )
        weight_error = torch.linalg.vector_norm(effective_weight - original_weight)
        original_norm = torch.linalg.vector_norm(original_weight)
        original_norm_value = original_norm.item()
        relative_weight_error = weight_error.item()
        if original_norm_value:
            relative_weight_error = weight_error.item() / original_norm_value
        stats["initial_effective_weight_relative_error"] = relative_weight_error
        parent_name, leaf_name = name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), leaf_name, geo_layer)
        target_metadata.append({
            "name": name,
            "weight_shape": list(module.weight.shape),
            "A_shape": list(A0.shape),
            "B_shape": list(B0.shape),
            "statistics": stats,
        })
        if progress_callback is not None:
            progress_callback(index, len(targets), name, stats)
        del A0, B0, original_weight, effective_weight, weight_error, original_norm

    return {
        "geora_checkpoint_format": 1,
        "rank": rank,
        "alpha": alpha,
        "rho": rho,
        "scaling": alpha / rank,
        "precision_policy": dict(PRECISION_POLICY),
        "forward_mode": "residual",
        "target_modules": target_metadata,
        "checkpoint_contents": "initial A0/B0 and current A/B; rebuild F from the original model",
    }


def export_adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot FP32 initial and current factors, without frozen base weights."""
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, GeoRALinear):
            for factor_name in ("A0", "B0", "A", "B"):
                factor = getattr(module, factor_name)
                if factor.dtype != torch.float32:
                    raise ValueError(f"{name}.{factor_name} must remain FP32")
                state[f"{name}.{factor_name}"] = factor.detach().cpu().contiguous().clone()
    if not state:
        raise ValueError("The model has no GeoRA layers")
    return state


@torch.no_grad()
def load_geora_state(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
    metadata: Mapping,
    *,
    forward_mode: str | None = None,
) -> None:
    """Restore adapters onto a fresh copy of the same FP32 original model.

    Residual mode rebuilds F from W_pre and saved A0/B0. Difference mode
    retains W_pre and restores the same initial/current factors. All modes
    require a fresh original model, without existing GeoRA layers.

    The manifest's forward_mode is authoritative when present. Untagged old
    checkpoints default to residual mode; an explicit forward_mode may select
    difference mode for those checkpoints. A conflicting explicit mode on a
    tagged checkpoint is rejected before any model modification.
    Module names, shapes, factor dtypes, and finite values are validated before
    changing the model. The caller must also pin the original model revision.
    """
    if metadata.get("geora_checkpoint_format") != 1:
        raise ValueError("Unsupported GeoRA checkpoint format")
    if "forward_mode" in metadata:
        recorded_forward_mode = metadata["forward_mode"]
        _validate_forward_mode(recorded_forward_mode)
        if forward_mode is not None and forward_mode != recorded_forward_mode:
            raise ValueError("Requested forward mode conflicts with the checkpoint manifest")
        resolved_forward_mode = recorded_forward_mode
    else:
        resolved_forward_mode = "residual" if forward_mode is None else forward_mode
    _validate_forward_mode(resolved_forward_mode)
    if any(isinstance(module, GeoRALinear) for module in model.modules()):
        raise ValueError("Load onto a fresh original model, without GeoRA layers")
    rank = metadata["rank"]
    alpha = metadata["alpha"]
    _validate_ratio(metadata["rho"])
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("Invalid checkpoint rank")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Invalid checkpoint alpha")
    if not math.isclose(metadata["scaling"], alpha / rank, rel_tol=1e-12):
        raise ValueError("Checkpoint scaling does not equal alpha/rank")
    expected_policy = PRECISION_POLICY
    if metadata.get("precision_policy") != expected_policy:
        raise ValueError("Checkpoint precision policy differs from this implementation")

    target_metadata = metadata["target_modules"]
    recorded_names = [target["name"] for target in target_metadata]
    actual_names = [name for name, _ in iter_target_linears(model)]
    names_match = (
        recorded_names == actual_names
        and len(set(recorded_names)) == len(recorded_names)
        and len(recorded_names) > 0
    )
    if not names_match:
        raise ValueError("Checkpoint targets do not match the fresh original model")
    expected_keys = {
        f"{name}.{factor_name}"
        for name in recorded_names
        for factor_name in ("A0", "B0", "A", "B")
    }
    if set(state) != expected_keys:
        raise ValueError("Checkpoint factor names are missing or unexpected")

    for target in target_metadata:
        name = target["name"]
        base_layer = model.get_submodule(name)
        original_dtype_matches = base_layer.weight.dtype == torch.float32
        original_device_matches = base_layer.weight.device.type == "cpu"
        if not original_device_matches or not original_dtype_matches:
            raise ValueError("Load the fresh model on CPU in FP32 before restoring GeoRA")
        if not torch.isfinite(base_layer.weight).all():
            raise ValueError(f"Non-finite original weights in {name}")
        if list(base_layer.weight.shape) != target["weight_shape"]:
            raise ValueError(f"Original weight shape differs for {name}")
        _validate_rank(rank, tuple(base_layer.weight.shape))
        expected_A_shape = [rank, base_layer.in_features]
        expected_B_shape = [base_layer.out_features, rank]
        shapes_match = (
            target["A_shape"] == expected_A_shape
            and target["B_shape"] == expected_B_shape
        )
        if not shapes_match:
            raise ValueError(f"Recorded factor shapes differ for {name}")
        for factor_name in ("A0", "B0", "A", "B"):
            factor = state[f"{name}.{factor_name}"]
            expected_shape = expected_B_shape
            if factor_name.startswith("A"):
                expected_shape = expected_A_shape
            if list(factor.shape) != expected_shape or factor.dtype != torch.float32:
                raise ValueError(f"Invalid shape or dtype for {name}.{factor_name}")
            if not torch.isfinite(factor).all():
                raise ValueError(f"Non-finite values in {name}.{factor_name}")

    model.requires_grad_(False)
    for name in recorded_names:
        geo_layer = GeoRALinear(
            model.get_submodule(name),
            state[f"{name}.A0"],
            state[f"{name}.B0"],
            scaling=metadata["scaling"],
            forward_mode=resolved_forward_mode,
        )
        geo_layer.A.copy_(state[f"{name}.A"])
        geo_layer.B.copy_(state[f"{name}.B"])
        parent_name, leaf_name = name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), leaf_name, geo_layer)
