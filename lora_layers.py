"""LoRA baseline with the same frozen-base/FP32-adapter precision as GeoRA.

Our explicit initialization is A ~ Normal(0, 0.02**2), B = 0 on CPU. The
Gaussian scale is a reproduction choice, not an author-verified GeoRA setting.
Only A/B are saved; restore them on the pinned original model, never on a
model containing GeoRA's residual weights or an already merged adapter.
"""

from collections.abc import Callable, Iterator, Mapping
import math

import torch
from torch import nn
from torch.nn import functional

from geora_layers import GeoRALinear, iter_target_linears


LORA_PRECISION_POLICY = {
    "initialization_device": "cpu",
    "initial_factor_dtype": "float32",
    "local_frozen_model_dtype": "float32",
    "training_frozen_model_dtype": "bfloat16",
    "trainable_adapter_dtype": "float32",
    "optimizer_state_dtype": "float32",
    "training_forward": "native_bfloat16_base_and_float32_adapter",
    "adapter_forward": "float32_autocast_disabled",
    "correction_before_base_addition": "cast_to_base_output_dtype",
}


def _validate_rank(rank: int, shape: tuple[int, int]) -> None:
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise TypeError("rank must be an integer")
    if rank < 1 or rank > min(shape):
        raise ValueError(f"rank={rank} is invalid for weight shape {shape}")


def _validate_positive(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


class LoRALinear(nn.Module):
    """For x [..., input], compute native Wpre x + scaling * B(Ax).

    A has shape [rank, input], B [output, rank]. The base remains frozen and
    retains its original values. A/B remain FP32; the branch and scaling use
    FP32 with autocast disabled. Cast only its output to the base output dtype.
    Both branches remain differentiable with respect to the input.
    """

    def __init__(self, base_layer: nn.Linear, A: torch.Tensor,
                 B: torch.Tensor, scaling: float) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("base_layer must be torch.nn.Linear")
        _validate_positive(scaling, "scaling")
        if A.ndim != 2 or B.ndim != 2 or not A.is_floating_point() or not B.is_floating_point():
            raise ValueError("A and B must be real floating-point matrices")
        rank = A.shape[0]
        _validate_rank(rank, tuple(base_layer.weight.shape))
        if tuple(A.shape) != (rank, base_layer.in_features) or tuple(B.shape) != (base_layer.out_features, rank):
            raise ValueError("Factor shapes differ from the base projection")
        if not torch.isfinite(A).all() or not torch.isfinite(B).all():
            raise ValueError("Factors must contain finite values")
        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)
        self.scaling = float(scaling)
        self.forward_mode = "lora"
        factor_device = base_layer.weight.device
        self.A = nn.Parameter(A.detach().to(device=factor_device, dtype=torch.float32).clone())
        self.B = nn.Parameter(B.detach().to(device=factor_device, dtype=torch.float32).clone())
        self.train(base_layer.training)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frozen_output = self.base_layer(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            input_fp32 = x.to(dtype=torch.float32)
            compressed_input = functional.linear(input_fp32, self.A)
            adapter_output = functional.linear(compressed_input, self.B)
            correction_fp32 = self.scaling * adapter_output
        return frozen_output + correction_fp32.to(dtype=frozen_output.dtype)


def iter_lora_layers(model: nn.Module) -> Iterator[tuple[str, LoRALinear]]:
    """Yield installed LoRA adapters with their original projection names."""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


@torch.no_grad()
def install_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 32,
    seed: int = 20261010,
    init_std: float = 0.02,
    progress_callback: Callable[[int, int, str, dict], None] | None = None,
    provenance: Mapping | None = None,
) -> dict:
    """Install seven projections per decoder block; return a compact manifest.

    Installation requires the fresh original model on CPU in FP32. A private
    CPU generator makes factor initialization reproducible without consuming
    the global RNG used later for rollout. Caller supplies verified base-model
    provenance and must check its pinned revision before loading a checkpoint.
    """
    _validate_positive(alpha, "alpha")
    _validate_positive(init_std, "init_std")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if any(isinstance(module, (GeoRALinear, LoRALinear)) for module in model.modules()):
        raise ValueError("Install LoRA onto a fresh original model without adapters")
    targets = list(iter_target_linears(model))
    if not targets:
        raise ValueError("No supported decoder projections found")
    for name, layer in targets:
        _validate_rank(rank, tuple(layer.weight.shape))
        if layer.weight.device.type != "cpu" or layer.weight.dtype != torch.float32:
            raise ValueError("Install LoRA on a fresh CPU FP32 model")
        if not torch.isfinite(layer.weight).all():
            raise ValueError(f"Non-finite base weight in {name}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    model.requires_grad_(False)
    target_metadata = []
    for index, (name, layer) in enumerate(targets, start=1):
        A = torch.randn((rank, layer.in_features), generator=generator, dtype=torch.float32) * init_std
        B = torch.zeros((layer.out_features, rank), dtype=torch.float32)
        lora_layer = LoRALinear(layer, A, B, scaling=alpha / rank)
        parent_name, leaf_name = name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), leaf_name, lora_layer)
        metadata = {
            "name": name,
            "weight_shape": list(layer.weight.shape),
            "A_shape": list(A.shape),
            "B_shape": list(B.shape),
        }
        target_metadata.append(metadata)
        if progress_callback is not None:
            progress_callback(index, len(targets), name, metadata)
    return {
        "lora_checkpoint_format": 1,
        "method": "lora",
        "forward_mode": "lora",
        "rank": rank,
        "alpha": alpha,
        "scaling": alpha / rank,
        "initialization": {
            "A_distribution": "normal",
            "A_mean": 0.0,
            "A_std": init_std,
            "B": "zeros",
            "seed": seed,
            "generator": "private_cpu_torch_Generator",
            "choice_status": "our_explicit_baseline_choice_not_author_verified",
        },
        "precision_policy": dict(LORA_PRECISION_POLICY),
        "base_model_provenance": dict(provenance or {}),
        "target_modules": target_metadata,
        "checkpoint_contents": "FP32 current A/B; load onto unchanged pinned original W_pre",
    }


def export_lora_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Clone only FP32 A/B onto CPU; frozen base weights are not serialized."""
    state = {}
    for name, layer in iter_lora_layers(model):
        for factor_name in ("A", "B"):
            factor = getattr(layer, factor_name)
            if factor.dtype != torch.float32 or not torch.isfinite(factor).all():
                raise ValueError(f"{name}.{factor_name} must be finite FP32")
            state[f"{name}.{factor_name}"] = factor.detach().cpu().contiguous().clone()
    if not state:
        raise ValueError("The model has no LoRA layers")
    return state


@torch.no_grad()
def load_lora_state(model: nn.Module, state: Mapping[str, torch.Tensor], metadata: Mapping,
                    *, expected_provenance: Mapping | None = None) -> None:
    """Restore on a fresh CPU FP32 original model, validating before mutation."""
    if metadata.get("lora_checkpoint_format") != 1 or metadata.get("method") != "lora" or metadata.get("forward_mode") != "lora":
        raise ValueError("Unsupported LoRA checkpoint format or forward mode")
    if metadata.get("precision_policy") != LORA_PRECISION_POLICY:
        raise ValueError("LoRA checkpoint precision policy differs")
    if expected_provenance is not None and metadata.get("base_model_provenance") != dict(expected_provenance):
        raise ValueError("LoRA checkpoint base-model provenance differs")
    if any(isinstance(module, (GeoRALinear, LoRALinear)) for module in model.modules()):
        raise ValueError("Load LoRA onto a fresh original model without adapters")
    rank, alpha = metadata["rank"], metadata["alpha"]
    _validate_positive(alpha, "alpha")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("Invalid checkpoint rank")
    _validate_positive(metadata["scaling"], "scaling")
    if not math.isclose(metadata["scaling"], alpha / rank, rel_tol=1e-12):
        raise ValueError("Checkpoint scaling differs from alpha/rank")
    targets = metadata["target_modules"]
    recorded_names = [target["name"] for target in targets]
    actual_names = [name for name, _ in iter_target_linears(model)]
    if not recorded_names or recorded_names != actual_names or len(set(recorded_names)) != len(recorded_names):
        raise ValueError("Checkpoint targets differ from the original model")
    expected_keys = {f"{name}.{factor}" for name in recorded_names for factor in ("A", "B")}
    if set(state) != expected_keys:
        raise ValueError("Checkpoint factor names are missing or unexpected")
    for target in targets:
        name = target["name"]
        layer = model.get_submodule(name)
        if layer.weight.device.type != "cpu" or layer.weight.dtype != torch.float32:
            raise ValueError("Restore LoRA on a fresh CPU FP32 original model")
        _validate_rank(rank, tuple(layer.weight.shape))
        if list(layer.weight.shape) != target["weight_shape"] or not torch.isfinite(layer.weight).all():
            raise ValueError(f"Invalid original weight in {name}")
        expected_shapes = {"A": [rank, layer.in_features], "B": [layer.out_features, rank]}
        for factor, expected_shape in expected_shapes.items():
            tensor = state[f"{name}.{factor}"]
            if target[f"{factor}_shape"] != expected_shape or list(tensor.shape) != expected_shape:
                raise ValueError(f"Invalid factor shape for {name}.{factor}")
            if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid factor dtype or value for {name}.{factor}")
    model.requires_grad_(False)
    for name in recorded_names:
        layer = LoRALinear(model.get_submodule(name), state[f"{name}.A"], state[f"{name}.B"], metadata["scaling"])
        parent_name, leaf_name = name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), leaf_name, layer)
