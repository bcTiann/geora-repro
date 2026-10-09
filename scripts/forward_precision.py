"""Temporary FP32 projection policy for a matched forward-only diagnostic.

Build each model from the original CPU FP32 checkpoint first. Restore GeoRA
from saved FP32 factors before calling prepare_fp32_projections(). An already
rounded BF16 residual cannot be repaired by converting it back to FP32.
"""

from contextlib import contextmanager
from functools import wraps

import torch
from torch import nn

from geora_layers import GeoRALinear, TARGET_PATTERN


def projection_modules(model: nn.Module):
    """Select Q/K/V/O and gate/up/down projections; exclude lm_head."""
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if TARGET_PATTERN.search(name)
        and isinstance(module, (nn.Linear, GeoRALinear))
    ]
    if not targets:
        raise ValueError("No supported decoder projections were found")
    return targets


@torch.no_grad()
def prepare_fp32_projections(model: nn.Module, device: str) -> list[str]:
    """Keep fresh FP32 projections; cast other frozen parameters to BF16.

    This changes parameter storage. The forward context below separately
    controls the operation dtype. It intentionally does not freeze A/B or
    replace the production precision policy.
    """
    targets = projection_modules(model)
    protected_parameters = {
        id(parameter)
        for _, module in targets
        for parameter in module.parameters()
    }
    for name, module in targets:
        tensors = list(module.named_parameters()) + list(module.named_buffers())
        for tensor_name, tensor in tensors:
            if tensor.is_floating_point() and tensor.dtype != torch.float32:
                raise ValueError(
                    f"{name}.{tensor_name} must already be FP32; rebuild the "
                    "model from the original CPU FP32 checkpoint and factors"
                )
    for name, parameter in model.named_parameters():
        if id(parameter) not in protected_parameters:
            if parameter.requires_grad:
                raise ValueError(f"Non-projection parameter is trainable: {name}")
            parameter.data = parameter.data.to(dtype=torch.bfloat16)
    # Retain existing non-parameter buffers, such as rotary inv_freq, exactly
    # as Transformers manages them. A0/B0 buffers were checked above as FP32.
    model.to(device=device)
    return [name for name, _ in targets]


def _fp32_projection_forward(original_forward):
    @wraps(original_forward)
    def forward(x: torch.Tensor) -> torch.Tensor:
        output_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            # For GeoRA this invokes the existing F x + c B(A x) forward.
            # Both the residual branch and the adapter branch compute FP32.
            output = original_forward(x.float())
        return output.to(dtype=output_dtype)

    return forward


@contextmanager
def fp32_projection_forwards(model: nn.Module):
    """Temporarily disable autocast inside target projections only.

    The caller must use inference_mode() for this forward-only experiment
    and must give reference and GeoRA models the same precision settings.
    No tensor is detached here, so this helper itself preserves autograd.
    It modifies instance forwards and is intended for single-threaded probes.
    """
    targets = projection_modules(model)
    for name, module in targets:
        for tensor_name, tensor in module.named_parameters():
            if tensor.dtype != torch.float32:
                raise ValueError(f"{name}.{tensor_name} must be FP32")
    previous = []
    try:
        for _, module in targets:
            # Restore an existing instance override if one was present. If
            # forward originally came from the class, remove our override.
            had_instance_forward = "forward" in module.__dict__
            instance_forward = module.__dict__.get("forward")
            original_forward = module.forward
            previous.append((module, had_instance_forward, instance_forward))
            module.forward = _fp32_projection_forward(original_forward)
        yield [name for name, _ in targets]
    finally:
        for module, had_instance_forward, instance_forward in reversed(previous):
            if had_instance_forward:
                module.forward = instance_forward
            else:
                del module.forward


def _eager_attention_fp32_scores(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Qwen eager attention with FP32 QK scores, retaining native PV dtype."""
    import transformers.models.qwen2.modeling_qwen2 as qwen

    key_states = qwen.repeat_kv(key, module.num_key_value_groups)
    value_states = qwen.repeat_kv(value, module.num_key_value_groups)
    with torch.autocast(device_type=query.device.type, enabled=False):
        scores = torch.matmul(query.float(), key_states.float().transpose(2, 3))
        scores = scores * scaling
        if attention_mask is not None:
            scores = scores + attention_mask.float()

    # This matches Qwen's existing FP32 softmax and conversion to query dtype.
    probabilities = nn.functional.softmax(scores, dim=-1, dtype=torch.float32)
    probabilities = probabilities.to(dtype=query.dtype)
    probabilities = nn.functional.dropout(
        probabilities, p=dropout, training=module.training
    )
    output = torch.matmul(probabilities, value_states)
    output = output.transpose(1, 2).contiguous()
    return output, probabilities


@contextmanager
def fp32_attention_scores():
    """Temporarily replace Qwen2's eager attention score calculation only.

    Every model used inside this context must select attn_implementation=eager.
    The module-global replacement is intended for sequential diagnostics, not
    concurrent model execution. The existing function is restored on exit.
    """
    import transformers.models.qwen2.modeling_qwen2 as qwen

    previous = qwen.eager_attention_forward
    try:
        qwen.eager_attention_forward = _eager_attention_fp32_scores
        yield
    finally:
        qwen.eager_attention_forward = previous
