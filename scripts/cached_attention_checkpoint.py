"""Checkpoint only pure native eager attention after DynamicCache.update.

Qwen2Attention has already projected/rotated Q/K and appended K/V when this
function runs. The checkpoint closure receives tensors; no cache update occurs
inside it. Forward and backward recomputation call the same bound native eager
function. Restore the module-global wrapper even when scoring raises.
"""

from contextlib import contextmanager

import torch
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen2 import modeling_qwen2

from grpo_rollout import require_zero_dropout


@contextmanager
def checkpoint_eager_attention(model, statistics=None):
    """Temporarily checkpoint native Qwen2's pure attention arithmetic.

    Scope a gradient-enabled cached scoring forward with this context; backward
    can run after it exits because each checkpoint closure binds the original
    native function. No-grad forwards remain native without checkpointing.
    Output-attention collection and nonzero dropout are deliberately unsupported.
    This is single-process/single-thread instrumentation, not a concurrent API.
    """
    require_zero_dropout(model)
    if model.config.model_type != "qwen2" or model.config._attn_implementation != "eager":
        raise ValueError("Checkpoint context supports native Qwen2 eager attention only")
    if getattr(model.config, "output_attentions", False):
        raise ValueError("Output-attention collection is unsupported by this checkpoint context")
    original = modeling_qwen2.eager_attention_forward
    if getattr(original, "_cached_attention_checkpoint", False):
        raise ValueError("Nested attention checkpoint contexts are unsupported")
    stats = statistics if statistics is not None else {}
    stats.setdefault("wrapper_forward_calls", 0)
    stats.setdefault("checkpoint_forward_calls", 0)
    stats.setdefault("recompute_calls", 0)
    stats.setdefault("native_no_grad_calls", 0)
    stats.setdefault("autocast_states_match", True)

    def attention(module, query, key, value, attention_mask, **kwargs):
        if kwargs.get("output_attentions", False):
            raise ValueError("Output-attention collection is unsupported")
        if kwargs.get("dropout", 0.0) != 0.0 or getattr(module, "attention_dropout", 0.0) != 0.0:
            raise ValueError("Pure attention checkpoint requires zero dropout")
        stats["wrapper_forward_calls"] += 1
        needs_gradient = torch.is_grad_enabled() and any(
            tensor.requires_grad for tensor in (query, key, value))
        if not needs_gradient:
            stats["native_no_grad_calls"] += 1
            return original(module, query, key, value, attention_mask, **kwargs)
        stats["checkpoint_forward_calls"] += 1
        calls = 0
        backend = query.device.type
        initial_autocast = (torch.is_autocast_enabled(backend), torch.get_autocast_dtype(backend))

        # Bind module/native kwargs per invocation. Backward must not consult the
        # restored global eager function or a later iteration's mutable state.
        def pure_attention(q, k, v, mask):
            nonlocal calls
            autocast = (torch.is_autocast_enabled(backend), torch.get_autocast_dtype(backend))
            stats["autocast_states_match"] &= autocast == initial_autocast
            if calls:
                stats["recompute_calls"] += 1
            calls += 1
            return original(module, q, k, v, mask, **kwargs)
        return checkpoint(pure_attention, query, key, value, attention_mask,
                          use_reentrant=False, preserve_rng_state=True,
                          determinism_check="default")

    attention._cached_attention_checkpoint = True
    modeling_qwen2.eager_attention_forward = attention
    try:
        yield stats
    finally:
        modeling_qwen2.eager_attention_forward = original


def score_completion_logps_checkpointed(model, rollout, device, frozen_dtype, *,
                                       gradient_enabled, trace_callback=None,
                                       checkpoint_statistics=None):
    """Use the same cached likelihoods with pure attention recomputed in backward.

    Inference forwards stay native. The temporary wrapper exists only during
    the differentiable forward; its stateless closures survive until backward.
    """
    from cached_score import score_completion_logps_cached
    if not gradient_enabled:
        return score_completion_logps_cached(
            model, rollout, device, frozen_dtype, gradient_enabled=False,
            trace_callback=trace_callback)
    with checkpoint_eager_attention(model, checkpoint_statistics):
        return score_completion_logps_cached(
            model, rollout, device, frozen_dtype, gradient_enabled=True,
            trace_callback=trace_callback)
