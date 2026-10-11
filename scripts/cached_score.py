"""Differentiable sequential likelihoods aligned with the cached sampler.

This candidate uses one DynamicCache per scoring call. Prefill is followed by
one-token teacher-forcing forwards, with the same inputs as Transformers
sampling. K/V tensors are never detached: every loss remains differentiable
through the earlier prompt and completion states. This changes neither the
existing sampler nor the validated fixed-width scorer.
"""

import math

import torch
from transformers.cache_utils import DynamicCache

from geora_check_utils import forward_context
from grpo_rollout import require_zero_dropout


def _validate_rollout(model, rollout):
    require_zero_dropout(model)
    if model.config.model_type != "qwen2" or model.config._attn_implementation != "eager":
        raise ValueError("This candidate is limited to native Qwen2 eager attention")
    if getattr(model.model, "has_sliding_layers", False):
        raise ValueError("Sliding attention is outside this candidate's scope")
    if getattr(model, "is_gradient_checkpointing", False):
        raise ValueError("Gradient checkpointing can disable or mutate the training cache")
    if not rollout.get("use_cache"):
        raise ValueError("Supply a rollout from the cached sampling path")
    ids, mask, completion_mask = (rollout[key] for key in
                                  ("input_ids", "attention_mask", "completion_mask"))
    width = rollout["prompt_width"]
    capacity, steps = rollout["max_new_tokens"], rollout["generated_steps"]
    if ids.ndim != 2 or ids.shape != mask.shape or ids.dtype != torch.long:
        raise ValueError("Input IDs and mask must be equal-shaped [batch, width]")
    if not isinstance(width, int) or width < 1 or width + capacity != ids.shape[1]:
        raise ValueError("Invalid prompt/completion width")
    if not 1 <= steps <= capacity <= 64:
        raise ValueError("Use 1-64 generated tokens for the bounded cached candidate")
    if completion_mask.dtype != torch.bool or completion_mask.shape != (ids.shape[0], capacity):
        raise ValueError("Invalid completion mask")
    if ids.device != mask.device or ids.device != completion_mask.device:
        raise ValueError("Rollout tensors must use the same device")
    if not ((mask == 0) | (mask == 1)).all().item():
        raise ValueError("Attention mask must be binary")
    prompt_mask = mask[:, :width]
    if not prompt_mask[:, -1].bool().all().item():
        raise ValueError("Each prompt must finish at the last prefill column")
    if (prompt_mask[:, :-1] > prompt_mask[:, 1:]).any().item():
        raise ValueError("Only left-padded prompts are supported")
    lengths = completion_mask.sum(-1)
    expected_mask = torch.arange(capacity, device=ids.device)[None, :] < lengths[:, None]
    if not torch.equal(completion_mask, expected_mask) or not (lengths >= 1).all().item():
        raise ValueError("Completion mask must contain a nonempty contiguous prefix")
    if (lengths > steps).any().item() or not torch.equal(mask[:, width:].bool(), completion_mask):
        raise ValueError("Completion mask exceeds generated steps or differs from stored layout")
    if not math.isfinite(rollout["temperature"]) or rollout["temperature"] <= 0:
        raise ValueError("Temperature must be finite and positive")


def score_completion_logps_cached(model, rollout, device, frozen_dtype, *,
                                  gradient_enabled, trace_callback=None):
    """Return [batch, max_new_tokens] actual cached-path selected logp.

    Completion tokens are fixed data; no sampling occurs. A fresh cache holds
    differentiable history only for this call. Finished rows remain in every
    forward, and generation's mask appends ones even after their first EOS.
    completion_mask alone controls which output logps contribute to the loss.

    trace_callback is an optional read-only observer of input metadata. It gets
    CPU lists/shapes, not the cache or tensors attached to the autograd graph.
    Model mode, parameters, RNG and installed hooks are left unchanged.
    """
    _validate_rollout(model, rollout)
    ids = rollout["input_ids"]
    if torch.device(device).type != ids.device.type:
        raise ValueError("Scoring device and rollout backend differ")
    width = rollout["prompt_width"]
    capacity = rollout["max_new_tokens"]
    steps = rollout["generated_steps"]
    prompt_mask = rollout["attention_mask"][:, :width].clone()
    # Native generate prepares positions before optionally dropping an all-one
    # mask; subsequent positions increase even in already-finished rows.
    positions = model._prepare_position_ids_for_generation(
        ids[:, :width], {"attention_mask": prompt_mask})
    generation_mask = None if (prompt_mask == 1).all().item() else prompt_mask
    cache = DynamicCache(config=model.config)
    model_kwargs = dict(past_key_values=cache, attention_mask=generation_mask,
                        position_ids=positions, use_cache=True, logits_to_keep=1)
    scores = []
    with torch.set_grad_enabled(gradient_enabled), forward_context(device, frozen_dtype):
        for step in range(steps):
            prefix_ids = ids[:, :width + step]
            prepared = model.prepare_inputs_for_generation(
                prefix_ids, next_sequence_length=None if step == 0 else 1,
                is_first_iteration=step == 0, **model_kwargs)
            if trace_callback is not None:
                trace_callback(dict(
                    step=step, input_shape=list(prepared["input_ids"].shape),
                    input_ids=prepared["input_ids"].detach().cpu().tolist(),
                    position_ids=prepared["position_ids"].detach().cpu().tolist(),
                    attention_mask=None if prepared.get("attention_mask") is None else
                    prepared["attention_mask"].detach().cpu().tolist(),
                    logits_to_keep=prepared["logits_to_keep"],
                    cache_length_before=cache.get_seq_length()))
            outputs = model(**prepared, return_dict=True)
            # Do not reuse a no_grad rollout cache or detach these new K/V.
            cache = outputs.past_key_values
            logits = outputs.logits[:, -1].float() / rollout["temperature"]
            selected = logits.log_softmax(-1).gather(
                1, ids[:, width + step:width + step + 1]).squeeze(1)
            scores.append(torch.where(rollout["completion_mask"][:, step], selected, 0.0))
            model_kwargs["past_key_values"] = cache
            positions = torch.cat([positions, positions[:, -1:] + 1], dim=-1)
            model_kwargs["position_ids"] = positions
            if generation_mask is not None:
                generation_mask = torch.cat([generation_mask, generation_mask.new_ones(
                    generation_mask.shape[0], 1)], dim=-1)
                model_kwargs["attention_mask"] = generation_mask
        zero = scores[0].new_zeros(ids.shape[0])
        scores.extend(zero for _ in range(capacity - steps))
        result = torch.stack(scores, dim=1)
    return result if gradient_enabled else result.detach()
