"""Cached native Qwen likelihoods with immutable block boundaries.

Generation and this scorer both prefill once and then execute one token at a
time. Blocks only change activation storage: non-reentrant checkpointing
reconstructs a fresh private DynamicCache from explicit K/V tensor inputs.
No cache object is captured or reused by backward recomputation.
"""

import math

import torch
from torch.utils.checkpoint import checkpoint
from transformers.cache_utils import DynamicCache

from geora_check_utils import forward_context
from grpo_rollout import require_zero_dropout


def _validate_rollout(model, rollout):
    require_zero_dropout(model)
    if model.config.model_type != "qwen2" or model.config._attn_implementation != "eager":
        raise ValueError("Block scoring supports native Qwen2 eager attention only")
    if getattr(model.model, "has_sliding_layers", False):
        raise ValueError("Sliding attention is outside the block scorer")
    if getattr(model, "is_gradient_checkpointing", False):
        raise ValueError("Do not combine mutable model gradient checkpointing with cached blocks")
    if not rollout.get("use_cache"):
        raise ValueError("Block scoring requires native cached sampling")
    ids = rollout["input_ids"]
    mask = rollout["attention_mask"]
    completion = rollout["completion_mask"]
    width = rollout["prompt_width"]
    capacity = rollout["max_new_tokens"]
    steps = rollout["generated_steps"]
    if ids.ndim != 2 or ids.shape != mask.shape or ids.dtype != torch.long:
        raise ValueError("IDs and attention mask must share a [batch, width] layout")
    if not isinstance(width, int) or width < 1 or width + capacity != ids.shape[1]:
        raise ValueError("Invalid prompt/completion storage width")
    if not 1 <= steps <= capacity <= 4096:
        raise ValueError("Block scoring supports at most 4096 completion tokens")
    if completion.dtype != torch.bool or completion.shape != (ids.shape[0], capacity):
        raise ValueError("Invalid completion mask")
    if ids.device != mask.device or ids.device != completion.device:
        raise ValueError("IDs and masks must use the same device")
    if not ((mask == 0) | (mask == 1)).all().item():
        raise ValueError("Attention mask must be binary")
    prompt_mask = mask[:, :width]
    if not prompt_mask[:, -1].bool().all().item():
        raise ValueError("Every prompt must finish in the last prefill column")
    if (prompt_mask[:, :-1] > prompt_mask[:, 1:]).any().item():
        raise ValueError("Only left-padded prompts are supported")
    lengths = completion.sum(-1)
    expected = torch.arange(capacity, device=ids.device)[None, :] < lengths[:, None]
    if not torch.equal(completion, expected) or not (lengths >= 1).all().item():
        raise ValueError("Completion mask must have a nonempty contiguous prefix")
    if (lengths > steps).any().item() or not torch.equal(mask[:, width:].bool(), completion):
        raise ValueError("Completion mask differs from stored token layout")
    if not math.isfinite(rollout["temperature"]) or rollout["temperature"] <= 0:
        raise ValueError("Temperature must be finite and positive")


def score_completion_logps_block_checkpointed(
    model, rollout, device, frozen_dtype, *, gradient_enabled,
    trace_callback=None, runtime_guard=None, checkpoint_statistics=None, block_size=16,
    progress_callback=None,
):
    """Return selected-token logps using the native one-token cache path.

    A block receives the previous boundary's K,V tensors as explicit inputs.
    Its new cache is private: native append allocates new tensors, leaving
    those input tensors unchanged. Return both selected logps and the next
    boundary tensors, so gradients from later tokens reach earlier history.
    Inference uses the same blocks without activation checkpointing.
    trace_callback observes detailed metadata on the original forward only.
    progress_callback receives a {"step": integer} dictionary on the original forward,
    avoiding tensor copies when callers need lightweight progress reporting.
    runtime_guard runs
    before/after token forwards, including backward checkpoint recomputation;
    it may raise to abort before an optimizer update.
    """
    _validate_rollout(model, rollout)
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    ids = rollout["input_ids"]
    if torch.device(device).type != ids.device.type:
        raise ValueError("Scoring device and rollout backend differ")
    width = rollout["prompt_width"]
    capacity = rollout["max_new_tokens"]
    steps = rollout["generated_steps"]
    prompt_mask = rollout["attention_mask"][:, :width]
    initial_positions = model._prepare_position_ids_for_generation(
        ids[:, :width], {"attention_mask": prompt_mask})
    all_one_mask = bool((prompt_mask == 1).all().item())
    statistics = checkpoint_statistics if checkpoint_statistics is not None else {}
    statistics.setdefault("checkpoint_blocks", 0)
    statistics.setdefault("recompute_blocks", 0)
    statistics.setdefault("autocast_states_match", True)
    statistics.setdefault("boundary_cache_tensor_elements", 0)
    statistics.setdefault("max_boundary_cache_tensor_elements", 0)
    statistics.setdefault("forward_cache_forwards", 0)
    statistics.setdefault("recompute_cache_forwards", 0)
    statistics.setdefault("runtime_guard_forward_checks", 0)
    statistics.setdefault("runtime_guard_recompute_checks", 0)
    statistics.update(block_size=block_size, generated_steps=steps,
                      boundary_cache_inputs_are_explicit=True)
    boundary = ()
    score_blocks = []

    def make_block(start, stop):
        # Each closure binds immutable token bounds. Its cache is recreated on
        # every invocation, including backward; later loop bounds cannot leak.
        calls = 0
        backend = ids.device.type
        initial_autocast = (torch.is_autocast_enabled(backend), torch.get_autocast_dtype(backend))

        def block(*cache_tensors):
            nonlocal calls
            recomputing = calls > 0
            calls += 1
            if recomputing:
                statistics["recompute_blocks"] += 1
            state = (torch.is_autocast_enabled(backend), torch.get_autocast_dtype(backend))
            statistics["autocast_states_match"] &= state == initial_autocast
            pairs = tuple(zip(cache_tensors[::2], cache_tensors[1::2]))
            cache = DynamicCache(ddp_cache_data=pairs or None, config=model.config)
            positions = torch.cat([
                initial_positions,
                initial_positions[:, -1:] + torch.arange(1, start + 1, device=ids.device)[None, :],
            ], dim=-1) if start else initial_positions
            generation_mask = None
            if not all_one_mask:
                generation_mask = torch.cat([
                    prompt_mask, prompt_mask.new_ones(prompt_mask.shape[0], start),
                ], dim=-1)
            model_kwargs = dict(past_key_values=cache, attention_mask=generation_mask,
                                position_ids=positions, use_cache=True, logits_to_keep=1)
            selected_logps = []
            for step in range(start, stop):
                prefix_ids = ids[:, :width + step]
                prepared = model.prepare_inputs_for_generation(
                    prefix_ids, next_sequence_length=None if step == 0 else 1,
                    is_first_iteration=step == 0, **model_kwargs)
                guard_metadata = dict(
                    step=step, recomputing=recomputing, block_start=start, block_stop=stop,
                )
                if runtime_guard is not None:
                    statistics["runtime_guard_recompute_checks" if recomputing else
                               "runtime_guard_forward_checks"] += 1
                    runtime_guard(dict(guard_metadata, phase="before_forward"))
                if progress_callback is not None and not recomputing:
                    progress_callback(dict(step=step))
                if trace_callback is not None and not recomputing:
                    trace_callback(dict(
                        step=step, input_shape=list(prepared["input_ids"].shape),
                        input_ids=prepared["input_ids"].detach().cpu().tolist(),
                        position_ids=prepared["position_ids"].detach().cpu().tolist(),
                        attention_mask=None if prepared.get("attention_mask") is None else
                        prepared["attention_mask"].detach().cpu().tolist(),
                        logits_to_keep=prepared["logits_to_keep"],
                        cache_length_before=cache.get_seq_length()))
                statistics["recompute_cache_forwards" if recomputing else
                           "forward_cache_forwards"] += 1
                outputs = model(**prepared, return_dict=True)
                if runtime_guard is not None:
                    statistics["runtime_guard_recompute_checks" if recomputing else
                               "runtime_guard_forward_checks"] += 1
                    runtime_guard(dict(guard_metadata, phase="after_forward"))
                cache = outputs.past_key_values
                logits = outputs.logits[:, -1].float() / rollout["temperature"]
                selected = logits.log_softmax(-1).gather(
                    1, ids[:, width + step:width + step + 1]).squeeze(1)
                selected_logps.append(torch.where(rollout["completion_mask"][:, step], selected, 0.0))
                model_kwargs["past_key_values"] = cache
                positions = torch.cat([positions, positions[:, -1:] + 1], dim=-1)
                model_kwargs["position_ids"] = positions
                if generation_mask is not None:
                    generation_mask = torch.cat([
                        generation_mask, generation_mask.new_ones(generation_mask.shape[0], 1),
                    ], dim=-1)
                    model_kwargs["attention_mask"] = generation_mask
            next_boundary = tuple(tensor for layer in cache for tensor in layer[:2])
            return (torch.stack(selected_logps, dim=1), *next_boundary)
        return block

    with torch.set_grad_enabled(gradient_enabled), forward_context(device, frozen_dtype):
        for start in range(0, steps, block_size):
            block = make_block(start, min(start + block_size, steps))
            if gradient_enabled:
                statistics["checkpoint_blocks"] += 1
                outputs = checkpoint(block, *boundary, use_reentrant=False,
                                     preserve_rng_state=True, determinism_check="default")
            else:
                outputs = block(*boundary)
            score_blocks.append(outputs[0])
            boundary = outputs[1:]
            count = sum(tensor.numel() for tensor in boundary)
            statistics["boundary_cache_tensor_elements"] += count
            statistics["max_boundary_cache_tensor_elements"] = max(
                statistics["max_boundary_cache_tensor_elements"], count)
        result = torch.cat(score_blocks, dim=1)
        if steps < capacity:
            result = torch.cat([result, result.new_zeros(ids.shape[0], capacity - steps)], dim=1)
    return result if gradient_enabled else result.detach()
