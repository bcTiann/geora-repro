"""Bounded KV-cache sampling for likelihood diagnostics, not training yet.

Keep the validated fixed-width sampler unchanged. This candidate uses the
Transformers generation loop and records the probabilities it actually samples.
Full-vocabulary scores are retained temporarily, so limit diagnostics to short
completions; a long-run sampler needs a more memory-efficient score recorder.
"""

from contextlib import contextmanager
import math
import time

import torch
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList

from geora_check_utils import forward_context
from grpo_rollout import require_zero_dropout


@contextmanager
def isolated_sampling_rng(generator, device):
    """Use a private generator without changing the process's global RNG.

    Transformers samples with the global RNG. Temporarily install our private
    state, copy its advanced state back, then restore the process state. This
    context is for a single sampling thread/process, on CPU or CUDA/ROCm.
    """
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("This candidate supports CPU and CUDA/ROCm only")
    if torch.device(generator.device).type != device.type:
        raise ValueError("Generator and model must use the same device backend")
    index = None
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        generator_index = torch.device(generator.device).index
        if generator_index is not None and generator_index != index:
            raise ValueError("Generator and model must use the same CUDA device")
    with torch.random.fork_rng(devices=[] if index is None else [index]):
        if index is None:
            torch.set_rng_state(generator.get_state())
        else:
            torch.cuda.set_rng_state(generator.get_state(), index)
        try:
            yield
        finally:
            advanced_state = (torch.get_rng_state() if index is None
                              else torch.cuda.get_rng_state(index))
            generator.set_state(advanced_state)


class RolloutDeadline(StoppingCriteria):
    """Fail instead of silently treating a timed-out answer as a completion."""

    def __init__(self, deadline):
        self.deadline = deadline

    def __call__(self, input_ids, scores, **kwargs):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Cached rollout deadline reached")
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


def sample_completions_cached(
    model, prompt_ids, pad_token_id, eos_token_id, max_new_tokens, device,
    frozen_dtype, *, temperature=1.0, generator, deadline=None,
):
    """Return the same fixed-width data layout as the validated sampler.

    Input prompts are already expanded to one row per answer. A first EOS is
    included in the completion mask; later padding is excluded by position,
    even when pad equals EOS or a pad ID is sampled as a legitimate token.
    KV caches belong to this call only and are never used for training scoring.
    """
    require_zero_dropout(model)
    if not prompt_ids or any(not tokens for tokens in prompt_ids):
        raise ValueError("Every prompt must contain at least one token")
    if not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 64:
        raise ValueError("Use 1-64 new tokens for this bounded diagnostic")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    eos_ids = [eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id)
    if not eos_ids or any(not isinstance(value, int) for value in eos_ids):
        raise ValueError("Supply one or more integer EOS token IDs")
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Cached rollout deadline reached before generation")

    batch_size = len(prompt_ids)
    prompt_width = max(map(len, prompt_ids))
    input_ids = torch.full(
        (batch_size, prompt_width + max_new_tokens), pad_token_id,
        dtype=torch.long, device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, tokens in enumerate(prompt_ids):
        start = prompt_width - len(tokens)
        input_ids[row, start:prompt_width] = torch.tensor(tokens, device=device)
        attention_mask[row, start:prompt_width] = 1

    # A fresh configuration avoids inheriting hidden model-specific penalties,
    # forced tokens or top-k filtering from generation_config.json.
    generation_config = GenerationConfig(
        do_sample=True, use_cache=True, cache_implementation="dynamic",
        max_new_tokens=max_new_tokens, temperature=temperature,
        top_k=0, top_p=1.0, typical_p=1.0, repetition_penalty=1.0,
        num_beams=1, num_return_sequences=1,
        pad_token_id=pad_token_id, eos_token_id=eos_ids,
        bos_token_id=model.config.bos_token_id,
        return_dict_in_generate=True, output_scores=True,
        output_logits=False, disable_compile=True,
    )
    criteria = StoppingCriteriaList()
    if deadline is not None:
        criteria.append(RolloutDeadline(deadline))
    # Do not use inference_mode: returned IDs will enter differentiable scoring.
    with isolated_sampling_rng(generator, device), torch.no_grad(), forward_context(device, frozen_dtype):
        generated = model.generate(
            input_ids=input_ids[:, :prompt_width],
            attention_mask=attention_mask[:, :prompt_width],
            generation_config=generation_config,
            stopping_criteria=criteria,
        )

    steps = len(generated.scores)
    if steps < 1 or steps > max_new_tokens:
        raise RuntimeError("Unexpected number of generation score tensors")
    generated_ids = generated.sequences[:, prompt_width:]
    if generated_ids.shape != (batch_size, steps):
        raise RuntimeError("Generated token and probability positions do not match")
    completion_mask = torch.zeros(
        (batch_size, max_new_tokens), dtype=torch.bool, device=device,
    )
    behavior_logps = torch.zeros(
        (batch_size, max_new_tokens), dtype=torch.float32, device=device,
    )
    active = torch.ones(batch_size, dtype=torch.bool, device=device)
    ended_with_eos = torch.zeros_like(active)
    eos_tensor = torch.tensor(eos_ids, dtype=torch.long, device=device)
    for step, scores in enumerate(generated.scores):
        scores = scores.float()
        if not torch.isfinite(scores).all().item():
            raise RuntimeError("Cached sampling scores contain non-finite values")
        sampled = generated_ids[:, step]
        selected_logps = scores.log_softmax(-1).gather(1, sampled[:, None]).squeeze(1)
        if not torch.isfinite(selected_logps).all().item():
            raise RuntimeError("Cached sampled-token log probabilities are not finite")
        input_ids[:, prompt_width + step] = torch.where(active, sampled, pad_token_id)
        attention_mask[:, prompt_width + step] = active.long()
        completion_mask[:, step] = active
        behavior_logps[:, step] = torch.where(active, selected_logps, 0.0)
        just_ended = active & torch.isin(sampled, eos_tensor)
        ended_with_eos |= just_ended
        active &= ~just_ended

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_ids": input_ids[:, prompt_width:].clone(),
        "completion_mask": completion_mask,
        "behavior_logps": behavior_logps,
        "prompt_width": prompt_width,
        "prompt_lengths": list(map(len, prompt_ids)),
        "completion_lengths": completion_mask.sum(-1),
        "ended_with_eos": ended_with_eos,
        "generated_steps": steps,
        "max_new_tokens": max_new_tokens,
        "fixed_width": prompt_width + max_new_tokens,
        "temperature": float(temperature),
        "eos_token_ids": eos_ids,
        "use_cache": True,
        "path": "transformers_generate_dynamic_cache_scores",
        "rng_path": "private_generator_via_fork_rng",
    }
