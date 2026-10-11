"""Native cached sampling with bounded live vocabulary-score memory.

A final generation LogitsProcessor applies the requested temperature and keeps
only one step's full-vocabulary log probabilities. At the next call, the token
actually sampled at the preceding step is available in input_ids; gather its
logp, then release that vocabulary tensor. Finalize the last token after
native generate returns. Saved rollout tensors grow only with batch*tokens.

This is a candidate training sampler, separate from both existing samplers.
"""

import math
import time

import torch
from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList
from transformers import StoppingCriteriaList

from cached_rollout import RolloutDeadline, isolated_sampling_rng
from geora_check_utils import forward_context
from grpo_rollout import require_zero_dropout


class StreamingTokenLogps(LogitsProcessor):
    """Record the distribution actually passed to native multinomial sampling.

    Temperature lives here; native generation temperature is 1 so no subsequent
    temperature warper can alter the recorded distribution. The fresh native
    config adds no other logits processors. This observer supports one ordinary
    non-beam decode only; it is not reusable across generation calls.
    """

    def __init__(self, batch_size, device, temperature, eos_ids, progress_callback=None):
        self.temperature = temperature
        self.progress_callback = progress_callback
        self.eos_ids = torch.tensor(eos_ids, device=device, dtype=torch.long)
        self.active = torch.ones(batch_size, device=device, dtype=torch.bool)
        self.ended = torch.zeros_like(self.active)
        self.pending = None
        self.selected_logps = []
        self.masks = []
        self.call_count = 0
        self.finalized = False
        self.max_live_vocabulary_elements = 0

    def consume(self, token):
        if self.pending is None:
            raise RuntimeError("There is no preceding sampling distribution")
        selected = self.pending.gather(1, token[:, None]).squeeze(1)
        self.selected_logps.append(torch.where(self.active, selected, 0.0))
        self.masks.append(self.active.clone())
        ended_now = self.active & torch.isin(token, self.eos_ids)
        self.ended |= ended_now
        self.active &= ~ended_now
        self.pending = None
        steps = len(self.selected_logps)
        if self.progress_callback is not None and (steps % 32 == 0 or not self.active.any().item()):
            self.progress_callback({"generated_steps": steps, "active_rows": self.active.sum().item()})

    def __call__(self, input_ids, scores):
        if self.finalized:
            raise RuntimeError("Cannot reuse a finalized probability recorder")
        if self.pending is not None:
            self.consume(input_ids[:, -1])
        transformed = scores.float() / self.temperature
        if not torch.isfinite(transformed).all().item():
            raise RuntimeError("Cached sampling logits contain non-finite values")
        self.pending = transformed.log_softmax(-1)
        self.max_live_vocabulary_elements = max(self.max_live_vocabulary_elements, self.pending.numel())
        self.call_count += 1
        return transformed

    def finalize(self, sequences):
        if self.finalized:
            raise RuntimeError("Probability recorder already finalized")
        self.consume(sequences[:, -1])
        self.finalized = True
        if self.call_count != len(self.selected_logps):
            raise RuntimeError("Generation and recorded probabilities have different step counts")


def sample_completions_cached_training(
    model, prompt_ids, pad_token_id, eos_token_id, max_new_tokens, device,
    frozen_dtype, *, temperature=1.0, generator, progress_callback=None, deadline=None,
):
    """Native DynamicCache rollout, 1-256 tokens, no saved vocabulary scores.

    The returned layout/RNG/EOS conventions match sample_completions_cached.
    First EOS is scored; later pad slots are excluded. No differentiable graph
    or rollout cache is retained. Use a fresh cached teacher-forcing forward to
    compute the current policy's differentiable selected-token probabilities.
    """
    require_zero_dropout(model)
    if not prompt_ids or any(not tokens for tokens in prompt_ids):
        raise ValueError("Every prompt must contain at least one token")
    if not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 256:
        raise ValueError("Use 1-256 new tokens for this bounded training candidate")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    eos_ids = [eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id)
    if not eos_ids or any(not isinstance(value, int) for value in eos_ids):
        raise ValueError("Supply one or more integer EOS token IDs")
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Cached rollout deadline reached before generation")

    batch_size = len(prompt_ids)
    prompt_width = max(map(len, prompt_ids))
    input_ids = torch.full((batch_size, prompt_width + max_new_tokens), pad_token_id,
                           dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for row, tokens in enumerate(prompt_ids):
        start = prompt_width - len(tokens)
        input_ids[row, start:prompt_width] = torch.tensor(tokens, device=device)
        attention_mask[row, start:prompt_width] = 1
    recorder = StreamingTokenLogps(batch_size, device, temperature, eos_ids, progress_callback)
    # Native temperature is 1. All transformations are disabled so the recorder
    # is the final/only probability transform before native softmax+multinomial.
    configuration = GenerationConfig(
        do_sample=True, use_cache=True, cache_implementation="dynamic",
        max_new_tokens=max_new_tokens, temperature=1.0,
        top_k=0, top_p=1.0, typical_p=1.0, repetition_penalty=1.0,
        num_beams=1, num_return_sequences=1,
        pad_token_id=pad_token_id, eos_token_id=eos_ids,
        bos_token_id=model.config.bos_token_id,
        return_dict_in_generate=True, output_scores=False, output_logits=False,
        disable_compile=True)
    criteria = StoppingCriteriaList()
    if deadline is not None:
        criteria.append(RolloutDeadline(deadline))
    with isolated_sampling_rng(generator, device), torch.no_grad(), forward_context(device, frozen_dtype):
        generated = model.generate(
            input_ids=input_ids[:, :prompt_width], attention_mask=attention_mask[:, :prompt_width],
            generation_config=configuration, stopping_criteria=criteria,
            logits_processor=LogitsProcessorList([recorder]))
        recorder.finalize(generated.sequences)
    if generated.scores is not None or generated.logits is not None:
        raise RuntimeError("Native generation unexpectedly retained vocabulary scores")
    steps = recorder.call_count
    if not 1 <= steps <= max_new_tokens:
        raise RuntimeError("Unexpected number of native generation steps")
    generated_ids = generated.sequences[:, prompt_width:]
    if generated_ids.shape != (batch_size, steps):
        raise RuntimeError("Generated token and probability positions do not match")
    completion_mask = torch.zeros((batch_size, max_new_tokens), dtype=torch.bool, device=device)
    behavior_logps = torch.zeros((batch_size, max_new_tokens), dtype=torch.float32, device=device)
    masks = torch.stack(recorder.masks, dim=1)
    selected = torch.stack(recorder.selected_logps, dim=1)
    if not torch.isfinite(selected).all().item():
        raise RuntimeError("Recorded selected-token probabilities are not finite")
    completion_mask[:, :steps] = masks
    behavior_logps[:, :steps] = selected
    input_ids[:, prompt_width:prompt_width + steps] = torch.where(masks, generated_ids, pad_token_id)
    attention_mask[:, prompt_width:prompt_width + steps] = masks.long()
    # Drop generate's inference K/V immediately; it must never enter backward.
    generated.past_key_values = None
    return {
        "input_ids": input_ids, "attention_mask": attention_mask,
        "completion_ids": input_ids[:, prompt_width:].clone(),
        "completion_mask": completion_mask, "behavior_logps": behavior_logps,
        "prompt_width": prompt_width, "prompt_lengths": list(map(len, prompt_ids)),
        "completion_lengths": completion_mask.sum(-1), "ended_with_eos": recorder.ended,
        "generated_steps": steps, "max_new_tokens": max_new_tokens,
        "fixed_width": prompt_width + max_new_tokens, "temperature": float(temperature),
        "eos_token_ids": eos_ids, "use_cache": True,
        "path": "transformers_generate_dynamic_cache_streamed_selected_logps",
        "rng_path": "private_generator_via_fork_rng",
        "score_storage": "one_pending_vocabulary_logp_plus_selected_batch_by_tokens",
        "max_pending_vocabulary_elements": recorder.max_live_vocabulary_elements,
        "native_vocabulary_scores_retained": False,
    }
