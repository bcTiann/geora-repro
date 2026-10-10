"""Sample and score a small Qwen batch through the same fixed-shape path.

This is a correctness path, not a throughput implementation. Every rollout
forward uses the final padded width and no KV cache. The final scorer uses
that same decoder shape and projects one prediction position at a time, so
the BF16 lm_head multiplication also keeps the same [batch, 1, hidden] shape.
"""

import math
import time

import torch

from geora_check_utils import forward_context


def require_zero_dropout(model):
    """Require deterministic train-mode forwards for the aligned smoke path."""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout) and module.p != 0:
            raise ValueError(f"Nonzero dropout at {name}: {module.p}")
        value = getattr(module, "attention_dropout", 0.0)
        if isinstance(value, (int, float)) and value != 0:
            raise ValueError(f"Nonzero attention dropout at {name}: {value}")


def position_ids_from_mask(attention_mask):
    """Left-padded valid tokens use positions 0, 1, ...; masked slots use 0."""
    position_ids = attention_mask.cumsum(dim=-1) - 1
    return position_ids.masked_fill(attention_mask == 0, 0)


def _decoder_hidden(model, input_ids, attention_mask):
    return model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids_from_mask(attention_mask),
        use_cache=False,
        return_dict=True,
    ).last_hidden_state


def _one_position_logits(model, hidden_states, prediction_position, temperature):
    # Both generation and scoring call the head with this exact batch shape.
    hidden = hidden_states[:, prediction_position:prediction_position + 1, :]
    return model.lm_head(hidden).squeeze(1).float() / temperature


def sample_completions(
    model,
    prompt_ids,
    pad_token_id,
    eos_token_id,
    max_new_tokens,
    device,
    frozen_dtype,
    *,
    temperature=1.0,
    generator=None,
    progress_callback=None,
    deadline=None,
):
    """Return a fixed-width rollout dictionary for an already-expanded batch.

    prompt_ids is a list of token lists; repeat each question G times before
    calling. All rows stay present after EOS. The first EOS is a valid scored
    completion token; later slots contain pad IDs and have zero masks/logp.
    A truncated answer keeps all max_new_tokens valid tokens.

    Sampling is plain FP32 softmax + multinomial. No top-k/top-p, penalties,
    or generation-config processors are used. deadline, when provided, is an
    absolute time.monotonic() deadline and is checked before each forward.
    The caller chooses train/eval mode; this function does not change it.
    """
    require_zero_dropout(model)
    if not prompt_ids or any(not tokens for tokens in prompt_ids):
        raise ValueError("Every prompt must contain at least one token")
    if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    eos_ids = [eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id)
    if not eos_ids or any(not isinstance(value, int) for value in eos_ids):
        raise ValueError("Supply one or more integer EOS token IDs")
    batch_size = len(prompt_ids)
    prompt_width = max(map(len, prompt_ids))
    full_width = prompt_width + max_new_tokens
    input_ids = torch.full(
        (batch_size, full_width), pad_token_id, dtype=torch.long, device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, tokens in enumerate(prompt_ids):
        start = prompt_width - len(tokens)
        input_ids[row, start:prompt_width] = torch.tensor(tokens, device=device)
        attention_mask[row, start:prompt_width] = 1
    completion_mask = torch.zeros(
        (batch_size, max_new_tokens), dtype=torch.bool, device=device,
    )
    behavior_logps = torch.zeros(
        (batch_size, max_new_tokens), dtype=torch.float32, device=device,
    )
    active = torch.ones(batch_size, dtype=torch.bool, device=device)
    ended_with_eos = torch.zeros_like(active)
    eos_tensor = torch.tensor(eos_ids, dtype=torch.long, device=device)
    steps = 0
    # no_grad, rather than inference_mode: returned IDs are subsequently used
    # by a differentiable embedding/decoder scoring forward.
    with torch.no_grad(), forward_context(device, frozen_dtype):
        for step in range(max_new_tokens):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Rollout deadline reached before an optimizer update")
            hidden_states = _decoder_hidden(model, input_ids, attention_mask)
            logits = _one_position_logits(
                model, hidden_states, prompt_width + step - 1, temperature,
            )
            if not torch.isfinite(logits).all().item():
                raise RuntimeError("Rollout logits contain non-finite values")
            probabilities = logits.softmax(dim=-1)
            sampled = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
            selected_logps = logits.log_softmax(dim=-1).gather(
                1, sampled[:, None],
            ).squeeze(1)
            if not torch.isfinite(selected_logps).all().item():
                raise RuntimeError("Sampled token log probabilities are not finite")
            sampled = torch.where(active, sampled, pad_token_id)
            input_ids[:, prompt_width + step] = sampled
            attention_mask[:, prompt_width + step] = active.long()
            completion_mask[:, step] = active
            behavior_logps[:, step] = torch.where(active, selected_logps, 0.0)
            just_ended = active & torch.isin(sampled, eos_tensor)
            ended_with_eos |= just_ended
            active &= ~just_ended
            steps = step + 1
            if progress_callback is not None and (steps % 32 == 0 or not active.any().item()):
                progress_callback({"generated_steps": steps, "active_rows": active.sum().item()})
            if not active.any().item():
                break
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_ids": input_ids[:, prompt_width:].clone(),
        "completion_mask": completion_mask,
        "behavior_logps": behavior_logps,
        "prompt_width": prompt_width,
        "prompt_lengths": list(map(len, prompt_ids)),
        "completion_lengths": completion_mask.sum(dim=-1),
        "ended_with_eos": ended_with_eos,
        "generated_steps": steps,
        "max_new_tokens": max_new_tokens,
        "fixed_width": full_width,
        "temperature": float(temperature),
        "eos_token_ids": eos_ids,
        "use_cache": False,
        "path": "fixed_full_width_decoder_and_one_position_lm_head",
    }


def score_completion_logps(model, rollout, device, frozen_dtype, *, gradient_enabled):
    """Return [batch, max_new_tokens] logp, zero outside completion_mask.

    One full fixed-width decoder forward is reused for all positions. Every
    lm_head call uses [batch, 1, hidden], matching the sampling head shape.
    Use gradient_enabled=False for fixed old/reference scores; True for the
    current policy. The caller retains model train/eval mode and zero dropout.
    """
    require_zero_dropout(model)
    input_ids = rollout["input_ids"]
    attention_mask = rollout["attention_mask"]
    completion_mask = rollout["completion_mask"]
    prompt_width = rollout["prompt_width"]
    token_count = rollout["max_new_tokens"]
    if input_ids.shape != attention_mask.shape:
        raise ValueError("Input IDs and attention mask shapes differ")
    if input_ids.shape[1] != prompt_width + token_count:
        raise ValueError("Scoring must retain the full rollout width")
    if completion_mask.shape != (input_ids.shape[0], token_count):
        raise ValueError("Invalid completion mask shape")
    with torch.set_grad_enabled(gradient_enabled), forward_context(device, frozen_dtype):
        hidden_states = _decoder_hidden(model, input_ids, attention_mask)
        scores = []
        for step in range(token_count):
            if step >= rollout["generated_steps"]:
                scores.append(hidden_states.new_zeros(input_ids.shape[0], dtype=torch.float32))
                continue
            logits = _one_position_logits(
                model, hidden_states, prompt_width + step - 1, rollout["temperature"],
            )
            sampled = input_ids[:, prompt_width + step:prompt_width + step + 1]
            selected = logits.log_softmax(dim=-1).gather(1, sampled).squeeze(1)
            scores.append(torch.where(completion_mask[:, step], selected, 0.0))
        result = torch.stack(scores, dim=1)
    return result if gradient_enabled else result.detach()


def likelihood_alignment_metrics(behavior_logps, score_logps, completion_mask):
    """Describe the actual sampling/scoring policy difference on valid tokens."""
    difference = (score_logps.detach() - behavior_logps.detach()).double()
    selected = difference[completion_mask]
    if selected.numel() == 0:
        raise ValueError("No valid completion tokens")
    ratios = selected.exp()
    sequence_differences = torch.where(completion_mask, difference, 0.0).sum(dim=-1)
    return {
        "valid_token_count": selected.numel(),
        "max_absolute_delta_logp": selected.abs().max().item(),
        "mean_absolute_delta_logp": selected.abs().mean().item(),
        "max_absolute_ratio_minus_one": (ratios - 1).abs().max().item(),
        "min_ratio": ratios.min().item(),
        "max_ratio": ratios.max().item(),
        "sequence_delta_logp": sequence_differences.cpu().tolist(),
        "finite": bool(torch.isfinite(selected).all().item() and torch.isfinite(ratios).all().item()),
        "exact": torch.equal(score_logps.detach()[completion_mask], behavior_logps.detach()[completion_mask]),
    }
