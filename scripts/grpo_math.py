"""Readable GRPO loss primitives; no model loading or optimizer steps.

Source: DeepSeekMath section 4.1, equations (3) and (4):
https://arxiv.org/html/2402.03300v3#S4.SS1

This implementation uses population group standard deviation, epsilon=1e-8,
and a token mean inside each completion followed by a completion mean. The
standard-deviation convention and epsilon are our explicit smoke-run choices.

The sampled-token k3 term is differentiated directly with its sampled token
held fixed. This is a GRPO surrogate convention, not the exact gradient of a
full-vocabulary KL expectation; the sampling-distribution derivative is absent.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def group_advantages(rewards: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Normalize each question's rewards, shape [questions, answers_per_question].

    Example: [1, 0, 0, 0] becomes approximately [sqrt(3), -1/sqrt(3),
    -1/sqrt(3), -1/sqrt(3)]. Identical rewards give an exactly zero group.
    Reward labels and the resulting advantages are fixed during optimization.
    """
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("Rewards must have shape [questions, group_size >= 2].")
    if epsilon <= 0:
        raise ValueError("Advantage epsilon must be positive.")
    fixed_rewards = rewards.detach().float()
    if not torch.isfinite(fixed_rewards).all():
        raise ValueError("Reward labels must be finite.")
    group_mean = fixed_rewards.mean(dim=1, keepdim=True)
    group_std = fixed_rewards.std(dim=1, correction=0, keepdim=True)
    return (fixed_rewards - group_mean) / (group_std + epsilon)


def completion_token_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_starts: torch.Tensor,
    eos_token_ids: Sequence[int],
) -> torch.Tensor:
    """Return a mask [answers, sequence_length] selecting completion token IDs.

    completion_starts[row] is the absolute token index of that row's first
    completion token, including any left padding. Prompt and attention=0
    positions are excluded. The first completion EOS is included; later tokens
    are excluded even when a caller accidentally marks them as attention=1.
    A truncated answer without EOS includes all of its non-padding tokens.

    For causal logits[:, :-1] predicting input_ids[:, 1:], use mask[:, 1:].
    The first completion token must have a preceding prompt token to predict it.
    """
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("Token IDs and attention mask must share shape [answers, length].")
    if completion_starts.shape != (input_ids.shape[0],):
        raise ValueError("Completion starts must have one absolute index per answer.")
    if torch.any(completion_starts < 1) or torch.any(completion_starts >= input_ids.shape[1]):
        raise ValueError("Each completion must begin after a prompt and inside the sequence.")
    if completion_starts.device != input_ids.device or attention_mask.device != input_ids.device:
        raise ValueError("Mask inputs must be on the same device.")
    positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    completion_positions = positions >= completion_starts.unsqueeze(1)
    candidate_mask = completion_positions & attention_mask.bool()
    if not eos_token_ids:
        return candidate_mask
    eos_ids = torch.tensor(list(eos_token_ids), dtype=input_ids.dtype, device=input_ids.device)
    completion_eos = torch.isin(input_ids, eos_ids) & candidate_mask
    eos_count_before_position = completion_eos.long().cumsum(dim=1) - completion_eos.long()
    return candidate_mask & (eos_count_before_position == 0)


def causal_token_log_probs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Gather FP32 causal log probabilities [answers, sequence_length - 1].

    logits[row, t] predicts input_ids[row, t+1]. This scores the provided token
    sequence; it does not generate a new answer. Old/reference scores must be
    computed without parameter updates, then detached before loss construction.
    This helper does not claim cache/padding paths have identical probabilities.
    """
    if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
        raise ValueError("Logits [answers, length, vocab] must match the token IDs.")
    next_token_log_probs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    next_token_ids = input_ids[:, 1:].unsqueeze(-1)
    return next_token_log_probs.gather(dim=-1, index=next_token_ids).squeeze(-1)


def grpo_loss(
    current_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    token_mask: torch.Tensor,
    clip_epsilon: float = 0.2,
    kl_coefficient: float = 0.001,
) -> dict[str, torch.Tensor]:
    """Clipped GRPO minimization loss and inspectable terms.

    Log probabilities and mask have shape [answers, completion_positions];
    advantages has shape [answers] (flatten group_advantages in the same order
    as the answers). EOS belongs in the mask; prompt/pad/post-EOS tokens do not.

    ratio = exp(current - old)
    surrogate = min(ratio * advantage, clip(ratio, 1-eps, 1+eps) * advantage)
    k3 = exp(reference - current) - (reference - current) - 1
    loss = mean_answers(mean_valid_tokens(-surrogate + beta * k3)).

    Old policy, reference and advantages are detached here as a second guard.
    Masked log probabilities are replaced before exp so irrelevant NaN/Inf
    positions cannot contaminate the loss or gradient. Valid non-finite values
    fail loudly. Optimizer and gradient clipping are intentionally separate.
    """
    if current_log_probs.ndim != 2:
        raise ValueError("Current log probabilities must have shape [answers, positions].")
    if old_log_probs.shape != current_log_probs.shape or reference_log_probs.shape != current_log_probs.shape:
        raise ValueError("Current, old and reference log probabilities must have identical shapes.")
    if token_mask.shape != current_log_probs.shape or advantages.shape != (current_log_probs.shape[0],):
        raise ValueError("Mask shape or one-advantage-per-answer shape is incorrect.")
    if not 0 < clip_epsilon < 1 or kl_coefficient < 0:
        raise ValueError("Clipping epsilon must lie in (0, 1); KL coefficient must be nonnegative.")
    if any(value.device != current_log_probs.device for value in (old_log_probs, reference_log_probs, advantages, token_mask)):
        raise ValueError("All loss inputs must share the same device.")
    valid_tokens = token_mask.bool()
    completion_lengths = valid_tokens.sum(dim=1)
    if torch.any(completion_lengths == 0):
        raise ValueError("Every completion must have at least one valid predicted token.")
    for values in (current_log_probs, old_log_probs, reference_log_probs):
        if not torch.isfinite(values[valid_tokens]).all():
            raise ValueError("Valid-token log probabilities must be finite.")
    if not torch.isfinite(advantages).all():
        raise ValueError("Advantages must be finite.")

    current = torch.where(valid_tokens, current_log_probs, 0.0)
    old = torch.where(valid_tokens, old_log_probs.detach(), 0.0)
    reference = torch.where(valid_tokens, reference_log_probs.detach(), 0.0)
    fixed_advantages = advantages.detach().unsqueeze(1)

    ratios = torch.exp(current - old)
    clipped_ratios = ratios.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    per_token_surrogate = torch.minimum(
        ratios * fixed_advantages,
        clipped_ratios * fixed_advantages,
    )
    reference_minus_current = reference - current
    # expm1(x)-x avoids subtracting two nearly equal numbers around x=0.
    per_token_kl = torch.expm1(reference_minus_current) - reference_minus_current
    if not torch.isfinite(ratios[valid_tokens]).all() or not torch.isfinite(per_token_kl[valid_tokens]).all():
        raise ValueError("Valid-token importance ratios or KL terms overflowed.")

    per_completion_policy_loss = -(per_token_surrogate * valid_tokens).sum(dim=1) / completion_lengths
    per_completion_kl = (per_token_kl * valid_tokens).sum(dim=1) / completion_lengths
    per_completion_loss = per_completion_policy_loss + kl_coefficient * per_completion_kl
    clipped_positions = ((fixed_advantages > 0) & (ratios > 1.0 + clip_epsilon)) | (
        (fixed_advantages < 0) & (ratios < 1.0 - clip_epsilon)
    )
    return {
        "loss": per_completion_loss.mean(),
        "policy_loss": per_completion_policy_loss.mean(),
        "kl": per_completion_kl.mean(),
        "per_completion_loss": per_completion_loss,
        "per_token_surrogate": per_token_surrogate,
        "per_token_kl": per_token_kl,
        "ratios": ratios,
        "completion_lengths": completion_lengths,
        "clip_fraction": (clipped_positions & valid_tokens).sum() / valid_tokens.sum(),
    }
