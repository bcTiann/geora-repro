"""Check initial GeoRA forwards through generation, caches, and left padding.

Every equality compares GeoRA with the original model using the same inputs,
layout, precision, and cache policy. Cache versus full-prefix calculations can
round differently even in the original model; those are recorded as controls.
No reward, task accuracy, or training stability is measured here.
"""

import torch

from geora_check_utils import forward_context, logits_difference


def left_padded_inputs(prompt_token_lists, pad_token_id, device):
    """Return IDs/mask/positions for unequal-length token lists, no tokenizer.

    Example: [[4, 5], [6, 7, 8]] becomes IDs [[pad, 4, 5], [6, 7, 8]],
    masks [[0, 1, 1], [1, 1, 1]], and positions [[0, 0, 1], [0, 1, 2]].
    Valid tokens keep their unpadded position numbers for direct forwards.
    """
    if not prompt_token_lists or any(len(tokens) < 2 for tokens in prompt_token_lists):
        raise ValueError("Supply nonempty prompts containing at least two tokens each.")
    width = max(map(len, prompt_token_lists))
    input_ids = torch.full(
        (len(prompt_token_lists), width), pad_token_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, tokens in enumerate(prompt_token_lists):
        input_ids[row, -len(tokens):] = torch.tensor(tokens, device=device)
        attention_mask[row, -len(tokens):] = 1
    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }


def _logits(model, inputs, device, frozen_dtype, **kwargs):
    model.eval()
    with torch.inference_mode(), forward_context(device, frozen_dtype):
        output = model(**inputs, **kwargs)
    return output.logits.detach().float().cpu(), output.past_key_values


def _generate(model, inputs, pad_token_id, eos_token_id, device, frozen_dtype,
              max_new_tokens, use_cache):
    model.eval()
    with torch.inference_mode(), forward_context(device, frozen_dtype):
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True,
        )
    sequences = output.sequences.detach().cpu()
    scores = [score.detach().float().cpu() for score in output.scores]
    raw_logits = [value.detach().float().cpu() for value in output.logits]
    return sequences, scores, raw_logits


def check_generation_paths(model, reference, prompt_token_lists, pad_token_id,
                           eos_token_id, device, frozen_dtype, report,
                           max_new_tokens=8):
    """Validate an untrained difference model against its native reference.

    Models must be independently constructed and already placed on the requested
    device/precision. Each call creates independent caches; a reference cache
    is never passed to GeoRA. The explicit three-step decode uses fixed tokens
    to exercise the cache even when ordinary greedy generation stops at EOS.
    """
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    padded = left_padded_inputs(prompt_token_lists, pad_token_id, device)
    if len(set(map(len, prompt_token_lists))) < 2:
        raise ValueError("Use unequal prompt lengths to exercise left padding.")
    result = {
        "scope": "untrained native-reference equality; backend controls are separate",
        "prompt_lengths": list(map(len, prompt_token_lists)),
        "max_new_tokens": max_new_tokens,
        "controls": {},
    }

    def exact(name, expected, actual):
        finite = torch.isfinite(expected).all().item() and torch.isfinite(actual).all().item()
        report.require("generation_" + name, finite and torch.equal(expected, actual),
                       **logits_difference(expected, actual), finite=finite)

    # Direct batch forward: ignore padded query positions in the comparison.
    reference_batch, _ = _logits(reference, padded, device, frozen_dtype, use_cache=False)
    model_batch, _ = _logits(model, padded, device, frozen_dtype, use_cache=False)
    valid = padded["attention_mask"].bool().cpu()
    exact("left_padded_valid_logits_exact", reference_batch[valid].unsqueeze(0),
          model_batch[valid].unsqueeze(0))

    # A batch and a single example may choose different matrix multiplication
    # shapes. Measure this on the reference before attributing it to GeoRA.
    single_controls = []
    for row, tokens in enumerate(prompt_token_lists):
        single = left_padded_inputs([tokens], pad_token_id, device)
        reference_single, _ = _logits(reference, single, device, frozen_dtype, use_cache=False)
        model_single, _ = _logits(model, single, device, frozen_dtype, use_cache=False)
        exact(f"single_{row}_logits_exact", reference_single, model_single)
        single_controls.append(logits_difference(
            reference_single, reference_batch[row, valid[row]].unsqueeze(0)
        ))
    result["controls"]["reference_single_vs_left_padded_batch"] = single_controls

    # Prefill a real mutable Transformer cache independently for each model.
    tokens = prompt_token_lists[0]
    single = left_padded_inputs([tokens], pad_token_id, device)
    reference_prefill, reference_cache = _logits(
        reference, single, device, frozen_dtype, use_cache=True
    )
    model_prefill, model_cache = _logits(model, single, device, frozen_dtype, use_cache=True)
    report.require("generation_separate_nonempty_caches",
                   reference_cache is not None and model_cache is not None
                   and reference_cache is not model_cache,
                   reference_cache_type=type(reference_cache).__name__,
                   model_cache_type=type(model_cache).__name__)
    exact("cache_prefill_logits_exact", reference_prefill, model_prefill)
    report.require("generation_cache_prefill_length",
                   reference_cache.get_seq_length() == model_cache.get_seq_length() == len(tokens),
                   reference_length=reference_cache.get_seq_length(),
                   model_length=model_cache.get_seq_length())
    explicit_decode_controls = []
    growing_tokens = list(tokens)
    # Reuse fixed prompt tokens; this is a cache mechanics check, not sampling.
    continuation = [tokens[-1], tokens[-2], tokens[-1]]
    for step, token in enumerate(continuation):
        position = len(growing_tokens)
        growing_tokens.append(token)
        decode_inputs = {
            "input_ids": torch.tensor([[token]], dtype=torch.long, device=device),
            "attention_mask": torch.ones((1, len(growing_tokens)), dtype=torch.long, device=device),
            "position_ids": torch.tensor([[position]], dtype=torch.long, device=device),
        }
        reference_cached, reference_cache = _logits(
            reference, decode_inputs, device, frozen_dtype,
            past_key_values=reference_cache, use_cache=True,
        )
        model_cached, model_cache = _logits(
            model, decode_inputs, device, frozen_dtype,
            past_key_values=model_cache, use_cache=True,
        )
        exact(f"cache_decode_{step}_logits_exact", reference_cached, model_cached)
        report.require(f"generation_cache_decode_{step}_length",
                       reference_cache is not model_cache
                       and reference_cache.get_seq_length() == model_cache.get_seq_length() == len(growing_tokens),
                       expected_length=len(growing_tokens), model_length=model_cache.get_seq_length())
        full_inputs = left_padded_inputs([growing_tokens], pad_token_id, device)
        reference_full, _ = _logits(reference, full_inputs, device, frozen_dtype, use_cache=False)
        model_full, _ = _logits(model, full_inputs, device, frozen_dtype, use_cache=False)
        exact(f"decode_{step}_full_prefix_logits_exact", reference_full[:, -1:], model_full[:, -1:])
        explicit_decode_controls.append(logits_difference(reference_full[:, -1:], reference_cached))
    result["controls"]["reference_cache_vs_full_prefix"] = explicit_decode_controls

    generated = {}
    # Use the padded batch for ordinary generation. Transformers derives valid
    # position IDs from the attention mask; direct forwards above use explicit IDs.
    for use_cache in (True, False):
        path = "cached" if use_cache else "uncached"
        expected_ids, expected_scores, expected_raw_logits = _generate(
            reference, padded, pad_token_id, eos_token_id, device, frozen_dtype,
            max_new_tokens, use_cache,
        )
        actual_ids, actual_scores, actual_raw_logits = _generate(
            model, padded, pad_token_id, eos_token_id, device, frozen_dtype,
            max_new_tokens, use_cache,
        )
        # Generation processors can legitimately replace scores with -inf
        # (for example, suppressed tokens). Require raw model logits finite.
        raw_logits_finite = all(torch.isfinite(value).all().item()
                                for value in expected_raw_logits + actual_raw_logits)
        raw_logits_exact = len(expected_raw_logits) == len(actual_raw_logits) and all(
            torch.equal(expected, actual)
            for expected, actual in zip(expected_raw_logits, actual_raw_logits)
        )
        scores_exact = len(expected_scores) == len(actual_scores) and all(
            torch.equal(expected, actual)
            for expected, actual in zip(expected_scores, actual_scores)
        )
        report.require(f"generation_{path}_greedy_sequences_and_scores_exact",
                       len(actual_scores) >= 1 and torch.equal(expected_ids, actual_ids)
                       and scores_exact and raw_logits_exact and raw_logits_finite,
                       generated_steps=len(actual_scores), raw_logits_finite=raw_logits_finite,
                       sequences_equal=torch.equal(expected_ids, actual_ids), scores_equal=scores_exact,
                       raw_logits_equal=raw_logits_exact)
        generated[path] = {"sequences": actual_ids.tolist(), "generated_steps": len(actual_scores)}
        result[path] = generated[path]
    result["controls"]["reference_cached_vs_uncached_greedy_sequences_equal"] = (
        generated["cached"]["sequences"] == generated["uncached"]["sequences"]
    )
    report.data["generation_paths"] = result
    report.write()
    return result
