"""Compare pre-softmax logits for identical cached and no-cache histories.

Use one frozen native model. Capture its actual generation lm_head outputs,
then reuse the validated fixed-width decoder/head path on the same token IDs.
The CPU FP64 analysis does not restore precision lost in BF16 model forwards.
"""

import hashlib
import time

import torch
from safetensors.torch import save_file

from cached_rollout import sample_completions_cached
from geora_check_utils import forward_context
from grpo_rollout import (
    _decoder_hidden, _one_position_logits, likelihood_alignment_metrics,
    require_zero_dropout,
)


def compare_native_logits(model, prompts, pad_id, eos_ids, new_tokens,
                          device, dtype, report, label, *, check_prefix=False):
    """Collect paired full-vocabulary logits and explain token probability gaps."""
    require_zero_dropout(model)
    report.require(label + "_one_frozen_native_model",
                   not model.training and all(not p.requires_grad for p in model.parameters())
                   and not any(type(m).__name__ in ("GeoRALinear", "LoRALinear")
                               for m in model.modules()))
    cached_logits = []
    head_shapes = []
    raw_dtypes = []

    def capture_head(module, inputs, output):
        if output.ndim != 3 or output.shape[0] != len(prompts):
            raise RuntimeError("Unexpected generation lm_head output shape")
        # The final prompt/new-token position predicts the next sampled token.
        cached_logits.append(output[:, -1, :].detach().float().cpu().clone())
        head_shapes.append(list(inputs[0].shape))
        raw_dtypes.append(str(output.dtype))

    started = time.perf_counter()
    handle = model.lm_head.register_forward_hook(capture_head)
    try:
        rollout = sample_completions_cached(
            model, prompts, pad_id, eos_ids, new_tokens, device, dtype,
            generator=torch.Generator(device=device).manual_seed(29),
        )
    finally:
        handle.remove()
    if device == "cuda":
        torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - started
    report.require(label + "_one_raw_logits_capture_per_generated_token",
                   len(cached_logits) == rollout["generated_steps"]
                   and all(z.shape == (len(prompts), model.config.vocab_size)
                           for z in cached_logits),
                   generated_steps=rollout["generated_steps"], head_input_shapes=head_shapes,
                   raw_output_dtypes=raw_dtypes)

    no_cache_logits = []
    reconstructed_cached_logps = torch.zeros_like(rollout["behavior_logps"])
    no_cache_logps = torch.zeros_like(rollout["behavior_logps"])
    same_input_repeat_max = 0.0
    started = time.perf_counter()
    with torch.no_grad(), forward_context(device, dtype):
        hidden = _decoder_hidden(model, rollout["input_ids"], rollout["attention_mask"])
        for step in range(rollout["generated_steps"]):
            tokens = rollout["completion_ids"][:, step:step + 1]
            valid = rollout["completion_mask"][:, step]
            original_logits = cached_logits[step].to(device)
            # Use the same device, FP32 routine and [batch,vocab] shape as the
            # generation probability recorder; do not use CPU for this gate.
            cached_logp = original_logits.log_softmax(-1)
            if step == 0:
                repeated = original_logits.log_softmax(-1)
                same_input_repeat_max = (cached_logp - repeated).abs().max().item()
            reconstructed_cached_logps[:, step] = torch.where(
                valid, cached_logp.gather(1, tokens).squeeze(1), 0.0,
            )
            logits = _one_position_logits(
                model, hidden, rollout["prompt_width"] + step - 1, 1.0,
            )
            no_cache_logits.append(logits.detach().cpu().clone())
            no_cache_logps[:, step] = torch.where(
                valid, logits.log_softmax(-1).gather(1, tokens).squeeze(1), 0.0,
            )
        del hidden
    if device == "cuda":
        torch.cuda.synchronize()
    recomputation_seconds = time.perf_counter() - started
    mask = rollout["completion_mask"].cpu()
    cached_gpu_logps = rollout["behavior_logps"].cpu()
    observed_delta = (no_cache_logps.cpu() - cached_gpu_logps).double()
    reconstruction_error = (
        reconstructed_cached_logps.cpu() - cached_gpu_logps
    )[mask].abs().max().item()
    report.require(label + "_captured_raw_logits_reproduce_generation_logps",
                   reconstruction_error <= 2e-6,
                   max_absolute_logp_difference=reconstruction_error)
    report.require(label + "_same_logits_same_device_normalization_repeat",
                   same_input_repeat_max == 0.0,
                   max_absolute_logp_difference=same_input_repeat_max)

    max_logit_difference = 0.0
    max_centered_difference = 0.0
    difference_sum = 0.0
    element_count = 0
    unequal_count = 0
    decomposition_error = 0.0
    normalization_residual = 0.0
    fp64_delta_values = []
    worst = None
    worst_ratio_error = -1.0
    for step in range(rollout["generated_steps"]):
        rows = mask[:, step].nonzero().flatten()
        if not rows.numel():
            continue
        cached = cached_logits[step][rows].double()
        actual = no_cache_logits[step][rows].double()
        sampled = rollout["completion_ids"].cpu()[rows, step]
        raw_difference = actual - cached
        centered_difference = raw_difference - raw_difference.mean(-1, keepdim=True)
        max_logit_difference = max(max_logit_difference, raw_difference.abs().max().item())
        max_centered_difference = max(max_centered_difference, centered_difference.abs().max().item())
        difference_sum += raw_difference.abs().sum().item()
        element_count += raw_difference.numel()
        unequal_count += (raw_difference != 0).sum().item()
        selected_difference = raw_difference.gather(1, sampled[:, None]).squeeze(1)
        normalizer_difference = actual.logsumexp(-1) - cached.logsumexp(-1)
        predicted_delta = selected_difference - normalizer_difference
        fp64_delta = (actual.log_softmax(-1) - cached.log_softmax(-1)).gather(
            1, sampled[:, None],
        ).squeeze(1)
        decomposition_error = max(decomposition_error,
                                  (fp64_delta - predicted_delta).abs().max().item())
        residual = observed_delta[rows, step] - fp64_delta
        normalization_residual = max(normalization_residual, residual.abs().max().item())
        fp64_delta_values.extend(fp64_delta.tolist())
        observed_ratios = observed_delta[rows, step].expm1().abs()
        index = observed_ratios.argmax().item()
        if observed_ratios[index].item() > worst_ratio_error:
            worst_ratio_error = observed_ratios[index].item()
            row = rows[index].item()
            token = sampled[index].item()
            prefix_width = rollout["prompt_width"] + step
            prefix_ids = rollout["input_ids"][row, :prefix_width].cpu()
            prefix_mask = rollout["attention_mask"][row, :prefix_width].cpu()
            worst = dict(
                row=row, completion_step_zero_based=step, sampled_token_id=token,
                prediction_column_zero_based=prefix_width - 1,
                padded_prefix_token_ids=prefix_ids.tolist(),
                prefix_attention_mask=prefix_mask.tolist(),
                semantic_prefix_token_ids=prefix_ids[prefix_mask.bool()].tolist(),
                cached_selected_logit=cached[index, token].item(),
                no_cache_selected_logit=actual[index, token].item(),
                selected_logit_difference=selected_difference[index].item(),
                cached_logsumexp=cached[index].logsumexp(-1).item(),
                no_cache_logsumexp=actual[index].logsumexp(-1).item(),
                normalizer_difference=normalizer_difference[index].item(),
                predicted_delta_logp_from_logits=predicted_delta[index].item(),
                fp64_delta_logp=fp64_delta[index].item(),
                observed_fp32_delta_logp=observed_delta[row, step].item(),
                observed_probability_ratio=observed_delta[row, step].exp().item(),
                fp64_probability_ratio=fp64_delta[index].exp().item(),
            )
    finite = all(torch.isfinite(z).all().item() for z in cached_logits + no_cache_logits)
    report.require(label + "_paired_logits_finite_and_fp64_decomposition",
                   finite and decomposition_error <= 1e-10,
                   max_absolute_logit_difference=max_logit_difference,
                   max_absolute_fp64_decomposition_residual=decomposition_error)

    snapshot = report.path.parent / (label + "_worst_logits.safetensors")
    row = worst["row"]
    step = worst["completion_step_zero_based"]
    save_file({"cached_logits": cached_logits[step][row].contiguous(),
               "no_cache_logits": no_cache_logits[step][row].contiguous()}, str(snapshot))
    worst["snapshot"] = dict(path=str(snapshot), stored_dtype="FP32",
                             bytes=snapshot.stat().st_size,
                             sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest())
    if check_prefix:
        # Keep all eight batch rows. Only sequence width is changed, while the
        # selected row's token history, attention mask and position IDs stay fixed.
        prefix_width = rollout["prompt_width"] + step
        with torch.no_grad(), forward_context(device, dtype):
            prefix_hidden = _decoder_hidden(
                model, rollout["input_ids"][:, :prefix_width],
                rollout["attention_mask"][:, :prefix_width],
            )
            prefix_logits = _one_position_logits(model, prefix_hidden, prefix_width - 1, 1.0)
        prefix_cpu = prefix_logits[row].double().cpu()
        cached_cpu = cached_logits[step][row].double()
        full_cpu = no_cache_logits[step][row].double()
        token = worst["sampled_token_id"]
        worst["prefix_width_control"] = dict(
            batch_size=len(prompts), prefix_width=prefix_width,
            fixed_width=rollout["fixed_width"],
            max_absolute_logits_vs_cached=(prefix_cpu - cached_cpu).abs().max().item(),
            max_absolute_logits_vs_fixed_width=(prefix_cpu - full_cpu).abs().max().item(),
            selected_delta_logp_vs_cached=(prefix_cpu.log_softmax(-1)[token]
                                          - cached_cpu.log_softmax(-1)[token]).item(),
        )
    fp64_deltas = torch.tensor(fp64_delta_values, dtype=torch.float64)
    record = dict(
        label=label, forward_dtype=str(dtype), probability_dtype="FP32",
        cpu_analysis_dtype="FP64", model_count=1, optimizer_steps=0,
        prompt_lengths=list(map(len, prompts)), batch_size=len(prompts),
        fixed_width=rollout["fixed_width"], generated_steps=rollout["generated_steps"],
        completion_lengths=rollout["completion_lengths"].cpu().tolist(),
        completion_ids=rollout["completion_ids"].cpu().tolist(),
        valid_token_count=mask.sum().item(),
        generation_with_hook_seconds=generation_seconds,
        fixed_width_recomputation_seconds=recomputation_seconds,
        pre_softmax=dict(max_absolute_logit_difference=max_logit_difference,
                        mean_absolute_logit_difference=difference_sum / element_count,
                        max_absolute_centered_logit_difference=max_centered_difference,
                        unequal_vocab_elements=unequal_count,
                        compared_vocab_elements=element_count,
                        logits_identical=(unequal_count == 0)),
        actual_fp32_probability_comparison=likelihood_alignment_metrics(
            cached_gpu_logps, no_cache_logps.cpu(), mask),
        fp64_probability_comparison=dict(
            max_absolute_delta_logp=fp64_deltas.abs().max().item(),
            max_absolute_ratio_minus_one=fp64_deltas.expm1().abs().max().item(),
            max_absolute_actual_minus_fp64_delta_logp=normalization_residual),
        worst_probability_position=worst,
    )
    report.data["cases"].append(record)
    report.write()
    print(label, "pre-softmax logits:", record["pre_softmax"], flush=True)
    print(label, "worst probability position:", worst, flush=True)
    return record
