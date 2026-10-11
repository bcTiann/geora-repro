"""Replay saved tokens and compare real cached/fixed logits in one dtype."""

import hashlib
import math
import time

import torch
from safetensors.torch import save_file
from transformers import GenerationConfig, LogitsProcessorList

from cache_layer_probe import SavedTokens, saved_batch
from geora_check_utils import forward_context
from grpo_rollout import _decoder_hidden, _one_position_logits, require_zero_dropout


def compare_saved_precision(model, prompts, case, pad_id, eos_id, device, dtype, report, label):
    """Keep old answers fixed; hooks capture logits before token forcing."""
    started = time.perf_counter()
    require_zero_dropout(model)
    ids, mask, width = saved_batch(prompts, case['completion_ids'], case['completion_lengths'], pad_id, device)
    steps = len(case['completion_ids'][0])
    versions = {name: p._version for name, p in model.named_parameters()}
    report.require(label + '_one_frozen_original_model',
        not model.training and all(not p.requires_grad for p in model.parameters())
        and not any(type(m).__name__ in ('GeoRALinear', 'LoRALinear') for m in model.modules()))
    report.require(label + '_saved_batch_dimensions_match',
        list(map(len, prompts)) == case['prompt_lengths'] and ids.shape[1] == case['fixed_width'])
    cached_logits = []
    cached_inputs = []
    fixed_inputs = []

    def capture_head(module, args, output):
        cached_logits.append(output[:, -1].detach().float().cpu().clone())

    def capture_inputs(target):
        def hook(module, args, kwargs):
            target.append({name: kwargs[name].detach().cpu().clone()
                           for name in ('input_ids', 'position_ids', 'attention_mask')})
        return hook

    config = GenerationConfig(
        do_sample=False, use_cache=True, cache_implementation='dynamic',
        max_new_tokens=steps, num_beams=1, num_return_sequences=1,
        pad_token_id=pad_id, eos_token_id=[eos_id], bos_token_id=model.config.bos_token_id,
        return_dict_in_generate=True, output_scores=False, output_logits=False, disable_compile=True)
    handles = [model.lm_head.register_forward_hook(capture_head),
               model.model.register_forward_pre_hook(capture_inputs(cached_inputs), with_kwargs=True)]
    print(label + ': replay saved answers with native KV cache', flush=True)
    try:
        with torch.no_grad(), forward_context(device, dtype):
            generated = model.generate(
                input_ids=ids[:, :width], attention_mask=mask[:, :width], generation_config=config,
                logits_processor=LogitsProcessorList([SavedTokens(width, ids[:, width:])]))
    finally:
        for handle in handles:
            handle.remove()
    report.require(label + '_exact_saved_tokens_replayed',
        torch.equal(generated.sequences[:, width:].cpu(), ids[:, width:].cpu()))
    report.require(label + '_all_real_vocab_logits_captured_before_forcing',
        len(cached_logits) == len(cached_inputs) == steps
        and all(x.shape == (len(prompts), model.config.vocab_size) for x in cached_logits))

    print(label + ': calculate the same fixed-width sequence without cache', flush=True)
    handle = model.model.register_forward_pre_hook(capture_inputs(fixed_inputs), with_kwargs=True)
    try:
        with torch.no_grad(), forward_context(device, dtype):
            hidden = _decoder_hidden(model, ids, mask)
            fixed_logits = [_one_position_logits(model, hidden, width + step - 1, 1.0)
                            .detach().cpu().clone() for step in range(steps)]
            del hidden
    finally:
        handle.remove()
    cached_ids = torch.cat([x['input_ids'] for x in cached_inputs], dim=1)
    cached_positions = torch.cat([x['position_ids'] for x in cached_inputs], dim=1)
    fixed = fixed_inputs[0]
    input_scope_ok = True
    for row, length in enumerate(case['completion_lengths']):
        end = width + length - 1
        valid = mask[row, :end].cpu().bool()
        input_scope_ok &= torch.equal(cached_ids[row, :end][valid], fixed['input_ids'][row, :end][valid])
        input_scope_ok &= torch.equal(cached_positions[row, :end][valid], fixed['position_ids'][row, :end][valid])
        for step in range(length):
            prefix_end = width + step
            input_scope_ok &= torch.equal(cached_inputs[step]['attention_mask'][row, :prefix_end],
                                          fixed['attention_mask'][row, :prefix_end])
    report.require(label + '_all_valid_prediction_histories_positions_and_input_masks_match', input_scope_ok)
    report.require(label + '_both_logits_finite',
        all(torch.isfinite(z).all().item() for z in cached_logits + fixed_logits))
    examples = []
    snapshot_values = {}
    identity_error = 0.0
    for step, (cached, actual) in enumerate(zip(cached_logits, fixed_logits)):
        cached = cached.double()
        actual = actual.double()
        logp_cached = cached.log_softmax(-1)
        logp_fixed = actual.log_softmax(-1)
        logz_cached = cached.logsumexp(-1)
        logz_fixed = actual.logsumexp(-1)
        for row, length in enumerate(case['completion_lengths']):
            if step >= length:
                continue
            token = case['completion_ids'][row][step]
            lc = logp_cached[row, token].item()
            lf = logp_fixed[row, token].item()
            delta_logit = (actual[row, token] - cached[row, token]).item()
            delta_logz = (logz_fixed[row] - logz_cached[row]).item()
            identity_error = max(identity_error, abs((lf-lc)-(delta_logit-delta_logz)))
            examples.append(dict(row=row, completion_step=step, token_id=token,
                cached_logp=lc, fixed_logp=lf, cached_probability=math.exp(lc), fixed_probability=math.exp(lf),
                relative_probability_difference=math.expm1(lf-lc), delta_logp=lf-lc,
                cached_logit=cached[row,token].item(), fixed_logit=actual[row,token].item(),
                selected_logit_difference=delta_logit, logsumexp_difference=delta_logz))
            # Keep only effective prediction positions: one full-vocab vector
            # per valid token. FP32 saves BF16 values without information loss.
            snapshot_values[f'cached.row{row}.step{step}'] = cached[row].float().contiguous()
            snapshot_values[f'fixed.row{row}.step{step}'] = actual[row].float().contiguous()
    report.require(label + '_all_valid_tokens_and_softmax_identity',
        len(examples) == sum(case['completion_lengths']) == case['valid_token_count'] and identity_error < 1e-12,
        token_count=len(examples), max_softmax_identity_residual=identity_error)
    report.require(label + '_first_eos_included_later_padding_excluded',
        all(not any(t == eos_id for t in tokens[:length-1]) and
            (length == steps or tokens[length-1] == eos_id) and
            all(t == pad_id for t in tokens[length:])
            for tokens, length in zip(case['completion_ids'],case['completion_lengths'])))
    report.require(label + '_parameters_and_gradients_unchanged',
        all(p._version == versions[name] and p.grad is None and not p.requires_grad
            for name,p in model.named_parameters()))
    path = report.path.parent / (label + '_valid_logits.safetensors')
    save_file(snapshot_values, str(path))
    record = dict(label=label, forward_dtype=str(dtype), analysis_dtype='CPU FP64',
        sampled_token_probabilities=examples, valid_token_count=len(examples), fixed_width=ids.shape[1],
        max_absolute_delta_logp=max(abs(x['delta_logp']) for x in examples),
        mean_absolute_delta_logp=sum(abs(x['delta_logp']) for x in examples)/len(examples),
        max_absolute_relative_probability_difference=max(abs(x['relative_probability_difference']) for x in examples),
        worst_probability_position=max(examples,key=lambda x:abs(x['relative_probability_difference'])),
        elapsed_seconds=time.perf_counter()-started,
        snapshot=dict(path=str(path), bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    report.data.setdefault('cases',[]).append(record)
    report.write()
    print(label + ': maximum relative probability difference = '
          f"{100*record['max_absolute_relative_probability_difference']:.8f}%", flush=True)
    return record, snapshot_values
