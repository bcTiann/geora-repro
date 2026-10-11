"""Controlled attention output interventions on a forced, saved token history.

Capture native probabilities P and repeated values V. Separate query/reduction
shape changes from historical-V changes. Target-only replacement measures a
conditional effect: other full-path historical states remain unchanged.
"""
from functools import partial
import hashlib
import math
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers.models.qwen2 import modeling_qwen2

from cache_layer_probe import LayerTrace, differences, saved_batch, trace_path
from geora_check_utils import forward_context


class AttentionCapture(LayerTrace):
    def __init__(self, *args, replacement=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.replacement = replacement
        self.native = {}
        self.patch_checks = []

    def attention_call(self, module, query, key, value, mask, original, **kwargs):
        output, probabilities = super().attention_call(
            module, query, key, value, mask, original, **kwargs)
        if module.layer_idx != 0:
            return output, probabilities
        # Only the last forced decode call is eligible for a replacement.
        final_call = (self.path != 'cached' or key.shape[2] == self.prefix_width)
        if not final_call:
            return output, probabilities
        repeated_value = modeling_qwen2.repeat_kv(value, module.num_key_value_groups)
        self.native = {
            'P': probabilities.detach().cpu().clone(),
            'V': repeated_value.detach().cpu().clone(),
            'output': output.detach().cpu().clone(),
        }
        if self.replacement is None:
            return output, probabilities
        column = -1 if self.path == 'cached' else self.target_column
        patched = output.clone()
        patched[self.row, column] = self.replacement.to(output.device, output.dtype)
        difference = patched != output
        difference[self.row, column] = False
        self.patch_checks.append(dict(unaffected_elements_exact=not difference.any().item(),
                                      patch_shape=list(self.replacement.shape)))
        return patched, probabilities


def product_controls(cached, fixed, row, column, prefix_width, valid, dtype, device):
    """2x2 query shape x historical values, plus a separate reduction-width check."""
    p = fixed.native['P'].to(device)
    value = fixed.native['V'].to(device)
    cached_value = cached.native['V'].to(device)
    valid = valid.to(device)
    swapped = value.clone()
    swapped[row, :, :prefix_width] = torch.where(
        valid[None, :, None], cached_value[row], value[row, :, :prefix_width])
    products = {}
    with torch.no_grad(), forward_context(device, dtype):
        reconstructed = torch.matmul(p, value).transpose(1, 2).contiguous()
        reconstructed_cached = torch.matmul(
            cached.native['P'].to(device), cached_value).transpose(1, 2).contiguous()
        for label, v in (('original', value), ('cached_V', swapped)):
            full = torch.matmul(p, v).transpose(1, 2).contiguous()
            one = torch.matmul(p[:, :, column:column+1].contiguous(), v)
            trimmed = torch.matmul(
                p[:, :, column:column+1, :prefix_width].contiguous(),
                v[:, :, :prefix_width].contiguous())
            products['full_' + label] = full[row, column].detach().cpu()
            products['one_' + label] = one[row, :, 0].detach().cpu()
            products['trimmed_' + label] = trimmed[row, :, 0].detach().cpu()
    baseline = products['full_original']
    oracle_p = fixed.native['P'][row, :, column].double()
    delta_v = (swapped[row].cpu().double() - fixed.native['V'][row].double())
    predicted_delta = torch.matmul(oracle_p[:, None, :], delta_v).squeeze(1)
    oracle_before = torch.matmul(oracle_p[:, None, :], fixed.native['V'][row].double()).squeeze(1)
    oracle_after = torch.matmul(oracle_p[:, None, :], swapped[row].cpu().double()).squeeze(1)
    native_delta = products['full_cached_V'].double() - baseline.double()
    interaction = ((products['one_cached_V'].double() - products['full_cached_V'].double())
                   - (products['one_original'].double() - baseline.double()))
    record = dict(
        cached_reconstructed_exact=torch.equal(reconstructed_cached.cpu(), cached.native['output']),
        fixed_reconstructed_exact=torch.equal(reconstructed.cpu(), fixed.native['output']),
        future_target_probability_zero=not p[row, :, column, prefix_width:].any().item(),
        P_target_cached_equals_fixed=torch.equal(
            cached.native['P'][row, :, -1], fixed.native['P'][row, :, column, :prefix_width]),
        value_difference=differences(value[row, :, :prefix_width].cpu()[:, valid.cpu()],
                                     cached_value[row].cpu()[:, valid.cpu()]),
        query_shape_effect=differences(baseline, products['one_original']),
        historical_V_effect=differences(baseline, products['full_cached_V']),
        crossed_effect=differences(baseline, products['one_cached_V']),
        reduction_width_effect=differences(products['one_original'], products['trimmed_original']),
        combined_difference_to_native_cached=differences(
            cached.native['output'][row, -1], products['trimmed_cached_V']),
        interaction_max_absolute=interaction.abs().max().item(),
        fp64_P_deltaV_identity_residual=(oracle_after-oracle_before-predicted_delta).abs().max().item(),
        fp64_predicted_delta_max_absolute=predicted_delta.abs().max().item(),
        native_delta_minus_fp64_prediction_max_absolute=(native_delta-predicted_delta).abs().max().item(),
        shapes=dict(full_P=list(p.shape), full_V=list(value.shape),
                    cached_P=list(cached.native['P'].shape), cached_V=list(cached_value.shape)),
    )
    return record, products


def inspect_interventions(model, prompts, case, row, step, pad_id, eos_ids,
                          dtype, device, report, label, expected_snapshot=None):
    ids, mask, prompt_width = saved_batch(prompts, case['completion_ids'],
                                         case['completion_lengths'], pad_id, device)
    prefix_width = prompt_width + step
    column = prefix_width - 1
    valid = mask[row, :prefix_width].bool().cpu()
    versions = {name: p._version for name, p in model.named_parameters()}
    report.require(label+'_frozen_no_dropout_model', not model.training
                   and all(not p.requires_grad for p in model.parameters())
                   and model.config.attention_dropout == 0)
    traces, logits = {}, {}
    for path in ('cached', 'fixed_no_cache'):
        traces[path], logits[path] = trace_path(
            model, ids, mask, prompt_width, step, row, pad_id, eos_ids,
            dtype, device, path, trace_factory=AttentionCapture)
    cached, fixed = traces['cached'], traces['fixed_no_cache']
    report.require(label+'_forced_saved_history', torch.equal(
        cached.generated_ids, ids[:, prompt_width:prompt_width+step+1].cpu()))
    joined_ids=sum((item['input_ids'] for item in cached.inputs),[])
    joined_positions=sum((item['position_ids'] for item in cached.inputs),[])
    report.require(label+'_actual_input_and_positions_equal',
                   joined_ids==fixed.inputs[0]['input_ids'][:prefix_width]
                   and joined_positions==fixed.inputs[0]['position_ids'][:prefix_width])
    report.require(label+'_actual_allowed_history_matches',
                   torch.equal(cached.attention['allowed_keys'], valid)
                   and torch.equal(fixed.attention['allowed_keys'], valid))
    if expected_snapshot is not None:
        report.require(label+'_previous_raw_logits_recovered',
                       torch.equal(logits['cached'], expected_snapshot['cached_logits'])
                       and torch.equal(logits['fixed_no_cache'], expected_snapshot['no_cache_logits']))
    controls, products = product_controls(cached, fixed, row, column, prefix_width, valid, dtype, device)
    report.require(label+'_native_products_recovered',
                   controls['cached_reconstructed_exact'] and controls['fixed_reconstructed_exact'])
    report.require(label+'_unseen_future_probability_zero', controls['future_target_probability_zero'])
    tensors = {}
    for path, trace in traces.items():
        for name, tensor in trace.native.items():
            tensors[path+'.'+name] = tensor.contiguous()
        tensors[path+'.logits'] = logits[path].contiguous()
    snapshot_path = report.path.parent / (label+'_attention.safetensors')
    save_file(tensors, str(snapshot_path))
    artifact = dict(path=str(snapshot_path), bytes=snapshot_path.stat().st_size,
                    sha256=hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
                    native_dtype=str(dtype), tensors={k:list(v.shape) for k,v in tensors.items()})
    token = case['completion_ids'][row][step]
    base_logp = logits['fixed_no_cache'].double().log_softmax(-1)[token].item()
    cached_logp = logits['cached'].double().log_softmax(-1)[token].item()
    interventions = []
    replacements = {
        'no_op': fixed.native['output'][row, column],
        'query_shape_only': products['one_original'],
        'historical_V_only': products['full_cached_V'],
        'query_shape_and_V': products['one_cached_V'],
        'native_cached_target_output': cached.native['output'][row, -1],
    }
    for name, replacement in replacements.items():
        trace, z = trace_path(
            model, ids, mask, prompt_width, step, row, pad_id, eos_ids, dtype, device,
            'fixed_no_cache', trace_factory=partial(AttentionCapture, replacement=replacement))
        report.require(label+'_'+name+'_only_target_patched', len(trace.patch_checks)==1
                       and all(item['unaffected_elements_exact'] for item in trace.patch_checks))
        if name == 'no_op':
            report.require(label+'_no_op_exact_logits', torch.equal(z, logits['fixed_no_cache']))
        logp = z.double().log_softmax(-1)[token].item()
        interventions.append(dict(
            label=name, logits_vs_fixed=differences(logits['fixed_no_cache'], z),
            logits_vs_cached=differences(logits['cached'], z),
            selected_logit=z[token].item(), selected_logp=logp,
            probability_ratio_to_fixed=math.exp(logp-base_logp),
            probability_ratio_to_cached=math.exp(logp-cached_logp)))
    report.require(label+'_parameter_versions_and_gradients_unchanged',
                   all(p._version==versions[name] and p.grad is None for name,p in model.named_parameters()))
    record = dict(label=label, row=row, completion_step=step, token_id=token,
                  controls=controls, interventions=interventions, raw_snapshot=artifact,
                  baseline_probability_ratio_fixed_to_cached=math.exp(base_logp-cached_logp),
                  scope='Conditional target-position layer0 interventions; other fixed-path states retained')
    report.data.setdefault('cases', []).append(record)
    report.write()
    print(label, 'local effects:', controls['query_shape_effect']['max_absolute'],
          controls['historical_V_effect']['max_absolute'], flush=True)
    return record
