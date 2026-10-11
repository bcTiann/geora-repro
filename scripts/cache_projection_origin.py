"""Separate native KV storage from shape-dependent historical V projection.

Replay a saved token sequence on one frozen native Qwen. Capture tensors before
cache.update and after cache retrieval. Then reuse identical BF16 input/weight
values in the original chunk shapes and in one full sequence multiplication.
This diagnostic never updates model parameters or changes training code.
"""

from contextlib import contextmanager
import hashlib
import math

import torch
from torch.nn import functional as F
from safetensors.torch import save_file

from cache_layer_probe import LayerTrace, differences, saved_batch, trace_path
from geora_check_utils import forward_context


class ProjectionTrace(LayerTrace):
    """Capture the native layer-0 V values before and after cache storage."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.projection_inputs = []
        self.projection_outputs = []
        self.projection_layouts = []
        self.cache_value = None

    def projection_input(self, module, args):
        tensor = args[0]
        self.projection_inputs.append(tensor.detach().cpu().contiguous().clone())
        self.projection_layouts.append(dict(shape=list(tensor.shape), stride=list(tensor.stride()),
                                            dtype=str(tensor.dtype), storage_offset=tensor.storage_offset(),
                                            contiguous=tensor.is_contiguous(), pointer_modulo_256=tensor.data_ptr() % 256))

    def projection_output(self, module, args, output):
        self.projection_outputs.append(output.detach().cpu().contiguous().clone())

    def attention_call(self, module, query, key, value, mask, original, **kwargs):
        output = super().attention_call(module, query, key, value, mask, original, **kwargs)
        if module.layer_idx == 0:
            self.cache_value = value.detach().cpu().contiguous().clone()
        return output

    @contextmanager
    def installed(self):
        with super().installed():
            projection = self.model.model.layers[0].self_attn.v_proj
            handles = [projection.register_forward_pre_hook(self.projection_input),
                       projection.register_forward_hook(self.projection_output)]
            try:
                yield self
            finally:
                for handle in handles:
                    handle.remove()


def restore_layout(tensor, layout, device):
    """Restore original shape and stride while preserving every numeric value."""
    result = torch.empty_strided(layout['shape'], layout['stride'], dtype=tensor.dtype, device=device)
    result.copy_(tensor.to(device))
    return result


def project_chunks(layer, tensors, layouts, dtype, device, *, promoted=False, contiguous=False):
    outputs = []
    with torch.no_grad():
        for tensor, layout in zip(tensors, layouts):
            inputs = restore_layout(tensor, layout, device)
            if contiguous:
                inputs = inputs.contiguous()
            if promoted:
                # Preserve the exact stored BF16 values, change arithmetic only.
                with torch.autocast(device_type=torch.device(device).type, enabled=False):
                    bias = None if layer.bias is None else layer.bias.float()
                    result = F.linear(inputs.float(), layer.weight.float(), bias)
            else:
                with forward_context(device, dtype):
                    result = layer(inputs)
            outputs.append(result.detach().cpu().contiguous())
    return torch.cat(outputs, dim=1)


def inspect_projection_origin(model, prompts, case, row, step, pad_id, eos_ids,
                              dtype, device, report, expected_snapshot=None):
    ids, mask, prompt_width = saved_batch(prompts, case['completion_ids'],
                                         case['completion_lengths'], pad_id, device)
    prefix_width = prompt_width + step
    valid = mask[row, :prefix_width].bool().cpu()
    versions = {name: parameter._version for name, parameter in model.named_parameters()}
    report.require('frozen_native_model_no_dropout', not model.training
                   and all(not p.requires_grad for p in model.parameters())
                   and model.config.attention_dropout == 0
                   and all(type(m).__name__ not in ('GeoRALinear', 'LoRALinear') for m in model.modules()))
    traces, logits = {}, {}
    for path in ('cached', 'fixed_no_cache'):
        traces[path], logits[path] = trace_path(
            model, ids, mask, prompt_width, step, row, pad_id, eos_ids, dtype, device,
            path, trace_factory=ProjectionTrace)
    cached = traces['cached']
    fixed = traces['fixed_no_cache']
    report.require('forced_saved_tokens_replayed', torch.equal(
        cached.generated_ids, ids[:, prompt_width:prefix_width + 1].cpu()))
    report.require('actual_target_tokens_and_positions_equal',
                   sum((entry['input_ids'] for entry in cached.inputs), [])
                   == fixed.inputs[0]['input_ids'][:prefix_width]
                   and sum((entry['position_ids'] for entry in cached.inputs), [])
                   == fixed.inputs[0]['position_ids'][:prefix_width])
    report.require('actual_target_allowed_history_equal',
                   torch.equal(cached.attention['allowed_keys'], valid)
                   and torch.equal(fixed.attention['allowed_keys'], valid))
    if expected_snapshot is not None:
        report.require('previous_raw_logits_reproduced_exactly',
                       torch.equal(logits['cached'], expected_snapshot['cached_logits'])
                       and torch.equal(logits['fixed_no_cache'], expected_snapshot['no_cache_logits']))

    joined_inputs = torch.cat(cached.projection_inputs, dim=1)
    joined_outputs = torch.cat(cached.projection_outputs, dim=1)
    fixed_inputs = fixed.projection_inputs[0]
    fixed_outputs = fixed.projection_outputs[0]
    report.require('all_batch_historical_projection_inputs_exactly_equal',
                   torch.equal(joined_inputs, fixed_inputs[:, :prefix_width]),
                   **differences(joined_inputs, fixed_inputs[:, :prefix_width]))
    for path, trace in traces.items():
        raw = torch.cat(trace.projection_outputs, dim=1)
        heads = trace.cache_value.shape[1]
        dimension = trace.cache_value.shape[-1]
        expected = raw.view(raw.shape[0], raw.shape[1], heads, dimension).transpose(1, 2)
        report.require(path + '_attention_V_exactly_preserves_raw_projection',
                       torch.equal(expected, trace.cache_value),
                       **differences(expected, trace.cache_value))

    layer = model.model.layers[0].self_attn.v_proj
    chunk_tensors = []
    offset = 0
    for layout in cached.projection_layouts:
        length = layout['shape'][1]
        chunk_tensors.append(fixed_inputs[:, offset:offset + length].contiguous())
        offset += length
    full_tensors = [fixed_inputs]
    original_chunk = project_chunks(layer, chunk_tensors, cached.projection_layouts, dtype, device)
    original_full = project_chunks(layer, full_tensors, fixed.projection_layouts, dtype, device)
    repeat_chunk = project_chunks(layer, chunk_tensors, cached.projection_layouts, dtype, device)
    repeat_full = project_chunks(layer, full_tensors, fixed.projection_layouts, dtype, device)
    report.require('same_inputs_chunk_shape_recovers_native_projection', torch.equal(original_chunk, joined_outputs))
    report.require('same_inputs_full_shape_recovers_native_projection', torch.equal(original_full, fixed_outputs))
    report.require('identical_shape_repeats_are_exact',
                   torch.equal(repeat_chunk, original_chunk) and torch.equal(repeat_full, original_full))
    contiguous_chunk = project_chunks(layer, chunk_tensors, cached.projection_layouts,
                                     dtype, device, contiguous=True)
    contiguous_full = project_chunks(layer, full_tensors, fixed.projection_layouts,
                                    dtype, device, contiguous=True)
    report.require('contiguous_copy_controls_preserve_results',
                   torch.equal(contiguous_chunk, original_chunk) and torch.equal(contiguous_full, original_full))

    promoted_chunk = project_chunks(layer, chunk_tensors, cached.projection_layouts,
                                   dtype, device, promoted=True)
    promoted_full = project_chunks(layer, full_tensors, fixed.projection_layouts,
                                  dtype, device, promoted=True)
    selected_fixed = fixed_outputs[row, :prefix_width]
    selected_cached = joined_outputs[row]
    coordinates = torch.nonzero((selected_cached != selected_fixed) & valid[:, None])
    weight = layer.weight.detach().cpu()
    bias = None if layer.bias is None else layer.bias.detach().cpu()
    mismatches = []
    for column, channel in coordinates.tolist():
        inputs64 = fixed_inputs[row, column].double()
        weight64 = weight[channel].double()
        dot64 = torch.dot(inputs64, weight64).item()
        bias64 = 0.0 if bias is None else bias[channel].double().item()
        oracle64 = dot64 + bias64
        a = selected_cached[column, channel].item()
        b = selected_fixed[column, channel].item()
        midpoint = (a + b) / 2
        adjacent = torch.nextafter(torch.tensor(a, dtype=dtype), torch.tensor(b, dtype=dtype)).item() == b
        rounded = torch.tensor(oracle64, dtype=torch.float64).to(dtype).item()
        mismatches.append(dict(
            padded_column=column, token_id=ids[row, column].item(),
            position_id=fixed.inputs[0]['position_ids'][column],
            projection_call='prefill' if column < prompt_width else 'decode',
            channel=channel, kv_head=channel // cached.cache_value.shape[-1],
            head_channel=channel % cached.cache_value.shape[-1],
            cached_V=a, fixed_V=b, fp64_dot=dot64, fp64_bias=bias64,
            fp64_dot_plus_bias=oracle64, fp64_rounded_to_storage_dtype=rounded,
            cached_error_to_fp64=a-oracle64, fixed_error_to_fp64=b-oracle64,
            midpoint_between_native_values=midpoint,
            native_values_are_adjacent=adjacent,
            oracle_minus_midpoint=oracle64-midpoint,
            distance_to_midpoint_in_native_gaps=abs(oracle64-midpoint)/abs(a-b),
            promoted_chunk_fp32=promoted_chunk[row, column, channel].item(),
            promoted_full_fp32=promoted_full[row, column, channel].item(),
            promoted_chunk_rounded=promoted_chunk[row, column, channel].to(dtype).item(),
            promoted_full_rounded=promoted_full[row, column, channel].to(dtype).item()))

    snapshot = {'fixed_inputs': fixed_inputs, 'weight': weight,
                'cached_outputs': joined_outputs, 'fixed_outputs': fixed_outputs,
                'cache_retrieved_V': cached.cache_value,
                'promoted_chunk_outputs': promoted_chunk, 'promoted_full_outputs': promoted_full,
                'cached_logits': logits['cached'], 'fixed_logits': logits['fixed_no_cache'],
                'input_ids': ids.cpu(), 'attention_mask': mask.cpu()}
    if bias is not None:
        snapshot['bias'] = bias
    snapshot_path = report.path.parent / 'projection_origin.safetensors'
    save_file({name: tensor.contiguous() for name, tensor in snapshot.items()}, str(snapshot_path))
    token = case['completion_ids'][row][step]
    logp_cached = logits['cached'].double().log_softmax(-1)[token].item()
    logp_fixed = logits['fixed_no_cache'].double().log_softmax(-1)[token].item()
    record = dict(row=row, completion_step=step, token_id=token,
                  prefix_width=prefix_width, prompt_width=prompt_width,
                  fixed_width=ids.shape[1], actual_call_layouts={
                      path: trace.projection_layouts for path, trace in traces.items()},
                  historical_V_difference=differences(selected_cached[valid], selected_fixed[valid]),
                  promoted_fp32_difference=differences(promoted_chunk[row, valid], promoted_full[row, :prefix_width][valid]),
                  promoted_then_rounded_difference=differences(promoted_chunk[row, valid].to(dtype),
                                                              promoted_full[row, :prefix_width][valid].to(dtype)),
                  mismatches=mismatches, fixed_to_cached_token_probability_ratio=math.exp(logp_fixed-logp_cached),
                  snapshot=dict(path=str(snapshot_path), bytes=snapshot_path.stat().st_size,
                                sha256=hashlib.sha256(snapshot_path.read_bytes()).hexdigest()),
                  scope='Layer-0 historical V origin, not a unique attribution of all final logits differences')
    report.require('model_parameters_and_gradients_unchanged',
                   all(parameter._version == versions[name] and parameter.grad is None
                       for name, parameter in model.named_parameters()))
    report.data['case'] = record
    report.write()
    print('Historical V mismatches:', len(mismatches), flush=True)
    for item in mismatches:
        print(item, flush=True)
    return record
