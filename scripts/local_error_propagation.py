"""Trace and separate two execution paths' hidden-state differences.

The counterfactual uses the fixed-width path but aligns the ENTIRE valid
prefix at a layer's input. It separates that layer's response to prior state
differences from its remaining execution-path difference at aligned inputs.
All FP64 analysis happens on CPU, never in the model's Metal forward.
"""

from contextlib import contextmanager
import hashlib
import math

import torch
from safetensors.torch import save_file

from cache_layer_probe import LayerTrace, differences, saved_batch, trace_path
from geora_check_utils import forward_context
from grpo_rollout import _decoder_hidden, _one_position_logits


class PropagationTrace(LayerTrace):
    """Read-only target, causal-history and attention observations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_attention = {}
        self.final_norm_batch = None
        self.post_normalized_full_batch = None

    def capture(self, name, tensor, history=False):
        super().capture(name, tensor, history)
        if name == 'final_norm':
            column = -1 if self.path == 'cached' else self.target_column
            self.final_norm_batch = tensor[:, column].detach().cpu().clone()
        if name == 'layer.0.post_attention_norm' and self.path == 'fixed_no_cache':
            self.post_normalized_full_batch = tensor.detach().clone()

    def attention_call(self, module, query, key, value, mask, original, **kwargs):
        output, probabilities = super().attention_call(
            module, query, key, value, mask, original, **kwargs)
        column = -1 if self.path == 'cached' else self.target_column
        self.all_attention[module.layer_idx] = {
            'query': query[self.row, :, column].detach().float().cpu().clone(),
            'key': key[self.row, :, :self.prefix_width].detach().float().cpu().clone(),
            'value': value[self.row, :, :self.prefix_width].detach().float().cpu().clone(),
            'probabilities': probabilities[self.row, :, column, :self.prefix_width].detach().float().cpu().clone(),
            'allowed_keys': (mask[self.row, 0, column, :self.prefix_width] == 0).detach().cpu().clone(),
        }
        return output, probabilities

    @contextmanager
    def installed(self):
        extra_handles = []
        with super().installed():
            try:
                for index, block in enumerate(self.model.model.layers):
                    prefix = f'layer.{index}'
                    extra_handles.append(block.register_forward_pre_hook(self.input_hook(prefix + '.input')))
                    extra_handles.append(block.post_attention_layernorm.register_forward_pre_hook(
                        self.input_hook(prefix + '.attention_residual')))
                    extra_handles.append(block.mlp.register_forward_hook(self.output_hook(prefix + '.mlp_output')))
                yield self
            finally:
                for handle in extra_handles:
                    handle.remove()


def distribution(logits, token):
    z = logits.cpu().double()
    normalizer = z.logsumexp(-1).item()
    logp = z[token].item() - normalizer
    return dict(selected_logit=z[token].item(), logsumexp=normalizer,
                selected_logp=logp, selected_probability=math.exp(logp))


def compare_distributions(cached_logits, fixed_logits, token):
    cached = distribution(cached_logits, token)
    fixed = distribution(fixed_logits, token)
    delta_selected = fixed['selected_logit'] - cached['selected_logit']
    delta_normalizer = fixed['logsumexp'] - cached['logsumexp']
    delta_logp = fixed['selected_logp'] - cached['selected_logp']
    return dict(cached=cached, fixed=fixed, selected_logit_difference=delta_selected,
                normalizer_difference=delta_normalizer, logp_difference=delta_logp,
                probability_relative_difference=math.expm1(delta_logp),
                identity_residual=delta_logp - (delta_selected - delta_normalizer))


def aligned_layer_forward(model, ids, mask, row, column, valid, replacement,
                          layer_index, dtype, device, *, target_only=False):
    """Run fixed-width forward, changing one layer's input activations only."""
    prefix_width = valid.numel()
    valid_device = valid.to(device)
    state = {}
    block = model.model.layers[layer_index]

    def replace_input(module, args, kwargs):
        original = args[0] if args else kwargs['hidden_states']
        patched = original.clone()
        layout_before = dict(shape=list(original.shape), stride=list(original.stride()),
                             contiguous=original.is_contiguous(), storage_offset=original.storage_offset())
        layout_after = dict(shape=list(patched.shape), stride=list(patched.stride()),
                            contiguous=patched.is_contiguous(), storage_offset=patched.storage_offset())
        state['layout_before'] = layout_before
        state['layout_after'] = layout_after
        state['layout_preserved'] = layout_before == layout_after
        values = replacement.to(device=device, dtype=original.dtype)
        if target_only:
            patched[row, column] = values[column]
        else:
            patched[row, :prefix_width] = torch.where(
                valid_device[:, None], values, original[row, :prefix_width])
        unaffected = patched != original
        if target_only:
            unaffected[row, column] = False
        else:
            unaffected[row, :prefix_width] &= ~valid_device[:, None]
        state['unaffected_elements_exact'] = not unaffected.any().item()
        actual = patched[row, :prefix_width].detach().float().cpu()
        state['actual_input'] = actual.clone()
        if target_only:
            state['replacement_exact'] = torch.equal(actual[column], replacement[column])
        else:
            state['replacement_exact'] = torch.equal(actual[valid], replacement[valid])
        state['input_call_count'] = state.get('input_call_count', 0) + 1
        if args:
            return (patched,) + args[1:], kwargs
        return args, {**kwargs, 'hidden_states': patched}

    def capture_output(module, args, output):
        state['layer_output'] = output[row, column].detach().float().cpu().clone()
        state['history_output'] = output[row, :prefix_width].detach().float().cpu().clone()

    handles = [block.register_forward_pre_hook(replace_input, with_kwargs=True),
               block.register_forward_hook(capture_output)]
    try:
        with torch.no_grad(), forward_context(device, dtype):
            hidden = _decoder_hidden(model, ids, mask)
            logits = _one_position_logits(model, hidden, column, 1.0)[row].detach().cpu()
    finally:
        for handle in handles:
            handle.remove()
    return state, logits


def diagnose_propagation(model, prompts, case, row, step, pad_id, eos_ids,
                         dtype, device, report, label, expected_snapshot=None,
                         *, intervene=True):
    ids, mask, width = saved_batch(prompts, case['completion_ids'], case['completion_lengths'], pad_id, device)
    prefix_width = width + step
    column = prefix_width - 1
    valid = mask[row, :prefix_width].bool().cpu()
    token = case['completion_ids'][row][step]
    versions = {name: p._version for name, p in model.named_parameters()}
    traces, raw = {}, {}
    print(f'{label}: capture cached and fixed-width native forwards', flush=True)
    for path in ('cached', 'fixed_no_cache'):
        traces[path], raw[path] = trace_path(model, ids, mask, width, step, row,
            pad_id, eos_ids, dtype, device, path, trace_factory=PropagationTrace)
    cached, fixed = traces['cached'], traces['fixed_no_cache']
    report.require(label + '_same_saved_tokens', torch.equal(
        cached.generated_ids, ids[:, width:width + step + 1].cpu()))
    cached_ids = torch.tensor(sum((x['input_ids'] for x in cached.inputs), []))
    fixed_ids = torch.tensor(fixed.inputs[0]['input_ids'][:prefix_width])
    cached_positions = torch.tensor(sum((x['position_ids'] for x in cached.inputs), []))
    fixed_positions = torch.tensor(fixed.inputs[0]['position_ids'][:prefix_width])
    report.require(label + '_same_valid_ids_positions_and_all_layer_attention_masks',
        torch.equal(cached_ids[valid], fixed_ids[valid])
        and torch.equal(cached_positions[valid], fixed_positions[valid])
        and all(torch.equal(cached.all_attention[i]['allowed_keys'], valid)
                and torch.equal(fixed.all_attention[i]['allowed_keys'], valid)
                for i in range(len(model.model.layers))))
    report.require(label + '_finite_native_logits', all(torch.isfinite(z).all().item() for z in raw.values()))
    if expected_snapshot is not None:
        report.require(label + '_hooks_preserve_previous_full_vocab_logits',
            torch.equal(raw['cached'], expected_snapshot['cached_logits'])
            and torch.equal(raw['fixed_no_cache'], expected_snapshot['no_cache_logits']))
    ordered = ['embedding']
    layer_rows = []
    layer_count = len(model.model.layers)
    for index in range(layer_count):
        prefix = f'layer.{index}'
        ordered.extend(prefix + '.' + suffix for suffix in (
            'input', 'input_norm', 'q_proj', 'k_proj', 'v_proj', 'attention_output',
            'o_proj', 'attention_residual', 'post_attention_norm', 'gate_proj',
            'up_proj', 'down_proj', 'mlp_output', 'output'))
        name = prefix + '.output'
        attn_metrics = {}
        for key in ('query', 'key', 'value', 'probabilities'):
            a, b = cached.all_attention[index][key], fixed.all_attention[index][key]
            if key in ('key', 'value', 'probabilities'):
                a, b = a[:, valid], b[:, valid]
            attn_metrics[key] = differences(a, b)
        layer_rows.append(dict(layer=index,
            input=differences(cached.target[prefix + '.input'], fixed.target[prefix + '.input']),
            output=differences(cached.target[name], fixed.target[name]),
            valid_history_output=differences(cached.joined_history(name)[valid], fixed.joined_history(name)[valid]),
            attention=attn_metrics))
    ordered.extend(('final_norm', 'lm_head'))
    boundaries = [dict(boundary=name, **differences(cached.target[name], fixed.target[name])) for name in ordered]
    endpoint = compare_distributions(raw['cached'], raw['fixed_no_cache'], token)
    report.require(label + '_softmax_exact_decomposition', abs(endpoint['identity_residual']) < 1e-12)
    record = dict(label=label, dtype=str(dtype), row=row, completion_step=step, token_id=token,
        target_prefix_column=column, prefix_width=prefix_width, valid_prefix_tokens=int(valid.sum()),
        fixed_width=ids.shape[1], boundaries=boundaries, layers=layer_rows,
        first_target_difference=next((x['boundary'] for x in boundaries if not x['exact']), None),
        endpoint=endpoint, interventions=[])
    tensors = {}
    for path, trace in traces.items():
        for name, values in trace.target.items():
            tensors[path + '.target.' + name] = values.contiguous()
        for name in trace.history:
            tensors[path + '.history.' + name] = trace.joined_history(name).contiguous()
        for index, attention in trace.all_attention.items():
            for name, value in attention.items():
                tensors[f'{path}.attention.{index}.{name}'] = value.contiguous()
    if intervene:
        # The target's layer-0 post-attention input is equal in the real BF16
        # cases. Keep its values/weights fixed; change only one-vs-full shape.
        local_inputs = fixed.post_normalized_full_batch
        local_controls = []
        for name, module in (('gate_proj', model.model.layers[0].mlp.gate_proj),
                             ('up_proj', model.model.layers[0].mlp.up_proj),
                             ('mlp_output', model.model.layers[0].mlp)):
            with torch.no_grad(), forward_context(device, dtype):
                one = module(local_inputs[:, column:column+1].contiguous())[row, 0].detach().float().cpu()
                full = module(local_inputs)[row, column].detach().float().cpu()
            boundary = 'layer.0.' + name
            local_controls.append(dict(module=boundary,
                identical_target_input_values=torch.equal(cached.target['layer.0.post_attention_norm'],
                                                           fixed.target['layer.0.post_attention_norm']),
                full_exactly_recovers_fixed_output=torch.equal(full, fixed.target[boundary]),
                one_exactly_recovers_cached_output=torch.equal(one, cached.target[boundary]),
                same_input_one_vs_full=differences(one, full),
                input_shapes=dict(one=[local_inputs.shape[0], 1, local_inputs.shape[-1]],
                                  full=list(local_inputs.shape))))
            tensors['layer0_shape_control.' + name + '.one'] = one.contiguous()
            tensors['layer0_shape_control.' + name + '.full'] = full.contiguous()
        record['layer0_mlp_same_input_shape_controls'] = local_controls
        report.require(label + '_layer0_shape_controls_recover_fixed_native_outputs',
                       all(x['full_exactly_recovers_fixed_output'] for x in local_controls))
        # A no-op replacement tests cloning/layout effects of the intervention.
        native_input = fixed.joined_history('embedding')
        state, no_op_z = aligned_layer_forward(model, ids, mask, row, column, valid,
            native_input, 0, dtype, device)
        report.require(label + '_no_op_input_replacement_exact_logits',
            state['replacement_exact'] and state['unaffected_elements_exact']
            and torch.equal(no_op_z, raw['fixed_no_cache']))
        for index in range(layer_count):
            previous = 'embedding' if index == 0 else f'layer.{index-1}.output'
            native_cached_input = cached.joined_history(previous)
            no_op_state, no_op_logits = aligned_layer_forward(model, ids, mask, row, column, valid,
                fixed.joined_history(previous), index, dtype, device)
            state, counterfactual_z = aligned_layer_forward(model, ids, mask, row, column, valid,
                native_cached_input, index, dtype, device)
            fixed_out = fixed.target[f'layer.{index}.output'].double()
            cached_out = cached.target[f'layer.{index}.output'].double()
            aligned_out = state['layer_output'].double()
            propagated = fixed_out - aligned_out
            fresh = aligned_out - cached_out
            total = fixed_out - cached_out
            fixed_history = fixed.joined_history(f'layer.{index}.output')[valid].double()
            cached_history = cached.joined_history(f'layer.{index}.output')[valid].double()
            aligned_history = state['history_output'][valid].double()
            history_propagated = fixed_history - aligned_history
            history_fresh = aligned_history - cached_history
            history_total = fixed_history - cached_history
            n_p, n_f = propagated.norm().item(), fresh.norm().item()
            decomposition_error = (total - propagated - fresh).abs().max().item()
            probability = compare_distributions(raw['cached'], counterfactual_z, token)
            item = dict(layer=index, input_values_aligned_exactly=state['replacement_exact'],
                unaffected_elements_exact=state['unaffected_elements_exact'], input_call_count=state['input_call_count'],
                layout_preserved=state['layout_preserved'], input_layout=state['layout_before'],
                no_op_exact_layer_output=torch.equal(no_op_state['layer_output'], fixed.target[f'layer.{index}.output']),
                no_op_exact_full_vocab_logits=torch.equal(no_op_logits, raw['fixed_no_cache']),
                prior_input_history_difference_l2=(fixed.joined_history(previous)[valid].double()
                                                   - native_cached_input[valid].double()).norm().item(),
                total_output_difference_l2=total.norm().item(),
                propagated_prior_difference_l2=n_p, same_input_path_difference_l2=n_f,
                component_cosine=None if n_p == 0 or n_f == 0 else float(torch.dot(propagated, fresh)/(n_p*n_f)),
                vector_decomposition_max_residual=decomposition_error,
                valid_history_decomposition=dict(
                    total_l2=history_total.norm().item(), propagated_prior_l2=history_propagated.norm().item(),
                    same_input_path_l2=history_fresh.norm().item(),
                    vector_identity_max_residual=(history_total-history_propagated-history_fresh).abs().max().item()),
                remaining_endpoint=probability)
            record['interventions'].append(item)
            tensors[f'intervention.{index}.propagated_prior_difference'] = propagated.contiguous()
            tensors[f'intervention.{index}.same_input_path_difference'] = fresh.contiguous()
            tensors[f'intervention.{index}.aligned_history_output'] = state['history_output'].contiguous()
            if index % 4 == 0 or index == layer_count - 1:
                print(f'{label}: aligned inputs at layer {index}; remaining token probability difference '
                      f"{100*probability['probability_relative_difference']:+.6f}%", flush=True)
        report.require(label + '_all_layer_input_interventions_are_scoped_and_exact',
            all(x['input_values_aligned_exactly'] and x['unaffected_elements_exact']
                and x['input_call_count'] == 1 for x in record['interventions']))
        report.require(label + '_all_layer_no_op_replacements_and_layout_controls_exact',
            all(x['layout_preserved'] and x['no_op_exact_layer_output'] and x['no_op_exact_full_vocab_logits']
                for x in record['interventions']))
        report.require(label + '_all_layer_vector_decompositions_exact',
            max(max(x['vector_decomposition_max_residual'],
                    x['valid_history_decomposition']['vector_identity_max_residual'])
                for x in record['interventions']) < 1e-12)
        # Hold the complete batch's final normalized hidden values identical.
        # Change only contiguous-vs-fixed-slice layout at the vocabulary head.
        values = cached.final_norm_batch.to(device)
        strided = torch.zeros((ids.shape[0], ids.shape[1], values.shape[-1]), dtype=values.dtype, device=device)
        strided[:, column] = values
        contiguous = values[:, None, :].contiguous()
        strided = strided[:, column:column+1, :]
        with torch.no_grad(), forward_context(device, dtype):
            z_contiguous = model.lm_head(contiguous)[row, 0].detach().float().cpu()
            z_strided = model.lm_head(strided)[row, 0].detach().float().cpu()
        record['head_layout_control'] = dict(
            values_identical=torch.equal(contiguous.cpu(), strided.cpu()),
            contiguous_stride=list(contiguous.stride()), strided_stride=list(strided.stride()),
            contiguous_recovers_native_cached_logits=torch.equal(z_contiguous, raw['cached']),
            logits=differences(z_contiguous, z_strided),
            endpoint=compare_distributions(z_contiguous, z_strided, token))
        report.require(label + '_head_control_same_input_and_native_recovery',
            record['head_layout_control']['values_identical']
            and record['head_layout_control']['contiguous_recovers_native_cached_logits'])
    report.require(label + '_no_parameter_update_or_gradient',
        all(p._version == versions[name] and p.grad is None and not p.requires_grad
            for name, p in model.named_parameters()))
    snapshot = report.path.parent / (label + '_propagation.safetensors')
    save_file(tensors, str(snapshot))
    record['snapshot'] = dict(path=str(snapshot), bytes=snapshot.stat().st_size,
        sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest())
    report.data.setdefault('cases', []).append(record)
    report.write()
    print(f'{label}: first target difference {record["first_target_difference"]}; '
          f'baseline token probability difference {100*endpoint["probability_relative_difference"]:+.6f}%', flush=True)
    return record
