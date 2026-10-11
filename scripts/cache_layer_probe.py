"""Locate cached/no-cache differences using saved tokens and one native Qwen.

Trace the target token and its complete causal history. Local Linear controls
reuse identical inputs/weights while changing only sequence shape. This is a
forward-only diagnostic; it does not change the production GRPO sampler.
"""

from contextlib import contextmanager

import torch
from torch.nn import functional as F
from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList
from transformers.models.qwen2 import modeling_qwen2

from geora_check_utils import forward_context
from grpo_rollout import _decoder_hidden, _one_position_logits, position_ids_from_mask


def differences(reference, actual):
    """Compare equal-shaped saved values in FP64, without an angle or clamp."""
    reference = reference.double()
    actual = actual.double()
    delta = actual - reference
    denominator = reference.norm().item()
    return dict(max_absolute=delta.abs().max().item(),
                relative_l2=delta.norm().item() / max(denominator, 1e-30),
                unequal_elements=(delta != 0).sum().item(),
                element_count=delta.numel(), exact=torch.equal(reference, actual))


def saved_batch(prompts, completion_ids, completion_lengths, pad_id, device):
    """Rebuild the old fixed-width batch; exclude slots after the first EOS."""
    width = max(map(len, prompts))
    completions = torch.tensor(completion_ids, dtype=torch.long, device=device)
    ids = torch.full((len(prompts), width + completions.shape[1]), pad_id,
                     dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for row, tokens in enumerate(prompts):
        start = width - len(tokens)
        ids[row, start:width] = torch.tensor(tokens, device=device)
        mask[row, start:width] = 1
        length = completion_lengths[row]
        ids[row, width:width + length] = completions[row, :length]
        mask[row, width:width + length] = 1
    return ids, mask, width


class SavedTokens(LogitsProcessor):
    """Force the already recorded token at each step; no new answer is sampled."""

    def __init__(self, prompt_width, completions):
        self.prompt_width = prompt_width
        self.completions = completions

    def __call__(self, input_ids, scores):
        step = input_ids.shape[1] - self.prompt_width
        forced = torch.full_like(scores, -torch.inf)
        forced.scatter_(1, self.completions[:, step:step + 1], 0.0)
        return forced


class LayerTrace:
    """Save one row's target vectors and per-layer complete prefix outputs."""

    def __init__(self, model, row, target_column, prefix_width, path):
        self.model = model
        self.row = row
        self.target_column = target_column
        self.prefix_width = prefix_width
        self.path = path
        self.target = {}
        self.history = {}
        self.inputs = []
        self.attention = {}
        self.normalized_full_batch = None
        self.handles = []
        self.order = []

    def capture(self, name, tensor, history=False):
        # Cached calls are prefill followed by single-token decoding. For the
        # no-cache calls, the target stays at its original padded column.
        selected_column = -1 if self.path == 'cached' else self.target_column
        self.target[name] = tensor[self.row, selected_column].detach().float().cpu().clone()
        if history:
            selected = tensor[self.row].detach().float().cpu().clone()
            self.history.setdefault(name, []).append(selected)

    def output_hook(self, name, history=False):
        def hook(module, args, output):
            self.capture(name, output, history)
            if name == 'layer.0.input_norm' and self.path != 'cached':
                self.normalized_full_batch = output.detach().clone()
        return hook

    def input_hook(self, name):
        def hook(module, args):
            self.capture(name, args[0])
        return hook

    def model_inputs(self, module, args, kwargs):
        ids = kwargs['input_ids']
        positions = kwargs.get('position_ids')
        mask = kwargs.get('attention_mask')
        past = kwargs.get('past_key_values')
        self.inputs.append(dict(input_ids=ids[self.row].detach().cpu().tolist(),
                                position_ids=positions[self.row].detach().cpu().tolist(),
                                attention_mask=mask[self.row].detach().cpu().tolist(),
                                cached_length_before=0 if past is None else past.get_seq_length(),
                                input_shape=list(ids.shape)))

    def attention_call(self, module, query, key, value, mask, original, **kwargs):
        output, probabilities = original(module, query, key, value, mask, **kwargs)
        # Capture actual post-RoPE Q and the cached/complete history K/V.
        if module.layer_idx == 0:
            column = -1 if self.path == 'cached' else self.target_column
            target_mask = mask[self.row, 0, column, :self.prefix_width]
            self.attention = dict(
                query=query[self.row, :, column:column + 1 if column >= 0 else None, :]
                    .detach().float().cpu().clone(),
                key=key[self.row, :, :self.prefix_width, :].detach().float().cpu().clone(),
                value=value[self.row, :, :self.prefix_width, :].detach().float().cpu().clone(),
                allowed_keys=(target_mask == 0).detach().cpu().clone(),
                probabilities=probabilities[self.row, :, column, :self.prefix_width]
                    .detach().float().cpu().clone(),
                query_shape=list(query.shape), key_shape=list(key.shape), mask_shape=list(mask.shape),
            )
        return output, probabilities

    @contextmanager
    def installed(self):
        original = modeling_qwen2.eager_attention_forward
        def attention(module, query, key, value, attention_mask, **kwargs):
            return self.attention_call(module, query, key, value, attention_mask, original, **kwargs)
        modeling_qwen2.eager_attention_forward = attention
        self.handles.append(self.model.model.register_forward_pre_hook(self.model_inputs, with_kwargs=True))
        specifications = [('embedding', self.model.model.embed_tokens, True)]
        for index, block in enumerate(self.model.model.layers):
            prefix = 'layer.' + str(index)
            specifications.extend([
                (prefix + '.input_norm', block.input_layernorm, index == 0),
                (prefix + '.q_proj', block.self_attn.q_proj, index == 0),
                (prefix + '.k_proj', block.self_attn.k_proj, index == 0),
                (prefix + '.v_proj', block.self_attn.v_proj, index == 0),
                (prefix + '.o_proj', block.self_attn.o_proj, False),
                (prefix + '.post_attention_norm', block.post_attention_layernorm, False),
                (prefix + '.gate_proj', block.mlp.gate_proj, False),
                (prefix + '.up_proj', block.mlp.up_proj, False),
                (prefix + '.down_proj', block.mlp.down_proj, False),
                (prefix + '.output', block, True),
            ])
            self.handles.append(block.self_attn.o_proj.register_forward_pre_hook(
                self.input_hook(prefix + '.attention_output')))
        specifications.append(('final_norm', self.model.model.norm, True))
        # Put the attention output boundary immediately before o_proj.
        for name, module, history in specifications:
            if name.endswith('.o_proj'):
                self.order.append(name.replace('.o_proj', '.attention_output'))
            self.order.append(name)
            self.handles.append(module.register_forward_hook(self.output_hook(name, history)))
        try:
            yield self
        finally:
            for handle in self.handles:
                handle.remove()
            modeling_qwen2.eager_attention_forward = original

    def joined_history(self, name):
        return torch.cat(self.history[name], dim=0)[:self.prefix_width]


def trace_path(model, ids, mask, prompt_width, step, row, pad_id, eos_ids, dtype, device, path):
    prefix_width = prompt_width + step
    column = prefix_width - 1
    trace = LayerTrace(model, row, column, prefix_width, path)
    with trace.installed(), torch.no_grad(), forward_context(device, dtype):
        if path == 'cached':
            config = GenerationConfig(
                do_sample=False, use_cache=True, cache_implementation='dynamic',
                max_new_tokens=step + 1, num_beams=1,
                pad_token_id=pad_id, eos_token_id=eos_ids, bos_token_id=model.config.bos_token_id,
                return_dict_in_generate=True, output_scores=False, disable_compile=True,
            )
            forced = SavedTokens(prompt_width, ids[:, prompt_width:])
            raw = []
            handle = model.lm_head.register_forward_hook(
                lambda module, args, output: raw.append(output[:, -1].detach().float().cpu().clone()))
            try:
                generated = model.generate(
                    input_ids=ids[:, :prompt_width], attention_mask=mask[:, :prompt_width],
                    generation_config=config, logits_processor=LogitsProcessorList([forced]),
                )
            finally:
                handle.remove()
            trace.generated_ids = generated.sequences[:, prompt_width:].detach().cpu()
            logits = raw[-1][row]
            trace.raw_head_shapes = len(raw)
        else:
            width = prefix_width if path == 'prefix_no_cache' else ids.shape[1]
            hidden = _decoder_hidden(model, ids[:, :width], mask[:, :width])
            logits = _one_position_logits(model, hidden, column, 1.0)[row].detach().cpu()
    trace.target['lm_head'] = logits.clone()
    trace.order.append('lm_head')
    return trace, logits


def trace_comparison(cached, actual, valid_prefix):
    rows = []
    for name in cached.order:
        metric = differences(cached.target[name], actual.target[name])
        rows.append(dict(boundary=name, **metric))
    histories = []
    for name in cached.history:
        a = cached.joined_history(name)[valid_prefix]
        b = actual.joined_history(name)[valid_prefix]
        histories.append(dict(boundary=name, **differences(a, b)))
    attention = {}
    for name in ('query', 'key', 'value', 'probabilities'):
        a, b = cached.attention[name], actual.attention[name]
        if name in ('key', 'value'):
            a, b = a[:, valid_prefix], b[:, valid_prefix]
        elif name == 'probabilities':
            a, b = a[:, valid_prefix], b[:, valid_prefix]
        attention[name] = differences(a, b)
    return dict(target_boundaries=rows, causal_history_boundaries=histories,
                first_target_difference=next((r['boundary'] for r in rows if not r['exact']), None),
                first_history_difference=next((r['boundary'] for r in histories if not r['exact']), None),
                layer0_attention=attention)


def common_input_linear_controls(model, trace, row, target_column, prefix_width, dtype, device):
    """Use the same normalized values in [B,1,H], [B,prefix,H], [B,full,H]."""
    inputs = trace.normalized_full_batch
    result = []
    with torch.no_grad():
        for name in ('q_proj', 'k_proj', 'v_proj'):
            layer = getattr(model.model.layers[0].self_attn, name)
            outputs = {}
            with forward_context(device, dtype):
                for shape, tensor, index in (
                    ('one', inputs[:, target_column:target_column + 1].contiguous(), 0),
                    ('prefix', inputs[:, :prefix_width].contiguous(), target_column),
                    ('full', inputs, target_column),
                ):
                    outputs[shape] = layer(tensor)[row, index].float().cpu()
            # Same numeric input/weight values, computation alone promoted.
            with torch.autocast(device_type=torch.device(device).type, enabled=False):
                weight = layer.weight.float()
                bias = None if layer.bias is None else layer.bias.float()
                fp32 = {}
                for shape, tensor, index in (
                    ('one', inputs[:, target_column:target_column + 1].contiguous(), 0),
                    ('full', inputs, target_column),
                ):
                    fp32[shape] = F.linear(tensor.float(), weight, bias)[row, index].cpu()
            result.append(dict(module='layer.0.' + name,
                               identical_selected_input=True, native_forward_dtype=str(dtype),
                               full_control_reproduces_trace_exact=torch.equal(
                                   outputs['full'], trace.target['layer.0.' + name]),
                               input_shapes=dict(one=[*inputs.shape[:1], 1, inputs.shape[-1]],
                                                 prefix=[inputs.shape[0], prefix_width, inputs.shape[-1]],
                                                 full=list(inputs.shape)),
                               forward_one_vs_full=differences(outputs['one'], outputs['full']),
                               forward_prefix_vs_full=differences(outputs['prefix'], outputs['full']),
                               fp32_one_vs_full=differences(fp32['one'], fp32['full']),
                               one_vs_fp32=differences(fp32['one'], outputs['one']),
                               full_vs_fp32=differences(fp32['full'], outputs['full'])))
    return result


def inspect_case(model, prompts, case, row, step, pad_id, eos_ids, dtype, device, report, label,
                 expected_snapshot=None):
    ids, mask, prompt_width = saved_batch(
        prompts, case['completion_ids'], case['completion_lengths'], pad_id, device)
    prefix_width = prompt_width + step
    column = prefix_width - 1
    valid = mask[row, :prefix_width].bool().cpu()
    report.require(label + '_one_frozen_native_model', not model.training
                   and all(not p.requires_grad for p in model.parameters())
                   and all(type(m).__name__ not in ('GeoRALinear', 'LoRALinear') for m in model.modules()))
    traces, logits = {}, {}
    for path in ('cached', 'prefix_no_cache', 'fixed_no_cache'):
        trace, output = trace_path(model, ids, mask, prompt_width, step, row,
                                   pad_id, eos_ids, dtype, device, path)
        traces[path], logits[path] = trace, output
        report.require(label + '_' + path + '_finite', torch.isfinite(output).all().item())
    cached = traces['cached']
    report.require(label + '_forced_replay_ids', torch.equal(
        cached.generated_ids, ids[:, prompt_width:prompt_width + step + 1].cpu()))
    # Verify semantic indexing from actual hook inputs, not inferred formulas.
    joined_ids = sum((entry['input_ids'] for entry in cached.inputs), [])
    joined_positions = sum((entry['position_ids'] for entry in cached.inputs), [])
    positions = position_ids_from_mask(mask)[row, :prefix_width].cpu().tolist()
    report.require(label + '_actual_valid_input_and_positions_match',
                   joined_ids == ids[row, :prefix_width].cpu().tolist()
                   and joined_positions == positions,
                   cached_inputs=cached.inputs,
                   no_cache_inputs=traces['fixed_no_cache'].inputs)
    allowed = cached.attention['allowed_keys']
    report.require(label + '_actual_target_attention_allowed_keys_match',
                   torch.equal(allowed, valid)
                   and all(torch.equal(trace.attention['allowed_keys'], allowed) for trace in traces.values()),
                   allowed_key_count=allowed.sum().item(), prefix_width=prefix_width,
                   shapes={p:{k:t.attention[k] for k in ('query_shape','key_shape','mask_shape')}
                           for p,t in traces.items()})
    if expected_snapshot is not None:
        report.require(label + '_replay_exactly_recovers_previous_cached_logits',
                       torch.equal(logits['cached'], expected_snapshot['cached_logits']),
                       **differences(expected_snapshot['cached_logits'], logits['cached']))
        report.require(label + '_full_path_exactly_recovers_previous_no_cache_logits',
                       torch.equal(logits['fixed_no_cache'], expected_snapshot['no_cache_logits']),
                       **differences(expected_snapshot['no_cache_logits'], logits['fixed_no_cache']))
    token = case['completion_ids'][row][step]
    logps = {p:z.double().log_softmax(-1)[token].item() for p,z in logits.items()}
    record = dict(label=label, dtype=str(dtype), row=row, completion_step=step,
                  sampled_token_id=token, prefix_width=prefix_width, fixed_width=ids.shape[1],
                  logits={p:z[token].item() for p,z in logits.items()}, logps=logps,
                  cached_to_fixed_probability_ratio=float(torch.exp(torch.tensor(
                      logps['fixed_no_cache'] - logps['cached'], dtype=torch.float64))),
                  comparisons={p:trace_comparison(cached,t,valid)
                               for p,t in traces.items() if p!='cached'},
                  common_input_linear_controls=common_input_linear_controls(
                      model, traces['fixed_no_cache'], row, column, prefix_width, dtype, device))
    report.require(label + '_common_input_full_controls_recover_trace', all(
        c['full_control_reproduces_trace_exact'] for c in record['common_input_linear_controls']))
    report.data.setdefault('cases', []).append(record)
    report.write()
    for path, comparison in record['comparisons'].items():
        print(label, path, 'first target difference:', comparison['first_target_difference'],
              'first causal-history difference:', comparison['first_history_difference'], flush=True)
    return record
