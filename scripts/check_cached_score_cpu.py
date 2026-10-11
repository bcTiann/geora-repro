"""CPU checks for differentiable cached teacher forcing, not task training.

Check actual sampling probabilities and actual forward metadata for every valid
position, before/after synthetic LoRA/GeoRA updates. A separate FP64 native Qwen
checks a final-token gradient against a full-prefix no-cache forward, so a
wrongly detached history cannot pass merely by preserving forward values.
"""

import argparse
import copy
from pathlib import Path
import sys
import time

import torch
import transformers
from transformers import Qwen2Config, Qwen2ForCausalLM

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIRECTORY))

from cached_rollout import sample_completions_cached
from cached_score import score_completion_logps_cached
from geora_layers import GeoRALinear, iter_target_linears
from lora_layers import LoRALinear
from geora_check_utils import CheckReport, move_with_precision
from grpo_rollout import likelihood_alignment_metrics


LOGP_LIMIT = 2e-5
RATIO_LIMIT = 2e-5
SEQUENCE_LIMIT = 1e-3


def native_model():
    torch.manual_seed(18)
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=37, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        attention_dropout=0.0, eos_token_id=2, pad_token_id=0))
    model.config._attn_implementation = "eager"
    return model


def tiny_adapter(kind, dtype):
    model = native_model()
    model.requires_grad_(False)
    for name, layer in list(iter_target_linears(model)):
        A = torch.randn(2, layer.in_features) * 0.02
        B = torch.randn(layer.out_features, 2) * 0.02
        if kind == "lora":
            replacement = LoRALinear(layer, A, torch.zeros_like(B), 2.0)
        else:
            replacement = GeoRALinear(layer, A, B, 2.0, forward_mode="difference")
        parent, leaf = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), leaf, replacement)
    move_with_precision(model, "cpu", dtype)
    return model.train()


def actual_input_observer(records):
    def hook(module, args, kwargs):
        cache = kwargs.get("past_key_values")
        mask = kwargs.get("attention_mask")
        records.append(dict(
            step=len(records), input_shape=list(kwargs["input_ids"].shape),
            input_ids=kwargs["input_ids"].detach().cpu().tolist(),
            position_ids=kwargs["position_ids"].detach().cpu().tolist(),
            attention_mask=None if mask is None else mask.detach().cpu().tolist(),
            logits_to_keep=kwargs["logits_to_keep"],
            cache_length_before=0 if cache is None else cache.get_seq_length()))
    return hook


def run_case(model, prompts, dtype, report, label, *, capacity=8, pad=0,
             eos=(2, 3, 4, 5, 6), backward=False):
    actual_inputs = []
    hook = model.register_forward_pre_hook(actual_input_observer(actual_inputs), with_kwargs=True)
    try:
        rollout = sample_completions_cached(
            model, prompts, pad, list(eos), capacity, "cpu", dtype,
            generator=torch.Generator().manual_seed(29))
    finally:
        hook.remove()
    original_mode = model.training
    rng = torch.get_rng_state().clone()
    scored_inputs = []
    current = score_completion_logps_cached(
        model, rollout, "cpu", dtype, gradient_enabled=True,
        trace_callback=scored_inputs.append)
    old = score_completion_logps_cached(
        model, rollout, "cpu", dtype, gradient_enabled=False)
    metrics = likelihood_alignment_metrics(
        rollout["behavior_logps"], current, rollout["completion_mask"])
    metrics["gate_passed"] = (metrics["finite"]
                              and metrics["max_absolute_delta_logp"] <= LOGP_LIMIT
                              and metrics["max_absolute_ratio_minus_one"] <= RATIO_LIMIT
                              and max(map(abs, metrics["sequence_delta_logp"])) <= SEQUENCE_LIMIT)
    report.require(label + "_all_valid_token_likelihood_gate", metrics["gate_passed"], **metrics)
    report.require(label + "_actual_generation_inputs_reproduced", actual_inputs == scored_inputs,
                   steps=len(scored_inputs), prefill_width=rollout["prompt_width"])
    report.require(label + "_autograd_and_no_grad_scores_equal", torch.equal(current.detach(), old)
                   and current.requires_grad and not old.requires_grad
                   and (current[~rollout["completion_mask"]] == 0).all().item())
    report.require(label + "_mode_rng_and_hooks_preserved", model.training == original_mode
                   and torch.equal(rng, torch.get_rng_state())
                   and not model._forward_pre_hooks and not model._forward_hooks)
    record = dict(label=label, dtype=str(dtype), likelihood=metrics,
                  completion_ids=rollout["completion_ids"].tolist(),
                  completion_lengths=rollout["completion_lengths"].tolist(),
                  behavior_logps=rollout["behavior_logps"].tolist(),
                  recomputed_logps=current.detach().tolist(),
                  forward_inputs=scored_inputs)
    report.data["cases"].append(record)
    report.write()
    if backward:
        snapshot = {name: p.detach().clone() for name, p in model.named_parameters()}
        buffers = {name: p.detach().clone() for name, p in model.named_buffers()}
        trainable = [p for p in model.parameters() if p.requires_grad]
        mask = rollout["completion_mask"]
        loss = -current[mask].mean()
        loss.backward()
        report.require(label + "_cached_history_backward_finite", all(
            p.grad is not None and p.grad.dtype == torch.float32
            and torch.isfinite(p.grad).all().item() for p in trainable)
            and any(p.grad.abs().sum().item() > 0 for p in trainable), loss=loss.item())
        optimizer = torch.optim.AdamW(trainable, lr=0.01, weight_decay=0.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        report.require(label + "_only_adapter_parameters_updated", all(
            torch.equal(p, snapshot[name]) for name, p in model.named_parameters()
            if not p.requires_grad) and all(torch.equal(p, buffers[name])
                                          for name, p in model.named_buffers())
            and any(not torch.equal(p, snapshot[name]) for name, p in model.named_parameters()
                    if p.requires_grad))
    return rollout


def gradient_reference(report):
    model = native_model().double().eval()
    model.requires_grad_(False)
    model.model.embed_tokens.weight.requires_grad_(True)
    independent = copy.deepcopy(model)
    ids = torch.tensor([[3, 4, 5, 10, 11, 12, 13], [6, 7, 8, 9, 14, 15, 16]])
    prompt_mask = torch.ones(2, 4, dtype=torch.long)
    rollout = dict(input_ids=ids, attention_mask=torch.cat([prompt_mask, torch.ones(2, 3, dtype=torch.long)], 1),
                   completion_mask=torch.ones(2, 3, dtype=torch.bool), prompt_width=4,
                   max_new_tokens=3, generated_steps=3, temperature=1.0, use_cache=True)
    sequential = score_completion_logps_cached(
        model, rollout, "cpu", torch.float64, gradient_enabled=True)
    (-sequential[:, -1].mean()).backward()
    # Independent full-prefix no-cache computation: one forward, no scorer
    # helper, no DynamicCache, and manual causal-position selection.
    mask = rollout["attention_mask"][:, :-1]
    positions = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 0)
    outputs = independent(input_ids=ids[:, :-1], attention_mask=None,
                          position_ids=positions, use_cache=False, logits_to_keep=1)
    manual = outputs.logits[:, -1].float().log_softmax(-1).gather(1, ids[:, -1:]).squeeze(1)
    (-manual.mean()).backward()
    actual = model.model.embed_tokens.weight.grad
    reference = independent.model.embed_tokens.weight.grad
    delta = actual - reference
    relative = delta.norm().item() / reference.norm().item()
    # Token 3 occurs only in the early prompt. The loss on the third answer
    # token can reach it only through attention to previous/history states.
    history_norm = actual[3].norm().item()
    report.require("fp64_native_history_gradient_matches_full_prefix_reference",
                   torch.isfinite(actual).all().item() and history_norm > 0
                   and relative < 2e-6 and delta.abs().max().item() < 2e-6,
                   gradient_relative_l2=relative, gradient_max_absolute=delta.abs().max().item(),
                   early_prompt_embedding_gradient_norm=history_norm,
                   max_logp_difference=(sequential[:, -1] - manual).abs().max().item(),
                   scope="FP64 model arithmetic; native RMSNorm and actual sampling log-softmax use FP32")
    # Value comparison alone cannot prove this: detaching the history would
    # remove the entire early-token gradient while preserving its forward V.
    report.require("gradient_reference_is_execution_path_comparison_not_fp64_policy",
                   sequential.dtype == torch.float32 and actual.dtype == torch.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = CheckReport(args.output_dir, "cached_score_cpu.json")
    report.data.update(stage="differentiable_cached_score_cpu_preflight", cases=[],
                       torch_version=torch.__version__, transformers_version=transformers.__version__,
                       scope="Tiny synthetic models; no GSM8K training or GPU allocation",
                       production_sampler_changed=False,
                       gates=dict(logp=LOGP_LIMIT, ratio=RATIO_LIMIT, sequence_logp=SEQUENCE_LIMIT))
    started = time.perf_counter()
    try:
        prompts = [[3, 4, 5]] * 4 + [[6, 7, 8, 9, 10]] * 4
        for dtype in (torch.float32, torch.bfloat16):
            for kind in ("lora", "geora"):
                model = tiny_adapter(kind, dtype)
                label = str(dtype).split(".")[-1] + "_" + kind
                initial = run_case(model, prompts, dtype, report, label + "_initial", backward=True)
                report.require(label + "_mixed_eos_lengths_and_truncation", initial["ended_with_eos"].any().item()
                               and (~initial["ended_with_eos"]).any().item()
                               and initial["completion_lengths"].unique().numel() > 1)
                run_case(model, prompts, dtype, report, label + "_updated_longer", capacity=24, eos=(36,))
                immediate = run_case(model, prompts, dtype, report, label + "_pad_equals_eos",
                                     pad=2, eos=tuple(range(37)))
                report.require(label + "_one_token_all_eos", immediate["generated_steps"] == 1
                               and (immediate["completion_lengths"] == 1).all().item())
                run_case(model, [[3, 4, 5]] * 8, dtype, report, label + "_no_left_padding")
                # Failures cannot leave hooks installed because the scorer
                # installs none; observer exceptions leave model state intact.
                def fail_observer(metadata):
                    raise RuntimeError("observer sentinel")
                failed = False
                try:
                    score_completion_logps_cached(model, initial, "cpu", dtype,
                                                   gradient_enabled=False, trace_callback=fail_observer)
                except RuntimeError as error:
                    failed = str(error) == "observer sentinel"
                report.require(label + "_exception_has_no_hooks_or_mutation", failed
                               and not model._forward_pre_hooks and not model._forward_hooks)
                model.model.layers[0].self_attn.attention_dropout = 0.1
                rejected = False
                try:
                    score_completion_logps_cached(model, initial, "cpu", dtype, gradient_enabled=False)
                except ValueError as error:
                    rejected = "dropout" in str(error)
                report.require(label + "_nonzero_dropout_rejected", rejected)
        gradient_reference(report)
        report.data["elapsed_seconds"] = time.perf_counter() - started
        report.finish()
    except Exception as error:
        report.data.update(status="failed", error=str(error), elapsed_seconds=time.perf_counter() - started)
        report.write()
        raise


if __name__ == "__main__":
    main()
