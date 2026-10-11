"""Check pure-attention checkpoint numerical/gradient/storage contracts on CPU.

These checks cover the actual cached scorer; DynamicCache.update is outside the
checkpoint. Saved-storage measurements count unique tensor storage retained by
autograd, not total process or GPU peak memory.
"""

import argparse
import copy
from pathlib import Path
import sys
import time

import torch
import transformers
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2 import modeling_qwen2

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIRECTORY))

from cached_attention_checkpoint import checkpoint_eager_attention
from cached_rollout import sample_completions_cached
from cached_score import score_completion_logps_cached
from check_cached_score_cpu import native_model, tiny_adapter
from geora_check_utils import CheckReport
from grpo_rollout import likelihood_alignment_metrics


def snapshot(model):
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def model_unchanged(model, original):
    return all(torch.equal(value, original[name]) for name, value in model.state_dict().items())


def compare_case(kind, dtype, report, *, updated=False):
    native = tiny_adapter(kind, dtype)
    if updated:
        with torch.no_grad():
            for name, p in native.named_parameters():
                if p.requires_grad:
                    # A deterministic nonzero adapter change exercises the
                    # active branch without consuming the sampling RNG.
                    p.add_(torch.linspace(-0.03, 0.03, p.numel()).reshape_as(p))
    checked = copy.deepcopy(native)
    initial_state = snapshot(native)
    label = str(dtype).split(".")[-1] + "_" + kind + ("_updated" if updated else "_initial")
    prompts = [[3, 4, 5]] * 4 + [[6, 7, 8, 9, 10]] * 4
    rollout = sample_completions_cached(native, prompts, 0, [2, 3, 4, 5, 6], 8, "cpu", dtype,
                                        generator=torch.Generator().manual_seed(29))
    rng = torch.get_rng_state().clone()
    native_logps = score_completion_logps_cached(native, rollout, "cpu", dtype, gradient_enabled=True)
    stats = {}
    original_attention = modeling_qwen2.eager_attention_forward
    with checkpoint_eager_attention(checked, stats):
        checked_logps = score_completion_logps_cached(checked, rollout, "cpu", dtype, gradient_enabled=True)
        no_grad_logps = score_completion_logps_cached(checked, rollout, "cpu", dtype, gradient_enabled=False)
    report.require(label + "_all_valid_logps_exact", torch.equal(native_logps.detach(), checked_logps.detach())
                   and torch.equal(no_grad_logps, checked_logps.detach()),
                   **likelihood_alignment_metrics(rollout["behavior_logps"], checked_logps,
                                                   rollout["completion_mask"]))
    report.require(label + "_wrapper_restored_before_backward",
                   modeling_qwen2.eager_attention_forward is original_attention)
    mask = rollout["completion_mask"]
    weights = torch.linspace(0.1, 1.0, mask.numel()).reshape_as(mask)
    (-(native_logps * weights)[mask].mean()).backward()
    (-(checked_logps * weights)[mask].mean()).backward()
    reference = dict(native.named_parameters())
    actual = dict(checked.named_parameters())
    names = [name for name, p in native.named_parameters() if p.requires_grad]
    a = torch.cat([actual[name].grad.flatten().double() for name in names])
    b = torch.cat([reference[name].grad.flatten().double() for name in names])
    max_delta = (a - b).abs().max().item()
    relative = (a - b).norm().item() / max(b.norm().item(), 1e-30)
    report.require(label + "_adapter_gradients_match", torch.isfinite(a).all().item()
                   and b.norm().item() > 0 and max_delta <= 1e-6 and relative <= 2e-5,
                   gradient_max_absolute=max_delta, gradient_relative_l2=relative,
                   absolute_limit=1e-6, relative_limit=2e-5)
    report.require(label + "_backward_recomputed_after_restoration", stats["recompute_calls"] > 0
                   and stats["checkpoint_forward_calls"] > 0 and stats["native_no_grad_calls"] > 0
                   and stats["autocast_states_match"]
                   and modeling_qwen2.eager_attention_forward is original_attention,
                   **stats)
    report.require(label + "_frozen_parameters_buffers_rng_mode_hooks_preserved",
                   model_unchanged(native, initial_state) and model_unchanged(checked, initial_state)
                   and torch.equal(rng, torch.get_rng_state()) and native.training and checked.training
                   and all(p.grad is None for p in checked.parameters() if not p.requires_grad)
                   and all(not m._forward_hooks and not m._forward_pre_hooks for m in checked.modules()))
    report.data["cases"].append(dict(label=label, completion_lengths=rollout["completion_lengths"].tolist(),
                                    valid_tokens=mask.sum().item(), checkpoint_statistics=stats,
                                    gradient_max_absolute=max_delta, gradient_relative_l2=relative))
    report.write()


def fp64_history_case(report):
    native = native_model().double().eval()
    native.requires_grad_(False)
    native.model.embed_tokens.weight.requires_grad_(True)
    checked = copy.deepcopy(native)
    ids = torch.tensor([[3, 4, 5, 10, 11, 12, 13], [6, 7, 8, 9, 14, 15, 16]])
    rollout = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                   completion_mask=torch.ones(2, 3, dtype=torch.bool), prompt_width=4,
                   max_new_tokens=3, generated_steps=3, temperature=1.0, use_cache=True)
    native_scores = score_completion_logps_cached(native, rollout, "cpu", torch.float64, gradient_enabled=True)
    stats = {}
    with checkpoint_eager_attention(checked, stats):
        checked_scores = score_completion_logps_cached(checked, rollout, "cpu", torch.float64, gradient_enabled=True)
    (-native_scores[:, -1].mean()).backward()
    (-checked_scores[:, -1].mean()).backward()
    a = checked.model.embed_tokens.weight.grad
    b = native.model.embed_tokens.weight.grad
    relative = (a - b).norm().item() / b.norm().item()
    maximum = (a - b).abs().max().item()
    report.require("fp64_history_gradient_and_probabilities_match", torch.equal(native_scores, checked_scores)
                   and a[3].norm().item() > 0 and maximum <= 2e-10 and relative <= 2e-9,
                   gradient_max_absolute=maximum, gradient_relative_l2=relative,
                   early_prompt_embedding_gradient_norm=a[3].norm().item(),
                   absolute_limit=2e-10, relative_limit=2e-9,
                   scope="FP64 model; native norm and sampled-token log-softmax include FP32")


def saved_storage_case(report):
    torch.manual_seed(18)
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=37, hidden_size=96, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=12, num_key_value_heads=2, attention_dropout=0.0,
        pad_token_id=0, eos_token_id=2))
    model.config._attn_implementation = "eager"
    model.requires_grad_(False)
    model.model.embed_tokens.weight.requires_grad_(True)
    model.eval()
    ids = torch.randint(3, 37, (8, 16 + 32))
    rollout = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                   completion_mask=torch.ones(8, 32, dtype=torch.bool), prompt_width=16,
                   max_new_tokens=32, generated_steps=32, temperature=1.0, use_cache=True)
    counts = {}
    for checkpointed in (False, True):
        unique = {}
        def save(tensor):
            # Keep the original tensor intact; this observer must not alter
            # dtype, device or layout, or change which values backward sees.
            storage = tensor.untyped_storage()
            unique[storage.data_ptr()] = storage.nbytes()
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor):
            if checkpointed:
                with checkpoint_eager_attention(model):
                    scores = score_completion_logps_cached(model, rollout, "cpu", torch.float32,
                                                           gradient_enabled=True)
            else:
                scores = score_completion_logps_cached(model, rollout, "cpu", torch.float32,
                                                       gradient_enabled=True)
            counts["checkpoint" if checkpointed else "native"] = sum(unique.values())
            # Release this forward's graph before constructing the next one.
            (-scores[:, -1].mean()).backward()
        model.zero_grad(set_to_none=True)
        del scores
    report.require("saved_autograd_tensor_storage_reduced", counts["checkpoint"] < counts["native"],
                   native_saved_unique_storage_bytes=counts["native"],
                   checkpoint_saved_unique_storage_bytes=counts["checkpoint"],
                   saved_storage_ratio=counts["checkpoint"] / counts["native"],
                   scope="Unique autograd-saved tensor storage, not total CPU/GPU allocation")


def failure_cases(report):
    model = tiny_adapter("geora", torch.float32)
    original = modeling_qwen2.eager_attention_forward
    before = snapshot(model)
    rng = torch.get_rng_state().clone()
    raised = False
    try:
        with checkpoint_eager_attention(model):
            raise RuntimeError("checkpoint sentinel")
    except RuntimeError as error:
        raised = str(error) == "checkpoint sentinel"
    report.require("exception_restores_wrapper_state_rng_and_hooks", raised
                   and modeling_qwen2.eager_attention_forward is original
                   and torch.equal(rng, torch.get_rng_state()) and model_unchanged(model, before)
                   and all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))
    model.config.output_attentions = True
    rejected = False
    try:
        with checkpoint_eager_attention(model):
            pass
    except ValueError as error:
        rejected = "Output-attention" in str(error)
    report.require("output_attentions_rejected_before_patch", rejected
                   and modeling_qwen2.eager_attention_forward is original)
    model.config.output_attentions = False
    model.model.layers[0].self_attn.attention_dropout = 0.1
    rejected = False
    try:
        with checkpoint_eager_attention(model):
            pass
    except ValueError as error:
        rejected = "dropout" in str(error)
    report.require("dropout_rejected_before_patch", rejected
                   and modeling_qwen2.eager_attention_forward is original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = CheckReport(args.output_dir, "cached_attention_checkpoint_cpu.json")
    report.data.update(stage="pure_attention_checkpoint_cpu_preflight", cases=[],
                       torch_version=torch.__version__, transformers_version=transformers.__version__,
                       production_scorer_changed=False,
                       scope="Tiny CPU only; no GPU allocation or real task update")
    started = time.perf_counter()
    try:
        for dtype in (torch.float32, torch.bfloat16):
            for kind in ("lora", "geora"):
                compare_case(kind, dtype, report)
                compare_case(kind, dtype, report, updated=True)
        fp64_history_case(report)
        saved_storage_case(report)
        failure_cases(report)
        report.data["elapsed_seconds"] = time.perf_counter() - started
        report.finish()
    except Exception as error:
        report.data.update(status="failed", error=str(error), elapsed_seconds=time.perf_counter() - started)
        report.write()
        raise


if __name__ == "__main__":
    main()
