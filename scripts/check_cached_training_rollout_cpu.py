"""Check streamed native probabilities against retained native generation scores.

Tiny CPU tests validate probability semantics/RNG/EOS contracts. They do not
measure full-model GPU memory or prove that 256-token differentiable cache
histories fit the training allocation.
"""

import argparse
from pathlib import Path
import time

import torch
import transformers
from transformers import GenerationConfig

from cached_rollout import isolated_sampling_rng, sample_completions_cached
from cached_score import score_completion_logps_cached
from cached_training_rollout import sample_completions_cached_training
from check_cached_score_cpu import tiny_adapter
from geora_check_utils import CheckReport, forward_context
from grpo_rollout import likelihood_alignment_metrics


def sampling_case(model, prompts, dtype, report, label, *, temperature, pad=0,
                  eos=(2, 3, 4, 5, 6), capacity=16):
    generation_processor_lists = []
    original = model._get_logits_processor
    def observe_processors(*args, **kwargs):
        processors = original(*args, **kwargs)
        generation_processor_lists.append([type(p).__name__ for p in processors])
        return processors
    model._get_logits_processor = observe_processors
    generator = torch.Generator().manual_seed(29)
    global_rng = torch.get_rng_state().clone()
    initial_private = generator.get_state().clone()
    try:
        actual = sample_completions_cached_training(
            model, prompts, pad, list(eos), capacity, "cpu", dtype,
            temperature=temperature, generator=generator)
    finally:
        del model._get_logits_processor
    advanced = generator.get_state().clone()
    report.require(label + "_recorder_is_final_and_only_processor",
                   generation_processor_lists == [["StreamingTokenLogps"]],
                   generation_processor_lists=generation_processor_lists)
    report.require(label + "_private_rng_isolated_advanced",
                   torch.equal(global_rng, torch.get_rng_state())
                   and not torch.equal(initial_private, advanced))
    generator.set_state(initial_private)
    expected = sample_completions_cached(
        model, prompts, pad, list(eos), capacity, "cpu", dtype,
        temperature=temperature, generator=generator)
    report.require(label + "_native_retained_scores_and_streamed_scores_exact",
                   all(torch.equal(actual[key], expected[key]) for key in (
                       "input_ids", "attention_mask", "completion_ids", "completion_mask",
                       "behavior_logps", "completion_lengths", "ended_with_eos"))
                   and torch.equal(advanced, generator.get_state()),
                   valid_tokens=actual["completion_mask"].sum().item(),
                   completion_lengths=actual["completion_lengths"].tolist())
    scored = score_completion_logps_cached(model, actual, "cpu", dtype, gradient_enabled=True)
    metrics = likelihood_alignment_metrics(actual["behavior_logps"], scored, actual["completion_mask"])
    report.require(label + "_differentiable_cached_score_direct_gate",
                   metrics["finite"] and metrics["max_absolute_delta_logp"] <= 2e-5
                   and metrics["max_absolute_ratio_minus_one"] <= 2e-5
                   and max(map(abs, metrics["sequence_delta_logp"])) <= 1e-3
                   and scored.requires_grad, **metrics)
    # Derive EOS semantics from returned IDs independently of the recorder.
    lengths = []
    for row in actual["completion_ids"].tolist():
        eos_index = next((i for i, token in enumerate(row[:actual["generated_steps"]]) if token in eos), None)
        lengths.append(actual["generated_steps"] if eos_index is None else eos_index + 1)
    expected_mask = torch.arange(capacity)[None, :] < torch.tensor(lengths)[:, None]
    report.require(label + "_first_eos_masks_and_zero_padding", torch.equal(
        actual["completion_mask"], expected_mask)
        and (actual["behavior_logps"][~expected_mask] == 0).all().item()
        and (actual["input_ids"][:, actual["prompt_width"]:][~expected_mask] == pad).all().item())
    report.require(label + "_no_vocabulary_score_sequence_retained",
                   not actual["native_vocabulary_scores_retained"]
                   and actual["max_pending_vocabulary_elements"] == len(prompts) * model.config.vocab_size
                   and not actual["behavior_logps"].requires_grad
                   and not actual["input_ids"].is_inference(),
                   pending_vocabulary_elements=actual["max_pending_vocabulary_elements"])
    report.data["cases"].append(dict(label=label, dtype=str(dtype), temperature=temperature,
                                    generated_steps=actual["generated_steps"],
                                    completion_lengths=lengths, likelihood=metrics))
    report.write()
    return actual


def long_sampling_case(report):
    model = tiny_adapter("geora", torch.float32)
    # Keep a valid vocabulary EOS token but make its probability negligible,
    # to exercise exactly 256 steps rather than stopping early by chance.
    model.lm_head.bias = torch.nn.Parameter(torch.zeros(model.config.vocab_size), requires_grad=False)
    model.lm_head.bias[36] = -80
    prompts = [[3, 4, 5], [6, 7, 8, 9, 10]]
    initial_rng = torch.Generator().manual_seed(51).get_state()
    generator = torch.Generator()
    generator.set_state(initial_rng)
    progress = []
    rollout = sample_completions_cached_training(
        model, prompts, 0, [36], 256, "cpu", torch.float32,
        temperature=0.7, generator=generator, progress_callback=progress.append)
    advanced = generator.get_state().clone()
    # Independent 256-step native retained-score oracle is small here (vocab37).
    # It does not use the capped <=64 old sampler or the streaming recorder.
    config = GenerationConfig(
        do_sample=True, use_cache=True, cache_implementation="dynamic",
        max_new_tokens=256, temperature=0.7, top_k=0, top_p=1.0,
        typical_p=1.0, repetition_penalty=1.0, num_beams=1, num_return_sequences=1,
        pad_token_id=0, eos_token_id=[36], bos_token_id=model.config.bos_token_id,
        return_dict_in_generate=True, output_scores=True, disable_compile=True)
    generator.set_state(initial_rng)
    with isolated_sampling_rng(generator, "cpu"), torch.no_grad(), forward_context("cpu", torch.float32):
        native = model.generate(input_ids=rollout["input_ids"][:, :5],
                                attention_mask=rollout["attention_mask"][:, :5],
                                generation_config=config)
    native_ids = native.sequences[:, 5:]
    native_logps = torch.stack([scores.float().log_softmax(-1).gather(1, native_ids[:, i:i+1]).squeeze(1)
                               for i, scores in enumerate(native.scores)], 1)
    report.require("long_256_native_actual_probabilities_and_rng_exact",
                   rollout["generated_steps"] == 256 and len(native.scores) == 256
                   and rollout["completion_mask"].all().item()
                   and torch.equal(native_ids, rollout["completion_ids"])
                   and torch.equal(native_logps, rollout["behavior_logps"])
                   and torch.equal(advanced, generator.get_state()),
                   valid_tokens=rollout["completion_mask"].sum().item(),
                   pending_vocabulary_elements=rollout["max_pending_vocabulary_elements"],
                   oracle_scope="Tiny vocab37 retains all scores only to verify streamed semantics")
    report.require("long_256_progress_callback_reports_bounded_steps",
                   [event["generated_steps"] for event in progress] == list(range(32, 257, 32))
                   and all(event["active_rows"] == 2 for event in progress))
    report.require("long_256_sampling_returns_no_graph_or_cache", not rollout["behavior_logps"].requires_grad
                   and "past_key_values" not in rollout and not rollout["native_vocabulary_scores_retained"])
    try:
        scores = score_completion_logps_cached(model, rollout, "cpu", torch.float32, gradient_enabled=True)
    except ValueError as error:
        # The separately owned candidate scorer initially caps at64. Record this
        # unfinished 256-history backward check explicitly, without bypassing it.
        report.data["long_cached_scoring"] = dict(status="not_run", reason=str(error),
                                                  required_change="Raise cached_score capacity guard from64 to256")
    else:
        metrics = likelihood_alignment_metrics(rollout["behavior_logps"], scores, rollout["completion_mask"])
        loss = -scores[rollout["completion_mask"]].mean()
        loss.backward()
        parameters = [p for p in model.parameters() if p.requires_grad]
        report.require("long_256_cached_score_and_backward", metrics["finite"]
                       and metrics["max_absolute_delta_logp"] <= 2e-5
                       and metrics["max_absolute_ratio_minus_one"] <= 2e-5
                       and all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in parameters),
                       **metrics)
        report.data["long_cached_scoring"] = dict(status="passed", **metrics)
    report.data["long_sampling"] = dict(capacity=256, generated_steps=rollout["generated_steps"],
                                        completion_lengths=rollout["completion_lengths"].tolist(),
                                        full_model_GPU_memory_verified=False)
    report.write()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = CheckReport(args.output_dir, "cached_training_rollout_cpu.json")
    report.data.update(stage="streamed_cached_sampling_cpu_preflight", cases=[],
                       torch_version=torch.__version__, transformers_version=transformers.__version__,
                       production_sampler_changed=False,
                       scope="Tiny synthetic CPU models; no task rewards or GPU allocation")
    started = time.perf_counter()
    try:
        prompts = [[3, 4, 5]] * 4 + [[6, 7, 8, 9, 10]] * 4
        for dtype in (torch.float32, torch.bfloat16):
            for kind in ("lora", "geora"):
                model = tiny_adapter(kind, dtype)
                for temperature in (1.0, 0.7):
                    label = str(dtype).split(".")[-1] + "_" + kind + "_t" + str(temperature)
                    sampling_case(model, prompts, dtype, report, label, temperature=temperature)
                sampling_case(model, prompts, dtype, report, str(dtype).split(".")[-1] + "_" + kind + "_pad_eos",
                              temperature=0.7, pad=2, eos=tuple(range(37)))
        long_sampling_case(report)
        report.data["elapsed_seconds"] = time.perf_counter() - started
        report.finish()
    except Exception as error:
        report.data.update(status="failed", error=str(error), elapsed_seconds=time.perf_counter() - started)
        report.write()
        raise


if __name__ == "__main__":
    main()
