"""One real GSM8K GRPO update with the validated difference implementation.

This bounded check starts from the saved UNTRAINED factors. It does not reuse
the CE diagnostic checkpoint, train for task scores, or compare LoRA/GeoRA.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIRECTORY))

from geora_layers import GeoRALinear, export_adapter_state, load_geora_state
from geora_check_utils import CheckReport, fresh_fp32_model, move_with_precision, pinned_configuration, validate_manifest
from check_difference_training import RUNTIME_PRECISION, effective_update_norm
from grpo_math import completion_token_mask, group_advantages, grpo_loss
from gsm8k_data import ANSWER_RULE, PROMPT_INSTRUCTION, build_prompt, file_sha256, load_prepared_rows, score_answer
from grpo_rollout import sample_completions, score_completion_logps


def logp_metrics(actual, expected, mask):
    difference = (actual.detach() - expected.detach())[mask]
    return {
        "max_absolute_logp_difference": difference.abs().max().item(),
        "mean_absolute_logp_difference": difference.abs().mean().item(),
        "max_absolute_ratio_minus_one": torch.expm1(difference).abs().max().item(),
        "sequence_logp_differences": ((actual.detach() - expected.detach()) * mask).sum(1).tolist(),
    }


def likelihood_gate(report, name, actual, expected, mask, config):
    metrics = logp_metrics(actual, expected, mask)
    report.require(name, torch.isfinite(actual[mask]).all().item()
                   and metrics["max_absolute_logp_difference"] <= config["max_absolute_logp_mismatch"]
                   and metrics["max_absolute_ratio_minus_one"] <= config["max_absolute_ratio_mismatch"]
                   and max(map(abs, metrics["sequence_logp_differences"])) <= config["max_absolute_sequence_logp_mismatch"],
                   **metrics)
    return metrics


def run(arguments, report):
    started = time.perf_counter()
    config = json.loads((PROJECT_DIRECTORY / "configs/grpo_smoke.json").read_text())
    if config["prompt_instruction"] != PROMPT_INSTRUCTION:
        raise RuntimeError("Smoke configuration and persisted data prompt must match.")
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("This full-model smoke check requires one visible ROCm GPU.")
    torch.set_num_threads(8)
    torch.manual_seed(config["seed"])
    device, frozen_dtype = "cuda", torch.bfloat16
    torch.cuda.reset_peak_memory_stats()
    configuration, original_checkpoint = pinned_configuration()
    checkpoint = arguments.checkpoint_dir
    staging = json.loads((checkpoint / "staging_record.json").read_text())
    if staging["status"] != "validated" or staging["model_revision"] != configuration["model_revision"]:
        raise RuntimeError("Checkpoint copy must pass CPU validation before this allocation.")
    if Path(staging["source"]).resolve() != original_checkpoint.resolve():
        raise RuntimeError("Staged source differs from the pinned original checkpoint.")
    saved_manifest = json.loads((arguments.init_dir / "manifest.json").read_text())
    validate_manifest(saved_manifest, configuration)
    manifest = dict(saved_manifest, forward_mode="difference", runtime_precision=RUNTIME_PRECISION)
    state = load_file(str(arguments.init_dir / "adapter.safetensors"))
    report.data.update(
        stage="gsm8k_first_real_grpo_update", config=config, runtime_precision=RUNTIME_PRECISION,
        forward_mode="difference", optimizer_steps=0, backward_calls=0, svd_calls=0,
        model_repository=configuration["model_repository"], model_revision=configuration["model_revision"],
        source_git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"), gpu_name=torch.cuda.get_device_name(0),
        torch_version=torch.__version__, hip_version=torch.version.hip,
        initialization_directory=str(arguments.init_dir), checkpoint_staging=staging,
        dataset_manifest=json.loads((arguments.data_dir / "manifest.json").read_text()),
        dataset_manifest_sha256=file_sha256(arguments.data_dir / "manifest.json"),
        attempted_batches=[], timings={}, scope=config["scope"],
    )
    report.require("starts_from_untrained_factors", all(
        torch.equal(state[f"{target['name']}.{factor}"], state[f"{target['name']}.{factor}0"])
        for target in manifest["target_modules"] for factor in ("A", "B")))
    load_started = time.perf_counter()
    reference = fresh_fp32_model(checkpoint, disable_mmap=True)
    model = fresh_fp32_model(checkpoint, disable_mmap=True)
    load_geora_state(model, state, manifest)
    del state
    move_with_precision(reference, device, frozen_dtype)
    move_with_precision(model, device, frozen_dtype)
    report.data["timings"]["load_seconds"] = time.perf_counter() - load_started
    reference.eval()
    model.train()
    targets = [(name, layer) for name, layer in model.named_modules() if isinstance(layer, GeoRALinear)]
    trainable = [p for p in model.parameters() if p.requires_grad]
    expected_names = {name + "." + factor for name, _ in targets for factor in ("A", "B")}
    actual_names = {name for name, p in model.named_parameters() if p.requires_grad}
    report.require("only_196_difference_targets_AB_trainable",
                   len(targets) == 196 and actual_names == expected_names
                   and sum(p.numel() for p in trainable) == 18464768
                   and all(layer.forward_mode == "difference" for _, layer in targets))
    report.require("BF16_frozen_FP32_factors_and_buffers", all(
        p.dtype == (torch.float32 if p.requires_grad else frozen_dtype) for p in model.parameters())
        and all(layer.A0.dtype == torch.float32 and layer.B0.dtype == torch.float32 for _, layer in targets))
    report.require("train_mode_zero_dropout_and_no_checkpointing", model.training
                   and model.config.attention_dropout == 0
                   and all(not isinstance(layer, torch.nn.Dropout) or layer.p == 0 for layer in model.modules())
                   and not model.is_gradient_checkpointing)
    report.require("original_reference_frozen", not reference.training
                   and all(not p.requires_grad and p.dtype == frozen_dtype for p in reference.parameters()))
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    candidates = load_prepared_rows(arguments.data_dir, "smoke")
    report.require("fixed_16_training_only_candidates", len(candidates) == config["candidate_question_count"]
                   and all(row["source_split"] == "train" for row in candidates))
    frozen_snapshot = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    fixed_snapshot = {name: p.detach().clone() for name, p in model.named_buffers() if name.endswith((".A0", ".B0"))}
    reference_snapshot = {name: p.detach().clone() for name, p in reference.named_parameters()}
    deadline = started + config["runtime_limit_seconds"]
    chosen = None
    for offset in range(0, len(candidates), config["questions_per_batch"]):
        if time.perf_counter() > deadline:
            raise RuntimeError("Bounded runtime reached before a mixed-reward group; no optimizer step.")
        questions = candidates[offset:offset + config["questions_per_batch"]]
        prompt_ids = []
        for row in questions:
            chat = tokenizer.apply_chat_template([{"role": "user", "content": build_prompt(row["question"])}],
                                                 tokenize=False, add_generation_prompt=True)
            ids = tokenizer.encode(chat, add_special_tokens=False)
            if len(ids) > config["max_prompt_tokens"]:
                raise RuntimeError("Prompt exceeds declared limit; do not silently truncate.")
            prompt_ids.extend([ids] * config["answers_per_question"])
        report.data["current_stage"] = f"rollout_batch_{offset // config['questions_per_batch']}"
        report.write()
        print("Sampling questions:", [row["id"] for row in questions], flush=True)
        rollout_started = time.perf_counter()
        rollout = sample_completions(model, prompt_ids, tokenizer.pad_token_id,
                                     [tokenizer.eos_token_id], config["max_new_tokens"], device, frozen_dtype,
                                     progress_callback=lambda progress: print("ROLLOUT:", progress, flush=True),
                                     deadline=time.monotonic() + max(0, deadline-time.perf_counter()))
        rows = []
        rewards = []
        for answer_index in range(len(prompt_ids)):
            question = questions[answer_index // config["answers_per_question"]]
            valid_ids = rollout["completion_ids"][answer_index][rollout["completion_mask"][answer_index]].tolist()
            text = tokenizer.decode(valid_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            grading = score_answer(text, question["target"])
            ended = any(token in valid_ids for token in [tokenizer.eos_token_id])
            reward = grading["reward"] if ended else config["truncated_answer_reward"]
            rewards.append(reward)
            rows.append(dict(question_id=question["id"], question=question["question"], completion=text,
                             token_ids=valid_ids, token_count=len(valid_ids), ended_with_eos=ended,
                             truncated=not ended, **grading, training_reward=reward))
        reward_groups = torch.tensor(rewards, device=device).reshape(len(questions), config["answers_per_question"])
        advantages = group_advantages(reward_groups, config["advantage_epsilon"]).flatten()
        batch = dict(questions=[row["id"] for row in questions], answers=rows,
                     rewards=reward_groups.tolist(), advantages=advantages.tolist(),
                     rollout_seconds=time.perf_counter()-rollout_started,
                     mixed_groups=(reward_groups.std(1, correction=0) > 0).tolist())
        report.data["attempted_batches"].append(batch)
        report.write()
        print("Real rewards:", batch["rewards"], "seconds:", round(batch["rollout_seconds"], 2), flush=True)
        if bool((advantages != 0).any().item()):
            chosen = rollout, advantages, batch
            break
    report.require("real_mixed_reward_group_found", chosen is not None)
    rollout, advantages, batch = chosen
    mask = rollout["completion_mask"]
    width = rollout["prompt_width"]
    independent_mask = completion_token_mask(rollout["input_ids"], rollout["attention_mask"],
                                            torch.full((mask.shape[0],), width, device=device),
                                            [tokenizer.eos_token_id])[:, width:width + mask.shape[1]]
    report.require("EOS_prompt_padding_mask_matches_independent_indices", torch.equal(mask, independent_mask))
    scoring_started = time.perf_counter()
    with torch.no_grad():
        old_logp = score_completion_logps(model, rollout, device, frozen_dtype, gradient_enabled=False).detach()
        ref_logp = score_completion_logps(reference, rollout, device, frozen_dtype, gradient_enabled=False).detach()
    likelihood_gate(report, "sampled_behavior_and_old_scoring_likelihood_align",
                    old_logp, rollout["behavior_logps"], mask, config)
    likelihood_gate(report, "initial_reference_and_policy_align", old_logp, ref_logp, mask, config)
    current_logp = score_completion_logps(model, rollout, device, frozen_dtype, gradient_enabled=True)
    likelihood_gate(report, "gradient_enabled_current_and_sampled_behavior_align",
                    current_logp, rollout["behavior_logps"], mask, config)
    likelihood_gate(report, "unchanged_current_old_ratio_one", current_logp, old_logp, mask, config)
    terms = grpo_loss(current_logp, old_logp, ref_logp, advantages, mask,
                      config["clip_epsilon"], config["kl_coefficient"])
    report.require("finite_GRPO_loss", torch.isfinite(terms["loss"]).item(),
                   loss=terms["loss"].item(), policy_loss=terms["policy_loss"].item(),
                   kl=terms["kl"].item(), clip_fraction=terms["clip_fraction"].item())
    report.data["timings"]["scoring_seconds"] = time.perf_counter() - scoring_started
    if time.perf_counter() > deadline:
        raise RuntimeError("Runtime limit reached before backward; no optimizer step.")
    optimizer = torch.optim.AdamW(trainable, lr=config["learning_rate"],
                                 betas=tuple(config["optimizer_betas"]), eps=config["optimizer_epsilon"],
                                 weight_decay=config["weight_decay"])
    optimizer.zero_grad(set_to_none=True)
    before = [p.detach().clone() for p in trainable]
    report.data["current_stage"] = "real_grpo_backward"
    report.write()
    backward_started = time.perf_counter()
    terms["loss"].backward()
    report.data["backward_calls"] += 1
    report.require("padded_batch_AB_gradients_finite_FP32_nonzero", all(
        p.grad is not None and p.grad.dtype == torch.float32 and torch.isfinite(p.grad).all().item()
        for p in trainable) and any(torch.count_nonzero(p.grad).item() > 0 for p in trainable))
    gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, config["gradient_clip_norm"], error_if_nonfinite=True).item()
    optimizer.step()
    report.data["optimizer_steps"] = 1
    report.data["timings"]["backward_and_step_seconds"] = time.perf_counter()-backward_started
    report.data["gradient_norm_before_clipping"] = gradient_norm
    report.require("AB_updated_finite", all(torch.isfinite(p).all().item() for p in trainable)
                   and any(not torch.equal(p, original) for p, original in zip(trainable, before)))
    layer_updates = [{"name": name, "effective_update_norm": effective_update_norm(layer)} for name, layer in targets]
    report.require("finite_nonzero_effective_update", all(math.isfinite(row["effective_update_norm"]) for row in layer_updates)
                   and any(row["effective_update_norm"] > 0 for row in layer_updates))
    parameters, buffers = dict(model.named_parameters()), dict(model.named_buffers())
    report.require("frozen_weights_A0_B0_reference_unchanged", all(
        torch.equal(parameters[name], original) and parameters[name].grad is None for name, original in frozen_snapshot.items())
        and all(torch.equal(buffers[name], original) for name, original in fixed_snapshot.items())
        and all(torch.equal(dict(reference.named_parameters())[name], original) for name, original in reference_snapshot.items()))
    with torch.no_grad():
        after_logp = score_completion_logps(model, rollout, device, frozen_dtype, gradient_enabled=False).detach()
        reference_after = score_completion_logps(reference, rollout, device, frozen_dtype, gradient_enabled=False).detach()
    report.require("reference_scores_unchanged", torch.equal(reference_after, ref_logp))
    report.require("updated_scoring_finite_changed", torch.isfinite(after_logp[mask]).all().item()
                   and not torch.equal(after_logp[mask], old_logp[mask]))
    post_terms = grpo_loss(after_logp, old_logp, ref_logp, advantages, mask,
                           config["clip_epsilon"], config["kl_coefficient"])
    report.data["post_update"] = {key: post_terms[key].item() for key in ("loss", "policy_loss", "kl", "clip_fraction")}
    report.data["effective_layer_updates"] = layer_updates
    report.data["selected_rollout"] = dict(
        **batch, prompt_width=width, fixed_input_width=rollout["input_ids"].shape[1],
        completion_mask=mask.tolist(), behavior_logp=rollout["behavior_logps"].tolist(),
        old_logp=old_logp.tolist(), reference_logp=ref_logp.tolist(), after_logp=after_logp.tolist())
    checkpoint_out = arguments.output_dir / "trained_adapter"
    checkpoint_out.mkdir(exist_ok=True)
    save_file(export_adapter_state(model), str(checkpoint_out / "adapter.safetensors"))
    manifest.update(training_stage="one_real_GSM8K_GRPO_update", optimizer_steps=1,
                    initialization_directory=str(arguments.init_dir), smoke_config=config,
                    checkpoint_contents="FP32 current and initial factors; reconstruct difference mode on original W_pre")
    (checkpoint_out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    torch.save(optimizer.state_dict(), checkpoint_out / "optimizer.pt")
    # A short post-update sampling gate is separate from before/after scoring.
    post_generation = sample_completions(model, prompt_ids[:2], tokenizer.pad_token_id,
                                        [tokenizer.eos_token_id], 8, device, frozen_dtype)
    report.require("post_update_stochastic_generation_finite", torch.isfinite(
        post_generation["behavior_logps"][post_generation["completion_mask"]]).all().item())
    report.data["post_update_generation"] = [tokenizer.decode(ids[valid].tolist(), skip_special_tokens=True)
                                               for ids, valid in zip(post_generation["completion_ids"], post_generation["completion_mask"])]
    torch.cuda.synchronize()
    report.data.update(current_stage="finished", elapsed_seconds=time.perf_counter()-started,
                       peak_allocated_memory_gib=torch.cuda.max_memory_allocated()/2**30,
                       trained_adapter_directory=str(checkpoint_out))
    report.finish()
    print("Real GRPO smoke passed:", report.path, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("init-dir", "data-dir", "checkpoint-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    arguments = parser.parse_args()
    report = CheckReport(arguments.output_dir, "grpo_checks.json")
    attempted_start = time.perf_counter()
    try:
        run(arguments, report)
    except Exception as error:
        report.data.update(status="failed", error=str(error), elapsed_seconds=time.perf_counter()-attempted_start)
        if torch.cuda.is_available():
            report.data["peak_allocated_memory_gib"] = torch.cuda.max_memory_allocated()/2**30
        report.write()
        raise
