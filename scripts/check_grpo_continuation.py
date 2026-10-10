"""Five real GSM8K GRPO steps and one replay of step three after restoration.

Both methods share prompts, data order, sampling, reward, loss and precision.
Each method starts from an untrained adapter. Equal-reward groups are retained.
This is a continuity/resume check, not a task-score experiment.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIRECTORY))

from geora_layers import GeoRALinear, export_adapter_state, load_geora_state
from lora_layers import LoRALinear, export_lora_state, install_lora
from geora_check_utils import CheckReport, fresh_fp32_model, move_with_precision, pinned_configuration, validate_manifest
from check_grpo_training import likelihood_gate
from grpo_math import completion_token_mask, group_advantages, grpo_loss
from grpo_rollout import sample_completions, score_completion_logps
from grpo_training_state import load_boundary_checkpoint, make_constant_scheduler, save_boundary_checkpoint
from gsm8k_data import ANSWER_RULE, PROMPT_INSTRUCTION, build_prompt, file_sha256, load_prepared_rows, score_answer


def adapter_digest(state):
    """Hash the actual initialized FP32 factors, including names and shapes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def clone_cpu(value):
    """state_dict tensors alias live state; deep snapshots must clone tensors."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_cpu(item) for item in value)
    return copy.deepcopy(value)


def trees_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        return isinstance(actual, torch.Tensor) and torch.equal(actual.detach().cpu(), expected)
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(
            trees_equal(actual[key], expected[key]) for key in expected)
    if isinstance(expected, (list, tuple)):
        return isinstance(actual, type(expected)) and len(actual) == len(expected) and all(
            trees_equal(left, right) for left, right in zip(actual, expected))
    return actual == expected


def prompt_token_ids(tokenizer, questions, config):
    expanded = []
    for row in questions:
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": build_prompt(row["question"])}],
            tokenize=False,
            add_generation_prompt=True,
        )
        tokens = tokenizer.encode(chat, add_special_tokens=False)
        if len(tokens) > config["max_prompt_tokens"]:
            raise RuntimeError("Prompt exceeds the fixed limit; no silent truncation")
        expanded.extend([tokens] * config["answers_per_question"])
    return expanded


def one_step(model, reference, tokenizer, questions, optimizer, scheduler,
             rollout_generator, config, report, label, deadline):
    """Sample one prescribed batch, keep every group, and take one AdamW step."""
    device, frozen_dtype = "cuda", torch.bfloat16
    started = time.perf_counter()
    prompt_ids = prompt_token_ids(tokenizer, questions, config)
    report.data["current_stage"] = label + "_rollout"
    report.write()
    print(label, "questions:", [row["id"] for row in questions], flush=True)
    rollout = sample_completions(
        model, prompt_ids, tokenizer.pad_token_id, [tokenizer.eos_token_id],
        config["max_new_tokens"], device, frozen_dtype,
        temperature=config["temperature"], generator=rollout_generator,
        progress_callback=lambda progress: print(label, "ROLLOUT:", progress, flush=True),
        deadline=deadline,
    )
    rollout_seconds = time.perf_counter() - started
    answers, rewards = [], []
    for index in range(len(prompt_ids)):
        row = questions[index // config["answers_per_question"]]
        mask_row = rollout["completion_mask"][index]
        ids = rollout["completion_ids"][index][mask_row].tolist()
        text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        grading = score_answer(text, row["target"])
        ended = bool(rollout["ended_with_eos"][index].item())
        reward = grading["reward"] if ended else config["truncated_answer_reward"]
        rewards.append(reward)
        answers.append(dict(question_id=row["id"], question=row["question"], completion=text,
                            token_ids=ids, token_count=len(ids), ended_with_eos=ended,
                            truncated=not ended, **grading, training_reward=reward))
    report.data["physical_rollouts"] += 1
    report.data["current_completed_rollout"] = dict(
        label=label, question_ids=[row["id"] for row in questions], answers=answers,
        rewards=rewards, rollout_seconds=rollout_seconds,
    )
    report.write()
    grouped_rewards = torch.tensor(rewards, device=device).reshape(
        len(questions), config["answers_per_question"])
    advantages = group_advantages(grouped_rewards, config["advantage_epsilon"]).flatten()
    mask = rollout["completion_mask"]
    width = rollout["prompt_width"]
    independent_mask = completion_token_mask(
        rollout["input_ids"], rollout["attention_mask"],
        torch.full((mask.shape[0],), width, device=device), [tokenizer.eos_token_id],
    )[:, width:width + mask.shape[1]]
    report.require(label + "_causal_completion_mask", torch.equal(mask, independent_mask))
    scoring_started = time.perf_counter()
    old_logp = score_completion_logps(model, rollout, device, frozen_dtype, gradient_enabled=False)
    reference_logp = score_completion_logps(reference, rollout, device, frozen_dtype, gradient_enabled=False)
    sampled_alignment = likelihood_gate(
        report, label + "_sampled_and_old_likelihood", old_logp,
        rollout["behavior_logps"], mask, config,
    )
    if label == "step_1":
        likelihood_gate(report, "initial_policy_matches_original_reference",
                        old_logp, reference_logp, mask, config)
    current_logp = score_completion_logps(model, rollout, device, frozen_dtype, gradient_enabled=True)
    likelihood_gate(report, label + "_current_and_old_likelihood", current_logp, old_logp, mask, config)
    terms = grpo_loss(current_logp, old_logp, reference_logp, advantages, mask,
                      config["clip_epsilon"], config["kl_coefficient"])
    report.require(label + "_finite_loss", all(torch.isfinite(terms[key]).item()
                   for key in ("loss", "policy_loss", "kl", "clip_fraction")),
                   loss=terms["loss"].item(), policy_loss=terms["policy_loss"].item(),
                   sampled_old_token_k3=terms["kl"].item())
    scoring_seconds = time.perf_counter() - scoring_started
    if time.monotonic() >= deadline:
        raise TimeoutError("Bounded runtime reached before the next optimizer step")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    previous = [parameter.detach().clone() for parameter in trainable]
    optimizer.zero_grad(set_to_none=True)
    backward_started = time.perf_counter()
    terms["loss"].backward()
    report.data["physical_backward_calls"] += 1
    report.require(label + "_finite_FP32_AB_gradients", all(
        parameter.grad is not None and parameter.grad.dtype == torch.float32
        and torch.isfinite(parameter.grad).all().item() for parameter in trainable))
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        trainable, config["gradient_clip_norm"], error_if_nonfinite=True,
    ).item()
    optimizer.step()
    scheduler.step()
    report.data["physical_optimizer_step_calls"] += 1
    changed = sum(not torch.equal(parameter, before) for parameter, before in zip(trainable, previous))
    report.require(label + "_finite_AB_parameters", all(
        torch.isfinite(parameter).all().item() for parameter in trainable),
        changed_parameter_tensors=changed, gradient_norm_before_clipping=gradient_norm)
    optimizer.zero_grad(set_to_none=True)
    backward_seconds = time.perf_counter() - backward_started
    # Complete answer records remain in JSON even if a later step times out.
    record = dict(
        label=label, question_ids=[row["id"] for row in questions], answers=answers,
        rewards=grouped_rewards.tolist(), advantages=advantages.tolist(),
        equal_reward_groups=(grouped_rewards.std(1, correction=0) == 0).tolist(),
        parse_failure_count=sum(answer["predicted_answer"] is None for answer in answers),
        truncated_answer_count=sum(answer["truncated"] for answer in answers),
        valid_completion_tokens=int(mask.sum().item()), fixed_input_width=rollout["fixed_width"],
        loss=terms["loss"].item(), policy_loss=terms["policy_loss"].item(),
        sampled_old_token_k3=terms["kl"].item(), clip_fraction=terms["clip_fraction"].item(),
        gradient_norm_before_clipping=gradient_norm, changed_parameter_tensors=changed,
        learning_rate=optimizer.param_groups[0]["lr"],
        rollout_seconds=rollout_seconds, scoring_seconds=scoring_seconds,
        backward_and_step_seconds=backward_seconds, likelihood_alignment=sampled_alignment,
        behavior_logp=rollout["behavior_logps"].tolist(), old_logp=old_logp.tolist(),
        reference_logp=reference_logp.tolist(), completion_mask=mask.tolist(),
    )
    print(label, "rewards:", record["rewards"], "grad:", gradient_norm,
          "seconds:", round(time.perf_counter() - started, 2), flush=True)
    return record


def run(arguments, report):
    started = time.perf_counter()
    config_path = PROJECT_DIRECTORY / "configs/grpo_continuation.json"
    config = json.loads(config_path.read_text())
    if config["prompt_instruction"] != PROMPT_INSTRUCTION:
        raise RuntimeError("Dataset and configuration prompt differ")
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("Run this full-model check on one allocated ROCm GPU")
    torch.set_num_threads(8)
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    torch.cuda.reset_peak_memory_stats()
    deadline = time.monotonic() + config["runtime_limit_seconds"]
    base_config, original_checkpoint = pinned_configuration()
    staging = json.loads((arguments.checkpoint_dir / "staging_record.json").read_text())
    if staging["status"] != "validated" or staging["model_revision"] != base_config["model_revision"]:
        raise RuntimeError("Checkpoint copy did not pass CPU validation")
    if Path(staging["source"]).resolve() != original_checkpoint.resolve():
        raise RuntimeError("Checkpoint copy has the wrong original source")
    report.data.update(
        stage="gsm8k_five_step_continuation_and_resume", method=arguments.method,
        config=config, answer_rule=ANSWER_RULE, dataset_manifest_sha256=file_sha256(arguments.data_dir / "manifest.json"),
        model_repository=base_config["model_repository"], model_revision=base_config["model_revision"],
        source_git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"), gpu_name=torch.cuda.get_device_name(0),
        torch_version=torch.__version__, hip_version=torch.version.hip,
        logical_optimizer_steps=0, physical_optimizer_step_calls=0, physical_backward_calls=0,
        physical_rollouts=0, svd_calls=0, batches=[], checkpoint_staging=staging,
        precision=dict(frozen_weights="BF16", factors="FP32", initial_GeoRA_buffers="FP32",
                       adapter_branch="FP32", correction_added_to_base="BF16",
                       log_softmax_and_loss="FP32", optimizer_moments="FP32"),
        scope=config["scope"], timings={},
    )
    load_started = time.perf_counter()
    reference = fresh_fp32_model(arguments.checkpoint_dir, disable_mmap=True)
    policy = fresh_fp32_model(arguments.checkpoint_dir, disable_mmap=True)
    if arguments.method == "geora":
        manifest = json.loads((arguments.init_dir / "manifest.json").read_text())
        validate_manifest(manifest, base_config)
        manifest = dict(manifest, forward_mode="difference")
        state = load_file(str(arguments.init_dir / "adapter.safetensors"))
        report.require("starts_from_untrained_GeoRA", all(
            torch.equal(state[target["name"] + "." + factor], state[target["name"] + "." + factor + "0"])
            for target in manifest["target_modules"] for factor in ("A", "B")))
        load_geora_state(policy, state, manifest)
        del state
        initialized_state = export_adapter_state(policy)
    else:
        manifest = install_lora(policy, rank=base_config["initialization"]["rank"],
                                alpha=base_config["initialization"]["alpha"],
                                seed=config["seed"], init_std=config["lora_initial_A_std"],
                                provenance={key: base_config[key] for key in ("model_repository", "model_revision")})
        initialized_state = export_lora_state(policy)
        report.require("starts_from_random_A_zero_B_LoRA", all(
            torch.count_nonzero(value).item() == 0 for name, value in initialized_state.items()
            if name.endswith(".B")))
    provenance = dict(
        method=arguments.method, model_repository=base_config["model_repository"],
        model_revision=base_config["model_revision"],
        dataset_manifest_sha256=report.data["dataset_manifest_sha256"],
        training_config_sha256=file_sha256(config_path), initialization_sha256=adapter_digest(initialized_state),
    )
    del initialized_state
    report.data.update(provenance=provenance, adapter_manifest=manifest)
    move_with_precision(reference, "cuda", torch.bfloat16)
    move_with_precision(policy, "cuda", torch.bfloat16)
    reference.eval()
    policy.train()
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    targets = [(name, layer) for name, layer in policy.named_modules()
               if isinstance(layer, (GeoRALinear, LoRALinear))]
    expected_names = {name + "." + factor for name, _ in targets for factor in ("A", "B")}
    actual_names = {name for name, parameter in policy.named_parameters() if parameter.requires_grad}
    report.require("same_196_targets_only_AB_trainable", len(targets) == 196
                   and actual_names == expected_names and sum(parameter.numel() for parameter in trainable) == 18464768)
    report.require("BF16_frozen_FP32_AB_and_initial_buffers", all(
        parameter.dtype == (torch.float32 if parameter.requires_grad else torch.bfloat16)
        for parameter in policy.parameters()) and all(
        value.dtype == torch.float32 for name, value in policy.named_buffers() if name.endswith((".A0", ".B0"))))
    report.require("train_mode_zero_dropout_original_frozen_reference", policy.training
                   and policy.config.attention_dropout == 0 and not policy.is_gradient_checkpointing
                   and all(not isinstance(layer, torch.nn.Dropout) or layer.p == 0 for layer in policy.modules())
                   and not reference.training and all(not parameter.requires_grad for parameter in reference.parameters()))
    optimizer = torch.optim.AdamW(trainable, lr=config["learning_rate"],
                                 betas=tuple(config["optimizer_betas"]), eps=config["optimizer_epsilon"],
                                 weight_decay=config["weight_decay"])
    scheduler = make_constant_scheduler(optimizer)
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    frozen = {name: parameter.detach().clone() for name, parameter in policy.named_parameters() if not parameter.requires_grad}
    fixed = {name: value.detach().clone() for name, value in policy.named_buffers() if name.endswith((".A0", ".B0"))}
    ref_parameters = {name: parameter.detach().clone() for name, parameter in reference.named_parameters()}
    tokenizer = AutoTokenizer.from_pretrained(arguments.checkpoint_dir, local_files_only=True)
    rows = load_prepared_rows(arguments.data_dir, "smoke")[:config["optimizer_steps"] * config["questions_per_batch"]]
    report.require("fixed_ten_train_questions_without_reward_filtering", len(rows) == 10
                   and len({row["id"] for row in rows}) == 10 and all(row["source_split"] == "train" for row in rows),
                   data_order=[row["id"] for row in rows])
    progress = dict(completed_optimizer_steps=0, next_data_position=0, completed_rollouts=0,
                    boundary="after_rollout_and_optional_update", rollout_in_flight=False,
                    resume_audit_completed=False)
    if arguments.resume_dir is not None:
        progress = load_boundary_checkpoint(arguments.resume_dir, model=policy, optimizer=optimizer,
            scheduler=scheduler, rollout_generator=generator, expected_provenance=provenance, device="cuda")
        report.data["resumed_from"] = str(arguments.resume_dir)
    report.data["timings"]["load_and_setup_seconds"] = time.perf_counter() - load_started
    resume_checkpoint = arguments.resume_dir if progress["completed_optimizer_steps"] == 2 else None
    for step in range(progress["completed_optimizer_steps"] + 1, config["optimizer_steps"] + 1):
        offset = progress["next_data_position"]
        if offset != (step - 1) * config["questions_per_batch"]:
            raise RuntimeError("Restored data position is inconsistent with the step count")
        questions = rows[offset:offset + config["questions_per_batch"]]
        batch = one_step(policy, reference, tokenizer, questions, optimizer, scheduler,
                         generator, config, report, f"step_{step}", deadline)
        if step == config["resume_after_step"] + 1 and not progress["resume_audit_completed"]:
            if resume_checkpoint is None:
                raise RuntimeError("Missing step-two boundary checkpoint for the replay")
            expected_parameters = {name: parameter.detach().cpu().clone()
                                   for name, parameter in policy.named_parameters() if parameter.requires_grad}
            expected_optimizer = clone_cpu(optimizer.state_dict())
            expected_scheduler = clone_cpu(scheduler.state_dict())
            expected_generator = generator.get_state().clone()
            expected_cpu_rng = torch.get_rng_state().clone()
            expected_device_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]
            expected_python_rng = random.getstate()
            restored = load_boundary_checkpoint(resume_checkpoint, model=policy, optimizer=optimizer,
                scheduler=scheduler, rollout_generator=generator, expected_provenance=provenance, device="cuda")
            report.require("resume_restores_step_two_and_data_position", restored["completed_optimizer_steps"] == 2
                           and restored["next_data_position"] == 4)
            replay = one_step(policy, reference, tokenizer, questions, optimizer, scheduler,
                              generator, config, report, "resume_replay_step_3", deadline)
            report.require("resume_reproduces_question_ids_tokens_rewards", replay["question_ids"] == batch["question_ids"]
                           and [answer["token_ids"] for answer in replay["answers"]] == [answer["token_ids"] for answer in batch["answers"]]
                           and replay["rewards"] == batch["rewards"] and replay["advantages"] == batch["advantages"])
            report.require("resume_reproduces_sampling_and_old_logp_exactly",
                           replay["behavior_logp"] == batch["behavior_logp"] and replay["old_logp"] == batch["old_logp"])
            report.require("resume_reproduces_next_AB_update_exactly", all(
                torch.equal(parameter.detach().cpu(), expected_parameters[name])
                for name, parameter in policy.named_parameters() if parameter.requires_grad))
            report.require("resume_reproduces_AdamW_scheduler_generator_exactly",
                           trees_equal(optimizer.state_dict(), expected_optimizer)
                           and trees_equal(scheduler.state_dict(), expected_scheduler)
                           and torch.equal(generator.get_state(), expected_generator))
            report.require("resume_reproduces_global_CPU_GPU_Python_RNG_exactly",
                           torch.equal(torch.get_rng_state(), expected_cpu_rng)
                           and all(torch.equal(actual, expected) for actual, expected
                                   in zip(torch.cuda.get_rng_state_all(), expected_device_rng))
                           and random.getstate() == expected_python_rng)
            report.data["resume_replay"] = replay
            progress["resume_audit_completed"] = True
            del expected_parameters, expected_optimizer, expected_scheduler, expected_generator
        report.data["batches"].append(batch)
        progress.update(completed_optimizer_steps=step, next_data_position=offset + len(questions), completed_rollouts=step)
        boundary_path = arguments.output_dir / f"checkpoint-step-{step}"
        save_started = time.perf_counter()
        save_boundary_checkpoint(boundary_path, model=policy, optimizer=optimizer, scheduler=scheduler,
            rollout_generator=generator, progress=progress, provenance=provenance, device="cuda")
        batch["checkpoint_seconds"] = time.perf_counter() - save_started
        batch["checkpoint_directory"] = str(boundary_path)
        if step == config["resume_after_step"]:
            resume_checkpoint = boundary_path
        report.data.update(logical_optimizer_steps=step, last_boundary_checkpoint=str(boundary_path))
        report.write()
    parameters = dict(policy.named_parameters())
    buffers = dict(policy.named_buffers())
    report.require("all_frozen_weights_initial_buffers_reference_unchanged", all(
        torch.equal(parameters[name], before) and parameters[name].grad is None for name, before in frozen.items())
        and all(torch.equal(buffers[name], before) for name, before in fixed.items())
        and all(torch.equal(dict(reference.named_parameters())[name], before)
                for name, before in ref_parameters.items()))
    report.require("five_logical_steps_and_verified_resume", progress["completed_optimizer_steps"] == 5
                   and progress["next_data_position"] == 10 and progress["resume_audit_completed"])
    report.require("at_least_one_real_nonzero_update_without_filtering", any(
        batch["gradient_norm_before_clipping"] > 0 and batch["changed_parameter_tensors"] > 0
        for batch in report.data["batches"]))
    # A final, short sampling check is separate from the training and consumes
    # a private generator, leaving the saved training RNG at its boundary.
    check_generator = torch.Generator(device="cuda").manual_seed(config["seed"] + 1)
    final_rollout = sample_completions(policy, prompt_token_ids(tokenizer, rows[:2], config),
        tokenizer.pad_token_id, [tokenizer.eos_token_id], 8, "cuda", torch.bfloat16,
        generator=check_generator, deadline=deadline)
    final_logp = score_completion_logps(policy, final_rollout, "cuda", torch.bfloat16, gradient_enabled=False)
    likelihood_gate(report, "final_updated_policy_generation_and_scoring", final_logp,
                    final_rollout["behavior_logps"], final_rollout["completion_mask"], config)
    torch.cuda.synchronize()
    batches = report.data["batches"]
    report.data.update(
        current_stage="finished", elapsed_seconds=time.perf_counter() - started,
        peak_allocated_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
        total_logical_completion_tokens=sum(batch["valid_completion_tokens"] for batch in batches),
        total_physical_completion_tokens=sum(batch["valid_completion_tokens"] for batch in batches)
            + report.data.get("resume_replay", {}).get("valid_completion_tokens", 0),
        equal_reward_group_count=sum(sum(batch["equal_reward_groups"]) for batch in batches),
        parse_failure_count=sum(batch["parse_failure_count"] for batch in batches),
        truncated_answer_count=sum(batch["truncated_answer_count"] for batch in batches),
        next_data_position=progress["next_data_position"],
    )
    report.finish()
    print("Five-step GRPO continuity passed:", report.path, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("geora", "lora"), required=True)
    for name in ("init-dir", "data-dir", "checkpoint-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resume-dir", type=Path)
    arguments = parser.parse_args()
    report = CheckReport(arguments.output_dir, "continuation_checks.json")
    attempted_start = time.perf_counter()
    try:
        run(arguments, report)
    except Exception as error:
        report.data.update(status="failed", error=str(error), elapsed_seconds=time.perf_counter() - attempted_start)
        if torch.cuda.is_available():
            report.data["peak_allocated_memory_gib"] = torch.cuda.max_memory_allocated() / 2**30
        report.write()
        raise
