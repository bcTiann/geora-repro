"""Evaluate a base or trained pilot checkpoint on persisted GSM8K validation.

Evaluation is greedy, batch one, native Transformers KV cache. It does not
provide probabilities to GRPO and never updates a parameter or training RNG.
Only the validation split is accepted; the final test split stays untouched.
"""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, GenerationConfig, __version__ as transformers_version

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIRECTORY))

from geora_layers import export_adapter_state, load_geora_state
from lora_layers import export_lora_state, install_lora
from check_grpo_continuation import adapter_digest
from geora_check_utils import forward_context, fresh_fp32_model, move_with_precision, pinned_configuration, validate_manifest
from gsm8k_data import ANSWER_RULE, PROMPT_INSTRUCTION, configured_answer_rule, build_prompt, file_sha256, load_prepared_rows, score_answer
from grpo_training_state import (
    BOUNDARY, CHECKPOINT_FORMAT, PROVENANCE_FIELDS,
    _validate_progress, _validate_provenance, canonical_json_sha256,
)
from pilot_runtime import atomic_json, exclusive_run, initialization_hash, training_config_hash, utc_now


def write_report(path, report):
    """Keep the previous complete JSON visible while the next one is written."""
    atomic_json(path, report)


def validation_rows(directory, limit):
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("Validation question limit must be a positive integer")
    rows = load_prepared_rows(directory, "validation")
    if limit > len(rows):
        raise ValueError("Requested evaluation exceeds the persisted validation split")
    train = load_prepared_rows(directory, "train")
    train_ids = {row["id"] for row in train}
    train_questions = {row["question"] for row in train}
    if any(row["source_split"] != "train" or row["id"] in train_ids
           or row["question"] in train_questions for row in rows):
        raise ValueError("Validation rows must be held out from the prepared training split")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate validation IDs")
    return rows[:limit]


def grade_completion(token_ids, eos_ids, text, row, answer_rule=ANSWER_RULE):
    """A numerically correct but capped answer receives zero, as in training."""
    if (not isinstance(token_ids, list) or not token_ids
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in token_ids)):
        raise ValueError("A generated completion must contain integer token IDs")
    if (not eos_ids or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                           for value in eos_ids)):
        raise ValueError("Evaluation requires one or more integer EOS IDs")
    if any(value in eos_ids for value in token_ids[:-1]):
        raise ValueError("Greedy generation must stop at the first EOS token")
    ended = bool(token_ids and token_ids[-1] in eos_ids)
    grading = score_answer(text, row["target"], answer_rule)
    return dict(
        question_id=row["id"], question=row["question"], completion=text,
        token_ids=token_ids, token_count=len(token_ids), ended_with_eos=ended,
        truncated=not ended, **grading,
        evaluation_reward=grading["reward"] if ended else 0.0,
    )


def evaluation_summary(answers, expected_count):
    count = len(answers)
    correct = sum(answer["evaluation_reward"] for answer in answers)
    return dict(
        evaluated_questions=count, requested_questions=expected_count,
        correct_count=int(correct), accuracy=None if count == 0 else correct / count,
        parse_failure_count=sum(answer["predicted_answer"] is None for answer in answers),
        parsed_incorrect_count=sum(answer["predicted_answer"] is not None and answer["reward"] == 0
                                   for answer in answers),
        correct_but_truncated_count=sum(answer["reward"] == 1 and answer["truncated"] for answer in answers),
        truncated_answer_count=sum(answer["truncated"] for answer in answers),
        generated_tokens=sum(answer["token_count"] for answer in answers),
        complete=count == expected_count,
    )


def validate_resume_identity(previous, identity):
    """Resume only the same persisted protocol; time/job controls may change."""
    if previous.get("status") not in ("running", "paused_between_questions"):
        raise ValueError("Only an interrupted or paused evaluation may resume")
    for name, expected in identity.items():
        if previous.get(name) != expected:
            raise ValueError(f"Evaluation resume identity differs: {name}")


def validate_answer_prefix(answers, rows, eos_ids, max_new_tokens, decode, answer_rule=ANSWER_RULE):
    """Do not trust a stored reward or allow reordered/skipped validation rows."""
    if not isinstance(answers, list) or len(answers) > len(rows):
        raise ValueError("Invalid saved evaluation answer count")
    for answer, row in zip(answers, rows):
        if answer.get("question_id") != row["id"]:
            raise ValueError("Saved evaluation answers are not the validation prefix")
        token_ids = answer.get("token_ids")
        if not isinstance(token_ids, list) or len(token_ids) > max_new_tokens:
            raise ValueError("Saved completion exceeds the configured token cap")
        text = decode(token_ids)
        expected = grade_completion(token_ids, eos_ids, text, row, answer_rule)
        if expected["truncated"] and len(token_ids) != max_new_tokens:
            raise ValueError("A saved non-EOS completion must reach the configured token cap")
        if any(answer.get(name) != value for name, value in expected.items()):
            raise ValueError(f"Saved evaluation answer/grading differs: {row['id']}")
        duration = answer.get("generation_seconds")
        if (isinstance(duration, bool) or not isinstance(duration, (int, float))
                or not math.isfinite(duration) or duration < 0):
            raise ValueError("Invalid saved question runtime")


def restore_adapter_for_evaluation(model, checkpoint, expected_provenance):
    """Read only adapter tensors; no optimizer or training RNG is restored."""
    _validate_provenance(expected_provenance)
    if set(expected_provenance) != set(PROVENANCE_FIELDS):
        raise ValueError("Evaluation requires exactly the six training provenance fields")
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    if (manifest.get("status") != "complete"
            or type(manifest.get("training_checkpoint_format")) is not int
            or manifest["training_checkpoint_format"] != CHECKPOINT_FORMAT
            or manifest.get("boundary") != BOUNDARY):
        raise ValueError("Expected a complete training boundary checkpoint")
    _validate_progress(manifest["progress"])
    if (manifest["provenance"] != expected_provenance
            or manifest.get("provenance_sha256") != canonical_json_sha256(expected_provenance)):
        raise ValueError("Evaluation checkpoint has different method/model/data/config/initialization")
    adapter_path = checkpoint / "adapter.safetensors"
    descriptor = manifest["files"]["adapter.safetensors"]
    if adapter_path.stat().st_size != descriptor["bytes"] or file_sha256(adapter_path) != descriptor["sha256"]:
        raise ValueError("Evaluation adapter checkpoint size/hash differs")
    state = load_file(str(adapter_path))
    parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
    initial_buffers = {name: value for name, value in model.named_buffers() if name.endswith((".A0", ".B0"))}
    if (not parameters or any(not name.endswith((".A", ".B")) for name in parameters)
            or any(value.dtype != torch.float32 or not torch.isfinite(value).all()
                   for value in list(parameters.values()) + list(initial_buffers.values()))):
        raise ValueError("Evaluation model factors must be named, finite FP32 A/B tensors")
    if (set(state) != set(parameters) | set(initial_buffers)
            or set(state) != set(manifest["adapter_tensors"])):
        raise ValueError("Evaluation adapter tensor names differ")
    # Validate the whole payload before copying any factor. A late corrupt B0
    # must not leave an earlier A/B partially restored in the caller's model.
    for name, tensor in state.items():
        destination = parameters.get(name) if name in parameters else initial_buffers[name]
        descriptor = manifest["adapter_tensors"][name]
        if (tensor.shape != destination.shape or tensor.dtype != torch.float32
                or descriptor.get("shape") != list(tensor.shape)
                or descriptor.get("dtype") != "torch.float32" or not torch.isfinite(tensor).all()):
            raise ValueError(f"Invalid evaluation factor: {name}")
        if name in initial_buffers and not torch.equal(tensor, destination):
            raise ValueError(f"Frozen GeoRA factor differs: {name}")
    with torch.no_grad():
        for name, destination in parameters.items():
            destination.copy_(state[name])
    return manifest


def run(arguments, report, report_path):
    started = time.perf_counter()
    previous = copy.deepcopy(report) if arguments.resume else None
    previous_elapsed = report.get("elapsed_seconds", 0.0) if previous is not None else 0.0
    deadline = time.monotonic() + arguments.runtime_limit_seconds
    config = json.loads(arguments.config.read_text())
    answer_rule = configured_answer_rule(config)
    for name in ("max_prompt_tokens", "max_new_tokens"):
        if isinstance(config[name], bool) or not isinstance(config[name], int) or config[name] < 1:
            raise ValueError(f"Evaluation {name} must be a positive integer")
    if arguments.method == "base":
        if arguments.adapter_checkpoint is not None or arguments.init_dir is not None:
            raise ValueError("Base evaluation does not accept adapter arguments")
    elif arguments.adapter_checkpoint is None:
        raise ValueError("A trained boundary checkpoint is required for adapter evaluation")
    elif arguments.method == "geora" and arguments.init_dir is None:
        raise ValueError("GeoRA evaluation requires the untrained initialization")
    elif arguments.method == "lora" and arguments.init_dir is not None:
        raise ValueError("LoRA evaluation does not use a GeoRA initialization directory")
    if config["prompt_instruction"] != PROMPT_INSTRUCTION:
        raise ValueError("Evaluation prompt differs from the prepared dataset")
    limit = config["validation_question_limit"] if arguments.question_limit is None else arguments.question_limit
    rows = validation_rows(arguments.data_dir, limit)
    device = arguments.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU evaluation requires one allocated visible GPU")
    dtype = torch.bfloat16
    torch.set_num_threads(8 if device == "cuda" else 2)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    base_config, original_checkpoint = pinned_configuration()
    if arguments.checkpoint_dir.resolve() != original_checkpoint.resolve():
        staging = json.loads((arguments.checkpoint_dir / "staging_record.json").read_text())
        if (staging["status"] != "validated" or staging["model_revision"] != base_config["model_revision"]
                or Path(staging["source"]).resolve() != original_checkpoint.resolve()):
            raise ValueError("Evaluation checkpoint copy has not passed the required validation")
    tokenizer = AutoTokenizer.from_pretrained(arguments.checkpoint_dir, local_files_only=True)
    eos_ids = [tokenizer.eos_token_id]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in eos_ids):
        raise ValueError("The tokenizer must provide an integer EOS token ID")
    generation_config = GenerationConfig(
        do_sample=False, max_new_tokens=config["max_new_tokens"], use_cache=True,
        eos_token_id=eos_ids, pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id, repetition_penalty=1.0,
    )
    source_paths = (
        "scripts/evaluate_gsm8k_pilot.py", "scripts/pilot_runtime.py",
        "scripts/grpo_training_state.py", "scripts/check_grpo_continuation.py",
        "scripts/geora_check_utils.py", "scripts/gsm8k_data.py", "geora_layers.py", "lora_layers.py",
        "configs/base_model.json", "third_party/verl/gsm8k.py", "third_party/verl/provenance.json",
    )
    identity = dict(
        stage="gsm8k_validation_greedy", method=arguments.method,
        model_repository=base_config["model_repository"], model_revision=base_config["model_revision"],
        training_config_sha256=training_config_hash(config),
        dataset_manifest_sha256=file_sha256(arguments.data_dir / "manifest.json"),
        question_ids=[row["id"] for row in rows], split="validation", answer_rule=answer_rule,
        source_git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_DIRECTORY, text=True).strip(),
        source_sha256={name: file_sha256(PROJECT_DIRECTORY / name) for name in source_paths},
        torch_version=str(torch.__version__), hip_version=torch.version.hip,
        transformers_version=transformers_version,
        device=device, gpu_name=torch.cuda.get_device_name(0) if device == "cuda" else None,
        precision=dict(frozen_weights="BF16", adapters="FP32", low_rank_branches="FP32"),
        decode=dict(batch_size=1, do_sample=False, use_cache=True, attention_backend="eager",
                    max_new_tokens=config["max_new_tokens"], repetition_penalty=1.0,
                    eos_token_ids=eos_ids, pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id),
        scope="Fixed validation pilot only; not final GSM8K test or GeoRA Table 8 scores",
        adapter_checkpoint=None if arguments.adapter_checkpoint is None else str(arguments.adapter_checkpoint.resolve()),
        adapter_checkpoint_sha256=None if arguments.adapter_checkpoint is None
            else file_sha256(arguments.adapter_checkpoint / "manifest.json"),
        adapter_tensor_file_sha256=None if arguments.adapter_checkpoint is None
            else file_sha256(arguments.adapter_checkpoint / "adapter.safetensors"),
        initialization_source=None if arguments.init_dir is None else dict(
            manifest_sha256=file_sha256(arguments.init_dir / "manifest.json"),
            adapter_sha256=file_sha256(arguments.init_dir / "adapter.safetensors")),
    )
    if previous is not None:
        validate_resume_identity(previous, identity)
        validate_answer_prefix(previous.get("answers"), rows, eos_ids, config["max_new_tokens"],
                               lambda tokens: tokenizer.decode(tokens, skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False), answer_rule)
        if (isinstance(previous_elapsed, bool) or not isinstance(previous_elapsed, (int, float))
                or not math.isfinite(previous_elapsed) or previous_elapsed < 0):
            raise ValueError("Invalid previous evaluation elapsed time")
    report.update(identity)
    report.update(status="running", config_file_sha256=file_sha256(arguments.config),
                  slurm_job_id=os.environ.get("SLURM_JOB_ID"))
    segment = dict(started_at_utc=utc_now(), slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                   resumed_after_questions=len(report["answers"]),
                   runtime_limit_seconds=arguments.runtime_limit_seconds,
                   config_file_sha256=report["config_file_sha256"])
    report.setdefault("segments", []).append(segment)
    report.update(evaluation_summary(report["answers"], limit))
    write_report(report_path, report)
    model = fresh_fp32_model(arguments.checkpoint_dir, disable_mmap=True)
    if arguments.method != "base":
        if arguments.adapter_checkpoint is None:
            raise ValueError("A trained boundary checkpoint is required for adapter evaluation")
        if arguments.method == "geora":
            if arguments.init_dir is None:
                raise ValueError("GeoRA evaluation requires the untrained initialization")
            manifest = json.loads((arguments.init_dir / "manifest.json").read_text())
            validate_manifest(manifest, base_config)
            manifest = dict(manifest, forward_mode="difference")
            state = load_file(str(arguments.init_dir / "adapter.safetensors"))
            if any(not torch.equal(state[target["name"] + "." + factor],
                                   state[target["name"] + "." + factor + "0"])
                   for target in manifest["target_modules"] for factor in ("A", "B")):
                raise ValueError("GeoRA source is not the untrained initialization")
            load_geora_state(model, state, manifest)
            del state
            initial_state = export_adapter_state(model)
        else:
            install_lora(model, rank=base_config["initialization"]["rank"],
                         alpha=base_config["initialization"]["alpha"], seed=config["seed"],
                         init_std=config["lora_initial_A_std"],
                         provenance={key: base_config[key] for key in ("model_repository", "model_revision")})
            initial_state = export_lora_state(model)
        expected_provenance = dict(
            method=arguments.method, model_repository=base_config["model_repository"],
            model_revision=base_config["model_revision"],
            dataset_manifest_sha256=report["dataset_manifest_sha256"],
            training_config_sha256=report["training_config_sha256"],
            initialization_sha256=initialization_hash(adapter_digest(initial_state),
                base_config["initialization"]["rank"], base_config["initialization"]["alpha"]),
        )
        del initial_state
        boundary = restore_adapter_for_evaluation(model, arguments.adapter_checkpoint, expected_provenance)
        report["completed_optimizer_steps"] = boundary["progress"]["completed_optimizer_steps"]
    elif arguments.adapter_checkpoint is not None or arguments.init_dir is not None:
        raise ValueError("Base evaluation does not accept adapter arguments")
    move_with_precision(model, device, dtype)
    model.requires_grad_(False)
    model.eval()
    for row in rows[len(report["answers"]):]:
        if time.monotonic() >= deadline:
            report["status"] = "paused_between_questions"
            break
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": build_prompt(row["question"])}],
            tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = tokenizer.encode(chat, add_special_tokens=False)
        if len(prompt_ids) > config["max_prompt_tokens"]:
            raise ValueError("Evaluation prompt exceeds the configured limit; no silent truncation")
        inputs = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generation_started = time.perf_counter()
        with torch.no_grad(), forward_context(device, dtype):
            output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                    generation_config=generation_config)
        token_ids = output[0, len(prompt_ids):].tolist()
        text = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        answer = grade_completion(token_ids, eos_ids, text, row, answer_rule)
        answer["generation_seconds"] = time.perf_counter() - generation_started
        report["answers"].append(answer)
        report.update(evaluation_summary(report["answers"], limit))
        segment["elapsed_seconds"] = time.perf_counter() - started
        segment["completed_questions"] = len(report["answers"]) - segment["resumed_after_questions"]
        report["elapsed_seconds"] = previous_elapsed + segment["elapsed_seconds"]
        write_report(report_path, report)
        print(f"Validation {len(report['answers'])}/{limit}: {row['id']} "
              f"reward={answer['evaluation_reward']} tokens={answer['token_count']} "
              f"truncated={answer['truncated']}", flush=True)
    report.update(evaluation_summary(report["answers"], limit))
    report["status"] = "completed" if report["complete"] else "paused_between_questions"
    segment["elapsed_seconds"] = time.perf_counter() - started
    segment["completed_questions"] = len(report["answers"]) - segment["resumed_after_questions"]
    segment["status"] = report["status"]
    report["elapsed_seconds"] = previous_elapsed + segment["elapsed_seconds"]
    if device == "cuda":
        report["peak_allocated_memory_gib"] = torch.cuda.max_memory_allocated() / 2**30
    write_report(report_path, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("base", "lora", "geora"), required=True)
    for name in ("data-dir", "checkpoint-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--init-dir", type=Path)
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_DIRECTORY / "configs/grpo_pilot_plan.json")
    parser.add_argument("--question-limit", type=int)
    parser.add_argument("--runtime-limit-seconds", type=int, default=1800)
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    parser.add_argument("--resume", action="store_true", help="Continue a paused evaluation without repeating completed questions.")
    arguments = parser.parse_args()
    if arguments.runtime_limit_seconds < 1:
        parser.error("runtime limit must be positive")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = arguments.output_dir / "validation.json"
    with exclusive_run(arguments.output_dir):
        if arguments.resume:
            if not report_path.exists():
                raise FileNotFoundError("Evaluation resume requires an existing validation.json")
            report = json.loads(report_path.read_text())
        else:
            if report_path.exists():
                raise FileExistsError("Use a fresh evaluation output directory or --resume")
            report = dict(status="running", answers=[])
        try:
            run(arguments, report, report_path)
        except Exception as error:
            report.update(status="failed", error=str(error))
            # A rejected resume request must not destroy an otherwise valid
            # saved prefix. Record this failed attempt in a separate artifact.
            failure_path = report_path if not arguments.resume else arguments.output_dir / (
                "validation-failed-attempt-" + str(time.time_ns()) + ".json")
            write_report(failure_path, report)
            print("Failed evaluation attempt:", failure_path, flush=True)
            raise
    print("Validation report:", report_path, flush=True)


if __name__ == "__main__":
    main()
