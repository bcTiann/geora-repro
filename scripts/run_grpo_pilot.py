"""Run or resume a bounded segment of the BF16 LoRA/GeoRA GSM8K pilot.

The existing one_step implementation owns sampling, likelihood gates, GRPO
loss and AdamW updates. This entry point adds deterministic train selection,
durable records and completed-boundary continuation. No SVD is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import uuid

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIRECTORY))
sys.path.insert(0, str(PROJECT_DIRECTORY / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, __version__ as transformers_version

from check_grpo_continuation import adapter_digest, one_step
from geora_check_utils import fresh_fp32_model, move_with_precision, pinned_configuration, validate_manifest
from geora_layers import GeoRALinear, export_adapter_state, load_geora_state
from grpo_training_state import load_boundary_checkpoint, make_constant_scheduler, save_boundary_checkpoint
from gsm8k_data import ANSWER_RULE, PROMPT_INSTRUCTION, configured_answer_rule, load_prepared_rows
from lora_layers import LoRALinear, export_lora_state, install_lora
from pilot_runtime import (
    COUNTERS, PilotReport, atomic_json, canonical_hash, exclusive_run, file_hash,
    initialization_hash, restore_record_history, run_training_segment, training_config_hash, utc_now,
)


def source_hashes():
    relative_paths = (
        "configs/base_model.json", "geora_layers.py", "lora_layers.py", "scripts/check_grpo_continuation.py",
        "scripts/check_grpo_training.py", "scripts/geora_check_utils.py",
        "scripts/grpo_math.py", "scripts/grpo_rollout.py", "scripts/grpo_training_state.py",
        "scripts/gsm8k_data.py",
    )
    sources = {name: file_hash(PROJECT_DIRECTORY / name) for name in relative_paths}
    sources["scripts/run_grpo_pilot.py"] = file_hash(Path(__file__))
    sources["scripts/pilot_runtime.py"] = file_hash(Path(__file__).with_name("pilot_runtime.py"))
    return sources


def tensor_digest(named_tensors, *, normalize_base_names=False):
    """Copy one tensor at a time to CPU; never retain a frozen GPU snapshot."""
    digest = hashlib.sha256()
    count = 0
    for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
        if normalize_base_names:
            name = name.replace(".base_layer.", ".")
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        cpu_tensor = tensor.detach().cpu().contiguous()
        # NumPy does not expose BF16, so hash its actual storage bytes.
        digest.update(cpu_tensor.view(torch.uint8).numpy().tobytes())
        count += tensor.numel()
        del cpu_tensor
    return dict(sha256=digest.hexdigest(), parameter_elements=count)


def immutable_digests(policy, reference):
    frozen = [(name.replace(".base_layer.", "."), value)
              for name, value in policy.named_parameters() if not value.requires_grad]
    return dict(
        frozen_policy=tensor_digest(frozen),
        initial_buffers=tensor_digest((name, value) for name, value in policy.named_buffers()
                                     if name.endswith((".A0", ".B0"))),
        frozen_reference=tensor_digest(reference.named_parameters()),
    )


def validate_config(config):
    if config["prompt_instruction"] != PROMPT_INSTRUCTION:
        raise ValueError("Configuration and prepared-data prompt instructions differ")
    for name in ("optimizer_steps", "questions_per_batch", "answers_per_question", "train_question_limit"):
        if isinstance(config[name], bool) or not isinstance(config[name], int) or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config.get("rollout_backend", "fixed_width") != "fixed_width":
        raise ValueError("This pilot preserves the accepted fixed_width probability path")
    if config.get("checkpoint_attention", False):
        raise ValueError("Attention checkpointing is not selected for this fixed_width pilot")
    if config["updates_per_rollout"] != 1 or config["group_std_correction"] != 0 or config["dropout"] != 0:
        raise ValueError("Only the validated one-update, population-std, zero-dropout policy is supported")
    if config["temperature"] != 1 or config["top_p"] != 1 or config["top_k"] != 0:
        raise ValueError("Changing the sampling transform requires new probability validation")


def checkpoint_source_record(directory, original, base_config):
    if directory.resolve() == original.resolve():
        return dict(status="pinned_original", source=str(original.resolve()))
    record = json.loads((directory / "staging_record.json").read_text())
    if record["status"] != "validated" or record["model_revision"] != base_config["model_revision"]:
        raise ValueError("Staged checkpoint has not passed CPU validation for this revision")
    if Path(record["source"]).resolve() != original.resolve():
        raise ValueError("Staged checkpoint has the wrong pinned original source")
    return record


def open_report(arguments, config, rows, source_record):
    directory = arguments.output_dir.resolve()
    path = directory / "pilot_report.json"
    identity = dict(
        method=arguments.method, dataset_manifest_sha256=file_hash(arguments.data_dir / "manifest.json"),
        training_config_sha256=training_config_hash(config),
        train_rows_sha256=canonical_hash(rows), train_question_ids=[row["id"] for row in rows],
        source_sha256=source_record, output_directory=str(directory),
    )
    if path.exists():
        if arguments.resume is None:
            raise ValueError("Existing pilot output requires --resume; previous records are preserved")
        data = json.loads(path.read_text())
        if data.get("identity") != identity or data.get("stage") != "gsm8k_bounded_grpo_pilot":
            raise ValueError("Output directory belongs to different data, method, configuration or source")
        resume = arguments.resume.resolve()
        if not resume.is_relative_to(directory / "checkpoints"):
            raise ValueError("Resume checkpoint must belong to this pilot output directory")
        return PilotReport(directory, data)
    if arguments.resume is not None:
        raise ValueError("Resume requires this pilot's original report and step records")
    if any(item.name != ".pilot.lock" for item in directory.iterdir()):
        raise ValueError("Refusing to use a nonempty output directory without a matching pilot report")
    data = dict(
        status="running", stage="gsm8k_bounded_grpo_pilot", checks=[],
        pilot_run_id=str(uuid.uuid4()), created_at_utc=utc_now(), identity=identity,
        config=config, answer_rule=configured_answer_rule(config), logical_optimizer_steps=0,
        next_data_position=0, completed_answer_tokens=0, active_step_records=[], segments=[],
        physical_tokens_accounted_rollout_count=0, svd_calls=0,
        **{key: 0 for key in COUNTERS},
    )
    report = PilotReport(directory, data)
    report.write()
    return report


def setup_models(arguments, config, report):
    base_config, original_checkpoint = pinned_configuration()
    staging = checkpoint_source_record(arguments.checkpoint_dir, original_checkpoint, base_config)
    reference = fresh_fp32_model(arguments.checkpoint_dir, disable_mmap=True)
    policy = fresh_fp32_model(arguments.checkpoint_dir, disable_mmap=True)
    if arguments.method == "geora":
        manifest = json.loads((arguments.init_dir / "manifest.json").read_text())
        validate_manifest(manifest, base_config)
        state = load_file(str(arguments.init_dir / "adapter.safetensors"))
        initial = all(torch.equal(state[target["name"] + "." + factor],
                                  state[target["name"] + "." + factor + "0"])
                      for target in manifest["target_modules"] for factor in ("A", "B"))
        report.require("rebuild_from_untrained_GeoRA_initialization", initial)
        manifest = dict(manifest, forward_mode="difference")
        load_geora_state(policy, state, manifest)
        del state
        initialized_state = export_adapter_state(policy)
    else:
        manifest = install_lora(
            policy, rank=base_config["initialization"]["rank"],
            alpha=base_config["initialization"]["alpha"], seed=config["seed"],
            init_std=config["lora_initial_A_std"],
            provenance={key: base_config[key] for key in ("model_repository", "model_revision")},
        )
        initialized_state = export_lora_state(policy)
        report.require("rebuild_from_random_A_zero_B_LoRA_initialization", all(
            torch.count_nonzero(value).item() == 0 for name, value in initialized_state.items()
            if name.endswith(".B")))
    provenance = dict(
        method=arguments.method, model_repository=base_config["model_repository"],
        model_revision=base_config["model_revision"],
        dataset_manifest_sha256=report.data["identity"]["dataset_manifest_sha256"],
        training_config_sha256=training_config_hash(config),
        initialization_sha256=initialization_hash(adapter_digest(initialized_state),
            base_config["initialization"]["rank"], base_config["initialization"]["alpha"]),
    )
    del initialized_state
    recorded = report.data.get("provenance")
    if recorded is not None and recorded != provenance:
        raise ValueError("Reconstructed initialization or model differs from the owned pilot")
    report.data.update(provenance=provenance, adapter_manifest=manifest, checkpoint_staging=staging)
    move_with_precision(reference, "cuda", torch.bfloat16)
    move_with_precision(policy, "cuda", torch.bfloat16)
    reference.eval()
    policy.train()
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    targets = [(name, layer) for name, layer in policy.named_modules()
               if isinstance(layer, (GeoRALinear, LoRALinear))]
    expected = {name + "." + factor for name, _ in targets for factor in ("A", "B")}
    actual = {name for name, parameter in policy.named_parameters() if parameter.requires_grad}
    report.require("same_196_targets_only_FP32_AB_trainable", len(targets) == 196 and actual == expected
                   and sum(parameter.numel() for parameter in trainable) == 18464768)
    report.require("BF16_frozen_FP32_AB_initial_buffers", all(
        parameter.dtype == (torch.float32 if parameter.requires_grad else torch.bfloat16)
        for parameter in policy.parameters()) and all(
        value.dtype == torch.float32 for name, value in policy.named_buffers() if name.endswith((".A0", ".B0"))))
    report.require("zero_dropout_no_gradient_checkpointing_original_frozen_reference", policy.training
                   and policy.config.attention_dropout == 0 and not policy.is_gradient_checkpointing
                   and all(not isinstance(layer, torch.nn.Dropout) or layer.p == 0 for layer in policy.modules())
                   and not reference.training and all(not parameter.requires_grad for parameter in reference.parameters()))
    report.data["precision"] = dict(
        frozen_weights="BF16", factors="FP32", initial_GeoRA_buffers="FP32",
        adapter_branch="FP32", correction_added_to_base="BF16", log_softmax_and_loss="FP32",
        optimizer_moments="FP32", checkpoint_adapter_storage="FP32",
    )
    optimizer = torch.optim.AdamW(
        trainable, lr=config["learning_rate"], betas=tuple(config["optimizer_betas"]),
        eps=config["optimizer_epsilon"], weight_decay=config["weight_decay"],
    )
    scheduler = make_constant_scheduler(optimizer)
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(arguments.checkpoint_dir, local_files_only=True)
    return policy, reference, tokenizer, optimizer, scheduler, generator, provenance


def run(arguments):
    process_started = time.perf_counter()
    process_deadline = time.monotonic() + arguments.runtime_limit_seconds
    config = json.loads(arguments.config.read_text())
    validate_config(config)
    if arguments.stop_after_step > config["optimizer_steps"]:
        raise ValueError("Segment target exceeds the final configured training target")
    config["checkpoint_interval"] = arguments.checkpoint_every
    rows = load_prepared_rows(arguments.data_dir, "train")[:config["train_question_limit"]]
    if len(rows) != config["train_question_limit"] or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Prepared train does not contain the complete requested unique subset")
    if any(row["source_split"] != "train" for row in rows):
        raise ValueError("Pilot data must come from the prepared train split")
    report = open_report(arguments, config, rows, source_hashes())
    prior_segment_count = len(report.data["segments"])
    try:
        if arguments.resume is not None:
            saved_progress = json.loads((arguments.resume / "manifest.json").read_text())["progress"]
            if arguments.stop_after_step < saved_progress["completed_optimizer_steps"]:
                raise ValueError("Segment target precedes its resume checkpoint")
        if torch.version.hip is None or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Run this pilot on one allocated ROCm GPU")
        torch.set_num_threads(8)
        torch.manual_seed(config["seed"])
        random.seed(config["seed"])
        torch.cuda.reset_peak_memory_stats()
        runtime = dict(
            slurm_job_id=os.environ.get("SLURM_JOB_ID"), gpu_name=torch.cuda.get_device_name(0),
            torch_version=str(torch.__version__), hip_version=torch.version.hip,
            transformers_version=transformers_version,
            runtime_limit_seconds=arguments.runtime_limit_seconds,
            absolute_stop_after_step=arguments.stop_after_step, checkpoint_every=arguments.checkpoint_every,
            source_git_commit=subprocess.check_output(
                ["git", "-C", str(PROJECT_DIRECTORY), "rev-parse", "HEAD"], text=True).strip(),
        )
        previous_runtime = report.data.get("runtime")
        if arguments.resume is not None and previous_runtime is not None:
            for key in ("gpu_name", "torch_version", "hip_version", "transformers_version"):
                if runtime[key] != previous_runtime.get(key):
                    raise ValueError(f"Exact continuation requires the same runtime: {key}")
        report.data["runtime"] = runtime
        load_started = time.perf_counter()
        policy, reference, tokenizer, optimizer, scheduler, generator, provenance = setup_models(arguments, config, report)
        before = immutable_digests(policy, reference)
        report.require("frozen_policy_is_exact_original_BF16_reference", before["frozen_policy"] == before["frozen_reference"])
        previous = report.data.get("initial_immutable_digests")
        if previous is not None:
            report.require("reconstructed_frozen_state_matches_original_run", before == previous)
        else:
            report.data["initial_immutable_digests"] = before
        if arguments.resume is not None:
            progress = load_boundary_checkpoint(
                arguments.resume, model=policy, optimizer=optimizer, scheduler=scheduler,
                rollout_generator=generator, expected_provenance=provenance, device="cuda",
            )
            restore_record_history(report, progress)
            report.data["resumed_from"] = str(arguments.resume.resolve())
        elif report.data["logical_optimizer_steps"] != 0:
            raise ValueError("An existing trained run requires its boundary checkpoint")
        report.data["load_and_setup_seconds"] = time.perf_counter() - load_started
        report.write()
        reserve = float(config.get("checkpoint_reserve_seconds", 60))
        if not math.isfinite(reserve) or reserve < 0:
            raise ValueError("checkpoint_reserve_seconds must be finite and nonnegative")
        try:
            status = run_training_segment(
                torch=torch, policy=policy, reference=reference, tokenizer=tokenizer,
                optimizer=optimizer, scheduler=scheduler, generator=generator,
                rows=rows, config=config, report=report, provenance=provenance,
                stop_after_step=arguments.stop_after_step, work_deadline=process_deadline - reserve,
                device="cuda", step_function=one_step, save_checkpoint=save_boundary_checkpoint,
            )
        finally:
            # Copy one frozen tensor at a time to CPU after the segment. The
            # reference, A0/B0 and frozen policy have no permanent GPU clones.
            after = immutable_digests(policy, reference)
            audit = dict(segment_id=report.data.get("current_segment_id"),
                         before=before, after=after, unchanged=before == after,
                         frozen_gradients_absent=all(parameter.grad is None for parameter in policy.parameters()
                                                     if not parameter.requires_grad),
                         reference_gradients_absent=all(parameter.grad is None for parameter in reference.parameters()))
            report.data.setdefault("immutability_audits", []).append(audit)
            report.require("frozen_policy_initial_buffers_reference_unchanged", audit["unchanged"]
                           and audit["frozen_gradients_absent"] and audit["reference_gradients_absent"])
        torch.cuda.synchronize()
        report.data.update(current_stage="segment_finished", status=status,
                           last_process_seconds=time.perf_counter() - process_started,
                           peak_allocated_memory_gib=torch.cuda.max_memory_allocated() / 2**30)
        report.data["segments"][-1]["process_seconds"] = report.data["last_process_seconds"]
        report.data["segments"][-1]["peak_allocated_memory_gib"] = report.data["peak_allocated_memory_gib"]
        report.data["segments"][-1]["runtime"] = dict(report.data["runtime"])
        report.data["cumulative_process_seconds"] = sum(
            segment.get("process_seconds", 0) for segment in report.data["segments"])
        segment = report.data["segments"][-1]
        atomic_json(report.directory / "segments" / (segment["id"] + ".json"), segment)
        report.write()
        print(json.dumps({key: report.data.get(key) for key in (
            "status", "logical_optimizer_steps", "physical_optimizer_step_calls", "physical_rollouts",
            "completed_answer_tokens", "physical_completion_tokens", "next_data_position", "latest_checkpoint",
        )}, indent=2), flush=True)
        print("Pilot report:", report.path, flush=True)
    except Exception as error:
        report.data.update(status="failed", error=str(error), last_process_seconds=time.perf_counter() - process_started)
        if len(report.data["segments"]) > prior_segment_count and report.data["segments"][-1].get("status") != "failed":
            report.data["segments"][-1].update(status="failed", error=str(error))
            segment = report.data["segments"][-1]
            atomic_json(report.directory / "segments" / (segment["id"] + ".json"), segment)
        if len(report.data["segments"]) > prior_segment_count:
            segment = report.data["segments"][-1]
            segment["process_seconds"] = report.data["last_process_seconds"]
            segment["runtime"] = dict(report.data.get("runtime", {}))
            atomic_json(report.directory / "segments" / (segment["id"] + ".json"), segment)
            report.data["cumulative_process_seconds"] = sum(
                item.get("process_seconds", 0) for item in report.data["segments"])
        report.write()
        raise


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("geora", "lora"), required=True)
    for name in ("init-dir", "data-dir", "checkpoint-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_DIRECTORY / "configs/grpo_pilot_plan.json")
    parser.add_argument("--resume", "--resume-dir", dest="resume", type=Path)
    parser.add_argument("--stop-after-step", type=int, default=10,
                        help="Absolute completed-step target for this segment; config controls the final target.")
    parser.add_argument("--runtime-limit-seconds", type=float, default=1000,
                        help="Elapsed process budget; reserve time for a completed-boundary checkpoint.")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    arguments = parser.parse_args()
    if arguments.stop_after_step < 0 or arguments.checkpoint_every < 1:
        parser.error("Step target must be nonnegative and checkpoint interval positive")
    if not math.isfinite(arguments.runtime_limit_seconds) or arguments.runtime_limit_seconds <= 0:
        parser.error("Runtime limit must be finite and positive")
    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    with exclusive_run(arguments.output_dir):
        run(arguments)
