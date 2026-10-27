"""Two real updates, with checkpoint1 and a new server/learner between them.

This diagnostic accepts a selected batch schedule; it cannot launch formal400.
Both processes retain max_steps=2 and the complete frozen training split.
Checkpoint25 initializes phase1 policy only, with a new optimizer at version0.
"""

from __future__ import annotations

import argparse
from collections import Counter
from importlib import import_module
from importlib.metadata import version
import json
from pathlib import Path
import socket
import time

import torch
from transformers import AutoTokenizer, set_seed

from checkpoint_audit import checkpoint_inventory, state_digest, verify_loaded_policy
from train_native import (ROOT, GSM8K_REWARD_VERSION, NativeGeoRATrainer, NativeLoRATrainer,
                          NativeMonitor, PerformanceRecorder, install_native_sync,
                          capacity_probe, load_model, make_audited_gsm8k_reward, prepare_dataset, score_three_way,
                          sha256_file, tensor_hashes, training_args, validate_fixed_rows, write_json)
from training_schedule import TrainingSchedule


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


class PhaseMonitor(NativeMonitor):
    def __init__(self, output, seconds_limit, phase):
        super().__init__(output, "schedule_acceptance", seconds_limit)
        self.phase = phase
        self.optimizer_initialization = None

    def on_train_begin(self, args, state, control, optimizer=None, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        require(optimizer is not None and state.global_step == self.phase - 1,
                "Unexpected optimizer/global step at training start")
        saved = optimizer.state_dict()
        require(isinstance(saved.get("state"), dict), "Cannot inspect actual optimizer state")
        if self.phase == 1:
            require(not saved["state"], "Phase1 must use a new optimizer with empty state")
        else:
            require(bool(saved["state"]), "Phase2 optimizer moments were not restored")
        self.optimizer_initialization = {"global_step": int(state.global_step),
            "fresh_empty_state": not saved["state"], "state_entries": len(saved["state"]),
            "state_sha256": state_digest(saved)}
        write_json(self.output / "optimizer_at_train_begin.json", self.optimizer_initialization)

    def on_step_end(self, args, state, control, **kwargs):
        super().on_step_end(args, state, control, **kwargs)
        if state.global_step >= self.phase:
            control.should_training_stop = True
            control.should_save = True


def actual_questions(records, schedule, start):
    require([row["optimizer_step"] for row in records] == [start] * schedule.rollouts_per_update,
            "Wrong actual rollout count or policy version")
    ids = [identifier for row in records for identifier in row["question_ids"]]
    require(all(isinstance(identifier, str) and identifier and identifier != "None" for identifier in ids),
            "Missing actual training question identities")
    require(len(ids) == 128 and len(set(ids)) == 16 and set(Counter(ids).values()) == {8},
            "Each phase must consume16 distinct questions with8 responses each")
    require(all(len(set(ids[index:index+8])) == 1 for index in range(0, len(ids), 8)),
            "Actual questions do not form consecutive G8 groups")
    return {"question_ids": ids, "unique_question_ids": sorted(set(ids)), "responses": len(ids),
            "question_counts": dict(Counter(ids)), "scope": "Actual inputs; stock skip is audited separately"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("lora", "geora_mixed"), required=True)
    parser.add_argument("--phase", type=int, choices=(1, 2), required=True)
    parser.add_argument("--micro-batch-size", type=int, required=True)
    parser.add_argument("--generation-batch-size", type=int, choices=(32, 64, 128), required=True)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--initialization-checkpoint", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "formal_config.json")
    parser.add_argument("--fixed-tokens", type=Path, default=ROOT / "fixtures/probability_controls.json")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--group-port", type=int, required=True)
    parser.add_argument("--seconds-limit", type=int, default=360)
    args = parser.parse_args()
    if ((args.phase == 1) != bool(args.initialization_checkpoint)
            or (args.phase == 2) != bool(args.resume_checkpoint)):
        parser.error("Phase1 requires only --initialization-checkpoint; phase2 requires only --resume-checkpoint")
    if args.seconds_limit <= 0:
        parser.error("seconds-limit must be positive")
    if args.output.exists():
        raise FileExistsError("Use a fresh phase output directory")
    config = read(args.config)
    require(config["training"]["answers_per_update"] == 128 and config["training"]["num_generations"] == 8,
            "This diagnostic retains128 responses/update and G8")
    schedule = TrainingSchedule(args.micro_batch_size, args.generation_batch_size, 128, 8)
    versions = {name: version(name) for name in config["runtime_versions"]}
    require(versions == config["runtime_versions"] and sha256_file(ROOT / "uv.lock") == config["runtime_lock_sha256"],
            "The accepted runtime or lock changed")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one learner GCD must be visible")
    torch.set_num_threads(4)
    # Reuse the frozen-manifest verifier only; no old fixed-batch training entry runs.
    base, original_checkpoint, frozen_protocol = import_module("02_train_split").verify_inputs(args)
    args.base_model = base
    require(args.phase != 1 or args.initialization_checkpoint.resolve() == original_checkpoint.resolve(),
            "Phase1 initialization differs from the frozen execution manifest")
    for name, digest in config["model"]["expected_base_files_sha256"].items():
        require(sha256_file(base / name) == digest, f"Base file changed: {name}")
    require(sha256_file(args.train_data) == config["data"]["train_file_sha256"]
            and config["training"]["reward"] == GSM8K_REWARD_VERSION
            and frozen_protocol["settings"]["learning_rate"] == config["training"]["learning_rate"],
            "Frozen data/reward/learning rate differs from the selected configuration")
    original_inventory = checkpoint_inventory(original_checkpoint, args.variant)
    require(original_inventory["global_step"] == 25, "Expected frozen initialization checkpoint25")
    resume_inventory = checkpoint_inventory(args.resume_checkpoint, args.variant) if args.resume_checkpoint else None
    if resume_inventory:
        require(resume_inventory["global_step"] == 1, "Phase2 requires this diagnostic's checkpoint1")
        require(args.output.resolve() != args.resume_checkpoint.resolve()
                and args.resume_checkpoint.resolve() not in args.output.resolve().parents,
                "Output must be outside the input checkpoint")
    fixed_rows = read(args.fixed_tokens)
    validate_fixed_rows(fixed_rows)
    require(len(fixed_rows) == 5, "All five frozen probability controls are required")
    contract = {"variant": args.variant, "schedule": schedule.as_dict(), "max_steps_for_both_phases": 2,
        "scientific_configuration_sha256": sha256_file(args.config), "versions": versions,
        "uv_lock_sha256": sha256_file(ROOT / "uv.lock"), "execution_manifest_sha256": sha256_file(args.execution_manifest),
        "original_policy_initialization": original_inventory, "train_data_sha256": sha256_file(args.train_data),
        "fixed_tokens_sha256": sha256_file(args.fixed_tokens), "base_files_sha256": config["model"]["expected_base_files_sha256"],
        "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(ROOT.rglob("*.py"))
                          if "__pycache__" not in path.parts}, "reward": GSM8K_REWARD_VERSION,
        "training_seed": config["training"]["seed"], "data_seed": config["training"]["data_seed"]}
    prior_root = args.resume_checkpoint.parent.parent if args.resume_checkpoint else None
    prior = read(prior_root / "result.json") if prior_root else None
    if args.phase == 2:
        require(isinstance(prior, dict) and prior.get("completed") is True, "Phase1 result is missing or invalid")
        require(read(prior_root / "run_protocol.json")["immutable_contract"] == contract,
                "Phase1/phase2 source, schedule, configuration or frozen inputs differ")
        require(prior["completed"] is True and prior["phase"] == 1 and prior["variant"] == args.variant
                and prior["total_global_step"] == 1 and prior["saved_inventory"] == resume_inventory,
                "Resume checkpoint is not the completed same-schedule phase1 output")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    monitor = None
    # Reuse training_args while making the bounded saving policy explicit below.
    args.mode, args.max_steps = "schedule_acceptance", 2
    recorder = PerformanceRecorder(args.output)
    write_json(args.output / "run_protocol.json", {"mode": args.mode, "phase": args.phase,
        "immutable_contract": contract, "resume_input": resume_inventory,
        "server_sampling_rng_restored": False, "bitwise_sampling_replay_guaranteed": False,
        "sampling_seed_policy": "training_seed + G * (global_optimizer_step * rollouts_per_update + rollout_index)"})
    write_json(args.output / "invocation.json", {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    try:
        write_json(args.output / "environment.json", {"hostname": socket.gethostname(), "versions": versions,
            "device_name": torch.cuda.get_device_properties(0).name, "hip_version": torch.version.hip})
        set_seed(config["training"]["seed"])
        with recorder.stage("model_initialization"):
            model, schema, parameters = load_model(args, config)
        frozen, adapters = tensor_hashes(model, False), tensor_hashes(model, True)
        loaded_policy = verify_loaded_policy(model, args.resume_checkpoint or original_checkpoint, args.variant)
        tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        dataset, counts = prepare_dataset(args.train_data, tokenizer, config["training"]["max_prompt_length"])
        dataset_order = state_digest(list(dataset["id"]))
        require(prior is None or prior["dataset_ordered_ids_sha256"] == dataset_order, "Resume dataset/order changed")
        set_seed(config["training"]["seed"])
        settings = training_args(args, config, schedule)
        settings.save_strategy = type(settings.save_strategy)("steps")
        settings.save_steps, settings.save_total_limit, settings.save_only_model = 1, 2, False
        require(not settings.ignore_data_skip and settings.max_steps == 2, "Stock data skip and target2 are required")
        monitor = PhaseMonitor(args.output, args.seconds_limit, args.phase)
        trainer_type = NativeLoRATrainer if args.variant == "lora" else NativeGeoRATrainer
        trainer = trainer_type(model=model, processing_class=tokenizer, train_dataset=dataset, args=settings,
            reward_funcs=make_audited_gsm8k_reward(args.output / "reward_audit.jsonl", num_generations=8),
            callbacks=[monitor], **({"base_model_path": base} if args.variant == "geora_mixed" else {}))
        trainer.performance, trainer.schedule, trainer.audit_output = recorder, schedule, args.output
        trainer.probability_records = []
        trainer.resume_audit_dir, trainer.resume_audit_variant = args.output / "resume", args.variant
        write_json(args.output / "resolved_training_options.json", {
            "fused_lm_head": getattr(trainer, "fused_lm_head", None),
            "use_liger_loss": getattr(settings, "use_liger_loss", None),
            "model_attention_implementation": getattr(model.config, "_attn_implementation", None),
            "bf16_autocast": settings.bf16, "fp16_autocast": settings.fp16,
            "micro_batch_size": settings.per_device_train_batch_size,
            "gradient_accumulation_steps": settings.gradient_accumulation_steps,
            "generation_batch_size": settings.generation_batch_size,
            "steps_per_generation": settings.steps_per_generation,
            "scope": "Resolved configuration flags; not a per-kernel precision profile"})
        capacity = None
        if args.phase == 1:
            # The shared probe includes the stock loss's entropy-return path.
            # It must remain outside actual rollouts/optimizer state and may
            # warm the allocator; this diagnostic makes no throughput claim.
            with recorder.stage("capacity_probe"):
                capacity = capacity_probe(trainer, args, config, output_path=args.output / "capacity_result.json")
            require(isinstance(capacity, dict) and capacity.get("passed") is True
                    and capacity.get("parameter_updates") == 0
                    and capacity.get("trainable_weights_unchanged") is True,
                    "Full-length capacity must pass without a parameter update")
            require(tensor_hashes(model, False) == frozen and tensor_hashes(model, True) == adapters,
                    "Capacity probe changed a frozen buffer/parameter or adapter")
            require(all(parameter.grad is None for parameter in model.parameters()),
                    "Capacity probe left gradients behind")
            set_seed(config["training"]["seed"])

        def probability_check(session, current):
            with recorder.stage("fixed_three_way_probability", optimizer_step=current):
                report = score_three_way(model, schema, fixed_rows, session.client,
                    device=trainer.accelerator.device, audit_path=session.audit_dir / f"three_way_v{current}.json",
                    version=current, version_getter=session._version)
            return {key: value for key, value in report.items() if key != "rows"}

        start, target = args.phase - 1, args.phase
        publications, generations = schedule.versions(start, target)
        session = install_native_sync(trainer, expected_schema=schema, audit_dir=args.output / "sync", fixed_rows=fixed_rows,
            expected_sync_versions=publications, expected_generation_versions=generations,
            fixed_token_checker=probability_check, base_sampling_seed=config["training"]["seed"],
            responses_per_rollout=schedule.generation_batch_size, num_generations=8,
            rollouts_per_update=schedule.rollouts_per_update)
        original_generate, original_sync = session.original_generate, session.sync_weights

        def timed_generate(*positional, **keywords):
            with recorder.stage("vllm_rollout", optimizer_step=int(trainer.state.global_step)):
                return original_generate(*positional, **keywords)

        def timed_sync():
            with recorder.stage("policy_sync", optimizer_step=int(trainer.state.global_step)):
                return original_sync()

        session.original_generate, trainer.vllm_generation.sync_weights = timed_generate, timed_sync
        recorder.set_stage("training")
        trained = trainer.train(resume_from_checkpoint=str(args.resume_checkpoint) if args.resume_checkpoint else None)
        require(trainer.state.global_step == target and len(monitor.records) == 1,
                "Each process must complete exactly its one requested optimizer update")
        questions = actual_questions(trainer.probability_records, schedule, start)
        if args.phase == 2:
            require(not set(questions["unique_question_ids"]) & set(prior["actual_questions"]["unique_question_ids"]),
                    "Restored data loader repeated phase1 questions")
        recorder.set_stage("final_policy_checks")
        session.sync_weights()
        synchronized = session.verify_final()
        require(tensor_hashes(model, False) == frozen, "Frozen parameters/buffers changed")
        changed = sum(value != adapters[name] for name, value in tensor_hashes(model, True).items())
        zero_advantages = all(row["nonzero_advantage_responses"] == 0 for row in trainer.probability_records)
        require(changed > 0 or zero_advantages, "Nonzero advantages produced no adapter changes")
        checkpoint = args.output / f"training/checkpoint-{target}"
        saved_inventory = checkpoint_inventory(checkpoint, args.variant)
        require(saved_inventory["global_step"] == target, "Saved checkpoint has the wrong global step")
        saved_policy = verify_loaded_policy(model, checkpoint, args.variant)
        restored = trainer.resume_audit_summary() if args.phase == 2 else None
        require(args.phase != 2 or (isinstance(restored, dict) and restored.get("passed") is True),
                "The five stock restoration checks did not all pass")
        v1_exact = None
        if args.phase == 2:
            previous, current = rows(prior_root / "sync/sync.jsonl")[-1], rows(args.output / "sync/sync.jsonl")[0]
            for field in ("canonical_tensor_sha256", "resident_checksums"):
                require(previous[field] == current[field], f"Restored v1 changed: {field}")
            require(checkpoint_inventory(args.resume_checkpoint, args.variant) == resume_inventory, "Resume input was mutated")
            v1_exact = True
        require(checkpoint_inventory(original_checkpoint, args.variant) == original_inventory, "Frozen checkpoint25 was mutated")
        result = {"completed": True, "mode": args.mode, "phase": args.phase, "variant": args.variant,
            "schedule": schedule.as_dict(), "max_steps_for_both_phases": 2,
            "optimizer_updates_this_process": 1, "total_global_step": target,
            "training_rollouts": schedule.rollouts_per_update, "training_responses": 128,
            "actual_completion_tokens": sum(row["active_tokens"] for row in trainer.probability_records),
            "stock_importance_sampling_verified": True, "frozen_parameters_and_buffers_unchanged": True,
            "finite_gradients_each_update": True, "changed_adapter_tensors": changed,
            "all_measured_advantages_zero": zero_advantages, "parameters": parameters,
            "optimizer_at_train_begin": monitor.optimizer_initialization, "updates": monitor.records,
            "capacity_probe_performed": args.phase == 1, "capacity_result": capacity,
            "loaded_policy_matches": loaded_policy, "saved_policy_matches": saved_policy,
            "saved_inventory": saved_inventory, "final_checkpoint": str(checkpoint.resolve()),
            "restore_audit": restored, "final_server_verification": synchronized,
            "actual_questions": questions, "dataset_ordered_ids_sha256": dataset_order, "data_counts": counts,
            "restored_v1_canonical_and_resident_exact": v1_exact,
            "v1_three_way_raw_paths": ([str(prior_root / "sync/three_way_v1.json"), str(args.output / "sync/three_way_v1.json")] if args.phase == 2 else None),
            "numerical_equivalence_claim": False, "formal400_training": False,
            "sampling_replay_bitwise_guaranteed": False, "performance_claim": False,
            "total_learner_wall_seconds": time.monotonic() - started, "train_metrics": trained.metrics}
        write_json(args.output / "result.json", result)
        recorder.set_stage("done")
        print(json.dumps({key: result[key] for key in ("completed", "phase", "variant", "schedule", "total_global_step", "training_responses")}), flush=True)
    except BaseException as error:
        write_json(args.output / "failure.json", {"completed": False, "error": repr(error), "seconds": time.monotonic() - started})
        raise
    finally:
        if monitor:
            monitor.restore_handlers()


if __name__ == "__main__":
    main()
