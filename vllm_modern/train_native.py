"""Native single-learner GRPO runner and bounded performance/capacity probes.

The frozen two-phase acceptance entry remains unchanged. Benchmark cases start
independently from the same checkpoint25 policy with fresh optimizer state.
Formal mode initializes from Base and enforces an immutable resume protocol.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import signal
import socket
import sys
import time

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from trl import GRPOConfig

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "frozen_training"))
from answer_scoring import GSM8K_REWARD_VERSION
from geora_paper_utils import (ExperimentGRPOTrainer, GeoRAGRPOTrainer,
                               install_geora_layers, linear_target_names, load_geora_layers)
from train_paper_grpo import prepare_dataset, sha256_file, validate_parameters
from checkpoint_audit import ResumeAuditMixin, checkpoint_inventory, verify_loaded_policy
from native_sync import install_native_sync
from reward_audit import make_audited_gsm8k_reward
from three_way_probability import score_three_way, validate_fixed_rows
from training_schedule import TrainingSchedule, resolve_schedule


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def tensor_hashes(model, trainable):
    import hashlib
    values = {name: value for name, value in model.named_parameters()
              if value.requires_grad == trainable}
    if not trainable:
        values.update(dict(model.named_buffers()))
    return {name: hashlib.sha256(value.detach().cpu().contiguous().reshape(-1)
                                .view(torch.uint8).numpy().tobytes()).hexdigest()
            for name, value in values.items()}


class PerformanceRecorder:
    def __init__(self, output):
        self.output = output
        self.current = "initialization"
        self.set_stage(self.current)

    def set_stage(self, stage):
        self.current = stage
        write_json(self.output / "stage.json", {"stage": stage, "epoch_time": time.time(),
                                              "run_directory": str(self.output)})

    @contextmanager
    def stage(self, stage, *, synchronize=True, **metadata):
        previous = self.current
        if synchronize:
            torch.cuda.synchronize()
        self.set_stage(stage)
        started = time.monotonic()
        try:
            yield
        finally:
            if synchronize:
                torch.cuda.synchronize()
            append_json(self.output / "performance.jsonl", {
                "stage": stage, "seconds": time.monotonic() - started,
                "epoch_time_end": time.time(), "nested_in": previous,
                "timing_semantics": "Inclusive wall time; nested stages must not be summed",
                **metadata})
            self.set_stage(previous)


class ProbabilityAndTimingMixin:
    def _get_per_token_logps_and_entropies(self, *args, **kwargs):
        stage = "learner_grad_forward" if torch.is_grad_enabled() else "learner_old_logprob"
        with self.performance.stage(stage, optimizer_step=int(self.state.global_step)):
            return super()._get_per_token_logps_and_entropies(*args, **kwargs)

    def training_step(self, *args, **kwargs):
        with self.performance.stage("training_microstep", optimizer_step=int(self.state.global_step)):
            return super().training_step(*args, **kwargs)

    def _generate_and_score_completions(self, inputs):
        with self.performance.stage("generate_and_score", optimizer_step=int(self.state.global_step)):
            result = super()._generate_and_score_completions(inputs)
        mask = result["completion_mask"].bool()
        old = result["old_per_token_logps"]
        behavior = result["sampling_per_token_logps"]
        ratio = result["importance_sampling_ratio"]
        if mask.shape[0] != self.schedule.generation_batch_size or not mask.any():
            raise ValueError("Unexpected or empty generation batch")
        for value in (old, behavior, ratio):
            if value.shape != mask.shape or not torch.isfinite(value[mask]).all():
                raise ValueError("Nonfinite/misaligned active probabilities")
        if not torch.equal((old - behavior).exp().clamp(max=2.0)[mask], ratio[mask]):
            raise ValueError("Stock token_truncate cap2 differs from its declared contract")
        if not torch.isfinite(result["advantages"]).all():
            raise ValueError("Nonfinite GRPO advantage")
        difference = (old - behavior)[mask].detach()
        record = {"optimizer_step": int(self.state.global_step),
                  "responses": int(mask.shape[0]), "question_ids": [str(row["id"]) for row in inputs],
                  "active_tokens": int(mask.sum()), "padded_completion_positions": mask.numel(),
                  "padding_fraction": 1 - mask.sum().item() / mask.numel(),
                  "mean_abs_log_ratio": difference.abs().mean().item(),
                  "max_abs_log_ratio": difference.abs().max().item(),
                  "fraction_ratio_above_2": (difference > math.log(2)).float().mean().item(),
                  "nonzero_advantage_responses": int((result["advantages"].abs() > 0).sum()),
                  "stock_token_truncate_verified": True,
                  "old_learner_logprobs": [row[active].detach().cpu().tolist() for row, active in zip(old, mask)],
                  "behavior_logprobs": [row[active].detach().cpu().tolist() for row, active in zip(behavior, mask)]}
        append_json(self.audit_output / "training_probabilities.jsonl", record)
        self.probability_records.append(record)
        return result


class NativeLoRATrainer(ResumeAuditMixin, ProbabilityAndTimingMixin, ExperimentGRPOTrainer):
    pass


class NativeGeoRATrainer(ResumeAuditMixin, ProbabilityAndTimingMixin, GeoRAGRPOTrainer):
    pass


class NativeMonitor(TrainerCallback):
    def __init__(self, output, mode, seconds_limit):
        self.output, self.mode, self.seconds_limit = output, mode, seconds_limit
        self.records = []
        self.started = time.monotonic()
        self.stop_signal = None
        self.handlers = {}

    def signal_stop(self, number, frame):
        self.stop_signal = signal.Signals(number).name

    def on_train_begin(self, args, state, control, **kwargs):
        for number in (signal.SIGTERM, signal.SIGUSR1):
            self.handlers[number] = signal.signal(number, self.signal_stop)

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        torch.cuda.synchronize()
        self.step_started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not grads or not torch.stack([torch.isfinite(g).all() for g in grads]).all().item():
            raise ValueError("Missing/nonfinite gradient")
        self.grad_norm = torch.stack([g.detach().float().square().sum() for g in grads]).sum().sqrt().item()

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        row = {"optimizer_step": int(state.global_step),
               "update_wall_seconds_including_sync_and_diagnostics": time.monotonic() - self.step_started,
               "post_clip_gradient_norm": self.grad_norm,
               "zero_gradient": self.grad_norm == 0,
               "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
               "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
        self.records.append(row)
        append_json(self.output / "updates.jsonl", row)
        print("[update] " + json.dumps(row), flush=True)
        if self.stop_signal or time.monotonic() - self.started >= self.seconds_limit:
            control.should_training_stop = True
            control.should_save = self.mode == "formal"
        if state.global_step >= args.max_steps and self.mode == "formal":
            control.should_save = True

    def restore_handlers(self):
        for number, handler in self.handlers.items():
            signal.signal(number, handler)


def training_args(args, config, schedule):
    t = config["training"]
    return GRPOConfig(
        output_dir=str(args.output / "training"),
        per_device_train_batch_size=schedule.micro_batch_size,
        gradient_accumulation_steps=schedule.gradient_accumulation_steps,
        generation_batch_size=schedule.generation_batch_size,
        num_generations=schedule.num_generations, num_iterations=1,
        max_completion_length=t["max_completion_length"], temperature=1.0, top_p=1.0, top_k=0,
        repetition_penalty=1.0, learning_rate=t["learning_rate"], optim="adamw_torch",
        weight_decay=0.0, lr_scheduler_type="constant", max_grad_norm=1.0, beta=0.0,
        loss_type="grpo", scale_rewards="group", epsilon=0.2, mask_truncated_completions=False,
        bf16=False, fp16=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, dataloader_pin_memory=True,
        remove_unused_columns=False, pad_to_multiple_of=64, torch_empty_cache_steps=None,
        use_vllm=True, vllm_mode="server", vllm_server_base_url=args.server_url,
        vllm_server_timeout=900.0, vllm_group_port=args.group_port,
        vllm_importance_sampling_correction=True, vllm_importance_sampling_mode="token_truncate",
        vllm_importance_sampling_clip_min=None, vllm_importance_sampling_clip_max=2.0,
        importance_sampling_level="token", max_steps=args.max_steps,
        eval_strategy="no", eval_on_start=False,
        save_strategy="steps" if args.mode == "formal" else "no",
        save_steps=config["checkpointing"]["save_every_optimizer_updates"], save_total_limit=2,
        save_only_model=False, logging_steps=1, report_to="none",
        seed=t["seed"], data_seed=t["data_seed"],
    )


def load_model(args, config):
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16, local_files_only=True)
    schema = {name: (tuple(value.shape), value.dtype) for name, value in model.state_dict().items()}
    targets = linear_target_names(model)
    t = config["training"]
    checkpoint = args.resume_checkpoint or args.initialization_checkpoint
    if checkpoint:
        if args.variant == "lora":
            model = PeftModel.from_pretrained(model, checkpoint, local_files_only=True, is_trainable=True)
        else:
            load_geora_layers(model, checkpoint, install=True, expected_base_model_path=args.base_model)
        verify_loaded_policy(model, checkpoint, args.variant)
    elif args.variant == "lora":
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", target_modules="all-linear",
                              r=t["rank"], lora_alpha=t["alpha"], lora_dropout=0.0, bias="none"))
    else:
        # Exact CPU FP32 SVD, before transferring the initialized policy to the GCD.
        install_geora_layers(model, t["rank"], t["rho"], t["alpha"])
    if args.variant == "lora":
        names = [name for name, layer in model.named_modules() if hasattr(layer, "lora_A")]
        targets = [next(name for name in names if name.endswith("." + target)) for target in targets]
    parameters = validate_parameters(model, targets, t["rank"])
    if parameters["target_layer_count"] != 196 or parameters["trainable_tensors"] != 392:
        raise ValueError("Unexpected Qwen adapter coverage")
    model.config.use_cache = False
    return model.to("cuda"), schema, parameters


def capacity_probe(trainer, args, config, *, output_path=None):
    """Fixed full-length synthetic tokens; no optimizer update or publication.

    Uses the pinned Trainer's actual differentiable token-logprob method with
    gradient checkpointing. This probes capacity, not an accuracy or speed score.
    """
    micro = trainer.schedule.micro_batch_size
    prompt = config["training"]["max_prompt_length"]
    completion = config["training"]["max_completion_length"]
    # pad_to_multiple_of64 is also used by normal training.
    length = ((prompt + completion + 63) // 64) * 64
    ids = torch.full((micro, length), 42, dtype=torch.long, device="cuda")
    attention = torch.ones_like(ids)
    model = trainer.model
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    torch.cuda.reset_peak_memory_stats()
    before = tensor_hashes(model, True)
    started = time.monotonic()
    passed, error, optimizer_state_reserve = False, None, []
    try:
        # AdamW allocates two FP32 moments per trainable parameter. Reserve the
        # same bytes without taking an optimizer step or changing any weight.
        optimizer_state_reserve = [torch.zeros_like(p) for p in model.parameters()
                                   if p.requires_grad for _ in range(2)]
        # Match the actual GRPO loss path, which also computes entropy for
        # diagnostics even when no entropy bonus is present in the objective.
        logps, entropies, _ = trainer._get_per_token_logps_and_entropies(
            model, ids, attention, completion, batch_size=micro, compute_entropy=True)
        if logps.shape != (micro, completion) or not torch.isfinite(logps).all():
            raise ValueError("Invalid full-length capacity log probabilities")
        if entropies is None or entropies.shape != logps.shape or not torch.isfinite(entropies).all():
            raise ValueError("Invalid full-length diagnostic entropy")
        (-logps.mean()).backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not grads or not torch.stack([torch.isfinite(g).all() for g in grads]).all().item():
            raise ValueError("Invalid full-length capacity gradients")
        torch.cuda.synchronize()
        passed = True
    except torch.OutOfMemoryError as exc:
        error = repr(exc)
    finally:
        peak_allocated = torch.cuda.max_memory_allocated() / 2**30
        peak_reserved = torch.cuda.max_memory_reserved() / 2**30
        model.zero_grad(set_to_none=True)
    unchanged = tensor_hashes(model, True) == before
    if not unchanged:
        raise ValueError("Capacity-only backward changed policy weights")
    result = {"completed": True, "passed": passed, "mode": "capacity", "variant": args.variant,
              "micro_batch_size": micro, "prompt_length": prompt, "completion_length": completion,
              "padded_total_length": length, "seconds": time.monotonic() - started,
              "peak_allocated_gib": peak_allocated, "peak_reserved_gib": peak_reserved,
              "parameter_updates": 0, "trainable_weights_unchanged": unchanged,
              "compute_entropy": True,
              "optimizer_moment_bytes_reserved": sum(p.numel() * p.element_size() for p in optimizer_state_reserve),
              "error": error, "synthetic_workload": True,
              "scope": "Actual TRL full-length differentiable logprob and diagnostic entropy forward/backward with AdamW moment-byte reserve; no optimizer step and not a full128-response update"}
    write_json(output_path or args.output / "result.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("benchmark", "capacity", "formal"), required=True)
    parser.add_argument("--variant", choices=("lora", "geora_mixed"), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "formal_config.json")
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--group-port", type=int, required=True)
    parser.add_argument("--initialization-checkpoint", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--generation-batch-size", type=int)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seconds-limit", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.max_steps is None:
        args.max_steps = config["training"]["max_steps"] if args.mode == "formal" else 3
    if args.seconds_limit is None:
        if args.mode == "formal":
            parser.error("Formal training requires an explicit soft wall limit bound to its allocation budget")
        args.seconds_limit = 1200
    if args.max_steps <= 0 or args.seconds_limit <= 0:
        parser.error("Steps and wall limit must be positive")
    if args.mode != "formal" and args.max_steps > 3:
        parser.error("Performance probes are bounded to three optimizer updates")
    if args.mode == "formal" and (args.initialization_checkpoint or not config["launchable"]):
        parser.error("Formal launch requires an approved final manifest and fresh Base initialization")
    if args.mode == "formal" and args.max_steps != config["training"]["max_steps"]:
        parser.error("Formal target steps must match the frozen configuration")
    if args.mode == "benchmark" and args.initialization_checkpoint is None:
        parser.error("Benchmark cases require an explicitly frozen checkpoint25 initialization")
    if args.initialization_checkpoint and args.resume_checkpoint:
        parser.error("Initialization and resume are distinct mutually exclusive operations")
    if args.resume_checkpoint and args.mode != "formal":
        parser.error("Performance cases must start independently with fresh optimizer state")
    if args.output.exists() and not args.resume_checkpoint:
        raise FileExistsError("Use a fresh output directory")
    schedule = resolve_schedule(config["training"], micro=args.micro_batch_size,
                                generation=args.generation_batch_size, require_frozen=args.mode == "formal")
    args.micro_batch_size, args.generation_batch_size = schedule.micro_batch_size, schedule.generation_batch_size
    if (version("trl"), version("vllm")) != ("1.17.0.dev0", "0.32.0+rocm723"):
        raise RuntimeError("Use the accepted immutable uv runtime")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one learner GCD must be visible")
    torch.set_num_threads(4)
    for name, digest in config["model"]["expected_base_files_sha256"].items():
        if sha256_file(args.base_model / name) != digest:
            raise ValueError(f"Base input drift: {name}")
    if sha256_file(args.train_data) != config["data"]["train_file_sha256"]:
        raise ValueError("Training split drift")
    if config["training"]["reward"] != GSM8K_REWARD_VERSION:
        raise ValueError("Reward contract drift")
    initialization = None
    if args.initialization_checkpoint:
        initialization = checkpoint_inventory(args.initialization_checkpoint, args.variant)
        if initialization["global_step"] != 25:
            raise ValueError("Benchmark initialization must be the frozen checkpoint25")
    args.output.mkdir(parents=True, exist_ok=bool(args.resume_checkpoint))
    started = time.monotonic()
    recorder = PerformanceRecorder(args.output)
    protocol = {"mode": args.mode, "variant": args.variant, "schedule": schedule.as_dict(),
                "max_optimizer_steps": args.max_steps,
                "scientific_configuration_sha256": sha256_file(args.config),
                "train_data_sha256": sha256_file(args.train_data),
                "base_revision": config["model"]["revision"],
                "base_files_sha256": config["model"]["expected_base_files_sha256"],
                "initialization_checkpoint": initialization,
                "sampling_seed_policy": "seed + G*(optimizer_step*rollouts_per_update + rollout_index)",
                "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(ROOT.rglob("*.py"))
                                  if "__pycache__" not in p.parts},
                "uv_lock_sha256": sha256_file(ROOT / "uv.lock"),
                "reward": GSM8K_REWARD_VERSION, "precision": config["training"]["precision_contract"],
                "primary_eval_backend": config["evaluation"]["primary_generation_backend"]}
    protocol_path = args.output / "run_protocol.json"
    if args.resume_checkpoint:
        if not protocol_path.is_file() or json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Immutable formal resume protocol changed")
    else:
        write_json(protocol_path, protocol)
    write_json(args.output / "invocation.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    set_seed(config["training"]["seed"])
    try:
        with recorder.stage("model_initialization"):
            model, schema, parameters = load_model(args, config)
        frozen = tensor_hashes(model, False)
        adapters = tensor_hashes(model, True)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        dataset, counts = prepare_dataset(args.train_data, tokenizer, config["training"]["max_prompt_length"])
        if args.mode == "benchmark":
            # Every case sees this same fixed48-question set in three updates;
            # generation grouping may change order and generated token counts.
            dataset = dataset.select(range(schedule.as_dict()["questions_per_update"] * args.max_steps))
            write_json(args.output / "benchmark_questions.json", {"ids": list(dataset["id"]),
                "selection": "First questions_per_update * max_steps of the already frozen training split",
                "tokens_identical_across_schedules_claim": False})
        set_seed(config["training"]["seed"])
        monitor = NativeMonitor(args.output, args.mode, args.seconds_limit)
        trainer_type = NativeLoRATrainer if args.variant == "lora" else NativeGeoRATrainer
        trainer = trainer_type(model=model, processing_class=tokenizer,
            args=training_args(args, config, schedule), train_dataset=dataset,
            reward_funcs=make_audited_gsm8k_reward(args.output / "reward_audit.jsonl", num_generations=schedule.num_generations),
            callbacks=[monitor], **({"base_model_path": args.base_model} if args.variant == "geora_mixed" else {}))
        trainer.performance = recorder
        trainer.schedule = schedule
        trainer.audit_output = args.output
        trainer.probability_records = []
        trainer.resume_audit_dir = args.output / "resume"
        trainer.resume_audit_variant = args.variant
        if args.mode == "capacity":
            capacity_probe(trainer, args, config)
            return
        start_step = (json.loads((args.resume_checkpoint / "trainer_state.json").read_text())["global_step"]
                      if args.resume_checkpoint else 0)
        publications, generations = schedule.versions(start_step, args.max_steps)
        fixed_rows = json.loads((ROOT / "fixtures/probability_controls.json").read_text())
        validate_fixed_rows(fixed_rows)

        def probability_check(session, current):
            # Keep identical overhead across benchmark cases; all resident hashes
            # remain mandatory at every publication. Full three-way at initial
            # and final policies, no expensive fixed replay in the timed middle.
            if current not in (start_step, args.max_steps):
                return {"status": "not_scheduled_at_this_version", "scope": "Full resident hash still checked"}
            with recorder.stage("fixed_three_way_probability", optimizer_step=current):
                report = score_three_way(model, schema, fixed_rows, session.client,
                    device=trainer.accelerator.device, audit_path=session.audit_dir / f"three_way_v{current}.json",
                    version=current, version_getter=session._version)
            return {key: value for key, value in report.items() if key != "rows"}

        session = install_native_sync(trainer, expected_schema=schema, audit_dir=args.output / "sync",
            fixed_rows=fixed_rows, expected_sync_versions=publications, expected_generation_versions=generations,
            fixed_token_checker=probability_check, base_sampling_seed=config["training"]["seed"],
            responses_per_rollout=schedule.generation_batch_size, num_generations=schedule.num_generations,
            rollouts_per_update=schedule.rollouts_per_update)
        original_generate = session.original_generate

        def timed_generate(*positional, **keywords):
            with recorder.stage("vllm_rollout", optimizer_step=int(trainer.state.global_step)):
                return original_generate(*positional, **keywords)

        session.original_generate = timed_generate
        original_sync = session.sync_weights

        def timed_sync():
            with recorder.stage("policy_sync", optimizer_step=int(trainer.state.global_step)):
                return original_sync()

        trainer.vllm_generation.sync_weights = timed_sync
        training_started = time.monotonic()
        recorder.set_stage("training")
        try:
            train_result = trainer.train(resume_from_checkpoint=str(args.resume_checkpoint) if args.resume_checkpoint else None)
        finally:
            monitor.restore_handlers()
        torch.cuda.synchronize()
        training_seconds = time.monotonic() - training_started
        final_step = int(trainer.state.global_step)
        if args.mode == "benchmark" and final_step != args.max_steps:
            raise ValueError("Bounded performance case did not complete all requested updates")
        # A wall-limited formal process may finish on an earlier complete boundary.
        if final_step != args.max_steps:
            session.expected_sync_versions = tuple(range(start_step, final_step + 1))
            session.expected_generation_versions = tuple(step for step in range(start_step, final_step)
                                                         for _ in range(schedule.rollouts_per_update))
        recorder.set_stage("final_policy_checks")
        session.sync_weights()
        sync_result = session.verify_final()
        if tensor_hashes(model, False) != frozen:
            raise ValueError("Frozen policy parameters/buffers changed")
        changed = sum(value != adapters[name] for name, value in tensor_hashes(model, True).items())
        if changed == 0 and any(row["nonzero_advantage_responses"] > 0 for row in trainer.probability_records):
            raise ValueError("Nonzero advantages produced no adapter parameter change")
        active_tokens = sum(row["active_tokens"] for row in trainer.probability_records)
        saved_inventory, saved_policy = None, None
        if args.mode == "formal":
            checkpoint = args.output / f"training/checkpoint-{final_step}"
            with recorder.stage("final_checkpoint_verification"):
                saved_inventory = checkpoint_inventory(checkpoint, args.variant)
                if saved_inventory["global_step"] != final_step:
                    raise ValueError("Final saved checkpoint has the wrong optimizer step")
                saved_policy = verify_loaded_policy(model, checkpoint, args.variant)
        result = {"completed": True, "mode": args.mode, "variant": args.variant, "schedule": schedule.as_dict(),
                  "global_step": final_step, "new_optimizer_updates": final_step - start_step,
                  "training_seconds_including_initial_audit": training_seconds,
                  "process_seconds": time.monotonic() - started, "updates": monitor.records,
                  "responses": sum(row["responses"] for row in trainer.probability_records),
                  "actual_completion_tokens": active_tokens, "changed_adapter_tensors": changed,
                  "all_measured_advantages_zero": all(row["nonzero_advantage_responses"] == 0
                                                       for row in trainer.probability_records),
                  "frozen_policy_unchanged": True, "parameters": parameters,
                  "sync_summary": sync_result, "data_counts": counts,
                  "warmup_policy": "First optimizer update is cold; later updates reported separately",
                  "accuracy_claim": False, "bitwise_batch_schedule_equivalence_claim": False,
                  "checkpoint_saved": saved_inventory is not None,
                  "saved_inventory": saved_inventory, "saved_policy_matches": saved_policy,
                  "train_metrics": train_result.metrics,
                  "resume_audit": trainer.resume_audit_summary() if args.resume_checkpoint else None}
        write_json(args.output / "result.json", result)
        recorder.set_stage("done")
        print(json.dumps({key: result[key] for key in ("completed", "mode", "variant", "schedule", "actual_completion_tokens", "changed_adapter_tensors")}), flush=True)
    except BaseException as error:
        write_json(args.output / "failure.json", {"error": repr(error), "seconds": time.monotonic() - started,
                                                "hostname": socket.gethostname()})
        raise


if __name__ == "__main__":
    main()
