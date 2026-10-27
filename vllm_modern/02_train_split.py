"""Two real GRPO updates with a server/learner restart at checkpoint1.

Checkpoint-25 initializes the policy only. Optimizer, scheduler, data order and
RNG are fresh, so this is policy version 0 -> 1 -> 2, not resumed steps 26/27.
The workload retains the frozen 4096-token comparison's batch accounting.
Phase 2 actually restores learner state; request seeds make server restart explicit.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from trl import GRPOConfig

CODE_ROOT = Path(__file__).resolve().parent / "frozen_training"
sys.path.insert(0, str(CODE_ROOT))

from answer_scoring import GSM8K_REWARD_VERSION
from geora_paper_utils import ExperimentGRPOTrainer, GeoRAGRPOTrainer
from geora_paper_utils import grpo_boxed_reward, load_geora_layers
from train_paper_grpo import prepare_dataset
from native_sync import install_native_sync
from three_way_probability import score_three_way, validate_fixed_rows
from checkpoint_audit import ResumeAuditMixin, checkpoint_inventory, verify_loaded_policy


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append_json(path: Path, payload: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(payload, allow_nan=False) + "\n")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def trainable_hashes(model) -> dict[str, str]:
    return {
        name: tensor_hash(value)
        for name, value in model.named_parameters()
        if value.requires_grad
    }


def frozen_hashes(model) -> dict[str, str]:
    frozen = {
        name: tensor_hash(value)
        for name, value in model.named_parameters()
        if not value.requires_grad
    }
    frozen.update({name: tensor_hash(value) for name, value in model.named_buffers()})
    return frozen


def verify_inputs(args):
    manifest = json.loads(args.execution_manifest.read_text())
    base = Path(manifest["base_model_path"])
    variant = manifest["variants"][args.variant]
    checkpoint = Path(variant["checkpoint_path"])
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    if state["global_step"] != 25:
        raise ValueError("The initialization checkpoint must be frozen checkpoint-25")
    for root, expected in (
        (base, manifest["base_files_sha256"]),
        (checkpoint, variant["checkpoint_files_sha256"]),
    ):
        for name, expected_hash in expected.items():
            if file_hash(root / name) != expected_hash:
                raise ValueError(f"Frozen input changed: {root / name}")
    protocol_path = Path(variant["training_protocol"]["path"])
    if file_hash(protocol_path) != variant["training_protocol"]["sha256"]:
        raise ValueError("The frozen training protocol changed")
    protocol = json.loads(protocol_path.read_text())
    if file_hash(args.train_data) != protocol["train_file_sha256"]:
        raise ValueError("The training data differs from the frozen split")
    if protocol["settings"]["reward"] != GSM8K_REWARD_VERSION:
        raise ValueError("The GSM8K reward contract differs from the frozen run")
    for name in ("geora_paper_utils.py", "train_paper_grpo.py", "answer_scoring.py"):
        if file_hash(CODE_ROOT / name) != protocol["source_sha256"][name]:
            raise ValueError(f"The frozen training helper changed: {name}")
    for protected in (base.resolve(), checkpoint.resolve()):
        if args.output.resolve() == protected or protected in args.output.resolve().parents:
            raise ValueError("The new output must be outside the frozen model/checkpoint")
    return base, checkpoint, protocol


def make_training_args(output: Path, server_url: str, group_port: int) -> GRPOConfig:
    """128 responses/update = microbatch 4 x 32 accumulated microsteps.

    Generation uses 32 responses/rollout = 4 questions x 8 samples. Four such
    rollouts belong to each optimizer update; the server syncs once per update.
    """
    return GRPOConfig(
        output_dir=str(output),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=32,
        generation_batch_size=32,
        num_generations=8,
        num_iterations=1,
        max_completion_length=4096,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
        learning_rate=1e-6,
        optim="adamw_torch",
        weight_decay=0.0,
        lr_scheduler_type="constant",
        max_grad_norm=1.0,
        beta=0.0,
        loss_type="grpo",
        scale_rewards="group",
        epsilon=0.2,
        mask_truncated_completions=False,
        bf16=False,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        pad_to_multiple_of=64,
        torch_empty_cache_steps=None,
        use_vllm=True,
        vllm_mode="server",
        vllm_server_base_url=server_url,
        vllm_server_timeout=900.0,
        vllm_group_port=group_port,
        vllm_importance_sampling_correction=True,
        vllm_importance_sampling_mode="token_truncate",
        vllm_importance_sampling_clip_min=None,
        vllm_importance_sampling_clip_max=2.0,
        importance_sampling_level="token",
        max_steps=2,
        eval_strategy="no",
        eval_on_start=False,
        save_strategy="steps",
        save_steps=1,
        save_total_limit=2,
        save_only_model=False,
        logging_steps=1,
        report_to="none",
        seed=42,
        data_seed=42,
    )


class ProbabilityAuditMixin:
    """Observe the pinned TRL result before backward or NaN masking can hide it."""

    def _generate_and_score_completions(self, inputs):
        result = super()._generate_and_score_completions(inputs)
        mask = result["completion_mask"].bool()
        old = result["old_per_token_logps"]
        behavior = result["sampling_per_token_logps"]
        ratio = result["importance_sampling_ratio"]
        if mask.shape[0] != 32 or not mask.any():
            raise ValueError("Each rollout must contain 32 nonempty responses")
        for value in (old, behavior, ratio):
            if value.shape != mask.shape or not torch.isfinite(value[mask]).all():
                raise ValueError("Missing/nonfinite active policy or behavior logprob")
        expected = (old - behavior).exp().clamp(max=2.0)
        if not torch.equal(expected[mask], ratio[mask]):
            raise ValueError("TRL correction differs from token_truncate with max=2")
        if not torch.isfinite(result["advantages"]).all():
            raise ValueError("Nonfinite group advantages")
        log_ratio = (old - behavior)[mask].detach()
        record = {
            "policy_version": int(self.state.global_step),
            "rollout_index": len(self.probability_records),
            "question_ids": [str(row.get("id")) for row in inputs],
            "responses": int(mask.shape[0]),
            "active_tokens": int(mask.sum().item()),
            "finite_active_logprobs": True,
            "stock_token_truncate_verified": True,
            "mean_abs_log_ratio": log_ratio.abs().mean().item(),
            "max_abs_log_ratio": log_ratio.abs().max().item(),
            "fraction_ratio_above_2": (log_ratio > math.log(2.0)).float().mean().item(),
            "advantages": result["advantages"].detach().cpu().tolist(),
        }
        self.probability_records.append(record)
        append_json(self.native_audit_dir / "training_probabilities.jsonl", record)
        return result


class ModernGeoRATrainer(ResumeAuditMixin, ProbabilityAuditMixin, GeoRAGRPOTrainer):
    pass


class ModernLoRATrainer(ResumeAuditMixin, ProbabilityAuditMixin, ExperimentGRPOTrainer):
    pass


class UpdateAudit(TrainerCallback):
    def __init__(self, output: Path, stop_at: int):
        self.output = output
        self.stop_at = stop_at
        self.records = []

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        torch.cuda.synchronize()
        self.started = time.monotonic()
        self.before = trainable_hashes(model)

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        squared_norm = 0.0
        gradient_tensors = 0
        for value in model.parameters():
            if value.grad is not None:
                if not torch.isfinite(value.grad).all():
                    raise ValueError("Nonfinite gradient before the optimizer update")
                squared_norm += value.grad.detach().float().square().sum().item()
                gradient_tensors += 1
        if gradient_tensors == 0 or squared_norm == 0.0:
            raise ValueError("This update has no nonzero training gradient")
        self.gradient_norm = math.sqrt(squared_norm)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        torch.cuda.synchronize()
        after = trainable_hashes(model)
        changed = sum(after[name] != previous for name, previous in self.before.items())
        if changed == 0:
            raise ValueError("The optimizer did not change any adapter tensor")
        record = {
            "optimizer_step": int(state.global_step),
            "wall_seconds_including_sync_and_audits": time.monotonic() - self.started,
            "finite_nonzero_gradients": True,
            "post_clip_gradient_norm": self.gradient_norm,
            "changed_adapter_tensors": changed,
        }
        self.records.append(record)
        append_json(self.output / "updates.jsonl", record)
        print(f"[update] policy version {state.global_step}: {changed} adapter tensors changed", flush=True)
        if state.global_step == self.stop_at:
            control.should_training_stop = True
            control.should_save = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--variant", choices=("geora_mixed", "lora"), default="geora_mixed")
    parser.add_argument("--fixed-tokens", type=Path, default=Path(__file__).resolve().parent / "fixtures/probability_controls.json")
    parser.add_argument("--phase", choices=(1, 2), type=int, required=True,
                        help="Phase 1 saves update1; phase 2 actually restores it and finishes update2")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--group-port", type=int, default=51216)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.phase == 2) != (args.resume_checkpoint is not None):
        raise ValueError("Only phase 2 requires --resume-checkpoint")
    start_step = args.phase - 1
    target_step = args.phase
    if args.output.exists():
        raise FileExistsError("Use a fresh output directory for this two-update diagnostic")
    if version("trl") != "1.17.0.dev0" or version("vllm") != "0.32.0+rocm723":
        raise RuntimeError("Use the pinned vllm_modern uv environment")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("The learner must see exactly one allocated GCD")
    torch.set_num_threads(4)
    base, checkpoint, protocol = verify_inputs(args)
    if protocol["settings"]["learning_rate"] != 1e-6:
        raise ValueError("The frozen learning rate differs from this diagnostic")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    try:
        properties = torch.cuda.get_device_properties(0)
        write_json(args.output / "environment.json", {
            "hostname": socket.gethostname(),
            "visible_device_count": torch.cuda.device_count(),
            "device_name": properties.name,
            "device_uuid": str(getattr(properties, "uuid", None)),
            "hip_version": torch.version.hip,
            "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
        })
        print("[learner] Loading checkpoint-25 as fresh diagnostic policy v0", flush=True)
        fixed_rows = json.loads(args.fixed_tokens.read_text())
        validate_fixed_rows(fixed_rows)
        if len(fixed_rows) < 5:
            raise ValueError("The five preregistered synthetic controls are required")
        tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        dataset, counts = prepare_dataset(args.train_data, tokenizer, max_prompt_length=1024)
        oversized = [row["id"] for row in dataset if len(tokenizer.apply_chat_template(
            row["prompt"], tokenize=True, add_generation_prompt=True, return_dict=False,
        )) > 1024]
        if oversized:
            raise ValueError(f"The frozen dataset helper retained oversized prompts: {oversized[:10]}")
        set_seed(42)
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, local_files_only=True)
        schema = {name: (tuple(value.shape), value.dtype) for name, value in model.state_dict().items()}
        if args.variant == "geora_mixed":
            load_geora_layers(model, checkpoint, install=True, expected_base_model_path=base)
        else:
            model = PeftModel.from_pretrained(model, checkpoint, local_files_only=True, is_trainable=True)
        model.config.use_cache = False
        model = model.to("cuda")
        trainable = [value for value in model.parameters() if value.requires_grad]
        if not trainable or any(value.dtype != torch.float32 for value in trainable):
            raise ValueError("The trainable adapter tensors must remain FP32")
        initial_frozen = frozen_hashes(model)
        training_args = make_training_args(args.output / "training", args.server_url, args.group_port)
        callback = UpdateAudit(args.output, stop_at=target_step)
        trainer_type = ModernGeoRATrainer if args.variant == "geora_mixed" else ModernLoRATrainer
        extra = {"base_model_path": base} if args.variant == "geora_mixed" else {}
        trainer = trainer_type(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=training_args,
            reward_funcs=grpo_boxed_reward,
            callbacks=[callback],
            **extra,
        )
        trainer.native_audit_dir = args.output
        trainer.probability_records = []
        trainer.resume_audit_dir = args.output / "resume"
        trainer.resume_audit_variant = args.variant

        def three_way_check(session, current):
            comparison = score_three_way(
                session.model, session.expected_schema, session.fixed_rows, session.client,
                device=trainer.accelerator.device,
                audit_path=session.audit_dir / f"three_way_v{current}.json",
                version=current, version_getter=session._version,
            )
            return {key: value for key, value in comparison.items() if key != "rows"}

        sync = install_native_sync(
            trainer,
            expected_schema=schema,
            audit_dir=args.output / "sync",
            fixed_rows=fixed_rows,
            expected_sync_versions=(start_step, target_step),
            expected_generation_versions=(start_step,) * 4,
            fixed_token_checker=three_way_check,
            base_sampling_seed=42,
        )
        write_json(args.output / "protocol.json", {
            "purpose": "Native synchronous vLLM/TRL two-update compatibility closure",
            "variant": args.variant,
            "initialization_checkpoint": str(checkpoint),
            "initialization_checkpoint_step": 25,
            "historical_training_resumed": False,
            "optimizer": "fresh AdamW" if args.phase == 1 else "AdamW restored from diagnostic checkpoint1",
            "policy_versions_this_process": [start_step, target_step],
            "phase": args.phase,
            "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint else None,
            "max_steps_for_both_phases": 2,
            "sampling_seed_policy": "42 + 8 * (global_optimizer_step * 4 + rollout_index_within_update)",
            "server_rng_state_required": False,
            "sampling_replay_bitwise_guaranteed": False,
            "max_completion_length": 4096,
            "max_prompt_length": 1024,
            "generation_batch_size": 32,
            "rollouts_per_update": 4,
            "responses_per_update": 128,
            "prompt_groups_per_update": 16,
            "num_generations": 8,
            "micro_batch_size": 4,
            "gradient_accumulation_steps": 32,
            "reward": GSM8K_REWARD_VERSION,
            "data_counts": counts,
            "execution_manifest_sha256": file_hash(args.execution_manifest),
            "train_data_sha256": file_hash(args.train_data),
            "fixed_tokens_sha256": file_hash(args.fixed_tokens),
            "versions": {name: version(name) for name in ("trl", "vllm", "torch", "transformers", "peft")},
        })
        set_seed(42)
        print(f"[learner] Phase {args.phase}: completing stock TRL update {target_step}", flush=True)
        if args.resume_checkpoint:
            previous_inventory = checkpoint_inventory(args.resume_checkpoint, args.variant)
            if previous_inventory["global_step"] != 1:
                raise ValueError("Phase 2 must restore diagnostic checkpoint1")
        train_result = trainer.train(resume_from_checkpoint=str(args.resume_checkpoint) if args.resume_checkpoint else None)
        if trainer.state.global_step != target_step or len(callback.records) != 1:
            raise ValueError("Each process must complete exactly one optimizer update")
        versions = [row["policy_version"] for row in trainer.probability_records]
        if versions != [start_step] * 4:
            raise ValueError(f"Expected four version{start_step} rollouts, got {versions}")
        print(f"[learner] Publishing final policy v{target_step} and checking the resident server", flush=True)
        sync.sync_weights()
        final_verification = sync.verify_final()
        if frozen_hashes(model) != initial_frozen:
            raise ValueError("A frozen base parameter or buffer changed during training")
        final_checkpoint = args.output / f"training/checkpoint-{target_step}"
        for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
            if not (final_checkpoint / name).is_file():
                raise ValueError(f"The final diagnostic checkpoint is incomplete: {name}")
        saved_policy = verify_loaded_policy(model, final_checkpoint, args.variant)
        saved_inventory = checkpoint_inventory(final_checkpoint, args.variant)
        restored = trainer.resume_audit_summary() if args.phase == 2 else None
        if args.resume_checkpoint and checkpoint_inventory(args.resume_checkpoint, args.variant) != previous_inventory:
            raise ValueError("Resume mutated its input checkpoint")
        report = {
            "completed": True,
            "scope": "One phase of a two-update save/restore native synchronization diagnostic; no accuracy or speedup claim",
            "phase": args.phase,
            "variant": args.variant,
            "optimizer_updates_this_process": 1,
            "total_global_step": trainer.state.global_step,
            "training_rollouts": 4,
            "training_responses": 128,
            "policy_sync_versions": [start_step, target_step],
            "stock_importance_sampling_verified": True,
            "finite_nonzero_gradients_each_update": True,
            "changed_adapter_tensors_each_update": [row["changed_adapter_tensors"] for row in callback.records],
            "frozen_parameters_and_buffers_unchanged": True,
            "final_server_verification": final_verification,
            "final_checkpoint": str(final_checkpoint),
            "saved_policy_matches": saved_policy,
            "saved_inventory": saved_inventory,
            "restore_audit": restored,
            "three_way_numerical_acceptance": "Descriptive measurements; no fitted equivalence threshold",
            "total_learner_wall_seconds": time.monotonic() - started,
            "train_metrics": train_result.metrics,
        }
        write_json(args.output / "result.json", report)
        print(json.dumps(report, indent=2), flush=True)
    except BaseException as error:
        write_json(args.output / "failure.json", {
            "completed": False,
            "error": f"{type(error).__name__}: {error}",
            "elapsed_seconds": time.monotonic() - started,
        })
        raise


if __name__ == "__main__":
    main()
