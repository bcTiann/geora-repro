"""Audit two five-step continuation reports and summarize measured cost.

No model loading, GPU work, or task-score inference is performed. This checks
the reports' identities, recorded gates, batch order, replay, and accounting.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


EXPECTED_STAGE = "gsm8k_five_step_continuation_and_resume"
REQUIRED_GATES = {
    "same_196_targets_only_AB_trainable",
    "BF16_frozen_FP32_AB_and_initial_buffers",
    "train_mode_zero_dropout_original_frozen_reference",
    "fixed_ten_train_questions_without_reward_filtering",
    "initial_policy_matches_original_reference",
    "resume_restores_step_two_and_data_position",
    "resume_reproduces_question_ids_tokens_rewards",
    "resume_reproduces_sampling_and_old_logp_exactly",
    "resume_reproduces_next_AB_update_exactly",
    "resume_reproduces_AdamW_scheduler_generator_exactly",
    "resume_reproduces_global_CPU_GPU_Python_RNG_exactly",
    "all_frozen_weights_initial_buffers_reference_unchanged",
    "five_logical_steps_and_verified_resume",
    "at_least_one_real_nonzero_update_without_filtering",
    "final_updated_policy_generation_and_scoring",
}
TIMING_FIELDS = (
    "rollout_seconds", "scoring_seconds", "backward_and_step_seconds",
    "checkpoint_seconds",
)


def finite_nonnegative(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def measured_times(batch):
    return {name: float(batch.get(name, 0.0)) for name in TIMING_FIELDS}


def allocation_record(report, jobs):
    if jobs is None:
        return None
    matching = [job for job in jobs.get("jobs", [])
                if str(job["job_id"]) == str(report["slurm_job_id"])]
    if len(matching) != 1:
        raise ValueError(f"Allocation ledger must contain one entry for job {report['slurm_job_id']}")
    job = matching[0]
    return {
        "source": "supplied Slurm job ledger",
        "job_id": str(job["job_id"]),
        "state": job["state"],
        "allocation_seconds": job["allocation_seconds"],
        "logical_gpu_count": job["allocated_resources"]["logical_gpu_count"],
        "logical_gpu_allocation_seconds": job["allocation_seconds"]
            * job["allocated_resources"]["logical_gpu_count"],
    }


def summarize_method(report, checks, jobs):
    method = report["method"]
    def require(name, condition, **details):
        checks.append(dict(name=method + "_" + name, passed=bool(condition), **details))

    require("passed_expected_stage", report["status"] == "passed"
            and report["stage"] == EXPECTED_STAGE)
    source_checks = report["checks"]
    names = [entry["name"] for entry in source_checks]
    require("all_recorded_gates_passed", bool(source_checks)
            and all(entry.get("passed") is True for entry in source_checks)
            and len(names) == len(set(names)), recorded_gate_count=len(source_checks))
    required_gates = REQUIRED_GATES | {
        label + "_" + suffix
        for label in [f"step_{step}" for step in range(1, 6)] + ["resume_replay_step_3"]
        for suffix in ("causal_completion_mask", "sampled_and_old_likelihood",
                       "current_and_old_likelihood", "finite_loss",
                       "finite_FP32_AB_gradients", "finite_AB_parameters")
    }
    required_gates.add({
        "geora": "starts_from_untrained_GeoRA",
        "lora": "starts_from_random_A_zero_B_LoRA",
    }[method])
    require("required_gates_present", required_gates <= set(names),
            missing=sorted(required_gates - set(names)))
    require("five_logical_six_physical", report["logical_optimizer_steps"] == 5
            and report["physical_optimizer_step_calls"] == 6
            and report["physical_backward_calls"] == 6
            and report["physical_rollouts"] == 6
            and report["next_data_position"] == 10)
    batches = report["batches"]
    require("five_ordered_batches", len(batches) == 5
            and [batch["label"] for batch in batches] == [f"step_{step}" for step in range(1, 6)])
    data_order = [question for batch in batches for question in batch["question_ids"]]
    declared_order = next((entry.get("data_order") for entry in source_checks
                           if entry["name"] == "fixed_ten_train_questions_without_reward_filtering"), None)
    require("ten_unique_training_questions", len(data_order) == 10
            and len(set(data_order)) == 10 and all(question.startswith("train:") for question in data_order)
            and data_order == declared_order)
    require("provenance_matches_report_identity", report["provenance"]["method"] == method
            and report["provenance"]["model_repository"] == report["model_repository"]
            and report["provenance"]["model_revision"] == report["model_revision"]
            and report["provenance"]["dataset_manifest_sha256"] == report["dataset_manifest_sha256"])
    q_count = report["config"]["questions_per_batch"]
    group_size = report["config"]["answers_per_question"]
    steps = []
    for batch in batches:
        answers = batch["answers"]
        flat_rewards = [reward for group in batch["rewards"] for reward in group]
        valid_tokens = sum(answer["token_count"] for answer in answers)
        equal_groups = [len(set(group)) == 1 for group in batch["rewards"]]
        parse_failures = sum(answer["predicted_answer"] is None for answer in answers)
        truncations = sum(answer["truncated"] for answer in answers)
        require(batch["label"] + "_answer_and_reward_accounting",
                len(answers) == q_count * group_size
                and len(batch["question_ids"]) == q_count
                and len(batch["rewards"]) == q_count
                and all(len(group) == group_size for group in batch["rewards"])
                and flat_rewards == [answer["training_reward"] for answer in answers]
                and all(reward in (0, 1) for reward in flat_rewards)
                and all(answer["question_id"] == batch["question_ids"][index // group_size]
                        and answer["token_count"] == len(answer["token_ids"])
                        and answer["truncated"] == (not answer["ended_with_eos"])
                        and answer["training_reward"] == (answer["reward"] if answer["ended_with_eos"] else 0)
                        for index, answer in enumerate(answers))
                and valid_tokens == batch["valid_completion_tokens"]
                and equal_groups == batch["equal_reward_groups"]
                and parse_failures == batch["parse_failure_count"]
                and truncations == batch["truncated_answer_count"])
        require(batch["label"] + "_finite_measurements",
                all(finite_nonnegative(batch.get(name, 0.0)) for name in TIMING_FIELDS)
                and finite_nonnegative(batch["gradient_norm_before_clipping"])
                and all(math.isfinite(batch[name]) for name in ("loss", "policy_loss", "sampled_old_token_k3")))
        mask = batch["completion_mask"]
        behavior_logp = batch["behavior_logp"]
        old_logp = batch["old_logp"]
        answer_count = q_count * group_size
        width = report["config"]["max_new_tokens"]
        shape_matches = all(
            len(rows) == answer_count and all(len(row) == width for row in rows)
            for rows in (mask, behavior_logp, old_logp)
        )
        require(batch["label"] + "_mask_and_likelihood_values",
                shape_matches
                and all(all(isinstance(value, bool) for value in row) for row in mask)
                and all(row == [True] * answer["token_count"]
                        + [False] * (width - answer["token_count"])
                        for row, answer in zip(mask, answers))
                and all(math.isfinite(sampled) and math.isfinite(old)
                        and abs(sampled - old) <= report["config"]["max_absolute_logp_mismatch"]
                        for mask_row, sampled_row, old_row in zip(mask, behavior_logp, old_logp)
                        for active, sampled, old in zip(mask_row, sampled_row, old_row) if active))
        steps.append({
            "step": int(batch["label"].split("_")[-1]),
            "question_ids": batch["question_ids"],
            "rewards": batch["rewards"],
            "gradient_norm_before_clipping": batch["gradient_norm_before_clipping"],
            "changed_parameter_tensors": batch["changed_parameter_tensors"],
            "equal_reward_groups": equal_groups,
            "parse_failure_count": parse_failures,
            "truncated_answer_count": truncations,
            "valid_completion_tokens": valid_tokens,
            "loss": batch["loss"],
            "sampled_old_token_k3": batch["sampled_old_token_k3"],
            "clip_fraction": batch["clip_fraction"],
            "timings": measured_times(batch),
        })
    replay = report["resume_replay"]
    original = batches[2]
    require("replay_records_exact", replay["question_ids"] == original["question_ids"]
            and [answer["token_ids"] for answer in replay["answers"]]
                == [answer["token_ids"] for answer in original["answers"]]
            and replay["rewards"] == original["rewards"]
            and replay["advantages"] == original["advantages"]
            and replay["behavior_logp"] == original["behavior_logp"]
            and replay["old_logp"] == original["old_logp"]
            and replay["completion_mask"] == original["completion_mask"])
    logical_tokens = sum(batch["valid_completion_tokens"] for batch in batches)
    physical_tokens = logical_tokens + replay["valid_completion_tokens"]
    require("token_and_failure_totals", logical_tokens == report["total_logical_completion_tokens"]
            and physical_tokens == report["total_physical_completion_tokens"]
            and sum(sum(batch["equal_reward_groups"]) for batch in batches) == report["equal_reward_group_count"]
            and sum(batch["parse_failure_count"] for batch in batches) == report["parse_failure_count"]
            and sum(batch["truncated_answer_count"] for batch in batches) == report["truncated_answer_count"])
    logical_times = {name: sum(batch.get(name, 0.0) for batch in batches) for name in TIMING_FIELDS}
    replay_times = measured_times(replay)
    load_seconds = report["timings"]["load_and_setup_seconds"]
    recorded_component_seconds = sum(logical_times.values()) + sum(replay_times.values()) + load_seconds
    require("total_runtime_and_memory_finite", finite_nonnegative(report["elapsed_seconds"])
            and finite_nonnegative(report["peak_allocated_memory_gib"])
            and report["elapsed_seconds"] + 1e-4 >= recorded_component_seconds)
    allocation = allocation_record(report, jobs)
    if allocation is not None:
        require("slurm_completed_one_logical_gpu", allocation["state"] == "COMPLETED"
                and allocation["logical_gpu_count"] == 1
                and allocation["allocation_seconds"] >= report["elapsed_seconds"] - 1.0)
    return {
        "method": method,
        "job_id": str(report["slurm_job_id"]),
        "reported_checks_passed": sum(entry.get("passed") is True for entry in source_checks),
        "reported_checks_total": len(source_checks),
        "data_order": data_order,
        "logical_optimizer_steps": 5,
        "physical_optimizer_step_calls": 6,
        "steps": steps,
        "cost": {
            "total_python_elapsed_seconds": report["elapsed_seconds"],
            "load_and_setup_seconds": load_seconds,
            "logical_training_timed_components_seconds": logical_times,
            "logical_training_component_sum_seconds": sum(logical_times.values()),
            "logical_valid_completion_tokens": logical_tokens,
            "logical_rollout_valid_tokens_per_second": logical_tokens / logical_times["rollout_seconds"]
                if logical_times["rollout_seconds"] > 0 else None,
            "replay_valid_completion_tokens": replay["valid_completion_tokens"],
            "replay_timed_components_seconds": replay_times,
            "replay_component_sum_seconds": sum(replay_times.values()),
            "physical_training_valid_completion_tokens": physical_tokens,
            "other_or_untimed_seconds": report["elapsed_seconds"] - recorded_component_seconds,
            "peak_allocated_memory_gib": report["peak_allocated_memory_gib"],
            "slurm_allocation": allocation,
        },
        "reward_diagnostics": {
            "equal_reward_group_count": report["equal_reward_group_count"],
            "parse_failure_count": report["parse_failure_count"],
            "truncated_answer_count": report["truncated_answer_count"],
            "correct_training_reward_answers": sum(answer["training_reward"] for batch in batches for answer in batch["answers"]),
            "logical_answer_count": sum(len(batch["answers"]) for batch in batches),
        },
    }


def analyze_reports(geora, lora, *, jobs=None, config_path=None):
    result = {
        "status": "running",
        "scope": "Independent checks of recorded five-step continuity/resume reports; no task-score or superiority claim",
        "checks": [],
        "limitations": [
            "Five steps on ten training questions are not a benchmark accuracy estimate.",
            "Reward zero combines incorrect values, unsupported final-answer format and truncated answers.",
            "Logical stage times exclude the replay; total Python/Slurm time includes it and restoration/audit overhead.",
            "Training token totals exclude the separate final eight-token generation diagnostic.",
            "Throughput describes this correctness-oriented fixed-width path, not the paper's efficiency protocol.",
        ],
    }
    checks = result["checks"]
    def require(name, condition, **details):
        checks.append(dict(name=name, passed=bool(condition), **details))
    require("reports_have_expected_methods", geora["method"] == "geora" and lora["method"] == "lora")
    require("same_training_configuration", geora["config"] == lora["config"])
    for field in ("model_repository", "model_revision", "dataset_manifest_sha256", "source_git_commit",
                  "torch_version", "hip_version", "gpu_name"):
        require("same_" + field, geora[field] == lora[field], geora=geora[field], lora=lora[field])
    require("same_config_and_data_provenance", all(
        geora["provenance"][field] == lora["provenance"][field]
        for field in ("model_repository", "model_revision", "dataset_manifest_sha256", "training_config_sha256")))
    if config_path is not None:
        raw = Path(config_path).read_bytes()
        require("actual_config_file_matches_reports", json.loads(raw) == geora["config"]
                and hashlib.sha256(raw).hexdigest() == geora["provenance"]["training_config_sha256"])
    methods = {
        "geora": summarize_method(geora, checks, jobs),
        "lora": summarize_method(lora, checks, jobs),
    }
    require("same_fixed_question_order", methods["geora"]["data_order"] == methods["lora"]["data_order"])
    first_geora, first_lora = geora["batches"][0], lora["batches"][0]
    require("initial_same_policy_sampling_tokens_exact", [answer["token_ids"] for answer in first_geora["answers"]]
            == [answer["token_ids"] for answer in first_lora["answers"]])
    require("initial_same_policy_behavior_logp_exact", first_geora["behavior_logp"] == first_lora["behavior_logp"]
            and first_geora["completion_mask"] == first_lora["completion_mask"]
            and first_geora["rewards"] == first_lora["rewards"])
    result["methods"] = methods
    allocations = [method["cost"]["slurm_allocation"] for method in methods.values()]
    result["combined_logical_gpu_allocation_seconds"] = (
        sum(record["logical_gpu_allocation_seconds"] for record in allocations)
        if all(record is not None for record in allocations) else None
    )
    result["status"] = "passed" if all(check["passed"] for check in checks) else "failed"
    result["checks_passed"] = sum(check["passed"] for check in checks)
    result["checks_total"] = len(checks)
    return result


def print_summary(result):
    print(f"Report audit: {result['status']} ({result.get('checks_passed', 0)}/{result.get('checks_total', 0)})")
    if "error" in result:
        print("ERROR:", result["error"])
    for name, method in result.get("methods", {}).items():
        cost = method["cost"]
        print(f"{name}: job {method['job_id']}, Python {cost['total_python_elapsed_seconds']:.2f}s, "
              f"peak {cost['peak_allocated_memory_gib']:.2f} GiB, "
              f"logical tokens {cost['logical_valid_completion_tokens']}, "
              f"replay tokens {cost['replay_valid_completion_tokens']}")
        for step in method["steps"]:
            print(f"  step {step['step']}: rewards={step['rewards']} "
                  f"grad={step['gradient_norm_before_clipping']:.6g} "
                  f"changed={step['changed_parameter_tensors']} "
                  f"truncated={step['truncated_answer_count']} parse_fail={step['parse_failure_count']}")
    for check in result.get("checks", []):
        if not check["passed"]:
            print("FAIL:", check["name"])
    print("This verifies the recorded training/resume process; it does not establish task improvement.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geora", type=Path, required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=Path)
    parser.add_argument("--config", type=Path)
    arguments = parser.parse_args()
    try:
        result = analyze_reports(
            json.loads(arguments.geora.read_text()), json.loads(arguments.lora.read_text()),
            jobs=None if arguments.jobs is None else json.loads(arguments.jobs.read_text()),
            config_path=arguments.config,
        )
    except Exception as error:
        result = {"status": "failed", "scope": "Continuation report audit", "error": str(error)}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print_summary(result)
    print("Analysis:", arguments.output)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
