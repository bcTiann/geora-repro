"""Audit the selected micro8/gen128 two-method, two-process-phase diagnostic.

Reads collected evidence only. It never loads checkpoint pickle/tensor files,
launches GPU work, writes formal configuration, or assumes exact vLLM replay.
Exit codes: 0 verified, 1 incomplete, 2 invalid/mismatched evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import math
from pathlib import Path

from summarize_batch_benchmark import (canonical_names, equal_number, file_hash,
    finite, float32, normalize_bdf, read, require, resident_names, rows, schedule, sha_mapping,
    value_hash, SHA)
from summarize_pair import COMPARISONS, exact_replay, raw_comparison

ROOT = Path(__file__).resolve().parent
METHODS = ("lora", "geora_mixed")
SCHEDULE = schedule(8, 128)
REWARD = "final_boxed_exact_rational_v2"


def checkpoint_inventory(value, variant, step):
    require(isinstance(value, dict) and value["variant"] == variant and value["global_step"] == step,
            "Wrong checkpoint inventory variant/global step")
    require(isinstance(value["checkpoint"], str) and value["checkpoint"].startswith("/")
            and (step == 25 or value["checkpoint"].endswith(f"checkpoint-{step}")),
            "Unexpected checkpoint path")
    names = {"trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"}
    names.update({"adapter_model.safetensors", "adapter_config.json"} if variant == "lora"
                 else {"geora.safetensors", "geora_config.json"})
    sha_mapping(value["files_sha256"], "complete checkpoint inventory", names)


def policy_check(value, variant):
    require(value["bitwise_equal"] is True and value["tensor_count"] == (392 if variant == "lora" else 672),
            "Runtime saved/loaded policy did not verify complete bitwise coverage")
    require(isinstance(value["state_sha256"], str) and SHA.fullmatch(value["state_sha256"]), "Missing policy state digest")


def three_way(report, version, controls):
    require(report["policy_version"] == version and report["structural_checks_passed"] is True
            and report["acceptance_policy"] is None and report["numerical_accepted"] is None,
            "Incomplete or undeclared numerical three-way acceptance")
    require(report["fixed_rows_sha256"] == value_hash(controls) and report["sequences"] == len(controls) == 5
            and len(report["rows"]) == 5, "Fixed probability controls changed")
    for row, fixed in zip(report["rows"], controls, strict=True):
        require(all(row[key] == fixed[key] for key in ("id", "category", "prompt_ids", "completion_ids"))
                and row["includes_eos"] is True and fixed["terminated_with_eos"] is True
                and row["completion_ids"][-1] == fixed["eos_token_id"], "Three-way token/EOS controls differ")
        require(all(len(row[field]) == len(row["completion_ids"])
                    and all(finite(value, "fixed logprob") <= 1e-6 for value in row[field])
                    for field in ("learner_logprobs", "hf_merged_logprobs", "vllm_logprobs")), "Invalid fixed-token probability")
    return {comparison: raw_comparison(report, comparison) for comparison in COMPARISONS}


def training_audits(learner, result, phase, native):
    probability, rollout, reward = (rows(learner / name) for name in
        ("training_probabilities.jsonl", "sync/rollouts.jsonl", "reward_audit.jsonl"))
    require(len(probability) == len(rollout) == len(reward) == 1, "Expected one128-response rollout per phase")
    p, r, reward = probability[0], rollout[0], reward[0]
    require(p["optimizer_step"] == r["policy_version"] == phase - 1 and p["responses"] == r["responses"] == 128,
            "Actual rollout policy/responses differ")
    require(r["generation_schedule"] == native and r["rollout_in_update"] == 0
            and r["request_seed"] == 42 + 8 * (phase - 1), "Native rollout schedule/request seed differs")
    require(p["stock_token_truncate_verified"] is True and r["all_behavior_logprobs_finite"] is True,
            "Stock importance correction or behavior probabilities failed")
    ids = p["question_ids"]
    require(len(ids) == 128 and all(isinstance(value, str) and value and value != "None" for value in ids)
            and len(set(ids)) == 16 and set(Counter(ids).values()) == {8}
            and all(len(set(ids[index:index+8])) == 1 for index in range(0, 128, 8)), "Actual inputs are not16 distinct G8 groups")
    declared = result["actual_questions"]
    require(declared["question_ids"] == ids and declared["unique_question_ids"] == sorted(set(ids))
            and declared["question_counts"] == dict(Counter(ids)) and declared["responses"] == 128,
            "Declared and actual question exposure differ")
    require(len(r["prompt_ids"]) == len(r["completion_ids"]) == len(r["behavior_logprobs"])
            == len(p["old_learner_logprobs"]) == len(p["behavior_logprobs"]) == 128, "Raw probability rows omitted")
    tokens, deltas = 0, []
    prompts = {}
    for identifier, prompt, completion, behavior, old, sampled in zip(ids, r["prompt_ids"], r["completion_ids"],
            r["behavior_logprobs"], p["old_learner_logprobs"], p["behavior_logprobs"], strict=True):
        require(0 < len(prompt) <= 1024 and 0 < len(completion) <= 4096
                and all(type(token) is int and token >= 0 for token in prompt + completion), "Invalid actual token IDs/length")
        require(identifier not in prompts or prompts[identifier] == prompt, "Question ID mapped to different prompt tokens")
        prompts[identifier] = prompt
        require(len(completion) == len(behavior) == len(old) == len(sampled), "Unscored active completion tokens")
        for a, b, c in zip(behavior, sampled, old, strict=True):
            require(all(finite(value, "active logprob") <= 1e-6 for value in (a, b, c)), "Invalid positive log probability")
            equal_number(a, b, "native/stock behavior logprob")
            deltas.append(float32(c - b))
        tokens += len(completion)
    require(tokens == p["active_tokens"] == r["tokens"] == result["actual_completion_tokens"], "Active token totals differ")
    require(type(p["padded_completion_positions"]) is int and p["padded_completion_positions"] >= tokens, "Invalid padding count")
    equal_number(sum(abs(value) for value in deltas) / tokens, p["mean_abs_log_ratio"], "mean raw log ratio")
    equal_number(max(abs(value) for value in deltas), p["max_abs_log_ratio"], "max raw log ratio")
    clipped = sum(value > float32(math.log(2)) for value in deltas) / tokens
    equal_number(clipped, p["fraction_ratio_above_2"], "stock cap2 trigger fraction")
    equal_number(1 - tokens / p["padded_completion_positions"], p["padding_fraction"], "padding fraction")
    require(reward["reward_contract"] == REWARD and reward["responses"] == 128 and reward["question_ids"] == ids
            and reward["global_step"] == phase - 1, "Reward audit identity/version differs")
    counts, grouping = reward["status_counts"], reward["grouping"]
    require(set(counts) == {"correct", "incorrect", "missing_or_incomplete_boxed", "invalid_prediction"}
            and all(type(value) is int and value >= 0 for value in counts.values()) and sum(counts.values()) == 128,
            "Reward statuses do not cover128 responses")
    require(grouping["status"] == "validated_consecutive_groups" and grouping["group_count"] == len(grouping["groups"]) == 16,
            "Reward groups were not identity validated")
    zero_groups, correct = 0, 0
    for index, group in enumerate(grouping["groups"]):
        mean = finite(group["reward_mean"], "group reward mean", minimum=0)
        require(group["question_id"] == ids[index*8] and group["responses"] == 8
                and mean <= 1 and math.isclose(mean*8, round(mean*8)), "Invalid binary reward group")
        require(group["zero_std"] == (mean in (0, 1)) and group["all_zero"] == (mean == 0)
                and group["all_one"] == (mean == 1), "Invalid zero-variance reward classification")
        zero_groups += group["zero_std"]
        correct += round(mean*8)
    require(correct == counts["correct"], "Reward group means/statuses differ")
    equal_number(zero_groups / 16, grouping["zero_std_fraction"], "zero variance fraction")
    equal_number(correct / 128, reward["reward_mean"], "reward mean")
    require(p["nonzero_advantage_responses"] == (16-zero_groups)*8
            and result["all_measured_advantages_zero"] == (zero_groups == 16), "Reward groups/actual advantages differ")
    return {"question_ids": ids, "prompt_tokens_by_question_sha256": value_hash(prompts), "tokens": tokens,
        "cap2_trigger_fraction": clipped, "max_abs_log_ratio": max(abs(value) for value in deltas),
        "reward_status_counts": counts, "zero_std_groups": zero_groups}


def phase_check(root, variant, phase, config, plan, controls):
    phase_dir, learner = root / variant / f"phase-{phase}", root / variant / f"phase-{phase}/learner"
    result, protocol, invocation = (read(learner / name) for name in ("result.json", "run_protocol.json", "invocation.json"))
    contract = protocol["immutable_contract"]
    require(result["completed"] is True and result["mode"] == protocol["mode"] == invocation["mode"] == "schedule_acceptance"
            and result["phase"] == protocol["phase"] == invocation["phase"] == phase
            and result["variant"] == contract["variant"] == invocation["variant"] == variant, "Phase/variant/mode differs")
    require(result["schedule"] == contract["schedule"] == SCHEDULE and invocation["micro_batch_size"] == 8
            and invocation["generation_batch_size"] == 128, "This is not the selected micro8/gen128 schedule")
    require(result["max_steps_for_both_phases"] == contract["max_steps_for_both_phases"] == invocation["max_steps"] == 2
            and result["optimizer_updates_this_process"] == result["training_rollouts"] == 1
            and result["total_global_step"] == phase and result["training_responses"] == 128, "Wrong optimizer/response budget")
    expected_sources = {name.removeprefix("vllm_modern/"): digest for name, digest in plan["source_files_sha256"].items()
                        if name.startswith("vllm_modern/") and name.endswith(".py")}
    require(contract["source_sha256"] == expected_sources, "Execution Python source differs from the frozen plan")
    require(contract["scientific_configuration_sha256"] == file_hash(root / "scientific-config.json")
            and contract["versions"] == config["runtime_versions"] == read(learner / "environment.json")["versions"]
            and contract["uv_lock_sha256"] == config["runtime_lock_sha256"] == plan["source_files_sha256"]["vllm_modern/uv.lock"]
            and contract["base_files_sha256"] == config["model"]["expected_base_files_sha256"]
            and contract["train_data_sha256"] == config["data"]["train_file_sha256"]
            and contract["fixed_tokens_sha256"] == plan["source_files_sha256"]["vllm_modern/fixtures/probability_controls.json"]
            and contract["reward"] == config["training"]["reward"] == REWARD
            and contract["training_seed"] == config["training"]["seed"] == 42
            and contract["data_seed"] == config["training"]["data_seed"] == 42, "Runtime/configuration/frozen inputs differ")
    for name in ("scientific_configuration_sha256", "uv_lock_sha256", "execution_manifest_sha256", "train_data_sha256", "fixed_tokens_sha256"):
        require(isinstance(contract[name], str) and SHA.fullmatch(contract[name]), "Missing input SHA: " + name)
    sha_mapping(contract["base_files_sha256"], "base input files")
    checkpoint_inventory(contract["original_policy_initialization"], variant, 25)
    saved = result["saved_inventory"]
    checkpoint_inventory(saved, variant, phase)
    require(result["final_checkpoint"] == saved["checkpoint"], "Saved inventory/final checkpoint differ")
    require(result["frozen_parameters_and_buffers_unchanged"] is True and result["finite_gradients_each_update"] is True
            and result["stock_importance_sampling_verified"] is True, "Runtime policy/gradient invariants failed")
    require(result["parameters"] == {"target_layer_count": 196, "trainable_tensors": 392,
            "trainable_elements": 18464768, "trainable_dtype": "float32"}, "Adapter coverage/storage changed")
    for name in ("loaded_policy_matches", "saved_policy_matches"):
        policy_check(result[name], variant)
    require(all(result[name] is False for name in ("numerical_equivalence_claim", "formal400_training", "sampling_replay_bitwise_guaranteed", "performance_claim")),
            "Diagnostic claims exceed their measured scope")
    optimizer = read(learner / "optimizer_at_train_begin.json")
    require(optimizer == result["optimizer_at_train_begin"] and optimizer["global_step"] == phase - 1
            and optimizer["fresh_empty_state"] is (phase == 1) and optimizer["state_entries"] == (0 if phase == 1 else 392)
            and SHA.fullmatch(optimizer["state_sha256"]), "Actual optimizer initialization/restoration proof differs")
    options = read(learner / "resolved_training_options.json")
    require(options["micro_batch_size"] == 8 and options["gradient_accumulation_steps"] == options["steps_per_generation"] == 16
            and options["generation_batch_size"] == 128 and options["bf16_autocast"] is False and options["fp16_autocast"] is False,
            "Resolved batch/precision options differ")
    update = rows(learner / "updates.jsonl")
    require(len(update) == 1 and update == result["updates"] and update[0]["optimizer_step"] == phase, "Actual update record differs")
    for name in ("update_wall_seconds_including_sync_and_diagnostics", "post_clip_gradient_norm", "peak_allocated_gib", "peak_reserved_gib"):
        finite(update[0][name], name, minimum=0)
    require(update[0]["zero_gradient"] == (update[0]["post_clip_gradient_norm"] == 0), "Gradient status differs")
    require(type(result["changed_adapter_tensors"]) is int and 0 <= result["changed_adapter_tensors"] <= 392
            and (result["changed_adapter_tensors"] > 0 or result["all_measured_advantages_zero"] is True), "No actual adapter update")
    capacity = None
    if phase == 1:
        capacity = read(learner / "capacity_result.json")
        require(result["capacity_probe_performed"] is True and result["capacity_result"] == capacity
                and capacity["completed"] is True and capacity["passed"] is True and capacity["error"] is None
                and capacity["variant"] == variant and capacity["mode"] == "capacity" and capacity["compute_entropy"] is True
                and capacity["micro_batch_size"] == 8 and capacity["prompt_length"] == 1024
                and capacity["completion_length"] == 4096 and capacity["padded_total_length"] == 5120
                and capacity["parameter_updates"] == 0 and capacity["trainable_weights_unchanged"] is True
                and capacity["optimizer_moment_bytes_reserved"] == 147718144 and capacity["synthetic_workload"] is True,
                "Phase1 entropy/full-length capacity was not accepted without an update")
        for name in ("peak_allocated_gib", "peak_reserved_gib", "seconds"):
            finite(capacity[name], "capacity " + name, minimum=0)
        require(protocol["resume_input"] is None and invocation["resume_checkpoint"] is None
                and invocation["initialization_checkpoint"] == contract["original_policy_initialization"]["checkpoint"]
                and result["restore_audit"] is None, "Phase1 did not start independently from checkpoint25")
    else:
        require(result["capacity_probe_performed"] is False and result["capacity_result"] is None
                and invocation["initialization_checkpoint"] is None, "Phase2 reran capacity or initialized from old policy")
        checkpoint_inventory(protocol["resume_input"], variant, 1)
        require(invocation["resume_checkpoint"] == protocol["resume_input"]["checkpoint"], "Resume invocation/inventory differ")
    native = {"responses_per_rollout": 128, "num_generations": 8, "questions_per_rollout": 16,
        "rollouts_per_update": 1, "responses_per_update": 128, "questions_per_update": 16,
        "request_seed_formula": "base_seed + num_generations * (global_step * rollouts_per_update + rollout_in_update)",
        "batch_invariant_sampling_claim": False, "training_micro_batch_size": 8,
        "gradient_accumulation_steps": 16, "steps_per_generation": 16}
    publications = rows(learner / "sync/sync.jsonl")
    require([row["policy_version"] for row in publications] == [phase - 1, phase], "Native publication versions differ")
    comparisons, raw_three_way = {}, []
    for publication in publications:
        require(publication["generation_schedule"] == native and publication["canonical_tensor_count"] == 339, "Native publication schedule/coverage differs")
        sha_mapping(publication["canonical_tensor_sha256"], "canonical policy", canonical_names())
        resident = publication["resident_verification"]
        require(resident["passed"] is True and resident["canonical_tensor_count"] == 339 and resident["resident_parameter_count"] == 198,
                "Native resident verification failed")
        sha_mapping(resident["expected_resident_checksums"], "resident policy", resident_names())
        require(publication["resident_checksums"] == resident["expected_resident_checksums"], "Actual resident bytes differ from the expected packed policy")
        raw = read(learner / f"sync/three_way_v{publication['policy_version']}.json")
        require(publication["fixed_token_comparison"] == {key: value for key, value in raw.items() if key != "rows"},
                "Publication/raw three-way report differs")
        comparisons[str(publication["policy_version"])] = three_way(raw, publication["policy_version"], controls)
        raw_three_way.append(raw)
    closure = result["final_server_verification"]
    require(closure["passed"] is True and closure["all_resident_weights_verified"] is True
            and closure["sync_versions"] == [phase - 1, phase] and closure["generation_schedule"] == native
            and closure["training_rollouts"] == 1 and closure["training_responses"] == 128, "Native final closure differs")
    audit = training_audits(learner, result, phase, native)
    devices = {role: read(phase_dir / f"{role}-device.json") for role in ("server", "learner")}
    separation = read(phase_dir / "separate-gcds-verified.json")
    require(separation["passed"] is True and all(separation[role] == value and value["visible_device_count"] == 1 for role, value in devices.items())
            and devices["server"]["hostname"] == devices["learner"]["hostname"]
            and normalize_bdf(devices["server"]["physical_pci_bdf"]) != normalize_bdf(devices["learner"]["physical_pci_bdf"]), "Physical one-GCD roles were not separate")
    performance = rows(learner / "performance.jsonl")
    require(sum(row["stage"] == "training_microstep" and row["optimizer_step"] == phase - 1 for row in performance) == 16,
            "Actual training did not use16 accumulated microsteps")
    require(sum(row["stage"] == "capacity_probe" for row in performance) == (1 if phase == 1 else 0), "Capacity timing scope differs")
    for name in ("failure.json", "sync/failures.jsonl"):
        path = learner / name
        require(not path.exists() or (path.suffix == ".jsonl" and not rows(path)), "Recorded runtime failure: " + name)
    require(isinstance(result["dataset_ordered_ids_sha256"], str) and SHA.fullmatch(result["dataset_ordered_ids_sha256"])
            and result["data_counts"]["raw_questions"] == config["data"]["training_rows"]
            and result["data_counts"]["kept_questions"] >= 32, "Full frozen training split/order was not recorded")
    return {"phase": phase, "variant": variant, "status": "verified", "training_responses": 128,
        "total_global_step": phase, "actual_questions": audit, "updates": update, "capacity": capacity,
        "optimizer_at_train_begin": optimizer, "saved_inventory": saved, "policy_bitwise_runtime_checks": True,
        "three_way_comparisons": comparisons, "dataset_ordered_ids_sha256": result["dataset_ordered_ids_sha256"],
        "data_counts": result["data_counts"], "devices": devices,
        "_result": result, "_protocol": protocol, "_contract": contract, "_publications": publications,
        "_three_way": raw_three_way, "_options": options}


def collect(root, plan_path, controls_path, benchmark_summary_path=None):
    root, plan_path, controls_path = Path(root), Path(plan_path), Path(controls_path)
    benchmark_summary_path = Path(benchmark_summary_path or ROOT.parent / "reports/batch_optimization_20261027/batch_summary.json")
    required = [root / "scientific-config.json", root / "job-source.sbatch", root / "all-phases.done", plan_path, controls_path, benchmark_summary_path]
    for variant in METHODS:
        required.append(root / variant / "both-phases.done")
        for phase in (1, 2):
            phase_dir = root / variant / f"phase-{phase}"
            required += [phase_dir / name for name in ("server-device.json", "learner-device.json", "separate-gcds-verified.json")]
            learner = phase_dir / "learner"
            required += [learner / name for name in ("result.json", "run_protocol.json", "invocation.json", "environment.json",
                "resolved_training_options.json", "optimizer_at_train_begin.json", "updates.jsonl", "performance.jsonl",
                "training_probabilities.jsonl", "reward_audit.jsonl", "sync/sync.jsonl", "sync/rollouts.jsonl",
                f"sync/three_way_v{phase-1}.json", f"sync/three_way_v{phase}.json")]
            required.append(learner / ("capacity_result.json" if phase == 1 else "resume/resume_audit.jsonl"))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {"status": "incomplete", "passed": False, "missing_files": missing, "root": str(root.resolve())}
    plan, config, controls = read(plan_path), read(root / "scientific-config.json"), read(controls_path)
    benchmark = read(benchmark_summary_path)
    require(benchmark["passed"] is True and benchmark["status"] == "verified", "Initialization reference matrix was not audited successfully")
    require(all(plan[key] == value for key, value in {"micro_batch_size": 8, "gradient_accumulation_steps": 16,
        "generation_batch_size": 128, "num_generations": 8, "answers_per_update": 128, "max_updates_per_method": 2,
        "max_gcds": 2, "max_allocation_seconds": 1800, "formal400_training": False}.items()) and set(plan["methods"]) == set(METHODS),
        "The launch plan does not describe the selected bounded schedule")
    require(plan["phase1_capacity"] == {"prompt": 1024, "completion": 4096, "compute_entropy": True,
            "adam_moment_bytes_reserved": True, "parameter_updates": 0}, "Declared entropy capacity plan differs")
    sha_mapping(plan["source_files_sha256"], "approved source snapshot")
    for path, source in ((root / "scientific-config.json", "vllm_modern/formal_config.json"),
                         (root / "job-source.sbatch", "vllm_modern/jobs/schedule_acceptance.sbatch"),
                         (controls_path, "vllm_modern/fixtures/probability_controls.json")):
        require(file_hash(path) == plan["source_files_sha256"][source], "Collected config/job/controls differ from the approved source snapshot")
    methods, pair_contract, reference_options, reference_dataset = {}, None, None, None
    for variant in METHODS:
        phases = [phase_check(root, variant, phase, config, plan, controls) for phase in (1, 2)]
        first, second = phases
        require(first["_contract"] == second["_contract"], "Save/restore immutable contract differs")
        original = first["_contract"]["original_policy_initialization"]
        require({key: value for key, value in original.items() if key != "checkpoint"}
                == benchmark["initialization_inventories_by_method"][variant], "Checkpoint25 inventory differs from the audited benchmark matrix")
        require(all(first["_contract"][name] == benchmark["common_protocol"][name] for name in
            ("scientific_configuration_sha256", "train_data_sha256", "base_files_sha256", "uv_lock_sha256", "reward")),
            "Scientific inputs differ from the audited benchmark matrix")
        common = {key: value for key, value in first["_contract"].items() if key not in ("variant", "original_policy_initialization")}
        require(pair_contract is None or common == pair_contract, "Methods used different source/config/runtime/frozen data")
        pair_contract = common
        require(second["_protocol"]["resume_input"] == first["saved_inventory"], "Restored checkpoint1 differs from the saved inventory")
        a, b = first["_result"]["saved_policy_matches"], second["_result"]["loaded_policy_matches"]
        require(a == b, "Runtime loaded policy differs from saved checkpoint1")
        restored = second["_result"]["restore_audit"]
        records = rows(root / variant / "phase-2/learner/resume/resume_audit.jsonl")
        require(restored["passed"] is True and restored["missing"] == [] and restored["records"] == records
                and len(records) == 5 and {row["kind"] for row in records} == {"policy", "optimizer", "scheduler", "rng", "training_state"},
                "The five actual stock restore hooks were not all verified")
        checkpoint = first["saved_inventory"]["checkpoint"]
        require(restored["checkpoints"] == [checkpoint] and all(row["passed"] is True and row["checkpoint"] == checkpoint for row in records),
                "Restore hooks came from different/incomplete checkpoints")
        by_kind = {row["kind"]: row for row in records}
        require(by_kind["policy"]["bitwise_equal"] is True and by_kind["policy"]["tensor_count"] == a["tensor_count"]
                and by_kind["policy"]["state_sha256"] == a["state_sha256"]
                and by_kind["optimizer"]["state_sha256"] == second["optimizer_at_train_begin"]["state_sha256"],
                "Actual loaded policy/optimizer digests differ from their runtime proofs")
        for kind in ("optimizer", "scheduler", "rng"):
            require(isinstance(by_kind[kind]["state_sha256"], str) and SHA.fullmatch(by_kind[kind]["state_sha256"]), "Missing restored state digest")
        require(by_kind["rng"]["keys"] == ["cpu", "cuda", "numpy", "python"] and by_kind["rng"]["server_rng_restored"] is False,
                "Learner RNG scope differs")
        state = by_kind["training_state"]
        require(state["global_step"] == 1 and state["epochs_trained"] == 0 and state["microbatches_skipped"] == 16
                and state["grpo_buffer_boundary_aligned"] is True and state["data_order_verified_here"] is False,
                "Stock skip16/buffer-boundary restore was not verified")
        for field in ("canonical_tensor_sha256", "resident_checksums"):
            require(first["_publications"][-1][field] == second["_publications"][0][field], "Restored v1 canonical/resident bytes differ")
        require(second["_result"]["restored_v1_canonical_and_resident_exact"] is True, "Runtime v1 repeat proof missing")
        require(not set(first["actual_questions"]["question_ids"]) & set(second["actual_questions"]["question_ids"]), "Phase2 repeated phase1 question exposure")
        for phase in phases:
            require(reference_options is None or reference_options == phase["_options"], "Resolved method/phase options differ")
            reference_options = phase["_options"]
            identity = (phase["dataset_ordered_ids_sha256"], phase["data_counts"])
            require(reference_dataset is None or reference_dataset == identity, "Full dataset/order differs across methods/phases")
            reference_dataset = identity
        methods[variant] = {"status": "verified", "optimizer_updates": 2, "responses": 256, "distinct_questions": 32,
            "publications_including_restart": [0, 1, 1, 2], "generation_policy_versions": [0, 1], "request_seeds": [42, 50],
            "all_339_canonical_and_198_resident_exact": True, "restored_v1_weights_exact": True,
            "actual_stock_restore_kinds": sorted(by_kind), "stock_microbatches_skipped": 16,
            "runtime_restored_state_sha256": {kind: by_kind[kind]["state_sha256"] for kind in ("policy", "optimizer", "scheduler", "rng")},
            "v1_raw_probability_replay": exact_replay(first["_three_way"][-1], second["_three_way"][0]),
            "phases": [{key: value for key, value in phase.items() if not key.startswith("_")} for phase in phases]}
    for index in (0, 1):
        a, b = (methods[variant]["phases"][index]["actual_questions"] for variant in METHODS)
        require(a["question_ids"] == b["question_ids"] and a["prompt_tokens_by_question_sha256"] == b["prompt_tokens_by_question_sha256"],
                "Paired methods consumed different actual question/prompt exposure")
    provenance = {str(path.relative_to(root)) if path.is_relative_to(root) else str(path.resolve()): file_hash(path) for path in required}
    return {"schema_version": 1, "status": "verified", "passed": True, "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root.resolve()), "source_commit": plan["source_commit"], "schedule": SCHEDULE, "methods": methods,
        "paired_actual_question_and_prompt_exposure_equal": True, "immutable_contract_common_to_methods": pair_contract,
        "frozen_initialization_matches_audited_batch_matrix": True,
        "resolved_training_options": reference_options, "artifact_sha256": provenance,
        "analyzer_sha256": {name: file_hash(ROOT / name) for name in ("collect_schedule_acceptance.py", "summarize_batch_benchmark.py", "summarize_pair.py", "collect_acceptance.py")},
        "checkpoint_evidence_scope": "Inventory file hashes and actual runtime bitwise policy/optimizer/scheduler/RNG checks under the frozen source; offline collector does not load or rehash checkpoint tensor/pickle files",
        "offline_checkpoint_tensor_or_optimizer_reload_performed": False,
        "numerical_equivalence_claim": False, "stochastic_replay_claim": False, "performance_claim": False,
        "formal_training_started_by_this_collector": False}


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=ROOT.parent / "reports/batch_optimization_20261027/selected_schedule_plan.json")
    parser.add_argument("--fixed-controls", type=Path, default=ROOT / "fixtures/probability_controls.json")
    parser.add_argument("--benchmark-summary", type=Path, default=ROOT.parent / "reports/batch_optimization_20261027/batch_summary.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "schedule_acceptance_summary.json"
    try:
        summary = collect(args.root, args.plan, args.fixed_controls, args.benchmark_summary)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, OverflowError) as error:
        summary = {"status": "invalid", "passed": False, "root": str(args.root), "error": str(error)}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    print(json.dumps({"status": summary["status"], "passed": summary["passed"], "output": str(output),
        "missing_files": summary.get("missing_files"), "error": summary.get("error"),
        "v1_raw_probability_replay": {name: value["v1_raw_probability_replay"] for name, value in summary.get("methods", {}).items()}}, indent=2))
    return 0 if summary["status"] == "verified" else 1 if summary["status"] == "incomplete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
