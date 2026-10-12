#!/usr/bin/env python3
"""Independently audit a long-pilot JSON bundle on CPU, using only stdlib.

Missing external evidence is UNVERIFIED, never silently treated as passed.
No model, CUDA, torch import, pickle loading, or training source import occurs.
Python 3.6+ supports execution in Setonix's ordinary login-node Python.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys

EXECUTION_ONLY = {
    "runtime_limit_seconds", "checkpoint_interval", "checkpoint_reserve_seconds",
    "step_time_budget_seconds", "checkpoint_boundary", "resume_after_step",
    "status", "implementation_status", "runtime_estimation", "selection_status", "scope",
}
RULE = "verl_gsm8k_strict_1d367a4f_v1"
SOURCE_KEYS = (
    "configs/base_model.json", "geora_layers.py", "lora_layers.py",
    "scripts/check_grpo_continuation.py", "scripts/check_grpo_training.py",
    "scripts/geora_check_utils.py", "scripts/grpo_math.py", "scripts/grpo_rollout.py",
    "scripts/grpo_training_state.py", "scripts/gsm8k_data.py",
    "scripts/cached_training_rollout.py", "scripts/cached_rollout.py",
    "scripts/cached_block_score.py", "scripts/long_grpo_step.py",
    "scripts/long_capacity_check.py", "third_party/verl/gsm8k.py",
    "third_party/verl/provenance.json", "scripts/run_grpo_pilot.py", "scripts/pilot_runtime.py",
)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def config_hash(config):
    return canonical_hash({k: v for k, v in config.items() if k not in EXECUTION_ONLY})


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


class Audit:
    def __init__(self):
        self.checks = []
        self.steps = []

    def check(self, name, condition, **details):
        self.checks.append(dict(name=name, status="PASS" if condition else "FAIL", **details))

    def unverified(self, name, reason):
        self.checks.append(dict(name=name, status="UNVERIFIED", reason=reason))

    def note(self, name, **details):
        self.checks.append(dict(name=name, status="INFO", **details))


def safe_relative(root, relative):
    root = root.resolve()
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError("Path escapes evidence root: " + str(relative))
    return path


def audit_capacity(audit, report):
    cap = report.get("capacity_check")
    if not isinstance(cap, dict):
        audit.note("capacity_absent", reason="LoRA or resumed run may not execute this GeoRA-only gate")
        return
    audit.note("capacity_is_not_GRPO", status_in_report=cap.get("status"),
               optimizer_steps=cap.get("optimizer_steps"),
               scope="Synthetic teacher-forced -mean(logp), not reward, sampling, or task accuracy")
    if cap.get("status") != "passed":
        audit.unverified("capacity_not_completed", "No claim of completed full capacity is made")
        return
    audit.check("capacity_scope_and_dimensions", cap.get("passed") is True
                and cap.get("full_1024_prompt_4096_completion_gate") is True
                and cap.get("cpu_fixture_only") is False
                and cap.get("synthetic_prompt_tokens") == 1024
                and cap.get("synthetic_completion_tokens") == 4096
                and cap.get("fixed_input_width") == 5120
                and cap.get("expected_final_cache_length") == 5119
                and cap.get("batch_size") == 1 and cap.get("checkpoint_block_tokens") == 16)
    audit.check("capacity_zero_updates_one_backward", cap.get("optimizer_steps") == 0
                and cap.get("backward_calls") == 1)
    audit.check("capacity_finite_FP32_values_gradients", cap.get("all_selected_logps_finite") is True
                and cap.get("selected_logp_dtype") == "torch.float32"
                and cap.get("expected_adapter_tensor_count") == 392
                and cap.get("finite_FP32_adapter_gradient_tensors") == 392
                and cap.get("adapter_parameter_count") == 18464768
                and finite(cap.get("adapter_gradient_l2_norm")) and cap["adapter_gradient_l2_norm"] > 0)
    memory = cap.get("peak_allocated_memory_gib")
    limit = cap.get("memory_guard_gib")
    audit.check("capacity_peak_within_guard", finite(memory) and finite(limit)
                and 0 < limit <= 55 and memory <= limit, peak_allocated_gib=memory, limit_gib=limit)
    stats = cap.get("checkpoint_statistics", {})
    audit.check("capacity_full_cache_replay", stats.get("checkpoint_blocks") == 256
                and stats.get("recompute_blocks") == 256
                and stats.get("forward_cache_forwards") == 4096
                and stats.get("recompute_cache_forwards") == 4096
                and stats.get("autocast_states_match") is True
                and stats.get("boundary_cache_inputs_are_explicit") is True
                and stats.get("runtime_guard_forward_checks", 0) >= 8192
                and stats.get("runtime_guard_recompute_checks", 0) >= 8192)
    # Qwen 28 layers, 2 KV heads, head_dim128, both K and V, batch1.
    elements = stats.get("max_boundary_cache_tensor_elements")
    audit.check("capacity_observed_final_cache_length", elements == 2 * 28 * 2 * 128 * 5119,
                cache_length_from_elements=None if not finite(elements) else elements / (2 * 28 * 2 * 128))
    restored = ("parameters_unchanged", "gradients_cleared", "CPU_rng_restored",
                "CUDA_rng_restored", "model_training_mode_unchanged")
    audit.check("capacity_caller_state_restored", all(cap.get(k) is True for k in restored))
    audit.note("capacity_duration", scoring_seconds=cap.get("scoring_seconds"),
               backward_seconds=cap.get("backward_seconds"), elapsed_seconds=cap.get("elapsed_seconds"))


def check_likelihood(audit, name, metrics, config):
    token = metrics.get("max_absolute_logp_difference")
    ratio = metrics.get("max_absolute_ratio_minus_one")
    sequences = metrics.get("sequence_logp_differences")
    sequences_valid = isinstance(sequences, list) and bool(sequences) and all(finite(x) for x in sequences)
    audit.check(name, finite(token) and finite(ratio) and sequences_valid
                and token <= config["max_absolute_logp_mismatch"]
                and ratio <= config["max_absolute_ratio_mismatch"]
                and max(abs(x) for x in sequences) <= config["max_absolute_sequence_logp_mismatch"],
                max_token_logp_error=token, max_probability_ratio_error=ratio,
                max_sequence_logp_error=max(abs(x) for x in sequences) if sequences_valid else None)


def audit_step(audit, record, reference, report, config, all_checks):
    step = record.get("logical_step")
    prefix = "step_" + str(step)
    group_size = config["answers_per_question"]
    questions = config["questions_per_batch"]
    expected_answers = group_size * questions
    answers = record.get("answers", [])
    train_ids = report["identity"]["train_question_ids"]
    expected_questions = [train_ids[((step - 1) * questions + k) % len(train_ids)]
                          for k in range(questions)]
    audit.check(prefix + "_record_identity", step == reference.get("logical_step")
                and record.get("pilot_run_id") == report.get("pilot_run_id")
                and record.get("label") == prefix and record.get("question_ids") == expected_questions)
    audit.check(prefix + "_complete_eight_answer_group", len(answers) == expected_answers
                and group_size == 8 and questions == 1)
    rewards = record.get("rewards")
    advantages = record.get("advantages")
    reward_shape = (isinstance(rewards, list) and len(rewards) == questions
                    and all(isinstance(g, list) and len(g) == group_size for g in rewards))
    advantage_shape = isinstance(advantages, list) and len(advantages) == expected_answers
    audit.check(prefix + "_reward_advantage_layout", reward_shape and advantage_shape)
    if not reward_shape or not advantage_shape or len(answers) != expected_answers:
        return
    flat_rewards = [value for group in rewards for value in group]
    audit.check(prefix + "_binary_finite_rewards", all(value in (0, 1) and finite(value) for value in flat_rewards))
    expected_advantages = []
    for group in rewards:
        mean = sum(group) / len(group)
        std = math.sqrt(sum((value - mean) ** 2 for value in group) / len(group))
        expected_advantages.extend((value - mean) / (std + config["advantage_epsilon"]) for value in group)
    error = max(abs(actual - expected) if finite(actual) else float("inf")
                for actual, expected in zip(advantages, expected_advantages))
    audit.check(prefix + "_independent_population_advantages", error < 1e-5, max_error=error)
    tokens = 0
    for index, answer in enumerate(answers):
        name = prefix + "_answer_" + str(index + 1)
        ids = answer.get("token_ids", [])
        count = answer.get("token_count")
        audit.check(name + "_length_eos", isinstance(ids, list) and integer(count)
                    and len(ids) == count and 1 <= count <= config["max_new_tokens"]
                    and isinstance(answer.get("ended_with_eos"), bool)
                    and answer.get("truncated") == (not answer.get("ended_with_eos")))
        tokens += count if integer(count) else 0
        matches = re.findall(r"#### (\-?[0-9\.\,]+)", answer.get("completion", "")[-300:])
        predicted = matches[-1].replace(",", "") if matches else None
        target = answer.get("target_answer")
        score = float(predicted is not None and predicted == target)
        training_reward = score if answer.get("ended_with_eos") else config["truncated_answer_reward"]
        audit.check(name + "_independent_verl_reward", answer.get("answer_rule") == RULE
                    and isinstance(target, str)
                    and answer.get("predicted_answer") == predicted and answer.get("reward") == score
                    and answer.get("training_reward") == training_reward
                    and flat_rewards[index] == training_reward)
        audit.check(name + "_question_membership", answer.get("question_id") == expected_questions[index // group_size])
        for suffix in ("causal_completion_mask", "sampled_and_old_likelihood", "current_and_old_likelihood",
                       "finite_loss", "functional_cache_blocks_recomputed_with_matching_autocast",
                       "accumulated_FP32_AB_gradients_finite"):
            matching = [c for c in all_checks if c.get("name") == name + "_" + suffix]
            audit.check(name + "_recorded_gate_" + suffix, bool(matching) and all(c.get("passed") is True for c in matching))
            if matching and suffix.endswith("likelihood"):
                check_likelihood(audit, name + "_gate_metrics_" + suffix, matching[-1], config)
        if step == 1:
            matching = [c for c in all_checks if c.get("name") == name + "_initial_policy_matches_original_reference"]
            audit.check(name + "_initial_reference_gate", bool(matching) and all(c.get("passed") is True for c in matching))
            if matching:
                check_likelihood(audit, name + "_initial_reference_metrics", matching[-1], config)
        old = record.get("old_logp", [])
        behavior = record.get("behavior_logp", [])
        masks = record.get("completion_mask", [])
        ref_logps = record.get("reference_logp", [])
        layout = (len(old) == expected_answers and len(behavior) == expected_answers
                  and len(masks) == expected_answers and len(ref_logps) == expected_answers
                  and all(len(arr[index]) == config["max_new_tokens"] for arr in (old, behavior, masks, ref_logps)))
        audit.check(name + "_stored_probability_layout", layout)
        if layout:
            mask = masks[index]
            audit.check(name + "_stored_contiguous_mask", mask == [j < count for j in range(config["max_new_tokens"])])
            valid = list(zip(old[index][:count], behavior[index][:count], ref_logps[index][:count]))
            values_finite = all(all(finite(value) for value in triple) for triple in valid)
            audit.check(name + "_stored_probabilities_finite", values_finite)
            if values_finite:
                differences = [a - b for a, b, _ in valid]
                metrics = dict(max_absolute_logp_difference=max(abs(x) for x in differences),
                               max_absolute_ratio_minus_one=max(abs(math.expm1(x)) for x in differences),
                               sequence_logp_differences=[math.fsum(differences)])
                check_likelihood(audit, name + "_independent_sampled_old_metrics", metrics, config)
    audit.check(prefix + "_token_accounting", record.get("valid_completion_tokens") == tokens
                and reference.get("valid_completion_tokens") == tokens)
    stats_cases = record.get("functional_block_checkpoint_statistics", [])
    audit.check(prefix + "_all_answers_checkpoint_stats", len(stats_cases) == expected_answers)
    for index, stats in enumerate(stats_cases):
        count = answers[index]["token_count"]
        blocks = (count + config["checkpoint_block_tokens"] - 1) // config["checkpoint_block_tokens"]
        audit.check(prefix + "_replay_answer_" + str(index + 1), stats.get("answer") == index + 1
                    and stats.get("checkpoint_blocks") == blocks and stats.get("recompute_blocks") == blocks
                    and stats.get("forward_cache_forwards") == count and stats.get("recompute_cache_forwards") == count
                    and stats.get("autocast_states_match") is True
                    and stats.get("boundary_cache_inputs_are_explicit") is True
                    and stats.get("runtime_guard_forward_checks", 0) >= 2 * count
                    and stats.get("runtime_guard_recompute_checks", 0) >= 2 * count)
    audit.check(prefix + "_single_update_geometry", record.get("generation_microbatch_size") == 1
                and record.get("gradient_microbatch_size") == 1
                and record.get("gradient_accumulation_answer_count") == expected_answers
                and record.get("mean_of_per_answer_token_means") is True)
    audit.check(prefix + "_finite_loss_update_metrics", all(finite(record.get(k)) for k in
                ("loss", "policy_loss", "sampled_old_token_k3", "clip_fraction", "gradient_norm_before_clipping"))
                and integer(record.get("changed_parameter_tensors")))
    audit.steps.append(dict(step=step, reward=sum(flat_rewards) / expected_answers,
        tokens=tokens, gradient_norm=record.get("gradient_norm_before_clipping"),
        changed_parameter_tensors=record.get("changed_parameter_tensors"),
        elapsed_seconds=record.get("step_elapsed_seconds"),
        equal_reward_group=all(value == flat_rewards[0] for value in flat_rewards)))


def run(arguments):
    audit = Audit()
    report_path = arguments.report.resolve()
    report = json.loads(report_path.read_text())
    config = report.get("config", {})
    identity = report.get("identity", {})
    audit.check("long_pilot_schema", report.get("stage") == "gsm8k_bounded_grpo_pilot"
                and config.get("rollout_backend") == "native_cached_block_checkpointed"
                and isinstance(identity, dict))
    audit.check("reported_training_config_hash", identity.get("training_config_sha256") == config_hash(config))
    if arguments.executed_config:
        executed = json.loads(arguments.executed_config.read_text())
        audit.check("executed_config_training_hash", config_hash(executed) == identity.get("training_config_sha256"))
        audit.note("executed_config_bytes", sha256=file_hash(arguments.executed_config),
                   reason="Runner records semantic training hash, not a separate config-file-byte hash")
    else:
        audit.unverified("external_executed_config", "No independently captured executed_config.json provided")
    audit.check("configured_pinned_answer_rule", report.get("answer_rule") == RULE and config.get("answer_rule") == RULE)
    provenance = report.get("provenance", {})
    audit.check("report_provenance_identity", provenance.get("method") == identity.get("method")
                and provenance.get("training_config_sha256") == identity.get("training_config_sha256")
                and provenance.get("dataset_manifest_sha256") == identity.get("dataset_manifest_sha256")
                and sha256(identity.get("train_rows_sha256"))
                and len(identity.get("train_question_ids", [])) == config.get("train_question_limit"))
    precision = report.get("precision", {})
    audit.check("reported_precision_protocol", precision.get("frozen_weights") == "BF16"
                and precision.get("factors") == "FP32" and precision.get("initial_GeoRA_buffers") == "FP32"
                and precision.get("adapter_branch") == "FP32" and precision.get("log_softmax_and_loss") == "FP32"
                and precision.get("optimizer_moments") == "FP32" and precision.get("checkpoint_adapter_storage") == "FP32")
    sources = identity.get("source_sha256", {})
    audit.check("source_digest_manifest", set(sources) == set(SOURCE_KEYS) and all(sha256(value) for value in sources.values()))
    if arguments.source_root:
        for relative, expected in sources.items():
            path = safe_relative(arguments.source_root, relative)
            audit.check("source_bytes_" + relative, path.is_file() and file_hash(path) == expected)
    else:
        audit.unverified("executed_source_bytes", "Source hashes have valid structure; bytes were not supplied for verification")
    audit_capacity(audit, report)
    all_checks = report.get("checks", [])
    audit.check("no_recorded_failed_gates", all(c.get("passed") is True for c in all_checks), recorded_count=len(all_checks))
    steps = report.get("logical_optimizer_steps")
    references = report.get("active_step_records", [])
    audit.check("logical_steps_and_active_records", integer(steps) and len(references) == steps
                and [r.get("logical_step") for r in references] == list(range(1, (steps or 0) + 1)))
    if arguments.minimum_steps is not None:
        audit.check("requested_minimum_steps", integer(steps) and steps >= arguments.minimum_steps,
                    observed=steps, required=arguments.minimum_steps)
    if not integer(steps):
        steps = 0
    audit.check("data_cursor", report.get("next_data_position") == steps * config.get("questions_per_batch", 0))
    audit.check("optimizer_logical_accounting", integer(report.get("physical_optimizer_step_calls"))
                and report["physical_optimizer_step_calls"] >= steps)
    audit.check("training_backward_excludes_capacity", integer(report.get("physical_backward_calls"))
                and report["physical_backward_calls"] >= steps * config.get("answers_per_question", 0) * config.get("questions_per_batch", 0))
    extras = report.get("superseded_step_records", []) or report.get("discarded_rollouts", [])
    finished_boundary = report.get("status") in ("paused_at_boundary", "completed")
    if finished_boundary and not extras:
        audit.check("boundary_optimizer_count_exact", report.get("physical_optimizer_step_calls") == steps)
        audit.check("boundary_backward_count_exact", report.get("physical_backward_calls") == steps * config["answers_per_question"] * config["questions_per_batch"])
    elif not finished_boundary:
        audit.unverified("final_boundary_not_reached", "Running/failed report cannot certify completion from partial backward calls")
    records_root = (arguments.records_root or report_path.parent).resolve()
    verified_tokens = 0
    records_verified = 0
    for reference in references:
        path = safe_relative(records_root, reference["relative_path"])
        if path.is_file():
            record_bytes = path.read_bytes()
        elif Path(str(path) + ".gz").is_file():
            # Archive-only compression; verify the original uncompressed bytes.
            with gzip.open(str(path) + ".gz", "rb") as handle:
                record_bytes = handle.read()
        else:
            audit.unverified("step_record_" + str(reference.get("logical_step")), "Missing external record: " + str(path))
            continue
        audit.check("record_bytes_" + str(reference["logical_step"]), hashlib.sha256(record_bytes).hexdigest() == reference.get("sha256"))
        record = json.loads(record_bytes)
        audit_step(audit, record, reference, report, config, all_checks)
        verified_tokens += record.get("valid_completion_tokens", 0)
        records_verified += 1
    if records_verified == len(references):
        audit.check("completed_answer_tokens", report.get("completed_answer_tokens") == verified_tokens)
    audit.check("physical_rollout_token_counts_monotonic", integer(report.get("physical_rollouts"))
                and report["physical_rollouts"] >= steps
                and integer(report.get("physical_completion_tokens"))
                and report["physical_completion_tokens"] >= report.get("completed_answer_tokens", 0))
    audits = report.get("immutability_audits", [])
    if finished_boundary:
        audit.check("finished_segment_has_freeze_audit", bool(audits))
    elif not audits:
        audit.unverified("post_segment_immutability", "Final frozen-state digest is only computed after a segment ends")
    for index, item in enumerate(audits):
        before, after = item.get("before"), item.get("after")
        digest_keys = {"frozen_policy", "initial_buffers", "frozen_reference"}
        digest_valid = (isinstance(before, dict) and set(before) == digest_keys
                        and all(sha256(value.get("sha256")) for value in before.values()))
        audit.check("immutable_segment_" + str(index + 1), item.get("unchanged") is True
                    and digest_valid and before == after
                    and before == report.get("initial_immutable_digests")
                    and item.get("frozen_gradients_absent") is True
                    and item.get("reference_gradients_absent") is True)
    if report.get("capacity_memory_peak_included_in_process_peak"):
        audit.note("process_peak_includes_synthetic_capacity", peak_gib=report.get("peak_allocated_memory_gib"))
    if finite(report.get("peak_allocated_memory_gib")):
        audit.check("final_process_peak_within_55GiB", report["peak_allocated_memory_gib"] <= config["scoring_memory_guard_gib"])
    manifest_path = arguments.checkpoint_manifest
    if manifest_path is None and report.get("latest_checkpoint"):
        manifest_path = records_root / "checkpoints" / Path(report["latest_checkpoint"]).name / "manifest.json"
    if manifest_path is not None and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        progress = manifest.get("progress", {})
        audit.check("checkpoint_manifest_boundary", manifest.get("status") == "complete"
                    and manifest.get("boundary") == "after_rollout_and_optional_update"
                    and progress.get("rollout_in_flight") is False)
        audit.check("checkpoint_training_identity", manifest.get("provenance") == report.get("provenance")
                    and manifest.get("provenance_sha256") == canonical_hash(report.get("provenance", {})))
        audit.check("checkpoint_step_cursor_records", progress.get("completed_optimizer_steps") == steps
                    and progress.get("next_data_position") == report.get("next_data_position")
                    and progress.get("active_step_records") == references
                    and progress.get("completed_answer_tokens") == report.get("completed_answer_tokens"))
        tensors = manifest.get("adapter_tensors", {})
        expected_tensors = 784 if identity.get("method") == "geora" else 392
        audit.check("checkpoint_declared_FP32_adapter", len(tensors) == expected_tensors
                    and all(t.get("dtype") == "torch.float32" for t in tensors.values())
                    and manifest.get("optimizer_type") == "AdamW"
                    and manifest.get("scheduler_kind") == "constant_v1")
        for name, descriptor in manifest.get("files", {}).items():
            path = safe_relative(manifest_path.parent, name)
            if path.is_file():
                audit.check("checkpoint_file_bytes_" + name, path.stat().st_size == descriptor.get("bytes")
                            and file_hash(path) == descriptor.get("sha256"))
            else:
                audit.unverified("checkpoint_file_" + name, "Manifest captured; tensor/state bytes were not supplied")
    else:
        audit.unverified("checkpoint_manifest", "No latest checkpoint manifest was supplied or found locally")
    counts = {status: sum(c["status"] == status for c in audit.checks) for status in ("PASS", "FAIL", "UNVERIFIED", "INFO")}
    status = "failed" if counts["FAIL"] else ("incomplete_evidence" if counts["UNVERIFIED"] else "passed")
    result = dict(status=status, counts=counts, scope="Independent stdlib CPU report/record audit; no model or task evaluation",
                  report_file=str(report_path), report_sha256=file_hash(report_path), method=identity.get("method"),
                  run_status=report.get("status"), logical_optimizer_steps=steps, step_summaries=audit.steps,
                  checks=audit.checks, script_sha256=file_hash(Path(__file__)))
    if arguments.output:
        arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    summary = {key: result[key] for key in ("status", "counts", "method", "run_status", "logical_optimizer_steps", "step_summaries")}
    summary["issues"] = [c for c in audit.checks if c["status"] in ("FAIL", "UNVERIFIED")][:20]
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 1 if counts["FAIL"] else (2 if arguments.strict_evidence and counts["UNVERIFIED"] else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("executed_config", nargs="?", type=Path)
    parser.add_argument("--records-root", type=Path, help="Directory containing segments/ and checkpoints/, defaults to report parent")
    parser.add_argument("--source-root", type=Path, help="Captured executed source tree, or unchanged executing repository")
    parser.add_argument("--checkpoint-manifest", type=Path)
    parser.add_argument("--minimum-steps", type=int)
    parser.add_argument("--strict-evidence", action="store_true", help="Exit2 when external evidence is missing; otherwise expose missing evidence in JSON")
    parser.add_argument("--output", type=Path, help="Write full audit JSON; terminal output remains compact")
    args = parser.parse_args()
    try:
        code = run(args)
    except (KeyError, ValueError, TypeError, IndexError, IOError, OverflowError) as error:
        print(json.dumps(dict(status="failed_schema_or_evidence", error_type=type(error).__name__, error=str(error)), indent=2))
        code = 1
    sys.exit(code)


if __name__ == "__main__":
    main()
