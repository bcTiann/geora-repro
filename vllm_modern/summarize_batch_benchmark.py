"""Revalidate collected batch probes; never launch jobs or select formal settings.

Usage: python summarize_batch_benchmark.py --root /collected/benchmark-root
Exit 0 means complete artifact checks passed, 1 means incomplete, and 2 means
invalid/mismatched artifacts. Telemetry remains optional and missing values are
null. All timings are host wall times from an instrumented three-update probe.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import struct


METHODS = ("lora", "geora_mixed")
CASE = re.compile(r"^(capacity|benchmark)-(lora|geora_mixed)-g(\d+)-m(\d+)$")
SHA = re.compile(r"^[0-9a-f]{64}$")
REWARD = "final_boxed_exact_rational_v2"
COMMON_FIELDS = ("scientific_configuration_sha256", "train_data_sha256", "base_revision",
                 "base_files_sha256", "source_sha256", "uv_lock_sha256", "reward", "precision",
                 "primary_eval_backend")
METRICS = ("gfx_utilization_percent", "memory_controller_utilization_percent", "vram_used_bytes",
           "vram_total_bytes", "vram_allocation_percent", "power_watts")


class InvalidEvidence(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InvalidEvidence(message)


def finite(value, label, *, minimum=None):
    require(type(value) in (int, float) and math.isfinite(value), f"Nonfinite/nonnumeric {label}")
    require(minimum is None or value >= minimum, f"Out-of-range {label}")
    return value


def equal_number(actual, recorded, label):
    require(math.isclose(actual, finite(recorded, label), rel_tol=2e-5, abs_tol=1e-6),
            f"Recorded statistic disagrees with raw data: {label}")


def float32(value):
    """Reproduce the saved torch.float32 subtraction/threshold, not float64."""
    return struct.unpack("!f", struct.pack("!f", value))[0]


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(InvalidEvidence(f"Invalid JSON number: {value}")))


def rows(path):
    return [json.loads(line, object_pairs_hook=_object,
                       parse_constant=lambda value: (_ for _ in ()).throw(InvalidEvidence(f"Invalid JSON number: {value}")))
            for line in Path(path).read_text().splitlines() if line.strip()]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def value_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def distribution(values):
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None, "p50": None, "p95": None, "p99": None}
    return {"count": len(values), "mean": statistics.fmean(values), "min": min(values), "max": max(values),
            "p50": percentile(values, .5), "p95": percentile(values, .95), "p99": percentile(values, .99)}


def schedule(micro, generation):
    require(micro > 0 and generation > 0 and generation % micro == 0
            and generation % 8 == 0 and 128 % generation == 0, "Invalid batch arithmetic")
    return {"micro_batch_size": micro, "generation_batch_size": generation, "answers_per_update": 128,
            "num_generations": 8, "learner_world_size": 1, "gradient_accumulation_steps": 128 // micro,
            "steps_per_generation": generation // micro, "rollouts_per_update": 128 // generation,
            "questions_per_update": 16}


def canonical_names():
    names = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for layer in range(28):
        prefix = f"model.layers.{layer}."
        names.update(prefix + f"self_attn.{part}_proj.weight" for part in ("q", "k", "v", "o"))
        names.update(prefix + f"self_attn.{part}_proj.bias" for part in ("q", "k", "v"))
        names.update(prefix + f"mlp.{part}_proj.weight" for part in ("gate", "up", "down"))
        names.update(prefix + part + ".weight" for part in ("input_layernorm", "post_attention_layernorm"))
    return names


def resident_names():
    names = {"model.embed_tokens.weight", "model.norm.weight"}
    for layer in range(28):
        prefix = f"model.layers.{layer}."
        names.update(prefix + part for part in ("self_attn.qkv_proj.weight", "self_attn.qkv_proj.bias",
            "self_attn.o_proj.weight", "mlp.gate_up_proj.weight", "mlp.down_proj.weight",
            "input_layernorm.weight", "post_attention_layernorm.weight"))
    return {"dp0:pp0:pcp0:tp0:" + name for name in names}


def sha_mapping(value, label, expected_names=None):
    require(isinstance(value, dict) and value and all(isinstance(key, str) and isinstance(digest, str)
            and SHA.fullmatch(digest) for key, digest in value.items()), f"Invalid hash map: {label}")
    require(expected_names is None or set(value) == expected_names, f"Incomplete tensor coverage: {label}")


def protocol_common(protocol):
    require(all(key in protocol for key in COMMON_FIELDS), "Missing immutable protocol fields")
    require(protocol["reward"] == REWARD, "Reward contract changed")
    for name in ("scientific_configuration_sha256", "train_data_sha256", "uv_lock_sha256"):
        require(isinstance(protocol[name], str) and SHA.fullmatch(protocol[name]), f"Invalid protocol hash: {name}")
    sha_mapping(protocol["source_sha256"], "source_sha256")
    sha_mapping(protocol["base_files_sha256"], "base_files_sha256")
    return {key: protocol[key] for key in COMMON_FIELDS}


def initialization_identity(protocol, variant):
    initial = protocol.get("initialization_checkpoint")
    require(isinstance(initial, dict) and initial.get("global_step") == 25 and initial.get("variant") == variant,
            "Every probe must initialize its policy from a declared checkpoint25")
    required_files = {"trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"}
    required_files.update({"adapter_model.safetensors", "adapter_config.json"} if variant == "lora"
                          else {"geora.safetensors", "geora_config.json"})
    sha_mapping(initial.get("files_sha256"), "initialization checkpoint", required_files)
    return {key: value for key, value in initial.items() if key != "checkpoint"}


def normalize_bdf(value):
    match = re.fullmatch(r"([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])", str(value))
    require(match is not None, f"Invalid physical BDF: {value}")
    domain, bus, device, function = (int(part, 16) for part in match.groups())
    return f"{domain:04x}:{bus:02x}:{device:02x}.{function:x}"


def telemetry(path, expected_bdf):
    if not path.is_file():
        return {"status": "missing", "samples": 0, "availability": None,
                "metrics": {name: distribution([]) for name in METRICS},
                "power_readings_informative": False, "power_readings_warning": "Power telemetry is missing",
                "power_scope": "Socket semantics unknown; no readings available; never sum across roles"}
    records = rows(path)
    samples = [row for row in records if row.get("event") == "sample"]
    for row in records:
        if row.get("event") == "monitor_start":
            require(normalize_bdf(row["target"]["physical_pci_bdf"]) == expected_bdf,
                    "GPU monitor identity differs from assigned role")
    for row in samples:
        require(normalize_bdf(row["physical_pci_bdf"]) == expected_bdf, "GPU sample belongs to another GCD")
    def summarize_subset(subset):
        output = {}
        for name in METRICS:
            values, invalid = [], 0
            for row in subset:
                value = row.get("metrics", {}).get(name)
                if value is None:
                    continue
                if (type(value) not in (int, float) or not math.isfinite(value) or value < 0
                        or (name.endswith("percent") and value > 100)):
                    invalid += 1
                else:
                    values.append(value)
            output[name] = {**distribution(values), "missing_count": len(subset) - len(values),
                            "invalid_readings_excluded": invalid,
                            "availability_fraction": len(values) / len(subset) if subset else None}
        return output
    stages = defaultdict(list)
    for row in samples:
        stages[str(row.get("stage"))].append(row)
    timestamps = [finite(row["epoch_time"], "telemetry timestamp") for row in samples]
    gaps = [b-a for a, b in zip(timestamps, timestamps[1:])]
    require(all(gap >= 0 for gap in gaps), "Telemetry timestamps moved backwards")
    statuses = dict(Counter(row.get("status", "missing") for row in samples))
    metrics = summarize_subset(samples)
    power_limits = [row.get("raw", {}).get("power_info", {}).get("power_limit") for row in samples
                    if isinstance(row.get("raw", {}).get("power_info"), dict)]
    power_limits = [value for value in power_limits if type(value) in (int, float) and math.isfinite(value) and value >= 0]
    power = metrics["power_watts"]
    informative = power["count"] > 0 and power["max"] > 0
    if not power["count"]:
        power_warning = "No usable numeric power readings; no energy or efficiency inference"
    elif power["max"] == 0:
        power_warning = "All recorded power readings are zero; likely unsupported/driver-zero telemetry, unusable for energy or efficiency inference"
        if power_limits and max(power_limits) == 0:
            power_warning += "; recorded power_limit is also zero"
        if metrics["gfx_utilization_percent"]["count"] and metrics["gfx_utilization_percent"]["max"] > 0:
            power_warning += "; GFX activity was observed despite the zero power readings"
    else:
        power_warning = None
    return {"status": "available" if samples else "unavailable", "samples": len(samples),
            "sample_status_counts": statuses, "metrics": metrics,
            "power_readings_informative": informative, "power_readings_warning": power_warning,
            "raw_power_limit_readings": distribution(power_limits),
            "by_learner_stage": {key: {"samples": len(value), "metrics": summarize_subset(value)}
                                 for key, value in stages.items()},
            "sample_gap_seconds": distribution(gaps),
            "power_source_fields": sorted({str(row.get("power_source_field")) for row in samples}),
            "power_scope": sorted({str(row.get("power_scope")) for row in samples}),
            "errors": [row for row in records if row.get("event") in ("monitor_unavailable", "monitor_shutdown_error")],
            "metric_error_counts": dict(Counter(name for row in samples for name in row.get("errors", {}))),
            "interpretation": "Sample-weighted summaries, not time-integrated utilization or energy; socket power is reported per role and is never summed"}


def role_telemetry(case):
    identity = {}
    for role in ("server", "learner"):
        row = read(case / f"{role}-device.json")
        require(row.get("visible_device_count") == 1, "A role did not see exactly one GCD")
        identity[role] = row
    require(identity["server"]["hostname"] == identity["learner"]["hostname"], "Roles are on different nodes")
    bdfs = {role: normalize_bdf(row["physical_pci_bdf"]) for role, row in identity.items()}
    require(bdfs["server"] != bdfs["learner"], "Roles share the same physical GCD")
    require(read(case / "distinct-gcds.json").get("passed") is True, "Missing physical GCD separation check")
    separation = read(case / "distinct-gcds.json")
    require(all(separation.get(role) == row for role, row in identity.items()), "GCD separation identity changed")
    return {role: {"physical_pci_bdf": bdfs[role], "hostname": identity[role]["hostname"],
                   **telemetry(case / f"{role}-gpu.jsonl", bdfs[role])} for role in bdfs}


def stage_times(performance):
    overall, per_step = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for row in performance:
        require(isinstance(row.get("stage"), str) and isinstance(row.get("nested_in"), str), "Invalid timed stage")
        seconds = finite(row["seconds"], "stage seconds", minimum=0)
        overall[row["stage"]].append(seconds)
        if "optimizer_step" in row:
            require(type(row["optimizer_step"]) is int and 0 <= row["optimizer_step"] <= 3,
                    "Unexpected timed optimizer version")
            per_step[row["optimizer_step"]][row["stage"]].append(seconds)
    def compact(mapping):
        return {name: {"calls": len(values), "inclusive_seconds": sum(values)} for name, values in mapping.items()}
    return {"overall": compact(overall), "by_policy_version": {str(key): compact(value) for key, value in per_step.items()},
            "semantics": "All stages are inclusive host wall time; nested stages must not be summed. Microstep includes preparation/backward; residual is not pure optimizer time. CUDA fences and JSON I/O instrument the workload."}


def reward_summary(rewards, probabilities):
    status_counts, group_counts = Counter(), Counter()
    scored_seconds = 0.0
    for reward, probability in zip(rewards, probabilities, strict=True):
        require(reward.get("reward_contract") == REWARD, "Reward audit changed scoring contract")
        require(reward["responses"] == probability["responses"], "Reward response count mismatch")
        require(reward.get("question_ids") == probability["question_ids"], "Reward IDs differ from probability IDs")
        require(reward.get("global_step") in (None, probability["optimizer_step"]), "Reward policy version mismatch")
        counts = reward["status_counts"]
        require(set(counts) == {"correct", "incorrect", "missing_or_incomplete_boxed", "invalid_prediction"}, "Unknown reward status")
        require(all(type(value) is int and value >= 0 for value in counts.values())
                and sum(counts.values()) == reward["responses"], "Reward statuses do not cover all responses")
        groups = reward["grouping"]
        require(groups.get("status") == "validated_consecutive_groups", "Reward groups were not identity-validated")
        require(groups["group_count"] == reward["responses"] // 8 == len(groups["groups"]), "Wrong reward group count")
        local = Counter()
        for index, group in enumerate(groups["groups"]):
            require(group["question_id"] == reward["question_ids"][index*8] and group["responses"] == 8,
                    "Reward group identity differs")
            mean = finite(group["reward_mean"], "group reward mean", minimum=0)
            require(mean <= 1 and math.isclose(mean * 8, round(mean * 8)), "Nonbinary group reward aggregate")
            require(group["all_zero"] == (mean == 0) and group["all_one"] == (mean == 1)
                    and group["zero_std"] == (mean in (0, 1)), "Inconsistent zero-variance group")
            local.update({"groups": 1, "all_zero": int(group["all_zero"]), "all_one": int(group["all_one"]),
                          "zero_std": int(group["zero_std"]), "correct_responses": round(mean * 8)})
        require(local["correct_responses"] == counts["correct"], "Reward means and statuses disagree")
        for name in ("all_zero", "all_one", "zero_std"):
            equal_number(local[name] / local["groups"], groups[name + "_fraction"], name + " fraction")
        equal_number(counts["correct"] / reward["responses"], reward["reward_mean"], "reward mean")
        status_counts.update(counts)
        group_counts.update(local)
        scored_seconds += finite(reward["scoring_seconds"], "reward scoring seconds", minimum=0)
    require(group_counts["groups"] == 48 and sum(status_counts.values()) == 384, "Incomplete reward audit")
    return {"status_counts": dict(status_counts), "reward_mean": status_counts["correct"] / 384,
            "groups": group_counts["groups"], "zero_std_fraction": group_counts["zero_std"] / 48,
            "all_zero_fraction": group_counts["all_zero"] / 48, "all_one_fraction": group_counts["all_one"] / 48,
            "scorer_only_seconds": scored_seconds,
            "scope": "Frozen reward observations, not an independent task-accuracy evaluation"}


def benchmark_case(case, variant, micro, generation):
    learner = case / "learner"
    result, protocol = read(learner / "result.json"), read(learner / "run_protocol.json")
    invocation = read(learner / "invocation.json")
    expected = schedule(micro, generation)
    require(result.get("completed") is True and result.get("mode") == protocol.get("mode") == "benchmark", "Incomplete/nonbenchmark result")
    require(result["variant"] == protocol["variant"] == variant, "Variant mismatch")
    require(result["schedule"] == protocol["schedule"] == expected, "Schedule differs from actual case")
    require(protocol["max_optimizer_steps"] == invocation["max_steps"] == 3, "Wrong benchmark step budget")
    require(invocation.get("resume_checkpoint") is None and invocation.get("initialization_checkpoint"), "Benchmark did not declare a fresh optimizer initialization")
    require(invocation["mode"] == "benchmark" and invocation["variant"] == variant
            and invocation["micro_batch_size"] == micro and invocation["generation_batch_size"] == generation, "Invocation/case mismatch")
    require(result["global_step"] == result["new_optimizer_updates"] == 3 and result["responses"] == 384, "Wrong update/response count")
    require(result["frozen_policy_unchanged"] is True and result.get("checkpoint_saved") is False, "Frozen state or fresh benchmark contract failed")
    require(result.get("resume_audit") is None, "Unexpected resumed optimizer in a benchmark")
    require(result["parameters"]["target_layer_count"] == 196 and result["parameters"]["trainable_tensors"] == 392,
            "Adapter coverage changed")
    common = protocol_common(protocol)
    initial = initialization_identity(protocol, variant)
    chosen = read(learner / "benchmark_questions.json")["ids"]
    require(len(chosen) == len(set(chosen)) == 48 and all(isinstance(item, str) and item for item in chosen), "Expected48 fixed unique questions")
    updates = rows(learner / "updates.jsonl")
    require([row["optimizer_step"] for row in updates] == [1, 2, 3] and updates == result["updates"], "Update records disagree")
    for row in updates:
        require(finite(row["update_wall_seconds_including_sync_and_diagnostics"], "update wall", minimum=0) > 0, "Zero update wall time")
        finite(row["post_clip_gradient_norm"], "gradient norm", minimum=0)
        finite(row["peak_allocated_gib"], "peak memory", minimum=0)
        finite(row["peak_reserved_gib"], "peak reserved memory", minimum=0)
    sync = rows(learner / "sync/sync.jsonl")
    require([row["policy_version"] for row in sync] == [0, 1, 2, 3], "Wrong publication versions")
    fixed_probability = []
    for publication in sync:
        require(publication["canonical_tensor_count"] == 339, "Wrong canonical tensor count")
        sha_mapping(publication["canonical_tensor_sha256"], "canonical tensors", canonical_names())
        resident = publication["resident_verification"]
        require(resident["passed"] is True and resident["canonical_tensor_count"] == 339
                and resident["resident_parameter_count"] == 198, "Resident verification failed")
        sha_mapping(resident["expected_resident_checksums"], "expected resident", resident_names())
        require(publication["resident_checksums"] == resident["expected_resident_checksums"], "Actual resident bytes differ from predicted packed policy")
        finite(publication["total_seconds"], "sync wall", minimum=0)
        for name, seconds in publication.get("phase_seconds", {}).items():
            finite(seconds, "sync phase " + name, minimum=0)
        fixed = publication["fixed_token_comparison"]
        if publication["policy_version"] in (0, 3):
            require(fixed["policy_version"] == publication["policy_version"] and fixed["structural_checks_passed"] is True
                    and fixed["numerical_accepted"] is None and fixed["acceptance_policy"] is None,
                    "Initial/final three-way audit was omitted or given an undeclared numerical acceptance policy")
            require(type(fixed["tokens"]) is int and fixed["tokens"] > 0
                    and type(fixed["sequences"]) is int and fixed["sequences"] > 0
                    and SHA.fullmatch(fixed["fixed_rows_sha256"]), "Incomplete fixed probability controls")
            require(set(fixed["comparisons"]) == {"learner_minus_hf_merged", "hf_merged_minus_vllm", "learner_minus_vllm"},
                    "Incomplete three-way comparisons")
            for comparison in fixed["comparisons"].values():
                require(comparison["tokens"] == fixed["tokens"], "Fixed probability token count differs")
                for field, value in comparison.items():
                    finite(value, "fixed probability " + field)
            fixed_probability.append(fixed)
        else:
            require(fixed.get("status") == "not_scheduled_at_this_version", "Middle fixed audit schedule changed")
    require(fixed_probability[0]["fixed_rows_sha256"] == fixed_probability[1]["fixed_rows_sha256"],
            "Initial/final fixed-token control rows differ")
    closure = result["sync_summary"]
    require(closure["passed"] is True and closure["all_resident_weights_verified"] is True
            and closure["sync_versions"] == [0, 1, 2, 3], "Final synchronization closure failed")
    require(closure["training_responses"] == 384 and closure["training_rollouts"] == 3 * expected["rollouts_per_update"], "Closure response count differs")
    probabilities = rows(learner / "training_probabilities.jsonl")
    rollouts = rows(learner / "sync/rollouts.jsonl")
    rewards = rows(learner / "reward_audit.jsonl")
    require(len(probabilities) == len(rollouts) == len(rewards) == 3 * expected["rollouts_per_update"], "Wrong rollout/audit count")
    versions = [step for step in range(3) for _ in range(expected["rollouts_per_update"])]
    require([row["optimizer_step"] for row in probabilities] == [row["policy_version"] for row in rollouts] == versions, "Rollout policy mismatch")
    exposure, per_step_exposure = Counter(), defaultdict(Counter)
    per_version = defaultdict(lambda: {"tokens": 0, "native_generation_seconds": 0.0, "responses": 0})
    prompt_by_question, sequence_lengths = {}, []
    deltas, active_positions, padded_positions, completion_lengths, flat_order = [], 0, 0, [], []
    for index, (probability, rollout) in enumerate(zip(probabilities, rollouts, strict=True)):
        require(probability["responses"] == rollout["responses"] == generation, "Wrong generation batch")
        require(probability["stock_token_truncate_verified"] is True and rollout["all_behavior_logprobs_finite"] is True, "Probability contract failed")
        ids = probability["question_ids"]
        require(len(ids) == len(rollout["completion_ids"]) == len(rollout["prompt_ids"]) == generation, "Misaligned rollout fields")
        for begin in range(0, generation, 8):
            require(len(set(ids[begin:begin+8])) == 1, "Question IDs do not form G8 groups")
            require(all(prompt == rollout["prompt_ids"][begin] for prompt in rollout["prompt_ids"][begin:begin+8]), "Grouped prompts differ")
            question, prompt = ids[begin], rollout["prompt_ids"][begin]
            require(isinstance(question, str) and question and prompt and len(prompt) <= 1024
                    and all(type(token) is int and token >= 0 for token in prompt), "Invalid/out-of-capacity prompt IDs")
            require(question not in prompt_by_question or prompt_by_question[question] == prompt,
                    "A question ID was associated with different input tokens")
            prompt_by_question[question] = prompt
        require(rollout["request_seed"] == 42 + 8 * index, "Request sampling seed schedule differs")
        require(rollout["rollout_in_update"] == index % expected["rollouts_per_update"], "Wrong rollout index within update")
        expected_generation = {"responses_per_rollout": generation, "num_generations": 8,
            "questions_per_rollout": generation // 8, "rollouts_per_update": expected["rollouts_per_update"],
            "responses_per_update": 128, "questions_per_update": 16, "training_micro_batch_size": micro,
            "gradient_accumulation_steps": expected["gradient_accumulation_steps"],
            "steps_per_generation": expected["steps_per_generation"], "batch_invariant_sampling_claim": False}
        for field, target in expected_generation.items():
            require(rollout["generation_schedule"][field] == target, "Native schedule metadata differs")
        exposure.update(ids)
        per_step_exposure[probability["optimizer_step"]].update(ids)
        flat_order.extend(ids)
        local_deltas, tokens = [], 0
        for prompt, completion, behavior, old, sampled in zip(rollout["prompt_ids"], rollout["completion_ids"], rollout["behavior_logprobs"],
                probability["old_learner_logprobs"], probability["behavior_logprobs"], strict=True):
            require(completion and len(completion) <= 4096 and len(completion) == len(behavior) == len(old) == len(sampled), "Omitted/out-of-capacity actual-token probabilities")
            require(all(type(token) is int and token >= 0 for token in completion), "Invalid completion token ID")
            completion_lengths.append(len(completion))
            sequence_lengths.append(len(prompt) + len(completion))
            for native, saved, current in zip(behavior, sampled, old, strict=True):
                finite(native, "native behavior logp")
                finite(saved, "TRL behavior logp")
                finite(current, "old learner logp")
                require(native <= 1e-6 and saved <= 1e-6 and current <= 1e-6, "Invalid positive log probability")
                equal_number(native, saved, "sampled behavior probability")
                local_deltas.append(float32(current - saved))
            tokens += len(completion)
        require(tokens == probability["active_tokens"] == rollout["tokens"], "Actual token count differs")
        require(type(probability["padded_completion_positions"]) is int and probability["padded_completion_positions"] >= tokens, "Invalid padding count")
        equal_number(statistics.fmean(abs(value) for value in local_deltas), probability["mean_abs_log_ratio"], "mean absolute log ratio")
        equal_number(max(abs(value) for value in local_deltas), probability["max_abs_log_ratio"], "max absolute log ratio")
        equal_number(sum(value > float32(math.log(2)) for value in local_deltas) / tokens,
                     probability["fraction_ratio_above_2"], "cap2 trigger fraction")
        equal_number(1 - tokens / probability["padded_completion_positions"], probability["padding_fraction"], "padding fraction")
        active_positions += tokens
        padded_positions += probability["padded_completion_positions"]
        deltas.extend(local_deltas)
        version = probability["optimizer_step"]
        per_version[version]["tokens"] += tokens
        per_version[version]["responses"] += generation
        per_version[version]["native_generation_seconds"] += finite(rollout["seconds"], "native generation wall", minimum=0)
    require(exposure == Counter({identifier: 8 for identifier in chosen}), "Actual question exposure differs from the fixed48-question set")
    require(all(row["generation_schedule"] == rollouts[0]["generation_schedule"] for row in rollouts + sync)
            and closure["generation_schedule"] == rollouts[0]["generation_schedule"],
            "Publication/rollout/closure native schedule differs")
    require(all(len(value) == 16 and set(value.values()) == {8} for value in per_step_exposure.values()),
            "An optimizer update did not use16 distinct G8 question groups")
    nonzero = [row["nonzero_advantage_responses"] for row in probabilities]
    require(all(type(value) is int and 0 <= value <= generation for value in nonzero), "Invalid advantage counts")
    require(result["all_measured_advantages_zero"] == all(value == 0 for value in nonzero), "Advantage closure differs")
    require(type(result["changed_adapter_tensors"]) is int and 0 <= result["changed_adapter_tensors"] <= 392
            and (result["changed_adapter_tensors"] > 0 or all(value == 0 for value in nonzero)),
            "Nonzero advantages did not change adapter tensors")
    require(result["actual_completion_tokens"] == active_positions, "Result total token count disagrees")
    performance = rows(learner / "performance.jsonl")
    timings = stage_times(performance)
    per_step = []
    for step, update in enumerate(updates):
        stages = timings["by_policy_version"].get(str(step), {})
        require(stages.get("vllm_rollout", {}).get("calls") == expected["rollouts_per_update"], "Missing rollout timing")
        require(stages.get("training_microstep", {}).get("calls") == expected["gradient_accumulation_steps"], "Wrong actual microstep count")
        require(stages.get("generate_and_score", {}).get("calls") == expected["rollouts_per_update"], "Wrong scoring batch count")
        require(per_version[step]["responses"] == 128, "A step did not consume128 responses")
        seconds = stages["vllm_rollout"]["inclusive_seconds"]
        require(seconds > 0, "Missing positive generation elapsed time")
        wall = update["update_wall_seconds_including_sync_and_diagnostics"]
        per_step.append({"optimizer_step": step + 1, "wall_seconds": wall, **per_version[step],
                         "vllm_call_seconds": seconds, "vllm_real_tokens_per_second": per_version[step]["tokens"] / seconds,
                         "update_real_tokens_per_second": per_version[step]["tokens"] / wall,
                         "learner_old_logprob_inclusive_seconds": stages.get("learner_old_logprob", {}).get("inclusive_seconds"),
                         "learner_grad_forward_inclusive_seconds": stages.get("learner_grad_forward", {}).get("inclusive_seconds"),
                         "training_microstep_inclusive_seconds": stages["training_microstep"]["inclusive_seconds"],
                         "peak_allocated_gib": update["peak_allocated_gib"], "peak_reserved_gib": update["peak_reserved_gib"],
                         "post_clip_gradient_norm": update["post_clip_gradient_norm"]})
    warm = per_step[1:]
    warm_tokens = sum(row["tokens"] for row in warm)
    warm_generation_seconds = sum(row["vllm_call_seconds"] for row in warm)
    warm_wall = sum(row["wall_seconds"] for row in warm)
    probability_stats = {"token_count": len(deltas), "absolute_log_ratio": distribution([abs(value) for value in deltas]),
                         "signed_log_ratio": distribution(deltas),
                         "cap2_trigger_count": sum(value > float32(math.log(2)) for value in deltas),
                         "cap2_trigger_fraction": sum(value > float32(math.log(2)) for value in deltas) / len(deltas),
                         "after_cap2_importance_ratio": distribution([math.exp(min(value, math.log(2))) for value in deltas]),
                         "scope": "Raw learner-minus-behavior logp before stock token_truncate cap2; no numerical equivalence claim"}
    result_summary = {"case": case.name, "status": "verified", "variant": variant, "schedule": expected,
        "optimizer_updates": 3, "responses": 384, "unique_questions": 48, "actual_tokens": active_positions,
        "frozen_policy_unchanged_runtime_assertion": True, "changed_adapter_tensors": result["changed_adapter_tensors"],
        "all_measured_advantages_zero": result["all_measured_advantages_zero"],
        "question_set_sha256": value_hash(sorted(chosen)), "actual_exposure_sha256": value_hash(sorted(exposure.items())),
        "actual_prompt_tokens_by_question_sha256": value_hash(prompt_by_question),
        "actual_question_order_sha256": value_hash(flat_order),
        "all_publications_resident_exact": True, "initial_policy_hashes": sync[0]["canonical_tensor_sha256"],
        "cold_step1": per_step[0], "warm_steps2_and3": warm,
        "warm_mean_wall_seconds": warm_wall / 2, "warm_real_tokens_per_second_in_vllm_call": warm_tokens / warm_generation_seconds,
        "warm_real_tokens_per_second_per_update_wall": warm_tokens / warm_wall,
        "warm_observations": 2, "stable_speedup_claim": False,
        "padding": {"active_tokens": active_positions, "padded_positions": padded_positions,
                    "fraction": 1 - active_positions / padded_positions},
        "completion_length": distribution(completion_lengths), "total_sequence_length": distribution(sequence_lengths),
        "probability": probability_stats,
        "fixed_rows_sha256": fixed_probability[0]["fixed_rows_sha256"],
        "fixed_three_way_probability": fixed_probability,
        "reward": reward_summary(rewards, probabilities), "inclusive_stage_timings": timings,
        "sync_wall_seconds_by_policy_version": [{"policy_version": row["policy_version"], "total_seconds": row["total_seconds"],
                                                    "phase_seconds": row.get("phase_seconds")} for row in sync],
        "gpu": role_telemetry(case),
        "workload_note": "Same48-question exposure, independently fresh optimizer; generated tokens and stochastic trajectories need not match across schedules",
        "process_seconds": finite(result["process_seconds"], "process wall", minimum=0)}
    return result_summary, common, initial


def capacity_case(case, variant, micro):
    root = case / "learner"
    result, protocol = read(root / "result.json"), read(root / "run_protocol.json")
    require(result.get("completed") is True and result.get("mode") == protocol.get("mode") == "capacity", "Incomplete capacity result")
    require(result["variant"] == protocol["variant"] == variant and result["micro_batch_size"] == micro, "Capacity variant/micro mismatch")
    require(protocol["schedule"] == schedule(micro, 128), "Capacity schedule mismatch")
    require(result["prompt_length"] == 1024 and result["completion_length"] == 4096 and result["padded_total_length"] == 5120,
            "Capacity probe did not test the declared worst length")
    require(result["parameter_updates"] == 0 and result["trainable_weights_unchanged"] is True, "Capacity probe updated parameters")
    require(type(result["passed"]) is bool, "Capacity pass status missing")
    if result["passed"]:
        require(result["optimizer_moment_bytes_reserved"] == 147718144, "Incomplete FP32 AdamW moment-byte reserve")
        require(result.get("error") is None, "Passing capacity probe records an error")
    else:
        error = str(result.get("error", ""))
        require("outofmemory" in error.lower() or "out of memory" in error.lower(), "Capacity failure is not an observed OOM")
    return {"case": case.name, "status": "verified", "outcome": "passed" if result["passed"] else "oom",
            "variant": variant, "micro_batch_size": micro, "passed": result["passed"],
            "peak_allocated_gib": finite(result["peak_allocated_gib"], "capacity memory", minimum=0),
            "peak_reserved_gib": finite(result["peak_reserved_gib"], "capacity memory", minimum=0),
            "seconds": finite(result["seconds"], "capacity seconds", minimum=0),
            "optimizer_moment_bytes_reserved": result["optimizer_moment_bytes_reserved"],
            "error": result.get("error"), "gpu": role_telemetry(case),
            "scope": "Synthetic full-length differentiable forward/backward with Adam moment-byte reserve, not a complete128-response optimizer update"}, protocol_common(protocol), initialization_identity(protocol, variant)


def collect(root):
    root = Path(root)
    plan = read(root / "optimization_plan.json")
    require(set(plan["benchmark"]["methods"]) == set(METHODS)
            and plan["benchmark"]["answers_per_update"] == 128
            and plan["benchmark"]["num_generations"] == 8
            and plan["benchmark"]["max_steps_per_case"] == 3, "Unexpected benchmark protocol")
    require(plan["max_gcds"] == 2 and plan["max_new_optimizer_updates"] == 24
            and plan["formal400_training"] is False and plan["capacity"]["micro_candidates"] == [4, 8]
            and plan["capacity"]["fallback_if_micro4_fails"] == 2
            and plan["capacity"]["prompt_tokens"] == 1024 and plan["capacity"]["completion_tokens"] == 4096
            and plan["capacity"]["optimizer_updates"] == 0 and plan["capacity"]["adam_moment_bytes_reserved"] is True
            and plan["benchmark"]["cases"] == [{"micro": 4, "generation": 32}, {"micro": 4, "generation": 64},
                {"micro": 4, "generation": 128}, {"micro": 8, "generation": 128}]
            and plan["benchmark"]["micro8_case_requires_capacity_pass_for_both_methods"] is True,
            "Capacity/budget/case plan differs from the bounded batch job")
    cases, errors, incomplete, provenance = {}, [], [], {}
    reference, initial_by_variant, initial_policy, questions, exposures, fixed_rows, prompt_tokens = None, {}, {}, None, None, None, None
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or not directory.name.startswith(("capacity-", "benchmark-")):
            continue
        match = CASE.fullmatch(directory.name)
        if match is None:
            errors.append({"case": directory.name, "error": "Unknown case name"})
            continue
        kind, variant, generation, micro = match.groups()
        generation, micro = int(generation), int(micro)
        if not ((kind == "capacity" and generation == 128 and micro in (2, 4, 8)) or
                (kind == "benchmark" and generation in (32, 64, 128) and micro in (2, 4, 8)
                 and (micro != 8 or generation == 128))):
            errors.append({"case": directory.name, "error": "Case is outside the bounded benchmark plan"})
            continue
        needed = ["learner/result.json", "learner/run_protocol.json", "server-device.json", "learner-device.json", "distinct-gcds.json"]
        if kind == "benchmark":
            needed += ["learner/invocation.json", "learner/benchmark_questions.json", "learner/updates.jsonl",
                       "learner/performance.jsonl", "learner/reward_audit.jsonl", "learner/training_probabilities.jsonl",
                       "learner/sync/rollouts.jsonl", "learner/sync/sync.jsonl"]
        missing = [name for name in needed if not (directory / name).is_file()]
        if missing:
            entry = {"case": directory.name, "status": "incomplete", "missing_files": missing}
            if (directory / "learner/failure.json").is_file():
                entry["recorded_failure"] = read(directory / "learner/failure.json")
            incomplete.append(entry)
            cases[directory.name] = entry
            continue
        try:
            if kind == "benchmark":
                value, common, initial = benchmark_case(directory, variant, micro, generation)
                if variant in initial_policy:
                    require(initial_policy[variant] == value["initial_policy_hashes"], "Initial canonical policy bytes differ across cases")
                initial_policy[variant] = value.pop("initial_policy_hashes")
                require(questions is None or questions == value["question_set_sha256"], "Fixed question set differs across cases/methods")
                require(exposures is None or exposures == value["actual_exposure_sha256"], "Actual question exposure differs across cases/methods")
                require(fixed_rows is None or fixed_rows == value["fixed_rows_sha256"], "Fixed probability control rows differ across cases")
                require(prompt_tokens is None or prompt_tokens == value["actual_prompt_tokens_by_question_sha256"], "Actual question input tokens differ across cases")
                questions, exposures = value["question_set_sha256"], value["actual_exposure_sha256"]
                fixed_rows = value["fixed_rows_sha256"]
                prompt_tokens = value["actual_prompt_tokens_by_question_sha256"]
            else:
                require(generation == 128, "Capacity generation label changed")
                value, common, initial = capacity_case(directory, variant, micro)
            require(reference is None or common == reference, "Full source/config/runtime/reward/base/train protocol differs across cases")
            reference = common
            require(variant not in initial_by_variant or initial_by_variant[variant] == initial, "Checkpoint25 inventory differs within a method")
            initial_by_variant[variant] = initial
            cases[directory.name] = value
            for path in directory.rglob("*"):
                if path.is_file() and path.suffix in (".json", ".jsonl"):
                    provenance[str(path.relative_to(root))] = file_hash(path)
        except (InvalidEvidence, KeyError, TypeError, ValueError, OverflowError) as error:
            entry = {"case": directory.name, "status": "invalid", "error": str(error)}
            errors.append(entry)
            cases[directory.name] = entry

    required, skipped = set(), []
    def capacity(variant, micro):
        name = f"capacity-{variant}-g128-m{micro}"
        required.add(name)
        value = cases.get(name)
        return value.get("passed") if value and value.get("status") == "verified" else None
    c4 = {variant: capacity(variant, 4) for variant in METHODS}
    c8 = {}
    for variant in METHODS:
        if c4[variant] is True:
            c8[variant] = capacity(variant, 8)
        elif c4[variant] is False:
            skipped.append({"case": f"capacity-{variant}-g128-m8", "reason": "micro4 capacity OOM; script skips micro8 for this method"})
    baseline = 2 if any(value is False for value in c4.values()) else 4 if all(value is True for value in c4.values()) else None
    if baseline == 2:
        for variant in METHODS:
            if capacity(variant, 2) is False:
                errors.append({"case": f"capacity-{variant}-g128-m2", "error": "Fallback capacity OOM; the job must not benchmark that schedule"})
    if baseline is not None:
        for variant in METHODS:
            for generation in (32, 64, 128):
                required.add(f"benchmark-{variant}-g{generation}-m{baseline}")
            if baseline == 4 and all(c8.get(method) is True for method in METHODS):
                required.add(f"benchmark-{variant}-g128-m8")
            elif baseline == 2 or any(value is False for value in c8.values()):
                skipped.append({"case": f"benchmark-{variant}-g128-m8", "reason": "Shared micro8 capacity gate did not pass"})
    for name in sorted(required):
        if name not in cases:
            incomplete.append({"case": name, "status": "incomplete", "missing_files": ["case directory"]})
        elif cases[name].get("status") == "invalid":
            continue
    completed_marker = (root / "all-cases.done").is_file()
    if not completed_marker:
        incomplete.append({"case": None, "status": "incomplete", "missing_files": ["all-cases.done"]})
    gates_known = all(value is not None for value in c4.values()) and all(value is not None for value in c8.values())
    for name in cases:
        if gates_known and name not in required:
            errors.append({"case": name, "error": "Case was not authorized by the observed capacity gates"})
    # Every present case was fully validated; source capacity gates determine the required set.
    for name, value in cases.items():
        if value.get("status") != "verified" or not name.startswith("benchmark-") or baseline is None:
            continue
        base = cases.get(f"benchmark-{value['variant']}-g32-m{baseline}")
        if base and base.get("status") == "verified":
            value["descriptive_relative_to_g32_baseline"] = {
                "warm_mean_wall_ratio_baseline_over_case": base["warm_mean_wall_seconds"] / value["warm_mean_wall_seconds"],
                "warm_vllm_token_rate_ratio_case_over_baseline": value["warm_real_tokens_per_second_in_vllm_call"] / base["warm_real_tokens_per_second_in_vllm_call"],
                "warm_observations_per_case": 2, "coupled_generated_tokens": False, "stable_speedup_claim": False}
    status = "invalid" if errors else "incomplete" if incomplete else "verified"
    return {"schema_version": 1, "audited_at_utc": datetime.now(timezone.utc).isoformat(),
            "root": str(root.resolve()), "status": status, "passed": status == "verified",
            "all_cases_done_marker": completed_marker, "baseline_micro_batch_size": baseline,
            "required_cases": sorted(required), "skipped_cases": skipped, "incomplete": incomplete,
            "protocol_errors": errors, "cases": cases, "common_protocol": reference,
            "initialization_inventories_by_method": initial_by_variant,
            "all_present_source_and_config_hashes_match": reference is not None and not errors,
            "all_source_and_config_hashes_match": status == "verified",
            "artifact_sha256": provenance, "analyzer_sha256": file_hash(__file__),
            "formal_config_modified": False, "formal_configuration_automatically_promoted": False,
            "limits": ["Only two warm update observations per case; no stable speedup or convergence claim",
                       "Same question exposure does not imply identical sampled tokens or training trajectories",
                       "Inclusive stages must not be summed; fences/hash checks/JSON I/O instrument throughput",
                       "Telemetry missing/unsupported readings stay null; power retains socket semantics and is never summed across GCD roles",
                       "Source equality covers the protocol's Python tree plus scientific/base/train/lock hashes; archive the job shell and non-Python inputs separately",
                       "Frozen-policy invariance and fresh optimizer initialization are checked runtime assertions under the matched source, not reconstructed from saved full tensors",
                       "This audits serialized evidence; it does not rerun GPU computation or math scoring"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "batch_summary.json"
    try:
        summary = collect(args.root)
    except (OSError, InvalidEvidence, KeyError, TypeError, ValueError, OverflowError) as error:
        summary = {"status": "invalid", "passed": False, "root": str(args.root), "error": str(error)}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    compact = {"status": summary["status"], "passed": summary["passed"], "output": str(output),
               "protocol_errors": summary.get("protocol_errors", summary.get("error")),
               "incomplete_cases": [item["case"] or "all-cases.done" for item in summary.get("incomplete", [])],
               "benchmarks": [{"case": name, "cold_seconds": row["cold_step1"]["wall_seconds"],
                                "warm_seconds_each": [step["wall_seconds"] for step in row["warm_steps2_and3"]],
                                "warm_vllm_real_tokens_per_second": row["warm_real_tokens_per_second_in_vllm_call"],
                                "padding_fraction": row["padding"]["fraction"]}
                               for name, row in summary.get("cases", {}).items()
                               if name.startswith("benchmark-") and row.get("status") == "verified"]}
    print(json.dumps(compact, indent=2, allow_nan=False))
    return 0 if summary["status"] == "verified" else 1 if summary["status"] == "incomplete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
