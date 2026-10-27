"""Revalidate and summarize a LoRA/GeoRA native two-update diagnostic pair.

Run beside the collected artifacts locally or on the remote machine. Paths are
ordinary filesystem paths; this program never launches jobs or fetches results.
The result describes compatibility measurements, not accuracy or speedup.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from collect_acceptance import collect, read, require, rows


COMPARISONS = ("learner_minus_hf_merged", "hf_merged_minus_vllm", "learner_minus_vllm")
LOGPROB_FIELDS = ("learner_logprobs", "hf_merged_logprobs", "vllm_logprobs")
PAIR_PROTOCOL_EXCEPTIONS = {"variant", "initialization_checkpoint", "resume_checkpoint"}
PHASE_PROTOCOL_EXCEPTIONS = {"phase", "optimizer", "policy_versions_this_process", "resume_checkpoint"}


def finite(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
            f"Nonfinite or nonnumeric measurement: {label}")
    return value


def normalized(protocol, exceptions):
    return {key: value for key, value in protocol.items() if key not in exceptions}


def exact_replay(before, after):
    require(before["policy_version"] == after["policy_version"] == 1, "Expected repeated policy v1")
    require(before["fixed_rows_sha256"] == after["fixed_rows_sha256"], "Repeated v1 controls changed")
    require(len(before["rows"]) == len(after["rows"]), "Repeated v1 controls changed in count")
    maximum = {field: 0.0 for field in LOGPROB_FIELDS}
    exact = {field: True for field in LOGPROB_FIELDS}
    for a, b in zip(before["rows"], after["rows"], strict=True):
        for field in ("id", "prompt_ids", "completion_ids", "includes_eos"):
            require(a[field] == b[field], f"Repeated v1 control differs: {field}")
        for field in LOGPROB_FIELDS:
            require(len(a[field]) == len(b[field]) == len(a["completion_ids"]),
                    f"Repeated v1 probability alignment differs: {field}")
            for x, y in zip(a[field], b[field], strict=True):
                finite(x, field)
                finite(y, field)
                maximum[field] = max(maximum[field], abs(x-y))
                exact[field] &= x == y
    return {"exact_repeat": all(exact.values()), "per_path_exact": exact,
            "per_path_max_abs_logprob_delta": maximum,
            "interpretation": "Exact comparison of serialized fixed-token measurements at saved and reloaded v1; separate from exact weight restoration and from stochastic training replay"}


def raw_comparison(report, comparison):
    left, right = {
        "learner_minus_hf_merged": ("learner_logprobs", "hf_merged_logprobs"),
        "hf_merged_minus_vllm": ("hf_merged_logprobs", "vllm_logprobs"),
        "learner_minus_vllm": ("learner_logprobs", "vllm_logprobs"),
    }[comparison]
    differences = []
    for row in report["rows"]:
        require(len(row[left]) == len(row[right]) == len(row["completion_ids"]),
                "Three-way raw token probability alignment differs")
        differences.extend(abs(finite(a, left)-finite(b, right))
                           for a, b in zip(row[left], row[right], strict=True))
    require(differences and len(differences) == report["tokens"], "Three-way token count differs")
    actual = {"mean_abs_logprob_delta": sum(differences)/len(differences),
              "max_abs_logprob_delta": max(differences), "tokens": len(differences)}
    recorded = report["comparisons"][comparison]
    for key in ("mean_abs_logprob_delta", "max_abs_logprob_delta"):
        require(math.isclose(actual[key], finite(recorded[key], key), rel_tol=1e-12, abs_tol=1e-12),
                f"Three-way summary disagrees with raw values: {comparison}/{key}")
    return actual


def summarize_variant(directory, variant):
    acceptance = collect(directory, variant)
    protocols, probability_rows, rollout_rows, update_rows = [], [], [], []
    publications = []
    provenance = {}
    for phase in (1, 2):
        root = directory / f"phase-{phase}/learner"
        protocol = read(root / "protocol.json")
        require(protocol["variant"] == variant and protocol["phase"] == phase,
                "Protocol belongs to another variant or phase")
        protocols.append(protocol)
        probabilities = rows(root / "training_probabilities.jsonl")
        rollouts = rows(root / "sync/rollouts.jsonl")
        updates = rows(root / "updates.jsonl")
        require(len(probabilities) == len(rollouts) == 4 and len(updates) == 1, "Wrong phase record counts")
        for index, (p, r) in enumerate(zip(probabilities, rollouts, strict=True)):
            require(p["policy_version"] == r["policy_version"] == phase-1,
                    "Probability audit and rollout policy versions differ")
            require(p["responses"] == r["responses"] == len(r["completion_ids"]) == 32,
                    "Probability audit and rollout response counts differ")
            require(p["rollout_index"] == index and r["rollout_in_update"] == index,
                    "Probability audit and rollout positions differ")
            token_count = sum(len(completion) for completion in r["completion_ids"])
            require(token_count == r["tokens"] == p["active_tokens"] > 0,
                    "Raw rollout tokens and active learner tokens differ")
            require(len(r["behavior_logprobs"]) == len(r["completion_ids"]), "Missing behavior probability rows")
            for tokens, logprobs in zip(r["completion_ids"], r["behavior_logprobs"], strict=True):
                require(len(tokens) == len(logprobs) > 0, "Missing active behavior probabilities")
                require(all(finite(value, "behavior_logprob") <= 1e-6 for value in logprobs),
                        "Invalid active behavior log probability")
            require(p["finite_active_logprobs"] and p["stock_token_truncate_verified"],
                    "Probability/correction audit failed")
            require(0 <= finite(p["fraction_ratio_above_2"], "clipped fraction") <= 1,
                    "Invalid clipped fraction")
            probability_rows.append(p)
            rollout_rows.append(r)
        update = updates[0]
        require(update["optimizer_step"] == phase and update["finite_nonzero_gradients"],
                "Wrong or invalid optimizer update")
        require(finite(update["post_clip_gradient_norm"], "gradient norm") > 0
                and update["changed_adapter_tensors"] > 0, "Missing nonzero gradient/adapter changes")
        update_rows.append(update)
        for version in (phase-1, phase):
            path = root / f"sync/three_way_v{version}.json"
            report = read(path)
            require(report["policy_version"] == version and report["structural_checks_passed"],
                    "Three-way report belongs to another version or is incomplete")
            require(report["acceptance_policy"] is None and report["numerical_accepted"] is None,
                    "Expected descriptive measurements without a fitted numerical threshold")
            publications.append({"phase": phase, "version": version, "report": report})
        for path in (root / "protocol.json", root / "result.json", root / "updates.jsonl",
                     root / "training_probabilities.jsonl", root / "sync/rollouts.jsonl", root / "sync/sync.jsonl"):
            provenance[str(path.relative_to(directory))] = hashlib.sha256(path.read_bytes()).hexdigest()
        for version in (phase-1, phase):
            path = root / f"sync/three_way_v{version}.json"
            provenance[str(path.relative_to(directory))] = hashlib.sha256(path.read_bytes()).hexdigest()

    require(normalized(protocols[0], PHASE_PROTOCOL_EXCEPTIONS)
            == normalized(protocols[1], PHASE_PROTOCOL_EXCEPTIONS),
            f"Invariant configuration drifted between save and restore: {variant}")
    fixed_hashes = {row["report"]["fixed_rows_sha256"] for row in publications}
    require(len(fixed_hashes) == 1, "Fixed-token controls changed between publications")
    repeat = exact_replay(publications[1]["report"], publications[2]["report"])
    # Version1 is measured twice to audit restart. Do not double-weight it in
    # the version0/1/2 descriptive range below.
    unique = (publications[0], publications[1], publications[3])
    comparisons = {}
    for comparison in COMPARISONS:
        by_version = {str(row["version"]): raw_comparison(row["report"], comparison) for row in unique}
        # Also validate the repeated raw data against its own recorded summary.
        raw_comparison(publications[2]["report"], comparison)
        means = [row["mean_abs_logprob_delta"] for row in by_version.values()]
        maxima = [row["max_abs_logprob_delta"] for row in by_version.values()]
        comparisons[comparison] = {"versions": by_version, "mean_abs_logprob_delta_range": [min(means), max(means)],
                                   "max_abs_logprob_delta": max(maxima)}

    active = sum(row["active_tokens"] for row in probability_rows)
    behavior = {
        "active_tokens": active,
        "token_weighted_mean_abs_learner_minus_behavior_logprob": sum(
            finite(row["mean_abs_log_ratio"], "mean log ratio") * row["active_tokens"] for row in probability_rows)/active,
        "max_abs_learner_minus_behavior_logprob": max(
            finite(row["max_abs_log_ratio"], "max log ratio") for row in probability_rows),
        "token_weighted_fraction_ratio_above_2": sum(
            row["fraction_ratio_above_2"] * row["active_tokens"] for row in probability_rows)/active,
        "importance_sampling": {"mode": "token_truncate", "clip_max": 2.0,
                                "scope": "Stock correction verified in every recorded rollout"},
        "aggregation": "Means/fractions weighted by active_tokens; original per-rollout float precision retained",
    }
    eos_ids = {row["completion_ids"][-1] for row in unique[0]["report"]["rows"] if row["includes_eos"]}
    require(eos_ids, "Fixed controls do not declare an EOS token")
    limit = protocols[0]["max_completion_length"]
    require(limit == 4096, "This summary expects the frozen4096-token diagnostic")
    completions = [completion for row in rollout_rows for completion in row["completion_ids"]]
    require(len(completions) == 256 and all(0 < len(row) <= limit for row in completions),
            "Training completion count/length is invalid")
    eos = sum(row[-1] in eos_ids for row in completions)
    hit_limit = sum(len(row) == limit for row in completions)
    limited_without_eos = sum(len(row) == limit and row[-1] not in eos_ids for row in completions)
    termination = {"responses": len(completions), "raw_generated_tokens": sum(map(len, completions)),
                   "declared_eos_token_ids": sorted(eos_ids), "ends_with_declared_eos": eos,
                   "length4096": hit_limit, "length4096_without_declared_eos": limited_without_eos,
                   "other_endings": sum(len(row) < limit and row[-1] not in eos_ids for row in completions),
                   "interpretation": "Inferred from raw token IDs and the controls' declared EOS; wire finish_reason was not recorded"}
    return {"acceptance": acceptance, "protocols": protocols,
            "question_ids_by_phase_and_rollout": [
                [row["question_ids"] for row in probability_rows[start:start+4]] for start in (0, 4)],
            "summary": {"variant": variant, "structural_acceptance_passed": True,
                        "three_way": comparisons, "restored_v1_fixed_token_repeat": repeat,
                        "actual_rollout_probabilities": behavior, "training_termination": termination,
                        "optimizer_updates": update_rows,
                        "changed_adapter_tensor_counts": [row["changed_adapter_tensors"] for row in update_rows],
                        "learner_wall_seconds_including_diagnostics": acceptance["learner_wall_seconds_including_diagnostics"],
                        "numerical_acceptance_policy": None, "numerical_equivalence_claim": False,
                        "provenance_sha256": provenance}}


def summarize_pair(lora_dir, geora_dir):
    data = {"lora": summarize_variant(Path(lora_dir), "lora"),
            "geora_mixed": summarize_variant(Path(geora_dir), "geora_mixed")}
    for phase in (0, 1):
        require(normalized(data["lora"]["protocols"][phase], PAIR_PROTOCOL_EXCEPTIONS)
                == normalized(data["geora_mixed"]["protocols"][phase], PAIR_PROTOCOL_EXCEPTIONS),
                f"LoRA and GeoRA diagnostic protocols differ in phase{phase+1}")
        require(data["lora"]["question_ids_by_phase_and_rollout"][phase]
                == data["geora_mixed"]["question_ids_by_phase_and_rollout"][phase],
                f"LoRA and GeoRA question order differs in phase{phase+1}")
    return {"schema_version": 1, "summarized_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed": True,
            "auditor_source_sha256": {Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                      for path in (__file__, collect.__code__.co_filename)},
            "scope": "Paired native TRL/vLLM two-update compatibility diagnostic with actual process restart at checkpoint1",
            "pair_protocols_equal_except": sorted(PAIR_PROTOCOL_EXCEPTIONS),
            "versions_data_and_fixed_controls_match": True, "per_rollout_training_question_order_matches": True,
            "all_restored_v1_raw_fixed_logprobs_repeat_exactly": all(
                value["summary"]["restored_v1_fixed_token_repeat"]["exact_repeat"] for value in data.values()),
            "methods": {name: value["summary"] for name, value in data.items()},
            "shared_protocol_by_phase": [normalized(protocol, PAIR_PROTOCOL_EXCEPTIONS)
                                         for protocol in data["lora"]["protocols"]],
            "numerical_acceptance_policy": None, "numerical_equivalence_claim": False,
            "scientific_accuracy_claim": False, "seed_robust_advantage_claim": False, "performance_claim": False,
            "timing_interpretation": "Learner wall includes loading, full weight hashing, three-way probability diagnostics and saving/restoration; it is not a throughput comparison. Full Slurm allocation, startup/import/input-preflight and queue costs are not reconstructed here.",
            "source_directories": {"lora": str(Path(lora_dir).resolve()), "geora_mixed": str(Path(geora_dir).resolve())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora-dir", type=Path, required=True)
    parser.add_argument("--geora-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for source in (args.lora_dir, args.geora_dir):
        require(source.resolve() != args.output.resolve()
                and source.resolve() not in args.output.resolve().parents,
                "Write the pair summary outside the read-only source run directories")
    summary = summarize_pair(args.lora_dir, args.geora_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"completed": True, "output": str(args.output),
                      "paired_protocol_and_question_order_match": True,
                      "v1_raw_fixed_logprobs_repeat_exactly": summary["all_restored_v1_raw_fixed_logprobs_repeat_exactly"],
                      "numerical_acceptance_policy": None}, indent=2))


if __name__ == "__main__":
    main()
