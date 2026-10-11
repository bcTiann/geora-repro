"""Independently verify paired GeoRA training precision runtime reports.

Only reads JSON files and recomputes statistics. No torch import, model loading,
GPU work, sampling, optimizer step, or assumption of unmeasured whole-job cost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


TIME_KEYS = (
    "forward_and_loss_seconds", "backward_clip_step_seconds", "total_seconds",
)
REQUIRED_SOURCE_NAMES = {
    "check_geora_precision_benchmark.py", "precision_benchmark_helpers.py",
    "geora_layers.py", "grpo_rollout.py", "grpo_math.py", "geora_check_utils.py",
}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def canonical_digest(value) -> str:
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode()).hexdigest()


def source_hashes_by_name(mapping: dict) -> dict:
    result = {}
    for source, digest in mapping.items():
        name = Path(source).name
        if name in result:
            raise ValueError(f"Ambiguous repeated source filename: {name}")
        if not isinstance(digest, str) or SHA256.fullmatch(digest) is None:
            raise ValueError(f"Invalid source SHA256 for {source}")
        result[name] = digest
    if set(result) != REQUIRED_SOURCE_NAMES:
        raise ValueError(f"Expected exactly the benchmark's six source files: {set(result)}")
    return result


def numeric(value, *, positive=False) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and (value > 0 if positive else value >= 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path,
                        help="Optional separately captured execution-source JSON record; never inserted into raw reports")
    args = parser.parse_args()
    output = {
        "status": "running", "checks": [],
        "scope": "fixed-data policy first-update runtime comparison; not whole GRPO or task quality",
        "report_paths": {"bf16": str(args.bf16), "fp32": str(args.fp32)},
        "report_sha256": {label: hashlib.sha256(path.read_bytes()).hexdigest()
                          for label, path in (("bf16", args.bf16), ("fp32", args.fp32))},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def persist():
        args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")

    def require(name, condition, **details):
        output["checks"].append({"name": name, "passed": bool(condition), **details})
        print("PASS" if condition else "FAIL", name, flush=True)
        if not condition:
            raise ValueError(f"Comparison check failed: {name}")

    try:
        reports = {label: json.loads(path.read_text())
                   for label, path in (("bf16", args.bf16), ("fp32", args.fp32))}
        b, f = reports["bf16"], reports["fp32"]
        external_sources = None
        if args.source_manifest is not None:
            external_record = json.loads(args.source_manifest.read_text())
            # The explicit wrapper leaves room for timestamps and capture method.
            # Its content is preserved verbatim as evidence, never copied into
            # either raw benchmark report or called an in-process capture.
            external_sources = source_hashes_by_name(external_record["source_sha256"])
            output["external_execution_source_record"] = {
                "path": str(args.source_manifest),
                "sha256": hashlib.sha256(args.source_manifest.read_bytes()).hexdigest(),
                "record": external_record,
                "provenance": "external_execution_source_record",
            }
        actual_sources = {}
        output["source_hash_provenance"] = {}
        for label, report in reports.items():
            if "source_sha256" in report:
                actual_sources[label] = source_hashes_by_name(report["source_sha256"])
                output["source_hash_provenance"][label] = "in_report_execution_source_record"
                if external_sources is not None:
                    require(label + "_internal_source_hashes_match_external_record",
                            actual_sources[label] == external_sources)
            else:
                require(label + "_missing_internal_hashes_have_external_record", external_sources is not None)
                actual_sources[label] = external_sources
                output["source_hash_provenance"][label] = "external_execution_source_record"
        require("both_completed_successfully", b["status"] == f["status"] == "passed")
        require("correct_precision_conditions", b["dtype"] == "bf16" and f["dtype"] == "fp32")
        hashes = (
            "input_sha256", "frozen_value_sha256", "initial_adapter_file_sha256",
            "initial_manifest_sha256", "source_report_sha256",
        )
        for key in hashes:
            require(key + "_valid_and_equal", all(isinstance(r[key], str)
                and SHA256.fullmatch(r[key]) is not None for r in (b, f)) and b[key] == f[key])
        require("all_six_source_hashes_equal", actual_sources["bf16"] == actual_sources["fp32"])
        equal_fields = (
            "device", "device_name", "torch_version", "transformers_version", "hip_version",
            "scope", "reference_anchor", "real_original_reference_model_resident",
            "attention_backend", "training_probability_path", "gradient_checkpointing",
            "optimizer_foreach", "float32_matmul_precision", "source_forward_mode",
            "actual_forward_mode", "frozen_parameter_count", "trainable_parameter_count",
            "question_ids", "complete_source_answer_lengths", "benchmark_completion_lengths",
            "truncation_is_runtime_workload_only", "advantages_from_original_complete_answers",
        )
        for key in equal_fields:
            require(key + "_equal", b[key] == f[key])
        require("intended_runtime_scope", b["real_original_reference_model_resident"] is False
            and b["actual_forward_mode"] == "difference"
            and b["training_probability_path"] == "fixed_width_no_cache"
            and b["gradient_checkpointing"] is False and b["optimizer_foreach"] is False)
        require("FP32_factors_optimizer_and_loss_in_both", all(
            r["precision"]["adapters_initial_buffers_gradients_optimizer"] == "FP32"
            and r["precision"]["logits_log_softmax_loss"] == "FP32"
            and r["precision"]["frozen_model"] == label
            and r["precision"]["shared_frozen_values"] == "BF16 rounded once; exact upcast for FP32"
            for label, r in reports.items()))
        require("nonzero_finite_fixed_advantages", len(b["advantages_from_original_complete_answers"]) > 0
            and all(isinstance(v, (int, float)) and math.isfinite(v)
                    for v in b["advantages_from_original_complete_answers"])
            and any(v != 0 for v in b["advantages_from_original_complete_answers"]))
        require("positive_parameter_counts", all(isinstance(b[key], int) and b[key] > 0
            for key in ("frozen_parameter_count", "trainable_parameter_count")))

        benchmark_fields = (
            "warmup_count", "measurement_count", "optimizer_steps_are_independent_first_step_trials",
            "valid_completion_tokens", "input_shape", "fixed_width",
        )
        for key in benchmark_fields:
            require("benchmark_" + key + "_equal", b["benchmark"][key] == f["benchmark"][key])
        shape = b["benchmark"]["input_shape"]
        require("valid_input_and_completion_layout", len(shape) == 2 and all(
            isinstance(n, int) and n > 0 for n in shape)
            and shape[0] == len(b["question_ids"]) == len(b["advantages_from_original_complete_answers"])
            and shape[0] == len(b["benchmark_completion_lengths"])
            and shape[1] == b["benchmark"]["fixed_width"]
            and sum(b["benchmark_completion_lengths"]) == b["benchmark"]["valid_completion_tokens"]
            and all(isinstance(n, int) and 0 < n <= shape[1] for n in b["benchmark_completion_lengths"]))
        computed = {}
        memory = {}
        for label, r in reports.items():
            bench = r["benchmark"]
            records = bench["measurements"]
            require(label + "_real_measured_trials", isinstance(bench["warmup_count"], int)
                and bench["warmup_count"] >= 1
                and isinstance(bench["measurement_count"], int) and bench["measurement_count"] >= 3
                and len(records) == bench["measurement_count"]
                and bench["optimizer_steps_are_independent_first_step_trials"] is True
                and all(record["warmup"] is False for record in records))
            expected_dtype = "torch.bfloat16" if label == "bf16" else "torch.float32"
            require(label + "_benchmark_dtype", bench["frozen_dtype"] == expected_dtype)
            require(label + "_finite_nonzero_gradients", all(
                numeric(record["gradient_norm_before_clipping"], positive=True)
                and isinstance(record["loss"], (int, float)) and math.isfinite(record["loss"])
                for record in records))
            require(label + "_timing_partition", all(all(numeric(record[key], positive=True)
                for key in TIME_KEYS) and math.isclose(record["total_seconds"],
                record["forward_and_loss_seconds"] + record["backward_clip_step_seconds"],
                rel_tol=1e-10, abs_tol=1e-10) for record in records))
            computed[label] = {}
            for key in TIME_KEYS:
                values = [record[key] for record in records]
                recomputed = {"median": statistics.median(values), "minimum": min(values), "maximum": max(values)}
                require(label + "_" + key + "_summary_recomputed", all(math.isclose(
                    recomputed[stat], bench["summary"][key][stat], rel_tol=1e-12, abs_tol=1e-12)
                    for stat in recomputed))
                computed[label][key] = recomputed
            if r["device"] == "cuda":
                require(label + "_GPU_peaks_valid", all(numeric(record["memory"]["peak_allocated_gib"], positive=True)
                    and numeric(record["memory"]["peak_reserved_gib"], positive=True)
                    and record["memory"]["peak_allocated_gib"] + 1e-9 >= record["start_memory"]["allocated_gib"]
                    and record["memory"]["peak_reserved_gib"] + 1e-9 >= record["memory"]["peak_allocated_gib"]
                    for record in records))
                memory[label] = {
                    "max_measured_peak_allocated_gib": max(record["memory"]["peak_allocated_gib"] for record in records),
                    "median_measured_peak_allocated_gib": statistics.median(record["memory"]["peak_allocated_gib"] for record in records),
                    "max_measured_peak_reserved_gib": max(record["memory"]["peak_reserved_gib"] for record in records),
                    "median_resident_allocated_gib": statistics.median(record["start_memory"]["allocated_gib"] for record in records),
                }
            else:
                memory[label] = {"GPU_peak_comparison_available": False,
                                 "reason": "This report is not CUDA/HIP; MPS snapshots are not peak measurements"}
        timing_ratios = {key: computed["fp32"][key]["median"] / computed["bf16"][key]["median"]
                         for key in TIME_KEYS}
        peak_difference = None
        if b["device"] == "cuda":
            bp, fp = (memory[label]["max_measured_peak_allocated_gib"] for label in ("bf16", "fp32"))
            peak_difference = {"max_peak_allocated_delta_gib": fp - bp,
                               "max_peak_allocated_ratio_fp32_over_bf16": fp / bp,
                               "resident_allocated_delta_gib": memory["fp32"]["median_resident_allocated_gib"]
                                   - memory["bf16"]["median_resident_allocated_gib"]}
        frozen, trainable = b["frozen_parameter_count"], b["trainable_parameter_count"]
        theoretical = {
            "one_policy_frozen_parameter_count": frozen,
            "trainable_AB_parameter_count": trainable,
            "one_policy_frozen_BF16_gib": frozen * 2 / 2**30,
            "one_policy_frozen_FP32_gib": frozen * 4 / 2**30,
            "one_policy_FP32_minus_BF16_gib": frozen * 2 / 2**30,
            "policy_plus_original_reference_FP32_minus_BF16_gib": frozen * 4 / 2**30,
            "hypothetical_original_reference_extra_resident_BF16_gib": frozen * 2 / 2**30,
            "hypothetical_original_reference_extra_resident_FP32_gib": frozen * 4 / 2**30,
            "FP32_AB_grads_two_moments_A0B0_total_mib": trainable * 4 * 5 / 2**20,
            "scope": "parameter and initial-buffer storage only; not measured whole-step peak or activation prediction",
        }
        tokens = b["benchmark"]["valid_completion_tokens"]
        output.update(
            status="passed", computed_timing_statistics=computed,
            median_time_ratio_FP32_over_BF16=timing_ratios,
            policy_update_effective_tokens_per_second={label: tokens / computed[label]["total_seconds"]["median"]
                                                       for label in ("bf16", "fp32")},
            measured_memory=memory, measured_peak_difference=peak_difference,
            theoretical_parameter_storage=theoretical,
            common_input_shape=shape, common_valid_completion_tokens=tokens,
            common_initial_adapter_sha256=b["initial_adapter_file_sha256"],
            common_input_sha256=b["input_sha256"], common_frozen_value_sha256=b["frozen_value_sha256"],
            common_sources_sha256=actual_sources["bf16"],
            interpretation_limits=[
                "One policy only; original reference model not resident or timed",
                "No generation, checkpoint writes, SVD or data download in timed current update",
                "Original complete-answer advantages on truncated fixed token workload",
                "FP32 changes frozen storage and model arithmetic, not FP32 factors/optimizer/loss",
                "Device timing ratio is not a whole-job or task-quality result",
                "Parameter storage estimate does not predict activation or differentiable-cache memory",
                *( ["Source hashes were captured in a separate external record after benchmark processes; raw reports lack in-process source hashes"]
                   if any(value == "external_execution_source_record" for value in output["source_hash_provenance"].values()) else [] ),
            ],
        )
        persist()
        print("Paired benchmark verified:", args.output, flush=True)
    except Exception as error:
        output.update(status="failed", error=str(error))
        persist()
        raise


if __name__ == "__main__":
    main()
