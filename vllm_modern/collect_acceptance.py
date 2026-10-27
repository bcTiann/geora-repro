"""Check both real-process phases before declaring a two-update closure."""

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def collect(directory, variant):
    phases, publications, question_sets, seeds = [], [], [], []
    for phase in (1, 2):
        root = directory / f"phase-{phase}"
        learner = root / "learner"
        report = read(learner / "result.json")
        require(report["completed"] and report["phase"] == phase and report["variant"] == variant,
                "A phase is missing or incomplete")
        require(report["optimizer_updates_this_process"] == 1 and report["total_global_step"] == phase,
                "Wrong optimizer update count")
        require(report["saved_policy_matches"]["bitwise_equal"] and report["frozen_parameters_and_buffers_unchanged"],
                "Policy save or frozen-state check failed")
        require(report["finite_nonzero_gradients_each_update"] and report["stock_importance_sampling_verified"],
                "Gradient or importance-correction check failed")
        require(read(root / "separate-gcds-verified.json")["passed"], "Physical GCD separation was not verified")
        sync = rows(learner / "sync/sync.jsonl")
        require([row["policy_version"] for row in sync] == [phase - 1, phase], "Wrong publications")
        for row in sync:
            resident = row["resident_verification"]
            require(resident["passed"] and resident["canonical_tensor_count"] == 339
                    and resident["resident_parameter_count"] == 198, "Incomplete resident policy")
            comparison = read(learner / f"sync/three_way_v{row['policy_version']}.json")
            require(comparison["structural_checks_passed"] and comparison["sequences"] == 5,
                    "Three-way fixed-token comparison did not complete")
            require(comparison["tokens"] > 200 and all(r["includes_eos"] for r in comparison["rows"]),
                    "Fixed-token controls omitted EOS or expected sequence coverage")
        publications.append(sync)
        rollout = rows(learner / "sync/rollouts.jsonl")
        require(len(rollout) == 4 and all(row["responses"] == 32 for row in rollout), "Wrong rollout count")
        require([row["policy_version"] for row in rollout] == [phase-1] * 4, "Wrong rollout policy")
        expected_seeds = [42 + 8 * ((phase - 1) * 4 + k) for k in range(4)]
        require([row["request_seed"] for row in rollout] == expected_seeds, "Sampling seeds did not resume")
        seeds.extend(expected_seeds)
        probabilities = rows(learner / "training_probabilities.jsonl")
        require(len(probabilities) == 4, "Missing probability audits")
        ids = [identifier for row in probabilities for identifier in row["question_ids"]]
        require(all(isinstance(identifier, str) and identifier != "None" for identifier in ids), "Missing question identities")
        unique = set(ids)
        require(len(ids) == 128 and len(unique) == 16 and all(ids.count(x) == 8 for x in unique),
                "GRPO groups do not contain 16 questions times eight responses")
        question_sets.append(unique)
        if phase == 2:
            require(report["restore_audit"]["passed"], "Actual checkpoint restore was not verified")
        phases.append(report)
    require(not (question_sets[0] & question_sets[1]), "Restored data loader repeated the previous questions")
    before, after = publications[0][-1], publications[1][0]
    for field in ("canonical_tensor_sha256", "resident_checksums"):
        require(before[field] == after[field], f"Restored v1 differs from saved v1: {field}")
    return {
        "completed": True, "variant": variant,
        "scope": "Two real updates with a new server/learner process and actual checkpoint1 restore between updates",
        "optimizer_updates": 2, "training_responses": 256, "distinct_training_questions": 32,
        "logical_policy_versions": [0, 1, 2], "publications_including_restart": [0, 1, 1, 2],
        "all_339_canonical_and_198_resident_weights_exact": True,
        "restored_v1_republished_identically": True, "actual_checkpoint_restore_passed": True,
        "data_progress_verified": True, "request_seeds": seeds,
        "three_way_probability_measurements_complete": True,
        "numerical_equivalence_claim": False,
        "performance_claim": False, "scientific_accuracy_claim": False,
        "learner_wall_seconds_including_diagnostics": sum(row["total_learner_wall_seconds"] for row in phases),
        "phases": phases,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--variant", choices=("lora", "geora_mixed"), required=True)
    args = parser.parse_args()
    result = collect(args.run, args.variant)
    (args.run / "acceptance.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "phases"}, indent=2))
