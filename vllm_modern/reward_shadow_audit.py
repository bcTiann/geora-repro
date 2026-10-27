"""CPU-only, descriptive reward comparison on already-generated responses.

This program does not load a language model, generate text, change a reward,
or download files. It joins raw completion IDs to the trainer's question-ID
records, decodes with a local tokenizer, and compares the frozen GSM8K numeric
reward with accuracy_reward from the exact accepted TRL source revision.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import sys
import time


TRL_COMMIT = "c6eac19f63be27498f2a7c8108d1e8a4dffde23f"
TRL_VERSION = "1.17.0.dev0"
# SHA256 of the complete official file fetched at TRL_COMMIT, not a guessed
# version-only identity. A different source must be audited explicitly.
OFFICIAL_REWARD_SOURCE_SHA256 = "242e73c532160f0325643e4a50274536d6aecb8cbbd1ba8dd6f069e37f004160"
MAX_COMPLETIONS = 384


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_hash(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


def jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object in {path}:{number}")
            yield value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def token_ids(values, label):
    require(isinstance(values, list) and values
            and all(isinstance(value, int) and not isinstance(value, bool) and value >= 0
                    for value in values), f"Invalid {label} token IDs")
    return values


def gold_map(path):
    result = {}
    for row in jsonl(path):
        require("id" in row and "target" in row, "Training rows need explicit id and target")
        require(isinstance(row["id"], (str, int)) and not isinstance(row["id"], bool)
                and str(row["id"]).strip(), "Invalid training question ID")
        key = str(row["id"])
        require(key not in result, f"Duplicate training question ID: {key}")
        result[key] = str(row["target"])
    require(result, "The training data is empty")
    return result


def aligned_completions(rollout_path, probability_path, targets, *, limit, num_generations):
    """Audit the selected prefix; IDs come only from recorded trainer inputs."""
    selected, seen_prompts, per_version = [], {}, Counter()
    pairs = zip(jsonl(rollout_path), jsonl(probability_path), strict=True)
    for rollout_index, (rollout, probability) in enumerate(pairs):
        version_fields = [name for name in ("policy_version", "optimizer_step") if name in probability]
        require(version_fields, "Trainer record needs policy_version or pre-update optimizer_step")
        require(all(isinstance(probability[name], int) and not isinstance(probability[name], bool)
                    and probability[name] >= 0 for name in version_fields),
                "Invalid trainer policy-version field")
        probability_version = probability[version_fields[0]]
        require(all(probability[name] == probability_version for name in version_fields),
                "Trainer policy_version and optimizer_step disagree")
        version_field = "+".join(version_fields)
        require(rollout["policy_version"] == probability_version,
                "Rollout/trainer policy versions differ")
        if "rollout_index" in probability:
            require(probability["rollout_index"] == rollout_index, "Trainer rollout order differs")
        version = rollout["policy_version"]
        if "rollout_in_update" in rollout:
            require(rollout["rollout_in_update"] == per_version[version],
                    "Rollout position within an update differs")
        per_version[version] += 1
        completions, prompts = rollout["completion_ids"], rollout["prompt_ids"]
        ids = probability.get("question_ids")
        require(isinstance(ids, list), "Missing recorded question IDs; refusing to infer golds")
        count = len(completions)
        require(count > 0 and count == len(prompts) == len(ids)
                == rollout["responses"] == probability["responses"],
                "Rollout/trainer response or question-ID counts differ")
        require(count % num_generations == 0, "Incomplete generation group")
        for start in range(0, count, num_generations):
            group_ids = ids[start:start + num_generations]
            require(all(isinstance(value, (str, int)) and not isinstance(value, bool)
                        and str(value).strip() for value in group_ids), "Invalid recorded question ID")
            group_ids = [str(value) for value in group_ids]
            question_id = group_ids[0]
            require(all(value == question_id for value in group_ids),
                    "A consecutive generation group mixes question IDs")
            require(question_id in targets, f"Recorded question absent from train data: {question_id}")
            group_prompts = [token_ids(value, "prompt")
                             for value in prompts[start:start + num_generations]]
            require(all(value == group_prompts[0] for value in group_prompts),
                    "Repeated question IDs do not have identical prompt tokens")
            prompt_hash = value_hash(group_prompts[0])
            require(question_id not in seen_prompts or seen_prompts[question_id] == prompt_hash,
                    "A question ID maps to different prompts within the audited prefix")
            seen_prompts[question_id] = prompt_hash
            for offset, completion in enumerate(completions[start:start + num_generations]):
                completion = token_ids(completion, "completion")
                if len(selected) < limit:
                    selected.append({
                        "index": len(selected), "rollout_index": rollout_index,
                        "response_index_in_rollout": start + offset,
                        "policy_version": version, "question_id": question_id,
                        "trainer_policy_version": probability_version,
                        "trainer_policy_version_field": version_field,
                        "prompt_ids_sha256": prompt_hash,
                        "completion_ids_sha256": value_hash(completion),
                        "completion_token_count": len(completion),
                        "completion_ids": completion, "gold_original": targets[question_id],
                    })
        if len(selected) == limit:
            break
    require(selected, "No completions are available for the shadow audit")
    return selected


def learner_directory(case_dir):
    """Accept either CASE or CASE/learner, based on the two required records."""
    candidates = [Path(case_dir), Path(case_dir) / "learner"]
    valid = [path for path in candidates if (path / "sync/rollouts.jsonl").is_file()
             and (path / "training_probabilities.jsonl").is_file()]
    require(len(valid) == 1,
            "case-dir must identify exactly one CASE/learner or direct learner record directory")
    return valid[0]


def installed_versions():
    return dict(sorted((distribution.metadata["Name"], distribution.version)
                       for distribution in importlib.metadata.distributions()
                       if distribution.metadata.get("Name")))


def mathematical_gold(text, parse_numeric_answer):
    value = parse_numeric_answer(text)
    require(value is not None, "The frozen GSM8K gold is not a valid exact numeric scalar")
    if value.denominator == 1:
        return f"${value.numerator}$"
    return "$\\frac{" + str(value.numerator) + "}{" + str(value.denominator) + "}$"


def compare_batches(rows, tokenizer, official_reward, strict_reward, extract_boxed,
                    parse_numeric_answer, *, batch_size):
    batches, output = [], []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        texts = tokenizer.batch_decode([row["completion_ids"] for row in batch],
                                      skip_special_tokens=True, clean_up_tokenization_spaces=False)
        solutions = [mathematical_gold(row["gold_original"], parse_numeric_answer) for row in batch]
        strict_started = time.perf_counter()
        strict = [strict_reward(text, row["gold_original"]) for text, row in zip(texts, batch, strict=True)]
        strict_seconds = time.perf_counter() - strict_started
        official_started, error = time.perf_counter(), None
        try:
            official = official_reward(
                completions=[[{"role": "assistant", "content": text}] for text in texts],
                solution=solutions,
            )
            require(isinstance(official, list) and len(official) == len(batch),
                    "Official reward returned the wrong count or container")
            require(all(value is None or (isinstance(value, (int, float))
                                          and not isinstance(value, bool) and math.isfinite(value)
                                          and value in (0.0, 1.0)) for value in official),
                    "Official accuracy_reward returned an unexpected value")
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
            official = [None] * len(batch)
        official_seconds = time.perf_counter() - official_started
        batches.append({"batch_index": len(batches), "responses": len(batch),
                        "strict_scoring_seconds": strict_seconds,
                        "official_scoring_seconds": official_seconds,
                        "official_batch_error": error})
        for row, text, solution, own, other in zip(batch, texts, solutions, strict, official, strict=True):
            record = {key: value for key, value in row.items() if key != "completion_ids"}
            eos = tokenizer.eos_token_id
            record.update({
                "decoded_text": text, "decoded_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "raw_ends_with_tokenizer_eos": row["completion_ids"][-1] == eos,
                "extracted_boxed": extract_boxed(text), "official_gold_math_text": solution,
                "strict": own, "official_reward": other,
                "official_error": error,
                "official_error_scope": "batch; no individual retry performed" if error else None,
                "comparison_status": "official_error" if error else "official_none" if other is None
                                     else "agreement" if own["reward"] == other else "mismatch",
                "mismatch": None if error or other is None else own["reward"] != other,
            })
            output.append(record)
    return output, batches


def audit(args):
    # This is a separate CPU scoring process. Hide devices before any optional
    # dependency import and force local-only tokenizer operation.
    for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        os.environ[name] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    learner = learner_directory(args.case_dir)
    paths = {"rollouts": learner / "sync/rollouts.jsonl",
             "trainer_probabilities": learner / "training_probabilities.jsonl",
             "train": args.train_data}
    for path in paths.values():
        require(path.is_file(), f"Missing input file: {path}")
    require(args.base_model.is_dir(), "base-model must be a local tokenizer directory")
    report = {
        "schema_version": 1, "scope": "Descriptive CPU-only shadow audit of existing generated responses",
        "training_reward_changed": False, "reward_equivalence_established": False,
        "model_weights_loaded": False, "generation_performed": False,
        "requested_limit": args.limit, "hard_limit": MAX_COMPLETIONS,
        "provided_case_directory": str(args.case_dir.resolve()),
        "resolved_learner_directory": str(learner.resolve()),
        "num_generations": args.num_generations,
        "selection_rule": "First completions in file order, without selecting for reward or disagreement",
        "alignment_scope": "Selected prefix; matching rollout versions/order/counts, explicit IDs, repeated prompt tokens",
        "python": sys.version, "platform": platform.platform(),
        "input_files": {key: {"path": str(path.resolve()), "sha256": file_hash(path)}
                        for key, path in paths.items()},
        "source_files": {"shadow_auditor_sha256": file_hash(__file__)},
        "official_expected": {"trl_version": TRL_VERSION, "trl_commit": TRL_COMMIT,
                              "reward_source_sha256": OFFICIAL_REWARD_SOURCE_SHA256},
        "completed": False,
    }
    try:
        versions = installed_versions()
        report["installed_distributions"] = versions
        report["installed_distributions_sha256"] = value_hash(versions)
        require(importlib.metadata.version("trl") == TRL_VERSION, "Unexpected TRL version")
        from transformers import AutoTokenizer
        from trl.rewards import accuracy_reward
        import frozen_training.answer_scoring as frozen

        official_path = Path(inspect.getsourcefile(accuracy_reward))
        official_sha = file_hash(official_path)
        report["source_files"].update({
            "official_reward_path": str(official_path), "official_reward_file_sha256": official_sha,
            "official_reward_function_sha256": hashlib.sha256(inspect.getsource(accuracy_reward).encode()).hexdigest(),
            "official_reward_signature": str(inspect.signature(accuracy_reward)),
            "frozen_reward_file_sha256": file_hash(frozen.__file__),
            "frozen_reward_contract": frozen.GSM8K_REWARD_VERSION,
        })
        require(official_sha == OFFICIAL_REWARD_SOURCE_SHA256,
                "Official reward source differs from the pinned TRL commit")
        tokenizer_files = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                           "added_tokens.json", "vocab.json", "merges.txt", "config.json", "generation_config.json")
        manifest = {name: file_hash(args.base_model / name) for name in tokenizer_files
                    if (args.base_model / name).is_file()}
        require("tokenizer.json" in manifest, "Missing local tokenizer.json")
        report["tokenizer"] = {"path": str(args.base_model.resolve()), "files_sha256": manifest,
                               "manifest_sha256": value_hash(manifest), "local_files_only": True}
        protocol = learner / "protocol.json"
        if protocol.is_file():
            report["input_files"]["protocol"] = {"path": str(protocol.resolve()), "sha256": file_hash(protocol)}
            settings = json.loads(protocol.read_text())
            if "num_generations" in settings:
                require(settings["num_generations"] == args.num_generations,
                        "CLI num-generations differs from the recorded protocol")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
        report["decoding"] = {
            "skip_special_tokens": True, "clean_up_tokenization_spaces": False,
            "input": "Every raw completion ID exactly as recorded, including EOS if present",
            "eos_rule": "EOS remains in raw completion hash/count; special tokens are removed only in decoded text",
            "manual_truncation_at_eos": False, "tokenizer_eos_token_id": tokenizer.eos_token_id,
            "gold_wrapping": "Frozen exact numeric scalar converted to $integer$ or $\\frac{numerator}{denominator}$",
        }
        rows = aligned_completions(paths["rollouts"], paths["trainer_probabilities"], gold_map(args.train_data),
                                   limit=args.limit, num_generations=args.num_generations)
        records, batches = compare_batches(rows, tokenizer, accuracy_reward, frozen.score_gsm8k,
                                          frozen.extract_final_boxed, frozen.parse_numeric_answer,
                                          batch_size=args.batch_size)
        counts = Counter(row["comparison_status"] for row in records)
        comparable = counts["agreement"] + counts["mismatch"]
        report.update({
            "completed": True, "responses": len(records),
            "comparison_counts": {name: counts[name] for name in ("agreement", "mismatch", "official_none", "official_error")},
            "official_comparison_complete_without_errors": counts["official_error"] == 0,
            "all_official_rewards_defined": counts["official_error"] == counts["official_none"] == 0,
            "comparable_responses": comparable,
            "mismatch_fraction_among_comparable": counts["mismatch"] / comparable if comparable else None,
            "strict_status_counts": dict(Counter(row["strict"]["status"] for row in records)),
            "trainer_policy_version_fields": dict(Counter(row["trainer_policy_version_field"] for row in records)),
            "strict_zero_official_one": sum(row["strict"]["reward"] == 0 and row["official_reward"] == 1
                                            and row["official_error"] is None for row in records),
            "strict_one_official_zero": sum(row["strict"]["reward"] == 1 and row["official_reward"] == 0
                                            and row["official_error"] is None for row in records),
            "strict_scoring_seconds": sum(row["strict_scoring_seconds"] for row in batches),
            "official_scoring_seconds": sum(row["official_scoring_seconds"] for row in batches),
            "batches": batches, "records": records,
            "interpretation": "Agreement on this finite sample does not establish parser equivalence; errors/None are not scored as disagreements",
        })
    except Exception as exception:
        report["fatal_error"] = f"{type(exception).__name__}: {exception}"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-generations", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.limit <= MAX_COMPLETIONS:
        parser.error(f"limit must be between1 and{MAX_COMPLETIONS}")
    if args.batch_size < 1 or args.batch_size > MAX_COMPLETIONS or args.num_generations < 1:
        parser.error("batch-size must be1..384 and num-generations must be positive")
    if args.output.exists():
        parser.error("Use a new output file; existing audit evidence is preserved")
    report = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"completed": report["completed"], "output": str(args.output),
                      "responses": report.get("responses"), "comparison_counts": report.get("comparison_counts"),
                      "fatal_error": report.get("fatal_error")}))
    return 0 if report["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
