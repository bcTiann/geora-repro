"""Repeat fixed-policy probabilities at controlled request concurrency; no train()."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import runpy
import time
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
ORDER = (1, 1, 4, 4, 1, 4)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def flattened(record, field):
    return [value for row in record["rows"] for value in row[field]]


def compare(left, right, field="vllm_logprobs"):
    require([(r["id"], r["completion_ids"]) for r in left["rows"]] ==
            [(r["id"], r["completion_ids"]) for r in right["rows"]], "Control token rows differ")
    a, b = flattened(left, field), flattened(right, field)
    delta = [abs(x-y) for x, y in zip(a, b, strict=True)]
    require(bool(delta), "No probability tokens to compare")
    locations = [(row["id"], index, token) for row in left["rows"]
                 for index, token in enumerate(row["completion_ids"])]
    worst = max(range(len(delta)), key=delta.__getitem__)
    return {"tokens": len(delta), "changed_tokens": sum(x != y for x, y in zip(a, b, strict=True)),
            "mean_abs_logprob_delta": sum(delta)/len(delta), "max_abs_logprob_delta": max(delta),
            "worst_token": {"row_id": locations[worst][0], "completion_index": locations[worst][1],
                            "token_id": locations[worst][2], "left_logprob": a[worst], "right_logprob": b[worst]}}


def repetition_summary(records):
    result = {}
    for concurrency in (1, 4):
        indices = [i for i, value in enumerate(ORDER) if value == concurrency]
        result[str(concurrency)] = [
            {"left_replay": a, "right_replay": b,
             **compare(records[a], records[b])} for a, b in itertools.combinations(indices, 2)]
    result["serial_vs_parallel"] = [{"left_replay": a, "right_replay": b,
                                     **compare(records[a], records[b])}
                                    for a in (0, 1, 4) for b in (2, 3, 5)]
    result["hf_learner_repeats"] = [compare(records[0], row, "learner_logprobs") for row in records[1:]]
    result["hf_merged_repeats"] = [compare(records[0], row, "hf_merged_logprobs") for row in records[1:]]
    return result


class ConcurrencyClient:
    def __init__(self, client, concurrency):
        self.client, self.concurrency = client, concurrency

    def get_sequence_logprobs(self, *args, **kwargs):
        kwargs["max_concurrent_requests"] = self.concurrency
        return self.client.get_sequence_logprobs(*args, **kwargs)


def run(args):
    # Imports and model construction occur only in the allocated learner role.
    import torch
    from importlib.metadata import version
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from checkpoint_audit import checkpoint_inventory, verify_loaded_policy
    from native_sync import install_native_sync, source_guard
    from three_way_probability import score_three_way, validate_fixed_rows

    require(version("trl") == "1.17.0.dev0" and version("vllm") == "0.32.0+rocm723", "Wrong pinned runtime")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one learner GCD is required")
    require(not args.output.exists(), "Use a fresh probability replay output directory")
    acceptance_root = args.acceptance_run.resolve()
    require(acceptance_root not in args.output.resolve().parents, "Output must not modify the original acceptance run")
    accepted = read(acceptance_root / "acceptance.json")
    require(accepted["completed"] and accepted["variant"] == "geora_mixed", "Use the completed GeoRA acceptance run")
    original = read(acceptance_root / "phase-1/learner/result.json")
    checkpoint = acceptance_root / "phase-1/learner/training/checkpoint-1"
    inventory = checkpoint_inventory(checkpoint, "geora_mixed")
    require(original["completed"] and original["total_global_step"] == 1
            and inventory == original["saved_inventory"], "Checkpoint1 differs from the accepted saved inventory")

    helper = runpy.run_path(str(HERE / "02_train_split.py"), run_name="probability_replay_training_helpers")
    base, _, _ = helper["verify_inputs"](SimpleNamespace(
        execution_manifest=args.execution_manifest, train_data=args.train_data,
        variant="geora_mixed", output=args.output))
    fixed_rows = read(args.fixed_tokens)
    validate_fixed_rows(fixed_rows)
    require(len(fixed_rows) == 5, "Use the preregistered five synthetic controls")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    try:
        torch.set_num_threads(4)
        set_seed(42)
        tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        dataset, counts = helper["prepare_dataset"](args.train_data, tokenizer, max_prompt_length=1024)
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, local_files_only=True)
        schema = {name: (tuple(value.shape), value.dtype) for name, value in model.state_dict().items()}
        helper["load_geora_layers"](model, checkpoint, install=True, expected_base_model_path=base)
        model.config.use_cache = False
        model.to("cuda")
        saved_policy = verify_loaded_policy(model, checkpoint, "geora_mixed")
        require(saved_policy == original["saved_policy_matches"], "Loaded policy differs from accepted checkpoint1")
        training_args = helper["make_training_args"](args.output / "unused_training_output", args.server_url, args.group_port)
        trainer = helper["ModernGeoRATrainer"](
            model=model, processing_class=tokenizer, train_dataset=dataset,
            args=training_args, reward_funcs=helper["grpo_boxed_reward"], base_model_path=base)
        trainer.state.global_step = 1
        require(trainer.optimizer is None and trainer.lr_scheduler is None, "Probability probe must not create an optimizer")
        sync = install_native_sync(
            trainer, expected_schema=schema, audit_dir=args.output / "sync", fixed_rows=fixed_rows,
            expected_sync_versions=(1,), expected_generation_versions=(), fixed_token_checker=lambda *_: {})
        sync.sync_weights()
        publication = sync.sync_records[0]
        require(publication["resident_verification"]["passed"]
                and publication["canonical_tensor_count"] == 339
                and publication["resident_verification"]["resident_parameter_count"] == 198,
                "The complete canonical/resident policy was not verified")
        guard = source_guard(model)
        trainable = helper["trainable_hashes"](model)
        frozen = helper["frozen_hashes"](model)
        server_info = sync.client.get_server_info()
        write(args.output / "server_config.json", server_info)
        write(args.output / "protocol.json", {
            "purpose": "Fixed GeoRA v1 serial/parallel probability repetitions without parameter updates",
            "concurrency_order": ORDER, "checkpoint_inventory": inventory,
            "fixed_tokens_sha256": helper["file_hash"](args.fixed_tokens), "data_counts": counts,
            "parameter_updates": 0, "train_called": False, "batch_invariance_enabled_by_probe": False,
            "version": 1, "server_config_sha256": digest(server_info),
            "versions": {name: version(name) for name in ("trl", "vllm", "torch", "transformers", "peft")},
            "canonical_tensor_sha256": publication["canonical_tensor_sha256"],
            "resident_checksums": publication["resident_checksums"],
        })
        records = []
        for index, concurrency in enumerate(ORDER):
            print(f"[probability-only] replay={index} concurrency={concurrency} policy=v1", flush=True)
            record = score_three_way(model, schema, fixed_rows, ConcurrencyClient(sync.client, concurrency),
                                     device=trainer.accelerator.device, version=1, version_getter=sync._version)
            record.update(replay_index=index, max_concurrent_requests=concurrency,
                          resident_checksums_sha256=digest(publication["resident_checksums"]))
            write(args.output / f"replay-{index}-concurrency-{concurrency}.json", record)
            records.append(record)
        resident_after = sync._request("POST", "/weight_checker", json={"action": "checksum"})["checksums"]
        require(resident_after == publication["resident_checksums"], "Resident weights changed during probability-only replay")
        require(source_guard(model) == guard and helper["trainable_hashes"](model) == trainable
                and helper["frozen_hashes"](model) == frozen, "Learner policy changed during probability replay")
        require(trainer.state.global_step == 1 and trainer.optimizer is None
                and trainer.lr_scheduler is None and not sync.generation_records,
                "Probability-only replay unexpectedly trained or generated rollouts")
        require(checkpoint_inventory(checkpoint, "geora_mixed") == inventory, "Input checkpoint was modified")
        result = {
            "completed": True, "parameter_updates": 0, "train_called": False, "policy_version": 1,
            "checkpoint_inventory": inventory, "saved_policy_matches": saved_policy,
            "fixed_tokens_sha256": helper["file_hash"](args.fixed_tokens),
            "canonical_tensor_sha256": publication["canonical_tensor_sha256"],
            "resident_checksums": resident_after, "weights_unchanged": True,
            "concurrency_order": ORDER, "repetition_comparisons": repetition_summary(records),
            "numerical_equivalence_claim": False, "elapsed_seconds": time.monotonic()-started,
        }
        write(args.output / "result.json", result)
        print(json.dumps({k: v for k, v in result.items() if k not in
                          ("canonical_tensor_sha256", "resident_checksums", "repetition_comparisons", "checkpoint_inventory")}, indent=2), flush=True)
    except BaseException as error:
        write(args.output / "failure.json", {"completed": False, "error": repr(error),
                                             "elapsed_seconds": time.monotonic()-started})
        raise


def collect(directory):
    results, records = [], []
    for lifetime in ("replay-a", "replay-b"):
        root = directory / lifetime
        require(read(root / "separate-gcds-verified.json")["passed"], "Physical GCD separation was not verified")
        result = read(root / "learner/result.json")
        require(result["completed"] and result["weights_unchanged"] and result["parameter_updates"] == 0,
                "An immutable probability replay did not complete")
        require(tuple(result["concurrency_order"]) == ORDER, "Control request order changed")
        results.append(result)
        records.append([read(root / f"learner/replay-{i}-concurrency-{concurrency}.json")
                        for i, concurrency in enumerate(ORDER)])
    for field in ("checkpoint_inventory", "saved_policy_matches", "fixed_tokens_sha256",
                  "canonical_tensor_sha256", "resident_checksums"):
        require(results[0][field] == results[1][field], f"Two server lifetimes differ in {field}")
    cross = {}
    for concurrency in (1, 4):
        indices = [i for i, value in enumerate(ORDER) if value == concurrency]
        cross[str(concurrency)] = [{"lifetime_a_replay": a, "lifetime_b_replay": b,
                                   **compare(records[0][a], records[1][b])} for a in indices for b in indices]
    summary = {"completed": True, "parameter_updates": 0, "server_lifetimes": 2,
               "same_checkpoint_and_canonical_and_resident_bytes": True,
               "within_lifetime": {key: value["repetition_comparisons"] for key, value in
                                   zip(("replay-a", "replay-b"), results, strict=True)},
               "cross_lifetime_vllm_comparisons": cross,
               "cross_lifetime_hf_learner": compare(records[0][0], records[1][0], "learner_logprobs"),
               "cross_lifetime_hf_merged": compare(records[0][0], records[1][0], "hf_merged_logprobs"),
               "numerical_equivalence_claim": False,
               "causal_interpretation": "Request concurrency is controlled; scheduling/kernel/cache causes remain unproven without recorded execution traces"}
    write(directory / "probability_replay_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", type=Path)
    parser.add_argument("--acceptance-run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--server-url")
    parser.add_argument("--group-port", type=int, default=51216)
    parser.add_argument("--fixed-tokens", type=Path, default=HERE / "fixtures/probability_controls.json")
    parser.add_argument("--execution-manifest", type=Path)
    parser.add_argument("--train-data", type=Path)
    args = parser.parse_args()
    if args.collect:
        collect(args.collect)
    else:
        if not args.acceptance_run or not args.output or not args.server_url or not args.execution_manifest or not args.train_data:
            parser.error("--acceptance-run, --output, --server-url, --execution-manifest and --train-data are required for a GPU replay")
        run(args)
