"""Copy a completed GeoRA/LoRA pilot boundary into a new source-owned directory.

This standalone, stdlib-only tool never changes the source run or checkpoint
bytes. It is an explicit lineage migration, not a source-identity bypass in the
production trainer. Checkpoint restoration remains the trainer's responsibility.
Run with Python >=3.10. No model is loaded and no GPU is allocated.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid


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
ALLOWED_CHANGED = {"scripts/cached_block_score.py", "scripts/long_grpo_step.py"}
PROVENANCE_FIELDS = {
    "method", "model_repository", "model_revision", "dataset_manifest_sha256",
    "training_config_sha256", "initialization_sha256",
}
COUNTERS = (
    "physical_optimizer_step_calls", "physical_backward_calls", "physical_rollouts",
    "physical_rollout_attempts", "physical_completion_tokens",
)
EXECUTION_ONLY_FIELDS = {
    "runtime_limit_seconds", "checkpoint_interval", "checkpoint_reserve_seconds",
    "step_time_budget_seconds", "checkpoint_boundary", "resume_after_step", "status",
    "implementation_status", "runtime_estimation", "selection_status", "scope",
}
TERMINAL_STATES = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
    "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED", "SPECIAL_EXIT",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_hash(config):
    return canonical_hash({key: value for key, value in config.items() if key not in EXECUTION_ONLY_FIELDS})


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def safe_relative(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts,
            "Unsafe relative artifact path: " + str(relative))
    candidate = (root / relative).resolve()
    require(candidate.is_relative_to(root.resolve()), "Artifact path escapes run directory")
    require(not (root / relative).is_symlink(), "Symlink artifact is unsupported")
    return candidate


def no_symlinks(root):
    require(not root.is_symlink(), "Copy source must not be a symlink")
    for path in root.rglob("*"):
        require(not path.is_symlink(), "Symlinks are unsupported in copied histories/checkpoints")


@contextmanager
def run_lock(directory, *, create=False):
    path = directory / ".pilot.lock"
    require(not path.is_symlink(), "Run lock must not be a symlink")
    if create:
        handle = path.open("x+b")
    else:
        require(path.is_file(), "Source run has no production .pilot.lock")
        # Opening an existing lock read-only preserves the source bundle bytes.
        handle = path.open("rb")
    with handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("A live process holds the run lock") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def verify_source_snapshot(snapshot_path, source_map):
    snapshot = read_json(snapshot_path)
    files = snapshot.get("files", {})
    require(set(files) == set(source_map), "Old source snapshot keys differ from report source identity")
    for relative, digest in source_map.items():
        item = files[relative]
        require(isinstance(item.get("content"), str), "Old snapshot needs UTF-8 source contents")
        require(item.get("sha256") == digest
                and hashlib.sha256(item["content"].encode("utf-8")).hexdigest() == digest,
                "Old source snapshot byte hash differs: " + relative)
    return snapshot


def current_source_snapshot(source_root, expected_keys):
    files = {}
    for relative in expected_keys:
        path = safe_relative(source_root, relative)
        require(path.is_file(), "Missing new source: " + relative)
        content = path.read_bytes()
        files[relative] = {"sha256": hashlib.sha256(content).hexdigest(), "content": content.decode("utf-8")}
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
            text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None  # Permits the deliberately non-git CPU fixture source tree.
    if commit is not None:
        for relative, item in files.items():
            committed = subprocess.check_output(
                ["git", "-C", str(source_root), "show", "HEAD:" + relative], stderr=subprocess.DEVNULL)
            require(hashlib.sha256(committed).hexdigest() == item["sha256"],
                    "New execution source has uncommitted bytes: " + relative)
    return {"execution_git_commit": commit, "source": "Actual migration-time source bytes", "files": files}


def validate_data(data_dir, report, new_config_path):
    identity = report["identity"]
    require(file_hash(data_dir / "manifest.json") == identity["dataset_manifest_sha256"],
            "Current dataset manifest differs from original training identity")
    metadata = read_json(data_dir / "manifest.json")["splits"]["train"]
    train_path = safe_relative(data_dir, metadata["relative_path"])
    require(file_hash(train_path) == metadata["sha256"], "Prepared train file checksum differs")
    all_rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line]
    require(len(all_rows) == metadata["rows"], "Prepared train row count differs")
    config = report["config"]
    rows = all_rows[:config["train_question_limit"]]
    ids = [row["id"] for row in rows]
    require(len(rows) == config["train_question_limit"] and len(ids) == len(set(ids)),
            "Training subset is incomplete or duplicated")
    require(all(row.get("source_split") == "train" for row in rows), "Training subset includes another split")
    require(canonical_hash(rows) == identity["train_rows_sha256"] and ids == identity["train_question_ids"],
            "Current training rows/order differ from original identity")
    expected = identity["training_config_sha256"]
    require(training_hash(config) == expected and training_hash(read_json(new_config_path)) == expected,
            "Training policy changed across migration")


def inspect_boundary(source_output, report):
    require(report.get("stage") == "gsm8k_bounded_grpo_pilot", "Wrong run stage")
    require(report.get("status") in {"paused_at_boundary", "completed"}, "Run report is not a completed boundary")
    # The production deadline handler deliberately retains discarded partial
    # diagnostics. They are not live samples after the job is terminal. Accept
    # only a verbatim diagnostic already attributed to the next aborted step.
    for current_key, discarded_key in (("current_partial_rollout", "partial_rollout"),
                                       ("current_partial_gradient_accumulation", "partial_gradient_accumulation")):
        diagnostic = report.get(current_key)
        if diagnostic:
            require(any(item.get("logical_step") == report.get("logical_optimizer_steps", -1) + 1
                        and item.get(discarded_key) == diagnostic for item in report.get("discarded_rollouts", [])),
                    "Partial diagnostic is not attributed to an explicitly discarded next step")
            if current_key == "current_partial_gradient_accumulation":
                require(diagnostic.get("optimizer_update_performed") is False,
                        "Partial gradient diagnostic claims an optimizer update")
    require(all(check.get("passed") is True for check in report.get("checks", [])), "Run contains a failed gate")
    require(report.get("identity", {}).get("output_directory") == str(source_output),
            "Original output identity does not match its directory")
    sources = report["identity"].get("source_sha256", {})
    require(set(sources) == set(SOURCE_KEYS), "Unexpected source-manifest keys for this long pilot")
    require(set(report.get("provenance", {})) == PROVENANCE_FIELDS, "Expected exactly six provenance fields")
    provenance = report["provenance"]
    for field in ("method", "dataset_manifest_sha256", "training_config_sha256"):
        require(provenance[field] == report["identity"][field], "Provenance and report identity differ: " + field)
    checkpoint = Path(report["latest_checkpoint"]).resolve()
    require(checkpoint.is_relative_to(source_output / "checkpoints") and checkpoint.is_dir(),
            "Latest checkpoint is outside original run")
    no_symlinks(checkpoint)
    manifest = read_json(checkpoint / "manifest.json")
    require(manifest.get("training_checkpoint_format") == 1 and manifest.get("status") == "complete"
            and manifest.get("boundary") == "after_rollout_and_optional_update", "Incomplete/unsupported checkpoint")
    require(manifest.get("scheduler_kind") == "constant_v1" and manifest.get("optimizer_type") == "AdamW",
            "Checkpoint optimizer/scheduler differs")
    require(manifest["provenance"] == provenance and manifest["provenance_sha256"] == canonical_hash(provenance),
            "Checkpoint provenance differs from report")
    progress = manifest["progress"]
    n = progress["completed_optimizer_steps"]
    require(type(n) is int and n >= 0 and progress["completed_rollouts"] == n,
            "Invalid logical checkpoint progress")
    require(progress.get("boundary") == "after_rollout_and_optional_update"
            and progress.get("rollout_in_flight") is False, "Checkpoint has in-flight samples")
    require(progress["pilot_run_id"] == report["pilot_run_id"], "Run UUID differs from checkpoint")
    references = progress["active_step_records"]
    require(len(references) == n and references == report["active_step_records"]
            and report["logical_optimizer_steps"] == n and report["latest_checkpoint_logical_step"] == n,
            "Checkpoint and report do not share their latest active history")
    q = report["config"]["questions_per_batch"]
    require(progress["next_data_position"] == report["next_data_position"] == n * q,
            "Data cursor does not match the completed boundary")
    token_total = 0
    segment_ids = [segment["id"] for segment in report["segments"]]
    require(segment_ids == [f"segment-{index:04d}" for index in range(1, len(segment_ids) + 1)],
            "Historical segment numbering is not contiguous")
    for segment in report["segments"]:
        segment_path = safe_relative(source_output, "segments/" + segment["id"] + ".json")
        require(read_json(segment_path) == segment, "Historical segment file differs from its report entry")
    journal = source_output / "step_journal.jsonl"
    require(journal.is_file() and not journal.is_symlink(), "Missing or symlinked completed-step journal")
    journal_entries = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines() if line]
    for step, reference in enumerate(references, 1):
        path = safe_relative(source_output, reference["relative_path"])
        require(path.is_file() and file_hash(path) == reference["sha256"], "Active step checksum differs")
        record = read_json(path)
        require(reference["logical_step"] == record["logical_step"] == step
                and record["pilot_run_id"] == report["pilot_run_id"], "Active history is not contiguous or same run")
        require(reference["valid_completion_tokens"] == record["valid_completion_tokens"], "Step token index differs")
        ids = report["identity"]["train_question_ids"]
        expected_ids = [ids[((step - 1) * q + offset) % len(ids)] for offset in range(q)]
        require(record["question_ids"] == expected_ids, "Step used a different deterministic data position")
        require(record.get("segment_id") in segment_ids, "Active step has no historical segment")
        expected_relative = f"segments/{record['segment_id']}/step-{step:05d}.json"
        require(reference["relative_path"] == expected_relative, "Active step path differs from its segment/step identity")
        require(any(event.get("event") == "completed_step" and event.get("segment_id") == record["segment_id"]
                    and all(event.get(key) == value for key, value in reference.items()) for event in journal_entries),
                "Active step is absent from the immutable completed-step journal")
        token_total += reference["valid_completion_tokens"]
    require(token_total == progress["completed_answer_tokens"] == report["completed_answer_tokens"],
            "Completed-answer token totals differ")
    for key in COUNTERS:
        require(type(report.get(key)) is int and report[key] >= progress["known_physical_counters"].get(key, 0),
                "Report physical counters precede checkpoint: " + key)
    require(report["physical_optimizer_step_calls"] >= n, "Physical optimizer count precedes logical count")
    require(report["physical_tokens_accounted_rollout_count"] == report["physical_rollouts"],
            "Completed rollout tokens have not all been accounted")
    require(set(manifest["files"]) == {"adapter.safetensors", "training_state.pt"}, "Unexpected checkpoint payload files")
    for relative, descriptor in manifest["files"].items():
        path = safe_relative(checkpoint, relative)
        require(path.stat().st_size == descriptor["bytes"] and file_hash(path) == descriptor["sha256"],
                "Checkpoint payload size/hash differs: " + relative)
    return checkpoint, manifest


def observed_job_ids(report):
    runtimes = [report.get("runtime", {})] + [segment.get("runtime", {}) for segment in report["segments"]]
    ids = {str(runtime["slurm_job_id"]) for runtime in runtimes if runtime.get("slurm_job_id") is not None}
    require(bool(ids), "No allocation IDs in the source run")
    return sorted(ids)


def slurm_record(report, supplied_path):
    ids = observed_job_ids(report)
    if supplied_path is None:
        command = ["sacct", "--noheader", "--parsable2", "-X", "-j", ",".join(ids),
                   "--format=JobIDRaw,State,ExitCode"]
        output = subprocess.check_output(command, text=True)
        jobs = []
        for line in output.splitlines():
            fields = line.strip().split("|")
            if len(fields) >= 3 and fields[0] in ids:
                jobs.append(dict(job_id=fields[0], state=fields[1], exit_code=fields[2]))
        record = dict(verified_at_utc=utc_now(), source="Actual sacct query at migration time",
                      command=command, raw_output=output, jobs=jobs)
    else:
        record = read_json(supplied_path)
        require(isinstance(record.get("source"), str) and record["source"], "Slurm record needs observation source")
    observed_at = datetime.fromisoformat(record["verified_at_utc"].replace("Z", "+00:00"))
    updated_at = datetime.fromisoformat(report["updated_at_utc"].replace("Z", "+00:00"))
    require(observed_at.tzinfo is not None and updated_at.tzinfo is not None and observed_at >= updated_at,
            "Slurm observation predates the final run report")
    require(observed_at <= datetime.now(timezone.utc), "Slurm observation is in the future")
    rows = {str(item["job_id"]): item for item in record["jobs"]}
    require(set(ids).issubset(rows), "Slurm completion record omits an original allocation")
    require(len(rows) == len(record["jobs"]), "Slurm completion record repeats an allocation")
    for job_id in ids:
        state = rows[job_id]["state"].split()[0].rstrip("+")
        require(state in TERMINAL_STATES, "Allocation remains live/nonterminal: " + job_id + " " + state)
        require(isinstance(rows[job_id].get("exit_code"), str), "Missing allocation exit code")
        if state == "COMPLETED":
            require(rows[job_id]["exit_code"] == "0:0", "Completed allocation has nonzero exit code")
    record["required_original_job_ids"] = ids
    return record


def copy_verified(source, destination):
    require(not source.is_symlink(), "Refusing a symlink copy source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    require(not destination.exists(), "Copy would overwrite a destination artifact")
    shutil.copy2(source, destination)
    require(source.stat().st_size == destination.stat().st_size and file_hash(source) == file_hash(destination),
            "Copied artifact differs from original bytes")


def tree_digest(root):
    return {str(path.relative_to(root)): dict(bytes=path.stat().st_size, sha256=file_hash(path))
            for path in sorted(root.rglob("*")) if path.is_file()}


def materialize_sources(snapshot, directory):
    for relative, item in snapshot["files"].items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(item["content"].encode("utf-8"))
        require(file_hash(path) == item["sha256"], "Materialized source bytes differ")


def migrate(arguments):
    old = arguments.source_output.resolve()
    require(not arguments.destination_output.is_symlink(), "New output must not be symlinked")
    new = arguments.destination_output.resolve()
    require(old.is_dir() and not arguments.source_output.is_symlink(), "Source output is missing or symlinked")
    require(not new.exists() and not new.is_symlink(), "New output must not already exist")
    require(not new.is_relative_to(old) and not old.is_relative_to(new), "Run directories cannot contain each other")
    with run_lock(old):
        report_path = old / "pilot_report.json"
        report = read_json(report_path)
        checkpoint, manifest = inspect_boundary(old, report)
        validate_data(arguments.data_dir.resolve(), report, arguments.new_config.resolve())
        completion = slurm_record(report, arguments.slurm_completion_record)
        old_sources = report["identity"]["source_sha256"]
        old_snapshot = verify_source_snapshot(arguments.old_source_snapshot, old_sources)
        new_snapshot = current_source_snapshot(arguments.new_source_root.resolve(), old_sources)
        new_sources = {key: item["sha256"] for key, item in new_snapshot["files"].items()}
        changed = {key for key in old_sources if old_sources[key] != new_sources[key]}
        require(changed == ALLOWED_CHANGED, "This migration permits exactly the two reviewed performance files")
        evidence = read_json(arguments.equivalence_evidence)
        require(evidence.get("status") == "passed", "Source-equivalence evidence has not passed")
        binding = evidence.get("source_binding", {})
        require(binding.get("old_source_sha256") == old_sources and binding.get("new_source_sha256") == new_sources,
                "Equivalence evidence is not bound to these exact old/new sources")
        no_symlinks(old / "segments")
        old_segment_bytes = tree_digest(old / "segments")
        original_report_hash = file_hash(report_path)
        new.parent.mkdir(parents=True, exist_ok=True)
        new.mkdir()  # Atomic no-overwrite reservation; FileExistsError wins races.
        with run_lock(new, create=True):
            try:
                shutil.copytree(old / "segments", new / "segments")
                require(tree_digest(new / "segments") == old_segment_bytes, "Historical segment bytes changed during copy")
                journal = old / "step_journal.jsonl"
                require(journal.is_file(), "Missing completed-step journal")
                copy_verified(journal, new / journal.name)
                new_checkpoint = new / "checkpoints" / checkpoint.name
                shutil.copytree(checkpoint, new_checkpoint)
                require(tree_digest(checkpoint) == tree_digest(new_checkpoint), "Copied checkpoint bytes differ")
                migration_dir = new / "migration"
                copy_verified(report_path, migration_dir / "original/pilot_report.json")
                copy_verified(arguments.old_source_snapshot, migration_dir / "original/source_snapshot.json")
                copy_verified(arguments.new_config, migration_dir / "current/executed_config.json")
                copy_verified(arguments.equivalence_evidence, migration_dir / "evidence/source_equivalence.json")
                additional_evidence = []
                for index, evidence_path in enumerate(arguments.evidence_file or []):
                    relative = f"migration/evidence/additional-{index:02d}-{evidence_path.name}"
                    copy_verified(evidence_path, new / relative)
                    additional_evidence.append(dict(original_name=evidence_path.name, relative_path=relative,
                                                    sha256=file_hash(evidence_path), bytes=evidence_path.stat().st_size))
                write_json_new(migration_dir / "current/source_snapshot.json", new_snapshot)
                write_json_new(migration_dir / "slurm_completion.json", completion)
                materialize_sources(old_snapshot, migration_dir / "original/source")
                materialize_sources(new_snapshot, migration_dir / "current/source")
                n = manifest["progress"]["completed_optimizer_steps"]
                migration = dict(
                    status="complete", kind="source_equivalent_migration", migration_id=str(uuid.uuid4()),
                    migrated_at_utc=utc_now(), original_output_directory=str(old), new_output_directory=str(new),
                    pilot_run_id=report["pilot_run_id"], preserved_completed_optimizer_steps=n,
                    next_data_position=manifest["progress"]["next_data_position"],
                    original_report_sha256=original_report_hash,
                    original_checkpoint_relative_path=str(checkpoint.relative_to(old)),
                    copied_checkpoint_relative_path=str(new_checkpoint.relative_to(new)),
                    checkpoint_bytes=tree_digest(new_checkpoint), original_history_bytes=old_segment_bytes,
                    old_source_sha256=old_sources, new_source_sha256=new_sources, changed_source_files=sorted(changed),
                    source_equivalence_evidence_relative_path="migration/evidence/source_equivalence.json",
                    source_equivalence_evidence_sha256=file_hash(arguments.equivalence_evidence),
                    additional_equivalence_evidence=additional_evidence,
                    slurm_completion_relative_path="migration/slurm_completion.json",
                    source_audit_ranges=[
                        dict(logical_steps=[1, n], segment_ids=[segment["id"] for segment in report["segments"]],
                             snapshot_relative_path="migration/original/source_snapshot.json",
                             source_root_relative_path="migration/original/source",
                             role="Historical source; historical segment git IDs and payload bytes remain unchanged"),
                        dict(logical_steps=[n + 1, report["config"]["optimizer_steps"]],
                             first_segment_id=f"segment-{len(report['segments']) + 1:04d}",
                             snapshot_relative_path="migration/current/source_snapshot.json",
                             source_root_relative_path="migration/current/source",
                             role="Future source; migration itself performs no optimizer update"),
                    ],
                    operational_note="Original run must remain immutable; resume only this new output. Locks are per directory.",
                    resource_accounting_note="Inherited cumulative counters describe one logical history. Deduplicate original and migrated GPU usage by Slurm job ID.",
                )
                write_json_new(migration_dir / "migration.json", migration)
                # Report copy carries the same UUID, counters, history, provenance,
                # RNG-bearing checkpoint and historical source-git runtime facts.
                report["identity"]["output_directory"] = str(new)
                report["identity"]["source_sha256"] = new_sources
                report["latest_checkpoint"] = str(new_checkpoint)
                report["source_equivalent_migration"] = dict(
                    migration_id=migration["migration_id"], relative_path="migration/migration.json",
                    sha256=file_hash(migration_dir / "migration.json"))
                report["updated_at_utc"] = migration["migrated_at_utc"]
                require(file_hash(report_path) == original_report_hash, "Original report changed during migration")
                write_json_new(new / "migration_ready.json", dict(status="ready_to_resume", latest_checkpoint=str(new_checkpoint),
                                                                  preserved_steps=n, pilot_run_id=report["pilot_run_id"]))
                write_json_new(new / "pilot_report.json", report)  # Publish resumability last.
            except BaseException as error:
                # Never delete the source or partially copied destination. A
                # published report only appears after all substantive checks.
                failure = new / "migration_failed.json"
                if not failure.exists():
                    write_json_new(failure, dict(status="failed", error=str(error), recorded_at_utc=utc_now()))
                raise
    return dict(status="ready_to_resume", output_directory=str(new), resume_checkpoint=str(new_checkpoint),
                completed_optimizer_steps=n, pilot_run_id=report["pilot_run_id"],
                migration_record=str(new / "migration/migration.json"))


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-output", "destination-output", "data-dir", "new-config", "new-source-root",
                 "old-source-snapshot", "equivalence-evidence"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--slurm-completion-record", type=Path,
                        help="Final actual observation JSON; omit to query sacct on this host.")
    parser.add_argument("--evidence-file", type=Path, action="append",
                        help="Optional additional CPU/GPU proof bytes to preserve in the new lineage bundle.")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(migrate(parse_arguments()), indent=2))
