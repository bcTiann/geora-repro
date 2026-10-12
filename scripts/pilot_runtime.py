"""Atomic records and completed-boundary control for the GeoRA pilot.

The training function is supplied by the existing five-step implementation.
No sampling, probability formula, optimizer policy or precision is redefined.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


EXECUTION_ONLY_FIELDS = {
    "runtime_limit_seconds", "checkpoint_interval", "checkpoint_reserve_seconds",
    "step_time_budget_seconds", "checkpoint_boundary", "resume_after_step",
    "status", "implementation_status", "runtime_estimation", "selection_status", "scope",
}


def training_config_hash(config):
    """Segment/runtime/storage controls do not identify the training policy."""
    return canonical_hash({k: v for k, v in config.items() if k not in EXECUTION_ONLY_FIELDS})


def initialization_hash(adapter_sha256, rank, alpha):
    """Tensor values alone cannot identify the adapter's alpha/r scaling."""
    return canonical_hash(dict(adapter_sha256=adapter_sha256, rank=rank, alpha=float(alpha)))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as destination:
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


@contextmanager
def exclusive_run(directory):
    """Hold one advisory lock for the entire process, including GPU setup."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".pilot.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process owns this pilot output directory") from error
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


class PilotReport:
    """The existing one_step only needs data, write(), require(), and path."""

    def __init__(self, directory, data=None):
        self.directory = Path(directory)
        self.path = self.directory / "pilot_report.json"
        self.data = data if data is not None else {"status": "running", "checks": []}

    def write(self):
        # The original one_step publishes complete rollout answers immediately
        # after incrementing physical_rollouts, including a rollout discarded
        # before backward because its deadline expired. Account each once.
        observed = self.data.get("physical_rollouts", 0)
        accounted = self.data.get("physical_tokens_accounted_rollout_count", 0)
        if observed > accounted:
            rollout = self.data.get("current_completed_rollout")
            if rollout is None or observed != accounted + 1:
                raise RuntimeError("Cannot account complete rollout tokens from the published record")
            count = sum(answer["token_count"] for answer in rollout["answers"])
            self.data["physical_completion_tokens"] = self.data.get("physical_completion_tokens", 0) + count
            self.data["physical_tokens_accounted_rollout_count"] = observed
        self.data["physical_completion_tokens_lower_bound"] = (
            self.data.get("physical_completion_tokens", 0)
            + self.data.get("discarded_partial_completion_tokens", 0)
            + self.data.get("discarded_incomplete_tokens_lower_bound", 0))
        self.data["updated_at_utc"] = utc_now()
        atomic_json(self.path, self.data)

    def require(self, name, condition, **measurements):
        self.data.setdefault("checks", []).append(
            {"name": name, "passed": bool(condition), **measurements})
        if not condition:
            self.data["status"] = "failed"
        self.write()
        print(f"{'PASS' if condition else 'FAIL'}: {name}", flush=True)
        if not condition:
            raise RuntimeError(f"Check failed: {name}; see {self.path}")


def selected_questions(rows, absolute_cursor, questions_per_batch):
    """Never filter rewards; cycle the fixed train subset by absolute cursor."""
    if not rows or absolute_cursor < 0 or questions_per_batch < 1:
        raise ValueError("Invalid deterministic train data selection")
    return [rows[(absolute_cursor + offset) % len(rows)]
            for offset in range(questions_per_batch)]


def record_path(directory, reference):
    path = (Path(directory) / reference["relative_path"]).resolve()
    if not path.is_relative_to(Path(directory).resolve()):
        raise ValueError("Step record escapes the owned output directory")
    if file_hash(path) != reference["sha256"]:
        raise ValueError("Saved step record checksum mismatch")
    return path


def validate_active_records(directory, references, completed_steps):
    if len(references) != completed_steps:
        raise ValueError("Active record count differs from completed logical steps")
    for step, reference in enumerate(references, 1):
        record = json.loads(record_path(directory, reference).read_text())
        if reference["logical_step"] != step or record["logical_step"] != step:
            raise ValueError("Active step records are not a contiguous logical history")
        if reference["valid_completion_tokens"] != record["valid_completion_tokens"]:
            raise ValueError("Saved step token count differs from its index")


def persist_completed_step(report, segment_id, logical_step, batch):
    relative = Path("segments") / segment_id / f"step-{logical_step:05d}.json"
    path = report.directory / relative
    if path.exists():
        raise FileExistsError("An immutable completed-step record already exists")
    record = dict(batch, logical_step=logical_step, segment_id=segment_id,
                  pilot_run_id=report.data["pilot_run_id"], persisted_at_utc=utc_now())
    atomic_json(path, record)
    reference = dict(relative_path=str(relative), sha256=file_hash(path), logical_step=logical_step,
                     valid_completion_tokens=batch["valid_completion_tokens"])
    event = dict(event="completed_step", segment_id=segment_id, **reference)
    # The single process holds the run lock. The append is flushed and fsynced;
    # complete records remain immutable even when a later checkpoint rolls back.
    with (report.directory / "step_journal.jsonl").open("a") as journal:
        journal.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        journal.flush()
        os.fsync(journal.fileno())
    report.data["active_step_records"].append(reference)
    return reference


COUNTERS = (
    "physical_optimizer_step_calls", "physical_backward_calls", "physical_rollouts",
    "physical_rollout_attempts", "physical_completion_tokens",
)


def capture_rng(torch, generator, device):
    return {
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_device": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
            if device == "cuda" else [],
        "python": random.getstate(), "generator": generator.get_state().cpu().clone(),
    }


def restore_rng(torch, generator, device, state):
    torch.set_rng_state(state["torch_cpu"])
    if device == "cuda":
        torch.cuda.set_rng_state_all(state["torch_device"])
    random.setstate(state["python"])
    generator.set_state(state["generator"])


def boundary_progress(report, cursor):
    steps = len(report.data["active_step_records"])
    return dict(
        boundary="after_rollout_and_optional_update", rollout_in_flight=False,
        completed_optimizer_steps=steps, completed_rollouts=steps, next_data_position=cursor,
        completed_answer_tokens=sum(r["valid_completion_tokens"] for r in report.data["active_step_records"]),
        pilot_run_id=report.data["pilot_run_id"], active_step_records=list(report.data["active_step_records"]),
        known_physical_counters={name: report.data[name] for name in COUNTERS},
    )


def save_current_boundary(report, model, optimizer, scheduler, generator, provenance,
                          cursor, device, save_checkpoint):
    progress = boundary_progress(report, cursor)
    step = progress["completed_optimizer_steps"]
    directory = report.directory / "checkpoints" / f"step-{step:05d}"
    if directory.exists():
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["progress"]["active_step_records"] != progress["active_step_records"]:
            # Resuming an earlier checkpoint can replay successful records
            # that were not in that checkpoint. Preserve their older boundary.
            directory = directory.with_name(directory.name + "-" + report.data["current_segment_id"])
            if directory.exists():
                manifest = json.loads((directory / "manifest.json").read_text())
                if manifest["progress"]["active_step_records"] != progress["active_step_records"]:
                    raise FileExistsError("Replay boundary directory belongs to different records")
        if manifest["provenance"] != provenance:
            raise ValueError("Existing checkpoint provenance differs")
        if directory.exists():
            for name, descriptor in manifest["files"].items():
                path = directory / name
                if path.stat().st_size != descriptor["bytes"] or file_hash(path) != descriptor["sha256"]:
                    raise ValueError("Existing boundary checksum differs")
    if not directory.exists():
        save_checkpoint(directory, model=model, optimizer=optimizer, scheduler=scheduler,
                        rollout_generator=generator, progress=progress, provenance=provenance, device=device)
    report.data["latest_checkpoint"] = str(directory.resolve())
    report.data["latest_checkpoint_logical_step"] = step
    report.write()
    return directory


def restore_record_history(report, progress):
    """Keep old attempts, but restore the checkpoint's authoritative history."""
    if progress.get("pilot_run_id") != report.data["pilot_run_id"]:
        raise ValueError("Resume checkpoint belongs to another pilot")
    restored = progress.get("active_step_records")
    if restored is None:
        raise ValueError("Resume checkpoint was not written by the general pilot runner")
    validate_active_records(report.directory, restored, progress["completed_optimizer_steps"])
    current = report.data["active_step_records"]
    if current[:len(restored)] != restored:
        raise ValueError("Resume checkpoint is not an ancestor of the current run history")
    abandoned = current[len(restored):]
    if abandoned:
        report.data.setdefault("superseded_step_records", []).extend(abandoned)
    report.data["active_step_records"] = list(restored)
    report.data["logical_optimizer_steps"] = progress["completed_optimizer_steps"]
    report.data["completed_answer_tokens"] = progress["completed_answer_tokens"]
    report.data["next_data_position"] = progress["next_data_position"]
    saved_counters = progress.get("known_physical_counters", {})
    for key in COUNTERS:
        if report.data[key] < saved_counters.get(key, 0):
            raise ValueError("Report physical counters precede the checkpoint")


def run_training_segment(*, torch, policy, reference, tokenizer, optimizer, scheduler,
                         generator, rows, config, report, provenance, stop_after_step,
                         work_deadline, device, step_function, save_checkpoint):
    """Reuse one_step; commit only a completed optimizer/scheduler boundary."""
    target = config["optimizer_steps"]
    starting_step = report.data["logical_optimizer_steps"]
    starting_answer_tokens = report.data["completed_answer_tokens"]
    if stop_after_step < starting_step or stop_after_step > target:
        raise ValueError("Absolute segment target must lie between current and final target")
    segment_id = f"segment-{len(report.data['segments']) + 1:04d}"
    starting_counts = {key: report.data[key] for key in COUNTERS}
    segment = dict(id=segment_id, status="running", started_at_utc=utc_now(),
                   start_logical_step=starting_step, absolute_stop_after_step=stop_after_step,
                   start_data_position=report.data["next_data_position"], runtime_only_controls_not_in_provenance=True)
    report.data["segments"].append(segment)
    report.data["current_segment_id"] = segment_id
    report.data["status"] = "running"
    report.data.pop("error", None)
    report.write()
    started = time.perf_counter()
    interval = config.get("checkpoint_interval", 10)
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
        raise ValueError("checkpoint_interval must be a positive integer")
    stop_reason = "absolute_segment_target"
    try:
        # The initial completed boundary makes a failure in the first segment
        # resumable; subsequent segments already have their entry checkpoint.
        if starting_step == 0:
            save_current_boundary(report, policy, optimizer, scheduler, generator, provenance,
                                  report.data["next_data_position"], device, save_checkpoint)
        while report.data["logical_optimizer_steps"] < stop_after_step:
            recent = []
            for ref in report.data["active_step_records"][-3:]:
                record = json.loads(record_path(report.directory, ref).read_text())
                recent.append(record["step_elapsed_seconds"])
            budget = max(recent) * 1.25 if recent else config.get("step_time_budget_seconds", 90.0)
            if time.monotonic() + budget >= work_deadline:
                stop_reason = "runtime_boundary_budget"
                break
            step = report.data["logical_optimizer_steps"] + 1
            cursor = report.data["next_data_position"]
            if cursor != (step - 1) * config["questions_per_batch"]:
                raise ValueError("Data cursor does not match the logical step")
            questions = selected_questions(rows, cursor, config["questions_per_batch"])
            rng = capture_rng(torch, generator, device)
            before_optimizer_calls = report.data["physical_optimizer_step_calls"]
            report.data.pop("current_completed_rollout", None)
            report.data["physical_rollout_attempts"] += 1
            report.write()
            step_started = time.perf_counter()
            try:
                batch = step_function(policy, reference, tokenizer, questions, optimizer, scheduler,
                                      generator, config, report, f"step_{step}", work_deadline)
            except TimeoutError as error:
                # A long microbatch may abort during backward. No optimizer
                # update has occurred; clear partial gradients and restore RNG.
                if report.data["physical_optimizer_step_calls"] != before_optimizer_calls:
                    raise RuntimeError("Deadline failure occurred after an optimizer update") from error
                restore_rng(torch, generator, device, rng)
                optimizer.zero_grad(set_to_none=True)
                partial = copy.deepcopy(report.data.get("current_partial_rollout"))
                known_partial_tokens = 0 if partial is None else sum(
                    answer["token_count"] for answer in partial.get("completed_answers", []))
                incomplete_lower_bound = 0 if partial is None else partial.get(
                    "incomplete_answer_valid_tokens_lower_bound", 0)
                report.data["discarded_partial_completion_tokens"] = report.data.get(
                    "discarded_partial_completion_tokens", 0) + known_partial_tokens
                report.data["discarded_incomplete_tokens_lower_bound"] = report.data.get(
                    "discarded_incomplete_tokens_lower_bound", 0) + incomplete_lower_bound
                report.data.setdefault("discarded_rollouts", []).append(dict(
                    segment_id=segment_id, logical_step=step, reason=str(error),
                    completed_rollout=copy.deepcopy(report.data.get("current_completed_rollout")),
                    partial_rollout=partial,
                    partial_gradient_accumulation=copy.deepcopy(report.data.get("current_partial_gradient_accumulation")),
                    known_partial_completion_tokens=known_partial_tokens,
                    incomplete_answer_tokens_lower_bound=incomplete_lower_bound,
                    partial_rollout_token_count_unknown=partial is None and report.data.get("current_completed_rollout") is None))
                stop_reason = "deadline_before_update_RNG_restored"
                break
            batch["step_elapsed_seconds"] = time.perf_counter() - step_started
            if report.data["physical_optimizer_step_calls"] != before_optimizer_calls + 1:
                raise RuntimeError("One successful step must perform exactly one optimizer update")
            persist_completed_step(report, segment_id, step, batch)
            report.data["logical_optimizer_steps"] = step
            report.data["next_data_position"] = cursor + config["questions_per_batch"]
            report.data["completed_answer_tokens"] += batch["valid_completion_tokens"]
            report.write()
            if step % interval == 0:
                save_current_boundary(report, policy, optimizer, scheduler, generator, provenance,
                                      report.data["next_data_position"], device, save_checkpoint)
        save_current_boundary(report, policy, optimizer, scheduler, generator, provenance,
                              report.data["next_data_position"], device, save_checkpoint)
        complete = report.data["logical_optimizer_steps"] == target
        segment["status"] = "completed" if complete else "paused_at_boundary"
        report.data["status"] = segment["status"]
        segment["stop_reason"] = "final_training_target" if complete else stop_reason
    except Exception as error:
        segment["status"] = report.data["status"] = "failed"
        segment["error"] = str(error)
        raise
    finally:
        segment.update(
            ended_at_utc=utc_now(), training_segment_seconds=time.perf_counter() - started,
            end_logical_step=report.data["logical_optimizer_steps"],
            segment_logical_optimizer_steps=report.data["logical_optimizer_steps"] - starting_step,
            end_data_position=report.data["next_data_position"],
            physical_counters={key: report.data[key] - starting_counts[key] for key in COUNTERS},
            cumulative_physical_counters={key: report.data[key] for key in COUNTERS},
            segment_committed_answer_tokens=report.data["completed_answer_tokens"] - starting_answer_tokens,
            cumulative_committed_answer_tokens=report.data["completed_answer_tokens"],
            latest_checkpoint=report.data.get("latest_checkpoint"),
        )
        atomic_json(report.directory / "segments" / f"{segment_id}.json", segment)
        report.write()
    return report.data["status"]
