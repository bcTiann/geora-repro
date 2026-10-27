"""Observe the frozen GSM8K reward without changing its scoring semantics.

Use one output file per learner process. The callback is a module-level callable
object so it can be pickled if a trainer hands reward evaluation to a worker.
Group statistics require one explicit question ID per completion; absent IDs
produce an explicit unavailable status rather than guessed group membership.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
import time

from frozen_training.answer_scoring import GSM8K_REWARD_VERSION, score_gsm8k


REWARD_STATUSES = ("correct", "incorrect", "missing_or_incomplete_boxed", "invalid_prediction")


def _question_ids(values, responses):
    if values is None:
        return None
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("Question IDs must be a sequence with one ID per completion")
    if len(values) != responses:
        raise ValueError("Question ID count differs from completion count")
    if any(not isinstance(value, (str, int)) or isinstance(value, bool)
           or not str(value).strip() for value in values):
        raise ValueError("Every supplied question ID must be a nonempty string or integer")
    return [str(value) for value in values]


def _groups(rewards, question_ids, num_generations):
    if question_ids is None:
        return {
            "status": "unavailable_missing_question_ids",
            "group_count": None,
            "zero_std_fraction": None,
            "all_zero_fraction": None,
            "all_one_fraction": None,
            "groups": None,
        }
    if len(rewards) % num_generations:
        raise ValueError("Completion count is not a multiple of num_generations")
    groups = []
    for start in range(0, len(rewards), num_generations):
        ids = question_ids[start:start + num_generations]
        if any(value != ids[0] for value in ids):
            raise ValueError("A consecutive generation group contains different question IDs")
        values = rewards[start:start + num_generations]
        all_zero = all(value == 0.0 for value in values)
        all_one = all(value == 1.0 for value in values)
        groups.append({
            "question_id": ids[0],
            "responses": len(values),
            "reward_mean": sum(values) / len(values),
            "all_zero": all_zero,
            "all_one": all_one,
            "zero_std": all_zero or all_one,
        })
    count = len(groups)
    return {
        "status": "validated_consecutive_groups",
        "group_count": count,
        "zero_std_fraction": sum(row["zero_std"] for row in groups) / count,
        "all_zero_fraction": sum(row["all_zero"] for row in groups) / count,
        "all_one_fraction": sum(row["all_one"] for row in groups) / count,
        "groups": groups,
    }


class AuditedGSM8KReward:
    """TRL-compatible callback using the existing score_gsm8k exactly once/answer."""

    # Preserve the existing metric name when TRL labels this callable's rewards.
    __name__ = "grpo_boxed_reward"

    def __init__(self, audit_path, *, num_generations=8):
        if (not isinstance(num_generations, int) or isinstance(num_generations, bool)
                or num_generations < 1):
            raise ValueError("num_generations must be a positive integer")
        self.audit_path = Path(audit_path)
        self.num_generations = num_generations

    def __call__(self, completions, correct_answer_text, trainer_state=None,
                 log_extra=None, **kwargs):
        if not completions:
            raise ValueError("Cannot audit an empty reward batch")
        question_ids = _question_ids(kwargs.get("id"), len(completions))
        global_step = getattr(trainer_state, "global_step", None)
        if global_step is not None and (not isinstance(global_step, int)
                                        or isinstance(global_step, bool) or global_step < 0):
            raise ValueError("Trainer global_step must be a nonnegative integer")

        started = time.perf_counter()
        records = [
            score_gsm8k(completion[-1]["content"], target)
            for completion, target in zip(completions, correct_answer_text, strict=True)
        ]
        scoring_seconds = time.perf_counter() - started
        rewards = [record["reward"] for record in records]
        grouping = _groups(rewards, question_ids, self.num_generations)
        payload = {
            "schema_version": 1,
            "reward_contract": GSM8K_REWARD_VERSION,
            "global_step": global_step,
            "trainer_state_available": global_step is not None,
            "responses": len(records),
            "question_ids": question_ids,
            "num_generations": self.num_generations,
            "status_counts": {
                status: sum(record["status"] == status for record in records)
                for status in REWARD_STATUSES
            },
            "reward_mean": sum(rewards) / len(rewards),
            "scoring_seconds": scoring_seconds,
            "timing_scope": "Frozen score_gsm8k calls only; excludes grouping, log_extra and file I/O",
            "grouping": grouping,
        }
        if log_extra is not None:
            log_extra("reward_status", [record["status"] for record in records])
            log_extra("extracted_answer", [record["extracted_answer"] for record in records])
        line = json.dumps(payload, allow_nan=False) + "\n"
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8") as stream:
            stream.write(line)
        return rewards


def make_audited_gsm8k_reward(audit_path, *, num_generations=8):
    """Create the audited callback; the runner should retain the dataset's id column."""
    return AuditedGSM8KReward(audit_path, num_generations=num_generations)
