"""Focused CPU safety checks for guards during native cache replay.

Tiny standard Qwen2 models execute the actual cached checkpoint scorer and
long-answer step. A clock expires only once backward starts, exercising the
real deadline guard rather than raising from the forward trace observer.
"""

import copy
import tempfile
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch

REPOSITORY = Path(os.environ.get("GEORA_REPOSITORY", Path.cwd())).resolve()
if not (REPOSITORY / "scripts/check_cached_score_cpu.py").exists():
    REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))
sys.path.insert(0, str(REPOSITORY))

from cached_block_score import score_completion_logps_block_checkpointed
from check_cached_block_score_cpu import fixed_rollout, oracle, require
from check_cached_score_cpu import tiny_adapter
import long_grpo_step


def observe_every_forward_and_replay(checks):
    model = tiny_adapter("geora", torch.float32)
    independent = copy.deepcopy(model)
    rollout = fixed_rollout(steps=9, dtype=torch.float32)
    expected = oracle.score_completion_logps_cached(
        independent, rollout, "cpu", torch.float32, gradient_enabled=True,
    )
    guard_calls, trace_calls = [], []
    statistics = {}
    actual_forwards = []
    hook = model.register_forward_pre_hook(
        lambda module, args, kwargs: actual_forwards.append(kwargs["input_ids"].shape[-1]),
        with_kwargs=True,
    )
    try:
        current = score_completion_logps_block_checkpointed(
            model, rollout, "cpu", torch.float32, gradient_enabled=True, block_size=4,
            trace_callback=trace_calls.append, runtime_guard=guard_calls.append,
            checkpoint_statistics=statistics,
        )
        original_forward_calls = len(actual_forwards)
        (-current[rollout["completion_mask"]].mean()).backward()
        (-expected[rollout["completion_mask"]].mean()).backward()
    finally:
        hook.remove()
    before_calls = [entry for entry in guard_calls if entry["phase"] == "before_forward"]
    forward_guards = [entry for entry in before_calls if not entry["recomputing"]]
    replay_guards = [entry for entry in before_calls if entry["recomputing"]]
    require(checks, "guard_precedes_every_actual_forward_and_backward_replay",
            len(before_calls) == len(actual_forwards)
            and len(forward_guards) == original_forward_calls == rollout["generated_steps"]
            and len(replay_guards) == len(actual_forwards) - original_forward_calls > 0,
            forward_token_calls=len(forward_guards), replay_token_calls=len(replay_guards),
            actual_model_forward_calls=len(actual_forwards), **statistics)
    require(checks, "trace_remains_original_forward_observer_only",
            len(trace_calls) == rollout["generated_steps"]
            and [entry["step"] for entry in trace_calls] == list(range(9)))
    require(checks, "guarded_backward_gradients_remain_finite",
            all(parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
                for parameter in model.parameters() if parameter.requires_grad))
    expected_parameters = dict(independent.named_parameters())
    require(checks, "active_runtime_guard_preserves_native_logps_and_gradients_exactly",
            torch.equal(current, expected)
            and all(torch.equal(parameter.grad, expected_parameters[name].grad)
                    for name, parameter in model.named_parameters() if parameter.requires_grad))


class CpuTokenizer:
    """Small deterministic tokenizer interface for exercising the step wiring."""

    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def encode(self, text, **kwargs):
        return [3, 4, 5]

    def decode(self, ids, **kwargs):
        return "#### " + str(sum(ids) % 2)


class MemoryReport:
    def __init__(self):
        self.data = dict(physical_rollouts=0, physical_backward_calls=0,
                         physical_optimizer_step_calls=0, checks=[])

    def write(self):
        pass

    def require(self, name, condition, **values):
        self.data["checks"].append(dict(name=name, passed=bool(condition), **values))
        if not condition:
            raise RuntimeError(name)


def abort_long_step_on_backward_deadline(checks):
    model = tiny_adapter("geora", torch.float32)
    reference = copy.deepcopy(model).requires_grad_(False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    before = [parameter.detach().clone() for parameter in trainable]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scheduler_before = copy.deepcopy(scheduler.state_dict())
    report = MemoryReport()
    config = dict(answers_per_question=2, max_prompt_tokens=8, max_new_tokens=9,
                  temperature=1.0, truncated_answer_reward=0.0, advantage_epsilon=1e-8,
                  checkpoint_block_tokens=4, scoring_memory_guard_gib=1.0,
                  max_absolute_logp_mismatch=2e-5, max_absolute_ratio_mismatch=2e-5,
                  max_absolute_sequence_logp_mismatch=1e-3,
                  clip_epsilon=0.2, kl_coefficient=0.001, gradient_clip_norm=1.0,
                  rollout_backend="native_cached_block_checkpointed")
    # Advance time only after the real step has entered backward. The earlier
    # outer before-backward deadline check therefore passes, and the inner
    # replay guard must be responsible for the eventual TimeoutError.
    original_time = long_grpo_step.time
    deadline = time.monotonic() + 60.0
    long_grpo_step.time = SimpleNamespace(
        perf_counter=time.perf_counter,
        monotonic=lambda: deadline + 1.0
        if report.data.get("current_stage", "").endswith("_backward") else time.monotonic(),
    )
    caught = None
    try:
        long_grpo_step.one_step(
            model, reference, CpuTokenizer(),
            [dict(id="cpu:0", question="What is zero?", target="0")],
            optimizer, scheduler, torch.Generator().manual_seed(29), config,
            report, "step_1", deadline, device="cpu", frozen_dtype=torch.float32,
        )
    except TimeoutError as error:
        caught = str(error)
    finally:
        long_grpo_step.time = original_time
    partial = report.data.get("current_partial_gradient_accumulation", {})
    statistics = partial.get("current_answer_checkpoint_statistics", {})
    require(checks, "actual_long_step_deadline_aborts_inside_backward_replay",
            caught is not None and "backward_cache_recompute_before_forward_token" in caught
            and statistics.get("runtime_guard_recompute_checks", 0) == 1
            and statistics.get("recompute_cache_forwards", -1) == 0,
            deadline_stage=report.data.get("deadline_stage"), error=caught,
            checkpoint_statistics=statistics)
    require(checks, "backward_timeout_leaves_parameters_optimizer_scheduler_unchanged",
            all(torch.equal(parameter, snapshot) for parameter, snapshot in zip(trainable, before))
            and len(optimizer.state) == 0 and scheduler.state_dict() == scheduler_before
            and report.data["physical_optimizer_step_calls"] == 0
            and not partial.get("optimizer_update_performed", True),
            optimizer_steps=report.data["physical_optimizer_step_calls"],
            completed_backward_answers=partial.get("completed_backward_answers"))
    require(checks, "long_step_uses_named_checkpoint_block_token_configuration",
            statistics.get("block_size") == config["checkpoint_block_tokens"] == 4)


def main():
    torch.set_num_threads(2)
    checks = []
    started = time.perf_counter()
    report = dict(stage="cached_runtime_guard_CPU", status="running", checks=checks,
                  scope="Tiny standard Qwen2 on CPU; guard wiring and aborted update, no GPU run")
    try:
        observe_every_forward_and_replay(checks)
        abort_long_step_on_backward_deadline(checks)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        Path(os.environ.get("GEORA_CHECK_OUTPUT", str(Path(tempfile.gettempdir()) / "geora_runtime_guard_cpu.json"))).write_text(
            json.dumps(report, indent=2) + "\n",
        )


if __name__ == "__main__":
    main()
