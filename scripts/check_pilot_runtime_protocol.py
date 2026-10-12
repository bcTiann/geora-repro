"""CPU-only protocol fixture for pilot segmentation and exact restoration.

This is a small AdamW network, not a model-quality or GPU performance test.
The production runner imports the existing full-model one_step unchanged.
"""

import json
from pathlib import Path
import random
import sys
import tempfile
import time

import torch

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIRECTORY))
sys.path.insert(0, str(PROJECT_DIRECTORY / "scripts"))

from lora_layers import LoRALinear
from grpo_training_state import load_boundary_checkpoint, make_constant_scheduler, save_boundary_checkpoint
from pilot_runtime import (
    COUNTERS, PilotReport, capture_rng, initialization_hash, restore_record_history, run_training_segment,
    selected_questions, training_config_hash,
)
from run_grpo_pilot import tensor_digest


CONFIG = dict(optimizer_steps=4, questions_per_batch=2, checkpoint_interval=2,
              step_time_budget_seconds=0.001, learning_rate=0.003)
ROWS = [dict(id=f"train-{i}", question=str(i), target=str(i), source_split="train") for i in range(3)]
PROVENANCE = dict(method="lora", model_repository="fixture/tiny", model_revision="fixture-v1",
                  dataset_manifest_sha256="a" * 64, training_config_sha256="b" * 64,
                  initialization_sha256="c" * 64)


def setup():
    torch.manual_seed(19)
    base = torch.nn.Linear(3, 2, bias=False)
    A = torch.randn(1, 3)
    B = torch.zeros(2, 1)
    model = torch.nn.Sequential(LoRALinear(base, A, B, scaling=1))
    model.train()
    optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad],
                                 lr=CONFIG["learning_rate"], weight_decay=0)
    scheduler = make_constant_scheduler(optimizer)
    generator = torch.Generator(device="cpu").manual_seed(37)
    torch.manual_seed(101)
    random.seed(103)
    return model, optimizer, scheduler, generator


def report_at(directory):
    directory.mkdir()
    report = PilotReport(directory, dict(
        status="running", checks=[], pilot_run_id=directory.name, active_step_records=[], segments=[],
        logical_optimizer_steps=0, next_data_position=0, completed_answer_tokens=0,
        physical_tokens_accounted_rollout_count=0, **{key: 0 for key in COUNTERS},
    ))
    report.write()
    return report


def mock_step(model, reference, tokenizer, questions, optimizer, scheduler, generator, config, report, label, deadline):
    tokens = torch.randint(0, 50, (4,), generator=generator).tolist()
    inputs = torch.randn(2, 3)
    target = torch.randn(2, 2) + random.random()
    report.data["physical_rollouts"] += 1
    report.data["current_completed_rollout"] = dict(
        label=label, question_ids=[row["id"] for row in questions],
        answers=[dict(token_count=2, token_ids=tokens[:2]), dict(token_count=2, token_ids=tokens[2:])],
    )
    report.write()
    optimizer.zero_grad(set_to_none=True)
    loss = (model(inputs) - target).square().mean()
    loss.backward()
    report.data["physical_backward_calls"] += 1
    optimizer.step()
    scheduler.step()
    report.data["physical_optimizer_step_calls"] += 1
    optimizer.zero_grad(set_to_none=True)
    return dict(question_ids=[row["id"] for row in questions], valid_completion_tokens=4,
                token_ids=tokens, loss=loss.item(), learning_rate=optimizer.param_groups[0]["lr"])


def segment(state, report, stop, step=mock_step):
    model, optimizer, scheduler, generator = state
    return run_training_segment(
        torch=torch, policy=model, reference=None, tokenizer=None, optimizer=optimizer,
        scheduler=scheduler, generator=generator, rows=ROWS, config=CONFIG, report=report,
        provenance=PROVENANCE, stop_after_step=stop, work_deadline=time.monotonic() + 30,
        device="cpu", step_function=step, save_checkpoint=save_boundary_checkpoint,
    )


def snapshot(state):
    model, optimizer, scheduler, generator = state
    return dict(model={name: value.detach().clone() for name, value in model.named_parameters()},
                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                rng=capture_rng(torch, generator, "cpu"))


def equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


def restore(state, report, path):
    model, optimizer, scheduler, generator = state
    progress = load_boundary_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler,
                                       rollout_generator=generator, expected_provenance=PROVENANCE, device="cpu")
    restore_record_history(report, progress)


def batches(report):
    return [json.loads((report.directory / row["relative_path"]).read_text())
            for row in report.data["active_step_records"]]


def main():
    checks = []

    def require(name, condition):
        assert condition, name
        checks.append(name)
        print("PASS:", name)

    require("runtime_segment_controls_do_not_change_training_hash", training_config_hash(CONFIG) == training_config_hash(
        dict(CONFIG, checkpoint_interval=1, runtime_limit_seconds=1, checkpoint_reserve_seconds=10)))
    require("learning_rate_changes_training_hash", training_config_hash(CONFIG) != training_config_hash(
        dict(CONFIG, learning_rate=0.004)))
    require("same_factor_values_with_different_alpha_have_different_identity",
            initialization_hash("a" * 64, 16, 32) != initialization_hash("a" * 64, 16, 64))
    require("absolute_cursor_cycles_without_filtering", [row["id"] for row in selected_questions(ROWS, 2, 2)] == ["train-2", "train-0"])
    require("BF16_CPU_digest_uses_real_storage", tensor_digest([("w", torch.ones(2, 3, dtype=torch.bfloat16))])["parameter_elements"] == 6)
    with tempfile.TemporaryDirectory(prefix="pilot-protocol-") as temporary:
        root = Path(temporary)
        full_state, full_report = setup(), report_at(root / "full")
        segment(full_state, full_report, 4)
        expected = snapshot(full_state)
        split_state, split_report = setup(), report_at(root / "split")
        require("early_segment_pauses_without_claiming_completion", segment(split_state, split_report, 2) == "paused_at_boundary")
        require("initial_and_segment_boundaries_are_written", (split_report.directory / "checkpoints/step-00000/manifest.json").is_file()
                and (split_report.directory / "checkpoints/step-00002/manifest.json").is_file())
        split_state = setup()
        restore(split_state, split_report, Path(split_report.data["latest_checkpoint"]))
        segment(split_state, split_report, 4)
        require("segmented_AB_AdamW_scheduler_and_all_RNG_equal_continuous", equal(snapshot(split_state), expected))
        require("all_four_question_groups_and_tokens_equal_continuous", all(
            a["question_ids"] == b["question_ids"] and a["token_ids"] == b["token_ids"] and a["loss"] == b["loss"]
            for a, b in zip(batches(full_report), batches(split_report))))
        require("logical_and_physical_steps_and_tokens_are_accounted", split_report.data["logical_optimizer_steps"] == 4
                and split_report.data["physical_optimizer_step_calls"] == 4 and split_report.data["physical_completion_tokens"] == 16
                and split_report.data["completed_answer_tokens"] == 16 and split_report.data["next_data_position"] == 8)
        # An older completed checkpoint preserves newer files and the append
        # journal, while restored records become the active logical history.
        old_records = list(split_report.data["active_step_records"])
        split_state = setup()
        restore(split_state, split_report, split_report.directory / "checkpoints/step-00002")
        segment(split_state, split_report, 4)
        require("old_records_preserved_when_replaying_older_boundary", all(
            (split_report.directory / row["relative_path"]).is_file() for row in old_records)
            and len(split_report.data["superseded_step_records"]) == 2
            and len((split_report.directory / "step_journal.jsonl").read_text().splitlines()) == 6)
        require("replay_boundary_immutable_and_physical_work_visible", equal(snapshot(split_state), expected)
                and split_report.data["physical_optimizer_step_calls"] == 6
                and "segment-0003" in split_report.data["latest_checkpoint"])
        paused_state, paused_report = setup(), report_at(root / "timeout")
        rng_before = capture_rng(torch, paused_state[-1], "cpu")

        def timeout_before_update(*args):
            generator, report = args[6], args[8]
            torch.randn(2)
            random.random()
            torch.randint(0, 50, (2,), generator=generator)
            report.data["physical_rollouts"] += 1
            report.data["current_completed_rollout"] = dict(answers=[dict(token_count=2)])
            report.write()
            raise TimeoutError("fixture completed rollout expired before backward")

        require("deadline_pauses_at_last_completed_boundary", segment(paused_state, paused_report, 4, timeout_before_update) == "paused_at_boundary"
                and paused_report.data["logical_optimizer_steps"] == 0
                and paused_report.data["physical_rollouts"] == 1 and paused_report.data["physical_completion_tokens"] == 2)
        require("deadline_rolls_back_every_RNG_before_retry", equal(capture_rng(torch, paused_state[-1], "cpu"), rng_before))
        segment(paused_state, paused_report, 4)
        require("retry_equals_continuous_and_discarded_work_retained", equal(snapshot(paused_state), expected)
                and paused_report.data["physical_rollouts"] == 5 and paused_report.data["physical_completion_tokens"] == 18
                and paused_report.data["completed_answer_tokens"] == 16)
        partial_state, partial_report = setup(), report_at(root / "partial_sampling_timeout")
        partial_before = snapshot(partial_state)

        def timeout_mid_group(*args):
            model, _, _, _, _, _, generator, _, report, _, _ = args
            torch.randn(2)
            random.random()
            torch.randint(0, 50, (2,), generator=generator)
            report.data["current_partial_rollout"] = dict(
                completed_answers=[dict(token_count=3)],
                incomplete_answer_valid_tokens_lower_bound=2,
                partial_answer_token_count_exact=False)
            model(torch.ones(1, 3)).sum().backward()
            report.data["physical_backward_calls"] += 1
            report.data["current_partial_gradient_accumulation"] = dict(
                completed_backward_answers=1, optimizer_update_performed=False)
            raise TimeoutError("fixture partial microbatch backward expired before update")

        require("partial_group_deadline_preserves_original_completed_boundary",
                segment(partial_state, partial_report, 4, timeout_mid_group) == "paused_at_boundary"
                and equal(snapshot(partial_state), partial_before)
                and all(p.grad is None for p in partial_state[0].parameters()))
        require("partial_group_tokens_and_gradient_attempt_are_not_lost",
                partial_report.data["discarded_partial_completion_tokens"] == 3
                and partial_report.data["discarded_incomplete_tokens_lower_bound"] == 2
                and partial_report.data["physical_completion_tokens_lower_bound"] == 5
                and partial_report.data["physical_backward_calls"] == 1
                and partial_report.data["physical_optimizer_step_calls"] == 0
                and partial_report.data["discarded_rollouts"][0]["partial_gradient_accumulation"]["completed_backward_answers"] == 1)
        segment(partial_state, partial_report, 4)
        require("retry_after_partial_backward_reproduces_continuous_update",
                equal(snapshot(partial_state), expected)
                and partial_report.data["physical_completion_tokens"] == 16
                and partial_report.data["physical_completion_tokens_lower_bound"] == 21)

        foreign = dict(PROVENANCE, initialization_sha256="d" * 64)
        try:
            model, optimizer, scheduler, generator = setup()
            load_boundary_checkpoint(Path(paused_report.data["latest_checkpoint"]), model=model,
                optimizer=optimizer, scheduler=scheduler, rollout_generator=generator,
                expected_provenance=foreign, device="cpu")
        except ValueError:
            require("different_initialization_is_rejected", True)
        else:
            require("different_initialization_is_rejected", False)
    print(json.dumps(dict(status="passed", scope="CPU segmentation/checkpoint protocol only", check_count=len(checks), checks=checks), indent=2))


if __name__ == "__main__":
    main()
