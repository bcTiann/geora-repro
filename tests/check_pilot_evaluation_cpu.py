"""CPU gates for validation holdout, strict grading, and atomic adapter restore.

The fixture creates real completed-boundary checkpoints with AdamW and tiny
GeoRA/LoRA layers. It downloads no data/model and uses no accelerator. During
staging, --support-repo supplies the unchanged shared modules from the repo.
"""

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile

PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument("--support-repo", type=Path)
PARSER.add_argument("--output", type=Path)
ARGUMENTS = PARSER.parse_args()
STAGED_ROOT = Path(__file__).resolve().parents[1]
SUPPORT_ROOT = ARGUMENTS.support_repo or STAGED_ROOT
for directory in (SUPPORT_ROOT, SUPPORT_ROOT / "scripts", STAGED_ROOT / "scripts"):
    sys.path.insert(0, str(directory))

import torch
from safetensors.torch import load_file, save_file

from geora_layers import GeoRALinear
from lora_layers import LoRALinear
from grpo_training_state import BOUNDARY, make_constant_scheduler, save_boundary_checkpoint
from gsm8k_data import ANSWER_RULE, SOURCE_COMMIT, file_sha256, write_jsonl
from pilot_runtime import training_config_hash

SPEC = importlib.util.spec_from_file_location(
    "pilot_evaluation_under_test", STAGED_ROOT / "scripts/evaluate_gsm8k_pilot.py")
EVALUATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATION)
CHECKS = []


def require(name, condition):
    CHECKS.append({"name": name, "passed": bool(condition)})
    if not condition:
        raise AssertionError(name)
    print("PASS:", name, flush=True)


def rejects(name, function):
    try:
        function()
    except (ValueError, RuntimeError, KeyError, FileNotFoundError):
        require(name, True)
    else:
        require(name, False)


def row(index, question=None, split="train"):
    return dict(id=f"{split}:{index:05d}", source_split=split,
                source_index=index, question=question or f"Fixture question {index}", target="42")


def dataset(directory, train, validation):
    directory.mkdir(parents=True, exist_ok=True)
    splits = {}
    for split, rows in (("train", train), ("validation", validation)):
        path = directory / f"{split}.jsonl"
        write_jsonl(path, rows)
        splits[split] = dict(relative_path=path.name, rows=len(rows), sha256=file_sha256(path))
    manifest = dict(source_commit=SOURCE_COMMIT, answer_rule=ANSWER_RULE, splits=splits)
    (directory / "manifest.json").write_text(json.dumps(manifest))


class TinyAdapter(torch.nn.Module):
    def __init__(self, method):
        super().__init__()
        torch.manual_seed(701)
        base = torch.nn.Linear(3, 2, bias=False)
        A = torch.randn(2, 3)
        B = torch.randn(2, 2)
        if method == "geora":
            self.projection = GeoRALinear(base, A, B, 2.0, forward_mode="difference")
        else:
            self.projection = LoRALinear(base, A, B, 2.0)

    def forward(self, inputs):
        return self.projection(inputs)


def checkpoint_fixture(directory, method):
    model = TinyAdapter(method)
    initial = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(
        [value for value in model.parameters() if value.requires_grad], lr=1e-3, weight_decay=0.0)
    scheduler = make_constant_scheduler(optimizer)
    inputs = torch.tensor([[0.2, -0.5, 1.0], [0.8, 0.3, -0.7]])
    (model(inputs) - 1.0).square().mean().backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    provenance = dict(method=method, model_repository="tiny/fixture", model_revision="fixture-v1",
                      dataset_manifest_sha256="1" * 64, training_config_sha256="2" * 64,
                      initialization_sha256="3" * 64)
    progress = dict(boundary=BOUNDARY, rollout_in_flight=False,
                    completed_optimizer_steps=1, completed_rollouts=1, next_data_position=2)
    save_boundary_checkpoint(directory, model=model, optimizer=optimizer, scheduler=scheduler,
                             rollout_generator=torch.Generator().manual_seed(91), progress=progress,
                             provenance=provenance, device="cpu")
    return initial, model, provenance


def rewrite_checkpoint(directory, *, tensor_change=None, manifest_change=None):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if tensor_change is not None:
        path = directory / "adapter.safetensors"
        tensors = load_file(str(path))
        tensor_change(tensors)
        save_file(tensors, str(path))
        manifest["files"][path.name] = dict(bytes=path.stat().st_size, sha256=file_sha256(path))
    if manifest_change is not None:
        manifest_change(manifest)
    manifest_path.write_text(json.dumps(manifest))


def model_snapshot(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def check_segmented_evaluation(root):
    """Exercise the real runner's persisted skip/resume flow without a model download."""
    directory = root / "segmented-data"
    dataset(directory, [row(1), row(2)], [row(10), row(11), row(12)])
    output = root / "segmented-output"
    output.mkdir()
    checkpoint = root / "tiny-base"
    checkpoint.mkdir()
    config_path = root / "evaluation-config.json"
    config = dict(prompt_instruction=EVALUATION.PROMPT_INSTRUCTION, validation_question_limit=3,
                  max_prompt_tokens=512, max_new_tokens=2)
    config_path.write_text(json.dumps(config))
    clock = [0.0]
    pause_after_next = [True]
    generated_question_indices = []

    class Tokenizer:
        eos_token_id, pad_token_id, bos_token_id = 99, 0, 1

        @staticmethod
        def apply_chat_template(messages, **kwargs):
            return messages[0]["content"]

        @staticmethod
        def encode(text, **kwargs):
            return [1, int(re.search(r"Fixture question ([0-9]+)", text).group(1))]

        @staticmethod
        def decode(tokens, **kwargs):
            return "#### 42"

    class GenerationModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)

        def generate(self, *, input_ids, **kwargs):
            generated_question_indices.append(input_ids[0, -1].item())
            if pause_after_next[0]:
                clock[0] = 1001.0
                pause_after_next[0] = False
            return torch.cat((input_ids, torch.tensor([[5, 99]])), dim=1)

    saved = {name: getattr(EVALUATION, name) for name in (
        "PROJECT_DIRECTORY", "AutoTokenizer", "pinned_configuration", "fresh_fp32_model", "file_sha256")}
    original_monotonic = EVALUATION.time.monotonic
    def mapped_hash(path):
        path = Path(path)
        relative = path.relative_to(SUPPORT_ROOT) if path.is_relative_to(SUPPORT_ROOT) else None
        if relative in (Path("scripts/evaluate_gsm8k_pilot.py"), Path("scripts/pilot_runtime.py")):
            path = STAGED_ROOT / relative
        return file_sha256(path)
    try:
        EVALUATION.PROJECT_DIRECTORY = SUPPORT_ROOT
        EVALUATION.AutoTokenizer = type("FixtureAutoTokenizer", (), {
            "from_pretrained": staticmethod(lambda *args, **kwargs: Tokenizer())})
        EVALUATION.pinned_configuration = lambda: (
            dict(model_repository="tiny/fixture", model_revision="fixture-v1"), checkpoint)
        EVALUATION.fresh_fp32_model = lambda *args, **kwargs: GenerationModel()
        EVALUATION.file_sha256 = mapped_hash
        EVALUATION.time.monotonic = lambda: clock[0]
        arguments = argparse.Namespace(method="base", config=config_path, question_limit=None,
            runtime_limit_seconds=1000, data_dir=directory, checkpoint_dir=checkpoint,
            output_dir=output, adapter_checkpoint=None, init_dir=None, device="cpu", resume=False)
        report_path = output / "validation.json"
        report = dict(status="running", answers=[])
        EVALUATION.run(arguments, report, report_path)
        paused = json.loads(report_path.read_text())
        require("real_runner_pauses_after_completed_question", paused["status"] == "paused_between_questions"
                and len(paused["answers"]) == 1 and generated_question_indices == [10])
        first_answer = copy.deepcopy(paused["answers"][0])
        arguments.resume = True
        clock[0] = 0.0
        EVALUATION.run(arguments, paused, report_path)
        completed = json.loads(report_path.read_text())
        require("real_runner_resume_generates_only_remaining_questions", generated_question_indices == [10, 11, 12]
                and completed["status"] == "completed" and completed["answers"][0] == first_answer)
        require("real_runner_resume_complete_summary_and_segments", completed["correct_count"] == 3
                and completed["accuracy"] == 1.0 and len(completed["segments"]) == 2
                and completed["segments"][1]["resumed_after_questions"] == 1)
    finally:
        EVALUATION.time.monotonic = original_monotonic
        for name, value in saved.items():
            setattr(EVALUATION, name, value)


def main():
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="geora-evaluation-cpu-") as temporary:
        root = Path(temporary)
        valid = root / "valid-data"
        train = [row(1), row(2)]
        validation = [row(10), row(11)]
        dataset(valid, train, validation)
        require("validation_selects_ordered_holdout_only", EVALUATION.validation_rows(valid, 1) == validation[:1])
        for value in (0, False, 3):
            rejects(f"invalid_validation_limit_{value!r}", lambda value=value: EVALUATION.validation_rows(valid, value))
        dataset(root / "id-leak", train, [row(1)])
        rejects("training_ID_cannot_enter_validation", lambda: EVALUATION.validation_rows(root / "id-leak", 1))
        dataset(root / "question-leak", train, [row(10, question=train[0]["question"])])
        rejects("same_question_under_another_ID_is_leakage", lambda: EVALUATION.validation_rows(root / "question-leak", 1))
        dataset(root / "test-leak", train, [row(10, split="test")])
        rejects("test_source_cannot_enter_validation", lambda: EVALUATION.validation_rows(root / "test-leak", 1))
        dataset(root / "late-leak", train, [row(10), row(1)])
        rejects("unselected_validation_tail_also_checked", lambda: EVALUATION.validation_rows(root / "late-leak", 1))
        dataset(root / "duplicate-ID", train, [row(10), row(10)])
        rejects("duplicate_validation_ID_rejected", lambda: EVALUATION.validation_rows(root / "duplicate-ID", 1))
        (valid / "validation.jsonl").write_text("{}\n")
        rejects("changed_data_checksum_rejected", lambda: EVALUATION.validation_rows(valid, 1))

        target = row(10)
        correct = EVALUATION.grade_completion([5, 99], [99], "Reasoning 7.\n#### 42", target)
        wrong = EVALUATION.grade_completion([6, 99], [99], "#### 43", target)
        truncated = EVALUATION.grade_completion([5, 7], [99], "#### 42", target)
        require("correct_strict_answer_with_EOS_reward_one", correct["evaluation_reward"] == 1)
        require("parsed_arithmetic_error_is_not_parse_failure", wrong["predicted_answer"] == "43" and wrong["evaluation_reward"] == 0)
        require("numeric_correctness_without_EOS_still_reward_zero", truncated["reward"] == 1 and truncated["evaluation_reward"] == 0)
        malformed = []
        for text in ("42", "#### 42 dollars", "#### 42\n#### 42", "#### 6 * 7"):
            answer = EVALUATION.grade_completion([8, 99], [99], text, target)
            malformed.append(answer)
            require("strict_format_rejected_" + text.replace("\n", "_"), answer["predicted_answer"] is None and answer["evaluation_reward"] == 0)
        rejects("tokens_after_EOS_rejected", lambda: EVALUATION.grade_completion([99, 5], [99], "#### 42", target))
        rejects("empty_completion_token_record_rejected", lambda: EVALUATION.grade_completion([], [99], "#### 42", target))
        summary = EVALUATION.evaluation_summary([correct, wrong, truncated, malformed[0]], 4)
        require("summary_separates_format_arithmetic_and_truncation", summary["correct_count"] == 1
                and summary["parse_failure_count"] == 1 and summary["parsed_incorrect_count"] == 1
                and summary["correct_but_truncated_count"] == 1 and summary["accuracy"] == 0.25)

        identity = dict(method="geora", training_config_sha256="2" * 64, question_ids=[target["id"]],
                        decode=dict(max_new_tokens=2, eos_token_ids=[99]), source_sha256={"script": "4" * 64})
        previous = dict(status="paused_between_questions", **copy.deepcopy(identity))
        EVALUATION.validate_resume_identity(previous, identity)
        require("same_evaluation_identity_can_resume", True)
        for key in ("training_config_sha256", "question_ids", "decode", "source_sha256", "method"):
            changed = copy.deepcopy(identity)
            changed[key] = "different"
            rejects("resume_identity_change_" + key, lambda changed=changed: EVALUATION.validate_resume_identity(previous, changed))
        rejects("completed_evaluation_not_restarted", lambda: EVALUATION.validate_resume_identity(dict(previous, status="completed"), identity))
        stored = dict(correct, generation_seconds=0.1)
        decoder = lambda tokens: "Reasoning 7.\n#### 42"
        EVALUATION.validate_answer_prefix([stored], [target], [99], 2, decoder)
        require("saved_completed_question_prefix_valid", True)
        for key, value in (("question_id", "train:99999"), ("evaluation_reward", 0.0), ("completion", "#### 43"), ("token_ids", [5])):
            changed = dict(stored, **{key: value})
            rejects("saved_prefix_corruption_" + key,
                    lambda changed=changed: EVALUATION.validate_answer_prefix([changed], [target], [99], 2, decoder))
        config = dict(max_new_tokens=256, runtime_limit_seconds=1000, scope="segment A")
        require("segment_controls_do_not_change_training_identity",
                training_config_hash(config) == training_config_hash(dict(config, runtime_limit_seconds=800, scope="segment B")))
        require("token_cap_changes_training_identity", training_config_hash(config) != training_config_hash(dict(config, max_new_tokens=512)))
        check_segmented_evaluation(root)

        for method in ("geora", "lora"):
            source = root / f"{method}-checkpoint"
            initial, trained, provenance = checkpoint_fixture(source, method)
            destination = copy.deepcopy(initial)
            frozen_before = {name: value.detach().clone() for name, value in destination.named_parameters() if not value.requires_grad}
            buffers_before = {name: value.clone() for name, value in destination.named_buffers()}
            parameter_ids = {name: id(value) for name, value in destination.named_parameters()}
            rng_before = torch.get_rng_state().clone()
            EVALUATION.restore_adapter_for_evaluation(destination, source, provenance)
            require(f"{method}_real_boundary_AB_restored", all(torch.equal(value, dict(trained.named_parameters())[name])
                    for name, value in destination.named_parameters() if value.requires_grad))
            require(f"{method}_Parameter_objects_preserved", all(id(value) == parameter_ids[name] for name, value in destination.named_parameters()))
            require(f"{method}_frozen_base_and_initial_buffers_unchanged", all(torch.equal(dict(destination.named_parameters())[name], value)
                    for name, value in frozen_before.items()) and all(torch.equal(dict(destination.named_buffers())[name], value)
                    for name, value in buffers_before.items()))
            require(f"{method}_restore_does_not_change_CPU_RNG", torch.equal(torch.get_rng_state(), rng_before))
            require(f"{method}_restored_AB_FP32", all(value.dtype == torch.float32 for value in destination.parameters() if value.requires_grad))
            if method != "geora":
                continue
            cases = {
                "wrong_tensor_name": dict(tensor_change=lambda tensors: tensors.__setitem__("unexpected.A", tensors.pop("projection.A"))),
                "wrong_shape": dict(tensor_change=lambda tensors: tensors.__setitem__("projection.A", tensors["projection.A"][:1].clone())),
                "BF16_factor": dict(tensor_change=lambda tensors: tensors.__setitem__("projection.B", tensors["projection.B"].bfloat16())),
                "nonfinite_late_factor": dict(tensor_change=lambda tensors: tensors.__setitem__("projection.B0", torch.full_like(tensors["projection.B0"], float("nan")))),
                "changed_initial_B0": dict(tensor_change=lambda tensors: tensors.__setitem__("projection.B0", tensors["projection.B0"] + 1)),
                "bad_provenance_hash": dict(manifest_change=lambda manifest: manifest.__setitem__("provenance_sha256", "0" * 64)),
                "wrong_method": dict(manifest_change=lambda manifest: manifest["provenance"].__setitem__("method", "lora")),
                "in_flight_boundary": dict(manifest_change=lambda manifest: manifest["progress"].__setitem__("rollout_in_flight", True)),
                "wrong_boundary": dict(manifest_change=lambda manifest: manifest.__setitem__("boundary", "during_rollout")),
                "bool_checkpoint_format": dict(manifest_change=lambda manifest: manifest.__setitem__("training_checkpoint_format", True)),
                "bad_manifest_shape": dict(manifest_change=lambda manifest: manifest["adapter_tensors"]["projection.B"].__setitem__("shape", [999, 2])),
                "bad_file_hash": dict(manifest_change=lambda manifest: manifest["files"]["adapter.safetensors"].__setitem__("sha256", "0" * 64)),
            }
            import shutil
            for name, changes in cases.items():
                corrupted = root / ("corrupt-" + name)
                shutil.copytree(source, corrupted)
                rewrite_checkpoint(corrupted, **changes)
                destination = copy.deepcopy(initial)
                before = model_snapshot(destination)
                rejects("corrupt_checkpoint_" + name, lambda destination=destination, corrupted=corrupted:
                        EVALUATION.restore_adapter_for_evaluation(destination, corrupted, provenance))
                require("no_partial_restore_" + name, all(torch.equal(value, before[key]) for key, value in destination.state_dict().items()))
            destination = copy.deepcopy(initial)
            destination.projection.A.data = destination.projection.A.data.bfloat16()
            rejects("BF16_destination_factor_rejected", lambda: EVALUATION.restore_adapter_for_evaluation(destination, source, provenance))
            destination = copy.deepcopy(initial)
            destination.projection.base_layer.weight.requires_grad_(True)
            rejects("unexpected_trainable_base_rejected", lambda: EVALUATION.restore_adapter_for_evaluation(destination, source, provenance))

    report = dict(status="passed", scope="CPU-only structural gates; no full-model evaluation or GPU execution",
                  passed_count=len(CHECKS), check_count=len(CHECKS), checks=CHECKS)
    if ARGUMENTS.output is not None:
        ARGUMENTS.output.parent.mkdir(parents=True, exist_ok=True)
        ARGUMENTS.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "scope", "passed_count", "check_count")}, indent=2))


if __name__ == "__main__":
    main()
