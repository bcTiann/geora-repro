"""CPU-only proof of two temporary long-GRPO efficiency candidates.

Real native Qwen sampling/scoring, the unchanged production reward parser,
loss, clipping, AdamW and scheduler run from snapshotted source. The toy
tokenizer maps tiny vocabulary samples to numeric fixture answers; it is
not a GSM8K task accuracy evaluation. No production source or GPU is touched.
"""

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import random
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "original"
CANDIDATE = ROOT / "candidate"
sys.dont_write_bytecode = True
sys.path.insert(0, str(ORIGINAL))
sys.path.insert(0, str(ORIGINAL / "scripts"))

import numpy as np
import torch
import transformers

from check_cached_score_cpu import tiny_adapter
from geora_check_utils import CheckReport


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


original_score = load_module("original_block_score", ORIGINAL / "scripts/cached_block_score.py")
candidate_score = load_module("candidate_block_score", CANDIDATE / "scripts/cached_block_score.py")
original_step = load_module("original_long_step", ORIGINAL / "scripts/long_grpo_step.py")
candidate_step = load_module("candidate_long_step", CANDIDATE / "scripts/long_grpo_step.py")


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: clone_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(clone_tree(item) for item in value)
    return copy.deepcopy(value)


def exact(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            exact(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            exact(a, b) for a, b in zip(left, right))
    return left == right


def rng_state():
    return dict(torch=torch.get_rng_state().clone(), python=random.getstate(),
                numpy=clone_tree(np.random.get_state()))


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])


def digest(value):
    return hashlib.sha256(value).hexdigest()


def require(report, name, passed, **values):
    item = dict(name=name, passed=bool(passed), **values)
    report["checks"].append(item)
    (ROOT / "proof_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(("PASS " if passed else "FAIL ") + name, flush=True)
    if not passed:
        raise AssertionError(name)


class FixtureTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def encode(self, text, **kwargs):
        return [6, 7, 8, 9, 10] if "fixture-b" in text else [3, 4, 5]

    def decode(self, ids, **kwargs):
        ordinary = [token for token in ids if token not in (0, 2)]
        return "Synthetic tiny-vocabulary fixture.\n#### " + str(sum(ordinary) % 2)


class RecordingAdamW(torch.optim.AdamW):
    def __init__(self, named_parameters, **kwargs):
        self.named_parameters = named_parameters
        self.step_calls = 0
        self.gradients_at_step = None
        super().__init__([parameter for _, parameter in named_parameters], **kwargs)

    def step(self, closure=None):
        self.step_calls += 1
        self.gradients_at_step = {name: parameter.grad.detach().clone()
                                 for name, parameter in self.named_parameters}
        return super().step(closure)


class RecordingScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer):
        self.step_calls = 0
        super().__init__(optimizer, lambda _: 1.0)
        self.step_calls = 0

    def step(self, *args, **kwargs):
        self.step_calls += 1
        return super().step(*args, **kwargs)


def run_step(module, scorer, policy_template, reference_template, config, initial_rng,
             private_state, run_name, *, label="step_2", tamper_behavior=False,
             expect_gate_failure=False):
    model = copy.deepcopy(policy_template)
    reference = copy.deepcopy(reference_template)
    named = [(name, parameter) for name, parameter in model.named_parameters()
             if parameter.requires_grad]
    optimizer = RecordingAdamW(named, lr=config["learning_rate"],
                              betas=tuple(config["optimizer_betas"]),
                              eps=config["optimizer_epsilon"], weight_decay=config["weight_decay"])
    scheduler = RecordingScheduler(optimizer)
    generator = torch.Generator().set_state(private_state)
    initial_model = clone_tree(model.state_dict())
    initial_reference = clone_tree(reference.state_dict())
    probability_calls = []
    sampled_rollouts = []
    gate_calls = []
    snapshots = {}
    original_sampler = module.sample_completions_cached_training
    original_gate = module.likelihood_gate

    def counted_probability(scoring_model, rollout, *args, **kwargs):
        probability_calls.append(dict(model="policy" if scoring_model is model else "reference",
                                      gradient_enabled=kwargs["gradient_enabled"],
                                      generated_steps=rollout["generated_steps"]))
        return scorer(scoring_model, rollout, *args, **kwargs)

    def recorded_sampler(*args, **kwargs):
        rollout = original_sampler(*args, **kwargs)
        if tamper_behavior:
            rollout["behavior_logps"][0, 0] += 0.01
        sampled_rollouts.append(clone_tree(rollout))
        return rollout

    def recorded_gate(report, name, actual, expected, mask, gate_config):
        gate_calls.append(dict(name=name, actual_requires_grad=actual.requires_grad,
                               same_storage=actual.data_ptr() == expected.data_ptr(),
                               actual=actual.detach().clone(), expected=expected.detach().clone(),
                               mask=mask.clone(), limits={key: gate_config[key] for key in (
                                   "max_absolute_logp_mismatch", "max_absolute_ratio_mismatch",
                                   "max_absolute_sequence_logp_mismatch")}))
        return original_gate(report, name, actual, expected, mask, gate_config)

    clip_function = torch.nn.utils.clip_grad_norm_

    def recorded_clip(parameters, *args, **kwargs):
        parameters = list(parameters)
        snapshots["before_clip"] = {name: parameter.grad.detach().clone()
                                    for name, parameter in named}
        norm = clip_function(parameters, *args, **kwargs)
        snapshots["after_clip"] = {name: parameter.grad.detach().clone()
                                   for name, parameter in named}
        snapshots["norm"] = norm.detach().clone()
        return norm

    module.score_completion_logps_block_checkpointed = counted_probability
    module.sample_completions_cached_training = recorded_sampler
    module.likelihood_gate = recorded_gate
    check_report = CheckReport(ROOT / "runs" / run_name, "checks.json")
    check_report.data.update(physical_rollouts=0, physical_backward_calls=0,
                             physical_optimizer_step_calls=0)
    restore_rng(initial_rng)
    started = time.perf_counter()
    log = io.StringIO()
    failure = None
    record = None
    try:
        with patch.object(torch.nn.utils, "clip_grad_norm_", recorded_clip), contextlib.redirect_stdout(log):
            record = module.one_step(
                model, reference, FixtureTokenizer(),
                [dict(id="fixture-a", question="fixture-a", target="1"),
                 dict(id="fixture-b", question="fixture-b", target="1")],
                optimizer, scheduler, generator, config, check_report, label,
                time.monotonic() + 120, device="cpu", frozen_dtype=next(
                    parameter.dtype for parameter in model.parameters() if not parameter.requires_grad))
        if expect_gate_failure:
            raise AssertionError("Deliberately corrupted behavior logp was not rejected")
        check_report.finish()
    except RuntimeError as error:
        if not expect_gate_failure:
            raise
        failure = str(error)
    finally:
        module.sample_completions_cached_training = original_sampler
        module.likelihood_gate = original_gate
        (ROOT / "runs" / run_name / "stdout.txt").write_text(log.getvalue())
    result = dict(record=record, failure=failure, config=copy.deepcopy(config), sampled_rollouts=sampled_rollouts,
                  probability_calls=probability_calls, gates=gate_calls,
                  initial_model=initial_model, initial_reference=initial_reference,
                  model=clone_tree(model.state_dict()), reference=clone_tree(reference.state_dict()),
                  gradients=snapshots, gradients_at_step=optimizer.gradients_at_step,
                  optimizer=clone_tree(optimizer.state_dict()), scheduler=clone_tree(scheduler.state_dict()),
                  optimizer_calls=optimizer.step_calls, scheduler_calls=scheduler.step_calls,
                  rng_initial=clone_tree(initial_rng), rng_after=rng_state(),
                  private_initial=private_state.clone(), private_after=generator.get_state().clone(),
                  counters={key: check_report.data[key] for key in (
                      "physical_rollouts", "physical_backward_calls", "physical_optimizer_step_calls")},
                  seconds=time.perf_counter()-started)
    torch.save(result, ROOT / "runs" / run_name / "full_evidence.pt")
    (ROOT / "runs" / run_name / "step_record.json").write_text(json.dumps(record, indent=2)+"\n")
    return result


def acceptance_gate_cases(report):
    config = json.loads((ROOT / "production_grpo_long_pilot.json").read_text())
    config.update(answers_per_question=4, max_prompt_tokens=64, max_new_tokens=24,
                  checkpoint_block_tokens=7)
    for kind in ("lora", "geora"):
        policy = tiny_adapter(kind, torch.bfloat16)
        policy.lm_head.bias = torch.nn.Parameter(torch.zeros(policy.config.vocab_size, dtype=torch.bfloat16),
                                                 requires_grad=False)
        policy.lm_head.bias[2] = 1.2
        reference = copy.deepcopy(policy).eval().requires_grad_(False)
        torch.manual_seed(8701)
        random.seed(8702)
        np.random.seed(8703)
        state = rng_state()
        private_state = torch.Generator().manual_seed(29).get_state()
        baseline = run_step(original_step, original_score.score_completion_logps_block_checkpointed,
                            policy, reference, config, state, private_state, kind+"_initial_original", label="step_1")
        candidate = run_step(candidate_step, candidate_score.score_completion_logps_block_checkpointed,
                             policy, reference, config, state, private_state, kind+"_initial_candidate", label="step_1")
        initial_gates = [gate for gate in candidate["gates"]
                         if gate["name"].endswith("_initial_policy_matches_original_reference")]
        require(report, kind+"_initial_reference_gate_and_step_one_results_preserved_exact",
                len(initial_gates) == 8 and all(not gate["same_storage"] for gate in initial_gates)
                and all(exact(baseline[key], candidate[key]) for key in (
                    "sampled_rollouts", "gradients", "gradients_at_step", "model", "reference",
                    "optimizer", "scheduler", "rng_after", "private_after"))
                and all(exact(baseline["record"][key], candidate["record"][key]) for key in (
                    "answers", "old_logp", "reference_logp", "loss", "sampled_old_token_k3")))
        rejected = run_step(candidate_step, candidate_score.score_completion_logps_block_checkpointed,
                            policy, reference, config, state, private_state, kind+"_tampered_behavior_candidate",
                            tamper_behavior=True, expect_gate_failure=True)
        require(report, kind+"_corrupted_recorded_behavior_rejected_by_independent_gate_before_update",
                rejected["failure"] is not None
                and "current_and_recorded_behavior_likelihood" in rejected["failure"]
                and rejected["optimizer_calls"] == rejected["scheduler_calls"] == 0
                and rejected["counters"]["physical_backward_calls"] == 0
                and exact(rejected["model"], rejected["initial_model"])
                and len(rejected["gates"]) == 1
                and rejected["gates"][0]["actual_requires_grad"]
                and not rejected["gates"][0]["same_storage"], injected_logp_difference=0.01)


def compare_step(report, kind, dtype, temperature):
    policy = tiny_adapter(kind, dtype)
    # A frozen head bias increases ordinary native EOS probability so the small
    # fixture has varied lengths, including EOS and capped answers.
    policy.lm_head.bias = torch.nn.Parameter(torch.zeros(policy.config.vocab_size, dtype=dtype),
                                             requires_grad=False)
    policy.lm_head.bias[2] = 1.2
    reference = copy.deepcopy(policy).eval().requires_grad_(False)
    perturbation_rng = torch.Generator().manual_seed(1901)
    with torch.no_grad():
        for parameter in policy.parameters():
            if parameter.requires_grad:
                parameter.add_(torch.randn(parameter.shape, generator=perturbation_rng) * 0.08)
    config = json.loads((ROOT / "production_grpo_long_pilot.json").read_text())
    config.update(answers_per_question=4, max_prompt_tokens=64, max_new_tokens=24,
                  checkpoint_block_tokens=7, temperature=temperature)
    torch.manual_seed(8701)
    random.seed(8702)
    np.random.seed(8703)
    initial_rng = rng_state()
    private_state = torch.Generator().manual_seed(29).get_state()
    name = kind + "_" + str(dtype).split(".")[-1] + "_temperature_" + str(temperature)
    baseline = run_step(original_step, original_score.score_completion_logps_block_checkpointed,
                        policy, reference, config, initial_rng, private_state, name + "_original")
    candidate = run_step(candidate_step, candidate_score.score_completion_logps_block_checkpointed,
                         policy, reference, config, initial_rng, private_state, name + "_candidate")
    original_record, candidate_record = baseline["record"], candidate["record"]
    fields = ["answers", "rewards", "advantages", "equal_reward_groups", "parse_failure_count",
              "truncated_answer_count", "valid_completion_tokens", "fixed_input_width", "loss",
              "policy_loss", "sampled_old_token_k3", "clip_fraction", "gradient_norm_before_clipping",
              "changed_parameter_tensors", "learning_rate", "likelihood_alignment", "behavior_logp",
              "old_logp", "reference_logp", "completion_mask", "functional_block_checkpoint_statistics"]
    unequal_fields = [field for field in fields if not exact(original_record[field], candidate_record[field])]
    require(report, name + "_native_rollouts_old_probabilities_losses_exact",
            not unequal_fields and exact(baseline["sampled_rollouts"], candidate["sampled_rollouts"]),
            differing_record_fields=unequal_fields)
    lengths = [answer["token_count"] for answer in original_record["answers"]]
    rewards = [answer["training_reward"] for answer in original_record["answers"]]
    require(report, name + "_nontrivial_variable_lengths_rewards_KL",
            len(set(lengths)) > 1 and set(rewards) == {0.0, 1.0}
            and any(a > 0 for a in original_record["advantages"])
            and any(a < 0 for a in original_record["advantages"])
            and original_record["sampled_old_token_k3"] > 0
            and all(torch.count_nonzero(parameter).item() > 0 for n, parameter in policy.named_parameters()
                    if n.endswith(".B") and parameter.requires_grad),
            completion_lengths=lengths, rewards=rewards,
            reference_KL=original_record["sampled_old_token_k3"],
            truncated_answers=original_record["truncated_answer_count"])
    require(report, name + "_gradients_before_and_after_clip_exact",
            exact(baseline["gradients"], candidate["gradients"])
            and exact(baseline["gradients_at_step"], candidate["gradients_at_step"]),
            gradient_max_absolute_difference=max((baseline["gradients"]["before_clip"][n]
                -candidate["gradients"]["before_clip"][n]).abs().max().item()
                for n in baseline["gradients"]["before_clip"]))
    require(report, name + "_one_AdamW_scheduler_updated_parameters_exact",
            exact(baseline["model"], candidate["model"])
            and exact(baseline["optimizer"], candidate["optimizer"])
            and exact(baseline["scheduler"], candidate["scheduler"])
            and baseline["optimizer_calls"] == candidate["optimizer_calls"] == 1
            and baseline["scheduler_calls"] == candidate["scheduler_calls"] == 1
            and exact(baseline["counters"], candidate["counters"])
            and all(state["step"].item() == 1 for state in candidate["optimizer"]["state"].values())
            and candidate_record["changed_parameter_tensors"] > 0)
    require(report, name + "_global_private_RNG_reference_frozen_state_exact",
            exact(baseline["rng_after"], candidate["rng_after"])
            and exact(initial_rng, candidate["rng_after"])
            and torch.equal(baseline["private_after"], candidate["private_after"])
            and not torch.equal(private_state, candidate["private_after"])
            and exact(baseline["reference"], candidate["reference"])
            and exact(candidate["initial_reference"], candidate["reference"])
            and all(exact(candidate["initial_model"][n], candidate["model"][n])
                    for n, parameter in policy.named_parameters() if not parameter.requires_grad)
            and all(exact(candidate["initial_model"][n], candidate["model"][n])
                    for n, _ in policy.named_buffers() if n in candidate["initial_model"]))
    count = len(original_record["answers"])
    baseline_old = [call for call in baseline["probability_calls"]
                    if call["model"] == "policy" and not call["gradient_enabled"]]
    candidate_old = [call for call in candidate["probability_calls"]
                     if call["model"] == "policy" and not call["gradient_enabled"]]
    require(report, name + "_exactly_duplicate_old_policy_pass_removed",
            len(baseline["probability_calls"]) == count*3
            and len(candidate["probability_calls"]) == count*2
            and len(baseline_old) == count and not candidate_old)
    require(report, name + "_independent_recorded_behavior_gate_limits_unchanged",
            len(candidate["gates"]) == count
            and all(gate["actual_requires_grad"] and not gate["same_storage"]
                    and gate["name"].endswith("_current_and_recorded_behavior_likelihood")
                    and exact(gate["limits"], baseline["gates"][0]["limits"])
                    for gate in candidate["gates"]))
    report["cases"].append(dict(name=name, dtype=str(dtype), temperature=temperature,
                                lengths=lengths, rewards=rewards,
                                reference_KL=original_record["sampled_old_token_k3"],
                                original_probability_calls=count*3, candidate_probability_calls=count*2,
                                original_CPU_seconds=baseline["seconds"], candidate_CPU_seconds=candidate["seconds"],
                                CPU_timing_is_not_GPU_speedup_evidence=True))


def fixed_rollout():
    width, steps, capacity = 5, 9, 12
    ids = torch.randint(7, 37, (2, width+capacity), generator=torch.Generator().manual_seed(908))
    attention = torch.ones_like(ids)
    ids[0, :2] = 0
    attention[0, :2] = 0
    lengths = torch.tensor([steps, steps//2])
    mask = torch.arange(capacity)[None, :] < lengths[:, None]
    ids[:, width:] = torch.where(mask, ids[:, width:], 0)
    for row, length in enumerate(lengths.tolist()):
        ids[row, width+length-1] = 2
    attention[:, width:] = mask.long()
    return dict(input_ids=ids, attention_mask=attention, completion_mask=mask,
                prompt_width=width, generated_steps=steps, max_new_tokens=capacity,
                temperature=0.7, use_cache=True)


def callback_case(report):
    template = tiny_adapter("geora", torch.bfloat16)
    rollout = fixed_rollout()
    cases = {}
    for name, source, detailed, light, gradient in (
        ("original_detailed", original_score, True, False, True),
        ("candidate_detailed", candidate_score, True, False, True),
        ("candidate_progress", candidate_score, False, True, True),
        ("candidate_both", candidate_score, True, True, True),
        ("original_inference", original_score, True, False, False),
        ("candidate_inference", candidate_score, True, True, False),
    ):
        model = copy.deepcopy(template)
        trace, progress, guards = [], [], []
        statistics = {}
        copies = []
        source_filename = str(Path(source.__file__).resolve())
        original_cpu = torch.Tensor.cpu

        def counted_cpu(tensor, *args, **kwargs):
            if sys._getframe(1).f_code.co_filename == source_filename:
                copies.append(True)
            return original_cpu(tensor, *args, **kwargs)

        kwargs = dict(gradient_enabled=gradient, runtime_guard=guards.append,
                      checkpoint_statistics=statistics, block_size=4)
        if detailed:
            kwargs["trace_callback"] = trace.append
        if light:
            kwargs["progress_callback"] = progress.append
        with patch.object(torch.Tensor, "cpu", counted_cpu):
            logps = source.score_completion_logps_block_checkpointed(model, rollout, "cpu", torch.bfloat16,
                                                                    **kwargs)
            before_backward_progress = copy.deepcopy(progress)
            if gradient:
                (-logps[rollout["completion_mask"]].mean()).backward()
        cases[name] = dict(logps=logps.detach().clone(), trace=trace, progress=progress,
                          before_backward_progress=before_backward_progress,
                          guard=guards, statistics=statistics, copies=len(copies),
                          gradients={n: p.grad.detach().clone() for n,p in model.named_parameters()
                                     if p.requires_grad and p.grad is not None})
    base = cases["original_detailed"]
    require(report, "default_detailed_trace_legacy_metadata_values_gradients_exact",
            all(exact(base[key], cases["candidate_detailed"][key])
                for key in ("logps", "trace", "guard", "statistics", "gradients", "copies"))
            and any(item["attention_mask"] is not None for item in base["trace"]))
    require(report, "progress_only_and_both_callbacks_preserve_values_gradients_guards",
            all(exact(base[key], cases[name][key]) for name in ("candidate_progress", "candidate_both")
                for key in ("logps", "guard", "statistics", "gradients"))
            and exact(base["trace"], cases["candidate_both"]["trace"]))
    require(report, "lightweight_progress_has_no_tensor_CPU_copy_and_no_backward_notifications",
            cases["candidate_progress"]["copies"] == 0
            and cases["candidate_progress"]["trace"] == []
            and base["copies"] == cases["candidate_detailed"]["copies"] == cases["candidate_both"]["copies"] > 0
            and all(cases[name]["progress"] == [{"step": i} for i in range(9)]
                    and cases[name]["progress"] == cases[name]["before_backward_progress"]
                    for name in ("candidate_progress", "candidate_both", "candidate_inference")),
            original_detailed_CPU_copy_calls=base["copies"], progress_only_CPU_copy_calls=0)
    require(report, "no_grad_inference_values_detailed_trace_runtime_guard_exact",
            all(exact(cases["original_inference"][key], cases["candidate_inference"][key])
                for key in ("logps", "trace", "guard", "statistics", "copies")))
    torch.save(cases, ROOT / "callback_evidence.pt")


def main():
    torch.set_num_threads(2)
    report = dict(status="running", stage="temporary_speed_candidate_CPU_proof",
                  scope="Tiny native Qwen CPU; no production source, config, allocation or GPU changed",
                  torch_version=torch.__version__, transformers_version=transformers.__version__,
                  cuda_available=torch.cuda.is_available(), checks=[], cases=[],
                  benefits="Removes one old-policy inference pass per answer and per-token detailed trace copies; GPU gain unmeasured",
                  limitation="GPU exact equivalence is not inferred from CPU; if recomputed old vs recorded old differ within production tolerance, substituting the actual recorded old can change numerical loss")
    started = time.perf_counter()
    try:
        for kind in ("lora", "geora"):
            for dtype, temperature in ((torch.float32, 1.0), (torch.bfloat16, 1.0), (torch.bfloat16, 0.7)):
                compare_step(report, kind, dtype, temperature)
        callback_case(report)
        acceptance_gate_cases(report)
        provenance_path = ROOT / "provenance.json"
        provenance = json.loads(provenance_path.read_text())
        repository = Path(provenance["repo"])
        provenance["production_source_after"] = {name:digest((repository/name).read_bytes())
                                                   for name in provenance["production_source_before"]}
        provenance["original_snapshot_sha256"] = {name:digest((ORIGINAL/name).read_bytes())
                                                   for name in provenance["production_source_before"]}
        provenance["candidate_sha256"] = {name:digest((CANDIDATE/name).read_bytes()) for name in (
            "scripts/cached_block_score.py", "scripts/long_grpo_step.py")}
        provenance["harness_sha256"] = digest(Path(__file__).read_bytes())
        provenance["production_config_snapshot_sha256"] = digest((ROOT / "production_grpo_long_pilot.json").read_bytes())
        provenance["production_config_after_sha256"] = digest((repository / "configs/grpo_long_pilot.json").read_bytes())
        provenance_path.write_text(json.dumps(provenance, indent=2)+"\n")
        require(report, "all_19_production_sources_unchanged_and_original_snapshots_exact",
                provenance["production_source_before"] == provenance["production_source_after"]
                == provenance["original_snapshot_sha256"], source_count=19)
        require(report, "production_training_configuration_unchanged",
                provenance["production_config_snapshot_sha256"] == provenance["production_config_after_sha256"])
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter()-started
        (ROOT / "proof_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(dict(status=report["status"], checks=len(report["checks"]),
                          cases=len(report["cases"]), output=str(ROOT)), indent=2), flush=True)


if __name__ == "__main__":
    main()
