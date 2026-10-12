"""Bounded MI250X comparison from two READ-ONLY real step-2 boundaries.

This wraps the existing setup_models, load_boundary_checkpoint and one_step.
Both diagnostic updates run only in memory. It never writes a training
checkpoint, pilot report, journal or production source. An overall pass needs
both methods to finish and all exact comparisons to succeed within the timer.
"""

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "original"
CANDIDATE = ROOT / "candidate"
sys.dont_write_bytecode = True
sys.path.insert(0, str(ORIGINAL))
sys.path.insert(0, str(ORIGINAL / "scripts"))


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def json_write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def checkpoint_hashes(directory):
    return {name: sha(directory / name) for name in (
        "manifest.json", "adapter.safetensors", "training_state.pt")}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(summary, name, condition, **measurements):
    summary["checks"].append(dict(name=name, passed=bool(condition), **measurements))
    json_write(Path(summary["output_dir"]) / "comparison.json", summary)
    if not condition:
        raise AssertionError(name)


def time_guard(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("The complete diagnostic exceeded its wall-clock limit")


def prepare_paths(args):
    config = json.loads((ORIGINAL / "configs/base_model.json").read_text())
    pinned = args.pinned_base_dir
    if pinned is None:
        staging = args.checkpoint_dir / "staging_record.json"
        pinned = Path(json.loads(staging.read_text())["source"]) if staging.exists() else args.checkpoint_dir
    pinned = pinned.resolve(strict=True)
    output = args.output_dir.resolve()
    protected = [args.checkpoint_dir, pinned, args.init_dir, args.data_dir,
                 args.geora_pilot_dir, args.lora_pilot_dir, ORIGINAL, CANDIDATE]
    if any(output == path.resolve() or output.is_relative_to(path.resolve()) for path in protected):
        raise ValueError("Diagnostic output must be outside all read-only input directories")
    if output.exists():
        raise FileExistsError("Use a fresh diagnostic output directory")
    output.mkdir(parents=True)
    link = ORIGINAL / config["checkpoint_directory"]
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        if link.resolve() != pinned:
            raise ValueError("Existing isolated pinned-base lookup points to another source")
    else:
        link.symlink_to(pinned, target_is_directory=True)
    return output


def run_variant(module, scorer, models, production_config, test_config, row,
                boundary, provenance, deadline, out, name, runtime):
    torch, runner, state, runtime_helpers = runtime
    policy, reference, tokenizer, optimizer, scheduler, generator = models
    time_guard(deadline)
    restore_started = time.perf_counter()
    progress = state.load_boundary_checkpoint(
        boundary, model=policy, optimizer=optimizer, scheduler=scheduler,
        rollout_generator=generator, expected_provenance=provenance, device="cuda")
    if progress["completed_optimizer_steps"] != 2:
        raise ValueError("The diagnostic requires a real completed step-2 boundary")
    restore_seconds = time.perf_counter() - restore_started

    from check_grpo_continuation import clone_cpu
    from geora_check_utils import CheckReport

    named = [(key, parameter) for key, parameter in policy.named_parameters() if parameter.requires_grad]
    initial = dict(parameters=clone_cpu(dict(named)), optimizer=clone_cpu(optimizer.state_dict()),
                   scheduler=clone_cpu(scheduler.state_dict()),
                   rng=runtime_helpers.capture_rng(torch, generator, "cuda"), progress=progress)
    before_immutable = runner.immutable_digests(policy, reference)
    before_buffers = runner.tensor_digest(policy.named_buffers())
    report = CheckReport(out / name, "stage_checks.json")
    report.data.update(physical_rollouts=0, physical_backward_calls=0,
                       physical_optimizer_step_calls=0, diagnostic_only=True,
                       real_training_optimizer_steps_added=0, restored_progress=progress,
                       production_config=production_config, diagnostic_config=test_config)
    calls, rollouts, gradients = [], [], {}
    counters = dict(optimizer=0, scheduler=0)
    instrumentation_seconds = [0.0]
    original_sampler = module.sample_completions_cached_training
    optimizer_step, scheduler_step = optimizer.step, scheduler.step
    clip = torch.nn.utils.clip_grad_norm_

    def snapshot_gradients(key):
        started = time.perf_counter()
        gradients[key] = clone_cpu({n: p.grad for n, p in named})
        instrumentation_seconds[0] += time.perf_counter()-started

    def scoring(scoring_model, rollout, *positional, **keywords):
        time_guard(deadline)
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = scorer(scoring_model, rollout, *positional, **keywords)
        torch.cuda.synchronize()
        seconds = time.perf_counter()-started
        calls.append(dict(model="policy" if scoring_model is policy else "reference",
                          gradient_enabled=keywords["gradient_enabled"],
                          stage=report.data["current_stage"], generated_steps=rollout["generated_steps"],
                          logps=clone_cpu(result), wall_seconds_GPU_synchronized=seconds))
        time_guard(deadline)
        return result

    def sampling(*positional, **keywords):
        result = original_sampler(*positional, **keywords)
        rollouts.append(clone_cpu(result))
        return result

    def clipping(parameters, *positional, **keywords):
        snapshot_gradients("before_clip")
        norm = clip(list(parameters), *positional, **keywords)
        snapshot_gradients("after_clip")
        gradients["norm"] = clone_cpu(norm)
        return norm

    def updating(*positional, **keywords):
        snapshot_gradients("at_optimizer_step")
        counters["optimizer"] += 1
        return optimizer_step(*positional, **keywords)

    def scheduling(*positional, **keywords):
        counters["scheduler"] += 1
        return scheduler_step(*positional, **keywords)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with patch.object(module, "score_completion_logps_block_checkpointed", scoring), \
         patch.object(module, "sample_completions_cached_training", sampling), \
         patch.object(torch.nn.utils, "clip_grad_norm_", clipping), \
         patch.object(optimizer, "step", updating), patch.object(scheduler, "step", scheduling):
        record = module.one_step(
            policy, reference, tokenizer, [row], optimizer, scheduler, generator,
            test_config, report, "step_3", deadline, device="cuda", frozen_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    time_guard(deadline)
    wall_seconds = time.perf_counter()-started
    final_rng = runtime_helpers.capture_rng(torch, generator, "cuda")
    peak = dict(allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    result = dict(initial=initial, record=record, rollouts=rollouts, probability_calls=calls,
                  gradients=gradients, parameters=clone_cpu(dict(named)),
                  optimizer=clone_cpu(optimizer.state_dict()), scheduler=clone_cpu(scheduler.state_dict()),
                  rng=final_rng, counters=counters, restore_seconds=restore_seconds,
                  step_wall_seconds_including_instrumentation=wall_seconds,
                  instrumentation_gradient_copy_seconds=instrumentation_seconds[0], peak=peak,
                  frozen_before=before_immutable, frozen_after=runner.immutable_digests(policy, reference),
                  buffers_before=before_buffers, buffers_after=runner.tensor_digest(policy.named_buffers()),
                  production_cursor_advanced=False, real_training_optimizer_steps_added=0,
                  in_memory_diagnostic_optimizer_updates=report.data["physical_optimizer_step_calls"],
                  physical_backward_calls=report.data["physical_backward_calls"],
                  restored_boundary_nonzero_B=all(torch.count_nonzero(initial["parameters"][n]).item() > 0
                      for n in initial["parameters"] if n.endswith(".B")))
    time_guard(deadline)
    temporary = out / name / "evidence.pt.tmp"
    torch.save(result, temporary)
    temporary.replace(out / name / "evidence.pt")
    json_write(out / name / "step_record.json", dict(record, diagnostic_only=True,
                real_training_optimizer_steps_added=0, production_cursor_advanced=False))
    report.data["status"] = "diagnostic_step_completed_pending_exact_comparison"
    report.write()
    return result


def compare_method(summary, method, baseline, candidate, torch):
    from check_grpo_continuation import trees_equal

    fields = ("answers", "rewards", "advantages", "equal_reward_groups", "parse_failure_count",
              "truncated_answer_count", "valid_completion_tokens", "fixed_input_width", "loss",
              "policy_loss", "sampled_old_token_k3", "clip_fraction", "gradient_norm_before_clipping",
              "changed_parameter_tensors", "learning_rate", "likelihood_alignment", "behavior_logp",
              "old_logp", "reference_logp", "completion_mask", "functional_block_checkpoint_statistics")
    differing = [key for key in fields if baseline["record"][key] != candidate["record"][key]]
    require(summary, method+"_tokens_actual_old_reference_loss_exact", not differing
            and trees_equal(baseline["rollouts"], candidate["rollouts"])
            and candidate["record"]["old_logp"] == candidate["record"]["behavior_logp"], differing_fields=differing)
    bcurrent = [item["logps"] for item in baseline["probability_calls"] if item["gradient_enabled"]]
    ccurrent = [item["logps"] for item in candidate["probability_calls"] if item["gradient_enabled"]]
    require(summary, method+"_independent_current_probabilities_exact", len(bcurrent)==len(ccurrent)==4
            and trees_equal(bcurrent, ccurrent))
    for key in ("initial", "gradients", "parameters", "optimizer", "scheduler", "rng", "counters"):
        require(summary, method+"_"+key+"_exact", trees_equal(baseline[key], candidate[key]))
    require(summary, method+"_frozen_reference_policy_and_all_buffers_unchanged",
            baseline["frozen_before"] == baseline["frozen_after"] == candidate["frozen_before"] == candidate["frozen_after"]
            and baseline["buffers_before"] == baseline["buffers_after"] == candidate["buffers_before"] == candidate["buffers_after"])
    require(summary, method+"_real_step2_nonzero_B_and_one_memory_only_update",
            baseline["restored_boundary_nonzero_B"] and candidate["restored_boundary_nonzero_B"]
            and baseline["counters"] == candidate["counters"] == dict(optimizer=1, scheduler=1)
            and baseline["physical_backward_calls"] == candidate["physical_backward_calls"] == 4
            and all(item["step"].item() == 3 for item in candidate["optimizer"]["state"].values()))
    require(summary, method+"_only_duplicate_old_policy_scoring_passes_removed",
            len(baseline["probability_calls"]) == 12 and len(candidate["probability_calls"]) == 8
            and sum(item["model"]=="policy" and not item["gradient_enabled"] for item in baseline["probability_calls"])==4
            and not any(item["model"]=="policy" and not item["gradient_enabled"] for item in candidate["probability_calls"]))
    initial_rng = candidate["initial"]["rng"]
    require(summary, method+"_global_CPU_CUDA_python_RNG_preserved_private_RNG_advanced",
            all(trees_equal(initial_rng[key], candidate["rng"][key]) for key in ("torch_cpu", "torch_device", "python"))
            and not torch.equal(initial_rng["generator"], candidate["rng"]["generator"]))
    summary["methods"][method]["result"] = dict(
        completion_lengths=[answer["token_count"] for answer in candidate["record"]["answers"]],
        rewards=candidate["record"]["rewards"], advantages=candidate["record"]["advantages"],
        all_equal_rewards=candidate["record"]["equal_reward_groups"],
        reference_KL=candidate["record"]["sampled_old_token_k3"],
        gradient_norm=candidate["record"]["gradient_norm_before_clipping"],
        original_timing={key:baseline["record"][key] for key in ("rollout_seconds", "scoring_seconds", "backward_and_step_seconds")},
        candidate_timing={key:candidate["record"][key] for key in ("rollout_seconds", "scoring_seconds", "backward_and_step_seconds")},
        original_step_wall_seconds=baseline["step_wall_seconds_including_instrumentation"],
        candidate_step_wall_seconds=candidate["step_wall_seconds_including_instrumentation"],
        original_peak=baseline["peak"], candidate_peak=candidate["peak"],
        timing_scope="One fixed question/four<=128-token answers; includes comparison instrumentation; not a 4096-token production runtime estimate")


def execute(args, summary, deadline):
    import torch
    import transformers
    import run_grpo_pilot as runner
    import grpo_training_state as state
    import pilot_runtime as runtime_helpers
    from geora_check_utils import CheckReport
    from gsm8k_data import load_prepared_rows

    if torch.version.hip is None or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError("This driver requires exactly one visible ROCm logical GPU")
    torch.set_num_threads(8)
    runtime = dict(gpu_name=torch.cuda.get_device_name(0), torch_version=str(torch.__version__),
                   hip_version=torch.version.hip, transformers_version=transformers.__version__)
    if runtime["gpu_name"] != "AMD Instinct MI250X":
        raise RuntimeError("The comparison requires the production MI250X runtime")
    summary["runtime"] = runtime
    bundle = json.loads((ROOT / "provenance.json").read_text())
    source_sha = {name:sha(ORIGINAL/name) for name in bundle["original_snapshot_sha256"]}
    candidate_sha = {name:sha(CANDIDATE/name) for name in bundle["candidate_sha256"]}
    require(summary, "isolated_original_and_candidate_sources_match_CPU_proof", source_sha==bundle["original_snapshot_sha256"]
            and candidate_sha==bundle["candidate_sha256"])
    source_files = {str(ORIGINAL/name):value for name,value in source_sha.items()}
    source_files.update({str(CANDIDATE/name):value for name,value in candidate_sha.items()})
    source_files[str(args.config.resolve())] = sha(args.config)
    source_files[str(args.init_dir.resolve()/"manifest.json")] = sha(args.init_dir/"manifest.json")
    source_files[str(args.init_dir.resolve()/"adapter.safetensors")] = sha(args.init_dir/"adapter.safetensors")
    config = json.loads(args.config.read_text())
    runner.validate_config(config)
    test_config = copy.deepcopy(config)
    test_config.update(answers_per_question=4, questions_per_batch=1, max_new_tokens=128)
    rows = load_prepared_rows(args.data_dir, "train")
    found = [row for row in rows if row["id"]=="train:01480"]
    if len(found)!=1 or found[0]["source_split"]!="train":
        raise ValueError("The checksum-verified training split must contain unique train:01480")
    row = found[0]
    summary.update(source_sha256=source_sha, candidate_sha256=candidate_sha,
                   driver_sha256=sha(Path(__file__)), production_config=config, diagnostic_config=test_config,
                   question=row, methods={})
    modules = {
        "original":(load_module("gpu_original_step", ORIGINAL/"scripts/long_grpo_step.py"),
                    load_module("gpu_original_score", ORIGINAL/"scripts/cached_block_score.py")),
        "candidate":(load_module("gpu_candidate_step", CANDIDATE/"scripts/long_grpo_step.py"),
                     load_module("gpu_candidate_score", CANDIDATE/"scripts/cached_block_score.py")),
    }
    for method in ("geora", "lora"):
        time_guard(deadline)
        pilot_dir = getattr(args, method+"_pilot_dir").resolve(strict=True)
        boundary = (getattr(args, method+"_boundary") or pilot_dir/"checkpoints/step-00002").resolve(strict=True)
        if not boundary.is_relative_to((pilot_dir/"checkpoints").resolve()):
            raise ValueError("Input step-2 checkpoint must belong to the specified pilot")
        pilot = json.loads((pilot_dir/"pilot_report.json").read_text())
        manifest = json.loads((boundary/"manifest.json").read_text())
        if pilot["identity"]["method"]!=method or manifest["provenance"]["method"]!=method:
            raise ValueError("Pilot/checkpoint method differs")
        if manifest["progress"]["completed_optimizer_steps"]!=2:
            raise ValueError("Expected completed step2, not another boundary")
        require(summary, method+"_runtime_and_original_source_match_production", all(
            runtime[key]==pilot["runtime"][key] for key in runtime)
            and pilot["identity"]["source_sha256"]==source_sha)
        checkpoint_before = checkpoint_hashes(boundary)
        setup_report = CheckReport(Path(summary["output_dir"])/method/"setup", "checks.json")
        setup_report.data.update(identity=dict(dataset_manifest_sha256=sha(args.data_dir/"manifest.json")),
                                 provenance=pilot["provenance"], diagnostic_only=True)
        setup_started = time.perf_counter()
        models_and_provenance = runner.setup_models(SimpleNamespace(
            method=method, checkpoint_dir=args.checkpoint_dir, init_dir=args.init_dir), config, setup_report)
        models, provenance = models_and_provenance[:6], models_and_provenance[6]
        policy, reference = models[:2]
        require(summary, method+"_reconstructed_original_frozen_reference_matches_pilot",
                runner.immutable_digests(policy,reference)==pilot["initial_immutable_digests"]
                and provenance==manifest["provenance"])
        cursor = manifest["progress"]["next_data_position"]
        summary["methods"][method] = dict(boundary=str(boundary), checkpoint_hashes_before=checkpoint_before,
            saved_progress=manifest["progress"], setup_seconds=time.perf_counter()-setup_started,
            fixed_question_differs_from_next_production_question=(rows[cursor]["id"]!=row["id"]),
            next_production_question_id=rows[cursor]["id"], production_cursor_advanced=False)
        evidence = {}
        for variant in ("original", "candidate"):
            time_guard(deadline)
            module, scorer = modules[variant]
            evidence[variant] = run_variant(module, scorer.score_completion_logps_block_checkpointed,
                models, config, test_config, row, boundary, provenance, deadline,
                Path(summary["output_dir"])/method, variant, (torch,runner,state,runtime_helpers))
        compare_method(summary,method,evidence["original"],evidence["candidate"],torch)
        require(summary,method+"_input_checkpoint_files_unchanged", checkpoint_hashes(boundary)==checkpoint_before)
        del evidence, policy, reference, models, models_and_provenance
        gc.collect()
        torch.cuda.empty_cache()
        time_guard(deadline)
    require(summary,"all_source_config_initialization_files_unchanged", all(sha(Path(path))==value for path,value in source_files.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir",type=Path,required=True,help="Existing verified staged base used by production")
    parser.add_argument("--pinned-base-dir",type=Path,help="Original base path referenced by staging_record; inferred when omitted")
    parser.add_argument("--init-dir",type=Path,required=True,help="Existing untrained GeoRA initialization; no SVD")
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--geora-pilot-dir",type=Path,required=True)
    parser.add_argument("--lora-pilot-dir",type=Path,required=True)
    parser.add_argument("--geora-boundary",type=Path,help="Default: geora pilot checkpoints/step-00002")
    parser.add_argument("--lora-boundary",type=Path,help="Default: lora pilot checkpoints/step-00002")
    parser.add_argument("--config",type=Path,default=ROOT/"production_grpo_long_pilot.json")
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--runtime-limit-seconds",type=float,default=1200)
    args = parser.parse_args()
    if not 0<args.runtime_limit_seconds<=1200:
        parser.error("runtime-limit-seconds must be positive and at most1200")
    started = time.monotonic()
    deadline = started+args.runtime_limit_seconds
    output = prepare_paths(args)
    summary = dict(status="running", output_dir=str(output), checks=[], diagnostic_only=True,
                   watchdog_run_id=os.environ.get("GEORA_SPEED_WATCHDOG_ID"),
                   real_training_optimizer_steps_added=0, production_cursor_advanced=False,
                   scope="Original/candidate memory-only update from each real step2 boundary; fixed train:01480,4answers,max128tokens",
                   wall_clock_limit_seconds=args.runtime_limit_seconds,
                   note="All exact checks required; rewards are actual, with no resampling or adjustment. All-equal rewards/zero KL or gradients limit test coverage.")
    json_write(output/"comparison.json",summary)

    def timeout_handler(signum, frame):
        raise TimeoutError("Hard diagnostic wall-clock limit reached")

    signal.signal(signal.SIGALRM,timeout_handler)
    signal.setitimer(signal.ITIMER_REAL,max(0.001,deadline-time.monotonic()))
    exit_code = 1
    try:
        execute(args,summary,deadline)
        time_guard(deadline)
        summary["status"]="passed"
        summary["in_memory_diagnostic_optimizer_updates"]=4
        exit_code=0
    except TimeoutError as error:
        summary["status"]="timed_out"
        summary["error"]=str(error)
        exit_code=124
    except Exception as error:
        summary["status"]="failed"
        summary["error"]=repr(error)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        summary["elapsed_seconds"]=time.monotonic()-started
        json_write(output/"comparison.json",summary)
    print(json.dumps(dict(status=summary["status"],diagnostic_only=True,
                         real_training_optimizer_steps_added=0,output_dir=str(output))),flush=True)
    return exit_code


if __name__=="__main__":
    sys.exit(main())
