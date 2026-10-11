"""Short MI250X cached-likelihood diagnosis; no optimizer or GRPO updates.

Compare the original model and untrained GeoRA against their own fixed-width
scorer. Keep each case's sampling probabilities, gate result, and timing. The
production rollout path remains unchanged even if this diagnostic succeeds.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import time

import torch
import transformers
from safetensors.torch import load_file
from transformers import AutoTokenizer

from check_cached_rollout_cpu import (
    LOGP_LIMIT, RATIO_LIMIT, SEQUENCE_LIMIT, measure_case,
)
from geora_layers import load_geora_state
from geora_check_utils import (
    CheckReport, fresh_fp32_model, move_with_precision,
    pinned_configuration, validate_manifest,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--init-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    arguments = parser.parse_args()
    if not 1 <= arguments.max_new_tokens <= 32:
        raise ValueError("Use 1-32 tokens for the full-model diagnostic")
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("Run on one allocated Setonix ROCm GPU")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("This full-model diagnostic requires a Slurm allocation")
    torch.set_num_threads(8)
    torch.cuda.reset_peak_memory_stats()
    report = CheckReport(arguments.output_dir, "cached_rollout_gpu.json")
    started = time.perf_counter()
    configuration, original_checkpoint = pinned_configuration()
    staging = json.loads((arguments.checkpoint_dir / "staging_record.json").read_text())
    report.data.update(
        stage="full_model_cached_likelihood_diagnostic", cases=[],
        scope="Short numerical measurement; no task reward or optimizer updates",
        torch_version=torch.__version__, transformers_version=transformers.__version__,
        hip_version=torch.version.hip, gpu_name=torch.cuda.get_device_name(0),
        model_repository=configuration["model_repository"],
        model_revision=configuration["model_revision"],
        source_git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        slurm_job_id=os.environ["SLURM_JOB_ID"], max_new_tokens=arguments.max_new_tokens,
        gates=dict(max_absolute_delta_logp=LOGP_LIMIT,
                   max_absolute_ratio_minus_one=RATIO_LIMIT,
                   max_absolute_sequence_delta_logp=SEQUENCE_LIMIT),
        production_sampler_changed=False, optimizer_steps=0,
        precision=dict(frozen_weights="BF16", A_B_A0_B0="FP32",
                       low_rank_correction="FP32 then BF16 before addition",
                       sampling_log_softmax="FP32", scoring_log_softmax="FP32"),
    )
    try:
        report.require("reuses_CPU_validated_checkpoint_copy",
                       staging["status"] == "validated"
                       and staging["model_revision"] == configuration["model_revision"]
                       and Path(staging["source"]).resolve() == original_checkpoint.resolve())
        manifest = json.loads((arguments.init_dir / "manifest.json").read_text())
        validate_manifest(manifest, configuration)
        state = load_file(str(arguments.init_dir / "adapter.safetensors"))
        report.require("reuses_untrained_FP32_GeoRA_initialization", all(
            torch.equal(state[target["name"] + "." + factor],
                        state[target["name"] + "." + factor + "0"])
            for target in manifest["target_modules"] for factor in ("A", "B")))
        tokenizer = AutoTokenizer.from_pretrained(arguments.checkpoint_dir, local_files_only=True)
        # Unequal prompt lengths exercise a real left-padded 8-answer batch.
        prompts = []
        for text in ("What is 7 + 5? Explain briefly.",
                     "A box has 3 red balls and 5 blue balls. How many balls are there in total?"):
            tokens = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True,
            )
            prompts.extend([tokens] * 4)
        report.data["prompt_lengths"] = list(map(len, prompts))

        loading_started = time.perf_counter()
        native_model = fresh_fp32_model(arguments.checkpoint_dir, disable_mmap=True)
        move_with_precision(native_model, "cuda", torch.bfloat16)
        native_model.eval()
        report.data["native_load_seconds"] = time.perf_counter() - loading_started
        native = measure_case(
            native_model, prompts, tokenizer.pad_token_id, [tokenizer.eos_token_id],
            arguments.max_new_tokens, "cuda", torch.bfloat16, report, "native",
        )
        del native_model
        gc.collect()
        torch.cuda.empty_cache()

        loading_started = time.perf_counter()
        policy = fresh_fp32_model(arguments.checkpoint_dir, disable_mmap=True)
        load_geora_state(policy, state, dict(manifest, forward_mode="difference"))
        del state
        move_with_precision(policy, "cuda", torch.bfloat16)
        policy.train()
        report.data["geora_load_seconds"] = time.perf_counter() - loading_started
        initial = measure_case(
            policy, prompts, tokenizer.pad_token_id, [tokenizer.eos_token_id],
            arguments.max_new_tokens, "cuda", torch.bfloat16, report, "geora_initial",
        )
        report.require("initial_cached_GeoRA_matches_original",
                       torch.equal(native["input_ids"], initial["input_ids"])
                       and torch.equal(native["behavior_logps"], initial["behavior_logps"]))
        report.data["mechanical_checks_passed"] = True
        report.data["likelihood_gate_passed"] = all(
            case["likelihood"]["gate_passed"] for case in report.data["cases"])
        report.data["decision"] = (
            "candidate_requires_updated_policy_and_training_validation"
            if report.data["likelihood_gate_passed"]
            else "keep_validated_fixed_width_training_sampler")
        report.data["peak_allocated_memory_gib"] = torch.cuda.max_memory_allocated() / 2**30
        report.data["elapsed_seconds"] = time.perf_counter() - started
        report.finish()
    except Exception as error:
        report.data.update(status="failed", error=str(error),
                           elapsed_seconds=time.perf_counter() - started)
        report.write()
        raise
    print("Mechanical checks passed; likelihood gate:", report.data["likelihood_gate_passed"])
    print("No optimizer updates. Production training sampler unchanged.")


if __name__ == "__main__":
    main()
