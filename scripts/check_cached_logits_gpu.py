"""One-native-model pre-softmax logits diagnostic on one allocated ROCm GPU."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import torch
import transformers
from transformers import AutoTokenizer

from cached_logits_diagnostic import compare_native_logits
from check_cached_rollout_gpu import diagnostic_prompt_token_ids
from geora_check_utils import CheckReport, fresh_fp32_model, move_with_precision, pinned_configuration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.version.hip is None or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Use one allocated Setonix ROCm GPU")
    torch.set_num_threads(8)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    report = CheckReport(args.output_dir, "cached_logits_gpu.json")
    configuration, original = pinned_configuration()
    report.data.update(
        stage="native_pre_softmax_logits_diagnostic", cases=[], model_count=1,
        optimizer_steps=0, production_sampler_changed=False,
        torch_version=torch.__version__, transformers_version=transformers.__version__,
        hip_version=torch.version.hip, gpu_name=torch.cuda.get_device_name(0),
        model_repository=configuration["model_repository"],
        model_revision=configuration["model_revision"],
        source_git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        slurm_job_id=os.environ["SLURM_JOB_ID"],
        scope="One frozen original model, same token histories, no GeoRA or training",
    )
    try:
        staging = json.loads((args.checkpoint_dir / "staging_record.json").read_text())
        report.require("CPU_validated_pinned_copy", staging["status"] == "validated"
                       and staging["model_revision"] == configuration["model_revision"]
                       and Path(staging["source"]).resolve() == original.resolve())
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, local_files_only=True)
        prompts = diagnostic_prompt_token_ids(tokenizer)
        model = fresh_fp32_model(args.checkpoint_dir, disable_mmap=True)
        move_with_precision(model, "cuda", torch.bfloat16)
        model.eval()
        record = compare_native_logits(
            model, prompts, tokenizer.pad_token_id, [tokenizer.eos_token_id],
            16, "cuda", torch.bfloat16, report, "native_bf16", check_prefix=True,
        )
        worst = record["worst_probability_position"]
        worst["sampled_token_text"] = tokenizer.decode([worst["sampled_token_id"]])
        worst["semantic_prefix_text"] = tokenizer.decode(worst["semantic_prefix_token_ids"])
        report.data.update(elapsed_seconds=time.perf_counter() - started,
                           peak_allocated_memory_gib=torch.cuda.max_memory_allocated() / 2**30)
        report.finish()
    except Exception as error:
        report.data.update(status="failed", error=str(error),
                           elapsed_seconds=time.perf_counter() - started)
        report.write()
        raise


if __name__ == "__main__":
    main()
