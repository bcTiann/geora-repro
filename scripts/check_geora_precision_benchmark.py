"""Draft standalone GeoRA runtime microbenchmark; no sampling or SVD.

Run BF16 and FP32 in separate processes. Measures the same fixed token batch,
FP32 factors, AdamW update and loss. Does not benchmark task quality or a whole
RL training job. Existing answer tokens may be truncated only as an explicit
runtime workload; original complete-answer advantages remain fixed labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--init-dir", type=Path)
    parser.add_argument("--source-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], required=True)
    parser.add_argument("--batch", type=int, choices=[4, 8], default=4)
    parser.add_argument("--response-limit", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    args.repo = args.repo.resolve()
    sys.path[:0] = [str(args.repo), str(args.repo / "scripts")]
    import torch
    import transformers
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from geora_layers import load_geora_state
    from geora_check_utils import fresh_fp32_model, move_with_precision
    from grpo_math import grpo_loss
    from grpo_rollout import score_completion_logps
    from gsm8k_data import build_prompt
    from precision_benchmark_helpers import benchmark, synchronize, memory_snapshot

    args.checkpoint_dir = args.checkpoint_dir or args.repo / "checkpoints/geora_base"
    args.init_dir = args.init_dir or args.repo / "outputs/geora_full_model_check"
    args.source_report = args.source_report or args.repo / "reports/grpo_cached_continuation/50660901.json"
    if not 1 <= args.response_limit <= 256:
        raise ValueError("Response limit must be in [1,256]")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Metal is unavailable; use an authorized unsandboxed local run")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP GPU unavailable")
    torch.set_num_threads(4)
    torch.manual_seed(20261011)
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "precision_benchmark.json"
    report = {
        "status": "running", "dtype": args.dtype, "device": args.device,
        "scope": "independent fixed-data first-step runtime trials; no rollout or task score",
        "real_original_reference_model_resident": False,
        "reference_anchor": "same-condition untrained fixed-path old probabilities",
        "source_report": str(args.source_report),
        "source_report_sha256": hashlib.sha256(args.source_report.read_bytes()).hexdigest(),
        "precision": {"adapters_initial_buffers_gradients_optimizer": "FP32",
                      "logits_log_softmax_loss": "FP32",
                      "frozen_model": args.dtype,
                      "shared_frozen_values": "BF16 rounded once; exact upcast for FP32"},
        "attention_backend": "eager", "training_probability_path": "fixed_width_no_cache",
        "gradient_checkpointing": False, "optimizer_foreach": False,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "hip_version": torch.version.hip,
        "slurm_job_id": __import__("os").environ.get("SLURM_JOB_ID"),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else args.device,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__), Path(__file__).with_name("precision_benchmark_helpers.py"),
                args.repo / "geora_layers.py", args.repo / "scripts/grpo_rollout.py",
                args.repo / "scripts/grpo_math.py", args.repo / "scripts/geora_check_utils.py",
            )
        },
    }
    def persist():
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    persist()
    try:
        source = json.loads(args.source_report.read_text())
        batch = source["batches"][0]
        answers = batch["answers"][:args.batch]
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, local_files_only=True)
        prompt_ids = []
        for answer in answers:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": build_prompt(answer["question"])}],
                tokenize=False, add_generation_prompt=True,
            )
            prompt_ids.append(tokenizer.encode(text, add_special_tokens=False))
        width = max(map(len, prompt_ids))
        capacity = args.response_limit
        ids = torch.full((args.batch, width + capacity), tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        completion_mask = torch.zeros((args.batch, capacity), dtype=torch.bool)
        lengths = []
        for index, (prompt, answer) in enumerate(zip(prompt_ids, answers)):
            ids[index, width-len(prompt):width] = torch.tensor(prompt)
            mask[index, width-len(prompt):width] = 1
            completion = answer["token_ids"][:capacity]
            lengths.append(len(completion))
            ids[index, width:width+len(completion)] = torch.tensor(completion)
            mask[index, width:width+len(completion)] = 1
            completion_mask[index, :len(completion)] = True
        input_digest = hashlib.sha256()
        for tensor in (ids, mask, completion_mask):
            input_digest.update(tensor.numpy().tobytes())
        report.update(
            input_sha256=input_digest.hexdigest(), question_ids=[a["question_id"] for a in answers],
            complete_source_answer_lengths=[len(a["token_ids"]) for a in answers],
            benchmark_completion_lengths=lengths,
            truncation_is_runtime_workload_only=True,
            advantages_from_original_complete_answers=batch["advantages"][:args.batch],
        )
        rollout = {"input_ids": ids.to(args.device), "attention_mask": mask.to(args.device),
                   "completion_mask": completion_mask.to(args.device), "prompt_width": width,
                   "max_new_tokens": capacity, "generated_steps": max(lengths),
                   "fixed_width": width+capacity, "temperature": source["config"]["temperature"]}
        advantages = torch.tensor(batch["advantages"][:args.batch], device=args.device)
        if not torch.count_nonzero(advantages).item():
            raise RuntimeError("Source batch has zero advantage; choose an explicit nonzero fixed workload")
        print("Loading one untrained GeoRA model", flush=True)
        model = fresh_fp32_model(args.checkpoint_dir)
        metadata = json.loads((args.init_dir / "manifest.json").read_text())
        report["source_forward_mode"] = metadata.get("forward_mode", "residual")
        report["actual_forward_mode"] = "difference"
        report["initial_adapter_file_sha256"] = hashlib.sha256((args.init_dir / "adapter.safetensors").read_bytes()).hexdigest()
        report["initial_manifest_sha256"] = hashlib.sha256((args.init_dir / "manifest.json").read_bytes()).hexdigest()
        metadata = dict(metadata, forward_mode="difference")
        state = load_file(str(args.init_dir / "adapter.safetensors"))
        if not all(torch.equal(state[t["name"]+"."+f], state[t["name"]+"."+f+"0"])
                   for t in metadata["target_modules"] for f in ("A", "B")):
            raise RuntimeError("Require an untrained initialization, not a trained adapter")
        load_geora_state(model, state, metadata)
        del state
        from geora_layers import GeoRALinear
        assert all(module.A0.dtype == torch.float32 and module.B0.dtype == torch.float32
                   for module in model.modules() if isinstance(module, GeoRALinear))
        # Both processes start from these same numeric frozen values.
        frozen_digest = hashlib.sha256()
        count = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                parameter.data = parameter.data.to(torch.bfloat16)
                frozen_digest.update(name.encode())
                frozen_digest.update(parameter.detach().view(torch.uint8).numpy().tobytes())
                count += parameter.numel()
        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
        move_with_precision(model, args.device, dtype)
        model.train()
        model.config.use_cache = False
        model.config.attention_dropout = 0.0
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert all(p.dtype == torch.float32 for p in trainable)
        assert all(value.dtype == torch.float32 for name, value in model.named_buffers()
                   if name.endswith((".A0", ".B0")))
        optimizer = torch.optim.AdamW(trainable, lr=1e-6, betas=(0.9,0.999),
                                      eps=1e-8, weight_decay=0.0, foreach=False)
        synchronize(args.device)
        report.update(frozen_value_sha256=frozen_digest.hexdigest(),
                      frozen_parameter_count=count,
                      trainable_parameter_count=sum(p.numel() for p in trainable),
                      initial_resident_memory=memory_snapshot(args.device),
                      setup_seconds=time.perf_counter()-started)
        persist()
        print("Compute fixed old probabilities outside current-update timing", flush=True)
        old_started = time.perf_counter()
        old = score_completion_logps(model, rollout, args.device, dtype, gradient_enabled=False)
        synchronize(args.device)
        report["one_old_probability_forward_seconds"] = time.perf_counter()-old_started
        reference = old.clone()
        print("Warmup and timed independent first updates", flush=True)
        report["benchmark"] = benchmark(
            model, optimizer, score_completion_logps, grpo_loss, rollout,
            args.device, dtype, old, reference, advantages,
            warmup=args.warmup, repeats=args.repeats,
        )
        if not all(torch.isfinite(p).all().item() for p in trainable):
            raise RuntimeError("Non-finite adapter parameter")
        if not all(s["exp_avg"].dtype == torch.float32 and s["exp_avg_sq"].dtype == torch.float32
                   for s in optimizer.state.values()):
            raise RuntimeError("Optimizer moments must remain FP32")
        report.update(status="passed", elapsed_seconds=time.perf_counter()-started)
        persist()
        print("Report:", report_path, flush=True)
    except Exception as error:
        report.update(status="failed", error=str(error), elapsed_seconds=time.perf_counter()-started)
        persist()
        raise


if __name__ == "__main__":
    main()
