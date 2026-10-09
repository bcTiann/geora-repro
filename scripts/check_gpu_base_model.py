"""Check one BF16 forward pass of the pinned base model on a ROCm GPU.

This does not install GeoRA, generate a full answer, or update parameters.
"""

import argparse
import json
import os
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()

    project_directory = Path(__file__).resolve().parents[1]
    configuration = json.loads(
        (project_directory / "configs/base_model.json").read_text()
    )
    checkpoint_directory = project_directory / configuration["checkpoint_directory"]
    metadata_path = (
        checkpoint_directory / ".cache/huggingface/download/model.safetensors.metadata"
    )
    if metadata_path.read_text().splitlines()[0] != configuration["model_revision"]:
        raise RuntimeError("Checkpoint revision does not match the pinned configuration.")
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("This check requires a visible ROCm GPU in a Slurm allocation.")

    started = time.perf_counter()
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Checkpoint:", checkpoint_directory.resolve(), flush=True)
    torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_directory,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_directory,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    )
    model.requires_grad_(False)
    model.eval()
    model.to("cuda")
    print("Model loaded in BF16 on GPU.", flush=True)

    messages = [{"role": "user", "content": "What is 2 + 3? Answer briefly."}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {name: value.to("cuda") for name, value in inputs.items()}

    # Forward computes vocabulary logits for each prompt position, not a full answer.
    with torch.inference_mode():
        logits = model(**inputs, use_cache=False).logits
    torch.cuda.synchronize()
    if not torch.isfinite(logits).all().item():
        raise RuntimeError("Forward produced non-finite logits.")

    report = {
        "stage": "base_model_bf16_forward_only",
        "status": "passed",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "model_repository": configuration["model_repository"],
        "model_revision": configuration["model_revision"],
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "gpu_name": torch.cuda.get_device_name(0),
        "parameter_dtype": str(next(model.parameters()).dtype),
        "logits_shape": list(logits.shape),
        "logits_dtype": str(logits.dtype),
        "logits_all_finite": True,
        "peak_allocated_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "elapsed_seconds": time.perf_counter() - started,
        "optimizer_steps": 0,
    }
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = arguments.output_dir / "base_model_forward.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print("Base-model GPU BF16 forward passed.", flush=True)
    print("Report:", report_path, flush=True)


if __name__ == "__main__":
    main()
