"""First vLLM exercise: load Qwen and generate one answer."""

# Print before imports, and avoid repeating this message in spawned workers.
if __name__ == "__main__":
    print("[01] Python script started.", flush=True)

import os


def main():
    # The GPU check runs before engine creation, so workers must use spawn.
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    print("[02] Importing PyTorch...", flush=True)
    import torch
    print("[03] PyTorch imported. Checking visible GPU...", flush=True)

    # PyTorch uses torch.cuda for ROCm GPUs as well as NVIDIA GPUs.
    gpu_available = torch.cuda.is_available()
    print(f"[04] GPU check finished: available={gpu_available}", flush=True)
    if not gpu_available:
        raise RuntimeError(
            "No GPU is visible. Run this script inside the container started "
            "from your allocated GPU compute-node terminal."
        )

    print("[05] Importing vLLM...", flush=True)
    from vllm import LLM, SamplingParams
    print("[06] vLLM imported.", flush=True)

    model_path = os.environ["GEORA_BASE_MODEL"]

    print("[07] Loading Qwen...", flush=True)
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        max_model_len=1024,           # Maximum input + output tokens per sequence.
        gpu_memory_utilization=0.4,   # Fraction of the visible GPU's memory.
        enforce_eager=True,           # Eager execution for this first exercise.
        generation_config="vllm",     # Use explicit sampling choices below.
    )
    print("[08] Qwen loaded. Generating an answer...", flush=True)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=64,                # Maximum number of generated tokens.
    )
    messages = [
        {"role": "user", "content": "What is 17 + 25?"},
    ]

    results = llm.chat(messages, sampling_params)
    print("[09] Generation finished.", flush=True)
    answer = results[0].outputs[0]
    print(answer.text, flush=True)


if __name__ == "__main__":
    main()
