"""Compare full greedy GSM8K answers using four native Transformers paths.

One frozen original model, not four trained models. Native DynamicCache is
provided by Transformers. No-cache generation recomputes the growing prefix.
All four paths start from the same BF16 checkpoint values. FP32 promotion
does not recover checkpoint bits. Only selected-token summaries are retained.
"""

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessor

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
from gsm8k_data import build_prompt, load_prepared_rows, score_answer


class RecordGreedy(LogitsProcessor):
    """Read the native next-token logits without changing any of them."""

    def __init__(self, label):
        self.label = label
        self.steps = []
        self.started = time.perf_counter()

    def __call__(self, input_ids, scores):
        # Generation casts logits to FP32 before processors. This changes
        # storage, not the already-computed BF16 forward values.
        logits = scores[0].detach().float()
        values, ids = logits.topk(8)
        # Match Transformers' two-dimensional argmax call exactly, including
        # the tie handling of the backend reduction kernel.
        selected_id = scores.argmax(dim=-1)[0]
        log_normalizer = torch.logsumexp(logits, dim=-1)
        logps = values - log_normalizer
        self.steps.append({
            "step": len(self.steps),
            "selected_id": int(selected_id.item()),
            "top8_ids": ids.cpu().tolist(),
            "top8_logits": values.cpu().tolist(),
            "top8_logps": logps.cpu().tolist(),
            "top1_probability": float(logps[0].exp().item()),
            "top1_minus_top2_logit": float((values[0] - values[1]).item()),
        })
        if len(self.steps) % 64 == 0:
            print(self.label, "tokens:", len(self.steps),
                  "seconds:", round(time.perf_counter() - self.started, 1), flush=True)
        return scores


def save(report, destination):
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(destination)


def comparisons(report):
    pairs = [("bf16_cache", "bf16_no_cache"), ("fp32_cache", "fp32_no_cache"),
             ("bf16_cache", "fp32_cache"), ("bf16_no_cache", "fp32_no_cache")]
    result = []
    for row in report["questions"]:
        for left_name, right_name in pairs:
            if left_name not in row["runs"] or right_name not in row["runs"]:
                continue
            left, right = row["runs"][left_name], row["runs"][right_name]
            left_ids, right_ids = left["completion_token_ids"], right["completion_token_ids"]
            common = 0
            for left_id, right_id in zip(left_ids, right_ids):
                if left_id != right_id:
                    break
                common += 1
            item = dict(question_id=row["id"], left=left_name, right=right_name,
                        token_sequences_equal=left_ids == right_ids,
                        final_numeric_answers_equal=left["grading"]["predicted_answer"] == right["grading"]["predicted_answer"],
                        common_prefix_tokens=common)
            # At the first divergence, both paths still have exactly the same
            # input text. Later different-prefix probabilities are not paired.
            shared_steps = min(common + 1, len(left["steps"]), len(right["steps"]))
            item["same_prefix_max_top1_absolute_probability_difference"] = max(
                (abs(left["steps"][i]["top1_probability"] - right["steps"][i]["top1_probability"])
                 for i in range(common)), default=0.0)
            if left_ids != right_ids and common < shared_steps:
                item["first_divergence"] = dict(
                    completion_position_zero_based=common,
                    left=left["steps"][common], right=right["steps"][common])
            result.append(item)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT / "checkpoints/geora_base")
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "datasets/gsm8k")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--device", choices=["mps", "cuda", "cpu"], default="mps")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not visible. Run outside the restrictive sandbox.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is unavailable.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "answers.json"
    rows = load_prepared_rows(args.data_dir, "smoke")[:args.count]
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_dir, dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True)
    model.requires_grad_(False)
    model.eval()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    report = {
        "status": "running", "experiment": "original_model_full_greedy_gsm8k_four_paths",
        "model_repository": "Qwen/Qwen2.5-1.5B-Instruct",
        "model_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "torch_version": torch.__version__, "transformers_version": __import__("transformers").__version__,
        "device": args.device, "attention_implementation": "eager", "batch_size": 1,
        "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else args.device,
        "hip_version": torch.version.hip,
        "parameter_count": parameters, "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy argmax, no stochastic sampling, no logits penalties",
        "softmax_summary_precision": "FP32 (not a separate model)",
        "no_cache_definition": "native generate(use_cache=False); growing full prefix, not fixed_width training",
        "cache_definition": "native Transformers DynamicCache",
        "question_selection": "first ten prepared smoke rows, selected before outcomes; no final test",
        "data_source_commit": "3101c7d5072418e28b9008a6636bde82a006892c",
        "questions": [{**row, "runs": {}} for row in rows],
    }
    model.to(args.device)
    total_started = time.perf_counter()
    for precision, dtype in [("bf16", torch.bfloat16), ("fp32", torch.float32)]:
        if dtype == torch.float32:
            # Verify each individual parameter's exact promotion without
            # retaining another full model in the 24 GiB unified memory.
            preserved = 0
            with torch.no_grad():
                for parameter in model.parameters():
                    previous = parameter.detach().cpu()
                    parameter.data = parameter.data.float()
                    assert torch.equal(parameter.detach().cpu(), previous.float())
                    preserved += parameter.numel()
            report["exact_promoted_parameter_values"] = preserved
            assert preserved == parameters
        for use_cache in [True, False]:
            name = precision + ("_cache" if use_cache else "_no_cache")
            for index, row in enumerate(report["questions"]):
                chat = tokenizer.apply_chat_template(
                    [{"role": "user", "content": build_prompt(row["question"])}],
                    tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(chat, return_tensors="pt", add_special_tokens=False).to(args.device)
                recorder = RecordGreedy(f"{name} question={index + 1}/{len(rows)} {row['id']}")
                generation = GenerationConfig(
                    do_sample=False, use_cache=use_cache, max_new_tokens=args.max_new_tokens,
                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id, repetition_penalty=1.0,
                    no_repeat_ngram_size=0, return_dict_in_generate=True,
                    output_scores=False, output_logits=False, disable_compile=True)
                context = torch.autocast(device_type=args.device, dtype=dtype) if dtype == torch.bfloat16 else nullcontext()
                started = time.perf_counter()
                print("START:", recorder.label, flush=True)
                with torch.inference_mode(), context:
                    output = model.generate(**inputs, generation_config=generation,
                                            logits_processor=[recorder], logits_to_keep=1)
                if args.device == "mps":
                    torch.mps.synchronize()
                elif args.device == "cuda":
                    torch.cuda.synchronize()
                ids = output.sequences[0, inputs.input_ids.shape[1]:].cpu().tolist()
                # Transformers 5.19 may defer the stop check by one step and
                # then undo that step. The observer also saw that discarded
                # computation; keep only steps belonging to returned tokens.
                discarded_observer_steps = len(recorder.steps) - len(ids)
                recorder.steps = recorder.steps[:len(ids)]
                recorded_ids = [step["selected_id"] for step in recorder.steps]
                if ids != recorded_ids:
                    mismatch = next((i for i, (a, b) in enumerate(zip(ids, recorded_ids)) if a != b), None)
                    raise RuntimeError(f"Recorder disagrees with generation: lengths={len(ids)},{len(recorded_ids)}; "
                                       f"first={mismatch}; actual={ids[mismatch] if mismatch is not None else None}; "
                                       f"recorded={recorded_ids[mismatch] if mismatch is not None else None}")
                ended = bool(ids and ids[-1] == tokenizer.eos_token_id)
                text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                row["runs"][name] = dict(
                    precision=precision, use_cache=use_cache, completion=text,
                    prompt_token_count=inputs.input_ids.shape[1], completion_token_ids=ids,
                    completion_token_count=len(ids), ended_with_eos=ended,
                    elapsed_seconds=time.perf_counter() - started, grading=score_answer(text, row["target"]),
                    deferred_stop_discarded_observer_steps=discarded_observer_steps,
                    steps=recorder.steps,
                    allocated_memory_gib=torch.mps.current_allocated_memory() / 2**30 if args.device == "mps" else None,
                )
                report["comparisons"] = comparisons(report)
                save(report, path)
                print("DONE:", recorder.label, "tokens:", len(ids), "EOS:", ended,
                      "answer:", row["runs"][name]["grading"], flush=True)
                del output, inputs
                if not ended:
                    report["status"] = "incomplete_token_limit"
                    save(report, path)
                    raise RuntimeError("Completion reached token limit: retained text, do not call it complete.")
    report["elapsed_seconds"] = time.perf_counter() - total_started
    report["status"] = "completed"
    save(report, path)
    print("ALL COMPLETED:", path, flush=True)


if __name__ == "__main__":
    main()
