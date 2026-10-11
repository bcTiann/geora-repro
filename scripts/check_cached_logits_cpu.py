"""Validate paired pre-softmax capture and token alignment without a GPU."""

import argparse
from pathlib import Path
import time

import torch
import transformers
from transformers import Qwen2Config, Qwen2ForCausalLM

from cached_logits_diagnostic import compare_native_logits
from geora_check_utils import CheckReport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = CheckReport(args.output_dir, "cached_logits_cpu.json")
    report.data.update(stage="tiny_native_paired_logits", cases=[],
                       torch_version=torch.__version__, transformers_version=transformers.__version__,
                       scope="CPU synthetic native models only; no GeoRA or optimizer updates")
    started = time.perf_counter()
    try:
        for dtype in (torch.float32, torch.bfloat16):
            torch.manual_seed(18)
            model = Qwen2ForCausalLM(Qwen2Config(
                vocab_size=37, hidden_size=32, intermediate_size=48,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                attention_dropout=0.0, eos_token_id=2, pad_token_id=0,
            ))
            model.config._attn_implementation = "eager"
            model.requires_grad_(False).eval().to(dtype=dtype)
            compare_native_logits(model, [[3, 4, 5]] * 4 + [[6, 7, 8, 9, 10]] * 4,
                                  0, [2, 3, 4, 5, 6], 8, "cpu", dtype,
                                  report, str(dtype).split(".")[-1], check_prefix=True)
            del model
        report.data["elapsed_seconds"] = time.perf_counter() - started
        report.finish()
    except Exception as error:
        report.data.update(status="failed", error=str(error))
        report.write()
        raise


if __name__ == "__main__":
    main()
