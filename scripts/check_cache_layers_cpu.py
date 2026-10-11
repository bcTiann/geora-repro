"""Validate saved-token replay, hooks and history alignment on tiny CPU Qwen."""

import argparse
from pathlib import Path
import time

import torch
import transformers
from safetensors.torch import load_file
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2 import modeling_qwen2

from cache_layer_probe import inspect_case
from cached_logits_diagnostic import compare_native_logits
from geora_check_utils import CheckReport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = CheckReport(args.output_dir, 'cache_layers_cpu.json')
    report.data.update(stage='tiny_cache_layer_replay', cases=[], optimizer_steps=0,
                       torch_version=torch.__version__, transformers_version=transformers.__version__)
    start = time.perf_counter()
    original_attention = modeling_qwen2.eager_attention_forward
    try:
        for dtype in (torch.float32, torch.bfloat16):
            torch.manual_seed(18)
            model = Qwen2ForCausalLM(Qwen2Config(
                vocab_size=37, hidden_size=32, intermediate_size=48,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                attention_dropout=0.0, eos_token_id=2, pad_token_id=0))
            model.config._attn_implementation = 'eager'
            model.requires_grad_(False).eval().to(dtype=dtype)
            prompts = [[3,4,5]] * 4 + [[6,7,8,9,10]] * 4
            source = CheckReport(args.output_dir / str(dtype).split('.')[-1], 'source.json')
            source.data['cases'] = []
            case = compare_native_logits(model, prompts, 0, [2,3,4,5,6], 8, 'cpu', dtype,
                                         source, 'source', check_prefix=False)
            worst = case['worst_probability_position']
            snapshot = load_file(worst['snapshot']['path'])
            label = str(dtype).split('.')[-1]
            inspect_case(model, prompts, case, worst['row'], worst['completion_step_zero_based'],
                         0, [2,3,4,5,6], dtype, 'cpu', report, label, snapshot)
            decoded_row = next(i for i,length in enumerate(case['completion_lengths']) if length >= 3)
            inspect_case(model, prompts, case, decoded_row, 2,
                         0, [2,3,4,5,6], dtype, 'cpu', report, label + '_decode')
            report.require(label + '_hooks_and_attention_wrapper_restored',
                           modeling_qwen2.eager_attention_forward is original_attention
                           and all(not m._forward_hooks and not m._forward_pre_hooks
                                   for m in model.modules()))
        report.data['elapsed_seconds'] = time.perf_counter() - start
        report.finish()
    except Exception as error:
        report.data.update(status='failed', error=str(error))
        report.write()
        raise


if __name__ == '__main__':
    main()
