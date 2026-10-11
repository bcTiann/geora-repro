"""Tiny CPU check of prefix alignment, hooks and exact vector decomposition."""
import argparse
import hashlib
from pathlib import Path
import time

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from geora_check_utils import CheckReport
from local_error_propagation import diagnose_propagation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = CheckReport(args.output_dir, 'structural_check.json')
    started = time.perf_counter()
    report.data.update(stage='tiny_CPU_structural_error_propagation', device='cpu', model_count=1,
        optimizer_steps=0, scope='two sequential synthetic 2-layer models; not pretrained Qwen measurements')
    for dtype in (torch.float32, torch.bfloat16):
        torch.manual_seed(18)
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=37, hidden_size=32, intermediate_size=48,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            attention_dropout=0.0, eos_token_id=2, pad_token_id=0))
        model.config._attn_implementation = 'eager'
        model.requires_grad_(False).eval().to(dtype=dtype)
        case = dict(completion_ids=[[11,12,13,14,15]]*8, completion_lengths=[5]*8)
        diagnose_propagation(model, [[3,4,5]]*4 + [[6,7,8,9,10]]*4,
            case, 1, 3, 0, [2], dtype, 'cpu', report, str(dtype).split('.')[-1])
    report.data.update(elapsed_seconds=time.perf_counter()-started,
        source_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                       for name in ('check_local_error_propagation_cpu.py','local_error_propagation.py')})
    report.finish()
    print('Tiny prefix replacement and decomposition checks passed.')


if __name__ == '__main__':
    main()
