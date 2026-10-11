"""Check saved-token/EOS replay before allocating an AMD GPU."""

import argparse
import hashlib
from pathlib import Path

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from cached_precision_comparison import compare_saved_precision
from geora_check_utils import CheckReport


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    report=CheckReport(args.output_dir,'cached_precision_cpu.json')
    report.data.update(stage='tiny_CPU_saved_token_precision_protocol',device='cpu',optimizer_steps=0)
    torch.manual_seed(18)
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=37,hidden_size=32,intermediate_size=48,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,attention_dropout=0.0,
        eos_token_id=2,pad_token_id=0))
    model.config._attn_implementation='eager'
    model.requires_grad_(False).eval().bfloat16()
    prompts=[[3,4,5]]*4+[[6,7,8,9,10]]*4
    tokens=[[2,0,0,0,0],[11,2,0,0,0]]+[[11,12,13,14,15]]*6
    case=dict(completion_ids=tokens,completion_lengths=[1,2]+[5]*6,
        prompt_lengths=list(map(len,prompts)),fixed_width=10,valid_token_count=33)
    compare_saved_precision(model,prompts,case,0,2,'cpu',torch.bfloat16,report,'BF16')
    model.float()
    compare_saved_precision(model,prompts,case,0,2,'cpu',torch.float32,report,'FP32')
    report.data['source_sha256']={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
        for name in ('cached_precision_comparison.py','check_cached_precision_cpu.py')}
    report.finish()


if __name__=='__main__':
    main()
