"""AMD same-weight BF16/FP32 cache comparison, all saved effective tokens."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import torch
import transformers
from safetensors.torch import load_file
from transformers import AutoTokenizer

from cached_precision_comparison import compare_saved_precision
from check_cached_rollout_gpu import diagnostic_prompt_token_ids
from geora_check_utils import CheckReport, fresh_fp32_model, move_with_precision, pinned_configuration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--source-report', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.version.hip is None or not os.getenv('SLURM_JOB_ID'):
        raise RuntimeError('Use one allocated Setonix ROCm GPU')
    torch.set_num_threads(8)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    report = CheckReport(args.output_dir, 'cached_precision_gpu.json')
    configuration, original = pinned_configuration()
    source = json.loads(args.source_report.read_text())
    case = source['cases'][0]
    report.data.update(stage='AMD_same_weights_BF16_FP32_all_saved_tokens', model_count=1, optimizer_steps=0,
        new_sampling=False, cases=[], source_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        model_repository=configuration['model_repository'], model_revision=configuration['model_revision'],
        torch_version=torch.__version__, transformers_version=transformers.__version__, hip_version=torch.version.hip,
        gpu_name=torch.cuda.get_device_name(0), slurm_job_id=os.environ['SLURM_JOB_ID'],
        source_report=str(args.source_report), source_report_sha256=hashlib.sha256(args.source_report.read_bytes()).hexdigest())
    try:
        staging = json.loads((args.checkpoint_dir/'staging_record.json').read_text())
        report.require('CPU_validated_pinned_checkpoint_copy', staging['status']=='validated'
            and staging['model_revision']==configuration['model_revision']
            and Path(staging['source']).resolve()==original.resolve())
        report.require('source_is_frozen_same_revision_original_model', source['status']=='passed'
            and source['model_count']==1 and source['optimizer_steps']==0
            and source['model_revision']==configuration['model_revision'])
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, local_files_only=True)
        prompts = diagnostic_prompt_token_ids(tokenizer)
        model = fresh_fp32_model(args.checkpoint_dir, disable_mmap=True)
        move_with_precision(model, 'cuda', torch.bfloat16)
        bf16, raw = compare_saved_precision(model,prompts,case,tokenizer.pad_token_id,tokenizer.eos_token_id,
                                            'cuda',torch.bfloat16,report,'BF16')
        previous = case['worst_probability_position']
        old_path = Path(previous['snapshot']['path'])
        report.require('old_AMD_snapshot_hash',hashlib.sha256(old_path.read_bytes()).hexdigest()==previous['snapshot']['sha256'])
        old = load_file(old_path)
        key = f"row{previous['row']}.step{previous['completion_step_zero_based']}"
        report.require('BF16_full_vocab_logits_reproduce_original_AMD_worst',
            torch.equal(raw['cached.'+key],old['cached_logits'])
            and torch.equal(raw['fixed.'+key],old['no_cache_logits']))
        del raw,old
        print('Promote every existing BF16 parameter value to FP32 without reloading weights',flush=True)
        checked = 0
        exact = True
        for parameter in model.parameters():
            before = parameter.detach().cpu().clone()
            parameter.data = parameter.data.float()
            exact &= torch.equal(parameter.detach().cpu(),before.float())
            checked += parameter.numel()
            del before
        report.require('all_parameter_values_preserved_by_FP32_promotion',exact,parameter_values_checked=checked)
        fp32,_ = compare_saved_precision(model,prompts,case,tokenizer.pad_token_id,tokenizer.eos_token_id,
                                         'cuda',torch.float32,report,'same_weight_values_FP32')
        for record in (bf16,fp32):
            for item in record['sampled_token_probabilities']:
                row,step = item['row'],item['completion_step']
                item.update(question=('What is 7 + 5? Explain briefly.' if row<4 else
                    'A box has 3 red balls and 5 blue balls. How many balls are there in total?'),
                    answer_prefix=tokenizer.decode(case['completion_ids'][row][:step],skip_special_tokens=False),
                    next_token=tokenizer.decode([item['token_id']],skip_special_tokens=False))
            record['original_AMD_worst_position'] = next(x for x in record['sampled_token_probabilities']
                if x['row']==previous['row'] and x['completion_step']==previous['completion_step_zero_based'])
        report.require('same_103_token_identities_in_both_precisions',
            [(x['row'],x['completion_step'],x['token_id']) for x in bf16['sampled_token_probabilities']]
            == [(x['row'],x['completion_step'],x['token_id']) for x in fp32['sampled_token_probabilities']]
            and bf16['valid_token_count']==fp32['valid_token_count']==103)
        torch.cuda.synchronize()
        report.data.update(elapsed_seconds=time.perf_counter()-started,
            peak_allocated_memory_gib=torch.cuda.max_memory_allocated()/2**30,
            source_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                for name in ('check_cached_precision_gpu.py','cached_precision_comparison.py','cache_layer_probe.py','grpo_rollout.py')})
        report.finish()
    except Exception as error:
        report.data.update(status='failed',error=str(error),elapsed_seconds=time.perf_counter()-started)
        report.write()
        raise


if __name__=='__main__':
    main()
