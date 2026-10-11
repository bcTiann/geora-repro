"""One-GPU layer localization from the saved native-model logits diagnostic."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import torch
import transformers
from safetensors.torch import load_file

from cache_layer_probe import inspect_case
from check_cached_rollout_gpu import diagnostic_prompt_token_ids
from geora_check_utils import CheckReport, fresh_fp32_model, move_with_precision, pinned_configuration
from transformers import AutoTokenizer


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
    start = time.perf_counter()
    report = CheckReport(args.output_dir, 'cache_layers_gpu.json')
    configuration, original = pinned_configuration()
    report.data.update(stage='native_cache_layer_localization', model_count=1, optimizer_steps=0,
                       cases=[], production_sampler_changed=False,
                       slurm_job_id=os.environ['SLURM_JOB_ID'],
                       source_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                       model_repository=configuration['model_repository'], model_revision=configuration['model_revision'],
                       torch_version=torch.__version__, transformers_version=transformers.__version__,
                       hip_version=torch.version.hip, gpu_name=torch.cuda.get_device_name(0),
                       source_report=str(args.source_report))
    try:
        staging = json.loads((args.checkpoint_dir/'staging_record.json').read_text())
        report.require('CPU_validated_pinned_checkpoint', staging['status']=='validated'
                       and staging['model_revision']==configuration['model_revision']
                       and Path(staging['source']).resolve()==original.resolve())
        source = json.loads(args.source_report.read_text())
        report.require('saved_original_model_forward_only_case', source['status']=='passed'
                       and source['model_count']==1 and source['optimizer_steps']==0
                       and source['model_revision']==configuration['model_revision'])
        case = source['cases'][0]
        worst = case['worst_probability_position']
        snapshot = load_file(worst['snapshot']['path'])
        import hashlib
        report.require('saved_logits_snapshot_hash', hashlib.sha256(
            Path(worst['snapshot']['path']).read_bytes()).hexdigest()==worst['snapshot']['sha256'])
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, local_files_only=True)
        prompts = diagnostic_prompt_token_ids(tokenizer)
        model = fresh_fp32_model(args.checkpoint_dir, disable_mmap=True)
        move_with_precision(model, 'cuda', torch.bfloat16)
        model.eval()
        report.require('dense_full_attention_Qwen', not model.model.has_sliding_layers)
        inspect_case(model, prompts, case, worst['row'], worst['completion_step_zero_based'],
                     tokenizer.pad_token_id, [tokenizer.eos_token_id], torch.bfloat16, 'cuda',
                     report, 'native_bf16', snapshot)
        # Bound the allocation: only run this control if the first case is quick.
        if time.perf_counter()-start < 45:
            # Exact dtype promotion preserves BF16 weight values. No reload from
            # an unrounded FP32 source; forward arithmetic is the changed factor.
            old = {name: p.detach().cpu().clone() for name,p in model.named_parameters()
                   if name=='model.layers.0.self_attn.q_proj.weight'}
            model.float()
            report.require('fp32_control_preserves_checked_q_weight_values', all(torch.equal(
                dict(model.named_parameters())[name].detach().cpu(), p.float()) for name,p in old.items()),
                           checked_tensor_count=len(old))
            inspect_case(model, prompts, case, worst['row'], worst['completion_step_zero_based'],
                         tokenizer.pad_token_id, [tokenizer.eos_token_id], torch.float32, 'cuda',
                         report, 'same_weight_values_fp32')
        else:
            report.data['fp32_control_skipped_reason']='BF16 case used at least 45s; preserve two-minute job budget'
        torch.cuda.synchronize()
        report.data.update(elapsed_seconds=time.perf_counter()-start,
                           peak_allocated_memory_gib=torch.cuda.max_memory_allocated()/2**30)
        report.finish()
    except Exception as error:
        report.data.update(status='failed', error=str(error), elapsed_seconds=time.perf_counter()-start)
        report.write()
        raise


if __name__=='__main__':
    main()
