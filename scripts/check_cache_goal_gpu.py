"""One-load MI250X attention interventions and cached-scoring candidate check.

The adapter update is a synthetic mechanical check, not GSM8K training. Native
attention attribution is completed before installing the untrained adapter.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import transformers
from safetensors.torch import load_file
from transformers import AutoTokenizer

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cache_attention_intervention import inspect_interventions
from cached_rollout import sample_completions_cached
from cached_score import score_completion_logps_cached
from check_cached_score_cpu import actual_input_observer, LOGP_LIMIT, RATIO_LIMIT, SEQUENCE_LIMIT
from check_cached_rollout_gpu import diagnostic_prompt_token_ids
from geora_check_utils import CheckReport, fresh_fp32_model, move_with_precision, pinned_configuration, validate_manifest
from geora_layers import load_geora_state
from grpo_math import group_advantages, grpo_loss
from grpo_rollout import likelihood_alignment_metrics


def alignment_case(model,prompts,tokenizer,capacity,report,label,update=False):
    inputs=[]
    handle=model.register_forward_pre_hook(actual_input_observer(inputs),with_kwargs=True)
    started=time.perf_counter()
    try:
        rollout=sample_completions_cached(model,prompts,tokenizer.pad_token_id,
            [tokenizer.eos_token_id],capacity,'cuda',torch.bfloat16,
            generator=torch.Generator(device='cuda').manual_seed(20261011),deadline=time.monotonic()+180)
    finally:handle.remove()
    generation_seconds=time.perf_counter()-started
    observer=[]
    scored_start=time.perf_counter()
    current=score_completion_logps_cached(model,rollout,'cuda',torch.bfloat16,
        gradient_enabled=True,trace_callback=observer.append)
    old=score_completion_logps_cached(model,rollout,'cuda',torch.bfloat16,gradient_enabled=False)
    scoring_seconds=time.perf_counter()-scored_start
    metric=likelihood_alignment_metrics(rollout['behavior_logps'],current,rollout['completion_mask'])
    passed=(metric['finite'] and metric['max_absolute_delta_logp']<=LOGP_LIMIT
        and metric['max_absolute_ratio_minus_one']<=RATIO_LIMIT
        and max(map(abs,metric['sequence_delta_logp']))<=SEQUENCE_LIMIT)
    report.require(label+'_all_valid_token_alignment',passed,**metric)
    report.require(label+'_actual_forward_inputs_equal',inputs==observer)
    report.require(label+'_gradient_enabled_and_no_grad_equal',current.requires_grad
                   and torch.equal(current.detach(),old) and not old.requires_grad)
    record=dict(label=label,capacity=capacity,valid_tokens=rollout['completion_mask'].sum().item(),
        completion_ids=rollout['completion_ids'].cpu().tolist(),
        completion_lengths=rollout['completion_lengths'].cpu().tolist(),
        behavior_logps=rollout['behavior_logps'].cpu().tolist(),
        recomputed_logps=current.detach().cpu().tolist(),likelihood=metric,
        generation_seconds=generation_seconds,scoring_seconds=scoring_seconds,
        scope='synthetic prompt alignment; not task reward or score')
    if update:
        trainable=[p for p in model.parameters() if p.requires_grad]
        versions={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
        buffers={n:p._version for n,p in model.named_buffers()}
        before=[p.detach().clone() for p in trainable]
        rewards=torch.tensor([[1.,0.,0.,0.],[0.,1.,0.,1.]],device='cuda')
        advantage=group_advantages(rewards).flatten()
        # Old is a synthetic reference here; real training keeps an original model.
        terms=grpo_loss(current,old,old,advantage,rollout['completion_mask'])
        optimizer=torch.optim.AdamW(trainable,lr=1e-6,weight_decay=0)
        terms['loss'].backward()
        finite=all(p.grad is not None and p.grad.dtype==torch.float32
                   and torch.isfinite(p.grad).all().item() for p in trainable)
        norm=torch.nn.utils.clip_grad_norm_(trainable,1.0)
        report.require(label+'_full_adapter_cached_backward_finite',finite
                       and torch.isfinite(norm).item() and norm.item()>0,
                       trainable_tensor_count=len(trainable),gradient_norm=norm.item())
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        report.require(label+'_only_trainable_update',all(p._version==versions[n]
            for n,p in model.named_parameters() if not p.requires_grad)
            and all(p._version==buffers[n] for n,p in model.named_buffers())
            and any(not torch.equal(a,b) for a,b in zip(trainable,before)))
        report.data['synthetic_optimizer_steps']+=1
        record['synthetic_loss']=terms['loss'].item()
    report.data.setdefault('alignment_cases',[]).append(record)
    report.write()
    del current,old,rollout
    gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir',type=Path,required=True)
    parser.add_argument('--init-dir',type=Path,required=True)
    parser.add_argument('--source-report',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if not torch.cuda.is_available() or torch.version.hip is None or not os.getenv('SLURM_JOB_ID'):
        raise RuntimeError('Use one allocated Setonix ROCm GPU')
    torch.set_num_threads(8);torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter();report=CheckReport(args.output_dir,'cache_goal_gpu.json')
    config,original=pinned_configuration()
    report.data.update(stage='cache_attention_intervention_and_execution_alignment',cases=[],alignment_cases=[],
        real_grpo_optimizer_steps=0,synthetic_optimizer_steps=0,production_sampler_changed=False,
        slurm_job_id=os.environ['SLURM_JOB_ID'],source_git_commit=subprocess.check_output(
            ['git','rev-parse','HEAD'],text=True).strip(),
        model_repository=config['model_repository'],model_revision=config['model_revision'],
        torch_version=torch.__version__,transformers_version=transformers.__version__,
        hip_version=torch.version.hip,gpu_name=torch.cuda.get_device_name(0),
        gates=dict(logp=LOGP_LIMIT,ratio=RATIO_LIMIT,sequence=SEQUENCE_LIMIT))
    try:
        staging=json.loads((args.checkpoint_dir/'staging_record.json').read_text())
        report.require('CPU_validated_pinned_checkpoint',staging['status']=='validated'
            and staging['model_revision']==config['model_revision']
            and Path(staging['source']).resolve()==original.resolve())
        source=json.loads(args.source_report.read_text());case=source['cases'][0]
        report.require('original_source_report',source['status']=='passed'
            and source['model_revision']==config['model_revision'] and source['model_count']==1)
        worst=case['worst_probability_position'];path=Path(worst['snapshot']['path'])
        report.require('raw_logits_hash',hashlib.sha256(path.read_bytes()).hexdigest()==worst['snapshot']['sha256'])
        tokenizer=AutoTokenizer.from_pretrained(args.checkpoint_dir,local_files_only=True)
        prompts=diagnostic_prompt_token_ids(tokenizer)
        model=fresh_fp32_model(args.checkpoint_dir,disable_mmap=True)
        move_with_precision(model,'cuda',torch.bfloat16);model.eval()
        inspect_interventions(model,prompts,case,worst['row'],worst['completion_step_zero_based'],
            tokenizer.pad_token_id,[tokenizer.eos_token_id],torch.bfloat16,'cuda',report,'native_bf16',load_file(path))
        manifest=json.loads((args.init_dir/'manifest.json').read_text());validate_manifest(manifest,config)
        state=load_file(str(args.init_dir/'adapter.safetensors'))
        report.require('untrained_FP32_factors',all(torch.equal(state[t['name']+'.'+factor],
            state[t['name']+'.'+factor+'0']) for t in manifest['target_modules'] for factor in ('A','B')))
        # Restoration API requires the original model on CPU in FP32.
        # Source weights were BF16; this promotion retains their numeric values.
        model.to(device='cpu',dtype=torch.float32)
        load_geora_state(model,state,dict(manifest,forward_mode='difference'))
        del state
        move_with_precision(model,'cuda',torch.bfloat16);model.train()
        alignment_case(model,prompts,tokenizer,16,report,'geora_initial_16',update=True)
        alignment_case(model,prompts,tokenizer,64,report,'geora_updated_64')
        torch.cuda.synchronize()
        report.data.update(elapsed_seconds=time.perf_counter()-start,
            peak_allocated_memory_gib=torch.cuda.max_memory_allocated()/2**30)
        report.finish()
    except Exception as error:
        report.data.update(status='failed',error=str(error),elapsed_seconds=time.perf_counter()-start)
        report.write();raise


if __name__=='__main__':main()
