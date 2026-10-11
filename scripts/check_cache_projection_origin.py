"""Bounded CPU or one-MI250X-GPU historical V origin check."""

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
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2 import modeling_qwen2

from cache_projection_origin import inspect_projection_origin
from check_cached_rollout_gpu import diagnostic_prompt_token_ids
from geora_check_utils import CheckReport, fresh_fp32_model, move_with_precision, pinned_configuration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--checkpoint-dir', type=Path)
    parser.add_argument('--source-report', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    arguments = parser.parse_args()
    started = time.perf_counter()
    report = CheckReport(arguments.output_dir, 'projection_origin.json')
    report.data.update(stage='historical_V_projection_before_cache_storage', model_count=1,
                       optimizer_steps=0, device=arguments.device,
                       torch_version=torch.__version__, transformers_version=transformers.__version__,
                       production_sampler_changed=False)
    attention_before = modeling_qwen2.eager_attention_forward
    try:
        if arguments.device == 'cuda':
            if not torch.cuda.is_available() or not torch.version.hip or not os.getenv('SLURM_JOB_ID'):
                raise RuntimeError('Use one allocated Setonix ROCm GPU')
            torch.set_num_threads(8)
            torch.cuda.reset_peak_memory_stats()
            configuration, original = pinned_configuration()
            staging = json.loads((arguments.checkpoint_dir / 'staging_record.json').read_text())
            report.require('CPU_validated_pinned_model_copy', staging['status'] == 'validated'
                           and staging['model_revision'] == configuration['model_revision']
                           and Path(staging['source']).resolve() == original.resolve())
            source = json.loads(arguments.source_report.read_text())
            report.require('original_frozen_model_source_case', source['status'] == 'passed'
                           and source['model_count'] == 1 and source['optimizer_steps'] == 0
                           and source['model_revision'] == configuration['model_revision'])
            case = source['cases'][0]
            worst = case['worst_probability_position']
            snapshot_path = Path(worst['snapshot']['path'])
            report.require('original_logits_snapshot_hash', hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
                           == worst['snapshot']['sha256'])
            expected = load_file(snapshot_path)
            tokenizer = AutoTokenizer.from_pretrained(arguments.checkpoint_dir, local_files_only=True)
            prompts = diagnostic_prompt_token_ids(tokenizer)
            model = fresh_fp32_model(arguments.checkpoint_dir, disable_mmap=True)
            move_with_precision(model, 'cuda', torch.bfloat16)
            row, step = worst['row'], worst['completion_step_zero_based']
            pad_id, eos_ids = tokenizer.pad_token_id, [tokenizer.eos_token_id]
            report.data.update(slurm_job_id=os.environ['SLURM_JOB_ID'], gpu_name=torch.cuda.get_device_name(0),
                               hip_version=torch.version.hip, model_revision=configuration['model_revision'],
                               source_report=str(arguments.source_report),
                               source_git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
        else:
            torch.set_num_threads(2)
            torch.manual_seed(18)
            model = Qwen2ForCausalLM(Qwen2Config(vocab_size=37, hidden_size=32, intermediate_size=48,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                attention_dropout=0.0, eos_token_id=2, pad_token_id=0))
            model.config._attn_implementation = 'eager'
            model.requires_grad_(False).eval().to(dtype=torch.bfloat16)
            prompts = [[3, 4, 5]] * 4 + [[6, 7, 8, 9, 10]] * 4
            case = dict(completion_ids=[[11, 12, 13, 14]] * 8, completion_lengths=[4] * 8)
            row, step, pad_id, eos_ids, expected = 1, 3, 0, [2], None
        inspect_projection_origin(model, prompts, case, row, step, pad_id, eos_ids,
                                  torch.bfloat16, arguments.device, report, expected)
        report.require('temporary_hooks_restored', modeling_qwen2.eager_attention_forward is attention_before
                       and all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules()))
        if arguments.device == 'cuda':
            torch.cuda.synchronize()
            report.data['peak_allocated_memory_gib'] = torch.cuda.max_memory_allocated() / 2**30
        report.data['elapsed_seconds'] = time.perf_counter() - started
        report.finish()
    except Exception as error:
        report.data.update(status='failed', error=str(error), elapsed_seconds=time.perf_counter() - started)
        report.write()
        raise


if __name__ == '__main__':
    main()
