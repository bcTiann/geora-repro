"""Local Metal layer-by-layer comparison and controlled input alignment.

Use saved real sampled tokens, a frozen original model and no optimizer.
Run from the repository; output JSON and raw diagnostic tensors separately.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
import transformers
from safetensors.torch import load_file
from transformers import AutoTokenizer

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
from check_cached_rollout_gpu import diagnostic_prompt_token_ids
from geora_check_utils import CheckReport, fresh_fp32_model, move_with_precision, pinned_configuration
from local_error_propagation import diagnose_propagation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--skip-fp32-control', action='store_true')
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError('Use local Metal; this script never submits a Setonix job')
    torch.set_num_threads(2)
    report = CheckReport(args.output_dir, 'local_error_propagation.json')
    start = time.perf_counter()
    configuration, checkpoint = pinned_configuration()
    source_path = PROJECT / 'reports/cached_logits/50659217.json'
    source = json.loads(source_path.read_text())
    source_case = source['cases'][0]
    metal_path = PROJECT / 'reports/cache_probabilities_mps/20261011.json'
    metal = json.loads(metal_path.read_text())
    report.data.update(stage='local_metal_error_propagation', cases=[], model_count=1,
        optimizer_steps=0, local_new_sampling=False, setonix_allocation_seconds=0,
        model_repository=configuration['model_repository'], model_revision=configuration['model_revision'],
        torch_version=torch.__version__, transformers_version=transformers.__version__,
        device='mps', source_report=str(source_path),
        source_report_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        previous_metal_report=str(metal_path),
        precision='BF16 native forward; framework FP32 internals retained; CPU FP64 analysis',
        intervention_scope='selected row, all valid prefix states, one decoder layer input; fixed-width continuation')
    try:
        report.require('source_and_metal_revision_and_frozen_model_match',
            all(r['status'] == 'passed' and r['model_count'] == 1 and r['optimizer_steps'] == 0
                and r['model_revision'] == configuration['model_revision'] for r in (source, metal)))
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        prompts = diagnostic_prompt_token_ids(tokenizer)
        model = fresh_fp32_model(checkpoint, disable_mmap=True)
        move_with_precision(model, 'mps', torch.bfloat16)
        report.require('one_frozen_dense_native_model',
            not model.training and not model.model.has_sliding_layers
            and all(not p.requires_grad for p in model.parameters())
            and all(type(m).__name__ not in ('GeoRALinear', 'LoRALinear') for m in model.modules()))
        target_specs = [
            ('This', metal['original_AMD_worst_position_on_metal']),
            ('End', metal['worst_metal_probability_position']),
        ]
        for label, target in target_specs:
            expected = None
            if label == 'This':
                raw_path = Path(metal['raw_snapshot']['path'])
                if not raw_path.is_absolute():
                    raw_path = PROJECT / raw_path
                report.require('previous_metal_logits_snapshot_hash',
                    hashlib.sha256(raw_path.read_bytes()).hexdigest() == metal['raw_snapshot']['sha256'])
                expected = load_file(raw_path)
            record = diagnose_propagation(model, prompts, source_case, target['row'], target['completion_step'],
                tokenizer.pad_token_id, [tokenizer.eos_token_id], torch.bfloat16, 'mps', report,
                'BF16_' + label, expected)
            record.update(question=target['question'], answer_prefix=target['answer_prefix'],
                          next_token=target['sampled_token_text'])
            report.require(label + '_same_previous_metal_selected_probabilities',
                abs(record['endpoint']['cached']['selected_probability'] - target['cached_probability']) < 1e-12
                and abs(record['endpoint']['fixed']['selected_probability'] - target['no_cache_probability']) < 1e-12)
            report.write()
            gc.collect()
        if not args.skip_fp32_control:
            print('Promoting the existing BF16 weight values to FP32; no checkpoint reload...', flush=True)
            # One parameter at a time bounds additional CPU storage. Check EVERY
            # parameter value so the precision control cannot silently change W.
            checked_parameters = 0
            values_preserved = True
            for parameter in model.parameters():
                before = parameter.detach().cpu().clone()
                parameter.data = parameter.data.float()
                values_preserved &= torch.equal(parameter.detach().cpu(), before.float())
                checked_parameters += parameter.numel()
                del before
            report.require('all_parameters_preserved_by_exact_fp32_promotion', values_preserved,
                           parameter_values_checked=checked_parameters)
            for label, target in target_specs:
                diagnose_propagation(model, prompts, source_case, target['row'], target['completion_step'],
                    tokenizer.pad_token_id, [tokenizer.eos_token_id], torch.float32, 'mps', report,
                    'same_weights_FP32_' + label, intervene=False)
        torch.mps.synchronize()
        report.require('all_temporary_hooks_restored', all(not m._forward_hooks and not m._forward_pre_hooks
                                                         for m in model.modules()))
        report.data.update(elapsed_seconds=time.perf_counter()-start,
            source_sha256={name:hashlib.sha256((PROJECT/'scripts'/name).read_bytes()).hexdigest()
                           for name in ('check_local_error_propagation.py', 'local_error_propagation.py', 'cache_layer_probe.py')},
            mps_memory_at_completion_gib=torch.mps.current_allocated_memory()/2**30,
            mps_driver_memory_at_completion_gib=torch.mps.driver_allocated_memory()/2**30)
        report.finish()
    except Exception as error:
        report.data.update(status='failed', error=str(error), elapsed_seconds=time.perf_counter()-start)
        report.write()
        raise


if __name__ == '__main__':
    main()
