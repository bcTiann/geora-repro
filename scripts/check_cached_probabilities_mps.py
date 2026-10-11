"""Compare Metal cached/no-cache probabilities for previously sampled tokens.

Replay the actual answers from the AMD source report. No new answer is sampled,
no parameters are updated, and Metal results do not replace AMD measurements.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoTokenizer, GenerationConfig, LogitsProcessorList

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
from cache_layer_probe import SavedTokens, saved_batch
from check_cached_rollout_gpu import diagnostic_prompt_token_ids
from geora_check_utils import CheckReport, forward_context, fresh_fp32_model, move_with_precision, pinned_configuration
from grpo_rollout import _decoder_hidden, _one_position_logits, require_zero_dropout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError('A working Metal device is required')
    torch.set_num_threads(2)
    report = CheckReport(args.output_dir, 'mps_probability_examples.json')
    started = time.perf_counter()
    config, checkpoint = pinned_configuration()
    source_path = PROJECT / 'reports/cached_logits/50659217.json'
    source = json.loads(source_path.read_text())
    case = source['cases'][0]
    report.data.update(
        stage='metal_replay_of_actual_AMD_sampled_answers', device='mps',
        torch_version=torch.__version__, transformers_version=transformers.__version__,
        model_revision=config['model_revision'], model_count=1, optimizer_steps=0,
        forward_dtype='BF16; native internal precision retained', analysis_dtype='CPU FP64',
        source_report=str(source_path), source_report_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        source_sampling_device='AMD MI250X', local_new_sampling=False,
    )
    try:
        report.require('source_is_frozen_original_model',
                       source['status'] == 'passed' and source['model_count'] == 1
                       and source['optimizer_steps'] == 0 and source['model_revision'] == config['model_revision'])
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        prompts = diagnostic_prompt_token_ids(tokenizer)
        ids, mask, width = saved_batch(prompts, case['completion_ids'], case['completion_lengths'], tokenizer.pad_token_id, 'mps')
        report.require('source_prompt_lengths_and_width_preserved',
                       list(map(len, prompts)) == case['prompt_lengths'] and ids.shape[1] == case['fixed_width'])
        print('Loading one frozen original model on Metal...', flush=True)
        model = fresh_fp32_model(checkpoint, disable_mmap=True)
        move_with_precision(model, 'mps', torch.bfloat16)
        require_zero_dropout(model)
        versions = {name: parameter._version for name, parameter in model.named_parameters()}
        raw_cached = []

        def capture_head(module, inputs, output):
            raw_cached.append(output[:, -1].detach().float().cpu().clone())

        generation_config = GenerationConfig(
            do_sample=False, use_cache=True, cache_implementation='dynamic',
            max_new_tokens=case['generated_steps'], num_beams=1, num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=[tokenizer.eos_token_id],
            bos_token_id=model.config.bos_token_id, return_dict_in_generate=True,
            output_scores=False, output_logits=False, disable_compile=True,
        )
        handle = model.lm_head.register_forward_hook(capture_head)
        print('Replaying saved tokens through native DynamicCache...', flush=True)
        try:
            with torch.no_grad(), forward_context('mps', torch.bfloat16):
                generated = model.generate(
                    input_ids=ids[:, :width], attention_mask=mask[:, :width],
                    generation_config=generation_config,
                    logits_processor=LogitsProcessorList([SavedTokens(width, ids[:, width:])]),
                )
        finally:
            handle.remove()
        torch.mps.synchronize()
        report.require('exact_saved_tokens_replayed',
                       torch.equal(generated.sequences[:, width:].cpu(), ids[:, width:].cpu()))
        report.require('one_raw_vocab_vector_per_prediction',
                       len(raw_cached) == case['generated_steps']
                       and all(z.shape == (8, model.config.vocab_size) for z in raw_cached))
        print('Computing the same saved sequence without cache...', flush=True)
        raw_fixed = []
        with torch.no_grad(), forward_context('mps', torch.bfloat16):
            hidden = _decoder_hidden(model, ids, mask)
            for step in range(case['generated_steps']):
                raw_fixed.append(_one_position_logits(model, hidden, width + step - 1, 1.0).detach().cpu().clone())
            del hidden
        torch.mps.synchronize()
        report.require('both_raw_logits_finite', all(torch.isfinite(z).all().item() for z in raw_cached + raw_fixed))
        questions = (
            'What is 7 + 5? Explain briefly.',
            'A box has 3 red balls and 5 blue balls. How many balls are there in total?',
        )
        examples = []
        for step, (cached, fixed) in enumerate(zip(raw_cached, raw_fixed)):
            # MPS does not run this FP64 analysis; move to CPU before conversion.
            cached_logp = cached.double().log_softmax(-1)
            fixed_logp = fixed.double().log_softmax(-1)
            for row, length in enumerate(case['completion_lengths']):
                if step >= length:
                    continue
                token = case['completion_ids'][row][step]
                pc = cached_logp[row, token].exp().item()
                pf = fixed_logp[row, token].exp().item()
                examples.append(dict(
                    row=row, completion_step=step, question=questions[row // 4], token_id=token,
                    answer_prefix=tokenizer.decode(case['completion_ids'][row][:step], skip_special_tokens=False),
                    sampled_token_text=tokenizer.decode([token], skip_special_tokens=False),
                    cached_probability=pc, no_cache_probability=pf,
                    probability_difference=pf-pc, relative_probability_difference=pf/pc-1,
                    cached_logit=cached[row,token].item(), no_cache_logit=fixed[row,token].item(),
                ))
        report.require('all_103_actual_sampled_tokens_compared', len(examples) == case['valid_token_count'] == 103)
        report.require('parameters_and_gradients_unchanged',
                       all(parameter._version == versions[name] and parameter.grad is None
                           for name, parameter in model.named_parameters()))
        worst = max(examples, key=lambda item: abs(item['relative_probability_difference']))
        source_worst = case['worst_probability_position']
        source_row = source_worst['row']
        source_step = source_worst['completion_step_zero_based']
        original_worst = next(item for item in examples if item['row'] == source_row and item['completion_step'] == source_step)
        report.require('original_worst_token_preserved', original_worst['token_id'] == source_worst['sampled_token_id'])
        snapshot = args.output_dir / 'original_AMD_worst_position_Metal_logits.safetensors'
        save_file({'cached_logits': raw_cached[source_step][source_row].contiguous(), 'no_cache_logits': raw_fixed[source_step][source_row].contiguous()}, str(snapshot))
        report.data.update(
            sampled_token_probabilities=examples, worst_metal_probability_position=worst,
            original_AMD_worst_position_on_metal=original_worst,
            actual_completions=[tokenizer.decode(tokens[:length], skip_special_tokens=True)
                                for tokens,length in zip(case['completion_ids'],case['completion_lengths'])],
            elapsed_seconds=time.perf_counter()-started,
            mps_current_allocated_memory_gib=torch.mps.current_allocated_memory()/2**30,
            mps_driver_allocated_memory_gib=torch.mps.driver_allocated_memory()/2**30,
            raw_snapshot=dict(path=str(snapshot), sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest()),
            source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        report.finish()
        print('Metal worst relative probability difference:', worst['relative_probability_difference'], flush=True)
        print('Original AMD worst position on Metal:', original_worst, flush=True)
    except Exception as error:
        report.data.update(status='failed', error=str(error), elapsed_seconds=time.perf_counter()-started)
        report.write()
        raise


if __name__ == '__main__':
    main()
