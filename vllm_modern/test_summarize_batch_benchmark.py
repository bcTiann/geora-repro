"""CPU-only artifact fixtures for the batch benchmark collector and its CLI.

Run from vllm_modern with: python -m unittest test_summarize_batch_benchmark
No GPU/runtime libraries, collected private artifacts, or remote paths needed.
"""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import summarize_batch_benchmark as summary

SOURCE = Path(__file__).resolve().parent
H = 'a' * 64


def put(path, value, lines=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in value) if lines else json.dumps(value))


def generation_schedule(micro, generation):
    return {'responses_per_rollout': generation, 'num_generations': 8,
        'questions_per_rollout': generation // 8, 'rollouts_per_update': 128 // generation,
        'responses_per_update': 128, 'questions_per_update': 16, 'training_micro_batch_size': micro,
        'gradient_accumulation_steps': 128 // micro, 'steps_per_generation': generation // micro,
        'request_seed_formula': 'base_seed + num_generations * (global_step * rollouts_per_update + rollout_in_update)',
        'batch_invariant_sampling_claim': False}


def case_fixture(root, kind, variant, micro, generation=128, passed=True):
    case = root / f'{kind}-{variant}-g{generation}-m{micro}'
    learner = case / 'learner'
    ident = {role: {'visible_device_count': 1, 'hostname': 'gpu-host', 'physical_pci_bdf': bdf}
             for role, bdf in [('server', '0000:01:00.0'), ('learner', '0000:02:00.0')]}
    for role, row in ident.items():
        put(case / f'{role}-device.json', row)
    put(case / 'distinct-gcds.json', {'passed': True, **ident})
    names = ['trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth']
    names += ['adapter_model.safetensors', 'adapter_config.json'] if variant == 'lora' else ['geora.safetensors', 'geora_config.json']
    protocol = {'mode': kind, 'variant': variant, 'schedule': summary.schedule(micro, generation),
        'max_optimizer_steps': 3, 'scientific_configuration_sha256': H, 'train_data_sha256': H,
        'base_revision': 'locked-base-revision', 'base_files_sha256': {'model.safetensors': H},
        'initialization_checkpoint': {'checkpoint': f'/frozen/{variant}/checkpoint-25', 'variant': variant,
            'global_step': 25, 'files_sha256': {name: H for name in names}},
        'source_sha256': {'train_native.py': H, 'native_sync.py': H}, 'uv_lock_sha256': H,
        'reward': summary.REWARD, 'precision': {'learner': 'bf16'}, 'primary_eval_backend': 'HF learner'}
    put(learner / 'run_protocol.json', protocol)
    if kind == 'capacity':
        put(learner / 'result.json', {'completed': True, 'mode': kind, 'variant': variant,
            'micro_batch_size': micro, 'prompt_length': 1024, 'completion_length': 4096,
            'padded_total_length': 5120, 'parameter_updates': 0, 'trainable_weights_unchanged': True,
            'passed': passed, 'optimizer_moment_bytes_reserved': 147718144 if passed else 0,
            'error': None if passed else 'OutOfMemoryError(cpu fixture)', 'peak_allocated_gib': 14.,
            'peak_reserved_gib': 16., 'seconds': 2.})
        return case
    put(learner / 'invocation.json', {'max_steps': 3, 'resume_checkpoint': None,
        'initialization_checkpoint': protocol['initialization_checkpoint']['checkpoint'],
        'mode': kind, 'variant': variant, 'micro_batch_size': micro, 'generation_batch_size': generation})
    chosen = [f'q{index}' for index in range(48)]
    put(learner / 'benchmark_questions.json', {'ids': chosen})
    updates = [{'optimizer_step': step, 'update_wall_seconds_including_sync_and_diagnostics': 30. if step == 1 else 10.,
        'post_clip_gradient_norm': .5, 'peak_allocated_gib': 20., 'peak_reserved_gib': 21.} for step in (1, 2, 3)]
    put(learner / 'updates.jsonl', updates, lines=True)
    gs = generation_schedule(micro, generation)
    native_hashes = {name: H for name in summary.canonical_names()}
    resident_hashes = {name: H for name in summary.resident_names()}
    pubs = [{'policy_version': version, 'canonical_tensor_count': 339, 'canonical_tensor_sha256': native_hashes,
        'resident_verification': {'passed': True, 'canonical_tensor_count': 339,
            'resident_parameter_count': 198, 'expected_resident_checksums': resident_hashes},
        'resident_checksums': resident_hashes, 'total_seconds': 1., 'phase_seconds': {'transfer': .5},
        'generation_schedule': gs} for version in range(4)]
    for publication in pubs:
        version = publication['policy_version']
        publication['fixed_token_comparison'] = ({'policy_version': version, 'structural_checks_passed': True,
            'numerical_accepted': None, 'acceptance_policy': None, 'tokens': 2, 'sequences': 1,
            'fixed_rows_sha256': H, 'comparisons': {name: {'tokens': 2, 'max_abs_logprob_delta': .1}
                for name in ['learner_minus_hf_merged', 'hf_merged_minus_vllm', 'learner_minus_vllm']}}
            if version in (0, 3) else {'status': 'not_scheduled_at_this_version'})
    put(learner / 'sync/sync.jsonl', pubs, lines=True)
    probabilities, rollouts, rewards, performance = [], [], [], []
    rpu = 128 // generation
    for index in range(3 * rpu):
        version = index // rpu
        group_begin = index * generation // 8
        group_ids = chosen[group_begin:group_begin + generation // 8]
        ids = [question for question in group_ids for _ in range(8)]
        probabilities.append({'optimizer_step': version, 'responses': generation, 'question_ids': ids,
            'stock_token_truncate_verified': True, 'old_learner_logprobs': [[-.875, -1.125]] * generation,
            'behavior_logprobs': [[-1., -2.]] * generation, 'active_tokens': generation * 2,
            'padded_completion_positions': generation * 3, 'padding_fraction': 1 / 3,
            'mean_abs_log_ratio': .5, 'max_abs_log_ratio': .875, 'fraction_ratio_above_2': .5,
            'nonzero_advantage_responses': generation})
        rollouts.append({'policy_version': version, 'responses': generation, 'all_behavior_logprobs_finite': True,
            'completion_ids': [[1, 2]] * generation, 'prompt_ids': [[3]] * generation,
            'behavior_logprobs': [[-1., -2.]] * generation, 'tokens': generation * 2, 'seconds': 1.5,
            'request_seed': 42 + 8 * index, 'rollout_in_update': index % rpu, 'generation_schedule': gs})
        rewards.append({'reward_contract': summary.REWARD, 'responses': generation, 'question_ids': ids,
            'global_step': version, 'status_counts': {'correct': generation // 2, 'incorrect': generation // 2,
                'missing_or_incomplete_boxed': 0, 'invalid_prediction': 0}, 'reward_mean': .5, 'scoring_seconds': .01,
            'grouping': {'status': 'validated_consecutive_groups', 'group_count': generation // 8,
                'zero_std_fraction': 0., 'all_zero_fraction': 0., 'all_one_fraction': 0.,
                'groups': [{'question_id': identifier, 'responses': 8, 'reward_mean': .5, 'all_zero': False,
                    'all_one': False, 'zero_std': False} for identifier in group_ids]}})
        for name, seconds in [('vllm_rollout', 1.), ('generate_and_score', 1.2), ('learner_old_logprob', .2)]:
            performance.append({'stage': name, 'seconds': seconds, 'nested_in': 'training_microstep', 'optimizer_step': version})
    for version in range(3):
        for _ in range(128 // micro):
            performance.append({'stage': 'training_microstep', 'seconds': .1, 'nested_in': 'training', 'optimizer_step': version})
    for name, value in [('training_probabilities.jsonl', probabilities), ('sync/rollouts.jsonl', rollouts),
                        ('reward_audit.jsonl', rewards), ('performance.jsonl', performance)]:
        put(learner / name, value, lines=True)
    put(learner / 'result.json', {'completed': True, 'mode': kind, 'variant': variant,
        'schedule': protocol['schedule'], 'global_step': 3, 'new_optimizer_updates': 3, 'responses': 384,
        'frozen_policy_unchanged': True, 'checkpoint_saved': False, 'resume_audit': None,
        'changed_adapter_tensors': 392, 'all_measured_advantages_zero': False,
        'parameters': {'target_layer_count': 196, 'trainable_tensors': 392}, 'updates': updates,
        'sync_summary': {'passed': True, 'all_resident_weights_verified': True, 'sync_versions': [0, 1, 2, 3],
            'training_responses': 384, 'training_rollouts': 3 * rpu, 'generation_schedule': gs},
        'actual_completion_tokens': 768, 'process_seconds': 100.})
    return case


def fixture(root, cap4=None, cap8=None):
    put(root / 'optimization_plan.json', json.loads((SOURCE / 'optimization_plan.json').read_text()))
    cap4 = cap4 or {method: True for method in summary.METHODS}
    cap8 = cap8 or {method: True for method in summary.METHODS}
    baseline = 2 if not all(cap4.values()) else 4
    for variant in summary.METHODS:
        case_fixture(root, 'capacity', variant, 4, passed=cap4[variant])
        if cap4[variant]:
            case_fixture(root, 'capacity', variant, 8, passed=cap8[variant])
        if baseline == 2:
            case_fixture(root, 'capacity', variant, 2)
        for generation in (32, 64, 128):
            case_fixture(root, 'benchmark', variant, baseline, generation)
        if baseline == 4 and all(cap8.values()):
            case_fixture(root, 'benchmark', variant, 8)
    (root / 'all-cases.done').touch()


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='geora-summary-test-')
        self.root = Path(self.tmp.name)
        fixture(self.root)
    def tearDown(self):
        self.tmp.cleanup()
    def test_complete_eight_benchmarks(self):
        value = summary.collect(self.root)
        self.assertEqual(value['status'], 'verified', value['protocol_errors'])
        bench = value['cases']['benchmark-lora-g32-m4']
        self.assertEqual(bench['actual_tokens'], 768)
        self.assertEqual(bench['probability']['cap2_trigger_fraction'], .5)
        self.assertEqual(bench['gpu']['learner']['metrics']['power_watts']['mean'], None)
        self.assertEqual(bench['warm_observations'], 2)
    def test_missing_case_is_incomplete(self):
        shutil.rmtree(self.root / 'benchmark-lora-g128-m8')
        self.assertEqual(summary.collect(self.root)['status'], 'incomplete')
    def test_missing_gate_keeps_present_benchmark_incomplete(self):
        shutil.rmtree(self.root / 'capacity-lora-g128-m4')
        self.assertEqual(summary.collect(self.root)['status'], 'incomplete')
    def test_source_hash_drift_invalid(self):
        path = self.root / 'benchmark-lora-g64-m4/learner/run_protocol.json'
        value = summary.read(path); value['source_sha256']['native_sync.py'] = 'b' * 64; put(path, value)
        self.assertEqual(summary.collect(self.root)['status'], 'invalid')
    def test_resident_wrong_even_with_passed_flag(self):
        path = self.root / 'benchmark-lora-g64-m4/learner/sync/sync.jsonl'
        value = summary.rows(path); value[1]['resident_checksums'][next(iter(value[1]['resident_checksums']))] = 'b' * 64
        put(path, value, lines=True)
        self.assertEqual(summary.collect(self.root)['status'], 'invalid')
    def test_inventory_drift_invalid(self):
        path = self.root / 'benchmark-lora-g64-m4/learner/run_protocol.json'
        value = summary.read(path); value['initialization_checkpoint']['files_sha256']['optimizer.pt'] = 'b' * 64; put(path, value)
        self.assertEqual(summary.collect(self.root)['status'], 'invalid')
    def test_wrong_microstep_count_invalid(self):
        path = self.root / 'benchmark-lora-g64-m4/learner/performance.jsonl'
        value = summary.rows(path); value.pop(); put(path, value, lines=True)
        self.assertEqual(summary.collect(self.root)['status'], 'invalid')
    def test_real_zero_telemetry_and_null_separate(self):
        case = self.root / 'benchmark-lora-g32-m4'
        put(case / 'learner-gpu.jsonl', [{'event': 'sample', 'epoch_time': 100., 'physical_pci_bdf': '0000:02:00.0',
            'status': 'partial', 'stage': 'training', 'metrics': {'gfx_utilization_percent': 0., 'power_watts': None},
            'power_scope': 'socket', 'power_source_field': None, 'errors': {'power': 'unsupported'}}], lines=True)
        value = summary.collect(self.root)
        self.assertEqual(value['status'], 'verified', value['protocol_errors'])
        metrics = value['cases'][case.name]['gpu']['learner']['metrics']
        self.assertEqual(metrics['gfx_utilization_percent']['mean'], 0.)
        self.assertIsNone(metrics['power_watts']['mean'])
        self.assertFalse(value['cases'][case.name]['gpu']['learner']['power_readings_informative'])
    def test_driver_zero_power_is_preserved_but_uninformative(self):
        case = self.root / 'benchmark-lora-g32-m4'
        samples = [{'event': 'sample', 'epoch_time': 100. + index,
            'physical_pci_bdf': '0000:02:00.0', 'status': 'ok', 'stage': 'training',
            'metrics': {'gfx_utilization_percent': activity, 'power_watts': 0.},
            'raw': {'power_info': {'average_socket_power': 0., 'power_limit': 0.}},
            'power_source_field': 'average_socket_power', 'power_scope': 'AMDSMI socket power', 'errors': {}}
            for index, activity in enumerate((0., 100., 25.))]
        put(case / 'learner-gpu.jsonl', samples, lines=True)
        value = summary.collect(self.root)
        self.assertEqual(value['status'], 'verified', value['protocol_errors'])
        telemetry = value['cases'][case.name]['gpu']['learner']
        self.assertFalse(telemetry['power_readings_informative'])
        self.assertEqual(telemetry['metrics']['power_watts']['count'], 3)
        self.assertEqual(telemetry['metrics']['power_watts']['mean'], 0.)
        self.assertEqual(telemetry['metrics']['power_watts']['availability_fraction'], 1.)
        self.assertEqual(telemetry['raw_power_limit_readings']['mean'], 0.)
        self.assertIn('driver-zero', telemetry['power_readings_warning'])
        self.assertIn('power_limit is also zero', telemetry['power_readings_warning'])
        self.assertIn('GFX activity', telemetry['power_readings_warning'])
        self.assertEqual(summary.rows(case / 'learner-gpu.jsonl'), samples)
    def test_positive_socket_power_is_informative_and_not_summed(self):
        path = self.root / 'one-role-gpu.jsonl'
        put(path, [{'event': 'sample', 'epoch_time': 100., 'physical_pci_bdf': '0000:02:00.0',
            'status': 'ok', 'metrics': {'power_watts': 125., 'gfx_utilization_percent': 100.},
            'power_source_field': 'average_socket_power', 'power_scope': 'AMDSMI socket power'}], lines=True)
        value = summary.telemetry(path, '0000:02:00.0')
        self.assertTrue(value['power_readings_informative'])
        self.assertIsNone(value['power_readings_warning'])
        self.assertEqual(value['power_scope'], ['AMDSMI socket power'])
    def test_missing_power_is_null_and_uninformative(self):
        value = summary.telemetry(self.root / 'missing-gpu.jsonl', '0000:02:00.0')
        self.assertFalse(value['power_readings_informative'])
        self.assertIsNone(value['metrics']['power_watts']['mean'])
        self.assertIn('missing', value['power_readings_warning'])
    def test_expected_tensor_and_schedule_schema(self):
        self.assertEqual(len(summary.canonical_names()), 339)
        self.assertEqual(len(summary.resident_names()), 198)
        self.assertIn('lm_head.weight', summary.canonical_names())
        self.assertIn('dp0:pp0:pcp0:tp0:model.layers.27.self_attn.qkv_proj.bias', summary.resident_names())
        self.assertEqual(summary.schedule(4, 32), {'micro_batch_size': 4, 'generation_batch_size': 32,
            'answers_per_update': 128, 'num_generations': 8, 'learner_world_size': 1,
            'gradient_accumulation_steps': 32, 'steps_per_generation': 8,
            'rollouts_per_update': 4, 'questions_per_update': 16})
    def test_publication_missing_is_invalid(self):
        path = self.root / 'benchmark-lora-g32-m4/learner/sync/sync.jsonl'
        value = summary.rows(path); value.pop(); put(path, value, lines=True)
        self.assertEqual(summary.collect(self.root)['status'], 'invalid')
    def test_raw_probability_statistic_tamper_is_invalid(self):
        path = self.root / 'benchmark-lora-g32-m4/learner/training_probabilities.jsonl'
        value = summary.rows(path); value[0]['mean_abs_log_ratio'] = 0.; put(path, value, lines=True)
        self.assertEqual(summary.collect(self.root)['status'], 'invalid')
    def test_question_order_can_differ_without_exposure_change(self):
        learner = self.root / 'benchmark-lora-g32-m4/learner'
        probability = summary.rows(learner / 'training_probabilities.jsonl')
        reward = summary.rows(learner / 'reward_audit.jsonl')
        for probability_row, reward_row in zip(probability, reward, strict=True):
            ids = [f'q{47 - int(identifier[1:])}' for identifier in probability_row['question_ids']]
            probability_row['question_ids'] = reward_row['question_ids'] = ids
            for index, group in enumerate(reward_row['grouping']['groups']):
                group['question_id'] = ids[index * 8]
        put(learner / 'training_probabilities.jsonl', probability, lines=True)
        put(learner / 'reward_audit.jsonl', reward, lines=True)
        value = summary.collect(self.root)
        self.assertEqual(value['status'], 'verified', value['protocol_errors'])
        first = value['cases']['benchmark-lora-g32-m4']
        second = value['cases']['benchmark-lora-g64-m4']
        self.assertEqual(first['actual_exposure_sha256'], second['actual_exposure_sha256'])
        self.assertNotEqual(first['actual_question_order_sha256'], second['actual_question_order_sha256'])
    def test_truncated_json_invalid(self):
        (self.root / 'benchmark-lora-g32-m4/learner/reward_audit.jsonl').write_text('{')
        self.assertEqual(summary.collect(self.root)['status'], 'invalid')
    def test_changed_plan_invalid(self):
        path = self.root / 'optimization_plan.json'
        value = summary.read(path); value['capacity']['completion_tokens'] = 100; put(path, value)
        with self.assertRaises(summary.InvalidEvidence):
            summary.collect(self.root)
    def test_oom_gates_and_fallback(self):
        shutil.rmtree(self.root); self.root.mkdir()
        fixture(self.root, cap8={'lora': False, 'geora_mixed': True})
        value = summary.collect(self.root)
        self.assertEqual(value['status'], 'verified', value['protocol_errors'])
        self.assertEqual(len(value['skipped_cases']), 2)
        shutil.rmtree(self.root); self.root.mkdir()
        fixture(self.root, cap4={'lora': False, 'geora_mixed': True})
        value = summary.collect(self.root)
        self.assertEqual(value['status'], 'verified', value['protocol_errors'])
        self.assertEqual(value['baseline_micro_batch_size'], 2)
    def test_fallback_oom_cannot_pass(self):
        shutil.rmtree(self.root); self.root.mkdir()
        fixture(self.root, cap4={'lora': False, 'geora_mixed': True})
        path = self.root / 'capacity-lora-g128-m2/learner/result.json'
        value = summary.read(path); value['passed'] = False; value['error'] = 'OutOfMemoryError'; put(path, value)
        self.assertEqual(summary.collect(self.root)['status'], 'invalid')
    def test_cli_exit_codes_and_default_json(self):
        command = [sys.executable, str(SOURCE / 'summarize_batch_benchmark.py'), '--root', str(self.root)]
        source_config = (SOURCE / 'formal_config.json').read_bytes()
        process = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
        self.assertTrue(summary.read(self.root / 'batch_summary.json')['passed'])
        self.assertEqual(json.loads(process.stdout)['status'], 'verified')
        (self.root / 'all-cases.done').unlink()
        self.assertEqual(subprocess.run(command, capture_output=True).returncode, 1)
        path = self.root / 'benchmark-lora-g32-m4/learner/run_protocol.json'
        value = summary.read(path); value['reward'] = 'changed'; put(path, value)
        self.assertEqual(subprocess.run(command, capture_output=True).returncode, 2)
        self.assertEqual((SOURCE / 'formal_config.json').read_bytes(), source_config)

if __name__ == '__main__':
    unittest.main(verbosity=2)
