"""CPU-only complete artifact fixtures and fail-closed collector/CLI checks.

Run in vllm_modern: python -m unittest test_collect_schedule_acceptance
Uses synthetic checkpoint evidence, never private reports, GPU work, or remote files.
"""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import collect_schedule_acceptance as c
from test_summarize_batch_benchmark import case_fixture, put

SOURCE = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def three_way(version, controls, shifted=False):
    rows = []
    for fixed in controls:
        count = len(fixed['completion_ids'])
        rows.append({key: fixed[key] for key in ('id', 'category', 'prompt_ids', 'completion_ids')})
        rows[-1].update({'includes_eos': True, 'learner_logprobs': [-1.] * count,
            'hf_merged_logprobs': [-1.1] * count, 'vllm_logprobs': [-1.2 - (.1 if shifted else 0.)] * count})
    comparisons = {}
    for name, left, right in [('learner_minus_hf_merged', 'learner_logprobs', 'hf_merged_logprobs'),
            ('hf_merged_minus_vllm', 'hf_merged_logprobs', 'vllm_logprobs'),
            ('learner_minus_vllm', 'learner_logprobs', 'vllm_logprobs')]:
        delta = [abs(a-b) for row in rows for a,b in zip(row[left],row[right])]
        comparisons[name] = {'tokens': len(delta), 'mean_abs_logprob_delta': sum(delta)/len(delta), 'max_abs_logprob_delta': max(delta)}
    return {'policy_version': version, 'structural_checks_passed': True, 'numerical_accepted': None,
        'acceptance_policy': None, 'fixed_rows_sha256': c.value_hash(controls), 'sequences': 5,
        'tokens': sum(len(row['completion_ids']) for row in rows), 'comparisons': comparisons, 'rows': rows}


def fixture(root):
    # Construct a local synthetic approval plan; no real execution report is an
    # input to these tests. Hashing files does not import the training runtime.
    source_paths = list(SOURCE.rglob('*.py')) + [SOURCE / name for name in
        ('formal_config.json', 'jobs/schedule_acceptance.sbatch', 'uv.lock', 'fixtures/probability_controls.json')]
    plan = {'source_commit': '0' * 40, 'micro_batch_size': 8, 'gradient_accumulation_steps': 16,
        'generation_batch_size': 128, 'answers_per_update': 128, 'num_generations': 8,
        'max_updates_per_method': 2, 'max_gcds': 2, 'max_allocation_seconds': 1800,
        'formal400_training': False, 'methods': ['lora', 'geora_mixed'],
        'phase1_capacity': {'prompt': 1024, 'completion': 4096, 'compute_entropy': True,
            'adam_moment_bytes_reserved': True, 'parameter_updates': 0},
        'source_files_sha256': {'vllm_modern/' + str(path.relative_to(SOURCE)): c.file_hash(path)
            for path in source_paths if '__pycache__' not in path.parts}}
    controls = c.read(SOURCE / 'fixtures/probability_controls.json')
    shutil.copyfile(SOURCE / 'formal_config.json', root / 'scientific-config.json')
    shutil.copyfile(SOURCE / 'jobs/schedule_acceptance.sbatch', root / 'job-source.sbatch')
    plan['source_files_sha256']['vllm_modern/formal_config.json'] = c.file_hash(root / 'scientific-config.json')
    plan['source_files_sha256']['vllm_modern/jobs/schedule_acceptance.sbatch'] = c.file_hash(root / 'job-source.sbatch')
    put(root / 'plan.json', plan)
    config = c.read(root / 'scientific-config.json')
    initial_reference = {}
    for variant in c.METHODS:
        source = case_fixture(root, 'benchmark', variant, 8, 128)
        base = source / 'learner'
        template = c.read(base / 'run_protocol.json')
        original = template['initialization_checkpoint']
        initial_reference[variant] = {key:value for key,value in original.items() if key != 'checkpoint'}
        contract = {'variant': variant, 'schedule': c.SCHEDULE, 'max_steps_for_both_phases': 2,
            'scientific_configuration_sha256': c.file_hash(root / 'scientific-config.json'),
            'versions': config['runtime_versions'], 'uv_lock_sha256': config['runtime_lock_sha256'],
            'execution_manifest_sha256': digest('exec-manifest'), 'original_policy_initialization': original,
            'train_data_sha256': config['data']['train_file_sha256'], 'fixed_tokens_sha256': c.file_hash(SOURCE / 'fixtures/probability_controls.json'),
            'base_files_sha256': config['model']['expected_base_files_sha256'],
            'source_sha256': {name.removeprefix('vllm_modern/'): value for name,value in plan['source_files_sha256'].items()
                if name.startswith('vllm_modern/') and name.endswith('.py')},
            'reward': c.REWARD, 'training_seed': 42, 'data_seed': 42}
        saved1 = None
        policy_states = {0: digest(variant + 'initial'), 1: digest(variant + 'first'), 2: digest(variant + 'second')}
        for phase in (1,2):
            phase_dir = root / variant / f'phase-{phase}'
            learner = phase_dir / 'learner'
            learner.mkdir(parents=True)
            devices = {role: c.read(source / f'{role}-device.json') for role in ('server','learner')}
            for role,value in devices.items(): put(phase_dir / f'{role}-device.json',value)
            put(phase_dir / 'separate-gcds-verified.json', {'passed': True, **devices})
            saved = {'checkpoint': f'/remote/{variant}/phase-{phase}/learner/training/checkpoint-{phase}',
                'variant': variant, 'global_step': phase, 'files_sha256': {name: digest(f'{variant}/{phase}/{name}') for name in original['files_sha256']}}
            if phase == 1: saved1 = saved
            put(learner / 'run_protocol.json', {'mode': 'schedule_acceptance', 'phase': phase,
                'immutable_contract': contract, 'resume_input': saved1 if phase == 2 else None,
                'server_sampling_rng_restored': False, 'bitwise_sampling_replay_guaranteed': False})
            put(learner / 'invocation.json', {'mode': 'schedule_acceptance', 'variant': variant, 'phase': phase,
                'micro_batch_size': 8, 'generation_batch_size': 128, 'max_steps': 2,
                'initialization_checkpoint': original['checkpoint'] if phase == 1 else None,
                'resume_checkpoint': saved1['checkpoint'] if phase == 2 else None})
            put(learner / 'environment.json', {'versions': config['runtime_versions']})
            put(learner / 'resolved_training_options.json', {'micro_batch_size': 8, 'gradient_accumulation_steps': 16,
                'generation_batch_size': 128, 'steps_per_generation': 16, 'bf16_autocast': False, 'fp16_autocast': False,
                'fused_lm_head': None, 'use_liger_loss': False, 'model_attention_implementation': 'sdpa'})
            optimizer = {'global_step': phase - 1, 'fresh_empty_state': phase == 1,
                'state_entries': 0 if phase == 1 else 392, 'state_sha256': digest(variant + ('empty' if phase == 1 else 'optimizer1'))}
            put(learner / 'optimizer_at_train_begin.json',optimizer)
            update = copy.deepcopy(c.rows(base / 'updates.jsonl')[0]); update['optimizer_step'] = phase; update['zero_gradient'] = False
            put(learner / 'updates.jsonl',[update],lines=True)
            capacity = {'completed': True, 'passed': True, 'mode': 'capacity', 'variant': variant, 'error': None,
                'compute_entropy': True, 'micro_batch_size': 8, 'prompt_length': 1024, 'completion_length': 4096,
                'padded_total_length': 5120, 'parameter_updates': 0, 'trainable_weights_unchanged': True,
                'optimizer_moment_bytes_reserved': 147718144, 'synthetic_workload': True,
                'peak_allocated_gib': 32., 'peak_reserved_gib': 40., 'seconds': 5.}
            if phase == 1: put(learner / 'capacity_result.json',capacity)
            probability = copy.deepcopy(c.rows(base / 'training_probabilities.jsonl')[phase - 1]); probability['optimizer_step'] = phase - 1
            rollout = copy.deepcopy(c.rows(base / 'sync/rollouts.jsonl')[phase - 1]); rollout['policy_version'] = phase - 1
            reward = copy.deepcopy(c.rows(base / 'reward_audit.jsonl')[phase - 1]); reward['global_step'] = phase - 1
            put(learner / 'training_probabilities.jsonl',[probability],lines=True)
            put(learner / 'sync/rollouts.jsonl',[rollout],lines=True)
            put(learner / 'reward_audit.jsonl',[reward],lines=True)
            performance = [{'stage': 'training_microstep', 'optimizer_step': phase - 1} for _ in range(16)]
            if phase == 1: performance += [{'stage': 'capacity_probe'}, {'stage': 'learner_grad_forward', 'optimizer_step': 0, 'nested_in': 'capacity_probe'}]
            put(learner / 'performance.jsonl',performance,lines=True)
            publications = copy.deepcopy(c.rows(base / 'sync/sync.jsonl')[phase - 1:phase + 1])
            for publication in publications:
                version = publication['policy_version']
                publication['canonical_tensor_sha256'] = {name: digest(f'{variant}/canonical/{version}') for name in c.canonical_names()}
                resident = {name: digest(f'{variant}/resident/{version}') for name in c.resident_names()}
                publication['resident_checksums'] = resident
                publication['resident_verification']['expected_resident_checksums'] = resident
                raw = three_way(version,controls,shifted=(phase == 2 and version == 1))
                put(learner / f'sync/three_way_v{version}.json',raw)
                publication['fixed_token_comparison'] = {key:value for key,value in raw.items() if key != 'rows'}
            put(learner / 'sync/sync.jsonl',publications,lines=True)
            audit = None
            tensor_count = 392 if variant == 'lora' else 672
            policy = {'bitwise_equal': True, 'tensor_count': tensor_count, 'state_sha256': policy_states[phase]}
            loaded_policy = {**policy, 'state_sha256': policy_states[phase - 1]}
            if phase == 2:
                checkpoint = saved1['checkpoint']
                records = [{'kind': kind, 'checkpoint': checkpoint, 'passed': True} for kind in ('policy','optimizer','scheduler','rng','training_state')]
                records[0].update(loaded_policy)
                records[1]['state_sha256'] = optimizer['state_sha256']
                records[2]['state_sha256'] = digest('scheduler1')
                records[3].update({'state_sha256': digest('rng1'), 'keys': ['cpu','cuda','numpy','python'], 'server_rng_restored': False})
                records[4].update({'global_step': 1, 'epochs_trained': 0, 'microbatches_skipped': 16,
                    'grpo_buffer_boundary_aligned': True, 'data_order_verified_here': False})
                put(learner / 'resume/resume_audit.jsonl',records,lines=True)
                audit = {'passed': True, 'missing': [], 'checkpoints': [checkpoint], 'records': records}
            ids = probability['question_ids']
            result = {'completed': True, 'mode': 'schedule_acceptance', 'phase': phase, 'variant': variant, 'schedule': c.SCHEDULE,
                'max_steps_for_both_phases': 2, 'optimizer_updates_this_process': 1, 'total_global_step': phase,
                'training_rollouts': 1, 'training_responses': 128, 'actual_completion_tokens': 256,
                'stock_importance_sampling_verified': True, 'frozen_parameters_and_buffers_unchanged': True,
                'finite_gradients_each_update': True, 'changed_adapter_tensors': 392, 'all_measured_advantages_zero': False,
                'parameters': {'target_layer_count': 196, 'trainable_tensors': 392, 'trainable_elements': 18464768, 'trainable_dtype': 'float32'},
                'optimizer_at_train_begin': optimizer, 'updates': [update], 'capacity_probe_performed': phase == 1,
                'capacity_result': capacity if phase == 1 else None, 'loaded_policy_matches': loaded_policy,
                'saved_policy_matches': policy, 'saved_inventory': saved, 'final_checkpoint': saved['checkpoint'], 'restore_audit': audit,
                'final_server_verification': {'passed': True, 'all_resident_weights_verified': True,
                    'sync_versions': [phase - 1,phase], 'generation_schedule': rollout['generation_schedule'], 'training_rollouts': 1, 'training_responses': 128},
                'actual_questions': {'question_ids': ids, 'unique_question_ids': sorted(set(ids)), 'responses': 128, 'question_counts': dict(c.Counter(ids))},
                'dataset_ordered_ids_sha256': digest('ordered dataset'), 'data_counts': {'raw_questions': 7345, 'kept_questions': 7345},
                'restored_v1_canonical_and_resident_exact': True if phase == 2 else None,
                'numerical_equivalence_claim': False, 'formal400_training': False, 'sampling_replay_bitwise_guaranteed': False, 'performance_claim': False}
            put(learner / 'result.json',result)
        (root / variant / 'both-phases.done').touch()
        shutil.rmtree(source)
    (root / 'all-phases.done').touch()
    put(root / 'benchmark_summary.json', {'passed': True, 'status': 'verified',
        'initialization_inventories_by_method': initial_reference,
        'common_protocol': {key:contract[key] for key in ('scientific_configuration_sha256', 'train_data_sha256', 'base_files_sha256', 'uv_lock_sha256', 'reward')}})


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='geora-schedule-collector-')
        self.root = Path(self.temp.name); fixture(self.root)
    def tearDown(self): self.temp.cleanup()
    def collect(self): return c.collect(self.root,self.root/'plan.json',SOURCE/'fixtures/probability_controls.json',self.root/'benchmark_summary.json')
    def test_complete_with_vllm_probability_drift(self):
        value = self.collect()
        self.assertTrue(value['passed'])
        replay = value['methods']['lora']['v1_raw_probability_replay']
        self.assertFalse(replay['exact_repeat'])
        self.assertEqual(replay['per_path_max_abs_logprob_delta']['learner_logprobs'],0.)
        self.assertAlmostEqual(replay['per_path_max_abs_logprob_delta']['vllm_logprobs'],.1)
    def test_missing_phase_incomplete(self):
        shutil.rmtree(self.root/'lora/phase-2')
        self.assertEqual(self.collect()['status'],'incomplete')
    def test_capacity_entropy_false_rejected(self):
        learner = self.root/'lora/phase-1/learner'
        capacity = c.read(learner/'capacity_result.json'); capacity['compute_entropy'] = False; put(learner/'capacity_result.json',capacity)
        result = c.read(learner/'result.json'); result['capacity_result'] = capacity; put(learner/'result.json',result)
        with self.assertRaisesRegex(ValueError,'capacity'): self.collect()
    def test_old_skip32_rejected(self):
        learner = self.root/'lora/phase-2/learner'
        records = c.rows(learner/'resume/resume_audit.jsonl'); records[-1]['microbatches_skipped'] = 32; put(learner/'resume/resume_audit.jsonl',records,lines=True)
        result = c.read(learner/'result.json'); result['restore_audit']['records'] = records; put(learner/'result.json',result)
        with self.assertRaisesRegex(ValueError,'skip16'): self.collect()
    def test_wrong_seed74_rejected(self):
        path = self.root/'lora/phase-2/learner/sync/rollouts.jsonl'
        value = c.rows(path); value[0]['request_seed'] = 74; put(path,value,lines=True)
        with self.assertRaisesRegex(ValueError,'seed'): self.collect()
    def test_resident_mismatch_rejected(self):
        path = self.root/'lora/phase-2/learner/sync/sync.jsonl'
        value = c.rows(path); value[0]['resident_checksums'][next(iter(value[0]['resident_checksums']))] = 'a'*64; put(path,value,lines=True)
        with self.assertRaisesRegex(ValueError,'resident bytes'): self.collect()
    def test_restored_v1_self_consistent_but_wrong_bytes_rejected(self):
        path = self.root/'lora/phase-2/learner/sync/sync.jsonl'
        value = c.rows(path); key = next(iter(value[0]['resident_checksums']))
        value[0]['resident_checksums'][key] = 'a'*64
        value[0]['resident_verification']['expected_resident_checksums'][key] = 'a'*64
        put(path,value,lines=True)
        with self.assertRaisesRegex(ValueError,'Restored v1'): self.collect()
    def test_saved_inventory_tamper_rejected(self):
        path = self.root/'lora/phase-2/learner/run_protocol.json'
        value = c.read(path); value['resume_input']['files_sha256']['optimizer.pt'] = 'a'*64; put(path,value)
        with self.assertRaisesRegex(ValueError,'saved inventory'): self.collect()
    def test_source_drift_rejected(self):
        path = self.root/'lora/phase-2/learner/run_protocol.json'
        value = c.read(path); value['immutable_contract']['source_sha256']['accept_schedule.py'] = 'a'*64; put(path,value)
        with self.assertRaisesRegex(ValueError,'source'): self.collect()
    def test_original_inventory_must_match_previous_matrix(self):
        path = self.root/'benchmark_summary.json'
        value = c.read(path); value['initialization_inventories_by_method']['lora']['files_sha256']['optimizer.pt'] = 'b'*64; put(path,value)
        with self.assertRaisesRegex(ValueError,'Checkpoint25 inventory'): self.collect()
    def test_optimizer_semantic_digest_mismatch_rejected(self):
        learner = self.root/'lora/phase-2/learner'
        records = c.rows(learner/'resume/resume_audit.jsonl'); records[1]['state_sha256'] = 'a'*64; put(learner/'resume/resume_audit.jsonl',records,lines=True)
        result = c.read(learner/'result.json'); result['restore_audit']['records'] = records; put(learner/'result.json',result)
        with self.assertRaisesRegex(ValueError,'digests'): self.collect()
    def test_missing_restore_kind_rejected(self):
        learner = self.root/'lora/phase-2/learner'
        records = c.rows(learner/'resume/resume_audit.jsonl'); records.pop(); put(learner/'resume/resume_audit.jsonl',records,lines=True)
        result = c.read(learner/'result.json'); result['restore_audit']['records'] = records; put(learner/'result.json',result)
        with self.assertRaisesRegex(ValueError,'five'): self.collect()
    def test_pair_question_mismatch_rejected(self):
        learner = self.root/'geora_mixed/phase-1/learner'
        p = c.rows(learner/'training_probabilities.jsonl'); p[0]['question_ids'] = [f'other-{q}' for q in p[0]['question_ids']]; put(learner/'training_probabilities.jsonl',p,lines=True)
        result = c.read(learner/'result.json'); ids = p[0]['question_ids']; result['actual_questions'] = {'question_ids': ids,'unique_question_ids': sorted(set(ids)),'question_counts':dict(c.Counter(ids)),'responses':128}; put(learner/'result.json',result)
        reward = c.rows(learner/'reward_audit.jsonl'); reward[0]['question_ids'] = ids
        for index,group in enumerate(reward[0]['grouping']['groups']): group['question_id'] = ids[index*8]
        put(learner/'reward_audit.jsonl',reward,lines=True)
        with self.assertRaisesRegex(ValueError,'Paired methods'): self.collect()
    def test_cli_exit_codes(self):
        command = [sys.executable,str(SOURCE/'collect_schedule_acceptance.py'),'--root',str(self.root),'--plan',str(self.root/'plan.json'), '--benchmark-summary', str(self.root/'benchmark_summary.json')]
        output = subprocess.run(command,capture_output=True,text=True)
        self.assertEqual(output.returncode,0,output.stderr+output.stdout)
        self.assertTrue(c.read(self.root/'schedule_acceptance_summary.json')['passed'])
        (self.root/'all-phases.done').unlink()
        self.assertEqual(subprocess.run(command,capture_output=True).returncode,1)
        (self.root/'all-phases.done').touch()
        (self.root/'lora/phase-1/learner/result.json').write_text('{}')
        self.assertEqual(subprocess.run(command,capture_output=True).returncode,2)


if __name__ == '__main__': unittest.main(verbosity=2)
