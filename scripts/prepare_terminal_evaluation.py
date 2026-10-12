#!/usr/bin/env python3
"""Read-only, stdlib-only terminal-100 preflight; never submits a GPU job.

This checks portable JSON, source bytes, checkpoint bytes and safetensors
headers. It does not deserialize training_state.pt, load a model, or certify
optimizer/RNG payload semantics. The existing evaluator revalidates factors.
Run on the same Setonix filesystem as the actual completed training output.
Python 3.6+; prints JSON only, with an sbatch argv only when every gate passes.
"""
import argparse
import ast
from contextlib import contextmanager
import fcntl
import gzip
import hashlib
import json
import math
from pathlib import Path
import shlex
import struct
import sys

EXECUTION_ONLY = {
    'runtime_limit_seconds', 'checkpoint_interval', 'checkpoint_reserve_seconds',
    'step_time_budget_seconds', 'checkpoint_boundary', 'resume_after_step',
    'status', 'implementation_status', 'runtime_estimation', 'selection_status', 'scope',
}
PROVENANCE_FIELDS = {
    'method', 'model_repository', 'model_revision', 'dataset_manifest_sha256',
    'training_config_sha256', 'initialization_sha256',
}
HELPERS = {'atomic_json', 'exclusive_run', 'initialization_hash', 'training_config_hash', 'utc_now', 'canonical_hash'}
BOUNDARY = 'after_rollout_and_optional_update'
RULE = 'verl_gsm8k_strict_1d367a4f_v1'
TRAINING_SOURCE_KEYS = {
    'configs/base_model.json', 'geora_layers.py', 'lora_layers.py',
    'scripts/check_grpo_continuation.py', 'scripts/check_grpo_training.py',
    'scripts/geora_check_utils.py', 'scripts/grpo_math.py', 'scripts/grpo_rollout.py',
    'scripts/grpo_training_state.py', 'scripts/gsm8k_data.py',
    'scripts/cached_training_rollout.py', 'scripts/cached_rollout.py',
    'scripts/cached_block_score.py', 'scripts/long_grpo_step.py',
    'scripts/long_capacity_check.py', 'third_party/verl/gsm8k.py',
    'third_party/verl/provenance.json', 'scripts/run_grpo_pilot.py', 'scripts/pilot_runtime.py',
}
EXPECTED_PRECISION = dict(frozen_weights='BF16', factors='FP32', initial_GeoRA_buffers='FP32',
    adapter_branch='FP32', correction_added_to_base='BF16', log_softmax_and_loss='FP32',
    optimizer_moments='FP32', checkpoint_adapter_storage='FP32')


def require(condition, name):
    if not condition:
        raise ValueError(name)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(config):
    return canonical_hash({key: value for key, value in config.items() if key not in EXECUTION_ONLY})


def inside(root, relative):
    root = Path(root).resolve()
    require(isinstance(relative, str) and not Path(relative).is_absolute(), 'Expected a relative artifact path')
    path = (root / relative).resolve()
    require(root in path.parents, 'Artifact path escapes owned output')
    return path


def descriptor_matches(path, descriptor):
    require(isinstance(descriptor, dict) and type(descriptor.get('bytes')) is int,
        'Malformed artifact byte descriptor')
    require(Path(path).is_file() and Path(path).stat().st_size == descriptor['bytes'],
        'Artifact byte length differs: ' + str(path))
    require(file_hash(path) == descriptor.get('sha256'), 'Artifact SHA256 differs: ' + str(path))


@contextmanager
def inactive_output(root):
    path = root / '.pilot.lock'
    require(path.is_file(), 'Missing actual training run lock')
    # Opening the existing file for reading neither creates nor rewrites it.
    with path.open('r') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            raise ValueError('A live training process still owns this output')
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def source_snapshot(path, expected):
    snapshot = read_json(path)
    files = snapshot.get('files')
    require(isinstance(files, dict) and set(files) == set(expected), 'Historical source snapshot names differ')
    for name, value in files.items():
        require(value.get('sha256') == expected[name]
            and hashlib.sha256(value['content'].encode()).hexdigest() == expected[name],
            'Historical source snapshot bytes differ: ' + name)
    return snapshot


def verify_migration(root, report):
    marker = report.get('source_equivalent_migration')
    if marker is None:
        return dict(present=False, source_history='Original report identity; no source transition asserted')
    migration_path = inside(root, marker['relative_path'])
    require(file_hash(migration_path) == marker['sha256'], 'Migration descriptor checksum differs')
    migration = read_json(migration_path)
    require(migration.get('status') == 'complete' and migration.get('kind') == 'source_equivalent_migration'
        and migration.get('migration_id') == marker['migration_id'], 'Incomplete or different source migration')
    require(migration.get('pilot_run_id') == report['pilot_run_id']
        and Path(migration['new_output_directory']).resolve() == root,
        'Migration output/run identity differs')
    old_path = root / 'migration/original/pilot_report.json'
    require(file_hash(old_path) == migration['original_report_sha256'], 'Archived old report differs')
    old = read_json(old_path)
    require(old['identity']['output_directory'] == migration['original_output_directory'],
        'Migration archived original output identity differs')
    copied = inside(root, migration['copied_checkpoint_relative_path'])
    for relative, descriptor in migration['checkpoint_bytes'].items():
        descriptor_matches(inside(copied, relative), descriptor)
    copied_manifest = read_json(copied / 'manifest.json')
    require(copied_manifest.get('provenance') == old['provenance']
        and copied_manifest.get('progress', {}).get('active_step_records') == old['active_step_records'],
        'Copied migration boundary is not the archived history')
    n = migration['preserved_completed_optimizer_steps']
    require(type(n) is int and 0 < n < 100 and old['logical_optimizer_steps'] == n,
        'Invalid preserved migration boundary')
    require(copied_manifest['progress'].get('completed_optimizer_steps') == n
        and copied_manifest['progress'].get('completed_rollouts') == n
        and copied_manifest['progress'].get('next_data_position') == n, 'Copied migration boundary progress differs')
    require(old['pilot_run_id'] == report['pilot_run_id'] and old['created_at_utc'] == report['created_at_utc']
        and old['provenance'] == report['provenance'] and config_hash(old['config']) == config_hash(report['config']),
        'Migration replaced prior run/config/provenance')
    require(report['active_step_records'][:n] == old['active_step_records']
        and report['segments'][:len(old['segments'])] == old['segments'],
        'Historical records or original segment runtime facts were rewritten')
    require(old['identity']['source_sha256'] == migration['old_source_sha256']
        and report['identity']['source_sha256'] == migration['new_source_sha256'],
        'Migration source identities differ')
    ranges = migration['source_audit_ranges']
    require(len(ranges) == 2 and ranges[0]['logical_steps'] == [1, n]
        and ranges[1]['logical_steps'] == [n + 1, 100]
        and ranges[0]['segment_ids'] == [item['id'] for item in old['segments']],
        'Missing precise historical/future source ranges')
    for item, expected in zip(ranges, [migration['old_source_sha256'], migration['new_source_sha256']]):
        snapshot = source_snapshot(inside(root, item['snapshot_relative_path']), expected)
        tree = inside(root, item['source_root_relative_path'])
        for name, descriptor in snapshot['files'].items():
            require(file_hash(inside(tree, name)) == descriptor['sha256'], 'Portable historical/current source differs')
    changed = sorted(name for name in migration['old_source_sha256']
        if migration['old_source_sha256'][name] != migration['new_source_sha256'][name])
    require(changed == migration['changed_source_files'], 'Declared source changes differ')
    for relative, descriptor in migration['original_history_bytes'].items():
        descriptor_matches(inside(root / 'segments', relative), descriptor)
    evidence_path = inside(root, migration['source_equivalence_evidence_relative_path'])
    require(file_hash(evidence_path) == migration['source_equivalence_evidence_sha256'], 'Archived equivalence evidence differs')
    evidence = read_json(evidence_path)
    require(evidence.get('status') == 'passed' and evidence.get('source_binding') == dict(
        old_source_sha256=migration['old_source_sha256'], new_source_sha256=migration['new_source_sha256']),
        'Equivalence evidence is not bound to actual old/new source bytes')
    for item in migration.get('additional_equivalence_evidence', []):
        descriptor_matches(inside(root, item['relative_path']), item)
    return dict(present=True, preserved_optimizer_steps=n, migration_sha256=file_hash(migration_path),
        old_source_sha256=migration['old_source_sha256'], new_source_sha256=migration['new_source_sha256'],
        source_audit_ranges=ranges,
        source_history='Historical segments remain bound to archived historical sources; later source identity is not applied retroactively')


def safetensors_header(path):
    with Path(path).open('rb') as handle:
        length_bytes = handle.read(8)
        require(len(length_bytes) == 8, 'Short safetensors file')
        length = struct.unpack('<Q', length_bytes)[0]
        require(0 < length < 16 * 1024 * 1024, 'Invalid safetensors header length')
        header = json.loads(handle.read(length).decode('utf-8'))
    entries = {key: value for key, value in header.items() if key != '__metadata__'}
    size = Path(path).stat().st_size - 8 - length
    offsets = []
    for name, descriptor in entries.items():
        require(descriptor.get('dtype') == 'F32', 'Expected FP32 adapter tensor: ' + name)
        shape = descriptor['shape']
        require(isinstance(shape, list) and len(shape) == 2 and all(type(n) is int and n > 0 for n in shape),
            'Invalid adapter shape: ' + name)
        start, end = descriptor['data_offsets']
        require(type(start) is int and type(end) is int and 0 <= start < end <= size
            and end - start == 4 * shape[0] * shape[1], 'Invalid tensor byte range: ' + name)
        offsets.append((start, end))
    ordered = sorted(offsets)
    require(bool(ordered) and ordered[0][0] == 0 and ordered[-1][1] == size
        and all(a[1] == b[0] for a, b in zip(ordered, ordered[1:])), 'Safetensors data is not fully contiguous')
    return entries, 8 + length


def tensor_byte_hash(path, data_start, descriptor, digest=None):
    start, end = descriptor['data_offsets']
    actual = hashlib.sha256() if digest is None else digest
    with Path(path).open('rb') as handle:
        handle.seek(data_start + start)
        remaining = end - start
        while remaining:
            chunk = handle.read(min(1024 * 1024, remaining))
            require(bool(chunk), 'Short tensor payload')
            actual.update(chunk)
            remaining -= len(chunk)
    return actual.hexdigest()


def geora_initialization(arguments, report, base, trained_header, trained_start, adapter_path):
    require(arguments.init_dir is not None, 'GeoRA requires the actual untrained initialization directory')
    root = arguments.init_dir.resolve()
    manifest = read_json(root / 'manifest.json')
    require(manifest.get('optimizer_steps') == 0 and manifest.get('geora_checkpoint_format') == 1,
        'GeoRA initialization is not an untrained format1 artifact')
    for name in ['model_repository', 'model_revision']:
        require(manifest.get(name) == base[name], 'GeoRA initialization base differs: ' + name)
    for name, value in base['initialization'].items():
        require(manifest.get(name) == value, 'GeoRA initialization setting differs: ' + name)
    expected_targets = {"model.layers.{}.{}".format(layer, projection)
        for layer in range(28) for projection in base['target_projections']}
    require(len(manifest['target_modules']) == 196
        and {item['name'] for item in manifest['target_modules']} == expected_targets,
        'GeoRA initialization target modules differ')
    source_tree = ast.parse((arguments.repo / 'geora_layers.py').read_text(encoding='utf-8'))
    policy = [ast.literal_eval(item.value) for item in source_tree.body if isinstance(item, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'PRECISION_POLICY' for target in item.targets)]
    require(len(policy) == 1 and manifest.get('precision_policy') == policy[0],
        'GeoRA full initialization precision policy differs from executed source')
    require(manifest.get('scaling') == base['initialization']['alpha'] / base['initialization']['rank'],
        'GeoRA initialization alpha/r scaling differs')
    precision = manifest['precision_policy']
    for name in ['mask_dtype', 'svd_dtype', 'initial_factor_dtype', 'residual_calculation_dtype', 'trainable_adapter_dtype']:
        require(precision.get(name) == 'float32', 'GeoRA initialization precision differs: ' + name)
    require(precision.get('initialization_device') == 'cpu'
        and precision.get('training_frozen_model_dtype') == 'bfloat16', 'GeoRA initialization device/base precision differs')
    path = root / 'adapter.safetensors'
    header, data_start = safetensors_header(path)
    expected_names = {target + '.' + factor for target in expected_targets for factor in ['A', 'B', 'A0', 'B0']}
    require(set(header) == expected_names, 'GeoRA initial tensor names differ')
    for target in manifest['target_modules']:
        name = target['name']
        require(header[name + '.A']['shape'] == target['A_shape']
            and header[name + '.B']['shape'] == target['B_shape']
            and target['A_shape'][0] == target['B_shape'][1] == base['initialization']['rank']
            and [target['B_shape'][0], target['A_shape'][1]] == target['weight_shape'],
            'GeoRA initial tensor shapes differ from target metadata')
    digest = hashlib.sha256()
    for name in sorted(header):
        digest.update(name.encode())
        digest.update(str(tuple(header[name]['shape'])).encode())
        tensor_byte_hash(path, data_start, header[name], digest)
    initialization_sha = canonical_hash(dict(adapter_sha256=digest.hexdigest(),
        rank=base['initialization']['rank'], alpha=float(base['initialization']['alpha'])))
    require(initialization_sha == report['provenance']['initialization_sha256'], 'GeoRA actual initialization digest differs')
    for target in expected_targets:
        for factor in ['A', 'B']:
            original = header[target + '.' + factor]
            initial = header[target + '.' + factor + '0']
            require(original['shape'] == initial['shape']
                and tensor_byte_hash(path, data_start, original) == tensor_byte_hash(path, data_start, initial),
                'GeoRA current factors were already trained in initialization')
            trained = trained_header.get(target + '.' + factor + '0')
            require(trained is not None and trained['shape'] == initial['shape']
                and tensor_byte_hash(adapter_path, trained_start, trained) == tensor_byte_hash(path, data_start, initial),
                'Frozen trained GeoRA A0/B0 differ from original initialization')
    return dict(directory=str(root), manifest_sha256=file_hash(root / 'manifest.json'),
        adapter_sha256=file_hash(path), initialization_sha256=initialization_sha,
        optimizer_steps=0, targets=196)


def helper_ast(content):
    tree = ast.parse(content)
    result = {item.name: ast.dump(item, include_attributes=False) for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name in HELPERS}
    require(set(result) == HELPERS, 'Missing baseline-evaluation helper definition')
    constants = {target.id: ast.dump(item.value, include_attributes=False)
        for item in tree.body if isinstance(item, ast.Assign)
        for target in item.targets if isinstance(target, ast.Name) and target.id == 'EXECUTION_ONLY_FIELDS'}
    result.update(constants)
    require('EXECUTION_ONLY_FIELDS' in result, 'Missing evaluation config-hash constant')
    # Compare only bindings used by evaluation helpers; training may import more.
    required = {'datetime', 'timezone', 'hashlib', 'json', 'tempfile', 'os', 'fcntl', 'contextmanager', 'Path'}
    imports = {}
    for item in tree.body:
        if isinstance(item, ast.Import):
            for alias in item.names:
                name = alias.asname or alias.name.split('.')[0]
                if name in required:
                    imports[name] = ('import', alias.name, alias.asname)
        elif isinstance(item, ast.ImportFrom):
            for alias in item.names:
                name = alias.asname or alias.name
                if name in required:
                    imports[name] = ('from', item.module, alias.name, alias.asname, item.level)
    require(set(imports) == required, 'Missing required evaluation-helper import binding')
    result['module_imports'] = imports
    return result


def baseline_sources(repo, baseline, snapshot):
    require(set(snapshot['executed_sources']) == set(baseline['source_sha256']), 'Baseline source inventory differs')
    compared = []
    for name, expected in sorted(baseline['source_sha256'].items()):
        old = snapshot['executed_sources'][name]
        require(hashlib.sha256(old.encode()).hexdigest() == expected, 'Baseline source snapshot differs: ' + name)
        current = inside(repo, name)
        actual = file_hash(current)
        if actual != expected:
            require(name == 'scripts/pilot_runtime.py'
                and helper_ast(old) == helper_ast(current.read_text(encoding='utf-8')),
                'Evaluation source differs from baseline: ' + name)
            comparison = 'Five imported helper ASTs plus their canonical_hash dependency, config-hash constant and required stdlib bindings identical; other training functions may differ'
        else:
            comparison = 'Exact file bytes identical'
        compared.append(dict(path=name, baseline_sha256=expected, current_sha256=actual, comparison=comparison))
    return compared


def prepared_rows(data_dir, manifest, split):
    descriptor = manifest['splits'][split]
    path = inside(data_dir, descriptor['relative_path'])
    require(file_hash(path) == descriptor['sha256'], 'Prepared ' + split + ' file SHA differs')
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
    require(len(rows) == descriptor['rows'], 'Prepared ' + split + ' row count differs')
    return rows


def inspect(arguments):
    root = arguments.training_output.resolve()
    repo = arguments.repo.resolve()
    config = read_json(arguments.config)
    baseline = read_json(arguments.baseline_report)
    baseline_snapshot = read_json(arguments.baseline_snapshot)
    base = read_json(repo / 'configs/base_model.json')
    require(config.get('optimizer_steps') == 100 and config.get('questions_per_batch') == 1
        and config.get('answers_per_question') == 8 and config.get('max_new_tokens') == 4096
        and config.get('validation_question_limit') == 128 and config.get('updates_per_rollout') == 1,
        'Not the fixed terminal100/eight-answer/4096/128 pilot configuration')
    require(config.get('answer_rule') == RULE and config.get('truncated_answer_reward') == 0,
        'Changed terminal validation reward rule')
    require(baseline.get('status') == 'completed' and baseline.get('complete') is True
        and baseline.get('method') == 'base' and baseline.get('evaluated_questions') == 128
        and baseline.get('requested_questions') == 128 and baseline.get('split') == 'validation',
        'Baseline128 is not completed')
    require(baseline['training_config_sha256'] == config_hash(config)
        and baseline['answer_rule'] == RULE, 'Baseline policy/config/answer rule differs')
    decode = baseline['decode']
    require(decode.get('batch_size') == 1 and decode.get('do_sample') is False
        and decode.get('use_cache') is True and decode.get('attention_backend') == 'eager'
        and decode.get('max_new_tokens') == 4096 and decode.get('repetition_penalty') == 1.0,
        'Baseline decoding protocol differs')
    require(baseline.get('precision') == dict(frozen_weights='BF16', adapters='FP32', low_rank_branches='FP32'),
        'Baseline precision differs')
    for key in ['model_repository', 'model_revision']:
        require(base[key] == baseline[key], 'Current pinned base differs from baseline: ' + key)
    require((arguments.base_checkpoint.resolve() / 'model.safetensors').is_file(), 'Actual base model weight file missing')
    metadata = arguments.base_checkpoint.resolve() / '.cache/huggingface/download/model.safetensors.metadata'
    require(metadata.read_text().splitlines()[0] == base['model_revision'], 'Actual base checkpoint revision differs')
    original_checkpoint = (repo / base['checkpoint_directory']).resolve()
    actual_base = arguments.base_checkpoint.resolve()
    if actual_base != original_checkpoint:
        staging = read_json(actual_base / 'staging_record.json')
        require(staging.get('status') == 'validated' and staging.get('model_revision') == base['model_revision']
            and Path(staging['source']).resolve() == original_checkpoint, 'Base copy lacks valid staging provenance')
    data_manifest = read_json(arguments.data_dir / 'manifest.json')
    data_sha = file_hash(arguments.data_dir / 'manifest.json')
    require(data_sha == baseline['dataset_manifest_sha256'], 'Dataset manifest differs from Base128')
    train = prepared_rows(arguments.data_dir, data_manifest, 'train')
    validation = prepared_rows(arguments.data_dir, data_manifest, 'validation')[:128]
    require([row['id'] for row in validation] == baseline['question_ids']
        and len({row['id'] for row in validation}) == 128, 'Ordered validation128 IDs differ from baseline')
    require(data_manifest['prompt_instruction'] == config['prompt_instruction'], 'Dataset prompt differs')
    train_ids = {row['id'] for row in train}
    train_questions = {row['question'] for row in train}
    require(all(row['source_split'] == 'train' and row['id'] not in train_ids
        and row['question'] not in train_questions for row in validation), 'Validation data leaks into training')
    source_comparison = baseline_sources(repo, baseline, baseline_snapshot)
    with inactive_output(root):
        report_path = root / 'pilot_report.json'
        report_sha = file_hash(report_path)
        report = read_json(report_path)
        require(report.get('status') == 'completed' and report.get('stage') == 'gsm8k_bounded_grpo_pilot'
            and report.get('logical_optimizer_steps') == 100 and report.get('next_data_position') == 100
            and report.get('latest_checkpoint_logical_step') == 100,
            'Training has not completed the actual fixed100 boundary')
        require(report.get('answer_rule') == RULE and report.get('precision') == EXPECTED_PRECISION,
            'Training reward rule or precision differs')
        require(config_hash(report['config']) == config_hash(config), 'Training configuration differs')
        method = report['identity']['method']
        require(method in ['lora', 'geora'], 'Terminal method must be LoRA or GeoRA')
        if method == 'lora':
            require(arguments.init_dir is None, 'LoRA evaluation cannot take a GeoRA initialization')
        identity = report['identity']
        require(Path(identity['output_directory']).resolve() == root
            and identity['dataset_manifest_sha256'] == data_sha
            and identity['training_config_sha256'] == config_hash(config), 'Current training identity differs')
        selected_train = train[:config['train_question_limit']]
        require(identity['train_question_ids'] == [row['id'] for row in selected_train]
            and identity['train_rows_sha256'] == canonical_hash(selected_train), 'Current train order/rows differ')
        require(set(identity['source_sha256']) == TRAINING_SOURCE_KEYS, 'Training source inventory differs')
        for name, expected in identity['source_sha256'].items():
            require(file_hash(inside(repo, name)) == expected, 'Current training source identity differs: ' + name)
        provenance = report['provenance']
        require(set(provenance) == PROVENANCE_FIELDS and provenance['method'] == method
            and provenance['training_config_sha256'] == config_hash(config)
            and provenance['dataset_manifest_sha256'] == data_sha
            and all(provenance[key] == base[key] for key in ['model_repository', 'model_revision']),
            'Terminal training provenance differs')
        checkpoint = Path(report['latest_checkpoint']).resolve()
        require(root in checkpoint.parents and (root / 'checkpoints').resolve() in checkpoint.parents,
            'Actual latest checkpoint is outside the owned training checkpoint tree')
        manifest_path = checkpoint / 'manifest.json'
        manifest = read_json(manifest_path)
        progress = manifest['progress']
        require(manifest.get('status') == 'complete' and type(manifest.get('training_checkpoint_format')) is int
            and manifest['training_checkpoint_format'] == 1 and manifest.get('boundary') == BOUNDARY
            and manifest.get('scheduler_kind') == 'constant_v1' and manifest.get('optimizer_type') == 'AdamW',
            'Actual latest manifest is not a complete compatible training boundary')
        require(progress.get('boundary') == BOUNDARY and progress.get('rollout_in_flight') is False
            and all(type(progress.get(key)) is int and progress[key] == 100 for key in
                ['completed_optimizer_steps', 'completed_rollouts', 'next_data_position'])
            and progress.get('pilot_run_id') == report['pilot_run_id'], 'Actual checkpoint is not at100 completed rollouts/updates/cursor')
        require(manifest['provenance'] == provenance
            and manifest['provenance_sha256'] == canonical_hash(provenance), 'Latest checkpoint provenance differs')
        records = report['active_step_records']
        require(len(records) == 100 and progress.get('active_step_records') == records,
            'Terminal checkpoint does not carry exactly the report100-step history')
        token_count = 0
        for step, reference in enumerate(records, 1):
            require(reference.get('logical_step') == step, 'Report step history is not contiguous')
            path = inside(root, reference['relative_path'])
            if path.is_file():
                payload = path.read_bytes()
            else:
                archive = path.with_suffix(path.suffix + '.gz')
                require(archive.is_file(), 'Missing actual immutable step record')
                payload = gzip.decompress(archive.read_bytes())
            require(hashlib.sha256(payload).hexdigest() == reference['sha256'], 'Actual step record SHA differs')
            record = json.loads(payload.decode('utf-8'))
            require(record.get('logical_step') == step and record.get('pilot_run_id') == report['pilot_run_id']
                and record.get('valid_completion_tokens') == reference['valid_completion_tokens']
                and record.get('question_ids') == [identity['train_question_ids'][(step - 1) % len(selected_train)]],
                'Actual step record identity/cursor differs')
            token_count += reference['valid_completion_tokens']
        require(report['completed_answer_tokens'] == progress['completed_answer_tokens'] == token_count,
            'Committed answer token totals differ')
        require(set(manifest['files']) == {'adapter.safetensors', 'training_state.pt'}, 'Incomplete checkpoint payload inventory')
        for name, descriptor in manifest['files'].items():
            descriptor_matches(checkpoint / name, descriptor)
        adapter_path = checkpoint / 'adapter.safetensors'
        header, data_start = safetensors_header(adapter_path)
        targets = {'model.layers.{}.{}'.format(layer, projection)
            for layer in range(28) for projection in base['target_projections']}
        factors = ['A', 'B', 'A0', 'B0'] if method == 'geora' else ['A', 'B']
        expected_names = {target + '.' + factor for target in targets for factor in factors}
        require(set(header) == set(manifest['adapter_tensors']) == expected_names,
            'Actual checkpoint adapter tensor names/196-target inventory differ')
        optimizer_names = [name for group in manifest['optimizer_parameter_names'] for name in group]
        require(len(optimizer_names) == 392 and set(optimizer_names) ==
            {target + '.' + factor for target in targets for factor in ['A', 'B']},
            'Checkpoint optimizer does not cover all392 trainable A/B exactly once')
        for name, item in header.items():
            require(manifest['adapter_tensors'][name] == dict(shape=item['shape'], dtype='torch.float32'),
                'Checkpoint adapter shape/storage descriptor differs: ' + name)
        initialization = geora_initialization(arguments, report, base, header, data_start, adapter_path) if method == 'geora' else dict(
            kind='Deterministic seed/rank/alpha/Astd LoRA initialization reconstructed and digest-checked by existing evaluator',
            initialization_sha256=provenance['initialization_sha256'])
        for segment in report['segments']:
            for name in ['torch_version', 'hip_version', 'transformers_version', 'gpu_name']:
                require(segment.get('runtime', {}).get(name) == baseline[name],
                    'Recorded training segment environment differs from baseline: ' + name)
        require(not any(item.get('passed') is False for item in report.get('checks', [])),
            'Training report retains a failed gate')
        migration = verify_migration(root, report)
        require(file_hash(report_path) == report_sha, 'Report changed while inspected')
    output = arguments.evaluation_output.resolve()
    require(output != root and root not in output.parents and output not in root.parents,
        'Validation output cannot overlap the training output')
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())), 'Fresh validation output must be absent or empty')
    require(arguments.runtime_seconds > 0 and arguments.wall_seconds > arguments.runtime_seconds + 90,
        'Evaluation needs a positive runtime with more than90s Slurm reserve')
    require(arguments.wall_seconds <= 86400, 'gpu partition supports at most24h walltime')
    wall = '{:02d}:{:02d}:{:02d}'.format(arguments.wall_seconds // 3600,
        arguments.wall_seconds % 3600 // 60, arguments.wall_seconds % 60)
    log = arguments.log_path if arguments.log_path is not None else output.parent / 'logs' / ('validation-' + method + '100-%j.log')
    argv = ['sbatch', '--parsable', '--account=' + arguments.account, '--partition=gpu', '--nodes=1', '--gres=gpu:1',
        '--time=' + wall, '--output=' + str(log.resolve()), str((repo / 'jobs/evaluate_gsm8k_pilot.sbatch').resolve()),
        method, str(arguments.data_dir.resolve()), str(actual_base), str(output), str(checkpoint),
        '' if arguments.init_dir is None else str(arguments.init_dir.resolve()), '128', 'fresh',
        str(arguments.config.resolve()), str(arguments.runtime_seconds)]
    return dict(status='passed', submitted_gpu_job=False, optimizer_steps=100, completed_rollouts=100,
        next_data_position=100, method=method, adapter_checkpoint=str(checkpoint),
        manifest_sha256=file_hash(manifest_path), checkpoint_files=manifest['files'],
        training_report_sha256=report_sha, config_sha256=config_hash(config), dataset_manifest_sha256=data_sha,
        baseline_report_sha256=file_hash(arguments.baseline_report), validation_question_ids=baseline['question_ids'],
        precision=report['precision'], initialization=initialization, source_comparison=source_comparison,
        migration=migration, recorded_training_segments=[dict(segment_id=item['id'],
            source_git_commit=item.get('runtime', {}).get('source_git_commit'),
            slurm_job_id=item.get('runtime', {}).get('slurm_job_id'), logical_range=[item['start_logical_step'] + 1, item['end_logical_step']])
            for item in report['segments']],
        baseline_environment={key:baseline[key] for key in ['torch_version', 'hip_version', 'transformers_version', 'gpu_name']},
        limitations=['Does not deserialize optimizer/RNG pickle; checks its byte hash and manifest only',
            'Does not replace independent per-step GRPO audit or final128 answer audit',
            'Current environment/factor finiteness are rechecked by evaluator; output contains baseline environment'],
        sbatch_argv=argv, shell_command=' '.join(shlex.quote(value) for value in argv),
        log_parent_must_exist=str(log.resolve().parent))


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['training-output', 'repo', 'config', 'baseline-report', 'baseline-snapshot', 'data-dir', 'base-checkpoint', 'evaluation-output']:
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--init-dir', type=Path)
    parser.add_argument('--account', default='pawsey0807-gpu')
    parser.add_argument('--runtime-seconds', type=int, default=3300)
    parser.add_argument('--wall-seconds', type=int, default=3600)
    parser.add_argument('--log-path', type=Path)
    return parser.parse_args()


if __name__ == '__main__':
    try:
        result = inspect(parse_arguments())
    except Exception as error:
        print(json.dumps(dict(status='failed', submitted_gpu_job=False, error=str(error)), indent=2))
        sys.exit(1)
    print(json.dumps(result, indent=2, sort_keys=True))
