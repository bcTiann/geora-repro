#!/usr/bin/env python3
"""CPU-only, independent audit of preserved GSM8K 256/4096-token generations.

No model class, GPU, SSH, evaluator, or repository scoring helper is imported.
The local tokenizer only verifies token/text identity and reconstructs prompts.
Pinned upstream reward source is executed after its SHA256 has been verified,
then checked against an independently written finditer implementation.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import html
import json
import math
from pathlib import Path
import re
import runpy
import shutil
import statistics
import subprocess


MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
MODEL_REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
DATA_REVISION = '3101c7d5072418e28b9008a6636bde82a006892c'
RAW_TRAIN_SHA256 = '17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465'
VERL_COMMIT = '1d367a4f459a43768f76312c0d3430a99ec46b8c'
VERL_SHA256 = '497c6e32b708d0bc8803b5a048ca5983de7756eb9c146abdc56e6f5d28767189'
OLD_RULE = 'single_final_hash_marker_exact_decimal_v1'
NEW_RULE = 'verl_gsm8k_strict_1d367a4f_v1'
PROMPT_INSTRUCTION = ('Solve the following math problem. Explain your reasoning, then end your '
                      'answer with a final line in the form #### <number>. Write only the numeric '
                      'answer after ####, without units.')
VERL_PATTERN = re.compile(r'#### (\-?[0-9\.\,]+)')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def json_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def require(audit, name, condition, **evidence):
    audit['checks'].append(dict(name=name, passed=bool(condition), **evidence))
    if not condition:
        raise AssertionError(name)


def numeric_string(text):
    """Historical grammar, implemented without the historical parser regex."""
    if not isinstance(text, str) or not text:
        return None
    unsigned = text[1:] if text[0] in '+-' else text
    if not unsigned or unsigned.count('.') > 1:
        return None
    integer, point, fraction = unsigned.partition('.')
    digits = set('0123456789')
    if point and (not fraction or not set(fraction) <= digits):
        return None
    if ',' in integer:
        groups = integer.split(',')
        if not 1 <= len(groups[0]) <= 3 or any(len(group) != 3 for group in groups[1:]):
            return None
        if any(not group or not set(group) <= digits for group in groups):
            return None
    elif integer:
        if not set(integer) <= digits:
            return None
    elif not point:
        return None
    return Decimal(text.replace(',', ''))


def decimal_extract(text):
    if text.count('####') != 1:
        return None
    return numeric_string(text.split('####', 1)[1].strip())


def canonical(number):
    if number is None:
        return None
    if number == 0:
        return '0'
    rendered = format(number, 'f')
    return rendered.rstrip('0').rstrip('.') if '.' in rendered else rendered


def strict_extract(text):
    """Independent finditer implementation of pinned verl strict semantics."""
    result = None
    for match in VERL_PATTERN.finditer(text[-300:]):
        result = match.group(1)
    return None if result is None else result.replace(',', '').replace('$', '')


def grade(text, target, ended, rule):
    if rule == OLD_RULE:
        numeric = decimal_extract(text)
        extracted = canonical(numeric)
        raw = float(numeric is not None and numeric == numeric_string(target))
    elif rule == NEW_RULE:
        extracted = strict_extract(text)
        raw = float(extracted is not None and extracted == target)
    else:
        raise ValueError(rule)
    return dict(extraction=extracted, raw_reward=raw, evaluation_reward=raw if ended else 0.0)


def mutually_exclusive_category(answer, grading):
    # Truncation is assigned first; the remaining groups contain EOS answers.
    if answer['truncated']:
        return 'truncated'
    if grading['extraction'] is None:
        return 'completed_without_extraction'
    if grading['raw_reward'] == 0:
        return 'completed_extracted_incorrect'
    return 'completed_correct'


def recount(answers, gradings):
    count = len(answers)
    categories = Counter(mutually_exclusive_category(answer, grading)
                         for answer, grading in zip(answers, gradings))
    tokens = [len(answer['token_ids']) for answer in answers]
    durations = [answer['generation_seconds'] for answer in answers]
    return dict(
        evaluated_questions=count, requested_questions=128,
        correct_count=sum(int(row['evaluation_reward']) for row in gradings),
        accuracy=sum(row['evaluation_reward'] for row in gradings) / count,
        parse_failure_count=sum(row['extraction'] is None for row in gradings),
        parsed_incorrect_count=sum(row['extraction'] is not None and row['raw_reward'] == 0 for row in gradings),
        correct_but_truncated_count=sum(row['raw_reward'] == 1 and answer['truncated']
                                      for answer, row in zip(answers, gradings)),
        truncated_answer_count=sum(answer['truncated'] for answer in answers),
        eos_answer_count=sum(answer['ended_with_eos'] for answer in answers),
        generated_tokens=sum(tokens), complete=count == 128,
        mutually_exclusive_categories={name: categories[name] for name in [
            'completed_correct', 'completed_extracted_incorrect', 'completed_without_extraction', 'truncated']},
        token_count=dict(min=min(tokens), median=statistics.median(tokens), mean=statistics.mean(tokens), max=max(tokens)),
        generation_seconds_sum=sum(durations),
    )


def source_comparison(report, repo):
    result = []
    for name, recorded in report['source_sha256'].items():
        path = repo / name
        current = sha256(path) if path.is_file() else None
        result.append(dict(relative_path=name, runtime_recorded_sha256=recorded,
                           current_checkout_sha256=current, current_equals_runtime_record=current == recorded))
    return result


def preflight(args, audit):
    repo = args.repo.resolve()
    dataset = repo / 'datasets/gsm8k'
    manifest_path = dataset / 'manifest.json'
    manifest = load_json(manifest_path)
    require(audit, 'pinned_dataset_revision_and_prompt', manifest['source_commit'] == DATA_REVISION
            and manifest['prompt_instruction'] == PROMPT_INSTRUCTION)
    require(audit, 'pinned_raw_train_sha256', sha256(dataset / 'raw/train.jsonl') == RAW_TRAIN_SHA256)
    groups = {name: json_lines(dataset / (name + '.jsonl')) for name in ['validation', 'train', 'smoke']}
    for name, rows in groups.items():
        require(audit, name + '_persisted_count_and_sha256',
                len(rows) == manifest['splits'][name]['rows']
                and sha256(dataset / (name + '.jsonl')) == manifest['splits'][name]['sha256'])
    rebuilt = []
    for index, original in enumerate(json_lines(dataset / 'raw/train.jsonl')):
        number = decimal_extract(original['answer'])
        if number is None:
            raise AssertionError('Unparseable original reference at index ' + str(index))
        rebuilt.append(dict(id=f'train:{index:05d}', source_split='train', source_index=index,
                            question=original['question'], target=canonical(number)))
    ordered = sorted(rebuilt, key=lambda row: hashlib.sha256(f"{manifest['split_seed']}:{row['id']}".encode()).hexdigest())
    require(audit, 'validation_128_exact_order_reconstructed_from_pinned_raw_train',
            len(groups['validation']) == 128 and groups['validation'] == ordered[:128]
            and manifest['validation_ids'] == [row['id'] for row in groups['validation']])
    require(audit, 'all_prepared_training_and_smoke_rows_reconstructed',
            groups['train'] == ordered[128:] and groups['smoke'] == ordered[128:144])
    held = groups['validation']
    require(audit, 'validation_IDs_and_question_texts_disjoint_from_train_and_smoke',
            all(not ({row['id'] for row in held} & {row['id'] for row in groups[name]})
                and not ({row['question'] for row in held} & {row['question'] for row in groups[name]})
                for name in ['train', 'smoke']))
    source_path = repo / 'third_party/verl/gsm8k.py'
    provenance = load_json(repo / 'third_party/verl/provenance.json')
    require(audit, 'pinned_verl_source_and_commit_metadata', sha256(source_path) == VERL_SHA256
            and provenance['commit'] == VERL_COMMIT and provenance['sha256'] == VERL_SHA256)
    upstream = runpy.run_path(str(source_path))
    probes = [
        ('#### 9.00', '9', '9.00', 0.0), ('#### 9', '9', '9', 1.0),
        ('#### 1,000 dollars', '1000', '1000', 1.0), ('#### 1,2,3', '123', '123', 1.0),
        ('#### 1\n#### 2', '2', '2', 1.0), ('answer is 2', '2', None, 0.0),
        ('####2', '2', None, 0.0), ('####  2', '2', None, 0.0),
        ('#### +2', '2', None, 0.0), ('#### -2', '-2', '-2', 1.0),
        ('#### .5', '.5', '.5', 1.0), ('#### 2. then explanation', '2', '2.', 0.0),
        ('#### 2\n' + 'x' * 301, '2', None, 0.0), ('x' * 301 + '#### 2', '2', '2', 1.0),
        ('#### 2/3', '2', '2', 1.0), ('#### $2', '2', None, 0.0),
        ('#### 2e3', '2', '2', 1.0), ('#### 2... end', '2', '2...', 0.0),
    ]
    audit['parser_probes'] = []
    for text, target, expected_extraction, expected_reward in probes:
        source_extraction = upstream['extract_solution'](text, method='strict')
        source_reward = float(upstream['compute_score'](text, target, method='strict', format_score=0.0, score=1.0))
        independent = grade(text, target, True, NEW_RULE)
        require(audit, 'strict_semantics_probe_' + str(len(audit['parser_probes'])),
                source_extraction == independent['extraction'] == expected_extraction
                and source_reward == independent['raw_reward'] == expected_reward)
        audit['parser_probes'].append(dict(text=text, target=target, extraction=source_extraction, reward=source_reward))
    require(audit, 'historical_numeric_representation_and_format_probe',
            grade('#### 9.00', '9', True, OLD_RULE)['raw_reward'] == 1
            and decimal_extract('#### 1\n#### 2') is None and numeric_string('1,00') is None)
    from transformers import AutoTokenizer
    checkpoint_config = load_json(repo / 'configs/base_model.json')
    checkpoint = (args.tokenizer or repo / checkpoint_config['checkpoint_directory']).resolve()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    audit['tokenizer'] = dict(path=str(checkpoint), loaded_model=False,
        files_sha256={name: sha256(checkpoint / name) for name in ['tokenizer.json', 'tokenizer_config.json',
            'special_tokens_map.json', 'vocab.json', 'merges.txt'] if (checkpoint / name).is_file()})
    prompts = []
    for row in held:
        prompt = f"{PROMPT_INSTRUCTION}\n\nQuestion:\n{row['question']}"
        chat = tokenizer.apply_chat_template([dict(role='user', content=prompt)], tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(chat, add_special_tokens=False)
        prompts.append(dict(question_id=row['id'], token_count=len(ids),
                            token_ids_sha256=hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()))
    require(audit, 'all_reconstructed_question_only_prompts_fit_1024_tokens', all(row['token_count'] <= 1024 for row in prompts))
    audit['prompt_reconstruction'] = dict(instruction=PROMPT_INSTRUCTION, min_tokens=min(row['token_count'] for row in prompts),
        max_tokens=max(row['token_count'] for row in prompts), answers=prompts,
        limitation='Prompt token IDs are reconstructed from the local tokenizer and question-only template; runtime prompt IDs were not persisted.')
    audit['dataset'] = dict(manifest_sha256=sha256(manifest_path), source_commit=DATA_REVISION,
                           validation_rows=128, prepared_train_rows=len(groups['train']), smoke_rows=len(groups['smoke']))
    old = load_json(args.old_report)
    old_gradings = audit_report(old, held, tokenizer, OLD_RULE, 256, upstream, audit, '256')
    regrade = load_json(args.old_regrade)
    require(audit, 'preserved_128_regrade_references_same_original_JSON',
            regrade['status'] == 'passed' and regrade['provenance']['original_validation_sha256'] == sha256(args.old_report)
            and regrade['provenance']['verl_source_sha256'] == VERL_SHA256)
    historical_parser = args.old_regrade.parent / 'provenance/historical_gsm8k_data.py'
    historical_config = args.old_report.parent / 'executed_protocol.json'
    require(audit, 'preserved_historical_scoring_source_matches_runtime_record',
            sha256(historical_parser) == old['source_sha256']['scripts/gsm8k_data.py']
            == regrade['provenance']['historical_parser_sha256'])
    require(audit, 'preserved_256_execution_config_snapshot_matches_runtime_record',
            sha256(historical_config) == old['config_file_sha256'])
    require(audit, 'preserved_regrade_all_128_rows_reproduced_independently',
            len(regrade['all_answers']) == 128 and all(
                all(row[key] == original[key] for key in ['question_id', 'question', 'target_answer', 'token_ids',
                                                         'completion', 'truncated', 'ended_with_eos'])
                and row['old_extraction'] == g_old['extraction'] and row['old_raw_reward'] == g_old['raw_reward']
                and row['old_evaluation_reward'] == g_old['evaluation_reward']
                and row['new_extraction'] == g_new['extraction'] and row['new_raw_reward'] == g_new['raw_reward']
                and row['new_evaluation_reward'] == g_new['evaluation_reward']
                for row, original, g_old, g_new in zip(regrade['all_answers'], old['answers'],
                    old_gradings[OLD_RULE], old_gradings[NEW_RULE])))
    return held, tokenizer, upstream, old, old_gradings


def audit_report(report, rows, tokenizer, actual_rule, cap, upstream, audit, label):
    require(audit, label + '_completed_original_base_validation_only', report['status'] == 'completed'
            and report['complete'] and report['method'] == 'base' and report['split'] == 'validation'
            and all(report.get(name) is None for name in ['adapter_checkpoint', 'adapter_checkpoint_sha256',
                                                        'adapter_tensor_file_sha256', 'initialization_source']))
    require(audit, label + '_pinned_model_and_dataset_manifest', report['model_repository'] == MODEL
            and report['model_revision'] == MODEL_REVISION
            and report['dataset_manifest_sha256'] == audit['dataset']['manifest_sha256'])
    require(audit, label + '_same_fixed_128_validation_IDs_and_order', len(report['answers']) == 128
            and report['question_ids'] == [row['id'] for row in rows]
            and [answer['question_id'] for answer in report['answers']] == report['question_ids'])
    decode = report['decode']
    require(audit, label + '_greedy_batch_one_native_cache_eager_BF16_protocol',
            decode['batch_size'] == 1 and decode['do_sample'] is False and decode['use_cache'] is True
            and decode['attention_backend'] == 'eager' and decode['repetition_penalty'] == 1.0
            and decode['max_new_tokens'] == cap and report['precision']['frozen_weights'] == 'BF16')
    require(audit, label + '_tokenizer_EOS_pad_bos_and_answer_rule', decode['eos_token_ids'] == [tokenizer.eos_token_id]
            and decode['pad_token_id'] == tokenizer.pad_token_id and decode['bos_token_id'] == tokenizer.bos_token_id
            and report['answer_rule'] == actual_rule)
    gradings = {OLD_RULE: [], NEW_RULE: []}
    for answer, row in zip(report['answers'], rows):
        name = label + '_' + row['id']
        ids = answer['token_ids']
        require(audit, name + '_question_target_and_token_validity', answer['question'] == row['question']
                and answer['target_answer'] == row['target'] and isinstance(ids, list) and bool(ids)
                and all(type(value) is int and value >= 0 for value in ids) and len(ids) <= cap)
        require(audit, name + '_saved_full_text_decodes_exactly', tokenizer.decode(ids,
                skip_special_tokens=True, clean_up_tokenization_spaces=False) == answer['completion'])
        ended = ids[-1] in decode['eos_token_ids']
        require(audit, name + '_first_EOS_or_exact_cap_and_saved_flags',
                not any(value in decode['eos_token_ids'] for value in ids[:-1]) and (ended or len(ids) == cap)
                and answer['token_count'] == len(ids) and answer['ended_with_eos'] is ended
                and answer['truncated'] is (not ended))
        for rule in gradings:
            gradings[rule].append(grade(answer['completion'], row['target'], ended, rule))
        strict = gradings[NEW_RULE][-1]
        require(audit, name + '_independent_strict_matches_pinned_upstream_source',
                strict['extraction'] == upstream['extract_solution'](answer['completion'], method='strict')
                and strict['raw_reward'] == float(upstream['compute_score'](answer['completion'], row['target'],
                                                        method='strict', format_score=0.0, score=1.0)))
        actual = gradings[actual_rule][-1]
        require(audit, name + '_recorded_grade_reproduced_independently',
                answer['predicted_answer'] == actual['extraction'] and answer['reward'] == actual['raw_reward']
                and answer['evaluation_reward'] == actual['evaluation_reward'] and answer['answer_rule'] == actual_rule)
        seconds = answer['generation_seconds']
        require(audit, name + '_valid_recorded_generation_duration', type(seconds) in (int, float)
                and math.isfinite(seconds) and seconds >= 0)
    summary = recount(report['answers'], gradings[actual_rule])
    stored_fields = ['evaluated_questions', 'requested_questions', 'correct_count', 'accuracy', 'parse_failure_count',
                     'parsed_incorrect_count', 'correct_but_truncated_count', 'truncated_answer_count', 'generated_tokens', 'complete']
    require(audit, label + '_stored_summary_reproduced_independently', all(report[name] == summary[name] for name in stored_fields))
    require(audit, label + '_mutually_exclusive_categories_cover_all_128', sum(summary['mutually_exclusive_categories'].values()) == 128)
    return gradings


def transition(values_before, values_after):
    return {f'{before} -> {after}': sum(a == before and b == after for a, b in zip(values_before, values_after))
            for before in [0, 1] for after in [0, 1]}


def make_comparison(old, new, old_grades, new_grades, audit):
    cells = {'256_decimal': recount(old['answers'], old_grades[OLD_RULE]),
             '256_verl': recount(old['answers'], old_grades[NEW_RULE]),
             '4096_decimal': recount(new['answers'], new_grades[OLD_RULE]),
             '4096_verl': recount(new['answers'], new_grades[NEW_RULE])}
    answer_rows = []
    for index, (short, long) in enumerate(zip(old['answers'], new['answers'])):
        old_ids, new_ids = short['token_ids'], long['token_ids']
        mismatch = next((i for i, (a, b) in enumerate(zip(old_ids, new_ids)) if a != b), None)
        prefix_equal = len(new_ids) >= len(old_ids) and new_ids[:len(old_ids)] == old_ids
        if mismatch is None and len(new_ids) < len(old_ids):
            mismatch = len(new_ids)
        score_cells = {name: grading[index] for name, grading in [
            ('256_decimal', old_grades[OLD_RULE]), ('256_verl', old_grades[NEW_RULE]),
            ('4096_decimal', new_grades[OLD_RULE]), ('4096_verl', new_grades[NEW_RULE])]}
        correctness_changed = len({g['evaluation_reward'] for g in score_cells.values()}) > 1
        raw_correctness_changed = len({g['raw_reward'] for g in score_cells.values()}) > 1
        truncation_changed = short['truncated'] != long['truncated']
        answer_rows.append(dict(question_id=short['question_id'], question=short['question'], target_answer=short['target_answer'],
            old_answer=short, new_answer=long, grading=score_cells,
            prefix_equal=prefix_equal, first_mismatch_token_index=mismatch,
            first_mismatch_old_token=None if mismatch is None or mismatch >= len(old_ids) else old_ids[mismatch],
            first_mismatch_new_token=None if mismatch is None or mismatch >= len(new_ids) else new_ids[mismatch],
            old_completed_full_tokens_equal=old_ids == new_ids if short['ended_with_eos'] else None,
            correctness_changed=correctness_changed, raw_correctness_changed=raw_correctness_changed,
            truncation_changed=truncation_changed,
            old_truncated_new_completed=short['truncated'] and long['ended_with_eos'],
            numeric_representation_rejection=any(g['raw_reward'] == 1
                and score_cells[name.replace('decimal', 'verl')]['raw_reward'] == 0
                and numeric_string(score_cells[name.replace('decimal', 'verl')]['extraction']) == numeric_string(short['target_answer'])
                for name, g in score_cells.items() if name.endswith('decimal'))))
    differences = [row for row in answer_rows if row['correctness_changed'] or row['raw_correctness_changed'] or row['truncation_changed'] or not row['prefix_equal']]
    final_rewards = lambda key: [row['grading'][key]['evaluation_reward'] for row in answer_rows]
    summary = dict(cells=cells, answer_count=128,
        delta_rule_on_256=cells['256_verl']['correct_count'] - cells['256_decimal']['correct_count'],
        delta_generation_cap_at_verl=cells['4096_verl']['correct_count'] - cells['256_verl']['correct_count'],
        delta_generation_cap_at_decimal=cells['4096_decimal']['correct_count'] - cells['256_decimal']['correct_count'],
        delta_rule_on_4096=cells['4096_verl']['correct_count'] - cells['4096_decimal']['correct_count'],
        delta_historical_to_final=cells['4096_verl']['correct_count'] - cells['256_decimal']['correct_count'],
        prefix_equal_count=sum(row['prefix_equal'] for row in answer_rows),
        prefix_mismatch_count=sum(not row['prefix_equal'] for row in answer_rows),
        old_completed_full_tokens_equal_count=sum(row['old_completed_full_tokens_equal'] is True for row in answer_rows),
        old_completed_full_tokens_differ_count=sum(row['old_completed_full_tokens_equal'] is False for row in answer_rows),
        old_truncated_new_completed_count=sum(row['old_truncated_new_completed'] for row in answer_rows),
        correctness_changed_count=sum(row['correctness_changed'] for row in answer_rows),
        raw_correctness_changed_count=sum(row['raw_correctness_changed'] for row in answer_rows),
        truncation_changed_count=sum(row['truncation_changed'] for row in answer_rows),
        reported_changed_answers_count=len(differences),
        remaining_truncated_question_ids=[row['question_id'] for row in answer_rows if row['new_answer']['truncated']],
        strict_length_transition=transition(final_rewards('256_verl'), final_rewards('4096_verl')),
        historical_to_final_transition=transition(final_rewards('256_decimal'), final_rewards('4096_verl')),
        strict_rule_transition_4096=transition(final_rewards('4096_decimal'), final_rewards('4096_verl')),
        completed_after_old_truncation_categories=dict(Counter(
            mutually_exclusive_category(row['new_answer'], row['grading']['4096_verl'])
            for row in answer_rows if row['old_truncated_new_completed'])),
    )
    audit['comparison_summary'] = summary
    audit['prefix_check'] = dict(all_128_prefixes_equal=summary['prefix_equal_count'] == 128,
        all_previously_completed_sequences_identical=summary['old_completed_full_tokens_differ_count'] == 0,
        mismatches=[{key: row[key] for key in ['question_id', 'first_mismatch_token_index', 'first_mismatch_old_token',
                                             'first_mismatch_new_token']} for row in answer_rows if not row['prefix_equal']],
        interpretation='A larger generation cap changes available continuation after token 256. Prefix equality is measured, not assumed.')
    return dict(status='passed', created_utc=audit['created_utc'], summary=summary,
                changed_answers=differences, all_answers=answer_rows)


def render(comparison, audit, output):
    summary = comparison['summary']
    cells = summary['cells']
    recovered = summary['completed_after_old_truncation_categories']
    lines = ['# GSM8K 128 题：4096-token 基线与独立复核', '',
             '复核在本地 CPU 上读取已经生成的 JSON，仅加载 tokenizer；未重新生成回答或执行训练。', '',
             '## 同一份回答分别应用两种规则', '',
             '| 生成上限 | 历史 Decimal 规则 | 固定 verl strict | 截断 |', '|---|---:|---:|---:|',
             f"| 256 token | {cells['256_decimal']['correct_count']}/128 ({cells['256_decimal']['accuracy']:.4%}) | {cells['256_verl']['correct_count']}/128 ({cells['256_verl']['accuracy']:.4%}) | {cells['256_verl']['truncated_answer_count']} |",
             f"| 4096 token | {cells['4096_decimal']['correct_count']}/128 ({cells['4096_decimal']['accuracy']:.4%}) | {cells['4096_verl']['correct_count']}/128 ({cells['4096_verl']['accuracy']:.4%}) | {cells['4096_verl']['truncated_answer_count']} |", '',
             'verl strict 只检查末尾 300 字符，取最后一次字面模式 `#### (-?[0-9.,]+)` 的匹配；删除逗号后直接比较字符串。',
             '历史规则要求唯一 `####`，其后全部内容必须为合法数字，以 Decimal 比较；两者均将未以 EOS 结束的回答最终奖励设为 0。', '',
             f"- 在旧 256-token 文本上更换规则：{summary['delta_rule_on_256']:+d} 题。",
             f"- 固定 verl strict，比较 256 → 4096 token：{summary['delta_generation_cap_at_verl']:+d} 题。",
             f"- 在新 4096-token 文本上更换规则：{summary['delta_rule_on_4096']:+d} 题。",
             f"- 从旧历史分数到新基线总变化：{summary['delta_historical_to_final']:+d} 题。", '',
             '## token、EOS 与数据隔离', '',
             f"全部 {summary['prefix_equal_count']}/128 份新回答保留旧 token 序列前缀；前缀不一致 {summary['prefix_mismatch_count']} 份。",
             f"旧已完成回答中，完整 token 序列相同 {summary['old_completed_full_tokens_equal_count']} 份，不同 {summary['old_completed_full_tokens_differ_count']} 份。",
             f"旧截断 → 新完成：{summary['old_truncated_new_completed_count']} 份；其中 {recovered.get('completed_correct', 0)} 份判对、{recovered.get('completed_extracted_incorrect', 0)} 份提取成功但判错、{recovered.get('completed_without_extraction', 0)} 份未提取到答案。",
             '已逐题核对 token_ids 解码得到完整文本、首个 EOS 停止、截断时恰好达到上限、报告奖励及汇总。',
             '固定 128 题顺序由原始 GSM8K train 文件和 SHA256 排序独立重建，并与准备训练集、smoke 集的 ID 与问题文本验证互斥。',
             f"用本地 tokenizer 重建的最长问题提示为 {audit['prompt_reconstruction']['max_tokens']} token，低于 1024 上限。运行时没有保存 prompt_ids，提示核对是本地重建证据。", '',
             '## 新基线：互斥结果分类', '', '| 分类 | 数量 |', '|---|---:|']
    labels = {'completed_correct': 'EOS 完成且判对', 'completed_extracted_incorrect': 'EOS 完成、成功提取但判错',
              'completed_without_extraction': 'EOS 完成、未提取到答案', 'truncated': '达到上限、未以 EOS 完成'}
    for name, count in cells['4096_verl']['mutually_exclusive_categories'].items():
        lines.append(f'| {labels[name]} | {count} |')
    runtime = audit['runtime_record']
    lines += ['', '四类互斥，合计 128 题。解析失败总数等指标另含截断回答，不能与截断数直接相加。', '',
              '## 运行记录与源码证据', '',
              f"- 新 GPU job：`{runtime.get('slurm_job_id')}`；GPU：`{runtime.get('gpu_name')}`；报告总时长：{runtime['elapsed_seconds']:.3f} 秒。",
              f"- 报告生成 token：{cells['4096_verl']['generated_tokens']}；峰值已分配显存：{runtime['peak_allocated_memory_gib']:.3f} GiB。",
              f"- 运行报告记录的 Git HEAD：`{runtime.get('source_git_commit')}`；执行文件字节另由逐文件 SHA256 记录。",
              f"- 固定 verl commit：`{VERL_COMMIT}`；其源码 SHA256：`{VERL_SHA256}`。",
              '- 源码哈希逐项保存在 independent_audit.json；当前工作区文件与运行时记录的比较只反映这些字节是否相等。',
              '- 本地工作区存在后续修改时，不据当前文件推断执行时全部源码一致；运行报告中的版本与哈希属于运行时记录，不能单独构成进程执行证明。',
              '- 这是固定验证子集基线，不是最终 GSM8K test 分数，也不是训练改善或 GeoRA 论文表格复现。', '']
    source_snapshot = audit.get('executed_source_snapshot')
    if source_snapshot is not None:
        lines += [f"远端保存的 {len(source_snapshot['computed_source_sha256'])} 份源码快照逐文件字节哈希均与运行报告记录相同。快照 SHA256：`{source_snapshot['sha256']}`。",
                  '当前本地工作区的哈希比较保存在 JSON 中；后续训练实现修改不改变这份执行源码快照。', '']
    lines += ['## 所有正确性或截断状态变化的题目，以及剩余截断：完整回答', '']
    articles = []
    # Show the representation case first, then recovered truncations, then other changes.
    changed = sorted(comparison['changed_answers'], key=lambda row: (
        not row['numeric_representation_rejection'], not row['old_truncated_new_completed'], row['question_id']))
    remaining = [row for row in comparison['all_answers'] if row['new_answer']['truncated']
                 and row['question_id'] not in {item['question_id'] for item in changed}]
    reported = changed + remaining
    lines += [f"变化题目 {len(changed)} 份；另补充仍截断的 {len(remaining)} 份回答全文。", '']
    for row in reported:
        qid = row['question_id']
        short, long = row['old_answer'], row['new_answer']
        grading = row['grading']
        tags = []
        if row['old_truncated_new_completed']:
            tags.append('旧截断 → 新完成')
        if row['numeric_representation_rejection']:
            tags.append('Decimal 判对 / 字符串判错')
        if long['truncated']:
            tags.append('仍达到 4096-token 上限')
        if not row['prefix_equal']:
            tags.append('token 前缀不同')
        title = qid + (' — ' + '；'.join(tags) if tags else '')
        details = (f"目标：{row['target_answer']}；token：{short['token_count']} → {long['token_count']}；"
                   f"EOS：{short['ended_with_eos']} → {long['ended_with_eos']}；截断：{short['truncated']} → {long['truncated']}；"
                   f"旧序列前缀相等：{row['prefix_equal']}。")
        lines += ['### ' + title, '', details, '', '| 上限 / 规则 | 提取 | 原始奖励 | EOS 门槛后奖励 |', '|---|---|---:|---:|']
        table = ['<table><tr><th>上限 / 规则</th><th>提取</th><th>原始奖励</th><th>EOS 门槛后奖励</th></tr>']
        display_names = {'256_decimal': '256 / 历史 Decimal', '256_verl': '256 / verl strict',
                         '4096_decimal': '4096 / 历史 Decimal', '4096_verl': '4096 / verl strict'}
        for name, value in grading.items():
            lines.append(f"| {display_names[name]} | `{value['extraction']}` | {value['raw_reward']} | {value['evaluation_reward']} |")
            table.append(f"<tr><td>{display_names[name]}</td><td><code>{html.escape(str(value['extraction']))}</code></td><td>{value['raw_reward']}</td><td>{value['evaluation_reward']}</td></tr>")
        table.append('</table>')
        lines += ['', '**问题**', '', row['question'], '', '**256-token 完整回答**', '',
                  '```text', short['completion'], '```', '', '**4096-token 完整回答**', '', '```text', long['completion'], '```', '']
        articles.append('<article id="' + qid.replace(':', '-') + '"><h2>' + html.escape(title) + '</h2><p>'
                        + html.escape(details) + '</p>' + '\n'.join(table) + '<h3>问题</h3><pre>'
                        + html.escape(row['question']) + '</pre><details><summary>256-token 完整回答</summary><pre>'
                        + html.escape(short['completion']) + '</pre></details><details open><summary>4096-token 完整回答</summary><pre>'
                        + html.escape(long['completion']) + '</pre></details></article>')
    if not changed:
        lines.append('没有正确性或截断状态变化。')
    (output / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    nav = ''.join('<a href="#' + row['question_id'].replace(':', '-') + '">' + row['question_id'] + '</a> ' for row in reported)
    page = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>GSM8K 4096-token 基线独立复核</title>',
            '<style>body{font-family:system-ui;max-width:1100px;margin:32px auto;padding:0 20px;line-height:1.65}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f7f9;padding:18px;border-radius:8px}table{border-collapse:collapse}th,td{border:1px solid #ccc;padding:8px}article{border-top:1px solid #ccc;margin-top:30px}summary{cursor:pointer;font-weight:600}nav a{display:inline-block;margin-right:12px}code{background:#eef1f5}a{color:#2258a8}</style>',
            '<h1>GSM8K 128 题：4096-token 基线独立复核</h1>',
            f"<p>固定 verl strict：{cells['256_verl']['correct_count']}/128 → <b>{cells['4096_verl']['correct_count']}/128 ({cells['4096_verl']['accuracy']:.4%})</b>；截断 {cells['256_verl']['truncated_answer_count']} → {cells['4096_verl']['truncated_answer_count']}。</p>",
            '<table><tr><th>生成上限</th><th>历史 Decimal</th><th>verl strict</th></tr>'
            + f"<tr><td>256</td><td>{cells['256_decimal']['correct_count']}/128</td><td>{cells['256_verl']['correct_count']}/128</td></tr>"
            + f"<tr><td>4096</td><td>{cells['4096_decimal']['correct_count']}/128</td><td>{cells['4096_verl']['correct_count']}/128</td></tr></table>",
            '<p>固定 verl strict 取最后 300 字符中最后一个 <code>#### 数字</code>，去逗号后比较字符串；<code>9.00</code> 与 <code>9</code> 不相等。截断回答最终奖励为 0。</p>',
            f"<p>逐题全文 token 解码、EOS、独立评分与汇总均通过。相同旧 token 前缀 {summary['prefix_equal_count']}/128；旧截断 → 新完成 {summary['old_truncated_new_completed_count']} 份。</p>",
            '<p>互斥分类：' + '；'.join(f"{labels[name]} {count}" for name, count in cells['4096_verl']['mutually_exclusive_categories'].items()) + '。详细证据与限制见 <a href="comparison.md">Markdown 报告</a> 与 <a href="independent_audit.json">复核 JSON</a>。</p>',
            f"<p>GPU job {runtime.get('slurm_job_id')}，报告运行时长 {runtime['elapsed_seconds']:.3f} 秒。复核为 CPU，仅加载 tokenizer。</p>",
            '<p>运行版本与源码哈希按原报告记录；当前工作区可能已修改，不能据当前文件推断执行时全部源码一致。本次是固定验证子集基线。</p>',
            f"<p>变化题目 {len(changed)} 份；另补充仍截断的 {len(remaining)} 份完整回答。仍截断 ID：{html.escape(', '.join(summary['remaining_truncated_question_ids']))}。</p>",
            '<h2>所有正确性或截断变化，以及剩余截断：完整回答</h2><nav>' + nav + '</nav>', *articles, '</html>']
    (output / 'comparison.html').write_text('\n'.join(page), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path('/Users/tianbaochen/geora-repro'))
    parser.add_argument('--new-report', type=Path, default=Path('/private/tmp/geora-base4096/validation.json'))
    parser.add_argument('--old-report', type=Path)
    parser.add_argument('--old-regrade', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--tokenizer', type=Path)
    parser.add_argument('--source-snapshot', type=Path, help='Optional remote UTF-8 source snapshot with executed_sources, observed_hashes, and config fields.')
    parser.add_argument('--output', type=Path, default=Path('/private/tmp/geora-base4096'))
    parser.add_argument('--expected-job-id', default='50697776')
    parser.add_argument('--preflight', action='store_true', help='Check only prerequisites and preserved 256-token result; never inspect or claim new result.')
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.old_report = args.old_report or args.repo / 'reports/pilot_base_validation/validation.json'
    args.old_regrade = args.old_regrade or args.repo / 'reports/regrade_128/comparison.json'
    args.config = args.config or args.repo / 'configs/grpo_long_pilot.json'
    default_source_snapshot = args.new_report.parent / 'source_snapshot.json'
    if args.source_snapshot is None and default_source_snapshot.is_file():
        args.source_snapshot = default_source_snapshot
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    audit = dict(status='running', created_utc=datetime.now(timezone.utc).isoformat(), checks=[],
        scope='CPU-only saved-generation audit. No inference, accelerator, SSH, training, or repository edits.',
        audit_script_sha256=sha256(Path(__file__)),
        source_provenance_limit='Runtime source hashes and commit are reported metadata. Current checkout matches are measured comparisons, not independent process-execution attestation.')
    audit_path = output / ('preflight.json' if args.preflight else 'independent_audit.json')
    try:
        rows, tokenizer, upstream, old, old_grades = preflight(args, audit)
        audit['inputs'] = {name: dict(path=str(path.resolve()), sha256=sha256(path)) for name, path in [
            ('validation256', args.old_report), ('regrade128', args.old_regrade)]}
        audit['old_256_summary'] = {rule: recount(old['answers'], grading) for rule, grading in old_grades.items()}
        if args.preflight:
            audit.update(status='passed', new_4096_result_audited=False)
        else:
            new = load_json(args.new_report)
            audit['inputs']['validation4096'] = dict(path=str(args.new_report.resolve()), sha256=sha256(args.new_report))
            audit['runtime_record'] = {key: new.get(key) for key in ['slurm_job_id', 'source_git_commit', 'source_sha256',
                'config_file_sha256', 'training_config_sha256', 'model_repository', 'model_revision', 'decode', 'precision',
                'torch_version', 'transformers_version', 'hip_version', 'gpu_name', 'elapsed_seconds',
                'peak_allocated_memory_gib', 'segments']}
            require(audit, 'expected_new_GPU_job_identity', str(new['slurm_job_id']) == args.expected_job_id)
            require(audit, 'new_runtime_record_pins_verl_source_sha256', new['source_sha256'].get('third_party/verl/gsm8k.py') == VERL_SHA256)
            config = load_json(args.config)
            require(audit, 'execution_config_snapshot_sha256_matches_runtime_record', sha256(args.config) == new['config_file_sha256'])
            require(audit, 'execution_config_4096_output_1024_prompt_and_fixed_verl_rule', config['max_new_tokens'] == 4096
                    and config['max_prompt_tokens'] == 1024 and config['prompt_instruction'] == PROMPT_INSTRUCTION
                    and config['answer_rule'] == NEW_RULE and config['validation_question_limit'] == 128)
            audit['inputs']['executed_config'] = dict(path=str(args.config.resolve()), sha256=sha256(args.config))
            new_grades = audit_report(new, rows, tokenizer, NEW_RULE, 4096, upstream, audit, '4096')
            audit['source_hash_comparison'] = source_comparison(new, args.repo)
            audit['current_checkout_git_commit'] = subprocess.check_output(['git', '-C', str(args.repo), 'rev-parse', 'HEAD'], text=True).strip()
            if args.source_snapshot is not None:
                snapshot = load_json(args.source_snapshot)
                require(audit, 'remote_source_snapshot_git_HEAD_and_config_match_runtime_record',
                        snapshot['source_git_commit'] == new['source_git_commit'] and snapshot['config'] == config)
                computed = {name: hashlib.sha256(text.encode('utf-8')).hexdigest()
                            for name, text in snapshot['executed_sources'].items()}
                require(audit, 'remote_source_snapshot_all_recorded_UTF8_bytes_match_runtime_SHA256',
                        all(computed.get(name) == recorded == snapshot['observed_hashes'].get(name)
                            for name, recorded in new['source_sha256'].items()))
                audit['executed_source_snapshot'] = dict(path=str(args.source_snapshot.resolve()),
                    sha256=sha256(args.source_snapshot), source_git_commit=snapshot['source_git_commit'],
                    computed_source_sha256=computed, all_recorded_source_bytes_match=True,
                    limitation='Snapshot bytes match runtime-recorded hashes; this independently checks bytes, not process execution.')
            comparison = make_comparison(old, new, old_grades, new_grades, audit)
            audit.update(status='passed', new_4096_result_audited=True, passed_check_count=len(audit['checks']))
            proof = output / 'provenance'
            proof.mkdir(exist_ok=True)
            copies = [('validation256.json', args.old_report), ('regrade128.json', args.old_regrade),
                      ('historical_gsm8k_data.py', args.old_regrade.parent / 'provenance/historical_gsm8k_data.py'),
                      ('executed_protocol256.json', args.old_report.parent / 'executed_protocol.json'),
                      ('verl_LICENSE', args.old_regrade.parent / 'provenance/verl_LICENSE'),
                      ('executed_protocol4096.json', args.config), ('manifest.json', args.repo / 'datasets/gsm8k/manifest.json'),
                      ('verl_gsm8k.py', args.repo / 'third_party/verl/gsm8k.py'),
                      ('verl_provenance.json', args.repo / 'third_party/verl/provenance.json')]
            if args.source_snapshot is not None:
                copies.append(('source_snapshot.json', args.source_snapshot))
            for name, source in copies:
                if source.resolve() != (proof / name).resolve():
                    shutil.copyfile(source, proof / name)
            audit['preserved_provenance_files'] = {name: sha256(proof / name) for name, _ in copies}
            comparison['provenance'] = dict(validation256_sha256=sha256(args.old_report), validation4096_sha256=sha256(args.new_report),
                verl_commit=VERL_COMMIT, verl_source_sha256=VERL_SHA256, audit_script_sha256=audit['audit_script_sha256'])
            write_json(output / 'comparison.json', comparison)
            render(comparison, audit, output)
    except Exception as error:
        audit.update(status='failed', error=str(error), error_type=type(error).__name__)
        raise
    finally:
        write_json(audit_path, audit)
    print(json.dumps(dict(status=audit['status'], preflight=args.preflight, audit=str(audit_path),
                         summary=audit.get('comparison_summary', audit.get('old_256_summary'))), ensure_ascii=False))


if __name__ == '__main__':
    main()
