"""Independently audit a saved base-model validation report; no inference.

Uses a separate numeric-format parser and summary implementation. Only the
tokenizer is instantiated, on CPU, to check saved token IDs against full text.
"""

from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path


REPO = Path('/Users/tianbaochen/geora-repro')
ROOT = Path('/private/tmp/geora-base-validation')
REPORT_PATH = ROOT / 'validation.json'
AUDIT_PATH = ROOT / 'independent_audit.json'
CHECKS = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(name, condition, **evidence):
    CHECKS.append(dict(name=name, passed=bool(condition), **evidence))
    if not condition:
        raise AssertionError(name)


def numeric_string(text):
    """Parse strict signed decimal/thousands grammar without the runtime regex."""
    if not text:
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
        if not (1 <= len(groups[0]) <= 3) or any(len(g) != 3 for g in groups[1:]):
            return None
        if any(not g or not set(g) <= digits for g in groups):
            return None
    elif integer:
        if not set(integer) <= digits:
            return None
    elif not point:
        return None
    return Decimal(text.replace(',', ''))


def final_number(text):
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


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    audit = dict(
        status='running', created_utc=datetime.now(timezone.utc).isoformat(),
        scope='CPU-only independent saved-result audit; no model forward, accelerator allocation, or repository edits.',
        parser='Independent ASCII numeric grammar with exact Decimal equality; does not call evaluate_gsm8k_pilot or gsm8k_data scoring/summary helpers.',
        checks=CHECKS,
    )
    try:
        parser_examples = {
            '1': Decimal(1), '+1,000.00': Decimal(1000), '-.5': Decimal('-.5'),
            '.5': Decimal('.5'), '01': Decimal(1), '1.': None,
            '1,00': None, '1e3': None, '1/2': None, 'NaN': None,
            '１２': None, '1 dollars': None, '': None,
        }
        require('independent_numeric_parser_edge_cases', all(numeric_string(s) == v for s, v in parser_examples.items()), cases=len(parser_examples))
        dataset = REPO / 'datasets/gsm8k'
        manifest_path = dataset / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        audit['dataset_manifest_sha256'] = sha(manifest_path)
        require('pinned_dataset_revision', manifest['source_commit'] == '3101c7d5072418e28b9008a6636bde82a006892c')
        require('pinned_original_train_sha256', sha(dataset / 'raw/train.jsonl') == '17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465')
        validation = lines(dataset / 'validation.jsonl')
        train = lines(dataset / 'train.jsonl')
        smoke = lines(dataset / 'smoke.jsonl')
        for name, rows in [('validation', validation), ('train', train), ('smoke', smoke)]:
            require(name + '_persisted_file_hash_and_count', sha(dataset / (name + '.jsonl')) == manifest['splits'][name]['sha256'] and len(rows) == manifest['splits'][name]['rows'])
        original = lines(dataset / 'raw/train.jsonl')
        independent_rows = []
        for index, row in enumerate(original):
            target = final_number(row['answer'])
            require('source_target_' + str(index), target is not None) if target is None else None
            independent_rows.append(dict(id=f'train:{index:05d}', source_split='train', source_index=index, question=row['question'], target=canonical(target)))
        ordered = sorted(independent_rows, key=lambda row: hashlib.sha256(f"{manifest['split_seed']}:{row['id']}".encode()).hexdigest())
        require('exact_128_validation_order_reconstructed_from_original_source', len(validation) == 128 and validation == ordered[:128] and manifest['validation_ids'] == [r['id'] for r in validation])
        require('exact_prepared_train_and_smoke_reconstructed', train == ordered[128:] and smoke == ordered[128:144])
        require('validation_held_out_from_prepared_training_and_smoke', not ({r['id'] for r in validation} & {r['id'] for r in train}) and not ({r['question'] for r in validation} & {r['question'] for r in train}) and not ({r['id'] for r in validation} & {r['id'] for r in smoke}))
        report = json.loads(REPORT_PATH.read_text())
        audit['input'] = dict(path=str(REPORT_PATH), sha256=sha(REPORT_PATH))
        audit['model'] = {k: report.get(k) for k in ['model_repository', 'model_revision', 'method', 'precision', 'decode', 'source_git_commit', 'slurm_job_id', 'gpu_name', 'hip_version']}
        require('completed_original_base_validation_only', report['status'] == 'completed' and report['method'] == 'base' and report['split'] == 'validation' and report['adapter_checkpoint'] is None and report['adapter_checkpoint_sha256'] is None and report['initialization_source'] is None)
        require('pinned_model_and_manifest_match', report['model_repository'] == 'Qwen/Qwen2.5-1.5B-Instruct' and report['model_revision'] == '989aa7980e4cf806f80c7fef2b1adb7bc71aa306' and report['dataset_manifest_sha256'] == sha(manifest_path))
        require('same_fixed_validation_ids_and_order', report['question_ids'] == [r['id'] for r in validation] and [a['question_id'] for a in report['answers']] == report['question_ids'])
        decode = report['decode']
        require('greedy_batch_one_native_cache_protocol', decode['batch_size'] == 1 and decode['do_sample'] is False and decode['use_cache'] is True and decode['attention_backend'] == 'eager' and decode['repetition_penalty'] == 1.0)
        require('BF16_original_model_precision', report['precision']['frozen_weights'] == 'BF16')
        require('strict_answer_rule_recorded', report['answer_rule'] == 'single_final_hash_marker_exact_decimal_v1')
        max_tokens = decode['max_new_tokens']
        require('valid_token_cap', type(max_tokens) is int and max_tokens > 0)
        from transformers import AutoTokenizer
        checkpoint_config = json.loads((REPO / 'configs/base_model.json').read_text())
        tokenizer = AutoTokenizer.from_pretrained(REPO / checkpoint_config['checkpoint_directory'], local_files_only=True)
        eos_ids = decode['eos_token_ids']
        require('tokenizer_and_report_EOS_agree', eos_ids == [tokenizer.eos_token_id])
        counts = Counter()
        answer_evidence = []
        for answer, row in zip(report['answers'], validation):
            ids = answer['token_ids']
            require(row['id'] + '_tokens_and_prompt_identity', answer['question_id'] == row['id'] and answer['question'] == row['question'] and isinstance(ids, list) and ids and all(type(i) is int and i >= 0 for i in ids) and len(ids) <= max_tokens)
            require(row['id'] + '_full_text_decode_exact', tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) == answer['completion'])
            require(row['id'] + '_first_EOS_or_exact_token_cap', not any(i in eos_ids for i in ids[:-1]) and (ids[-1] in eos_ids or len(ids) == max_tokens))
            ended = ids[-1] in eos_ids
            predicted = final_number(answer['completion'])
            target = numeric_string(row['target'])
            reward = float(predicted is not None and predicted == target)
            evaluation_reward = reward if ended else 0.0
            expected = dict(token_count=len(ids), ended_with_eos=ended, truncated=not ended, predicted_answer=canonical(predicted), target_answer=canonical(target), reward=reward, evaluation_reward=evaluation_reward, answer_rule='single_final_hash_marker_exact_decimal_v1')
            require(row['id'] + '_strict_grading_and_truncation_match', all(answer.get(k) == v for k, v in expected.items()))
            seconds = answer.get('generation_seconds')
            require(row['id'] + '_valid_duration', type(seconds) in (float, int) and math.isfinite(seconds) and seconds >= 0)
            counts['questions'] += 1
            counts['correct'] += int(evaluation_reward)
            counts['parse_failure'] += int(predicted is None)
            counts['parsed_incorrect'] += int(predicted is not None and reward == 0)
            counts['correct_but_truncated'] += int(reward == 1 and not ended)
            counts['truncated'] += int(not ended)
            counts['eos'] += int(ended)
            counts['tokens'] += len(ids)
            answer_evidence.append(dict(question_id=row['id'], independent_predicted_answer=canonical(predicted), target_answer=row['target'], independent_reward=reward, independent_evaluation_reward=evaluation_reward, ended_with_eos=ended, token_count=len(ids)))
        summary = dict(evaluated_questions=counts['questions'], requested_questions=128, correct_count=counts['correct'], accuracy=counts['correct'] / counts['questions'], parse_failure_count=counts['parse_failure'], parsed_incorrect_count=counts['parsed_incorrect'], correct_but_truncated_count=counts['correct_but_truncated'], truncated_answer_count=counts['truncated'], generated_tokens=counts['tokens'], complete=counts['questions'] == 128)
        require('all_summary_fields_match_independent_recount', all(report.get(k) == v for k, v in summary.items()), independent_summary=summary)
        require('all_128_questions_evaluated', counts['questions'] == 128)
        require('mutually_exclusive_scoring_categories_cover_answers', counts['correct'] + counts['parse_failure'] + counts['parsed_incorrect'] + counts['correct_but_truncated'] == 128)
        audit.update(status='passed', independent_summary=summary, eos_answer_count=counts['eos'], answers=answer_evidence)
    except Exception as error:
        audit.update(status='failed', error=str(error))
        raise
    finally:
        AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(AUDIT_PATH, audit['status'], audit['independent_summary'])


if __name__ == '__main__':
    main()
