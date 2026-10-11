#!/usr/bin/env python3
"""Regrade preserved GSM8K generations without loading a model or using a GPU.

Inputs are a validation report, the historical parser file, and the vendored
verbatim verl source. Outputs preserve original generations, source provenance,
all 128 verdicts, changed examples, and independently checked source semantics.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import shutil

VERL_COMMIT = '1d367a4f459a43768f76312c0d3430a99ec46b8c'
VERL_SOURCE_SHA256 = '497c6e32b708d0bc8803b5a048ca5983de7756eb9c146abdc56e6f5d28767189'
VERL_LICENSE_SHA256 = 'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30'
NEW_RULE = 'verl_gsm8k_strict_1d367a4f_v1'
VERL_RELATIVE_SOURCE = 'verl/utils/reward_score/gsm8k.py'
VERL_STRICT_PATTERN = re.compile(r'#### (\-?[0-9\.\,]+)')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source_namespace(path):
    # Postponed annotations let the historical parser run on Python 3.9 as well.
    source = 'from __future__ import annotations\n' + path.read_text()
    namespace = {'__name__': 'preserved_scoring_source', '__file__': str(path)}
    exec(compile(source, str(path), 'exec'), namespace)
    return namespace


def independent_verl_extract(text):
    # Same specified semantics; an independent finditer implementation.
    tail = text[-300:]
    last_match = None
    for match in VERL_STRICT_PATTERN.finditer(tail):
        last_match = match.group(1)
    if last_match is None:
        return None
    return last_match.replace(',', '').replace('$', '')


def independent_verl_score(text, target):
    answer = independent_verl_extract(text)
    return float(answer is not None and answer == target)


def verdict_counts(rows, prefix):
    extracted = sum(row[prefix + '_extraction'] is not None for row in rows)
    raw_correct = sum(row[prefix + '_raw_reward'] == 1 for row in rows)
    evaluation_correct = sum(row[prefix + '_evaluation_reward'] == 1 for row in rows)
    return {
        'extracted_count': extracted,
        'parse_failure_count': len(rows) - extracted,
        'raw_correct_count': raw_correct,
        'evaluation_correct_count': evaluation_correct,
        'evaluation_accuracy': evaluation_correct / len(rows),
        'raw_correct_but_truncated_count': sum(
            row[prefix + '_raw_reward'] == 1 and row['truncated'] for row in rows
        ),
        'parsed_incorrect_count': sum(
            row[prefix + '_extraction'] is not None and row[prefix + '_raw_reward'] == 0
            for row in rows
        ),
        'nontruncated_parse_failure_count': sum(
            row[prefix + '_extraction'] is None and not row['truncated'] for row in rows
        ),
    }


def write_markdown(report, destination):
    summary = report['summary']
    changed = report['changed_answers']
    lines = [
        '# 已有 128 份 GSM8K 回答：新旧判分比较', '',
        '这是 CPU 对已有文本的重新判定。没有重新生成回答，没有训练或 GPU 运算。', '',
        '## 判定规则', '',
        '- 旧规则：唯一 `####`，其后全部内容必须是一个数；使用 Decimal 数值相等。',
        '- 新规则：固定版本 verl strict，只检查末尾 300 字符，取最后一个 `#### 数字` 匹配；去逗号后按字符串相等。',
        '- 两者都保留原评估门槛：未以 EOS 结束的截断回答，最终评估奖励为 0。',
        '- 提取和奖励都直接执行保存的官方源码，并与独立实现核对。', '',
        f"verl commit: `{VERL_COMMIT}`", '',
        '## 汇总', '',
        '| 指标 | 旧规则 | 新规则 |', '|---|---:|---:|',
    ]
    labels = [
        ('evaluation_correct_count', '最终判对'),
        ('extracted_count', '成功提取数值'),
        ('parse_failure_count', '无法提取'),
        ('parsed_incorrect_count', '提取成功但数值不对'),
        ('nontruncated_parse_failure_count', '未截断但无法提取'),
        ('raw_correct_but_truncated_count', '原始奖励正确但被截断门槛排除'),
    ]
    for key, label in labels:
        lines.append(f"| {label} | {summary['old'][key]} | {summary['new'][key]} |")
    lines += [
        f"| 最终正确率 | {summary['old']['evaluation_accuracy']:.4%} | {summary['new']['evaluation_accuracy']:.4%} |", '',
        f"截断回答：{summary['truncated_count']} / {summary['answer_count']}。", '',
        f"提取结果变化：{summary['extraction_changed_count']} 份；原始奖励变化：{summary['raw_reward_changed_count']} 份；最终评估奖励变化：{summary['evaluation_reward_changed_count']} 份。", '',
        '## 最终奖励转移', '',
        '| 旧奖励 → 新奖励 | 数量 |', '|---|---:|',
    ]
    for transition, count in summary['evaluation_transition_matrix'].items():
        lines.append(f'| {transition} | {count} |')
    lines += ['', '## 每一份提取或奖励改变的回答', '']
    for row in changed:
        lines += [
            f"### {row['question_id']}", '',
            f"标准答案：`{row['target_answer']}`；截断：{row['truncated']}；EOS：{row['ended_with_eos']}。", '',
            f"旧提取：`{row['old_extraction']}`；新提取：`{row['new_extraction']}`。", '',
            f"原始奖励：{row['old_raw_reward']} → {row['new_raw_reward']}；最终评估奖励：{row['old_evaluation_reward']} → {row['new_evaluation_reward']}。", '',
            '**问题**', '', row['question'], '', '**完整原回答**', '',
            '```text', row['completion'], '```', '',
        ]
    lines += [
        '## 证据与限制', '',
        '- 源 JSON、原始 token_ids、完整回答、旧代码、verl 源码及许可证均已复制保存。',
        '- 这里只测判分规则变化，不说明模型能力提高。',
        '- 新规则不接受没有 `####` 的正确自然语言答案；`#### 2.0` 与标签 `2` 也不会得到相同奖励。',
        '- 旧回答的生成上限仍是 256 token；4096 token 基线需另行运行。',
        f"- verl 源码 SHA256：`{report['provenance']['verl_source_sha256']}`。", '',
    ]
    destination.write_text('\n'.join(lines), encoding='utf-8')


def write_html(report, destination):
    s = report['summary']
    def esc(value):
        return html.escape(str(value))
    sections = [
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
        '<title>GSM8K 128 份回答判分比较</title>',
        '<style>body{font-family:system-ui;max-width:1000px;margin:30px auto;padding:0 20px;line-height:1.6}pre{white-space:pre-wrap;background:#f5f6f8;padding:16px;border-radius:8px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px}article{border-top:1px solid #ddd;margin-top:30px}code{background:#f3f3f3}</style>',
        '<h1>已有 128 份 GSM8K 回答：新旧判分比较</h1>',
        '<p>CPU 重新判分，未重新生成或训练，未使用 GPU。截断回答最终奖励仍为 0。</p>',
        f"<p>判对：{s['old']['evaluation_correct_count']} → {s['new']['evaluation_correct_count']}；正确率：{s['old']['evaluation_accuracy']:.4%} → {s['new']['evaluation_accuracy']:.4%}。</p>",
        f"<p>提取变化 {s['extraction_changed_count']} 份；最终奖励变化 {s['evaluation_reward_changed_count']} 份；截断 {s['truncated_count']} 份。</p>",
        '<p>新规则取末尾 300 字符最后一个 <code>#### 数字</code>，去逗号后比较字符串。源码及许可证保存在 provenance/。</p>',
    ]
    for row in report['changed_answers']:
        sections += [
            '<article>', f"<h2>{esc(row['question_id'])}</h2>",
            f"<p>目标：<code>{esc(row['target_answer'])}</code>；截断：{row['truncated']}；EOS：{row['ended_with_eos']}</p>",
            f"<p>旧提取：<code>{esc(row['old_extraction'])}</code> → 新提取：<code>{esc(row['new_extraction'])}</code></p>",
            f"<p>原始奖励：{row['old_raw_reward']} → {row['new_raw_reward']}；最终奖励：{row['old_evaluation_reward']} → {row['new_evaluation_reward']}</p>",
            '<h3>问题</h3>', f"<pre>{esc(row['question'])}</pre>",
            '<h3>完整原回答</h3>', f"<pre>{esc(row['completion'])}</pre>", '</article>',
        ]
    sections.append('</html>')
    destination.write_text('\n'.join(sections), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--historical-parser', type=Path, required=True)
    parser.add_argument('--verl-source', type=Path, required=True)
    parser.add_argument('--verl-license', type=Path, required=True)
    parser.add_argument('--verl-commit-metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    provenance_dir = output / 'provenance'
    provenance_dir.mkdir(parents=True, exist_ok=True)
    filenames = {
        'validation_original.json': args.input,
        'historical_gsm8k_data.py': args.historical_parser,
        'verl_gsm8k.py': args.verl_source,
        'verl_LICENSE': args.verl_license,
        'verl_commit.json': args.verl_commit_metadata,
    }
    saved = {}
    for name, source in filenames.items():
        destination = provenance_dir / name
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
        saved[name] = destination
    # Include this executable so the analysis itself is reproducible.
    script_destination = output / Path(__file__).name
    if Path(__file__).resolve() != script_destination.resolve():
        shutil.copyfile(__file__, script_destination)
    source_report = json.loads(saved['validation_original.json'].read_text())
    commit_metadata = json.loads(saved['verl_commit.json'].read_text())
    assert commit_metadata['sha'] == VERL_COMMIT
    assert sha256(saved['verl_gsm8k.py']) == VERL_SOURCE_SHA256
    assert sha256(saved['verl_LICENSE']) == VERL_LICENSE_SHA256
    assert len(source_report['answers']) == 128
    assert source_report['complete'] and source_report['status'] in ('passed', 'completed')
    assert sha256(saved['historical_gsm8k_data.py']) == source_report['source_sha256']['scripts/gsm8k_data.py']
    old_source = load_source_namespace(saved['historical_gsm8k_data.py'])
    verl_source = load_source_namespace(saved['verl_gsm8k.py'])
    checks = []
    rows = []
    for original in source_report['answers']:
        text = original['completion']
        target = original['target_answer']
        old = old_source['score_answer'](text, target)
        new_extract = verl_source['extract_solution'](text, method='strict')
        new_raw = float(verl_source['compute_score'](text, target, method='strict', format_score=0.0, score=1.0))
        assert new_extract == independent_verl_extract(text), original['question_id']
        assert new_raw == independent_verl_score(text, target), original['question_id']
        assert old['predicted_answer'] == original['predicted_answer'], original['question_id']
        assert old['reward'] == original['reward'], original['question_id']
        assert original['truncated'] == (not original['ended_with_eos'])
        old_evaluation = old['reward'] if original['ended_with_eos'] else 0.0
        assert old_evaluation == original['evaluation_reward'], original['question_id']
        row = dict(original)
        row.update(
            old_extraction=old['predicted_answer'], new_extraction=new_extract,
            old_raw_reward=old['reward'], new_raw_reward=new_raw,
            old_evaluation_reward=old_evaluation,
            new_evaluation_reward=new_raw if original['ended_with_eos'] else 0.0,
        )
        row['extraction_changed'] = row['old_extraction'] != row['new_extraction']
        row['raw_reward_changed'] = row['old_raw_reward'] != row['new_raw_reward']
        row['evaluation_reward_changed'] = row['old_evaluation_reward'] != row['new_evaluation_reward']
        rows.append(row)
    probes = [
        ('#### 2', '2'), ('#### 2.0', '2'), ('#### 2.0', '2.0'),
        ('#### 1,000 dollars', '1000'), ('#### 1,2,3', '123'),
        ('#### 1\n#### 2', '2'), ('answer is 2', '2'),
        ('####2', '2'), ('####  2', '2'), ('#### +2', '2'),
        ('#### -2', '-2'), ('#### .5', '.5'), ('#### 2. then explanation', '2'),
        ('#### 2\n' + 'x' * 301, '2'), ('x' * 301 + '#### 2', '2'),
        ('#### 2/3', '2'), ('#### $2', '2'), ('#### 2e3', '2'),
        ('#### 2... end', '2'), ('#### 2\nThe result is 3.', '2'),
    ]
    for text, target in probes:
        actual_extract = verl_source['extract_solution'](text, method='strict')
        actual_score = float(verl_source['compute_score'](text, target, method='strict', format_score=0.0, score=1.0))
        assert actual_extract == independent_verl_extract(text)
        assert actual_score == independent_verl_score(text, target)
        checks.append({'text': text, 'target': target, 'source_extraction': actual_extract,
                       'source_reward': actual_score, 'independent_agrees': True})
    old_counts = verdict_counts(rows, 'old')
    new_counts = verdict_counts(rows, 'new')
    assert old_counts['evaluation_correct_count'] == source_report['correct_count']
    assert old_counts['parse_failure_count'] == source_report['parse_failure_count']
    matrix = {f'{old} → {new}': sum(
        row['old_evaluation_reward'] == old and row['new_evaluation_reward'] == new for row in rows
    ) for old in [0, 1] for new in [0, 1]}
    extraction_matrix = dict(Counter(
        ('parsed' if row['old_extraction'] is not None else 'unparsed') + ' → ' +
        ('parsed' if row['new_extraction'] is not None else 'unparsed') for row in rows
    ))
    summary = {
        'answer_count': len(rows), 'truncated_count': sum(row['truncated'] for row in rows),
        'old': old_counts, 'new': new_counts,
        'extraction_changed_count': sum(row['extraction_changed'] for row in rows),
        'raw_reward_changed_count': sum(row['raw_reward_changed'] for row in rows),
        'evaluation_reward_changed_count': sum(row['evaluation_reward_changed'] for row in rows),
        'evaluation_transition_matrix': matrix, 'extraction_transition_matrix': extraction_matrix,
    }
    provenance = {
        'verl_repository': 'https://github.com/volcengine/verl',
        'verl_commit': VERL_COMMIT,
        'verl_source_url': f'https://github.com/volcengine/verl/blob/{VERL_COMMIT}/{VERL_RELATIVE_SOURCE}',
        'verl_source_raw_url': f'https://raw.githubusercontent.com/volcengine/verl/{VERL_COMMIT}/{VERL_RELATIVE_SOURCE}',
        'verl_source_sha256': sha256(saved['verl_gsm8k.py']),
        'verl_license_sha256': sha256(saved['verl_LICENSE']),
        'historical_parser_sha256': sha256(saved['historical_gsm8k_data.py']),
        'original_validation_sha256': sha256(saved['validation_original.json']),
        'original_generation_max_new_tokens': source_report['decode']['max_new_tokens'],
        'original_generation_model': source_report['model_repository'],
        'original_generation_revision': source_report['model_revision'],
        'original_generation_slurm_job_id': source_report['slurm_job_id'],
        'source_generation_git_commit': source_report['source_git_commit'],
        'generation_inputs_and_outputs_changed': False,
        'analysis_device': 'cpu_stdlib_only',
        'analysis_source_sha256': sha256(script_destination),
        'saved_file_sha256': {name: sha256(path) for name, path in saved.items()},
    }
    report = {
        'status': 'passed', 'stage': 'existing_128_answer_rule_comparison',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'old_rule': 'single_final_hash_marker_exact_decimal_v1',
        'new_rule': NEW_RULE,
        'evaluation_gate': 'zero_reward_unless_original_ended_with_eos',
        'summary': summary, 'provenance': provenance,
        'checks': {
            'original_128_rewards_reproduced': True,
            'original_128_extractions_reproduced': True,
            'verbatim_verl_and_independent_agree_on_128': True,
            'verbatim_verl_and_independent_agree_on_20_probes': True,
            'all_original_token_ids_and_text_preserved': True,
            'no_generation_no_gpu_no_training': True,
        },
        'synthetic_source_semantics_probes': checks,
        'changed_answers': [row for row in rows if row['extraction_changed'] or row['raw_reward_changed'] or row['evaluation_reward_changed']],
        'all_answers': rows,
    }
    (output / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    write_markdown(report, output / 'comparison.md')
    write_html(report, output / 'comparison.html')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('Output:', output)


if __name__ == '__main__':
    main()
