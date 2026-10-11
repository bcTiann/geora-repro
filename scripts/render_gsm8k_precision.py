"""Render real GSM8K precision reports without truncating any answer.

This uses the exact strict answer rule in geora-repro/scripts/gsm8k_data.py.
It verifies recorded grading rather than replacing it with a lenient parser.
The input reports are independent greedy generations of one original model.
No model inference, accelerator, third-party package, or network is required.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import gzip
import html
import json
from pathlib import Path
import re


ANSWER_RULE = "single_final_hash_marker_exact_decimal_v1"
NUMBER = re.compile(
    r"[+-]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?|\.[0-9]+)\Z"
)
RUN_NAMES = ("bf16_cache", "bf16_no_cache", "fp32_cache", "fp32_no_cache")
RUN_LABELS = {
    "bf16_cache": "BF16 · 有 KV cache",
    "bf16_no_cache": "BF16 · 无 KV cache",
    "fp32_cache": "FP32 · 有 KV cache",
    "fp32_no_cache": "FP32 · 无 KV cache",
}
BACKEND_LABELS = {"local": "Metal", "amd": "AMD"}
STATUS_LABELS = {
    "correct": "正确（严格规则）",
    "numeric_mismatch": "数字不匹配（可解析）",
    "format_failure": "格式无法解析",
    "missing": "尚无结果",
}


def canonical_decimal(value):
    if value == 0:
        return "0"
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def strict_grade(completion, target):
    """Match the training reward rule exactly; never infer a last number."""
    if not isinstance(target, str) or NUMBER.fullmatch(target) is None:
        raise ValueError(f"Unsupported target: {target!r}")
    expected = Decimal(target.replace(",", ""))
    predicted = None
    markers = completion.count("####")
    reason = None
    if markers == 0:
        reason = "缺少 #### 标记。"
    elif markers != 1:
        reason = f"有 {markers} 个 #### 标记，严格规则要求恰好一个。"
    else:
        suffix = completion.split("####", 1)[1].strip()
        if not suffix:
            reason = "#### 后没有数字。"
        elif NUMBER.fullmatch(suffix) is None:
            reason = "#### 后的全部非空白文字不是单个受支持的十进制数字。"
        else:
            predicted = Decimal(suffix.replace(",", ""))
    grading = {
        "reward": float(predicted is not None and predicted == expected),
        "predicted_answer": None if predicted is None else canonical_decimal(predicted),
        "target_answer": canonical_decimal(expected),
        "answer_rule": ANSWER_RULE,
    }
    status = "format_failure" if predicted is None else (
        "correct" if predicted == expected else "numeric_mismatch"
    )
    return grading, status, reason


def load_report(path, backend, allow_partial, expected_count):
    stored = path.read_bytes()
    raw = gzip.decompress(stored) if path.suffix == ".gz" else stored
    report = json.loads(raw)
    if not allow_partial and report.get("status") != "completed":
        raise ValueError(f"{path}: report status is {report.get('status')!r}, not completed")
    expected_device = "mps" if backend == "local" else "cuda"
    if report.get("device") != expected_device:
        raise ValueError(f"{path}: expected {expected_device}, got {report.get('device')!r}")
    if backend == "amd" and not report.get("hip_version"):
        raise ValueError(f"{path}: AMD report must identify its HIP version")
    rows = report.get("questions")
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError(f"{path}: expected {expected_count} question rows")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate question IDs")
    for row in rows:
        runs = row.get("runs", {})
        if set(runs) - set(RUN_NAMES):
            raise ValueError(f"{path}, {row['id']}: unknown run keys")
        if not allow_partial and set(runs) != set(RUN_NAMES):
            raise ValueError(f"{path}, {row['id']}: four conditions are required")
        for name, run in runs.items():
            precision = name.split("_", 1)[0]
            use_cache = not name.endswith("_no_cache")
            if run.get("precision") != precision or run.get("use_cache") is not use_cache:
                raise ValueError(f"{path}, {row['id']}, {name}: condition metadata disagrees")
            completion = run.get("completion")
            token_ids = run.get("completion_token_ids")
            if not isinstance(completion, str):
                raise ValueError(f"{path}, {row['id']}, {name}: missing completion text")
            if not isinstance(token_ids, list) or not token_ids or any(
                type(token_id) is not int for token_id in token_ids
            ):
                raise ValueError(f"{path}, {row['id']}, {name}: missing/invalid complete token IDs")
            if run.get("completion_token_count") != len(token_ids):
                raise ValueError(f"{path}, {row['id']}, {name}: token count disagrees")
            if type(run.get("ended_with_eos")) is not bool:
                raise ValueError(f"{path}, {row['id']}, {name}: missing/invalid EOS flag")
            if not allow_partial and run.get("ended_with_eos") is not True:
                raise ValueError(f"{path}, {row['id']}, {name}: no EOS; not a complete answer")
            grading, status, reason = strict_grade(completion, row["target"])
            if run.get("grading") != grading:
                raise ValueError(
                    f"{path}, {row['id']}, {name}: recorded grading differs from exact rule; "
                    f"recorded={run.get('grading')!r}, recomputed={grading!r}"
                )
            run["presentation_status"] = status
            run["format_failure_reason"] = reason
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "stored_sha256": hashlib.sha256(stored).hexdigest(),
        "compression": "gzip" if path.suffix == ".gz" else "none",
        "report": report,
    }


def check_alignment(sources):
    """Require the same weights, data, questions and generation settings."""
    if len(sources) < 2:
        return
    local = sources["local"]["report"]
    amd = sources["amd"]["report"]
    settings = (
        "experiment", "model_repository", "model_revision", "data_source_commit",
        "max_new_tokens", "decoding", "attention_implementation", "batch_size",
        "cache_definition", "no_cache_definition", "parameter_count",
    )
    for key in settings:
        if key not in local or key not in amd or local[key] != amd[key]:
            raise ValueError(f"Local/AMD reports do not match for {key!r}")
    for before, after in zip(local["questions"], amd["questions"]):
        for key in ("id", "question", "target", "source_index", "source_split"):
            if before.get(key) != after.get(key):
                raise ValueError(f"Local/AMD question rows do not match for {key!r}")
        for name in set(before["runs"]) & set(after["runs"]):
            if before["runs"][name].get("prompt_token_count") != after["runs"][name].get("prompt_token_count"):
                raise ValueError(f"Local/AMD prompt token counts differ for {before['id']}, {name}")


def condition_label(condition):
    backend, name = condition
    return BACKEND_LABELS[backend] + " · " + RUN_LABELS[name]


def condition_short(condition):
    backend, name = condition
    return ("M" if backend == "local" else "A") + (
        "16" if name.startswith("bf16") else "32"
    ) + ("C" if not name.endswith("_no_cache") else "N")


def get_run(sources, condition, index):
    backend, name = condition
    if backend not in sources:
        return None
    return sources[backend]["report"]["questions"][index]["runs"].get(name)


def make_summaries(sources, conditions, rows):
    summary = []
    for condition in conditions:
        counts = Counter()
        tokens = 0
        seconds = 0.0
        for index in range(len(rows)):
            run = get_run(sources, condition, index)
            if run is None:
                counts["missing"] += 1
                continue
            counts[run["presentation_status"]] += 1
            counts["present"] += 1
            counts["eos"] += int(run["ended_with_eos"])
            tokens += run["completion_token_count"]
            seconds += run.get("elapsed_seconds", 0)
        summary.append({
            "condition": condition_label(condition),
            "condition_short": condition_short(condition),
            "counts": dict(counts), "tokens": tokens, "seconds": seconds,
        })
    pairs = []
    per_backend = ((RUN_NAMES[0], RUN_NAMES[1]), (RUN_NAMES[2], RUN_NAMES[3]),
                   (RUN_NAMES[0], RUN_NAMES[2]), (RUN_NAMES[1], RUN_NAMES[3]))
    for backend in BACKEND_LABELS:
        for left, right in per_backend:
            pairs.append(((backend, left), (backend, right)))
    pairs.extend((("local", name), ("amd", name)) for name in RUN_NAMES)
    comparisons = []
    for left_condition, right_condition in pairs:
        counts = Counter()
        records = []
        for index, row in enumerate(rows):
            left = get_run(sources, left_condition, index)
            right = get_run(sources, right_condition, index)
            if left is None or right is None:
                counts["missing_pair"] += 1
                continue
            left_ids, right_ids = left["completion_token_ids"], right["completion_token_ids"]
            common = 0
            for first, second in zip(left_ids, right_ids):
                if first != second:
                    break
                common += 1
            left_answer = left["grading"]["predicted_answer"]
            right_answer = right["grading"]["predicted_answer"]
            both_parsed = left_answer is not None and right_answer is not None
            # Two parse failures never count as valid answer agreement.
            numeric_equal = both_parsed and left_answer == right_answer
            token_equal = left_ids == right_ids
            counts["available_pairs"] += 1
            counts["exact_tokens_equal"] += int(token_equal)
            counts["both_parsed"] += int(both_parsed)
            counts["valid_numeric_equal"] += int(numeric_equal)
            counts["unparsed_pairs"] += int(not both_parsed)
            counts["both_correct"] += int(
                left["grading"]["reward"] == 1.0 and right["grading"]["reward"] == 1.0
            )
            records.append({
                "question_id": row["id"], "exact_tokens_equal": token_equal,
                "both_parsed": both_parsed, "valid_numeric_equal": numeric_equal,
                "common_prefix_tokens": common,
                "left_numeric_answer": left_answer, "right_numeric_answer": right_answer,
            })
        comparisons.append({
            "left": condition_label(left_condition), "right": condition_label(right_condition),
            "pair_short": condition_short(left_condition) + " / " + condition_short(right_condition),
            "counts": dict(counts), "questions": records,
        })
    return summary, comparisons


def h(value):
    return html.escape(str(value), quote=True)


def md(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def rendered_cell(run):
    if run is None:
        return "待完成"
    answer = run["grading"]["predicted_answer"]
    numeric = "解析失败" if answer is None else answer
    eos = "EOS" if run["ended_with_eos"] else "无 EOS"
    return f"{numeric} · R={run['grading']['reward']:.0f} · {eos}"


def render_markdown(sources, conditions, rows, summary, comparisons, partial):
    out = [
        f"# GSM8K：{len(rows)} 题完整回答的精度与 KV cache 对照",
        "",
        "**预览：结果尚未全部完成。**" if partial else f"两份输入报告均已完成；{len(rows) * len(conditions)} 个回答均保留原文，见同目录 HTML。",
        "",
        "## 实验范围与读法", "",
        "同一个未训练的原始 Qwen 模型，在 Metal 和 AMD 上分别用 BF16 / FP32、"
        "有 / 无 KV cache 独立进行贪心生成。没有 GeoRA 训练，也没有改变权重数值："
        "FP32 路径是对 BF16 检查点数值的精确提升，不能恢复原先已舍入的信息。",
        "",
        "有 cache 使用 Transformers DynamicCache；无 cache 每一步重算增长的完整前文。"
        "回答若已分岔，后续输入前文也不同。本表不是固定同一前文下的逐 token 概率比较。"
        "完整 token 一致也不证明 logits 或概率数值完全相同。",
        "",
        "严格 reward 要求回答中恰好一个 `####`，其后全部非空白文字必须是一个受支持的十进制数字，"
        "且数值等于目标。格式解析失败与可解析数字不匹配分别统计；不从推理正文宽松抽取数字。"
        "这些是预先选定的训练集 smoke 题目，不是最终测试集成绩。",
        "",
        "标签：M=Metal，A=AMD，16=BF16，32=FP32，C=有 cache，N=无 cache。"
        "R 是现有训练规则的严格 reward；EOS 表示自然生成结束标记。",
        "", "## 按条件汇总", "",
        "| 条件 | 已生成 | 严格正确 | 格式解析失败 | 可解析数字不匹配 | EOS | token 总数 | 实测生成秒数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary:
        counts = item["counts"]
        out.append(f"| {md(item['condition'])} | {counts.get('present', 0)} | "
                   f"{counts.get('correct', 0)} | {counts.get('format_failure', 0)} | "
                   f"{counts.get('numeric_mismatch', 0)} | {counts.get('eos', 0)} | "
                   f"{item['tokens']} | {item['seconds']:.1f} |")
    out.extend([
        "", "生成时间包含记录统计的开销，各条件输出长度也可能不同；这不是公平的训练性能基准。",
        "", "## 每题最终数字 / 严格 reward / EOS", "",
        "| 题目 ID | 目标 | " + " | ".join(condition_short(c) for c in conditions) + " |",
        "| --- | ---: | " + " | ".join("---" for _ in conditions) + " |",
    ])
    for index, row in enumerate(rows):
        cells = [rendered_cell(get_run(sources, condition, index)) for condition in conditions]
        out.append("| " + " | ".join(md(v) for v in [row["id"], row["target"], *cells]) + " |")
    out.extend([
        "", "## 两条件比较：完整 token 是否一模一样", "",
        "| 条件对 | 有两边结果 | 完整 token 相同 | 两边均能解析 | 有效最终数字相同 | 至少一边解析失败 | 两边均严格正确 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for item in comparisons:
        counts = item["counts"]
        out.append(f"| {item['pair_short']} | {counts.get('available_pairs', 0)} | "
                   f"{counts.get('exact_tokens_equal', 0)} | {counts.get('both_parsed', 0)} | "
                   f"{counts.get('valid_numeric_equal', 0)} | {counts.get('unparsed_pairs', 0)} | "
                   f"{counts.get('both_correct', 0)} |")
    out.extend([
        "", "完整 token 比较包含 EOS，长度与每一个 token ID 都必须相同。有效数字一致要求两边都成功解析；"
        "`None == None` 不计为有效数字一致。两个回答可以给出相同的错误数字，所以数字一致与正确分列。",
        "", "## 格式失败的具体原因", "",
    ])
    failures = []
    for index, row in enumerate(rows):
        for condition in conditions:
            run = get_run(sources, condition, index)
            if run and run["presentation_status"] == "format_failure":
                failures.append(f"- {row['id']} / {condition_short(condition)}：{run['format_failure_reason']}")
    out.extend(failures or ["没有已完成回答出现格式解析失败。"])
    out.extend(["", "## 输入与可追溯信息", ""])
    for backend, source in sources.items():
        report = source["report"]
        out.extend([
            f"### {BACKEND_LABELS[backend]}", "",
            f"- 输入：`{source['path']}`",
            f"- SHA256：`{source['sha256']}`",
            f"- 报告状态：`{report['status']}`；设备：`{report.get('device_name')}`；HIP：`{report.get('hip_version')}`",
            f"- PyTorch：`{report.get('torch_version')}`；Transformers：`{report.get('transformers_version')}`",
            f"- 模型：`{report.get('model_repository')}`；版本：`{report.get('model_revision')}`",
            f"- 数据版本：`{report.get('data_source_commit')}`；最多新 token：`{report.get('max_new_tokens')}`",
            "",
        ])
    return "\n".join(out) + "\n"


def render_html(sources, conditions, rows, summary, comparisons, partial):
    out = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'>",
           "<meta name='viewport' content='width=device-width,initial-scale=1'>",
           "<title>GSM8K 完整回答：精度与 KV cache</title>",
           """<style>
*{box-sizing:border-box}body{margin:0;background:#f4f6f8;color:#172534;font:16px/1.65 system-ui,sans-serif}
main{max-width:1450px;margin:auto;padding:24px}h1,h2,h3{line-height:1.3}h1{font-size:28px}h2{margin-top:0}
a{color:#1758a5}.intro,.question,.summary{background:white;border:1px solid #d8e0e7;border-radius:10px;padding:22px;margin:20px 0}
nav{position:sticky;top:0;background:#e8edf2f5;border-bottom:1px solid #c9d4df;padding:12px;z-index:1;display:flex;flex-wrap:wrap;gap:8px}
nav a{background:white;border:1px solid #c9d4df;border-radius:5px;padding:3px 9px;text-decoration:none}
.question{scroll-margin-top:120px}.question-text{white-space:pre-wrap;margin:14px 0}.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
.card{border:1px solid #d8e0e7;border-top:4px solid #3878b4;border-radius:7px;padding:16px;min-width:0;background:#fbfcfd}
.card h3{margin:0 0 10px;font-size:18px}.card.amd{border-top-color:#7656a4}.badges{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px}
.badge{border-radius:4px;padding:2px 7px;background:#e8edf2;font-size:13px}.correct{background:#dff2e6;color:#185b32}
.numeric_mismatch{background:#ffe7de;color:#923c20}.format_failure{background:#fff1cb;color:#775217}.missing{background:#ececec;color:#555}
pre.answer{white-space:pre-wrap;overflow-wrap:anywhere;word-break:normal;margin:16px 0 0;font:15px/1.7 ui-monospace,monospace;border-top:1px solid #d8e0e7;padding-top:14px}
.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #d8e0e7;padding:8px;text-align:left;white-space:nowrap}th{background:#eef2f6}
.muted{color:#536779;font-size:14px}.notice{background:#fff1cb;padding:10px;border-radius:5px}.metadata{overflow-wrap:anywhere}.metric{font-size:14px}
@media(max-width:800px){main{padding:12px}.cards{grid-template-columns:1fr}.intro,.question,.summary{padding:16px}h1{font-size:23px}nav{position:static}.question{scroll-margin-top:10px}}
@media print{nav{position:static}.cards{grid-template-columns:1fr}.card{break-inside:avoid}body{background:white}}
</style>""",
           "<main><h1>GSM8K 完整回答：BF16 / FP32 × 有 / 无 KV cache</h1>"]
    if partial:
        out.append("<p class='notice'>这是部分结果预览。缺失结果和未出现 EOS 的回答不会称为完整输出。</p>")
    out.append("<section class='intro'><h2>我们在比较什么</h2>"
               "<p>同一个未训练的原始 Qwen 模型，在 Metal 与 AMD 上运行 8 个计算条件，"
               "各自从相同题目独立贪心生成。没有 GeoRA 训练。FP32 路径精确提升同一 BF16 检查点的数值，"
               "不恢复已丢失的精度。有 cache 使用 Transformers DynamicCache；无 cache 每一步重算增长的完整前文。</p>"
               "<p>这里展示全部回答原文，没有省略或截断。EOS 表示自然生成结束标记。回答分岔以后前文不同，"
               "本页不是固定前文下的概率差比较；完整 token 一致也不代表概率数值完全相同。</p>"
               "<p>严格 reward 要求恰好一个 <code>####</code>，其后全部非空白文字是单个受支持的十进制数字，"
               "且等于目标。格式无法解析与可解析数字不匹配分别统计，不从推理正文宽松提取答案。"
               "这 10 题来自预先选定的训练集 smoke 子集，不是最终测试成绩。</p></section>")
    out.append("<nav aria-label='题目导航'>")
    for index, row in enumerate(rows):
        out.append(f"<a href='#q{index + 1}'>第 {index + 1} 题 · {h(row['id'])}</a>")
    out.append("</nav><section class='summary'><h2>按计算条件汇总</h2><div class='table-wrap'><table>"
               "<tr><th>条件</th><th>已生成</th><th>严格正确</th><th>格式失败</th><th>数字不匹配</th><th>EOS</th><th>token 总数</th><th>实测秒数</th></tr>")
    for item in summary:
        counts = item["counts"]
        values = [item["condition"], counts.get("present", 0), counts.get("correct", 0),
                  counts.get("format_failure", 0), counts.get("numeric_mismatch", 0),
                  counts.get("eos", 0), item["tokens"], f"{item['seconds']:.1f}"]
        out.append("<tr>" + "".join(f"<td>{h(v)}</td>" for v in values) + "</tr>")
    out.append("</table></div><p class='muted'>生成时间包含记录统计的开销，输出长度可能不同；不是公平的训练性能基准。</p>"
               "<h3>两条件比较</h3><p class='muted'>M=Metal，A=AMD，16=BF16，32=FP32，C=有 cache，N=无 cache。"
               "完整 token 比较包含 EOS。有效数字一致要求两边均解析成功；None==None 不计为一致。相同错误答案仍可数字一致。</p>"
               "<div class='table-wrap'><table><tr><th>条件对</th><th>两边有结果</th><th>完整 token 相同</th>"
               "<th>两边均可解析</th><th>有效数字相同</th><th>至少一边解析失败</th><th>两边均正确</th></tr>")
    for item in comparisons:
        counts = item["counts"]
        values = [item["pair_short"], *[counts.get(key, 0) for key in (
            "available_pairs", "exact_tokens_equal", "both_parsed", "valid_numeric_equal", "unparsed_pairs", "both_correct")]]
        out.append("<tr>" + "".join(f"<td>{h(v)}</td>" for v in values) + "</tr>")
    out.append("</table></div></section>")
    for index, row in enumerate(rows):
        out.append(f"<section class='question' id='q{index + 1}'><h2>第 {index + 1} 题 · {h(row['id'])}</h2>"
                   f"<div class='question-text'>{h(row['question'])}</div>"
                   f"<p><strong>目标数字：{h(row['target'])}</strong>"
                   f" <span class='muted'>来源：{h(row.get('source_split'))} / {h(row.get('source_index'))}</span></p><div class='cards'>")
        for condition in conditions:
            backend, name = condition
            run = get_run(sources, condition, index)
            out.append(f"<article class='card {backend}'><h3>{h(condition_label(condition))}</h3>")
            if run is None:
                out.append("<span class='badge missing'>尚无结果</span></article>")
                continue
            status = run["presentation_status"]
            parsed = run["grading"]["predicted_answer"]
            out.append(f"<div class='badges'><span class='badge {status}'>{h(STATUS_LABELS[status])}</span>"
                       f"<span class='badge'>严格 reward = {run['grading']['reward']:.0f}</span>"
                       f"<span class='badge'>数字：{h('无法解析' if parsed is None else parsed)}</span>"
                       f"<span class='badge'>{'EOS ✓' if run['ended_with_eos'] else '未 EOS：不称完整'}</span></div>"
                       f"<div class='metric'>回答 token：{run['completion_token_count']}（包含返回的特殊 token）；"
                       f"输入 token：{run['prompt_token_count']}；生成耗时：{run.get('elapsed_seconds', 0):.2f} 秒。</div>")
            if run["format_failure_reason"]:
                out.append(f"<p class='metric'>{h(run['format_failure_reason'])} R=0 不等于证明推理正文的数字算错。</p>")
            elif status == "numeric_mismatch":
                out.append(f"<p class='metric'>解析数字 {h(parsed)} 不等于目标 {h(row['target'])}。这里只核对最终数字，不自动判定是哪一步推理出了错。</p>")
            # Never slice, shorten, normalize, or reformat the recorded answer.
            out.append(f"<pre class='answer'><span>{h(run['completion'])}</span></pre></article>")
        out.append("</div></section>")
    out.append("<section class='summary metadata'><h2>输入报告与版本</h2>")
    for backend, source in sources.items():
        report = source["report"]
        out.append(f"<h3>{h(BACKEND_LABELS[backend])}</h3><ul>")
        entries = [
            ("输入", source["path"]), ("SHA256", source["sha256"]),
            ("状态", report["status"]), ("设备", report.get("device_name")),
            ("HIP", report.get("hip_version")), ("PyTorch", report.get("torch_version")),
            ("Transformers", report.get("transformers_version")),
            ("模型", report.get("model_repository")), ("模型版本", report.get("model_revision")),
            ("数据版本", report.get("data_source_commit")), ("最多新 token", report.get("max_new_tokens")),
        ]
        for label, value in entries:
            out.append(f"<li>{h(label)}：<code>{h(value)}</code></li>")
        out.append("</ul>")
    out.append("</section></main></html>\n")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-report", type=Path, required=True)
    parser.add_argument("--amd-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=10)
    parser.add_argument("--allow-partial", action="store_true", help="Explicitly label a partial preview; never invent missing outputs.")
    args = parser.parse_args()
    if args.expected_count <= 0:
        parser.error("--expected-count must be positive")
    if args.amd_report is None and not args.allow_partial:
        parser.error("--amd-report is required for a complete eight-condition comparison")
    sources = {"local": load_report(args.local_report, "local", args.allow_partial, args.expected_count)}
    if args.amd_report:
        sources["amd"] = load_report(args.amd_report, "amd", args.allow_partial, args.expected_count)
    check_alignment(sources)
    rows = sources["local"]["report"]["questions"]
    conditions = [(backend, name) for backend in BACKEND_LABELS for name in RUN_NAMES]
    summary, comparisons = make_summaries(sources, conditions, rows)
    partial = len(sources) != 2 or any(
        source["report"]["status"] != "completed" for source in sources.values()
    ) or any(
        get_run(sources, condition, index) is None
        or not get_run(sources, condition, index)["ended_with_eos"]
        for condition in conditions for index in range(len(rows))
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_html = args.output_dir / "gsm8k_precision_full_answers.html"
    output_md = args.output_dir / "gsm8k_precision_summary.md"
    output_json = args.output_dir / "gsm8k_precision_summary.json"
    output_html.write_text(render_html(sources, conditions, rows, summary, comparisons, partial), encoding="utf-8")
    output_md.write_text(render_markdown(sources, conditions, rows, summary, comparisons, partial), encoding="utf-8")
    output_json.write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(), "partial_preview": partial,
        "question_count": len(rows), "condition_count": len(conditions),
        "answer_rule": ANSWER_RULE,
        "raw_legacy_comparisons_used": False,
        "sources": {backend: {k: v for k, v in source.items() if k != "report"} for backend, source in sources.items()},
        "conditions": summary, "comparisons": comparisons,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Partial preview:" if partial else "Complete comparison:", len(rows), "questions ×", len(conditions), "conditions")
    print("Full answers:", output_html)
    print("Summary:", output_md)
    print("Machine-readable comparisons:", output_json)


if __name__ == "__main__":
    main()
