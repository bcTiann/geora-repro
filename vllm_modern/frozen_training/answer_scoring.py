"""Frozen answer contracts: GSM8K exact rational values; MATH500 Math-Verify.

The GSM8K parser is deliberately a small numeric grammar, not a prose or
symbolic-expression evaluator. No eval(), floating point, units, or guessing.
"""

from decimal import Decimal, InvalidOperation
from fractions import Fraction
from functools import lru_cache
from importlib.metadata import version
import re

GSM8K_REWARD_VERSION = "final_boxed_exact_rational_v2"
MATH500_METRIC_VERSION = "math_verify_0.8.0_math500_precision6_v1"
MAX_NUMERIC_CHARS = 256
MAX_EXPONENT = 100
_BOX = re.compile(r"\\boxed(?![A-Za-z])")
_NUMBER = re.compile(
    r"[+-]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]*)?|\.[0-9]+)"
    r"(?:[eE]([+-]?[0-9]+))?"
)
_LATEX_FRACTION = re.compile(r"([+-]?)\s*\\(?:frac|dfrac|tfrac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}")


def extract_final_boxed(answer_text):
    """Return the last boxed command including braces, or None if malformed.

    A later incomplete command invalidates extraction; never rescue an earlier
    box. Text after a complete box does not change the designated answer field.
    """
    matches = list(_BOX.finditer(answer_text))
    if not matches:
        return None
    match = matches[-1]
    opening = match.end()
    while opening < len(answer_text) and answer_text[opening].isspace():
        opening += 1
    if opening == len(answer_text) or answer_text[opening] != "{":
        return None
    depth = 1
    for position in range(opening + 1, len(answer_text)):
        if answer_text[position] == "{":
            depth += 1
        elif answer_text[position] == "}":
            depth -= 1
            if depth == 0:
                return answer_text[match.start():position + 1]
    return None


def _numeric_atom(text):
    text = text.strip()
    match = _NUMBER.fullmatch(text)
    if match is None:
        return None
    exponent = match.group(1)
    if exponent is not None and abs(int(exponent)) > MAX_EXPONENT:
        return None
    try:
        return Fraction(Decimal(text.replace(",", "")))
    except (InvalidOperation, ValueError, OverflowError):
        return None


def parse_numeric_answer(text):
    """Parse a whole numeric scalar exactly, with bounded input and exponent.

    Accept signed decimals/scientific notation with valid thousands grouping,
    p/q, and LaTeX frac/dfrac/tfrac with numeric atoms. Reject prose, units,
    percentages, equations, lists, arithmetic, and nonfinite values.
    """
    text = str(text)
    if len(text) > MAX_NUMERIC_CHARS:
        return None
    text = text.strip()
    fraction = _LATEX_FRACTION.fullmatch(text)
    sign = 1
    if fraction:
        sign = -1 if fraction.group(1) == "-" else 1
        numerator, denominator = fraction.group(2), fraction.group(3)
    elif text.count("/") == 1:
        numerator, denominator = text.split("/")
    else:
        return _numeric_atom(text)
    numerator, denominator = _numeric_atom(numerator), _numeric_atom(denominator)
    if numerator is None or denominator is None or denominator == 0:
        return None
    return sign * numerator / denominator


@lru_cache(maxsize=16384)
def _gold_value(text):
    value = parse_numeric_answer(text)
    if value is None:
        raise ValueError(f"Invalid GSM8K numeric gold: {text!r}")
    return value


def score_gsm8k(answer_text, correct_answer_text):
    """Return reward and extraction diagnostics; invalid gold is a data error."""
    gold = _gold_value(str(correct_answer_text))
    boxed = extract_final_boxed(answer_text)
    extracted = None if boxed is None else boxed[boxed.index("{") + 1:-1].strip()
    prediction = None if extracted is None else parse_numeric_answer(extracted)
    if boxed is None:
        status = "missing_or_incomplete_boxed"
    elif prediction is None:
        status = "invalid_prediction"
    else:
        status = "correct" if prediction == gold else "incorrect"
    return {"reward": float(status == "correct"), "status": status,
            "extracted_answer": extracted,
            "normalized_prediction": None if prediction is None else str(prediction),
            "normalized_gold": str(gold)}


def correctness_reward(answer_text, correct_answer_text):
    return score_gsm8k(answer_text, correct_answer_text)["reward"]


@lru_cache(maxsize=1)
def _math500_metric():
    if version("math-verify") != "0.8.0":
        raise RuntimeError("MATH500 protocol requires math-verify==0.8.0")
    from math_verify import ExprExtractionConfig, LatexExtractionConfig
    from math_verify.metric import math_metric
    return math_metric(
        gold_extraction_target=(LatexExtractionConfig(boxed_match_priority=0),),
        pred_extraction_target=(LatexExtractionConfig(), ExprExtractionConfig()),
        precision=6,
    )


def score_math500(answer_text, correct_answer_text):
    """Official task metric settings; boxed-only correctness is secondary.

    This is a scorer, not a claim to reproduce the entire official evaluation
    harness/prompt/decoding protocol. Gold is LaTeX, not GSM8K numeric grammar.
    """
    metric = _math500_metric()
    from math_verify import LatexExtractionConfig, parse, verify
    gold_text = "$" + str(correct_answer_text) + "$"
    gold = parse(gold_text, extraction_config=[LatexExtractionConfig()],
                 extraction_mode="first_match", fallback_mode="no_fallback")
    if not gold:
        raise ValueError(f"Invalid MATH500 gold: {correct_answer_text!r}")
    score, extractions = metric([gold_text], [answer_text])
    # Preserve the existing notebook's secondary metric: its last-box search
    # recognizes an opening brace, unlike the new GSM8K command-level policy.
    matches = list(re.finditer(r"\\boxed\s*\{", answer_text))
    boxed = None
    if matches:
        match = matches[-1]
        depth = 1
        for position in range(match.end(), len(answer_text)):
            if answer_text[position] == "{":
                depth += 1
            elif answer_text[position] == "}":
                depth -= 1
                if depth == 0:
                    boxed = answer_text[match.start():position + 1]
                    break
    prediction = [] if boxed is None else parse(
        "$" + boxed + "$", extraction_config=[LatexExtractionConfig(boxed_match_priority=0)],
        extraction_mode="first_match", fallback_mode="no_fallback")
    return {"math_verify_score": float(score),
            "strict_boxed_score": float(bool(prediction) and verify(gold, prediction, float_rounding=6)),
            "has_complete_boxed": boxed is not None, "boxed_parseable": bool(prediction),
            "boxed_answer": boxed, "metric_extractions": extractions}
