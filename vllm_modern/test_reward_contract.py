"""Pure scoring contract checks; no model, dataset, GPU, or network access."""

from fractions import Fraction
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent / "frozen_training"))

from answer_scoring import (  # noqa: E402
    correctness_reward,
    extract_final_boxed,
    parse_numeric_answer,
    score_gsm8k,
)


class FinalBoxExtractionTests(unittest.TestCase):
    def test_final_box_and_nested_braces(self):
        cases = [
            (r"Reasoning 17. Final: \boxed{2}", r"\boxed{2}"),
            (r"\boxed{1} then \boxed{2}", r"\boxed{2}"),
            (r"Final \boxed{\frac{1}{2}}.", r"\boxed{\frac{1}{2}}"),
            ("Final \\boxed \n { 2 }.", "\\boxed \n { 2 }"),
            (r"\boxed{2} \boxedness is prose", r"\boxed{2}"),
            (r"\boxed{}", r"\boxed{}"),
        ]
        for answer, expected in cases:
            with self.subTest(answer=answer):
                self.assertEqual(extract_final_boxed(answer), expected)

    def test_missing_or_malformed_last_command_has_no_rescue(self):
        cases = [
            "", "The answer is 2.", "#### 2", r"\fbox{2}",
            r"\boxedness{2}", r"\boxed{2", r"\boxed{\frac{1}{2}",
            r"\boxed{2} but finally \boxed{3", r"\boxed{2} then \boxed",
            r"\boxed{2} then \boxed broken", r"\boxed{2} then \boxed 3",
        ]
        for answer in cases:
            with self.subTest(answer=answer):
                self.assertIsNone(extract_final_boxed(answer))


class NumericGrammarTests(unittest.TestCase):
    def test_equivalent_scalar_representations(self):
        cases = [
            ("0", Fraction(0)), ("-0.0", Fraction(0)),
            ("+12", Fraction(12)), ("  -12  ", Fraction(-12)),
            (".5", Fraction(1, 2)), ("-.5", Fraction(-1, 2)),
            ("5.", Fraction(5)), ("0.125", Fraction(1, 8)),
            ("1,234", Fraction(1234)), ("-1,234.50", Fraction(-2469, 2)),
            ("1,234,567", Fraction(1234567)), ("1e3", Fraction(1000)),
            ("-1.25E+2", Fraction(-125)), ("1e-3", Fraction(1, 1000)),
            ("1e100", Fraction(10**100)), ("1e-100", Fraction(1, 10**100)),
            ("1/2", Fraction(1, 2)), ("-3/6", Fraction(-1, 2)),
            ("3/-6", Fraction(-1, 2)), ("-3/-6", Fraction(1, 2)),
            ("1.2/0.3", Fraction(4)), ("1e2/2", Fraction(50)),
            (r"\frac{1}{2}", Fraction(1, 2)),
            (r"\dfrac{-3}{6}", Fraction(-1, 2)),
            (r"\tfrac{3}{-6}", Fraction(-1, 2)),
            (r"-\frac{-3}{6}", Fraction(1, 2)),
            (r"+\frac{1.2}{0.3}", Fraction(4)),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(parse_numeric_answer(text), expected)

    def test_rejects_prose_units_and_ambiguous_or_non_scalar_answers(self):
        cases = [
            "", " ", "the answer is 2", "2 dollars", "4 million", "4 thousand",
            "50%", r"50\%", "$2$", "$$2$$", r"\(2\)", r"\[2\]",
            r"\left(2\right)", "(2)", "[2]", "{2}", "x=2", "1+1",
            "2*3", "2**3", "2^3", "sqrt(4)", r"\sqrt{4}", "1,2",
            "(1, 2)", "[1,2]", "1 or 2", "1 and 2", "1/2/3", "1 1/2",
            "1 000", "1,23", "1234,567", "1,,000", "1,234,56", "1.2,3",
            "NaN", "nan", "inf", "Infinity", "-Infinity", r"\infty",
            "+-2", "--2", "2e", "2e1.5", "1e101", "1e-101", "0e101",
            "1/0", "1/-0", r"\frac{1}{0}", r"\frac{1}{-0.0}",
            r"\frac{\frac{1}{2}}{3}", r"\frac{1}{2+3}", r"\frac12",
            r"\frac{1}{2} dollars", r"\boxed{2}", r"\unknown{2}",
            "1" * 257, " " * 256 + "1",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertIsNone(parse_numeric_answer(text))


class GSM8KScoringTests(unittest.TestCase):
    def assert_score(self, answer, gold, reward, status):
        result = score_gsm8k(answer, gold)
        self.assertEqual(result["reward"], reward)
        self.assertEqual(result["status"], status)
        self.assertIn("extracted_answer", result)
        self.assertEqual(correctness_reward(answer, gold), reward)
        return result

    def test_equivalent_numbers_receive_full_reward(self):
        for answer, gold in [
            (r"\boxed{1,234.00}", "1234"),
            (r"\boxed{\frac{1}{2}}", "0.5"),
            (r"\boxed{5e-1}", ".5"),
            (r"\boxed{3/-6}", "-0.5"),
            (r"\boxed{-0.0}", 0),
            (r"Reasoning says 9, but the final result is \boxed{2}.", 2),
        ]:
            with self.subTest(answer=answer, gold=gold):
                result = self.assert_score(answer, gold, 1.0, "correct")
                self.assertIsNotNone(result["extracted_answer"])

    def test_exact_equality_has_no_tolerance_or_percentage_rescaling(self):
        for prediction, gold in [
            ("7.9999998", "8"), ("2.0000001", "2"),
            ("0.3333333333333333", "1/3"),
            ("100", "1"), ("0.01", "1"),
            ("9007199254740993", "9007199254740992"),
        ]:
            with self.subTest(prediction=prediction, gold=gold):
                self.assert_score(r"\boxed{" + prediction + "}", gold, 0.0, "incorrect")

    def test_final_answer_controls_reward(self):
        self.assert_score(r"\boxed{2} correction: \boxed{3}", "2", 0.0, "incorrect")
        self.assert_score(r"\boxed{3} correction: \boxed{2}", "2", 1.0, "correct")
        self.assert_score(r"\boxed{2} correction: \boxed{3", "2", 0.0,
                          "missing_or_incomplete_boxed")

    def test_missing_format_and_invalid_scalar_are_distinct(self):
        for answer in ["2", "Answer: 2", r"\boxed{2", r"\boxed{2} \boxed broken"]:
            with self.subTest(answer=answer):
                self.assert_score(answer, "2", 0.0, "missing_or_incomplete_boxed")
        for prediction in ["", "2 dollars", "4 million", "100%", "$2$", "1+1", "1/0"]:
            with self.subTest(prediction=prediction):
                self.assert_score(r"\boxed{" + prediction + "}", "2", 0.0,
                                  "invalid_prediction")

    def test_invalid_gold_is_a_data_error_even_when_prediction_is_missing(self):
        for gold in ["", "two", "2 dollars", "NaN", "Infinity", "1/0", "1e101"]:
            for answer in [r"\boxed{2}", "no boxed answer"]:
                with self.subTest(answer=answer, gold=gold):
                    with self.assertRaises(ValueError):
                        score_gsm8k(answer, gold)


if __name__ == "__main__":
    unittest.main()
