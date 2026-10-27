"""CPU contracts for audit observations, grouping identity and unchanged rewards."""

import json
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from frozen_training.answer_scoring import GSM8K_REWARD_VERSION, score_gsm8k
from reward_audit import make_audited_gsm8k_reward


def messages(texts):
    return [[{"role": "assistant", "content": text}] for text in texts]


class RewardAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "nested/rewards.jsonl"
        self.reward = make_audited_gsm8k_reward(self.path)

    def records(self):
        return [json.loads(line) for line in self.path.read_text().splitlines()]

    def test_scores_match_frozen_contract_and_preserve_log_extra(self):
        texts = [r"\boxed{1,234.00}", r"\boxed{\frac{1}{2}}", r"\boxed{2 dollars}",
                 "The answer is 2.", r"\boxed{2} then \boxed{3", r"\boxed{7.9999998}"]
        golds = ["1234", "0.5", "2", "2", "2", "8"]
        expected = [score_gsm8k(text, gold) for text, gold in zip(texts, golds)]
        extras = {}
        with patch("reward_audit.score_gsm8k", wraps=score_gsm8k) as scorer:
            actual = self.reward(messages(texts), golds, log_extra=extras.__setitem__)
        self.assertEqual(scorer.call_count, len(texts))
        self.assertEqual(actual, [row["reward"] for row in expected])
        self.assertEqual(extras["reward_status"], [row["status"] for row in expected])
        self.assertEqual(extras["extracted_answer"], [row["extracted_answer"] for row in expected])
        record = self.records()[0]
        self.assertEqual(record["reward_contract"], GSM8K_REWARD_VERSION)
        self.assertEqual(record["status_counts"], {
            "correct": 2, "incorrect": 1, "invalid_prediction": 1, "missing_or_incomplete_boxed": 2})
        self.assertEqual(record["reward_mean"], 2 / 6)
        self.assertEqual(record["grouping"]["status"], "unavailable_missing_question_ids")
        self.assertIsNone(record["grouping"]["zero_std_fraction"])
        self.assertIsNone(record["question_ids"])
        self.assertIsNone(record["global_step"])
        self.assertFalse(record["trainer_state_available"])

    def test_g8_counts_all_zero_all_one_and_mixed_groups_with_actual_ids(self):
        values = [0] * 8 + [1] * 8 + [1, 1, 0, 0, 0, 0, 0, 0]
        ids = ["q0"] * 8 + ["q1"] * 8 + ["q2"] * 8
        texts = [r"\boxed{2}" if value else r"\boxed{3}" for value in values]
        actual = self.reward(messages(texts), ["2"] * 24, id=ids,
                             trainer_state=SimpleNamespace(global_step=17))
        self.assertEqual(actual, values)
        record = self.records()[0]
        self.assertEqual(record["question_ids"], ids)
        self.assertEqual(record["global_step"], 17)
        self.assertTrue(record["trainer_state_available"])
        group = record["grouping"]
        self.assertEqual(group["status"], "validated_consecutive_groups")
        self.assertEqual(group["group_count"], 3)
        self.assertEqual(group["zero_std_fraction"], 2 / 3)
        self.assertEqual(group["all_zero_fraction"], 1 / 3)
        self.assertEqual(group["all_one_fraction"], 1 / 3)
        self.assertEqual([row["reward_mean"] for row in group["groups"]], [0, 1, 0.25])
        self.assertEqual([row["question_id"] for row in group["groups"]], ["q0", "q1", "q2"])

    def test_mixed_ids_inside_a_group_fail_closed_without_audit_record(self):
        with self.assertRaisesRegex(ValueError, "different question IDs"):
            self.reward(messages([r"\boxed{2}"] * 8), ["2"] * 8,
                        id=["a"] * 7 + ["b"])
        self.assertFalse(self.path.exists())

    def test_supplied_ids_need_one_nonempty_identifier_per_completion(self):
        bad_ids = ["question", ["a"] * 7, ["a"] * 9, ["a"] * 7 + [None],
                   ["a"] * 7 + [""], ["a"] * 7 + [True], ["a"] * 7 + [1.5]]
        for ids in bad_ids:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.reward(messages([r"\boxed{2}"] * 8), ["2"] * 8, id=ids)
        self.assertFalse(self.path.exists())

    def test_incomplete_generation_group_fails_instead_of_guessing(self):
        with self.assertRaisesRegex(ValueError, "not a multiple"):
            self.reward(messages([r"\boxed{2}"] * 7), ["2"] * 7, id=["a"] * 7)
        self.assertFalse(self.path.exists())

    def test_integer_zero_id_and_step_zero_are_preserved(self):
        self.reward(messages([r"\boxed{2}"] * 8), ["2"] * 8, id=[0] * 8,
                    trainer_state=SimpleNamespace(global_step=0))
        record = self.records()[0]
        self.assertEqual(record["global_step"], 0)
        self.assertTrue(record["trainer_state_available"])
        self.assertEqual(record["grouping"]["groups"][0]["question_id"], "0")

    def test_score_timing_excludes_logging_and_records_append(self):
        texts = messages([r"\boxed{2}"] * 8)
        with patch("reward_audit.time.perf_counter", side_effect=[10.0, 10.25]):
            self.reward(texts, ["2"] * 8, id=["a"] * 8)
        self.reward(texts, ["2"] * 8, id=["b"] * 8)
        records = self.records()
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["scoring_seconds"], 0.25)
        self.assertGreaterEqual(records[1]["scoring_seconds"], 0)
        self.assertEqual(records[1]["grouping"]["groups"][0]["question_id"], "b")

    def test_invalid_gold_or_length_does_not_create_a_success_record(self):
        for golds in (["bad"] * 8, ["2"] * 7):
            with self.subTest(golds=golds), self.assertRaises(ValueError):
                self.reward(messages([r"\boxed{2}"] * 8), golds, id=["a"] * 8)
        self.assertFalse(self.path.exists())

    def test_factory_result_is_picklable_and_keeps_metric_name(self):
        restored = pickle.loads(pickle.dumps(self.reward))
        self.assertEqual(restored.__name__, "grpo_boxed_reward")
        self.assertEqual(restored(messages([r"\boxed{2}"] * 8), ["2"] * 8,
                                  id=["a"] * 8), [1.0] * 8)
        self.assertEqual(self.records()[0]["grouping"]["all_one_fraction"], 1)

    def test_invalid_generation_counts_empty_batches_and_steps_fail(self):
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                make_audited_gsm8k_reward(self.path, num_generations=count)
        with self.assertRaises(ValueError):
            self.reward([], [])
        for step in (-1, True, 1.5):
            with self.subTest(step=step), self.assertRaises(ValueError):
                self.reward(messages([r"\boxed{2}"] * 8), ["2"] * 8,
                            trainer_state=SimpleNamespace(global_step=step), id=["a"] * 8)
        self.assertFalse(self.path.exists())


if __name__ == "__main__":
    unittest.main()
