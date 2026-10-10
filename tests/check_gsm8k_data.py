"""CPU checks for exact rewards, deterministic splits and data provenance."""

import argparse
from decimal import Decimal
import json
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIRECTORY / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gsm8k_data import (
    ANSWER_RULE,
    SOURCE_COMMIT,
    build_prompt,
    extract_final_answer,
    file_sha256,
    load_prepared_rows,
    prepare_dataset,
    score_answer,
    split_train_rows,
    write_jsonl,
)


class RewardTests(unittest.TestCase):
    def test_exact_numerical_equivalence(self):
        cases = [
            ("12 + 13 = 25.\n#### 25", "25"),
            ("Costs 3 and 7, total 10.\n#### 1,234.500", "1234.5"),
            ("First 2, then -3.\n#### -0.50", "-.5"),
            ("#### +.5", "0.5"),
            ("#### -0.000", "0"),
            ("#### 12345678901234567890.123456789", "12345678901234567890.123456789"),
        ]
        for completion, target in cases:
            with self.subTest(completion=completion):
                self.assertEqual(score_answer(completion, target)["reward"], 1.0)

    def test_incorrect_answer_is_zero(self):
        self.assertEqual(score_answer("Reasoning has 5.\n#### 6", "5")["reward"], 0.0)
        self.assertEqual(score_answer("#### 0.30000000000000004", "0.3")["reward"], 0.0)

    def test_no_final_marker_fallback(self):
        for completion in ["The answer is 5", "5", "Intermediate result 5. Now continue.", ""]:
            with self.subTest(completion=completion):
                self.assertIsNone(extract_final_answer(completion))
                self.assertEqual(score_answer(completion, "5")["reward"], 0.0)

    def test_ambiguous_or_unsupported_formats(self):
        invalid = [
            "#### 5\n#### 6", "#### 5 cats", "#### $5", "#### 5,00", "#### 1,23,456",
            "#### 1/2", "#### 1e3", "#### NaN", "#### Infinity", "#### 2 + 3",
            "#### 5\nMore reasoning", "#### ", "#### ５", "#### 0.5%",
        ]
        for completion in invalid:
            with self.subTest(completion=completion):
                self.assertIsNone(extract_final_answer(completion))

    def test_whitespace_and_decimal_parse(self):
        self.assertEqual(extract_final_answer("Reasoning: 19.\n####  -1,250.75 \n\n"), Decimal("-1250.75"))

    def test_bad_reference_fails_instead_of_scoring(self):
        with self.assertRaises(ValueError):
            score_answer("#### 5", "five")

    def test_prompt_excludes_reference_and_reasoning(self):
        question = "SENTINEL_QUESTION: how many apples?"
        row = {"question": question, "target": "987654321", "answer": "SECRET_REASONING"}
        prompt = build_prompt(row["question"])
        self.assertIn(question, prompt)
        self.assertIn("#### <number>", prompt)
        self.assertNotIn(row["target"], prompt)
        self.assertNotIn(row["answer"], prompt)
        with self.assertRaises(ValueError):
            build_prompt("")


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"id": f"train:{index:05d}", "question": str(index)} for index in range(40)]

    def test_split_independent_of_input_order(self):
        train, validation = split_train_rows(self.rows, seed=42, validation_count=8)
        reversed_train, reversed_validation = split_train_rows(list(reversed(self.rows)), seed=42, validation_count=8)
        self.assertEqual(train, reversed_train)
        self.assertEqual(validation, reversed_validation)
        self.assertEqual(len(train), 32)
        self.assertEqual(len(validation), 8)
        self.assertFalse({row["id"] for row in train} & {row["id"] for row in validation})
        self.assertEqual({row["id"] for row in train + validation}, {row["id"] for row in self.rows})

    def test_split_leaves_fixed_smoke_pool(self):
        with self.assertRaises(ValueError):
            split_train_rows(self.rows, seed=42, validation_count=25)

    def test_existing_policy_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest_path = directory / "manifest.json"
            manifest_path.write_text('{"source_commit": "another revision"}')
            before = manifest_path.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "policy differs"):
                prepare_dataset(directory, offline=True)
            self.assertEqual(manifest_path.read_bytes(), before)
            self.assertFalse((directory / "raw").exists())

    def test_unicode_line_separator_stays_in_one_json_record(self):
        row = {"question": "First line\u2028Second line", "target": "5"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rows.jsonl"
            write_jsonl(path, [row])
            lines = path.read_text().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0]), row)


def check_prepared_data(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["source_commit"] == SOURCE_COMMIT
    assert manifest["answer_rule"] == ANSWER_RULE
    groups = {split: load_prepared_rows(directory, split) for split in ("train", "validation", "test", "smoke")}
    assert {split: len(rows) for split, rows in groups.items()} == {"train": 7345, "validation": 128, "test": 1319, "smoke": 16}
    ids = {split: {row["id"] for row in rows} for split, rows in groups.items()}
    assert not ids["train"] & ids["validation"]
    assert not ids["train"] & ids["test"]
    assert not ids["validation"] & ids["test"]
    assert ids["smoke"] <= ids["train"]
    assert not ids["smoke"] & (ids["validation"] | ids["test"])
    assert all(row["source_split"] == "train" for row in groups["smoke"] + groups["validation"])
    assert len(ids["train"] | ids["validation"]) == 7473
    assert [row["id"] for row in groups["smoke"]] == manifest["smoke_ids"]
    assert [row["id"] for row in groups["validation"]] == manifest["validation_ids"]
    for split in ("train", "validation", "test"):
        for row in groups[split]:
            assert score_answer(f"#### {row['target']}", row["target"])["reward"] == 1.0
            assert set(row) == {"id", "source_split", "source_index", "question", "target"}
    # Detect corruption independently of the nominal successful dataset path.
    with tempfile.TemporaryDirectory() as temporary:
        copy_directory = Path(temporary)
        (copy_directory / "manifest.json").write_text(json.dumps(manifest))
        (copy_directory / "smoke.jsonl").write_bytes((directory / "smoke.jsonl").read_bytes() + b" ")
        try:
            load_prepared_rows(copy_directory, "smoke")
        except RuntimeError as error:
            assert "checksum failed" in str(error)
        else:
            raise AssertionError("Modified smoke data escaped the hash check.")
    first_manifest_hash = file_sha256(directory / "manifest.json")
    prepare_dataset(directory, offline=True)
    assert file_sha256(directory / "manifest.json") == first_manifest_hash
    return {
        "source_commit": SOURCE_COMMIT,
        "dataset_manifest_sha256": first_manifest_hash,
        "counts": {split: len(rows) for split, rows in groups.items()},
        "split_ids_disjoint": True,
        "smoke_is_training_only": True,
        "all_8792_reference_targets_parse": True,
        "modified_smoke_checksum_rejected": True,
        "offline_reprepare_identical_manifest": True,
        "smoke_ids": manifest["smoke_ids"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"stage": "gsm8k_data_reward_cpu", "status": "passed" if result.wasSuccessful() else "failed", "unit_tests": result.testsRun}
    if result.wasSuccessful() and arguments.data_dir is not None:
        try:
            report["dataset_checks"] = check_prepared_data(arguments.data_dir)
        except Exception as error:
            report["status"] = "failed"
            report["error"] = f"{type(error).__name__}: {error}"
    if arguments.report is not None:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
