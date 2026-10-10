"""Prepare pinned original GSM8K data and score explicit numeric final answers.

This uses only the Python standard library. Raw data and derived splits belong
under an ignored datasets directory (or Setonix scratch), never in the source
repository. The final-answer rule is our smoke-test choice, not an author-
verified GeoRA prompt or grading protocol.
"""

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
from urllib.request import urlopen


SOURCE_REPOSITORY = "openai/grade-school-math"
SOURCE_COMMIT = "3101c7d5072418e28b9008a6636bde82a006892c"
SPLIT_SEED = 20261010
VALIDATION_COUNT = 128
SMOKE_COUNT = 16
SOURCE_FILES = {
    "train.jsonl": {
        "path": "grade_school_math/data/train.jsonl",
        "sha256": "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465",
        "rows": 7473,
    },
    "test.jsonl": {
        "path": "grade_school_math/data/test.jsonl",
        "sha256": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
        "rows": 1319,
    },
    "LICENSE": {
        "path": "LICENSE",
        "sha256": "86bbb73e855821d7c401912fd4bf82e34313e6e3b6fd6f909f2b6cc9e209a53b",
    },
}
ANSWER_RULE = "single_final_hash_marker_exact_decimal_v1"
PROMPT_INSTRUCTION = (
    "Solve the following math problem. Explain your reasoning, then end your "
    "answer with a final line in the form #### <number>. Write only the numeric "
    "answer after ####, without units."
)

# Commas must be valid thousands separators. Scientific notation, fractions,
# units and expressions are rejected; no fallback to the last reasoning number.
NUMBER = re.compile(
    r"[+-]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?|\.[0-9]+)\Z"
)


def canonical_decimal(value: Decimal) -> str:
    """Stable display string; comparison itself uses exact Decimal equality."""
    if value == 0:
        return "0"
    numeric_text = format(value, "f")
    return numeric_text.rstrip("0").rstrip(".") if "." in numeric_text else numeric_text


def extract_final_answer(completion: str) -> Decimal | None:
    """Return the explicit final number, or None for absent/ambiguous formats.

    There must be exactly one #### marker, and everything after it except
    whitespace must be a single supported decimal number. Intermediate numbers
    before the marker do not matter. More than one marker is rejected rather
    than arbitrarily choosing an answer. The marker need not start a line.
    """
    if completion.count("####") != 1:
        return None
    numeric_text = completion.split("####", 1)[1].strip()
    if NUMBER.fullmatch(numeric_text) is None:
        return None
    return Decimal(numeric_text.replace(",", ""))


def score_answer(completion: str, target: str) -> dict:
    """Reward is 1 exactly when the parsed final number equals the target."""
    if NUMBER.fullmatch(target) is None:
        raise ValueError("The reference target must be a supported numeric string.")
    expected = Decimal(target.replace(",", ""))
    predicted = extract_final_answer(completion)
    return {
        "reward": float(predicted is not None and predicted == expected),
        "predicted_answer": None if predicted is None else canonical_decimal(predicted),
        "target_answer": canonical_decimal(expected),
        "answer_rule": ANSWER_RULE,
    }


def build_prompt(question: str) -> str:
    """Only the question enters the model prompt; target/reasoning stay outside."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("A nonempty question string is required.")
    return f"{PROMPT_INSTRUCTION}\n\nQuestion:\n{question}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_original_rows(path: Path, split: str) -> list[dict]:
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8").split("\n")):
        if not line:
            continue
        original = json.loads(line)
        if set(original) != {"question", "answer"}:
            raise ValueError(f"Unexpected source fields at {split}:{index}.")
        target = extract_final_answer(original["answer"])
        if target is None:
            raise ValueError(f"Unparseable reference answer at {split}:{index}.")
        rows.append({
            "id": f"{split}:{index:05d}",
            "source_split": split,
            "source_index": index,
            "question": original["question"],
            "target": canonical_decimal(target),
        })
    expected_count = SOURCE_FILES[f"{split}.jsonl"]["rows"]
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} {split} rows, got {len(rows)}.")
    if len({row["question"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate questions in source {split}.")
    return rows


def split_train_rows(rows: list[dict], *, seed: int, validation_count: int) -> tuple[list[dict], list[dict]]:
    """SHA256 ordering avoids random-library/version dependence in split IDs."""
    if validation_count < 1 or validation_count + SMOKE_COUNT > len(rows):
        raise ValueError("Validation size must leave at least 16 training questions.")
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest(),
    )
    return ordered[validation_count:], ordered[:validation_count]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def prepare_dataset(directory: Path, *, offline: bool = False, seed: int = SPLIT_SEED,
                    validation_count: int = VALIDATION_COUNT) -> dict:
    directory = directory.resolve()
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        required_policy = {
            "source_commit": SOURCE_COMMIT,
            "split_seed": seed,
            "validation_count": validation_count,
            "answer_rule": ANSWER_RULE,
            "prompt_instruction": PROMPT_INSTRUCTION,
        }
        if any(existing.get(key) != value for key, value in required_policy.items()):
            raise RuntimeError("Existing dataset policy differs; prepare into a new directory.")
    raw_directory = directory / "raw"
    raw_directory.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for filename, specification in SOURCE_FILES.items():
        source_url = f"https://raw.githubusercontent.com/{SOURCE_REPOSITORY}/{SOURCE_COMMIT}/{specification['path']}"
        destination = raw_directory / filename
        if not destination.exists():
            if offline:
                raise FileNotFoundError(f"Offline source file absent: {destination}")
            temporary = destination.with_name(destination.name + ".part")
            try:
                with urlopen(source_url, timeout=60) as response, temporary.open("wb") as output:
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        output.write(block)
                if file_sha256(temporary) != specification["sha256"]:
                    raise RuntimeError(f"Pinned source checksum failed: {filename}")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        if file_sha256(destination) != specification["sha256"]:
            raise RuntimeError(f"Pinned source checksum failed: {filename}")
        provenance[filename] = {
            "url": source_url,
            "relative_path": f"raw/{filename}",
            "sha256": specification["sha256"],
            "bytes": destination.stat().st_size,
        }
    original_train = read_original_rows(raw_directory / "train.jsonl", "train")
    original_test = read_original_rows(raw_directory / "test.jsonl", "test")
    train, validation = split_train_rows(original_train, seed=seed, validation_count=validation_count)
    groups = {"train": train, "validation": validation, "test": original_test, "smoke": train[:SMOKE_COUNT]}
    split_metadata = {}
    for split, rows in groups.items():
        path = directory / f"{split}.jsonl"
        write_jsonl(path, rows)
        split_metadata[split] = {
            "relative_path": path.name,
            "rows": len(rows),
            "sha256": file_sha256(path),
        }
    manifest = {
        "dataset": "GSM8K",
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "source_license": "MIT",
        "source_files": provenance,
        "split_seed": seed,
        "split_rule": "sort source train IDs by SHA256(seed:ID); first validation_count are validation; remainder are train",
        "validation_count": validation_count,
        "smoke_rule": "first 16 remaining training IDs; no reward-dependent selection",
        "smoke_ids": [row["id"] for row in groups["smoke"]],
        "validation_ids": [row["id"] for row in validation],
        "answer_rule": ANSWER_RULE,
        "prompt_instruction": PROMPT_INSTRUCTION,
        "test_usage": "held out for final GSM8K evaluation; not smoke candidates or validation",
        "splits": split_metadata,
    }
    # Changing data policy in an existing directory requires an explicit new
    # directory so an earlier experiment's provenance is not silently replaced.
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("Existing dataset manifest differs; prepare into a new directory.")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_prepared_rows(directory: Path, split: str) -> list[dict]:
    """Check persisted provenance/hash before exposing rows to a run."""
    if split not in {"train", "validation", "test", "smoke"}:
        raise ValueError(f"Unknown split: {split}")
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["source_commit"] != SOURCE_COMMIT or manifest["answer_rule"] != ANSWER_RULE:
        raise RuntimeError("Prepared dataset provenance or answer rule differs.")
    metadata = manifest["splits"][split]
    path = directory / metadata["relative_path"]
    if file_sha256(path) != metadata["sha256"]:
        raise RuntimeError(f"Prepared {split} checksum failed.")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]
    if len(rows) != metadata["rows"]:
        raise RuntimeError(f"Prepared {split} row count differs.")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--offline", action="store_true", help="Require already downloaded and checksum-valid raw sources.")
    arguments = parser.parse_args()
    manifest = prepare_dataset(arguments.data_dir, offline=arguments.offline)
    print(json.dumps({
        "status": "passed",
        "data_directory": str(arguments.data_dir.resolve()),
        "source_commit": SOURCE_COMMIT,
        "split_counts": {split: item["rows"] for split, item in manifest["splits"].items()},
        "smoke_ids": manifest["smoke_ids"],
        "answer_rule": ANSWER_RULE,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
