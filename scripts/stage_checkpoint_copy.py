"""Stream the pinned checkpoint to a temporary directory without a GPU.

Use only when shared-scratch reads are slow. This adds a temporary full copy;
the original data stays on scratch. Validate it with check_staged_checkpoint.py
before passing it to the continued-test job, and remove it after testing.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geora_check_utils import pinned_configuration


def run(output):
    _, source = pinned_configuration()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Use an empty temporary output directory")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    record = {"status": "running", "source": str(source.resolve()), "staged": str(output), "files": {}}
    for file in source.iterdir():
        if not file.is_file():
            continue
        digest = hashlib.sha256()
        size = 0
        previous = 0
        destination = output / file.name
        temporary = output / (file.name + ".part")
        with file.open("rb") as incoming, temporary.open("wb") as outgoing:
            while chunk := incoming.read(16 * 2**20):
                outgoing.write(chunk)
                digest.update(chunk)
                size += len(chunk)
                if size - previous >= 256 * 2**20:
                    print("STAGING", file.name, round(size / 2**20), "MiB", flush=True)
                    previous = size
        temporary.replace(destination)
        record["files"][file.name] = {"bytes": size, "source_sha256": digest.hexdigest()}
    record.update(status="copied", elapsed_seconds=time.perf_counter()-started)
    (output / "staging_record.json").write_text(json.dumps(record, indent=2) + "\n")
    print("Copied; run CPU checksum/value validation next:", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args().output_dir)
