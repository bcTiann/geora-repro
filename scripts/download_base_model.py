"""Download the pinned GeoRA base model, including its tokenizer.

Run from the repository root:
    uv run python scripts/download_base_model.py

On Setonix, use its container environment and pass --output-dir to select
the checkpoint directory under scratch. This script does not load a model.
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Checkpoint destination; defaults to checkpoints/geora_base.",
    )
    arguments = parser.parse_args()

    # Read the recorded model revision rather than downloading a moving main.
    project_directory = Path(__file__).resolve().parents[1]
    configuration_path = project_directory / "configs/base_model.json"
    configuration = json.loads(configuration_path.read_text())
    output_directory = arguments.output_dir
    if output_directory is None:
        output_directory = project_directory / configuration["checkpoint_directory"]
    output_directory = output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    # This pinned 1.5B checkpoint has one weight file and six configuration
    # or tokenizer files. No optimizer state or unrelated models are needed.
    filenames = (
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors",
    )
    print("Repository:", configuration["model_repository"])
    print("Revision:", configuration["model_revision"])
    print("Destination:", output_directory)
    for filename in filenames:
        print("Downloading:", filename, flush=True)
        downloaded_path = hf_hub_download(
            repo_id=configuration["model_repository"],
            revision=configuration["model_revision"],
            filename=filename,
            local_dir=output_directory,
        )
        print("Saved:", downloaded_path, flush=True)

    # The full-model notebook reads this Hugging Face provenance record too.
    record_path = (
        output_directory
        / ".cache/huggingface/download/model.safetensors.metadata"
    )
    recorded_revision = record_path.read_text().splitlines()[0]
    if recorded_revision != configuration["model_revision"]:
        raise RuntimeError("Downloaded weight revision differs from the configuration.")
    print("Pinned model download finished.")


if __name__ == "__main__":
    main()
