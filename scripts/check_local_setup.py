"""Check local imports, checkpoint paths, and saved adapter metadata.

Run with:
    uv run python scripts/check_local_setup.py

Only Safetensors headers are inspected; this does not load the full model
or rerun its SVD, forward pass, or training.
"""

import json
from pathlib import Path
import sys

import huggingface_hub
import torch
import transformers
from safetensors import safe_open


def main() -> None:
    project_directory = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_directory))
    from geora_layers import GeoRALinear, PRECISION_POLICY

    configuration = json.loads(
        (project_directory / "configs/base_model.json").read_text()
    )
    checkpoint_directory = (
        project_directory / configuration["checkpoint_directory"]
    )

    print("Python:", sys.executable)
    print("PyTorch:", torch.__version__)
    print("Transformers:", transformers.__version__)
    print("Hugging Face Hub:", huggingface_hub.__version__)
    print("GeoRA class:", GeoRALinear.__name__)
    print("Checkpoint:", checkpoint_directory.resolve())

    # Confirm the pinned weight provenance and read a representative header.
    record_path = (
        checkpoint_directory
        / ".cache/huggingface/download/model.safetensors.metadata"
    )
    assert record_path.read_text().splitlines()[0] == configuration["model_revision"]
    model_config = json.loads((checkpoint_directory / "config.json").read_text())
    assert model_config["num_hidden_layers"] == 28
    assert model_config["hidden_size"] == 1536
    with safe_open(
        checkpoint_directory / "model.safetensors",
        framework="pt",
        device="cpu",
    ) as checkpoint:
        q_slice = checkpoint.get_slice("model.layers.0.self_attn.q_proj.weight")
        assert q_slice.get_shape() == [1536, 1536]
        assert q_slice.get_dtype() == "BF16"
        print("Base weight tensors:", len(checkpoint.keys()))

    # A fresh clone can build this artifact by running the full-model notebook.
    artifact_directory = project_directory / "outputs/geora_full_model_check"
    adapter_path = artifact_directory / "adapter.safetensors"
    if adapter_path.is_file():
        manifest = json.loads((artifact_directory / "manifest.json").read_text())
        assert manifest["model_repository"] == configuration["model_repository"]
        assert manifest["model_revision"] == configuration["model_revision"]
        assert manifest["precision_policy"] == PRECISION_POLICY
        for setting in ("rank", "alpha", "rho"):
            assert manifest[setting] == configuration["initialization"][setting]
        expected_shapes = {}
        for target in manifest["target_modules"]:
            for factor in ("A0", "B0", "A", "B"):
                shape_key = "A_shape" if factor.startswith("A") else "B_shape"
                expected_shapes[f"{target['name']}.{factor}"] = target[shape_key]
        assert len(manifest["target_modules"]) == 196
        with safe_open(adapter_path, framework="pt", device="cpu") as adapter:
            assert set(adapter.keys()) == set(expected_shapes)
            for name, expected_shape in expected_shapes.items():
                factor_slice = adapter.get_slice(name)
                assert factor_slice.get_shape() == expected_shape
                assert factor_slice.get_dtype() == "F32"
            print("Saved adapter tensors:", len(adapter.keys()))
        print("Saved adapter size (MB):", round(adapter_path.stat().st_size / 1e6, 1))
    else:
        print("Saved adapter: not present; generate it with geora_full_model_check.ipynb.")

    print("Local imports, paths, and checkpoint headers passed.")


if __name__ == "__main__":
    main()
