"""Initialize all 196 GeoRA targets on CPU and save their FP32 factors.

Run this with bounded CPU threads; it performs two SVDs per target layer.
On this project, initialization is permitted on the Setonix login node.
"""

import argparse
import json
import os
import socket
from pathlib import Path
import time

import torch
from safetensors.torch import save_file

from geora_check_utils import CheckReport, fresh_fp32_model, pinned_configuration, validate_manifest
from geora_layers import export_adapter_state, install_geora


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    report = CheckReport(arguments.output_dir, "initialization_checks.json")
    try:
        run(arguments.output_dir, report)
    except Exception as error:
        report.data.update(status="failed", error=str(error))
        report.write()
        raise


def run(output_directory, report) -> None:
    started = time.perf_counter()
    configuration, checkpoint_directory = pinned_configuration()
    if (output_directory / "adapter.safetensors").exists():
        raise FileExistsError("Use a new output directory to preserve the existing initialization.")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    print("CPU initialization host:", socket.gethostname(), flush=True)
    print("CPU threads:", torch.get_num_threads(), flush=True)
    torch.manual_seed(0)
    model = fresh_fp32_model(checkpoint_directory)

    def progress(completed, total, name, statistics):
        print(f"[{completed}/{total}] {name}; elapsed={time.perf_counter()-started:.1f}s", flush=True)

    manifest = install_geora(
        model,
        **configuration["initialization"],
        progress_callback=progress,
    )
    manifest.update(
        model_repository=configuration["model_repository"],
        model_revision=configuration["model_revision"],
        torch_version=torch.__version__,
        optimizer_steps=0,
        initialization_slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        initialization_host=socket.gethostname(),
        initialization_cpu_threads=torch.get_num_threads(),
    )
    validate_manifest(manifest, configuration)
    report.require("all_196_expected_targets", True, target_count=len(manifest["target_modules"]))

    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    report.require("trainable_parameter_count", trainable_count == 18464768, count=trainable_count)
    maximum_error = max(
        target["statistics"]["initial_effective_weight_relative_error"]
        for target in manifest["target_modules"]
    )
    report.require("fp32_initial_effective_weights", maximum_error <= 1e-6, max_relative_error=maximum_error)

    state = export_adapter_state(model)
    report.require("fp32_finite_factors", all(
        factor.dtype == torch.float32 and torch.isfinite(factor).all().item()
        for factor in state.values()
    ), tensor_count=len(state))
    initial_matches = all(
        torch.equal(state[f"{target['name']}.{factor}"], state[f"{target['name']}.{factor}0"])
        for target in manifest["target_modules"]
        for factor in ("A", "B")
    )
    report.require("current_factors_equal_initial_factors", initial_matches)
    save_file(state, str(output_directory / "adapter.safetensors"))
    (output_directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report.data.update(
        stage="cpu_geora_initialization",
        hostname=socket.gethostname(),
        cpu_threads=torch.get_num_threads(),
        optimizer_steps=0,
        elapsed_seconds=time.perf_counter() - started,
    )
    report.finish()
    print("GeoRA CPU initialization passed.", flush=True)
    print("Initialization:", output_directory, flush=True)


if __name__ == "__main__":
    main()
