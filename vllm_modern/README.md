# TRL + vLLM experiments

This directory contains the independent GeoRA and LoRA GRPO experiments using a separate vLLM rollout server and learner on two Setonix MI250X GCDs. The implementation includes dense policy transfer (including GeoRA residual buffers), resident weight hash checks, token log-probability comparison, and checkpoint recovery checks. The selected configuration and its limits are described in the repository's main README. The 400-step experiment has not started.

The diagnostic runs used Python 3.12, ROCm 7.2.3, vLLM `0.32.0+rocm723`, and TRL `1.17.0.dev0` at commit `c6eac19f63be27498f2a7c8108d1e8a4dffde23f`. Versions are pinned in `pyproject.toml` and `uv.lock`; the exact runtime depends on the target cluster and container. The scripts are research code, not a portable one-command setup. Check cluster policy and paths before submitting any job.

## Inputs and cluster configuration

- Obtain the pinned base model and GSM8K training data separately; model weights and datasets are not included.
- For the historical two-step and batch acceptance jobs, prepare an execution manifest and matching checkpoint artifacts in the format expected by the scripts. These artifacts are not included; the benchmark starts from historical checkpoint 25, not a fresh base model.
- On Setonix, set `MYSOFTWARE`, `MYSCRATCH`, `GEORA_MODERN_DIR` (this directory), `GEORA_BASE_MODEL`, `GEORA_TRAIN_DATA`, and `GEORA_EXECUTION_MANIFEST` to your own locations. `GEORA_ENV_ROOT` may point to a shared pinned environment; otherwise the source directory is used. The standalone generation exercise uses `GEORA_BASE_MODEL`.
- Paths passed to containerized commands must be accessible through the `MYSOFTWARE` or `MYSCRATCH` bind mounts.
- The Slurm examples have no account specified. Pass your own with `sbatch --account=<project-account> ...` and confirm resource limits and job arguments. The scripts export the submission environment. They write new outputs and must be reviewed before use.
- `setup_setonix.sh` prepares the pinned container and dependencies without submitting a GPU job. It requires `MYSOFTWARE` and `MYSCRATCH`; it downloads software and writes an environment. Consult the script before running it.

`formal_config.json` records the selected experimental configuration; `launchable` remains false. Formal training requires fresh-base initialization and frozen source, data, prompt, evaluator, and budget manifests. The scripts and configuration in this public preview have been adapted to remove personal paths and have not themselves been rerun on a GPU after that adaptation.

The existing offline Python tests cover contracts and collection logic; they do not replace cluster validation. The original aggregate observations are from historical runs using the original source snapshot, not from a fresh run of this sanitized version.
