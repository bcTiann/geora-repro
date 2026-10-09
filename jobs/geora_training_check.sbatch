#!/bin/bash --login
#SBATCH --job-name=geora-gpu-check
#SBATCH --partition=gpu-dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00

set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Submit with jobs/submit_gpu_check.sh INITIALIZATION_DIRECTORY [--diagnose]." >&2
  exit 2
fi
# Capture the argument before loading modules or entering the container.
geora_initialization="$1"
geora_mode_arguments=(--init-dir "$geora_initialization")
if [[ $# == 2 ]]; then
  if [[ "$2" != --diagnose-only ]]; then
    echo "Unknown check mode: $2" >&2
    exit 2
  fi
  geora_mode_arguments+=(--diagnose-only)
fi
echo "Initialization directory: $geora_initialization"
export SLURM_EXPORT_ENV=ALL
module load pytorch/2.7.1-rocm6.3.3
export TMPDIR="/tmp/${USER}-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
export HF_HOME="$MYSCRATCH/geora/cache/huggingface"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

cd "$MYSOFTWARE/geora/code"
geora_python="$MYSOFTWARE/manual/software/geora-environments/py312-rocm633/bin/python"
geora_output="$MYSCRATCH/geora/runs/geora-check-${SLURM_JOB_ID}"

srun --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --gres=gpu:1 --gpus-per-task=1 --gpu-bind=closest \
  pytorch-exec "$geora_python" -u scripts/check_geora_training.py \
  "${geora_mode_arguments[@]}" \
  --output-dir "$geora_output"
