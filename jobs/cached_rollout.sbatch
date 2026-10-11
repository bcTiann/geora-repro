#!/bin/bash --login
#SBATCH --job-name=geora-cache-diag
#SBATCH --partition=gpu-dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:02:00

set -euo pipefail
geora_initialization="${1:?Supply the untrained GeoRA initialization.}"
geora_checkpoint="${2:?Supply the CPU-validated checkpoint copy.}"
module load pytorch/2.7.1-rocm6.3.3
export SLURM_EXPORT_ENV=ALL
export TMPDIR="/tmp/${USER}-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
export HF_HOME="$MYSCRATCH/geora/cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DEACTIVATE_ASYNC_LOAD=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd "$MYSOFTWARE/geora/code"
geora_python="$MYSOFTWARE/manual/software/geora-environments/py312-rocm633/bin/python"
srun --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --gres=gpu:1 --gpus-per-task=1 --gpu-bind=closest \
  pytorch-exec "$geora_python" -u scripts/check_cached_rollout_gpu.py \
  --init-dir "$geora_initialization" --checkpoint-dir "$geora_checkpoint" \
  --output-dir "$MYSCRATCH/geora/runs/cached-rollout-${SLURM_JOB_ID}" \
  --max-new-tokens 16
