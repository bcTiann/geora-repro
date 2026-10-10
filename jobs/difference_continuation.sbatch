#!/bin/bash --login
#SBATCH --job-name=geora-diff-next
#SBATCH --partition=gpu-dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:03:00

set -euo pipefail
geora_initialization="${1:?Supply the saved initialization directory.}"
module load pytorch/2.7.1-rocm6.3.3
export SLURM_EXPORT_ENV=ALL
export TMPDIR="/tmp/${USER}-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
export HF_HOME="$MYSCRATCH/geora/cache/huggingface"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
cd "$MYSOFTWARE/geora/code"
geora_python="$MYSOFTWARE/manual/software/geora-environments/py312-rocm633/bin/python"
geora_output="$MYSCRATCH/geora/runs/difference-next-${SLURM_JOB_ID}"
srun --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --gres=gpu:1 --gpus-per-task=1 --gpu-bind=closest \
  pytorch-exec "$geora_python" -u scripts/check_difference_continuation.py \
  --init-dir "$geora_initialization" --output-dir "$geora_output"
