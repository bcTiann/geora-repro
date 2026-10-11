#!/bin/bash --login
#SBATCH --job-name=geora-v-origin
#SBATCH --partition=gpu-dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:02:00

set -euo pipefail
geora_checkpoint="${1:?CPU-validated checkpoint copy}"
geora_source="${2:?Saved S12 raw-logits report}"
geora_probe="${3:?Directory containing the diagnostic scripts}"
module load pytorch/2.7.1-rocm6.3.3
export SLURM_EXPORT_ENV=ALL
export TMPDIR="/tmp/${USER}-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
export HF_HOME="$MYSCRATCH/geora/cache/huggingface"
export HF_HUB_OFFLINE=1 HF_DEACTIVATE_ASYNC_LOAD=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd "$MYSOFTWARE/geora/code"
export PYTHONPATH="$PWD/scripts:$PWD"
geora_python="$MYSOFTWARE/manual/software/geora-environments/py312-rocm633/bin/python"
srun --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --gpus-per-task=1 --gpu-bind=closest \
  pytorch-exec "$geora_python" -u "$geora_probe/check_cache_projection_origin.py" \
  --device cuda --checkpoint-dir "$geora_checkpoint" --source-report "$geora_source" \
  --output-dir "$MYSCRATCH/geora/runs/projection-origin-${SLURM_JOB_ID}"
