#!/bin/bash --login
#SBATCH --job-name=geora-cpu-init
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=00:30:00

set -euo pipefail
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
geora_initialization="$MYSCRATCH/geora/initializations/${SLURM_JOB_ID}"

srun --nodes=1 --ntasks=1 --cpus-per-task=8 \
  pytorch-exec "$geora_python" scripts/prepare_geora_initialization.py \
  --output-dir "$geora_initialization"
