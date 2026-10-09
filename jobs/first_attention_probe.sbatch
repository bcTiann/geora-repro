#!/bin/bash --login
#SBATCH --job-name=geora-attn0-probe
#SBATCH --partition=gpu-dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:01:00

set -euo pipefail
geora_initialization="${1:?Supply the saved initialization directory.}"
module load pytorch/2.7.1-rocm6.3.3
export SLURM_EXPORT_ENV=ALL
export TMPDIR="/tmp/${USER}-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
export HF_HUB_OFFLINE=1
export HF_HOME="$MYSCRATCH/geora/cache/huggingface"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
cd "$MYSOFTWARE/geora/code"
geora_python="$MYSOFTWARE/manual/software/geora-environments/py312-rocm633/bin/python"
geora_report="$MYSCRATCH/geora/runs/attention-probe-${SLURM_JOB_ID}/attention_probe.json"
srun --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --gres=gpu:1 --gpus-per-task=1 --gpu-bind=closest \
  pytorch-exec "$geora_python" -u scripts/probe_first_attention.py \
  --init-dir "$geora_initialization" --output "$geora_report"
