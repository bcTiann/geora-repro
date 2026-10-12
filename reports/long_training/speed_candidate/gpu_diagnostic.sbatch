#!/bin/bash --login
#SBATCH --job-name=geora-speed-proof
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00

set -euo pipefail
module load pytorch/2.7.1-rocm6.3.3
export SLURM_EXPORT_ENV=ALL
export TMPDIR="/tmp/${USER}-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
export HF_HOME="$MYSCRATCH/geora/cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DEACTIVATE_ASYNC_LOAD=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1
geora_python="$MYSOFTWARE/manual/software/geora-environments/py312-rocm633/bin/python"
geora_bundle="$MYSCRATCH/geora/tmp/speed-candidate-cpu-20261012"
geora_output="$MYSCRATCH/geora/runs/speed-candidate-gpu-${SLURM_JOB_ID}"
cd "$geora_bundle"
srun --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --gres=gpu:1 --gpus-per-task=1 --gpu-bind=closest \
  pytorch-exec "$geora_python" -u run_candidates_gpu_bounded.py \
  --checkpoint-dir "$MYSCRATCH/geora/checkpoints/geora_base" \
  --pinned-base-dir "$MYSCRATCH/geora/checkpoints/geora_base" \
  --init-dir "$MYSCRATCH/geora/initializations/login-20261009T115357Z-2781659" \
  --data-dir "$MYSCRATCH/geora/datasets/gsm8k" \
  --geora-pilot-dir "$MYSCRATCH/geora/runs/long-pilot-geora" \
  --lora-pilot-dir "$MYSCRATCH/geora/runs/long-pilot-lora" \
  --config "$geora_bundle/production_grpo_long_pilot.json" \
  --output-dir "$geora_output" \
  --runtime-limit-seconds 1200
