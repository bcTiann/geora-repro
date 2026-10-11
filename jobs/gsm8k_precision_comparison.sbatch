#!/bin/bash --login
#SBATCH --job-name=geora-precision
#SBATCH --partition=gpu-dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00

set -euo pipefail
module load pytorch/2.7.1-rocm6.3.3
export SLURM_EXPORT_ENV=ALL
export TMPDIR="/tmp/${USER}-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
export HF_HOME="$MYSCRATCH/geora/cache/huggingface"
export HF_HUB_OFFLINE=1 HF_DEACTIVATE_ASYNC_LOAD=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$MYSOFTWARE/geora/code"
geora_python="$MYSOFTWARE/manual/software/geora-environments/py312-rocm633/bin/python"
geora_output="$MYSCRATCH/geora/runs/gsm8k-precision-${SLURM_JOB_ID}"
mkdir -p "$geora_output"

for geora_dtype in bf16 fp32; do
  srun --nodes=1 --ntasks=1 --cpus-per-task=8 \
    --gres=gpu:1 --gpus-per-task=1 --gpu-bind=closest \
    pytorch-exec "$geora_python" -u scripts/check_geora_precision_benchmark.py \
    --repo "$MYSOFTWARE/geora/code" --device cuda --dtype "$geora_dtype" \
    --checkpoint-dir "$MYSCRATCH/geora/checkpoints/geora_base" \
    --init-dir "$MYSCRATCH/geora/initializations/login-20261009T115357Z-2781659" \
    --source-report "$MYSCRATCH/geora/runs/precision-benchmark-source-50660901.json" \
    --batch 4 --response-limit 64 --warmup 2 --repeats 3 \
    --output-dir "$geora_output/benchmark-$geora_dtype"
done

srun --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --gres=gpu:1 --gpus-per-task=1 --gpu-bind=closest \
  pytorch-exec "$geora_python" -u scripts/compare_gsm8k_precision.py \
  --device cuda --checkpoint-dir "$MYSCRATCH/geora/checkpoints/geora_base" \
  --data-dir "$MYSCRATCH/geora/datasets/gsm8k" --output-dir "$geora_output/answers"
