#!/bin/bash --login
#SBATCH --job-name=geora-pilot
#SBATCH --partition=gpu-dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00

set -euo pipefail
geora_method="${1:?Supply geora or lora.}"
geora_initialization="${2:?Supply the untrained GeoRA initialization.}"
geora_data="${3:?Supply the prepared GSM8K directory.}"
geora_checkpoint="${4:?Supply the CPU-validated checkpoint copy.}"
geora_output="${5:?Supply a persistent pilot output directory.}"
geora_stop_step="${6:-10}"
geora_resume="${7:-}"

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
cd "$MYSOFTWARE/geora/code"
geora_python="$MYSOFTWARE/manual/software/geora-environments/py312-rocm633/bin/python"
geora_arguments=(
  --method "$geora_method"
  --init-dir "$geora_initialization"
  --data-dir "$geora_data"
  --checkpoint-dir "$geora_checkpoint"
  --output-dir "$geora_output"
  --config configs/grpo_pilot_plan.json
  --stop-after-step "$geora_stop_step"
  --runtime-limit-seconds 1000
  --checkpoint-every 10
)
if [[ -n "$geora_resume" ]]; then
  geora_arguments+=(--resume "$geora_resume")
fi
srun --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --gres=gpu:1 --gpus-per-task=1 --gpu-bind=closest \
  pytorch-exec "$geora_python" -u scripts/run_grpo_pilot.py "${geora_arguments[@]}"
