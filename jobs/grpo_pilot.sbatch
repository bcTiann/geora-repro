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
geora_config="${8:-configs/grpo_pilot_plan.json}"
geora_runtime="${9:-1000}"
geora_checkpoint_every="${10:-10}"
geora_capacity_check="${11:-skip}"

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
  --config "$geora_config"
  --stop-after-step "$geora_stop_step"
  --runtime-limit-seconds "$geora_runtime"
  --checkpoint-every "$geora_checkpoint_every"
)
if [[ "$geora_capacity_check" == "check" ]]; then
  geora_arguments+=(--capacity-check)
elif [[ "$geora_capacity_check" != "skip" ]]; then
  echo "Capacity option must be check or skip." >&2
  exit 2
fi
if [[ -n "$geora_resume" ]]; then
  geora_arguments+=(--resume "$geora_resume")
fi
srun --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --gres=gpu:1 --gpus-per-task=1 --gpu-bind=closest \
  pytorch-exec "$geora_python" -u scripts/run_grpo_pilot.py "${geora_arguments[@]}"
