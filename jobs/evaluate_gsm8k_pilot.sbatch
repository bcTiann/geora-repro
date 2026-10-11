#!/bin/bash --login
#SBATCH --job-name=geora-validation
#SBATCH --partition=gpu-dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00

set -euo pipefail
geora_method="${1:?Supply base, geora, or lora.}"
geora_data="${2:?Supply the prepared GSM8K directory.}"
geora_checkpoint="${3:?Supply the CPU-validated checkpoint copy.}"
geora_output="${4:?Supply a validation output directory.}"
geora_adapter="${5:-}"
geora_initialization="${6:-}"
geora_question_limit="${7:-128}"
geora_resume="${8:-fresh}"
geora_config="${9:-configs/grpo_pilot_plan.json}"
geora_runtime="${10:-1000}"
if [[ "$geora_resume" != "fresh" && "$geora_resume" != "resume" ]]; then
  echo "The final argument must be fresh or resume." >&2
  exit 2
fi

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
  --data-dir "$geora_data"
  --checkpoint-dir "$geora_checkpoint"
  --output-dir "$geora_output"
  --config "$geora_config"
  --question-limit "$geora_question_limit"
  --runtime-limit-seconds "$geora_runtime"
)
if [[ -n "$geora_adapter" ]]; then
  geora_arguments+=(--adapter-checkpoint "$geora_adapter")
fi
if [[ -n "$geora_initialization" ]]; then
  geora_arguments+=(--init-dir "$geora_initialization")
fi
if [[ "$geora_resume" == "resume" ]]; then
  geora_arguments+=(--resume)
fi
srun --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --gres=gpu:1 --gpus-per-task=1 --gpu-bind=closest \
  pytorch-exec "$geora_python" -u scripts/evaluate_gsm8k_pilot.py "${geora_arguments[@]}"
