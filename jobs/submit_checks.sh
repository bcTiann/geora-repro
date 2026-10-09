#!/bin/bash
# Submit CPU initialization first, then a single GPU check after successful initialization.
# Run from a Setonix login shell: /bin/bash jobs/submit_checks.sh

set -euo pipefail
cd "${MYSOFTWARE:?}/geora/code"
geora_logs="${MYSCRATCH:?}/geora/runs/logs"
mkdir -p "$geora_logs"

geora_cpu_account="${GEORA_CPU_ACCOUNT:-${PAWSEY_PROJECT:?}}"
geora_gpu_account="${GEORA_GPU_ACCOUNT:-${PAWSEY_PROJECT}-gpu}"
geora_cpu_id=$(sbatch --parsable --account="$geora_cpu_account" \
  --output="$geora_logs/initialization-%j.log" jobs/prepare_initialization.sbatch)
geora_cpu_id="${geora_cpu_id%%;*}"
echo "CPU initialization job: $geora_cpu_id"

export GEORA_INIT_DIR="$MYSCRATCH/geora/initializations/$geora_cpu_id"
geora_gpu_id=$(sbatch --parsable --account="$geora_gpu_account" \
  --dependency="afterok:$geora_cpu_id" --kill-on-invalid-dep=yes \
  --output="$geora_logs/geora-check-%j.log" jobs/geora_training_check.sbatch)
geora_gpu_id="${geora_gpu_id%%;*}"
echo "GPU validation job: $geora_gpu_id"
echo "The GPU job waits for successful CPU initialization; neither job keeps an interactive shell."
echo "CPU log: $geora_logs/initialization-$geora_cpu_id.log"
echo "GPU log: $geora_logs/geora-check-$geora_gpu_id.log"
echo "GPU report: $MYSCRATCH/geora/runs/geora-check-$geora_gpu_id/gpu_checks.json"
