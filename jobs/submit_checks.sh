#!/bin/bash
# Run bounded CPU initialization on the login node, then submit one GPU check.
# Run from a Setonix login shell: /bin/bash jobs/submit_checks.sh

set -euo pipefail
cd "${MYSOFTWARE:?}/geora/code"
geora_logs="${MYSCRATCH:?}/geora/runs/logs"
mkdir -p "$geora_logs"

geora_init_threads="${GEORA_INIT_THREADS:-2}"
if ! [[ "$geora_init_threads" =~ ^[1-9][0-9]*$ ]]; then
  echo "GEORA_INIT_THREADS must be a positive integer." >&2
  exit 2
fi

geora_run_id="login-$(date -u +%Y%m%dT%H%M%SZ)-$$"
geora_cpu_log="$geora_logs/initialization-${geora_run_id}.log"
export GEORA_INIT_DIR="$MYSCRATCH/geora/initializations/$geora_run_id"
export TMPDIR="$MYSCRATCH/geora/tmp/$geora_run_id"
mkdir -p "$TMPDIR"

module load pytorch/2.7.1-rocm6.3.3
export HF_HOME="$MYSCRATCH/geora/cache/huggingface"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="$geora_init_threads"
export MKL_NUM_THREADS="$geora_init_threads"
export OPENBLAS_NUM_THREADS="$geora_init_threads"
geora_python="$MYSOFTWARE/manual/software/geora-environments/py312-rocm633/bin/python"

echo "CPU initialization: login node, $geora_init_threads threads (no Slurm allocation)"
echo "Initialization directory: $GEORA_INIT_DIR"
echo "CPU log: $geora_cpu_log"

# pipefail preserves Python's failure status even when tee successfully saves the log.
pytorch-exec "$geora_python" -u scripts/prepare_geora_initialization.py \
  --output-dir "$GEORA_INIT_DIR" 2>&1 | tee "$geora_cpu_log"

# Submit only after the synchronous initialization command has succeeded.
geora_gpu_account="${GEORA_GPU_ACCOUNT:-${PAWSEY_PROJECT:?}-gpu}"
geora_gpu_id=$(sbatch --parsable --account="$geora_gpu_account" \
  --output="$geora_logs/geora-check-%j.log" jobs/geora_training_check.sbatch)
geora_gpu_id="${geora_gpu_id%%;*}"
echo "GPU validation job: $geora_gpu_id"
echo "GPU log: $geora_logs/geora-check-$geora_gpu_id.log"
echo "GPU report: $MYSCRATCH/geora/runs/geora-check-$geora_gpu_id/gpu_checks.json"
