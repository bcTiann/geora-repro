#!/bin/bash
# Submit GPU validation using an existing CPU initialization directory.
# Usage: /bin/bash jobs/submit_gpu_check.sh /absolute/path/to/initialization

set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: /bin/bash jobs/submit_gpu_check.sh INITIALIZATION_DIRECTORY" >&2
  exit 2
fi

# Resolve the path before changing to the repository directory.
geora_initialization=$(cd -- "$1" && pwd -P)
for geora_file in adapter.safetensors manifest.json initialization_checks.json; do
  if [[ ! -f "$geora_initialization/$geora_file" ]]; then
    echo "Missing initialization file: $geora_initialization/$geora_file" >&2
    exit 2
  fi
done

cd "${MYSOFTWARE:?}/geora/code"
geora_logs="${MYSCRATCH:?}/geora/runs/logs"
mkdir -p "$geora_logs"
geora_gpu_account="${GEORA_GPU_ACCOUNT:-${PAWSEY_PROJECT:?}-gpu}"

# The initialization path is an explicit batch-script argument, not an env variable.
geora_gpu_id=$(sbatch --parsable --export=ALL --account="$geora_gpu_account" \
  --output="$geora_logs/geora-check-%j.log" \
  jobs/geora_training_check.sbatch "$geora_initialization")
geora_gpu_id="${geora_gpu_id%%;*}"
echo "Reusing initialization: $geora_initialization"
echo "GPU validation job: $geora_gpu_id"
echo "GPU log: $geora_logs/geora-check-$geora_gpu_id.log"
echo "GPU report: $MYSCRATCH/geora/runs/geora-check-$geora_gpu_id/gpu_checks.json"
