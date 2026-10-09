#!/bin/bash
# Submit GPU validation using an existing CPU initialization directory.
# Usage: /bin/bash jobs/submit_gpu_check.sh /absolute/path/to/initialization

set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: /bin/bash jobs/submit_gpu_check.sh INIT_DIR [--diagnose] [--no-follow]" >&2
  exit 2
fi
geora_input_directory="$1"
shift
geora_follow=1
geora_diagnose=0
for geora_option in "$@"; do
  case "$geora_option" in
    --no-follow) geora_follow=0 ;;
    --diagnose) geora_diagnose=1 ;;
    *) echo "Unknown option: $geora_option" >&2; exit 2 ;;
  esac
done

# Resolve the path before changing to the repository directory.
geora_initialization=$(cd -- "$geora_input_directory" && pwd -P)
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

# Keep the argument array nonempty for Bash 3.2 with nounset as well as newer Bash.
geora_batch_arguments=(jobs/geora_training_check.sbatch "$geora_initialization")
if [[ "$geora_diagnose" == 1 ]]; then
  geora_batch_arguments+=(--diagnose-only)
fi
# The initialization path is an explicit batch-script argument, not an env variable.
geora_gpu_id=$(sbatch --parsable --export=ALL --account="$geora_gpu_account" \
  --output="$geora_logs/geora-check-%j.log" \
  "${geora_batch_arguments[@]}")
geora_gpu_id="${geora_gpu_id%%;*}"
echo "Reusing initialization: $geora_initialization"
echo "GPU validation job: $geora_gpu_id"
echo "GPU log: $geora_logs/geora-check-$geora_gpu_id.log"
echo "GPU report: $MYSCRATCH/geora/runs/geora-check-$geora_gpu_id/gpu_checks.json"

if [[ "$geora_follow" == 1 ]]; then
  /bin/bash jobs/watch_gpu_check.sh "$geora_gpu_id" "$geora_logs/geora-check-$geora_gpu_id.log"
fi
