#!/bin/bash --login
#SBATCH --job-name=geora-logp-replay
#SBATCH --partition=gpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --time=00:20:00
#SBATCH --export=ALL

# No training: two fresh server/learner lifetimes load the same accepted v1.
set -euo pipefail
: "${MYSOFTWARE:?Set MYSOFTWARE before submitting}"
: "${MYSCRATCH:?Set MYSCRATCH before submitting}"
export SLURM_EXPORT_ENV=ALL
geora_modern="${GEORA_MODERN_DIR:?Set GEORA_MODERN_DIR to the vllm_modern directory}"
export GEORA_ENV_ROOT="${GEORA_ENV_ROOT:-$geora_modern}"
geora_base="${GEORA_BASE_MODEL:?Set GEORA_BASE_MODEL}"

if [[ "${1:-}" == --task ]]; then
    geora_run="${2:?fresh lifetime directory}"
    geora_accepted="${3:?completed GeoRA acceptance directory}"
    geora_lifetime="${4:?lifetime index}"
    geora_port=$((20000 + SLURM_JOB_ID % 20000 + 2 * geora_lifetime))
    geora_group_port=$((geora_port + 1))
    case "${SLURM_PROCID:?Slurm task rank required}" in
        0) geora_role=server ;;
        1) geora_role=learner ;;
        *) echo 'Exactly two task roles are required.' >&2; exit 2 ;;
    esac
    # Retain both GPUs in the step cgroup for HIP IPC, but show one per process.
    export ROCR_VISIBLE_DEVICES="${SLURM_LOCALID:?local task rank required}"
    exec > "$geora_run/$geora_role.log" 2>&1
    export SINGULARITYENV_OMP_NUM_THREADS=4
    export SINGULARITYENV_MKL_NUM_THREADS=4
    export SINGULARITYENV_OPENBLAS_NUM_THREADS=4
    export SINGULARITYENV_HF_HUB_OFFLINE=1
    export SINGULARITYENV_PYTHONUNBUFFERED=1
    # Preserve the accepted default kernels; this probe does not enable invariance.
    unset VLLM_BATCH_INVARIANT SINGULARITYENV_VLLM_BATCH_INVARIANT
    bash "$geora_modern/run_in_container.sh" uv run --locked python \
        "$geora_modern/device_identity.py" "$geora_run/$geora_role-device.json"
    if [[ "$geora_role" == server ]]; then
        export SINGULARITYENV_VLLM_SERVER_DEV_MODE=1
        bash "$geora_modern/run_in_container.sh" uv run --locked vllm serve "$geora_base" \
            --host 127.0.0.1 --port "$geora_port" --served-model-name geora-native-policy \
            --dtype bfloat16 --max-model-len 5121 --max-num-seqs 128 \
            --gpu-memory-utilization 0.5 --enforce-eager --generation-config vllm \
            --logprobs-mode processed_logprobs --max-logprobs -1 \
            --weight-transfer-config '{"backend":"nccl"}' &
        geora_server_pid=$!
        geora_cleanup() {
            kill "$geora_server_pid" 2>/dev/null || true
            wait "$geora_server_pid" 2>/dev/null || true
            touch "$geora_run/server.exited"
        }
        trap geora_cleanup EXIT INT TERM
        while [[ ! -e "$geora_run/learner.done" ]]; do
            if ! kill -0 "$geora_server_pid" 2>/dev/null; then
                wait "$geora_server_pid" || true
                echo 'Server exited before probability replay completed.' >&2
                exit 1
            fi
            sleep 2
        done
    else
        unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE PMI_RANK PMI_SIZE PMIX_RANK
        unset OMPI_COMM_WORLD_RANK OMPI_COMM_WORLD_LOCAL_RANK
        unset OMPI_COMM_WORLD_SIZE OMPI_COMM_WORLD_LOCAL_SIZE
        bash "$geora_modern/run_in_container.sh" uv run --locked python - \
            "$geora_port" "$geora_run" <<'PY'
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

port, run = sys.argv[1], Path(sys.argv[2])
deadline = time.monotonic() + 600
while True:
    if (run / "server.exited").exists():
        raise RuntimeError("The probability server exited before becoming ready")
    if time.monotonic() > deadline:
        raise TimeoutError("Probability server was not healthy within 600 seconds")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
            if response.status == 200:
                break
    except (urllib.error.URLError, TimeoutError):
        pass
    time.sleep(2)
server = json.loads((run / "server-device.json").read_text())
learner = json.loads((run / "learner-device.json").read_text())
if server["physical_pci_bdf"] == learner["physical_pci_bdf"] or server["hostname"] != learner["hostname"]:
    raise RuntimeError("Replay requires two distinct physical GCDs on one node")
(run / "separate-gcds-verified.json").write_text(json.dumps({"passed": True, "server": server, "learner": learner}, indent=2) + "\n")
PY
        bash "$geora_modern/run_in_container.sh" uv run --locked python -u \
            "$geora_modern/probability_replay.py" \
            --acceptance-run "$geora_accepted" --output "$geora_run/learner" \
            --execution-manifest "${GEORA_EXECUTION_MANIFEST:?Set GEORA_EXECUTION_MANIFEST}" --train-data "${GEORA_TRAIN_DATA:?Set GEORA_TRAIN_DATA}" \
            --server-url "http://127.0.0.1:$geora_port" --group-port "$geora_group_port"
        touch "$geora_run/learner.done"
    fi
    exit 0
fi

geora_run="${1:?Pass a fresh probability replay output directory}"
geora_accepted="${2:?Pass the completed GeoRA acceptance directory}"
if [[ -e "$geora_run" || ! -f "$geora_accepted/acceptance.json" ]]; then
    echo 'Output must be fresh and the completed acceptance.json must exist.' >&2
    exit 2
fi
mkdir -p -- "$(dirname -- "$geora_run")"
mkdir -- "$geora_run"
geora_lifetime=0
for geora_label in replay-a replay-b; do
    geora_lifetime=$((geora_lifetime + 1))
    mkdir -- "$geora_run/$geora_label"
    srun --nodes=1 --ntasks=2 --cpus-per-task=8 --gres=gpu:2 \
        --cpu-bind=cores --kill-on-bad-exit=1 \
        /bin/bash --login "$geora_modern/jobs/probability_replay.sbatch" \
        --task "$geora_run/$geora_label" "$geora_accepted" "$geora_lifetime"
done
bash "$geora_modern/run_in_container.sh" uv run --locked python \
    "$geora_modern/probability_replay.py" --collect "$geora_run"
printf 'Probability replay complete: %s\n' "$geora_run/probability_replay_summary.json"
