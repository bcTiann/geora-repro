#!/bin/bash --login
#SBATCH --job-name=geora-native-split
#SBATCH --partition=gpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --time=00:30:00
#SBATCH --export=ALL

# One persistent inference process and one independent single-GCD learner.
# Submit with --account and the environment variables documented in README.md.
set -euo pipefail

: "${MYSOFTWARE:?Set MYSOFTWARE before submitting}"
: "${MYSCRATCH:?Set MYSCRATCH before submitting}"
export SLURM_EXPORT_ENV=ALL
geora_modern="${GEORA_MODERN_DIR:?Set GEORA_MODERN_DIR to the vllm_modern directory}"
export GEORA_ENV_ROOT="${GEORA_ENV_ROOT:-$geora_modern}"
geora_base="${GEORA_BASE_MODEL:?Set GEORA_BASE_MODEL}"

if [[ "${1:-}" == --task ]]; then
    geora_run="${2:?fresh run directory}"
    geora_variant="${3:-geora_mixed}"
    geora_phase="${4:?phase 1 or 2}"
    geora_parent="${5:?parent run directory}"
    geora_port=$((20000 + SLURM_JOB_ID % 20000 + 2 * geora_phase))
    geora_group_port=$((geora_port + 1))
    case "${SLURM_PROCID:?Slurm task rank is required}" in
        0) geora_role=server ;;
        1) geora_role=learner ;;
        *) echo 'This diagnostic requires exactly two Slurm tasks.' >&2; exit 2 ;;
    esac
    # Setonix's manual binding keeps peer GPUs in the step cgroup for HIP IPC.
    # Per-task GPU cgroups break native RCCL; each process still sees one GCD.
    export ROCR_VISIBLE_DEVICES="${SLURM_LOCALID:?local task rank required}"
    exec > "$geora_run/$geora_role.log" 2>&1
    if [[ -z "${ROCR_VISIBLE_DEVICES:-}" || "$ROCR_VISIBLE_DEVICES" == *,* ]]; then
        echo "Each role must be bound to exactly one Slurm-assigned GCD." >&2
        exit 2
    fi
    printf '%s\n' "$ROCR_VISIBLE_DEVICES" > "$geora_run/$geora_role.rocr-visible-devices.txt"
    hostname > "$geora_run/$geora_role.hostname.txt"
    export SINGULARITYENV_OMP_NUM_THREADS=4
    export SINGULARITYENV_MKL_NUM_THREADS=4
    export SINGULARITYENV_OPENBLAS_NUM_THREADS=4
    export SINGULARITYENV_HF_HUB_OFFLINE=1
    export SINGULARITYENV_PYTHONUNBUFFERED=1
    bash "$geora_modern/run_in_container.sh" uv run --locked python \
        "$geora_modern/device_identity.py" "$geora_run/$geora_role-device.json"

    if [[ "$geora_role" == server ]]; then
        export SINGULARITYENV_VLLM_SERVER_DEV_MODE=1
        bash "$geora_modern/run_in_container.sh" uv run --locked vllm serve "$geora_base" \
            --host 127.0.0.1 \
            --port "$geora_port" \
            --served-model-name geora-native-policy \
            --dtype bfloat16 \
            --max-model-len 5121 \
            --max-num-seqs 128 \
            --gpu-memory-utilization 0.5 \
            --enforce-eager \
            --generation-config vllm \
            --logprobs-mode processed_logprobs \
            --max-logprobs -1 \
            --weight-transfer-config '{"backend":"nccl"}' &
        geora_server_pid=$!
        geora_cleanup_server() {
            kill "$geora_server_pid" 2>/dev/null || true
            wait "$geora_server_pid" 2>/dev/null || true
            touch "$geora_run/server.exited"
        }
        trap geora_cleanup_server EXIT INT TERM
        while [[ ! -e "$geora_run/learner.done" ]]; do
            if ! kill -0 "$geora_server_pid" 2>/dev/null; then
                wait "$geora_server_pid" || true
                echo 'The native vLLM server exited before the learner completed.' >&2
                exit 1
            fi
            sleep 2
        done
    else
        # Task 1 is an independent learner, not rank 1 of a training world.
        unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE PMI_RANK PMI_SIZE PMIX_RANK
        unset OMPI_COMM_WORLD_RANK OMPI_COMM_WORLD_LOCAL_RANK
        unset OMPI_COMM_WORLD_SIZE OMPI_COMM_WORLD_LOCAL_SIZE
        bash "$geora_modern/run_in_container.sh" uv run --locked python - \
            "$geora_port" "$geora_run" <<'PY'
from pathlib import Path
import json
import sys
import time
import urllib.error
import urllib.request

port = sys.argv[1]
run = Path(sys.argv[2])
deadline = time.monotonic() + 600
while True:
    if (run / "server.exited").exists():
        raise RuntimeError("The native vLLM server exited before training")
    if time.monotonic() > deadline:
        raise TimeoutError("The native vLLM server was not healthy within 600 seconds")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
            if response.status == 200:
                break
    except (urllib.error.URLError, TimeoutError):
        pass
    time.sleep(2)
server = json.loads((run / "server-device.json").read_text())
learner = json.loads((run / "learner-device.json").read_text())
if server["physical_pci_bdf"] == learner["physical_pci_bdf"]:
    raise RuntimeError("The two roles share the same physical GCD")
if server["hostname"] != learner["hostname"]:
    raise RuntimeError("This bounded diagnostic requires both roles on the same node")
(run / "separate-gcds-verified.json").write_text(json.dumps({"passed": True, "server": server, "learner": learner}, indent=2) + "\n")
print(f"[ready] server PCI={server['physical_pci_bdf']}, learner PCI={learner['physical_pci_bdf']}", flush=True)
PY
        geora_resume=()
        if [[ "$geora_phase" == 2 ]]; then
            geora_resume=(--resume-checkpoint "$geora_parent/phase-1/learner/training/checkpoint-1")
        fi
        bash "$geora_modern/run_in_container.sh" uv run --locked python -u \
            "$geora_modern/02_train_split.py" \
            --variant "$geora_variant" \
            --phase "$geora_phase" \
            --execution-manifest "${GEORA_EXECUTION_MANIFEST:?Set GEORA_EXECUTION_MANIFEST}" --train-data "${GEORA_TRAIN_DATA:?Set GEORA_TRAIN_DATA}" \
            --server-url "http://127.0.0.1:$geora_port" \
            --group-port "$geora_group_port" \
            "${geora_resume[@]}" \
            --output "$geora_run/learner"
        touch "$geora_run/learner.done"
    fi
    exit 0
fi

geora_run="${1:-$geora_modern/validation/native-split-$SLURM_JOB_ID}"
geora_variant="${2:-geora_mixed}"
if [[ "$geora_variant" != geora_mixed && "$geora_variant" != lora ]]; then
    echo 'Variant must be geora_mixed or lora.' >&2
    exit 2
fi
if [[ -e "$geora_run" ]]; then
    echo 'Use a fresh run directory; this diagnostic does not resume.' >&2
    exit 2
fi
mkdir -p -- "$(dirname -- "$geora_run")"
mkdir -- "$geora_run"
printf 'Native two-update diagnostic: %s\n' "$geora_run"

for geora_phase in 1 2; do
    mkdir -- "$geora_run/phase-$geora_phase"
    srun --nodes=1 --ntasks=2 --cpus-per-task=8 \
        --gres=gpu:2 --cpu-bind=cores --kill-on-bad-exit=1 \
        /bin/bash --login "$geora_modern/jobs/modern_split.sbatch" \
        --task "$geora_run/phase-$geora_phase" "$geora_variant" "$geora_phase" "$geora_run"
done
bash "$geora_modern/run_in_container.sh" uv run --locked python \
    "$geora_modern/collect_acceptance.py" --run "$geora_run" --variant "$geora_variant"
printf 'Two-update save/restore diagnostic completed: %s\n' "$geora_run/acceptance.json"
