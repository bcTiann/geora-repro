#!/bin/bash --login
#SBATCH --job-name=geora-schedule-accept
#SBATCH --partition=gpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --time=00:30:00
#SBATCH --export=ALL

# Explicit submission only: sbatch jobs/schedule_acceptance.sbatch NEW_RUN MICRO GEN
# Four fresh server/learner lifetimes: two methods, two real update/save phases each.
set -euo pipefail
: "${MYSOFTWARE:?Set MYSOFTWARE before submitting}"
: "${MYSCRATCH:?Set MYSCRATCH before submitting}"
export SLURM_EXPORT_ENV=ALL
geora_modern="${GEORA_MODERN_DIR:?Set GEORA_MODERN_DIR to the vllm_modern directory}"
export GEORA_ENV_ROOT="${GEORA_ENV_ROOT:-$geora_modern}"
geora_base="${GEORA_BASE_MODEL:?Set GEORA_BASE_MODEL}"
geora_data="${GEORA_TRAIN_DATA:?Set GEORA_TRAIN_DATA}"
geora_manifest="${GEORA_EXECUTION_MANIFEST:?Set GEORA_EXECUTION_MANIFEST}"

if [[ "${1:-}" == --task ]]; then
    geora_phase_run="${2:?fresh phase directory}"
    geora_variant="${3:?variant}"
    geora_phase="${4:?phase}"
    geora_parent="${5:?method run directory}"
    geora_micro="${6:?micro batch}"
    geora_generation="${7:?generation batch}"
    geora_checkpoint="${8:?frozen initialization checkpoint}"
    geora_method_index="${9:?method index}"
    geora_port=$((20000 + SLURM_JOB_ID % 19000 + 4 * geora_method_index + 2 * geora_phase))
    geora_group_port=$((geora_port + 1))
    case "${SLURM_PROCID:?task rank}" in
        0) geora_role=server ;;
        1) geora_role=learner ;;
        *) echo 'Exactly two Slurm tasks are required.' >&2; exit 2 ;;
    esac
    # Verified Setonix manual binding: step sees both peer GCDs for HIP IPC,
    # each role sees one GCD. Do not add per-task GPU cgroups/gpu-bind.
    export ROCR_VISIBLE_DEVICES="${SLURM_LOCALID:?local task rank}"
    exec > "$geora_phase_run/$geora_role.log" 2>&1
    export SINGULARITYENV_OMP_NUM_THREADS=4 SINGULARITYENV_MKL_NUM_THREADS=4
    export SINGULARITYENV_OPENBLAS_NUM_THREADS=4 SINGULARITYENV_HF_HUB_OFFLINE=1
    export SINGULARITYENV_PYTHONUNBUFFERED=1
    bash "$geora_modern/run_in_container.sh" uv run --locked python \
        "$geora_modern/device_identity.py" "$geora_phase_run/$geora_role-device.json"
    bash "$geora_modern/run_in_container.sh" uv run --locked python \
        "$geora_modern/gpu_monitor.py" --identity-json "$geora_phase_run/$geora_role-device.json" \
        --output "$geora_phase_run/$geora_role-gpu.jsonl" --stage-file "$geora_phase_run/learner/stage.json" &
    geora_monitor_pid=$!
    geora_server_pid=''
    geora_cleanup() {
        kill "$geora_monitor_pid" 2>/dev/null || true
        wait "$geora_monitor_pid" 2>/dev/null || true
        if [[ -n "$geora_server_pid" ]]; then
            kill "$geora_server_pid" 2>/dev/null || true
            wait "$geora_server_pid" 2>/dev/null || true
            touch "$geora_phase_run/server.exited"
        fi
    }
    trap geora_cleanup EXIT INT TERM
    if [[ "$geora_role" == server ]]; then
        export SINGULARITYENV_VLLM_SERVER_DEV_MODE=1
        bash "$geora_modern/run_in_container.sh" uv run --locked vllm serve "$geora_base" \
            --host 127.0.0.1 --port "$geora_port" --served-model-name geora-native-policy \
            --dtype bfloat16 --max-model-len 5121 --max-num-seqs 128 \
            --gpu-memory-utilization 0.5 --enforce-eager --generation-config vllm \
            --logprobs-mode processed_logprobs --max-logprobs -1 \
            --weight-transfer-config '{"backend":"nccl"}' &
        geora_server_pid=$!
        while [[ ! -e "$geora_phase_run/learner.done" ]]; do
            if ! kill -0 "$geora_server_pid" 2>/dev/null; then
                wait "$geora_server_pid" || true
                echo 'Server exited before the learner completed.' >&2
                exit 1
            fi
            sleep 2
        done
    else
        unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE PMI_RANK PMI_SIZE PMIX_RANK
        unset OMPI_COMM_WORLD_RANK OMPI_COMM_WORLD_LOCAL_RANK OMPI_COMM_WORLD_SIZE OMPI_COMM_WORLD_LOCAL_SIZE
        bash "$geora_modern/run_in_container.sh" uv run --locked python - "$geora_port" "$geora_phase_run" <<'PY'
import json, sys, time, urllib.error, urllib.request
from pathlib import Path
port, run = sys.argv[1], Path(sys.argv[2])
deadline = time.monotonic() + 180
while True:
    if (run / "server.exited").exists() or time.monotonic() > deadline:
        raise RuntimeError("Server did not become healthy")
    try:
        with urllib.request.urlopen("http://127.0.0.1:" + port + "/health", timeout=3) as response:
            if response.status == 200: break
    except (urllib.error.URLError, TimeoutError): pass
    time.sleep(2)
server = json.loads((run / "server-device.json").read_text())
learner = json.loads((run / "learner-device.json").read_text())
if (server["physical_pci_bdf"] == learner["physical_pci_bdf"] or server["hostname"] != learner["hostname"]
        or server["visible_device_count"] != 1 or learner["visible_device_count"] != 1):
    raise RuntimeError("Expected separate one-GCD roles on the same node")
(run / "separate-gcds-verified.json").write_text(json.dumps({"passed": True, "server": server, "learner": learner}) + "\n")
PY
        geora_load=()
        if [[ "$geora_phase" == 1 ]]; then
            geora_load=(--initialization-checkpoint "$geora_checkpoint")
        else
            geora_load=(--resume-checkpoint "$geora_parent/phase-1/learner/training/checkpoint-1")
        fi
        bash "$geora_modern/run_in_container.sh" uv run --locked python -u \
            "$geora_modern/accept_schedule.py" --variant "$geora_variant" --phase "$geora_phase" \
            --micro-batch-size "$geora_micro" --generation-batch-size "$geora_generation" \
            --execution-manifest "$geora_manifest" --train-data "$geora_data" \
            --server-url "http://127.0.0.1:$geora_port" --group-port "$geora_group_port" \
            --seconds-limit 360 "${geora_load[@]}" --output "$geora_phase_run/learner"
        touch "$geora_phase_run/learner.done"
    fi
    exit 0
fi

if [[ "$#" -ne 3 ]]; then
    echo 'Usage: sbatch jobs/schedule_acceptance.sbatch NEW_RUN MICRO GEN' >&2
    exit 2
fi
geora_run="$1"
geora_micro="$2"
geora_generation="$3"
case "$geora_micro" in 2|4|8) ;; *) echo 'MICRO must be a measured candidate: 2, 4 or8.' >&2; exit 2 ;; esac
case "$geora_generation" in 32|64|128) ;; *) echo 'GEN must be32, 64 or128.' >&2; exit 2 ;; esac
[[ "$geora_run" == /* && ! -e "$geora_run" ]] || { echo 'Use a fresh absolute NEW_RUN path.' >&2; exit 2; }
mkdir -p -- "$(dirname -- "$geora_run")"
mkdir -- "$geora_run"
cp "$geora_modern/jobs/schedule_acceptance.sbatch" "$geora_run/job-source.sbatch"
cp "$geora_modern/formal_config.json" "$geora_run/scientific-config.json"
geora_method_index=0
for geora_variant in lora geora_mixed; do
    geora_parent="$geora_run/$geora_variant"
    geora_checkpoint="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["variants"][sys.argv[2]]["checkpoint_path"])' "$geora_manifest" "$geora_variant")"
    mkdir -- "$geora_parent"
    for geora_phase in 1 2; do
        mkdir -- "$geora_parent/phase-$geora_phase"
        # Each step gets a new server/communicator and an independent learner.
        srun --nodes=1 --ntasks=2 --cpus-per-task=8 --gres=gpu:2 --cpu-bind=cores \
            --time=00:07:00 --kill-on-bad-exit=1 \
            /bin/bash --login "$geora_modern/jobs/schedule_acceptance.sbatch" --task \
            "$geora_parent/phase-$geora_phase" "$geora_variant" "$geora_phase" "$geora_parent" \
            "$geora_micro" "$geora_generation" "$geora_checkpoint" "$geora_method_index"
    done
    touch "$geora_parent/both-phases.done"
    geora_method_index=$((geora_method_index + 1))
done
touch "$geora_run/all-phases.done"
printf 'All four bounded phases completed; collect artifacts before accepting the schedule: %s\n' "$geora_run"
