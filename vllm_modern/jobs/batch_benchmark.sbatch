#!/bin/bash --login
#SBATCH --job-name=geora-batch-bench
#SBATCH --partition=gpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --export=ALL

# At most24 diagnostic optimizer updates, not formal400-step training.
set -euo pipefail
: "${MYSOFTWARE:?Set MYSOFTWARE before submitting}"
: "${MYSCRATCH:?Set MYSCRATCH before submitting}"
export SLURM_EXPORT_ENV=ALL
geora_modern="${GEORA_MODERN_DIR:?Set GEORA_MODERN_DIR to the vllm_modern directory}"
export GEORA_ENV_ROOT="${GEORA_ENV_ROOT:-$geora_modern}"
geora_base="${GEORA_BASE_MODEL:?Set GEORA_BASE_MODEL}"
geora_data="${GEORA_TRAIN_DATA:?Set GEORA_TRAIN_DATA}"
geora_initial_manifest="${GEORA_EXECUTION_MANIFEST:?Set GEORA_EXECUTION_MANIFEST}"

if [[ "${1:-}" == --task ]]; then
    geora_case="${2:?case directory}"
    geora_mode="${3:?benchmark or capacity}"
    geora_variant="${4:?variant}"
    geora_micro="${5:?micro}"
    geora_generation="${6:?generation responses}"
    geora_checkpoint="${7:?frozen checkpoint}"
    geora_port=$((20000 + SLURM_JOB_ID % 20000))
    geora_group_port=$((geora_port + 1))
    case "${SLURM_PROCID:?task rank}" in
        0) geora_role=server ;;
        1) geora_role=learner ;;
        *) exit 2 ;;
    esac
    export ROCR_VISIBLE_DEVICES="${SLURM_LOCALID:?local task rank}"
    exec > "$geora_case/$geora_role.log" 2>&1
    export SINGULARITYENV_OMP_NUM_THREADS=4 SINGULARITYENV_MKL_NUM_THREADS=4
    export SINGULARITYENV_OPENBLAS_NUM_THREADS=4 SINGULARITYENV_HF_HUB_OFFLINE=1
    export SINGULARITYENV_PYTHONUNBUFFERED=1
    bash "$geora_modern/run_in_container.sh" uv run --locked python \
        "$geora_modern/device_identity.py" "$geora_case/$geora_role-device.json"
    bash "$geora_modern/run_in_container.sh" uv run --locked python \
        "$geora_modern/gpu_monitor.py" --identity-json "$geora_case/$geora_role-device.json" \
        --output "$geora_case/$geora_role-gpu.jsonl" --stage-file "$geora_case/learner/stage.json" &
    geora_monitor_pid=$!
    geora_server_pid=''
    geora_cleanup() {
        kill "$geora_monitor_pid" 2>/dev/null || true
        wait "$geora_monitor_pid" 2>/dev/null || true
        if [[ -n "$geora_server_pid" ]]; then
            kill "$geora_server_pid" 2>/dev/null || true
            wait "$geora_server_pid" 2>/dev/null || true
            touch "$geora_case/server.exited"
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
        while [[ ! -e "$geora_case/learner.done" ]]; do
            if ! kill -0 "$geora_server_pid" 2>/dev/null; then
                wait "$geora_server_pid" || true
                echo 'Server exited before the learner completed' >&2
                exit 1
            fi
            sleep 2
        done
    else
        unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE PMI_RANK PMI_SIZE PMIX_RANK
        unset OMPI_COMM_WORLD_RANK OMPI_COMM_WORLD_LOCAL_RANK OMPI_COMM_WORLD_SIZE OMPI_COMM_WORLD_LOCAL_SIZE
        bash "$geora_modern/run_in_container.sh" uv run --locked python - "$geora_port" "$geora_case" <<'PY'
import json, sys, time, urllib.request, urllib.error
from pathlib import Path
port, root = sys.argv[1], Path(sys.argv[2])
deadline = time.monotonic() + 600
while True:
    if (root / "server.exited").exists() or time.monotonic() > deadline:
        raise RuntimeError("Server did not become healthy")
    try:
        with urllib.request.urlopen("http://127.0.0.1:" + port + "/health", timeout=3) as response:
            if response.status == 200: break
    except (urllib.error.URLError, TimeoutError): pass
    time.sleep(2)
server = json.loads((root / "server-device.json").read_text())
learner = json.loads((root / "learner-device.json").read_text())
assert server["physical_pci_bdf"] != learner["physical_pci_bdf"]
assert server["hostname"] == learner["hostname"]
(root / "distinct-gcds.json").write_text(json.dumps({"passed": True, "server": server, "learner": learner}) + "\n")
PY
        bash "$geora_modern/run_in_container.sh" uv run --locked python -u \
            "$geora_modern/train_native.py" --mode "$geora_mode" --variant "$geora_variant" \
            --base-model "$geora_base" --train-data "$geora_data" \
            --initialization-checkpoint "$geora_checkpoint" --output "$geora_case/learner" \
            --server-url "http://127.0.0.1:$geora_port" --group-port "$geora_group_port" \
            --micro-batch-size "$geora_micro" --generation-batch-size "$geora_generation" \
            --max-steps 3 --seconds-limit 900
        touch "$geora_case/learner.done"
    fi
    exit 0
fi

geora_root="${1:?fresh output root}"
[[ ! -e "$geora_root" ]] || { echo 'Use a fresh result directory' >&2; exit 2; }
mkdir -p "$geora_root"
cp "$geora_modern/optimization_plan.json" "$geora_root/optimization_plan.json"
geora_case_run() {
    local geora_kind="$1" geora_variant="$2" geora_micro="$3" geora_generation="$4"
    local geora_case="$geora_root/$geora_kind-$geora_variant-g$geora_generation-m$geora_micro"
    local geora_checkpoint
    geora_checkpoint="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["variants"][sys.argv[2]]["checkpoint_path"])' "$geora_initial_manifest" "$geora_variant")"
    mkdir "$geora_case"
    srun --nodes=1 --ntasks=2 --cpus-per-task=8 --gres=gpu:2 --cpu-bind=cores --kill-on-bad-exit=1 \
        /bin/bash --login "$geora_modern/jobs/batch_benchmark.sbatch" --task \
        "$geora_case" "$geora_kind" "$geora_variant" "$geora_micro" "$geora_generation" "$geora_checkpoint"
}
geora_capacity_passed() {
    python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["passed"] else 1)' \
        "$geora_root/capacity-$1-g128-m$2/learner/result.json"
}

for geora_variant in lora geora_mixed; do
    geora_case_run capacity "$geora_variant" 4 128
    if geora_capacity_passed "$geora_variant" 4; then
        geora_case_run capacity "$geora_variant" 8 128
    fi
done
geora_baseline_micro=4
if ! geora_capacity_passed lora 4 || ! geora_capacity_passed geora_mixed 4; then
    geora_baseline_micro=2
    for geora_variant in lora geora_mixed; do
        geora_case_run capacity "$geora_variant" 2 128
        geora_capacity_passed "$geora_variant" 2 || exit 1
    done
fi
for geora_variant in lora geora_mixed; do
    for geora_generation in 32 64 128; do
        geora_case_run benchmark "$geora_variant" "$geora_baseline_micro" "$geora_generation"
    done
    if [[ "$geora_baseline_micro" == 4 ]] && geora_capacity_passed lora 8 && geora_capacity_passed geora_mixed 8; then
        geora_case_run benchmark "$geora_variant" 8 128
    fi
done
touch "$geora_root/all-cases.done"
