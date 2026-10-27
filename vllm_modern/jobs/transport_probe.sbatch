#!/bin/bash --login
#SBATCH --job-name=geora-transport-probe
#SBATCH --partition=gpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --time=00:08:00
#SBATCH --export=ALL

set -euo pipefail
: "${MYSOFTWARE:?Set MYSOFTWARE before submitting}"
: "${MYSCRATCH:?Set MYSCRATCH before submitting}"
export SLURM_EXPORT_ENV=ALL
geora_modern="${GEORA_MODERN_DIR:?Set GEORA_MODERN_DIR to the vllm_modern directory}"
export GEORA_ENV_ROOT="${GEORA_ENV_ROOT:-$geora_modern}"

if [[ "${1:-}" == --task ]]; then
    geora_mode="${2:?mode}"
    geora_output="${3:?mode output directory}"
    geora_port="${4:?metadata port}"
    geora_preset="${5:-default}"
    exec > "$geora_output/rank-${SLURM_PROCID:?}.log" 2>&1
    if [[ "$geora_preset" == manual-binding ]]; then
        # Setonix documented IPC-safe binding: retain the step's two-GCD cgroup,
        # select one GCD per process via ROCr rather than per-task cgroups.
        export ROCR_VISIBLE_DEVICES="${SLURM_LOCALID:?local task rank required}"
    fi
    if [[ -z "${ROCR_VISIBLE_DEVICES:-}" || "$ROCR_VISIBLE_DEVICES" == *,* ]]; then
        echo 'Each transport probe task requires one Slurm-assigned GCD.' >&2
        exit 2
    fi
    unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE PMI_RANK PMI_SIZE PMIX_RANK
    unset OMPI_COMM_WORLD_RANK OMPI_COMM_WORLD_LOCAL_RANK OMPI_COMM_WORLD_SIZE OMPI_COMM_WORLD_LOCAL_SIZE
    export SINGULARITYENV_OMP_NUM_THREADS=4
    export SINGULARITYENV_MKL_NUM_THREADS=4
    export SINGULARITYENV_OPENBLAS_NUM_THREADS=4
    export SINGULARITYENV_PYTHONUNBUFFERED=1
    export SINGULARITYENV_NCCL_DEBUG=INFO
    export SINGULARITYENV_NCCL_DEBUG_SUBSYS=INIT,ENV,P2P,SHM,NET,GRAPH
    case "$geora_preset" in
        default) ;;
        legacy-ipc) export SINGULARITYENV_HSA_ENABLE_IPC_MODE_LEGACY=1 ;;
        no-p2p) export SINGULARITYENV_NCCL_P2P_DISABLE=1 ;;
        task-sharing) ;;
        manual-binding) ;;
        *) echo 'Unknown transport diagnostic preset.' >&2; exit 2 ;;
    esac
    exec bash "$geora_modern/run_in_container.sh" uv run --locked python -u \
        "$geora_modern/transport_probe.py" --mode "$geora_mode" --port "$geora_port" \
        --rank "$SLURM_PROCID" --timeout 60 --output "$geora_output"
fi

geora_run="${1:-$MYSCRATCH/geora-trl-paper/transport-probes/probe-$SLURM_JOB_ID}"
geora_preset="${2:-default}"
if [[ -e "$geora_run" ]]; then
    echo 'Use a fresh transport probe directory.' >&2
    exit 2
fi
mkdir -p -- "$geora_run"
geora_index=0
for geora_mode in torch pynccl; do
    mkdir -- "$geora_run/$geora_mode"
    geora_port=$((20000 + SLURM_JOB_ID % 20000 + 2 * geora_index))
    geora_index=$((geora_index + 1))
    # Keep both modes independent: a failed torch group does not suppress the
    # subsequent model-free PyNccl test. Every step is externally bounded.
    geora_sharing=()
    geora_gpu_binding=(--gres=gpu:2 --gpus-per-task=1 --gpu-bind=closest)
    if [[ "$geora_preset" == task-sharing ]]; then
        geora_sharing=(--gres-flags=allow-task-sharing)
    elif [[ "$geora_preset" == manual-binding ]]; then
        geora_gpu_binding=(--gres=gpu:2 --cpu-bind=cores)
    fi
    if timeout --signal=TERM --kill-after=10s 150s \
        srun --nodes=1 --ntasks=2 --cpus-per-task=8 --time=00:03:00 --wait=10 \
        "${geora_gpu_binding[@]}" --kill-on-bad-exit=1 \
        "${geora_sharing[@]}" \
        /bin/bash --login "$geora_modern/jobs/transport_probe.sbatch" \
        --task "$geora_mode" "$geora_run/$geora_mode" "$geora_port" "$geora_preset"; then
        geora_exit=0
    else
        geora_exit=$?
    fi
    printf '%s\n' "$geora_exit" > "$geora_run/$geora_mode/srun-exit-code.txt"
done
python3 - "$geora_run" "$geora_preset" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
result = {"scope": "Model-free two-rank transport isolation; explicit single-variable preset", "preset": sys.argv[2], "modes": {}}
for mode in ("torch", "pynccl"):
    path = root / mode
    ranks = {}
    for rank in (0, 1):
        report = path / f"rank-{rank}.json"
        ranks[str(rank)] = json.loads(report.read_text()) if report.is_file() else {"passed": False, "error": "No rank JSON; inspect timeout and rank log"}
    exit_code = int((path / "srun-exit-code.txt").read_text())
    result["modes"][mode] = {"passed": exit_code == 0 and all(row["passed"] for row in ranks.values()),
                              "srun_exit_code": exit_code, "ranks": ranks}
result["passed"] = all(row["passed"] for row in result["modes"].values())
(root / "transport-summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps({"run": str(root), "passed": result["passed"], "modes": {key: value["passed"] for key, value in result["modes"].items()}}, indent=2))
sys.exit(0 if result["passed"] else 1)
PY
