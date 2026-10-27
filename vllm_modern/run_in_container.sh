#!/bin/bash --login
# Open the project runtime or execute a command; uv owns the persistent .venv.
set -euo pipefail
geora_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$geora_script_dir/runtime_config.sh"
if [[ "$#" -eq 0 ]]; then
    set -- /bin/bash --noprofile --norc
fi
if [[ ! -f "$geora_image" || ! -f "$geora_image.sha256" ]]; then
    echo 'Run setup_setonix.sh before using this runtime.' >&2
    exit 2
fi
mkdir -p -- "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR"
module load singularity/4.1.0-nompi
unset SINGULARITYENV_LD_LIBRARY_PATH SINGULARITYENV_LD_PRELOAD SINGULARITY_ROCM
unset SINGULARITYENV_HIP_VISIBLE_DEVICES SINGULARITYENV_CUDA_VISIBLE_DEVICES
geora_syslibs="$geora_environment_root/.system-libs/root"
if [[ -f "$geora_syslibs/.validated-modern-runtime" ]]; then
    export SINGULARITYENV_LD_LIBRARY_PATH="$geora_syslibs/usr/lib/x86_64-linux-gnu:$geora_syslibs/usr/lib/x86_64-linux-gnu/openmpi/lib:/opt/rocm/lib:/opt/rocm/lib64"
fi
export SINGULARITYENV_ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-}"
if [[ -n "${GEORA_BASE_MODEL:-}" ]]; then
    export SINGULARITYENV_GEORA_BASE_MODEL="$GEORA_BASE_MODEL"
fi
export SINGULARITYENV_PYTHONNOUSERSITE=1
export SINGULARITYENV_UV_CACHE_DIR="$UV_CACHE_DIR"
export SINGULARITYENV_UV_PYTHON_INSTALL_DIR="$UV_PYTHON_INSTALL_DIR"
export SINGULARITYENV_UV_PROJECT_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT"
export SINGULARITYENV_UV_LINK_MODE=copy
# --cleanenv requires explicit forwarding into the engine and its workers.
export SINGULARITYENV_XDG_CACHE_HOME="$XDG_CACHE_HOME"
export SINGULARITYENV_TRITON_CACHE_DIR="$TRITON_CACHE_DIR"
export SINGULARITYENV_VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT"
export SINGULARITYENV_TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR"
export SINGULARITYENV_VLLM_ROCM_USE_AITER=0
export SINGULARITYENV_TOKENIZERS_PARALLELISM=false
export SINGULARITYENV_UV_BIN="$geora_uv_bin"
export SINGULARITYENV_PREPEND_PATH="$(dirname -- "$geora_uv_bin")"
if [[ "$1" == uv ]]; then
    shift
    set -- "$geora_uv_bin" "$@"
fi
# Do not inject host ROCm libraries into the new userspace runtime.
exec singularity exec --cleanenv \
    --bind "$MYSOFTWARE:$MYSOFTWARE" \
    --bind "$MYSCRATCH:$MYSCRATCH" \
    --pwd "$geora_modern_dir" "$geora_image" "$@"
