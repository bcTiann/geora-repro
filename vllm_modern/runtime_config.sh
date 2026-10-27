#!/bin/bash
# The container supplies Linux/ROCm libraries; uv owns all Python dependencies.
geora_modern_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
geora_uv_software_root="${MYSOFTWARE:?Run on Setonix}"
geora_scratch_root="${MYSCRATCH:?Run on Setonix}/geora-trl-paper"
geora_runtime_root="$geora_scratch_root/vllm-modern-runtime-20261026"
geora_image="$geora_runtime_root/rocm7.2.3-ubuntu24.04-complete.sif"
geora_image_uri='docker://rocm/dev-ubuntu-24.04@sha256:ec1b59bf75ec1122e7a091c0be82301ba12458b038499ef96a5c115876bd78d2'
geora_uv_bin="${GEORA_UV_BIN:-$(command -v uv || true)}"
if [[ -z "$geora_uv_bin" ]]; then
    geora_uv_bin="$MYSOFTWARE/uv/bin/uv"
fi
if [[ ! -x "$geora_uv_bin" ]]; then
    echo 'uv was not found. Set GEORA_UV_BIN to its absolute path.' >&2
    return 2
fi
export UV_CACHE_DIR="$geora_scratch_root/cache/uv-vllm-modern"
export UV_PYTHON_INSTALL_DIR="$geora_scratch_root/environments/uv-python-modern"
# Keep source snapshots isolated while explicitly reusing the accepted lockfile environment.
geora_environment_root="${GEORA_ENV_ROOT:-$geora_modern_dir}"
export UV_PROJECT_ENVIRONMENT="$geora_environment_root/.venv"
export UV_LINK_MODE=copy
export PYTHONNOUSERSITE=1

# Runtime compiler caches must not use the small home filesystem.
export XDG_CACHE_HOME="$geora_scratch_root/cache/vllm-modern"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
