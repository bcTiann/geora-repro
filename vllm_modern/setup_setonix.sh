#!/bin/bash --login
# Prepare the pinned userspace and install the independent uv project, without a GPU job.
set -euo pipefail
geora_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$geora_script_dir/runtime_config.sh"
module load singularity/4.1.0-nompi
mkdir -p "$geora_runtime_root" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"
export SINGULARITY_CACHEDIR="$geora_runtime_root/cache"
if [[ ! -f "$geora_image" ]]; then
    geora_pull_tmp="$(mktemp -d "/tmp/${USER}-geora-modern-pull.XXXXXX")"
    export SINGULARITY_TMPDIR="$geora_pull_tmp"
    trap 'rm -rf -- "$geora_pull_tmp"' EXIT
    geora_pull_cpus="$(/usr/bin/python3 -c 'import os; print(",".join(str(x) for x in sorted(os.sched_getaffinity(0))[:2]))')"
    taskset -c "$geora_pull_cpus" singularity pull "$geora_image" "$geora_image_uri"
    sha256sum "$geora_image" > "$geora_image.sha256"
fi
if [[ ! -f "$geora_image.sha256" ]]; then
    echo 'Image exists without its receipt; refusing to assume it is the pinned runtime.' >&2
    exit 2
fi
sha256sum --check "$geora_image.sha256"
bash "$geora_modern_dir/run_in_container.sh" /bin/bash install_inside.sh
