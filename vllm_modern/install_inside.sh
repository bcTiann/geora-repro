#!/bin/bash
# Install only. No Slurm submission, GPU probe, inference, or training is run here.
set -euo pipefail
mkdir -p evidence
"$UV_BIN" python install 3.12 --managed-python
geora_python="$("$UV_BIN" python find 3.12 --managed-python)"
"$geora_python" check_environment.py --platform-only --report evidence/platform.json
"$UV_BIN" sync --locked --python 3.12 --managed-python
"$UV_BIN" pip check --python .venv/bin/python
if [[ ! -f .system-libs/root/.validated-modern-runtime ]]; then
    if [[ -f evidence/system-libraries-packages.tsv ]]; then
        export GEORA_SYSTEM_LIBS_REUSE_CACHE=1
    fi
    bash setup_system_libraries.sh
fi
geora_syslibs="$PWD/.system-libs/root/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="$geora_syslibs:$geora_syslibs/openmpi/lib:/opt/rocm/lib:/opt/rocm/lib64"
"$UV_BIN" run python check_environment.py \
    --installed --report evidence/installed.json
"$UV_BIN" run python check_imports.py
"$UV_BIN" --version > evidence/uv-version.txt
echo 'Modern uv environment installed. GPU inference and training have not been tested.'
