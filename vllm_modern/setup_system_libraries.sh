#!/bin/bash
# Download signed Ubuntu runtime packages into a private directory; never install them.
set -euo pipefail

geora_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
geora_syslibs_dir="$geora_project_dir/.system-libs"
geora_apt_dir="$geora_syslibs_dir/apt"
geora_extract_dir="$geora_syslibs_dir/root"
geora_evidence_dir="$geora_project_dir/evidence"
geora_python="$geora_project_dir/.venv/bin/python"

if [[ ! -r /etc/os-release ]]; then
    echo 'This helper must run inside the pinned Ubuntu 24.04 container.' >&2
    exit 2
fi
source /etc/os-release
if [[ "${ID:-}" != ubuntu || "${VERSION_ID:-}" != 24.04 ]]; then
    echo 'This helper supports only the pinned Ubuntu 24.04 runtime.' >&2
    exit 2
fi
if [[ "$(dpkg --print-architecture)" != amd64 ]]; then
    echo 'The modern ROCm project requires the amd64 container.' >&2
    exit 2
fi
if [[ ! -x "$geora_python" ]]; then
    echo 'Create the uv project environment before resolving system libraries.' >&2
    exit 2
fi
for geora_command in apt-get apt-cache dpkg-deb sha256sum ldd; do
    command -v "$geora_command" >/dev/null
done

mkdir -p "$geora_apt_dir/state/lists/partial"
mkdir -p "$geora_apt_dir/cache/archives/partial"
mkdir -p "$geora_apt_dir/log"
mkdir -p "$geora_evidence_dir"

# APT reads this file after the container defaults. Clear hooks that could write
# into the image's system cache; package signature checks remain enabled.
cat > "$geora_apt_dir/private.conf" <<EOF
Dir::State "$geora_apt_dir/state";
Dir::State::lists "$geora_apt_dir/state/lists";
Dir::State::status "/var/lib/dpkg/status";
Dir::Cache "$geora_apt_dir/cache";
Dir::Cache::archives "$geora_apt_dir/cache/archives";
Dir::Log "$geora_apt_dir/log";
Debug::NoLocking "true";
APT::Sandbox::User "$(id -un)";
APT::Get::List-Cleanup "false";
Acquire::AllowInsecureRepositories "false";
Acquire::AllowDowngradeToInsecureRepositories "false";
APT::Get::AllowUnauthenticated "false";
#clear APT::Update::Post-Invoke;
#clear APT::Update::Post-Invoke-Success;
#clear DPkg::Post-Invoke;
EOF

geora_apt_options=(-c "$geora_apt_dir/private.conf")
if [[ "${GEORA_SYSTEM_LIBS_REUSE_CACHE:-0}" == 1 ]]; then
    # Reuse only the exact archives recorded by a previous signed APT download.
    # This is useful when changing the base image without repeating downloads.
    "$geora_python" - "$geora_evidence_dir/system-libraries-packages.tsv" "$geora_apt_dir/cache/archives" <<'PY'
import csv
import hashlib
import pathlib
import sys

manifest = pathlib.Path(sys.argv[1])
if not manifest.is_file():
    raise SystemExit("No previous package receipt is available for cache reuse.")
with manifest.open() as stream:
    rows = list(csv.DictReader(stream, delimiter="\t"))
if not rows:
    raise SystemExit("The previous package receipt is empty.")
for row in rows:
    archive = pathlib.Path(row["archive"])
    if hashlib.sha256(archive.read_bytes()).hexdigest() != row["sha256"]:
        raise SystemExit(f"Cached archive failed verification: {archive}")
recorded = {pathlib.Path(row["archive"]).resolve() for row in rows}
actual = {path.resolve() for path in pathlib.Path(sys.argv[2]).glob("*.deb")}
if actual != recorded:
    raise SystemExit("Cached archives differ from the signed-download receipt.")
PY
else
apt-get "${geora_apt_options[@]}" update 2>&1 | tee "$geora_evidence_dir/system-libraries-update.log"

# Check candidates before downloading so a renamed or unavailable package fails
# with a clear diagnostic. libopenmpi3t64 owns libmpi_cxx.so.40 on Noble.
for geora_package in libopenmpi3t64 libdw1t64; do
    geora_candidate="$(apt-cache "${geora_apt_options[@]}" policy "$geora_package" | awk '/Candidate:/ {print $2; exit}')"
    if [[ -z "$geora_candidate" || "$geora_candidate" == '(none)' ]]; then
        echo "No signed Ubuntu candidate for $geora_package." >&2
        exit 3
    fi
done
apt-get "${geora_apt_options[@]}" --simulate --no-install-recommends \
    install libopenmpi3t64 libdw1t64 \
    2>&1 | tee "$geora_evidence_dir/system-libraries-plan.log"
apt-get "${geora_apt_options[@]}" --download-only --no-install-recommends --yes \
    install libopenmpi3t64 libdw1t64 \
    2>&1 | tee "$geora_evidence_dir/system-libraries-download.log"
fi

geora_manifest_tmp="$geora_evidence_dir/system-libraries-packages.tsv.tmp"
printf 'package\tversion\tarchitecture\tbytes\tsha256\tarchive\n' > "$geora_manifest_tmp"
geora_excluded_tmp="$geora_evidence_dir/system-libraries-excluded.tsv.tmp"
printf 'package\tversion\tsha256\treason\n' > "$geora_excluded_tmp"
geora_extract_archives=()
shopt -s nullglob
geora_archives=("$geora_apt_dir/cache/archives/"*.deb)
if [[ "${#geora_archives[@]}" -eq 0 ]]; then
    echo 'APT downloaded no packages; refusing an unrecorded runtime.' >&2
    exit 3
fi
for geora_deb in "${geora_archives[@]}"; do
    geora_package="$(dpkg-deb --field "$geora_deb" Package)"
    geora_version="$(dpkg-deb --field "$geora_deb" Version)"
    geora_architecture="$(dpkg-deb --field "$geora_deb" Architecture)"
    case "$geora_package" in
        libc6|libc-bin|libc6-dev|libc-dev-bin|locales|locales-all)
            echo "Refusing private replacement of the container C runtime: $geora_package." >&2
            exit 3
            ;;
    esac
    if [[ "$geora_architecture" != amd64 && "$geora_architecture" != all ]]; then
        echo "Unexpected archive architecture: $geora_architecture." >&2
        exit 3
    fi
    geora_sha256="$(sha256sum "$geora_deb" | awk '{print $1}')"
    geora_bytes="$(stat -c %s "$geora_deb")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$geora_package" "$geora_version" "$geora_architecture" \
        "$geora_bytes" "$geora_sha256" "$geora_deb" >> "$geora_manifest_tmp"
    # Ubuntu's UCX dependency closure can pull ROCm 5.7. The pinned container
    # owns the entire GPU runtime; those archives must never enter our LD path.
    case "$geora_package" in
        libamd*|libhsa*|libhsakmt*|libhip*|libroc*|librccl*|libmiopen*|miopen*|rocm*|hip*|libllvm*|llvm-*|clang-*|libclang*|lld-*)
            printf '%s\t%s\t%s\t%s\n' "$geora_package" "$geora_version" \
                "$geora_sha256" 'GPU or LLVM runtime is supplied by the pinned ROCm container' >> "$geora_excluded_tmp"
            ;;
        *)
            geora_extract_archives+=("$geora_deb")
            ;;
    esac
done

# Extract only after validating every archive. Package maintainer scripts are
# never executed; dpkg's system database and the container remain unchanged.
geora_stage_dir="$(mktemp -d "$geora_syslibs_dir/root.pending.XXXXXXXX")"
for geora_deb in "${geora_extract_archives[@]}"; do
    dpkg-deb --extract "$geora_deb" "$geora_stage_dir"
done
mv "$geora_manifest_tmp" "$geora_evidence_dir/system-libraries-packages.tsv"
mv "$geora_excluded_tmp" "$geora_evidence_dir/system-libraries-excluded.tsv"

geora_lib_dir="$geora_stage_dir/usr/lib/x86_64-linux-gnu"
geora_clean_ld_path=''
IFS=: read -r -a geora_old_ld_paths <<< "${LD_LIBRARY_PATH:-}"
for geora_ld_path in "${geora_old_ld_paths[@]}"; do
    if [[ -n "$geora_ld_path" && "$geora_ld_path" != "$geora_syslibs_dir"/* ]]; then
        geora_clean_ld_path="${geora_clean_ld_path:+$geora_clean_ld_path:}$geora_ld_path"
    fi
done
export LD_LIBRARY_PATH="$geora_lib_dir:$geora_lib_dir/openmpi/lib:/opt/rocm/lib:/opt/rocm/lib64${geora_clean_ld_path:+:$geora_clean_ld_path}"
for geora_library in libmpi_cxx.so.40 libmpi.so.40 libdw.so.1; do
    if [[ ! -e "$geora_lib_dir/$geora_library" ]]; then
        echo "The extracted runtime is missing $geora_library." >&2
        exit 4
    fi
done

geora_torch_lib="$geora_project_dir/.venv/lib/python3.12/site-packages/torch/lib"
geora_ldd_log="$geora_evidence_dir/system-libraries-ldd.log"
: > "$geora_ldd_log"
for geora_library in libtorch_cpu.so libtorch_python.so libtorch_hip.so; do
    if [[ ! -f "$geora_torch_lib/$geora_library" ]]; then
        echo "The uv Torch installation is missing $geora_library." >&2
        exit 4
    fi
    printf '\n%s\n' "$geora_library" >> "$geora_ldd_log"
    ldd "$geora_torch_lib/$geora_library" >> "$geora_ldd_log" 2>&1
done
if rg -q '=> not found' "$geora_ldd_log" 2>/dev/null || grep -q '=> not found' "$geora_ldd_log"; then
    cat "$geora_ldd_log" >&2
    echo 'Critical Torch dependencies remain unresolved.' >&2
    exit 4
fi

"$geora_python" - "$geora_project_dir" "$geora_stage_dir" <<'PY'
import datetime
import hashlib
import json
import pathlib
import re
import sys

project = pathlib.Path(sys.argv[1])
evidence = project / "evidence"
sources = []
source_paths = [pathlib.Path("/etc/apt/sources.list")]
source_paths.extend(sorted(pathlib.Path("/etc/apt/sources.list.d").glob("*")))
for path in source_paths:
    if path.is_file():
        data = path.read_bytes()
        sources.append({"path": str(path), "sha256": hashlib.sha256(data).hexdigest()})
manifest = evidence / "system-libraries-packages.tsv"
ldd_text = (evidence / "system-libraries-ldd.log").read_text()
hsa_paths = re.findall(r"libhsa-runtime64\.so\.\d+\s+=>\s+(\S+)", ldd_text)
if not hsa_paths:
    raise SystemExit("ldd did not identify the ROCm HSA runtime; refusing promotion.")
rocm_root = pathlib.Path("/opt/rocm").resolve()
version_files = [rocm_root / ".info" / "version", rocm_root / ".info" / "version-dev"]
rocm_versions = {
    str(path): path.read_text().strip()
    for path in version_files
    if path.is_file()
}
if not any(value.startswith("7.2.3") for value in rocm_versions.values()):
    raise SystemExit(f"Expected the pinned ROCm 7.2.3 runtime, found {rocm_versions}")
for value in hsa_paths:
    if not pathlib.Path(value).resolve().is_relative_to(rocm_root):
        raise SystemExit(f"HSA must resolve inside the pinned /opt/rocm runtime: {value}")
excluded = evidence / "system-libraries-excluded.tsv"
report = {
    "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "method": "signed_apt_download_only_then_dpkg_deb_extract",
    "requested_packages": ["libopenmpi3t64", "libdw1t64"],
    "private_root": str(project / ".system-libs" / "root"),
    "package_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "excluded_manifest_sha256": hashlib.sha256(excluded.read_bytes()).hexdigest(),
    "gpu_library_source": str(rocm_root),
    "rocm_version_files": rocm_versions,
    "hsa_library_paths": sorted(set(hsa_paths)),
    "apt_sources": sources,
    "critical_torch_ldd_missing": [],
    "staging_root": sys.argv[2],
    "system_packages_installed": False,
    "gpu_tested": False,
}
(evidence / "system-libraries.json").write_text(json.dumps(report, indent=2) + "\n")
PY

# Preserve our earlier extraction for diagnosis, then publish a fresh root.
# Excluded GPU libraries cannot survive from a previous attempt.
if [[ -e "$geora_extract_dir" ]]; then
    geora_archive_dir="$geora_syslibs_dir/root.previous.$(date -u +%Y%m%dT%H%M%SZ).$$"
    mv "$geora_extract_dir" "$geora_archive_dir"
    printf '%s\n' "$geora_archive_dir" > "$geora_evidence_dir/system-libraries-previous-root.txt"
fi
mv "$geora_stage_dir" "$geora_extract_dir"
touch "$geora_extract_dir/.validated-modern-runtime"
geora_lib_dir="$geora_extract_dir/usr/lib/x86_64-linux-gnu"
echo 'Private system libraries extracted and critical Torch linkage checked.'
echo "Runtime library path: $geora_lib_dir:$geora_lib_dir/openmpi/lib"
