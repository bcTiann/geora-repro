"""Check the modern ROCm environment without probing or initializing a GPU."""

import argparse
import importlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import sys


EXPECTED_VERSIONS = {
    "vllm": "0.32.0+rocm723",
    "torch": "2.13.0+git733fca1",
}
MINIMUM_GLIBC = (2, 39)


def inspect_platform():
    """Require the platform encoded by the Linux ROCm wheel collection."""
    system = platform.system()
    machine = platform.machine()
    python_version = tuple(sys.version_info[:3])
    libc_name, libc_version = platform.libc_ver()
    version_match = re.fullmatch(r"(\d+)\.(\d+)(?:\.\d+)*", libc_version)
    libc_numbers = None
    if version_match is not None:
        libc_numbers = tuple(int(part) for part in version_match.groups())
    checks = {
        "linux": system == "Linux",
        "x86_64": machine == "x86_64",
        "python_3_12": python_version[:2] == (3, 12),
        "glibc_2_39_or_newer": (
            libc_name == "glibc"
            and libc_numbers is not None
            and libc_numbers >= MINIMUM_GLIBC
        ),
    }
    details = {
        "system": system,
        "machine": machine,
        "python": ".".join(str(part) for part in python_version),
        "python_executable": sys.executable,
        "libc_name": libc_name,
        "libc_version": libc_version,
        "minimum_glibc": ".".join(str(part) for part in MINIMUM_GLIBC),
    }
    return details, checks


def inspect_installed():
    """Read package metadata and Torch build information, without GPU calls."""
    packages = {}
    errors = []
    checks = {}
    for name, expected in EXPECTED_VERSIONS.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = None
        packages[name] = {"expected": expected, "installed": actual}
        checks[name + "_version_matches"] = actual == expected

    nvidia_packages = []
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name", "")
        normalized_name = re.sub(r"[-_.]+", "-", name).lower()
        if normalized_name.startswith("nvidia-"):
            nvidia_packages.append({"name": name, "version": distribution.version})
    nvidia_packages.sort(key=lambda item: item["name"].lower())
    checks["no_nvidia_packages"] = not nvidia_packages
    torch_build = {"checked": False}

    # Import only the selected build. Reading torch.version does not query a GPU.
    if checks["torch_version_matches"]:
        try:
            torch = importlib.import_module("torch")
            torch_build = {
                "checked": True,
                "version": str(torch.__version__),
                "hip": torch.version.hip,
                "cuda": torch.version.cuda,
            }
            checks["torch_import_version_matches"] = (
                torch_build["version"] == EXPECTED_VERSIONS["torch"]
            )
            checks["torch_has_hip"] = bool(torch_build["hip"])
            checks["torch_has_no_cuda_build"] = torch_build["cuda"] is None
        except Exception as error:
            checks["torch_import_succeeded"] = False
            errors.append("Torch import failed: " + type(error).__name__ + ": " + str(error))
    else:
        torch_build["skip_reason"] = "Torch package metadata does not match the project pin."

    details = {
        "packages": packages,
        "nvidia_packages": nvidia_packages,
        "torch_build": torch_build,
    }
    return details, checks, errors


def check_environment(*, installed=False):
    """Return a serializable preflight report, keeping GPU validation separate."""
    platform_details, checks = inspect_platform()
    report = {
        "schema_version": 1,
        "phase": "installed" if installed else "platform_only",
        "passed": False,
        "platform": platform_details,
        "checks": checks,
        "errors": [],
        "gpu": {
            "tested": False,
            "initialized_by_this_check": False,
            "inference_tested": False,
            "status": "not_tested",
        },
    }
    if installed:
        if all(checks.values()):
            details, installed_checks, errors = inspect_installed()
            report["installed"] = details
            report["checks"].update(installed_checks)
            report["errors"].extend(errors)
        else:
            report["installed"] = {
                "skipped": True,
                "reason": "Platform requirements failed; Torch was not imported.",
            }
    report["passed"] = all(report["checks"].values()) and not report["errors"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument("--platform-only", action="store_true")
    phase.add_argument("--installed", action="store_true")
    parser.add_argument("--report", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()
    report = check_environment(installed=args.installed)
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
