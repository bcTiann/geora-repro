"""Read AMDSMI telemetry only for the single GCD in a trusted HIP identity file.

Run inside the same bound task/container as device_identity.py. The monitor never
enumerates all processor handles, changes clocks/power limits, or creates a HIP
context. Missing APIs and unsupported readings are explicit nulls/errors, never
synthetic zero utilization. Monitoring failure exits successfully so it need not
stop the training process that launched it.

API names/units checked against ROCm/amdsmi tag rocm-7.2.3:
  py-interface/amdsmi_interface.py and include/amd_smi/amdsmi.h.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from enum import Enum
import hashlib
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import json
import math
from numbers import Real
import os
from pathlib import Path
import re
import signal
import socket
import sys
import threading
import time


API_SOURCE = "https://github.com/ROCm/amdsmi/blob/rocm-7.2.3/py-interface/amdsmi_interface.py"
UNIT_SOURCE = "https://github.com/ROCm/amdsmi/blob/rocm-7.2.3/include/amd_smi/amdsmi.h"
_BDF = re.compile(r"^([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])$")
_METRICS = (
    "gfx_utilization_percent", "memory_controller_utilization_percent", "power_watts",
    "vram_used_bytes", "vram_total_bytes", "vram_allocation_percent",
)


def normalize_bdf(value):
    match = _BDF.fullmatch(str(value).strip())
    if match is None:
        raise ValueError("Identity requires a complete PCI domain:bus:device.function address")
    domain, bus, device, function = (int(part, 16) for part in match.groups())
    if domain > 0xFFFF or device > 0x1F:
        raise ValueError("PCI domain or device is outside its supported range")
    return f"{domain:04x}:{bus:02x}:{device:02x}.{function:x}"


def validate_identity(path):
    """Reject stale/other-task identity files before touching AMDSMI."""
    raw = Path(path).read_bytes()
    identity = json.loads(raw)
    if identity.get("visible_device_count") != 1:
        raise ValueError("The HIP identity must describe exactly one visible GCD")
    if identity.get("hostname") != socket.gethostname():
        raise ValueError("The HIP identity belongs to another host")
    binding = identity.get("environment", {}).get("ROCR_VISIBLE_DEVICES")
    if (not isinstance(binding, str) or not binding.strip() or "," in binding
            or binding != os.environ.get("ROCR_VISIBLE_DEVICES")):
        raise ValueError("Monitor must inherit the same single-GCD ROCR_VISIBLE_DEVICES binding as the HIP identity")
    return {"physical_pci_bdf": normalize_bdf(identity["physical_pci_bdf"]),
            "hostname": identity["hostname"], "rocr_visible_devices": binding,
            "identity_sha256": hashlib.sha256(raw).hexdigest()}


def json_safe(value):
    """Preserve raw API structure while making exceptional scalar values explicit."""
    if isinstance(value, Enum):
        return {"enum": type(value).__name__, "name": value.name, "value": json_safe(value.value)}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Real):
        return float(value) if math.isfinite(value) else {"nonfinite": str(value)}
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if hasattr(value, "value"):
        return json_safe(value.value)
    return {"type": type(value).__name__, "repr": repr(value)}


def error_record(error):
    result = {"type": type(error).__name__, "message": str(error)}
    for name in ("err_code", "status"):
        if hasattr(error, name):
            result[name] = json_safe(getattr(error, name))
    return result


def _number(value, *, maximum=None, integer=False, power=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        return None
    # Power struct docs use UINT32_MAX; its Python wrapper also recognizes
    # UINT16_MAX. VRAM functions return uint64 bytes and may expose UINT64_MAX.
    if value == 0xFFFFFFFFFFFFFFFF or (power and value in (0xFFFF, 0xFFFFFFFF)):
        return None
    if maximum is not None and value > maximum:
        return None
    if integer and int(value) != value:
        return None
    return int(value) if integer else float(value)


def _api_call(api, name, errors, *args):
    try:
        return getattr(api, name)(*args)
    except Exception as error:
        errors[name] = error_record(error)
        return None


def sample_gpu(api, handle):
    """All metric calls receive only the already-verified assigned handle."""
    errors = {}
    activity = _api_call(api, "amdsmi_get_gpu_activity", errors, handle)
    power = _api_call(api, "amdsmi_get_power_info", errors, handle)
    try:
        vram_type = api.AmdSmiMemoryType.VRAM
    except Exception as error:
        errors["AmdSmiMemoryType.VRAM"] = error_record(error)
        used = total = None
    else:
        used = _api_call(api, "amdsmi_get_gpu_memory_usage", errors, handle, vram_type)
        total = _api_call(api, "amdsmi_get_gpu_memory_total", errors, handle, vram_type)
    raw = {"activity": activity, "power_info": power, "vram_used_bytes": used, "vram_total_bytes": total}
    activity = activity if isinstance(activity, Mapping) else {}
    power = power if isinstance(power, Mapping) else {}
    metrics = {
        "gfx_utilization_percent": _number(activity.get("gfx_activity"), maximum=100),
        "memory_controller_utilization_percent": _number(activity.get("umc_activity"), maximum=100),
        "power_watts": None,
        "vram_used_bytes": _number(used, integer=True),
        "vram_total_bytes": _number(total, integer=True),
        "vram_allocation_percent": None,
    }
    power_field = None
    # Header identifies average_socket_power for MI200 and earlier, current for
    # MI300+. Retain the selected source instead of assuming both are supported.
    for field in ("average_socket_power", "current_socket_power", "socket_power"):
        reading = _number(power.get(field), power=True)
        if reading is not None:
            metrics["power_watts"], power_field = reading, field
            break
    if metrics["vram_total_bytes"] == 0:
        metrics["vram_total_bytes"] = None
    used_value, total_value = metrics["vram_used_bytes"], metrics["vram_total_bytes"]
    if used_value is not None and total_value is not None:
        if used_value <= total_value:
            metrics["vram_allocation_percent"] = 100.0 * used_value / total_value
        else:
            errors["vram_consistency"] = {"type": "InvalidReading", "message": "Reported VRAM usage exceeds total"}
            metrics["vram_used_bytes"] = None
    unsupported = [name for name in _METRICS if metrics[name] is None]
    return {
        "status": "unsupported" if len(unsupported) == len(_METRICS) else "partial" if unsupported or errors else "ok",
        "metrics": metrics, "raw": raw, "errors": errors, "unsupported_metrics": unsupported,
        "power_source_field": power_field,
        "power_scope": "AMDSMI socket power for the selected handle; do not assume independent per-GCD readings or sum across GCDs without validating hardware semantics",
        "memory_utilization_definition": "UMC activity percent; VRAM allocated capacity is reported separately",
    }


def stage_snapshot(path):
    if path is None:
        return {"stage": None, "stage_epoch_time": None}
    try:
        value = json.loads(Path(path).read_text())
        if not isinstance(value, dict) or not isinstance(value.get("stage"), str):
            raise ValueError("Stage file must be an object with a string stage")
        epoch = value.get("epoch_time")
        if isinstance(epoch, bool) or not isinstance(epoch, Real) or not math.isfinite(epoch):
            raise ValueError("Stage file epoch_time must be finite")
        return {"stage": value["stage"], "stage_epoch_time": float(epoch)}
    except Exception as error:
        return {"stage": None, "stage_epoch_time": None, "stage_error": error_record(error)}


def monitor(identity_path, output_path, *, interval=2.0, stage_path=None, stop_event=None,
            max_samples=None, api=None):
    """Return a status summary; hardware/API failures are recorded, not raised."""
    stop_event = stop_event or threading.Event()
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("Sampling interval must be positive and finite")
    if max_samples is not None and (type(max_samples) is not int or max_samples < 1):
        raise ValueError("max_samples must be a positive integer")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count, initialized, target = 0, False, None
    summary = {"status": "unsupported", "samples": 0}
    sample_status_counts = {"ok": 0, "partial": 0, "unsupported": 0}
    completion_reason = "monitor_unavailable"
    with path.open("a", buffering=1) as stream:
        def emit(payload):
            record = {"epoch_time": time.time(), "monotonic_time": time.monotonic(),
                      "pid": os.getpid(), **payload}
            stream.write(json.dumps(json_safe(record), allow_nan=False) + "\n")
            stream.flush()

        phase = "identity_validation"
        try:
            target = validate_identity(identity_path)
            phase = "amdsmi_import"
            api = import_module("amdsmi") if api is None else api
            phase = "amdsmi_init"
            api.amdsmi_init()
            initialized = True
            phase = "assigned_bdf_lookup"
            handle = api.amdsmi_get_processor_handle_from_bdf(target["physical_pci_bdf"])
            actual_bdf = normalize_bdf(api.amdsmi_get_gpu_device_bdf(handle))
            if actual_bdf != target["physical_pci_bdf"]:
                raise ValueError("AMDSMI returned a handle for a different physical GCD")
            metadata_errors = {}
            library_version = _api_call(api, "amdsmi_get_lib_version", metadata_errors)
            try:
                package_version = version("amdsmi")
            except PackageNotFoundError:
                package_version = None
            emit({"event": "monitor_start", "status": "ok", "target": target,
                  "interval_seconds": interval, "amdsmi_package_version": package_version,
                  "amdsmi_library_version": library_version, "metadata_errors": metadata_errors,
                  "api_source": API_SOURCE, "unit_source": UNIT_SOURCE,
                  "selection": "Direct BDF lookup and exact identity recheck; no processor enumeration or metrics for other handles"})
            phase = "sampling"
            deadline = time.monotonic()
            while not stop_event.is_set():
                began_epoch, began_mono = time.time(), time.monotonic()
                stage = stage_snapshot(stage_path)
                sample = sample_gpu(api, handle)
                sample_status_counts[sample["status"]] += 1
                emit({"event": "sample", "sample_index": count,
                      "physical_pci_bdf": actual_bdf, "sample_started_epoch_time": began_epoch,
                      "sample_started_monotonic_time": began_mono,
                      "collection_wall_seconds": time.monotonic() - began_mono, **stage, **sample})
                count += 1
                if max_samples is not None and count >= max_samples:
                    break
                deadline += interval
                remaining = deadline - time.monotonic()
                # If an API sample exceeded its interval, resume a normal cadence
                # rather than issuing a burst of catch-up hardware queries.
                if remaining <= 0:
                    deadline = time.monotonic() + interval
                    remaining = interval
                if stop_event.wait(remaining):
                    break
            status = ("unsupported" if sample_status_counts["unsupported"] == count else
                      "partial" if sample_status_counts["partial"] or sample_status_counts["unsupported"] else "ok")
            summary = {"status": status, "samples": count, "sample_status_counts": sample_status_counts}
            completion_reason = "stop_requested" if stop_event.is_set() else "max_samples"
        except Exception as error:
            emit({"event": "monitor_unavailable", "status": "unsupported", "phase": phase,
                  "target": target, "error": error_record(error),
                  "interpretation": "No zero readings substituted; training may continue without this telemetry"})
            summary = {"status": "unsupported", "samples": count}
        finally:
            if initialized:
                try:
                    api.amdsmi_shut_down()
                except Exception as error:
                    emit({"event": "monitor_shutdown_error", "status": "unsupported", "error": error_record(error)})
            emit({"event": "monitor_stop", **summary,
                  "stop_reason": completion_reason})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage-file", type=Path)
    parser.add_argument("--interval-seconds", type=float, default=2.0)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    stop = threading.Event()

    def request_stop(signum, frame):
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        monitor(args.identity_json, args.output, interval=args.interval_seconds,
                stage_path=args.stage_file, stop_event=stop, max_samples=args.max_samples)
    except Exception as error:
        # Includes an unwritable output path. Keep this standalone optional
        # process from turning monitoring into a training failure dependency.
        print(json.dumps({"event": "monitor_unavailable", "status": "unsupported",
                          "epoch_time": time.time(), "error": error_record(error)}, default=str), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
