"""Identify the physical HIP device; Slurm-local device indices can both be 0."""

import ctypes
import json
import os
from pathlib import Path
import socket
import sys
import uuid


def physical_device():
    library = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
    library.hipInit.argtypes = [ctypes.c_uint]
    library.hipInit.restype = ctypes.c_int
    library.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    library.hipGetDeviceCount.restype = ctypes.c_int
    library.hipDeviceGetPCIBusId.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    library.hipDeviceGetPCIBusId.restype = ctypes.c_int
    if library.hipInit(0) != 0:
        raise RuntimeError("Cannot initialize HIP on the assigned GCD")
    count = ctypes.c_int()
    if library.hipGetDeviceCount(ctypes.byref(count)) != 0 or count.value != 1:
        raise RuntimeError(f"Expected one visible physical GCD, got {count.value}")
    bus = ctypes.create_string_buffer(64)
    if library.hipDeviceGetPCIBusId(bus, len(bus), 0) != 0:
        raise RuntimeError("Cannot identify the assigned physical GCD")
    raw = (ctypes.c_ubyte * 16)()
    library.hipDeviceGetUuid.argtypes = [ctypes.c_void_p, ctypes.c_int]
    library.hipDeviceGetUuid.restype = ctypes.c_int
    status = library.hipDeviceGetUuid(ctypes.byref(raw), 0)
    return {
        "hostname": socket.gethostname(),
        "physical_pci_bdf": bus.value.decode(),
        "hip_uuid": str(uuid.UUID(bytes=bytes(raw))) if status == 0 else None,
        "visible_device_count": count.value,
        "environment": {name: os.environ.get(name) for name in (
            "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
            "SLURM_PROCID", "SLURM_LOCALID", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS",
        )},
    }


if __name__ == "__main__":
    result = physical_device()
    Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
