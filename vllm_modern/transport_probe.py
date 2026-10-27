"""Bounded two-rank ROCm transport probe without model loading or training."""

from __future__ import annotations

import argparse
import ctypes
from datetime import timedelta
from importlib.metadata import version
import json
import os
from pathlib import Path
import signal
import socket
import time
import traceback


def library_maps():
    names = ("libamdhip64", "librccl", "libhsa-runtime64", "libamd_comgr")
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        if any(name in line for name in names):
            path = line.split()[-1]
            if path.startswith("/"):
                paths.add(path)
    return sorted(paths)


def device_identity_from_loaded_runtime():
    """Read the BDF without introducing a second HIP runtime into the process."""
    paths = [path for path in library_maps() if "libamdhip64" in path]
    if len(paths) != 1:
        raise RuntimeError(f"Expected one already-loaded HIP runtime, found {paths}")
    library = ctypes.CDLL(paths[0], mode=os.RTLD_NOLOAD)
    library.hipDeviceGetPCIBusId.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    library.hipDeviceGetPCIBusId.restype = ctypes.c_int
    bus = ctypes.create_string_buffer(64)
    if library.hipDeviceGetPCIBusId(bus, len(bus), 0) != 0:
        raise RuntimeError("Cannot identify the Slurm-assigned physical GCD")
    return {"physical_pci_bdf": bus.value.decode(), "loaded_hip_runtime": paths[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("torch", "pynccl"), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--rank", type=int, default=int(os.environ.get("SLURM_PROCID", "-1")))
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rank not in (0, 1) or not 1 <= args.timeout <= 60:
        raise ValueError("Require rank0/1 and timeout at most60 seconds")
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / f"rank-{args.rank}.json"
    events_path = args.output / f"rank-{args.rank}-events.jsonl"
    started = time.monotonic()
    result = {"passed": False, "mode": args.mode, "rank": args.rank, "world_size": 2,
              "hostname": socket.gethostname(), "pid": os.getpid(), "events": []}
    result["kernel"] = os.uname().release
    driver = Path("/sys/module/amdgpu/version")
    result["amdgpu_driver_version"] = driver.read_text().strip() if driver.exists() else None
    topology = []
    for node in sorted(Path("/sys/class/kfd/kfd/topology/nodes").glob("*")):
        entry = {"node": node.name}
        try:
            text = (node / "properties").read_text()
            entry["properties_bytes"] = len(text.encode())
            entry["selected_properties"] = {line.split()[0]: line.split()[1:]
                                            for line in text.splitlines() if line.split()
                                            and line.split()[0] in ("location_id", "domain", "unique_id", "vendor_id", "device_id")}
            entry["gpu_id"] = (node / "gpu_id").read_text().strip()
        except OSError as error:
            entry["read_errno"] = error.errno
        topology.append(entry)
    result["kfd_topology"] = topology

    def save():
        temporary = result_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(result_path)

    def event(stage, **values):
        item = {"stage": stage, "elapsed_seconds": time.monotonic() - started, **values}
        result["last_stage"] = stage
        result["events"].append(item)
        with events_path.open("a") as stream:
            stream.write(json.dumps(item, allow_nan=False) + "\n")
        save()
        print(json.dumps({"mode": args.mode, "rank": args.rank, **item}), flush=True)

    def interrupted(signum, _frame):
        raise TimeoutError(f"Transport probe interrupted by signal{signum}")

    signal.signal(signal.SIGALRM, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    # Native RCCL calls may not return to Python promptly. The Slurm launcher
    # additionally bounds each entire step with GNU timeout and kill-after.
    signal.alarm(args.timeout + 30)
    try:
        event("import_torch")
        import torch
        import torch.distributed as dist

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Each task must see exactly one Slurm-assigned GCD")
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        result["versions"] = {"torch": version("torch"), "vllm": version("vllm"), "hip": torch.version.hip}
        result["device"] = device_identity_from_loaded_runtime()
        result["environment"] = {key: value for key, value in os.environ.items()
                                 if key.startswith(("NCCL_", "RCCL_", "TORCH_NCCL_", "HSA_"))
                                 or key in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                                            "VLLM_NCCL_SO_PATH", "LD_LIBRARY_PATH")}
        event("local_tensor", mapped_libraries=library_maps())
        local = torch.arange(16, dtype=torch.float32, device=device)
        torch.cuda.synchronize()
        if not torch.equal(local.cpu(), torch.arange(16, dtype=torch.float32)):
            raise ValueError("Local GPU tensor sanity check failed")
        del local
        communicator = None
        group = None
        event("communicator_init_start", mapped_libraries=library_maps())
        if args.mode == "torch":
            dist.init_process_group("nccl", init_method=f"tcp://{args.host}:{args.port}",
                                    rank=args.rank, world_size=2, timeout=timedelta(seconds=args.timeout),
                                    device_id=device)
        else:
            from vllm.distributed.utils import StatelessProcessGroup
            from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

            group = StatelessProcessGroup.create(args.host, args.port, args.rank, 2,
                                                  store_timeout=args.timeout)
            event("metadata_group_ready", mapped_libraries=library_maps())
            communicator = PyNcclCommunicator(group, device=device)
            if not communicator.available or communicator.disabled:
                raise RuntimeError("PyNcclCommunicator silently disabled the requested transport")
            result["pynccl_library"] = str(communicator.nccl.lib._name)
            result["pynccl_nccl_version"] = communicator.nccl.ncclGetVersion()
        event("communicator_init_complete", mapped_libraries=library_maps())
        outcomes = []
        for dtype in (torch.float32, torch.bfloat16):
            for count in (1, 4096, 1048576):
                label = f"{str(dtype).removeprefix('torch.')}/{count}"
                value = torch.full((count,), args.rank + 1, dtype=dtype, device=device)
                event("all_reduce_start", tensor=label)
                if args.mode == "torch":
                    dist.all_reduce(value)
                    reduced = value
                else:
                    reduced = communicator.all_reduce(value)
                torch.cuda.synchronize()
                if not torch.equal(reduced, torch.full_like(reduced, 3)):
                    raise ValueError(f"Incorrect all_reduce result: {label}")
                event("all_reduce_complete", tensor=label)
                del reduced, value
                value = torch.full((count,), 17 if args.rank == 0 else -2, dtype=dtype, device=device)
                event("broadcast_start", tensor=label)
                if args.mode == "torch":
                    dist.broadcast(value, src=0)
                else:
                    communicator.broadcast(value, src=0)
                torch.cuda.synchronize()
                if not torch.equal(value, torch.full_like(value, 17)):
                    raise ValueError(f"Incorrect broadcast result: {label}")
                event("broadcast_complete", tensor=label)
                outcomes.append({"dtype": str(dtype), "elements": count, "all_reduce_exact": True,
                                 "broadcast_exact": True})
                del value
        event("collectives_complete", mapped_libraries=library_maps())
        if args.mode == "torch":
            dist.barrier()
            dist.destroy_process_group()
        else:
            group.barrier(timeout=args.timeout)
            communicator.destroy()
        result["outcomes"] = outcomes
        result["passed"] = True
        result["scope"] = "No model or training; local GPU sanity plus two-rank all_reduce and broadcast"
        event("completed", mapped_libraries=library_maps())
    except BaseException as error:
        result["passed"] = False
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
        result["mapped_libraries_at_failure"] = library_maps()
        result["elapsed_seconds"] = time.monotonic() - started
        save()
        print(result["traceback"], flush=True)
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
