"""No-hardware checks for allocation isolation and truthful unavailable readings."""

import json
import os
from pathlib import Path
import socket
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gpu_monitor import monitor, normalize_bdf, sample_gpu, stage_snapshot


class FakeAMDSMI:
    AmdSmiMemoryType = SimpleNamespace(VRAM="vram")

    def __init__(self, bdf="0000:5d:00.0"):
        self.bdf = bdf
        self.lookups = []
        self.metric_handles = []
        self.initialized = 0
        self.shutdowns = 0

    def amdsmi_init(self):
        self.initialized += 1

    def amdsmi_shut_down(self):
        self.shutdowns += 1

    def amdsmi_get_processor_handles(self):
        raise AssertionError("Must not enumerate other allocated/unallocated GPUs")

    def amdsmi_get_processor_handle_from_bdf(self, bdf):
        self.lookups.append(bdf)
        return "selected-handle"

    def amdsmi_get_gpu_device_bdf(self, handle):
        if handle != "selected-handle":
            raise AssertionError("Wrong handle")
        return self.bdf

    def amdsmi_get_lib_version(self):
        return {"major": 26, "minor": 2, "release": 2}

    def checked(self, handle):
        if handle != "selected-handle":
            raise AssertionError("Attempted metrics on another GPU")
        self.metric_handles.append(handle)

    def amdsmi_get_gpu_activity(self, handle):
        self.checked(handle)
        return {"gfx_activity": 0, "umc_activity": 50, "mm_activity": "N/A"}

    def amdsmi_get_power_info(self, handle):
        self.checked(handle)
        return {"average_socket_power": 105, "current_socket_power": 0xFFFFFFFF}

    def amdsmi_get_gpu_memory_usage(self, handle, memory_type):
        self.checked(handle)
        assert memory_type == self.AmdSmiMemoryType.VRAM
        return 16 * 1024**3

    def amdsmi_get_gpu_memory_total(self, handle, memory_type):
        self.checked(handle)
        assert memory_type == self.AmdSmiMemoryType.VRAM
        return 64 * 1024**3


def identity(path, binding="1"):
    path.write_text(json.dumps({"visible_device_count": 1, "hostname": socket.gethostname(),
                                "physical_pci_bdf": "0000:5D:00.0",
                                "environment": {"ROCR_VISIBLE_DEVICES": binding}}))


class GpuMonitorTests(unittest.TestCase):
    def test_domain_is_required_and_bdf_canonicalized(self):
        self.assertEqual(normalize_bdf("00000000:5D:00.0"), "0000:5d:00.0")
        for value in ("5d:00.0", "0000:5d:ff.0", "00010000:5d:00.0"):
            with self.assertRaises(ValueError):
                normalize_bdf(value)

    def test_wrong_task_identity_never_initializes_hardware(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"ROCR_VISIBLE_DEVICES": "0"}):
            root, api = Path(directory), FakeAMDSMI()
            identity(root / "identity.json", binding="1")
            result = monitor(root / "identity.json", root / "samples.jsonl", max_samples=1, api=api)
            self.assertEqual(result["status"], "unsupported")
            self.assertEqual(api.initialized, 0)
            self.assertEqual(api.lookups, [])
            rows = [json.loads(row) for row in (root / "samples.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["phase"], "identity_validation")

    def test_bdf_lookup_mismatch_never_reads_metric(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"ROCR_VISIBLE_DEVICES": "1"}):
            root, api = Path(directory), FakeAMDSMI(bdf="0000:6d:00.0")
            identity(root / "identity.json")
            result = monitor(root / "identity.json", root / "samples.jsonl", max_samples=1, api=api)
            self.assertEqual(result["status"], "unsupported")
            self.assertEqual(api.lookups, ["0000:5d:00.0"])
            self.assertEqual(api.metric_handles, [])
            self.assertEqual(api.shutdowns, 1)

    def test_sentinels_and_na_are_not_zero_measurements(self):
        api = FakeAMDSMI()
        api.amdsmi_get_gpu_activity = lambda handle: {"gfx_activity": "N/A", "umc_activity": 0xFFFFFFFF}
        api.amdsmi_get_power_info = lambda handle: {"average_socket_power": 0xFFFFFFFF,
                                                   "current_socket_power": 0xFFFF,
                                                   "socket_power": 0xFFFFFFFFFFFFFFFF}
        api.amdsmi_get_gpu_memory_usage = lambda handle, kind: 0xFFFFFFFFFFFFFFFF
        api.amdsmi_get_gpu_memory_total = lambda handle, kind: 0xFFFFFFFFFFFFFFFF
        result = sample_gpu(api, "selected-handle")
        self.assertEqual(result["status"], "unsupported")
        self.assertTrue(all(value is None for value in result["metrics"].values()))
        self.assertEqual(result["raw"]["activity"]["gfx_activity"], "N/A")

    def test_api_failure_preserves_other_valid_measurements_and_true_zero(self):
        api = FakeAMDSMI()

        def unsupported(handle):
            raise RuntimeError("AMDSMI_STATUS_NOT_SUPPORTED")

        api.amdsmi_get_power_info = unsupported
        result = sample_gpu(api, "selected-handle")
        self.assertEqual(result["status"], "partial")
        self.assertIsNone(result["metrics"]["power_watts"])
        self.assertEqual(result["metrics"]["gfx_utilization_percent"], 0.0)
        self.assertEqual(result["metrics"]["vram_allocation_percent"], 25.0)
        self.assertIn("amdsmi_get_power_info", result["errors"])

    def test_two_samples_record_stage_and_only_selected_gcd(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"ROCR_VISIBLE_DEVICES": "1"}):
            root, api = Path(directory), FakeAMDSMI()
            identity(root / "identity.json")
            (root / "stage.json").write_text('{"stage":"rollout","epoch_time":123.0}')
            result = monitor(root / "identity.json", root / "samples.jsonl", interval=0.001,
                             stage_path=root / "stage.json", max_samples=2, api=api)
            self.assertEqual(result["status"], "ok")
            rows = [json.loads(row) for row in (root / "samples.jsonl").read_text().splitlines()]
            samples = [row for row in rows if row["event"] == "sample"]
            self.assertEqual(len(samples), 2)
            self.assertTrue(all(row["physical_pci_bdf"] == "0000:5d:00.0" for row in samples))
            self.assertTrue(all(row["stage"] == "rollout" and row["stage_epoch_time"] == 123.0 for row in samples))
            self.assertTrue(all(row["metrics"]["power_watts"] == 105.0 for row in samples))
            self.assertTrue(all(row["epoch_time"] > 0 and row["monotonic_time"] > 0 for row in samples))
            self.assertEqual(set(api.metric_handles), {"selected-handle"})
            self.assertEqual(api.shutdowns, 1)
            self.assertEqual(rows[-1]["stop_reason"], "max_samples")

    def test_stop_request_finishes_current_sample_flushes_and_shuts_down(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"ROCR_VISIBLE_DEVICES": "1"}):
            root, api, stop = Path(directory), FakeAMDSMI(), threading.Event()
            identity(root / "identity.json")
            activity = api.amdsmi_get_gpu_activity

            def stopping_activity(handle):
                stop.set()
                return activity(handle)

            api.amdsmi_get_gpu_activity = stopping_activity
            result = monitor(root / "identity.json", root / "samples.jsonl", stop_event=stop, api=api)
            self.assertEqual(result["samples"], 1)
            self.assertEqual(api.shutdowns, 1)
            last = json.loads((root / "samples.jsonl").read_text().splitlines()[-1])
            self.assertEqual(last["stop_reason"], "stop_requested")

    def test_partial_stage_write_is_explicit_and_does_not_invent_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage.json"
            path.write_text('{"stage":')
            result = stage_snapshot(path)
            self.assertIsNone(result["stage"])
            self.assertIn("stage_error", result)


if __name__ == "__main__":
    unittest.main()
