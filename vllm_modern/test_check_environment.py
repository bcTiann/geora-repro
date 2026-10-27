"""Preflight tests: reject incompatible builds without touching GPU devices."""

from importlib import metadata
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import check_environment as subject


class EnvironmentCheckTests(unittest.TestCase):
    def setUp(self):
        self.platform_patches = [
            patch.object(subject.platform, "system", return_value="Linux"),
            patch.object(subject.platform, "machine", return_value="x86_64"),
            patch.object(subject.platform, "libc_ver", return_value=("glibc", "2.39")),
            patch.object(subject.sys, "version_info", (3, 12, 11)),
        ]
        for item in self.platform_patches:
            item.start()
            self.addCleanup(item.stop)

    def test_platform_only_does_not_inspect_or_import_packages(self):
        with patch.object(subject, "inspect_installed") as installed:
            with patch.object(subject.importlib, "import_module") as importer:
                report = subject.check_environment()
        self.assertTrue(report["passed"])
        self.assertEqual(report["phase"], "platform_only")
        self.assertEqual(report["gpu"]["status"], "not_tested")
        installed.assert_not_called()
        importer.assert_not_called()

    def test_rejects_old_unknown_and_non_glibc_platforms(self):
        for libc in (("glibc", "2.38"), ("musl", "2.40"), ("glibc", "")):
            with self.subTest(libc=libc):
                with patch.object(subject.platform, "libc_ver", return_value=libc):
                    report = subject.check_environment()
                self.assertFalse(report["passed"])
                self.assertFalse(report["checks"]["glibc_2_39_or_newer"])

    def test_accepts_newer_glibc_and_rejects_wrong_system_arch_python(self):
        with patch.object(subject.platform, "libc_ver", return_value=("glibc", "2.100")):
            self.assertTrue(subject.check_environment()["passed"])
        invalid_values = [
            (subject.platform, "system", "Darwin", "linux"),
            (subject.platform, "machine", "aarch64", "x86_64"),
            (subject.sys, "version_info", (3, 13, 0), "python_3_12"),
        ]
        for owner, name, value, check in invalid_values:
            with self.subTest(check=check):
                if owner is subject.sys:
                    value_patch = patch.object(owner, name, value)
                else:
                    value_patch = patch.object(owner, name, return_value=value)
                with value_patch:
                    report = subject.check_environment()
                self.assertFalse(report["passed"])
                self.assertFalse(report["checks"][check])

    def test_installed_failure_skips_torch_on_incompatible_platform(self):
        with patch.object(subject.platform, "libc_ver", return_value=("glibc", "2.38")):
            with patch.object(subject, "inspect_installed") as installed:
                report = subject.check_environment(installed=True)
        self.assertFalse(report["passed"])
        self.assertTrue(report["installed"]["skipped"])
        installed.assert_not_called()

    def installed_report(self, *, versions=None, hip="7.2.3", cuda=None, distributions=(), import_error=None):
        if versions is None:
            versions = subject.EXPECTED_VERSIONS
        fake_torch = SimpleNamespace(
            __version__=versions.get("torch"),
            version=SimpleNamespace(hip=hip, cuda=cuda),
        )
        importer_result = {"return_value": fake_torch}
        if import_error is not None:
            importer_result = {"side_effect": import_error}
        with patch.object(subject.metadata, "version", side_effect=lambda name: versions.get(name)):
            with patch.object(subject.metadata, "distributions", return_value=distributions):
                with patch.object(subject.importlib, "import_module", **importer_result) as importer:
                    report = subject.check_environment(installed=True)
        return report, importer

    def test_exact_installed_rocm_build_passes_without_gpu_api(self):
        # This fake has no torch.cuda attribute: GPU access would fail the test.
        report, importer = self.installed_report()
        self.assertTrue(report["passed"])
        self.assertEqual(report["phase"], "installed")
        self.assertFalse(report["gpu"]["tested"])
        self.assertFalse(report["gpu"]["inference_tested"])
        importer.assert_called_once_with("torch")

    def test_wrong_torch_pin_is_rejected_without_import(self):
        versions = dict(subject.EXPECTED_VERSIONS)
        versions["torch"] = "2.13.0"
        report, importer = self.installed_report(versions=versions)
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["torch_version_matches"])
        importer.assert_not_called()

    def test_missing_package_is_rejected(self):
        def package_version(name):
            if name == "vllm":
                raise metadata.PackageNotFoundError(name)
            return subject.EXPECTED_VERSIONS[name]

        with patch.object(subject.metadata, "version", side_effect=package_version):
            with patch.object(subject.metadata, "distributions", return_value=[]):
                with patch.object(subject.importlib, "import_module", return_value=SimpleNamespace(
                    __version__=subject.EXPECTED_VERSIONS["torch"],
                    version=SimpleNamespace(hip="7.2.3", cuda=None),
                )):
                    report = subject.check_environment(installed=True)
        self.assertFalse(report["passed"])
        self.assertIsNone(report["installed"]["packages"]["vllm"]["installed"])

    def test_cuda_or_non_hip_build_is_rejected(self):
        for hip, cuda in ((None, None), (None, "13.0"), ("7.2.3", "13.0")):
            with self.subTest(hip=hip, cuda=cuda):
                report, importer = self.installed_report(hip=hip, cuda=cuda)
                self.assertFalse(report["passed"])

    def test_nvidia_package_names_are_normalized_and_rejected(self):
        distributions = [
            SimpleNamespace(metadata={"Name": "NVIDIA_Cublas-Cu13"}, version="13.0"),
            SimpleNamespace(metadata={"Name": "numpy"}, version="2.4.0"),
        ]
        report, importer = self.installed_report(distributions=distributions)
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["no_nvidia_packages"])
        self.assertEqual(len(report["installed"]["nvidia_packages"]), 1)

    def test_torch_import_error_is_reported_as_failure(self):
        report, importer = self.installed_report(import_error=OSError("missing libamdhip64.so"))
        self.assertFalse(report["passed"])
        self.assertIn("missing libamdhip64.so", report["errors"][0])
        self.assertFalse(report["gpu"]["tested"])

    def test_cli_emits_json_and_failure_exit_status(self):
        with patch.object(subject.sys, "argv", ["check_environment.py", "--platform-only"]):
            with patch.object(subject.platform, "libc_ver", return_value=("glibc", "2.38")):
                with patch("sys.stdout", new_callable=io.StringIO) as output:
                    result = subject.main()
        self.assertEqual(result, 1)
        report = json.loads(output.getvalue())
        self.assertFalse(report["passed"])
        self.assertEqual(report["phase"], "platform_only")


if __name__ == "__main__":
    unittest.main()
