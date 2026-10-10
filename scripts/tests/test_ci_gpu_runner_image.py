"""CPU-only image identity and registry-failure policy checks."""

import importlib.util
import re
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "ci_gpu_runner_image", ROOT / "scripts/ci_gpu_runner_image.py"
)
assert SPEC and SPEC.loader
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)
BUILD_INPUTS = helper.BUILD_INPUTS
BASE = "ghcr.io/example/runner@sha256:" + "a" * 64
IMAGE = "ghcr.io/example/smg:ci-gpu-runner-test"


class ImageTagTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in BUILD_INPUTS:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)

    def tag(self, base=BASE, driver="580"):
        return helper.image_tag(self.root, base, driver)

    def test_same_inputs_ignore_checkout_location_and_other_sources(self):
        expected = self.tag()
        (self.root / "unrelated.py").write_text("changed SMG source")
        self.assertEqual(expected, self.tag())
        other = self.root / "another-checkout"
        for name in BUILD_INPUTS:
            path = other / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        self.assertEqual(expected, helper.image_tag(other, BASE, "580"))

    def test_base_digest_and_driver_change_identity(self):
        self.assertNotEqual(self.tag(), self.tag(base=BASE[:-1] + "b"))
        self.assertNotEqual(self.tag(), self.tag(driver="590"))

    def test_forced_candidates_do_not_replace_the_recipe_tag(self):
        stable = self.tag()
        forced = helper.image_tag(self.root, BASE, "580", "12345-1")
        self.assertEqual(forced, f"{stable}-run-12345-1")
        self.assertNotEqual(forced, helper.image_tag(self.root, BASE, "580", "12345-2"))
        self.assertEqual(stable, self.tag())
        with self.assertRaises(ValueError):
            helper.image_tag(self.root, BASE, "580", "latest")

    def test_every_baked_input_changes_identity(self):
        original = self.tag()
        for name in BUILD_INPUTS:
            with self.subTest(name=name):
                path = self.root / name
                path.write_text(name + "changed")
                self.assertNotEqual(original, self.tag())
                path.write_text(name)

    def test_missing_input_and_unpinned_profiles_fail(self):
        for base in ("", "ubuntu:24.04", BASE + "\nimage=other"):
            with self.subTest(base=base), self.assertRaises(ValueError):
                self.tag(base=base)
        for driver in ("", "latest", "580\nimage=other"):
            with self.subTest(driver=driver), self.assertRaises(ValueError):
                self.tag(driver=driver)
        (self.root / BUILD_INPUTS[0]).unlink()
        with self.assertRaises(FileNotFoundError):
            self.tag()

    def test_copy_inputs_and_push_paths_cover_the_recipe(self):
        dockerfile = (ROOT / "docker/ci-gpu-runner.Dockerfile").read_text()
        copied = re.findall(r"scripts/[A-Za-z0-9_]+\.sh", dockerfile)
        self.assertTrue(set(copied) <= set(BUILD_INPUTS))
        workflow = (ROOT / ".github/workflows/ci-gpu-runner-image.yml").read_text()
        for name in BUILD_INPUTS:
            self.assertIn(f"- '{name}'", workflow)

    def test_cuda_layout_probe_rejects_each_global_nvcc_path(self):
        workflow = (ROOT / ".github/workflows/ci-gpu-runner-image.yml").read_text()
        guard = re.search(
            r"(?ms)^ +test ! -x /usr/local/cuda/bin/nvcc.*?^ +test -x /usr/local/cuda-13.0/bin/nvcc$",
            workflow,
        ).group()
        for index, (global_nvcc, path_nvcc, expected) in enumerate(
            ((False, False, 0), (True, False, 1), (False, True, 1))
        ):
            with self.subTest(global_nvcc=global_nvcc, path_nvcc=path_nvcc):
                folder = self.root / str(index)
                folder.mkdir()
                toolkit = folder / "versioned-nvcc"
                toolkit.touch()
                toolkit.chmod(0o755)
                unversioned = folder / "unversioned-nvcc"
                if global_nvcc:
                    unversioned.touch()
                    unversioned.chmod(0o755)
                if path_nvcc:
                    (folder / "nvcc").touch()
                    (folder / "nvcc").chmod(0o755)
                script = guard.replace("/usr/local/cuda/bin/nvcc", shlex.quote(str(unversioned)))
                script = script.replace("/usr/local/cuda-13.0/bin/nvcc", shlex.quote(str(toolkit)))
                result = subprocess.run(
                    ["/bin/bash", "-euo", "pipefail", "-c", script],
                    env={"PATH": str(folder)},
                    capture_output=True,
                )
                self.assertEqual(result.returncode, expected, result.stderr)


class ImageExistsTests(unittest.TestCase):
    def inspect(self, code, stderr=""):
        result = subprocess.CompletedProcess([], code, stdout="", stderr=stderr)
        with patch.object(helper.subprocess, "run", return_value=result) as run:
            exists = helper.image_exists(IMAGE)
        self.assertEqual(run.call_args.args[0], ["docker", "manifest", "inspect", IMAGE])
        return exists

    def test_existing_manifest_skips(self):
        self.assertTrue(self.inspect(0))

    def test_only_missing_manifest_requests_build(self):
        self.assertFalse(self.inspect(1, f"no such manifest: {IMAGE}\n"))
        self.assertFalse(self.inspect(1, "manifest unknown: manifest unknown"))

    def test_auth_network_and_unexpected_errors_fail(self):
        errors = (
            "unauthorized",
            "denied",
            "TLS handshake timeout",
            "connection refused",
            "manifest unknown\nunauthorized",
        )
        for error in errors:
            with (
                self.subTest(error=error),
                self.assertRaisesRegex(RuntimeError, "Cannot check image availability"),
            ):
                self.inspect(1, error)

    def test_missing_docker_fails(self):
        with patch.object(helper.subprocess, "run", side_effect=FileNotFoundError):
            with self.assertRaises(FileNotFoundError):
                helper.image_exists(IMAGE)

    def test_manifest_inspection_timeout_fails(self):
        with patch.object(
            helper.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired("docker", 60),
        ):
            with self.assertRaises(subprocess.TimeoutExpired):
                helper.image_exists(IMAGE)


if __name__ == "__main__":
    unittest.main()
