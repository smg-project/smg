"""CPU-only tests for scripts/ci_setup_jit_cache.sh (persistent node-NVMe JIT cache)."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/ci_setup_jit_cache.sh"


class JitCacheSetupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.packages = self.root / "packages"
        self.packages.mkdir()
        self.cache_root = self.root / "models/.ci-cache/jit"
        self.github_env = self.root / "github.env"
        self.github_env.touch()

        self.write_tool(
            "nvidia-smi",
            '#!/bin/sh\necho "${FAKE_NVIDIA_SMI:-9.0, 580.65.06}"\n',
        )
        self.write_tool(
            "rocm_agent_enumerator",
            '#!/bin/sh\nprintf "%s\\n" gfx000 "${FAKE_ROCM_ARCH:-gfx942}"\n',
        )
        self.cuda_home = self.root / "cuda-13.0"
        (self.cuda_home / "bin").mkdir(parents=True)
        nvcc = self.cuda_home / "bin/nvcc"
        nvcc.write_text(
            '#!/bin/sh\necho "Cuda compilation tools, ${FAKE_NVCC_RELEASE:-release 13.0, V13.0.88}"\n'
        )
        nvcc.chmod(0o755)

        for pkg, ver in {
            "vllm": "0.31.0",
            "torch": "2.13.0+cu130",
            "triton": "3.8.0",
            "flashinfer-python": "0.7.0",
        }.items():
            self.metadata(pkg, ver)
        self.packages.joinpath("torch.py").write_text(
            "class version:\n    cuda = '13.0'\n    hip = None\n"
        )

    def write_tool(self, name: str, content: str) -> None:
        path = self.bin / name
        path.write_text(content)
        path.chmod(0o755)

    def metadata(self, name: str, version: str) -> None:
        dist = self.packages / f"{name.replace('-', '_')}.dist-info"
        dist.mkdir(exist_ok=True)
        dist.joinpath("METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
        )

    def run_setup(
        self, engine: str = "vllm", **overrides: str
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, str]]:
        self.github_env.write_text("")
        env = (
            os.environ
            | {
                "PATH": f"{self.bin}:{os.environ['PATH']}",
                "PYTHONPATH": str(self.packages),
                "SMG_JIT_CACHE_ROOT": str(self.cache_root),
                "SMG_JIT_CACHE_MIN_FREE_GB": "0",
                "SMG_JIT_CACHE_PYTHON": sys.executable,
                "CUDA_HOME": str(self.cuda_home),
                "GITHUB_ENV": str(self.github_env),
            }
            | overrides
        )
        result = subprocess.run(
            ["bash", str(SCRIPT), engine],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        exported: dict[str, str] = {}
        for line in self.github_env.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                exported[k] = v
        return result, exported

    def test_nvidia_exports_all_jit_cache_dirs_and_writes_manifest(self):
        result, exported = self.run_setup("vllm")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Enabled persistent JIT cache", result.stdout)
        triton_dir = Path(exported["TRITON_CACHE_DIR"])
        entry = triton_dir.parent
        self.assertTrue(entry.name.startswith("vllm-sm90-drv580-"), entry.name)
        self.assertEqual(exported["FLASHINFER_WORKSPACE_BASE"], str(entry / "flashinfer"))
        self.assertEqual(exported["VLLM_CACHE_ROOT"], str(entry / "vllm"))
        self.assertEqual(exported["TORCHINDUCTOR_CACHE_DIR"], str(entry / "torchinductor"))
        self.assertEqual(exported["TORCHINDUCTOR_FX_GRAPH_CACHE"], "1")
        self.assertEqual(exported["TVM_FFI_CACHE_DIR"], str(entry / "tvm-ffi"))
        self.assertEqual(exported["CUDA_CACHE_PATH"], str(entry / "cuda"))
        self.assertEqual(exported["CUDA_CACHE_MAXSIZE"], "4294967296")
        for key in (
            "TRITON_CACHE_DIR",
            "FLASHINFER_WORKSPACE_BASE",
            "VLLM_CACHE_ROOT",
            "TORCHINDUCTOR_CACHE_DIR",
            "TVM_FFI_CACHE_DIR",
            "CUDA_CACHE_PATH",
        ):
            self.assertTrue(Path(exported[key]).is_dir(), key)
        manifest = (entry / "manifest.txt").read_text()
        self.assertIn("pkg.vllm=0.31.0", manifest)
        self.assertIn("gpu_arch=sm90-drv580", manifest)

    def test_amd_keys_by_gcn_arch_and_omits_cuda_cache_path(self):
        result, exported = self.run_setup("vllm", E2E_GPU_VENDOR="amd", FAKE_ROCM_ARCH="gfx942")
        self.assertEqual(result.returncode, 0, result.stderr)
        entry = Path(exported["TRITON_CACHE_DIR"]).parent
        self.assertTrue(entry.name.startswith("vllm-gfx942"), entry.name)
        self.assertNotIn("CUDA_CACHE_PATH", exported)
        self.assertEqual(exported["VLLM_CACHE_ROOT"], str(entry / "vllm"))

    def test_version_driver_and_arch_changes_isolate_cache_keys(self):
        _, base = self.run_setup("vllm")
        _, new_arch = self.run_setup("vllm", FAKE_NVIDIA_SMI="10.0, 580.65.06")
        _, new_drv = self.run_setup("vllm", FAKE_NVIDIA_SMI="9.0, 590.10.01")
        _, new_nvcc = self.run_setup("vllm", FAKE_NVCC_RELEASE="release 13.1, V13.1.0")
        self.metadata("vllm", "0.32.0")
        _, new_vllm = self.run_setup("vllm")
        keys = {
            Path(d["TRITON_CACHE_DIR"]).parent.name
            for d in (base, new_arch, new_drv, new_nvcc, new_vllm)
        }
        self.assertEqual(len(keys), 5, keys)

    def test_degrades_softly_when_disabled_unwritable_or_low_disk(self):
        res, exp = self.run_setup("vllm", SMG_DISABLE_JIT_CACHE="1")
        self.assertEqual(res.returncode, 0)
        self.assertEqual(exp, {})

        not_dir = self.root / "file-blocker"
        not_dir.write_text("x")
        res, exp = self.run_setup("vllm", SMG_JIT_CACHE_ROOT=str(not_dir / "jit"))
        self.assertEqual(res.returncode, 0)
        self.assertEqual(exp, {})

        res, exp = self.run_setup("vllm", SMG_JIT_CACHE_MIN_FREE_GB="999999999")
        self.assertEqual(res.returncode, 0)
        self.assertIn("GiB free", res.stdout)
        self.assertEqual(exp, {})

    def test_stale_entries_are_swept_while_recent_and_active_survive(self):
        self.cache_root.mkdir(parents=True, exist_ok=True)
        stale = self.cache_root / "vllm-sm90-drv580-stale0000000000"
        recent = self.cache_root / "vllm-sm90-drv580-recent000000000"
        stale.mkdir()
        recent.mkdir()
        old_time = time.time() - 10 * 86400
        os.utime(stale, (old_time, old_time))
        res, exp = self.run_setup("vllm", SMG_JIT_CACHE_MAX_AGE_DAYS="7")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertFalse(stale.exists())
        self.assertTrue(recent.exists())
        self.assertTrue(Path(exp["TRITON_CACHE_DIR"]).is_dir())


if __name__ == "__main__":
    unittest.main()
