"""Only new owned-origin/observer integration; never execute SDK or patch."""

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "owned_sdk_installer", SCRIPTS / "ci_install_sglang_gptoss20_sdk.py"
)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)
observer = sys.modules["ci_sglang_mxfp4_fallback"]


def sha(body):
    return hashlib.sha256(body).hexdigest()


class OwnedSdk(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=SCRIPTS.parent.parent)
        self.addCleanup(self.temp.cleanup)
        self.scope = Path(self.temp.name)
        self.prefix = self.scope / "interpreter"
        site = self.prefix / "lib/python3.12/site-packages"
        site.mkdir(parents=True)
        self.root = self.scope / "sdk/sglang"
        self.root.mkdir(parents=True)
        (self.root / "__init__.py").write_text("# fixture only\n")
        (self.root / "probe.py").write_text("post")
        self.manifest = site / "smg-cohort.json"
        self.manifest.write_text(
            json.dumps({"interpreter_prefix": str(self.prefix), "isolated_children": True})
        )
        self.boot = types.SimpleNamespace(
            __file__=str(site / "ci_sglang_cohort.py"),
            SMG_MANIFEST_SHA256=sha(self.manifest.read_bytes()),
            SMG_COHORT={"sglang": {"module_path": str(self.root / "__init__.py")}},
        )
        for context in (
            patch.dict(sys.modules, {"ci_sglang_cohort": self.boot, "sitecustomize": self.boot}),
            patch.object(sys, "prefix", str(self.prefix)),
            patch.object(installer, "sdk_root", return_value=self.root),
            patch.object(importlib.metadata, "version", return_value="0.5.21"),
            patch.object(observer, "FILES", {"probe.py": (sha(b"pre"), sha(b"post"))}),
            patch.object(installer, "FILES", {"probe.py": (sha(b"pre"), sha(b"post"))}),
            patch.dict(
                os.environ,
                {
                    "E2E_ENGINE": "sglang",
                    "E2E_GPU_VENDOR": "amd",
                    observer.ROW_ENV: "1",
                    "SMG_SGLANG_ROCM_IMAGE": observer.IMAGE,
                },
                clear=True,
            ),
        ):
            context.start()
            self.addCleanup(context.stop)
        context = patch.object(
            installer.subprocess, "run", side_effect=AssertionError("No process")
        )
        self.no_process = context.start()
        self.addCleanup(context.stop)

    def invoke(self):
        output = self.scope / "report.json"
        code = installer.main(["--verify-only", "--phase", "prepared", "--output", str(output)])
        self.no_process.assert_not_called()
        self.assertEqual((self.root / "probe.py").read_text(), "post")
        return code, json.loads(output.read_text())

    def test_exact_owned_sdk_verification_uses_shared_observer(self):
        code, report = self.invoke()
        self.assertEqual(code, 0)
        self.assertEqual(report["sdk"]["state"], "patched")
        self.assertEqual(report["sdk"]["sdk_root"], str(self.root))
        self.assertEqual(report["patch_sha256"], observer.PATCH_SHA)
        self.assertFalse(hasattr(observer, "install"))
        self.assertFalse(hasattr(observer, "main"))

    def test_missing_active_bootstrap_refuses_before_process_or_sdk_write(self):
        with patch.dict(sys.modules, {"sitecustomize": types.ModuleType("foreign")}):
            code, report = self.invoke()
        self.assertEqual(code, 1)
        self.assertIn("Active owned cohort", report["error"]["message"])

    def test_wrong_manifest_hash_refuses_before_process_or_sdk_write(self):
        self.boot.SMG_MANIFEST_SHA256 = "0" * 64
        code, report = self.invoke()
        self.assertEqual(code, 1)
        self.assertIn("manifest differs", report["error"]["message"])

    def test_foreign_sdk_origin_refuses_before_process_or_sdk_write(self):
        with patch.object(installer, "sdk_root", return_value=self.scope / "borrowed"):
            code, report = self.invoke()
        self.assertEqual(code, 1)
        self.assertIn("owned staged copy", report["error"]["message"])

    def test_direct_install_cannot_target_a_borrowed_sdk(self):
        borrowed = self.scope / "borrowed"
        borrowed.mkdir()
        sentinel = borrowed / "sentinel"
        sentinel.write_bytes(b"unchanged borrowed bytes")
        with self.assertRaisesRegex(RuntimeError, "Foreign SDK install target"):
            installer.install(borrowed)
        self.no_process.assert_not_called()
        self.assertEqual(sentinel.read_bytes(), b"unchanged borrowed bytes")

    def test_nvidia_context_is_rejected_before_origin_or_process(self):
        with patch.dict(os.environ, {"E2E_GPU_VENDOR": "nvidia"}):
            code, report = self.invoke()
        self.assertEqual(code, 1)
        self.assertIn("Explicit AMD SGLang", report["error"]["message"])


if __name__ == "__main__":
    unittest.main()
