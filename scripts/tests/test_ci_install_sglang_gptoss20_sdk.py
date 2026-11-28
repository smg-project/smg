"""Owned SDK guards and harmless patch fixtures; never import the real SDK."""

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
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.scope = Path(self.temp.name).resolve()
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
        self.no_process_context = context
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

    def test_vendored_patch_bytes_match_shared_observer(self):
        self.assertEqual(sha(installer.PATCH.read_bytes()), observer.PATCH_SHA)

    def test_real_patch_on_owned_toy_sources_then_idempotent(self):
        self.no_process_context.stop()
        (self.root / "probe.py").write_bytes(b"pre\n")
        payload = self.scope / "toy.patch"
        payload.write_text(
            "--- a/python/sglang/probe.py\n+++ b/python/sglang/probe.py\n@@ -1 +1 @@\n-pre\n+post\n"
        )
        files = {"probe.py": (sha(b"pre\n"), sha(b"post\n"))}
        actual_run = installer.subprocess.run
        with (
            patch.object(installer, "FILES", files),
            patch.object(observer, "FILES", files),
            patch.object(installer, "PATCH", payload),
            patch.object(installer, "PATCH_SHA", sha(payload.read_bytes())),
            patch.object(installer.subprocess, "run", wraps=actual_run) as calls,
        ):
            report = installer.install(self.root)
            self.assertEqual(report["state"], "patched")
            self.assertEqual((self.root / "probe.py").read_bytes(), b"post\n")
            self.assertEqual(calls.call_count, 1)
            installer.install(self.root)
            self.assertEqual(calls.call_count, 1)

    def test_mixed_toy_pair_refuses_before_patch_process(self):
        (self.root / "other.py").write_bytes(b"pre-other")
        files = {
            "probe.py": (sha(b"pre"), sha(b"post")),
            "other.py": (sha(b"pre-other"), sha(b"post-other")),
        }
        with patch.object(installer, "FILES", files), patch.object(observer, "FILES", files):
            with self.assertRaisesRegex(RuntimeError, "unknown or mixed"):
                installer.install(self.root)
        self.no_process.assert_not_called()

    def test_launcher_and_patched_guard_share_effective_cache(self):
        # Compile only these stdlib checkpoint guards from the actual vendored
        # payload. Never import/execute the SDK module or its engine imports.
        payload = installer.PATCH.read_text()

        def definition(name, following):
            block = payload.split("\n+def " + name + "(", 1)[1]
            block = "+def " + name + "(" + block.split("\n+def " + following + "(", 1)[0]
            return "\n".join(line[1:] for line in block.splitlines() if line.startswith("+"))

        worker_spec = importlib.util.spec_from_file_location(
            "sdk_checkpoint_worker", SCRIPTS.parent / "e2e_test/infra/sglang_gptoss.py"
        )
        worker = importlib.util.module_from_spec(worker_spec)
        worker_spec.loader.exec_module(worker)
        documents = {
            "config.json": b"{}",
            "model.safetensors.index.json": b'{"weight_map":{"toy":"model.safetensors"}}',
        }
        namespace = {
            "Path": Path,
            "os": os,
            "hashlib": hashlib,
            "json": json,
            "_GPTOSS_REVISION": worker.REVISION,
            "_GPTOSS_METADATA_SHA256": {name: sha(body) for name, body in documents.items()},
        }
        guards = definition("_original_shared_blob", "_validate_gptoss_local_checkpoint")
        guards += "\n" + definition(
            "_validate_gptoss_local_checkpoint", "_validate_gfx942_gptoss_bf16_model"
        )
        exec(compile(guards, str(installer.PATCH) + ":checkpoint-guards", "exec"), namespace)
        scenarios = (
            (
                {
                    "HF_HUB_CACHE": str(self.scope / "cache-a"),
                    "HF_HOME": str(self.scope / "ignored"),
                },
                {},
            ),
            ({"HF_HOME": str(self.scope / "home-b")}, {}),
            (
                {"HF_HUB_CACHE": str(self.scope / "old-cache")},
                {"HF_HUB_CACHE": str(self.scope / "cache-c")},
            ),
        )
        for inherited, extra in scenarios:
            with self.subTest(inherited=inherited, extra=extra):
                effective = {**inherited, **extra}
                cache = Path(effective.get("HF_HUB_CACHE") or Path(effective["HF_HOME"]) / "hub")
                snapshot = cache / "models--openai--gpt-oss-20b/snapshots" / worker.REVISION
                snapshot.mkdir(parents=True)
                for name, body in documents.items():
                    (snapshot / name).write_bytes(body)
                (snapshot / "model.safetensors").touch()
                env = {
                    **os.environ,
                    **inherited,
                    "ROUTER_LOCAL_MODEL_PATH": str(self.scope / "unused-map"),
                }
                with patch.dict(os.environ, env, clear=True):
                    selected, _ = worker.prepare_launch(
                        "sglang", worker.MODEL, worker.MODEL, {}, [], extra
                    )
                with patch.dict(os.environ, {**env, **extra}, clear=True):
                    validate = namespace["_validate_gptoss_local_checkpoint"]
                    self.assertEqual(validate(selected)["resolved_path"], str(snapshot))
                    with self.assertRaisesRegex(ValueError, "Unrecognized"):
                        validate(selected.replace(worker.REVISION, "0" * 40))
                    (snapshot / "config.json").write_bytes(b"changed")
                    with self.assertRaisesRegex(ValueError, "metadata differs"):
                        validate(selected)
                    outside = self.scope / "foreign-config.json"
                    outside.write_bytes(documents["config.json"])
                    (snapshot / "config.json").unlink()
                    (snapshot / "config.json").symlink_to(outside)
                    with self.assertRaisesRegex(ValueError, "escapes its original repository"):
                        validate(selected)


if __name__ == "__main__":
    unittest.main()
