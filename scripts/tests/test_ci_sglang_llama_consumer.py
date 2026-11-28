"""Only new ordinary consumer ownership/readonly/phase controls; no SDK imports."""

import ast
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]
BOOTSTRAP = SCRIPTS / "ci_sglang_llama_cohort.py"
PROFILE = SCRIPTS / "ci_sglang_llama_profile.py"


def load_bootstrap():
    module = types.ModuleType("ordinary_cohort_source_tests")
    module.__file__ = str(BOOTSTRAP)
    exec(compile(BOOTSTRAP.read_bytes(), str(BOOTSTRAP), "exec"), module.__dict__)
    return module


class OrdinaryOwnership(unittest.TestCase):
    def setUp(self):
        self.boot = load_bootstrap()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.env = {
            "RUNNER_TEMP": str(self.root),
            "GITHUB_RUN_ID": "123",
            "GITHUB_RUN_ATTEMPT": "1",
            "E2E_JOB_MARKER": "owned-unit-fixture",
        }
        self.patch = mock.patch.dict(os.environ, self.env)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.addCleanup(self.temp.cleanup)

    def owned(self):
        root = self.root / "only-owned"
        root.mkdir()
        owner = {"root": str(root), "nonce": "unit-known-nonce", "job": self.boot.job_identity()}
        (root / "ordinary-owner.json").write_text(json.dumps(owner))
        (root / "data").write_text("owned-fixture")
        return root

    def test_exact_owned_cleanup(self):
        root = self.owned()
        self.assertFalse(self.boot.cleanup(root, "unit-known-nonce")["exists"])

    def test_wrong_nonce_keeps_directory(self):
        root = self.owned()
        with self.assertRaisesRegex(ValueError, "identity differs"):
            self.boot.cleanup(root, "foreign")
        self.assertTrue((root / "data").exists())

    def test_other_job_keeps_directory(self):
        root = self.owned()
        with mock.patch.dict(os.environ, {"GITHUB_RUN_ID": "456"}):
            with self.assertRaisesRegex(ValueError, "identity differs"):
                self.boot.cleanup(root, "unit-known-nonce")
        self.assertTrue(root.exists())

    def test_symlink_scope_is_never_adopted(self):
        root = self.owned()
        alias = self.root / "alias"
        alias.symlink_to(root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Unaliased"):
            self.boot.cleanup(alias, "unit-known-nonce")
        self.assertTrue(root.exists())

    def test_missing_nonce_is_never_adopted(self):
        root = self.owned()
        with self.assertRaisesRegex(ValueError, "nonce required"):
            self.boot.cleanup(root, "")
        self.assertTrue(root.exists())

    def test_readonly_roots_without_remount_capability(self):
        # These are explicit VFS/capability fixtures, not an image observation.
        with mock.patch.object(Path, "read_text", return_value="CapEff:\t00000000\n"):
            with mock.patch.object(
                os, "statvfs", return_value=types.SimpleNamespace(f_flag=os.ST_RDONLY)
            ):
                self.boot.readonly_providers([self.root])

    def test_writable_root_rejected(self):
        with mock.patch.object(Path, "read_text", return_value="CapEff:\t00000000\n"):
            with mock.patch.object(os, "statvfs", return_value=types.SimpleNamespace(f_flag=0)):
                with self.assertRaisesRegex(ValueError, "Read-only provider"):
                    self.boot.readonly_providers([self.root])

    def test_remount_capability_rejected(self):
        with mock.patch.object(Path, "read_text", return_value="CapEff:\t00200000\n"):
            with self.assertRaisesRegex(ValueError, "remountable"):
                self.boot.readonly_providers([self.root])

    def test_actual_cli_requires_all_creator_inputs(self):
        import subprocess

        result = subprocess.run(
            ["/usr/bin/python3", "-I", "-S", "-B", str(BOOTSTRAP)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Complete packet path/hash", result.stderr)


class OrdinaryContext(unittest.TestCase):
    def context(self):
        source = PROFILE.read_text()
        function = next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == "context"
        )
        observer = ast.parse((SCRIPTS / "ci_sglang_mxfp4_fallback.py").read_text())
        row_name = next(
            ast.literal_eval(node.value)
            for node in observer.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "ROW_ENV" for target in node.targets
            )
        )
        image = load_bootstrap().packet_image()
        self.metadata = mock.Mock(
            side_effect=lambda name: {"sglang": "0.5.21", "torch": "2.11.0+rocm10.0.0"}[name]
        )
        namespace = {
            "os": os,
            "md": types.SimpleNamespace(version=self.metadata),
            "sdk": types.SimpleNamespace(ROW_ENV=row_name),
            "native": types.SimpleNamespace(IMAGE=image),
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(PROFILE), "exec"), namespace)
        return namespace["context"], image

    def test_explicit_ordinary_context(self):
        context, image = self.context()
        with mock.patch.dict(
            os.environ,
            {"E2E_ENGINE": "sglang", "E2E_GPU_VENDOR": "amd", "SMG_SGLANG_ROCM_IMAGE": image},
            clear=True,
        ):
            context()
        self.assertEqual(self.metadata.call_count, 2)

    def test_20b_inputs_rejected_before_metadata(self):
        for key, value in [
            ("E2E_SGLANG_GPTOSS_MXFP4_BF16", "1"),
            ("SGLANG_GPTOSS_MXFP4_BF16_FALLBACK", "1"),
            ("SMG_GPTOSS_COHORT_MANIFEST", "/foreign20b/manifest.json"),
        ]:
            with self.subTest(key=key):
                context, image = self.context()
                env = {
                    "E2E_ENGINE": "sglang",
                    "E2E_GPU_VENDOR": "amd",
                    "SMG_SGLANG_ROCM_IMAGE": image,
                    key: value,
                }
                with mock.patch.dict(os.environ, env, clear=True):
                    with self.assertRaisesRegex(ValueError, "ordinary AMD SGLang context"):
                        context()
                self.metadata.assert_not_called()


class OrdinaryPhaseReports(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.calls = []
        self.native = types.SimpleNamespace(verify=self.native_fixture)
        source = PROFILE.read_text()
        function = next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == "verify"
        )
        namespace = {"context": lambda: None, "native": self.native, "json": json}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(PROFILE), "exec"), namespace)
        self.verify = namespace["verify"]
        self.cohort = {"sha256": "fixture-only", "sources": {}, "workflow_sha256": "fixture-only"}

    def native_fixture(self, report, **kwargs):
        self.calls.append(kwargs)
        report.update(versions={"fixture": "only"}, pip_check={"returncode": 0})

    def test_private_prepared_final_same_contract(self):
        result = self.verify("prepared", self.root, cohort=self.cohort)
        (self.root / "llama-cohort-prepared.json").write_text(json.dumps(result))
        self.verify("final", self.root, cohort=self.cohort)
        self.assertEqual(self.calls, [{"private_cohort": True}, {"private_cohort": True}])

    def test_final_rejects_changed_manifest(self):
        result = self.verify("prepared", self.root, cohort=self.cohort)
        result["contract"]["manifest_sha256"] = "changed-fixture"
        (self.root / "llama-cohort-prepared.json").write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, "Prepared/final ordinary cohort differs"):
            self.verify("final", self.root, cohort=self.cohort)

    def test_partial_native_failure_is_preserved(self):
        def fail(report, **kwargs):
            report["partial_fixture"] = "saved"
            raise RuntimeError("fixture-native-failure")

        self.native.verify = fail
        with self.assertRaisesRegex(RuntimeError, "fixture-native-failure"):
            self.verify("prepared", self.root, cohort=self.cohort)
        report = json.loads((self.root / "sglang-native-environment-prepared.json").read_text())
        self.assertEqual(report["partial_fixture"], "saved")
        self.assertFalse(report["verified"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
