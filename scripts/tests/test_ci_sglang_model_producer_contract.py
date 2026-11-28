"""Validate only the changed SDK Source contract; no provider or SDK execution."""

import hashlib
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


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bootstrap = load("ci_sglang_cohort_bootstrap")
delivery = load("ci_sglang_provider_delivery")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Contract(unittest.TestCase):
    def packet(self):
        return {
            "workflow_sha256": sha(SCRIPTS.parent / ".github/workflows/e2e-gpu-job.yml"),
            "source_sha256": {name: sha(SCRIPTS / name) for name in bootstrap.SOURCES},
        }

    def test_exact_new_installer_is_part_of_the_delivered_source_contract(self):
        packet = self.packet()
        self.assertIn("ci_install_sglang_gptoss20_sdk.py", packet["source_sha256"])
        self.assertNotIn("ci_sglang_gptoss20_profile.py", packet["source_sha256"])
        delivery.source_contract(packet, SCRIPTS.parent)

    def test_legacy_packet_is_rejected_before_any_provider_loader(self):
        packet = self.packet()
        packet["source_sha256"].pop("ci_install_sglang_gptoss20_sdk.py")
        packet["source_sha256"]["ci_sglang_gptoss20_profile.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "checkout Source"):
            delivery.source_contract(packet, SCRIPTS.parent)
        flags = types.SimpleNamespace(isolated=True, no_site=True, dont_write_bytecode=True)
        with tempfile.TemporaryDirectory(dir=SCRIPTS.parent.parent) as folder:
            source = Path(folder) / "legacy.json"
            source.write_text(json.dumps(packet))
            with (
                patch.object(bootstrap, "sys", types.SimpleNamespace(flags=flags)),
                patch.object(
                    bootstrap, "source_module", side_effect=AssertionError("No provider")
                ) as loader,
                self.assertRaisesRegex(ValueError, "Source contract differs"),
            ):
                bootstrap.build(source, sha(source), Path(folder) / "new-cohort")
            loader.assert_not_called()
            self.assertFalse((Path(folder) / "new-cohort").exists())


class ActualBootstrapManifest(unittest.TestCase):
    def test_production_helper_emits_mapping_consumed_by_real_edb_validator(self):
        cohort = load("ci_sglang_cohort")
        with patch.object(sys, "path", [str(SCRIPTS), *sys.path]):
            private_pip = load("ci_sglang_private_pip")
        with tempfile.TemporaryDirectory(
            prefix="actual-bootstrap-", dir=SCRIPTS.parent.parent
        ) as temp:
            root = Path(temp) / "runtime"
            root.mkdir()
            sdk_parent = root / "sdk"
            sdk_parent.mkdir()
            hook = Path(temp) / "inert-backend-hook.py"
            hook.write_text("raise RuntimeError('fixture hook is never executed')\n")
            # This is the production helper called by build(), including real
            # no-pip stdlib venv creation and the real cohort manifest writer.
            # No provider module, SDK, pip or backend is imported or launched.
            python, manifest = bootstrap.create_private_interpreter(
                cohort,
                root,
                sdk_parent,
                [],
                [],
                {"path": str(hook), "sha256": sha(hook)},
                image=private_pip.IMAGE,
            )
            produced = json.loads(manifest.read_text())
            projects = ("crates/grpc_client/python", "grpc_servicer")
            self.assertEqual(
                set(produced["editable_projects"]),
                {str((SCRIPTS.parent / name).resolve()) for name in projects},
            )
            constraint = Path(temp) / "inert-fixture-constraint.txt"
            constraint.write_text("torch==0+inert-fixture\n")
            arguments = ["--no-cache-dir", "-e", projects[0], "-e", projects[1]]
            with (
                patch.object(private_pip, "CONSTRAINT", constraint),
                patch.object(private_pip, "CONSTRAINT_SHA", sha(constraint)),
                patch.dict(os.environ, {"PIP_CONSTRAINT": str(constraint)}, clear=True),
                patch.object(
                    private_pip.subprocess, "run", side_effect=AssertionError("No pip")
                ) as no_pip,
            ):
                command, environment = private_pip.install_command(
                    manifest, arguments, cwd=SCRIPTS.parent
                )
                self.assertEqual(command[0], str(python))
                self.assertEqual(command[-len(arguments) :], arguments)
                self.assertEqual(command[command.index("--prefix") + 1], str(root / "native"))
                # The negative control removes only the mapping from a copy of
                # the produced manifest. The validator and target remain strict.
                produced["editable_projects"] = {}
                missing = manifest.with_name("missing-editable-map.json")
                missing.write_text(json.dumps(produced))
                with self.assertRaisesRegex(ValueError, "explicit cohort import mapping"):
                    private_pip.install_command(missing, arguments, cwd=SCRIPTS.parent)
                no_pip.assert_not_called()


if __name__ == "__main__":
    unittest.main()
