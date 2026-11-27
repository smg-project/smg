"""The workflow's exact profile guards and runtime names must agree."""

import importlib.util
import os
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


class StartupProfileNamesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[2] / "e2e_test/infra/constants.py"
        spec = importlib.util.spec_from_file_location("startup_profile_name_contract", path)
        cls.constants = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.constants
        spec.loader.exec_module(cls.constants)

    def declaration(self, value):
        with patch.dict(os.environ, {"E2E_STARTUP_PROFILE": value}):
            return self.constants._startup_profile_declaration()

    def test_workflow_guards_cover_exactly_the_registered_names(self):
        workflow = Path(__file__).resolve().parents[2] / ".github/workflows/e2e-gpu-job.yml"
        guarded = set(re.findall(r"inputs\.startup_profile\s*==\s*'([^']+)'", workflow.read_text()))
        self.assertEqual(guarded, set(self.constants._STARTUP_PROFILES))

    def test_blank_profile_keeps_ordinary_behavior(self):
        for value in ("", " ", "\t\n"):
            with self.subTest(value=value):
                self.assertIsNone(self.declaration(value))

    def test_canonical_registered_profiles_remain_accepted(self):
        self.assertIn("vllm-mooncake-pd2", self.constants._STARTUP_PROFILES)
        for name in self.constants._STARTUP_PROFILES:
            with self.subTest(name=name):
                self.assertEqual(self.declaration(name)[0], name)

    def test_noncanonical_registered_profiles_are_rejected(self):
        for name in self.constants._STARTUP_PROFILES:
            for value in (" " + name, name + " ", "\t" + name, name + "\n"):
                with self.subTest(value=value):
                    with self.assertRaisesRegex(ValueError, "canonical"):
                        self.declaration(value)


if __name__ == "__main__":
    unittest.main()
