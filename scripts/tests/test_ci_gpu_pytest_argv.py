"""CPU-only argv/security regressions for the actual GPU test-step Bash body."""

import json
import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/e2e-gpu-job.yml"
FIXED_BEFORE_FILTER = ["--ignore=e2e_test/benchmarks", "--ignore=e2e_test/bindings_go"]
FIXED_AFTER_FILTER = ["--reruns", "2", "--reruns-delay", "5", "-s", "-vv"]


class GPUArgvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = WORKFLOW.read_text()
        start = source.index("      - name: Run E2E tests\n")
        end = source.index("      - name: Worker failure diagnostics\n", start)
        cls.step = source[start:end]
        cls.body = textwrap.dedent(cls.step.split("        run: |\n", 1)[1])

    def run_step(self, dirs="e2e_test/", filter="", vendor="nvidia", exit_code=0):
        temporary = tempfile.TemporaryDirectory(prefix="smg-gpu-argv-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root / "scripts").mkdir()
        (root / "bin").mkdir()
        # Only private recording stubs run: never the repository GPU cleanup or pytest.
        (root / "scripts/ci_killall_sglang.sh").write_text(
            '#!/bin/bash\nprintf "%s\\n" "$1" >> cleanup.txt\n'
        )
        pytest = root / "bin/pytest"
        pytest.write_text(
            "#!/usr/bin/env python3\nimport json,os,pathlib,sys\n"
            'pathlib.Path("argv.json").write_text(json.dumps(sys.argv[1:]))\n'
            'sys.exit(int(os.environ["STUB_PYTEST_EXIT"]))\n'
        )
        pytest.chmod(0o755)
        env = {
            "PATH": str(root / "bin") + os.pathsep + os.environ["PATH"],
            "HOME": str(root),
            "GPU_VENDOR": vendor,
            "TEST_DIRS": dirs,
            "TEST_FILTER": filter,
            "AMD_DEPENDENCY_DIRECTORY": "evidence with spaces",
            "STUB_PYTEST_EXIT": str(exit_code),
        }
        # Model Actions' literal expansion for an intentional paired old-body check.
        # With the fixed body these input tokens are absent, so all values stay in env.
        body = self.body
        for key, value in [("gpu_vendor", vendor), ("test_dirs", dirs), ("test_filter", filter)]:
            body = body.replace("${{ inputs." + key + " }}", value)
        proc = subprocess.run(
            ["bash", "-e", "-o", "pipefail", "-c", body],
            cwd=root,
            env=env,
            capture_output=True,
            timeout=10,
        )
        capture = root / "argv.json"
        argv = json.loads(capture.read_text()) if capture.exists() else None
        cleanup = (
            (root / "cleanup.txt").read_text().splitlines()
            if (root / "cleanup.txt").exists()
            else []
        )
        return proc, argv, cleanup, root

    def test_inputs_are_bound_as_environment_data(self):
        for name in ("GPU_VENDOR", "TEST_DIRS", "TEST_FILTER"):
            key = name.lower()
            self.assertIn(name + ": ${{ inputs." + key + " }}", self.step)
            self.assertNotIn("${{ inputs." + key + " }}", self.body)

    def test_quoted_filter_and_paths_keep_argument_boundaries(self):
        proc, argv, cleanup, _ = self.run_step(
            dirs='"suite with spaces" e2e_test/router',
            filter="-k \"not slow and (http or grpc)\" -m 'not serial'",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(
            argv,
            ["suite with spaces", "e2e_test/router"]
            + FIXED_BEFORE_FILTER
            + ["-k", "not slow and (http or grpc)", "-m", "not serial"]
            + FIXED_AFTER_FILTER,
        )
        self.assertEqual(cleanup, ["nuke_gpus"])

    def test_amd_keeps_junit_single_argument_and_rocm_cleanup(self):
        proc, argv, cleanup, _ = self.run_step(vendor="amd")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(
            argv,
            ["e2e_test/"]
            + FIXED_BEFORE_FILTER
            + FIXED_AFTER_FILTER
            + ["--junitxml=evidence with spaces/e2e.xml"],
        )
        self.assertEqual(cleanup, ["rocm"])

    def test_command_substitution_stays_literal(self):
        proc, argv, _, root = self.run_step(
            filter='-k "$(touch injected)" -m "`touch injected2`"',
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertFalse((root / "injected").exists())
        self.assertFalse((root / "injected2").exists())
        self.assertEqual(
            argv,
            ["e2e_test/"]
            + FIXED_BEFORE_FILTER
            + ["-k", "$(touch injected)", "-m", "`touch injected2`"]
            + FIXED_AFTER_FILTER,
        )

    def test_invalid_quotes_fail_before_pytest(self):
        proc, argv, _, _ = self.run_step(filter='-k "unterminated')
        self.assertNotEqual(proc.returncode, 0)
        self.assertIsNone(argv)

    def test_pytest_nonzero_exit_is_preserved(self):
        proc, _, _, _ = self.run_step(exit_code=7)
        self.assertEqual(proc.returncode, 7)


if __name__ == "__main__":
    unittest.main()
