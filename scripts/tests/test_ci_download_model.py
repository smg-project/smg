"""Real downloader Bash, owned cold cache and network-free HF invocation fixture."""

import ast
import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/ci_download_model.sh"
REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
MOVED_MAIN = "f" * 40
MODEL = "openai/gpt-oss-20b"
FAKE_HF = r"""#!/usr/bin/python3
import fcntl, json, os, sys
from pathlib import Path
args = sys.argv[1:]
assert args[0] == "download" and args[-1] == "--quiet", args
model = args[1]
lock = Path(os.environ["HF_HOME"]) / ".locks" / (model.replace("/", "_") + ".lock")
with lock.open("a") as stream:
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        pass
    else:
        raise RuntimeError("real downloader did not hold its model lock")
log = Path(os.environ["TEST_HF_LOG"])
with log.open("a") as stream:
    stream.write(json.dumps(args) + "\n")
if os.environ.get("TEST_FAIL_FIRST") == "1" and len(log.read_text().splitlines()) == 1:
    sys.exit(23)
repo = Path(os.environ["HF_HOME"]) / "hub" / ("models--" + model.replace("/", "--"))
revision = args[args.index("--revision") + 1] if "--revision" in args else "f" * 40
snapshot = repo / "snapshots" / revision
snapshot.mkdir(parents=True)
(snapshot / "invocation-fixture.txt").write_text(model)
if "--revision" not in args:
    (repo / "refs").mkdir(exist_ok=True)
    (repo / "refs/main").write_text(revision)
"""


class DownloadModelInvocationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ci-download-model-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        (self.bin / "hf").write_text(FAKE_HF)
        (self.bin / "hf").chmod(0o700)
        (self.bin / "sleep").write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$TEST_SLEEP_LOG"\n')
        (self.bin / "sleep").chmod(0o700)
        self.home = self.root / "hf"
        self.log = self.root / "calls.jsonl"
        self.sleep_log = self.root / "sleep.log"
        self.env = {
            "PATH": f"{self.bin}:/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "HF_HOME": str(self.home),
            "HF_XET_CACHE": str(self.root / "xet"),
            "RUNNER_TEMP": str(self.root),
            "TMPDIR": str(self.root),
            "TEST_HF_LOG": str(self.log),
            "TEST_SLEEP_LOG": str(self.sleep_log),
            "PYTHONDONTWRITEBYTECODE": "1",
        }

    def run_downloader(self, *args, pin=False, fail_first=False):
        env = dict(self.env)
        if pin:
            env["CI_SGLANG_GPTOSS_REVISION"] = REVISION
        if fail_first:
            env["TEST_FAIL_FIRST"] = "1"
        result = subprocess.run(
            ["/usr/bin/bash", str(SCRIPT), *args],
            env=env,
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(result.returncode, 0, (result.stdout, result.stderr))
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_cold_cache_pinned_model_downloads_only_required_snapshot(self):
        self.assertFalse(self.home.exists())
        self.assertEqual(
            self.run_downloader(MODEL, pin=True),
            [["download", MODEL, "--revision", REVISION, "--quiet"]],
        )
        snapshots = self.home / "hub/models--openai--gpt-oss-20b/snapshots"
        self.assertEqual([p.name for p in snapshots.iterdir()], [REVISION])
        self.assertFalse((snapshots / MOVED_MAIN).exists())

    def test_default_call_keeps_floating_gptoss_download(self):
        self.assertEqual(self.run_downloader(MODEL), [["download", MODEL, "--quiet"]])
        repo = self.home / "hub/models--openai--gpt-oss-20b"
        self.assertEqual((repo / "refs/main").read_text(), MOVED_MAIN)
        self.assertFalse((repo / "snapshots" / REVISION).exists())

    def test_other_model_ignores_job_specific_gptoss_revision(self):
        self.assertEqual(
            self.run_downloader("example/other-model", pin=True),
            [["download", "example/other-model", "--quiet"]],
        )

    def test_pin_stays_inside_existing_retry_lock_and_stale_cleanup(self):
        folders = [
            self.home / "hub/models--openai--gpt-oss-20b",
            self.home / "hub/.locks/models--openai--gpt-oss-20b",
        ]
        for folder in folders:
            folder.mkdir(parents=True)
            (folder / "old.lock").write_text("owned stale fixture")
            (folder / "fresh.lock").write_text("owned fresh fixture")
            stale = time.time() - 3600
            os.utime(folder / "old.lock", (stale, stale))
        self.assertEqual(
            self.run_downloader(MODEL, pin=True, fail_first=True),
            [["download", MODEL, "--revision", REVISION, "--quiet"]] * 2,
        )
        self.assertEqual(self.sleep_log.read_text().splitlines(), ["30"])
        for folder in folders:
            self.assertFalse((folder / "old.lock").exists())
            self.assertTrue((folder / "fresh.lock").exists())

    def test_gpu_tier_selection_preserves_tp_floor_and_skip_flag(self):
        package = self.root / "metadata/e2e_test/infra"
        package.mkdir(parents=True)
        (package.parent / "__init__.py").write_text("")
        (package / "__init__.py").write_text("")
        specs = {
            MODEL: {"tp": 1},
            "example/one-gpu": {"tp": 1},
            "example/two-gpu": {"tp": 2},
            "example/skipped": {"tp": 1, "skip_tier_download": True},
        }
        (package / "model_specs.py").write_text("MODEL_SPECS = " + repr(specs))
        self.env["PYTHONPATH"] = str(package.parents[1])
        self.assertEqual(
            self.run_downloader("--gpu-tier", "1", pin=True),
            [
                ["download", MODEL, "--revision", REVISION, "--quiet"],
                ["download", "example/one-gpu", "--quiet"],
            ],
        )

    def test_only_amd_sglang_final_environment_steps_supply_pin(self):
        import yaml

        workflow = yaml.safe_load((ROOT / ".github/workflows/e2e-gpu-job.yml").read_text())
        steps = workflow["jobs"]["run"]["steps"]
        downloads = [
            s for s in steps if s.get("name") in ["Download models", "Download extra models"]
        ]
        self.assertEqual(len(downloads), 2)
        gate = (
            "${{ inputs.final_environment_check && inputs.engine == 'sglang' && "
            "inputs.gpu_vendor == 'amd' && '" + REVISION + "' || '' }}"
        )
        for step in downloads:
            self.assertEqual(step["env"]["CI_SGLANG_GPTOSS_REVISION"], gate)
        numeric = ast.parse((ROOT / "scripts/ci_sglang_mxfp4_numeric.py").read_text())
        revisions = [
            node.value.value
            for node in numeric.body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "REVISION" for t in node.targets)
        ]
        self.assertEqual(revisions, [REVISION])


if __name__ == "__main__":
    unittest.main()
