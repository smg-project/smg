"""CPU-only tests for scripts/ci_prune_gpu_runner_images.sh."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/ci_prune_gpu_runner_images.sh"
TAG_SCRIPT = REPO / "scripts/ci_gpu_runner_image.py"
RUNNER_BASE = (
    "ghcr.io/actions/actions-runner:2.337.0"
    "@sha256:5036480998280bb21e32ade9fe1b02b493861ac314b62ba1aea320b94f56ec97"
)
CARRIER_BASE = (
    "ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55"
)


class PruneGpuRunnerImagesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.deletes = self.root / "deletes.log"
        self.deletes.touch()
        self.versions_tsv = self.root / "versions.tsv"

        gh = self.bin / "gh"
        gh.write_text(
            "#!/bin/bash\n"
            'if [ "$1" = "api" ] && [ "$2" = "--method" ] && [ "$3" = "DELETE" ]; then\n'
            '  echo "$4" >> "$FAKE_DELETES"\n'
            '  [ "${FAKE_DELETE_FAIL:-0}" = 1 ] && { echo "forbidden" >&2; exit 1; }\n'
            "  exit 0\n"
            "fi\n"
            'cat "$FAKE_VERSIONS_TSV"\n'
        )
        gh.chmod(0o755)

    def current_tag(self, backend: str) -> str:
        cmd = [
            "python3",
            str(TAG_SCRIPT),
            "tag",
            "--backend",
            backend,
            "--base-image",
            RUNNER_BASE if backend == "all" else CARRIER_BASE,
            "--cuda-driver-version",
            "580.65.06",
        ]
        if backend == "all":
            cmd += ["--carrier-base-image", CARRIER_BASE]
        return subprocess.check_output(cmd, text=True).strip()

    def run_prune(self, rows: list[tuple[str, str, str]], **overrides: str):
        self.deletes.write_text("")
        self.versions_tsv.write_text("\n".join("\t".join(r) for r in rows) + "\n")
        env = (
            os.environ
            | {
                "PATH": f"{self.bin}:{os.environ['PATH']}",
                "FAKE_DELETES": str(self.deletes),
                "FAKE_VERSIONS_TSV": str(self.versions_tsv),
                "GITHUB_REPOSITORY_OWNER": "smg-project",
            }
            | overrides
        )
        result = subprocess.run(
            ["bash", str(SCRIPT)],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        deleted = [line for line in self.deletes.read_text().splitlines() if line]
        return result, deleted

    def test_prunes_per_prefix_protects_current_and_ignores_other_tags(self):
        cur_vllm = self.current_tag("vllm")
        cur_trt = self.current_tag("trtllm")
        rows = [
            # vllm: 4 versions, current is oldest -> keep 2 newest + current, delete 103
            ("101", "2026-11-27T06:00:00Z", "ci-gpu-backend-vllm-newest1"),
            ("102", "2026-11-26T06:00:00Z", "ci-gpu-backend-vllm-newest2"),
            ("103", "2026-11-25T06:00:00Z", "ci-gpu-backend-vllm-old"),
            ("104", "2026-11-20T06:00:00Z", cur_vllm),
            # trtllm: 3 versions -> keep 2 newest (one is current), delete 203
            ("201", "2026-11-27T06:00:00Z", cur_trt),
            ("202", "2026-11-26T06:00:00Z", "ci-gpu-backend-trtllm-prev"),
            ("203", "2026-11-24T06:00:00Z", "ci-gpu-backend-trtllm-stale"),
            # mixed-tag version & unrelated tags must never be deleted
            ("301", "2026-11-01T06:00:00Z", "ci-gpu-backend-vllm-mixed,latest"),
            ("302", "2026-11-01T06:00:00Z", "nightly"),
            ("303", "2026-11-01T06:00:00Z", "ci-tokenspeed-20261102"),
        ]
        res, deleted = self.run_prune(rows, SMG_GPU_IMAGE_KEEP="2")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(
            deleted,
            [
                "/orgs/smg-project/packages/container/smg/versions/103",
                "/orgs/smg-project/packages/container/smg/versions/203",
            ],
        )

    def test_dry_run_and_delete_errors_never_fail(self):
        rows = [
            ("401", "2026-11-27T06:00:00Z", "ci-gpu-runner-a"),
            ("402", "2026-11-26T06:00:00Z", "ci-gpu-runner-b"),
            ("403", "2026-11-25T06:00:00Z", "ci-gpu-runner-c"),
        ]
        res, deleted = self.run_prune(rows, DRY_RUN="1", SMG_GPU_IMAGE_KEEP="2")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(deleted, [])
        self.assertIn("would delete ci-gpu-runner-c", res.stdout)

        res, _ = self.run_prune(rows, FAKE_DELETE_FAIL="1", SMG_GPU_IMAGE_KEEP="2")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("WARNING: could not delete ci-gpu-runner-c", res.stdout)


if __name__ == "__main__":
    unittest.main()
