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
        self.calls = self.root / "calls.log"
        self.calls.touch()
        self.deletes = self.root / "deletes.log"
        self.deletes.touch()
        self.versions_tsv = self.root / "versions.tsv"

        gh = self.bin / "gh"
        gh.write_text(
            "#!/bin/bash\n"
            'echo "$*" >> "$FAKE_CALLS"\n'
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
            CARRIER_BASE,
            "--cuda-driver-version",
            "580.65.06",
        ]
        return subprocess.check_output(cmd, text=True).strip()

    def run_prune(self, rows: list[tuple[str, str, str]], **overrides: str):
        self.calls.write_text("")
        self.deletes.write_text("")
        self.versions_tsv.write_text("\n".join("\t".join(r) for r in rows) + "\n")
        env = (
            os.environ
            | {
                "PATH": f"{self.bin}:{os.environ['PATH']}",
                "FAKE_CALLS": str(self.calls),
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

    def test_prunes_per_carrier_prefix_protects_current_and_leaves_combined_runner_untouched(self):
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
            # combined runner tags must never be deleted automatically (fleet rollout may still pull them)
            ("401", "2026-11-27T06:00:00Z", "ci-gpu-runner-a"),
            ("402", "2026-11-26T06:00:00Z", "ci-gpu-runner-b"),
            ("403", "2026-11-25T06:00:00Z", "ci-gpu-runner-deployed"),
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

    def test_honours_nondefault_image_repository_and_skips_unparseable_repo(self):
        rows = [
            ("501", "2026-11-27T06:00:00Z", "ci-gpu-backend-sglang-a"),
            ("502", "2026-11-26T06:00:00Z", "ci-gpu-backend-sglang-b"),
            ("503", "2026-11-25T06:00:00Z", "ci-gpu-backend-sglang-c"),
        ]
        res, deleted = self.run_prune(
            rows,
            SMG_GPU_IMAGE_KEEP="2",
            SMG_CI_GPU_RUNNER_IMAGE_REPOSITORY="ghcr.io/custom-org/ci/carriers",
        )
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn(
            "/orgs/custom-org/packages/container/ci%2Fcarriers/versions?per_page=100",
            self.calls.read_text(),
        )
        self.assertEqual(
            deleted,
            ["/orgs/custom-org/packages/container/ci%2Fcarriers/versions/503"],
        )

        # Invalid override must skip instead of pruning the default package.
        res, deleted = self.run_prune(
            rows,
            SMG_GPU_IMAGE_KEEP="2",
            SMG_CI_GPU_RUNNER_IMAGE_REPOSITORY="docker.io/other/smg",
        )
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(deleted, [])
        self.assertEqual(self.calls.read_text().strip(), "")

    def test_dry_run_and_delete_errors_never_fail(self):
        rows = [
            ("601", "2026-11-27T06:00:00Z", "ci-gpu-backend-vllm-a"),
            ("602", "2026-11-26T06:00:00Z", "ci-gpu-backend-vllm-b"),
            ("603", "2026-11-25T06:00:00Z", "ci-gpu-backend-vllm-c"),
        ]
        res, deleted = self.run_prune(rows, DRY_RUN="1", SMG_GPU_IMAGE_KEEP="2")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(deleted, [])
        self.assertIn("would delete ci-gpu-backend-vllm-c", res.stdout)

        res, _ = self.run_prune(rows, FAKE_DELETE_FAIL="1", SMG_GPU_IMAGE_KEEP="2")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("WARNING: could not delete ci-gpu-backend-vllm-c", res.stdout)


if __name__ == "__main__":
    unittest.main()
