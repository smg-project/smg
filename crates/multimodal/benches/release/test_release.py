"""A test of the release runner that takes no measurement."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("release_run", HERE / "run.py")
assert SPEC and SPEC.loader
release_run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release_run)


class ReleaseRunnerTests(unittest.TestCase):
    def test_failed_window_is_not_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory) / "window"

            def fail(command, **_kwargs):
                Path(command[-1]).write_text('{"kind":"meta"}\n')
                return SimpleNamespace(returncode=9)

            with (
                mock.patch.object(release_run, "sibling_cpus", return_value=[]),
                mock.patch.object(release_run.subprocess, "run", side_effect=fail),
                self.assertRaisesRegex(RuntimeError, "failed after 3 attempts"),
            ):
                release_run.window([], 0, stem, 10.0, 1, mock.Mock(), "test")
            self.assertFalse(Path(f"{stem}.jsonl").exists())
            self.assertTrue(Path(f"{stem}.jsonl.bad3").exists())
