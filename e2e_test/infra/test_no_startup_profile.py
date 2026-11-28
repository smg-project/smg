"""A startup profile left in the shell fails no unit test that drives worker setup."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from infra.constants import _STARTUP_PROFILES, ENV_STARTUP_PROFILE

ROOT = Path(__file__).resolve().parents[2]


def test_a_profile_left_in_the_shell_fails_no_setup_test() -> None:
    """A real session, as CI starts it, over a file that drives worker setup."""
    env = dict(os.environ)
    env[ENV_STARTUP_PROFILE] = next(iter(_STARTUP_PROFILES))
    env["PYTHONPATH"] = "e2e_test"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--noconftest",
            "-p",
            "no:cacheprovider",
            "e2e_test/infra/test_startup_timeout.py",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
