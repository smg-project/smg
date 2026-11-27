"""The shared-runner device guard must survive Python optimization."""

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("actual", [0, 1, 2])
def test_shared_device_count_guard(optimized, actual):
    workflow = Path(__file__).resolve().parents[2] / ".github/workflows/e2e-gpu-job.yml"
    steps = yaml.safe_load(workflow.read_text(encoding="utf-8"))["jobs"]["run"]["steps"]
    guard = next(s for s in steps if s.get("name") == "Verify shared GPU admission")
    assert guard["if"] == "inputs.gpu_lease"
    command = next(line for line in guard["run"].splitlines() if line.startswith("python3 -c "))
    code = shlex.split(command)[2]
    # Run the actual workflow snippet, substituting only device enumeration.
    wrapped = (
        "from unittest.mock import patch\n"
        f"with patch('pathlib.Path.glob', return_value={['device'] * actual!r}):\n"
        f"    exec({code!r})\n"
    )
    result = subprocess.run(
        [sys.executable, *(["-O"] if optimized else []), "-c", wrapped],
        env={**os.environ, "AMD_CI_GPU_COUNT": "1"},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == (0 if actual == 1 else 1), result.stderr
    if actual != 1:
        assert "shared GPU device count mismatch" in result.stderr
