"""Real CPU subprocess stdio controls, without gateway/native/engine proof."""

import importlib.util
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryFile

import pytest

SPEC = importlib.util.spec_from_file_location(
    "epd_gateway_stdio", Path(__file__).with_name("gateway_proof.py")
)
PROOF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROOF)


@contextmanager
def child(**streams):
    with subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.buffer.read()"],
        stdin=subprocess.PIPE,
        **streams,
    ) as process:
        try:
            yield process
        finally:
            process.communicate(timeout=5)


def test_inherited_stdout_and_stderr_are_bound_separately(capfd):
    with child() as process:
        record = PROOF.stdio_binding(process, None)
        assert record["stdio_mode"] == "inherited_parent"
        assert record["log"] is None
        assert record["stdout"] == record["parent_stdout"]
        assert record["stderr"] == record["parent_stderr"]
        assert record["parent_stdout"] != record["parent_stderr"]


def test_captured_stdout_retains_log_inode_check(tmp_path):
    log_path = tmp_path / "router.log"
    with log_path.open("w") as log, child(stdout=log, stderr=subprocess.STDOUT) as process:
        record = PROOF.stdio_binding(process, log_path)
        assert record["stdio_mode"] == "captured_log"
        assert record["log"] == str(log_path)
        assert record["stdout"] == record["stderr"]
        assert record["stdout"] == {
            "device": log_path.stat().st_dev,
            "inode": log_path.stat().st_ino,
        }


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_mismatched_inherited_stream_is_rejected(stream):
    with TemporaryFile() as wrong, child(**{stream: wrong}) as process:
        with pytest.raises(ValueError, match=f"gateway {stream}/parent differs"):
            PROOF.stdio_binding(process, None)


def test_captured_stdout_with_another_log_is_rejected(tmp_path):
    wrong_path = tmp_path / "wrong.log"
    wrong_path.touch()
    with TemporaryFile() as actual, child(stdout=actual) as process:
        with pytest.raises(ValueError, match="gateway stdout/log differs"):
            PROOF.stdio_binding(process, wrong_path)
