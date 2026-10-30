"""Model-download retries preserve active cache locks; all paths are temporary."""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "ci_download_model.sh"
MODEL = "test-org/test-model"


def download_options(root: Path, exits: list[int]) -> dict:
    tools = root / "tools"
    tools.mkdir(exist_ok=True)
    (tools / "hf").write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "calls = Path(os.environ['DOWNLOAD_CALLS'])\n"
        "previous = calls.read_text().splitlines() if calls.exists() else []\n"
        "with calls.open('a') as f: f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "raise SystemExit(json.loads(os.environ['DOWNLOAD_EXITS'])[len(previous)])\n"
    )
    (tools / "sleep").write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "with open(os.environ['SLEEP_CALLS'], 'a') as f: f.write(sys.argv[1] + '\\n')\n"
    )
    for tool in tools.iterdir():
        tool.chmod(0o755)
    return {
        "args": ["bash", str(SCRIPT), MODEL],
        "env": {
            **os.environ,
            "PATH": f"{tools}:{os.environ['PATH']}",
            "HF_HOME": str(root / "cache"),
            "HF_XET_CACHE": str(root / "xet"),
            "DOWNLOAD_CALLS": str(root / "downloads"),
            "DOWNLOAD_EXITS": json.dumps(exits),
            "SLEEP_CALLS": str(root / "sleeps"),
        },
        "text": True,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
    }


def download(root: Path, exits: list[int]) -> subprocess.CompletedProcess:
    return subprocess.run(**download_options(root, exits), timeout=10, check=False)


def lock_owner(lock: Path) -> subprocess.Popen:
    # Pass an explicitly writable FD, then close the parent's copy before waiting.
    with lock.open("a") as shared:
        return subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import fcntl, sys; fcntl.flock(int(sys.argv[1]), fcntl.LOCK_EX); "
                "print('locked', flush=True); sys.stdin.read(1)",
                str(shared.fileno()),
            ],
            pass_fds=(shared.fileno(),),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )


def contender(lock: Path) -> int:
    return subprocess.run(["flock", "-n", str(lock), "true"], timeout=5, check=False).returncode


def require_append_denied(lock: Path) -> None:
    try:
        with lock.open("a"):
            pass
    except PermissionError:
        return
    pytest.skip("this process can bypass the read-only mode bits")


@pytest.mark.parametrize(
    "location", ["models--test-org--test-model", ".locks/models--test-org--test-model"]
)
def test_retry_preserves_old_lock_held_by_another_process(tmp_path, location):
    lock = tmp_path / "cache/hub" / location / "blob.lock"
    lock.parent.mkdir(parents=True)
    owner = lock_owner(lock)
    try:
        assert owner.stdout is not None
        assert select.select([owner.stdout], [], [], 5)[0], "lock holder did not become ready"
        assert owner.stdout.readline().strip() == "locked"
        assert contender(lock) != 0
        os.utime(lock, (1, 1))
        original_inode = lock.stat().st_ino
        result = download(tmp_path, [1, 0])
        assert result.returncode == 0, result.stdout + result.stderr
        second_acquired = contender(lock) == 0
        observation = {
            "original_inode": original_inode,
            "current_inode": lock.stat().st_ino,
            "owner_alive": owner.poll() is None,
            "second_acquired_while_owner_alive": second_acquired,
        }
        print(json.dumps(observation, sort_keys=True))
        assert observation["owner_alive"]
        assert observation["current_inode"] == original_inode
        assert not second_acquired
        assert (tmp_path / "sleeps").read_text().splitlines() == ["30"]
        assert len((tmp_path / "downloads").read_text().splitlines()) == 2
    finally:
        owner.communicate("x", timeout=5)
    assert owner.returncode == 0
    assert contender(lock) == 0


def test_failed_download_still_tries_three_times_with_original_delay(tmp_path):
    result = download(tmp_path, [1, 1, 1])
    assert result.returncode != 0
    assert (tmp_path / "sleeps").read_text().splitlines() == ["30", "30"]
    calls = [json.loads(line) for line in (tmp_path / "downloads").read_text().splitlines()]
    assert calls == [["download", MODEL, "--quiet"]] * 3
    assert "flock -w 1800 200" in SCRIPT.read_text()


def test_readonly_lock_waits_for_writable_holder_before_download(tmp_path):
    lock = tmp_path / "cache/.locks/test-org_test-model.lock"
    lock.parent.mkdir(parents=True)
    lock.write_bytes(b"existing shared lock\n")
    owner = lock_owner(lock)
    downloader = None
    try:
        assert owner.stdout is not None
        assert select.select([owner.stdout], [], [], 5)[0], "lock holder did not become ready"
        assert owner.stdout.readline().strip() == "locked"
        lock.chmod(0o444)
        original = lock.stat()
        require_append_denied(lock)
        downloader = subprocess.Popen(**download_options(tmp_path, [0]))
        assert downloader.stdout is not None
        # Read the real fallback message before checking that flock still blocks.
        prefix = b""
        deadline = time.monotonic() + 5
        while b"Using existing read-only lock" not in prefix:
            remaining = deadline - time.monotonic()
            assert remaining > 0, "downloader did not reach the read-only fallback"
            assert select.select([downloader.stdout], [], [], remaining)[0]
            block = os.read(downloader.stdout.fileno(), 4096)
            assert block, prefix.decode()
            prefix += block
        assert owner.poll() is None
        assert contender(lock) != 0
        with pytest.raises(subprocess.TimeoutExpired):
            downloader.wait(timeout=0.2)
        assert not (tmp_path / "downloads").exists()
    finally:
        owner.communicate("x", timeout=5)
        if downloader is not None:
            try:
                stdout, stderr = downloader.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                downloader.kill()
                downloader.communicate()
                raise
    assert owner.returncode == 0
    assert downloader is not None
    assert downloader.returncode == 0, stdout + stderr
    calls = [json.loads(line) for line in (tmp_path / "downloads").read_text().splitlines()]
    assert calls == [["download", MODEL, "--quiet"]]
    current = lock.stat()
    assert (current.st_dev, current.st_ino, current.st_mode, current.st_uid, current.st_gid) == (
        original.st_dev,
        original.st_ino,
        original.st_mode,
        original.st_uid,
        original.st_gid,
    )
    assert lock.read_bytes() == b"existing shared lock\n"
    assert contender(lock) == 0


@pytest.mark.parametrize("kind", ["symlink", "missing"])
def test_readonly_fallback_rejects_invalid_existing_lock(tmp_path, kind):
    lock = tmp_path / "cache/.locks/test-org_test-model.lock"
    lock.parent.mkdir(parents=True)
    target = tmp_path / "existing-lock"
    target.write_bytes(b"untouched lock target\n")
    target.chmod(0o444)
    original = target.stat()
    if kind == "symlink":
        lock.symlink_to(target)
    lock.parent.chmod(0o555)
    try:
        # Both cases must reach the fallback after an actual failed append-open.
        require_append_denied(lock)
        result = download(tmp_path, [0])
        assert result.returncode != 0
        assert "Cannot open shared lock" in result.stderr
        assert not (tmp_path / "downloads").exists()
        assert not (tmp_path / "sleeps").exists()
        if kind == "symlink":
            assert lock.is_symlink() and lock.readlink() == target
        else:
            assert not lock.exists() and not lock.is_symlink()
        current = target.stat()
        assert (current.st_ino, current.st_mode, current.st_uid, current.st_gid) == (
            original.st_ino,
            original.st_mode,
            original.st_uid,
            original.st_gid,
        )
        assert target.read_bytes() == b"untouched lock target\n"
    finally:
        lock.parent.chmod(0o755)
