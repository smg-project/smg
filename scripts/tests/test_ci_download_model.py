"""Model-download retries preserve active cache locks; all paths are temporary."""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "ci_download_model.sh"
MODEL = "test-org/test-model"


def download(root: Path, exits: list[int]) -> subprocess.CompletedProcess:
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
    return subprocess.run(
        ["bash", str(SCRIPT), MODEL],
        env={
            **os.environ,
            "PATH": f"{tools}:{os.environ['PATH']}",
            "HF_HOME": str(root / "cache"),
            "HF_XET_CACHE": str(root / "xet"),
            "DOWNLOAD_CALLS": str(root / "downloads"),
            "DOWNLOAD_EXITS": json.dumps(exits),
            "SLEEP_CALLS": str(root / "sleeps"),
        },
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )


def contender(lock: Path) -> int:
    return subprocess.run(["flock", "-n", str(lock), "true"], timeout=5, check=False).returncode


@pytest.mark.parametrize(
    "location", ["models--test-org--test-model", ".locks/models--test-org--test-model"]
)
def test_retry_preserves_old_lock_held_by_another_process(tmp_path, location):
    lock = tmp_path / "cache/hub" / location / "blob.lock"
    lock.parent.mkdir(parents=True)
    owner = subprocess.Popen(
        [
            "flock",
            "-x",
            str(lock),
            sys.executable,
            "-c",
            "import sys; print('locked', flush=True); sys.stdin.read(1)",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
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
