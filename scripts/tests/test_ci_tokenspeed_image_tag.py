"""The carrier tag and version scripts/ci_tokenspeed_image_tag.sh derives from the nightly pin."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "ci_tokenspeed_image_tag.sh"
PIN = Path(__file__).resolve().parents[2] / ".github" / "versions" / "tokenspeed-nightly.txt"


def run(pin: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        env={"PATH": "/usr/bin:/bin", "TOKENSPEED_NIGHTLY_PIN": str(pin)},
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_tag_is_the_pinned_nightly_date(tmp_path):
    pin = tmp_path / "pin"
    pin.write_text("0.1.1.post20261021\n")
    result = run(pin)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "ci-tokenspeed-20261021\n"


def test_version_prints_the_pin_itself(tmp_path):
    pin = tmp_path / "pin"
    pin.write_text("  0.1.1.post20261021  \n")
    result = run(pin, "--version")
    assert result.returncode == 0, result.stderr
    assert result.stdout == "0.1.1.post20261021\n"


@pytest.mark.parametrize(
    "content", ["", "latest\n", "0.1.1\n", "0.1.1.post2026102\n", "dd70f5dce4372d64\n"]
)
def test_anything_but_a_dated_nightly_is_refused(tmp_path, content):
    pin = tmp_path / "pin"
    pin.write_text(content)
    result = run(pin)
    assert result.returncode == 1
    assert result.stdout == ""
    assert "dated TokenSpeed nightly version" in result.stderr


def test_the_repository_pin_is_a_dated_nightly():
    result = run(PIN, "--version")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().count(".post") == 1
