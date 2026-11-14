"""scripts/ci_install_wasm_tools.sh: the wasm-tools CLI from its release, pinned to the release of the
wit-component crate the policy crate's tests depend on, checked against its sha256 before anything is unpacked.

Run with: pytest -q scripts/tests/test_ci_install_wasm_tools.py
"""

from __future__ import annotations

import hashlib
import io
import os
import platform
import re
import subprocess
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "ci_install_wasm_tools.sh"
POLICY_MANIFEST = REPO / "crates" / "policy_controller" / "Cargo.toml"

VERSION = re.search(r'^VERSION="([^"]+)"$', SCRIPT.read_text(), re.MULTILINE).group(1)
ARCH = {"x86_64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}[platform.machine()]
NAME = f"wasm-tools-{VERSION}-{ARCH}-linux"
ASSET = f"{NAME}.tar.gz"

# The wasm-tools of a release archive: answers the pinned version.
RELEASE_WASM_TOOLS = f"#!/bin/bash\necho 'wasm-tools {VERSION} (stub)'\n"


class Bed:
    """A release directory with a stub archive, a prefix and a GITHUB_PATH file of the test's own."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.prefix = root / "prefix"
        self.github_path = root / "github_path"
        release = root / "release"
        release.mkdir()
        body = RELEASE_WASM_TOOLS.encode()
        with tarfile.open(release / ASSET, "w:gz") as archive:
            info = tarfile.TarInfo(f"{NAME}/wasm-tools")
            info.mode = 0o755
            info.size = len(body)
            archive.addfile(info, io.BytesIO(body))
        self.sha256 = hashlib.sha256((release / ASSET).read_bytes()).hexdigest()

    def run(self, **extra: str) -> subprocess.CompletedProcess[str]:
        env = {
            "PATH": os.environ["PATH"],
            "HOME": str(self.root),
            "CI_WASM_TOOLS_PREFIX": str(self.prefix),
            "CI_WASM_TOOLS_URL_BASE": (self.root / "release").as_uri(),
            "GITHUB_PATH": str(self.github_path),
        }
        env.update(extra)
        return subprocess.run(
            ["bash", str(SCRIPT)], env=env, capture_output=True, text=True, check=False
        )


@pytest.fixture
def bed(tmp_path: Path) -> Bed:
    return Bed(tmp_path)


def test_the_release_is_installed_under_the_prefix_and_put_on_the_path(bed: Bed) -> None:
    result = bed.run(CI_WASM_TOOLS_SHA256=bed.sha256)
    assert result.returncode == 0, result.stderr
    tool = bed.prefix / "bin" / "wasm-tools"
    assert tool.exists() and os.access(tool, os.X_OK)
    assert str(bed.prefix / "bin") in bed.github_path.read_text().splitlines()
    assert f"installed wasm-tools {VERSION}" in result.stdout


def test_a_release_with_the_wrong_hash_installs_nothing(bed: Bed) -> None:
    result = bed.run(CI_WASM_TOOLS_SHA256="0" * 64)
    assert result.returncode != 0
    assert not (bed.prefix / "bin" / "wasm-tools").exists()
    assert not bed.github_path.exists()
    assert "does not match its pinned sha256" in result.stderr


def test_the_pinned_archive_is_not_the_stub(bed: Bed) -> None:
    # Without the test's own hash the script holds the stub to the real
    # release's pinned sha256, and refuses it.
    result = bed.run()
    assert result.returncode != 0
    assert not (bed.prefix / "bin" / "wasm-tools").exists()


def test_the_pin_is_the_release_of_the_policy_crates_wit_component() -> None:
    # One version of the component tooling: the encoder the crate's tests
    # link (wit-component 0.x.y) and the CLI that wraps the example plugin
    # (wasm-tools 1.x.y) come from one release.
    manifest = POLICY_MANIFEST.read_text()
    crate = re.search(
        r'^wit-component = \{ version = "0\.([0-9]+\.[0-9]+)"', manifest, re.MULTILINE
    )
    assert crate, "the policy crate no longer names wit-component with a 0.x.y version"
    assert VERSION == f"1.{crate.group(1)}"


def test_both_architectures_carry_a_pinned_hash() -> None:
    script = SCRIPT.read_text()
    pins = re.findall(r'pinned="([0-9a-f]{64})"', script)
    assert len(pins) == 2 and len(set(pins)) == 2, pins
