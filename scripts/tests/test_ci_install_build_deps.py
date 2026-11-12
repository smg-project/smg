"""scripts/ci_install_build_deps.sh: the build's native dependencies, only the missing ones, and protoc from its
release when the archive does not answer.

Run with: pytest -q scripts/tests/test_ci_install_build_deps.py
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "ci_install_build_deps.sh"

# The utilities the script and its helpers call, linked into the bed's own
# PATH so the host's compiler, pkg-config and protoc are never found by chance.
UTILITIES = [
    "bash",
    "sh",
    "cat",
    "chmod",
    "cmp",
    "cp",
    "curl",
    "cut",
    "dirname",
    "basename",
    "env",
    "grep",
    "head",
    "mkdir",
    "mktemp",
    "mv",
    "rm",
    "sed",
    "sha256sum",
    "sleep",
    "sort",
    "tail",
    "tee",
    "timeout",
    "tr",
    "uname",
    "unzip",
    "wc",
]

STUB = "#!/bin/bash\nexit 0\n"

# A stub apt-get: one line per call in CALLS, the exit code of the n-th call
# from EXITS (0 past its end).
APT_GET = """\
#!/bin/bash
echo "$*" >> "$CALLS"
n=$(wc -l < "$CALLS")
code=$(sed -n "${n}p" "$EXITS")
[ -n "$code" ] || code=0
exit "$code"
"""

# A stub pkg-config: `--exists openssl` answers as OPENSSL says.
PKG_CONFIG = """\
#!/bin/bash
if [ "${1:-}" = "--exists" ]; then exit "${OPENSSL:-0}"; fi
exit 0
"""

# The protoc of a release zip: answers the version the archive ships.
RELEASE_PROTOC = "#!/bin/bash\necho 'libprotoc 3.21.12'\n"

ARCH = {"x86_64": "x86_64", "aarch64": "aarch_64", "arm64": "aarch_64"}[platform.machine()]
ASSET = f"protoc-21.12-linux-{ARCH}.zip"


class Bed:
    """A host with the named tools on PATH and nothing else of the real one."""

    def __init__(self, root: Path, tools: list[str], openssl: bool, apt_exits: list[int]) -> None:
        self.root = root
        self.bin = root / "bin"
        self.bin.mkdir()
        for name in UTILITIES:
            real = shutil.which(name)
            if real:
                os.symlink(real, self.bin / name)
        self.calls = root / "calls"
        self.calls.touch()
        self.exits = root / "exits"
        self.exits.write_text("".join(f"{code}\n" for code in apt_exits))
        # No apt.conf.d under the apt root: the mirror script leaves the host alone.
        self.apt_root = root / "etc-apt"
        self.apt_root.mkdir()
        self.prefix = root / "prefix"
        self.github_path = root / "github_path"
        self.openssl = openssl
        for name in tools:
            self.stub(name, PKG_CONFIG if name == "pkg-config" else STUB)
        self.stub("apt-get", APT_GET)
        self.stub("dpkg", STUB)

    def stub(self, name: str, text: str) -> None:
        path = self.bin / name
        path.write_text(text)
        path.chmod(0o755)

    def release(self) -> str:
        """A release zip of a stub protoc under root/release; its sha256."""
        release = self.root / "release"
        release.mkdir()
        with zipfile.ZipFile(release / ASSET, "w") as archive:
            info = zipfile.ZipInfo("bin/protoc")
            info.external_attr = 0o755 << 16
            archive.writestr(info, RELEASE_PROTOC)
            archive.writestr("include/google/protobuf/empty.proto", 'syntax = "proto3";\n')
        return hashlib.sha256((release / ASSET).read_bytes()).hexdigest()

    def run(self, **extra: str) -> subprocess.CompletedProcess[str]:
        env = {
            "PATH": str(self.bin),
            "HOME": str(self.root),
            "CALLS": str(self.calls),
            "EXITS": str(self.exits),
            "OPENSSL": "0" if self.openssl else "1",
            "CI_APT_ROOT": str(self.apt_root),
            "CI_APT_INSTALL_RETRY_DELAY": "0",
            "CI_APT_UPDATE_RETRY_DELAY": "0",
            "CI_PROTOC_PREFIX": str(self.prefix),
            "CI_PROTOC_URL_BASE": (self.root / "release").as_uri(),
            "GITHUB_PATH": str(self.github_path),
        }
        env.update(extra)
        return subprocess.run(
            ["bash", str(SCRIPT)], env=env, capture_output=True, text=True, check=False
        )

    def apt_calls(self) -> list[str]:
        return self.calls.read_text().splitlines()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path


def test_a_complete_host_runs_no_apt(root: Path) -> None:
    bed = Bed(root, ["cc", "c++", "make", "pkg-config", "protoc"], openssl=True, apt_exits=[])
    result = bed.run()
    assert result.returncode == 0, result.stderr
    assert bed.apt_calls() == []
    assert "apt is not needed" in result.stdout


def test_a_host_without_a_cpp_compiler_gets_build_essential(root: Path) -> None:
    # A C compiler alone does not build the workspace: the tokenizers crate
    # compiles C++ sources, so the probe asks for c++ as well.
    bed = Bed(root, ["cc", "make", "pkg-config", "protoc"], openssl=True, apt_exits=[0] * 8)
    result = bed.run()
    assert result.returncode == 0, result.stderr
    installs = [call for call in bed.apt_calls() if call.startswith("install")]
    assert installs == ["install -y build-essential"], bed.apt_calls()
    assert "missing: build-essential" in result.stdout


def test_a_missing_protoc_comes_from_the_archive(root: Path) -> None:
    bed = Bed(root, ["cc", "c++", "make", "pkg-config"], openssl=True, apt_exits=[0] * 8)
    result = bed.run()
    assert result.returncode == 0, result.stderr
    installs = [call for call in bed.apt_calls() if call.startswith("install")]
    assert installs == ["install -y protobuf-compiler"], bed.apt_calls()
    assert not bed.prefix.exists(), "no release fallback while the archive answers"


def test_a_missing_protoc_comes_from_its_release_when_the_archive_is_down(root: Path) -> None:
    bed = Bed(root, ["cc", "c++", "make", "pkg-config"], openssl=True, apt_exits=[100] * 12)
    sha = bed.release()
    result = bed.run(CI_PROTOC_SHA256=sha)
    assert result.returncode == 0, result.stderr
    # The index refresh failed, so nothing was asked of the archive beyond it.
    assert bed.apt_calls() and bed.apt_calls()[0] == "update", bed.apt_calls()
    assert not any(call.startswith("install") for call in bed.apt_calls())
    protoc = bed.prefix / "bin" / "protoc"
    assert protoc.exists() and os.access(protoc, os.X_OK)
    assert (bed.prefix / "include" / "google" / "protobuf" / "empty.proto").exists()
    assert str(bed.prefix / "bin") in bed.github_path.read_text().splitlines()
    assert "from its release" in result.stdout


def test_a_release_with_the_wrong_hash_installs_nothing(root: Path) -> None:
    bed = Bed(root, ["cc", "c++", "make", "pkg-config"], openssl=True, apt_exits=[100] * 12)
    bed.release()
    result = bed.run(CI_PROTOC_SHA256="0" * 64)
    assert result.returncode != 0
    assert not (bed.prefix / "bin" / "protoc").exists()
    assert "does not match its pinned sha256" in result.stderr


def test_more_than_protoc_missing_and_the_archive_down_fails_the_step(root: Path) -> None:
    bed = Bed(root, ["cc", "c++", "make"], openssl=False, apt_exits=[100] * 12)
    bed.release()
    result = bed.run()
    assert result.returncode == 1
    assert not bed.prefix.exists(), "the release carries protoc alone"
    assert "could not install" in result.stderr
    assert bed.apt_calls() and bed.apt_calls()[0] == "update", bed.apt_calls()
    assert "missing: pkg-config libssl-dev protobuf-compiler" in result.stdout
