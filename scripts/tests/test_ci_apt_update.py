"""scripts/ci_apt_update.sh: the index refresh of a CI install step survives a third-party source.

Run with: pytest -q scripts/tests/test_ci_apt_update.py
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "ci_apt_update.sh"

UBUNTU_SOURCES = """\
Types: deb
URIs: http://eu-frankfurt-1.clouds.archive.ubuntu.com/ubuntu/
Suites: noble noble-updates noble-backports
Components: main universe restricted multiverse
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg

Types: deb
URIs: http://eu-frankfurt-1.clouds.archive.ubuntu.com/ubuntu/
Suites: noble-security
Components: main universe restricted multiverse
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
"""

PPA_SOURCES = """\
Types: deb
URIs: https://ppa.launchpadcontent.net/git-core/ppa/ubuntu/
Suites: noble
Components: main
Signed-By: /etc/apt/keyrings/git-core.gpg
"""

# A stub apt-get: one line per call in CALLS, and when a call carries the
# fallback's source directory, a listing of what it holds in TREE (the script
# removes that directory on exit, so the stub reads it while it exists).
APT_GET = """\
#!/bin/bash
echo "$*" >> "$CALLS"
for arg in "$@"; do
    case "$arg" in
        -o) ;;
        Dir::Etc::sourceparts=*)
            dir="${arg#Dir::Etc::sourceparts=}"
            for f in "$dir"/*; do
                [ -e "$f" ] || continue
                echo "== $(basename "$f")" >> "$TREE"
                cat "$f" >> "$TREE"
            done ;;
        Dir::Etc::sourcelist=*)
            echo "== sources.list" >> "$TREE"
            cat "${arg#Dir::Etc::sourcelist=}" >> "$TREE" ;;
    esac
done
n=$(wc -l < "$CALLS")
code=$(sed -n "${n}p" "$EXITS")
[ -n "$code" ] || code=0
exit "$code"
"""


class Bed:
    def __init__(self, root: Path, exits: list[int]) -> None:
        self.root = root
        self.apt_root = root / "etc-apt"
        (self.apt_root / "sources.list.d").mkdir(parents=True)
        tools = root / "tools"
        tools.mkdir()
        stub = tools / "apt-get"
        stub.write_text(APT_GET)
        stub.chmod(0o755)
        self.calls = root / "calls"
        self.tree = root / "tree"
        exits_file = root / "exits"
        exits_file.write_text("".join(f"{code}\n" for code in exits))
        self.env = {
            **os.environ,
            "PATH": f"{tools}:{os.environ['PATH']}",
            "CI_APT_ROOT": str(self.apt_root),
            "CALLS": str(self.calls),
            "TREE": str(self.tree),
            "EXITS": str(exits_file),
            "TMPDIR": str(root),
            # The retry helper's first delay: no waiting in a unit test.
            "CI_APT_UPDATE_RETRY_DELAY": "0",
        }

    def source(self, name: str, text: str) -> None:
        path = self.apt_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(SCRIPT), *args], env=self.env, capture_output=True, text=True, check=False
        )

    def call_lines(self) -> list[str]:
        return self.calls.read_text().splitlines() if self.calls.exists() else []

    def staged(self) -> str:
        return self.tree.read_text() if self.tree.exists() else ""


def test_a_clean_update_is_one_call(tmp_path: Path) -> None:
    bed = Bed(tmp_path, exits=[0])
    bed.source("sources.list.d/ubuntu.sources", UBUNTU_SOURCES)
    bed.source("sources.list.d/git-core.sources", PPA_SOURCES)
    result = bed.run()
    assert result.returncode == 0, result.stderr
    assert bed.call_lines() == ["update"]
    assert "refreshing the Ubuntu archive" not in result.stderr


def test_a_failed_update_is_retried_on_the_ubuntu_sources_alone(tmp_path: Path) -> None:
    bed = Bed(tmp_path, exits=[100, 0])
    bed.source("sources.list.d/ubuntu.sources", UBUNTU_SOURCES)
    bed.source("sources.list.d/git-core.sources", PPA_SOURCES)
    bed.source(
        "sources.list.d/docker.list",
        "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] "
        "https://download.docker.com/linux/ubuntu noble stable\n",
    )
    result = bed.run()
    assert result.returncode == 0, result.stderr
    calls = bed.call_lines()
    assert len(calls) == 2 and calls[0] == "update"
    assert calls[1].startswith("update -o APT::Get::List-Cleanup=false -o Dir::Etc::sourcelist=")
    assert "-o Dir::Etc::sourceparts=" in calls[1]
    staged = bed.staged()
    assert "== ubuntu.sources" in staged
    assert "eu-frankfurt-1.clouds.archive.ubuntu.com" in staged
    assert "launchpadcontent" not in staged
    assert "docker" not in staged
    assert "1 Ubuntu source file(s) kept" in result.stderr


def test_the_one_line_form_keeps_the_archive_lines_only(tmp_path: Path) -> None:
    bed = Bed(tmp_path, exits=[100, 0])
    bed.source(
        "sources.list",
        "deb http://archive.ubuntu.com/ubuntu jammy main universe\n"
        "deb-src http://archive.ubuntu.com/ubuntu jammy main\n"
        "deb http://security.ubuntu.com/ubuntu jammy-security main\n"
        "deb https://packages.microsoft.com/repos/azure-cli/ jammy main\n",
    )
    bed.source(
        "sources.list.d/mixed.list",
        "deb [signed-by=/k.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /\n"
        "deb http://ports.ubuntu.com/ubuntu-ports jammy main\n",
    )
    bed.source(
        "sources.list.d/ppa.list",
        "deb https://ppa.launchpadcontent.net/git-core/ppa/ubuntu jammy main\n",
    )
    result = bed.run()
    assert result.returncode == 0, result.stderr
    staged = bed.staged()
    assert staged.count("== ") == 2, staged
    assert "== sources.list" in staged and "== mixed.list" in staged
    assert "== ppa.list" not in staged
    assert "archive.ubuntu.com" in staged and "security.ubuntu.com" in staged
    assert "ports.ubuntu.com" in staged
    assert "microsoft" not in staged and "nvidia" not in staged and "launchpadcontent" not in staged


def test_a_mixed_deb822_file_keeps_its_archive_stanzas_alone(tmp_path: Path) -> None:
    bed = Bed(tmp_path, exits=[100, 0])
    bed.source("sources.list.d/mixed.sources", UBUNTU_SOURCES + "\n" + PPA_SOURCES)
    bed.source("sources.list.d/git-core.sources", PPA_SOURCES)
    result = bed.run()
    assert result.returncode == 0, result.stderr
    staged = bed.staged()
    assert "== mixed.sources" in staged and "== git-core.sources" not in staged, staged
    assert staged.count("Types: deb") == 2, staged
    assert staged.count("eu-frankfurt-1.clouds.archive.ubuntu.com") == 2, staged
    assert "launchpadcontent" not in staged
    assert "1 Ubuntu source file(s) kept" in result.stderr


def test_a_folded_uris_field_is_read_whole(tmp_path: Path) -> None:
    bed = Bed(tmp_path, exits=[100, 0])
    bed.source(
        "sources.list.d/folded-vendor.sources",
        "Types: deb\nURIs: http://archive.ubuntu.com/ubuntu/\n http://vendor.example.com/ubuntu/\n"
        "Suites: noble\nComponents: main\n",
    )
    bed.source(
        "sources.list.d/folded-archive.sources",
        "Types: deb\nURIs:\n http://archive.ubuntu.com/ubuntu/\n\thttp://security.ubuntu.com/ubuntu/\n"
        "Suites: noble\nComponents: main\n",
    )
    result = bed.run()
    assert result.returncode == 0, result.stderr
    staged = bed.staged()
    assert "== folded-archive.sources" in staged and "security.ubuntu.com" in staged, staged
    assert "== folded-vendor.sources" not in staged and "vendor.example.com" not in staged, staged
    assert "1 Ubuntu source file(s) kept" in result.stderr


def test_a_host_that_only_resembles_the_archive_is_not_the_archive(tmp_path: Path) -> None:
    bed = Bed(tmp_path, exits=[100, 0])
    bed.source(
        "sources.list",
        "deb http://notubuntu.com/ubuntu noble main\n"
        "deb http://ubuntu.com.example.net/ubuntu noble main\n"
        "deb http://archive.ubuntu.com/ubuntu noble main\n",
    )
    result = bed.run()
    assert result.returncode == 0, result.stderr
    staged = bed.staged()
    assert staged.count("deb ") == 1 and "archive.ubuntu.com" in staged, staged


def test_a_failure_of_the_retried_fallback_is_the_steps(tmp_path: Path) -> None:
    bed = Bed(tmp_path, exits=[100, 100, 100, 100])
    bed.source("sources.list.d/ubuntu.sources", UBUNTU_SOURCES)
    result = bed.run()
    assert result.returncode == 100
    # The update as configured, then the Ubuntu-only refresh three times.
    calls = bed.call_lines()
    assert len(calls) == 4 and calls[0] == "update"
    assert all(call.startswith("update -o APT::Get::List-Cleanup=false") for call in calls[1:])
    assert "failed after 3 attempts" in result.stderr


def test_a_blip_of_the_archive_during_the_fallback_is_retried(tmp_path: Path) -> None:
    bed = Bed(tmp_path, exits=[100, 100, 0])
    bed.source("sources.list.d/ubuntu.sources", UBUNTU_SOURCES)
    result = bed.run()
    assert result.returncode == 0, result.stderr
    assert len(bed.call_lines()) == 3


MIRRORS = """\
# the GitHub-hosted images list the archive's hosts here
http://azure.archive.ubuntu.com/ubuntu/\tpriority:1
http://archive.ubuntu.com/ubuntu/\tpriority:2
http://security.ubuntu.com/ubuntu/\tpriority:3
"""


def mirror_sources(path: Path) -> str:
    return f"""\
Types: deb
URIs: mirror+file:{path}
Suites: noble noble-updates noble-backports noble-security
Components: main universe restricted multiverse
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
"""


def test_a_mirror_file_uri_is_the_archive_when_every_host_in_it_is(tmp_path: Path) -> None:
    bed = Bed(tmp_path, exits=[100, 0])
    mirrors = tmp_path / "apt-mirrors.txt"
    mirrors.write_text(MIRRORS)
    bed.source("sources.list.d/ubuntu.sources", mirror_sources(mirrors))
    bed.source("sources.list.d/git-core.sources", PPA_SOURCES)
    bed.source("sources.list.d/legacy.list", f"deb mirror+file:{mirrors} noble main\n")
    result = bed.run()
    assert result.returncode == 0, result.stderr
    staged = bed.staged()
    assert "== ubuntu.sources" in staged and "== legacy.list" in staged
    assert "launchpadcontent" not in staged
    assert "2 Ubuntu source file(s) kept" in result.stderr


def test_a_mirror_file_with_a_foreign_host_is_not_the_archive(tmp_path: Path) -> None:
    bed = Bed(tmp_path, exits=[100])
    mirrors = tmp_path / "apt-mirrors.txt"
    mirrors.write_text(
        "http://archive.ubuntu.com/ubuntu/\thttp://mirror.example.net/ubuntu/\n"
        "http://mirror.example.net/ubuntu/\n"
    )
    bed.source("sources.list.d/ubuntu.sources", mirror_sources(mirrors))
    result = bed.run()
    assert result.returncode == 100
    assert bed.call_lines() == ["update"]
    assert "nothing to fall back to" in result.stderr


def test_nothing_to_fall_back_to_fails_as_the_update_did(tmp_path: Path) -> None:
    bed = Bed(tmp_path, exits=[100])
    bed.source("sources.list.d/git-core.sources", PPA_SOURCES)
    result = bed.run()
    assert result.returncode == 100
    assert bed.call_lines() == ["update"]
    assert "nothing to fall back to" in result.stderr


@pytest.mark.parametrize("word", ["sudo", ""])
def test_the_sudo_word_prefixes_every_call(tmp_path: Path, word: str) -> None:
    bed = Bed(tmp_path, exits=[100, 0])
    bed.source("sources.list.d/ubuntu.sources", UBUNTU_SOURCES)
    # A stub sudo that runs its command and leaves a mark of its own.
    marks = tmp_path / "sudo-marks"
    sudo = tmp_path / "tools" / "sudo"
    sudo.write_text(f'#!/bin/bash\necho "$1" >> "{marks}"\nexec "$@"\n')
    sudo.chmod(0o755)
    result = bed.run(*([word] if word else []))
    assert result.returncode == 0, result.stderr
    assert len(bed.call_lines()) == 2
    marked = marks.read_text().splitlines() if marks.exists() else []
    assert marked == (["apt-get", "apt-get"] if word else [])
