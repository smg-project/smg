"""scripts/ci_apt_install.sh: an install from the Ubuntu archive is one retried unit (the mirror choice, the
index refresh, the install), and an attempt that follows a failed one reads another mirror.

Run with: pytest -q scripts/tests/test_ci_apt_install.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"

# The utilities the scripts call, linked into the bed's own PATH so the host's
# apt-get and sudo are never found by chance.
UTILITIES = [
    "awk",
    "basename",
    "bash",
    "cat",
    "dirname",
    "grep",
    "mkdir",
    "mktemp",
    "rm",
    "sed",
    "tr",
    "uname",
]

# The mirror in use is the word in MIRROR: "bad" answers its probe and serves
# 404, as one did for hours; a re-probe moves to "good" unless ONE_MIRROR says
# the host has no other.
MIRROR = """\
#!/bin/bash
echo "$*" >> "$MIRROR_CALLS"
if [ "${1:-}" = "--reprobe" ] && [ -z "${ONE_MIRROR:-}" ]; then echo good > "$MIRROR"; fi
"""

# The index refresh, with the line the red step read.
UPDATE = """\
#!/bin/bash
echo "$*" >> "$UPDATE_CALLS"
if [ "$(cat "$MIRROR")" = bad ] && [ "${BAD_FAILS:-}" = update ]; then
    echo "E: The repository 'http://mirror.example/ubuntu noble Release' no longer has a Release file." >&2
    exit 100
fi
"""

# apt-get on the bad mirror. BAD_FAILS=install: the line the red step read, the
# index names a file the mirror's pool lacks. BAD_FAILS=outage: no call passes.
APT_GET = """\
#!/bin/bash
echo "$*" >> "$INSTALL_CALLS"
[ "$(cat "$MIRROR")" = bad ] || exit 0
case "${BAD_FAILS:-}" in
    install)
        echo "E: Failed to fetch http://mirror.example/ubuntu/pool/main/p/pkg/pkg_1_amd64.deb  404  Not Found" >&2
        exit 100 ;;
    outage)
        echo "E: Could not connect to mirror.example:80, connection timed out" >&2
        exit 100 ;;
esac
"""

# One source of the archive, for the cases that run the real refresh script.
ARCHIVE_SOURCE = """\
Types: deb
URIs: http://mirror.example.ubuntu.com/ubuntu/
Suites: noble
Components: main
"""

SUDO = '#!/bin/bash\necho "$*" >> "$SUDO_CALLS"\nexec "$@"\n'

# The retry helper's pauses, recorded instead of slept.
SLEEP = '#!/bin/bash\necho "$1" >> "$SLEEPS"\n'


class Bed:
    """The helper beside the real retry helper and stand-ins of its two siblings, on a PATH of its own."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.scripts = root / "scripts"
        self.scripts.mkdir()
        shutil.copy(SCRIPTS / "ci_apt_install.sh", self.scripts)
        shutil.copy(SCRIPTS / "ci_retry.sh", self.scripts)
        (self.scripts / "ci_apt_mirror.sh").write_text(MIRROR)
        (self.scripts / "ci_apt_update.sh").write_text(UPDATE)
        self.bin = root / "bin"
        self.bin.mkdir()
        for name in UTILITIES:
            real = shutil.which(name)
            if real:
                os.symlink(real, self.bin / name)
        for name, text in {"apt-get": APT_GET, "sudo": SUDO, "sleep": SLEEP}.items():
            path = self.bin / name
            path.write_text(text)
            path.chmod(0o755)
        (root / "mirror").write_text("bad\n")

    def with_the_real_refresh(self) -> None:
        """The real refresh script in place of its stand-in, over one archive source."""
        shutil.copy(SCRIPTS / "ci_apt_update.sh", self.scripts)
        sources = self.root / "etc-apt" / "sources.list.d"
        sources.mkdir(parents=True)
        (sources / "ubuntu.sources").write_text(ARCHIVE_SOURCE)

    def run(self, *command: str, **extra: str) -> subprocess.CompletedProcess[str]:
        env = {
            "PATH": str(self.bin),
            "TMPDIR": str(self.root),
            "CI_APT_ROOT": str(self.root / "etc-apt"),
            "MIRROR": str(self.root / "mirror"),
            "MIRROR_CALLS": str(self.root / "mirror_calls"),
            "UPDATE_CALLS": str(self.root / "update_calls"),
            "INSTALL_CALLS": str(self.root / "install_calls"),
            "SUDO_CALLS": str(self.root / "sudo_calls"),
            "SLEEPS": str(self.root / "sleeps"),
            **extra,
        }
        return subprocess.run(
            list(command), cwd=self.root, env=env, capture_output=True, text=True, check=False
        )

    def install(self, *arguments: str, **extra: str) -> subprocess.CompletedProcess[str]:
        return self.run("bash", "scripts/ci_apt_install.sh", *arguments, **extra)

    def lines(self, name: str) -> list[str]:
        path = self.root / name
        return path.read_text().splitlines() if path.exists() else []


def test_a_good_mirror_takes_one_attempt_and_no_reprobe(tmp_path: Path) -> None:
    bed = Bed(tmp_path)
    result = bed.install("-y", "--no-install-recommends", "pkg")
    assert result.returncode == 0, result.stderr
    assert bed.lines("mirror_calls") == [""]
    assert bed.lines("update_calls") == [""]
    assert bed.lines("install_calls") == ["install -y --no-install-recommends pkg"]
    assert bed.lines("sleeps") == []


def test_a_refresh_that_fails_on_one_mirror_passes_on_another(tmp_path: Path) -> None:
    bed = Bed(tmp_path)
    result = bed.install("-y", "pkg", BAD_FAILS="update")
    assert result.returncode == 0, result.stderr
    assert "no longer has a Release file" in result.stderr
    assert bed.lines("mirror_calls") == ["", "--reprobe"]
    assert len(bed.lines("update_calls")) == 2
    assert bed.lines("install_calls") == ["install -y pkg"]
    assert bed.lines("sleeps") == ["10"]


def test_an_install_that_reads_404_on_one_mirror_passes_on_another(tmp_path: Path) -> None:
    bed = Bed(tmp_path)
    result = bed.install("-y", "pkg", BAD_FAILS="install")
    assert result.returncode == 0, result.stderr
    assert "404  Not Found" in result.stderr
    assert bed.lines("mirror_calls") == ["", "--reprobe"]
    # The second attempt read fresh lists: the refresh ran again before it.
    assert len(bed.lines("update_calls")) == 2
    assert bed.lines("install_calls") == ["install -y pkg"] * 2
    assert bed.lines("sleeps") == ["10"]


def test_a_failure_on_every_mirror_is_the_steps(tmp_path: Path) -> None:
    bed = Bed(tmp_path)
    result = bed.install("-y", "pkg", BAD_FAILS="install", ONE_MIRROR="1")
    assert result.returncode == 100
    assert bed.lines("mirror_calls") == ["", "--reprobe", "--reprobe"]
    assert bed.lines("install_calls") == ["install -y pkg"] * 3
    assert bed.lines("sleeps") == ["10", "20"]


def test_a_retry_of_the_install_alone_reads_the_same_mirror(tmp_path: Path) -> None:
    # The form the helper replaced: the mirror chosen once, the refresh once, and the install alone
    # behind the retry helper. Every attempt read the mirror the first one failed on.
    bed = Bed(tmp_path)
    old = (
        "bash scripts/ci_apt_mirror.sh && bash scripts/ci_apt_update.sh sudo"
        " && bash scripts/ci_retry.sh 3 10 sudo apt-get install -y pkg"
    )
    result = bed.run("bash", "-c", old, BAD_FAILS="install")
    assert result.returncode == 100
    assert bed.lines("mirror_calls") == [""]
    assert bed.lines("install_calls") == ["install -y pkg"] * 3
    assert result.stderr.count("404  Not Found") == 3


def test_an_outage_costs_one_fallback_of_the_refresh_per_attempt(tmp_path: Path) -> None:
    # The real refresh script inside the unit. Alone, its fallback is retried three times. Inside
    # the unit that would multiply with the unit's retry, on the mirror that just failed.
    bed = Bed(tmp_path)
    bed.with_the_real_refresh()
    result = bed.install("-y", "pkg", BAD_FAILS="outage", ONE_MIRROR="1")
    assert result.returncode == 100
    calls = bed.lines("install_calls")
    # Three attempts, each the update as configured and the archive's sources alone, once.
    assert len(calls) == 6 and all(call.startswith("update") for call in calls)
    assert bed.lines("sleeps") == ["10", "20"]


def test_a_failed_refresh_moves_to_another_mirror_after_one_fallback(tmp_path: Path) -> None:
    bed = Bed(tmp_path)
    bed.with_the_real_refresh()
    result = bed.install("-y", "pkg", BAD_FAILS="outage")
    assert result.returncode == 0, result.stderr
    assert bed.lines("mirror_calls") == ["", "--reprobe"]
    calls = [call.split()[0] for call in bed.lines("install_calls")]
    assert calls == ["update", "update", "update", "install"]
    assert bed.lines("sleeps") == ["10"]


def test_the_sudo_word_prefixes_the_refresh_and_the_install(tmp_path: Path) -> None:
    bed = Bed(tmp_path)
    result = bed.install("sudo", "-y", "pkg")
    assert result.returncode == 0, result.stderr
    assert bed.lines("update_calls") == ["sudo"]
    assert bed.lines("sudo_calls") == ["apt-get install -y pkg"]
    assert bed.lines("install_calls") == ["install -y pkg"]


def test_a_call_with_nothing_to_install_is_refused(tmp_path: Path) -> None:
    bed = Bed(tmp_path)
    for arguments in ((), ("sudo",)):
        result = bed.install(*arguments)
        assert result.returncode == 2
        assert "usage" in result.stderr
    assert bed.lines("install_calls") == []


OPENCV_PACKAGES = "clang libclang-dev cmake ninja-build pkg-config libopencv-dev"
OPENCV_INSTALL = f"install -y --no-install-recommends {OPENCV_PACKAGES}"


def test_the_opencv_script_installs_through_the_helper_in_ci(tmp_path: Path) -> None:
    # Its last package pulls hundreds of files from the pool, so a mirror that lacks one fails it.
    bed = Bed(tmp_path)
    shutil.copy(SCRIPTS / "install_opencv.sh", bed.scripts)
    result = bed.run("bash", "scripts/install_opencv.sh", CI="true", BAD_FAILS="install")
    assert result.returncode == 0, result.stderr
    assert bed.lines("mirror_calls") == ["", "--reprobe"]
    assert bed.lines("install_calls") == [OPENCV_INSTALL] * 2


def test_the_opencv_script_outside_ci_installs_as_before(tmp_path: Path) -> None:
    bed = Bed(tmp_path)
    shutil.copy(SCRIPTS / "install_opencv.sh", bed.scripts)
    result = bed.run("bash", "scripts/install_opencv.sh", AUTO_INSTALL="1")
    assert result.returncode == 0, result.stderr
    assert bed.lines("mirror_calls") == []
    assert bed.lines("install_calls") == ["update", OPENCV_INSTALL]


def test_the_libaio_package_follows_the_release(tmp_path: Path) -> None:
    # The service dependencies install it in one unit with their tools, under the name the release
    # has: a fallback from one name to the other would retry a name that does not exist.
    release = tmp_path / "os-release"
    script = (
        "CI_AGENTIC_SVC_DEPS_LIB_ONLY=1 source scripts/ci_agentic_svc_deps.sh && ci_libaio_package"
    )
    for text, package in (
        ('VERSION_ID="24.04"\n', "libaio1t64"),
        ('VERSION_ID="22.04"\n', "libaio1"),
        ("", "libaio1"),
    ):
        release.write_text(text)
        result = subprocess.run(
            ["bash", "-c", script],
            cwd=REPO,
            env={**os.environ, "CI_OS_RELEASE": str(release)},
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.stdout.split() == [package], result.stderr


# Where an install stays outside the helper, and why.
#
# The workflows whose installs run in a job of another kind: the review job and
# the release containers.
OTHER_JOBS = {
    "codex-code-review.yml",
    "nightly-artifacts.yml",
    "release-pypi.yml",
    "release-pypi-dev.yml",
}
# The lines that stay as they are, each for the reason above it.
KEPT = {
    # The install takes the vendor repository's packages in the same transaction, and the script
    # refreshes both hosts' lists, with a mirror move of its own, right before it.
    "scripts/ci_install_trtllm.sh": {
        "$RETRY 3 10 sudo apt-get install -y libopenmpi-dev libnvinfer10 cuda-toolkit-13-0",
    },
    # The branch a developer runs by hand: CI's branch of the same function calls the helper.
    "scripts/install_opencv.sh": {
        'run $sudo apt-get install -y --no-install-recommends "${packages[@]}"',
    },
    # A developer's make target: no job runs it.
    "scripts/setup-sccache.sh": {
        'echo "  sudo apt-get install -y sccache (Debian/Ubuntu)"',
        "sudo apt-get update -y && sudo apt-get install -y sccache",
    },
}
# What an image must copy before a script it copies can call the helper: the
# helper and the three scripts it runs.
UNIT = {
    "scripts/ci_apt_install.sh",
    "scripts/ci_apt_mirror.sh",
    "scripts/ci_apt_update.sh",
    "scripts/ci_retry.sh",
}


def image_scripts() -> dict[str, set[str]]:
    """The scripts each image of docker/ copies, by its COPY lines."""
    copied: dict[str, set[str]] = {}
    for dockerfile in sorted((REPO / "docker").glob("*.Dockerfile")):
        lines = dockerfile.read_text().replace("\\\n", " ").splitlines()
        words = [word for line in lines if line.startswith("COPY ") for word in line.split()[1:-1]]
        scripts = {word for word in words if re.fullmatch(r"scripts/\w+\.sh", word)}
        if scripts:
            copied[dockerfile.name] = scripts
    return copied


def test_a_script_an_image_copies_calls_the_helper_only_if_the_image_copies_it() -> None:
    # An image copies a fixed list of scripts and runs them at its build. A copied script that
    # called the helper would end that build at a file the image does not have.
    missing = [
        f"{dockerfile} copies {name}, which calls the helper, without {sorted(UNIT - scripts)}"
        for dockerfile, scripts in sorted(image_scripts().items())
        for name in sorted(scripts)
        if "ci_apt_install.sh" in (REPO / name).read_text() and not UNIT <= scripts
    ]
    assert missing == []


def test_every_archive_install_of_ci_goes_through_the_helper() -> None:
    # A script an image copies keeps its installs until that image copies the helper.
    held = {name for scripts in image_scripts().values() if not UNIT <= scripts for name in scripts}
    files = [
        path for path in sorted((REPO / ".github").rglob("*.yml")) if path.name not in OTHER_JOBS
    ] + [path for path in sorted(SCRIPTS.glob("*.sh")) if path.name != "ci_apt_install.sh"]
    bare = []
    for path in files:
        name = str(path.relative_to(REPO))
        if name in held:
            continue
        bare += [
            f"{name}:{number}: {line.strip()}"
            for number, line in enumerate(path.read_text().splitlines(), 1)
            if "apt-get install" in line
            and not line.lstrip().startswith("#")
            and line.strip() not in KEPT.get(name, set())
        ]
    assert bare == []


def test_strict_mirror_failure_never_refreshes_or_installs(tmp_path: Path) -> None:
    bed = Bed(tmp_path)
    (bed.scripts / "ci_apt_mirror.sh").write_text(
        '#!/bin/bash\necho "$*" >> "$MIRROR_CALLS"\nexit 71\n'
    )
    result = bed.install("--strict-mirror", "sudo", "-y", "pkg")
    assert result.returncode == 71, result.stderr
    assert bed.lines("mirror_calls") == ["", "--reprobe", "--reprobe"]
    assert bed.lines("update_calls") == []
    assert bed.lines("install_calls") == []
    assert bed.lines("sleeps") == ["10", "20"]


def test_default_mirror_failure_still_uses_existing_sources(tmp_path: Path) -> None:
    bed = Bed(tmp_path)
    (bed.scripts / "ci_apt_mirror.sh").write_text(
        '#!/bin/bash\necho "$*" >> "$MIRROR_CALLS"\nexit 71\n'
    )
    result = bed.install("sudo", "-y", "pkg")
    assert result.returncode == 0, result.stderr
    assert "mirror probe failed" in result.stderr
    assert bed.lines("update_calls") == ["sudo"]
    assert bed.lines("install_calls") == ["install -y pkg"]


def test_strict_mirror_still_requires_packages(tmp_path: Path) -> None:
    bed = Bed(tmp_path)
    for arguments in (("--strict-mirror",), ("--strict-mirror", "sudo")):
        result = bed.install(*arguments)
        assert result.returncode == 2
    assert bed.lines("mirror_calls") == []
    assert bed.lines("install_calls") == []


def mirror_bed(root: Path) -> Bed:
    """The real mirror selector over private sources, with only curl replaced."""
    bed = Bed(root)
    shutil.copy(SCRIPTS / "ci_apt_mirror.sh", bed.scripts)
    for name in ("head", "tee", "cmp", "chmod", "mv"):
        real = shutil.which(name)
        assert real is not None, name
        os.symlink(real, bed.bin / name)
    (root / "etc-apt" / "apt.conf.d").mkdir(parents=True)
    (root / "etc-apt" / "sources.list.d").mkdir()
    (root / "os-release").write_text("ID=ubuntu\nVERSION_CODENAME=noble\n")
    curl = bed.bin / "curl"
    curl.write_text('#!/bin/bash\nprintf \'%s\\n\' "${!#}" >> "$MIRROR_PROBE_LOG"\n')
    curl.chmod(0o755)
    return bed


def run_mirror(bed: Bed, mirrors: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return bed.run(
        "bash",
        "scripts/ci_apt_mirror.sh",
        *arguments,
        CI_OS_RELEASE=str(bed.root / "os-release"),
        CI_APT_MIRRORS=mirrors,
        MIRROR_PROBE_LOG=str(bed.root / "mirror_probes"),
    )


@pytest.mark.parametrize("field", ["URIs:", "URIs: ", "uRiS:\t"])
def test_deb822_uri_values_preserve_field_spacing_and_unmanaged_urls(
    tmp_path: Path, field: str
) -> None:
    bed = mirror_bed(tmp_path)
    main = "http://archive.ubuntu.com/ubuntu/"
    security = "https://security.ubuntu.com/ubuntu/"
    chosen = "http://azure.archive.ubuntu.com/ubuntu"
    unrelated = "https://mirror.example/a.[x]+(repo)&extras/"
    ports = "http://ports.ubuntu.com/ubuntu-ports"
    source = tmp_path / "etc-apt" / "sources.list.d" / "ubuntu.sources"
    original = (
        f"Types: deb\n{field}{main}\t {unrelated}  {ports}\t{security} \n"
        "Suites: noble noble-security\nComponents: main\n"
    )
    source.write_text(original)
    result = run_mirror(bed, chosen)
    assert result.returncode == 0, result.stderr
    assert source.read_text() == original.replace(main, chosen + "/").replace(
        security, chosen + "/"
    ), result.stdout
    assert bed.lines("install_calls") == []
    assert bed.lines("update_calls") == []


@pytest.mark.parametrize(
    "current,alternate",
    [
        (
            "http://azure.archive.ubuntu.com/ubuntu/",
            "http://eu-frankfurt-1.clouds.archive.ubuntu.com/ubuntu/",
        ),
        ("https://mirror.example/a.[x]+(repo)/", "https://backup.example/repo&extras/"),
    ],
)
def test_trailing_slash_mirrors_rewrite_then_reprobe_to_another_origin(
    tmp_path: Path, current: str, alternate: str
) -> None:
    bed = mirror_bed(tmp_path)
    main = "http://archive.ubuntu.com/ubuntu/"
    security = "http://security.ubuntu.com/ubuntu/"
    legacy = tmp_path / "etc-apt" / "sources.list"
    deb822 = tmp_path / "etc-apt" / "sources.list.d" / "ubuntu.sources"
    unrelated = tmp_path / "etc-apt" / "sources.list.d" / "unrelated.list"
    legacy_original = f"deb {main} noble main\ndeb {security} noble-security main\n"
    deb822_original = f"Types: deb\nURIs: {main}\t{security}\nSuites: noble noble-security\n"
    untouched = (
        "deb http://ports.ubuntu.com/ubuntu-ports noble main\n"
        "deb https://mirror.example/aZxrepo/ noble main\n"
    )
    legacy.write_text(legacy_original)
    deb822.write_text(deb822_original)
    unrelated.write_text(untouched)
    mirrors = current + " " + alternate
    first = run_mirror(bed, mirrors)
    assert first.returncode == 0, first.stderr
    first_sources = (legacy.read_text(), deb822.read_text())
    first_marker = (tmp_path / "etc-apt" / "apt.conf.d" / "99-ci-mirror").read_text()
    (tmp_path / "mirror_probes").write_text("")
    second = run_mirror(bed, mirrors, "--reprobe")
    assert second.returncode == 0, second.stderr
    expected_first = tuple(
        text.replace(main, current).replace(security, current)
        for text in (legacy_original, deb822_original)
    )
    assert first_sources == expected_first, (first.stdout, second.stdout)
    assert (legacy.read_text(), deb822.read_text()) == tuple(
        text.replace(main, alternate).replace(security, alternate)
        for text in (legacy_original, deb822_original)
    ), second.stdout
    assert f"mirror={current}" in first_marker
    assert (
        f"mirror={alternate}" in (tmp_path / "etc-apt" / "apt.conf.d" / "99-ci-mirror").read_text()
    )
    assert unrelated.read_text() == untouched
    probes = bed.lines("mirror_probes")
    assert len(probes) == 2 and all(url.startswith(alternate) for url in probes)
    assert bed.lines("install_calls") == []
    assert bed.lines("update_calls") == []
