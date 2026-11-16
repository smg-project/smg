"""The rustup installs of the CI jobs go through scripts/ci_retry.sh: the toolchain install of
scripts/ci_install_rust.sh and the wasm32-wasip2 target install of the WASM fixture build each survive a
failed download and a burst of them, a failure that stays is still the step's, every install takes the same
attempts, and no workflow step installs a toolchain or a target by a bare rustup call or leaves the policy
guest's target to the guest's own build script.

Run with: pytest -q scripts/tests/test_ci_rustup_retry.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
INSTALL = REPO / "scripts" / "ci_install_rust.sh"
RETRY = REPO / "scripts" / "ci_retry.sh"
FIXTURES = REPO / "crates" / "wasm" / "tests" / "fixtures" / "build_fixtures.sh"
WORKFLOW = REPO / ".github" / "workflows" / "pr-test-rust.yml"

# The attempts of every rustup install and the pauses between them (the helper
# doubles its delay): 630 s in all. The release host fails in bursts of
# minutes, and three attempts 10 s and 20 s apart gave up inside one.
ATTEMPTS = 7
PAUSES = ["10", "20", "40", "80", "160", "320"]

# The utilities the scripts, the retry helper and the stubs call, linked into
# the bed's own PATH so the host's rustup, cargo and curl are never found by
# chance.
UTILITIES = [
    "bash",
    "sh",
    "basename",
    "cat",
    "cp",
    "cut",
    "dirname",
    "du",
    "echo",
    "grep",
    "mkdir",
    "sed",
    "tr",
    "wc",
    "xargs",
]

STUB = "#!/bin/bash\nexit 0\n"

# One line per call in CALLS; the exit code of the n-th call from EXITS (0
# past its end).
NTH_EXIT = """\
echo "$*" >> "$CALLS"
code=$(sed -n "$(wc -l < "$CALLS")p" "$EXITS")
[ -z "$code" ] || [ "$code" = 0 ] || exit "$code"
"""

# What the stub curl hands to `sh`: an installer that fails as EXITS says and
# otherwise leaves the environment file the script sources.
INSTALLER = NTH_EXIT + 'mkdir -p "$HOME/.cargo/bin"\n: > "$HOME/.cargo/env"\n'

# A stub curl: the n-th fetch fails as FETCH_EXITS says, before a byte of the
# installer is written.
CURL = """\
#!/bin/bash
echo fetch >> "$FETCHES"
code=$(sed -n "$(wc -l < "$FETCHES")p" "$FETCH_EXITS")
[ -z "$code" ] || [ "$code" = 0 ] || exit "$code"
cat "$INSTALLER"
"""

# A stub rustup: the installed targets from INSTALLED, any other call by EXITS.
RUSTUP = f"""\
#!/bin/bash
if [ "$1 $2" = "target list" ]; then cat "$INSTALLED"; exit 0; fi
{NTH_EXIT}"""

# A stub cargo: the component a guest's build leaves, named after the guest.
CARGO = """\
#!/bin/bash
echo "$PWD" >> "$BUILDS"
mkdir -p target/wasm32-wasip2/release
: > "target/wasm32-wasip2/release/$(basename "$PWD" | tr - _).wasm"
"""

# The retry helper's pauses, recorded instead of slept.
SLEEP = '#!/bin/bash\necho "$1" >> "$SLEEPS"\n'


def run(
    script: Path, root: Path, stubs: dict[str, str], exits: list[int], **extra: str
) -> subprocess.CompletedProcess[str]:
    """The script on a PATH of the utilities and the stubs, and nothing else of the host."""
    bin_dir = root / "bin"
    bin_dir.mkdir()
    for name in UTILITIES:
        real = shutil.which(name)
        if real:
            os.symlink(real, bin_dir / name)
    for name, text in {"sleep": SLEEP, **stubs}.items():
        path = bin_dir / name
        path.write_text(text)
        path.chmod(0o755)
    (root / "exits").write_text("".join(f"{code}\n" for code in exits))
    env = {
        "PATH": str(bin_dir),
        "HOME": str(root / "home"),
        "CALLS": str(root / "calls"),
        "EXITS": str(root / "exits"),
        "SLEEPS": str(root / "sleeps"),
        **extra,
    }
    return subprocess.run(
        ["bash", str(script)], env=env, capture_output=True, text=True, check=False
    )


def install(
    root: Path, exits: list[int], fetch_exits: tuple[int, ...] = ()
) -> subprocess.CompletedProcess[str]:
    """ci_install_rust.sh beside the retry helper, its dependency step a no-op."""
    scripts = root / "scripts"
    scripts.mkdir()
    shutil.copy(INSTALL, scripts)
    shutil.copy(RETRY, scripts)
    (scripts / "ci_install_build_deps.sh").write_text(STUB)
    (root / "installer").write_text(INSTALLER)
    (root / "fetch_exits").write_text("".join(f"{code}\n" for code in fetch_exits))
    return run(
        scripts / INSTALL.name,
        root,
        {"curl": CURL, "rustc": STUB, "cargo": STUB, "protoc": STUB},
        exits,
        INSTALLER=str(root / "installer"),
        FETCHES=str(root / "fetches"),
        FETCH_EXITS=str(root / "fetch_exits"),
        GITHUB_PATH=str(root / "github_path"),
    )


def build_fixtures(root: Path, exits: list[int]) -> subprocess.CompletedProcess[str]:
    """The fixture build in a copy of the tree's layout: the guests' directories, the retry
    helper, a toolchain without the wasm target."""
    repo = root / "repo"
    script = repo / FIXTURES.relative_to(REPO)
    script.parent.mkdir(parents=True)
    shutil.copy(FIXTURES, script)
    (repo / "scripts").mkdir()
    shutil.copy(RETRY, repo / "scripts")
    for guest in (REPO / "examples" / "wasm").iterdir():
        if guest.is_dir():
            (repo / "examples" / "wasm" / guest.name).mkdir(parents=True)
    (root / "installed").touch()
    return run(
        script,
        root,
        {"rustup": RUSTUP, "cargo": CARGO},
        exits,
        INSTALLED=str(root / "installed"),
        BUILDS=str(root / "builds"),
    )


def lines(path: Path) -> list[str]:
    return path.read_text().splitlines() if path.exists() else []


def test_the_toolchain_install_is_retried_after_a_failed_download(tmp_path: Path) -> None:
    result = install(tmp_path, exits=[1])
    assert result.returncode == 0, result.stderr
    calls = lines(tmp_path / "calls")
    assert len(calls) == 2 and calls[0] == calls[1]
    assert calls[0].startswith("-y --default-toolchain ")
    assert lines(tmp_path / "sleeps") == ["10"]
    assert lines(tmp_path / "github_path") == [f"{tmp_path}/home/.cargo/bin"]


def test_a_failed_fetch_of_the_installer_is_retried_too(tmp_path: Path) -> None:
    # The installer comes through a pipe: the fetch's failure has to reach the
    # retry helper, or an empty installer passes for a finished install.
    result = install(tmp_path, exits=[], fetch_exits=(22,))
    assert result.returncode == 0, result.stderr
    assert len(lines(tmp_path / "fetches")) == 2
    assert len(lines(tmp_path / "calls")) == 1
    assert lines(tmp_path / "sleeps") == ["10"]


def test_the_toolchain_install_outlasts_a_burst_of_failed_downloads(tmp_path: Path) -> None:
    # Every attempt but the last fails: the install still ends well, after
    # every pause.
    result = install(tmp_path, exits=[1] * (ATTEMPTS - 1))
    assert result.returncode == 0, result.stderr
    assert len(lines(tmp_path / "calls")) == ATTEMPTS
    assert lines(tmp_path / "sleeps") == PAUSES
    assert lines(tmp_path / "github_path") == [f"{tmp_path}/home/.cargo/bin"]


def test_a_toolchain_download_that_keeps_failing_ends_the_step(tmp_path: Path) -> None:
    result = install(tmp_path, exits=[1] * ATTEMPTS)
    assert result.returncode == 1
    assert len(lines(tmp_path / "calls")) == ATTEMPTS
    assert lines(tmp_path / "sleeps") == PAUSES
    assert lines(tmp_path / "github_path") == []


def test_the_wasm_target_install_is_retried_after_a_failed_download(tmp_path: Path) -> None:
    result = build_fixtures(tmp_path, exits=[1])
    assert result.returncode == 0, result.stderr
    assert lines(tmp_path / "calls") == ["target add wasm32-wasip2"] * 2
    assert lines(tmp_path / "sleeps") == ["10"]
    assert lines(tmp_path / "builds")


def test_a_wasm_target_download_that_keeps_failing_ends_the_step(tmp_path: Path) -> None:
    result = build_fixtures(tmp_path, exits=[1] * ATTEMPTS)
    assert result.returncode == 1
    assert len(lines(tmp_path / "calls")) == ATTEMPTS
    assert lines(tmp_path / "sleeps") == PAUSES
    assert lines(tmp_path / "builds") == []


# A workflow step has no bed: its text is the contract. The calls that install
# a toolchain or a target download from the release host.
BARE_INSTALL = re.compile(r"^\s*rustup (toolchain install|target add|component add --toolchain) ")


def test_no_workflow_installs_a_toolchain_or_a_target_by_a_bare_rustup_call() -> None:
    bare = [
        f"{path.relative_to(REPO)}:{number}: {line.strip()}"
        for path in sorted((REPO / ".github").rglob("*.yml"))
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if BARE_INSTALL.match(line)
    ]
    assert bare == []


# Every install behind the helper: the script's installer pipe, and each
# rustup call of the fixture build and of the workflows.
RETRIED_PIPE = re.compile(r'ci_retry\.sh" (\d+) (\d+) bash -o pipefail -c')
RETRIED_RUSTUP = re.compile(r'ci_retry\.sh"? (\d+) (\d+) rustup ')


def test_every_rustup_install_takes_the_same_attempts() -> None:
    takes = RETRIED_PIPE.findall(INSTALL.read_text())
    for path in [FIXTURES, *sorted((REPO / ".github").rglob("*.yml"))]:
        takes += RETRIED_RUSTUP.findall(path.read_text())
    assert len(takes) >= 6
    assert set(takes) == {(str(ATTEMPTS), PAUSES[0])}


def test_the_policy_plugin_step_adds_the_guest_target_before_the_script_that_would() -> None:
    # The guest's build.sh is the copy a user starts from: it adds a missing
    # target by one attempt. The step adds it first, behind the helper.
    step = WORKFLOW.read_text().split("- name: Build the example policy plugin", 1)[1]
    step = step.split("\n      - name:", 1)[0]
    install = f"bash scripts/ci_retry.sh {ATTEMPTS} 10 rustup target add wasm32-unknown-unknown"
    assert install in step
    assert step.index(install) < step.index("bash examples/wasm/wasm-guest-policy/build.sh")
