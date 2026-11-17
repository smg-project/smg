"""The rustup installs of the CI jobs go through scripts/ci_retry.sh: the toolchain install of
scripts/ci_install_rust.sh and the wasm32-wasip2 target install of the WASM fixture build each survive a
failed download, a failure that stays is still the step's, and no workflow step installs a toolchain, a
target or a component by a bare rustup call or leaves the policy guest's target to the guest's own build script.

The calls of CI are bounded too: each attempt runs under a ceiling, the install's two curl calls under a
connect limit from a config file, and the worst case of one step fits inside its job's own limit.

Run with: pytest -q scripts/tests/test_ci_rustup_retry.py
"""

from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
INSTALL = REPO / "scripts" / "ci_install_rust.sh"
RETRY = REPO / "scripts" / "ci_retry.sh"
FIXTURES = REPO / "crates" / "wasm" / "tests" / "fixtures" / "build_fixtures.sh"
WORKFLOW = REPO / ".github" / "workflows" / "pr-test-rust.yml"

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

# The config file a curl of the install reads its connect limit from, recorded
# once a call.
LIMIT = 'cat "$CURL_HOME/.curlrc" >> "$LIMITS"\n'

# What the stub curl hands to `sh`: an installer that fails as EXITS says and
# otherwise leaves the environment file the script sources. The real one
# fetches by a curl of its own: the config file that curl would read is
# recorded.
INSTALLER = LIMIT + NTH_EXIT + 'mkdir -p "$HOME/.cargo/bin"\n: > "$HOME/.cargo/env"\n'

# A stub curl: the n-th fetch fails as FETCH_EXITS says, before a byte of the
# installer is written. It is the first curl of the install: the config file
# it runs under is recorded.
CURL = f"""\
#!/bin/bash
echo fetch >> "$FETCHES"
{LIMIT}code=$(sed -n "$(wc -l < "$FETCHES")p" "$FETCH_EXITS")
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

# The ceiling of an attempt, recorded instead of kept: the attempt runs at once.
TIMEOUT = '#!/bin/bash\necho "$1" >> "$CEILINGS"\nshift\nexec "$@"\n'


def run(
    script: Path, root: Path, stubs: dict[str, str], exits: list[int], **extra: str
) -> subprocess.CompletedProcess[str]:
    """The script on a PATH of the utilities and the stubs, and nothing else of the host."""
    bin_dir = root / "bin"
    bin_dir.mkdir()
    (root / "tmp").mkdir()
    for name in UTILITIES:
        real = shutil.which(name)
        if real:
            os.symlink(real, bin_dir / name)
    for name, text in {"sleep": SLEEP, "timeout": TIMEOUT, **stubs}.items():
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
        "CEILINGS": str(root / "ceilings"),
        "LIMITS": str(root / "limits"),
        "RUNNER_TEMP": str(root / "tmp"),
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
    assert lines(tmp_path / "ceilings") == ["180", "180"]
    # Both curl calls of both attempts, the fetch and the installer's own.
    assert lines(tmp_path / "limits") == ["connect-timeout = 10"] * 4


def test_a_failed_fetch_of_the_installer_is_retried_too(tmp_path: Path) -> None:
    # The installer comes through a pipe: the fetch's failure has to reach the
    # retry helper, or an empty installer passes for a finished install.
    result = install(tmp_path, exits=[], fetch_exits=(22,))
    assert result.returncode == 0, result.stderr
    assert len(lines(tmp_path / "fetches")) == 2
    assert len(lines(tmp_path / "calls")) == 1
    assert lines(tmp_path / "sleeps") == ["10"]


def test_a_toolchain_download_that_keeps_failing_ends_the_step(tmp_path: Path) -> None:
    result = install(tmp_path, exits=[1] * 5)
    assert result.returncode == 1
    assert len(lines(tmp_path / "calls")) == 5
    assert lines(tmp_path / "sleeps") == ["10", "20", "40", "80"]
    assert lines(tmp_path / "github_path") == []


def test_the_wasm_target_install_is_retried_after_a_failed_download(tmp_path: Path) -> None:
    result = build_fixtures(tmp_path, exits=[1])
    assert result.returncode == 0, result.stderr
    assert lines(tmp_path / "calls") == ["target add wasm32-wasip2"] * 2
    assert lines(tmp_path / "sleeps") == ["10"]
    assert lines(tmp_path / "builds")
    # Run by hand too: the script asks no ceiling of an attempt.
    assert lines(tmp_path / "ceilings") == []


def test_a_wasm_target_download_that_keeps_failing_ends_the_step(tmp_path: Path) -> None:
    result = build_fixtures(tmp_path, exits=[1] * 5)
    assert result.returncode == 1
    assert len(lines(tmp_path / "calls")) == 5
    assert lines(tmp_path / "sleeps") == ["10", "20", "40", "80"]
    assert lines(tmp_path / "builds") == []


# A workflow step has no bed: its text is the contract. The calls that install
# a toolchain, a target or a component download from the release host.
BARE_INSTALL = re.compile(r"^\s*rustup (toolchain install|target add|component add) ")


def test_no_workflow_installs_a_toolchain_or_a_target_by_a_bare_rustup_call() -> None:
    bare = [
        f"{path.relative_to(REPO)}:{number}: {line.strip()}"
        for path in sorted((REPO / ".github").rglob("*.yml"))
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if BARE_INSTALL.match(line)
    ]
    assert bare == []


def test_the_policy_plugin_step_adds_the_guest_target_before_the_script_that_would() -> None:
    # The guest's build.sh is the copy a user starts from: it adds a missing
    # target by one attempt. The step adds it first, behind the helper.
    step = WORKFLOW.read_text().split("- name: Build the example policy plugin", 1)[1]
    step = step.split("\n      - name:", 1)[0]
    install = "bash scripts/ci_retry.sh 5 10 timeout 180 rustup target add wasm32-unknown-unknown"
    assert install in step
    assert step.index(install) < step.index("bash examples/wasm/wasm-guest-policy/build.sh")


# A retried rustup call of a workflow: its attempts, its first wait, the ceiling of an attempt.
RETRIED = re.compile(r'ci_retry\.sh"? (\d+) (\d+) (?:timeout (\d+) )?rustup ')


def test_every_retried_rustup_call_of_a_workflow_has_five_attempts_under_a_ceiling() -> None:
    calls = [
        match.groups()
        for path in sorted((REPO / ".github").rglob("*.yml"))
        for match in RETRIED.finditer(path.read_text())
    ]
    assert calls and set(calls) == {("5", "10", "180")}, calls


def test_an_attempt_that_never_answers_is_ended_at_its_ceiling(tmp_path: Path) -> None:
    # The form the calls take, with a ceiling of one second and the real timeout: a fetch that
    # never answers ends each attempt there, and the step ends with timeout's own status.
    if not shutil.which("timeout"):
        pytest.skip("timeout (coreutils) is not installed")
    hang = tmp_path / "hang"
    hang.write_text(f'#!/bin/bash\necho call >> "{tmp_path}/hung"\nexec sleep 600\n')
    hang.chmod(0o755)
    result = subprocess.run(
        ["bash", str(RETRY), "2", "0", "timeout", "1", str(hang)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 124
    assert len(lines(tmp_path / "hung")) == 2


def test_a_connect_that_never_answers_ends_at_the_config_files_limit(tmp_path: Path) -> None:
    # The form the install takes, with the real curl and a limit of one second: a listener
    # that never speaks holds the handshake, and curl ends there with its timeout status.
    curl = shutil.which("curl")
    if not curl:
        pytest.skip("curl is not installed")
    (tmp_path / ".curlrc").write_text("connect-timeout = 1\n")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        started = time.monotonic()
        result = subprocess.run(
            [curl, "-sS", f"https://127.0.0.1:{listener.getsockname()[1]}/"],
            env={"CURL_HOME": str(tmp_path), "PATH": os.environ["PATH"]},
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        waited = time.monotonic() - started
    assert result.returncode == 28, result.stderr
    assert waited < 30


def test_the_worst_case_of_one_step_fits_inside_its_jobs_limit() -> None:
    # Every attempt of every install of one step ended at its ceiling: the ceilings and the
    # waits between them, with five minutes of the job's own limit left for the step's work.
    calls = {
        match.groups()
        for path in sorted((REPO / ".github").rglob("*.yml"))
        for match in RETRIED.finditer(path.read_text())
    }
    assert len(calls) == 1 and None not in next(iter(calls)), calls
    attempts, first_wait, ceiling = (int(number) for number in next(iter(calls)))
    assert f'ci_retry.sh" {attempts} {first_wait} timeout {ceiling} bash ' in INSTALL.read_text()
    one_install = attempts * ceiling + first_wait * (2 ** (attempts - 1) - 1)
    held, over = 0, []
    for path in sorted((REPO / ".github" / "workflows").glob("*.yml")):
        for name, job in (yaml.safe_load(path.read_text()).get("jobs") or {}).items():
            limit = job.get("timeout-minutes")
            for step in job.get("steps") or []:
                run_text, uses = str(step.get("run", "")), str(step.get("uses", ""))
                installs = len(RETRIED.findall(run_text))
                installs += ("ci_install_rust.sh" in run_text) + ("actions/setup-rust" in uses)
                held += installs
                if isinstance(limit, int) and installs * one_install + 300 > limit * 60:
                    over.append(f"{path.name}: {name}: {installs * one_install} s of {limit * 60}")
    assert held and over == []
