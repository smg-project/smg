"""CPU contracts for the AMD Go caller; no backend/model/GPU execution."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


def _python_loader_environment():
    # Keep an explicitly configured interpreter's shared-library loader path.
    # Do not copy the caller's credentials or unrelated environment.
    return {name: os.environ[name] for name in ("LD_LIBRARY_PATH",) if name in os.environ}


@pytest.mark.parametrize(
    "change, expected",
    [
        ({}, True),
        ({"cancelled": True}, False),
        ({"SMG_RUN_AMD_LEGS": "false"}, False),
        ({"SMG_RUN_AMD_GO_LEGS": "false"}, False),
        ({"SMG_RUN_AMD_ON_PRS": "false"}, False),
        ({"fork": True}, False),
        ({"wheel": "failure"}, False),
        ({"detect": "failure"}, False),
        ({"skip": "true"}, False),
        ({"event": "push", "SMG_RUN_AMD_ON_PRS": "false", "detect": "skipped"}, True),
    ],
)
def test_go_amd_eligibility_keeps_optins_and_fork_filter(change, expected):
    # This is an eligibility expression, not an external runner/credential trust boundary.
    root = Path(__file__).resolve().parents[2]
    job = yaml.safe_load((root / ".github/workflows/pr-test-rust.yml").read_text())["jobs"][
        "e2e-go-bindings-amd"
    ]
    values = {
        "SMG_RUN_AMD_LEGS": "true",
        "SMG_RUN_AMD_GO_LEGS": "true",
        "SMG_RUN_AMD_ON_PRS": "true",
    }
    values.update({key: value for key, value in change.items() if key.startswith("SMG_")})
    ns = SimpleNamespace
    context = {
        "vars": ns(**values),
        "github": ns(
            event_name=change.get("event", "pull_request"),
            repository="owner/repo",
            event=ns(
                pull_request=ns(
                    head=ns(repo=ns(full_name="fork/repo" if change.get("fork") else "owner/repo"))
                )
            ),
        ),
        "needs": {
            "build-wheel": ns(result=change.get("wheel", "success")),
            "detect-changes": ns(
                result=change.get("detect", "success"),
                outputs=ns(**{"skip-go-bindings": change.get("skip", "false")}),
            ),
        },
        "always": lambda: True,
        "cancelled": lambda: change.get("cancelled", False),
    }
    expression = " ".join(job["if"].split()).replace("&&", " and ").replace("||", " or ")
    expression = re.sub(r"!(?!=)", " not ", expression)
    for name in ["build-wheel", "detect-changes"]:
        expression = expression.replace("needs." + name, f"needs[{name!r}]")
    expression = expression.replace(
        ".outputs.skip-go-bindings", ".outputs.__dict__['skip-go-bindings']"
    )
    assert eval(expression, {"__builtins__": {}}, context) is expected


def test_go_amd_original_suite_and_finish_are_not_advisory():
    root = Path(__file__).resolve().parents[2]
    jobs = yaml.safe_load((root / ".github/workflows/pr-test-rust.yml").read_text())["jobs"]
    job = jobs["e2e-go-bindings-amd"]
    step = next(s for s in job["steps"] if s.get("name") == "Run original Go OAI server assertions")
    assert step["timeout-minutes"] == 45
    assert "e2e_test/bindings_go -s -vv" in step["run"]
    assert "--reruns 2 --reruns-delay 5" in step["run"]
    assert "-k " not in step["run"]
    assert job["env"]["E2E_MIN_SELECTED"] == "23"
    assert job["env"]["E2E_AMD_REQUIRE_PASSING_CALL"] == "1"
    assert job["env"]["SPT_NOENV"] == "1"
    assert "e2e-go-bindings-amd" in jobs["finish"]["needs"]
    check = next(s["run"] for s in jobs["finish"]["steps"] if s.get("name") == "Check CI result")
    assert '"${{ needs.e2e-go-bindings-amd.result }}" =~ $bad' in check
    assert "failure|cancelled" in check


@pytest.mark.parametrize("returncode, marker", [(0, "go-amd-owned"), (23, "go-amd-owned"), (0, "")])
def test_go_amd_strict_cleanup_propagates_errors(tmp_path, returncode, marker):
    root = Path(__file__).resolve().parents[2]
    job = yaml.safe_load((root / ".github/workflows/pr-test-rust.yml").read_text())["jobs"][
        "e2e-go-bindings-amd"
    ]
    step = next(s for s in job["steps"] if s.get("name") == "Reap owned processes")
    assert step["if"] == "always()"
    assert not step.get("continue-on-error", False)
    binary = tmp_path / "bin"
    binary.mkdir()
    stub = binary / "python3"
    stub.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$ARGV_FILE"\nexit "$STUB_EXIT"\n')
    stub.chmod(0o755)
    log = tmp_path / "argv"
    result = subprocess.run(
        ["/usr/bin/bash", "-e", "-c", step["run"]],
        cwd=root,
        env={
            "PATH": str(binary),
            "E2E_JOB_MARKER": marker,
            "ARGV_FILE": str(log),
            "STUB_EXIT": str(returncode),
        },
        capture_output=True,
        text=True,
        timeout=5,
    )
    if not marker:
        assert result.returncode != 0
        assert not log.exists()
    else:
        assert result.returncode == returncode
        assert log.read_text().splitlines() == [
            "scripts/ci_reap_job_processes.py",
            "--strict",
            marker,
        ]


def test_go_amd_marker_uses_a_fresh_uuid_for_each_invocation(tmp_path):
    root = Path(__file__).resolve().parents[2]
    job = yaml.safe_load((root / ".github/workflows/pr-test-rust.yml").read_text())["jobs"][
        "e2e-go-bindings-amd"
    ]
    command = next(s["run"] for s in job["steps"] if s.get("name") == "Mark owned processes")
    output = tmp_path / "github-env"
    for _ in range(2):
        result = subprocess.run(
            ["/usr/bin/bash", "-e", "-c", command],
            env={
                "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin",
                "GITHUB_ENV": str(output),
                **_python_loader_environment(),
            },
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode == 0, result.stderr
    values = output.read_text().splitlines()
    assert len(values) == 2 and len(set(values)) == 2
    assert all(re.fullmatch(r"E2E_JOB_MARKER=go-amd-[0-9a-f]{32}", value) for value in values)


@pytest.mark.parametrize("drop_preservation", [False, True])
def test_go_amd_caller_marker_survives_real_process_title_only_with_preservation(
    tmp_path, drop_preservation
):
    # This is a bounded own child, not Go/FFI/model/worker execution. The runtime
    # is selected from PATH so matching caller-environment qualification can
    # bind it; ordinary tooling without this optional native package is unproved.
    root = Path(__file__).resolve().parents[2]
    job = yaml.safe_load((root / ".github/workflows/pr-test-rust.yml").read_text())["jobs"][
        "e2e-go-bindings-amd"
    ]
    command = next(s["run"] for s in job["steps"] if s.get("name") == "Mark owned processes")
    output = tmp_path / "github-env"
    environment = {
        "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin",
        "GITHUB_ENV": str(output),
        **_python_loader_environment(),
    }
    if "SPT_NOENV" in job["env"]:
        environment["SPT_NOENV"] = job["env"]["SPT_NOENV"]
    marked = subprocess.run(
        ["/usr/bin/bash", "-e", "-c", command],
        env=environment,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert marked.returncode == 0, marked.stderr
    (marker,) = output.read_text().splitlines()
    key, value = marker.split("=", 1)
    assert key == "E2E_JOB_MARKER" and re.fullmatch(r"go-amd-[0-9a-f]{32}", value)
    environment[key] = value
    if drop_preservation:
        environment.pop("SPT_NOENV", None)
    marker_python = shutil.which("python3")
    assert marker_python is not None
    probe = subprocess.run(
        [marker_python, "-I", "-B", "-c", "import setproctitle"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if probe.returncode:
        pytest.skip("setproctitle unavailable: actual caller-environment coverage not established")
    child = subprocess.run(
        [
            marker_python,
            "-I",
            "-B",
            "-c",
            "import os,pathlib,json,setproctitle; "
            "marker=os.environ['E2E_JOB_MARKER']; "
            "setproctitle.setproctitle('smg::router'); "
            "print(json.dumps({'present': ('E2E_JOB_MARKER='+marker).encode() "
            "in pathlib.Path('/proc/self/environ').read_bytes()}))",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout)["present"] is (not drop_preservation)
