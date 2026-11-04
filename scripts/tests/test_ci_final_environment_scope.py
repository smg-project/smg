"""Exercise the actual opt-in scope Bash without a container or backend."""

import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _document():
    return yaml.safe_load((ROOT / ".github/workflows/e2e-gpu-job.yml").read_text())


def _scope():
    steps = [
        step
        for job in _document()["jobs"].values()
        for step in job.get("steps", [])
        if step.get("name") == "Check final-environment verification scope"
    ]
    assert len(steps) == 1
    return steps[0]


def _original_pin():
    source = (ROOT / "scripts/ci_verify_sglang_rocm_cohort.py").read_text()
    pins = re.findall(r'^IMAGE = "([^"\n]+)"$', source, re.MULTILINE)
    assert len(pins) == 1
    return source, pins[0]


def _run(tmp_path, *, requested=None, declaration=None, vendor="amd", engine="sglang", helper=True):
    source, pin = _original_pin()
    work = tmp_path / "work"
    scripts = work / "scripts"
    scripts.mkdir(parents=True)
    marker = tmp_path / "unexpected-helper"
    if helper:
        # Later stack layers add an execution wrapper to the same scope guard.
        # Supply every helper present at this layer, with a marker that still
        # rejects execution: this test exercises the guard, never the helpers.
        for name in (
            "ci_setup_sglang_rocm_image.sh",
            "ci_setup_sglang_rocm_e2e.sh",
            "ci_verify_sglang_rocm_e2e.py",
        ):
            if (ROOT / "scripts" / name).is_file():
                (scripts / name).write_text('#!/bin/sh\nprintf called > "$PROBE_MARKER"\nexit 97\n')
    if declaration != "MISSING_FILE":
        if declaration is not None:
            source = source.replace('IMAGE = "' + pin + '"', declaration, 1)
        (scripts / "ci_verify_sglang_rocm_cohort.py").write_text(source)
    binary = tmp_path / "bin"
    binary.mkdir()
    # Closed command path: only the scope's ordinary grep is available.
    (binary / "grep").symlink_to("/usr/bin/grep")
    script = work / "scope.sh"
    script.write_text(_scope()["run"])
    environment = {
        "PATH": str(binary),
        "CHECK_ENGINE": engine,
        "CHECK_VENDOR": vendor,
        "CHECK_IMAGE": pin if requested is None else requested,
        "PROBE_MARKER": str(marker),
    }
    result = subprocess.run(
        ["/bin/bash", "--noprofile", "--norc", "-e", "-o", "pipefail", str(script)],
        cwd=work,
        env=environment,
        capture_output=True,
        text=True,
        timeout=3,
    )
    assert not marker.exists(), "scope must not execute the helper"
    return result


@pytest.mark.parametrize("requested", ["new-pin", "old-pin"])
def test_scope_follows_the_verifier_pin(tmp_path, requested):
    _, old = _original_pin()
    new = old[:-64] + "a" * 64
    assert new != old
    result = _run(
        tmp_path,
        requested=new if requested == "new-pin" else old,
        declaration='IMAGE = "' + new + '"',
    )
    assert result.returncode == (0 if requested == "new-pin" else 1), result.stderr


@pytest.mark.parametrize("engine,vendor", [("vllm", "amd"), ("sglang", "nvidia")])
def test_scope_retains_engine_and_vendor_boundaries(tmp_path, engine, vendor):
    assert _run(tmp_path, engine=engine, vendor=vendor).returncode != 0


@pytest.mark.parametrize("state", ["empty", "short-digest", "newline-pattern", "shell-looking"])
def test_scope_rejects_malformed_input_as_data(tmp_path, state):
    _, pin = _original_pin()
    requested = {
        "empty": "",
        "short-digest": pin[:-1],
        "newline-pattern": pin + '"\nignored',
        "shell-looking": "$(printf unexpected)",
    }[state]
    assert _run(tmp_path, requested=requested).returncode != 0


@pytest.mark.parametrize(
    "state",
    [
        "missing-file",
        "missing-pin",
        "duplicate",
        "unclosed",
        "computed",
        "multiline",
        "single-quoted",
        "trailing-comment",
    ],
)
def test_scope_rejects_unsupported_verifier_pin_source(tmp_path, state):
    _, pin = _original_pin()
    literal = 'IMAGE = "' + pin + '"'
    declaration = {
        "missing-file": "MISSING_FILE",
        "missing-pin": "# no IMAGE declaration",
        "duplicate": literal + "\n" + literal,
        "unclosed": literal[:-1],
        "computed": literal + ' + ""',
        "multiline": 'IMAGE = (\n    "' + pin + '"\n)',
        "single-quoted": "IMAGE = '" + pin + "'",
        "trailing-comment": literal + "  # alternate format",
    }[state]
    assert _run(tmp_path, declaration=declaration).returncode != 0


def test_scope_requires_the_setup_helper(tmp_path):
    assert _run(tmp_path, helper=False).returncode != 0


def test_scope_stays_opt_in_with_typed_inputs():
    document = _document()
    trigger = document.get("on", document.get(True))
    flag = trigger["workflow_call"]["inputs"]["final_environment_check"]
    assert flag["type"] == "boolean" and flag["default"] is False
    scope = _scope()
    assert scope["if"] == "inputs.final_environment_check"
    assert scope["env"] == {
        "CHECK_ENGINE": "${{ inputs.engine }}",
        "CHECK_VENDOR": "${{ inputs.gpu_vendor }}",
        "CHECK_IMAGE": "${{ inputs.container_image }}",
    }
