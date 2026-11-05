"""Exercise the actual opt-in scope Bash without a container or backend."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

# Closed fixture environments still need the setup interpreter's loader paths.
INTERPRETER_ENV = {
    name: os.environ[name]
    for name in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH")
    if name in os.environ
}

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
        (scripts / "ci_setup_sglang_rocm_image.sh").write_text(
            '#!/bin/sh\nprintf called > "$PROBE_MARKER"\nexit 97\n'
        )
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
    setups = {step["name"]: step for step in document["jobs"]["run"]["steps"] if "name" in step}
    assert setups["Setup SGLang backend"]["if"] == (
        "inputs.engine == 'sglang' && !inputs.final_environment_check"
    )
    assert (
        setups["Setup AMD SGLang in its fixed ROCm image"]["if"] == "inputs.final_environment_check"
    )
    assert scope["env"] == {
        "CHECK_ENGINE": "${{ inputs.engine }}",
        "CHECK_VENDOR": "${{ inputs.gpu_vendor }}",
        "CHECK_IMAGE": "${{ inputs.container_image }}",
    }


def _evidence_step():
    steps = [
        step
        for step in _document()["jobs"]["run"]["steps"]
        if step.get("id") == "evidence-identity"
    ]
    assert len(steps) == 1
    return steps[0]


def _effective_evidence_inputs(inputs):
    document = _document()
    trigger = document.get("on", document.get(True))["workflow_call"]
    defaults = {
        name: spec.get("default", {"boolean": False, "number": 0}.get(spec["type"], ""))
        for name, spec in trigger["inputs"].items()
    }
    return {**defaults, **inputs}


def _evidence_identity(tmp_path, inputs, attempt="1"):
    inputs = _effective_evidence_inputs(inputs)
    work = tmp_path / str(len(list(tmp_path.iterdir())))
    work.mkdir()
    binary = work / "bin"
    binary.mkdir()
    (binary / "python3").symlink_to(sys.executable)
    output = work / "output"
    script = work / "identity.bash"
    script.write_text(_evidence_step()["run"])
    result = subprocess.run(
        ["/bin/bash", "--noprofile", "--norc", "-e", "-o", "pipefail", str(script)],
        env={
            **INTERPRETER_ENV,
            "PATH": str(binary),
            "EVIDENCE_INPUTS": json.dumps(inputs),
            "GITHUB_RUN_ATTEMPT": attempt,
            "GITHUB_OUTPUT": str(output),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        cwd=work,
        capture_output=True,
        text=True,
        timeout=3,
    )
    assert result.returncode == 0, result.stderr
    fields = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert set(fields) == {"complete_artifact", "worker_artifact"}
    complete = fields["complete_artifact"].removeprefix("sglang-complete-results-")
    worker = fields["worker_artifact"].removeprefix("e2e-worker-logs-")
    assert complete == worker
    assert re.fullmatch(r"[A-Za-z0-9_.-]+", complete)
    contract, actual_attempt, unique = complete.rsplit("-", 2)
    assert actual_attempt == attempt
    assert re.fullmatch(r"[0-9a-f]{32}", unique)
    return contract, unique, fields


def test_evidence_exposes_exact_artifact_outputs():
    document = _document()
    trigger = document.get("on", document.get(True))["workflow_call"]
    job = document["jobs"]["run"]
    for public, internal in [
        ("complete_evidence_artifact", "complete_artifact"),
        ("worker_log_artifact", "worker_artifact"),
    ]:
        assert trigger["outputs"][public]["value"] == "${{ jobs.run.outputs." + public + " }}"
        output = job["outputs"][public]
        upload = (
            "upload-complete-evidence"
            if public == "complete_evidence_artifact"
            else "upload-worker-logs"
        )
        assert output == (
            "${{ steps." + upload + ".outputs.artifact-id != '' && "
            "steps.evidence-identity.outputs." + internal + " || '' }}"
        )
    complete_upload = [
        step for step in job["steps"] if step.get("id") == "upload-complete-evidence"
    ]
    assert len(complete_upload) == 1
    assert complete_upload[0]["if"] == (
        "always() && inputs.final_environment_check && inputs.engine == 'sglang' && inputs.gpu_vendor == 'amd'"
    )
    uploads = [
        step["with"]["name"]
        for step in job["steps"]
        if step.get("uses", "").startswith("actions/upload-artifact@")
    ]
    assert "${{ steps.evidence-identity.outputs.complete_artifact }}" in uploads
    assert "${{ steps.evidence-identity.outputs.worker_artifact }}" in uploads
    assert _evidence_step()["env"] == {"EVIDENCE_INPUTS": "${{ toJSON(inputs) }}"}


def test_evidence_readable_contract_preserves_invocation_uniqueness(tmp_path):
    inputs = {"engine": "sglang", "gpu_tier": "2", "test_dirs": "e2e_test/chat_completions"}
    first = _evidence_identity(tmp_path, inputs)
    second = _evidence_identity(tmp_path, inputs)
    assert first[0] == second[0]
    assert "sglang-e2e_test-chat_completions" in first[0]
    assert first[1] != second[1]
    assert first[2] != second[2]


@pytest.mark.parametrize("count", ["", "1", "2"])
def test_evidence_readable_contract_identifies_grouped_zmq(tmp_path, count):
    contract, _, artifacts = _evidence_identity(
        tmp_path,
        {
            "engine": "vllm",
            "gpu_tier": "1",
            "gpu_vendor": "nvidia",
            "test_dirs": "e2e_test/chat_completions",
            "connection_mode": "zmq",
            "zmq_engine_count": count,
        },
    )
    assert ("-dp2-" in contract) is (count == "2")
    assert "-dp1-" not in contract
    assert all(contract in name for name in artifacts.values())


def test_evidence_contract_ignores_json_key_order(tmp_path):
    inputs = {"engine": "sglang", "gpu_tier": "2", "test_dirs": "e2e_test/router"}
    assert (
        _evidence_identity(tmp_path, inputs)[0]
        == _evidence_identity(tmp_path, dict(reversed(list(inputs.items()))))[0]
    )


@pytest.mark.parametrize(
    "field",
    ["test_dirs", "test_filter", "kv_backend", "connection_mode", "servicer_impl", "mm_processing"],
)
def test_evidence_contract_distinguishes_effective_rows(tmp_path, field):
    inputs = {"engine": "sglang", "gpu_tier": "2", "test_dirs": "e2e_test/chat_completions"}
    changed = {**inputs, field: "changed"}
    assert _evidence_identity(tmp_path, inputs)[0] != _evidence_identity(tmp_path, changed)[0]


def test_evidence_attempt_and_source_looking_json_remain_data(tmp_path):
    marker = tmp_path / "must-not-execute"
    inputs = {
        "engine": "sglang",
        "gpu_tier": "2",
        "test_dirs": "e2e_test/chat: weird/paths",
        "test_filter": "$(touch " + str(marker) + "); `false`\n'\"",
    }
    first = _evidence_identity(tmp_path, inputs, "1")
    second = _evidence_identity(tmp_path, inputs, "2")
    assert first[0] == second[0]
    assert first[2] != second[2]
    assert not marker.exists()


def test_evidence_identity_retains_opt_in_or_failure_boundary():
    step = _evidence_step()
    assert step["if"] == (
        "always() && ((inputs.final_environment_check && inputs.engine == 'sglang' "
        "&& inputs.gpu_vendor == 'amd') || failure() || cancelled())"
    )
    names = [step.get("name") for step in _document()["jobs"]["run"]["steps"]]
    assert names.index("Identify row evidence") < names.index("Check final AMD SGLang environment")
    assert names.index("Check final AMD SGLang environment") < names.index(
        "Stop every process this job started"
    )


@pytest.mark.parametrize("engine", ["sglang", "tokenspeed", "trtllm"])
def test_evidence_non_vllm_defaults_do_not_claim_nixl(tmp_path, engine):
    inputs = {"engine": engine, "gpu_tier": "1", "test_dirs": "e2e_test/chat_completions"}
    assert _effective_evidence_inputs(inputs)["vllm_kv_backend"] == "nixl"
    contract = _evidence_identity(tmp_path, inputs)[0]
    assert "-auto-" in contract
    assert "-nixl-" not in contract


def test_evidence_vllm_keeps_its_configured_default_backend(tmp_path):
    contract = _evidence_identity(tmp_path, {"engine": "vllm", "gpu_tier": "1"})[0]
    assert "-nixl-" in contract


@pytest.mark.parametrize("engine", ["sglang", "vllm"])
def test_evidence_explicit_common_backend_wins_over_vllm_default(tmp_path, engine):
    contract = _evidence_identity(
        tmp_path, {"engine": engine, "gpu_tier": "2", "kv_backend": "mooncake"}
    )[0]
    assert "-mooncake-" in contract and "-nixl-" not in contract


@pytest.mark.parametrize(
    "mode,vendor", [("read", "amd"), ("write", "amd"), ("read", "nvidia"), ("write", "nvidia")]
)
def test_evidence_readable_label_keeps_moriio_mode_and_vendor(tmp_path, mode, vendor):
    contract = _evidence_identity(
        tmp_path,
        {
            "engine": "vllm",
            "gpu_tier": "2",
            "test_dirs": "e2e_test/router",
            "vllm_kv_backend": "moriio",
            "vllm_moriio_mode": mode,
            "gpu_vendor": vendor,
        },
    )[0]
    assert "-moriio-" + mode + "-" + vendor + "-" in contract


def test_evidence_empty_moriio_mode_uses_declared_read_default(tmp_path):
    inputs = {
        "engine": "vllm",
        "gpu_tier": "2",
        "test_dirs": "e2e_test/router",
        "vllm_kv_backend": "moriio",
        "gpu_vendor": "amd",
    }
    assert _effective_evidence_inputs(inputs)["vllm_moriio_mode"] == ""
    contract = _evidence_identity(tmp_path, inputs)[0]
    assert "-moriio-read-amd-" in contract
    assert "-moriio-auto-" not in contract
