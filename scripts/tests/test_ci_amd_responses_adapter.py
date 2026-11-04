"""Execute the actual AMD Responses shell routes without services or GPUs."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/e2e-gpu-job.yml").read_text())
JOB = next(iter(WORKFLOW["jobs"].values()))
STEPS = {step.get("name"): step for step in JOB["steps"]}
PROFILE = {
    "CHECK_AMD_RESPONSES": "true",
    "CHECK_RESPONSES_EVIDENCE": "true",
    "CHECK_FINAL_ENV": "true",
    "CHECK_GPU_VENDOR": "amd",
    "CHECK_GPU_TIER": "1",
    "CHECK_ENGINE": "sglang",
    "CHECK_SETUP_AGENTIC": "true",
    "CHECK_TEST_DIRS": "e2e_test/responses",
    "CHECK_TEST_FILTER": "",
    "CHECK_WIRE": "",
    "CHECK_MIN_SELECTED": "128",
}


@pytest.mark.parametrize(
    ("key", "value", "accepted"),
    [("CHECK_TEST_DIRS", "e2e_test/responses/", True), ("CHECK_MIN_SELECTED", "129", True)]
    + [
        (key, "false", False)
        for key in (
            "CHECK_AMD_RESPONSES",
            "CHECK_RESPONSES_EVIDENCE",
            "CHECK_FINAL_ENV",
            "CHECK_SETUP_AGENTIC",
        )
    ]
    + [
        ("CHECK_GPU_VENDOR", "nvidia", False),
        ("CHECK_ENGINE", "vllm", False),
        ("CHECK_GPU_TIER", "2", False),
        ("CHECK_TEST_DIRS", "e2e_test/responses/partial", False),
        ("CHECK_TEST_FILTER", "-k partial", False),
        ("CHECK_WIRE", "zmq", False),
        ("CHECK_MIN_SELECTED", "127", False),
        ("CHECK_MIN_SELECTED", "128x", False),
    ],
)
def test_actual_early_profile_guard(key, value, accepted):
    result = subprocess.run(
        ["/bin/bash", str(ROOT / "scripts/ci_validate_amd_responses.sh")],
        env={**os.environ, **PROFILE, key: value},
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert (result.returncode == 0) is accepted, result.stderr
    if not accepted:
        assert "requires" in result.stderr


FAKE_PYTEST = r"""
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if Path(sys.argv[0]).name == "python3":
    assert args[:2] == ["-m", "pytest"], args
    args = args[2:]
Path("observed.json").write_text(json.dumps({"argv": args, "evidence": os.environ.get("RESPONSES_EVIDENCE_DIR")}))
print("fixture pytest stdout", flush=True)
print("fixture pytest stderr", file=sys.stderr, flush=True)
raise SystemExit(int(os.environ["CI_FIXTURE_PYTEST_RC"]))
"""


@pytest.mark.parametrize(
    ("final", "evidence", "directory", "status"),
    [
        (True, True, "e2e_test/responses", 0),
        (True, True, "e2e_test/responses", 7),
        (True, False, "e2e_test/chat_completions", 0),
        (True, False, "e2e_test/router", 0),
        (False, False, "e2e_test/chat_completions", 0),
        (False, True, "e2e_test/responses", 0),
    ],
)
def test_actual_run_shell_routes_observer_before_final_capture_exit(
    tmp_path, final, evidence, directory, status
):
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("python3", "pytest"):
        tool = tools / name
        tool.write_text(f"#!{sys.executable}\n" + FAKE_PYTEST)
        tool.chmod(0o755)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    # The actual cleanup command resolves exclusively to this inert local stub.
    (scripts / "ci_killall_sglang.sh").write_text("#!/bin/bash\nexit 0\n")
    run = STEPS["Run E2E tests"]["run"]
    values = {
        "inputs.provider_api_keys": "false",
        "inputs.gpu_vendor": "amd",
        "inputs.test_dirs": directory,
        "inputs.test_filter": "",
    }
    run = re.sub(r"\$\{\{\s*(.*?)\s*\}\}", lambda m: values[m[1]], run)
    result = subprocess.run(
        ["/bin/bash", "-euo", "pipefail", "-c", run],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": str(tools) + os.pathsep + os.environ["PATH"],
            "SMG_PROVIDER_OPENAI_API_KEY": "",
            "SMG_PROVIDER_XAI_API_KEY": "",
            "FINAL_ENVIRONMENT_CHECK": str(int(final)),
            "RESPONSES_EVIDENCE_ENABLED": str(int(evidence)),
            "RESPONSES_EVIDENCE_DIRECTORY": "e2e-logs/responses-fixture",
            "CI_FIXTURE_PYTEST_RC": str(status),
        },
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == status, result.stderr
    observed = json.loads((tmp_path / "observed.json").read_text())
    args = observed["argv"]
    assert args[0] == directory
    assert args.count("--reruns") == 1 and args[args.index("--reruns") + 1] == "2"
    junit = [value for value in args if value.startswith("--junitxml=")]
    if evidence:
        assert args[args.index("-p") + 1] == "scripts.ci_responses_evidence"
        assert junit == ["--junitxml=e2e-logs/responses-fixture/junit.xml"]
        assert observed["evidence"] == "e2e-logs/responses-fixture"
    else:
        assert "scripts.ci_responses_evidence" not in args
        assert observed["evidence"] is None
        assert junit == (["--junitxml=e2e-logs/sglang-results.xml"] if final else [])
    assert "fixture pytest stdout" in result.stdout
    assert "fixture pytest stderr" in result.stderr
    if final:
        assert (tmp_path / "e2e-logs/pytest.stdout").read_text() == "fixture pytest stdout\n"
        assert (tmp_path / "e2e-logs/pytest.stderr").read_text() == "fixture pytest stderr\n"


@pytest.mark.parametrize("uid", [0, 1000])
@pytest.mark.parametrize("fallback", [False, True])
def test_actual_oracle_apt_prefix_root_and_nonroot(tmp_path, uid, fallback):
    source = (ROOT / "scripts/ci_agentic_svc_deps.sh").read_text()
    prefix = source.split("cmd_setup_oracle_client() {", 1)[1].split("    LIBAIO_PATH=", 1)[0]
    # EUID is readonly; only the extracted fixture substitutes a simulated UID.
    prefix = prefix.replace('"$EUID"', '"$CI_FIXTURE_EUID"')
    tools = tmp_path / "tools"
    tools.mkdir()
    fake = r"""
import json, os, sys
from pathlib import Path
name = Path(sys.argv[0]).name
with open(os.environ["CI_FIXTURE_CALLS"], "a") as stream:
    stream.write(json.dumps([name, *sys.argv[1:]]) + "\n")
if name == "sudo":
    os.execvp(sys.argv[1], sys.argv[1:])
if name == "apt-get" and sys.argv[-1:] == ["libaio1t64"] and os.environ["CI_FIXTURE_FALLBACK"] == "1":
    raise SystemExit(13)
"""
    for name in ("bash", "sudo", "apt-get"):
        tool = tools / name
        tool.write_text(f"#!{sys.executable}\n" + fake)
        tool.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    result = subprocess.run(
        [
            "/bin/bash",
            "-euo",
            "pipefail",
            "-c",
            "setup_fixture() {\n" + prefix + "\n}\nsetup_fixture",
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": str(tools) + os.pathsep + os.environ["PATH"],
            "CI_FIXTURE_EUID": str(uid),
            "CI_FIXTURE_FALLBACK": str(int(fallback)),
            "CI_FIXTURE_CALLS": str(calls),
        },
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in calls.read_text().splitlines()]
    apt = [item[1:] for item in records if item[0] == "apt-get"]
    assert apt == [
        ["update"],
        ["install", "-y", "unzip", "wget"],
        ["install", "-y", "libaio1t64"],
    ] + ([["install", "-y", "libaio1"]] if fallback else [])
    assert sum(item[0] == "sudo" for item in records) == (len(apt) if uid else 0)


def test_services_and_always_verifier_keep_unique_evidence_contract():
    services = JOB["services"]
    assert (
        "@sha256:d86d09794ae138a8951e97d7ee010778dcd43088e8d0f83a51002abab8c6d7fd"
        in services["oracle-db"]["image"]
    )
    assert (
        "@sha256:c7fcb7c3f794d8860362c6bcbc2ad959339d07e57503d1a8fc1a5aadf159ac6b"
        in services["brave-search-mcp"]["image"]
    )
    assert services["brave-search-mcp"]["env"]["BRAVE_API_KEY"] == "${{ secrets.BRAVE_API_KEY }}"
    verify = STEPS["Verify complete Responses business evidence"]
    upload = STEPS["Upload Responses evidence"]
    assert verify["if"].startswith("always() && inputs.responses_evidence")
    assert "steps.e2e_tests.outcome != 'skipped'" in verify["if"]
    assert verify["run"].endswith(
        '"$RESPONSES_EVIDENCE_DIR/junit.xml" scripts/ci_responses_original128.json '
        "scripts/ci_responses_required_cap5.json"
    )
    assert upload["with"]["name"] == "${{ steps.responses-evidence.outputs.artifact }}"
    assert upload["with"]["if-no-files-found"] == "error"
    assert STEPS["Run E2E tests"]["id"] == "e2e_tests"
    names = list(STEPS)
    assert names.index("Validate AMD Responses configuration") < names.index(
        "Setup AMD SGLang in its fixed ROCm image"
    )


def test_complete_upload_includes_responses_verdict_even_when_verification_fails():
    names = list(STEPS)
    assert names.index("Run E2E tests") < names.index("Verify complete Responses business evidence")
    assert names.index("Verify complete Responses business evidence") < names.index(
        "Upload AMD SGLang complete test and final-environment evidence"
    )
    verify = STEPS["Verify complete Responses business evidence"]
    upload = STEPS["Upload AMD SGLang complete test and final-environment evidence"]
    assert verify["if"].startswith("always() && inputs.responses_evidence")
    assert upload["if"] == (
        "always() && inputs.final_environment_check && inputs.engine == 'sglang' "
        "&& inputs.gpu_vendor == 'amd'"
    )
    assert upload["id"] == "upload-complete-evidence"
    assert upload["with"]["path"] == "e2e-logs/"
    assert (
        "steps.upload-complete-evidence.outputs.artifact-id != ''"
        in JOB["outputs"]["complete_evidence_artifact"]
    )
