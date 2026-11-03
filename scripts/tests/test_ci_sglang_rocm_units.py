"""Exercise the real shell wrapper with a private timeout command fixture."""

import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "ci_run_sglang_rocm_units.sh"


def run_units(tmp_path, *, status=0, xml=True, verify_status=0):
    binary = tmp_path / "bin"
    binary.mkdir(exist_ok=True)
    timeout = binary / "timeout"
    timeout.write_text(
        "#!/usr/bin/python3\n"
        "import json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['UNIT_ARGS']).write_text(json.dumps(sys.argv[1:]))\n"
        "if os.environ['UNIT_XML'] == '1':\n"
        "    pathlib.Path('unit-junit/sglang.xml').write_text('<testsuites/>')\n"
        "raise SystemExit(int(os.environ['UNIT_STATUS']))\n"
    )
    timeout.chmod(0o755)
    checker = binary / "bash"
    checker.write_text(
        "#!/usr/bin/python3\n"
        "import json, os, pathlib, sys\n"
        "assert pathlib.Path(sys.argv[1]).name == 'ci_setup_sglang_rocm_image.sh'\n"
        "assert sys.argv[2:] == ['--verify-only', 'unit-junit/native-environment.json']\n"
        "pathlib.Path(os.environ['UNIT_COHORT_ARGS']).write_text(json.dumps(sys.argv[1:]))\n"
        "raise SystemExit(int(os.environ['UNIT_VERIFY_STATUS']))\n"
    )
    checker.chmod(0o755)
    result = subprocess.run(
        ["/bin/bash", str(SCRIPT)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        env={
            **os.environ,
            "PATH": str(binary) + ":/usr/bin:/bin",
            "GITHUB_OUTPUT": str(tmp_path / "output"),
            "UNIT_ARGS": str(tmp_path / "arguments.json"),
            "UNIT_STATUS": str(status),
            "UNIT_VERIFY_STATUS": str(verify_status),
            "UNIT_COHORT_ARGS": str(tmp_path / "cohort-arguments.json"),
            "UNIT_XML": "1" if xml else "0",
        },
    )
    args_file = tmp_path / "arguments.json"
    return result, json.loads(args_file.read_text()) if args_file.exists() else []


@pytest.mark.parametrize(
    ("status", "xml", "expected"), [(0, True, 0), (7, True, 7), (124, False, 124), (0, False, 1)]
)
def test_complete_unit_selection_timeout_and_original_failure_code(tmp_path, status, xml, expected):
    result, arguments = run_units(tmp_path, status=status, xml=xml)
    assert result.returncode == expected, result.stderr
    assert arguments == [
        "--signal=INT",
        "--kill-after=30s",
        "600s",
        "pytest",
        "-vv",
        "grpc_servicer/tests",
        "-p",
        "no:cacheprovider",
        "-rs",
        "-o",
        "faulthandler_timeout=120",
        "--junitxml=unit-junit/sglang.xml",
    ]
    artifact = (tmp_path / "output").read_text()
    assert re.fullmatch(r"artifact=sglang-unit-results-[0-9a-f]{32}\n", artifact)


def test_artifacts_do_not_collide_for_equivalent_matrix_inputs(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    assert run_units(first)[0].returncode == run_units(second)[0].returncode == 0
    assert (first / "output").read_text() != (second / "output").read_text()


@pytest.mark.parametrize(
    ("engine", "vendor", "final", "failure", "last", "code"),
    [
        ("sglang", "amd", "true", 0, "bash", 0),
        ("sglang", "nvidia", "true", 0, "pytest", 0),
        ("sglang", "amd", "false", 0, "pytest", 0),
        ("sglang", "amd", "true", 44, "bash", 44),
    ],
)
def test_real_callee_routes_only_opted_in_rocm_units_and_keeps_failures(
    tmp_path, engine, vendor, final, failure, last, code
):
    jobs = yaml.safe_load((SCRIPT.parents[1] / ".github/workflows/e2e-gpu-job.yml").read_text())[
        "jobs"
    ]
    steps = next(iter(jobs.values()))["steps"]
    step = next(step for step in steps if step["name"] == "Run engine-gated servicer unit tests")
    upload = next(step for step in steps if step["name"] == "Upload AMD SGLang unit results")
    assert step["id"] == "engine-units"
    assert upload["with"]["name"] == "${{ steps.engine-units.outputs.artifact }}"
    assert "outputs.artifact != ''" in upload["if"]
    binary = tmp_path / "bin"
    binary.mkdir()
    dispatcher = r"""#!/usr/bin/python3
import json, os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
record = {'name': name, 'args': sys.argv[1:], 'image': os.environ.get('SMG_SGLANG_ROCM_IMAGE')}
with open(os.environ['ROUTE_LOG'], 'a') as output:
    output.write(json.dumps(record) + '\n')
if name == 'bash':
    assert sys.argv[1:] == ['scripts/ci_run_sglang_rocm_units.sh']
    raise SystemExit(int(os.environ['ROUTE_FAILURE']))
"""
    for name in ("python3", "pip", "bash", "pytest"):
        path = binary / name
        path.write_text(dispatcher)
        path.chmod(0o755)
    log = tmp_path / "route.jsonl"
    result = subprocess.run(
        ["/bin/bash", "-e", "-c", step["run"].replace("${{ inputs.engine }}", engine)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        env={
            **os.environ,
            "PATH": str(binary) + ":/usr/bin:/bin",
            "UNIT_ENGINE": engine,
            "UNIT_VENDOR": vendor,
            "UNIT_FINAL_ENV": final,
            "UNIT_SGLANG_IMAGE": "pinned-fixture",
            "ROUTE_LOG": str(log),
            "ROUTE_FAILURE": str(failure),
        },
    )
    assert result.returncode == code, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert [call["name"] for call in calls] == ["python3", "pip", last]
    if last == "bash":
        assert calls[-1]["image"] == "pinned-fixture"


def test_failed_post_install_cohort_check_blocks_unit_calls(tmp_path):
    result, arguments = run_units(tmp_path, verify_status=44)
    assert result.returncode == 44
    assert arguments == []
    assert not (tmp_path / "unit-junit/sglang.xml").exists()
    assert json.loads((tmp_path / "cohort-arguments.json").read_text())[1:] == [
        "--verify-only",
        "unit-junit/native-environment.json",
    ]


@pytest.mark.parametrize("status", [0, 44])
def test_real_before_e2e_recheck_follows_oracle_and_model_preparation(tmp_path, status):
    jobs = yaml.safe_load((SCRIPT.parents[1] / ".github/workflows/e2e-gpu-job.yml").read_text())[
        "jobs"
    ]
    steps = next(iter(jobs.values()))["steps"]
    names = [step["name"] for step in steps]
    name = "Recheck AMD SGLang cohort before E2E"
    assert names.index("Setup Oracle") < names.index(name) < names.index("Run E2E tests")
    assert names.index("Download extra models") < names.index(name)
    step = steps[names.index(name)]
    assert step["if"] == "inputs.final_environment_check"
    binary = tmp_path / "bin"
    binary.mkdir()
    checker = binary / "bash"
    checker.write_text(
        "#!/usr/bin/python3\n"
        "import json, os, pathlib, sys\n"
        "assert sys.argv[1:] == [\n"
        "    'scripts/ci_setup_sglang_rocm_image.sh', '--verify-only',\n"
        "    'e2e-logs/sglang-native-environment.json']\n"
        "pathlib.Path(os.environ['CHECK_LOG']).write_text(os.environ['SMG_SGLANG_ROCM_IMAGE'])\n"
        "raise SystemExit(int(os.environ['CHECK_STATUS']))\n"
    )
    checker.chmod(0o755)
    log = tmp_path / "check.log"
    result = subprocess.run(
        ["/bin/bash", "-e", "-c", step["run"]],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=5,
        env={
            **os.environ,
            "PATH": str(binary) + ":/usr/bin:/bin",
            "CHECK_STATUS": str(status),
            "CHECK_LOG": str(log),
            "SMG_SGLANG_ROCM_IMAGE": "pinned-fixture",
        },
    )
    assert result.returncode == status, result.stderr
    assert log.read_text() == "pinned-fixture"
