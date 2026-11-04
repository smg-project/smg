"""Exercise the real shell wrapper with a private timeout command fixture."""

import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "ci_run_sglang_rocm_units.sh"


def run_units(tmp_path, *, status=0, xml=True, verify_status=0, pip_status=0, provenance_status=0):
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
    python = binary / "python3"
    python.write_text(
        "#!/usr/bin/python3\n"
        "import json, os, pathlib, sys\n"
        "if sys.argv[1] == '-c':\n"
        "    exec(sys.argv[2])\n"
        "elif sys.argv[1:] == ['-m', 'pip', 'check']:\n"
        "    raise SystemExit(int(os.environ['UNIT_PIP_STATUS']))\n"
        "else:\n"
        "    assert pathlib.Path(sys.argv[1]).name == 'ci_verify_sglang_rocm_e2e.py'\n"
        "    assert sys.argv[2:5] == ['--verify', 'setup-baseline.json', '--root']\n"
        "    pathlib.Path(os.environ['UNIT_PROVENANCE_ARGS']).write_text(json.dumps(sys.argv[1:]))\n"
        "    raise SystemExit(int(os.environ['UNIT_PROVENANCE_STATUS']))\n"
    )
    python.chmod(0o755)
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
            "UNIT_PIP_STATUS": str(pip_status),
            "UNIT_PROVENANCE_STATUS": str(provenance_status),
            "UNIT_PROVENANCE_ARGS": str(tmp_path / "provenance-arguments.json"),
            "SMG_CI_SGLANG_E2E_SNAPSHOT": "setup-baseline.json",
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


@pytest.mark.parametrize("probe,code", [("pip", 43), ("provenance", 47)])
def test_failed_pip_or_persistent_baseline_blocks_unit_calls_with_original_code(
    tmp_path, probe, code
):
    result, arguments = run_units(tmp_path, **{probe + "_status": code})
    assert result.returncode == code, result.stderr
    assert arguments == []
    assert not (tmp_path / "unit-junit/sglang.xml").exists()
    statuses = json.loads((tmp_path / "unit-junit/environment-status.json").read_text())
    assert statuses == {
        "pip_returncode": code if probe == "pip" else 0,
        "cohort_returncode": 0,
        "provenance_returncode": code if probe == "provenance" else 0,
    }
    assert (tmp_path / "cohort-arguments.json").exists()
    assert (tmp_path / "provenance-arguments.json").exists()


@pytest.mark.parametrize("phase", ["prepared", "before-e2e", "final"])
@pytest.mark.parametrize("failed", [None, "pip", "cohort", "provenance"])
def test_real_environment_stages_capture_all_probes_and_block_on_failure(tmp_path, phase, failed):
    jobs = yaml.safe_load((SCRIPT.parents[1] / ".github/workflows/e2e-gpu-job.yml").read_text())[
        "jobs"
    ]
    steps = next(iter(jobs.values()))["steps"]
    names = [step["name"] for step in steps]
    name = {
        "prepared": "Check prepared AMD SGLang environment",
        "before-e2e": "Recheck AMD SGLang cohort before E2E",
        "final": "Check final AMD SGLang environment",
    }[phase]
    if phase == "before-e2e":
        assert names.index("Setup Oracle") < names.index(name) < names.index("Run E2E tests")
        assert names.index("Download extra models") < names.index(name)
    step = steps[names.index(name)]
    assert "inputs.final_environment_check" in step["if"]
    binary = tmp_path / "bin"
    binary.mkdir()
    dispatcher = r"""#!/usr/bin/python3
import json, os, pathlib, sys
args = sys.argv[1:]
if pathlib.Path(sys.argv[0]).name == 'bash':
    assert args[0] == 'scripts/ci_setup_sglang_rocm_image.sh'
    assert args[1] == '--verify-only'
    probe = 'cohort'
elif args == ['-m', 'pip', 'check']:
    probe = 'pip'
else:
    assert args == ['scripts/ci_verify_sglang_rocm_e2e.py', '--verify', 'setup-baseline.json', '--root', '.']
    probe = 'provenance'
with open(os.environ['CHECK_LOG'], 'a') as output:
    output.write(json.dumps({'probe': probe, 'image': os.environ['SMG_SGLANG_ROCM_IMAGE']}) + '\n')
print(probe + ' stdout')
print(probe + ' stderr', file=sys.stderr)
raise SystemExit({'pip': 43, 'cohort': 44, 'provenance': 47}[probe] if probe == os.environ['FAILED_PROBE'] else 0)
"""
    for command in ("bash", "python3"):
        checker = binary / command
        checker.write_text(dispatcher)
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
            "FAILED_PROBE": failed or "",
            "CHECK_LOG": str(log),
            "SMG_SGLANG_ROCM_IMAGE": "pinned-fixture",
            "SMG_CI_SGLANG_E2E_SNAPSHOT": "setup-baseline.json",
        },
    )
    assert (result.returncode != 0) == (failed is not None), result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls == [
        {"probe": probe, "image": "pinned-fixture"} for probe in ("pip", "cohort", "provenance")
    ]
    status = json.loads((tmp_path / f"e2e-logs/{phase}-environment-status.json").read_text())
    assert status == {
        probe + "_returncode": code if failed == probe else 0
        for probe, code in (("pip", 43), ("cohort", 44), ("provenance", 47))
    }
    for probe in ("pip", "cohort", "provenance"):
        assert (tmp_path / f"e2e-logs/{phase}-{probe}.stderr").read_text() == probe + " stderr\n"
