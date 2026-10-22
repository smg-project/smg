"""Real-shell cleanup controls with real closure snapshots; no package installs."""

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import runtime_environment as runtime

ROOT = Path(__file__).resolve().parents[2]
CLOSURE = ROOT / ".github/amd-qa/closure_audit.py"


def closure_snapshot(path):
    spec = importlib.util.spec_from_file_location("actual_closure_audit", CLOSURE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.snapshot(path)


def cleanup_body():
    workflow = (ROOT / ".github/workflows/pr-test-rust.yml").read_text()
    section = workflow.split("      - name: Reap owned processes", 1)[1]
    section = section.split("      - name:", 1)[0]
    assert "shell: bash" in section
    return textwrap.dedent(section.split("        run: |\n", 1)[1])


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "diagnostic-results"
    output.mkdir()
    monkeypatch.setattr(runtime, "OUT", output)
    # The snapshot's only Git operation is rev-parse; reuse the real checkout identity.
    (tmp_path / ".git").write_text((ROOT / ".git").read_text())
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for key, value in {
        "GITHUB_RUN_ID": "17",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_SHA": head,
        "RUNNER_NAME": "cpu-control",
        "E2E_ENGINE": "vllm",
        "SMG_QA_IMAGE": "fixture",
        "E2E_JOB_MARKER": "owned-fixture",
    }.items():
        monkeypatch.setenv(key, value)
    fake = tmp_path / "bin/python3"
    fake.parent.mkdir()
    fake.write_text(
        f"#!{sys.executable}\n"
        + f"ROOT={str(ROOT)!r}\nCLOSURE={str(CLOSURE)!r}\n"
        + """
import json,os,subprocess,sys
from pathlib import Path
args=sys.argv[1:]
with open('cleanup-calls.jsonl','a') as stream:
    stream.write(json.dumps(args)+'\\n')
if args[:2]==['-m','pip']:
    assert args==['-m','pip','check']
    print('No broken requirements found.')
elif args[:1]==['.github/amd-qa/closure_audit.py']:
    raise SystemExit(subprocess.call([sys.executable,CLOSURE,*args[1:]]))
elif args[:1]==['.github/amd-pd-env/runtime_environment.py']:
    assert args[1:]==['final']
    sys.path.insert(0,ROOT+'/.github/amd-pd-env')
    import runtime_environment
    runtime_environment.constraint_state('final')
else:
    assert args==['scripts/ci_reap_job_processes.py','owned-fixture']
"""
    )
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake.parent) + ":" + os.environ["PATH"])
    return output


@pytest.mark.parametrize("original", [None, "", "file"])
@pytest.mark.parametrize("damage", [None, "missing", "changed", "unexported"])
def test_final_bash_restores_and_checks_exact_constraints(
    prepared, tmp_path, monkeypatch, original, damage
):
    if original == "file":
        original = str(tmp_path / "image-constraints.txt")
        Path(original).write_text("image-package==1.0\n")
    if original is None:
        monkeypatch.delenv("PIP_CONSTRAINT", raising=False)
    else:
        monkeypatch.setenv("PIP_CONSTRAINT", original)
    closure_snapshot(prepared / "image-before.json")
    before = {
        "records": [{"name": "torch", "version": "1"}, {"name": "vllm", "version": "2"}],
        "active": {"torch": {"version": "1"}, "vllm": {"version": "2"}},
    }
    runtime.runtime_pins(before)
    plan = json.loads((prepared / "constraint-plan.json").read_text())
    assert plan["original_environment"] == original
    assert (
        json.loads((prepared / "constraint-state-before.json").read_text())["environment"]
        == original
    )
    with monkeypatch.context() as installed:
        installed.setenv("PIP_CONSTRAINT", plan["effective_environment"])
        closure_snapshot(prepared / "installed-packages.json")
        runtime.constraint_state("installed")
    (prepared / "install-complete").touch()
    pins = prepared / "runtime-pins.txt"
    body = cleanup_body()
    if damage == "missing":
        pins.unlink()
    elif damage == "changed":
        pins.write_text("torch==changed\n")
    elif damage == "unexported":
        body = "\n".join(
            ":" if line.lstrip().startswith("export PIP_CONSTRAINT=") else line
            for line in body.splitlines()
        )
    result = subprocess.run(["bash", "-e", "-c", body], text=True, capture_output=True)
    (tmp_path / "cleanup.stdout").write_text(result.stdout)
    (tmp_path / "cleanup.stderr").write_text(result.stderr)
    (tmp_path / "cleanup-result.json").write_text(
        json.dumps({"exit_code": result.returncode, "damage": damage, "original": original})
    )
    calls = [
        json.loads(line) for line in (tmp_path / "cleanup-calls.jsonl").read_text().splitlines()
    ]
    assert calls[0] == ["scripts/ci_reap_job_processes.py", "owned-fixture"]
    assert ["-m", "pip", "check"] in calls
    assert [".github/amd-pd-env/runtime_environment.py", "final"] in calls
    final = json.loads((prepared / "final-packages.json").read_text())
    if damage is None:
        assert result.returncode == 0, result.stderr
        assert final["constraints"] == plan["effective"]
        state = json.loads((prepared / "constraint-state-final.json").read_text())
        assert state == {"environment": plan["effective_environment"], "entries": plan["effective"]}
    else:
        assert result.returncode != 0
        assert (prepared / "pip-check-final.txt").read_text() == "No broken requirements found.\n"


def test_snapshot_cannot_claim_different_constraint_bytes(prepared, monkeypatch):
    pins = prepared / "runtime-pins.txt"
    pins.write_text("torch==1\n")
    monkeypatch.setenv("PIP_CONSTRAINT", str(pins))
    expected = runtime.constraint_entries(str(pins))
    (prepared / "constraint-plan.json").write_text(
        json.dumps(
            {
                "effective_environment": str(pins),
                "effective": expected,
            }
        )
    )
    closure_snapshot(prepared / "final-packages.json")
    snapshot = json.loads((prepared / "final-packages.json").read_text())
    snapshot["constraints"][0]["sha256"] = "0" * 64
    (prepared / "final-packages.json").write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="closure snapshot constraint identity differs"):
        runtime.constraint_state("final")
