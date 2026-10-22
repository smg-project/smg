"""Offline timing and exact full-helper shell controls; installers are doubles."""

import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import budget
import pytest

HERE = Path(__file__).resolve().parent


@pytest.fixture
def job(monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    monkeypatch.setenv("RUNNER_NAME", "offline")
    return {
        "id": 17,
        "name": "cairo-product-setup",
        "runner_name": "offline",
        "head_sha": "a" * 40,
        "status": "in_progress",
        "started_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }


@pytest.mark.parametrize("bad", [None, "expired", "future", "head", "runner", "rerun", "extra"])
def test_actual_job_identity_and_shared_absolute_budget(job, monkeypatch, bad):
    now = time.time()
    jobs = [job]
    if bad in ["expired", "future"]:
        job["started_at"] = datetime.datetime.fromtimestamp(
            now + (1 if bad == "future" else -236), datetime.UTC
        ).isoformat()
    elif bad in ["head", "runner"]:
        job["head_sha" if bad == "head" else "runner_name"] = "wrong"
    elif bad == "rerun":
        monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    elif bad == "extra":
        jobs.append(dict(job))
    if bad:
        with pytest.raises(ValueError):
            budget.deadline(jobs, now)
    else:
        record = budget.deadline(jobs, now)
        assert record["hard_deadline_epoch"] - record["soft_deadline_epoch"] == 5
        assert record["job_deadline_epoch"] - record["hard_deadline_epoch"] == 60


@pytest.mark.parametrize("bad", ["not-ready", "changed-budget"])
def test_closed_or_changed_manifest_stops_before_api(tmp_path, monkeypatch, bad):
    inputs = json.loads(budget.INPUTS.read_text())
    if bad == "changed-budget":
        inputs.update(ready=True, core_hard_seconds=241)
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(inputs))
    monkeypatch.setattr(budget, "INPUTS", path)
    monkeypatch.setenv("SMG_QA_IMAGE", budget.IMAGE)
    monkeypatch.setattr(budget, "get_jobs", lambda: pytest.fail("API accessed"))
    with pytest.raises(ValueError):
        budget.main()


def test_real_timeout_includes_elapsed_job_time(job, tmp_path, monkeypatch):
    inputs = json.loads(budget.INPUTS.read_text())
    inputs["ready"] = True
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(inputs))
    monkeypatch.setattr(budget, "INPUTS", path)
    monkeypatch.setattr(budget, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SMG_QA_IMAGE", budget.IMAGE)
    job["started_at"] = datetime.datetime.fromtimestamp(time.time() - 234, datetime.UTC).isoformat()
    monkeypatch.setattr(budget, "get_jobs", lambda: [job])
    monkeypatch.setattr(budget, "COMMAND", ["bash", "-c", "trap 'exit 0' INT; sleep 5"])
    started = time.monotonic()
    assert budget.main() == 124
    assert time.monotonic() - started < 3
    assert json.loads((tmp_path / "cairo-results/job-budget.json").read_text())["job_id"] == 17


@pytest.mark.parametrize("failure", ["", "helper", "after", "native", "pip-after"])
def test_actual_full_helper_and_exit_trap_stay_red(tmp_path, failure):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ["ci_setup_vllm_rocm_image.sh", "ci_retry.sh"]:
        shutil.copyfile(HERE.parents[1] / "scripts" / name, scripts / name)
    binary = tmp_path / "bin"
    binary.mkdir()
    fake = binary / "python3"
    fake.write_text(
        f"#!{sys.executable}\n"
        + """
import json,os,sys
from pathlib import Path
args=sys.argv[1:]
with open('commands.jsonl','a') as stream:stream.write(json.dumps([Path(sys.argv[0]).name,*args])+'\\n')
failure=os.environ['FAILURE']
if Path(sys.argv[0]).name=='pip':
    assert os.environ['ACTIVATED']=='yes'
    raise SystemExit(37 if failure=='helper' else 0)
if args[:2]==['-m','pip']:
    stage='after' if Path('pip-before-seen').exists() else 'before'
    Path('pip-before-seen').touch()
    print('fixture pipcheck '+stage)
    raise SystemExit(1 if stage=='before' or failure=='pip-after' else 0)
if args[:1]==['.github/amd-cairo-product/proof.py']:
    Path('cairo-results').mkdir(exist_ok=True)
    raise SystemExit(1 if args[1]==failure else 0)
if args[:2]==['-m','venv']:
    target=Path('.venv/bin');target.mkdir(parents=True)
    (target/'activate').write_text('export ACTIVATED=yes\\n')
elif args[:1]==['-c'] and 'print(Version(version("vllm"))' in args[1]:
    print('0.31.0+rocm 1.78.0')
"""
    )
    fake.chmod(0o755)
    (binary / "pip").symlink_to(fake)
    (binary / "sleep").write_text("#!/bin/sh\nexit 0\n")
    (binary / "sleep").chmod(0o755)
    env = dict(
        os.environ,
        PATH=str(binary) + ":" + os.environ["PATH"],
        FAILURE=failure,
        GITHUB_ENV="",
        GITHUB_PATH="",
    )
    result = subprocess.run(
        ["bash", str(HERE / "run.sh")],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
    )
    calls = [json.loads(line) for line in (tmp_path / "commands.jsonl").read_text().splitlines()]
    assert (result.returncode != 0) == bool(failure)
    assert ["python3", ".github/amd-cairo-product/proof.py", "native"] in calls
    assert calls[-1] == ["python3", ".github/amd-cairo-product/proof.py", "devices-after"]
    assert (tmp_path / "cairo-results/pip-check-after.txt").is_file()
    attempts = [call for call in calls if call[:2] == ["pip", "install"]]
    assert len(attempts) == (3 if failure == "helper" else 2)
    assert (scripts / "ci_setup_vllm_rocm_image.sh").read_bytes() == (
        HERE.parents[1] / "scripts/ci_setup_vllm_rocm_image.sh"
    ).read_bytes()
