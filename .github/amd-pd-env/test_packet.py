"""CPU-only packet controls. No network, installer, backend or model is invoked."""

import datetime
import io
import json
import sys
import time
import zipfile
from copy import deepcopy
from pathlib import Path

import diagnostic_budget as budget_runner
import probe_evidence
import pytest
import reuse_foundation as reuse


@pytest.fixture
def job(monkeypatch, tmp_path):
    for key, value in {
        "GITHUB_RUN_ID": "17",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_SHA": "a" * 40,
        "RUNNER_NAME": "offline-runner",
        "RUNNER_TEMP": str(tmp_path),
    }.items():
        monkeypatch.setenv(key, value)
    now = datetime.datetime.now(datetime.UTC)
    own = dict(
        id=19,
        name="pd-nixl-env-close",
        runner_name="offline-runner",
        status="in_progress",
        head_sha="a" * 40,
        started_at=(now - datetime.timedelta(seconds=60)).isoformat(),
    )

    def api(path):
        assert path == "actions/runs/17/attempts/2/jobs"
        return io.StringIO(json.dumps({"jobs": [own]}))

    monkeypatch.setattr(reuse, "api_get", api)
    monkeypatch.setattr(reuse, "ROOT", tmp_path)
    return own


def test_job_budget_includes_time_before_core(job):
    record = reuse.consumer_budget()
    start = datetime.datetime.fromisoformat(job["started_at"]).timestamp()
    assert record["diagnostic_deadline_epoch"] == start + 585
    assert record["diagnostic_deadline_epoch"] + 15 + 300 == start + 900
    assert record["diagnostic_deadline_epoch"] - time.time() < 525


@pytest.mark.parametrize("change", ["head", "runner", "expired", "future"])
def test_bad_consumer_cannot_launch(job, monkeypatch, change):
    if change in ["head", "runner"]:
        job["head_sha" if change == "head" else "runner_name"] = "wrong"
    else:
        job["started_at"] = (
            datetime.datetime.now(datetime.UTC)
            + datetime.timedelta(seconds=60 if change == "future" else -601)
        ).isoformat()
    monkeypatch.setattr(
        budget_runner.subprocess, "run", lambda *a, **k: pytest.fail("core launched")
    )
    with pytest.raises(ValueError):
        budget_runner.main()


def test_unfrozen_artifacts_fail_before_api(job, tmp_path, monkeypatch):
    record = reuse.consumer_budget()
    p = tmp_path / "budget.json"
    p.write_text(json.dumps(record))
    monkeypatch.setenv("SMG_PD_ENV_BUDGET_FILE", str(p))
    monkeypatch.setattr(reuse, "api_get", lambda *a: pytest.fail("producer API accessed"))
    with pytest.raises(ValueError, match="ready=false"):
        reuse.main()


def test_budget_receipt_cannot_be_extended(job, tmp_path, monkeypatch):
    record = reuse.consumer_budget()
    record["diagnostic_deadline_epoch"] += 60
    p = tmp_path / "budget.json"
    p.write_text(json.dumps(record))
    monkeypatch.setenv("SMG_PD_ENV_BUDGET_FILE", str(p))
    with pytest.raises(ValueError, match="budget policy"):
        reuse.load_budget()


def test_actual_archive_bytes_and_metadata_fail_closed():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("payload", b"known")
    data = stream.getvalue()
    artifact = dict(
        id=1,
        name="fixture",
        bytes=len(data),
        sha256=reuse.digest(data),
        files={"payload": dict(bytes=5, sha256=reuse.digest(b"known"))},
    )
    assert reuse.unpack(data, artifact) == {"payload": b"known"}
    for payload in [data[:-1], data[:-1] + bytes([data[-1] ^ 1])]:
        with pytest.raises(ValueError):
            reuse.unpack(payload, artifact)
    inputs = dict(run_id=9, producer_sha="a" * 40)
    meta = dict(
        id=1,
        name="fixture",
        expired=False,
        size_in_bytes=len(data),
        digest="sha256:" + artifact["sha256"],
        expires_at=(datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1)).isoformat(),
        workflow_run=dict(id=9, head_sha="a" * 40),
    )
    reuse.check_metadata(meta, artifact, inputs)
    for field, value in [
        ("expired", True),
        ("digest", "sha256:" + "0" * 64),
        ("expires_at", "2000-01-01T00:00:00Z"),
        ("id", 2),
    ]:
        bad = deepcopy(meta)
        bad[field] = value
        with pytest.raises(ValueError):
            reuse.check_metadata(bad, artifact, inputs)


@pytest.mark.parametrize("limited", [False, True])
def test_real_shell_uses_one_deadline_for_reuse_setup_and_pytest(
    job, tmp_path, monkeypatch, limited
):
    script = Path(__file__).with_name("run_pd_close.sh")
    fake = tmp_path / "python3"
    fake.write_text(
        f"#!{sys.executable}\n"
        + """
import json,os,signal,sys,time
from pathlib import Path

def interrupt(*args):
    Path('interrupted').write_text('SIGINT')
    raise SystemExit(130)

signal.signal(signal.SIGINT,interrupt)
args=sys.argv[1:]
with open('commands.jsonl','a') as log:log.write(json.dumps(args)+'\\n')
record=json.loads(Path(os.environ['SMG_PD_ENV_BUDGET_FILE']).read_text())
assert record['consumer_job']==19
Path('diagnostic-results').mkdir(exist_ok=True)
if os.environ['SLOW']=='1':time.sleep(.7)
if args[:1]==['-c'] and 'importlib.metadata' in args[1]:
    print('0.31.0+rocm 1.78.0')
if args[:2]==['-m','venv']:
    path=Path(args[-1])/'bin';path.mkdir(parents=True)
    (path/'activate').write_text(':\\n')
"""
    )
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + ":/usr/bin:/bin")
    monkeypatch.setenv("SLOW", "1" if limited else "0")
    monkeypatch.setattr(budget_runner, "COMMAND", ["bash", str(script)])
    if limited:
        job["started_at"] = (
            datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=583.8)
        ).isoformat()
    before = time.monotonic()
    result = budget_runner.main()
    elapsed = time.monotonic() - before
    commands = [json.loads(line) for line in (tmp_path / "commands.jsonl").read_text().splitlines()]
    assert commands[0] == [".github/amd-pd-env/reuse_foundation.py"]
    assert (
        json.loads((tmp_path / "diagnostic-results/job-budget.json").read_text())[
            "consumer_started_at"
        ]
        == job["started_at"]
    )
    tests = [args for args in commands if args[:2] == ["-m", "pytest"]]
    if limited:
        assert result == 124 and elapsed < 3
        assert (tmp_path / "interrupted").read_text() == "SIGINT"
        assert tests == []
    else:
        assert result == 0 and len(tests) == 1
        assert tests[0][tests[0].index("--reruns") + 1] == "0"
        for args in commands:
            if args[:3] == ["-m", "pip", "install"]:
                assert args[args.index("--retries") + 1] == "0"
        assert any("vllm==0.31.0+rocm" in args for args in commands)
        assert any("grpcio-reflection==1.78.0" in args for args in commands)


@pytest.mark.parametrize(
    "bad", ["skip", "empty", "duplicate", "wrong-case", "unknown-export", "none"]
)
def test_result_gate_rejects_skips_and_incomplete_exports(tmp_path, monkeypatch, bad):
    monkeypatch.setattr(probe_evidence, "OUT", tmp_path)
    body = (
        '<testcase classname="router.test_pd_topologies.TestPDDecodeAbortProbe" '
        'name="test_decode_client_close[pd_http]">{}</testcase>'
    )
    cases = body.format("<skipped/>" if bad == "skip" else "")
    if bad == "empty":
        cases = ""
    if bad == "duplicate":
        cases += cases
    if bad == "wrong-case":
        cases = cases.replace("test_decode_client_close", "test_other")
    (tmp_path / "pd-close.xml").write_text(
        "<testsuites><testsuite>" + cases + "</testsuite></testsuites>"
    )
    export = dict(accepted=False, error=None, backend="nixl", decode_abort=dict(client_closed=True))
    if bad == "unknown-export":
        export["error"] = "incomplete"
    (tmp_path / "pd-close.log").write_text(
        "SMG_PD_PROBE_BEGIN\n" + json.dumps(export) + "\nSMG_PD_PROBE_END\n"
    )
    if bad == "none":
        probe_evidence.results()
        assert json.loads((tmp_path / "probe-export.json").read_text()) == export
    else:
        with pytest.raises(ValueError):
            probe_evidence.results()


def test_budget_expiry_during_bootstrap_never_launches(job, tmp_path, monkeypatch):
    record = reuse.consumer_budget()
    monkeypatch.setattr(budget_runner.time, "time", lambda: record["diagnostic_deadline_epoch"] + 1)
    monkeypatch.setattr(
        budget_runner.subprocess, "run", lambda *a, **k: pytest.fail("core launched")
    )
    with pytest.raises(ValueError, match="before execution"):
        budget_runner.main()
    assert (tmp_path / "diagnostic-results/job-budget.json").is_file()
