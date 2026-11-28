"""Guard/report tests only: no provider, pip, native or model is executed."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "ci_sglang_responses_native_phase.py"


@pytest.fixture
def phase():
    spec = importlib.util.spec_from_file_location("responses_phase_fixture", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cohort(phase, tmp_path, monkeypatch):
    scope = tmp_path / "owned"
    prefix = scope / "interpreter"
    site = prefix / "lib/python3.12/site-packages"
    site.mkdir(parents=True)
    path = site / "smg-cohort.json"
    for key, value in (
        ("RUNNER_TEMP", str(tmp_path)),
        ("GITHUB_RUN_ID", "100"),
        ("GITHUB_RUN_ATTEMPT", "1"),
        ("E2E_JOB_MARKER", "e2e-fixture"),
        ("E2E_ENGINE", "sglang"),
        ("E2E_GPU_VENDOR", "amd"),
    ):
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("SMG_SGLANG_ROCM_IMAGE", phase.IMAGE)
    nonce = "a" * 32
    spec = dict(
        interpreter_prefix=str(prefix),
        isolated_children=True,
        image=phase.IMAGE,
        import_roots=[str(SOURCE.parent)],
        responses_binding=dict(
            source_sha256=phase.source_hashes(),
            job=phase.job_identity(),
            nonce=nonce,
            workflow_sha256=phase.digest(SOURCE.parents[1] / ".github/workflows/e2e-gpu-job.yml"),
        ),
    )
    boot = SimpleNamespace(__file__=str(site / "ci_sglang_cohort.py"))
    monkeypatch.setattr(
        phase,
        "sys",
        SimpleNamespace(
            prefix=str(prefix),
            flags=SimpleNamespace(isolated=True, dont_write_bytecode=True),
            modules={"ci_sglang_cohort": boot, "sitecustomize": boot},
        ),
    )

    def bind():
        path.write_text(json.dumps(spec))
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        boot.SMG_MANIFEST_SHA256 = sha
        return sha

    return spec, path, nonce, bind


@pytest.fixture
def installed(phase, cohort, tmp_path, monkeypatch):
    spec, path, nonce, bind = cohort
    sha = bind()
    directory = tmp_path / "logs"
    identity = {
        "manifest": str(path),
        "sha256": sha,
        "responses_binding": spec["responses_binding"],
        "consumer_sha256": None,
        "producer_contract_sha256": None,
    }

    def complete(context, output):
        assert context["sha256"] == sha
        (output / "responses-native-manifest-commit-2.json").write_text(
            json.dumps({**identity, "previous_sha256": sha})
        )

    monkeypatch.setattr(phase, "install_native", complete)
    argv = [
        "--directory",
        str(directory),
        "--manifest",
        str(path),
        "--sha256",
        sha,
        "--nonce",
        nonce,
    ]
    assert phase.main(["install", *argv]) == 0
    receipt = json.loads((directory / "responses-cohort-install.json").read_text())
    assert receipt["status"] == "passed" and receipt["contract"] == identity
    return phase.active_manifest(path, sha, nonce), directory, argv


def test_exact_active_cohort_binding(phase, cohort):
    _, path, nonce, bind = cohort
    assert phase.active_manifest(path, bind(), nonce)["path"] == path


@pytest.mark.parametrize("field", ["source_sha256", "workflow_sha256", "job", "nonce"])
def test_self_consistent_mutation_cannot_rebind_the_job(phase, cohort, field):
    spec, path, nonce, bind = cohort
    spec["responses_binding"][field] = "different"
    with pytest.raises(ValueError, match="Source, workflow or job nonce"):
        phase.active_manifest(path, bind(), nonce)


def test_changed_startup_manifest_is_rejected(phase, cohort):
    _, path, nonce, bind = cohort
    sha = bind()
    phase.sys.modules["ci_sglang_cohort"].SMG_MANIFEST_SHA256 = "0" * 64
    with pytest.raises(ValueError, match="Matching active isolated"):
        phase.active_manifest(path, sha, nonce)


def test_wrong_nonce_prevents_the_install_call(phase, cohort, tmp_path, monkeypatch):
    _, path, _, bind = cohort

    def forbidden(*args):
        pytest.fail("installer was reached despite wrong nonce")

    monkeypatch.setattr(phase, "install_native", forbidden)
    output = tmp_path / "logs"
    assert (
        phase.main(
            [
                "install",
                "--directory",
                str(output),
                "--manifest",
                str(path),
                "--sha256",
                bind(),
                "--nonce",
                "b" * 32,
            ]
        )
        == 1
    )
    report = json.loads((output / "responses-cohort-install.json").read_text())
    assert report["status"] == "failed"
    assert report["model_qualification_not_proved"] == list(phase.MODELS)


def test_missing_manifest_has_no_legacy_install_route(phase, tmp_path, monkeypatch):
    monkeypatch.setattr(phase, "install_native", lambda *args: pytest.fail("legacy install"))
    with pytest.raises(SystemExit) as error:
        phase.main(["install", "--directory", str(tmp_path)])
    assert error.value.code == 2


def test_native_failure_preserves_partial_report(phase, installed, monkeypatch):
    context, directory, _ = installed

    def failed(report, *, private_cohort):
        assert private_cohort is True
        report["retained_anchor"] = "actual-prefix-check-started"
        raise RuntimeError("native fixture refusal")

    monkeypatch.setitem(sys.modules, "ci_verify_sglang_rocm_cohort", SimpleNamespace(verify=failed))
    with pytest.raises(RuntimeError, match="native fixture refusal"):
        phase.verify_phase("final", directory, context)
    report = json.loads((directory / "responses-native-final.json").read_text())
    assert report["verified"] is False
    assert report["retained_anchor"] == "actual-prefix-check-started"
    assert report["error"]["message"] == "native fixture refusal"


def test_final_refuses_a_different_prepared_contract(phase, installed, monkeypatch):
    context, directory, _ = installed
    monkeypatch.setitem(
        sys.modules,
        "ci_verify_sglang_rocm_cohort",
        SimpleNamespace(verify=lambda report, **kwargs: report.update(anchor="fixed")),
    )
    (directory / "responses-cohort-prepared.json").write_text(
        json.dumps({"status": "passed", "contract": {"manifest_sha256": "older"}})
    )
    monkeypatch.setattr(phase, "job_identity", lambda: {"job": "current"})
    with pytest.raises(ValueError, match="Prepared/final Responses"):
        phase.verify_phase("final", directory, context)


@pytest.mark.parametrize(
    "invalid", ["failed", "stale_contract", "missing_stage2", "stale_commit", "pending"]
)
def test_prepared_requires_a_completed_install_of_this_identity(
    phase, installed, monkeypatch, invalid
):
    context, directory, argv = installed
    receipt_path = directory / "responses-cohort-install.json"
    commit_path = directory / "responses-native-manifest-commit-2.json"
    receipt = json.loads(receipt_path.read_text())
    commit = json.loads(commit_path.read_text())
    if invalid == "failed":
        receipt["status"] = "failed"
    elif invalid == "stale_contract":
        receipt["contract"]["sha256"] = "0" * 64
    elif invalid == "missing_stage2":
        commit_path.unlink()
    elif invalid == "stale_commit":
        commit["sha256"] = "0" * 64
        commit_path.write_text(json.dumps(commit))
    else:
        marker = Path(context["spec"]["interpreter_prefix"]).parent / ".private-pip-transaction"
        marker.write_text("incomplete install fixture")
    receipt_path.write_text(json.dumps(receipt))
    monkeypatch.setitem(
        sys.modules,
        "ci_verify_sglang_rocm_cohort",
        SimpleNamespace(
            verify=lambda *args, **kwargs: pytest.fail("native reached incomplete install")
        ),
    )
    assert phase.main(["prepared", *argv]) == 1
    result = json.loads((directory / "responses-cohort-prepared.json").read_text())
    assert result["status"] == "failed"
    assert "Incomplete Responses native installation" in result["error"]["message"]


def test_completed_install_can_reach_matching_prepared_and_final(phase, installed, monkeypatch):
    _, directory, argv = installed
    monkeypatch.setitem(
        sys.modules,
        "ci_verify_sglang_rocm_cohort",
        SimpleNamespace(verify=lambda report, **kwargs: report.update(anchor="fixture-only")),
    )
    assert phase.main(["prepared", *argv]) == 0
    assert phase.main(["final", *argv]) == 0
    prepared = json.loads((directory / "responses-cohort-prepared.json").read_text())
    final = json.loads((directory / "responses-cohort-final.json").read_text())
    assert prepared["contract"] == final["contract"]
