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


def test_native_failure_preserves_partial_report(phase, tmp_path, monkeypatch):
    def failed(report, *, private_cohort):
        assert private_cohort is True
        report["retained_anchor"] = "actual-prefix-check-started"
        raise RuntimeError("native fixture refusal")

    monkeypatch.setitem(sys.modules, "ci_verify_sglang_rocm_cohort", SimpleNamespace(verify=failed))
    with pytest.raises(RuntimeError, match="native fixture refusal"):
        phase.verify_phase("final", tmp_path, {})
    report = json.loads((tmp_path / "responses-native-final.json").read_text())
    assert report["verified"] is False
    assert report["retained_anchor"] == "actual-prefix-check-started"
    assert report["error"]["message"] == "native fixture refusal"


def test_final_refuses_a_different_prepared_contract(phase, tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "ci_verify_sglang_rocm_cohort",
        SimpleNamespace(verify=lambda report, **kwargs: report.update(anchor="fixed")),
    )
    (tmp_path / "responses-cohort-prepared.json").write_text(
        json.dumps({"status": "passed", "contract": {"manifest_sha256": "older"}})
    )
    monkeypatch.setattr(phase, "job_identity", lambda: {"job": "current"})
    with pytest.raises(ValueError, match="Prepared/final Responses"):
        phase.verify_phase("final", tmp_path, {"sha256": "current"})
