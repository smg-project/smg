"""New explicit-context controls; no native/pip/backend execution."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[1] / "ci_sglang_responses_native_phase.py"


@pytest.fixture
def phase():
    spec = importlib.util.spec_from_file_location("native_context_subject", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def put_manifest(phase, tmp_path, context="chat"):
    prefix = tmp_path / "scope" / "interpreter"
    prefix.mkdir(parents=True)
    path = prefix / "smg-cohort.json"
    binding = {
        "source_sha256": phase.source_hashes(),
        "workflow_sha256": phase.digest(SOURCE.parents[1] / ".github/workflows/e2e-gpu-job.yml"),
        "job": {"GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1", "E2E_JOB_MARKER": "job"},
        "nonce": "a" * 32,
    }
    spec = {
        "interpreter_prefix": str(prefix),
        "image": phase.IMAGE,
        "isolated_children": True,
        "import_roots": [str(SOURCE.parent.resolve())],
        context + "_binding": binding,
    }
    if context == "chat":
        spec.update(
            native_consumer_context="chat",
            chat_consumer_sha256="c" * 64,
            chat_producer_contract_sha256="d" * 64,
        )
    path.write_text(json.dumps(spec))
    result = {"path": path, "sha256": phase.digest(path), "spec": spec}
    if context == "chat":
        result["context"] = context
    return result


def activate(phase, monkeypatch, tmp_path, cohort, context):
    boot = types.ModuleType("ci_sglang_cohort")
    boot.__file__ = str(cohort["path"].with_name("sitecustomize.py"))
    boot.SMG_MANIFEST_SHA256 = cohort["sha256"]
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "ci_sglang_cohort", boot)
        m.setitem(sys.modules, "sitecustomize", boot)
        m.setattr(sys, "prefix", cohort["spec"]["interpreter_prefix"])
        m.setattr(sys, "flags", types.SimpleNamespace(isolated=1, dont_write_bytecode=1))
        for key, value in {
            "GITHUB_RUN_ID": "123",
            "GITHUB_RUN_ATTEMPT": "1",
            "E2E_JOB_MARKER": "job",
        }.items():
            m.setenv(key, value)
        m.setenv("RUNNER_TEMP", str(tmp_path))
        m.setenv("E2E_ENGINE", "sglang")
        m.setenv("E2E_GPU_VENDOR", "amd")
        m.setenv("SMG_SGLANG_ROCM_IMAGE", phase.IMAGE)
        return phase.active_manifest(cohort["path"], cohort["sha256"], "a" * 32, context)


@pytest.mark.parametrize("context", ["responses", "chat"])
def test_actual_active_context_retains_exact_source_job_guards(
    phase, tmp_path, monkeypatch, context
):
    cohort = put_manifest(phase, tmp_path, context)
    result = activate(phase, monkeypatch, tmp_path, cohort, context)
    assert result == cohort
    identity = phase.manifest_identity(cohort)
    assert (context + "_binding") in identity
    assert ("consumer_context" in identity) == (context == "chat")


@pytest.mark.parametrize("context, foreign", [("chat", "responses"), ("responses", "chat")])
@pytest.mark.parametrize("field", ["binding", "consumer_sha256", "producer_contract_sha256"])
def test_cross_context_fields_rejected(phase, tmp_path, context, foreign, field):
    cohort = put_manifest(phase, tmp_path, context)
    cohort["spec"][foreign + "_" + field] = {}
    with pytest.raises(ValueError, match="context"):
        phase.manifest_identity(cohort)


@pytest.mark.parametrize("mutation", ["context", "consumer_pin", "producer_pin"])
def test_chat_requires_explicit_context_and_both_source_pins(phase, tmp_path, mutation):
    cohort = put_manifest(phase, tmp_path)
    key = {
        "context": "native_consumer_context",
        "consumer_pin": "chat_consumer_sha256",
        "producer_pin": "chat_producer_contract_sha256",
    }[mutation]
    del cohort["spec"][key]
    with pytest.raises(ValueError):
        phase.manifest_identity(cohort)


@pytest.mark.parametrize("key", ["source_sha256", "workflow_sha256", "nonce", "job"])
def test_actual_active_binding_drift_refused(phase, tmp_path, monkeypatch, key):
    cohort = put_manifest(phase, tmp_path)
    cohort["spec"]["chat_binding"][key] = "changed"
    cohort["path"].write_text(json.dumps(cohort["spec"]))
    cohort["sha256"] = phase.digest(cohort["path"])
    with pytest.raises(ValueError, match="Source, workflow or job nonce"):
        activate(phase, monkeypatch, tmp_path, cohort, "chat")


def test_pending_transaction_remains_fail_closed(phase, tmp_path, monkeypatch):
    cohort = put_manifest(phase, tmp_path)
    (cohort["path"].parent.parent / ".private-pip-transaction").touch()
    with pytest.raises(ValueError, match="transaction"):
        activate(phase, monkeypatch, tmp_path, cohort, "chat")


def complete(phase, cohort, logs):
    identity = phase.manifest_identity(cohort)
    phase.report_path(logs, cohort, "cohort-install").write_text(
        json.dumps({"status": "passed", "phase": "install", "contract": identity})
    )
    phase.report_path(logs, cohort, "native-manifest-commit-2").write_text(json.dumps(identity))


def test_chat_completion_receipts_are_separate_and_cannot_adopt_responses(phase, tmp_path):
    cohort = put_manifest(phase, tmp_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    complete(phase, cohort, logs)
    phase.require_completed_install(logs, cohort)
    receipt = phase.report_path(logs, cohort, "native-manifest-commit-2")
    old = json.loads(receipt.read_text())
    old["responses_binding"] = old.pop("chat_binding")
    receipt.write_text(json.dumps(old))
    with pytest.raises(ValueError, match="identity"):
        phase.require_completed_install(logs, cohort)


def test_chat_prepared_final_names_and_contract_remain_bound(phase, tmp_path, monkeypatch):
    cohort = put_manifest(phase, tmp_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    complete(phase, cohort, logs)
    fake = types.ModuleType("ci_verify_sglang_rocm_cohort")
    fake.verify = lambda report, private_cohort: report.update(observed=True)
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setattr(phase, "job_identity", lambda: {"job": "owned"})
    prepared = phase.verify_phase("prepared", logs, cohort)
    phase.report_path(logs, cohort, "cohort-prepared").write_text(json.dumps(prepared))
    assert phase.verify_phase("final", logs, cohort)["contract"]["consumer_context"] == "chat"
    assert sorted(p.name for p in logs.glob("*-native-*.json")) == [
        "chat-native-final.json",
        "chat-native-manifest-commit-2.json",
        "chat-native-prepared.json",
    ]
    assert not list(logs.glob("responses-*"))


@pytest.mark.parametrize("context", ["responses", "chat"])
def test_two_stage_install_commits_only_selected_context_env_and_names(
    phase, tmp_path, monkeypatch, context
):
    cohort = put_manifest(phase, tmp_path, context)
    logs = tmp_path / "logs"
    logs.mkdir()
    envfile = tmp_path / "env"
    envfile.touch()
    monkeypatch.setenv("GITHUB_ENV", str(envfile))
    monkeypatch.setenv("SMG_SGLANG_ROCM_IMAGE", phase.IMAGE)
    monkeypatch.setattr(phase, "readonly_borrowed_providers", lambda spec: None)
    monkeypatch.setattr(phase.subprocess, "run", lambda *a, **kw: None)
    calls = []
    pip = types.ModuleType("ci_sglang_private_pip")
    pip.install_command = lambda *a: (["inert-python"], {})
    pip.install = lambda *a, **kw: calls.append(a[1])
    native = types.ModuleType("ci_verify_sglang_rocm_cohort")
    native.IMAGE, native.SDIST_URL, native.SDIST_SHA = (
        phase.IMAGE,
        "https://inert.invalid/source",
        "e" * 64,
    )
    monkeypatch.setitem(sys.modules, pip.__name__, pip)
    monkeypatch.setitem(sys.modules, native.__name__, native)

    def rebind(current, unused):
        current["spec"]["test_stage"] = len(calls)
        current["path"].write_text(json.dumps(current["spec"]))
        current["sha256"] = phase.digest(current["path"])

    monkeypatch.setattr(phase, "rebind_native", rebind)
    phase.install_native(cohort, logs)
    assert len(calls) == 2
    assert "--no-deps" in calls[0] and "--no-deps" not in calls[1]
    lines = envfile.read_text().splitlines()
    keys = [line.split("=", 1)[0] for line in lines]
    assert keys == list(phase.CONTEXTS[context]["env"]) * 2
    assert len(list(logs.glob(context + "-native-manifest-commit-*.json"))) == 2
    assert phase.report_path(logs, cohort, "manifest-before-install").is_file()


def test_default_cli_preserves_three_argument_active_call_and_responses_report(
    phase, tmp_path, monkeypatch
):
    cohort = put_manifest(phase, tmp_path, "responses")
    monkeypatch.setattr(phase, "active_manifest", lambda path, sha, nonce: cohort)
    monkeypatch.setattr(phase, "install_native", lambda *args: None)
    logs = tmp_path / "logs"
    assert (
        phase.main(
            [
                "install",
                "--directory",
                str(logs),
                "--manifest",
                str(cohort["path"]),
                "--sha256",
                cohort["sha256"],
                "--nonce",
                "a" * 32,
            ]
        )
        == 0
    )
    result = json.loads((logs / "responses-cohort-install.json").read_text())
    assert result["model_qualification_not_proved"] == list(phase.MODELS)
    assert "consumer_context" not in result["contract"]


@pytest.mark.parametrize("context", ["../chat", "unknown", None])
def test_no_arbitrary_context_or_report_namespace(phase, context):
    with pytest.raises(ValueError):
        phase.context_settings(context)


def test_chat_missing_stage2_receipt_refused(phase, tmp_path):
    cohort = put_manifest(phase, tmp_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    complete(phase, cohort, logs)
    phase.report_path(logs, cohort, "native-manifest-commit-2").unlink()
    with pytest.raises(ValueError, match="receipts missing"):
        phase.require_completed_install(logs, cohort)
