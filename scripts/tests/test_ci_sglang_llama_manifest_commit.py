"""Committed manifest recovery over inert metadata; no pip or SDK child runs."""

import ast
import hashlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

PROFILE = Path(__file__).parents[1] / "ci_sglang_llama_profile.py"
ENV_KEY = "SMG_SGLANG_LLAMA_COHORT_MANIFEST_SHA256"


def transition(tmp_path, monkeypatch, *, telemetry_failure=False, export_failure=False):
    spec = {
        "python_version": list(sys.version_info[:2]),
        "install_prefix": str(tmp_path / "native"),
        "interpreter_prefix": str(tmp_path / "interpreter"),
        "packages": [],
    }
    site = tmp_path / "native/lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
    site /= "site-packages"
    anchors = {}
    for distribution, version, module in (
        ("apache_tvm_ffi", "0.1.12", "tvm_ffi"),
        ("tilelang", "0.1.15", "tilelang"),
        ("torch_c_dlpack_ext", "0.1.5", "torch_c_dlpack_ext"),
    ):
        source = site / module / "__init__.py"
        source.parent.mkdir(parents=True)
        source.write_text("# inert fixture, never imported\n")
        metadata = site / f"{distribution}-{version}.dist-info/METADATA"
        metadata.parent.mkdir()
        metadata.write_text("fixture-only:" + distribution + "\n")
        anchors[str(metadata.relative_to(site))] = hashlib.sha256(metadata.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(spec))
    initial = hashlib.sha256(manifest.read_bytes()).hexdigest()
    environment = tmp_path / "github-env"
    environment.write_text(ENV_KEY + "=" + initial + "\n")
    if export_failure:
        environment = tmp_path / "missing-parent/github-env"
    monkeypatch.setenv("GITHUB_ENV", str(environment))
    calls = []
    observed = {}

    def run(argv, **kwargs):
        calls.append(argv)
        if "opentelemetry-api" in argv[-1]:
            observed["environment_before_telemetry"] = (
                environment.read_text() if environment.exists() else ""
            )
            if telemetry_failure:
                raise subprocess.CalledProcessError(7, argv)
            return types.SimpleNamespace(stdout="1.2.3\n")
        return types.SimpleNamespace(returncode=0)

    private = types.ModuleType("ci_sglang_private_pip")
    private.install_command = lambda *a, **kw: (["fixture-python"], {})
    private.install = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "ci_sglang_private_pip", private)
    namespace = {
        "Path": Path,
        "NamedTemporaryFile": NamedTemporaryFile,
        "os": os,
        "json": json,
        "hashlib": hashlib,
        "subprocess": types.SimpleNamespace(run=run),
        "native": types.SimpleNamespace(ANCHORS=anchors),
        "context": lambda: None,
        "SDIST": "fixture-only-not-downloaded",
        "__file__": str(PROFILE),
    }
    names = {"native_site", "available_scope", "publish_manifest", "prepare_private"}
    functions = [
        node for node in ast.parse(PROFILE.read_text()).body if isinstance(node, ast.FunctionDef)
    ]
    exec(
        compile(
            ast.Module([node for node in functions if node.name in names], []), str(PROFILE), "exec"
        ),
        namespace,
    )
    cohort = {"path": manifest, "spec": spec}

    def invoke():
        return namespace["prepare_private"](tmp_path / "install.json", cohort)

    return invoke, manifest, environment, initial, observed, calls


def test_late_telemetry_failure_retains_exported_committed_identity(tmp_path, monkeypatch):
    invoke, manifest, environment, initial, observed, _ = transition(
        tmp_path, monkeypatch, telemetry_failure=True
    )
    with pytest.raises(subprocess.CalledProcessError):
        invoke()
    committed = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert committed != initial
    assert ENV_KEY + "=" + committed in observed["environment_before_telemetry"]
    assert environment.read_text().splitlines()[-1] == ENV_KEY + "=" + committed


def test_normal_telemetry_preserves_identity_and_companion_export(tmp_path, monkeypatch):
    invoke, manifest, environment, initial, observed, _ = transition(tmp_path, monkeypatch)
    invoke()
    committed = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert committed != initial
    assert ENV_KEY + "=" + committed in observed["environment_before_telemetry"]
    assert "SMG_CI_OTEL_API_VERSION=1.2.3" in environment.read_text().splitlines()


def test_export_io_failure_stops_before_followup_child(tmp_path, monkeypatch):
    invoke, manifest, _, initial, observed, calls = transition(
        tmp_path, monkeypatch, export_failure=True
    )
    with pytest.raises(FileNotFoundError):
        invoke()
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() != initial
    assert "environment_before_telemetry" not in observed
    assert len(calls) == 1
