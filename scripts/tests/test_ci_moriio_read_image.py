"""READ source/image gates without GPU imports, server startup or image pulls."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "moriio_image", ROOT / "scripts/ci_moriio_read_image.py"
)
assert spec and spec.loader
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


@pytest.mark.parametrize(
    "image",
    [
        "registry.example:5000/team/read-image:v1@sha256:" + "a" * 64,
        "vllm/read-image@sha256:" + "b" * 64,
    ],
)
def test_digest_reference_is_captured_exactly(image):
    assert gate.read_image(image) == image


@pytest.mark.parametrize(
    "image",
    [
        "",
        "vllm/read:latest",
        "vllm/read@sha256:abc",
        "bad image@sha256:" + "a" * 64,
        "/vllm/read@sha256:" + "a" * 64,
        "vllm/read/@sha256:" + "a" * 64,
        "vllm/read@sha256:" + "A" * 64,
        "vllm/read@sha256:" + gate.OLD_IMAGE_DIGEST,
    ],
)
def test_unpinned_malformed_and_old_image_are_rejected(image):
    with pytest.raises(ValueError):
        gate.read_image(image)


@pytest.mark.parametrize(
    "env,required",
    [
        ({"E2E_VLLM_KV_BACKEND": "moriio"}, True),
        ({"E2E_VLLM_KV_BACKEND": "moriio", "E2E_VLLM_MORIIO_MODE": ""}, True),
        ({"E2E_VLLM_KV_BACKEND": "moriio", "E2E_VLLM_MORIIO_MODE": "write"}, False),
        ({"E2E_KV_BACKEND": " MORIIO ", "E2E_VLLM_KV_BACKEND": "nixl"}, True),
        ({"E2E_KV_BACKEND": "nixl", "E2E_VLLM_KV_BACKEND": "moriio"}, False),
        ({"E2E_VLLM_KV_BACKEND": "nixl", "E2E_VLLM_EXTRA_KV_BACKENDS": "mooncake,moriio"}, True),
        ({"E2E_VLLM_KV_BACKEND": "nixl"}, False),
    ],
)
def test_effective_read_includes_empty_mode_lane_override_and_extra_fleet(env, required):
    assert gate.needs_read_preflight(env) is required


def test_moriio_invalid_mode_fails():
    with pytest.raises(ValueError):
        gate.needs_read_preflight({"E2E_VLLM_KV_BACKEND": "moriio", "E2E_VLLM_MORIIO_MODE": "both"})


def test_missing_and_unreviewed_source_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        gate.verify_source_root(tmp_path)
    for relative in gate.SOURCE_SHA256:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# source with no reviewed fence\n")
    with pytest.raises(ValueError, match="does not match"):
        gate.verify_source_root(tmp_path)


def test_finder_does_not_import_vllm_and_rejects_source_shadow(tmp_path, monkeypatch):
    package = tmp_path / "vllm"
    package.mkdir()
    (package / "__init__.py").write_text("raise RuntimeError('GPU package must not be imported')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "vllm", raising=False)

    class Distribution:
        def locate_file(self, relative):
            return tmp_path / "other-installation" / relative

    monkeypatch.setattr(gate.importlib.metadata, "distribution", lambda name: Distribution())
    with pytest.raises(ValueError, match="shadowing"):
        gate.verify_installed()


def test_reusable_gate_runs_after_install_before_models_and_workers():
    workflow = yaml.safe_load((ROOT / ".github/workflows/e2e-gpu-job.yml").read_text())
    steps = workflow["jobs"]["run"]["steps"]
    names = [step.get("name") for step in steps]
    guard = names.index("Verify installed MoRI READ source cohort")
    assert names.index("Install wheel and test dependencies") < guard
    assert guard < names.index("Run engine-gated servicer unit tests")
    assert guard < names.index("Download models") < names.index("Run E2E tests")
    assert steps[guard]["if"] == "inputs.engine == 'vllm' && inputs.gpu_vendor == 'amd'"
    assert "verify --image" in steps[guard]["run"]
    assert steps[guard]["env"]["MORIIO_READ_IMAGE"] == "${{ inputs.container_image }}"
    assert steps[guard]["env"]["E2E_GPU_VENDOR"] == "${{ inputs.gpu_vendor }}"
    assert '--qualification-env "$GITHUB_ENV"' in steps[guard]["run"]


def test_read_image_consumes_cpu_matrix_value_not_another_variable_lookup():
    workflow = yaml.safe_load((ROOT / ".github/workflows/nightly-pd.yml").read_text())
    expression = workflow["jobs"]["pd-topologies"]["with"]["container_image"]
    assert expression.startswith("${{ matrix.container_image || ")
    assert "vars." not in expression
    assert gate.OLD_IMAGE_DIGEST in expression
    plan = workflow["jobs"]["plan"]
    assert plan["steps"][0]["uses"] == "actions/checkout@v7"


_READ_IMAGE = "example.invalid/moriio-read@sha256:" + "a" * 64


def _qualification_env(monkeypatch, **overrides):
    env = {
        "E2E_RUNTIME": "vllm",
        "E2E_GPU_VENDOR": "amd",
        "E2E_KV_BACKEND": "",
        "E2E_VLLM_KV_BACKEND": "moriio",
        "E2E_VLLM_MORIIO_MODE": "read",
        "E2E_VLLM_MORIIO_BACKEND": "rdma",
        "E2E_VLLM_EXTRA_KV_BACKENDS": "",
    } | overrides
    for name in (gate.QUALIFICATION_ENV, "MORIIO_READ_IMAGE", "MORI_DISABLE_AUTO_XGMI"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return env


def _exports(path):
    return dict(line.split("=", 1) for line in path.read_text().splitlines())


@pytest.mark.parametrize("mode,backend", [("read", "rdma"), ("", "")])
def test_successful_actual_preflight_exports_first_attempt_to_actions_env(
    monkeypatch, tmp_path, mode, backend
):
    env = _qualification_env(
        monkeypatch, E2E_VLLM_MORIIO_MODE=mode, E2E_VLLM_MORIIO_BACKEND=backend
    )
    calls = []
    monkeypatch.setattr(
        gate, "verify_installed", lambda: calls.append("source") or dict(gate.SOURCE_SHA256)
    )
    output = tmp_path / "github-env"
    monkeypatch.setattr(
        sys, "argv", ["gate", "verify", "--image", _READ_IMAGE, "--qualification-env", str(output)]
    )
    assert gate.main() == 0
    exported = _exports(output)
    assert exported["MORI_DISABLE_AUTO_XGMI"] == "1"
    assert exported["MORIIO_READ_IMAGE"] == _READ_IMAGE
    receipt = json.loads(exported[gate.QUALIFICATION_ENV])
    assert receipt["purpose"] == "first-strict-rdma-read-qualification-attempt"
    assert receipt["source_sha256"] == gate.SOURCE_SHA256
    assert gate.qualification_attempt(env | exported) is True
    assert calls == ["source", "source"]  # Preflight and actual consumer both verify.


@pytest.mark.parametrize("bad", ["image", "source"])
def test_failed_preflight_clears_stale_intent_and_never_exports_success(monkeypatch, tmp_path, bad):
    _qualification_env(monkeypatch)
    output = tmp_path / "github-env"
    output.write_text(f"{gate.QUALIFICATION_ENV}=stale-success\n")

    def invalid_source():
        raise ValueError("installed source mismatch")

    monkeypatch.setattr(gate, "verify_installed", invalid_source)
    image = "example.invalid/read:latest" if bad == "image" else _READ_IMAGE
    monkeypatch.setattr(
        sys, "argv", ["gate", "verify", "--image", image, "--qualification-env", str(output)]
    )
    with pytest.raises(SystemExit) as error:
        gate.main()
    assert error.value.code == 1
    assert _exports(output) == {gate.QUALIFICATION_ENV: ""}


@pytest.mark.parametrize(
    "override",
    [
        {"E2E_VLLM_MORIIO_MODE": "write"},
        {"E2E_VLLM_MORIIO_BACKEND": "xgmi"},
        {"E2E_GPU_VENDOR": "nvidia"},
        {"E2E_RUNTIME": "sglang"},
        {"E2E_VLLM_KV_BACKEND": "nixl", "E2E_VLLM_EXTRA_KV_BACKENDS": "moriio"},
    ],
)
def test_inapplicable_preflight_does_not_grant_restart_eligibility(monkeypatch, tmp_path, override):
    _qualification_env(monkeypatch, **override)
    monkeypatch.setattr(gate, "verify_installed", lambda: dict(gate.SOURCE_SHA256))
    output = tmp_path / "github-env"
    monkeypatch.setattr(
        sys, "argv", ["gate", "verify", "--image", _READ_IMAGE, "--qualification-env", str(output)]
    )
    assert gate.main() == 0
    assert _exports(output) == {gate.QUALIFICATION_ENV: ""}


@pytest.mark.parametrize("bad", ["bare_flag", "image_changed", "receipt_source_changed"])
def test_bare_or_inconsistent_intent_is_not_qualification(monkeypatch, bad):
    env = _qualification_env(monkeypatch)
    exported = gate._qualification_exports(_READ_IMAGE, dict(gate.SOURCE_SHA256), env)
    if bad == "bare_flag":
        exported[gate.QUALIFICATION_ENV] = "1"
    elif bad == "image_changed":
        exported["MORIIO_READ_IMAGE"] = "example.invalid/other@sha256:" + "b" * 64
    else:
        record = json.loads(exported[gate.QUALIFICATION_ENV])
        record["source_sha256"] = {}
        exported[gate.QUALIFICATION_ENV] = json.dumps(record)
    monkeypatch.setattr(gate, "verify_installed", lambda: dict(gate.SOURCE_SHA256))
    with pytest.raises(ValueError):
        gate.qualification_attempt(env | exported)


@pytest.mark.parametrize("shadow", [False, True])
def test_qualification_consumer_rechecks_real_missing_source_and_shadow(
    monkeypatch, tmp_path, shadow
):
    import importlib.machinery

    env = _qualification_env(monkeypatch)
    exported = gate._qualification_exports(_READ_IMAGE, dict(gate.SOURCE_SHA256), env)
    package = tmp_path / "vllm"
    package.mkdir()
    (package / "__init__.py").write_text("raise RuntimeError('must not import GPU package')\n")
    found = importlib.machinery.ModuleSpec("vllm", loader=None, is_package=True)
    found.origin = str(package / "__init__.py")
    found.submodule_search_locations = [str(package)]
    original_find = gate.importlib.util.find_spec
    monkeypatch.setattr(
        gate.importlib.util,
        "find_spec",
        lambda name: found if name == "vllm" else original_find(name),
    )
    monkeypatch.delitem(sys.modules, "vllm", raising=False)

    class Distribution:
        def locate_file(self, name):
            return (tmp_path / "other" if shadow else tmp_path) / name

    monkeypatch.setattr(gate.importlib.metadata, "distribution", lambda name: Distribution())
    with pytest.raises(ValueError, match="shadowing" if shadow else "missing"):
        gate.qualification_attempt(env | exported)
    assert "vllm" not in sys.modules
