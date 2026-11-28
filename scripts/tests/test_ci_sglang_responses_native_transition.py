"""Real A5 metadata/origin selection, with no pip, Torch or native execution."""

import base64
import csv
import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
NATIVES = (
    ("apache_tvm_ffi", "0.1.12", "tvm_ffi"),
    ("tilelang", "0.1.15", "tilelang"),
    ("torch_c_dlpack_ext", "0.1.5", "torch_c_dlpack_ext"),
)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def package(root, distribution, version, module):
    source = root / module / "__init__.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("raise AssertionError('metadata fixture must never import')\n")
    folder = root / f"{distribution}-{version}.dist-info"
    folder.mkdir(exist_ok=True)
    metadata = folder / "METADATA"
    metadata.write_text(f"Name: {distribution.replace('_', '-')}\nVersion: {version}\n")
    files = [source, metadata]
    if module == "tvm_ffi":
        for name in ("tvm-ffi-config", "tvm-ffi-stubgen"):
            console = root.parents[2] / "bin" / name
            console.parent.mkdir(exist_ok=True)
            console.write_text("metadata-only fixture\n")
            files.append(console)
    with (folder / "RECORD").open("w") as output:
        writer = csv.writer(output)
        for path in files:
            body = path.read_bytes()
            sha = base64.urlsafe_b64encode(hashlib.sha256(body).digest()).rstrip(b"=").decode()
            name = (
                "../../../bin/" + path.name
                if path.parent.name == "bin"
                else str(path.relative_to(root))
            )
            writer.writerow((name, "sha256=" + sha, str(len(body))))
        writer.writerow((str((folder / "RECORD").relative_to(root)), "", ""))
    return dict(
        module=module,
        distribution=distribution.replace("_", "-"),
        version=version,
        package_root=str(root),
        module_path=str(source),
        metadata_path=str(metadata),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
    )


@pytest.fixture
def transition(tmp_path, monkeypatch):
    phase = load("responses_transition_fixture", SCRIPTS / "ci_sglang_responses_native_phase.py")
    native = load("native_metadata_fixture", SCRIPTS / "ci_verify_sglang_rocm_cohort.py")
    boot = load("bootstrap_metadata_fixture", SCRIPTS / "ci_sglang_cohort.py")
    prefix = tmp_path / "interpreter"
    python_minor = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site = prefix / "lib" / python_minor / "site-packages"
    site.mkdir(parents=True)
    copy = site / "ci_sglang_cohort.py"
    copy.write_bytes((SCRIPTS / "ci_sglang_cohort.py").read_bytes())
    monkeypatch.setattr(boot, "__file__", str(copy))
    # This unit exercises activate's actual origin/metadata finder, not the
    # separately qualified launcher, provider mount or interpreter identity.
    monkeypatch.setattr(boot, "_check_child_launch", lambda spec: None)
    (site / "00-smg-cohort.pth").write_text(boot.BOOTSTRAP_PTH)
    cache, marker = prefix / "cache", prefix / "borrowed-metadata"
    cache.mkdir()
    marker.mkdir()
    borrowed = tmp_path / "borrowed/lib" / python_minor / "site-packages"
    borrowed.mkdir(parents=True)
    target = tmp_path / "native"
    owned = target / "lib" / python_minor / "site-packages"
    owned.mkdir(parents=True)
    initial = [package(borrowed, dist, "0.0.1", mod) for dist, _, mod in NATIVES]
    initial.append(package(borrowed, "torch", "2.11.0", "torch"))
    spec = dict(
        interpreter_prefix=str(prefix),
        python_version=list(sys.version_info[:2]),
        install_prefix=str(target),
        isolated_children=True,
        image=phase.IMAGE,
        import_roots=[str(owned)],
        borrowed_sites=[str(borrowed)],
        borrowed_metadata_path=str(marker),
        pycache_prefix=str(cache),
        bootstrap_sha256=phase.digest(copy),
        packages=initial,
        responses_binding={"source": "unchanged", "job": "unchanged", "nonce": "unchanged"},
    )
    manifest = site / "smg-cohort.json"
    manifest.write_text(json.dumps(spec))
    original = manifest.read_bytes()
    cohort = dict(path=manifest, spec=spec, sha256=phase.digest(manifest))
    env = tmp_path / "job-env"
    env.touch()
    logs = tmp_path / "logs"
    logs.mkdir()
    monkeypatch.setenv("GITHUB_ENV", str(env))
    monkeypatch.setenv("SMG_SGLANG_ROCM_IMAGE", phase.IMAGE)
    monkeypatch.setattr(phase, "readonly_borrowed_providers", lambda spec: None)
    monkeypatch.setattr(phase.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "ci_verify_sglang_rocm_cohort", native)
    state = SimpleNamespace(calls=[], corruption=None, failure=None)

    def activate():
        with monkeypatch.context() as context:
            # Simulate the isolated child while keeping this metadata-only
            # fixture independent of pytest's own interpreter launch flags.
            flags = SimpleNamespace(
                **{
                    name: getattr(sys.flags, name)
                    for name in dir(sys.flags)
                    if not name.startswith("_")
                }
            )
            flags.isolated = flags.dont_write_bytecode = 1
            context.setattr(sys, "flags", flags)
            context.setattr(sys, "dont_write_bytecode", True)
            context.setattr(sys, "prefix", str(prefix))
            context.setattr(sys, "path", list(sys.path))
            context.setattr(sys, "meta_path", list(sys.meta_path))
            context.setattr(sys, "pycache_prefix", None)
            return boot.activate(manifest)

    def install(path, arguments, *, cwd):
        observed = activate()  # Before pip, just as in each isolated child.
        state.calls.append(observed)
        stage = len(state.calls)
        if state.failure == stage:
            raise RuntimeError("pip fixture stage failure")
        if stage == 1:
            current = [package(owned, *item) for item in NATIVES]
            anchors = {
                str(Path(item[key]).relative_to(owned)): item[sha]
                for item in current
                for key, sha in (
                    ("module_path", "source_sha256"),
                    ("metadata_path", "metadata_sha256"),
                )
            }
            monkeypatch.setattr(native, "ANCHORS", anchors)
            importlib.invalidate_caches()
        if state.corruption == stage:
            (owned / "tilelang/__init__.py").write_text("changed fixture payload\n")

    monkeypatch.setitem(
        sys.modules,
        "ci_sglang_private_pip",
        SimpleNamespace(install_command=lambda *a: (["fixture-python"], {}), install=install),
    )
    return phase, cohort, logs, env, state, original, owned


def committed_environment(expected):
    return "".join(
        key + "=" + expected + "\n"
        for key in (
            "SMG_RESPONSES_SHA256",
            "SMG_RESPONSES_COHORT_SHA256",
            "SMG_GPTOSS_COHORT_MANIFEST_SHA256",
        )
    )


def test_borrowed_native_rebind_precedes_resolver_bootstrap(transition):
    phase, cohort, logs, env, state, original, owned = transition
    phase.install_native(cohort, logs)
    assert len(state.calls) == 2
    assert Path(state.calls[0]["tvm_ffi"]["module_path"]).is_relative_to(owned) is False
    assert Path(state.calls[1]["tvm_ffi"]["module_path"]).is_relative_to(owned)
    before = json.loads(original)
    after = json.loads(cohort["path"].read_text())
    assert {k: v for k, v in after.items() if k != "packages"} == {
        k: v for k, v in before.items() if k != "packages"
    }
    assert after["packages"][0] == before["packages"][-1]  # Torch protection retained.
    assert (logs / "responses-manifest-before-install.json").read_bytes() == original
    assert env.read_text() == committed_environment(phase.digest(cohort["path"])) * 2


@pytest.mark.parametrize("stage", [1, 2])
def test_invalid_owned_native_data_never_advances(transition, stage):
    phase, cohort, logs, env, state, original, _ = transition
    state.corruption = stage
    with pytest.raises((ValueError, RuntimeError), match="RECORD mismatch|payload anchor"):
        phase.install_native(cohort, logs)
    assert len(state.calls) == stage
    # A failed stage publishes nothing, but cannot erase the earlier known commit.
    assert env.read_text() == committed_environment(phase.digest(cohort["path"])) * (stage - 1)
    assert (logs / "responses-manifest-before-install.json").read_bytes() == original


@pytest.mark.parametrize("stage", [1, 2])
def test_failed_pip_stage_never_exports_failed_stage_or_starts_next_child(transition, stage):
    phase, cohort, logs, env, state, original, _ = transition
    state.failure = stage
    with pytest.raises(RuntimeError, match="pip fixture stage failure"):
        phase.install_native(cohort, logs)
    assert len(state.calls) == stage
    # A failed stage publishes nothing, but cannot erase the earlier known commit.
    assert env.read_text() == committed_environment(phase.digest(cohort["path"])) * (stage - 1)
    assert (logs / "responses-manifest-before-install.json").read_bytes() == original


def actual_private_pip_bridge(transition, monkeypatch):
    """Run new A5 topology/marker code; keep only pip/Torch execution doubled."""
    phase, cohort, _, _, _, _, _ = transition
    backend = sys.modules["ci_sglang_private_pip"].install
    private_pip = load("responses_868_private_pip_fixture", SCRIPTS / "ci_sglang_private_pip.py")
    monkeypatch.setattr(
        private_pip,
        "install_command",
        lambda path, arguments, **kwargs: (["fixture-python", "-m", "pip", *arguments], {}),
    )

    def run(command, **kwargs):
        if command[1] != "-c":
            backend(cohort["path"], command[3:], cwd=kwargs["cwd"])

    monkeypatch.setattr(private_pip.subprocess, "run", run)
    monkeypatch.setitem(sys.modules, "ci_sglang_private_pip", private_pip)
    return private_pip


def test_868_record_postcheck_returns_before_native_rebind_and_next_child(transition, monkeypatch):
    phase, cohort, logs, env, state, original, owned = transition
    private_pip = actual_private_pip_bridge(transition, monkeypatch)
    phase.install_native(cohort, logs)
    assert len(state.calls) == 2  # Two pip children; post-check starts none.
    assert Path(state.calls[1]["tvm_ffi"]["module_path"]).is_relative_to(owned)
    assert len(private_pip.installed_records(cohort["path"])) == 3
    assert not (
        Path(cohort["spec"]["interpreter_prefix"]).parent / ".private-pip-transaction"
    ).exists()
    assert json.loads(original)["python_version"] == list(sys.version_info[:2])
    assert cohort["spec"]["python_version"] == list(sys.version_info[:2])
    assert env.read_text() == committed_environment(phase.digest(cohort["path"])) * 2


def test_868_pip_failure_marker_blocks_reusing_the_scope(transition, monkeypatch):
    phase, cohort, logs, env, state, _, _ = transition
    private_pip = actual_private_pip_bridge(transition, monkeypatch)
    state.failure = 1
    with pytest.raises(RuntimeError, match="pip fixture stage failure"):
        phase.install_native(cohort, logs)
    marker = Path(cohort["spec"]["interpreter_prefix"]).parent / ".private-pip-transaction"
    assert json.loads(marker.read_text())["sha256"] == phase.digest(cohort["path"])
    with pytest.raises(FileExistsError):
        private_pip.install(cohort["path"], ["same-declared-arguments"], cwd=SCRIPTS)
    assert len(state.calls) == 1
    assert marker.is_file()
    assert not env.read_text()


@pytest.mark.parametrize("next_phase", ["prepared", "final"])
def test_partial_second_stage_cannot_become_a_verified_phase(transition, monkeypatch, next_phase):
    phase, cohort, logs, env, state, _, _ = transition
    actual_private_pip_bridge(transition, monkeypatch)
    state.failure = 2
    with pytest.raises(RuntimeError, match="pip fixture stage failure"):
        phase.install_native(cohort, logs)
    assert len(state.calls) == 2
    assert env.read_text() == committed_environment(cohort["sha256"])
    marker = Path(cohort["spec"]["interpreter_prefix"]).parent / ".private-pip-transaction"
    assert marker.is_file()
    assert (logs / "responses-native-manifest-commit-1.json").is_file()
    assert not (logs / "responses-native-manifest-commit-2.json").exists()
    # The public CLI preserves this failed install result after the exception.
    (logs / "responses-cohort-install.json").write_text(
        json.dumps({"status": "failed", "phase": "install"})
    )
    verified = []
    native = sys.modules["ci_verify_sglang_rocm_cohort"]
    monkeypatch.setattr(native, "verify", lambda report, **kwargs: verified.append(True))
    monkeypatch.setattr(phase, "job_identity", lambda: {"job": "fixture"})
    # Even a matching older prepared report must not authorize the final phase.
    (logs / "responses-cohort-prepared.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "contract": {
                    "native": {"verified": True},
                    "manifest_sha256": cohort["sha256"],
                    "source_sha256": phase.source_hashes(),
                    "job": phase.job_identity(),
                },
            }
        )
    )
    with pytest.raises(ValueError, match="Incomplete Responses native installation"):
        phase.verify_phase(next_phase, logs, cohort)
    assert not verified
