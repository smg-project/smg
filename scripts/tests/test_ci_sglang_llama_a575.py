"""Only changed ordinary/A5-75 API seams; no real pip, SDK or native imports."""

import ast
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[2]
SCRIPTS = ROOT / "scripts"


def source_module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def profile_functions(*names):
    nodes = ast.parse((SCRIPTS / "ci_sglang_llama_profile.py").read_text()).body
    namespace = {
        "Path": Path,
        "os": os,
        "json": json,
        "NamedTemporaryFile": NamedTemporaryFile,
    }
    exec(
        compile(
            ast.Module(
                body=[n for n in nodes if isinstance(n, ast.FunctionDef) and n.name in names],
                type_ignores=[],
            ),
            "ordinary-profile-seams",
            "exec",
        ),
        namespace,
    )
    return namespace


@pytest.fixture
def editable_checkout(tmp_path):
    checkout = tmp_path / "checkout"
    projects = []
    for relative, module in [
        ("crates/grpc_client/python", "smg_grpc_proto"),
        ("grpc_servicer", "smg_grpc_servicer"),
    ]:
        project = checkout / relative
        (project / module).mkdir(parents=True)
        (project / module / "__init__.py").write_text("# inert package, never imported\n")
        (project / "pyproject.toml").write_bytes((ROOT / relative / "pyproject.toml").read_bytes())
        projects.append(project)
    return checkout, projects


def test_real_two_editable_declarations_reach_new_create_and_private_command(
    editable_checkout, tmp_path, monkeypatch
):
    checkout, projects = editable_checkout
    creator = source_module("ci_sglang_llama_cohort")
    api = source_module("ci_sglang_cohort")
    monkeypatch.syspath_prepend(str(SCRIPTS))
    pip = source_module("ci_sglang_private_pip")
    scope = tmp_path / "owned"
    site = (
        scope / f"native/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    )
    site.mkdir(parents=True)
    roots = [site, *projects]
    mapping = creator.editable_projects(checkout, roots, api.canonical)
    python = api.create(
        scope / "interpreter", roots, [], isolated_children=True, editable_projects=mapping
    )
    manifest = next((scope / "interpreter/lib").glob("python*/site-packages/smg-cohort.json"))
    spec = json.loads(manifest.read_text())
    assert spec["python_version"] == list(sys.version_info[:2])
    assert spec["editable_projects"] == {str(project): [str(project)] for project in projects}
    spec.update(image=pip.IMAGE, install_prefix=str(scope / "native"))
    manifest.write_text(json.dumps(spec))
    for name in ["home", "tmp", "cache", "build-tracker"]:
        (scope / name).mkdir()
    constraint = tmp_path / "constraint.txt"
    constraint.write_text("torch==fixture\n")
    monkeypatch.setattr(pip, "CONSTRAINT", constraint)
    monkeypatch.setattr(pip, "CONSTRAINT_SHA", hashlib.sha256(constraint.read_bytes()).hexdigest())
    monkeypatch.setenv("PIP_CONSTRAINT", str(constraint))
    args = ["-e", "crates/grpc_client/python/", "-e", "grpc_servicer/"]
    command, _ = pip.install_command(manifest, args, cwd=checkout)
    assert command[0] == str(python) and command[-4:] == args
    del spec["editable_projects"][str(projects[1])]
    manifest.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="explicit cohort import mapping"):
        pip.install_command(manifest, args, cwd=checkout)


@pytest.mark.parametrize(
    "bad", ["missing_root", "changed_where", "missing_source", "aliased_project"]
)
def test_unknown_editable_root_is_rejected_instead_of_guessed(editable_checkout, bad):
    checkout, roots = editable_checkout
    creator = source_module("ci_sglang_llama_cohort")
    api = source_module("ci_sglang_cohort")
    roots = list(roots)
    if bad == "missing_root":
        roots.pop()
    elif bad == "changed_where":
        path = roots[1] / "pyproject.toml"
        path.write_text(path.read_text().replace('where = ["."]', 'where = ["src"]'))
    elif bad == "missing_source":
        (roots[1] / "smg_grpc_servicer/__init__.py").unlink()
    else:
        project = roots[1]
        moved = project.with_name("servicer-target")
        project.rename(moved)
        project.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError):
        creator.editable_projects(checkout, roots, api.canonical)


def test_native_scheme_comes_from_manifest_minor_not_data_reader_host(tmp_path):
    native_site = profile_functions("native_site")["native_site"]
    assert (
        native_site({"python_version": [3, 9], "install_prefix": str(tmp_path)})
        == tmp_path / "lib/python3.9/site-packages"
    )


@pytest.mark.parametrize("version", [None, [3], [3, True], [3, -1]])
def test_missing_or_invalid_manifest_minor_is_not_inferred(tmp_path, version):
    native_site = profile_functions("native_site")["native_site"]
    with pytest.raises(ValueError, match="Explicit cohort Python version"):
        native_site({"python_version": version, "install_prefix": str(tmp_path)})


def test_active_consumer_rejects_another_interpreter_minor_before_provider_access(tmp_path):
    nodes = ast.parse((SCRIPTS / "ci_sglang_llama_profile.py").read_text()).body
    sources = next(
        ast.literal_eval(n.value)
        for n in nodes
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "PRIVATE_SOURCES" for t in n.targets)
    )
    # Inert fixture identity; this API unit does not deliver an activation packet.
    image = (
        "lmsysorg/sglang:v0.5.21-rocm10-mi30x@sha256:"
        "4420a7e492fce623285c635b9a6a7e09ccf77a3c812006494e4414655d46662d"
    )
    prefix = tmp_path / "interpreter"
    prefix.mkdir()
    path = prefix / "smg-cohort.json"
    spec = {
        "image": image,
        "isolated_children": True,
        "llama_source_sha256": {
            name: hashlib.sha256((SCRIPTS / name).read_bytes()).hexdigest() for name in sources
        },
        "workflow_sha256": hashlib.sha256(
            (ROOT / ".github/workflows/e2e-gpu-job.yml").read_bytes()
        ).hexdigest(),
        "python_version": [sys.version_info.major, sys.version_info.minor + 1],
        "interpreter_prefix": str(prefix),
        "install_prefix": str(tmp_path / "native"),
    }
    body = json.dumps(spec).encode()
    path.write_bytes(body)
    ns = profile_functions("native_site", "available_scope", "private_manifest")
    ns.update(
        __file__=str(SCRIPTS / "ci_sglang_llama_profile.py"),
        hashlib=hashlib,
        PRIVATE_SOURCES=sources,
        native=SimpleNamespace(IMAGE=image),
        sys=sys,
    )
    with pytest.raises(ValueError, match="interpreter minor differs"):
        ns["private_manifest"](path, hashlib.sha256(body).hexdigest(), active=True)


@pytest.mark.parametrize("linked", [False, True])
def test_failed_scope_stops_the_consumer_before_any_next_command(tmp_path, monkeypatch, linked):
    prefix = tmp_path / "interpreter"
    prefix.mkdir()
    marker = tmp_path / ".private-pip-transaction"
    if linked:
        marker.symlink_to(tmp_path / "absent")
    else:
        marker.write_text("failed owned fixture transaction")
    ns = profile_functions("available_scope", "prepare_private")
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "ci_sglang_private_pip",
        SimpleNamespace(install_command=lambda *a, **k: calls.append((a, k))),
    )
    ns.update(
        context=lambda: None,
        SDIST="fixture-data-only",
        __file__=str(SCRIPTS / "ci_sglang_llama_profile.py"),
    )
    with pytest.raises(ValueError, match="must be discarded"):
        ns["prepare_private"](tmp_path / "report", {"spec": {"interpreter_prefix": str(prefix)}})
    assert calls == [] and marker.is_symlink() == linked


@pytest.mark.parametrize("fail", [False, True])
def test_manifest_publication_is_complete_or_retains_the_previous_contract(
    tmp_path, monkeypatch, fail
):
    publish = profile_functions("publish_manifest")["publish_manifest"]
    path = tmp_path / "smg-cohort.json"
    original = b'{"previous": true}\n'
    path.write_bytes(original)
    real = os.replace
    observed = []

    def replace(source, target):
        assert path.read_bytes() == original
        assert json.loads(Path(source).read_text()) == {"updated": "complete"}
        observed.append(True)
        if fail:
            raise OSError("fixture replace refusal")
        real(source, target)

    monkeypatch.setattr(os, "replace", replace)
    if fail:
        with pytest.raises(OSError):
            publish(path, {"updated": "complete"})
        assert path.read_bytes() == original
    else:
        publish(path, {"updated": "complete"})
        assert json.loads(path.read_text()) == {"updated": "complete"}
    assert observed == [True] and not list(tmp_path.glob(".ordinary-manifest-*"))


def test_actual_create_call_passes_mapping_and_profile_installs_share_checkout_cwd():
    creator = ast.parse((SCRIPTS / "ci_sglang_llama_cohort.py").read_text())
    create = next(
        n
        for n in ast.walk(creator)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "create"
    )
    assert "editable_projects" in {kw.arg for kw in create.keywords}
    profile = ast.parse((SCRIPTS / "ci_sglang_llama_profile.py").read_text())
    calls = [
        n
        for n in ast.walk(profile)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in {"install_command", "install"}
    ]
    assert len(calls) == 3
    assert all(
        any(
            k.arg == "cwd" and isinstance(k.value, ast.Name) and k.value.id == "checkout"
            for k in n.keywords
        )
        for n in calls
    )
    shell = (SCRIPTS / "ci_setup_sglang_rocm_image.sh").read_text()
    assert "install -e crates/grpc_client/python/ -e grpc_servicer/" in shell
