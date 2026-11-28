"""Ordinary consumer seams over inert metadata; no pip, backend or SDK runs."""

import ast
import importlib.util
import json
import os
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[2]
SCRIPTS = ROOT / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def profile_functions(*names):
    nodes = ast.parse((SCRIPTS / "ci_sglang_llama_profile.py").read_text()).body
    namespace = {"Path": Path, "os": os, "json": json}
    exec(
        compile(
            ast.Module(
                body=[n for n in nodes if isinstance(n, ast.FunctionDef) and n.name in names],
                type_ignores=[],
            ),
            "ordinary-694-consumer-seams",
            "exec",
        ),
        namespace,
    )
    return namespace


def dist(root, name, version, *, direct=None):
    folder = root / (name.replace("-", "_") + "-" + version + ".dist-info")
    folder.mkdir()
    (folder / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n")
    paths = [folder.name + "/METADATA", folder.name + "/RECORD"]
    if direct is not None:
        (folder / "direct_url.json").write_text(json.dumps(direct))
        paths.append(folder.name + "/direct_url.json")
    (folder / "RECORD").write_text("".join(path + ",,\n" for path in paths))
    return folder


def egg(project, name, version, kind="directory"):
    path = project / (name.replace("-", "_") + ".egg-info")
    body = f"Name: {name}\nVersion: {version}\n"
    if kind == "directory":
        path.mkdir()
        (path / "PKG-INFO").write_text(body)
    else:
        path.write_text(body)
    return path


@pytest.fixture
def ordinary(tmp_path, monkeypatch):
    # Inert fixture identity; this API unit does not deliver an activation packet.
    image = (
        "lmsysorg/sglang:v0.5.21-rocm10-mi30x@sha256:"
        "4420a7e492fce623285c635b9a6a7e09ccf77a3c812006494e4414655d46662d"
    )
    monkeypatch.setitem(
        sys.modules,
        "ci_verify_sglang_rocm_cohort",
        SimpleNamespace(CONSTRAINT=tmp_path / "unused", CONSTRAINT_SHA="fixture", IMAGE=image),
    )
    api = load("ci_sglang_private_pip")
    creator = load("ci_sglang_llama_cohort")
    checkout = tmp_path / "checkout"
    projects = []
    for relative, module in (
        ("crates/grpc_client/python", "smg_grpc_proto"),
        ("grpc_servicer", "smg_grpc_servicer"),
    ):
        project = checkout / relative
        (project / module).mkdir(parents=True)
        (project / module / "__init__.py").write_text("# inert, never imported\n")
        (project / "pyproject.toml").write_bytes((ROOT / relative / "pyproject.toml").read_bytes())
        projects.append(project)
    e2e = checkout / "e2e_test"
    e2e.mkdir()
    (e2e / "pyproject.toml").write_bytes((ROOT / "e2e_test/pyproject.toml").read_bytes())
    scope = tmp_path / "scope"
    prefix = scope / "interpreter"
    prefix.mkdir(parents=True)
    site = (
        scope / f"native/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    )
    site.mkdir(parents=True)
    staged = scope / "sdk"
    staged.mkdir()
    roots = [site, staged, *projects, e2e]
    spec = {
        "interpreter_prefix": str(prefix),
        "install_prefix": str(scope / "native"),
        "python_version": list(sys.version_info[:2]),
        "import_roots": [str(p) for p in roots],
        "packages": [],
        "editable_projects": creator.editable_projects(checkout, roots, api.directory),
    }
    manifest = prefix / "smg-cohort.json"
    manifest.write_text(json.dumps(spec))
    commands = []

    def command(path, arguments, *, cwd=None):
        commands.append((list(arguments), cwd))
        return ["fixture-python", "-m", "pip", "install", *arguments], {}

    # Launcher/constraint construction is already qualified separately. This
    # fixture exercises the real new inventory/install transaction, not pip.
    monkeypatch.setattr(api, "install_command", command)
    return SimpleNamespace(
        api=api,
        checkout=checkout,
        projects=projects,
        e2e=e2e,
        scope=scope,
        prefix=prefix,
        site=site,
        staged=staged,
        manifest=manifest,
        spec=spec,
        commands=commands,
    )


@pytest.mark.parametrize("kind", ["directory", "file"])
def test_two_real_editable_mappings_survive_694_owned_inventory(ordinary, monkeypatch, kind):
    c = ordinary
    names = [
        tomllib.loads((p / "pyproject.toml").read_text())["project"]["name"] for p in c.projects
    ]
    before = [egg(p, name, "1.0", kind) for p, name in zip(c.projects, names)]

    def write_editables(*args, **kwargs):
        for project, name in zip(c.projects, names):
            dist(
                c.site,
                name,
                "1.0",
                direct={"dir_info": {"editable": True}, "url": project.as_uri()},
            )

    monkeypatch.setattr(c.api, "subprocess", SimpleNamespace(run=write_editables))
    arguments = ["-e", "crates/grpc_client/python/", "-e", "grpc_servicer/"]
    c.api.install(c.manifest, arguments, cwd=c.checkout)
    assert c.commands == [(arguments, c.checkout)]
    assert all(path.exists() for path in before)
    assert not (c.scope / ".private-pip-transaction").exists()


def test_native_consumer_stops_before_next_stage_on_694_conflict(ordinary, monkeypatch):
    c = ordinary
    dist(c.staged, "tilelang", "0.1.14")
    trace = []

    def incompatible_native(*args, **kwargs):
        trace.append("pip-double")
        dist(c.site, "tilelang", "0.1.15")

    monkeypatch.setattr(c.api, "subprocess", SimpleNamespace(run=incompatible_native))
    monkeypatch.setitem(sys.modules, "ci_sglang_private_pip", c.api)
    ns = profile_functions("available_scope", "prepare_private")
    ns.update(
        context=lambda: None,
        SDIST="fixture-sdist-not-executed",
        __file__=str(SCRIPTS / "ci_sglang_llama_profile.py"),
        subprocess=SimpleNamespace(run=lambda *a, **k: trace.append("torch-double")),
    )
    original = c.manifest.read_bytes()
    with pytest.raises(ValueError, match="Cross-root|Conflicting|Ambiguous"):
        ns["prepare_private"](c.scope / "report.json", {"path": c.manifest, "spec": c.spec})
    assert trace == ["torch-double", "pip-double"]
    assert c.manifest.read_bytes() == original
    assert not (c.scope / "llama-private-input-before-install.json").exists()
    with pytest.raises(ValueError, match="must be discarded"):
        ns["available_scope"](c.prefix)


def test_noneditable_e2e_shadowing_is_refused_and_cannot_reuse_scope(ordinary, monkeypatch):
    c = ordinary
    name = tomllib.loads((c.e2e / "pyproject.toml").read_text())["project"]["name"]
    source_metadata = egg(c.e2e, name, "0.1.0")
    calls = []

    def normal_wheel_metadata(*args, **kwargs):
        calls.append(1)
        dist(c.site, name, "0.1.0", direct={"dir_info": {}, "url": c.e2e.as_uri()})

    monkeypatch.setattr(c.api, "subprocess", SimpleNamespace(run=normal_wheel_metadata))
    with pytest.raises(ValueError, match="Cross-root"):
        c.api.install(c.manifest, ["e2e_test/"], cwd=c.checkout)
    assert source_metadata.is_dir() and (c.scope / ".private-pip-transaction").is_file()
    with pytest.raises(FileExistsError):
        c.api.install(c.manifest, ["e2e_test/"], cwd=c.checkout)
    assert calls == [1]
