"""Only the ordinary consumer's changed command/target seams; never pip or SDK."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_ci_sglang_llama_a575 import editable_checkout as make_checkout
from test_ci_sglang_llama_a575 import profile_functions, source_module

ROOT = Path(__file__).parents[2]
SCRIPTS = ROOT / "scripts"


@pytest.fixture
def ordinary_command(tmp_path, monkeypatch):
    checkout, _ = make_checkout.__wrapped__(tmp_path)
    bracketed = checkout.with_name("checkout[ordinary]")
    checkout.rename(bracketed)
    projects = [bracketed / relative for relative in ("crates/grpc_client/python", "grpc_servicer")]
    constraint = tmp_path / "torch-constraint.txt"
    constraint.write_text("torch==fixture-only\n")
    # Inert fixture identity; this API unit does not deliver an activation packet.
    image = (
        "lmsysorg/sglang:v0.5.21-rocm10-mi30x@sha256:"
        "4420a7e492fce623285c635b9a6a7e09ccf77a3c812006494e4414655d46662d"
    )
    monkeypatch.setitem(
        sys.modules,
        "ci_verify_sglang_rocm_cohort",
        SimpleNamespace(
            IMAGE=image,
            CONSTRAINT=constraint,
            CONSTRAINT_SHA=hashlib.sha256(constraint.read_bytes()).hexdigest(),
        ),
    )
    api = source_module("ci_sglang_cohort")
    private = source_module("ci_sglang_private_pip")
    creator = source_module("ci_sglang_llama_cohort")
    scope = tmp_path / "owned"
    target = scope / "native"
    site = target / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    site.mkdir(parents=True)
    (target / "bin").mkdir()
    roots = [site, *projects]
    python = api.create(
        scope / "interpreter",
        roots,
        [],
        isolated_children=True,
        editable_projects=creator.editable_projects(bracketed, roots, api.canonical),
    )
    manifest = next((scope / "interpreter/lib").glob("python*/site-packages/smg-cohort.json"))
    spec = json.loads(manifest.read_text())
    spec.update(image=image, install_prefix=str(target))
    manifest.write_text(json.dumps(spec))
    for name in ("home", "tmp", "cache", "build-tracker"):
        (scope / name).mkdir()
    monkeypatch.setenv("PIP_CONSTRAINT", str(constraint))
    return private, manifest, python, bracketed, scope, target, spec


def test_edb_accepts_original_two_editables_in_the_actual_sibling_scheme(ordinary_command):
    api, manifest, python, checkout, _, target, spec = ordinary_command
    arguments = ["-e", "crates/grpc_client/python/", "-e", "grpc_servicer/"]
    command, _ = api.install_command(manifest, arguments, cwd=checkout)
    assert command[0] == str(python)
    assert command[-4:] == arguments
    assert command[command.index("--prefix") + 1] == str(target)
    assert not Path(spec["interpreter_prefix"]).is_relative_to(target)
    assert len(spec["editable_projects"]) == 2


def test_edb_scheme_alias_stops_the_ordinary_consumer_before_torch_or_pip(
    ordinary_command, monkeypatch
):
    api, manifest, python, _, scope, target, spec = ordinary_command
    (target / "include").symlink_to(python.parent, target_is_directory=True)
    calls = []
    monkeypatch.setattr(api, "subprocess", SimpleNamespace(run=lambda *a, **k: calls.append("pip")))
    monkeypatch.setitem(sys.modules, "ci_sglang_private_pip", api)
    ns = profile_functions("available_scope", "prepare_private")
    ns.update(
        __file__=str(SCRIPTS / "ci_sglang_llama_profile.py"),
        context=lambda: None,
        SDIST="fixture-only-sdist-never-executed",
        subprocess=SimpleNamespace(run=lambda *a, **k: calls.append("torch")),
    )
    with pytest.raises(ValueError, match="symlink alias"):
        ns["prepare_private"](scope / "report.json", {"path": manifest, "spec": spec})
    assert calls == []
    assert not (scope / ".private-pip-transaction").exists()
    assert (target / "include").is_symlink()
