"""Only the new private-install scheme, ownership and input boundaries.

No pip/backend/SDK/native code is executed. The original setup fixture creates
an empty stdlib venv; transaction subprocesses are harmless owned file writers.
"""

import json
import sys
import types

import pytest
from test_ci_sglang_cohort import cohort as make_cohort
from test_ci_sglang_cohort import private_pip as make_private_pip
from test_ci_sglang_cohort import (
    test_start_hashes_exactly_the_manifest_bytes_it_validated as check_start_bytes,
)


@pytest.fixture
def private_pip(tmp_path, monkeypatch):
    cohort = make_cohort.__wrapped__(tmp_path)
    return make_private_pip.__wrapped__(cohort, monkeypatch)


def install_site(target):
    return target / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"


def distribution(site, name, files):
    folder = site / f"{name}-1.0.dist-info"
    folder.mkdir(exist_ok=True)
    (folder / "METADATA").write_text(f"Name: {name}\nVersion: 1.0\n")
    paths = [f"{folder.name}/METADATA", f"{folder.name}/RECORD"]
    for relative, content in files.items():
        path = site / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        paths.append(relative)
    (folder / "RECORD").write_text("".join(p + ",,\n" for p in paths))


@pytest.mark.parametrize("collision", [False, True])
def test_record_claims_are_disjoint_before_success(private_pip, monkeypatch, collision):
    api, cohort, target = private_pip
    site = install_site(target)
    distribution(site, "first", {"shared.py": "VALUE='first'\n"})
    other = "shared.py" if collision else "another.py"
    calls = []

    def writer(*args, **kwargs):
        calls.append(True)
        distribution(site, "second", {other: "VALUE='second'\n"})

    monkeypatch.setattr(api, "subprocess", types.SimpleNamespace(run=writer))
    marker = cohort.root / ".private-pip-transaction"
    if collision:
        with pytest.raises(ValueError, match="RECORD path.*multiple distributions"):
            api.install(cohort.manifest, ["fixture-input"], cwd=cohort.root)
        assert marker.is_file() and calls == [True]
        with pytest.raises(FileExistsError):
            api.install(cohort.manifest, ["fixture-input"], cwd=cohort.root)
        assert calls == [True]
    else:
        api.install(cohort.manifest, ["fixture-input"], cwd=cohort.root)
        assert not marker.exists()
        assert "first" in (site / "shared.py").read_text()
        assert "second" in (site / "another.py").read_text()


def test_all_prefix_schemes_must_be_outside_interpreter(private_pip, monkeypatch):
    api, cohort, _ = private_pip
    site = install_site(cohort.root)
    site.mkdir(parents=True)
    spec = json.loads(cohort.manifest.read_bytes())
    spec["install_prefix"] = str(cohort.root)
    spec["import_roots"].append(str(site))
    cohort.manifest.write_text(json.dumps(spec))
    calls = []
    monkeypatch.setattr(
        api, "subprocess", types.SimpleNamespace(run=lambda *a, **k: calls.append(1))
    )
    with pytest.raises(ValueError, match="target must not contain the interpreter"):
        api.install(cohort.manifest, ["fixture-input"], cwd=cohort.root)
    assert not calls and not (cohort.root / ".private-pip-transaction").exists()


@pytest.mark.parametrize("relative", ["bin", "include/nested", "share/data-entry"])
def test_existing_scheme_alias_is_refused_before_pip(private_pip, monkeypatch, relative):
    api, cohort, target = private_pip
    link = target / relative
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(cohort.python.parent, target_is_directory=True)
    calls = []
    monkeypatch.setattr(
        api, "subprocess", types.SimpleNamespace(run=lambda *a, **k: calls.append(1))
    )
    with pytest.raises(ValueError, match="symlink alias"):
        api.install(cohort.manifest, ["fixture-input"], cwd=cohort.root)
    assert not calls and not (cohort.root / ".private-pip-transaction").exists()
    assert link.is_symlink()


def test_unaliased_scheme_directories_remain_accepted(private_pip):
    api, cohort, target = private_pip
    for path in (target / "bin", target / "include/site", target / "share/data-entry"):
        path.mkdir(parents=True)
    command, _ = api.install_command(cohort.manifest, ["fixture-input"], cwd=cohort.root)
    assert command[-1] == "fixture-input"


@pytest.mark.parametrize(
    "suffix, option", [("", "-e"), ("[extra]", "-e"), ("[extra]", "--editable=")]
)
def test_editable_parent_brackets_and_trailing_extras(private_pip, suffix, option):
    api, cohort, _ = private_pip
    project = cohort.root / "build[rocm]/project"
    project.mkdir(parents=True)
    spec = json.loads(cohort.manifest.read_bytes())
    spec["editable_projects"][str(project)] = [str(project)]
    spec["import_roots"].append(str(project))
    cohort.manifest.write_text(json.dumps(spec))
    value = str(project) + suffix
    arguments = [option + value] if option.endswith("=") else [option, value]
    command, _ = api.install_command(cohort.manifest, arguments, cwd=cohort.root)
    assert command[-len(arguments) :] == arguments


def test_short_editable_equals_is_rejected_before_marker(private_pip, monkeypatch):
    api, cohort, _ = private_pip
    # If pip parsed the short attached value, it would see this other project.
    (cohort.root / "=fixture-project").mkdir()
    calls = []
    monkeypatch.setattr(
        api, "subprocess", types.SimpleNamespace(run=lambda *a, **k: calls.append(1))
    )
    with pytest.raises(ValueError, match="Use separate -e"):
        api.install(cohort.manifest, ["-e=fixture-project"], cwd=cohort.root)
    assert not calls and not (cohort.root / ".private-pip-transaction").exists()


@pytest.mark.parametrize("initially_present", [False, True])
def test_start_fixture_restores_sitecustomize(private_pip, monkeypatch, initially_present):
    _, cohort, _ = private_pip
    sentinel = object()
    if initially_present:
        monkeypatch.setitem(sys.modules, "sitecustomize", sentinel)
    else:
        monkeypatch.delitem(sys.modules, "sitecustomize", raising=False)
    with pytest.MonkeyPatch.context() as inner:
        check_start_bytes(cohort, inner)
    assert ("sitecustomize" in sys.modules) is initially_present
    if initially_present:
        assert sys.modules["sitecustomize"] is sentinel
