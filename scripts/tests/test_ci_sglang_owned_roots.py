"""Cross-root transaction regressions; fake writers, never pip or an SDK."""

import json
import types
from pathlib import Path

import pytest
from test_ci_sglang_cohort import child
from test_ci_sglang_cohort import cohort as cohort_setup
from test_ci_sglang_cohort import private_pip as private_pip_setup


@pytest.fixture
def private_environment(tmp_path, monkeypatch):
    c = cohort_setup.__wrapped__(tmp_path)
    return private_pip_setup.__wrapped__(c, monkeypatch)


def distribution(root, name, version, *, module=True, direct=None):
    folder = root / f"{name}-{version}.dist-info"
    folder.mkdir()
    (folder / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n")
    paths = [f"{folder.name}/METADATA", f"{folder.name}/RECORD"]
    if module:
        (root / f"{name}.py").write_text(f"VALUE={version!r}\n")
        paths.append(f"{name}.py")
    if direct is not None:
        (folder / "direct_url.json").write_text(json.dumps(direct))
        paths.append(f"{folder.name}/direct_url.json")
    (folder / "RECORD").write_text("".join(p + ",,\n" for p in paths))
    return folder


def install_site(api, c):
    spec = json.loads(c.manifest.read_text())
    major, minor = spec["python_version"]
    return Path(spec["install_prefix"]) / f"lib/python{major}.{minor}/site-packages"


def test_other_owned_provider_replacement_refuses_and_poisons_scope(
    private_environment, monkeypatch
):
    api, c, _ = private_environment
    site = install_site(api, c)
    previous = distribution(c.owned, "widget", "1.0")
    original = (previous / "METADATA").read_bytes()
    calls = []

    def write_new(*args, **kwargs):
        calls.append(1)
        distribution(site, "widget", "2.0")

    monkeypatch.setattr(api, "subprocess", types.SimpleNamespace(run=write_new))
    with pytest.raises(ValueError, match="Cross-root|Conflicting|Ambiguous"):
        api.install(c.manifest, ["widget==2.0", "--upgrade"], cwd=c.root)
    assert (previous / "METADATA").read_bytes() == original
    assert (c.root / ".private-pip-transaction").is_file()
    with pytest.raises(FileExistsError):
        api.install(c.manifest, ["widget==2.0"], cwd=c.root)
    assert calls == [1]


def test_unrelated_owned_and_borrowed_providers_do_not_block_additions(
    private_environment, monkeypatch
):
    api, c, _ = private_environment
    site = install_site(api, c)
    distribution(c.owned, "unrelated", "1.0")
    distribution(c.borrowed, "widget", "1.0")
    monkeypatch.setattr(
        api,
        "subprocess",
        types.SimpleNamespace(run=lambda *a, **k: distribution(site, "widget", "2.0")),
    )
    api.install(c.manifest, ["widget==2.0"], cwd=c.root)
    assert not (c.root / ".private-pip-transaction").exists()
    result = child(
        c.python,
        "import widget,importlib.metadata as m; "
        "assert widget.VALUE=='2.0'; assert m.version('widget')=='2.0'",
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("kind", ["directory", "file"])
def test_same_declared_editable_project_metadata_is_not_a_second_provider(
    private_environment, monkeypatch, kind
):
    api, c, _ = private_environment
    site = install_site(api, c)
    project = c.root / "fixture-project"
    (project / "widget.py").write_text("VALUE='editable'\n")
    egg = project / "widget.egg-info"
    text = "Name: widget\nVersion: 1.0\n"
    if kind == "directory":
        egg.mkdir()
        (egg / "PKG-INFO").write_text(text)
    else:
        egg.write_text(text)
    direct = {"dir_info": {"editable": True}, "url": project.as_uri()}
    monkeypatch.setattr(
        api,
        "subprocess",
        types.SimpleNamespace(
            run=lambda *a, **k: distribution(site, "widget", "1.0", module=False, direct=direct)
        ),
    )
    api.install(c.manifest, ["-e", str(project)], cwd=c.root)
    assert egg.exists() and not (c.root / ".private-pip-transaction").exists()
    result = child(c.python, "import widget; assert widget.VALUE=='editable'")
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "kind", ["older_version", "not_editable", "wrong_project", "unrelated_egg", "relative_url"]
)
def test_equal_name_alone_never_proves_an_editable_origin(private_environment, monkeypatch, kind):
    api, c, _ = private_environment
    site = install_site(api, c)
    project = c.root / "fixture-project"
    egg_root = c.owned if kind == "unrelated_egg" else project
    egg = egg_root / "widget.egg-info"
    egg.mkdir()
    (egg / "PKG-INFO").write_text(
        "Name: widget\nVersion: " + ("0.9" if kind == "older_version" else "1.0") + "\n"
    )
    foreign = c.root / "foreign-project"
    foreign.mkdir()
    direct = {
        "dir_info": {"editable": kind != "not_editable"},
        "url": (foreign if kind == "wrong_project" else project).as_uri(),
    }
    if kind == "relative_url":
        direct["url"] = "file:fixture-project"
    monkeypatch.setattr(
        api,
        "subprocess",
        types.SimpleNamespace(
            run=lambda *a, **k: distribution(site, "widget", "1.0", module=False, direct=direct)
        ),
    )
    with pytest.raises(ValueError, match="Cross-root|Conflicting|Editable"):
        api.install(c.manifest, ["-e", str(project)], cwd=c.root)
    assert egg.is_dir() and (c.root / ".private-pip-transaction").is_file()


def test_missing_other_owned_metadata_is_detected_without_deleting_its_modules(
    private_environment, monkeypatch
):
    api, c, _ = private_environment
    site = install_site(api, c)
    previous = distribution(c.owned, "widget", "1.0")

    def remove_metadata(*a, **k):
        for path in previous.iterdir():
            path.unlink()
        previous.rmdir()
        distribution(site, "widget", "2.0")

    monkeypatch.setattr(api, "subprocess", types.SimpleNamespace(run=remove_metadata))
    with pytest.raises(ValueError, match="outside the target disappeared"):
        api.install(c.manifest, ["widget==2.0"], cwd=c.root)
    assert (c.owned / "widget.py").read_text() == "VALUE='1.0'\n"
    assert (c.root / ".private-pip-transaction").is_file()


def test_explicit_owned_metadata_root_outside_import_search_is_also_checked(
    private_environment, monkeypatch
):
    api, c, _ = private_environment
    spec = json.loads(c.manifest.read_text())
    entry = spec["packages"][0]
    old = Path(entry["metadata_path"]).parent
    pins = c.owned / "metadata-pins"
    pins.mkdir()
    relocated = pins / old.name
    old.rename(relocated)
    entry["metadata_path"] = str(relocated / "METADATA")
    c.manifest.write_text(json.dumps(spec))
    site = install_site(api, c)
    monkeypatch.setattr(
        api,
        "subprocess",
        types.SimpleNamespace(run=lambda *a, **k: distribution(site, "ownedfixture", "2.0")),
    )
    with pytest.raises(ValueError, match="Cross-root|Conflicting|Ambiguous"):
        api.install(c.manifest, ["ownedfixture==2.0"], cwd=c.root)
    assert relocated.is_dir() and (c.root / ".private-pip-transaction").is_file()
