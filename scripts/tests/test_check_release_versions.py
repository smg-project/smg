"""Tests for scripts/check_release_versions.py.

The verdicts run against a throwaway repository built here: a tag, then one crate bumped
and synced, one changed but unbumped, one with a version-only manifest edit, one new, one
moved under crates/ without a content change, the gateway with a breaking change, and two
Python packages, one changed. Nothing here touches the real repository.
"""

from __future__ import annotations

import importlib.util
import io
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "check_release_versions.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_release_versions", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_release_versions"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load()


# --- the version lines --------------------------------------------------------


def test_reads_the_first_version_line_of_a_manifest(mod):
    text = '[package]\nname = "a"\nversion = "1.2.3"\n[dependencies]\nb = { version = "9.9.9" }\n'
    assert mod.read_version("cargo", text, "a/Cargo.toml") == "1.2.3"
    assert mod.read_version("pyproject", 'name = "p"\nversion = "0.4.0"\n', "p.toml") == "0.4.0"
    assert mod.read_version("python", '__version__ = "2.0.1"\n', "v.py") == "2.0.1"
    assert (
        mod.read_version(
            "helm", 'apiVersion: v2\nversion: 1.5.0\nappVersion: "1.4.0"\n', "Chart.yaml"
        )
        == "1.5.0"
    )
    assert mod.read_version("workflow", "env:\n  SMG_RELEASE_TAG: v1.13.0\n", "w.yml") == "1.13.0"


def test_workspace_versioning_is_refused(mod):
    with pytest.raises(mod.Fatal, match="workspace versioning"):
        mod.read_version("cargo", "[package]\nversion.workspace = true\n", "a/Cargo.toml")


def test_rewrites_touch_only_the_version(mod):
    manifest = '[package]\nversion = "1.0.0"\n[dependencies]\nb = { version = "1.0.0" }\n'
    assert mod.write_version("cargo", manifest, "1.0.0", "1.1.0") == manifest.replace(
        'version = "1.0.0"\n[', 'version = "1.1.0"\n['
    )
    chart = 'version: 1.0.0\nappVersion: "0.9.0"\n'
    assert (
        mod.write_version("helm", chart, "1.0.0", "1.1.0")
        == 'version: 1.1.0\nappVersion: "1.1.0"\n'
    )
    workflow = "# e.g. smg=v1.0.0-vllm\nenv:\n  SMG_RELEASE_TAG: v1.0.0\n"
    assert mod.write_version("workflow", workflow, "1.0.0", "1.1.0") == workflow.replace(
        "SMG_RELEASE_TAG: v1.0.0", "SMG_RELEASE_TAG: v1.1.0"
    )
    assert mod.write_version("workflow", "nothing here\n", "1.0.0", "1.1.0") is None
    root = 'a = { version = "1.0.0", path = "crates/a" }\nab = { version = "1.0.0", path = "crates/ab" }\n'
    assert mod.workspace_dep_version(root, "a") == "1.0.0"
    assert mod.write_workspace_dep_version(root, "a", "1.0.0", "2.0.0") == root.replace(
        'a = { version = "1.0.0"', 'a = { version = "2.0.0"', 1
    )


# --- the bump -----------------------------------------------------------------


def test_bump_level_follows_conventional_commits(mod):
    assert mod.bump_level([("fix(a): x", ""), ("refactor: y", "")]) == "patch"
    assert mod.bump_level([("fix: x", ""), ("feat(a): y", "")]) == "minor"
    assert mod.bump_level([("feat(a)!: y", "")]) == "major"
    assert mod.bump_level([("fix: x", "body\n\nBREAKING CHANGE: z")]) == "major"
    assert mod.bump_level([]) == "patch"


def test_bumped_versions(mod):
    assert mod.bumped("1.2.3", "major") == "2.0.0"
    assert mod.bumped("1.2.3", "minor") == "1.3.0"
    assert mod.bumped("1.2.3", "patch") == "1.2.4"
    with pytest.raises(mod.Fatal):
        mod.bumped("1.2", "patch")


# --- a throwaway repository ---------------------------------------------------


def git(root: Path, *args: str, date: str = "2026-01-01T00:00:00Z") -> str:
    env = {
        "GIT_AUTHOR_DATE": date,
        "GIT_COMMITTER_DATE": date,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
        "HOME": str(root),
        "PATH": "/usr/bin:/bin",
    }
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True, env=env
    ).stdout


def put(root: Path, path: str, text: str) -> None:
    (root / path).parent.mkdir(parents=True, exist_ok=True)
    (root / path).write_text(text)


def manifest(name: str, version: str) -> str:
    return f'[package]\nname = "{name}"\nversion = "{version}"\nedition = "2021"\n'


def root_manifest(pins: dict[str, str]) -> str:
    lines = [f'{k} = {{ version = "{v}", path = "crates/{k}" }}' for k, v in pins.items()]
    return (
        '[workspace]\nmembers = ["crates/*", "model_gateway"]\n\n[workspace.dependencies]\n'
        + "\n".join(lines)
        + "\n"
    )


@pytest.fixture
def repo(tmp_path, mod):
    """A tagged repository and the changes since the tag the check must judge."""
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    pins = {"alpha": "1.0.0", "beta": "1.0.0", "gamma": "1.0.0", "moved": "1.0.0"}
    put(root, "Cargo.toml", root_manifest(pins))
    for name in ("alpha", "beta", "gamma"):
        put(root, f"crates/{name}/Cargo.toml", manifest(name, "1.0.0"))
        put(root, f"crates/{name}/src/lib.rs", f"// {name}\n")
    put(root, "moved/Cargo.toml", manifest("moved", "1.0.0"))  # lives at the legacy path at the tag
    put(root, "moved/src/lib.rs", "// moved\n")
    put(root, "model_gateway/Cargo.toml", manifest("smg", "1.0.0"))
    put(root, "model_gateway/src/main.rs", "fn main() {}\n")
    put(root, "pkg/pyproject.toml", '[project]\nname = "pkg"\nversion = "0.1.0"\n')
    put(root, "pkg/pkg.py", "x = 1\n")
    put(root, "quiet/pyproject.toml", '[project]\nname = "quiet"\nversion = "0.1.0"\n')
    put(root, "quiet/quiet.py", "y = 1\n")
    put(root, "bindings/Cargo.toml", manifest("smg-binding", "1.0.0"))
    put(root, "bindings/pyproject.toml", '[project]\nname = "smg"\nversion = "1.0.0"\n')
    put(root, "bindings/smg/version.py", '__version__ = "1.0.0"\n')
    put(
        root, "chart/Chart.yaml", 'apiVersion: v2\nname: smg\nversion: 1.0.0\nappVersion: "1.0.0"\n'
    )
    put(
        root,
        "wf.yml",
        "run-name: smg=${{ env.SMG_RELEASE_TAG }}\nenv:\n  SMG_RELEASE_TAG: v1.0.0\n",
    )
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "chore: initial")
    git(root, "tag", "v1.0.0")

    # alpha: changed, bumped, and the root pin follows
    put(root, "crates/alpha/src/lib.rs", "// alpha 2\n")
    put(root, "crates/alpha/Cargo.toml", manifest("alpha", "1.1.0"))
    pins["alpha"] = "1.1.0"
    put(root, "Cargo.toml", root_manifest(pins))
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "feat(alpha): more")
    # beta: changed, not bumped, by a fix
    put(root, "crates/beta/src/lib.rs", "// beta 2\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "fix(beta): less")
    # gamma: only its version line moved (a bump without a change is not a change)
    put(root, "crates/gamma/Cargo.toml", manifest("gamma", "1.0.1"))
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "chore(gamma): version only")
    # moved: under crates/ now, byte-identical
    git(root, "mv", "moved", "crates/moved")
    git(root, "commit", "-q", "-m", "refactor: move the crate")
    # delta: new since the tag, with a root pin that disagrees
    put(root, "crates/delta/Cargo.toml", manifest("delta", "0.2.0"))
    put(root, "crates/delta/src/lib.rs", "// delta\n")
    pins["delta"] = "0.1.0"
    put(root, "Cargo.toml", root_manifest(pins))
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "feat(delta): new crate")
    # the gateway: a breaking change, not bumped
    put(root, "model_gateway/src/main.rs", "fn main() { println!(); }\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "feat(gateway)!: a new wire")
    # pkg: a feat and then a fix, not bumped (the older feat still decides); quiet: untouched
    put(root, "pkg/pkg.py", "x = 2\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "feat(pkg): more")
    put(root, "pkg/pkg.py", "x = 3\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "fix(pkg): typo")
    return root


def tables(mod):
    # the gateway first: the mirror target is found by its name, not by its place in the table
    crates = (
        mod.crate("smg", "model_gateway"),
        mod.crate("alpha", "crates/alpha", "alpha"),
        mod.crate("beta", "crates/beta", "beta"),
        mod.crate("gamma", "crates/gamma", "gamma"),
        mod.crate("moved", "crates/moved", "moved"),
        mod.crate("delta", "crates/delta", "delta"),
    )
    packages = (
        mod.package("pkg", "pkg", "pkg/pyproject.toml"),
        mod.package("quiet", "quiet", "quiet/pyproject.toml"),
    )
    mirrors = (
        mod.Mirror("binding", "bindings/Cargo.toml", "cargo"),
        mod.Mirror("pyproject", "bindings/pyproject.toml", "pyproject"),
        mod.Mirror("version.py", "bindings/smg/version.py", "python"),
        mod.Mirror("chart", "chart/Chart.yaml", "helm"),
        mod.Mirror("workflow", "wf.yml", "workflow"),
    )
    return crates, packages, mirrors


def run(mod, root, *, mode="check", tag=None, override=None, answer="n", ask=None):
    stream = io.StringIO()
    crates, packages, mirrors = tables(mod)
    code = mod.run(
        root,
        tag,
        mode,
        override,
        mod.Out(stream),
        ask=ask or (lambda _prompt: answer),
        crates=crates,
        packages=packages,
        mirrors=mirrors,
    )
    return code, stream.getvalue()


def test_the_repo_reads_its_root_whatever_git_variables_the_caller_carries(mod, repo, monkeypatch):
    """A hook's GIT_DIR / GIT_WORK_TREE would point every git call at another repository
    regardless of ``cwd``; the tool's calls drop them and read the root they were given."""
    monkeypatch.setenv("GIT_DIR", "/nonexistent/.git")
    monkeypatch.setenv("GIT_WORK_TREE", "/nonexistent")
    r = mod.Repo(repo)
    assert "GIT_DIR" not in r.env and "GIT_WORK_TREE" not in r.env
    assert r.newest_tag() is not None
    assert r.has_ref("HEAD")
    assert r.show("HEAD", "Cargo.toml") is not None


@pytest.mark.parametrize("variable", ["GIT_SHALLOW_FILE", "GIT_GRAFT_FILE"])
def test_a_shallow_or_graft_file_the_caller_carries_does_not_cut_the_history(
    mod, repo, monkeypatch, tmp_path, variable
):
    """`GIT_SHALLOW_FILE` and `GIT_GRAFT_FILE` make git treat the listed commits as roots
    (a graft line naming HEAD alone gives it no parents), so an inherited one would cut
    the commits between the tag and HEAD from the history and with them the bump they
    ask for; the tool drops both."""
    head = git(repo, "rev-parse", "HEAD").strip()
    roots = tmp_path / "roots"
    roots.write_text(head + "\n")
    monkeypatch.setenv(variable, str(roots))
    subjects = [subject for subject, _ in mod.Repo(repo).commits_since("v1.0.0", ["crates/alpha/"])]
    assert any(subject.startswith("feat(alpha)") for subject in subjects), subjects


def test_the_verdicts(mod, repo):
    code, out = run(mod, repo)
    assert code == 1
    assert "  ✓ alpha (crates/alpha/) — v1.0.0 → v1.1.0 (1 file(s) changed)" in out
    assert (
        "  ! beta (crates/beta/) — 1 file(s) changed but version not bumped (v1.0.0) [patch]" in out
    )
    assert "gamma" not in out, "a version-only manifest edit is not a change"
    assert "moved" not in out, "a pure move under crates/ is not a change"
    assert "  ✓ delta (crates/delta/) — new crate (v0.2.0), 2 file(s) changed" in out
    assert "    ✗ workspace Cargo.toml has delta v0.1.0, expected v0.2.0" in out
    assert (
        "  ! smg (model_gateway/) — 1 file(s) changed but version not bumped (v1.0.0) [major]"
        in out
    )
    assert "  ! pkg (pkg/) — 1 file(s) changed but version not bumped (v0.1.0) [minor]" in out
    assert "quiet" not in out
    for label, file in (
        ("binding", "bindings/Cargo.toml"),
        ("version.py", "bindings/smg/version.py"),
        ("chart", "chart/Chart.yaml"),
        ("workflow", "wf.yml"),
    ):
        assert f"  ! {label} ({file}) — v1.0.0, expected v2.0.0" in out
    assert (
        "Summary: 4 crate(s) with changes, 2 unchanged; 1 Python package(s) with changes, 1 unchanged"
        in out
    )
    assert "9 issue(s) found." in out
    assert "Proposed fixes" not in out, "--check never proposes"


def test_the_installed_packages_version_file_is_a_checked_mirror(mod):
    """smg.__version__ is read from bindings/python/src/smg/version.py; the bump writes it and this check reads it."""
    mirror = next(m for m in mod.MIRRORS if m.file == "bindings/python/src/smg/version.py")
    assert mirror.kind == "python"
    assert mod.read_version(mirror.kind, '__version__ = "1.2.3"\n', mirror.file) == "1.2.3"
    assert (
        mod.write_version(mirror.kind, '__version__ = "1.2.3"\n', "1.2.3", "1.3.0")
        == '__version__ = "1.3.0"\n'
    )


def test_the_prompt_declined_changes_nothing(mod, repo):
    before = (repo / "Cargo.toml").read_text()
    code, out = run(mod, repo, mode="prompt", answer="n")
    assert code == 1
    assert "  patch beta v1.0.0 → v1.0.1 (crates/beta/Cargo.toml)" in out
    assert "       sync workspace Cargo.toml beta → v1.0.1" in out
    assert "  sync workspace Cargo.toml delta v0.1.0 → v0.2.0" in out
    assert "  major smg v1.0.0 → v2.0.0 (model_gateway/Cargo.toml)" in out
    assert "  minor pkg v0.1.0 → v0.2.0 (pkg/pyproject.toml)" in out
    assert "  sync chart v1.0.0 → v2.0.0 (chart/Chart.yaml)" in out
    assert out.rstrip().endswith("No changes made.")
    assert (repo / "Cargo.toml").read_text() == before


def test_a_closed_stdin_at_the_prompt_is_a_no(mod, repo):
    def closed(_prompt):
        raise EOFError

    before = (repo / "Cargo.toml").read_text()
    code, out = run(mod, repo, mode="prompt", ask=closed)
    assert code == 1
    assert out.rstrip().endswith("No changes made.")
    assert (repo / "Cargo.toml").read_text() == before


def test_the_fixes_make_the_tree_consistent(mod, repo):
    code, out = run(mod, repo, mode="fix")
    assert code == 0, out
    assert "Done. Re-run to verify." in out
    assert mod.read_version("cargo", (repo / "crates/beta/Cargo.toml").read_text(), "") == "1.0.1"
    root = (repo / "Cargo.toml").read_text()
    assert mod.workspace_dep_version(root, "beta") == "1.0.1"
    assert mod.workspace_dep_version(root, "delta") == "0.2.0"
    assert mod.workspace_dep_version(root, "alpha") == "1.1.0"
    assert mod.read_version("cargo", (repo / "model_gateway/Cargo.toml").read_text(), "") == "2.0.0"
    assert mod.read_version("pyproject", (repo / "pkg/pyproject.toml").read_text(), "") == "0.2.0"
    assert (repo / "bindings/smg/version.py").read_text() == '__version__ = "2.0.0"\n'
    assert (
        repo / "chart/Chart.yaml"
    ).read_text() == 'apiVersion: v2\nname: smg\nversion: 2.0.0\nappVersion: "2.0.0"\n'
    assert (
        repo / "wf.yml"
    ).read_text() == "run-name: smg=${{ env.SMG_RELEASE_TAG }}\nenv:\n  SMG_RELEASE_TAG: v2.0.0\n"
    code, out = run(mod, repo)
    assert code == 0, out
    assert "All versions consistent." in out


def test_the_plan_lists_the_bump_quietly_and_changes_nothing(mod, repo):
    """Before a cut, main reads every changed crate as unbumped by design: the plan names what the
    cut's bump writes, without the verdicts and the issue count, and exits 0."""
    files = ("Cargo.toml", "crates/beta/Cargo.toml", "model_gateway/Cargo.toml", "chart/Chart.yaml")
    before = {path: (repo / path).read_text() for path in files}
    code, out = run(mod, repo, mode="plan", tag="v1.0.0")
    assert code == 0
    assert (
        "The bump the next cut writes against v1.0.0: the crates and packages by their conventional"
        " commits since it, the gateway and its mirrors by the checker's own step (the cut writes the"
        " VERSION it is given there: pass it as VERSION_OVERRIDE for those lines). Nothing is changed here"
        in out
    )
    assert "  patch beta v1.0.0 → v1.0.1 (crates/beta/Cargo.toml)" in out
    assert "       sync workspace Cargo.toml beta → v1.0.1" in out
    assert "  sync workspace Cargo.toml delta v0.1.0 → v0.2.0" in out
    assert "  major smg v1.0.0 → v2.0.0 (model_gateway/Cargo.toml)" in out
    assert "  minor pkg v0.1.0 → v0.2.0 (pkg/pyproject.toml)" in out
    assert "  sync chart v1.0.0 → v2.0.0 (chart/Chart.yaml)" in out
    assert out.rstrip().endswith(
        "9 component(s) and mirror(s) move (a crate's workspace pin follows it)."
        " Bumped already since v1.0.0, left as they are: alpha v1.1.0, delta v0.2.0 (new)."
    )
    for quiet in ("not bumped", "issue(s) found", "Checking workspace", "Summary:", "✓", "!"):
        assert quiet not in out, quiet
    assert {path: (repo / path).read_text() for path in files} == before
    # the cut's own version for the gateway and its mirrors: a candidate is allowed here, and only here; the gateway
    # is set to it whatever its conventional commits say (the fixture's are breaking: the checker alone says major)
    code, out = run(mod, repo, mode="plan", tag="v1.0.0", override="1.1.0-rc.1")
    assert code == 0
    assert "the gateway and its mirrors to 1.1.0-rc.1. Nothing is changed here" in out
    assert "  set smg v1.0.0 → v1.1.0-rc.1 (model_gateway/Cargo.toml)" in out
    assert "major smg" not in out
    assert "  sync chart v1.0.0 → v1.1.0-rc.1 (chart/Chart.yaml)" in out
    assert "  patch beta v1.0.0 → v1.0.1 (crates/beta/Cargo.toml)" in out
    assert "9 component(s) and mirror(s) move" in out
    with pytest.raises(mod.Fatal, match="X.Y.Z format"):
        run(mod, repo, override="1.1.0-rc.1")
    with pytest.raises(mod.Fatal, match="X.Y.Z or X.Y.Z-rc.N format"):
        run(mod, repo, mode="plan", override="1.1.0-rc1")
    assert {path: (repo / path).read_text() for path in files} == before
    # once the tree is consistent there is nothing to plan, still exit 0
    assert run(mod, repo, mode="fix")[0] == 0
    code, out = run(mod, repo, mode="plan", tag="v1.0.0")
    assert code == 0
    assert out.startswith(
        "Nothing to bump against v1.0.0: every crate and package changed since it is bumped already ("
    )
    assert "alpha v1.1.0" in out and "beta v1.0.1" in out and "smg v2.0.0" in out
    # a gateway bumped already (2.0.0 against v1.0.0) is still set to the cut's version, with its mirrors: the cut
    # writes that version unconditionally, so the plan never lists the gateway as left alone when it is given one
    code, out = run(mod, repo, mode="plan", tag="v1.0.0", override="2.1.0-rc.1")
    assert code == 0
    assert "  set smg v2.0.0 → v2.1.0-rc.1 (model_gateway/Cargo.toml)" in out
    assert "  sync chart v2.0.0 → v2.1.0-rc.1 (chart/Chart.yaml)" in out
    assert "smg v2.0.0" not in out.split("Bumped already")[-1]
    assert "6 component(s) and mirror(s) move" in out


def test_the_plan_reads_the_candidate_from_mains_development_version(mod, repo):
    """Between cuts main reads the next development version (1.1.0-dev once release/1.0 is cut): without
    VERSION_OVERRIDE the plan takes its first candidate as what the cut writes, and says where it read it."""
    assert mod.first_candidate("1.15.0-dev") == "1.15.0-rc.1"
    assert mod.first_candidate("1.15.0") is None and mod.first_candidate("1.15.0-rc.1") is None
    put(repo, "model_gateway/Cargo.toml", manifest("smg", "1.1.0-dev"))
    for path, text in (
        ("bindings/Cargo.toml", manifest("smg-binding", "1.1.0-dev")),
        ("bindings/pyproject.toml", '[project]\nname = "smg"\nversion = "1.1.0-dev"\n'),
        ("bindings/smg/version.py", '__version__ = "1.1.0-dev"\n'),
        (
            "chart/Chart.yaml",
            'apiVersion: v2\nname: smg\nversion: 1.1.0-dev\nappVersion: "1.1.0-dev"\n',
        ),
        (
            "wf.yml",
            "run-name: smg=${{ env.SMG_RELEASE_TAG }}\nenv:\n  SMG_RELEASE_TAG: v1.1.0-dev\n",
        ),
    ):
        put(repo, path, text)
    code, out = run(mod, repo, mode="plan", tag="v1.0.0")
    assert code == 0
    assert (
        "the gateway and its mirrors to 1.1.0-rc.1 (the first candidate of main's development version"
        " 1.1.0-dev; VERSION_OVERRIDE names another). Nothing is changed here" in out
    )
    assert "  set smg v1.1.0-dev → v1.1.0-rc.1 (model_gateway/Cargo.toml)" in out
    assert "  sync chart v1.1.0-dev → v1.1.0-rc.1 (chart/Chart.yaml)" in out
    assert "major smg" not in out and "smg v1.1.0-dev" not in out.split("Bumped already")[-1]
    # a version given still wins, and the plan does not say it was read from main
    code, out = run(mod, repo, mode="plan", tag="v1.0.0", override="1.2.0-rc.1")
    assert code == 0
    assert "the gateway and its mirrors to 1.2.0-rc.1. Nothing is changed here" in out
    assert "  set smg v1.1.0-dev → v1.2.0-rc.1 (model_gateway/Cargo.toml)" in out
    assert "development version" not in out
    # the check reads a development version as consistent: no release tag carries it, the mirrors carry the string
    code, out = run(mod, repo, mode="check", tag="v1.0.0")
    assert code == 1, "the fixture's other crates are still unbumped"
    assert "✓ smg (model_gateway/) — v1.0.0 → v1.1.0-dev" in out
    for mirror in ("binding", "pyproject", "version.py", "chart", "workflow"):
        assert f"✓ {mirror} (" in out, mirror
    assert "expected v1.1" not in out, "no mirror disagrees with the development version"


def test_the_plan_lists_a_gateway_already_at_the_cuts_version_among_the_bumped(mod, repo):
    """After a local ``make bump-version VERSION=<cut>`` the gateway carries the cut's version: the plan prints no
    ``set`` line for it and names it among the components bumped already, in both closing lines."""
    put(repo, "model_gateway/Cargo.toml", manifest("smg", "1.1.0-rc.1"))
    code, out = run(mod, repo, mode="plan", tag="v1.0.0", override="1.1.0-rc.1")
    assert code == 0
    assert "set smg" not in out and "major smg" not in out
    assert "  sync chart v1.0.0 → v1.1.0-rc.1 (chart/Chart.yaml)" in out
    assert out.rstrip().endswith(
        "8 component(s) and mirror(s) move (a crate's workspace pin follows it)."
        " Bumped already since v1.0.0, left as they are: smg v1.1.0-rc.1, alpha v1.1.0, delta v0.2.0 (new)."
    )
    # the whole tree at the cut's version (the crates bumped, the gateway and its mirrors set): nothing to bump,
    # the gateway named among the bumped
    assert run(mod, repo, mode="fix")[0] == 0
    put(repo, "model_gateway/Cargo.toml", manifest("smg", "2.1.0-rc.1"))
    put(repo, "bindings/Cargo.toml", manifest("smg-binding", "2.1.0-rc.1"))
    put(repo, "bindings/pyproject.toml", '[project]\nname = "smg"\nversion = "2.1.0-rc.1"\n')
    put(repo, "bindings/smg/version.py", '__version__ = "2.1.0-rc.1"\n')
    put(
        repo,
        "chart/Chart.yaml",
        'apiVersion: v2\nname: smg\nversion: 2.1.0-rc.1\nappVersion: "2.1.0-rc.1"\n',
    )
    put(
        repo,
        "wf.yml",
        "run-name: smg=${{ env.SMG_RELEASE_TAG }}\nenv:\n  SMG_RELEASE_TAG: v2.1.0-rc.1\n",
    )
    code, out = run(mod, repo, mode="plan", tag="v1.0.0", override="2.1.0-rc.1")
    assert code == 0
    assert out.rstrip() == (
        "Nothing to bump against v1.0.0: every crate and package changed since it is bumped already"
        " (smg v2.1.0-rc.1, alpha v1.1.0, beta v1.0.1, delta v0.2.0 (new), pkg v0.2.0); the mirrors agree."
    )


def test_the_plan_compares_against_the_highest_final_tag_by_default(mod, repo):
    put(repo, "crates/beta/src/lib.rs", "// beta 3\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "chore: a candidate", date="2026-06-01T00:00:00Z")
    git(
        repo, "tag", "v1.1.0-rc.1"
    )  # the newest tag by date, a candidate: not what a cut compares against
    code, out = run(mod, repo, mode="plan")
    assert code == 0 and "against v1.0.0" in out
    assert "against tag: v1.1.0-rc.1" in run(mod, repo)[1], (
        "--check keeps the newest tag as its default"
    )
    git(
        repo, "tag", "v1.0.9", "v1.0.0"
    )  # a patch of an older line, lower than v1.1.0-rc.1 by date and by value
    assert "against v1.0.9" in run(mod, repo, mode="plan")[1]


def test_the_override_applies_to_the_gateway_only(mod, repo):
    code, out = run(mod, repo, mode="prompt", override="3.5.0")
    assert "Version override: 3.5.0" in out
    assert "  major smg v1.0.0 → v3.5.0 (model_gateway/Cargo.toml)" in out
    assert "  patch beta v1.0.0 → v1.0.1 (crates/beta/Cargo.toml)" in out
    assert "  ! chart (chart/Chart.yaml) — v1.0.0, expected v3.5.0" in out


def test_a_bad_tag_or_override_is_refused(mod, repo):
    with pytest.raises(mod.Fatal, match="does not exist"):
        run(mod, repo, tag="v9.9.9")
    with pytest.raises(mod.Fatal, match="X.Y.Z"):
        run(mod, repo, override="3.5")


def test_the_newest_tag_is_the_default(mod, repo):
    put(repo, "crates/beta/src/lib.rs", "// beta 3\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "chore: tagged state", date="2026-06-01T00:00:00Z")
    git(repo, "tag", "v1.1.0")
    code, out = run(mod, repo)
    assert "against tag: v1.1.0" in out
    assert (
        "Summary: 0 crate(s) with changes, 6 unchanged; 0 Python package(s) with changes, 2 unchanged"
        in out
    )


# --- the versions release tags carry ----------------------------------------------


def test_version_keys_order_candidates_below_their_finals(mod):
    versions = ("1.9.0", "1.10.0-rc.1", "1.10.0-rc.2", "1.10.0", "1.10.1")
    keys = [mod.version_key(v) for v in versions]
    assert keys == sorted(keys)
    assert mod.version_key("1.2") is None and mod.version_key("v1.2.3") is None


def test_texts_at_reads_every_ref_and_path_in_one_call(mod, repo):
    wanted = [
        ("v1.0.0", "crates/alpha/Cargo.toml"),
        ("v1.0.0", "crates/delta/Cargo.toml"),  # not at the tag
        ("HEAD", "crates/delta/Cargo.toml"),
    ]
    texts = mod.Repo(repo).texts_at(wanted)
    assert set(texts) == {wanted[0], wanted[2]}
    assert mod.read_version("cargo", texts[wanted[0]], "") == "1.0.0"
    assert mod.read_version("cargo", texts[wanted[2]], "") == "0.2.0"


def side_line(repo):
    """Another release line's tag, off v1.0.0: beta patched to 1.0.1 there, alpha at the 1.1.0
    main gave it (with v1.0.0's code), gamma at the 1.0.1 main gave it with the very code main
    carries (the same fix picked), and a tag of another shape that published nothing."""
    put(repo, "crates/gamma/src/lib.rs", "// gamma fixed\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "fix(gamma): real")
    git(repo, "switch", "-q", "-c", "release/1.0", "v1.0.0")
    put(repo, "crates/beta/src/lib.rs", "// beta patched on the line\n")
    put(repo, "crates/beta/Cargo.toml", manifest("beta", "1.0.1"))
    put(repo, "crates/alpha/Cargo.toml", manifest("alpha", "1.1.0"))
    put(repo, "crates/gamma/src/lib.rs", "// gamma fixed\n")
    put(repo, "crates/gamma/Cargo.toml", manifest("gamma", "1.0.1"))
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "chore(release): bump versions for v1.0.9")
    git(repo, "tag", "v1.0.9")
    put(repo, "crates/alpha/Cargo.toml", manifest("alpha", "1.2.0"))  # never published
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "chore: a trial tag")
    git(repo, "tag", "v1.0.9-qa")
    git(repo, "switch", "-q", "main")


def test_a_version_any_release_tag_carries_is_not_a_bump(mod, repo):
    """Two release lines share the crate versions and a registry takes a version once. Against
    v1.0.0: beta, unbumped there, is proposed 1.0.2, past the 1.0.1 the other line's tag carries;
    alpha, raised to 1.1.0 since v1.0.0 but at the version that tag carries with other code,
    reads unbumped too and its minor bump lands on 1.2.0 (the 1.2.0 of a tag of another shape
    counts for nothing); gamma, changed since v1.0.0 but at the version that tag carries with
    this very code, needs no bump, and its workspace pin, still at 1.0.0, is the one thing it
    needs (the comparison tag's path reads the pin of every changed crate; this path must too).
    Against the other line's tag, beta's 1.0.0 sits below its 1.0.1, a difference the equality
    rule read as a bump: it is v1.0.0's version, so unbumped."""
    side_line(repo)
    assert mod.Repo(repo).release_tags() == ["v1.0.0", "v1.0.9"]
    code, out = run(mod, repo, tag="v1.0.0")
    assert code == 1
    assert (
        "  ! beta (crates/beta/) — 1 file(s) changed but version not bumped (v1.0.0) [patch]" in out
    )
    assert (
        "  ! alpha (crates/alpha/) — 1 file(s) changed but version not bumped (v1.1.0, v1.0.9's) [minor]"
        in out
    )
    assert "gamma (crates/gamma/) — v1.0.1 is v1.0.9's, with this code" in out
    assert "    ✗ workspace Cargo.toml has gamma v1.0.0, expected v1.0.1" in out
    assert "not bumped (v1.0.1" not in out, "the tag's copy of gamma is this very code: no bump"
    assert "Summary: 5 crate(s) with changes, 1 unchanged" in out, (
        "gamma's pin counts it as changed"
    )
    code, out = run(mod, repo, tag="v1.0.9")
    assert (
        "  ! beta (crates/beta/) — 1 file(s) changed but version not bumped (v1.0.0, v1.0.0's) [patch]"
        in out
    )
    code, out = run(mod, repo, tag="v1.0.0", mode="prompt", answer="n")
    assert "  patch beta v1.0.0 → v1.0.2 (crates/beta/Cargo.toml; v1.0.1 is v1.0.9's)" in out
    assert "       sync workspace Cargo.toml beta → v1.0.2" in out
    assert "  minor alpha v1.1.0 → v1.2.0 (crates/alpha/Cargo.toml)" in out
    assert "  sync workspace Cargo.toml gamma v1.0.0 → v1.0.1" in out
    code, out = run(mod, repo, tag="v1.0.0", mode="fix")
    assert code == 0, out
    assert mod.read_version("cargo", (repo / "crates/beta/Cargo.toml").read_text(), "") == "1.0.2"
    assert mod.read_version("cargo", (repo / "crates/alpha/Cargo.toml").read_text(), "") == "1.2.0"
    root = (repo / "Cargo.toml").read_text()
    assert mod.workspace_dep_version(root, "beta") == "1.0.2"
    assert mod.workspace_dep_version(root, "alpha") == "1.2.0"
    assert mod.workspace_dep_version(root, "gamma") == "1.0.1"
    code, out = run(mod, repo, tag="v1.0.0")
    assert code == 0, out
    assert "All versions consistent." in out


def test_a_component_whose_files_went_is_named_loudly(mod, repo):
    """A crate deleted since the tag but still in the table: its files read as changed and its
    version file is gone; the check says which entry to drop rather than printing a green line."""
    git(repo, "rm", "-q", "-r", "crates/beta")
    git(repo, "commit", "-q", "-m", "refactor: beta goes")
    with pytest.raises(mod.Fatal, match="crates/beta/Cargo.toml is gone: drop beta from the table"):
        run(mod, repo, tag="v1.0.0")


# --- a component's symlinks -----------------------------------------------------


def test_what_a_components_symlinks_point_at_is_its_content(mod, tmp_path):
    """A package that ships another directory's files through a link (the proto package ships
    the .proto files that way) ships their changes: they count as the package's, with the
    level of the commits that made them. A link that points at a link, a target directory
    holding one, or a link whose path runs through a directory link is followed to the end of
    the chain; a directory link crossed on the way counts, what lies behind it beyond the path
    does not; a link crossed twice on one path is no loop, a loop ends at the hop limit. A link
    into the package itself or out of the repository adds nothing."""
    import os

    root = tmp_path / "linked"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    put(root, "Cargo.toml", root_manifest({}))
    put(root, "model_gateway/Cargo.toml", manifest("smg", "1.0.0"))
    put(root, "model_gateway/src/main.rs", "fn main() {}\n")
    put(root, "pkg/pyproject.toml", '[project]\nname = "pkg"\nversion = "0.1.0"\n')
    put(root, "pkg/pkg.py", "x = 1\n")
    put(root, "shared/schema.proto", 'syntax = "proto3";\n')
    put(root, "LICENSE", "license\n")
    os.symlink("../shared", root / "pkg" / "schema")
    os.symlink("pkg.py", root / "pkg" / "alias.py")
    os.symlink("../../elsewhere", root / "pkg" / "outside")
    os.symlink("/etc/hostname", root / "pkg" / "absolute")
    put(root, "real/x", "x\n")
    put(root, "real/y", "y\n")
    put(root, "real/w", "w\n")  # in the linked directory, shipped by nothing
    os.symlink("../real/x", root / "shared" / "chain")  # a link inside the target directory
    os.symlink("real", root / "hop")  # a directory link ...
    os.symlink("../hop/y", root / "pkg" / "p")  # ... with a path that runs through it
    os.symlink("../hop/y", root / "real" / "z")  # a link that crosses the directory link again
    os.symlink("../hop/z", root / "pkg" / "q")  # ... reached through it twice
    os.symlink("loop_b", root / "loop_a")
    os.symlink("loop_a", root / "loop_b")
    os.symlink("../loop_a", root / "pkg" / "l")  # a genuine loop
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "chore: initial")
    git(root, "tag", "v1.0.0")
    repo = mod.Repo(root)
    assert repo.resolve_path("pkg/../hop/y") == ("real/y", ["hop"])
    assert repo.resolve_path("pkg/../hop/z") == ("real/y", ["hop", "real/z", "hop"])
    assert repo.resolve_path("pkg/../../elsewhere") == (None, [])
    looped, crossed = repo.resolve_path("pkg/../loop_a")
    assert looped is None and len(crossed) == 40 and set(crossed) == {"loop_a", "loop_b"}
    chain = ["loop_a", "loop_b", "hop", "real/y", "real/z", "shared", "real/x"]
    assert repo.symlink_targets("pkg") == chain
    assert repo.symlink_targets("pkg/") == chain
    assert mod.component_paths(repo, mod.package("pkg", "pkg", "pkg/pyproject.toml")) == [
        "pkg/",
        *chain,
    ]
    assert repo.symlink_targets("model_gateway") == []

    crates = (mod.crate("smg", "model_gateway"),)
    packages = (mod.package("pkg", "pkg", "pkg/pyproject.toml"),)
    stream = io.StringIO()
    assert (
        mod.run(
            root,
            "v1.0.0",
            "check",
            None,
            mod.Out(stream),
            crates=crates,
            packages=packages,
            mirrors=(),
        )
        == 0
    )
    assert "pkg" not in stream.getvalue(), "nothing changed since the tag"

    put(root, "shared/schema.proto", 'syntax = "proto3";\nmessage M {}\n')
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "feat(shared): a message")
    stream = io.StringIO()
    code = mod.run(
        root, "v1.0.0", "check", None, mod.Out(stream), crates=crates, packages=packages, mirrors=()
    )
    out = stream.getvalue()
    assert code == 1
    assert "  ! pkg (pkg/) — 1 file(s) changed but version not bumped (v0.1.0) [minor]" in out
    assert "1 Python package(s) with changes, 0 unchanged" in out

    # a change at the end of a chain of links is the package's as much as a direct target's
    put(root, "real/x", "x2\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "fix(real): x")
    stream = io.StringIO()
    code = mod.run(
        root, "v1.0.0", "check", None, mod.Out(stream), crates=crates, packages=packages, mirrors=()
    )
    out = stream.getvalue()
    assert code == 1
    assert "  ! pkg (pkg/) — 2 file(s) changed but version not bumped (v0.1.0) [minor]" in out

    # and so is a change behind a link whose path runs through a directory link
    put(root, "real/y", "y2\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "fix(real): y")
    stream = io.StringIO()
    mod.run(
        root, "v1.0.0", "check", None, mod.Out(stream), crates=crates, packages=packages, mirrors=()
    )
    assert "  ! pkg (pkg/) — 3 file(s) changed but version not bumped (v0.1.0) [minor]" in (
        stream.getvalue()
    )

    # but not a change elsewhere in the directory behind that link: the package ships one file of it
    put(root, "real/w", "w2\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "fix(real): w")
    stream = io.StringIO()
    mod.run(
        root, "v1.0.0", "check", None, mod.Out(stream), crates=crates, packages=packages, mirrors=()
    )
    assert "  ! pkg (pkg/) — 3 file(s) changed but version not bumped (v0.1.0) [minor]" in (
        stream.getvalue()
    )
