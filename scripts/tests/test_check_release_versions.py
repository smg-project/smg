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
        ("chart", "chart/Chart.yaml"),
        ("workflow", "wf.yml"),
    ):
        assert f"  ! {label} ({file}) — v1.0.0, expected v2.0.0" in out
    assert (
        "Summary: 4 crate(s) with changes, 2 unchanged; 1 Python package(s) with changes, 1 unchanged"
        in out
    )
    assert "8 issue(s) found." in out
    assert "Proposed fixes" not in out, "--check never proposes"


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
    assert "The bump the next cut writes against v1.0.0 (nothing is changed here" in out
    assert "  patch beta v1.0.0 → v1.0.1 (crates/beta/Cargo.toml)" in out
    assert "       sync workspace Cargo.toml beta → v1.0.1" in out
    assert "  sync workspace Cargo.toml delta v0.1.0 → v0.2.0" in out
    assert "  major smg v1.0.0 → v2.0.0 (model_gateway/Cargo.toml)" in out
    assert "  minor pkg v0.1.0 → v0.2.0 (pkg/pyproject.toml)" in out
    assert "  sync chart v1.0.0 → v2.0.0 (chart/Chart.yaml)" in out
    assert out.rstrip().endswith(
        "8 version line(s) move. Bumped already since v1.0.0, left as they are: alpha v1.1.0, delta v0.2.0 (new)."
    )
    for quiet in ("not bumped", "issue(s) found", "Checking workspace", "Summary:", "✓", "!"):
        assert quiet not in out, quiet
    assert {path: (repo / path).read_text() for path in files} == before
    # once the tree is consistent there is nothing to plan, still exit 0
    assert run(mod, repo, mode="fix")[0] == 0
    code, out = run(mod, repo, mode="plan", tag="v1.0.0")
    assert code == 0
    assert out.startswith(
        "Nothing to bump against v1.0.0: every crate and package changed since it is bumped already ("
    )
    assert "alpha v1.1.0" in out and "beta v1.0.1" in out and "smg v2.0.0" in out


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
