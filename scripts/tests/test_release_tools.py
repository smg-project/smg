"""Tests for scripts/release/release.sh and scripts/release/sync_versions.py.

Every test runs the tool on a throwaway repository built here: an origin (a bare repository)
and a clone that carries the Makefile, the version checker and the release scripts of this
tree, the version files the checker reads, one crate and a few commits past a v1.13.0 tag.
``gh`` and ``cargo`` are stubs on PATH that record their arguments. Nothing here touches the
real repository or the network.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
TOOL = "scripts/release/release.sh"
SYNC = "scripts/release/sync_versions.py"
VERSION_FILES = (
    "model_gateway/Cargo.toml",
    "bindings/golang/Cargo.toml",
    "bindings/python/Cargo.toml",
    "bindings/python/pyproject.toml",
    "bindings/python/src/smg/version.py",
    "deploy/helm/smg/Chart.yaml",
    ".github/workflows/release-engine-docker.yml",
)
MAKE_TARGETS = {
    "release-cut": "cut",
    "release-bump": "bump",
    "release-pick": "pick",
    "release-pick-pr": "pick-pr",
    "release-status": "status",
    "release-tag": "tag",
    "release-notes": "notes",
    "release-bump-main": "bump-main",
    "release-publish-plan": "publish-plan",
    "release-check": "check",
}

GH_STUB = r"""#!/usr/bin/env bash
printf 'gh %s\n' "$*" >> "$GH_STUB_LOG"
case "$1 $2" in
  "pr create")
    for ((i = 1; i <= $#; i++)); do
      if [ "${!i}" = "--body-file" ]; then j=$((i + 1)); cp "${!j}" "$GH_STUB_LOG.body"; fi
    done
    echo "https://example.invalid/pull/1" ;;
  "pr list") echo "#7  fix(router): y  pick/1.14-0000000" ;;
  "repo view") echo "example/repo" ;;
  "api --paginate") printf '%b\n' "${GH_STUB_CHECKS:-success a\nsuccess b}" ;;
  "api repos/example/repo/actions/workflows/pr-test-rust.yml/runs?head_sha="*)
    case "${GH_STUB_RUNS-}" in
      none) ;;
      "") echo "success https://example.invalid/runs/1" ;;
      *) printf '%b\n' "$GH_STUB_RUNS" ;;
    esac ;;
esac
"""
CARGO_STUB = r"""#!/usr/bin/env bash
printf 'cargo %s\n' "$*" >> "$CARGO_STUB_LOG"
"""
DOCKER_NOTES_STUB = """#!/usr/bin/env bash
echo "### Docker images"
echo ""
echo "- stub images for $1"
"""


CURL_STUB = r"""#!/usr/bin/env bash
# PyPI as the release-grpc selection reads it: CURL_STUB lists "<name>/<version>=complete|wheel";
# anything else is a 404.
url=${*: -1}; key=${url#https://pypi.org/pypi/}; key=${key%/json}
case " $CURL_STUB " in
  *" $key=complete "*) printf '{"urls": [{"packagetype": "bdist_wheel"}, {"packagetype": "sdist"}]}' ;;
  *" $key=wheel "*) printf '{"urls": [{"packagetype": "bdist_wheel"}]}' ;;
  *) exit 22 ;;
esac
"""

# The lines of the engine-image workflow the tooling reads: the release line the version checker
# bumps and the release-base table the publishing plan turns into dispatches.
ENGINE_WORKFLOW = """env:
  SMG_RELEASE_TAG: v1.13.0
jobs:
  plan:
    steps:
      - run: |
          engines='{
            "vllm": {"release_bases": ["vllm/vllm-openai:v0.31.0", "vllm/vllm-openai:v0.27.1"]},
            "sglang": {"release_bases": ["lmsysorg/sglang:v0.5.21"]},
            "tokenspeed": {"release_bases": [""]}
          }'
"""


def versioned(name: str, version: str) -> str:
    return f'[package]\nname = "{name}"\nversion = "{version}"\n'


class Sandbox:
    """An origin, a clone with the tooling, the stubs, and the helpers the tests share."""

    def __init__(self, base: Path) -> None:
        self.base = base
        self.origin = base / "origin.git"
        self.work = base / "work"
        self.bin = base / "bin"
        self.gh_log = base / "gh.log"
        self.cargo_log = base / "cargo.log"
        self.bin.mkdir()
        for name, text in (("gh", GH_STUB), ("cargo", CARGO_STUB)):
            stub = self.bin / name
            stub.write_text(text)
            stub.chmod(0o755)
        # the maintainer's git configuration the tool must not depend on: push.default=upstream
        # sends `git push origin topic` of a tracking branch to its upstream
        (base / ".gitconfig").write_text(
            "[push]\n\tdefault = upstream\n[init]\n\tdefaultBranch = main\n"
            "[advice]\n\tdetachedHead = false\n"
        )
        self.git("init", "--quiet", "--bare", str(self.origin), cwd=base)
        self.git("init", "--quiet", "-b", "main", str(self.work), cwd=base)
        self.git("remote", "add", "origin", str(self.origin))
        self.seed()

    # --- running things -------------------------------------------------------

    def env(self, **extra: str) -> dict[str, str]:
        env = {
            "PATH": os.pathsep.join(
                [str(self.bin), str(Path(sys.executable).parent), "/usr/bin", "/bin"]
            ),
            "HOME": str(self.base),
            "GIT_AUTHOR_NAME": "Release Owner",
            "GIT_AUTHOR_EMAIL": "owner@example.invalid",
            "GIT_COMMITTER_NAME": "Release Owner",
            "GIT_COMMITTER_EMAIL": "owner@example.invalid",
            "GIT_EDITOR": "true",
            "GH_STUB_LOG": str(self.gh_log),
            "CARGO_STUB_LOG": str(self.cargo_log),
            "PYTHONDONTWRITEBYTECODE": "1",
            "TMPDIR": str(self.base),
        }
        env.update(extra)
        return env

    def git(self, *args: str, cwd: Path | None = None, check: bool = True) -> str:
        done = subprocess.run(
            ["git", *args],
            cwd=cwd or self.work,
            env=self.env(),
            check=check,
            capture_output=True,
            text=True,
        )
        return done.stdout.strip()

    def tool(self, *args: str, check: bool = True, **env: str) -> subprocess.CompletedProcess[str]:
        done = subprocess.run(
            [str(self.work / TOOL), *args],
            cwd=self.work,
            env=self.env(**env),
            capture_output=True,
            text=True,
        )
        if check and done.returncode != 0:
            raise AssertionError(
                f"release.sh {' '.join(args)} failed ({done.returncode}):\n"
                f"{done.stdout}\n{done.stderr}"
            )
        return done

    # --- the tree -----------------------------------------------------------------

    def put(self, path: str, text: str) -> None:
        file = self.work / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text)

    def read(self, path: str, ref: str | None = None) -> str:
        if ref is None:
            return (self.work / path).read_text()
        return self.git("show", f"{ref}:{path}")

    def version_of(self, path: str, ref: str | None = None) -> str:
        for line in self.read(path, ref).splitlines():
            for prefix in ('version = "', '__version__ = "', "version: ", "  SMG_RELEASE_TAG: v"):
                if line.startswith(prefix):
                    return line[len(prefix) :].strip().strip('"')
        raise AssertionError(f"no version line in {path}")

    def commit_main(self, subject: str, path: str, text: str) -> str:
        """A commit on main, pushed; the clone is left on main."""
        self.git("switch", "--quiet", "main")
        self.git("pull", "--quiet", "--ff-only", "origin", "main")
        self.put(path, text)
        self.git("add", "-A")
        self.git("commit", "--quiet", "-m", subject)
        self.git("push", "--quiet", "origin", "main")
        return self.git("rev-parse", "HEAD")

    def seed(self) -> None:
        """The v1.13.0 tree, tagged and pushed, then two commits of development on main."""
        self.put(
            "Cargo.toml",
            '[workspace]\nmembers = ["model_gateway", "crates/tool_parser"]\n\n'
            '[workspace.dependencies]\ntool-parser = { version = "0.5.0", path = "crates/tool_parser" }\n',
        )
        self.put("model_gateway/Cargo.toml", versioned("smg", "1.13.0"))
        self.put("model_gateway/src/main.rs", 'fn main() {\n    println!("one");\n}\n')
        self.put("crates/tool_parser/Cargo.toml", versioned("tool-parser", "0.5.0"))
        self.put("crates/tool_parser/src/lib.rs", "pub fn parse() {}\n")
        self.put("bindings/golang/Cargo.toml", versioned("smg-golang", "1.13.0"))
        self.put("bindings/python/Cargo.toml", versioned("smg-python", "1.13.0"))
        self.put("bindings/python/pyproject.toml", '[project]\nname = "smg"\nversion = "1.13.0"\n')
        self.put("bindings/python/src/smg/version.py", '__version__ = "1.13.0"\n')
        self.put(
            "deploy/helm/smg/Chart.yaml",
            'apiVersion: v2\nname: smg\nversion: 1.13.0\nappVersion: "1.13.0"\n',
        )
        self.put(".github/workflows/release-engine-docker.yml", ENGINE_WORKFLOW)
        shutil.copy(ROOT / "Makefile", self.work / "Makefile")
        (self.work / "scripts" / "release").mkdir(parents=True)
        shutil.copy(ROOT / "scripts" / "check_release_versions.py", self.work / "scripts")
        for name in ("release.sh", "sync_versions.py"):
            shutil.copy(ROOT / "scripts" / "release" / name, self.work / "scripts" / "release")
        self.put("scripts/release_docker_notes.sh", DOCKER_NOTES_STUB)
        (self.work / "scripts" / "release_docker_notes.sh").chmod(0o755)
        self.git("add", "-A")
        self.git("commit", "--quiet", "-m", "chore: the v1.13.0 tree")
        self.git("tag", "v1.13.0")
        self.git("push", "--quiet", "origin", "main", "--tags")
        self.commit_main(
            "feat(tool_parser): more",
            "crates/tool_parser/src/lib.rs",
            "pub fn parse() {}\npub fn more() {}\n",
        )
        self.commit_main(
            "fix(router): two",
            "model_gateway/src/main.rs",
            'fn main() {\n    println!("two");\n}\n',
        )

    # --- the release branch on the origin ----------------------------------------------

    def cut(self, version: str = "1.14.0-rc.1") -> None:
        self.tool("cut", version, PUSH="1")

    def land_on_release(self, release: str = "1.14") -> None:
        """The pull request of the checked-out topic branch 'merges': origin's branch moves to it."""
        self.git("push", "--quiet", "origin", f"HEAD:refs/heads/release/{release}")

    def origin_ref(self, ref: str) -> str | None:
        shown = self.git("rev-parse", "--verify", "--quiet", ref, cwd=self.origin, check=False)
        return shown or None

    def gh_calls(self) -> list[str]:
        return self.gh_log.read_text().splitlines() if self.gh_log.exists() else []

    def pr_body(self) -> str:
        return Path(f"{self.gh_log}.body").read_text()


@pytest.fixture
def repo(tmp_path: Path) -> Sandbox:
    return Sandbox(tmp_path)


# --- cut --------------------------------------------------------------------------


def test_cut_creates_the_branch_and_the_bump_commit(repo: Sandbox) -> None:
    done = repo.tool("cut", "1.14.0-rc.1")
    assert repo.git("rev-parse", "--abbrev-ref", "HEAD") == "release/1.14"
    for path in VERSION_FILES:
        assert repo.version_of(path) == "1.14.0-rc.1", path
    assert (
        repo.version_of("crates/tool_parser/Cargo.toml") == "0.6.0"
    )  # a feat since v1.13.0: minor
    assert 'tool-parser = { version = "0.6.0"' in repo.read("Cargo.toml")
    message = repo.git("log", "-1", "--format=%B")
    assert message.startswith("chore(release): bump versions for v1.14.0-rc.1\n")
    assert "Signed-off-by: Release Owner <owner@example.invalid>" in message
    assert (
        repo.git("log", "-1", "--format=%an <%ae> %cn")
        == "Release Owner <owner@example.invalid> Release Owner"
    )
    assert repo.git("rev-parse", "HEAD~1") == repo.origin_ref("refs/heads/main")
    assert repo.origin_ref("refs/heads/release/1.14") is None, (
        "the cut is not pushed without PUSH=1"
    )
    assert "git push origin refs/heads/release/1.14:refs/heads/release/1.14" in done.stdout
    assert "nothing is published" in done.stdout


def test_cut_pushes_the_branch_only_on_request(repo: Sandbox) -> None:
    repo.tool("cut", "1.14.0-rc.1", PUSH="1")
    assert repo.origin_ref("refs/heads/release/1.14") == repo.git("rev-parse", "HEAD")
    assert repo.origin_ref("refs/heads/main") == repo.git("rev-parse", "HEAD~1"), (
        "main is untouched"
    )


def test_cut_starts_from_the_named_commit(repo: Sandbox) -> None:
    older = repo.git("rev-parse", "origin/main~1")
    repo.tool("cut", "1.14.0-rc.1", older)
    assert repo.git("rev-parse", "HEAD~1") == older
    # the gateway did not change between v1.13.0 and that commit, its version still moves
    assert repo.version_of("model_gateway/Cargo.toml") == "1.14.0-rc.1"


def test_cut_refusals(repo: Sandbox) -> None:
    assert "not a version" in repo.tool("cut", "1.14", check=False).stderr
    assert "not a version" in repo.tool("cut", "1.14.0-beta.1", check=False).stderr
    assert "unknown commit" in repo.tool("cut", "1.14.0-rc.1", "no-such-ref", check=False).stderr
    repo.git("switch", "--quiet", "-c", "side", "origin/main")
    repo.put("model_gateway/src/main.rs", "fn main() {}\n")
    repo.git("commit", "--quiet", "-am", "feat: side")
    assert "not on origin/main" in repo.tool("cut", "1.14.0-rc.1", "side", check=False).stderr
    repo.put("model_gateway/src/main.rs", "fn main() { /* dirty */ }\n")
    assert "uncommitted changes" in repo.tool("cut", "1.14.0-rc.1", check=False).stderr
    repo.git("checkout", "--quiet", "--", "model_gateway/src/main.rs")
    repo.cut()
    done = repo.tool("cut", "1.14.0-rc.2", check=False)
    assert done.returncode == 1 and "exists" in done.stderr and "bump 1.14.0-rc.2" in done.stderr


def test_cut_refreshes_the_lock_file_through_cargo(repo: Sandbox) -> None:
    repo.commit_main("chore: a lock file", "Cargo.lock", "# lock\n")
    repo.tool("cut", "1.14.0-rc.1")
    assert repo.cargo_log.read_text().splitlines() == ["cargo update --workspace --offline"]
    (repo.bin / "cargo").unlink()
    repo.git("switch", "--quiet", "main")
    repo.git("branch", "--quiet", "-D", "release/1.14")
    done = repo.tool("cut", "1.14.0-rc.1", check=False)
    assert done.returncode == 1 and "Cargo.lock needs a refresh" in done.stderr


# --- bump ---------------------------------------------------------------------------


def test_bump_opens_a_pull_request_against_the_release_branch(repo: Sandbox) -> None:
    repo.cut()
    branch_head = repo.origin_ref("refs/heads/release/1.14")
    repo.tool("bump", "1.14.0-rc.2")
    assert repo.git("rev-parse", "--abbrev-ref", "HEAD") == "chore/release-1.14.0-rc.2"
    for path in VERSION_FILES:
        assert repo.version_of(path) == "1.14.0-rc.2", path
    assert repo.git("log", "-1", "--format=%s") == "chore(release): bump versions for v1.14.0-rc.2"
    assert repo.origin_ref("refs/heads/chore/release-1.14.0-rc.2") == repo.git("rev-parse", "HEAD")
    assert repo.origin_ref("refs/heads/release/1.14") == branch_head, (
        "the release branch moves by merges only"
    )
    assert repo.gh_calls()[-1] == (
        "gh pr create --base release/1.14 --head chore/release-1.14.0-rc.2"
        " --title chore(release): bump versions for v1.14.0-rc.2 --body-file "
        + repo.gh_calls()[-1].split("--body-file ")[1]
    )
    body = repo.pr_body()
    assert "## Release gate" in body and "nothing is published by the merge" in body
    for family in ("DeepSeek", "GLM", "Kimi", "MiniMax"):
        assert f"{family} provider verifier: <record link>" in body


def test_bump_without_push_prints_the_commands(repo: Sandbox) -> None:
    repo.cut()
    done = repo.tool("bump", "1.14.0", PUSH="0")
    assert "gh pr create --base release/1.14 --head chore/release-1.14.0" in done.stdout
    assert (
        "--title chore\\(release\\):\\ bump\\ versions\\ for\\ v1.14.0 --body-file " in done.stdout
    ), "the printed title is shell-quoted (a bump-main title carries an apostrophe)"
    assert repo.origin_ref("refs/heads/chore/release-1.14.0") is None
    assert repo.gh_calls() == []
    assert (
        "the four model-provider verifiers ran against this release line's candidate build"
        in repo.read(done.stdout.split("--body-file ")[1].split()[0])
    )


def test_bump_refusals(repo: Sandbox) -> None:
    assert "no release/1.14 on origin" in repo.tool("bump", "1.14.0-rc.2", check=False).stderr
    repo.cut()
    assert "already reads 1.14.0-rc.1" in repo.tool("bump", "1.14.0-rc.1", check=False).stderr


# --- pick -------------------------------------------------------------------------------


def test_pick_cherry_picks_from_main_and_opens_the_pull_request(repo: Sandbox) -> None:
    repo.cut()
    branch_head = repo.origin_ref("refs/heads/release/1.14")
    three = repo.commit_main(
        "fix(router): three (#123)",
        "model_gateway/src/main.rs",
        'fn main() {\n    println!("three");\n}\n',
    )
    repo.tool("pick", "1.14", three)
    topic = f"pick/1.14-{three[:7]}"
    assert repo.git("rev-parse", "--abbrev-ref", "HEAD") == topic
    message = repo.git("log", "-1", "--format=%B", "HEAD~1")
    assert message.startswith("fix(router): three (#123)\n")
    assert f"(cherry picked from commit {three})" in message
    assert "Signed-off-by: Release Owner <owner@example.invalid>" in message
    assert repo.git("rev-parse", "HEAD~2") == branch_head
    # The branch records its picks in a file the squash merge keeps (the repository merges under the title alone).
    assert repo.git("log", "-1", "--format=%s") == "chore(release): record 1 pick into release/1.14"
    assert repo.read("scripts/release/picks.txt") == f"{three} fix(router): three (#123)\n"
    assert repo.origin_ref(f"refs/heads/{topic}") == repo.git("rev-parse", "HEAD")
    assert repo.origin_ref("refs/heads/release/1.14") == branch_head
    assert repo.gh_calls()[-1].startswith(
        f"gh pr create --base release/1.14 --head {topic} --title fix(router): three (#123) --body-file"
    )
    assert f"(cherry picked from commit {three})" in repo.pr_body()


def test_pick_takes_several_commits_oldest_first(repo: Sandbox) -> None:
    repo.cut()
    first = repo.commit_main("fix(a): first", "a.txt", "a\n")
    second = repo.commit_main("fix(b): second", "b.txt", "b\n")
    repo.tool("pick", "1.14", f"{second},{first}")
    assert repo.git("log", "--format=%s", "-3") == (
        "chore(release): record 2 picks into release/1.14\nfix(b): second\nfix(a): first"
    )
    assert (
        repo.read("scripts/release/picks.txt")
        == f"{first} fix(a): first\n{second} fix(b): second\n"
    )
    assert repo.git("rev-parse", "--abbrev-ref", "HEAD") == f"pick/1.14-{first[:7]}-{second[:7]}"
    assert (
        repo.gh_calls()[-1]
        .split(" --title ")[1]
        .startswith("chore(release): pick 2 commits from main into release/1.14")
    )


def test_pick_refusals(repo: Sandbox) -> None:
    repo.cut()
    assert "unknown commit" in repo.tool("pick", "1.14", "0000000", check=False).stderr
    bump = repo.origin_ref("refs/heads/release/1.14")
    assert "not on origin/main" in repo.tool("pick", "1.14", bump, check=False).stderr
    already = repo.git("rev-parse", "origin/main")  # the cut point is on the branch
    assert "already on" in repo.tool("pick", "1.14", already, check=False).stderr
    assert "not a release" in repo.tool("pick", "x.y", already, check=False).stderr


def test_pick_conflict_stops_with_the_steps_and_pick_pr_finishes(repo: Sandbox) -> None:
    repo.cut()
    repo.git("switch", "--quiet", "release/1.14")
    repo.put("model_gateway/src/main.rs", 'fn main() {\n    println!("branch only");\n}\n')
    repo.git("commit", "--quiet", "-am", "fix(router): branch only")
    repo.land_on_release()
    four = repo.commit_main(
        "fix(router): four (#125)",
        "model_gateway/src/main.rs",
        'fn main() {\n    println!("four");\n}\n',
    )
    done = repo.tool("pick", "1.14", four, check=False)
    assert done.returncode == 1
    assert f"the pick of {four[:7]} (fix(router): four (#125)) conflicts on:" in done.stdout
    assert "    model_gateway/src/main.rs" in done.stdout
    assert "git cherry-pick --continue" in done.stdout
    assert "make release-pick-pr RELEASE=1.14" in done.stdout
    assert repo.gh_calls() == [], "no pull request for an unresolved pick"
    topic = f"pick/1.14-{four[:7]}"
    assert repo.git("rev-parse", "--abbrev-ref", "HEAD") == topic
    assert "still in progress" in repo.tool("pick-pr", "1.14", check=False).stderr
    repo.put("model_gateway/src/main.rs", 'fn main() {\n    println!("four");\n}\n')
    repo.git("add", "model_gateway/src/main.rs")
    repo.git("cherry-pick", "--continue")
    assert f"(cherry picked from commit {four})" in repo.git("log", "-1", "--format=%B")
    repo.tool("pick-pr", "1.14")
    assert repo.origin_ref(f"refs/heads/{topic}") == repo.git("rev-parse", "HEAD")
    assert repo.gh_calls()[-1].startswith(
        f"gh pr create --base release/1.14 --head {topic} --title fix(router): four"
    )


def test_pick_pr_refuses_other_branches(repo: Sandbox) -> None:
    repo.cut()
    assert "not on a pick/1.14-* branch" in repo.tool("pick-pr", "1.14", check=False).stderr


# --- status ------------------------------------------------------------------------------


def test_status_lists_what_main_has_and_the_branch_does_not(repo: Sandbox) -> None:
    repo.cut()
    three = repo.commit_main(
        "fix(router): three (#123)",
        "model_gateway/src/main.rs",
        'fn main() {\n    println!("three");\n}\n',
    )
    other = repo.commit_main(
        "feat(tool_parser): x (#124)", "crates/tool_parser/src/lib.rs", "pub fn x() {}\n"
    )
    repo.tool("pick", "1.14", three)
    repo.land_on_release()
    out = repo.tool("status", "1.14").stdout
    assert f"  {other[:7]} feat(tool_parser): x (#124)" in out
    assert three[:7] not in out.split("== on release/1.14")[0].split("== on origin/main")[1]
    assert "(1 commit(s))" in out
    assert "== versions: 1.14.0-rc.1" in out
    assert (
        "chore(release): bump versions for v1.14.0-rc.1" in out.split("the branch's own commits")[1]
    )
    assert "  v1.13.0" in out.split("== tags")[1]
    assert "#7  fix(router): y  pick/1.14-0000000" in out.split("open pull requests")[1]


# --- tag ----------------------------------------------------------------------------------


def test_tag_checks_the_versions_and_tags_the_branch_head(repo: Sandbox) -> None:
    repo.cut()
    done = repo.tool("tag", "1.14.0-rc.1")
    head = repo.origin_ref("refs/heads/release/1.14")
    assert repo.git("rev-parse", "v1.14.0-rc.1^{commit}") == head
    assert repo.git("cat-file", "-t", "v1.14.0-rc.1") == "tag", "annotated"
    tag_message = repo.git("tag", "-l", "--format=%(contents)", "v1.14.0-rc.1")
    assert tag_message.startswith("SMG v1.14.0-rc.1\n")
    assert "Signed-off-by: Release Owner <owner@example.invalid>" in tag_message
    assert repo.origin_ref("refs/tags/v1.14.0-rc.1") is None
    assert "git push origin refs/tags/v1.14.0-rc.1" in done.stdout
    assert "checking the versions at" in done.stderr
    assert "exists locally" in repo.tool("tag", "1.14.0-rc.1", check=False).stderr
    repo.git("tag", "-d", "v1.14.0-rc.1")
    repo.tool("tag", "1.14.0-rc.1", PUSH="1")
    assert repo.origin_ref("refs/tags/v1.14.0-rc.1") is not None
    repo.git("tag", "-d", "v1.14.0-rc.1")
    # the tool fetches the tags first, so the pushed tag is back and refused as existing
    assert "the tag v1.14.0-rc.1 exists" in repo.tool("tag", "1.14.0-rc.1", check=False).stderr


def test_tag_refuses_a_head_that_reads_another_version(repo: Sandbox) -> None:
    repo.cut()
    done = repo.tool("tag", "1.14.0", check=False)
    assert (
        done.returncode == 1 and "reads 1.14.0-rc.1" in done.stderr and "bump 1.14.0" in done.stderr
    )
    assert repo.git("tag", "-l", "v1.14.0") == ""


def test_tag_refuses_inconsistent_versions_at_the_head(repo: Sandbox) -> None:
    repo.cut()
    repo.git("switch", "--quiet", "release/1.14")
    repo.put(
        "deploy/helm/smg/Chart.yaml",
        'apiVersion: v2\nname: smg\nversion: 1.13.0\nappVersion: "1.13.0"\n',
    )
    repo.git("commit", "--quiet", "-am", "chore: the chart slipped")
    repo.land_on_release()
    done = repo.tool("tag", "1.14.0-rc.1", check=False)
    assert done.returncode == 1 and "inconsistent" in done.stderr
    assert not list(repo.base.glob("tmp*/.git")), "the scratch worktree is removed"


# --- notes --------------------------------------------------------------------------------


def finished_release(repo: Sandbox) -> None:
    """A candidate, a pick, the final: the branch is tagged v1.14.0-rc.1 and v1.14.0."""
    repo.cut()
    repo.tool("tag", "1.14.0-rc.1", PUSH="1")
    three = repo.commit_main(
        "fix(router): three (#123)",
        "model_gateway/src/main.rs",
        'fn main() {\n    println!("three");\n}\n',
    )
    repo.commit_main(
        "feat(tool_parser): x (#124)", "crates/tool_parser/src/lib.rs", "pub fn x() {}\n"
    )
    repo.tool("pick", "1.14", three)
    repo.land_on_release()
    repo.tool("bump", "1.14.0")
    repo.land_on_release()
    repo.tool("tag", "1.14.0", PUSH="1")


def next_minor_cut(repo: Sandbox) -> None:
    """After v1.14.0: main follows the release, a crate changes again on main, release/1.15 is cut."""
    finished_release(repo)
    repo.tool("bump-main", "1.14.0")
    repo.git("push", "--quiet", "origin", "HEAD:refs/heads/main")  # the pull request merges
    repo.commit_main(
        "fix(tool_parser): after the release (#125)",
        "crates/tool_parser/src/lib.rs",
        "pub fn x() {}\npub fn after() {}\n",
    )
    repo.cut("1.15.0-rc.1")


def test_the_next_cut_compares_against_the_previous_final_on_its_own_branch(repo: Sandbox) -> None:
    next_minor_cut(repo)
    assert "v1.14.0" not in repo.git("tag", "--merged", "origin/main").split(), (
        "the premise: a release tag is made on its branch and never reaches main"
    )
    assert repo.git("rev-parse", "--abbrev-ref", "HEAD") == "release/1.15"
    for path in VERSION_FILES:
        assert repo.version_of(path) == "1.15.0-rc.1", path
    # Against v1.14.0 (where tool-parser read 0.6.0): a feat and a fix since, so a minor bump.
    # Against v1.13.0 (the newest tag reachable from main), 0.6.0 would read as already bumped
    # and the crate would ship 1.15 at the version crates.io already has.
    assert repo.version_of("crates/tool_parser/Cargo.toml") == "0.7.0"
    assert 'tool-parser = { version = "0.7.0"' in repo.read("Cargo.toml")
    repo.git("push", "--quiet", "origin", "HEAD:refs/heads/release/1.15")
    tagged = repo.tool("tag", "1.15.0-rc.1", PUSH="1")
    assert "checking the versions at" in tagged.stderr and "against v1.14.0" in tagged.stderr
    checked = repo.tool("check", "1.15.0-rc.1")
    assert "[ok]   versions: the branch reads 1.15.0-rc.1" in checked.stdout
    assert "consistent with v1.15.0-rc.1" in checked.stdout, "the branch's own tag once it has one"
    bumped = repo.tool("bump", "1.15.0-rc.2", PUSH="0")
    assert "versions since v1.15.0-rc.1" in bumped.stderr


def test_a_patch_bump_compares_against_the_final_not_its_candidate(repo: Sandbox) -> None:
    finished_release(repo)  # the branch's own tags: v1.14.0-rc.1, v1.14.0
    bumped = repo.tool("bump", "1.14.1", PUSH="0")
    assert "versions since v1.14.0" in bumped.stderr


def test_a_cut_before_the_previous_releases_bump_main_merged_is_refused(repo: Sandbox) -> None:
    """Against the previous final, a crate bumped for that release reads as bumped already: a cut from a main that
    does not carry the release's versions yet would ship nothing for the crate, and nothing would say so."""
    finished_release(repo)  # v1.14.0 on its branch; main still reads 1.13.0
    repo.commit_main(
        "fix(tool_parser): after the release (#125)",
        "crates/tool_parser/src/lib.rs",
        "pub fn x() {}\npub fn after() {}\n",
    )
    refused = repo.tool("cut", "1.15.0-rc.1", check=False)
    assert refused.returncode == 1
    assert (
        "origin/main reads 1.13.0, below 1.14.0: merge the release-bump-main pull request for v1.14.0 first"
        in refused.stderr
    )
    assert repo.git("branch", "--list", "release/1.15") == "", "nothing was cut"


def land_on_release_squashed(repo: Sandbox, release: str = "1.14") -> None:
    """The checked-out pick pull request merges as a squash: one commit on origin's branch under the pull request's
    title alone, no body (the repository's `squash_merge_commit_message` is BLANK), so no `(cherry picked from
    commit …)` line reaches the branch; the picks file the pull request carries does."""
    topic = repo.git("rev-parse", "--abbrev-ref", "HEAD")
    repo.git("switch", "--quiet", "--detach", f"origin/release/{release}")
    repo.git("merge", "--quiet", "--squash", topic)
    repo.git(
        "commit",
        "--quiet",
        "-m",
        f"chore(release): pick 2 commits from main into release/{release} (#9)",
    )
    repo.git("push", "--quiet", "origin", f"HEAD:refs/heads/release/{release}")
    repo.git("switch", "--quiet", "main")


def test_a_squash_landed_pick_is_read_as_picked_by_status_and_the_next_notes(repo: Sandbox) -> None:
    repo.cut()
    three = repo.commit_main(
        "fix(router): three (#123)",
        "model_gateway/src/main.rs",
        'fn main() {\n    println!("three");\n}\n',
    )
    four = repo.commit_main(
        "fix(router): four (#126)",
        "model_gateway/src/four.rs",
        "pub fn four() {}\n",
    )
    repo.tool("pick", "1.14", f"{three},{four}")
    land_on_release_squashed(repo)
    assert "cherry picked" not in repo.git("log", "-1", "--format=%B", "origin/release/1.14")
    assert sorted(
        repo.read("scripts/release/picks.txt", "origin/release/1.14").splitlines()
    ) == sorted([f"{three} fix(router): three (#123)", f"{four} fix(router): four (#126)"]), (
        "the pick order follows the commit dates, which the sandbox writes within one second"
    )
    status = repo.tool("status", "1.14").stdout
    assert "fix(router): three" not in status and "fix(router): four" not in status
    assert "  (0 commit(s))" in status, "two picks in one squash commit: both read off its trailers"
    repo.tool("bump", "1.14.0")
    repo.land_on_release()
    repo.tool("tag", "1.14.0", PUSH="1")
    # The release's own notes name the two fixes, under Fixes, not the squash commit's subject under Maintenance.
    notes = repo.tool("notes", "1.14.0").stdout
    assert "- fix(router): three (#123)\n" in notes and "- fix(router): four (#126)\n" in notes
    assert "chore(release): pick 2 commits" not in notes
    assert notes.index("### Fixes") < notes.index("- fix(router): three (#123)")
    assert "6 commits since v1.13.0" in notes, "the two originals count, the squash commit does not"
    repo.tool("bump-main", "1.14.0")
    repo.git("push", "--quiet", "origin", "HEAD:refs/heads/main")  # the pull request merges
    repo.commit_main(
        "fix(tool_parser): after the release (#125)",
        "crates/tool_parser/src/lib.rs",
        "pub fn after() {}\n",
    )
    repo.cut("1.15.0-rc.1")
    notes = repo.tool("notes", "1.15.0-rc.1").stdout
    assert "fix(router): three" not in notes and "fix(router): four" not in notes
    assert "- fix(tool_parser): after the release (#125)\n" in notes


def test_notes_of_the_next_release_skip_the_fixes_picked_into_the_previous_one(
    repo: Sandbox,
) -> None:
    next_minor_cut(repo)
    out = repo.tool("notes", "1.15.0-rc.1").stdout
    assert "commits since v1.14.0 on `release/1.15`" in out
    assert "fix(router): three (#123)" not in out, "picked into 1.14.0 with -x: not news in 1.15"
    assert "- feat(tool_parser): x (#124)\n" in out
    assert "- fix(tool_parser): after the release (#125)\n" in out
    assert "fix(router): two" not in out, "shipped in 1.14.0"


def test_notes_group_the_merged_subjects_and_name_the_release_command(repo: Sandbox) -> None:
    finished_release(repo)
    done = repo.tool("notes", "1.14.0")
    out = done.stdout
    assert out.startswith("## SMG v1.14.0\n")
    assert "commits since v1.13.0 on `release/1.14`" in out
    assert "### Features\n\n- feat(tool_parser): more\n" in out
    assert "### Fixes\n\n- fix(router): three (#123)\n- fix(router): two\n" in out
    assert "chore(release): bump versions" not in out
    assert "### Docker images\n\n- stub images for v1.14.0" in out
    assert "gh release create v1.14.0 --verify-tag --title 'SMG v1.14.0'" in done.stderr
    assert "--prerelease" not in done.stderr
    candidate = repo.tool("notes", "1.14.0-rc.1")
    assert "gh release create v1.14.0-rc.1 --verify-tag --prerelease" in candidate.stderr
    since = repo.tool("notes", "1.14.0", "v1.14.0-rc.1").stdout
    assert "fix(router): two" not in since and "fix(router): three (#123)" in since


# --- publish-plan ----------------------------------------------------------------------------


def test_publish_plan_prints_the_dispatches_and_runs_nothing(repo: Sandbox) -> None:
    finished_release(repo)
    calls = len(repo.gh_calls())
    out = repo.tool("publish-plan", "1.14.0-rc.1").stdout
    assert len(repo.gh_calls()) == calls, "the plan runs nothing"
    for workflow in (
        "release-pypi.yml",
        "release-crates.yml",
        "release-docker.yml",
        "release-helm.yml",
        "release-grpc.yml",
    ):
        assert f"gh workflow run {workflow} --ref v1.14.0-rc.1" in out
    assert "gh workflow run release-engine-docker.yml --ref v1.14.0-rc.1 -f engine=vllm" in out
    assert "gh workflow run release-go.yml --ref v1.14.0-rc.1 -f version=1.14.0-rc.1" in out
    assert "smg==1.14.0rc1" in out
    assert "latest does not move" in out
    assert "     smg 1.14.0-rc.1\n" in out and "     tool-parser 0.6.0\n" in out
    final = repo.tool("publish-plan", "1.14.0").stdout
    assert "smg==1.14.0," in final and "and latest" in final


def test_publish_plan_prints_one_engine_dispatch_per_release_base(repo: Sandbox) -> None:
    finished_release(repo)
    out = repo.tool("publish-plan", "1.14.0").stdout
    dispatches = [
        line.strip()
        for line in out.splitlines()
        if line.strip().startswith("gh workflow run release-engine-docker.yml")
    ]
    assert dispatches == [
        "gh workflow run release-engine-docker.yml --ref v1.14.0 -f engine=vllm"
        "   # vllm/vllm-openai:v0.31.0",
        "gh workflow run release-engine-docker.yml --ref v1.14.0 -f engine=vllm"
        " -f base_image_ref=vllm/vllm-openai:v0.27.1   # vllm/vllm-openai:v0.27.1",
        "gh workflow run release-engine-docker.yml --ref v1.14.0 -f engine=sglang"
        "   # lmsysorg/sglang:v0.5.21",
        "gh workflow run release-engine-docker.yml --ref v1.14.0 -f engine=tokenspeed"
        "   # the latest nightly carrier",
    ]
    assert "every other base is a dispatch of its own" in out


def test_publish_plan_refuses_a_version_the_branch_does_not_carry(repo: Sandbox) -> None:
    repo.cut()
    assert (
        "reads 1.14.0-rc.1, not 1.14.0" in repo.tool("publish-plan", "1.14.0", check=False).stderr
    )


# --- check -----------------------------------------------------------------------------------


def test_check_reads_the_gate_items_and_leaves_the_verifier_lines_to_the_owner(
    repo: Sandbox,
) -> None:
    repo.cut()
    repo.commit_main(
        "feat(tool_parser): x (#124)", "crates/tool_parser/src/lib.rs", "pub fn x() {}\n"
    )
    done = repo.tool("check", "1.14.0-rc.1")
    out = done.stdout
    assert "[ok]   versions: the branch reads 1.14.0-rc.1" in out
    assert "[ok]   CI: 2 check(s) green" in out
    assert "[ok]   PR Test (SMG): a successful run on" in out
    assert "[....] tag: v1.14.0-rc.1 not created yet" in out
    assert "[note] main: 1 commit(s) on origin/main are not on release/1.14" in out
    for family in ("DeepSeek", "GLM", "Kimi", "MiniMax"):
        assert f"       [ ] {family} provider verifier: <record link>" in out
    assert out.rstrip().endswith("the verifier lines are the release owner's.")
    repo.tool("tag", "1.14.0-rc.1")
    assert (
        "[ok]   tag: v1.14.0-rc.1 is at the branch head" in repo.tool("check", "1.14.0-rc.1").stdout
    )


def test_check_fails_on_a_version_mismatch_or_red_ci(repo: Sandbox) -> None:
    repo.cut()
    done = repo.tool("check", "1.14.0-rc.2", check=False)
    assert done.returncode == 1
    assert "[FAIL] versions: the branch reads 1.14.0-rc.1, not 1.14.0-rc.2" in done.stdout
    red = repo.tool(
        "check", "1.14.0-rc.1", check=False, GH_STUB_CHECKS="failure unit tests\\nsuccess lint"
    )
    assert red.returncode == 1
    assert "       red: unit tests" in red.stdout
    assert "[FAIL] CI: 1 green, 1 red, 0 pending" in red.stdout
    pending = repo.tool("check", "1.14.0-rc.1", check=False, GH_STUB_CHECKS="in_progress build")
    assert pending.returncode == 1 and "1 pending" in pending.stdout


def test_check_requires_a_run_of_the_test_matrix_on_the_head(repo: Sandbox) -> None:
    """A documentation-only merge to the branch runs no matrix (the push trigger's paths-ignore) and the pull
    request's run was for the merge ref: the gate wants a successful run for the head commit itself."""
    repo.cut()
    none = repo.tool("check", "1.14.0-rc.1", check=False, GH_STUB_RUNS="none")
    assert none.returncode == 1
    assert "[FAIL] PR Test (SMG): no run on" in none.stdout
    assert "gh workflow run pr-test-rust.yml --ref release/1.14" in none.stdout
    pending = repo.tool(
        "check",
        "1.14.0-rc.1",
        check=False,
        GH_STUB_RUNS="in_progress https://example.invalid/runs/2",
    )
    assert pending.returncode == 1
    assert "[FAIL] PR Test (SMG): no successful run on" in pending.stdout


def test_check_without_gh_names_the_branch_page(repo: Sandbox) -> None:
    repo.cut()
    (repo.bin / "gh").unlink()
    out = repo.tool("check", "1.14.0-rc.1").stdout
    assert "[....] CI: not readable here" in out


# --- bump-main ---------------------------------------------------------------------------------


def test_bump_main_moves_mains_versions_to_the_release(repo: Sandbox) -> None:
    finished_release(repo)
    main_head = repo.commit_main("docs: after the release", "README.md", "# smg\n")
    repo.tool("bump-main", "1.14.0")
    assert repo.git("rev-parse", "--abbrev-ref", "HEAD") == "chore/main-versions-1.14.0"
    assert repo.git("rev-parse", "HEAD~1") == main_head
    for path in VERSION_FILES:
        assert repo.version_of(path) == "1.14.0", path
    assert repo.version_of("crates/tool_parser/Cargo.toml") == "0.6.0"
    assert 'tool-parser = { version = "0.6.0"' in repo.read("Cargo.toml")
    assert repo.git("log", "-1", "--format=%s") == "chore(release): main's versions follow v1.14.0"
    assert repo.gh_calls()[-1].startswith(
        "gh pr create --base main --head chore/main-versions-1.14.0"
        " --title chore(release): main's versions follow v1.14.0"
    )
    assert repo.origin_ref("refs/heads/main") == main_head, "main moves by its pull request only"


def test_bump_main_refuses_a_candidate(repo: Sandbox) -> None:
    repo.cut()
    assert "follows a final release" in repo.tool("bump-main", "1.14.0-rc.1", check=False).stderr


def test_bump_main_after_a_patch_of_an_older_line_keeps_mains_higher_version(repo: Sandbox) -> None:
    """v1.15.0 is out and main follows it; a v1.14.1 patch of the older line releases a tool-parser fix. bump-main for
    the patch moves that crate alone: main's gateway stays at 1.15.0 (the nightly and the next cut derive from it) and
    the next cut is accepted against v1.15.0. Run again once merged, it has nothing to move and says so."""
    repo.cut()  # 1.14.0-rc.1: tool-parser 0.6.0 for the feat on main since v1.13.0
    repo.tool("tag", "1.14.0-rc.1", PUSH="1")
    repo.tool("bump", "1.14.0")
    repo.land_on_release()
    repo.tool("tag", "1.14.0", PUSH="1")
    repo.tool("bump-main", "1.14.0")
    repo.git("push", "--quiet", "origin", "HEAD:refs/heads/main")
    repo.commit_main(  # nothing of tool-parser changes on main before 1.15: it stays 0.6.0 there
        "fix(router): gateway only (#130)",
        "model_gateway/src/main.rs",
        'fn main() {\n    println!("five");\n}\n',
    )
    repo.cut("1.15.0-rc.1")
    repo.tool("bump", "1.15.0")
    repo.land_on_release("1.15")
    repo.tool("tag", "1.15.0", PUSH="1")
    repo.tool("bump-main", "1.15.0")
    repo.git("push", "--quiet", "origin", "HEAD:refs/heads/main")
    assert repo.version_of("model_gateway/Cargo.toml", "origin/main") == "1.15.0"
    assert repo.version_of("crates/tool_parser/Cargo.toml", "origin/main") == "0.6.0"
    # The patch of the older line: a tool-parser fix picked into release/1.14, bumped there, released as v1.14.1.
    fix = repo.commit_main(
        "fix(tool_parser): patch (#131)",
        "crates/tool_parser/src/lib.rs",
        "pub fn x() {}\npub fn patch() {}\n",
    )
    repo.tool("pick", "1.14", fix)
    repo.land_on_release("1.14")
    repo.tool("bump", "1.14.1")
    repo.land_on_release("1.14")
    repo.tool("tag", "1.14.1", PUSH="1")
    assert repo.version_of("crates/tool_parser/Cargo.toml", "v1.14.1") == "0.6.1"
    done = repo.tool("bump-main", "1.14.1")
    assert "main reads 1.15.0, above 1.14.1: the gateway keeps it" in done.stderr
    assert "the gateway reads 1.15.0, above 1.14.1: it and its mirrors keep that" in done.stdout
    for path in VERSION_FILES:
        assert repo.version_of(path) == "1.15.0", path
    assert repo.version_of("crates/tool_parser/Cargo.toml") == "0.6.1"
    assert 'tool-parser = { version = "0.6.1"' in repo.read("Cargo.toml")
    assert repo.git("log", "-1", "--format=%b").startswith(
        "The gateway keeps 1.15.0 on main (above v1.14.1)"
    )
    repo.git("push", "--quiet", "origin", "HEAD:refs/heads/main")  # the pull request merges
    repo.git("switch", "--quiet", "main")
    repo.git("branch", "--quiet", "-D", "chore/main-versions-1.14.1")
    again = repo.tool("bump-main", "1.14.1", check=False)
    assert again.returncode == 1 and "nothing to move" in again.stderr
    assert repo.git("rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert repo.git("branch", "--list", "chore/main-versions-1.14.1") == ""
    repo.cut("1.16.0-rc.1")  # accepted: main reads the previous final's version, v1.15.0's
    assert repo.git("rev-parse", "--abbrev-ref", "HEAD") == "release/1.16"
    for path in VERSION_FILES:
        assert repo.version_of(path) == "1.16.0-rc.1", path


def grpc_selection(tmp_path: Path, pypi: str, packages: str = "auto") -> dict[str, str]:
    """The detect step of release-grpc.yml over a tree whose two pyproject files read 1.14.0, with a curl that
    answers for PyPI."""
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "release-grpc.yml").read_text())
    step = next(s for s in workflow["jobs"]["detect"]["steps"] if s.get("id") == "changes")
    tree = tmp_path / f"selection-{len(list(tmp_path.iterdir()))}"
    for file in ("crates/grpc_client/python/pyproject.toml", "grpc_servicer/pyproject.toml"):
        (tree / file).parent.mkdir(parents=True)
        (tree / file).write_text('[project]\nversion = "1.14.0"\n')
    bin_dir = tree / "bin"
    bin_dir.mkdir()
    (bin_dir / "curl").write_text(CURL_STUB)
    (bin_dir / "curl").chmod(0o755)
    output = tree / "output"
    done = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", step["run"]],
        cwd=tree,
        env={
            "PATH": os.pathsep.join(
                [str(bin_dir), str(Path(sys.executable).parent), "/usr/bin", "/bin"]
            ),
            "GITHUB_OUTPUT": str(output),
            "CURL_STUB": pypi,
            "PACKAGES": packages,
        },
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    return dict(line.split("=", 1) for line in output.read_text().splitlines())


def test_the_grpc_release_selects_a_package_until_pypi_holds_both_its_files(tmp_path: Path) -> None:
    """A version PyPI shows after a failed upload may hold one of its two files; a selection by the version's presence
    alone would skip the package for good. Selected unless the wheel and the sdist are both there; the packages input
    names them outright."""
    complete = "smg-grpc-proto/1.14.0=complete smg-grpc-servicer/1.14.0=complete"
    assert grpc_selection(tmp_path, complete) == {"proto": "false", "servicer": "false"}
    half = "smg-grpc-proto/1.14.0=wheel smg-grpc-servicer/1.14.0=complete"
    assert grpc_selection(tmp_path, half) == {"proto": "true", "servicer": "false"}
    assert grpc_selection(tmp_path, "") == {"proto": "true", "servicer": "true"}
    assert grpc_selection(tmp_path, complete, packages="servicer") == {
        "proto": "false",
        "servicer": "true",
    }
    assert grpc_selection(tmp_path, complete, packages="both") == {
        "proto": "true",
        "servicer": "true",
    }


def test_sync_versions_never_lowers_a_version_main_already_raised(repo: Sandbox) -> None:
    repo.cut()
    repo.commit_main(
        "chore(tool_parser): 0.7.0 on main",
        "crates/tool_parser/Cargo.toml",
        versioned("tool-parser", "0.7.0"),
    )
    done = subprocess.run(
        [sys.executable, str(repo.work / SYNC), "origin/release/1.14", "1.14.0"],
        cwd=repo.work,
        env=repo.env(),
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    assert repo.version_of("crates/tool_parser/Cargo.toml") == "0.7.0"
    assert repo.version_of("model_gateway/Cargo.toml") == "1.14.0"
    assert repo.version_of("deploy/helm/smg/Chart.yaml") == "1.14.0"
    assert "model_gateway/Cargo.toml: 1.13.0 -> 1.14.0" in done.stdout
    assert "tool_parser" not in done.stdout
    again = subprocess.run(
        [sys.executable, str(repo.work / SYNC), "origin/release/1.14", "1.14.0"],
        cwd=repo.work,
        env=repo.env(),
        capture_output=True,
        text=True,
    )
    assert "already follows" in again.stdout


def test_sync_versions_usage_and_version_order() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("sync_versions", ROOT / SYNC)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    key = module.version_key
    assert key("1.14.0-rc.1") < key("1.14.0-rc.2") < key("1.14.0") < key("1.14.1-rc.1")
    with pytest.raises(SystemExit):
        key("1.14")
    assert module.main([]) == 2


# --- the shape of the tooling -------------------------------------------------------------------


def test_help_and_unknown_commands() -> None:
    done = subprocess.run([str(ROOT / TOOL), "help"], capture_output=True, text=True, cwd=ROOT)
    assert done.returncode == 0
    for command in MAKE_TARGETS.values():
        assert f"release.sh {command} " in done.stdout, command
    unknown = subprocess.run(
        [str(ROOT / TOOL), "frobnicate"], capture_output=True, text=True, cwd=ROOT
    )
    assert unknown.returncode == 1 and "unknown command: frobnicate" in unknown.stderr


def test_shellcheck_is_clean() -> None:
    if shutil.which("shellcheck") is None:
        pytest.skip("shellcheck is not installed")
    done = subprocess.run(["shellcheck", str(ROOT / TOOL)], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout


@pytest.mark.parametrize("target,command", sorted(MAKE_TARGETS.items()))
def test_make_dry_run_of_every_target(target: str, command: str) -> None:
    if shutil.which("make") is None:
        pytest.skip("make is not installed")
    done = subprocess.run(
        ["make", "-n", target, "VERSION=1.14.0-rc.1", "RELEASE=1.14", "SHA=abc1234", "PUSH=1"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    assert f"scripts/release/release.sh {command} " in done.stdout


def test_make_help_lists_every_target() -> None:
    if shutil.which("make") is None:
        pytest.skip("make is not installed")
    out = subprocess.run(["make", "help"], cwd=ROOT, capture_output=True, text=True).stdout
    for target in MAKE_TARGETS:
        assert target in out, target
