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
import time
from datetime import UTC, datetime, timedelta
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
api_error() { # <message> <status>: an API error as gh reports it: the body on stdout, the status on stderr, exit 1
  printf '{"message":"%s","documentation_url":"https://docs.github.com/rest","status":"%s"}\n' "$1" "$2"
  echo "gh: $1 (HTTP $2)" >&2
  exit 1
}
case "$1 $2" in
  "pr create")
    for ((i = 1; i <= $#; i++)); do
      if [ "${!i}" = "--body-file" ]; then j=$((i + 1)); cp "${!j}" "$GH_STUB_LOG.body"; fi
    done
    echo "https://example.invalid/pull/1" ;;
  "pr list") echo "#7  fix(router): y  pick/1.14-0000000" ;;
  "repo view") echo "example/repo" ;;
  "api --paginate")
    [ -z "${GH_STUB_API_ERROR-}" ] || api_error "$GH_STUB_API_ERROR" "${GH_STUB_API_STATUS:-422}"
    printf '%b\n' "${GH_STUB_CHECKS:-success a\nsuccess b}" ;;
  "api repos/example/repo/actions/workflows/pr-test-rust.yml/runs?head_sha="*)
    [ -z "${GH_STUB_RUNS_ERROR-}" ] || api_error "$GH_STUB_RUNS_ERROR" 401
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
# The registries as the release tool reads them: crates.io's sparse index (one JSON line per version) from
# CRATES_STUB ("name=v1,v2 other=v3@2026-01-01T00:00:00Z": a version may carry its publish time), PyPI's JSON from
# PYPI_STUB (the same shape); "down" answers nothing (exit 7); an unlisted name is a 404. With -w the http code
# follows the body on its own line, as curl prints it.
REGISTRY_STUB = r"""#!/usr/bin/env bash
printf 'curl %s\n' "$*" >> "$CURL_STUB_LOG"
body_of() { # <crate|package> <v1[@time],v2>: the body for those versions, a publish time when given
  local items="" v t p
  for v in ${2//,/ }; do
    t=""; case $v in *@*) t=${v#*@}; v=${v%%@*} ;; esac
    if [ "$1" = crate ]; then
      p=""; [ -z "$t" ] || p=",\"pubtime\":\"$t\""
      items="$items"$'\n'"{\"vers\":\"$v\",\"yanked\":false$p}"
    else
      p=""; [ -z "$t" ] || p="{\"upload_time_iso_8601\":\"$t\"}"
      items="$items,\"$v\":[$p]"
    fi
  done
  if [ "$1" = crate ]; then printf '%s' "${items#$'\n'}"; else printf '{"releases":{%s}}' "${items#,}"; fi
}
url=${*: -1}; code=404; body=""
case "$url" in
  https://index.crates.io/*)
    name=${url##*/}; [ "${CRATES_STUB-}" = down ] && exit 7
    for entry in ${CRATES_STUB-}; do
      [ "${entry%%=*}" = "$name" ] && { code=200; body=$(body_of crate "${entry#*=}"); }
    done ;;
  https://pypi.org/pypi/*/json)
    name=${url#https://pypi.org/pypi/}; name=${name%/json}; [ "${PYPI_STUB-}" = down ] && exit 7
    for entry in ${PYPI_STUB-}; do
      [ "${entry%%=*}" = "$name" ] && { code=200; body=$(body_of package "${entry#*=}"); }
    done ;;
esac
case " $* " in
  *" -w "*) printf '%s\n%s' "$body" "$code" ;;
  *) [ "$code" = 200 ] && printf '%s' "$body" || exit 22 ;;
esac
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


def utc_second_after(iso: str) -> str:
    """One second after a git ``%cI`` date, in the UTC spelling the tool prints for a publish time."""
    moment = datetime.fromisoformat(iso).astimezone(UTC) + timedelta(seconds=1)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


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
        self.curl_log = base / "curl.log"
        self.bin.mkdir()
        for name, text in (("gh", GH_STUB), ("cargo", CARGO_STUB), ("curl", REGISTRY_STUB)):
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
            "CURL_STUB_LOG": str(self.curl_log),
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
        # a crate the checker lists that cargo never publishes (the real one is a generator's scaffold)
        self.put(
            "clients/openapi-gen/Cargo.toml",
            versioned("openapi-gen", "0.2.0") + "publish = false\n",
        )
        self.put("clients/openapi-gen/src/main.rs", "fn main() {}\n")
        # cargo's other spelling of never published, under [package] like the first (the sandbox's, not the crate's)
        self.put("clients/rust/Cargo.toml", versioned("smg-client", "0.1.0") + "publish = []\n")
        self.put("clients/rust/src/lib.rs", "pub fn client() {}\n")
        # a Python package the checker lists, published to PyPI by its own dispatch
        self.put(
            "grpc_servicer/pyproject.toml",
            '[project]\nname = "smg-grpc-servicer"\nversion = "0.14.0"\n',
        )
        self.put("grpc_servicer/servicer.py", "x = 1\n")
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

    def curl_calls(self) -> list[str]:
        return self.curl_log.read_text().splitlines() if self.curl_log.exists() else []

    def pr_body(self) -> str:
        return Path(f"{self.gh_log}.body").read_text()


@pytest.fixture
def repo(tmp_path: Path) -> Sandbox:
    return Sandbox(tmp_path)


# --- cut --------------------------------------------------------------------------


def test_cut_creates_the_branch_and_the_bump_commit(repo: Sandbox) -> None:
    done = repo.tool("cut", "1.14.0-rc.1")
    # the branch is behind you: the next commands run from the checkout the cut started on
    assert repo.git("rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert "release/1.14 is at " in done.stdout and "(You are back on main;" in done.stdout
    branch = "release/1.14"
    for path in VERSION_FILES:
        assert repo.version_of(path, branch) == "1.14.0-rc.1", path
        assert repo.version_of(path) == "1.13.0", f"{path}: main's tree is untouched"
    assert (
        repo.version_of("crates/tool_parser/Cargo.toml", branch) == "0.6.0"
    )  # a feat since v1.13.0: minor
    assert 'tool-parser = { version = "0.6.0"' in repo.read("Cargo.toml", branch)
    message = repo.git("log", "-1", "--format=%B", branch)
    assert message.startswith("chore(release): bump versions for v1.14.0-rc.1\n")
    assert "Signed-off-by: Release Owner <owner@example.invalid>" in message
    assert (
        repo.git("log", "-1", "--format=%an <%ae> %cn", branch)
        == "Release Owner <owner@example.invalid> Release Owner"
    )
    assert repo.git("rev-parse", f"{branch}~1") == repo.origin_ref("refs/heads/main")
    assert repo.origin_ref("refs/heads/release/1.14") is None, (
        "the cut is not pushed without PUSH=1"
    )
    assert "git push origin refs/heads/release/1.14:refs/heads/release/1.14" in done.stdout
    assert "nothing is published" in done.stdout


def test_cut_pushes_the_branch_only_on_request(repo: Sandbox) -> None:
    repo.tool("cut", "1.14.0-rc.1", PUSH="1")
    assert repo.origin_ref("refs/heads/release/1.14") == repo.git("rev-parse", "release/1.14")
    assert repo.origin_ref("refs/heads/main") == repo.git("rev-parse", "release/1.14~1"), (
        "main is untouched"
    )
    assert repo.git("rev-parse", "--abbrev-ref", "HEAD") == "main"


def test_cut_starts_from_the_named_commit(repo: Sandbox) -> None:
    older = repo.git("rev-parse", "origin/main~1")
    repo.tool("cut", "1.14.0-rc.1", older)
    assert repo.git("rev-parse", "release/1.14~1") == older
    # the gateway did not change between v1.13.0 and that commit, its version still moves
    assert repo.version_of("model_gateway/Cargo.toml", "release/1.14") == "1.14.0-rc.1"


def test_cut_from_a_detached_head_leaves_you_detached_there(repo: Sandbox) -> None:
    start = repo.git("rev-parse", "origin/main")
    repo.git("switch", "--quiet", "--detach", start)
    done = repo.tool("cut", "1.14.0-rc.1")
    assert repo.git("rev-parse", "HEAD") == start
    assert repo.git("rev-parse", "--abbrev-ref", "HEAD") == "HEAD"
    assert f"(You are back on {repo.git('rev-parse', '--short', start)};" in done.stdout
    assert repo.git("rev-parse", "release/1.14~1") == start


def test_cut_comes_back_to_the_branch_when_a_tag_shares_its_name(repo: Sandbox) -> None:
    """A tag named like the branch makes `git symbolic-ref --short HEAD` answer `heads/main`, which `git switch` does
    not take as the branch: the cut reads the start with `git branch --show-current` and comes back to the branch."""
    repo.git("tag", "main")
    done = repo.tool("cut", "1.14.0-rc.1")
    assert repo.git("branch", "--show-current") == "main"
    assert "(You are back on main;" in done.stdout


def test_cut_reports_the_branch_beside_a_tag_of_its_name(repo: Sandbox) -> None:
    """A tag named like the release branch answers for the bare name: the closing lines read the
    branch by its full ref, so they name the bump commit, not the tag's."""
    older = repo.git("rev-parse", "origin/main~1")
    repo.git("tag", "release/1.14", older)
    done = repo.tool("cut", "1.14.0-rc.1")
    bump = repo.git("rev-parse", "--short", "refs/heads/release/1.14")
    assert (
        f"release/1.14 is at {bump}: chore(release): bump versions for v1.14.0-rc.1" in done.stdout
    )
    assert repo.git("rev-parse", "--short", older) not in done.stdout


def test_cut_comes_back_without_a_head_reflog(repo: Sandbox) -> None:
    """`git switch -` and `@{-1}` read the HEAD reflog, which a clone can be without (core.logAllRefUpdates=false, or
    the logs expired): the cut reads where it started before it makes the branch, and comes back to that."""
    repo.git("config", "core.logAllRefUpdates", "false")
    (repo.work / ".git" / "logs" / "HEAD").unlink()
    done = repo.tool("cut", "1.14.0-rc.1")
    assert repo.git("rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert "(You are back on main;" in done.stdout
    assert not (repo.work / ".git" / "logs" / "HEAD").exists(), (
        "the sandbox stayed without a HEAD reflog"
    )


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
    assert repo.read(f"scripts/release/picks/{three}") == "fix(router): three (#123)\n"
    # A second run of pick-pr (a pull request creation that failed): the record commit is no pick, so the title,
    # the count and the body read the one cherry-pick as before, and nothing is recorded twice.
    repo.tool("pick-pr", "1.14")
    assert repo.git("log", "-1", "--format=%s") == "chore(release): record 1 pick into release/1.14"
    assert repo.git("rev-parse", "HEAD~2") == branch_head, "no second record commit"
    assert repo.gh_calls()[-1].startswith(
        f"gh pr create --base release/1.14 --head {topic} --title fix(router): three (#123) --body-file"
    )
    assert repo.pr_body().count("cherry picked from commit") == 1
    # A tracked file left modified (a conflict resolution's leftovers) is refused, never swept into the record.
    repo.put("Cargo.toml", repo.read("Cargo.toml") + "# edited\n")
    refused = repo.tool("pick-pr", "1.14", check=False)
    assert (
        refused.returncode == 1
        and "uncommitted changes: discard them (git restore <file>)" in refused.stderr
    )
    assert f"commit them on {topic} and say so in the pull request" in refused.stderr
    repo.git("checkout", "--", "Cargo.toml")
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
    assert repo.read(f"scripts/release/picks/{first}") == "fix(a): first\n"
    assert repo.read(f"scripts/release/picks/{second}") == "fix(b): second\n"
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
    topic = f"pick/1.14-{four[:7]}"
    assert f"make release-pick-pr RELEASE=1.14 PICK={topic}" in done.stdout
    assert repo.gh_calls() == [], "no pull request for an unresolved pick"
    assert repo.git("rev-parse", "--abbrev-ref", "HEAD") == topic
    assert "still in progress" in repo.tool("pick-pr", "1.14", check=False).stderr
    repo.put("model_gateway/src/main.rs", 'fn main() {\n    println!("four");\n}\n')
    repo.git("add", "model_gateway/src/main.rs")
    repo.git("cherry-pick", "--continue")
    assert f"(cherry picked from commit {four})" in repo.git("log", "-1", "--format=%B")
    # finished from main, the branch named: the pick branch may lack the release targets
    repo.git("switch", "--quiet", "main")
    again = repo.tool(
        "pick", "1.14", four, check=False
    )  # the same pick once more: the hint names the branch
    assert (
        f"the branch {topic} exists: delete it or finish it (make release-pick-pr RELEASE=1.14 PICK={topic})"
        in again.stderr
    )
    repo.tool("pick-pr", "1.14", topic)
    assert repo.git("rev-parse", "--abbrev-ref", "HEAD") == topic
    assert repo.origin_ref(f"refs/heads/{topic}") == repo.git("rev-parse", "HEAD")
    assert repo.gh_calls()[-1].startswith(
        f"gh pr create --base release/1.14 --head {topic} --title fix(router): four"
    )


def test_pick_pr_refuses_other_branches(repo: Sandbox) -> None:
    repo.cut()
    repo.git(
        "switch", "--quiet", "release/1.14"
    )  # the cut left the clone where it was; the test tries pick-pr from the branch
    refused = repo.tool("pick-pr", "1.14", check=False)
    assert (
        "not on a pick/1.14-* branch (on release/1.14): name it, release.sh pick-pr 1.14 <pick branch>"
        in refused.stderr
    )
    assert (
        "no branch pick/1.14-0000000 here"
        in repo.tool("pick-pr", "1.14", "pick/1.14-0000000", check=False).stderr
    )
    # a wrong name is refused before the clone moves (a later `make release-*` would run from the named branch)
    assert (
        "main is not a pick/1.14-* branch (the pick branches are pick/1.14-<sha>...)"
        in repo.tool("pick-pr", "1.14", "main", check=False).stderr
    )
    assert repo.git("rev-parse", "--abbrev-ref", "HEAD") == "release/1.14", (
        "a wrong name is not switched to"
    )
    repo.git("branch", "--quiet", "pick/1.14-0000001")
    repo.put("Cargo.toml", repo.read("Cargo.toml") + "# edited\n")
    dirty = repo.tool("pick-pr", "1.14", "pick/1.14-0000001", check=False)
    assert (
        "uncommitted changes: commit or discard them before switching to pick/1.14-0000001"
        in dirty.stderr
    )
    assert repo.git("rev-parse", "--abbrev-ref", "HEAD") == "release/1.14", (
        "a dirty tree is not switched"
    )
    repo.git("checkout", "--", "Cargo.toml")


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
    tags = out.split("== tags made on release/1.14")[1].split("== open pull requests")[0]
    assert tags.startswith(" (the versions are compared against v1.13.0):\n  (none yet)\n")
    assert "v1.13.0" not in tags.split("\n", 1)[1], (
        "main's history of release tags is not the branch's"
    )
    assert "#7  fix(router): y  pick/1.14-0000000" in out.split("open pull requests")[1]
    repo.tool("tag", "1.14.0-rc.1", PUSH="1")
    tagged = repo.tool("status", "1.14").stdout
    tags = tagged.split("== tags made on release/1.14")[1].split("== open pull requests")[0]
    assert tags == " (the versions are compared against v1.14.0-rc.1):\n  v1.14.0-rc.1\n"


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
    repo.git(
        "switch", "--quiet", "release/1.15"
    )  # the cut left the clone where it was; the test reads the branch
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
    assert (
        repo.read(f"scripts/release/picks/{three}", "origin/release/1.14")
        == "fix(router): three (#123)"
    )
    assert (
        repo.read(f"scripts/release/picks/{four}", "origin/release/1.14")
        == "fix(router): four (#126)"
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


def test_two_pick_pull_requests_against_one_branch_land_in_turn_without_a_conflict(
    repo: Sandbox,
) -> None:
    """One file per original: the second pick pull request, opened before the first merged, squash-merges onto the
    branch the first moved without touching the same path; both picks are read."""
    repo.cut()
    three = repo.commit_main(
        "fix(router): three (#123)",
        "model_gateway/src/main.rs",
        'fn main() {\n    println!("three");\n}\n',
    )
    four = repo.commit_main(
        "fix(tool_parser): four (#126)", "crates/tool_parser/src/lib.rs", "pub fn four() {}\n"
    )
    repo.tool("pick", "1.14", three)
    first = repo.git("rev-parse", "--abbrev-ref", "HEAD")
    repo.git("switch", "--quiet", "main")
    repo.tool("pick", "1.14", four)
    second = repo.git("rev-parse", "--abbrev-ref", "HEAD")
    for topic in (first, second):
        repo.git("switch", "--quiet", "--detach", "origin/release/1.14")
        repo.git("merge", "--quiet", "--squash", topic)  # no conflict: different paths under picks/
        repo.git("commit", "--quiet", "-m", f"{topic} squashed (#9)")
        repo.git("push", "--quiet", "origin", "HEAD:refs/heads/release/1.14")
    repo.git("switch", "--quiet", "main")
    assert (
        repo.read(f"scripts/release/picks/{three}", "origin/release/1.14")
        == "fix(router): three (#123)"
    )
    assert (
        repo.read(f"scripts/release/picks/{four}", "origin/release/1.14")
        == "fix(tool_parser): four (#126)"
    )
    assert "  (0 commit(s))" in repo.tool("status", "1.14").stdout


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
    assert "openapi-gen" not in out, (
        "a crate with publish = false is never published: not in the plan"
    )
    assert "smg-client" not in out, (
        "a crate with publish = [] is never published either: not in the plan"
    )
    assert "publish = false or publish = [] is never published and not listed" in out
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


def test_check_reads_an_api_error_as_an_error_not_as_a_pending_check(repo: Sandbox) -> None:
    """gh leaves the API's error body on stdout: a head the remote lacks (422), a rate limit or a bad token
    is reported as the failed read it is, with the API's message, never counted as one pending check."""
    repo.cut()
    head = repo.git("rev-parse", "--short", "origin/release/1.14")
    calls_before = len(repo.gh_calls())
    unpushed = repo.tool(
        "check", "1.14.0-rc.1", check=False, GH_STUB_API_ERROR=f"No commit found for SHA: {head}"
    )
    assert unpushed.returncode == 1
    assert (
        f"[....] CI: {head} is not on example/repo, the repository gh reads (origin is {repo.origin}):"
        " either the head is not pushed yet, or gh is pointed at another repository than origin"
        " (GH_REPO, gh repo set-default, a fork as the remote)" in unpushed.stdout
    )
    assert "pending" not in unpushed.stdout and "[FAIL] PR Test (SMG)" not in unpushed.stdout
    assert '"message"' not in unpushed.stdout
    calls = repo.gh_calls()[calls_before:]
    assert any("check-runs" in call for call in calls)
    assert not any("actions/workflows" in call for call in calls), (
        "no matrix-run read after a failed check-runs read"
    )
    limited = repo.tool(
        "check",
        "1.14.0-rc.1",
        check=False,
        GH_STUB_API_ERROR="API rate limit exceeded for 203.0.113.9.",
        GH_STUB_API_STATUS="403",
    )
    assert limited.returncode == 1
    assert (
        "[FAIL] CI: the check-runs read failed (API rate limit exceeded for 203.0.113.9.);"
        " the checks and the PR Test (SMG) run of " in limited.stdout
    )
    assert "https://github.com/example/repo/commits/" in limited.stdout
    assert "pending" not in limited.stdout
    runs = repo.tool("check", "1.14.0-rc.1", check=False, GH_STUB_RUNS_ERROR="Bad credentials")
    assert runs.returncode == 1
    assert "[ok]   CI: 2 check(s) green" in runs.stdout
    assert (
        "[FAIL] PR Test (SMG): the workflow-runs read failed (Bad credentials); the runs of "
        in runs.stdout
    )


def test_check_names_every_pending_check_and_reads_a_startup_failure_as_red(repo: Sandbox) -> None:
    repo.cut()
    done = repo.tool(
        "check",
        "1.14.0-rc.1",
        check=False,
        GH_STUB_CHECKS="in_progress build\\nqueued e2e (mock)\\nstartup_failure finish\\nsuccess lint",
    )
    assert done.returncode == 1
    assert "       pending: build (in_progress)" in done.stdout
    assert "       pending: e2e (mock) (queued)" in done.stdout
    assert "       red: finish" in done.stdout
    assert "[FAIL] CI: 1 green, 1 red, 2 pending on" in done.stdout
    green = repo.tool("check", "1.14.0-rc.1")
    assert "pending:" not in green.stdout and "red:" not in green.stdout


def test_check_reads_the_registries_for_versions_published_already(repo: Sandbox) -> None:
    """A crate or package at a version crates.io or PyPI hold already is skipped by its publish dispatch: with
    changes after that version landed the release would ship none of them, so the gate reads the registries."""
    repo.cut()  # tool-parser 0.6.0 by the cut's bump, smg 1.14.0-rc.1, smg-grpc-servicer 0.14.0 unchanged
    done = repo.tool("check", "1.14.0-rc.1")
    assert (
        "[ok]   registry: no crate or package sits at a version crates.io or PyPI hold already"
        in done.stdout
    )
    assert "(3 of 3 read)" in done.stdout
    urls = sorted(call.split()[-1] for call in repo.curl_calls())
    assert (
        urls
        == [  # crates.io through the sparse index cargo reads (its path scheme), PyPI's JSON
            "https://index.crates.io/3/s/smg",
            "https://index.crates.io/to/ol/tool-parser",
            "https://pypi.org/pypi/smg-grpc-servicer/json",
        ]
    )
    # published already, nothing changed after the bump: the dispatch skips it and nothing is lost
    quiet = repo.tool(
        "check",
        "1.14.0-rc.1",
        CRATES_STUB="tool-parser=0.5.0,0.6.0",
        PYPI_STUB="smg-grpc-servicer=0.14.0",
    )
    assert "[ok]   registry:" in quiet.stdout
    assert (
        "[note] registry: at a published version with nothing changed after it (the dispatch skips them,"
        " nothing is lost): smg-grpc-servicer 0.14.0, tool-parser 0.6.0" in quiet.stdout
    )
    # a change lands on the branch after the bump: the published version would ship without it
    repo.git("switch", "--quiet", "release/1.14")
    repo.put(
        "crates/tool_parser/src/lib.rs", "pub fn parse() {}\npub fn more() {}\npub fn late() {}\n"
    )
    repo.git("commit", "--quiet", "-am", "fix(tool_parser): late")
    repo.land_on_release()
    repo.git("switch", "--quiet", "main")
    held = repo.tool("check", "1.14.0-rc.1", check=False, CRATES_STUB="tool-parser=0.6.0")
    assert held.returncode == 1
    assert (
        "[FAIL] registry: at a version crates.io or PyPI hold already, with changes after it (the publish"
        " dispatch skips them and ships none of those changes; a bump pull request against release/1.14"
        " moves them): tool-parser 0.6.0 (1 commit(s) after its bump; the registry gave no publish time)"
        in held.stdout
    )
    # the registry held 0.6.0 before the candidate was tagged (published on main the older way, or by another
    # line): the tag's dispatch skipped it, so the commits after the publish time keep counting, tag or no tag
    repo.tool("tag", "1.14.0-rc.1", PUSH="1")
    early = "2000-01-01T00:00:00Z"
    since_early = repo.git(
        "rev-list", "--count", f"--after={early}", "origin/release/1.14", "--", "crates/tool_parser"
    )
    held = repo.tool("check", "1.14.0-rc.1", check=False, CRATES_STUB=f"tool-parser=0.6.0@{early}")
    assert (
        f"tool-parser 0.6.0 ({since_early} commit(s) after it was published ({early}))"
        in held.stdout
    )
    # published by the tag's dispatch, after the branch's last commit: nothing is missing...
    published = utc_second_after(repo.git("log", "-1", "--format=%cI", "origin/release/1.14"))
    tagged = repo.tool(
        "check", "1.14.0-rc.1", check=False, CRATES_STUB=f"tool-parser=0.6.0@{published}"
    )
    assert "[ok]   registry:" in tagged.stdout
    assert "tool-parser 0.6.0" in tagged.stdout.split("[note] registry:")[-1]
    # ...until the next change lands, counted from that publish time
    time.sleep(1.1)
    repo.git("switch", "--quiet", "release/1.14")
    repo.put(
        "crates/tool_parser/src/lib.rs", "pub fn parse() {}\npub fn more() {}\npub fn later() {}\n"
    )
    repo.git("commit", "--quiet", "-am", "fix(tool_parser): later")
    repo.land_on_release()
    repo.git("switch", "--quiet", "main")
    later = repo.tool(
        "check", "1.14.0-rc.1", check=False, CRATES_STUB=f"tool-parser=0.6.0@{published}"
    )
    assert f"tool-parser 0.6.0 (1 commit(s) after it was published ({published}))" in later.stdout
    # the registries do not answer: the gate says so and does not pass
    down = repo.tool("check", "1.14.0-rc.1", check=False, CRATES_STUB="down")
    assert down.returncode == 1
    assert (
        "[....] registry: 2 of 3 read(s) of crates.io / PyPI did not answer; read again before the tag"
        in down.stdout
    )
    calls = len(repo.curl_calls())
    skipped = repo.tool("check", "1.14.0-rc.1", check=False, REGISTRY="0")
    assert "[....] registry: crates.io and PyPI not read (REGISTRY=0)" in skipped.stdout
    assert len(repo.curl_calls()) == calls, "REGISTRY=0 reads nothing"


def test_check_fails_the_registry_item_when_the_component_list_cannot_be_read(
    repo: Sandbox,
) -> None:
    """The crates and packages come from a helper the checker's module reads; a manifest it refuses (workspace
    versioning) fails the item with the reason, it never shortens the list behind a green line."""
    repo.cut()
    repo.git("switch", "--quiet", "release/1.14")
    repo.put(
        "crates/tool_parser/Cargo.toml",
        '[package]\nname = "tool-parser"\nversion.workspace = true\n',
    )
    repo.git("commit", "--quiet", "-am", "chore: workspace versioning")
    repo.land_on_release()
    repo.git("switch", "--quiet", "main")
    done = repo.tool("check", "1.14.0-rc.1", check=False)
    assert done.returncode == 1
    assert "[FAIL] registry: the crate and package list at " in done.stdout
    assert "workspace versioning" in done.stdout and "[ok]   registry" not in done.stdout
    assert repo.curl_calls() == [], "nothing is compared when the list is unknown"
    plan = repo.tool("publish-plan", "1.14.0-rc.1", check=False)
    assert plan.returncode == 1
    assert "could not be read" in plan.stderr and "workspace versioning" in plan.stderr


def test_publish_plan_marks_the_versions_the_registries_hold(repo: Sandbox) -> None:
    finished_release(repo)
    repo.git("switch", "--quiet", "main")
    out = repo.tool(
        "publish-plan",
        "1.14.0",
        CRATES_STUB="tool-parser=0.6.0",
        PYPI_STUB="smg-grpc-servicer=0.14.0",
    ).stdout
    assert "     tool-parser 0.6.0 (on crates.io already: the dispatch skips it)\n" in out
    assert "     smg 1.14.0\n" in out
    assert "smg-grpc-servicer==0.14.0 (on PyPI already: the dispatch skips it)" in out
    plain = repo.tool("publish-plan", "1.14.0", REGISTRY="0").stdout
    assert (
        "     tool-parser 0.6.0\n" in plain
        and "(the registries were not read: REGISTRY=0)" in plain
    )


def test_check_without_gh_names_the_branch_page(repo: Sandbox) -> None:
    repo.cut()
    (repo.bin / "gh").unlink()
    out = repo.tool("check", "1.14.0-rc.1").stdout
    assert "[....] CI: not readable here" in out


# --- a rehearsal: the cut before its push ------------------------------------------------------------


def test_the_read_only_commands_read_a_local_cut_before_its_push(repo: Sandbox) -> None:
    """A cut without PUSH=1 is local. check, status, publish-plan and notes read the local branch and say so,
    which rehearses a release on a clone; bump, pick and tag still need the branch on the remote."""
    repo.tool("cut", "1.14.0-rc.1")
    assert repo.origin_ref("refs/heads/release/1.14") is None
    repo.git("switch", "--quiet", "main")
    said = (
        "release: release/1.14 is not on origin yet: reading the local branch"
        " (the push: git push origin refs/heads/release/1.14:refs/heads/release/1.14)"
    )
    check = repo.tool("check", "1.14.0-rc.1")
    assert said in check.stderr and "[ok]   versions: the branch reads 1.14.0-rc.1" in check.stdout
    status = repo.tool("status", "1.14")
    assert said in status.stderr and "== versions: 1.14.0-rc.1" in status.stdout
    plan = repo.tool("publish-plan", "1.14.0-rc.1")
    assert (
        said in plan.stderr and "gh workflow run release-pypi.yml --ref v1.14.0-rc.1" in plan.stdout
    )
    notes = repo.tool("notes", "1.14.0-rc.1")
    assert said in notes.stderr and "## SMG v1.14.0-rc.1" in notes.stdout
    assert repo.origin_ref("refs/heads/release/1.14") is None, "a read pushes nothing"
    on_main = repo.git("rev-parse", "origin/main")
    for command in (("bump", "1.14.0-rc.2"), ("pick", "1.14", on_main), ("tag", "1.14.0-rc.1")):
        refused = repo.tool(*command, check=False)
        assert refused.returncode == 1, command
        assert "no release/1.14 on origin: cut it first" in refused.stderr, command
    assert "no release/1.15 on origin or here" in repo.tool("status", "1.15", check=False).stderr
    # Once pushed, the remote's branch is read and nothing is said.
    repo.git("push", "--quiet", "origin", "refs/heads/release/1.14:refs/heads/release/1.14")
    assert "reading the local branch" not in repo.tool("check", "1.14.0-rc.1").stderr


def test_a_branch_the_remote_has_but_this_clone_never_fetched_is_not_read_as_a_rehearsal(
    repo: Sandbox,
) -> None:
    """A single-branch clone or a narrowed fetch refspec never creates origin/release/<x.y>, while the local
    branch the cut left behind goes stale as pull requests move the remote: the remote is asked before a local
    branch is read."""
    repo.cut()  # pushed: the remote has release/1.14
    first = repo.origin_ref("refs/heads/release/1.14")
    repo.git("update-ref", "-d", "refs/remotes/origin/release/1.14")
    repo.git("config", "remote.origin.fetch", "+refs/heads/main:refs/remotes/origin/main")
    for command in (("check", "1.14.0-rc.1"), ("status", "1.14"), ("publish-plan", "1.14.0-rc.1")):
        refused = repo.tool(*command, check=False)
        assert refused.returncode == 1, command
        assert (
            "release/1.14 is on origin but not fetched here:"
            " git remote set-branches --add origin release/1.14 && git fetch origin"
            in refused.stderr
        ), command
        assert "reading the local branch" not in refused.stderr, command
    repo.git("remote", "set-branches", "--add", "origin", "release/1.14")
    repo.git("fetch", "--quiet", "origin")
    done = repo.tool("status", "1.14")
    assert "reading the local branch" not in done.stderr
    assert f"== release/1.14 is at {first[:7]}" in done.stdout
    # The branch is part of the refspec now: when the remote moves (a bump merges), the tool's own fetch follows it.
    repo.tool("bump", "1.14.0-rc.2")
    repo.land_on_release()
    moved = repo.origin_ref("refs/heads/release/1.14")
    assert moved != first
    # The push above came from this clone and moved the remote-tracking ref itself (the branch is in the refspec
    # now): set it back, so only the tool's own fetch can bring it forward, and the read below proves that fetch.
    repo.git("update-ref", "refs/remotes/origin/release/1.14", first)
    assert repo.git("rev-parse", "refs/remotes/origin/release/1.14") == first
    repo.git("switch", "--quiet", "main")
    done = repo.tool("status", "1.14")
    assert f"== release/1.14 is at {moved[:7]}" in done.stdout
    assert (
        "== versions: 1.14.0-rc.2" in done.stdout and "reading the local branch" not in done.stderr
    )


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
    assert repo.git("rev-parse", "--abbrev-ref", "HEAD") == "main"
    for path in VERSION_FILES:
        assert repo.version_of(path, "release/1.16") == "1.16.0-rc.1", path


def test_a_patch_of_the_older_line_steps_past_the_version_the_newer_line_published(
    repo: Sandbox,
) -> None:
    """Two live lines share tool-parser's versions. 1.15's candidate ships the crate at 0.6.1 (two
    fixes on main since v1.14.0); one fix picked into release/1.14 afterwards is bumped against
    that line's own tag, where the crate reads 0.6.0. The plain patch, 0.6.1, is on crates.io
    from the candidate: the bump steps past it to 0.6.2 and says so. Then release-bump-main for
    the patch moves main's tool-parser to 0.6.2, a version published with the other line's
    code; the next cut, which finds no change since v1.15.0, bumps it past that too (0.6.3),
    so the 1.16 line never ships main's code under the 1.14 line's number."""
    repo.cut()  # 1.14.0-rc.1: tool-parser 0.6.0 for the feat on main since v1.13.0
    repo.tool("tag", "1.14.0-rc.1", PUSH="1")
    repo.tool("bump", "1.14.0")
    repo.land_on_release()
    repo.tool("tag", "1.14.0", PUSH="1")
    repo.tool("bump-main", "1.14.0")
    repo.git("push", "--quiet", "origin", "HEAD:refs/heads/main")
    fix = repo.commit_main(
        "fix(tool_parser): one (#130)",
        "crates/tool_parser/src/lib.rs",
        "pub fn parse() {}\npub fn more() {}\npub fn one() {}\n",
    )
    repo.commit_main(
        "fix(tool_parser): three (#131)",
        "crates/tool_parser/src/lib.rs",
        "pub fn parse() {}\npub fn more() {}\npub fn one() {}\npub fn three() {}\n",
    )
    repo.cut("1.15.0-rc.1")  # the clone stays on main; the branch carries the bump
    assert repo.version_of("crates/tool_parser/Cargo.toml", "release/1.15") == "0.6.1", (
        "the fixes since v1.14.0"
    )
    repo.tool("tag", "1.15.0-rc.1", PUSH="1")
    repo.tool("pick", "1.14", fix)
    repo.land_on_release("1.14")
    bumped = repo.tool("bump", "1.14.1")
    assert repo.version_of("crates/tool_parser/Cargo.toml") == "0.6.2"
    assert 'tool-parser = { version = "0.6.2"' in repo.read("Cargo.toml")
    assert (
        "tool-parser v0.6.0 → v0.6.2 (crates/tool_parser/Cargo.toml; v0.6.1 is v1.15.0-rc.1's)"
        in (bumped.stdout)
    )
    for path in VERSION_FILES:
        assert repo.version_of(path) == "1.14.1", path
    repo.land_on_release("1.14")
    tagged = repo.tool("tag", "1.14.1", PUSH="1")
    assert "against v1.14.0" in tagged.stderr, "the line's own newest tag"
    assert repo.version_of("crates/tool_parser/Cargo.toml", "v1.14.1") == "0.6.2"
    # main follows the patch: tool-parser 0.6.2 there, with main's code (the 1.15 line's)
    repo.tool("bump-main", "1.14.1")
    repo.git("push", "--quiet", "origin", "HEAD:refs/heads/main")
    assert repo.version_of("crates/tool_parser/Cargo.toml", "origin/main") == "0.6.2"
    repo.tool("bump", "1.15.0")
    repo.land_on_release("1.15")
    repo.tool("tag", "1.15.0", PUSH="1")
    assert repo.version_of("crates/tool_parser/Cargo.toml", "v1.15.0") == "0.6.1"
    repo.tool("bump-main", "1.15.0")
    repo.git("push", "--quiet", "origin", "HEAD:refs/heads/main")
    assert repo.version_of("crates/tool_parser/Cargo.toml", "origin/main") == "0.6.2", (
        "the sync never lowers a version main already carries"
    )
    # the next cut: nothing of tool-parser changed since v1.15.0, but 0.6.2 is v1.14.1's, other code
    cut = repo.tool("cut", "1.16.0-rc.1", PUSH="1")
    assert (
        "tool-parser (crates/tool_parser/) — 1 file(s) changed but version not bumped (v0.6.2, v1.14.1's) [patch]"
        in (cut.stdout)
    )
    assert repo.version_of("crates/tool_parser/Cargo.toml", "release/1.16") == "0.6.3"
    assert 'tool-parser = { version = "0.6.3"' in repo.read("Cargo.toml", "release/1.16")


def grpc_detect(
    tmp_path: Path, pypi: str, packages: str = "auto", ref_type: str = "tag"
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """The detect step of release-grpc.yml over a tree whose two pyproject files read 1.14.0, with a curl that
    answers for PyPI, dispatched on a ref of the given type; the step's result and its output file."""
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
            "GITHUB_REF_TYPE": ref_type,
            "GITHUB_REF_NAME": "v1.14.0" if ref_type == "tag" else "release/1.14",
            "CURL_STUB": pypi,
            "PACKAGES": packages,
        },
        capture_output=True,
        text=True,
    )
    return done, output


def grpc_selection(tmp_path: Path, pypi: str, packages: str = "auto") -> dict[str, str]:
    """The step's selection on a tag dispatch."""
    done, output = grpc_detect(tmp_path, pypi, packages)
    assert done.returncode == 0, done.stderr
    return dict(line.split("=", 1) for line in output.read_text().splitlines())


def go_guard(
    ref_type: str, ref_name: str, version: str = "1.14.0"
) -> subprocess.CompletedProcess[str]:
    """The first step of release-go.yml, run as the workflow runs it: dispatched with the version input, on a ref
    of the given type and name."""
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "release-go.yml").read_text())
    step = workflow["jobs"]["release"]["steps"][0]
    assert step["env"] == {"VERSION": "${{ inputs.version }}"}, (
        "the input reaches the step through env"
    )
    return subprocess.run(
        ["bash", "-eo", "pipefail", "-c", step["run"]],
        env={
            "PATH": os.pathsep.join(["/usr/bin", "/bin"]),
            "GITHUB_REF_TYPE": ref_type,
            "GITHUB_REF_NAME": ref_name,
            "VERSION": version,
        },
        capture_output=True,
        text=True,
    )


def test_the_go_module_tag_is_made_on_a_dispatch_against_a_tag_only() -> None:
    """release-go.yml publishes a Go module tag: no push or pull request runs it (the job used to be skipped by its
    `if`, after a run that did nothing), and its first step refuses a dispatch whose ref is not the tag of the version
    it publishes: a branch moves between a dispatch and a retry, and another release's tag would give the module tag
    that release's code (the proxies keep what they fetched first)."""
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "release-go.yml").read_text())
    assert list(workflow[True]) == ["workflow_dispatch"], "the yaml key `on` reads as True"
    job = workflow["jobs"]["release"]
    assert "if" not in job, "nothing to skip once the dispatch is the only trigger"
    assert "tag" in job["steps"][0]["name"]
    header = (ROOT / ".github" / "workflows" / "release-go.yml").read_text().splitlines()[2]
    assert header.startswith("# Published only on a maintainer's dispatch against a release tag")
    assert go_guard("tag", "v1.14.0").returncode == 0
    branch = go_guard("branch", "release/1.14")
    assert branch.returncode == 1
    assert "release tag" in branch.stderr and "release/1.14 is a branch" in branch.stderr
    other = go_guard("tag", "v1.14.0", version="1.15.0")
    assert other.returncode == 1
    assert "against v1.15.0" in other.stderr and "v1.14.0 is another tag" in other.stderr


def test_the_grpc_release_refuses_a_dispatch_against_a_branch(tmp_path: Path) -> None:
    """A branch can move between a failed upload and its retry, and the retry would complete the version on PyPI with
    the missing file built from another commit; the step refuses a ref that is not a tag before it selects anything."""
    done, output = grpc_detect(tmp_path, "", ref_type="branch")
    assert done.returncode == 1
    assert "release tag" in done.stderr and "release/1.14 is a branch" in done.stderr
    assert not output.exists()


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


def test_make_release_plan_runs_the_checker_quietly() -> None:
    if shutil.which("make") is None:
        pytest.skip("make is not installed")
    done = subprocess.run(
        ["make", "-n", "release-plan", "TAG=v1.13.0", "VERSION=1.14.0-rc.1"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    assert (
        'VERSION_OVERRIDE="1.14.0-rc.1" python3 scripts/check_release_versions.py --plan v1.13.0'
        in done.stdout
    )
    assert "release.sh" not in done.stdout, "the plan reads; nothing of the release tool runs"
    help_text = subprocess.run(["make", "help"], cwd=ROOT, capture_output=True, text=True).stdout
    assert "release-plan" in help_text


def test_make_help_lists_every_target() -> None:
    if shutil.which("make") is None:
        pytest.skip("make is not installed")
    out = subprocess.run(["make", "help"], cwd=ROOT, capture_output=True, text=True).stdout
    for target in MAKE_TARGETS:
        assert target in out, target
