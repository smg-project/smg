"""Tests for scripts/release/release.sh and scripts/release/sync_versions.py.

Every test runs the tool on a throwaway repository built here: an origin (a bare repository)
and a clone that carries the Makefile, the version checker and the release scripts of this
tree, the version files the checker reads, one crate and a few commits past a v1.13.0 tag.
``gh`` and ``cargo`` are stubs on PATH that record their arguments. Nothing here touches the
real repository or the network.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
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
    "release-ship": "ship",
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
      case "${!i}" in
        --body-file) j=$((i + 1)); cp "${!j}" "$GH_STUB_LOG.body" ;;
        --head) j=$((i + 1)); head=${!j} ;;
        --base) j=$((i + 1)); base=${!j} ;;
      esac
    done
    printf '%s %s\n' "${head-}" "${base-}" > "$GH_STUB_LOG.pr"
    echo "https://example.invalid/pull/1" ;;
  "pr list")
    [ -z "${GH_STUB_LIST_ERROR-}" ] || { echo "gh: $GH_STUB_LIST_ERROR" >&2; exit 1; }
    case " $* " in
      *" --head "*) # the open pull requests on a head (--json number --jq .[].number): the cut's and the ship's reads
        for ((i = 1; i <= $#; i++)); do
          if [ "${!i}" = "--head" ]; then j=$((i + 1)); head=${!j}; fi
          if [ "${!i}" = "--base" ]; then j=$((i + 1)); base=${!j}; fi
        done
        [ -z "${base-}" ] || printf '%s %s\n' "$head" "$base" > "$GH_STUB_LOG.pr"
        case " ${GH_STUB_OPEN_HEADS-} " in *" $head "*) echo 8 ;; esac ;;
      *) echo "#7  fix(router): y  pick/1.14-0000000" ;;
    esac ;;
  "pr view") # the bump pull request the ship waits for; GH_STUB_MERGE=1 merges it (origin's base moves to its head)
    if [ "${GH_STUB_MERGE-}" = 1 ] && [ -s "$GH_STUB_LOG.pr" ]; then
      read -r head base < "$GH_STUB_LOG.pr"
      git -C "$GH_STUB_ORIGIN" update-ref "refs/heads/$base" "refs/heads/$head"
      echo MERGED
    else
      echo "${GH_STUB_PR_STATE:-OPEN}"
    fi ;;
  "api user") [ -z "${GH_STUB_AUTH_ERROR-}" ] || { echo "gh: HTTP 401: Bad credentials" >&2; exit 1; }; echo 1 ;;
  "api repos/example/repo/branches/"*) echo "${GH_STUB_PROTECTED:-false}" ;;
  "api repos/example/repo/rules/branches/"*) [ -z "${GH_STUB_RULES-}" ] || printf '%b\n' "$GH_STUB_RULES" ;;
  "api repos/example/repo/rulesets/"*) [ -z "${GH_STUB_BYPASS-}" ] || printf '%b\n' "$GH_STUB_BYPASS" ;;
  "release view")
    case " ${GH_STUB_RELEASES-} " in
      *" $3 "*) echo "https://example.invalid/releases/$3 (prerelease: ${GH_STUB_RELEASE_PRE:-false})" ;;
      *) echo "release not found" >&2; exit 1 ;;
    esac ;;
  "release create") echo "https://example.invalid/releases/$3" ;;
  "workflow run") ;;
  "run list") # the runs on a tag: GH_STUB_TAG_RUNS holds "file;id;title;status;conclusion" lines, --workflow narrows
    [ -z "${GH_STUB_RUN_LIST_ERROR-}" ] || { echo "gh: $GH_STUB_RUN_LIST_ERROR" >&2; exit 1; }
    workflow=""
    for ((i = 1; i <= $#; i++)); do
      if [ "${!i}" = "--workflow" ]; then j=$((i + 1)); workflow=${!j}; fi
    done
    while IFS=';' read -r file id title status conclusion; do
      [ -n "$file" ] || continue
      [ -z "$workflow" ] || [ "$file" = "$workflow" ] || continue
      if [ -n "$workflow" ]; then printf '%s\t%s\t%s\t%s\n' "$id" "$title" "$status" "$conclusion"
      else printf '%s\t%s\t%s\t%s\t%s\n' "$id" "$file" "$status" "$conclusion" "$title"; fi
    done <<< "${GH_STUB_TAG_RUNS-}" ;;
  "repo view") echo "example/repo" ;;
  "api graphql") # the notes' batch read of pull request bodies: GH_STUB_PULLS/<number>.md is a body, no file a null
    [ -z "${GH_STUB_GRAPHQL_ERROR-}" ] || { echo "gh: $GH_STUB_GRAPHQL_ERROR" >&2; exit 1; }
    if [ -n "${GH_STUB_GRAPHQL_NULL_REPO-}" ]; then # a repository the token cannot see: null, an error, exit 1
      echo '{"data":{"repository":null},"errors":[{"type":"NOT_FOUND","message":"Could not resolve to a Repository"}]}'
      echo "gh: Could not resolve to a Repository with the name 'example/repo'. (NOT_FOUND)" >&2
      exit 1
    fi
    python3 - "${GH_STUB_PULLS:-/nonexistent}" "$*" <<'PY'
import json
import re
import sys
from pathlib import Path

pulls, query = Path(sys.argv[1]), sys.argv[2]
data = {}
for alias, number in re.findall(r"(p\d+): pullRequest\(number: (\d+)\)", query):
    body = pulls / f"{number}.md"
    data[alias] = (
        {"number": int(number), "title": "", "body": body.read_text(), "author": {"login": "someone"}}
        if body.is_file()
        else None
    )
print(json.dumps({"data": {"repository": data}}))
PY
    ;;
  "api --paginate")
    [ -z "${GH_STUB_API_ERROR-}" ] || api_error "$GH_STUB_API_ERROR" "${GH_STUB_API_STATUS:-422}"
    # one commit's own check runs when a test sets GH_STUB_CHECKS_<sha> (a bump's parent); else GH_STUB_CHECKS
    sha=${3#*commits/}; sha=${sha%%/*}; of_sha="GH_STUB_CHECKS_$sha"; checks=${!of_sha-${GH_STUB_CHECKS-}}
    printf '%b\n' "${checks:-success a\nsuccess b}" ;;
  "api repos/example/repo/actions/workflows/pr-test-rust.yml/runs?head_sha="*)
    [ -z "${GH_STUB_RUNS_ERROR-}" ] || api_error "$GH_STUB_RUNS_ERROR" 401
    # one commit's own runs when a test sets GH_STUB_RUNS_<sha> (a version-only head's parent); else GH_STUB_RUNS;
    # a line is "<conclusion> <url>", with the run's event as a third word when a test gives one: a pull_request
    # line is dropped when the read selects on the event, as the API would
    sha=${2#*head_sha=}; sha=${sha%%&*}; of_sha="GH_STUB_RUNS_$sha"; runs=${!of_sha-${GH_STUB_RUNS-}}
    case "$runs" in
      none) ;;
      "") echo "success https://example.invalid/runs/1" ;;
      *) printf '%b\n' "$runs" | while read -r conclusion url event; do
           [[ "$event" == pull_request && "$4" == *'.event != "pull_request"'* ]] || echo "$conclusion $url"
         done ;;
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


def interpreter_library_path() -> dict[str, str]:
    """The LD_LIBRARY_PATH of this process, when it has one. The sandboxes below put this
    interpreter's directory on PATH so the tools under test run the same python; on a
    runner whose python is a relocatable tool-cache build, that interpreter finds its
    shared library through LD_LIBRARY_PATH, which an environment built from scratch
    would drop."""
    library_path = os.environ.get("LD_LIBRARY_PATH")
    return {"LD_LIBRARY_PATH": library_path} if library_path else {}


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
            "GH_STUB_ORIGIN": str(self.origin),
            "CARGO_STUB_LOG": str(self.cargo_log),
            "CURL_STUB_LOG": str(self.curl_log),
            "PYTHONDONTWRITEBYTECODE": "1",
            "TMPDIR": str(self.base),
        }
        env.update(interpreter_library_path())
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
        self.put("bindings/golang/go.mod", f"module {GO_MODULE}\n\ngo 1.24.0\n")
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
        for name in ("check_release_versions.py", "ci_version_only.py"):
            shutil.copy(ROOT / "scripts" / name, self.work / "scripts")
        for name in ("release.sh", "sync_versions.py", "notes.py", "NOTES-TEMPLATE.md"):
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
    assert repo.cargo_log.read_text().splitlines() == ["cargo update --workspace --offline"] * 2, (
        "the branch's bump and main's move each refresh the lock"
    )
    (repo.bin / "cargo").unlink()
    repo.git("switch", "--quiet", "main")
    repo.git("branch", "--quiet", "-D", "release/1.14")
    repo.git("branch", "--quiet", "-D", "chore/main-versions-1.15.0-dev")
    done = repo.tool("cut", "1.14.0-rc.1", check=False)
    assert done.returncode == 1 and "Cargo.lock needs a refresh" in done.stderr


def land_main_move(repo: Sandbox, dev: str) -> None:
    """The pull request a cut opened to move main 'merges': origin's main moves to the topic; the clone follows."""
    repo.git("push", "--quiet", "origin", f"refs/heads/chore/main-versions-{dev}:refs/heads/main")
    repo.git("switch", "--quiet", "main")
    repo.git("pull", "--quiet", "--ff-only", "origin", "main")


def test_cut_moves_main_to_the_next_development_version_by_a_pull_request(repo: Sandbox) -> None:
    """The trunk is next: the cut leaves the release branch behind and, with PUSH=1, pushes the topic that moves main
    above it and opens its pull request. main itself moves by that pull request alone."""
    main_head = repo.git("rev-parse", "origin/main")
    done = repo.tool("cut", "1.14.0-rc.1", PUSH="1")
    assert repo.git("rev-parse", "--abbrev-ref", "HEAD") == "main"
    topic = "chore/main-versions-1.15.0-dev"
    assert repo.origin_ref(f"refs/heads/{topic}") == repo.git("rev-parse", topic)
    assert repo.origin_ref("refs/heads/main") == main_head, "main moves by its pull request only"
    assert repo.git("rev-parse", f"{topic}~1") == main_head
    for path in VERSION_FILES:
        assert repo.version_of(path, topic) == "1.15.0-dev", path
        assert repo.version_of(path, "release/1.14") == "1.14.0-rc.1", path
        assert repo.version_of(path) == "1.13.0", f"{path}: the checkout's main is untouched"
    assert repo.version_of("crates/tool_parser/Cargo.toml", topic) == "0.6.0", "the branch's number"
    assert 'tool-parser = { version = "0.6.0"' in repo.read("Cargo.toml", topic)
    message = repo.git("log", "-1", "--format=%B", topic)
    assert message.startswith("chore(release): main reads 1.15.0-dev, above release/1.14\n")
    assert (
        "The gateway reads 1.15.0-dev on main and every crate and package at least the version"
        " release/1.14 carries" in message
    )
    assert "Signed-off-by: Release Owner <owner@example.invalid>" in message
    assert repo.gh_calls()[-1].startswith(
        f"gh pr create --base main --head {topic}"
        " --title chore(release): main reads 1.15.0-dev, above release/1.14"
    )
    body = repo.pr_body()
    assert "`release/1.14` is cut at" in body
    assert "reads 1.14.0-rc.1; main moves above it the same moment" in body
    assert "1.15.0.dev0 once Python tooling normalizes it" in body
    assert (
        "cut: main moves above release/1.14: the gateway and its mirrors to 1.15.0-dev"
        in done.stderr
    )
    # the checker reads the moved main as consistent against the previous final
    repo.git("switch", "--quiet", topic)
    checked = subprocess.run(
        [sys.executable, "scripts/check_release_versions.py", "--check", "v1.13.0"],
        cwd=repo.work,
        env=repo.env(),
        capture_output=True,
        text=True,
    )
    assert checked.returncode == 0, checked.stdout


def test_cut_without_push_leaves_the_move_local_and_prints_its_commands(repo: Sandbox) -> None:
    done = repo.tool("cut", "1.14.0-rc.1")
    topic = "chore/main-versions-1.15.0-dev"
    assert repo.origin_ref("refs/heads/release/1.14") is None
    assert repo.origin_ref(f"refs/heads/{topic}") is None, "the move follows the cut's switch"
    assert repo.gh_calls() == [], "no pull request without the push"
    assert repo.version_of("model_gateway/Cargo.toml", topic) == "1.15.0-dev"
    assert "git push origin refs/heads/release/1.14:refs/heads/release/1.14" in done.stdout
    assert f"git push -u origin refs/heads/{topic}:refs/heads/{topic}" in done.stdout
    assert f"gh pr create --base main --head {topic}" in done.stdout
    assert repo.git("rev-parse", "--abbrev-ref", "HEAD") == "main"


def test_cut_refuses_when_the_move_topic_exists(repo: Sandbox) -> None:
    repo.git("branch", "--quiet", "chore/main-versions-1.15.0-dev", "origin/main")
    refused = repo.tool("cut", "1.14.0-rc.1", check=False)
    assert refused.returncode == 1
    assert "the branch chore/main-versions-1.15.0-dev exists" in refused.stderr
    assert repo.git("branch", "--list", "release/1.14") == "", "nothing was cut"


def nothing_was_cut(repo: Sandbox) -> None:
    """A refusal before anything is made: no release branch here or on the origin, the clone on main and clean,
    no pull request opened."""
    assert repo.git("branch", "--list", "release/1.14") == ""
    assert repo.origin_ref("refs/heads/release/1.14") is None
    assert repo.git("branch", "--show-current") == "main"
    assert repo.git("status", "--porcelain", "--untracked-files=no") == ""
    assert not any(call.startswith("gh pr create") for call in repo.gh_calls())


def test_cut_refuses_when_the_move_topic_exists_on_the_remote(repo: Sandbox) -> None:
    """The local check alone would let a cut push `release/<x.y>` and fail at the topic's push after it."""
    topic = "chore/main-versions-1.15.0-dev"
    repo.git("push", "--quiet", "origin", f"origin/main:refs/heads/{topic}")
    for env in ({}, {"PUSH": "1"}):
        refused = repo.tool("cut", "1.14.0-rc.1", check=False, **env)
        assert refused.returncode == 1, env
        assert f"the branch {topic} exists on origin" in refused.stderr, env
        nothing_was_cut(repo)


def test_a_pushing_cut_refuses_without_gh_before_the_branch_is_pushed(repo: Sandbox) -> None:
    """With PUSH=1 the release branch is pushed before main's move: what the move's pull request needs is
    checked first, not by open_pr once the branch is live."""
    refused = repo.tool("cut", "1.14.0-rc.1", PUSH="1", GH="gh-missing", check=False)
    assert refused.returncode == 1
    assert "'gh-missing' is not installed and PUSH=1 opens the pull request" in refused.stderr
    nothing_was_cut(repo)


def test_a_pushing_cut_shows_ghs_own_error_when_the_pull_request_read_fails(repo: Sandbox) -> None:
    """The read runs before anything is made, so gh's own stderr is the useful text: it is not swallowed."""
    refused = repo.tool(
        "cut",
        "1.14.0-rc.1",
        PUSH="1",
        GH_STUB_LIST_ERROR="could not determine base repo: more than one remote",
        check=False,
    )
    assert refused.returncode == 1
    assert "gh: could not determine base repo: more than one remote" in refused.stderr
    assert "gh pr list failed (its message is above)" in refused.stderr
    assert refused.stderr.index("gh: could not determine") < refused.stderr.index(
        "gh pr list failed"
    )
    nothing_was_cut(repo)


def test_a_pushing_cut_refuses_an_open_pull_request_on_the_move_topic(repo: Sandbox) -> None:
    topic = "chore/main-versions-1.15.0-dev"
    refused = repo.tool("cut", "1.14.0-rc.1", PUSH="1", GH_STUB_OPEN_HEADS=topic, check=False)
    assert refused.returncode == 1
    assert f"an open pull request already carries {topic} (#8)" in refused.stderr
    assert repo.gh_calls() == [
        f"gh pr list --head {topic} --state open --json number --jq .[].number"
    ]
    nothing_was_cut(repo)


def test_a_failed_commit_of_the_move_stops_the_cut_and_leaves_main_untouched(repo: Sandbox) -> None:
    """`move_main` runs under `if !` in the cut and under `|| die` in bump-main, where `set -e` is off: a git step
    that fails must stop the command itself, or the cut ends 0 with the version edits carried back to the start
    branch and the move reported as made. The failure names the move's whole recipe as its finish lines, so the
    way back under the cut (the drop steps, the topic recreated, the lines run) completes the move."""
    topic = "chore/main-versions-1.15.0-dev"
    hook = repo.work / ".git" / "hooks" / "pre-commit"
    hook.write_text(
        "#!/bin/sh\n"
        f'[ "$(git branch --show-current)" = {topic} ] || exit 0\n'
        'echo "pre-commit: refused" >&2\n'
        "exit 1\n"
    )
    hook.chmod(0o755)
    done = repo.tool("cut", "1.14.0-rc.1", PUSH="1", check=False)
    assert done.returncode == 1
    assert (
        "the commit 'chore(release): main reads 1.15.0-dev, above release/1.14' failed:"
        f" the version edits are staged on {topic}. Once the cause is fixed, finish by hand:"
        in done.stderr
    )
    assert done.stderr.count("git add --update && git commit") == 1, (
        "the commit is among the finish lines, once: the message never asks for it twice"
    )
    # the lines open_pr would have printed with PUSH=0, with the body written before the commit
    assert f"  git push -u origin refs/heads/{topic}:refs/heads/{topic}\n" in done.stderr
    body_line = next(line for line in done.stderr.splitlines() if "gh pr create" in line)
    assert body_line.startswith(f"  gh pr create --base main --head {topic} --title chore")
    body_file = Path(body_line.split("--body-file ")[1])
    assert body_file.is_file() and "`release/1.14` is cut at" in body_file.read_text()
    assert (
        f"switch back to where you started and delete {topic} (git branch -D {topic})"
        in done.stderr
    )
    assert "nothing to move" not in done.stdout
    assert repo.origin_ref("refs/heads/release/1.14") == repo.git("rev-parse", "release/1.14"), (
        "the branch was cut and pushed before the move"
    )
    assert repo.origin_ref(f"refs/heads/{topic}") is None, "nothing of the move is pushed"
    assert repo.gh_calls() == [
        f"gh pr list --head {topic} --state open --json number --jq .[].number"
    ], "no pull request is opened"
    assert repo.git("branch", "--show-current") == topic, "the staged edits stay on the topic"
    assert repo.git("rev-parse", topic) == repo.git("rev-parse", "origin/main")
    assert "M  model_gateway/Cargo.toml" in repo.git("status", "--porcelain")
    assert repo.git("rev-parse", "main") == repo.git("rev-parse", "origin/main"), (
        "main is untouched"
    )
    assert repo.version_of("model_gateway/Cargo.toml", "main") == "1.13.0"
    # the finish lines are the move's whole recipe, so the way back the cut's rerun text names completes it: the
    # hook gone, the drop steps, the recreate line, the lines as printed
    lines = finish_lines(done.stderr)
    assert lines[0] == "python3 scripts/release/sync_versions.py refs/heads/release/1.14 1.15.0-dev"
    assert lines[1].startswith("git add --update && git commit --signoff --message chore")
    assert f"or {drop_steps(topic, CUT_RERUN)}" in done.stderr
    hook.unlink()
    repo.git("restore", "--staged", "--worktree", ".")
    repo.git("switch", "--quiet", "main")
    repo.git("branch", "--quiet", "-D", topic)
    assert repo.git("status", "--porcelain") == ""
    for line in [f"git switch --no-track -c {topic} origin/main", *lines]:
        ran = subprocess.run(
            ["bash", "-c", line], cwd=repo.work, env=repo.env(), capture_output=True, text=True
        )
        assert ran.returncode == 0, f"{line}\n{ran.stdout}\n{ran.stderr}"
    assert repo.origin_ref(f"refs/heads/{topic}") == repo.git("rev-parse", topic)
    assert repo.git("rev-parse", f"{topic}~1") == repo.git("rev-parse", "origin/main")
    for path in VERSION_FILES:
        assert repo.version_of(path, topic) == "1.15.0-dev", path
    assert repo.gh_calls()[-1].startswith(f"gh pr create --base main --head {topic}")


def test_a_failed_bump_commit_names_the_finish_lines_and_the_rerun_path(repo: Sandbox) -> None:
    """The same message serves bump: its topic blocks a rerun through has_ref, so the user is told to finish by
    hand (the push and the pull request, with the body written before the commit) or to delete it first."""
    repo.cut()
    topic = "chore/release-1.14.0-rc.2"
    hook = repo.work / ".git" / "hooks" / "pre-commit"
    hook.write_text(
        "#!/bin/sh\n"
        f'[ "$(git branch --show-current)" = {topic} ] || exit 0\n'
        'echo "pre-commit: refused" >&2\n'
        "exit 1\n"
    )
    hook.chmod(0o755)
    repo.git("switch", "--quiet", "main")
    before = repo.gh_calls()
    done = repo.tool("bump", "1.14.0-rc.2", check=False)
    assert done.returncode == 1
    assert (
        "the commit 'chore(release): bump versions for v1.14.0-rc.2' failed:"
        f" the version edits are staged on {topic}." in done.stderr
    )
    assert f"  git push -u origin refs/heads/{topic}:refs/heads/{topic}\n" in done.stderr
    assert f"  gh pr create --base release/1.14 --head {topic} --title chore" in done.stderr
    # the commit is the first finish line and the message asks for it once
    lines = finish_lines(done.stderr)
    assert lines[0].startswith("git add --update && git commit --signoff --message chore")
    assert lines[1] == f"git push -u origin refs/heads/{topic}:refs/heads/{topic}"
    assert done.stderr.count("git add --update && git commit") == 1
    assert (
        f"delete {topic} (git branch -D {topic}): a rerun is refused while it exists" in done.stderr
    )
    assert repo.origin_ref(f"refs/heads/{topic}") is None
    assert repo.gh_calls() == before, "no pull request for the bump"
    assert repo.git("branch", "--show-current") == topic
    assert "M  model_gateway/Cargo.toml" in repo.git("status", "--porcelain")


def drop_steps(topic: str, rerun: str) -> str:
    """The second way out every failure of a move names once its topic exists: the text `commit_all` prints, then
    what the command offers as a rerun."""
    return (
        f"drop them (git restore --staged --worktree .), switch back to where you started and delete {topic}"
        f" (git branch -D {topic}): {rerun}"
    )


CUT_RERUN = (
    "a rerun of the cut is refused while release/1.14 exists (it stays, pushed or not): to make"
    " the move later, recreate the topic (git switch --no-track -c chore/main-versions-1.15.0-dev"
    " origin/main) and run the lines above"
)
MAIN_RERUN = "then 'release.sh bump-main 1.14.0' again"
# the sandbox's sync_versions.py: the real one, or a tool that stops half way while a marker file exists
SYNC_STUB = (
    "#!/usr/bin/env python3\n"
    "import pathlib\n"
    "import runpy\n"
    "import sys\n"
    "\n"
    "root = pathlib.Path(__file__).resolve().parents[2]\n"
    "if (root / '.sync-stops').exists():\n"
    "    manifest = root / 'model_gateway' / 'Cargo.toml'\n"
    "    manifest.write_text(manifest.read_text().replace('1.13.0', '1.15.0-dev'))\n"
    "    sys.exit('sync_versions: FAILED to set bindings/python/pyproject.toml to 1.15.0-dev')\n"
    "sys.argv[0] = str(root / 'scripts' / 'release' / 'sync_versions.real.py')\n"
    "runpy.run_path(sys.argv[0], run_name='__main__')\n"
)


def finish_lines(stderr: str) -> list[str]:
    """The lines between 'finish by hand' and 'or drop them' of a failure message: what the owner runs."""
    lines = stderr.splitlines()
    start = next(i for i, line in enumerate(lines) if "finish by hand" in line)
    end = next(i for i, line in enumerate(lines) if i > start and line.startswith("or drop them"))
    return [line.strip() for line in lines[start + 1 : end]]


def test_a_failed_sync_of_the_move_names_the_finish_lines_and_they_complete_it(
    repo: Sandbox,
) -> None:
    """`move_main` sits on its topic when `sync_versions.py` runs: a failure there leaves what it moved before it
    stopped on the topic. Under the cut nothing reruns the move (the release branch is made, and pushed with PUSH=1),
    so the message names the lines that finish it by hand, the pull request's with its body among them; run once the
    cause is gone, they complete the move."""
    topic = "chore/main-versions-1.15.0-dev"
    repo.put("scripts/release/sync_versions.real.py", repo.read("scripts/release/sync_versions.py"))
    repo.commit_main(
        "chore: a sync that stops while a marker exists",
        "scripts/release/sync_versions.py",
        SYNC_STUB,
    )
    (repo.work / ".sync-stops").touch()
    done = repo.tool("cut", "1.14.0-rc.1", PUSH="1", check=False)
    assert done.returncode == 1
    assert "sync_versions: FAILED to set bindings/python/pyproject.toml" in done.stderr
    assert (
        f"sync_versions.py failed: the version files of {topic} are moved in part at most."
        " Once the cause is fixed, finish by hand (a rerun of the sync completes what it started):"
        in done.stderr
    )
    assert f"or {drop_steps(topic, CUT_RERUN)}" in done.stderr
    assert repo.git("branch", "--show-current") == topic
    assert repo.git("diff", "--name-only") == "model_gateway/Cargo.toml", (
        "the stub's edit, unstaged"
    )
    assert repo.origin_ref("refs/heads/release/1.14") is not None, "the cut's first half is done"
    assert repo.origin_ref(f"refs/heads/{topic}") is None
    lines = finish_lines(done.stderr)
    assert lines[0] == "python3 scripts/release/sync_versions.py refs/heads/release/1.14 1.15.0-dev"
    assert "cargo update --workspace" not in lines, "no lock file in this tree"
    assert lines[1].startswith("git add --update && git commit --signoff --message chore")
    assert lines[2] == f"git push -u origin refs/heads/{topic}:refs/heads/{topic}"
    assert lines[3].startswith(f"gh pr create --base main --head {topic} --title chore")
    body_file = Path(lines[3].split("--body-file ")[1])
    assert "`release/1.14` is cut at" in body_file.read_text(), "the body exists with its intro"
    # the cause is gone: the printed lines, run as printed, complete the move
    (repo.work / ".sync-stops").unlink()
    for line in lines:
        ran = subprocess.run(
            ["bash", "-c", line], cwd=repo.work, env=repo.env(), capture_output=True, text=True
        )
        assert ran.returncode == 0, f"{line}\n{ran.stdout}\n{ran.stderr}"
    assert repo.origin_ref(f"refs/heads/{topic}") == repo.git("rev-parse", topic)
    assert repo.git("rev-parse", f"{topic}~1") == repo.git("rev-parse", "origin/main")
    for path in VERSION_FILES:
        assert repo.version_of(path, topic) == "1.15.0-dev", path
    assert repo.version_of("crates/tool_parser/Cargo.toml", topic) == "0.6.0"
    assert repo.gh_calls()[-1].startswith(f"gh pr create --base main --head {topic}")


def test_a_failed_lock_refresh_of_the_move_names_the_finish_lines_and_the_drop_steps(
    repo: Sandbox,
) -> None:
    """The lock refresh runs once the sync moved the version files: its own `die` names the same ways out, the lock
    line among the finish lines. The drop steps clean the clone, and under the cut that is where they lead: the
    release branch stays and refuses the cut's rerun, as the message says; the way back it names (the topic
    recreated, the lines above run once the cause is gone) completes the move."""
    topic = "chore/main-versions-1.15.0-dev"
    repo.commit_main("chore: a lock file", "Cargo.lock", "# lock\n")
    # the branch's refresh passes, the move's fails
    failing = (
        f'[ "$(git branch --show-current)" = {topic} ] || exit 0\n'
        'echo "error: failed to select a version for the requirement" >&2\n'
        "exit 101\n"
    )
    (repo.bin / "cargo").write_text(CARGO_STUB + failing)
    done = repo.tool("cut", "1.14.0-rc.1", PUSH="1", check=False)
    assert done.returncode == 1
    assert "error: failed to select a version" in done.stderr, "cargo's own reason stays visible"
    assert (
        "cargo update --workspace failed: Cargo.lock does not follow the versions;"
        f" the version edits sit on {topic}. Once the cause is fixed, finish by hand:"
        in done.stderr
    )
    assert finish_lines(done.stderr)[:2] == [
        "python3 scripts/release/sync_versions.py refs/heads/release/1.14 1.15.0-dev",
        "cargo update --workspace",
    ]
    assert f"or {drop_steps(topic, CUT_RERUN)}" in done.stderr
    assert repo.git("branch", "--show-current") == topic
    assert "model_gateway/Cargo.toml" in repo.git("diff", "--name-only").splitlines(), (
        "the sync's edits"
    )
    assert repo.origin_ref(f"refs/heads/{topic}") is None
    # the drop steps leave a clean clone on main; the cut's rerun is refused by the release branch, as said
    repo.git("restore", "--staged", "--worktree", ".")
    repo.git("switch", "--quiet", "main")
    repo.git("branch", "--quiet", "-D", topic)
    assert repo.git("status", "--porcelain") == ""
    rerun = repo.tool("cut", "1.14.0-rc.1", PUSH="1", check=False)
    assert rerun.returncode == 1 and "release/1.14 exists locally" in rerun.stderr
    # the cause gone, the way back the message names: the topic recreated, the lines above run as printed
    (repo.bin / "cargo").write_text(CARGO_STUB)
    recreate = f"git switch --no-track -c {topic} origin/main"
    assert f"recreate the topic ({recreate}) and run the lines above" in done.stderr
    for line in [recreate, *finish_lines(done.stderr)]:
        ran = subprocess.run(
            ["bash", "-c", line], cwd=repo.work, env=repo.env(), capture_output=True, text=True
        )
        assert ran.returncode == 0, f"{line}\n{ran.stdout}\n{ran.stderr}"
    assert repo.origin_ref(f"refs/heads/{topic}") == repo.git("rev-parse", topic)
    assert repo.git("rev-parse", f"{topic}~1") == repo.git("rev-parse", "origin/main")
    for path in VERSION_FILES:
        assert repo.version_of(path, topic) == "1.15.0-dev", path
    assert repo.gh_calls()[-1].startswith(f"gh pr create --base main --head {topic}")


def test_a_failed_body_file_of_the_move_names_the_topic_to_delete(repo: Sandbox) -> None:
    """The body file is made first, before any edit: its failure leaves a clean topic, the finish lines name a body
    file of the owner's own, and bump-main's message names its own rerun."""
    repo.cut()
    topic = "chore/main-versions-1.14.0"
    done = repo.tool("bump-main", "1.14.0", TMPDIR=str(repo.base / "missing"), check=False)
    assert done.returncode == 1
    assert (
        f"cannot make a temporary file for the pull request body of {topic} (nothing is edited yet)."
        " Once the cause is fixed, finish by hand, with a body file of your own (body.md below):"
        in done.stderr
    )
    lines = [line.strip() for line in done.stderr.splitlines() if line.startswith("  ")]
    assert lines[0] == "python3 scripts/release/sync_versions.py origin/release/1.14 1.15.0-dev"
    assert lines[-1].startswith(f"gh pr create --base main --head {topic} --title chore")
    assert lines[-1].endswith("--body-file body.md")
    assert (
        f"or switch back to where you started and delete {topic} (git branch -D {topic}): {MAIN_RERUN}"
        in done.stderr
    )
    assert repo.git("branch", "--show-current") == topic
    assert repo.git("status", "--porcelain") == "", "nothing is edited yet"
    assert repo.origin_ref(f"refs/heads/{topic}") is None


def test_a_failed_push_of_the_move_names_the_lines_that_finish_it(repo: Sandbox) -> None:
    """After the commit, a failed push (the topic taken on the remote by another history here) names the push and
    the pull request lines, the body file the tool wrote among them."""
    repo.cut()
    topic = "chore/main-versions-1.14.0"
    repo.git("push", "--quiet", "origin", f"refs/heads/release/1.14:refs/heads/{topic}")
    before = repo.gh_calls()
    done = repo.tool("bump-main", "1.14.0", check=False)
    assert done.returncode == 1
    assert (
        f"the push of {topic} to origin failed; nothing opened. Once the cause is fixed, finish by hand:"
        in done.stderr
    )
    assert f"  git push -u origin refs/heads/{topic}:refs/heads/{topic}\n" in done.stderr
    body_line = next(line for line in done.stderr.splitlines() if "gh pr create" in line)
    assert body_line.startswith(f"  gh pr create --base main --head {topic} --title chore")
    body = Path(body_line.split("--body-file ")[1]).read_text()
    assert "After the v1.14.0 release" in body and "model_gateway/Cargo.toml" in body
    assert repo.gh_calls() == before, "no pull request is opened"
    assert repo.git("branch", "--show-current") == topic
    assert repo.git("log", "-1", "--format=%s") == "chore(release): main's versions follow v1.14.0"


def test_a_move_without_gh_names_the_lines_to_run_by_hand(repo: Sandbox) -> None:
    """bump-main runs no pull-request precondition: without `gh` the move is committed and its message names the
    push and the pull request lines."""
    repo.cut()
    topic = "chore/main-versions-1.14.0"
    done = repo.tool("bump-main", "1.14.0", GH="gh-missing", check=False)
    assert done.returncode == 1
    assert (
        f"'gh-missing' is not installed: push {topic} and open the pull request against main by hand:"
        in done.stderr
    )
    assert f"  git push -u origin refs/heads/{topic}:refs/heads/{topic}\n" in done.stderr
    assert f"  gh pr create --base main --head {topic} --title chore" in done.stderr
    assert repo.origin_ref(f"refs/heads/{topic}") is None


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
    assert not [call for call in repo.gh_calls() if "chore/release-1.14.0" in call], (
        "no pull request without the push (the cut's own, moving main, is the only call)"
    )
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
    assert not [call for call in repo.gh_calls() if "pick/" in call], (
        "no pull request for an unresolved pick (the cut's own, moving main, is the only call)"
    )
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
    does not read above the release yet (the move its cut opened never merged) would ship nothing for the crate,
    and nothing would say so."""
    finished_release(repo)  # v1.14.0 on its branch; main still reads 1.13.0
    repo.commit_main(
        "fix(tool_parser): after the release (#125)",
        "crates/tool_parser/src/lib.rs",
        "pub fn x() {}\npub fn after() {}\n",
    )
    refused = repo.tool("cut", "1.15.0-rc.1", check=False)
    assert refused.returncode == 1
    assert (
        "origin/main reads 1.13.0, below 1.14.0: merge the pull request that moves main's versions first"
        " (release.sh bump-main 1.14.0 opens it)" in refused.stderr
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
    changed = notes[notes.index("## What's Changed") :]
    assert "- fix(router): three (#123)\n" in changed and "- fix(router): four (#126)\n" in changed
    assert "chore(release): pick 2 commits" not in notes
    assert changed.index("### Fixes") < changed.index("- fix(router): three (#123)")
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
    assert out.startswith("# Shepherd Model Gateway v1.14.0\n")
    assert "commits since v1.13.0 on `release/1.14`" in out
    assert "### Features\n\n- feat(tool_parser): more\n" in out
    assert "### Fixes\n\n- fix(router): three (#123)\n- fix(router): two\n" in out
    assert "chore(release): bump versions" not in out
    assert "### Docker images\n\n- stub images for v1.14.0" in out
    assert "gh release create v1.14.0 --verify-tag --title 'v1.14.0'" in done.stderr, (
        "the tag alone is the title"
    )
    assert "### Storage" not in out, (
        "no pull request declares a storage change: no Storage subsection"
    )
    assert "--prerelease" not in done.stderr
    candidate = repo.tool("notes", "1.14.0-rc.1")
    assert (
        "gh release create v1.14.0-rc.1 --verify-tag --prerelease --title 'v1.14.0-rc.1'"
        in candidate.stderr
    )
    since = repo.tool("notes", "1.14.0", "v1.14.0-rc.1").stdout
    assert "fix(router): two" not in since and "fix(router): three (#123)" in since


# A dozen-and-three merged pull requests, one per bucket of the notes: the squash subjects and the bodies GH_STUB_PULLS
# serves (None: a number the repository has no pull request for).
NOTED_PULLS: tuple[tuple[str, str | None], ...] = (
    (
        "feat(health): the readiness body names each served model (#201)",
        "## Description\n\nx\n\n## Release note\n\nThe readiness body names each served model with a reason.\n\n"
        "## Numbers\n\n| a | b |\n",
    ),
    (
        "perf(tokenizer): native encode for byte-level vocabularies (#202)",
        "## Release note\n\nEncoding a 32k-token prompt takes a sixth of the CPU it took.\n",
    ),
    (
        "feat(symphony): Llama 3.1 reads with the json table (#203)",
        "## Release notes\n\n- Symphony reads the Llama 3.1 tool syntax.\n- Under its own id.\n",
    ),
    (
        "fix(servicer): the start deadline follows the engine's liveness (#204)",
        "**Release note:** an engine that takes an hour to load no longer restarts from zero.\n\nMore text.\n",
    ),
    (
        "fix(multimodal): an image to a text-only worker is refused up front (#205)",
        "## Release note\n\nAn image sent to a model without a vision encoder is refused with 400.\n",
    ),
    (
        "fix(metrics): the KV event lag is a histogram (#206)",
        "## Release note\n\n`smg_kv_event_lag_seconds` is a histogram;"
        " the `smg_kv_event_tail_depth` metric is removed.\n",
    ),
    (
        "feat(python): Router takes parser (#207)",
        "## Release note\n\n`Router(...)` takes `parser=`.\n",
    ),
    (
        "feat(helm)!: the router runs as UID 65532 (#208)",
        "## Release note\n\nThe chart runs the router as UID 65532.\n\n## Upgrade\n\n"
        "BREAKING CHANGE: a custom image must run as that user.\n",
    ),
    (
        "fix(config): a malformed control-plane key list is a start-up error (#209)",
        "## Release note\n\nA malformed `--control-plane-api-keys` entry refuses to start instead of being dropped.\n",
    ),
    (
        "feat(gateway): admission is bounded by default (#210)",
        "## Release note\n\nAn unset `--max-concurrent-requests` now defaults to 1,024 in-flight requests per core.\n",
    ),
    ("docs(readme): the upgrade page (#211)", "## Release note\n\nnone\n"),
    (
        "fix(router): the retry charge is one per request (#212)",
        "## Description\n\nno release-note section here\n",
    ),
    ("chore(ci): pin the runner image (#213)", None),
    (
        "fix(storage): the memory backend is bounded (#214)",
        "## Release note\n\nThe memory history backend is bounded at 10,000 responses.\n\n"
        "## Compatibility\n\nBREAKING CHANGE: an evicted id answers 404.\n",
    ),
    ("feat(gateway): a test-only change (#215)", "## Release note\n\nNone: a test-only change.\n"),
    (
        "fix(gateway): a fix whose note was left blank (#216)",
        "## Description\n\nx\n\n## Release note\n\n<!-- one or two sentences on what a user gets -->\n\n"
        "## Test Plan\n\ny\n",
    ),
)
SKELETON = (
    "# Shepherd Model Gateway v1.14.0",
    "## Before you upgrade",
    "## Highlights",
    "### How this release was validated",
    "## Notable fixes",
    "## Upgrade notes",
    "### Storage",
    "### Configuration that now refuses to start",
    "### Changed defaults",
    "### API behavior",
    "### Metrics",
    "### Engine side",
    "### Bindings and Rust crates",
    "## Packages",
    "## Known limits",
    "## What's Changed",
)


def noted_release(
    repo: Sandbox, pulls: Path, extra: tuple[tuple[str, str | None], ...] = ()
) -> None:
    """The sixteen pull requests merged on main (and extra ones after them), their bodies where GH_STUB_PULLS
    serves them, v1.14.0 tagged."""
    pulls.mkdir()
    for index, (subject, body) in enumerate(NOTED_PULLS + extra):
        repo.commit_main(subject, f"model_gateway/src/change_{index}.rs", f"// {index}\n")
        if body is not None:
            (pulls / f"{subject.rsplit('#', 1)[1].rstrip(')')}.md").write_text(body)
    repo.cut()
    repo.tool("bump", "1.14.0")
    repo.land_on_release()
    repo.tool("tag", "1.14.0", PUSH="1")


def section(text: str, heading: str, until: str) -> str:
    return text[text.index(heading) : text.index(until)]


def subsection(text: str, heading: str) -> str:
    """A ### subsection: from its heading to the next heading of any level, or to the unsorted candidates."""
    start = text.index(heading) + len(heading)
    rest = text[start:]
    next_heading = re.search(
        r"^#{2,3} |^<!-- AUTHOR: candidates with no obvious subsection", rest, re.M
    )
    return rest[: next_heading.start()] if next_heading else rest


def load_notes_module():
    spec = importlib.util.spec_from_file_location(
        "release_notes", ROOT / "scripts" / "release" / "notes.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["release_notes"] = (
        module  # the dataclass resolves its string annotations through sys.modules
    )
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_notes_render_the_template_seeded_by_the_pull_requests_release_notes(
    repo: Sandbox, tmp_path: Path
) -> None:
    pulls = tmp_path / "pulls"
    noted_release(repo, pulls)
    done = repo.tool("notes", "1.14.0", GH_STUB_PULLS=str(pulls))
    out = done.stdout
    headings = [line for line in out.splitlines() if line.startswith("#")]
    assert tuple(line for line in headings if line in SKELETON) == SKELETON, (
        "every section, in the published order"
    )
    assert "{{" not in out
    assert "chore(release): bump versions" not in out
    highlights = section(out, "## Highlights", "### How this release was validated")
    assert "**Gateway**\n\n" in highlights
    assert "- The readiness body names each served model with a reason. (#201)\n" in highlights
    assert (
        "- An unset `--max-concurrent-requests` now defaults to 1,024 in-flight requests per core. (#210)"
        in highlights
    )
    assert (
        "**Tokenizer**\n\n- Encoding a 32k-token prompt takes a sixth of the CPU it took. (#202)"
        in highlights
    )
    assert "- Symphony reads the Llama 3.1 tool syntax. Under its own id. (#203)" in highlights, (
        "the list joined"
    )
    assert "**Bindings and crates**\n\n- `Router(...)` takes `parser=`. (#207)" in highlights
    assert "**Helm and deploy**\n\n- The chart runs the router as UID 65532. (#208)" in highlights
    assert (
        "- feat(tool_parser): more <!-- AUTHOR: no pull request number in the subject: write the sentence or drop"
        in highlights
    )
    assert "#215" not in highlights, "a note that says none seeds nothing"
    assert "#211" not in highlights, "documentation seeds nothing"
    assert "#204" not in highlights and "#212" not in highlights, "fixes are not highlights"
    fixes = section(out, "## Notable fixes", "## Upgrade notes")
    assert "These bugs shipped in v1.13.0." in fixes
    assert (
        "- fix(router): the retry charge is one per request (#212) <!-- AUTHOR: no release note in the pull request:"
        " write the sentence or drop the line -->" in fixes
    )
    assert (
        "**Servicers and engines**\n\n- an engine that takes an hour to load no longer restarts from zero. (#204)"
        in fixes
    )
    assert (
        "**Multimodal**\n\n- An image sent to a model without a vision encoder is refused with 400. (#205)"
        in fixes
    )
    assert "**Metrics and observability**\n\n- `smg_kv_event_lag_seconds` is a histogram;" in fixes
    assert "- fix(router): two <!-- AUTHOR: no pull request number in the subject" in fixes
    assert (
        "- fix(gateway): a fix whose note was left blank (#216) <!-- AUTHOR: the release-note section is empty:"
        " write the sentence or drop the line -->" in fixes
    ), "an empty section is a marked subject, never a silent drop"
    assert "#201" not in fixes
    changed = out[out.index("## What's Changed") :]
    assert "commits since v1.13.0 on `release/1.14` (the merged pull requests, by type):" in changed
    assert (
        "### Features\n\n" in changed and "- feat(gateway): a test-only change (#215)\n" in changed
    )
    assert "### Documentation\n\n- docs(readme): the upgrade page (#211)\n" in changed
    assert "### Maintenance\n\n- chore(ci): pin the runner image (#213)\n" in changed
    assert (
        "18 pull requests: 11 carry a release note, 2 declare none, 1 leave the section empty, 1 have no"
        " release-note section, 1 could not be read, 2 subjects name no pull request" in done.stderr
    )
    assert "release.sh notes --check <file>" in done.stderr
    assert (
        "gh release create v1.14.0 --verify-tag --title 'v1.14.0' --notes-file <file>"
        in done.stderr
    )


def test_notes_seed_upgrade_notes_from_what_the_pull_requests_declare(
    repo: Sandbox, tmp_path: Path
) -> None:
    pulls = tmp_path / "pulls"
    noted_release(repo, pulls)
    out = repo.tool("notes", "1.14.0", GH_STUB_PULLS=str(pulls)).stdout
    upgrade = section(out, "## Upgrade notes", "## Packages")
    declared = "<!-- AUTHOR: declared: {}; keep it here, move it or drop it -->"
    assert "- The memory history backend is bounded at 10,000 responses. (#214) " + declared.format(
        "a breaking change"
    ) in subsection(upgrade, "### Storage")
    assert "(#209) " + declared.format("a configuration that now refuses to start") in subsection(
        upgrade, "### Configuration that now refuses to start"
    )
    assert "(#210) " + declared.format("a changed default") in subsection(
        upgrade, "### Changed defaults"
    )
    assert "(#206) " + declared.format("a removed or renamed flag or setting") in subsection(
        upgrade, "### Metrics"
    )
    empty = "<!-- AUTHOR: nothing declared by the pull requests under this heading: write it or delete the heading. -->"
    for heading in ("### API behavior", "### Engine side", "### Bindings and Rust crates"):
        assert subsection(upgrade, heading).strip() == empty, heading
    unsorted = upgrade[upgrade.index("<!-- AUTHOR: candidates with no obvious subsection") :]
    assert (
        "- The chart runs the router as UID 65532. (#208) " + declared.format("a breaking change")
        in unsorted
    )
    assert "#201" not in upgrade and "#205" not in upgrade, "nothing declared, nothing seeded"


def test_notes_name_the_packages_and_engines_at_the_tag(repo: Sandbox, tmp_path: Path) -> None:
    pulls = tmp_path / "pulls"
    noted_release(repo, pulls)
    out = repo.tool("notes", "1.14.0", GH_STUB_PULLS=str(pulls)).stdout
    packages = section(out, "## Packages", "## Known limits")
    assert (
        "- `smg` 1.14.0 on PyPI and crates.io, with container images and Helm chart 1.14.0. Release images build on"
        " vLLM (vllm/vllm-openai:v0.31.0 and vllm/vllm-openai:v0.27.1), SGLang (lmsysorg/sglang:v0.5.21) and"
        " TokenSpeed (the latest nightly carrier).\n" in packages
    )
    assert (
        "- `smg-grpc-servicer` 0.14.0. <!-- AUTHOR: the upgrade order between gateways and engine pods"
        in packages
    )
    assert (
        "- The library crates moved to new versions: `tool-parser` 0.6.0 (from 0.5.0). <!-- AUTHOR: name"
        in packages
    )
    assert "Building from source" not in packages, "no rust-version in this tree: no line"
    assert "### Docker images\n\n- stub images for v1.14.0" in packages
    assert "[Full changelog](https://github.com/example/repo/compare/v1.13.0...v1.14.0)" in out


def test_notes_keep_what_a_sentence_that_reads_as_none_declares(
    repo: Sandbox, tmp_path: Path
) -> None:
    """A conditional sentence ("as long as") is a release note with its declaration; one that reads as none and
    goes on ("No behavior change, but the default moves") seeds no candidate line, and what it declares reaches
    Upgrade notes in its own words."""
    pulls = tmp_path / "pulls"
    conditional = "No user-visible change as long as `--foo` is unset; with it set, the gateway now refuses to start."
    none_but = "No behavior change, but the default of `--x` moves to 5."
    noted_release(
        repo,
        pulls,
        (
            (
                "fix(gateway): a set `--foo` needs its peer (#217)",
                f"## Release note\n\n{conditional}\n",
            ),
            ("feat(gateway): the `--x` default moves (#218)", f"## Release note\n\n{none_but}\n"),
            ("feat(gateway)!: remove the `--legacy-foo` flag (#219)", "## Release note\n\nnone\n"),
        ),
    )
    out = repo.tool("notes", "1.14.0", GH_STUB_PULLS=str(pulls)).stdout
    declared = "<!-- AUTHOR: declared: {}; keep it here, move it or drop it -->"
    assert f"- {conditional} (#217)" in section(out, "## Notable fixes", "## Upgrade notes")
    assert "(#218)" not in section(out, "## Highlights", "## Upgrade notes"), "it reads as none"
    upgrade = section(out, "## Upgrade notes", "## Packages")
    assert f"- {conditional} (#217) " + declared.format(
        "a configuration that now refuses to start"
    ) in subsection(upgrade, "### Configuration that now refuses to start")
    assert "- feat(gateway): the `--x` default moves (#218) " + declared.format(
        f'a changed default; the release note says "{none_but}"'
    ) in subsection(upgrade, "### Changed defaults"), (
        "the subject leads, the sentence rides in the comment"
    )
    assert "- feat(gateway)!: remove the `--legacy-foo` flag (#219) " + declared.format(
        'a breaking change, a removed or renamed flag or setting; the release note says "none"'
    ) in subsection(upgrade, "### API behavior"), "a bare none: the subject says what broke"
    assert "(#219)" not in section(out, "## Highlights", "## Upgrade notes")


def test_notes_read_each_pull_request_body_once(repo: Sandbox, tmp_path: Path) -> None:
    pulls = tmp_path / "pulls"
    noted_release(repo, pulls)
    first = repo.tool("notes", "1.14.0", GH_STUB_PULLS=str(pulls)).stdout
    reads = [call for call in repo.gh_calls() if call.startswith("gh api graphql")]
    assert (
        len(reads) == 1 and "p201: pullRequest(number: 201)" in reads[0] and "p215:" in reads[0]
    ), "one batch"
    second = repo.tool("notes", "1.14.0", GH_STUB_PULLS=str(pulls)).stdout
    assert second == first
    assert len([call for call in repo.gh_calls() if call.startswith("gh api graphql")]) == 1, (
        "the cache answered"
    )


def test_notes_render_without_the_bodies_when_gh_cannot_read_them(
    repo: Sandbox, tmp_path: Path
) -> None:
    pulls = tmp_path / "pulls"
    noted_release(repo, pulls)
    offline = repo.tool(
        "notes",
        "1.14.0",
        GH_STUB_PULLS=str(pulls),
        GH_STUB_GRAPHQL_ERROR="boom",
        NOTES_CACHE=str(tmp_path / "c1"),
    )
    assert "the read of 16 pull request bodies failed (gh: boom)" in offline.stderr
    assert "0 carry a release note" in offline.stderr
    assert "16 of them fall back to their subjects" in offline.stderr
    assert (
        "- feat(health): the readiness body names each served model (#201) <!-- AUTHOR: no release note in the pull"
        " request: write the sentence or drop the line -->" in offline.stdout
    )
    assert (
        "[Full changelog](https://github.com/example/repo/compare/v1.13.0...v1.14.0)"
        in offline.stdout
    )
    without_gh = repo.tool(
        "notes", "1.14.0", GH="/nonexistent/gh", NOTES_CACHE=str(tmp_path / "c2")
    )
    assert "the repository is unknown" in without_gh.stderr
    assert (
        "<!-- AUTHOR: the compare link of the previous final and this tag -->" in without_gh.stdout
    )
    assert "## What's Changed" in without_gh.stdout


def test_notes_check_refuses_a_placeholder_and_passes_a_filled_draft(
    repo: Sandbox, tmp_path: Path
) -> None:
    pulls = tmp_path / "pulls"
    noted_release(repo, pulls)
    out = repo.tool("notes", "1.14.0", GH_STUB_PULLS=str(pulls)).stdout
    draft = tmp_path / "notes.md"
    draft.write_text(out)
    refused = repo.tool("notes", "--check", str(draft), check=False)
    assert refused.returncode == 1
    assert (
        "still carries" in refused.stderr
        and "author comment(s), 0 template token(s)" in refused.stderr
    )
    assert "  line 3: <!-- AUTHOR: the opening paragraph" in refused.stderr
    filled = re.sub(r"<!--\s*AUTHOR:.*?-->", "written", out, flags=re.S)
    draft.write_text(filled)
    assert "carries no placeholder" in repo.tool("notes", "--check", str(draft)).stderr
    draft.write_text(filled + "\n{{highlights}}\n")
    token = repo.tool("notes", "--check", str(draft), check=False)
    assert token.returncode == 1 and "0 author comment(s), 1 template token(s)" in token.stderr
    draft.write_text("")
    assert "is empty" in repo.tool("notes", "--check", str(draft), check=False).stderr
    missing = repo.tool("notes", "--check", str(tmp_path / "nope.md"), check=False)
    assert missing.returncode == 1 and "no such file" in missing.stderr
    usage = repo.tool("notes", "--check", check=False)
    assert usage.returncode == 1 and "usage: release.sh notes --check <file>" in usage.stderr
    draft.write_text(
        filled + "\n- none because tests only (#5)\n"
    )  # a candidate line that says none
    none_line = repo.tool("notes", "--check", str(draft), check=False)
    assert none_line.returncode == 1 and "1 line(s) that say none" in none_line.stderr


def test_notes_refuse_a_template_token_the_render_does_not_fill(repo: Sandbox) -> None:
    finished_release(repo)
    repo.put("scripts/release/NOTES-TEMPLATE.md", "# v{{version}}\n\n{{nope}}\n")
    refused = repo.tool("notes", "1.14.0", check=False)
    assert refused.returncode == 1
    assert "the template names {{nope}}, which the render does not fill" in refused.stderr
    assert "the draft was not rendered" in refused.stderr and refused.stdout == ""


def test_the_release_note_of_a_pull_request_body() -> None:
    notes = load_notes_module()
    assert (
        notes.release_note("## Release note\n\nOne sentence.\n\n## Numbers\n\nx") == "One sentence."
    )
    assert notes.release_note("### Release notes\n\n- a\n- b\n") == "a b"
    assert notes.release_note("**Release note:** inline.\nstill.\n\nnot this\n") == "inline. still."
    assert (
        notes.release_note("## Release note: on the heading\n\nand below\n")
        == "on the heading and below"
    )
    assert notes.release_note("## Release note\n\n<!-- a comment -->\nkept\n") == "kept"
    for none in (
        "none",
        "N/A",
        "No user-visible change.",
        "None: a test-only change.",
        "(none)",
        "nothing",
    ):
        assert notes.release_note(f"## Release note\n\n{none}\n") == "", none
    assert (
        notes.release_note(
            "## Release note\n\nNone of the flags change, but the metric is renamed.\n"
        )
        != ""
    )
    assert notes.release_note("## Description\n\nno section\n") is None
    assert notes.release_note("") is None
    # an empty section (nothing, or a placeholder comment alone) is no note: the subject is seeded and marked
    assert notes.release_note("## Release note\n\n\n\n## Test Plan\n\nx") is None
    assert notes.release_note("## Release note\n\n## Test Plan\n") is None
    assert (
        notes.release_note("## Release note\n\n<!-- a comment alone -->\n\n## Test Plan\n") is None
    )
    assert notes.note_section("## Release note\n\n<!-- alone -->\n") == (True, None)
    assert notes.note_section("## Description\n\nx") == (False, None)
    assert notes.note_section("## Release note\n\nnone\n") == (True, "")
    # the four bodies of the pull request template's section: untouched (its comment alone) -> missing, seeded and
    # marked; a sentence -> as written; "none" with a reason -> declared none; the section deleted -> missing
    template_comment = (
        '<!-- One sentence on what a user gets, written to be pasted into the release notes as it stands, or "none"'
        " with the reason. The release notes are written from these sentences. -->"
    )
    untouched = (
        f"## Description\n\nx\n\n## Release note\n\n{template_comment}\n\n## Test Plan\n\ny\n"
    )
    assert notes.release_note(untouched) is None
    assert (
        notes.release_note(untouched.replace(template_comment, "A sentence, as written."))
        == "A sentence, as written."
    )
    assert (
        notes.release_note(untouched.replace(template_comment, "none: a test-only change.")) == ""
    )
    assert notes.release_note("## Description\n\nx\n\n## Test Plan\n\ny\n") is None
    # "none" with a reason, in the spellings the template invites: a dash of any width or a word opens the reason
    for declared in (
        "none because it is a test-only change",
        "None \u2014 tests only",
        "none \u2013 CI only",
        "none, a refactor",
        "N/A \u2014 refactor",
        "No user-facing change.",
    ):
        assert notes.release_note(f"## Release note\n\n{declared}\n") == "", declared
    # a sentence that goes on in its own words is a release note: "as" and "since" open no reason ("as long as",
    # "as of", "since 1.13" are conditions and times)
    for sentence in (
        "None of the defaults change in this release.",
        "Nothing breaks when the flag is unset.",
        "No user-visible change as long as `--foo` is unset; with it set, the gateway now refuses to start.",
        "No behaviour change as long as the default of `--max-batch` is kept; it now defaults to 64.",
        "No visible change since 1.13 for callers; the flag now defaults to on.",
        "Nothing as of this release; the flag is removed in the next.",
    ):
        assert notes.release_note(f"## Release note\n\n{sentence}\n") == sentence, sentence
    # what a sentence that reads as none goes on to declare still reaches Upgrade notes, in its own words,
    # while it seeds no candidate line
    pull = notes.Pull.of("feat(gateway): the `--x` default moves (#7)")
    pull.read = True
    pull.take("## Release note\n\nNo behavior change, but the default of `--x` moves to 5.\n")
    pull.declare()
    assert pull.note == "" and not pull.seeds and pull.kinds == ["a changed default"]
    assert pull.upgrade_line == (
        "- feat(gateway): the `--x` default moves (#7) <!-- AUTHOR: declared: a changed default; the release"
        ' note says "No behavior change, but the default of `--x` moves to 5."; keep it here, move it or drop'
        " it -->"
    ), "the subject leads when the section reads as none; the sentence rides in the comment"
    # a bare none, or one with a reason: the subject says what the declaration refers to
    pull = notes.Pull.of("refactor(config): remove the unused `--foo` flag (#10)")
    pull.read = True
    pull.take("## Release note\n\nnone: internal refactor\n")
    pull.declare()
    assert pull.kinds == ["a removed or renamed flag or setting"]
    assert pull.upgrade_line == (
        "- refactor(config): remove the unused `--foo` flag (#10) <!-- AUTHOR: declared: a removed or renamed"
        ' flag or setting; the release note says "none: internal refactor"; keep it here, move it or drop'
        " it -->"
    )
    pull = notes.Pull.of("fix(gateway): a set `--foo` needs its peer (#8)")
    pull.read = True
    pull.take(
        "## Release note\n\nNo user-visible change as long as `--foo` is unset; with it set, the gateway"
        " now refuses to start.\n"
    )
    pull.declare()
    assert pull.seeds and pull.kinds == ["a configuration that now refuses to start"]
    assert notes.placeholders("- none because tests only (#5)\n- A real sentence. (#6)\n") == [
        (1, "- none because tests only (#5)")
    ]
    assert (
        notes.release_note("## Release note\n\nA --> closer first, then a <!-- opener.\n")
        == "A --&gt; closer first, then a &lt;!-- opener."
    )
    absorbed = "<!-- (an opener without a closer: it hides everything up to the next closer)"
    assert notes.placeholders("a <!-- x\n<!-- AUTHOR: y -->") == [
        (1, absorbed),
        (2, "<!-- AUTHOR: y -->"),
    ], "the opener without a closer, and the author comment that absorbed it"
    assert notes.placeholders("a <!-- x\nprose the draft loses\n<!-- plain -->\nmore") == [
        (1, absorbed)
    ], "a plain comment that absorbs an unclosed opener does not pass it"
    assert notes.placeholders("text <!-- open\nmore") == [
        (1, "<!-- (an opener without a closer: it hides everything after it)")
    ]
    assert notes.placeholders("a `<!--` in code\n<!-- plain -->\n") == [], "code spans open nothing"
    assert notes.placeholders("<!-- see AUTHOR: later -->\n<!-- not one -->") == [
        (1, "<!-- see AUTHOR: later -->")
    ]
    pull = notes.Pull.of("feat(gateway, python)!: x (#7)")
    assert (pull.number, pull.type, pull.scope, pull.bang, pull.area) == (
        7,
        "feat",
        "gateway, python",
        True,
        "Gateway",
    )
    assert notes.Pull.of("docs(gateway): x (#8)").area is None
    assert notes.Pull.of("fix(e2e): x (#9)").area is None
    assert notes.Pull.of("fix(whatever): x (#9)").area == "Other areas"
    unparsed = notes.Pull.of("Add a thing")
    assert (unparsed.number, unparsed.type, unparsed.area) == (None, "", "Other areas")
    assert notes.placeholders("a\n<!-- AUTHOR: x -->\n{{t}}\n<!-- not one -->") == [
        (2, "<!-- AUTHOR: x -->"),
        (3, "{{t}}"),
    ]
    quoted = "## Release note\n\nThe template writes `<!-- x -->` as text, and `-->` alone.\n"
    assert (
        notes.release_note(quoted) == "The template writes `<!-- x -->` as text, and `-->` alone."
    )
    mixed = "## Release note\n\n<!-- hidden --> kept, with `<!-- shown -->` and <!-- see `x` --> gone.\n"
    assert notes.release_note(mixed) == "kept, with `<!-- shown -->` and gone."
    assert notes.placeholders("an opener in code: `<!--` hides nothing\n") == []
    assert notes.placeholders("`<!-- AUTHOR: quoted -->` is not a placeholder") == []


def test_notes_render_without_the_bodies_when_the_repository_is_not_visible(
    repo: Sandbox, tmp_path: Path
) -> None:
    """A token that cannot see the repository gets "repository": null and no data: the fallback, not a crash."""
    pulls = tmp_path / "pulls"
    noted_release(repo, pulls)
    done = repo.tool("notes", "1.14.0", GH_STUB_PULLS=str(pulls), GH_STUB_GRAPHQL_NULL_REPO="1")
    assert (
        "the read of 16 pull request bodies failed (gh: Could not resolve to a Repository"
        in done.stderr
    )
    assert "0 carry a release note" in done.stderr and "## What's Changed" in done.stdout
    assert "(#201) <!-- AUTHOR: no release note in the pull request" in done.stdout


def test_notes_refresh_reads_every_body_again_and_a_cut_cache_file_is_read_again(
    repo: Sandbox, tmp_path: Path
) -> None:
    pulls = tmp_path / "pulls"
    noted_release(repo, pulls)
    cache = tmp_path / "cache"
    env = {"GH_STUB_PULLS": str(pulls), "NOTES_CACHE": str(cache)}
    first = repo.tool("notes", "1.14.0", **env).stdout
    assert "(#212) <!-- AUTHOR: no release note in the pull request" in first
    (pulls / "212.md").write_text("## Release note\n\nThe retry charge is one per request.\n")
    cached = repo.tool("notes", "1.14.0", **env).stdout
    assert "(#212) <!-- AUTHOR: no release note" in cached, "the cache answers until a refresh"
    refreshed = repo.tool("notes", "1.14.0", NOTES_REFRESH="1", **env).stdout
    assert "- The retry charge is one per request. (#212)" in refreshed
    reads = [call for call in repo.gh_calls() if call.startswith("gh api graphql")]
    assert len(reads) == 2 and "p201:" in reads[1], "the refresh asked for every number again"
    (cache / "example-repo" / "201.json").write_text("{")  # a write cut short
    again = repo.tool("notes", "1.14.0", **env).stdout
    assert "- The readiness body names each served model with a reason. (#201)" in again
    reads = [call for call in repo.gh_calls() if call.startswith("gh api graphql")]
    assert len(reads) == 3 and "p201:" in reads[2] and "p202:" not in reads[2], (
        "only the cut file is read again"
    )
    (cache / "example-repo" / "205.json").unlink()  # one body with no earlier read left
    failed = repo.tool("notes", "1.14.0", NOTES_REFRESH="1", GH_STUB_GRAPHQL_ERROR="boom", **env)
    assert "the read of 16 pull request bodies failed (gh: boom)" in failed.stderr
    assert (
        "15 of them render from the earlier read; the refresh did not reach them" in failed.stderr
    )
    assert (
        "1 of them fall back to their subjects" in failed.stderr
        and "2 could not be read" in failed.stderr
    )
    assert "- The retry charge is one per request. (#212)" in failed.stdout, (
        "the earlier read stands, and says so"
    )
    assert "(#205) <!-- AUTHOR: no release note in the pull request" in failed.stdout
    assert "- chore(ci): pin the runner image (#213)" in failed.stdout, (
        "#213 is no pull request: unread as always"
    )


# --- publish-plan ----------------------------------------------------------------------------


GO_MODULE = "github.com/smg-project/smg/bindings/golang"  # what bindings/golang/go.mod declares


def plan_items(out: str) -> list[int]:
    """The numbers of a publishing plan's items, in the order printed."""
    return [int(line.split(".", 1)[0]) for line in out.splitlines() if re.match(r"\d+\. ", line)]


def test_publish_plan_prints_the_dispatches_and_runs_nothing(repo: Sandbox) -> None:
    finished_release(repo)
    calls = len(repo.gh_calls())
    out = repo.tool("publish-plan", "1.14.0").stdout
    assert len(repo.gh_calls()) == calls, "the plan runs nothing"
    for workflow in (
        "release-pypi.yml",
        "release-crates.yml",
        "release-docker.yml",
        "release-helm.yml",
        "release-grpc.yml",
    ):
        assert f"gh workflow run {workflow} --ref v1.14.0" in out
    assert "gh workflow run release-engine-docker.yml --ref v1.14.0 -f engine=vllm" in out
    assert "gh workflow run release-go.yml --ref v1.14.0 -f version=1.14.0" in out
    assert "smg==1.14.0," in out and "and latest" in out
    assert "     smg 1.14.0\n" in out and "     tool-parser 0.6.0\n" in out
    assert "openapi-gen" not in out, (
        "a crate with publish = false is never published: not in the plan"
    )
    assert "smg-client" not in out, (
        "a crate with publish = [] is never published either: not in the plan"
    )
    assert "publish = false or publish = [] is never published and not listed" in out
    assert plan_items(out) == [1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_publish_plan_leaves_pypi_and_crates_io_out_of_a_candidate(repo: Sandbox) -> None:
    """A candidate is never published to PyPI or crates.io: its plan carries the images, the chart, the engine
    images, the Go module tag and the GitHub pre-release, says once why the three registry dispatches are not
    there, reads no registry, and numbers its items without a gap."""
    finished_release(repo)
    out = repo.tool("publish-plan", "1.14.0-rc.1").stdout
    for workflow in ("release-docker.yml", "release-helm.yml"):
        assert f"gh workflow run {workflow} --ref v1.14.0-rc.1" in out
    assert "gh workflow run release-engine-docker.yml --ref v1.14.0-rc.1 -f engine=vllm" in out
    assert "gh workflow run release-go.yml --ref v1.14.0-rc.1 -f version=1.14.0-rc.1" in out
    assert "latest does not move" in out
    for workflow in ("release-pypi.yml", "release-crates.yml", "release-grpc.yml"):
        assert f"gh workflow run {workflow}" not in out, workflow
    assert "A candidate is never published to PyPI or crates.io" in out
    assert "smg==" not in out and "tool-parser" not in out
    assert repo.curl_calls() == [], "no registry is read for a plan that publishes to none"
    assert plan_items(out) == [1, 2, 3, 4, 5, 6]


def test_publish_plan_leaves_the_bindings_tag_out_where_the_module_path_is_old(
    repo: Sandbox,
) -> None:
    """A patch of a line whose bindings/golang/go.mod still declares another module path: the plan prints no
    `bindings/golang/v<version>` tag item (the tag would name a go.mod that go get refuses) and says why; the
    other items count without a gap."""
    finished_release(repo)
    repo.tool("bump", "1.14.1")
    repo.put("bindings/golang/go.mod", "module github.com/lightseek/smg/go-grpc-sdk\n\ngo 1.24.0\n")
    repo.git("add", "-A")
    repo.git("commit", "--quiet", "-m", "chore: the older line keeps its module path")
    repo.land_on_release()
    out = repo.tool("publish-plan", "1.14.1").stdout
    assert "git tag bindings/golang/v1.14.1" not in out
    assert (
        "no Go bindings module tag: bindings/golang/go.mod at origin/release/1.14 does not declare "
        f"{GO_MODULE}, so a tag there would not resolve" in out
    )
    assert "gh workflow run release-go.yml --ref v1.14.1 -f version=1.14.1" in out
    assert plan_items(out) == [1, 2, 3, 4, 5, 6, 7, 8]


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


def test_check_reports_that_a_candidate_never_reaches_the_registries(repo: Sandbox) -> None:
    repo.cut()
    out = repo.tool("check", "1.14.0-rc.1").stdout
    assert (
        "[note] publish: a candidate is never published to PyPI or crates.io (release-pypi,"
        " release-crates and release-grpc refuse a ref named -rc.; release.sh publish-plan 1.14.0-rc.1"
        " leaves them out): its artifacts are the images, the chart, the two Go module tags and the GitHub"
        " pre-release" in out
    )
    repo.tool("bump", "1.14.0")
    repo.land_on_release()
    out = repo.tool("check", "1.14.0").stdout
    assert (
        "[note] publish: the final publishes to PyPI and crates.io by its dispatches"
        " (release.sh publish-plan 1.14.0 lists them; a candidate never does)" in out
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
    # the parent's run is pending too: the bump has no card tier to inherit
    pending = repo.tool(
        "check",
        "1.14.0-rc.1",
        check=False,
        GH_STUB_CHECKS="in_progress build",
        GH_STUB_RUNS="in_progress https://example.invalid/runs/2",
    )
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


def test_check_takes_the_parents_run_for_a_head_that_changes_nothing_but_version_strings(
    repo: Sandbox,
) -> None:
    """The cut's bump commit changes nothing but version strings on main's tip: the workflow lets it inherit the
    card tier from that commit when its run passed, and the gate reads that run while the head's own host rows are
    pending or not started. A red on the head, a parent whose run did not pass or whose only passed run is a
    pull request's, or code beside the bump: nothing is inherited."""
    repo.cut()
    head = repo.origin_ref("refs/heads/release/1.14")
    assert head is not None
    parent = repo.git("rev-parse", f"{head}^")
    # the parent: green on every check of its own (the head's checks come from every workflow), its run passed
    passed = {
        f"GH_STUB_RUNS_{parent}": "success https://example.invalid/runs/2",
        f"GH_STUB_CHECKS_{parent}": "success build\\nsuccess pr-test-mlx\\nskipped e2e (mock)",
    }
    out = repo.tool(
        "check",
        "1.14.0-rc.1",
        GH_STUB_CHECKS="in_progress build\\nsuccess lint",
        GH_STUB_RUNS="queued https://example.invalid/runs/3",
        **passed,
    ).stdout
    assert (
        f"[ok]   CI: 1 green, 1 pending on {head[:7]}: the head changes nothing but version strings on"
        f" {parent[:7]}, green on every check, whose PR Test (SMG) run passed (https://example.invalid/runs/2)"
        in out
    )
    assert (
        f"[ok]   PR Test (SMG): the card tier of {head[:7]} is {parent[:7]}'s successful run"
        " (https://example.invalid/runs/2)" in out
    )
    assert "the head's own run (queued https://example.invalid/runs/3) is expected to end" in out
    assert out.rstrip().endswith("the verifier lines are the release owner's.")
    none = repo.tool(
        "check", "1.14.0-rc.1", GH_STUB_CHECKS="queued build", GH_STUB_RUNS="none", **passed
    )
    assert "the head's own run (not started yet) is expected to end" in none.stdout
    assert "[FAIL]" not in none.stdout
    # a parent whose PR Test (SMG) run passed but whose other workflow is red, or still pending, covers nothing:
    # the same red is queued on the head
    for parent_checks in (
        "success build\\nfailure pr-test-mlx",
        "success build\\nin_progress pr-test-mlx",
    ):
        other = repo.tool(
            "check",
            "1.14.0-rc.1",
            check=False,
            GH_STUB_CHECKS="in_progress build\\nsuccess lint",
            GH_STUB_RUNS="queued https://example.invalid/runs/3",
            **{**passed, f"GH_STUB_CHECKS_{parent}": parent_checks},
        )
        assert other.returncode == 1, parent_checks
        assert "[FAIL] CI: 1 green, 0 red, 1 pending" in other.stdout, parent_checks
        assert "[FAIL] PR Test (SMG): no successful run on" in other.stdout, parent_checks
    red = repo.tool(
        "check",
        "1.14.0-rc.1",
        check=False,
        GH_STUB_CHECKS="failure build",
        GH_STUB_RUNS="failure https://example.invalid/runs/3",
        **passed,
    )
    assert red.returncode == 1
    assert "[FAIL] CI: 0 green, 1 red, 0 pending" in red.stdout
    assert "[FAIL] PR Test (SMG): no successful run on" in red.stdout
    cancelled = repo.tool(
        "check",
        "1.14.0-rc.1",
        check=False,
        GH_STUB_CHECKS="in_progress build",
        GH_STUB_RUNS="queued https://example.invalid/runs/3",
        **{**passed, f"GH_STUB_RUNS_{parent}": "cancelled https://example.invalid/runs/2"},
    )
    assert cancelled.returncode == 1
    assert "[FAIL] CI: 0 green, 0 red, 1 pending" in cancelled.stdout
    assert "[FAIL] PR Test (SMG): no successful run on" in cancelled.stdout
    # a pull request's run can pass with the card lanes skipped (the docs-only shortcut): the parent's push run
    # is the one read, whatever the order the API lists them in, and a pull request's run alone covers nothing
    push = repo.tool(
        "check",
        "1.14.0-rc.1",
        GH_STUB_CHECKS="in_progress build",
        GH_STUB_RUNS="queued https://example.invalid/runs/3",
        **{
            **passed,
            f"GH_STUB_RUNS_{parent}": "success https://example.invalid/runs/9 pull_request"
            "\\nsuccess https://example.invalid/runs/2 push",
        },
    ).stdout
    assert (
        f"[ok]   PR Test (SMG): the card tier of {head[:7]} is {parent[:7]}'s successful run"
        " (https://example.invalid/runs/2)" in push
    )
    pull_request = repo.tool(
        "check",
        "1.14.0-rc.1",
        check=False,
        GH_STUB_CHECKS="in_progress build",
        GH_STUB_RUNS="queued https://example.invalid/runs/3",
        **{
            **passed,
            f"GH_STUB_RUNS_{parent}": "success https://example.invalid/runs/9 pull_request",
        },
    )
    assert pull_request.returncode == 1
    assert "[FAIL] CI: 0 green, 0 red, 1 pending" in pull_request.stdout
    assert "[FAIL] PR Test (SMG): no successful run on" in pull_request.stdout
    # code lands on the branch after the bump: that head inherits nothing, whatever the bump's run
    repo.git("switch", "--quiet", "--detach", "origin/release/1.14")
    repo.put("model_gateway/src/main.rs", 'fn main() {\n    println!("three");\n}\n')
    repo.git("add", "-A")
    repo.git("commit", "--quiet", "-m", "fix(router): three on the branch (#10)")
    repo.git("push", "--quiet", "origin", "HEAD:refs/heads/release/1.14")
    repo.git("switch", "--quiet", "main")
    code = repo.tool(
        "check",
        "1.14.0-rc.1",
        check=False,
        GH_STUB_CHECKS="in_progress build",
        GH_STUB_RUNS="queued https://example.invalid/runs/4",
        **{f"GH_STUB_RUNS_{head}": "success https://example.invalid/runs/3"},
    )
    assert code.returncode == 1
    assert "[FAIL] CI: 0 green, 0 red, 1 pending" in code.stdout
    assert "[FAIL] PR Test (SMG): no successful run on" in code.stdout


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
    # a candidate's plan reads neither the list nor a registry, so this manifest cannot stop it
    candidate = repo.tool("publish-plan", "1.14.0-rc.1")
    assert "A candidate is never published to PyPI or crates.io" in candidate.stdout
    assert repo.curl_calls() == [], "a candidate's plan reads no registry"
    # the final's plan reads the list, and fails the same way as the gate (the bump is written by hand: the
    # checker refuses this manifest)
    repo.git("switch", "--quiet", "release/1.14")
    repo.put("model_gateway/Cargo.toml", versioned("smg", "1.14.0"))
    repo.git("commit", "--quiet", "-am", "chore(release): bump versions to 1.14.0")
    repo.land_on_release()
    repo.git("switch", "--quiet", "main")
    plan = repo.tool("publish-plan", "1.14.0", check=False)
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


# --- ship -------------------------------------------------------------------------------------


def ship_lines(out: str) -> list[str]:
    return [line[len("ship: ") :] for line in out.splitlines() if line.startswith("ship: ")]


ENGINE_RUN = "SMG+{engine} | base={base} | engine=latest | smg=release tag |  by @owner"
# a candidate's plan: the Python package, the crates and the gRPC packages wait for the final
CANDIDATE_DISPATCHES = [
    "gh workflow run release-docker.yml --ref v1.14.0-rc.2",
    "gh workflow run release-helm.yml --ref v1.14.0-rc.2",
    "gh workflow run release-engine-docker.yml --ref v1.14.0-rc.2 -f engine=vllm",
    "gh workflow run release-engine-docker.yml --ref v1.14.0-rc.2 -f engine=vllm"
    " -f base_image_ref=vllm/vllm-openai:v0.27.1",
    "gh workflow run release-engine-docker.yml --ref v1.14.0-rc.2 -f engine=sglang",
    "gh workflow run release-engine-docker.yml --ref v1.14.0-rc.2 -f engine=tokenspeed",
    "gh workflow run release-go.yml --ref v1.14.0-rc.2 -f version=1.14.0-rc.2",
]
TITLE = "# Shepherd Model Gateway v1.14.0-rc.2"  # the template's first line for the candidate


def written_notes(repo: Sandbox) -> Path:
    """Notes the author wrote for v1.14.0-rc.2: the template's title and a sentence, no placeholder."""
    path = repo.base / "notes.md"
    path.write_text(f"{TITLE}\n\nWhat changed, in the author's words.\n")
    return path


def written(notes: Path) -> None:
    """The author's pass over a rendered draft: every placeholder comment becomes a sentence."""
    notes.write_text(re.sub(r"<!--.*?-->", "Written.", notes.read_text(), flags=re.S))


def test_ship_dry_run_lists_every_step_and_writes_nothing(repo: Sandbox) -> None:
    """A dry run reads the branch, the rules and the plan, prints each step with what it would do, and
    changes nothing; a candidate's release is a prerelease, a final's is latest."""
    repo.cut()
    head = repo.origin_ref("refs/heads/release/1.14")
    calls = len(repo.gh_calls())
    done = repo.tool("ship", "1.14.0-rc.2", "--dry-run")
    lines = ship_lines(done.stdout)
    assert lines[0].startswith("v1.14.0-rc.2, a candidate on release/1.14 at ")
    assert lines[1] == (
        "dry run: each step prints what it would do with what it reads now; nothing is written"
    )
    assert "[ok]   CI: 2 check(s) green" in done.stdout and "[ok]   PR Test (SMG)" in done.stdout
    assert "DeepSeek provider verifier: <record link>" in done.stdout
    steps = [
        "preflight: the CI of release/1.14's head",
        "would commit 'chore(release): bump versions for v1.14.0-rc.2' on chore/release-1.14.0-rc.2",
        "would tag v1.14.0-rc.2 at release/1.14's head once it reads 1.14.0-rc.2",
        "would render the notes (release.sh notes 1.14.0-rc.2) into ",
        "would create the GitHub release: gh release create v1.14.0-rc.2 --verify-tag --prerelease"
        " --title 'v1.14.0-rc.2' --notes-file ",
        "publish: the plan is read at ",
        "would dispatch: gh workflow run release-docker.yml --ref v1.14.0-rc.2",
        "dry run of v1.14.0-rc.2 done: nothing was committed, pushed, tagged, created or dispatched",
    ]
    found = [next(i for i, line in enumerate(lines) if line.startswith(step)) for step in steps]
    assert found == sorted(found), "the steps print in their order"
    bump = lines[found[1]]
    assert bump.endswith(
        "and move release/1.14 to it directly: no rule the caller cannot bypass requires"
        " a pull request on release/1.14"
    )
    assert [
        line[len("would dispatch: ") :] for line in lines if line.startswith("would dispatch: ")
    ] == (CANDIDATE_DISPATCHES)
    assert repo.origin_ref("refs/heads/release/1.14") == head
    assert repo.origin_ref("refs/tags/v1.14.0-rc.2") is None
    assert repo.git("branch", "--list", "chore/release-1.14.0-rc.2") == ""
    assert repo.git("branch", "--show-current") == "main"
    writes = [
        call
        for call in repo.gh_calls()[calls:]
        if call.split()[1:3] in (["pr", "create"], ["release", "create"], ["workflow", "run"])
    ]
    assert writes == [], "a dry run reads"
    final = ship_lines(repo.tool("ship", "1.14.0", "--dry-run").stdout)
    assert final[0].startswith("v1.14.0, the final on release/1.14 at ")
    release = [line for line in final if line.startswith("would create the GitHub release: ")]
    assert len(release) == 1 and " --latest " in release[0] and "--prerelease" not in release[0]
    assert "would dispatch: gh workflow run release-go.yml --ref v1.14.0 -f version=1.14.0" in final
    assert "would dispatch: gh workflow run release-pypi.yml --ref v1.14.0" in final, (
        "the final's plan carries the registry dispatches a candidate's leaves out"
    )


def test_ship_plans_the_bump_by_a_pull_request_when_a_rule_the_caller_cannot_bypass_applies(
    repo: Sandbox,
) -> None:
    repo.cut()

    def plan(**env: str) -> str:
        return repo.tool("ship", "1.14.0-rc.2", "--dry-run", **env).stdout

    assert (
        "by a pull request: the rule 'pull_request' of ruleset 11 applies to release/1.14 and the"
        " caller is not on its bypass list" in plan(GH_STUB_RULES="pull_request 11")
    )
    assert "directly: no rule the caller cannot bypass" in plan(
        GH_STUB_RULES="pull_request 11", GH_STUB_BYPASS="7\\n1"
    ), "the caller (user 1) is on the ruleset's always-bypass list"
    assert "by a pull request: release/1.14 carries branch protection (the classic kind)" in plan(
        GH_STUB_PROTECTED="true"
    )
    assert "directly (BUMP_PUSH=1)" in plan(GH_STUB_RULES="pull_request 11", BUMP_PUSH="1")
    assert "by a pull request (BUMP_PUSH=0)" in plan(BUMP_PUSH="0")


def test_ship_takes_a_candidate_from_the_bump_to_the_dispatches_and_a_rerun_changes_nothing(
    repo: Sandbox,
) -> None:
    repo.cut()
    parent = repo.origin_ref("refs/heads/release/1.14")
    cut_calls = len(repo.gh_calls())
    first = repo.tool("ship", "1.14.0-rc.2", check=False)
    out = first.stdout
    head = repo.origin_ref("refs/heads/release/1.14")
    assert head != parent and repo.git("rev-parse", f"{head}^") == parent, (
        "the bump commit was pushed onto the branch"
    )
    for path in VERSION_FILES:
        assert repo.version_of(path, head) == "1.14.0-rc.2", path
    assert repo.git("log", "-1", "--format=%s", head) == (
        "chore(release): bump versions for v1.14.0-rc.2"
    )
    assert "bump: pushing chore/release-1.14.0-rc.2 to release/1.14 directly" in out
    assert "[ok]   versions: the branch reads 1.14.0-rc.2" in out and "[ok]   registry:" in out
    assert repo.origin_ref("refs/tags/v1.14.0-rc.2^{commit}") == head
    assert repo.git("branch", "--show-current") == "main"
    assert repo.git("branch", "--list", "chore/release-1.14.0-rc.2") == "", (
        "the topic is on the branch now"
    )
    notes = repo.work / ".git" / "release" / "notes-v1.14.0-rc.2.md"
    assert notes.read_text().startswith(f"{TITLE}\n")
    # the fresh render carries the template's author placeholders: the ship stops at the notes with the
    # check's lines for the author, and nothing is published
    assert first.returncode == 1 and f"notes: rendered into {notes}" in out
    assert "not ready for publication: still carries " in first.stderr
    assert " author comment(s), 0 template token(s), 0 line(s) that say none (listed above)" in (
        first.stderr
    )
    assert "(fill them, then rerun the ship: --from notes)" in first.stderr
    assert "  line " in first.stderr and "<!-- AUTHOR:" in first.stderr
    calls = repo.gh_calls()[cut_calls:]
    assert not [call for call in calls if call.startswith("gh pr create")], (
        "no pull request: the push was allowed (the cut's own, moving main, came before)"
    )
    assert not [
        call for call in calls if call.split()[1:3] in (["release", "create"], ["workflow", "run"])
    ], "nothing is published on unwritten notes"
    written(notes)
    out = repo.tool("ship", "1.14.0-rc.2").stdout
    assert f"notes: {notes}, rendered by an earlier run" in out
    calls = repo.gh_calls()[cut_calls:]
    assert (
        "gh release create v1.14.0-rc.2 --verify-tag --prerelease --title v1.14.0-rc.2"
        f" --notes-file {notes}" in calls
    )
    assert [call for call in calls if call.startswith("gh workflow run ")] == CANDIDATE_DISPATCHES
    assert "publish: 7 dispatched, 0 there already, 0 failed" in out
    assert "the runs on v1.14.0-rc.2 (gh run list --branch v1.14.0-rc.2):" in out
    assert out.rstrip().endswith(
        "7 new dispatch(es) (0 were there already); a run that fails is dispatched again by"
        " 'release.sh ship 1.14.0-rc.2 --from publish'"
    )
    # the rerun: the branch, the tag, the release and the runs are there; a failed or cancelled run is
    # dispatched again, nothing else is
    runs = "\n".join(
        [
            "release-docker.yml;3;Release Docker Image;completed;failure",
            "release-helm.yml;4;Release Helm Chart;completed;success",
            "release-engine-docker.yml;5;"
            + ENGINE_RUN.format(engine="vllm", base="engine default")
            + ";queued;",
            "release-engine-docker.yml;6;"
            + ENGINE_RUN.format(engine="vllm", base="vllm/vllm-openai:v0.27.1")
            + ";queued;",
            "release-engine-docker.yml;7;"
            + ENGINE_RUN.format(engine="sglang", base="engine default")
            + ";completed;cancelled",
            "release-engine-docker.yml;8;"
            + ENGINE_RUN.format(engine="tokenspeed", base="engine default")
            + ";queued;",
            "release-go.yml;10;Release gRPC goLang packages;completed;success",
        ]
    )
    before = len(repo.gh_calls())
    again = repo.tool(
        "ship",
        "1.14.0-rc.2",
        GH_STUB_RELEASES="v1.14.0-rc.2",
        GH_STUB_RELEASE_PRE="true",
        GH_STUB_TAG_RUNS=runs,
    ).stdout
    assert "preflight: release/1.14 reads 1.14.0-rc.2 already; the CI of the bump's parent" in again
    assert "bump: done, release/1.14 reads 1.14.0-rc.2 at" in again
    assert "tag: done, v1.14.0-rc.2 is on origin at" in again
    assert "notes: the GitHub release v1.14.0-rc.2 exists (" in again and (
        "its notes are out, nothing is read or rendered" in again
    ), "the release is out: the kept render is not read again"
    assert (
        "release: done, https://example.invalid/releases/v1.14.0-rc.2 (prerelease: true)" in again
    )
    assert (
        "publish: done, gh workflow run release-helm.yml --ref v1.14.0-rc.2: run 4 (success)"
        in again
    )
    assert (
        "publish: done, gh workflow run release-engine-docker.yml --ref v1.14.0-rc.2 -f engine=vllm:"
        " run 5 (queued)" in again
    )
    assert (
        "publish: done, gh workflow run release-engine-docker.yml --ref v1.14.0-rc.2 -f engine=vllm"
        " -f base_image_ref=vllm/vllm-openai:v0.27.1: run 6 (queued)" in again
    )
    new_calls = repo.gh_calls()[before:]
    assert [call for call in new_calls if call.startswith("gh workflow run ")] == [
        "gh workflow run release-docker.yml --ref v1.14.0-rc.2",
        "gh workflow run release-engine-docker.yml --ref v1.14.0-rc.2 -f engine=sglang",
    ]
    assert not [
        call for call in new_calls if call.split()[1:3] in (["release", "create"], ["pr", "create"])
    ]
    assert "publish: 2 dispatched, 5 there already, 0 failed" in again
    assert repo.origin_ref("refs/heads/release/1.14") == head
    assert "  4            release-helm.yml                     completed success" in again


def test_ship_opens_the_bump_pull_request_when_the_remote_refuses_the_push_and_waits_for_its_merge(
    repo: Sandbox,
) -> None:
    """The rules read as pushable and the remote says otherwise (a hook here, a rule there): the bump goes
    by its pull request, the ship waits for the merge, then tags the merged head."""
    repo.cut()
    hook = repo.origin / "hooks" / "pre-receive"
    hook.write_text(
        "#!/usr/bin/env bash\n"
        "while read -r old new ref; do\n"
        '  case "$ref" in refs/heads/release/*) echo "a pull request is required" >&2; exit 1 ;; esac\n'
        "done\n"
        "exit 0\n"
    )
    hook.chmod(0o755)
    out = repo.tool(
        "ship", "1.14.0-rc.2", GH_STUB_MERGE="1", MERGE_POLL="0", NOTES=str(written_notes(repo))
    ).stdout
    assert (
        "bump: origin refused the push (" in out and "opening the bump pull request instead" in out
    )
    assert (
        "bump: pull request #1 opened (https://example.invalid/pull/1): waiting for its merge"
        " (120 min at most, read every 0 s)" in out
    )
    calls = repo.gh_calls()
    assert any(
        call.startswith(
            "gh pr create --base release/1.14 --head chore/release-1.14.0-rc.2"
            " --title chore(release): bump versions for v1.14.0-rc.2 --body-file "
        )
        for call in calls
    )
    assert "gh pr view 1 --json state --jq .state" in calls
    assert "## Release gate" in repo.pr_body()
    head = repo.origin_ref("refs/heads/release/1.14")
    assert repo.version_of("model_gateway/Cargo.toml", head) == "1.14.0-rc.2"
    assert "bump: release/1.14 reads 1.14.0-rc.2 at" in out
    assert repo.origin_ref("refs/tags/v1.14.0-rc.2^{commit}") == head
    assert [call for call in calls if call.startswith("gh workflow run ")] == CANDIDATE_DISPATCHES


def test_ship_stops_when_the_bump_pull_request_is_not_merged_in_time_and_resumes_on_it(
    repo: Sandbox,
) -> None:
    repo.cut()
    done = repo.tool("ship", "1.14.0-rc.2", BUMP_PUSH="0", MERGE_WAIT="0", check=False)
    assert done.returncode == 1
    assert (
        "the bump pull request #1 is not merged after 0 min: merge it, then rerun the ship"
        in done.stderr
    )
    assert (
        "bump: opening the pull request of chore/release-1.14.0-rc.2 against release/1.14"
        " (by a pull request (BUMP_PUSH=0))" in done.stdout
    )
    assert repo.origin_ref("refs/heads/chore/release-1.14.0-rc.2") is not None, (
        "the topic is pushed and its pull request open"
    )
    assert repo.origin_ref("refs/tags/v1.14.0-rc.2") is None
    assert not [call for call in repo.gh_calls() if call.startswith("gh release create")]
    again = repo.tool(
        "ship",
        "1.14.0-rc.2",
        GH_STUB_OPEN_HEADS="chore/release-1.14.0-rc.2",
        GH_STUB_MERGE="1",
        MERGE_POLL="0",
        NOTES=str(written_notes(repo)),
    ).stdout
    assert (
        "bump: the pull request #8 of chore/release-1.14.0-rc.2 is open (an earlier run):"
        " waiting for its merge" in again
    )
    head = repo.origin_ref("refs/heads/release/1.14")
    assert repo.version_of("model_gateway/Cargo.toml", head) == "1.14.0-rc.2"
    assert repo.origin_ref("refs/tags/v1.14.0-rc.2^{commit}") == head
    assert repo.git("branch", "--list", "chore/release-1.14.0-rc.2") == ""
    closed = repo.tool("ship", "1.14.0", BUMP_PUSH="0", GH_STUB_PR_STATE="CLOSED", check=False)
    assert closed.returncode == 1 and "was closed without a merge" in closed.stderr


def test_ship_stops_on_a_red_ci_row_and_goes_on_past_a_pending_one(repo: Sandbox) -> None:
    repo.cut()
    red = repo.tool(
        "ship",
        "1.14.0-rc.2",
        "--dry-run",
        check=False,
        GH_STUB_CHECKS="failure unit tests\\nsuccess lint",
    )
    assert red.returncode == 1 and "a CI row of " in red.stderr and " is red (above)" in red.stderr
    assert "       red: unit tests" in red.stdout and "would commit" not in red.stdout
    pending = repo.tool(
        "ship",
        "1.14.0-rc.2",
        "--dry-run",
        GH_STUB_CHECKS="in_progress e2e (card)\\nsuccess lint",
        GH_STUB_RUNS="in_progress https://example.invalid/runs/2",
    ).stdout
    assert "       pending: e2e (card) (in_progress)" in pending
    assert "preflight: the pending rows above do not hold the ship" in pending
    assert any(line.startswith("would commit") for line in ship_lines(pending))
    ended = repo.tool(
        "ship",
        "1.14.0-rc.2",
        "--dry-run",
        check=False,
        GH_STUB_RUNS="cancelled https://example.invalid/runs/3",
    )
    assert ended.returncode == 1, "a matrix that ended without a success is red, not pending"
    absent = repo.tool("ship", "1.14.0-rc.2", "--dry-run", check=False, GH_STUB_RUNS="none")
    assert absent.returncode == 1 and "[FAIL] PR Test (SMG): no run on" in absent.stdout, (
        "a matrix that never ran is red, not pending: nothing arrives by waiting"
    )
    logged_out = repo.tool("ship", "1.14.0-rc.2", "--dry-run", check=False, GH_STUB_AUTH_ERROR="1")
    assert logged_out.returncode == 1 and "is not logged in" in logged_out.stderr
    assert "preflight" not in logged_out.stdout, "nothing is read before gh is known to answer"


def test_ship_refuses_a_tag_elsewhere_and_a_topic_of_a_bump_that_did_not_land(
    repo: Sandbox,
) -> None:
    repo.cut()
    repo.git("tag", "v1.14.0-rc.2", "origin/main")
    repo.git("push", "--quiet", "origin", "refs/tags/v1.14.0-rc.2")
    done = repo.tool("ship", "1.14.0-rc.2", "--dry-run", "--from", "tag", check=False)
    assert done.returncode == 1 and "tags never move; the ship stops" in done.stderr
    repo.git("push", "--quiet", "origin", ":refs/tags/v1.14.0-rc.2")
    repo.git("tag", "-d", "v1.14.0-rc.2")
    repo.git("branch", "chore/release-1.14.0-rc.2", "origin/release/1.14")
    stale = repo.tool("ship", "1.14.0-rc.2", check=False)
    assert stale.returncode == 1
    assert "the branch chore/release-1.14.0-rc.2 exists (a bump that did not land)" in stale.stderr
    assert repo.origin_ref("refs/heads/release/1.14") == repo.git(
        "rev-parse", "origin/release/1.14"
    )


def test_ship_from_resumes_at_a_step_and_publishes_the_authored_notes(repo: Sandbox) -> None:
    repo.cut()
    repo.tool("bump", "1.14.0-rc.2", PUSH="0")
    repo.land_on_release()
    repo.git("switch", "--quiet", "main")
    repo.tool("tag", "1.14.0-rc.2", PUSH="1")
    authored = written_notes(repo)
    lines = ship_lines(
        repo.tool("ship", "1.14.0-rc.2", "--from", "notes", NOTES=str(authored)).stdout
    )
    assert lines[1:4] == [
        "preflight: skipped (--from notes)",
        "bump: skipped (--from notes)",
        "tag: skipped (--from notes)",
    ]
    assert f"notes: {authored} (NOTES)" in lines
    assert (
        "gh release create v1.14.0-rc.2 --verify-tag --prerelease --title v1.14.0-rc.2"
        f" --notes-file {authored}" in repo.gh_calls()
    )
    assert "publish: 7 dispatched, 0 there already, 0 failed" in lines
    wrong = repo.base / "wrong.md"
    wrong.write_text("# notes\n")
    stopped = repo.tool("ship", "1.14.0-rc.2", "--from", "notes", NOTES=str(wrong), check=False)
    assert stopped.returncode == 1
    assert f"not ready for publication: the first line is not '{TITLE}'" in stopped.stderr
    half = repo.base / "half.md"
    half.write_text(f"{TITLE}\n\n<!-- AUTHOR: the opening paragraph -->\n\nA sentence.\n")
    creates = len([call for call in repo.gh_calls() if call.startswith("gh release create")])
    stopped = repo.tool("ship", "1.14.0-rc.2", "--from", "notes", NOTES=str(half), check=False)
    assert stopped.returncode == 1
    assert (
        "not ready for publication: still carries 1 placeholder(s): 1 author comment(s),"
        " 0 template token(s), 0 line(s) that say none (listed above) (fill them, then rerun the"
        " ship: --from notes)" in stopped.stderr
    )
    assert "  line 3: <!-- AUTHOR: the opening paragraph -->" in stopped.stderr
    assert len([call for call in repo.gh_calls() if call.startswith("gh release create")]) == (
        creates
    ), "the template's check holds the release"
    # from the release step the notes of the earlier run are read, not rendered again
    later = repo.tool("ship", "1.14.0-rc.2", "--from", "release", NOTES=str(authored)).stdout
    assert f"notes: {authored} (NOTES)" in later and "release: created" in later
    assert (
        "unknown step 'frobnicate'"
        in repo.tool("ship", "1.14.0-rc.2", "--from", "frobnicate", check=False).stderr
    )
    assert (
        "no release/1.15 on origin: cut it first"
        in repo.tool("ship", "1.15.0-rc.1", "--dry-run", check=False).stderr
    )


def test_ship_rerun_stands_down_at_the_notes_once_the_release_exists(repo: Sandbox) -> None:
    """A rerun after an interruption in publish, with no kept render (the first run used NOTES, or the render was
    deleted) or with a NOTES file that is not ready: the release is out with its notes, so the notes step reads
    and renders nothing and the ship goes on to the dispatches instead of asking for notes that are published."""
    repo.cut()
    repo.tool("bump", "1.14.0-rc.2", PUSH="0")
    repo.land_on_release()
    repo.git("switch", "--quiet", "main")
    repo.tool("tag", "1.14.0-rc.2", PUSH="1")
    notes = repo.work / ".git" / "release" / "notes-v1.14.0-rc.2.md"
    out = repo.tool(
        "ship", "1.14.0-rc.2", GH_STUB_RELEASES="v1.14.0-rc.2", GH_STUB_RELEASE_PRE="true"
    ).stdout
    assert (
        "notes: the GitHub release v1.14.0-rc.2 exists (https://example.invalid/releases/v1.14.0-rc.2"
        " (prerelease: true)): its notes are out, nothing is read or rendered"
    ) in ship_lines(out)
    assert not notes.exists(), "nothing is rendered"
    assert "release: done, https://example.invalid/releases/v1.14.0-rc.2 (prerelease: true)" in out
    assert "publish: 7 dispatched, 0 there already, 0 failed" in out
    assert not [call for call in repo.gh_calls() if call.startswith("gh release create")]
    half = repo.base / "half.md"
    half.write_text(f"{TITLE}\n\n<!-- AUTHOR: the opening paragraph -->\n")
    again = repo.tool(
        "ship",
        "1.14.0-rc.2",
        "--from",
        "notes",
        NOTES=str(half),
        GH_STUB_RELEASES="v1.14.0-rc.2",
        GH_STUB_RELEASE_PRE="true",
    ).stdout
    assert "its notes are out, nothing is read or rendered" in again and "release: done" in again, (
        "a NOTES file that is not ready does not matter once the release is out"
    )


def test_ship_dispatches_nothing_when_the_plan_or_a_run_list_cannot_be_read(repo: Sandbox) -> None:
    """The plan is read whole, with its status, before the first dispatch, and a failed read of a workflow's runs is
    not an empty list: either failure stops the ship with nothing dispatched (a rerun would double a run otherwise).
    The plan half runs against a final: its plan prints the registry dispatch lines, then reads the component list."""
    repo.cut()
    repo.tool("bump", "1.14.0-rc.2", PUSH="0")
    repo.land_on_release()
    repo.git("switch", "--quiet", "main")
    repo.tool("tag", "1.14.0-rc.2", PUSH="1")
    before = len(repo.gh_calls())
    unread = repo.tool(
        "ship", "1.14.0-rc.2", "--from", "publish", check=False, GH_STUB_RUN_LIST_ERROR="no answer"
    )
    assert unread.returncode == 1
    assert "the runs of release-docker.yml on v1.14.0-rc.2 could not be read" in unread.stderr
    assert "gh: no answer" in unread.stderr, "gh's own message is shown"
    assert not [call for call in repo.gh_calls()[before:] if call.startswith("gh workflow run ")]
    # the plan of a final reads the component list after it has printed the first dispatch lines: a list that cannot
    # be read stops the plan there (a candidate's plan never reads it, so this half runs against the final)
    repo.tool("bump", "1.14.0", PUSH="0")
    repo.land_on_release()
    repo.git("switch", "--quiet", "main")
    repo.tool("tag", "1.14.0", PUSH="1")
    before = len(repo.gh_calls())
    broken = repo.tool("ship", "1.14.0", "--from", "publish", check=False, PYTHON="false")
    assert broken.returncode == 1
    assert "the crates and packages at v1.14.0 are unknown" in broken.stderr, (
        "the final's plan died in its component list, not in the engine table"
    )
    assert "the publish plan of v1.14.0 could not be read" in broken.stderr
    assert not [call for call in repo.gh_calls()[before:] if call.startswith("gh workflow run ")]


def test_ship_dispatches_nothing_when_the_engine_table_cannot_be_read(repo: Sandbox) -> None:
    """The engine table is read inside the plan, whose failures `set -e` does not see under `if`: a table the
    Python cannot parse stops the plan by name, so the ship dispatches nothing (not the registry workflows
    alone, with the engine images missing)."""
    repo.commit_main(
        "ci: a trailing comma in the engine table",
        ".github/workflows/release-engine-docker.yml",
        ENGINE_WORKFLOW.replace(
            '"tokenspeed": {"release_bases": [""]}', '"tokenspeed": {"release_bases": [""]},'
        ),
    )
    repo.cut()
    repo.tool("bump", "1.14.0-rc.2", PUSH="0")
    repo.land_on_release()
    repo.git("switch", "--quiet", "main")
    repo.tool("tag", "1.14.0-rc.2", PUSH="1")
    before = len(repo.gh_calls())
    done = repo.tool("ship", "1.14.0-rc.2", "--from", "publish", check=False)
    assert done.returncode == 1
    assert (
        "the engine table of .github/workflows/release-engine-docker.yml at v1.14.0-rc.2 could not be read"
        in done.stderr
    )
    assert "the publish plan of v1.14.0-rc.2 could not be read" in done.stderr
    assert not [call for call in repo.gh_calls()[before:] if call.startswith("gh workflow run ")]
    plan = repo.tool("publish-plan", "1.14.0-rc.2", check=False)
    assert plan.returncode == 1 and "no engine dispatch listed" in plan.stderr


def test_ship_leaves_no_notes_file_behind_when_the_render_fails(repo: Sandbox) -> None:
    """The notes are rendered beside their place and moved there once whole: a render that stops halfway
    leaves no file a rerun would keep as "rendered by an earlier run"."""
    repo.cut()
    repo.tool("bump", "1.14.0-rc.2", PUSH="0")
    repo.land_on_release()
    repo.git("switch", "--quiet", "main")
    repo.tool("tag", "1.14.0-rc.2", PUSH="1")
    first_final = repo.git("rev-parse", "v1.13.0^{commit}")
    repo.git("push", "--quiet", "origin", ":refs/tags/v1.13.0")
    repo.git("tag", "-d", "v1.13.0")
    failed = repo.tool("ship", "1.14.0-rc.2", "--from", "notes", check=False)
    assert failed.returncode == 1 and "no final release tag below v1.14.0-rc.2" in failed.stderr
    notes = repo.work / ".git" / "release" / "notes-v1.14.0-rc.2.md"
    assert not notes.exists()
    repo.git("tag", "v1.13.0", first_final)
    repo.git("push", "--quiet", "origin", "refs/tags/v1.13.0")
    again = repo.tool("ship", "1.14.0-rc.2", "--from", "notes", check=False)
    assert f"notes: rendered into {notes}" in again.stdout
    assert again.returncode == 1 and "not ready for publication: still carries " in again.stderr, (
        "the fresh render carries the author's placeholders: the ship stops at the notes"
    )
    assert notes.read_text().startswith(f"{TITLE}\n")
    assert not notes.with_name(notes.name + ".partial").exists()


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
        said in plan.stderr
        and "gh workflow run release-docker.yml --ref v1.14.0-rc.1" in plan.stdout
    )
    notes = repo.tool("notes", "1.14.0-rc.1")
    assert said in notes.stderr and "# Shepherd Model Gateway v1.14.0-rc.1" in notes.stdout
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


def test_bump_main_gives_a_main_still_below_the_release_its_development_version(
    repo: Sandbox,
) -> None:
    """main never merged the move its cut opened and reads 1.13.0 under v1.14.0: bump-main moves the crates the
    release raised and the gateway to the development version after the release, never to the release itself."""
    finished_release(repo)
    main_head = repo.commit_main("docs: after the release", "README.md", "# smg\n")
    done = repo.tool("bump-main", "1.14.0")
    assert repo.git("rev-parse", "--abbrev-ref", "HEAD") == "chore/main-versions-1.14.0"
    assert repo.git("rev-parse", "HEAD~1") == main_head
    assert (
        "main reads 1.13.0, below 1.15.0-dev: the gateway and its mirrors move to 1.15.0-dev"
        in done.stderr
    )
    for path in VERSION_FILES:
        assert repo.version_of(path) == "1.15.0-dev", path
    assert repo.version_of("crates/tool_parser/Cargo.toml") == "0.6.0"
    assert 'tool-parser = { version = "0.6.0"' in repo.read("Cargo.toml")
    assert repo.git("log", "-1", "--format=%s") == "chore(release): main's versions follow v1.14.0"
    assert repo.git("log", "-1", "--format=%b").startswith(
        "The gateway reads 1.15.0-dev on main and every crate and package at least the version"
        " v1.14.0 carries"
    )
    assert repo.gh_calls()[-1].startswith(
        "gh pr create --base main --head chore/main-versions-1.14.0"
        " --title chore(release): main's versions follow v1.14.0"
    )
    assert repo.origin_ref("refs/heads/main") == main_head, "main moves by its pull request only"


def test_bump_main_refuses_a_candidate(repo: Sandbox) -> None:
    repo.cut()
    assert "follows a final release" in repo.tool("bump-main", "1.14.0-rc.1", check=False).stderr


def test_bump_main_after_a_patch_of_an_older_line_keeps_mains_higher_version(repo: Sandbox) -> None:
    """v1.15.0 is out and main reads 1.16.0-dev; a v1.14.1 patch of the older line releases a tool-parser fix.
    bump-main for the patch moves that crate alone: main's gateway stays at 1.16.0-dev (the nightly and the next cut
    derive from it) and the next cut is accepted against v1.15.0. Run again once merged, it has nothing to move and
    says so."""
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
    # main read 1.15.0-dev, below its own line's final: it takes the development version after it
    repo.tool("bump-main", "1.15.0")
    repo.git("push", "--quiet", "origin", "HEAD:refs/heads/main")
    assert repo.version_of("model_gateway/Cargo.toml", "origin/main") == "1.16.0-dev"
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
    assert (
        "main reads 1.16.0-dev, not below 1.15.0-dev: the gateway and its mirrors keep it"
        in done.stderr
    )
    assert (
        "the gateway reads 1.16.0-dev, above 1.15.0-dev: it and its mirrors keep that"
        in done.stdout
    )
    for path in VERSION_FILES:
        assert repo.version_of(path) == "1.16.0-dev", path
    assert repo.version_of("crates/tool_parser/Cargo.toml") == "0.6.1"
    assert 'tool-parser = { version = "0.6.1"' in repo.read("Cargo.toml")
    assert repo.git("log", "-1", "--format=%b").startswith(
        "The gateway keeps 1.16.0-dev on main (not below 1.15.0-dev)"
    )
    repo.git("push", "--quiet", "origin", "HEAD:refs/heads/main")  # the pull request merges
    repo.git("switch", "--quiet", "main")
    repo.git("branch", "--quiet", "-D", "chore/main-versions-1.14.1")
    again = repo.tool("bump-main", "1.14.1", check=False)
    assert again.returncode == 1 and "nothing to move" in again.stderr
    assert repo.git("rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert repo.git("branch", "--list", "chore/main-versions-1.14.1") == ""
    repo.cut("1.16.0-rc.1")  # accepted: main reads 1.16.0-dev, above the previous final v1.15.0
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
            **interpreter_library_path(),
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


# --- a candidate never reaches PyPI or crates.io --------------------------------------------------

# The publish workflows the rule holds, by the jobs a dispatch enters: their `if` skips a candidate's ref and every
# other job needs them, so the dispatch is a no-op; their first step refuses the ref with the reason when the job is
# reached anyway.
REGISTRY_WORKFLOWS = {
    "release-pypi.yml": ("build", "build-sdist"),
    "release-crates.yml": ("tier1",),
    "release-grpc.yml": ("detect",),
}
CANDIDATE_IF = "${{ !contains(github.ref_name, '-rc.') }}"
GUARD_STEP = "Refuse a release candidate"
# a version or a ref name, and whether it is a candidate: the tag's form with the v and the bare version
CANDIDATE_CASES = (
    ("v1.14.0", False),
    ("1.14.0", False),
    ("v1.14.0-rc.1", True),
    ("1.14.0-rc.1", True),
    ("v1.14.0-rc.12", True),
    ("1.14.0-rc.12", True),
    ("release/1.14", False),
    ("main", False),
)


def is_candidate(name: str) -> bool:
    """release.sh's predicate, run as the script defines it (its one line, in bash), on a version or a tag name."""
    source = (ROOT / TOOL).read_text().splitlines()
    line = next(text for text in source if text.startswith("is_candidate()"))
    done = subprocess.run(
        ["bash", "-c", f'{line}\nis_candidate "$1"', "is_candidate", name],
        capture_output=True,
        text=True,
    )
    assert done.returncode in (0, 1), done.stderr
    return done.returncode == 0


def candidate_guard(workflow: str, job: str, name: str) -> subprocess.CompletedProcess[str]:
    """The first step of a publish workflow's entry job, run as the workflow runs it, dispatched on the named ref."""
    loaded = yaml.safe_load((ROOT / ".github" / "workflows" / workflow).read_text())
    step = loaded["jobs"][job]["steps"][0]
    assert step["name"] == GUARD_STEP, (workflow, job)
    assert step["shell"] == "bash", (workflow, job, "the wheels' matrix has a Windows runner")
    return subprocess.run(
        ["bash", "-eo", "pipefail", "-c", step["run"]],
        env={
            "PATH": os.pathsep.join(["/usr/bin", "/bin"]),
            "GITHUB_REF_TYPE": "tag" if name.startswith("v") else "branch",
            "GITHUB_REF_NAME": name,
        },
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("name,candidate", CANDIDATE_CASES)
def test_the_candidate_predicate_reads_a_version_or_a_tag_name(name: str, candidate: bool) -> None:
    """One rule in release.sh and in the three publish workflows: a name with `-rc.` in it is a candidate, with the
    tag's v or without; a final, a release branch and main are not."""
    assert is_candidate(name) is candidate
    for workflow, jobs in REGISTRY_WORKFLOWS.items():
        for job in jobs:
            guard = candidate_guard(workflow, job, name)
            assert guard.returncode == (1 if candidate else 0), (workflow, job, guard.stderr)
            if candidate:
                assert f"{name} is a release candidate" in guard.stderr, (workflow, job)
                assert "PyPI" in guard.stderr or "crates.io" in guard.stderr, (workflow, job)
            else:
                assert guard.stderr == "", (workflow, job)


def test_the_publish_workflows_skip_a_candidate_before_their_first_step() -> None:
    """The jobs a dispatch enters carry the rule as their `if`, and every other job of the workflow needs one of
    them, so a candidate's dispatch builds and uploads nothing; a job that runs under always() reads the needed
    job's outputs or result, which a skipped job leaves empty."""
    for workflow, entries in REGISTRY_WORKFLOWS.items():
        loaded = yaml.safe_load((ROOT / ".github" / "workflows" / workflow).read_text())
        assert list(loaded[True]) == ["workflow_dispatch"], workflow
        jobs = loaded["jobs"]

        def upstream(name: str) -> set[str]:
            needs = jobs[name].get("needs", [])
            needs = [needs] if isinstance(needs, str) else list(needs)
            found = set(needs)
            for need in needs:
                found |= upstream(need)
            return found

        for entry in entries:
            assert jobs[entry]["if"] == CANDIDATE_IF, (workflow, entry)
            assert jobs[entry]["steps"][0]["name"] == GUARD_STEP, (workflow, entry)
        for name, job in jobs.items():
            if name in entries:
                continue
            assert upstream(name) & set(entries), (workflow, name, "no entry job gates it")
            condition = str(job.get("if", ""))
            if "always()" in condition:
                assert "needs." in condition, (workflow, name, "always() without reading its needs")


def test_the_rule_end_to_end_the_trunk_is_next(repo: Sandbox) -> None:
    """cut 1.14 -> main 1.15.0-dev (the move merged) -> the final 1.14.0 with a crate bumped on the branch ->
    bump-main moves that crate alone and keeps the gateway -> the next cut, from 1.15.0-dev, writes 1.15.0-rc.1 on
    the branch and 1.16.0-dev on main. The gate's versions item and drift note read a development main as before."""
    repo.cut()
    land_main_move(repo, "1.15.0-dev")
    assert repo.version_of("model_gateway/Cargo.toml", "origin/main") == "1.15.0-dev"
    repo.tool("tag", "1.14.0-rc.1", PUSH="1")
    fix = repo.commit_main(
        "fix(tool_parser): x (#124)", "crates/tool_parser/src/lib.rs", "pub fn x() {}\n"
    )
    repo.tool("pick", "1.14", fix)
    repo.land_on_release()
    repo.tool("bump", "1.14.0")
    assert repo.version_of("crates/tool_parser/Cargo.toml") == "0.6.1", (
        "the branch bumps the picked crate"
    )
    repo.land_on_release()
    repo.tool("tag", "1.14.0", PUSH="1")
    checked = repo.tool("check", "1.14.0")
    assert "[ok]   versions: the branch reads 1.14.0" in checked.stdout
    assert "[note] main: 1 commit(s) on origin/main are not on release/1.14" in checked.stdout, (
        "the move; the picked fix is excluded"
    )
    done = repo.tool("bump-main", "1.14.0")
    assert (
        "main reads 1.15.0-dev, not below 1.15.0-dev: the gateway and its mirrors keep it"
        in done.stderr
    )
    for path in VERSION_FILES:
        assert repo.version_of(path) == "1.15.0-dev", path
    assert repo.version_of("crates/tool_parser/Cargo.toml") == "0.6.1"
    assert 'tool-parser = { version = "0.6.1"' in repo.read("Cargo.toml")
    assert repo.git("log", "-1", "--format=%b").startswith(
        "The gateway keeps 1.15.0-dev on main (not below 1.15.0-dev)"
    )
    repo.git("push", "--quiet", "origin", "HEAD:refs/heads/main")  # the pull request merges
    repo.git("switch", "--quiet", "main")
    repo.git("branch", "--quiet", "-D", "chore/main-versions-1.14.0")
    again = repo.tool("bump-main", "1.14.0", check=False)
    assert again.returncode == 1 and "nothing to move" in again.stderr
    assert repo.git("rev-parse", "--abbrev-ref", "HEAD") == "main"
    # the next cut: from 1.15.0-dev, accepted against v1.14.0; the branch reads 1.15.0-rc.1, main moves to 1.16.0-dev
    repo.commit_main(
        "fix(router): after (#125)",
        "model_gateway/src/main.rs",
        'fn main() {\n    println!("after");\n}\n',
    )
    cut = repo.tool("cut", "1.15.0-rc.1", PUSH="1")
    assert "the versions move from v1.14.0 to 1.15.0-rc.1" in cut.stderr
    for path in VERSION_FILES:
        assert repo.version_of(path, "release/1.15") == "1.15.0-rc.1", path
        assert repo.version_of(path, "chore/main-versions-1.16.0-dev") == "1.16.0-dev", path
    assert repo.version_of("crates/tool_parser/Cargo.toml", "release/1.15") == "0.6.1", (
        "nothing of it changed since v1.14.0: it keeps the number v1.14.0 published"
    )


def test_a_cut_is_refused_while_main_reads_the_development_version_of_a_line_already_final(
    repo: Sandbox,
) -> None:
    """1.15.0-dev sorts below 1.15.0, as a pre-release does: once v1.15.0 is out, a main still at 1.15.0-dev (the
    1.15 cut's move never merged) is below the previous final, and the cut of 1.16 is refused until bump-main 1.15.0
    moves it to 1.16.0-dev."""
    repo.cut()
    land_main_move(repo, "1.15.0-dev")
    repo.tool("tag", "1.14.0-rc.1", PUSH="1")
    repo.tool("bump", "1.14.0")
    repo.land_on_release()
    repo.tool("tag", "1.14.0", PUSH="1")
    assert "nothing to move" in repo.tool("bump-main", "1.14.0", check=False).stderr, (
        "main moved at the cut; the final raised no crate"
    )
    repo.commit_main(
        "fix(router): y (#130)", "model_gateway/src/main.rs", 'fn main() {\n    println!("y");\n}\n'
    )
    repo.cut("1.15.0-rc.1")  # main stays 1.15.0-dev: its move (to 1.16.0-dev) is not merged
    repo.tool("tag", "1.15.0-rc.1", PUSH="1")
    repo.tool("bump", "1.15.0")
    repo.land_on_release("1.15")
    repo.tool("tag", "1.15.0", PUSH="1")
    refused = repo.tool("cut", "1.16.0-rc.1", check=False)
    assert refused.returncode == 1
    assert (
        "origin/main reads 1.15.0-dev, below 1.15.0: merge the pull request that moves main's versions first"
        " (release.sh bump-main 1.15.0 opens it)" in refused.stderr
    )
    assert repo.git("branch", "--list", "release/1.16") == "", "nothing was cut"
    done = repo.tool("bump-main", "1.15.0")
    assert (
        "main reads 1.15.0-dev, below 1.16.0-dev: the gateway and its mirrors move to 1.16.0-dev"
        in done.stderr
    )
    repo.git("push", "--quiet", "origin", "HEAD:refs/heads/main")
    repo.git("switch", "--quiet", "main")
    repo.tool("cut", "1.16.0-rc.1")
    assert repo.version_of("model_gateway/Cargo.toml", "release/1.16") == "1.16.0-rc.1"
    assert (
        repo.version_of("model_gateway/Cargo.toml", "chore/main-versions-1.17.0-dev")
        == "1.17.0-dev"
    )


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


def test_sync_versions_moves_main_to_the_next_development_version_when_a_release_is_cut(
    repo: Sandbox,
) -> None:
    """The moment release/1.14 is cut, main moves above everything that branch will release: the gateway and its
    mirrors to 1.15.0-dev (one string in every file: cargo and Helm read a pre-release below 1.15.0, Python tooling
    normalizes it to 1.15.0.dev0, as it normalizes a candidate), and every crate and package to the version the cut
    gave the branch, so the next cut's bump starts from the numbers the registries will hold."""
    repo.cut()
    main_head = repo.git("rev-parse", "origin/main")
    done = subprocess.run(
        [sys.executable, str(repo.work / SYNC), "origin/release/1.14", "1.15.0-dev"],
        cwd=repo.work,
        env=repo.env(),
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    for path in VERSION_FILES:
        assert repo.version_of(path) == "1.15.0-dev", path
    assert repo.version_of("crates/tool_parser/Cargo.toml") == "0.6.0", (
        "the branch's number, not above it"
    )
    assert 'tool-parser = { version = "0.6.0"' in repo.read("Cargo.toml")
    assert "model_gateway/Cargo.toml: 1.13.0 -> 1.15.0-dev" in done.stdout
    assert "crates/tool_parser/Cargo.toml: 0.5.0 -> 0.6.0" in done.stdout
    assert repo.git("rev-parse", "origin/main") == main_head, (
        "the files move; main moves by a pull request"
    )
    # the checker reads the moved tree as consistent: 1.15.0-dev is a version no release tag carries,
    # and the mirrors carry the gateway's string as written
    checked = subprocess.run(
        [sys.executable, "scripts/check_release_versions.py", "--check", "v1.13.0"],
        cwd=repo.work,
        env=repo.env(),
        capture_output=True,
        text=True,
    )
    assert checked.returncode == 0, checked.stdout
    assert "All versions consistent." in checked.stdout
    # the next cut's guard and bump-main's comparison read the development version above the previous final
    again = subprocess.run(
        [sys.executable, str(repo.work / SYNC), "origin/release/1.14", "1.14.0"],
        cwd=repo.work,
        env=repo.env(),
        capture_output=True,
        text=True,
    )
    assert again.returncode == 0, again.stderr
    assert (
        "the gateway reads 1.15.0-dev, above 1.14.0: it and its mirrors keep that" in again.stdout
    )
    assert repo.version_of("model_gateway/Cargo.toml") == "1.15.0-dev"


def test_sync_versions_carries_the_dependency_floors_a_release_raised(repo: Sandbox) -> None:
    """A release branch raises the floor a package puts on another of ours when it ships fields the
    code reads (the servicer's floor on the proto package). After the release, main's floor follows
    the branch's: the whole line, comment included, where the requirement is alone on its line
    (the comment states the floor's reason), the requirement alone where it sits inline in an
    extra, the rest of that line staying main's; a floor on a package that is not ours is left to
    the maintainer; a floor main already raised higher stays; a second run changes nothing."""
    # main's floor quotes an older floor in its comment and has an entry after it
    floor = '    "smg-grpc-proto>=0.4.0",  # up from "smg-grpc-proto>=0.3.0"\n'
    extra = 'mlx = ["smg-grpc-proto>=0.4.0", "mlx>=0.22.0"]\n'
    # an extra whose first entry is a literal string with a URL fragment: its hash opens no comment
    frag = "cu = ['wheel @ https://example.org/w.whl#sha256=abc', \"smg-grpc-proto>=0.4.0\"]\n"
    repo.commit_main(
        "chore(servicer): the floors",
        "grpc_servicer/pyproject.toml",
        '[project]\nname = "smg-grpc-servicer"\nversion = "0.14.0"\n'
        f'dependencies = [\n{floor}    "grpcio>=1.60",\n]\n\n[project.optional-dependencies]\n{extra}{frag}',
    )
    repo.cut()
    # the branch's bump raised the floors with what it published (by hand, as a cut's review does):
    # the dependencies entry, last in its array and without a comma there, the extra's inline
    # requirement, and one that is not ours
    repo.git("switch", "--quiet", "release/1.14")
    raised = '    "smg-grpc-proto>=0.5.0"  # 0.5.0 adds the fields the servicer reads\n'
    repo.put(
        "grpc_servicer/pyproject.toml",
        repo.read("grpc_servicer/pyproject.toml")
        .replace(floor + '    "grpcio>=1.60",\n', '    "grpcio>=1.70",\n' + raised)
        .replace(extra, 'mlx = ["smg-grpc-proto>=0.5.0", "mlx>=0.23.0"]\n')
        .replace(frag, frag.replace("0.4.0", "0.5.0")),
    )
    repo.git("add", "-A")
    repo.git("commit", "--quiet", "-m", "chore(release): the servicer's floors")
    repo.land_on_release()
    repo.git("switch", "--quiet", "main")
    done = subprocess.run(
        [sys.executable, str(repo.work / SYNC), "origin/release/1.14", "1.14.0"],
        cwd=repo.work,
        env=repo.env(),
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    text = repo.read("grpc_servicer/pyproject.toml")
    # the branch's requirement and comment, main's comma (an entry follows on main): one line
    followed = '    "smg-grpc-proto>=0.5.0",  # 0.5.0 adds the fields the servicer reads\n'
    assert followed in text, text
    assert text.count("smg-grpc-proto>=0.5.0") == 3 and "0.3.0" not in text, text
    assert 'mlx = ["smg-grpc-proto>=0.5.0", "mlx>=0.22.0"]\n' in text, text
    assert frag.replace("0.4.0", "0.5.0") in text, text
    assert tomllib.loads(text)["project"]["dependencies"] == [
        "smg-grpc-proto>=0.5.0",
        "grpcio>=1.60",
    ]
    assert '"grpcio>=1.60"' in text, "a floor on a package that is not ours is the maintainer's"
    assert done.stdout.count("grpc_servicer/pyproject.toml smg-grpc-proto>=: 0.4.0 -> 0.5.0") == 3
    assert "0.4.0 -> 0.5.0 (requirement 3)" in done.stdout, done.stdout
    again = subprocess.run(
        [sys.executable, str(repo.work / SYNC), "origin/release/1.14", "1.14.0"],
        cwd=repo.work,
        env=repo.env(),
        capture_output=True,
        text=True,
    )
    assert again.returncode == 0 and "already follows" in again.stdout, again.stdout
    # a floor main raised above the branch's stays where main put it
    higher = text.replace(followed, '    "smg-grpc-proto>=0.6.0",  # main went further\n')
    assert higher != text
    repo.put("grpc_servicer/pyproject.toml", higher)
    kept = subprocess.run(
        [sys.executable, str(repo.work / SYNC), "origin/release/1.14", "1.14.0"],
        cwd=repo.work,
        env=repo.env(),
        capture_output=True,
        text=True,
    )
    assert kept.returncode == 0 and "already follows" in kept.stdout, kept.stdout
    assert repo.read("grpc_servicer/pyproject.toml") == higher


def test_sync_versions_refuses_a_floor_it_cannot_pair_by_position(repo: Sandbox) -> None:
    """A name that occurs a different number of times in main's file and the release's (an extra
    added on one side since the cut, placed before an older one) cannot be paired by position: the
    sync names the file, the name and the counts, moves nothing for that name and fails, so the
    release flow stops for a move by hand rather than writing a floor onto the wrong entry."""
    floor = '    "smg-grpc-proto>=0.4.0",\n'
    extra = 'mlx = ["smg-grpc-proto>=0.4.0", "mlx>=0.22.0"]\n'
    repo.commit_main(
        "chore(servicer): the floors",
        "grpc_servicer/pyproject.toml",
        '[project]\nname = "smg-grpc-servicer"\nversion = "0.14.0"\n'
        f'dependencies = [\n{floor}    "grpcio>=1.60",\n]\n\n[project.optional-dependencies]\n{extra}',
    )
    repo.cut()
    repo.git("switch", "--quiet", "release/1.14")
    repo.put(
        "grpc_servicer/pyproject.toml",
        repo.read("grpc_servicer/pyproject.toml").replace("0.4.0", "0.5.0"),
    )
    repo.git("add", "-A")
    repo.git("commit", "--quiet", "-m", "chore(release): the servicer's floors")
    repo.land_on_release()
    repo.git("switch", "--quiet", "main")
    # main added an extra with its own requirement on the proto package, before the mlx extra
    repo.put(
        "grpc_servicer/pyproject.toml",
        repo.read("grpc_servicer/pyproject.toml").replace(
            extra, 'tpu = ["smg-grpc-proto>=0.4.0"]\n' + extra
        ),
    )
    repo.git("add", "-A")
    repo.git("commit", "--quiet", "-m", "feat(servicer): a tpu extra")
    done = subprocess.run(
        [sys.executable, str(repo.work / SYNC), "origin/release/1.14", "1.14.0"],
        cwd=repo.work,
        env=repo.env(),
        capture_output=True,
        text=True,
    )
    assert done.returncode == 1, done.stdout + done.stderr
    assert (
        "sync_versions: grpc_servicer/pyproject.toml: smg-grpc-proto>= occurs 3 time(s) here and 2"
        " in origin/release/1.14: the pairing is ambiguous, not moved (move it by hand)"
    ) in done.stderr, done.stderr
    text = repo.read("grpc_servicer/pyproject.toml")
    assert "0.5.0" not in text and text.count("smg-grpc-proto>=0.4.0") == 3, text
    # the move by hand the message asks for: main carries the release's floors, the counts still
    # differ (the release never takes main's new extra), nothing is pending, and the rerun passes
    repo.put("grpc_servicer/pyproject.toml", text.replace("0.4.0", "0.5.0"))
    repo.git("add", "-A")
    repo.git("commit", "--quiet", "-m", "chore(servicer): the floors by hand")
    again = subprocess.run(
        [sys.executable, str(repo.work / SYNC), "origin/release/1.14", "1.14.0"],
        cwd=repo.work,
        env=repo.env(),
        capture_output=True,
        text=True,
    )
    assert again.returncode == 0, again.stdout + again.stderr
    assert "ambiguous" not in again.stderr and "already follows" in again.stdout, again.stdout
    # main dropping its extras is the same case the other way (one occurrence against two)
    repo.put(
        "grpc_servicer/pyproject.toml",
        repo.read("grpc_servicer/pyproject.toml")
        .replace('tpu = ["smg-grpc-proto>=0.5.0"]\n', "")
        .replace(extra.replace("0.4.0", "0.5.0"), ""),
    )
    assert repo.read("grpc_servicer/pyproject.toml").count("smg-grpc-proto>=") == 1
    repo.git("add", "-A")
    repo.git("commit", "--quiet", "-m", "chore(servicer): the extras dropped")
    dropped = subprocess.run(
        [sys.executable, str(repo.work / SYNC), "origin/release/1.14", "1.14.0"],
        cwd=repo.work,
        env=repo.env(),
        capture_output=True,
        text=True,
    )
    assert dropped.returncode == 0 and "ambiguous" not in dropped.stderr, dropped.stderr


def test_sync_versions_usage_and_version_order() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("sync_versions", ROOT / SYNC)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    key = module.version_key
    assert key("1.14.0-rc.1") < key("1.14.0-rc.2") < key("1.14.0") < key("1.14.1-rc.1")
    # main's development version: below the candidates and the final of its number, above every release before it
    assert (
        key("1.14.0") < key("1.15.0-dev") < key("1.15.0-rc.1") < key("1.15.0") < key("1.16.0-dev")
    )
    for bad in ("1.14", "1.15.0-dev0", "1.15.0.dev0", "1.15.0-dev.1"):
        with pytest.raises(SystemExit):
            key(bad)
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
