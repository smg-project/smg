#!/usr/bin/env python3
"""Pre-release version check for the workspace crates and the Python packages.

For every component (a workspace crate or a Python package), since the given tag:
whether its files changed, whether its version was bumped, and, for a crate, whether
the workspace root's dependency entry agrees. A component's files include what the
symlinks under its path point at: a package that ships another directory's files through
a link (the proto package ships crates/grpc_client/proto that way) ships their changes
too, so they move its version. A version is bumped when no release tag
``v<x.y.z>`` (the given tag, or one another release line made) carries it with other
code: the registries take a version once. A crate held at its published version is
republished when a workspace dependency's new version leaves a requirement its published
copy carries (cargo's caret): at a patch step, or at its own breaking step when the crate's
public API carries the dependency. Then the files that must mirror the gateway's
version. Issues are listed with the bump conventional commits ask for (``feat!:`` or
``BREAKING CHANGE``: major; ``feat:``: minor; otherwise patch), stepped past the
versions release tags carry, and, on request, applied.

Usage:
    check_release_versions.py [tag] [--check | --fix | --plan]

    tag       the release to compare against (default: the newest tag; for --plan the
              highest final release tag, what a cut compares against)
    --check   report only, never prompt (for CI and dry runs)
    --fix     apply the proposed fixes without prompting
    --plan    the bump the next cut writes against the tag, quietly: the proposed versions
              only, nothing changed, exit 0 (on main every crate changed since the last
              release reads as unbumped by design; the cut bumps them); VERSION_OVERRIDE
              may be the cut's candidate version here (X.Y.Z-rc.N), what the cut writes
              to the gateway and its mirrors; without it, main's development version
              names it (1.15.0-dev: the first candidate, 1.15.0-rc.1)

Environment:
    VERSION_OVERRIDE  X.Y.Z: the gateway's new version instead of the detected bump

Exit code 0 = all versions consistent (or fixed, or --plan), 1 = issues found and not fixed.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import subprocess
import sys
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

# --- what is checked ----------------------------------------------------------


@dataclass(frozen=True)
class Component:
    """A versioned unit of the repository: a workspace crate or a Python package."""

    kind: str  # "crate" | "package"
    name: str
    path: str  # its directory, relative to the repository root
    version_file: str  # the file carrying its version, relative to the root
    dep_key: str | None = None  # its key under [workspace.dependencies]; None = no entry


def crate(name: str, path: str, dep_key: str | None = None) -> Component:
    return Component("crate", name, path, f"{path}/Cargo.toml", dep_key)


def package(name: str, path: str, version_file: str) -> Component:
    return Component("package", name, path, version_file)


@dataclass(frozen=True)
class Mirror:
    """A file that must carry the gateway's version."""

    label: str
    file: str
    kind: str  # "cargo" | "pyproject" | "python" | "helm" | "workflow"


GATEWAY = "smg"

CRATES = (
    crate("openai-protocol", "crates/protocols", "openai-protocol"),
    crate("smg-mm-rdma", "crates/mm_rdma", "smg-mm-rdma"),
    crate("engine-zmq-client", "crates/engine_zmq_client", "engine-zmq-client"),
    crate("reasoning-parser", "crates/reasoning_parser", "reasoning-parser"),
    crate("tool-parser", "crates/tool_parser", "tool-parser"),
    crate("smg-symphony", "crates/symphony", "smg-symphony"),
    crate("smg-response-template", "crates/response_template", "smg-response-template"),
    crate("wfaas", "crates/workflow", "wfaas"),
    crate("llm-tokenizer", "crates/tokenizer", "llm-tokenizer"),
    crate("smg-auth", "crates/auth", "smg-auth"),
    crate("smg-mcp", "crates/mcp", "smg-mcp"),
    crate("kv-index", "crates/kv_index", "kv-index"),
    crate("smg-radix-tree", "crates/radix_tree", "smg-radix-tree"),
    crate("data-connector", "crates/data_connector", "smg-data-connector"),
    crate("llm-multimodal", "crates/multimodal", "llm-multimodal"),
    crate("smg-wasm", "crates/wasm", "smg-wasm"),
    crate("smg-mesh", "crates/mesh", "smg-mesh"),
    crate("smg-grpc-client", "crates/grpc_client", "smg-grpc-client"),
    crate("smg-external-router", "crates/external_router", "smg-external-router"),
    crate("smg-rl", "crates/rl", "smg-rl"),
    crate("engine-zmq-adapter", "crates/engine_zmq_adapter", "engine-zmq-adapter"),
    crate("engine-servicer", "crates/engine_servicer", "engine-servicer"),
    crate("smg-client", "clients/rust"),
    crate("openapi-gen", "clients/openapi-gen"),
    crate(GATEWAY, "model_gateway"),
)

PACKAGES = (
    package(
        "smg-grpc-proto", "crates/grpc_client/python", "crates/grpc_client/python/pyproject.toml"
    ),
    package("smg-grpc-servicer", "grpc_servicer", "grpc_servicer/pyproject.toml"),
)

MIRRORS = (
    Mirror("smg-python", "bindings/python/Cargo.toml", "cargo"),
    Mirror("smg-golang", "bindings/golang/Cargo.toml", "cargo"),
    Mirror("smg pyproject", "bindings/python/pyproject.toml", "pyproject"),
    # smg.__version__ as the installed package reports it; `make bump-version` writes it, this check reads it
    Mirror("smg version.py", "bindings/python/src/smg/version.py", "python"),
    Mirror("helm chart", "deploy/helm/smg/Chart.yaml", "helm"),
    Mirror("engine-docker", ".github/workflows/release-engine-docker.yml", "workflow"),
)


class Fatal(Exception):
    """A condition the check cannot proceed from; its text is the message."""


# --- version lines: read and rewrite ------------------------------------------

SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
RELEASE_VERSION = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:-rc\.(\d+))?$")
CUT_VERSION = re.compile(r"^\d+\.\d+\.\d+(-rc\.\d+)?$")  # what a cut writes: a final or a candidate
# what main reads between cuts: the next development version (1.15.0-dev once release/1.14 is cut)
DEV_VERSION = re.compile(r"^(\d+\.\d+\.\d+)-dev$")
FINAL_TAG = re.compile(r"^v\d+\.\d+\.\d+$")
TOML_VERSION = re.compile(r'^version\s*=\s*"([^"]*)"', re.M)
TOML_WORKSPACE_VERSION = re.compile(
    r"^version\.workspace\s*=\s*true|^version\s*=\s*\{\s*workspace\s*=\s*true", re.M
)
DUNDER_VERSION = re.compile(r'__version__\s*=\s*"([^"]*)"')
HELM_VERSION = re.compile(r'^version:\s*"?([^"\s]*)"?\s*$', re.M)
HELM_APP_VERSION = re.compile(r"^appVersion:.*$", re.M)
WORKFLOW_VERSION = re.compile(r"SMG_RELEASE_TAG:\s*v([0-9]\S*)")


def version_key(version: str) -> tuple[int, int, int, int, int] | None:
    """A sort key for ``X.Y.Z`` and ``X.Y.Z-rc.N`` (a candidate below its final); None otherwise."""
    match = RELEASE_VERSION.match(version)
    if not match:
        return None
    major, minor, patch, candidate = match.groups()
    return (int(major), int(minor), int(patch), 0 if candidate else 1, int(candidate or 0))


def first_candidate(version: str) -> str | None:
    """The candidate a cut writes from main's development version: ``1.15.0-dev`` -> ``1.15.0-rc.1``;
    None for any other spelling (a cut names its version then)."""
    match = DEV_VERSION.match(version)
    return f"{match.group(1)}-rc.1" if match else None


def version_kind(file: str) -> str:
    """How a file spells its version: Cargo.toml and pyproject.toml alike, or ``__version__``."""
    return "toml" if file.endswith(".toml") else "python"


def read_version(kind: str, text: str, file: str) -> str:
    """The version a file carries; ``kind`` is a Mirror kind or a version_kind()."""
    if kind == "cargo" and TOML_WORKSPACE_VERSION.search(text):
        raise Fatal(
            f"{file} uses workspace versioning; this tool expects explicit version strings."
        )
    pattern = {
        "cargo": TOML_VERSION,
        "pyproject": TOML_VERSION,
        "toml": TOML_VERSION,
        "python": DUNDER_VERSION,
        "helm": HELM_VERSION,
        "workflow": WORKFLOW_VERSION,
    }[kind]
    match = pattern.search(text)
    if not match:
        raise Fatal(f"{file}: no version found.")
    return match.group(1)


def write_version(kind: str, text: str, old: str, new: str) -> str | None:
    """The file's text with its version set to ``new``; None when the rewrite did not take."""
    if kind in ("cargo", "pyproject", "toml"):
        text = TOML_VERSION.sub(f'version = "{new}"', text, count=1)
        done = re.search(rf'^version = "{re.escape(new)}"', text, re.M)
    elif kind == "python":
        text = DUNDER_VERSION.sub(f'__version__ = "{new}"', text, count=1)
        done = re.search(rf'__version__ = "{re.escape(new)}"', text)
    elif kind == "helm":
        # the chart's appVersion follows unconditionally, so drift between the two heals
        text = re.sub(rf"^(version:\s*){re.escape(old)}", rf"\g<1>{new}", text, count=1, flags=re.M)
        text = HELM_APP_VERSION.sub(f'appVersion: "{new}"', text, count=1)
        done = re.search(rf"^version:\s*{re.escape(new)}", text, re.M) and re.search(
            rf'^appVersion: "{re.escape(new)}"', text, re.M
        )
    elif kind == "workflow":
        # the one SMG_RELEASE_TAG line pins the release; the run names and fallbacks read it from there
        text = re.sub(
            rf"(SMG_RELEASE_TAG:\s*)v{re.escape(old)}(?=\s|$)",
            rf"\g<1>v{new}",
            text,
            count=1,
            flags=re.M,
        )
        done = re.search(rf"SMG_RELEASE_TAG:\s*v{re.escape(new)}(?=\s|$)", text, re.M)
    else:
        raise ValueError(kind)
    return text if done else None


def workspace_dep_version(root_toml: str, dep_key: str) -> str | None:
    """The version the root manifest pins for a workspace dependency."""
    match = re.search(rf'^{re.escape(dep_key)} .*?version = "([^"]*)"', root_toml, re.M)
    return match.group(1) if match else None


def write_workspace_dep_version(root_toml: str, dep_key: str, old: str, new: str) -> str | None:
    key = re.escape(dep_key)
    was, now = rf'^{key} = {{ version = "{re.escape(old)}"', f'{dep_key} = {{ version = "{new}"'
    text = re.sub(was, now, root_toml, count=1, flags=re.M)
    return text if re.search(rf'^{key} .* version = "{re.escape(new)}"', text, re.M) else None


# --- the bump conventional commits ask for ------------------------------------

BREAKING_SUBJECT = re.compile(r"^[a-z]+(\([^)]*\))?!:")
FEAT_SUBJECT = re.compile(r"^feat(\([^)]*\))?:")


def bump_level(commits: Iterable[tuple[str, str]]) -> str:
    """major on a breaking subject or footer, minor on a feat, patch otherwise."""
    level = "patch"
    for subject, body in commits:
        if BREAKING_SUBJECT.search(subject) or "BREAKING CHANGE" in body:
            return "major"
        if FEAT_SUBJECT.search(subject):
            level = "minor"
    return level


def bumped(version: str, level: str) -> str:
    if not SEMVER.match(version):
        raise Fatal(f"bump only supports X.Y.Z versions, got: {version}")
    major, minor, patch = (int(part) for part in version.split("."))
    return {
        "major": f"{major + 1}.0.0",
        "minor": f"{major}.{minor + 1}.0",
        "patch": f"{major}.{minor}.{patch + 1}",
    }[level]


# --- the repository -----------------------------------------------------------


def with_legacy(path: str) -> list[str]:
    """The path and, for the crates moved under crates/, where tags older than the move keep it."""
    return [path, path[len("crates/") :]] if path.startswith("crates/") else [path]


# The variables through which a hook or a wrapper points git at another repository; they
# take precedence over ``cwd``, so the tool's own calls drop them and read ``root`` alone.
_GIT_LOCATION_VARS = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_NAMESPACE",
    # the variables that change which commits git sees
    "GIT_SHALLOW_FILE",
    "GIT_GRAFT_FILE",
    "GIT_REPLACE_REF_BASE",
    "GIT_NO_REPLACE_OBJECTS",
)


class Repo:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.env = {k: v for k, v in os.environ.items() if k not in _GIT_LOCATION_VARS}

    def _run(self, *args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
        # git's output carries file content (the tag's copy of a version file, the diff,
        # the commit messages): read as UTF-8, like the working tree, under any locale.
        return subprocess.run(
            ["git", *args],
            cwd=self.root,
            env=self.env,
            check=check,
            capture_output=True,
            encoding="utf-8",
        )

    def git(self, *args: str) -> str:
        return self._run(*args, check=True).stdout

    def has_ref(self, ref: str) -> bool:
        return self._run("rev-parse", "--verify", "-q", ref).returncode == 0

    def newest_tag(self) -> str | None:
        tags = self.git("tag", "--sort=-creatordate").split()
        return tags[0] if tags else None

    def highest_final_tag(self) -> str | None:
        """The highest vX.Y.Z tag; a candidate (-rc.N) is not a final."""
        finals = [t for t in self.git("tag", "--list", "v[0-9]*").split() if FINAL_TAG.match(t)]
        return (
            max(finals, key=lambda t: tuple(int(n) for n in t[1:].split("."))) if finals else None
        )

    def show(self, ref: str, path: str) -> str | None:
        """A file's text at a ref; None when it is not there."""
        result = self._run("show", f"{ref}:{path}")
        return result.stdout if result.returncode == 0 else None

    def release_tags(self) -> list[str]:
        """Every release tag (``v<x.y.z>`` or ``v<x.y.z>-rc.<n>``), whichever line made it; a tag
        of another shape (``v1.14.0-qa``) published nothing and is not one."""
        tags = self.git("tag", "--list", "v[0-9]*").split()
        return [tag for tag in tags if version_key(tag[1:]) is not None]

    def texts_at(self, wanted: Sequence[tuple[str, str]]) -> dict[tuple[str, str], str]:
        """The text of each (ref, path) that exists, read in one ``cat-file --batch`` call."""
        out = subprocess.run(
            ["git", "cat-file", "--batch"],
            cwd=self.root,
            env=self.env,
            check=True,
            input="".join(f"{ref}:{path}\n" for ref, path in wanted).encode(),
            capture_output=True,
        ).stdout
        texts: dict[tuple[str, str], str] = {}
        position = 0
        # one answer per request, in order: "<sha> blob <size>\n<text>\n", or "<request> missing"
        for ref_path in wanted:
            end = out.index(b"\n", position)
            header = out[position:end].decode("utf-8")
            position = end + 1
            if header.endswith((" missing", " ambiguous")):
                continue
            size = int(header.rsplit(" ", 1)[1])
            texts[ref_path] = out[position : position + size].decode("utf-8")
            position += size + 1
        return texts

    def resolve_path(self, path: str) -> tuple[str | None, list[str]]:
        """``path`` at HEAD with every symlink along it resolved, one component at a time: a
        prefix that is a link has its pointee spliced in where it sits, ``..`` is collapsed only
        once the component before it is resolved, and the links crossed come back beside the
        result. None when the path leaves the repository, reaches an absolute pointee or crosses
        more links than the OS follows (a loop; a link crossed twice on one path is not one)."""
        resolved: list[str] = []
        crossed: list[str] = []
        remaining = path.split("/")
        while remaining:
            part = remaining.pop(0)
            if part in ("", "."):
                continue
            if part == "..":
                if not resolved:
                    return None, crossed
                resolved.pop()
                continue
            candidate = "/".join([*resolved, part])
            if not self.git("ls-tree", "HEAD", "--", candidate).startswith("120000 "):
                resolved.append(part)
                continue
            if len(crossed) >= 40:
                return None, crossed
            crossed.append(candidate)
            pointee = self.show("HEAD", candidate) or ""
            if not pointee or os.path.isabs(pointee):
                return None, crossed
            remaining = [*pointee.split("/"), *remaining]
        return "/".join(resolved), crossed

    def symlink_targets(self, path: str) -> list[str]:
        """What the symlinks under ``path`` at HEAD point at, as repository paths, followed to
        the end of every chain: a target that is a link, a directory holding one, or a path
        running through a directory link adds what that points at, each read once. The links
        crossed on the way count as targets too (a retarget changes what ships) but are not
        read further: what lies behind a directory link beyond the path through it is not the
        component's. A link into ``path`` itself, outside the repository or absolute is left
        out."""
        here, inside = path.rstrip("/"), path.rstrip("/") + "/"
        targets: list[str] = []
        pending = [path]
        while pending:
            entries = self.git("ls-tree", "-r", "-z", "HEAD", "--", pending.pop(0)).split("\0")
            for entry in entries:
                if not entry.startswith("120000 "):
                    continue
                link = entry.split("\t", 1)[1]
                pointee = self.show("HEAD", link) or ""
                if not pointee or os.path.isabs(pointee):
                    continue
                target, crossed = self.resolve_path(os.path.join(os.path.dirname(link), pointee))
                for found in [*crossed, target]:
                    if found is None or found == here or found.startswith(inside):
                        continue
                    if found not in targets:
                        targets.append(found)
                        if found == target:
                            pending.append(found)
        return targets

    def changed_since(self, tag: str, paths: Sequence[str]) -> list[str]:
        """``git diff --name-status`` lines since the tag, renames detected."""
        return self.git("diff", "--name-status", "-M", f"{tag}..HEAD", "--", *paths).splitlines()

    def diff_since(self, tag: str, paths: Sequence[str]) -> str:
        return self.git("diff", "-M", f"{tag}..HEAD", "--", *paths)

    def commits_since(self, tag: str, paths: Sequence[str]) -> list[tuple[str, str]]:
        """(subject, body) of every non-merge commit since the tag touching the paths."""
        log = self.git("log", f"{tag}..HEAD", "--no-merges", "--format=%s%x00%b%x1e", "--", *paths)
        entries = (
            entry.lstrip("\n").partition("\x00") for entry in log.split("\x1e") if entry.strip()
        )
        return [(subject, body) for subject, _, body in entries]

    def read(self, path: str) -> str:
        return (self.root / path).read_text(encoding="utf-8")

    def write(self, path: str, text: str) -> None:
        (self.root / path).write_text(text, encoding="utf-8")


# --- one component's verdict --------------------------------------------------


@dataclass
class Verdict:
    component: Component
    # against the tag the verdict is judged by; 0 = no files to publish (the carrier path may still check the pin)
    files_changed: int
    version: str = ""
    tag_version: str | None = None  # None = not at the tag: a new component
    tagged: dict[str, str] = field(default_factory=dict)  # version -> the release tag carrying it
    carried_by: str | None = None  # the release tag, not the given one, that carries the version
    unbumped: bool = False  # the version is a release tag's, carried there with other code
    level: str | None = None  # the bump asked for when the version was not bumped
    workspace_version: str | None = None  # the root's entry when it disagrees
    # why a held crate moves: the published requirement a dependency's new version leaves
    dependency: str | None = None
    # the manifest's own requirements on path dependencies that the versions written leave (cargo
    # refuses the tree over them): the entry's key, its requirement, the version to write
    requirements: list[tuple[str, str, str]] = field(default_factory=list)


def version_file_has_other_changes(repo: Repo, tag: str, file: str) -> bool:
    """Whether the version file changed beyond its version line (a pure rename is no change)."""
    paths = with_legacy(file)
    if not any(repo.changed_since(tag, paths)):
        return False
    version_line = re.compile(
        r"^[+-]\s*version\s*=" if file.endswith(".toml") else r"^[+-].*__version__"
    )
    for line in repo.diff_since(tag, paths).splitlines():
        if line[:1] in ("+", "-") and not (len(line) >= 3 and set(line[:3]) <= {"+", "-"}):
            if not version_line.match(line):
                return True
    return False


def component_paths(repo: Repo, component: Component) -> list[str]:
    """The paths whose changes are the component's: its directory (and the place a crate moved
    under crates/ had before), then what the symlinks under it point at."""
    return [f"{p}/" for p in with_legacy(component.path)] + repo.symlink_targets(component.path)


def files_changed(repo: Repo, tag: str, component: Component) -> int:
    """The component's files changed since the tag, the targets of its symlinks among them; pure
    renames and a version-only edit of the component's own version file do not count (a manifest
    of the same name a link reaches is content)."""
    version_files = with_legacy(component.version_file)
    count = sum(
        1
        for line in repo.changed_since(tag, component_paths(repo, component))
        if not line.startswith("R100") and line.rsplit("\t", 1)[-1] not in version_files
    )
    if version_file_has_other_changes(repo, tag, component.version_file):
        count += 1
    return count


def manifest_kind(component: Component) -> str:
    return version_kind(component.version_file) if component.kind == "package" else "cargo"


def tagged_versions(repo: Repo, components: Sequence[Component]) -> dict[str, dict[str, str]]:
    """Per component name, {version: release tag} over every release tag that carries it (the
    newest tag for a version several carry), read in one git call; a tag whose copy has no
    readable version is skipped."""
    tags = sorted(repo.release_tags(), key=lambda tag: version_key(tag[1:]) or (-1, 0, 0, 0, 0))
    wanted = [
        (tag, path) for c in components for tag in tags for path in with_legacy(c.version_file)
    ]
    texts = repo.texts_at(wanted)
    table: dict[str, dict[str, str]] = {}
    for component in components:
        found: dict[str, str] = {}
        for tag in tags:
            for path in with_legacy(component.version_file):
                text = texts.get((tag, path))
                if text is None:
                    continue
                try:
                    found[tag] = read_version(manifest_kind(component), text, f"{path} at {tag}")
                except Fatal:
                    pass
                break
        table[component.name] = {found[tag]: tag for tag in tags if tag in found}
    return table


def check(
    repo: Repo, tag: str, component: Component, root_toml: str, tagged: dict[str, str]
) -> Verdict:
    verdict = Verdict(component, files_changed(repo, tag, component), tagged=tagged)
    if not (repo.root / component.version_file).is_file():
        if verdict.files_changed:  # its files went since the tag: the table still names it
            raise Fatal(f"{component.version_file} is gone: drop {component.name} from the table.")
        return verdict  # a table entry this tree never had
    kind = manifest_kind(component)
    verdict.version = read_version(kind, repo.read(component.version_file), component.version_file)
    for path in with_legacy(component.version_file):
        text = repo.show(tag, path)
        if text is not None:
            verdict.tag_version = read_version(kind, text, f"{path} at {tag}")
            break
    # The release tag whose copy of the component carries HEAD's version: the given tag, or one
    # another line made (a patch of an older line synced to main, a cut before that patch). The
    # version is bumped only when no tag carries it, or the carrier's code is this code already.
    carrier = tag if verdict.version == verdict.tag_version else tagged.get(verdict.version)
    changed_since_tag = verdict.files_changed
    if carrier is not None and carrier != tag:
        verdict.carried_by = carrier
        verdict.files_changed = files_changed(repo, carrier, component)
    if verdict.files_changed == 0:
        # Changed since the comparison tag, at a version another line's tag carries with this very
        # code: nothing to publish, but the workspace pin still has to follow the version (the
        # comparison tag's path reads it below; this path must not skip it).
        if changed_since_tag and component.dep_key:
            pinned = workspace_dep_version(root_toml, component.dep_key)
            if pinned != verdict.version:
                verdict.workspace_version = pinned or ""
        return verdict
    verdict.unbumped = carrier is not None
    if verdict.unbumped:
        verdict.level = bump_level(repo.commits_since(tag, component_paths(repo, component)))
    elif component.dep_key:
        pinned = workspace_dep_version(root_toml, component.dep_key)
        if pinned != verdict.version:
            verdict.workspace_version = pinned or ""
    return verdict


# --- the report, the proposal, the fixes --------------------------------------


class Out:
    """Writes the report; colours only on a terminal."""

    COLOURS = {
        "red": "0;31",
        "green": "0;32",
        "yellow": "1;33",
        "blue": "0;34",
        "cyan": "0;36",
        "bold": "1",
    }

    def __init__(self, stream) -> None:
        self.stream = stream
        self.tty = hasattr(stream, "isatty") and stream.isatty()

    def paint(self, colour: str, text: str) -> str:
        return f"\033[{self.COLOURS[colour]}m{text}\033[0m" if self.tty else text

    def line(self, text: str = "") -> None:
        print(text, file=self.stream)

    def level(self, level: str) -> str:
        return self.paint({"major": "red", "minor": "yellow", "patch": "cyan"}[level], level)


@dataclass
class Plan:
    """What the check found and what fixing it means."""

    verdicts: list[Verdict]
    mirrors: list[tuple[Mirror, str]]  # the mirrors that disagree, with the version they carry
    gateway_target: str
    override: str | None
    gateway_version: str = ""  # what the gateway's manifest reads now
    cut_version: str | None = (
        None  # a plan only: what the cut writes to the gateway and its mirrors, whatever the verdict
    )
    # a plan only: the cut version was read from main's development version, not given as VERSION_OVERRIDE
    cut_version_derived: bool = False

    @property
    def gateway_set(self) -> bool:
        """Whether a plan moves the gateway to the cut's version (the cut always writes it; a line when it differs)."""
        return self.cut_version is not None and self.gateway_version != self.cut_version

    @property
    def issues(self) -> int:
        return (
            sum(1 for v in self.verdicts if v.unbumped)
            + sum(1 for v in self.verdicts if v.workspace_version is not None)
            + sum(len(v.requirements) for v in self.verdicts)
            + len(self.mirrors)
        )

    def proposal(self, verdict: Verdict) -> tuple[str, list[str]]:
        """The version to write, and the versions stepped past because a release tag carries them."""
        if self.override and verdict.component.name == GATEWAY:
            return self.override, []
        new, stepped = bumped(verdict.version, verdict.level or "patch"), []
        while new in verdict.tagged:
            stepped.append(f"v{new} is {verdict.tagged[new]}'s")
            new = bumped(new, "patch")
        return new, stepped

    def new_version(self, verdict: Verdict) -> str:
        return self.proposal(verdict)[0]


def report(verdict: Verdict, out: Out) -> None:
    c, name = verdict.component, out.paint("bold", verdict.component.name)
    where, n = f"({c.path}/)", f"{verdict.files_changed} file(s) changed"
    ok, warn, bad = out.paint("green", "✓"), out.paint("yellow", "!"), out.paint("red", "✗")
    plural = "s" if "; " in (verdict.dependency or "") else ""
    if verdict.dependency and not verdict.files_changed:
        out.line(
            f"  {warn} {name} {where} — unchanged, but its published requirement{plural}"
            f" {verdict.dependency} [{out.level(verdict.level)}]"
        )
    elif verdict.unbumped:
        carried = f", {verdict.carried_by}'s" if verdict.carried_by else ""
        out.line(
            f"  {warn} {name} {where} — {n} but version not bumped (v{verdict.version}{carried})"
            f" [{out.level(verdict.level)}]"
        )
        if verdict.dependency:
            out.line(f"    {warn} and its published requirement{plural} {verdict.dependency}")
    elif verdict.tag_version is None:
        out.line(f"  {ok} {name} {where} — new {c.kind} (v{verdict.version}), {n}")
    elif verdict.carried_by and not verdict.files_changed:
        out.line(
            f"  {ok} {name} {where} — v{verdict.version} is {verdict.carried_by}'s, with this code"
        )
    else:
        out.line(f"  {ok} {name} {where} — v{verdict.tag_version} → v{verdict.version} ({n})")
    if verdict.workspace_version is not None:
        out.line(
            f"    {bad} workspace Cargo.toml has {c.dep_key} v{verdict.workspace_version}, expected v{verdict.version}"
        )
    for key, requirement, new in verdict.requirements:
        out.line(f"    {bad} {c.version_file} has {key} v{requirement}, expected v{new}")


def caret_accepts(requirement: str, version: str) -> bool:
    """Whether cargo's default requirement on ``requirement`` (a caret) accepts ``version``: not
    older, and the same major; the same minor below 1.0; the same patch below 0.1."""
    old = tuple(int(n) for n in requirement.split("-")[0].split("."))
    new = tuple(int(n) for n in version.split("-")[0].split("."))
    if new < old:
        return False
    if old[0] > 0:
        return new[0] == old[0]
    if old[1] > 0:
        return new[:2] == old[:2]
    return new == old


LEVELS = ("patch", "minor", "major")


def breaking_level(version: str) -> str:
    """The step that leaves a version's own caret: major from 1.0 on, minor at 0.y, patch at 0.0.z.
    A candidate steps as its line does (the suffix carries no level), so a plan from one candidate
    to the next reads a carried dependency crossing its caret as a final would."""
    if not CUT_VERSION.match(version):
        raise Fatal(
            f"the dependency rule only steps X.Y.Z versions and their candidates, got: {version}"
        )
    major, minor, _ = (int(part) for part in version.split("-")[0].split("."))
    return "major" if major else "minor" if minor else "patch"


def manifest_at(repo: Repo, ref: str, component: Component) -> dict:
    """The component's manifest at the ref, parsed; empty when it is not there."""
    text = next(
        (text for path in with_legacy(component.version_file) if (text := repo.show(ref, path))),
        None,
    )
    if text is None:
        return {}
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise Fatal(f"{component.version_file} at {ref} is not TOML: {error}") from None


def dependency_entries(manifest: dict) -> list[tuple[str, object]]:
    """The entries of a manifest's published dependency tables: ``[dependencies]`` and
    ``[build-dependencies]``, at the top and under every ``[target.<cfg>]``; dev-dependencies
    are not published requirements."""
    tables = [manifest, *(t for t in manifest.get("target", {}).values() if isinstance(t, dict))]
    return [
        (key, spec)
        for table in tables
        for section in ("dependencies", "build-dependencies")
        for key, spec in table.get(section, {}).items()
    ]


def published_requirements(
    repo: Repo, ref: str, component: Component, crates: Sequence[Component]
) -> dict[str, str]:
    """A crate's requirements on the workspace crates as its copy at the ref (the release tag
    that carries its version) carries them, per crate name: the root's pin for a
    ``workspace = true`` entry (keyed as the root keys it), the entry's own version otherwise
    (a direct entry names the package: its ``package =``, else its key)."""
    root = repo.show(ref, "Cargo.toml") or ""
    by_key = {c.dep_key or c.name: c.name for c in crates}
    packages = {c.name for c in crates}
    requirements: dict[str, str] = {}
    for key, spec in dependency_entries(manifest_at(repo, ref, component)):
        if isinstance(spec, dict) and spec.get("workspace"):
            # a workspace entry is keyed as the root keys it
            name, pinned = by_key.get(key), workspace_dep_version(root, key)
        else:
            # a direct entry names the package: its `package =`, else its key
            package = spec.get("package", key) if isinstance(spec, dict) else key
            name = package if package in packages else None
            pinned = spec.get("version") if isinstance(spec, dict) else spec
        if name is None or name == component.name:
            continue
        if isinstance(pinned, str) and SEMVER.match(pinned):
            requirements[name] = pinned
    return requirements


def direct_requirements(
    repo: Repo, component: Component, crates: Sequence[Component]
) -> list[tuple[str, str, str]]:
    """The requirements the component's manifest writes itself on the workspace crates it reaches
    by path (a ``version`` beside the ``path``; the root pins a ``workspace = true`` entry), in
    every dependency table cargo resolves, dev-dependencies too: the entry's key, the crate's name,
    the requirement. A direct entry names the package (its ``package =``, else its key), not the
    root's workspace key. Read from the working tree, as cargo reads it and as the fix leaves it."""
    try:
        manifest = tomllib.loads(repo.read(component.version_file))
    except FileNotFoundError:
        return []
    except tomllib.TOMLDecodeError as error:
        raise Fatal(f"{component.version_file} is not TOML: {error}") from None
    packages = {
        c.name for c in crates
    }  # a direct entry names the package: its `package =`, else its key
    tables = [manifest, *(t for t in manifest.get("target", {}).values() if isinstance(t, dict))]
    found: list[tuple[str, str, str]] = []
    for table in tables:
        for section in ("dependencies", "build-dependencies", "dev-dependencies"):
            for key, spec in table.get(section, {}).items():
                if not isinstance(spec, dict) or spec.get("workspace") or not spec.get("path"):
                    continue
                name, requirement = spec.get("package", key), spec.get("version")
                if name not in packages or name == component.name:
                    continue
                if isinstance(requirement, str) and SEMVER.match(requirement):
                    found.append((key, name, requirement))
    return found


def write_dep_version(manifest: str, key: str, old: str, new: str) -> str | None:
    """The member manifest with the inline entry ``key``'s version moved from ``old`` to ``new``;
    None when no such line is there (an entry written as its own table is left to the operator)."""
    line = rf'^(\s*{re.escape(key)}\s*=\s*\{{[^\n]*?\bversion\s*=\s*"){re.escape(old)}(")'
    text, moved = re.subn(line, rf"\g<1>{new}\g<2>", manifest, count=1, flags=re.M)
    return text if moved else None


def source_name(repo: Repo, dependent: Component, dep: Component) -> str:
    """The name the dependent's sources use for the dependency: the key it declares the
    dependency under when that key renames the package (its own ``package =``, or the
    workspace entry's), else the dependency's library name (``[lib] name``, by default the
    package name). Both manifests are read at HEAD, with the sources."""
    key = root_key = dep.dep_key or dep.name
    for own_key, spec in dependency_entries(manifest_at(repo, "HEAD", dependent)):
        package = spec.get("package") if isinstance(spec, dict) else None
        # a rename, the root's key (a workspace entry) or a direct entry under the package name
        if package == dep.name or (package is None and own_key in (root_key, dep.name)):
            key = own_key
    if key != dep.name:
        return key.replace("-", "_")
    lib = manifest_at(repo, "HEAD", dep).get("lib", {})
    return lib.get("name") or dep.name.replace("-", "_")


COMMENTS = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
USE_TREE = re.compile(r"\buse\s+(?:::)?([A-Za-z_][A-Za-z0-9_]*)\s*(?:::\s*([^;]*?))?\s*;", re.S)
NAME = r"[A-Za-z_][A-Za-z0-9_]*"
CONSTANT = re.compile(r"\s*(?:const|static)\b(?!\s+(?:fn|unsafe|async|extern)\b)")


def use_leaves(tree: str, prefix: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], str | None]]:
    """The leaves of a ``use`` tree below the crate, each as its path segments and its ``as``
    alias: ``a::{b, c::{d, self}}`` gives a::b, a::c::d and a::c::self."""
    items: list[str] = []
    depth = start = 0
    for i, char in enumerate(tree):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        elif char == "," and depth == 0:
            items.append(tree[start:i])
            start = i + 1
    items.append(tree[start:])
    leaves: list[tuple[tuple[str, ...], str | None]] = []
    for item in items:
        item = item.strip()
        if not item:
            continue
        if item.endswith("}"):
            head, _, inner = item.partition("{")
            segments = tuple(s.strip() for s in head.strip().rstrip(":").split("::") if s.strip())
            leaves += use_leaves(inner[:-1], prefix + segments)
            continue
        alias = None
        if re.search(r"\s+as\s+", item):
            item, alias = (part.strip() for part in re.split(r"\s+as\s+", item, maxsplit=1))
        leaves.append((prefix + tuple(s.strip() for s in item.split("::") if s.strip()), alias))
    return leaves


def imported_names(text: str, ident: str) -> set[str] | None:
    """The names the ``use`` statements of a source bring in from the crate ``ident``: the leaf
    of each path or its ``as`` alias, the segment before a ``self``; ``_`` (a trait imported
    for its methods) brings in none. None when a glob import leaves them unknown."""
    names: set[str] = set()
    for match in USE_TREE.finditer(text):
        if match[1] != ident:
            continue
        for segments, alias in use_leaves(match[2] or ""):
            if "*" in segments:
                return None
            if segments and segments[-1] == "self":
                segments = segments[:-1]
            name = alias or (segments[-1] if segments else ident)
            if name not in ("_", ident) and re.fullmatch(NAME, name):
                names.add(name)
    return names


def public_spans(text: str) -> list[str]:
    """The heads of a source's public items: from a bare ``pub`` (``pub(crate)`` and its kin are
    not public) or an ``impl`` that begins an item (an ``impl Trait`` inside a signature opens no
    head of its own: in a public one it lies inside the ``pub`` span, in a private one it is no
    API) to the ``{``, ``;``, ``,`` or ``}`` that ends the head outside
    every bracket, across wrapped lines (``<=``, ``->``, ``=>`` and ``>=`` are operators, not
    brackets, and ``<<`` is a shift when whitespace follows it, as rustfmt writes one, else two
    brackets as in ``Vec<<T as Tr>::A>``). A ``const`` or ``static`` head ends at its ``=``: the
    initializer is not API. A ``pub enum`` or ``pub trait`` keeps its body, whose variants and
    methods are public without the word; a trait ``impl`` keeps the ``type … = …;`` lines of its
    body."""
    spans: list[str] = []
    for match in re.finditer(r"\bpub\b(?!\s*\()|^[ \t]*(?:unsafe[ \t]+)?(impl)\b", text, re.M):
        start = i = match.end()
        depth = 0
        constant = CONSTANT.match(text, start) is not None
        while i < len(text):
            char, before, after = text[i], text[i - 1 : i], text[i + 1 : i + 2]
            if char == "<":
                if after == "=" or (after == "<" and text[i + 2 : i + 3].isspace()):
                    i += 2
                    continue
                depth += 1
            elif char == ">":
                if before in ("-", "=") or after == "=":
                    i += 1
                    continue
                depth = max(depth - 1, 0)
            elif char in "([":
                depth += 1
            elif char in ")]":
                depth = max(depth - 1, 0)
            elif depth == 0 and (char in "{;,}" or (char == "=" and constant)):
                break
            i += 1
        head = text[start:i]
        if i < len(text) and text[i] == "{":
            end, depth = i, 0
            while end < len(text):
                depth += {"{": 1, "}": -1}.get(text[end], 0)
                end += 1
                if depth == 0:
                    break
            if re.match(r"\s*(enum|trait)\b", head):
                head = text[start:end]
            elif match[1] and re.search(r"\bfor\b", head):
                head += " ".join(re.findall(r"\btype\s+[^;]*;", text[i:end]))
        spans.append(head)
    return spans


def carries_in_public_api(repo: Repo, component: Component, dep: Component) -> bool:
    """Whether the crate's sources at HEAD carry the dependency in a public position: its path
    or the whole crate under the name the sources use for it (``source_name``), or a name a
    ``use`` of it brought into the file, in the head of a public item or an ``impl`` (the body
    of a public enum or trait, the associated types of a trait impl). A read of the text; a
    glob import of the dependency leaves it blind and reads as carried."""
    ident = source_name(repo, component, dep)
    by_path = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(ident)}::")
    whole = re.compile(rf"\s*use\s+(?:::)?{re.escape(ident)}\b")
    files = repo.git("ls-tree", "-r", "--name-only", "HEAD", "--", f"{component.path}/src")
    for path in files.splitlines():
        if not path.endswith(".rs"):
            continue
        text = COMMENTS.sub(" ", repo.show("HEAD", path) or "")
        names = imported_names(text, ident)
        if names is None:
            return True
        alternatives = "|".join(map(re.escape, sorted(names)))
        by_name = (
            re.compile(rf"(?<![A-Za-z0-9_:])(?:{alternatives})(?![A-Za-z0-9_])") if names else None
        )
        for span in public_spans(text):
            if whole.match(span) or by_path.search(span):
                return True
            if by_name is not None and by_name.search(span):
                return True
    return False


def dependency_moves(repo: Repo, tag: str, plan: Plan, crates: Sequence[Component]) -> None:
    """The dependency rule, repeated until no crate moves or rises: a crate held at a published
    version (the given tag's, or another line's tag that carries it) is republished when a
    workspace dependency's new version (the plan's, where the dependency moves here) leaves a
    requirement its published copy carries, and a crate already moving for its own changes
    takes at least the step such a requirement asks. Every requirement is judged, and the step
    is the highest they ask: the crate's own breaking step where its public API carries the
    dependency, patch otherwise. The move's own pin write supersedes a stale-pin sync."""
    by_name = {v.component.name: v for v in plan.verdicts if v.component.kind == "crate"}
    held = {
        v.component.name
        for v in by_name.values()
        if v.files_changed == 0
        and (v.carried_by or (v.tag_version is not None and v.version == v.tag_version))
    }
    moving = {v.component.name for v in by_name.values() if v.unbumped and v.files_changed}
    judged = [v for v in by_name.values() if v.component.name in held | moving]
    requirements = {
        v.component.name: published_requirements(repo, v.carried_by or tag, v.component, crates)
        for v in judged
    }
    levels = {v.component.name: LEVELS.index(v.level or "patch") for v in judged if v.unbumped}
    carried: dict[tuple[str, str], bool] = {}
    changed = True
    while changed:
        changed = False
        for verdict in judged:
            name, level = verdict.component.name, -1
            left: list[str] = []
            for dep, requirement in requirements[name].items():
                target = by_name.get(dep)
                if target is None:
                    continue
                new = plan.new_version(target) if target.unbumped else target.version
                if caret_accepts(requirement, new):
                    continue
                if (name, dep) not in carried:
                    carried[name, dep] = carries_in_public_api(
                        repo, verdict.component, target.component
                    )
                step = breaking_level(verdict.version) if carried[name, dep] else "patch"
                level = max(level, LEVELS.index(step))
                why = ", carried in its public API" if carried[name, dep] else ""
                left.append(f"{dep} ^{requirement} does not accept v{new}{why}")
            if not left:
                continue
            rises = level > levels.get(name, -1)
            if name in held or rises:
                verdict.dependency = "; ".join(left)
            if rises:
                levels[name] = level
                verdict.unbumped = True
                verdict.workspace_version = None
                verdict.level = LEVELS[level]
                changed = True


def requirement_moves(repo: Repo, plan: Plan, crates: Sequence[Component]) -> None:
    """The requirements the fix carries along: a member manifest's own ``version`` on a path
    dependency, when the version the plan writes for that crate (or the one it has) leaves the
    requirement's caret. Cargo refuses the tree over such an entry, which a read of the versions
    alone passes; the entry is an issue until it holds, and the fix writes the new version to it
    as it writes the root's pin."""
    by_name = {v.component.name: v for v in plan.verdicts if v.component.kind == "crate"}
    for verdict in by_name.values():
        for key, name, requirement in direct_requirements(repo, verdict.component, crates):
            target = by_name.get(name)
            if target is None:
                continue
            new = plan.new_version(target) if target.unbumped else target.version
            if not caret_accepts(requirement, new):
                verdict.requirements.append((key, requirement, new))


def examine(
    repo: Repo,
    tag: str,
    override: str | None,
    out: Out,
    crates: Sequence[Component] = CRATES,
    packages: Sequence[Component] = PACKAGES,
    mirrors: Sequence[Mirror] = MIRRORS,
    cut_version: str | None = None,
) -> Plan:
    """Walk the components and the mirrors, printing each verdict as the shell did.

    With ``cut_version`` (a plan) the gateway and its mirrors are measured against that version, as the cut
    writes it unconditionally (``make bump-version`` first), not against the checker's own step.
    """
    root_toml = repo.read("Cargo.toml")
    tagged = tagged_versions(repo, (*crates, *packages))
    verdicts = []
    for component in (*crates, *packages):
        verdict = check(repo, tag, component, root_toml, tagged.get(component.name, {}))
        verdicts.append(verdict)

    gateway = next((v for v in verdicts if v.component.name == GATEWAY), None)
    gateway_component = next(c for c in crates if c.name == GATEWAY)
    target = read_version(
        "cargo", repo.read(gateway_component.version_file), gateway_component.version_file
    )
    plan = Plan(verdicts, [], target, override, gateway_version=target, cut_version=cut_version)
    dependency_moves(repo, tag, plan, crates)
    requirement_moves(repo, plan, crates)
    for verdict in verdicts:
        if verdict.files_changed or verdict.workspace_version is not None or verdict.dependency:
            report(verdict, out)
        elif verdict.requirements:
            report(verdict, out)
    if cut_version is not None:
        plan.gateway_target = cut_version
    elif gateway is not None and gateway.unbumped:
        plan.gateway_target = plan.new_version(gateway)
    for mirror in mirrors:
        carried = read_version(mirror.kind, repo.read(mirror.file), mirror.file)
        if carried != plan.gateway_target:
            out.line(
                f"  {out.paint('yellow', '!')} {out.paint('bold', mirror.label)} ({mirror.file}) — "
                f"v{carried}, expected v{plan.gateway_target}"
            )
            plan.mirrors.append((mirror, carried))
        else:
            out.line(
                f"  {out.paint('green', '✓')} {out.paint('bold', mirror.label)} ({mirror.file}) — v{carried}"
            )

    def count(kind: str, changed: bool) -> int:
        # a verdict with a pin to sync is "with changes" too: the Summary matches the list above
        return sum(
            1
            for v in verdicts
            if v.component.kind == kind
            and bool(
                v.files_changed or v.workspace_version is not None or v.dependency or v.requirements
            )
            == changed
        )

    out.line()
    out.line(
        out.paint("bold", "Summary:")
        + f" {count('crate', True)} crate(s) with changes, {count('crate', False)} unchanged;"
        f" {count('package', True)} Python package(s) with changes, {count('package', False)} unchanged"
    )
    return plan


def propose(plan: Plan, out: Out, title: str = "Proposed fixes:") -> None:
    out.line()
    out.line(out.paint("bold", title))
    sync = out.paint("blue", "sync")
    for v in plan.verdicts:
        c = v.component
        if v.unbumped and c.kind == "crate" and not (c.name == GATEWAY and plan.cut_version):
            new, stepped = plan.proposal(v)
            note = f"; {', '.join(stepped)}" if stepped else ""
            out.line(
                f"  {out.level(v.level)} {c.name} v{v.version} → v{new} ({c.version_file}{note})"
            )
            if c.dep_key:
                out.line(f"       sync workspace Cargo.toml {c.dep_key} → v{new}")
    for v in plan.verdicts:
        if v.workspace_version is not None:
            out.line(
                f"  {sync} workspace Cargo.toml {v.component.dep_key} v{v.workspace_version} → v{v.version}"
            )
    for v in plan.verdicts:
        for key, requirement, new in v.requirements:
            out.line(f"  {sync} {v.component.version_file} {key} v{requirement} → v{new}")
    for v in plan.verdicts:
        c = v.component
        if v.unbumped and c.kind == "package":
            new, stepped = plan.proposal(v)
            note = f"; {', '.join(stepped)}" if stepped else ""
            out.line(
                f"  {out.level(v.level)} {c.name} v{v.version} → v{new} ({c.version_file}{note})"
            )
    if plan.gateway_set:
        # the cut writes its version to the gateway whatever the conventional commits say: no level, a set
        manifest = next(
            v.component.version_file for v in plan.verdicts if v.component.name == GATEWAY
        )
        out.line(
            f"  {out.paint('blue', 'set')} {GATEWAY} v{plan.gateway_version} → v{plan.cut_version} ({manifest})"
        )
    for mirror, carried in plan.mirrors:
        out.line(f"  {sync} {mirror.label} v{carried} → v{plan.gateway_target} ({mirror.file})")


def show_plan(plan: Plan, tag: str, out: Out) -> None:
    """The bump the next cut writes against ``tag``: the proposal alone, nothing judged.

    The crates and packages move by their conventional commits since the tag, as the cut's ``--fix`` moves them.
    The gateway and its mirrors take the version the cut is given (``make bump-version VERSION=…`` runs first): with
    ``VERSION_OVERRIDE`` the plan shows that version, without it the checker's own step for the gateway.
    """
    gateway_unbumped = any(v.component.name == GATEWAY and v.unbumped for v in plan.verdicts)
    bumped_already = [
        f"{v.component.name} v{v.version}" + (" (new)" if v.tag_version is None else "")
        for v in plan.verdicts
        if v.files_changed
        and not v.unbumped
        and not (v.component.name == GATEWAY and plan.gateway_set)
    ]
    # what the proposal lists: the issues, the gateway counted once as a set when the cut's version is known (a gateway
    # already at that version is set nowhere: it stays among the components bumped already)
    moving = (
        plan.issues
        - (1 if gateway_unbumped and plan.cut_version else 0)
        + (1 if plan.gateway_set else 0)
    )
    if moving == 0:
        out.line(
            out.paint("bold", f"Nothing to bump against {tag}:")
            + " every crate and package changed since it is bumped already"
            + (f" ({', '.join(bumped_already)})" if bumped_already else "")
            + "; the mirrors agree."
        )
        return
    if plan.cut_version and plan.cut_version_derived:
        gateway = (
            f"the gateway and its mirrors to {plan.cut_version} (the first candidate of main's development version"
            f" {plan.gateway_version}; VERSION_OVERRIDE names another)"
        )
    elif plan.cut_version:
        gateway = f"the gateway and its mirrors to {plan.cut_version}"
    else:
        gateway = (
            "the gateway and its mirrors by the checker's own step (the cut writes the VERSION it is given there:"
            " pass it as VERSION_OVERRIDE for those lines)"
        )
    propose(
        plan,
        out,
        f"The bump the next cut writes against {tag}: the crates and packages by their conventional commits since it,"
        f" {gateway}. Nothing is changed here; `make release-cut` writes it:",
    )
    out.line()
    out.line(
        out.paint(
            "bold",
            f"{moving} component(s) and mirror(s) move (a crate's workspace pin follows it).",
        )
        + (
            f" Bumped already since {tag}, left as they are: {', '.join(bumped_already)}."
            if bumped_already
            else ""
        )
    )


def apply(plan: Plan, repo: Repo, out: Out) -> int:
    """Write the proposed versions; returns the number of rewrites that did not take."""
    failed = 0
    ok = out.paint("green", "✓")

    def rewrite(path: str, kind: str, old: str, new: str) -> bool:
        text = write_version(kind, repo.read(path), old, new)
        if text is None:
            out.line(f"    {out.paint('red', f'FAILED to update {path}')}")
            return False
        repo.write(path, text)
        out.line(f"  {ok} {path} → v{new}")
        return True

    def pin(dep_key: str, new: str) -> bool:
        old = workspace_dep_version(repo.read("Cargo.toml"), dep_key) or ""
        text = write_workspace_dep_version(repo.read("Cargo.toml"), dep_key, old, new)
        if text is None:
            out.line(
                f"    {out.paint('red', f'FAILED to update {dep_key} in workspace Cargo.toml to v{new}')}"
            )
            return False
        repo.write("Cargo.toml", text)
        out.line(f"  {ok} Cargo.toml {dep_key} → v{new}")
        return True

    for v in plan.verdicts:
        c = v.component
        if v.unbumped:
            kind = "cargo" if c.kind == "crate" else version_kind(c.version_file)
            new = plan.new_version(v)
            failed += not rewrite(c.version_file, kind, v.version, new)
            if c.kind == "crate" and c.dep_key:
                failed += not pin(c.dep_key, new)
    for v in plan.verdicts:
        if v.workspace_version is not None:
            failed += not pin(v.component.dep_key or "", v.version)
    for v in plan.verdicts:
        for key, old, new in v.requirements:
            path = v.component.version_file
            text = write_dep_version(repo.read(path), key, old, new)
            if text is None:
                out.line(f"    {out.paint('red', f'FAILED to update {key} in {path} to v{new}')}")
                failed += 1
                continue
            repo.write(path, text)
            out.line(f"  {ok} {path} {key} → v{new}")
    for mirror, carried in plan.mirrors:
        failed += not rewrite(mirror.file, mirror.kind, carried, plan.gateway_target)
    return failed


# --- the command --------------------------------------------------------------


def confirmed(ask) -> bool:
    """The prompt's answer; a closed stdin or an interrupt is a no."""
    try:
        return ask("Apply fixes? [y/N] ").strip().lower() == "y"
    except (EOFError, KeyboardInterrupt):
        return False


def run(
    root: Path,
    tag: str | None,
    mode: str,
    override: str | None,
    out: Out,
    ask=input,
    crates: Sequence[Component] = CRATES,
    packages: Sequence[Component] = PACKAGES,
    mirrors: Sequence[Mirror] = MIRRORS,
) -> int:
    """The check against ``tag`` in ``root``; mode is "check", "fix", "prompt" or "plan"."""
    repo = Repo(root)
    if override and not (CUT_VERSION if mode == "plan" else SEMVER).match(override):
        expected = "X.Y.Z or X.Y.Z-rc.N" if mode == "plan" else "X.Y.Z"
        raise Fatal(f"ERROR: VERSION_OVERRIDE must be in {expected} format, got: '{override}'")
    if tag is None:
        # a plan compares against what a cut compares against: the last final release
        tag = repo.highest_final_tag() if mode == "plan" else repo.newest_tag()
        if tag is None:
            raise Fatal("No tags found in repository. Pass a tag explicitly.")
    if not repo.has_ref(tag):
        raise Fatal(f"Tag '{tag}' does not exist.")
    if mode == "plan":
        # the verdicts are walked in silence: on main before a cut every changed crate reads as unbumped by design
        silent = Out(io.StringIO())
        gateway = next(c for c in crates if c.name == GATEWAY)
        derived = None
        if override is None and (root / gateway.version_file).is_file():
            # main reads the next development version between cuts: its first candidate is what the cut writes
            derived = first_candidate(
                read_version("cargo", repo.read(gateway.version_file), gateway.version_file)
            )
        plan = examine(
            repo, tag, None, silent, crates, packages, mirrors, cut_version=override or derived
        )
        plan.cut_version_derived = derived is not None
        show_plan(plan, tag, out)
        return 0

    out.line(
        out.paint("bold", "Checking workspace versions against tag: ") + out.paint("blue", tag)
    )
    if override:
        out.line(
            out.paint("bold", "Version override: ")
            + out.paint("cyan", override)
            + " (ignoring conventional commit detection)"
        )
    out.line()
    plan = examine(repo, tag, override, out, crates, packages, mirrors)
    if plan.issues == 0:
        out.line(out.paint("green", out.paint("bold", "All versions consistent.")))
        return 0
    out.line(out.paint("red", out.paint("bold", f"{plan.issues} issue(s) found.")))
    if mode == "check":
        return 1
    propose(plan, out)
    out.line()
    if mode == "prompt" and not confirmed(ask):
        out.line("No changes made.")
        return 1
    out.line()
    failed = apply(plan, repo, out)
    out.line()
    if failed:
        out.line(
            out.paint("red", out.paint("bold", f"{failed} fix(es) failed. Check output above."))
        )
        return 1
    out.line(out.paint("green", out.paint("bold", "Done. Re-run to verify.")))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("tag", nargs="?", help="the tag to compare against (default: the newest)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="report only, never prompt")
    group.add_argument(
        "--fix", action="store_true", help="apply the proposed fixes without prompting"
    )
    group.add_argument(
        "--plan",
        action="store_true",
        help="the bump the next cut writes against the tag, quietly; nothing is changed",
    )
    args = parser.parse_args(argv)
    mode = "check" if args.check else "fix" if args.fix else "plan" if args.plan else "prompt"
    out = Out(sys.stdout)
    try:
        return run(
            Path(__file__).resolve().parents[1],
            args.tag,
            mode,
            os.environ.get("VERSION_OVERRIDE") or None,
            out,
        )
    except Fatal as error:
        print(out.paint("red", str(error)), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
