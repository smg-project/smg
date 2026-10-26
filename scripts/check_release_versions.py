#!/usr/bin/env python3
"""Pre-release version check for the workspace crates and the Python packages.

For every component (a workspace crate or a Python package), since the given tag:
whether its files changed, whether its version was bumped, and, for a crate, whether
the workspace root's dependency entry agrees. Then the files that must mirror the
gateway's version. Issues are listed with the bump conventional commits ask for
(``feat!:`` or ``BREAKING CHANGE``: major; ``feat:``: minor; otherwise patch) and,
on request, applied.

Usage:
    check_release_versions.py [tag] [--check | --fix]

    tag       the release to compare against (default: the newest tag)
    --check   report only, never prompt (for CI and dry runs)
    --fix     apply the proposed fixes without prompting

Environment:
    VERSION_OVERRIDE  X.Y.Z: the gateway's new version instead of the detected bump

Exit code 0 = all versions consistent (or fixed), 1 = issues found and not fixed.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
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
    kind: str  # "cargo" | "pyproject" | "helm" | "workflow"


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
    Mirror("helm chart", "deploy/helm/smg/Chart.yaml", "helm"),
    Mirror("engine-docker", ".github/workflows/release-engine-docker.yml", "workflow"),
)


class Fatal(Exception):
    """A condition the check cannot proceed from; its text is the message."""


# --- version lines: read and rewrite ------------------------------------------

SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
TOML_VERSION = re.compile(r'^version\s*=\s*"([^"]*)"', re.M)
TOML_WORKSPACE_VERSION = re.compile(
    r"^version\.workspace\s*=\s*true|^version\s*=\s*\{\s*workspace\s*=\s*true", re.M
)
DUNDER_VERSION = re.compile(r'__version__\s*=\s*"([^"]*)"')
HELM_VERSION = re.compile(r'^version:\s*"?([^"\s]*)"?\s*$', re.M)
HELM_APP_VERSION = re.compile(r"^appVersion:.*$", re.M)
WORKFLOW_VERSION = re.compile(r"SMG_RELEASE_TAG:\s*v([0-9]\S*)")


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

    def show(self, ref: str, path: str) -> str | None:
        """A file's text at a ref; None when it is not there."""
        result = self._run("show", f"{ref}:{path}")
        return result.stdout if result.returncode == 0 else None

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
    files_changed: int  # 0 = nothing to check
    version: str = ""
    tag_version: str | None = None  # None = not at the tag: a new component
    level: str | None = None  # the bump asked for when the version was not bumped
    workspace_version: str | None = None  # the root's entry when it disagrees

    @property
    def unbumped(self) -> bool:
        return self.tag_version is not None and self.tag_version == self.version


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


def files_changed(repo: Repo, tag: str, component: Component) -> int:
    """The component's files changed since the tag; pure renames and a version-only edit do not count."""
    dirs = [f"{p}/" for p in with_legacy(component.path)]
    version_name = os.path.basename(component.version_file)
    count = sum(
        1
        for line in repo.changed_since(tag, dirs)
        if not line.startswith("R100") and not line.endswith(version_name)
    )
    if version_file_has_other_changes(repo, tag, component.version_file):
        count += 1
    return count


def check(repo: Repo, tag: str, component: Component, root_toml: str) -> Verdict:
    verdict = Verdict(component, files_changed(repo, tag, component))
    if verdict.files_changed == 0:
        return verdict
    kind = version_kind(component.version_file) if component.kind == "package" else "cargo"
    verdict.version = read_version(kind, repo.read(component.version_file), component.version_file)
    for path in with_legacy(component.version_file):
        text = repo.show(tag, path)
        if text is not None:
            verdict.tag_version = read_version(kind, text, f"{path} at {tag}")
            break
    if verdict.unbumped:
        verdict.level = bump_level(
            repo.commits_since(tag, [f"{p}/" for p in with_legacy(component.path)])
        )
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

    @property
    def issues(self) -> int:
        return (
            sum(1 for v in self.verdicts if v.unbumped)
            + sum(1 for v in self.verdicts if v.workspace_version is not None)
            + len(self.mirrors)
        )

    def new_version(self, verdict: Verdict) -> str:
        if self.override and verdict.component.name == GATEWAY:
            return self.override
        return bumped(verdict.version, verdict.level or "patch")


def report(verdict: Verdict, out: Out) -> None:
    c, name = verdict.component, out.paint("bold", verdict.component.name)
    where, n = f"({c.path}/)", f"{verdict.files_changed} file(s) changed"
    ok, warn, bad = out.paint("green", "✓"), out.paint("yellow", "!"), out.paint("red", "✗")
    if verdict.tag_version is None:
        out.line(f"  {ok} {name} {where} — new {c.kind} (v{verdict.version}), {n}")
    elif verdict.unbumped:
        out.line(
            f"  {warn} {name} {where} — {n} but version not bumped (v{verdict.version}) [{out.level(verdict.level)}]"
        )
    else:
        out.line(f"  {ok} {name} {where} — v{verdict.tag_version} → v{verdict.version} ({n})")
    if verdict.workspace_version is not None:
        out.line(
            f"    {bad} workspace Cargo.toml has {c.dep_key} v{verdict.workspace_version}, expected v{verdict.version}"
        )


def examine(
    repo: Repo,
    tag: str,
    override: str | None,
    out: Out,
    crates: Sequence[Component] = CRATES,
    packages: Sequence[Component] = PACKAGES,
    mirrors: Sequence[Mirror] = MIRRORS,
) -> Plan:
    """Walk the components and the mirrors, printing each verdict as the shell did."""
    root_toml = repo.read("Cargo.toml")
    verdicts = []
    for component in (*crates, *packages):
        verdict = check(repo, tag, component, root_toml)
        verdicts.append(verdict)
        if verdict.files_changed:
            report(verdict, out)

    gateway = next((v for v in verdicts if v.component.name == GATEWAY), None)
    gateway_component = next(c for c in crates if c.name == GATEWAY)
    target = read_version(
        "cargo", repo.read(gateway_component.version_file), gateway_component.version_file
    )
    plan = Plan(verdicts, [], target, override)
    if gateway is not None and gateway.unbumped:
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
        return sum(
            1 for v in verdicts if v.component.kind == kind and bool(v.files_changed) == changed
        )

    out.line()
    out.line(
        out.paint("bold", "Summary:")
        + f" {count('crate', True)} crate(s) with changes, {count('crate', False)} unchanged;"
        f" {count('package', True)} Python package(s) with changes, {count('package', False)} unchanged"
    )
    return plan


def propose(plan: Plan, out: Out) -> None:
    out.line()
    out.line(out.paint("bold", "Proposed fixes:"))
    sync = out.paint("blue", "sync")
    for v in plan.verdicts:
        c = v.component
        if v.unbumped and c.kind == "crate":
            out.line(
                f"  {out.level(v.level)} {c.name} v{v.version} → v{plan.new_version(v)} ({c.version_file})"
            )
            if c.dep_key:
                out.line(f"       sync workspace Cargo.toml {c.dep_key} → v{plan.new_version(v)}")
    for v in plan.verdicts:
        if v.workspace_version is not None:
            out.line(
                f"  {sync} workspace Cargo.toml {v.component.dep_key} v{v.workspace_version} → v{v.version}"
            )
    for v in plan.verdicts:
        c = v.component
        if v.unbumped and c.kind == "package":
            out.line(
                f"  {out.level(v.level)} {c.name} v{v.version} → v{plan.new_version(v)} ({c.version_file})"
            )
    for mirror, carried in plan.mirrors:
        out.line(f"  {sync} {mirror.label} v{carried} → v{plan.gateway_target} ({mirror.file})")


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
    """The check against ``tag`` in ``root``; mode is "check", "fix" or "prompt"."""
    repo = Repo(root)
    if override and not SEMVER.match(override):
        raise Fatal(f"ERROR: VERSION_OVERRIDE must be in X.Y.Z format, got: '{override}'")
    if tag is None:
        tag = repo.newest_tag()
        if tag is None:
            raise Fatal("No tags found in repository. Pass a tag explicitly.")
    if not repo.has_ref(tag):
        raise Fatal(f"Tag '{tag}' does not exist.")

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
    args = parser.parse_args(argv)
    mode = "check" if args.check else "fix" if args.fix else "prompt"
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
