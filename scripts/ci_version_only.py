#!/usr/bin/env python3
"""Whether a change moves nothing but the release version carriers.

A release bump (`release.sh cut`, `bump` and `bump-main`, `check_release_versions.py --fix`) rewrites
version strings and nothing else: the manifests' `version = "..."` lines and the root manifest's pins
of the workspace crates, Cargo.lock's entries for those crates, the Python packages' version fields,
the chart's version and appVersion, the engine-image workflow's release tag. Such a change leaves
every tested code path as the commit it was made on had it. The PR Test workflow lets it inherit that
commit's GPU e2e lanes (the card tier) when that commit's own run passed, and `release.sh check`
reads that run for the head's card tier.

This script is the decision. Every file changed between the two commits must be a version carrier,
and every changed line of it a version line whose only difference is the version string; Cargo.lock
is compared entry by entry, and only the entries of path packages (the workspace's own crates) may
move. A file added or removed, a line beyond the version, a dependency moved in a manifest or in the
lock: none of that is version-only. The carrier tables are scripts/check_release_versions.py's,
loaded from beside this file.

Usage:
    ci_version_only.py [-C <repository>] <base> [<head>]      (head defaults to HEAD)

Exit code 0 = version-only (the versions that moved are printed), 1 = not (the first thing that is
more is printed), 2 = the commits or their files could not be read.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = "Cargo.lock"
ROOT_MANIFEST = "Cargo.toml"
# A version as the carriers spell it: 1.14.0, 1.14.0-rc.2, 1.15.0-dev, 1.14.0rc1.
VERSION = re.compile(r"[0-9][0-9A-Za-z.+-]*")
# The version line of each carrier kind, split as (prefix)(version)(suffix): a change may touch
# the middle group alone. "toml" is a manifest's or a pyproject's own version line.
VERSION_LINES = {
    "toml": re.compile(rf'^(\s*version\s*=\s*")({VERSION.pattern})(".*)$'),
    "python": re.compile(rf'^(\s*__version__\s*=\s*")({VERSION.pattern})(".*)$'),
    "helm": re.compile(rf'^((?:version|appVersion):\s*"?)({VERSION.pattern})("?\s*)$'),
    "workflow": re.compile(rf"^(\s*SMG_RELEASE_TAG:\s*v)({VERSION.pattern})(\s*)$"),
}
# A dependency pinned by version in an inline table, `tool-parser = { version = "1.10.1", path =
# ... }`: a version line when the key is a workspace crate's (the root manifest's pins, a crate's
# dependency on a sibling); a third-party dependency's pin is a dependency moved.
PIN_LINE = re.compile(
    rf'^(\s*([A-Za-z0-9_.-]+)\s*=\s*\{{[^}}]*?\bversion\s*=\s*")({VERSION.pattern})(".*)$'
)
STATUS = {"A": "added", "D": "removed", "T": "its type changed", "U": "unmerged"}


def _checker():
    """The version checker's tables and git reader, loaded from scripts/check_release_versions.py."""
    path = Path(__file__).resolve().with_name("check_release_versions.py")
    spec = importlib.util.spec_from_file_location("check_release_versions", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_release_versions"] = module
    spec.loader.exec_module(module)
    return module


class Unreadable(Exception):
    """A file of the diff that one of the commits does not carry; its text is the message."""


@dataclass(frozen=True)
class Verdict:
    version_only: bool
    reason: str  # version-only: the versions that moved; otherwise the first thing that is more


def carriers(crv) -> dict[str, str]:
    """Path -> kind of every file a bump may rewrite: the checker's crates, packages and mirrors,
    the root manifest (its pins of the workspace crates) and the lock file."""
    kinds = {component.version_file: "toml" for component in (*crv.CRATES, *crv.PACKAGES)}
    for mirror in crv.MIRRORS:
        kinds[mirror.file] = "toml" if mirror.kind in ("cargo", "pyproject") else mirror.kind
    kinds[ROOT_MANIFEST] = "workspace"
    kinds[LOCK] = "lock"
    return kinds


def workspace_keys(crv) -> frozenset[str]:
    """The names a workspace crate is pinned under: its [workspace.dependencies] key and its own."""
    return frozenset(key for crate in crv.CRATES for key in (crate.dep_key, crate.name) if key)


def version_span(kind: str, line: str, keys: frozenset[str]) -> tuple[str, str, str] | None:
    """The line split around its version string when it is a version line of the kind; None
    otherwise. The root manifest has no version of its own: its pins alone are version lines."""
    if kind != "workspace":
        match = VERSION_LINES[kind].match(line)
        if match:
            return match.group(1), match.group(2), match.group(3)
    if kind in ("toml", "workspace"):
        match = PIN_LINE.match(line)
        if match and match.group(2) in keys:
            return match.group(1), match.group(3), match.group(4)
    return None


def text_reason(
    path: str, kind: str, base: str, head: str, keys: frozenset[str], moved: set[tuple[str, str]]
) -> str | None:
    """Why a text carrier's change is more than its version lines; None when it is not (the
    versions that moved are added to ``moved``)."""
    before, after = base.splitlines(), head.splitlines()
    if len(before) != len(after):
        return f"{path}: a line was added or removed"
    for number, (old, new) in enumerate(zip(before, after), 1):
        if old == new:
            continue
        span = version_span(kind, old, keys)
        if span is None:
            return f"{path}:{number}: not a version line"
        prefix, was, suffix = span
        now = new[len(prefix) : len(new) - len(suffix)]
        if not (new.startswith(prefix) and new.endswith(suffix) and VERSION.fullmatch(now)):
            return f"{path}:{number}: more than the version changed"
        moved.add((was, now))
    return None


def lock_reason(base: str, head: str, moved: set[tuple[str, str]]) -> str | None:
    """Why the lock file's change is more than the workspace crates' versions; None when it is
    not. A path package (no ``source``) is one of the workspace's own crates: its version may
    move; every other entry, and every other table of the file, must stay as it was."""
    try:
        before, after = tomllib.loads(base), tomllib.loads(head)
    except tomllib.TOMLDecodeError as error:
        return f"{LOCK}: not readable as TOML: {error}"
    old, new = before.pop("package", []), after.pop("package", [])
    if before != after:
        return f"{LOCK}: a table other than the packages changed"
    if len(old) != len(new):
        return f"{LOCK}: the number of entries changed ({len(old)} -> {len(new)})"
    for was, now in zip(old, new):
        if was == now:
            continue
        name = was.get("name")
        if name != now.get("name"):
            return f"{LOCK}: an entry was added or removed ({name}, {now.get('name')})"
        if "source" in was or "source" in now:
            return (
                f"{LOCK}: {name} {was.get('version')} -> {now.get('version')}: a dependency moved"
            )
        if {k: v for k, v in was.items() if k != "version"} != {
            k: v for k, v in now.items() if k != "version"
        }:
            return f"{LOCK}: {name}: more than its version changed"
        moved.add((str(was.get("version")), str(now.get("version"))))
    return None


def changed(repo, base: str, head: str) -> list[tuple[str, str]]:
    """(status, path) of every file that differs between the commits; a rename reads as a removal
    and an addition."""
    lines = repo.git("diff", "--name-status", "--no-renames", base, head).splitlines()
    return [(line[:1], line.split("\t", 1)[1]) for line in lines if "\t" in line]


def classify(repo, base: str, head: str, crv=None) -> Verdict:
    """Whether ``head`` moves nothing but version strings on ``base``."""
    crv = crv or _checker()
    kinds, keys = carriers(crv), workspace_keys(crv)
    changes = changed(repo, base, head)
    if not changes:
        return Verdict(False, f"nothing changed between {base} and {head}")
    moved: set[tuple[str, str]] = set()
    for status, path in changes:
        kind = kinds.get(path)
        if kind is None:
            return Verdict(False, f"{path}: not a version carrier")
        if status != "M":
            return Verdict(False, f"{path}: {STATUS.get(status, status)}")
        before, after = repo.show(base, path), repo.show(head, path)
        if before is None or after is None:
            raise Unreadable(f"{path}: not readable at both {base} and {head}")
        if kind == "lock":
            reason = lock_reason(before, after, moved)
        else:
            reason = text_reason(path, kind, before, after, keys, moved)
        if reason:
            return Verdict(False, reason)
    if not moved:
        return Verdict(False, "no version string moved")
    moves = ", ".join(f"{was} -> {now}" for was, now in sorted(moved))
    return Verdict(True, f"{len(changes)} file(s) move nothing but versions: {moves}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Whether a change moves nothing but the release version carriers."
    )
    parser.add_argument(
        "-C",
        dest="repository",
        type=Path,
        default=ROOT,
        help="the repository to read (default: the one this script is in)",
    )
    parser.add_argument("base", help="the commit the change was made on")
    parser.add_argument(
        "head", nargs="?", default="HEAD", help="the commit carrying the change (default: HEAD)"
    )
    args = parser.parse_args(argv)
    crv = _checker()
    try:
        verdict = classify(crv.Repo(args.repository.resolve()), args.base, args.head, crv)
    except subprocess.CalledProcessError as error:
        print(f"ci_version_only: {error.stderr.strip() or error}", file=sys.stderr)
        return 2
    except Unreadable as error:
        print(f"ci_version_only: {error}", file=sys.stderr)
        return 2
    print(("version-only: " if verdict.version_only else "not version-only: ") + verdict.reason)
    return 0 if verdict.version_only else 1


if __name__ == "__main__":
    sys.exit(main())
