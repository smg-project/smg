#!/usr/bin/env python3
"""Move the checkout's version files to a release's.

After a release ships from its branch, main's version files follow it: the gateway and the
files that mirror its version read the release's version, and every other crate or package
reads the release's version when that is higher than main's (crates.io and PyPI never take a
version twice, so the next cut must start above what the release published). A component
main already carries at a higher version keeps it, the gateway included: after a patch of an
older line (v1.14.1 once v1.15.0 is out) main never goes back, since the nightly and the next
cut derive their versions from it. The dependency floors a package puts on another of ours
(a ``"smg-grpc-proto>=X.Y.Z"`` line of a pyproject) follow the release's as well: a release
raises a floor when it ships fields the code reads, and a fresh install from main must not
accept an older package; a floor main already raised higher stays.

Usage:
    sync_versions.py <ref> <gateway-version>

    ref               the release's tag or branch (its version files are read with git show)
    gateway-version   the version the gateway and its mirrors take: X.Y.Z after a final, or main's
                      next development version X.Y.Z-dev when a release branch is cut (main reads
                      above everything the branch will release: 1.15.0-dev once release/1.14 exists)

Exit code 0 = the files follow the release (a report of what moved is printed), 1 = a rewrite
did not take.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION_PARTS = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:-rc\.(\d+)|-(dev))?$")
# a dependency floor line of a pyproject: the indent and quote, the name, the floor, the rest of the line
FLOOR = re.compile(r'^(\s*")([A-Za-z0-9_.-]+)>=(\d+\.\d+\.\d+)(".*)$')


def _checker():
    """The version checker's tables and file readers, loaded from scripts/check_release_versions.py."""
    path = ROOT / "scripts" / "check_release_versions.py"
    spec = importlib.util.spec_from_file_location("check_release_versions", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_release_versions"] = module
    spec.loader.exec_module(module)
    return module


def version_key(version: str) -> tuple[int, ...]:
    """Orders X.Y.Z, X.Y.Z-rc.N and X.Y.Z-dev: a final sorts above its candidates, a candidate above the
    development version of the same number (1.15.0-dev < 1.15.0-rc.1 < 1.15.0), as cargo and PEP 440 order them."""
    match = VERSION_PARTS.match(version)
    if not match:
        raise SystemExit(
            f"sync_versions: not a version: {version!r} (X.Y.Z, X.Y.Z-rc.N or X.Y.Z-dev)"
        )
    major, minor, patch, rc, dev = match.groups()
    stage = 0 if dev else 1 if rc else 2
    return (int(major), int(minor), int(patch), stage, int(rc or 0))


def floors(text: str, names: set[str]) -> dict[str, list[tuple[str, str]]]:
    """Per package name of ours, the floor lines of ``text`` in order: each line as written and
    the floor it names."""
    found: dict[str, list[tuple[str, str]]] = {}
    for line in text.split("\n"):
        match = FLOOR.match(line)
        if match and match[2] in names:
            found.setdefault(match[2], []).append((line, match[3]))
    return found


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    ref, gateway_version = args
    version_key(gateway_version)
    crv = _checker()
    repo = crv.Repo(ROOT)
    moved: list[str] = []
    failed = 0

    def set_version(path: str, kind: str, new: str) -> None:
        nonlocal failed
        if not (ROOT / path).is_file():
            return
        text = repo.read(path)
        old = crv.read_version(kind, text, path)
        if old == new:
            return
        written = crv.write_version(kind, text, old, new)
        if written is None:
            print(f"sync_versions: FAILED to set {path} to {new}", file=sys.stderr)
            failed += 1
            return
        repo.write(path, written)
        moved.append(f"{path}: {old} -> {new}")

    def pin(dep_key: str, new: str) -> None:
        nonlocal failed
        root_toml = repo.read("Cargo.toml")
        old = crv.workspace_dep_version(root_toml, dep_key)
        if old is None or old == new:
            return
        written = crv.write_workspace_dep_version(root_toml, dep_key, old, new)
        if written is None:
            print(f"sync_versions: FAILED to pin {dep_key} to {new} in Cargo.toml", file=sys.stderr)
            failed += 1
            return
        repo.write("Cargo.toml", written)
        moved.append(f"Cargo.toml {dep_key}: {old} -> {new}")

    gateway = next(c for c in crv.CRATES if c.name == crv.GATEWAY)
    ours = (
        crv.read_version("cargo", repo.read(gateway.version_file), gateway.version_file)
        if (ROOT / gateway.version_file).is_file()
        else gateway_version
    )
    if version_key(ours) > version_key(gateway_version):
        print(
            f"sync_versions: the gateway reads {ours}, above {gateway_version}:"
            " it and its mirrors keep that (main never goes back)"
        )
    else:
        set_version(gateway.version_file, "cargo", gateway_version)
        for mirror in crv.MIRRORS:  # the installed package's version.py among them
            set_version(mirror.file, mirror.kind, gateway_version)

    for component in (*crv.CRATES, *crv.PACKAGES):
        if component.name == crv.GATEWAY:
            continue
        theirs_text = repo.show(ref, component.version_file)
        if theirs_text is None or not (ROOT / component.version_file).is_file():
            continue
        kind = "cargo" if component.kind == "crate" else crv.version_kind(component.version_file)
        theirs = crv.read_version(kind, theirs_text, f"{component.version_file} at {ref}")
        ours = crv.read_version(kind, repo.read(component.version_file), component.version_file)
        if version_key(theirs) <= version_key(ours):
            continue
        set_version(component.version_file, kind, theirs)
        if component.dep_key:
            pin(component.dep_key, theirs)

    # the floors a package puts on another of ours: its k-th floor line for a name takes the
    # release's k-th, as the release wrote it, when the release's floor is higher
    names = {component.name for component in crv.PACKAGES}
    for component in crv.PACKAGES:
        theirs_text = repo.show(ref, component.version_file)
        if theirs_text is None or not (ROOT / component.version_file).is_file():
            continue
        theirs = floors(theirs_text, names)
        lines = repo.read(component.version_file).split("\n")
        seen: dict[str, int] = {}
        changed = False
        for i, line in enumerate(lines):
            match = FLOOR.match(line)
            if not match or match[2] not in names:
                continue
            k = seen.get(match[2], 0)
            seen[match[2]] = k + 1
            if k >= len(theirs.get(match[2], [])):
                continue
            their_line, their_floor = theirs[match[2]][k]
            if version_key(their_floor) <= version_key(match[3]):
                continue
            lines[i] = their_line
            moved.append(f"{component.version_file} {match[2]}>=: {match[3]} -> {their_floor}")
            changed = True
        if changed:
            repo.write(component.version_file, "\n".join(lines))

    if moved:
        print(f"sync_versions: {len(moved)} version line(s) follow {ref}:")
        for line in moved:
            print(f"  {line}")
    else:
        print(f"sync_versions: every version file already follows {ref}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
