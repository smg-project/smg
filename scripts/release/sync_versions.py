#!/usr/bin/env python3
"""Move the checkout's version files to a release's.

After a release ships from its branch, main's version files follow it: the gateway and the
files that mirror its version read the release's version, and every other crate or package
reads the release's version when that is higher than main's (crates.io and PyPI never take a
version twice, so the next cut must start above what the release published). A component
main already carries at a higher version keeps it.

Usage:
    sync_versions.py <ref> <gateway-version>

    ref               the release's tag or branch (its version files are read with git show)
    gateway-version   the version the gateway and its mirrors take (X.Y.Z)

Exit code 0 = the files follow the release (a report of what moved is printed), 1 = a rewrite
did not take.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# bumped by make bump-version; the checker does not list it among its mirrors
VERSION_PY = "bindings/python/src/smg/version.py"
VERSION_PARTS = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:-rc\.(\d+))?$")


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
    """Orders X.Y.Z and X.Y.Z-rc.N: a final sorts above its candidates."""
    match = VERSION_PARTS.match(version)
    if not match:
        raise SystemExit(f"sync_versions: not a version: {version!r} (X.Y.Z or X.Y.Z-rc.N)")
    major, minor, patch, rc = match.groups()
    return (int(major), int(minor), int(patch), 1 if rc is None else 0, int(rc or 0))


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
    set_version(gateway.version_file, "cargo", gateway_version)
    for mirror in crv.MIRRORS:
        set_version(mirror.file, mirror.kind, gateway_version)
    set_version(VERSION_PY, "python", gateway_version)

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

    if moved:
        print(f"sync_versions: {len(moved)} version line(s) follow {ref}:")
        for line in moved:
            print(f"  {line}")
    else:
        print(f"sync_versions: every version file already follows {ref}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
