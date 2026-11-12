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
did not take, or a floor could not be paired between the two files (named on stderr, with the
counts, to move by hand; a rerun passes once main carries the release's floors).
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION_PARTS = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:-rc\.(\d+)|-(dev))?$")
# a quoted dependency floor of a pyproject, wherever it sits on a line (a dependencies entry on a
# line of its own, an extra's inline array): the name, the floor, the rest of the requirement
FLOOR = re.compile(r'"([A-Za-z0-9_.-]+)>=(\d+\.\d+\.\d+)([^"]*)"')
# a line holding one quoted requirement and, at most, a comment: its comment states the floor's reason
ALONE = re.compile(r'^\s*"[^"]+",?\s*(#.*)?$')


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


def line_of(text: str, start: int, end: int) -> tuple[int, int]:
    """The bounds of the line holding ``text[start:end]``, without its newline."""
    first = text.rfind("\n", 0, start) + 1
    last = text.find("\n", end)
    return first, len(text) if last < 0 else last


def in_comment(text: str, start: int) -> bool:
    """Whether ``text[start]`` sits in a TOML comment: a ``#`` outside quotes earlier on its line
    (a comment that quotes a floor names no requirement; a ``#`` inside a basic or a literal
    string, a URL's fragment, opens no comment)."""
    quote = ""
    for char in text[text.rfind("\n", 0, start) + 1 : start]:
        if char in "\"'" and quote in ("", char):
            quote = "" if quote else char  # a basic or a literal string opens and closes
        elif char == "#" and not quote:
            return True
    return False


def floors(text: str, names: set[str]) -> dict[str, list[tuple[str, str, str | None]]]:
    """Per package name of ours, the floor requirements of ``text`` in order of appearance: each
    quoted requirement as written, the floor it names, and the whole line when the requirement
    is alone on it (None when it shares the line, as in an extra's inline array). A floor quoted
    inside a comment is not one."""
    found: dict[str, list[tuple[str, str, str | None]]] = {}
    for match in FLOOR.finditer(text):
        if match[1] in names and not in_comment(text, match.start()):
            first, last = line_of(text, match.start(), match.end())
            line = text[first:last]
            alone = line if ALONE.match(line) else None
            found.setdefault(match[1], []).append((match[0], match[2], alone))
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

    # the floors a package puts on another of ours: its k-th requirement on a name, wherever it
    # sits on a line, takes the release's k-th as the release wrote it, when the release's floor
    # is higher: the whole line, comment included, where the requirement is alone on its line in
    # both files (the comment states the floor's reason); the requirement alone where it sits in
    # an extra's inline array, the rest of that line staying main's
    names = {component.name for component in crv.PACKAGES}
    for component in crv.PACKAGES:
        theirs_text = repo.show(ref, component.version_file)
        if theirs_text is None or not (ROOT / component.version_file).is_file():
            continue
        theirs = floors(theirs_text, names)
        text = repo.read(component.version_file)
        mine = floors(text, names)
        # a name that occurs a different number of times in the two files cannot be paired by
        # position (an extra added on one side since the cut may sit before an older one): while
        # one of main's floors is below the release's highest, a move is pending under some
        # pairing, so the name is refused with its counts, for a move by hand, and the run
        # reports the failure; once main carries the release's floors there is nothing to pair
        # and the name passes, so the recovery the message asks for clears the refusal
        ambiguous = {
            name
            for name in mine.keys() & theirs.keys()
            if len(mine[name]) != len(theirs[name])
            and min(version_key(floor) for _, floor, _ in mine[name])
            < max(version_key(floor) for _, floor, _ in theirs[name])
        }
        for name in sorted(ambiguous):
            print(
                f"sync_versions: {component.version_file}: {name}>= occurs"
                f" {len(mine.get(name, []))} time(s) here and {len(theirs.get(name, []))} in {ref}:"
                " the pairing is ambiguous, not moved (move it by hand)",
                file=sys.stderr,
            )
            failed += 1
        seen: dict[str, int] = {}
        pieces: list[str] = []
        end = 0
        for match in FLOOR.finditer(text):
            name, floor = match[1], match[2]
            if name not in names or name in ambiguous or in_comment(text, match.start()):
                continue
            k = seen.get(name, 0)
            seen[name] = k + 1
            if match.start() < end or k >= len(theirs.get(name, [])):
                continue
            their_requirement, their_floor, their_line = theirs[name][k]
            if version_key(their_floor) <= version_key(floor):
                continue
            first, last = line_of(text, match.start(), match.end())
            if their_line is not None and ALONE.match(text[first:last]):
                # the release's requirement and comment, main's separator: main may have put an
                # entry after the floor the release's array ends with
                comma = "," if text[match.end() : match.end() + 1] == "," else ""
                head = re.match(r'(\s*"[^"]+"),?', their_line)
                assert head is not None  # ALONE matched the line
                pieces += [text[end:first], head[1] + comma + their_line[head.end() :]]
                end = last
            else:
                pieces += [text[end : match.start()], their_requirement]
                end = match.end()
            nth = f" (requirement {k + 1})" if k else ""
            moved.append(f"{component.version_file} {name}>=: {floor} -> {their_floor}{nth}")
        if end:
            repo.write(component.version_file, "".join(pieces) + text[end:])

    if moved:
        print(f"sync_versions: {len(moved)} version line(s) follow {ref}:")
        for line in moved:
            print(f"  {line}")
    else:
        print(f"sync_versions: every version file already follows {ref}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
