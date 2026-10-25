"""Opt-in image metadata and private pip-constraint continuity for one CI job."""

from contextlib import contextmanager
import hashlib
import importlib.metadata as metadata
import json
import os
import re
import shutil
import sys
from pathlib import Path

PINNED = {"torch", "vllm", "grpcio", "protobuf", "grpcio-reflection"}
OPTIONAL = "grpcio-health-checking"
OBSERVED = PINNED | {OPTIONAL, "pygobject"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def entries(environment):
    result = []
    for value in (environment or "").split():
        path = Path(value).resolve(strict=True)
        raw = path.read_bytes()
        result.append({"path": str(path), "sha256": sha(raw), "text": raw.decode()})
    return result


def snapshot():
    records = []
    for dist in metadata.distributions():
        name = re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower()
        if name not in OBSERVED:
            continue
        root = Path(dist._path)
        origin = root if root.is_file() else root / "METADATA"
        if not origin.is_file():
            origin = root / "PKG-INFO"
        raw = origin.read_bytes()
        records.append(
            {
                "name": name,
                "version": dist.version,
                "origin": str(origin.resolve()),
                "metadata_sha256": sha(raw),
                "raw_metadata": raw.decode(),
            }
        )
    require({r["name"] for r in records} >= PINNED | {"pygobject"}, "image metadata missing")
    active = {}
    for name in sorted(OBSERVED):
        if not any(row["name"] == name for row in records):
            active[name] = None
            continue
        dist = metadata.distribution(name)
        active[name] = {"version": dist.version, "origin": str(Path(dist._path).resolve())}
    return {
        "python": sys.executable,
        "prefix": sys.prefix,
        "active": active,
        "records": sorted(records, key=lambda r: (r["name"], r["version"], r["origin"])),
    }


@contextmanager
def evidence_owner(value=None):
    """Opt-in numeric owner for private CI evidence creation; default is unchanged."""
    if value is None:
        value = os.environ.get("SMG_CI_RUNTIME_EVIDENCE_OWNER")
    if value is None:
        yield
        return
    require(re.fullmatch(r"[1-9][0-9]*:[1-9][0-9]*", value) is not None, "invalid evidence owner")
    uid, gid = map(int, value.split(":"))
    require(uid < 2**31 and gid < 2**31 and os.geteuid() == 0, "root evidence writer required")
    previous = os.geteuid(), os.getegid(), os.getgroups()
    try:
        os.setgroups([])
        os.setegid(gid)
        os.seteuid(uid)
        yield
    finally:
        # Attempt every restoration even when entry or the original operation failed.
        try:
            os.seteuid(previous[0])
        finally:
            try:
                os.setegid(previous[1])
            finally:
                os.setgroups(previous[2])


def prepare(directory):
    directory.mkdir(mode=0o700)
    before = snapshot()
    original_environment = os.environ.get("PIP_CONSTRAINT")
    require("\n" not in (original_environment or ""), "multiline constraint environment")
    pins = []
    for name in sorted(PINNED | {OPTIONAL}):
        selected = before["active"][name]
        if selected is None:
            version = before["active"]["grpcio-reflection"]["version"]
        else:
            version = selected["version"]
            require(
                {r["version"] for r in before["records"] if r["name"] == name} == {version},
                "ambiguous image versionbag: " + name,
            )
        pins.append(name + "==" + version)
    path = directory / "runtime-pins.txt"
    path.write_text("\n".join(pins) + "\n")
    effective = (original_environment + " " if original_environment else "") + str(path)
    plan = {
        "original_environment": original_environment,
        "original": entries(original_environment),
        "effective_environment": effective,
        "effective": entries(effective),
        "before": before,
        "allowed_missing_companion": before["active"][OPTIONAL] is None,
    }
    (directory / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(effective)


def verify(directory, stage):
    plan = json.loads((directory / "plan.json").read_text())
    environment = os.environ.get("PIP_CONSTRAINT")
    current = {
        "environment": environment,
        "constraints": entries(environment),
        "metadata": snapshot(),
    }
    (directory / (stage + ".json")).write_text(json.dumps(current, indent=2) + "\n")
    require(environment == plan["effective_environment"], "constraint environment changed")
    require(current["constraints"] == plan["effective"], "constraint path, bytes or order changed")
    before, after = plan["before"], current["metadata"]
    stable = OBSERVED - ({OPTIONAL} if plan["allowed_missing_companion"] else set())
    for name in stable:
        require(
            [r for r in before["records"] if r["name"] == name]
            == [r for r in after["records"] if r["name"] == name],
            "metadata bag changed: " + name,
        )
        require(before["active"][name] == after["active"][name], "active metadata changed: " + name)
    if plan["allowed_missing_companion"]:
        rows = [r for r in after["records"] if r["name"] == OPTIONAL]
        version = before["active"]["grpcio-reflection"]["version"]
        require(len(rows) == 1 and rows[0]["version"] == version, "added companion differs")
        require(
            Path(rows[0]["origin"]).is_relative_to(Path(sys.prefix).resolve()),
            "added companion outside job venv",
        )
        added = {"records": rows, "active": after["active"][OPTIONAL]}
        anchor = directory / "added-companion.json"
        if anchor.exists():
            require(json.loads(anchor.read_text()) == added, "added companion metadata changed")
        else:
            anchor.write_text(json.dumps(added, indent=2) + "\n")


def main():
    mode, path, *rest = sys.argv[1:]
    directory = Path(path).resolve()
    if mode == "prepare":
        prepare(directory)
    elif mode == "verify":
        stage, evidence = rest
        require(stage in {"installed", "final"}, "unknown audit stage")
        try:
            verify(directory, stage)
        except Exception as error:
            (Path(evidence) / (stage + "-runtime-error.json")).write_text(
                json.dumps({"error": str(error)}) + "\n"
            )
            raise
        finally:
            if directory.is_dir():
                shutil.copytree(directory, Path(evidence) / "runtime-evidence", dirs_exist_ok=True)
    else:
        raise ValueError("unknown runtime constraint mode")


if __name__ == "__main__":
    with evidence_owner():
        main()
