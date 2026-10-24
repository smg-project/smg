"""Fixed-image private wheel transaction; never replace the image's installation."""

import base64
import csv
import hashlib
import importlib.metadata as metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import zipfile

SPEC_SHA = "40bb0378fa335481ec5870fd898726f2e64642dde19d62673c8ee505673f2644"

# Exact cached749f OpenCV bootstrap; this directory must remain absent.
CV2_LOADER_ROOT = Path("/usr/local/lib/python3.12/dist-packages/cv2")
CV2_LOADER_PREFIX = "/usr/local/lib/python3.12/dist-packages/cv2/../../lib64"
CV2_LOADER_RESOLVED = "/usr/local/lib/python3.12/lib64"
CV2_LOADER_SOURCES = {
    "__init__.py": (6630, "69192904dead3ada37c8b305ee850ac7474924730881a9d5e0b2390944eb5aae"),
    "config.py": (111, "974e2d4096ee1a9a9a341df1bf33e16973683bf1ac733006de27ff2f23bc584d"),
    "config-3.py": (724, "9a7aadf724b822001f5e963b01fd4e375d45e2cbbe328d2ca7b42a440c083a1c"),
    "version.py": (91, "e718587212508acdcb296839c287c7fa346edb621005ffb199c26397592544ed"),
}


def require(value, message):
    if not value:
        raise ValueError(message)


def identity(path):
    with path.open("rb") as stream:
        return {"bytes": path.stat().st_size, "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}


def comparable_private_state(raw):
    """Compare unchanged loader order, allowing one proven-inert cv2 prepend.

    Keep every raw state/plan byte: this is comparison only, never an env reset.
    A /proc environment still describes launch state, not a child import trace.
    """
    current = dict(raw)
    value = current["LD_LIBRARY_PATH"]
    require(isinstance(value, str), "private loader path is not a string")
    if value.startswith(CV2_LOADER_PREFIX + ":"):
        for name, (size, digest) in CV2_LOADER_SOURCES.items():
            path = CV2_LOADER_ROOT / name
            require(path.is_file() and not path.is_symlink() and path.resolve() == path,
                    "cv2 loader source origin changed")
            require(identity(path) == {"bytes": size, "sha256": digest},
                    "cv2 loader source changed: " + name)
        require(os.path.realpath(CV2_LOADER_PREFIX) == CV2_LOADER_RESOLVED,
                "cv2 loader directory resolution changed")
        require(not os.path.lexists(CV2_LOADER_PREFIX) and not os.path.lexists(CV2_LOADER_RESOLVED),
                "cv2 loader directory is no longer inert")
        value = value[len(CV2_LOADER_PREFIX) + 1:]
    require(value.split(":")[:3] == ["/work/prefix/usr/lib/x86_64-linux-gnu", "/work/prefix/usr/lib", "/opt/rocm/lib"],
            "private loader path changed")
    current["LD_LIBRARY_PATH"] = value
    return current


def spec():
    raw = Path(__file__).with_name("private-wheel.json").read_bytes()
    require(hashlib.sha256(raw).hexdigest() == SPEC_SHA, "private wheel specification changed")
    return json.loads(raw)


def state():
    root = Path(os.environ["SMG_CI_RUNTIME_STATE_DIR"]).resolve(strict=True)
    require(os.environ.get("SMG_CI_PRIVATE_MOONCAKE") == "1", "explicit private wheel opt-in required")
    return root, root / "private-mooncake-site"


def readonly_mount(path):
    matches = [line.split() for line in Path("/proc/self/mountinfo").read_text().splitlines()]
    require(any(row[4] == str(path) and "ro" in row[5].split(",") for row in matches), "required read-only bind mount: " + str(path))


def prefix_leaves(prefix, declared, empty_directories=()):
    require(0 < len(declared) <= 55, "private prefix declaration cap")
    directories = {parent for name in declared for parent in Path(name).parents if parent != Path(".")}
    directories.update(Path(name) for name in empty_directories)
    entry_limit = 2 * (len(declared) + len(directories))
    depth_limit = max(len(Path(name).parts) for name in declared) + 2
    leaves, pending, seen_directories, entries_seen = set(), [prefix], set(), 0
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                entries_seen += 1
                require(entries_seen <= entry_limit, "private prefix walk entry cap")
                path = Path(entry.path)
                require(len(path.relative_to(prefix).parts) <= depth_limit, "private prefix walk depth cap")
                if entry.is_dir(follow_symlinks=False):
                    require(path.relative_to(prefix) in directories, "unexpected private prefix directory")
                    seen_directories.add(path.relative_to(prefix))
                    pending.append(path)
                else:
                    leaves.add(path.relative_to(prefix).as_posix())
                    require(len(leaves) <= len(declared), "private prefix leaf cap")
    require(seen_directories == directories, "private prefix directory set changed")
    return leaves


def verify_prefix(prefix, declared, empty_directories=()):
    require(prefix_leaves(prefix, declared, empty_directories) == set(declared), "private prefix leaf set changed")
    for name, item in declared.items():
        path = prefix / name
        require(path.resolve(strict=True).is_relative_to(prefix), "prefix symlink escaped")
        if item["kind"] == "symlink":
            require(path.is_symlink() and os.readlink(path) == item["target"], "prefix symlink changed")
        else:
            require(path.is_file() and not path.is_symlink() and identity(path) == {key: item[key] for key in ("bytes", "sha256")}, "prefix bytes changed: " + name)


def record_files(root, record):
    rows = list(csv.reader(io.StringIO(record)))
    require(len({row[0] for row in rows}) == len(rows), "duplicate RECORD entry")
    for name, encoded, size in rows:
        item = PurePosixPath(name)
        require(name and not item.is_absolute() and ".." not in item.parts and "\\" not in name, "unsafe RECORD path")
        path = root / name
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root), "missing/escaped RECORD file")
        if name.endswith(".dist-info/RECORD"):
            require(encoded == size == "", "RECORD self entry")
        else:
            value = identity(path)
            digest = base64.urlsafe_b64encode(bytes.fromhex(value["sha256"])).decode().rstrip("=")
            require(encoded == "sha256=" + digest and size == str(value["bytes"]), "RECORD mismatch: " + name)


def verify(stage):
    require(stage in {"prepare", "installed", "final", "unit-before", "unit-after", "pytest-before", "pytest-after", "canary"}, "unknown private audit stage")
    config = spec()
    root, target = state()
    wheel, prefix = Path(config["wheel_mount"]), Path(config["runtime_prefix"])
    readonly_mount(wheel)
    readonly_mount(prefix)
    require(identity(wheel) == {key: config["wheel"][key] for key in ("bytes", "sha256")}, "wheel differs")
    verify_prefix(prefix, config["private_prefix"], config["private_prefix_empty_directories"])
    for item in config["base_files"]:
        require(identity(Path(item["path"])) == {key: item[key] for key in ("bytes", "sha256")}, "image installation changed")
    curl = [block for block in Path("/var/lib/dpkg/status").read_text().split("\n\n") if "Package: libcurl4\n" in block + "\n"]
    require(len(curl) == 1 and "\nVersion: " + config["base_curl_version"] + "\n" in "\n" + curl[0] + "\n", "base curl version changed")
    if stage == "prepare":
        require(not target.exists(), "private target already exists")
        with zipfile.ZipFile(wheel) as archive:
            require(set(archive.namelist()) == set(config["wheel_members"]), "wheel member set changed")
            for name, item in config["wheel_members"].items():
                raw = archive.read(name)
                require({"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} == item, "wheel member differs")
        subprocess.run([sys.executable, "-m", "pip", "--isolated", "--disable-pip-version-check", "install", "--no-index", "--no-deps", "--no-cache-dir", "--no-compile", "--retries", "0", "--target", str(target), str(wheel)], check=True, timeout=60)
    require(target.is_dir() and str(target) == os.environ["PYTHONPATH"].split(":")[0], "private import path changed")
    comparable_private_state({"LD_LIBRARY_PATH": os.environ["LD_LIBRARY_PATH"]})
    dist_info = target / ("mooncake_transfer_engine_rocm-" + config["version"] + ".dist-info")
    dist = metadata.distribution("mooncake-transfer-engine-rocm")
    require(Path(dist._path).resolve() == dist_info and dist.version == config["version"], "active private metadata origin/version differs")
    record = dist.read_text("RECORD")
    record_files(target, record)
    for name, item in config["wheel_members"].items():
        if not name.endswith(".dist-info/RECORD"):
            require(identity(target / name) == item, "installed wheel file changed: " + name)
    direct = json.loads(dist.read_text("direct_url.json"))
    require(direct["url"] == wheel.as_uri() and direct["archive_info"].get("hashes", {}).get("sha256") == config["wheel"]["sha256"], "private wheel provenance differs")
    current = {"site": str(target), "version": dist.version, "RECORD": record, "direct_url": direct, "PYTHONPATH": os.environ["PYTHONPATH"], "LD_LIBRARY_PATH": os.environ["LD_LIBRARY_PATH"]}
    plan = root / "private-mooncake-plan.json"
    if stage == "prepare":
        with plan.open("x") as stream:
            stream.write(json.dumps(current, indent=2) + "\n")
    else:
        require(comparable_private_state(json.loads(plan.read_text())) == comparable_private_state(current), "private installed state changed")
    (root / (stage + "-private-mooncake.json")).write_text(json.dumps(current, indent=2) + "\n")
    return config, target


def require_mapped(children):
    config, target = spec(), state()[1]
    engine = target / "mooncake/engine.cpython-312-x86_64-linux-gnu.so"
    mapped, cores = [], []
    for child in children:
        libraries = child["loaded_native"]
        core = any(re.fullmatch(r"VLLM::EngineCore(?:_DP\d+)?", value) for value in child["argv"])
        if core:
            cores.append(child["pid"])
        mooncake = [name for name in libraries if "/mooncake/engine" in name]
        if not mooncake:
            require(not core, "owned EngineCore lacks private Mooncake mapping")
            continue
        require(mooncake == [str(engine)], "owned role loaded a different Mooncake engine")
        require(comparable_private_state(child["private_environment"]) == comparable_private_state({name: os.environ[name] for name in ("PYTHONPATH", "LD_LIBRARY_PATH")}), "transport process private environment differs")
        expected = {str(target / "mooncake" / name): item for name, item in config["raw_native"].items() if name != "nvlink_allocator.so"}
        expected.update({"/work/prefix/usr/lib/x86_64-linux-gnu/" + name: {key: item[key] for key in ("bytes", "sha256")} for name, item in config["runtime_libraries"].items()})
        require(all(libraries.get(name) == item for name, item in expected.items()), "owned Mooncake process lacks exact private runtime mappings")
        mapped.append(child["pid"])
    require(len(cores) == 1 and set(cores).issubset(mapped), "TP1 role lacks exactly one mapped owned EngineCore")
    return mapped


if __name__ == "__main__":
    try:
        verify(sys.argv[1])
    except Exception as error:
        root = Path(os.environ["SMG_CI_RUNTIME_STATE_DIR"])
        (root / (sys.argv[1] + "-private-mooncake-error.json")).write_text(json.dumps({"error": str(error)}) + "\n")
        raise
