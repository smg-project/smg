"""Default-off, exact auxiliary inputs for the fixed-image MC4 job only."""
import base64
import csv
import hashlib
import importlib
import importlib.metadata as metadata
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import zipfile

SPEC_SHA = "f3519bade67f15b8bf0fe30f60b84313128175b14ca12d5c431168e3c1e50e7f"
SERVICER_SOURCE_SHA = "e79fde207845a729312197903c55b07d7912b5157db87d6d88ea02149a6eda86"
ROOT = Path(os.environ["GITHUB_WORKSPACE"]).resolve(strict=True)


def require(value, message):
    if not value:
        raise ValueError(message)


def identity(path):
    require(path.is_file() and path.stat().st_size <= 16 * 1024**2, "input file size/type")
    with path.open("rb") as stream:
        return {"bytes": path.stat().st_size, "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}


def enabled():
    return os.environ.get("SMG_CI_MC4_RUNTIME_INPUTS") == "1"


def spec():
    require(enabled() and os.environ.get("SMG_CI_PRIVATE_MOONCAKE") == "1", "private MC4 opt-ins required")
    path = ROOT / ".github/amd-shared-product/runtime-inputs.json"
    require(identity(path)["sha256"] == SPEC_SHA, "runtime inputs declaration changed")
    value = json.loads(path.read_text())
    require(value["policy"] == "fixed749f_mc4_auxiliary_normal_resolver_cairo_v1", "runtime input policy")
    return value, Path(value["bundle"])


def exact_tree(root, declared_files, declared_links, declared_dirs, limits):
    require(root.resolve(strict=True) == root and os.statvfs(root).f_flag & os.ST_RDONLY, "read-only exact bundle required")
    files, links, directories = set(), set(), set()
    count = 0
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                count += 1
                relative = path.relative_to(root).as_posix()
                require(count <= limits["entries"] and len(Path(relative).parts) <= limits["depth"], "bundle walk limit")
                if entry.is_symlink():
                    require(relative in declared_links and os.readlink(path) == declared_links[relative], "unknown/changed bundle link")
                    require(not Path(os.readlink(path)).is_absolute() and path.resolve(strict=True).is_relative_to(root), "bundle link escaped")
                    links.add(relative)
                elif entry.is_dir(follow_symlinks=False):
                    require(relative in declared_dirs, "unknown bundle directory")
                    directories.add(relative)
                    pending.append(path)
                else:
                    require(relative in declared_files and entry.is_file(follow_symlinks=False) and identity(path) == declared_files[relative], "unknown/changed bundle file")
                    files.add(relative)
    require(files == set(declared_files) and links == set(declared_links) and directories == set(declared_dirs), "bundle exact set differs")


def verify_bundle():
    value, root = spec()
    exact_tree(root, value["files"], value["links"], value["directories"], value["walk_limits"])
    require(sum(x["bytes"] for x in value["files"].values()) <= value["walk_limits"]["total_regular_bytes"], "bundle byte limit")
    for prefix, view in (("cairo-runtime", Path("/work/cairo-runtime")), ("private-five", Path("/input/private-five"))):
        f = {n.removeprefix(prefix + "/"): x for n, x in value["files"].items() if n.startswith(prefix + "/")}
        links = {n.removeprefix(prefix + "/"): x for n, x in value["links"].items() if n.startswith(prefix + "/")}
        dirs = [n.removeprefix(prefix + "/") for n in value["directories"] if n.startswith(prefix + "/")]
        exact_tree(view, f, links, dirs, value["walk_limits"])
    inventory = json.loads((root / "cairo-target-inventory.json").read_text())
    require(inventory["status"] == "COMPLETE_METADATA_ONLY" and not inventory["missing_or_ambiguous_SONAMEs"], "Cairo metadata closure incomplete")
    for item in inventory["libraries"]:
        if item["sha256"] != value["cairo"]["seed_sha256"]:
            require(identity(Path(item["path"])) == {k: item[k] for k in ("bytes", "sha256")}, "target Cairo closure changed")
    return value, root


def state():
    path = Path(os.environ["SMG_CI_RUNTIME_STATE_DIR"])
    run, attempt = os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_RUN_ATTEMPT"]
    require(run.isdigit() and attempt.isdigit() and path.parent == Path("/tmp") and path.name == "smg-mooncake4-runtime-" + run + "-" + attempt, "exact job state path")
    return path


def constraints():
    value, _ = verify_bundle()
    path = state().with_name(state().name + "-cohort-constraints.txt")
    with state().with_name(state().name + "-original-image-environment.json").open("x") as stream:
        json.dump({name: os.environ.get(name) for name in ("PIP_CONSTRAINT", "LD_LIBRARY_PATH")}, stream)
    raw = "".join(name + "==" + item["version"] + "\n" for name, item in sorted(value["packages"].items()))
    with path.open("x") as stream:
        stream.write(raw)
    print(path)  # The caller appends it before the existing prepare snapshot.


def installed_record(dist):
    record = dist.read_text("RECORD")
    require(record, "installed RECORD missing")
    rows = list(csv.reader(io.StringIO(record)))
    require(len(rows) == len({row[0] for row in rows}), "duplicate installed RECORD entry")
    base = Path(dist.locate_file("")).resolve()
    for relative, digest, size in rows:
        require(relative and not Path(relative).is_absolute(), "absolute installed RECORD path")
        path = Path(dist.locate_file(relative)).resolve(strict=True)
        require(path.is_relative_to(Path(sys.prefix).resolve()) or path.is_relative_to(base), "RECORD escaped active installation")
        if relative.endswith(".dist-info/RECORD"):
            require(digest == size == "", "installed RECORD self entry")
        elif digest:
            found = identity(path)
            encoded = base64.urlsafe_b64encode(bytes.fromhex(found["sha256"])).rstrip(b"=").decode()
            require(digest == "sha256=" + encoded and size == str(found["bytes"]), "installed RECORD bytes changed")
        else:
            require("__pycache__" in Path(relative).parts and relative.endswith(".pyc"), "unhashed installed RECORD member")
    return record


def distribution_proof(name, item, root):
    dist = metadata.distribution(name)
    require(dist.version == item["version"], "active auxiliary version: " + name)
    record = installed_record(dist)
    direct = None
    if name == "smg-grpc-proto":
        direct = json.loads(dist.read_text("direct_url.json") or "null")
        require(isinstance(direct, dict) and set(direct) == {"url", "archive_info"}, "proto direct URL schema")
        require(direct["url"] == (root / item["wheel"]).as_uri(), "proto exact wheel file URI")
        require(direct["archive_info"].get("hashes", {}).get("sha256") == identity(root / item["wheel"])["sha256"], "proto direct archive SHA")
        require(Path(dist._path).resolve().is_relative_to(Path(sys.prefix).resolve()), "proto active metadata is outside the job venv")
    with zipfile.ZipFile(root / item["wheel"]) as wheel:
        for member in wheel.infolist():
            require(not Path(member.filename).is_absolute() and ".." not in Path(member.filename).parts, "wheel member path")
            if member.is_dir():
                continue  # Genuine setproctitle/PyYAML wheels include directory entries.
            if member.filename.endswith(".dist-info/RECORD"):
                continue  # pip adds installer/direct_url/console entries to this file.
            raw = wheel.read(member)
            path = Path(dist.locate_file(member.filename))
            require(not path.is_symlink() and identity(path) == {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}, "installed source wheel differs: " + name)
            if name == "smg-grpc-proto" and member.filename.startswith("smg_grpc_proto/proto/") and member.filename.endswith(".proto"):
                require((ROOT / "crates/grpc_client/proto" / Path(member.filename).name).read_bytes() == raw, "current eight-schema source differs")
    return {"version": dist.version, "metadata_origin": str(dist._path), "RECORD_sha256": hashlib.sha256(record.encode()).hexdigest(), "wheel": item["wheel"], "direct_url": direct}


def editable_servicer():
    declaration = ROOT / ".github/amd-shared-product/editable-servicer-source.json"
    require(identity(declaration)["sha256"] == SERVICER_SOURCE_SHA, "editable servicer source declaration changed")
    value = json.loads(declaration.read_text())
    require(value["policy"] == "legacy_c2ac_same_root_editable_servicer_exact89_v1" and len(value["files"]) == 89, "legacy editable source scope")
    source = ROOT / "grpc_servicer"
    hashes = {}
    for name, item in value["files"].items():
        path = source / name
        if "symlink" in item:
            require(name == "LICENSE" and path.is_symlink() and os.readlink(path) == "../LICENSE" and path.resolve(strict=True) == ROOT / "LICENSE", "editable LICENSE target changed")
        else:
            require(not path.is_symlink() and path.resolve(strict=True).is_relative_to(source), "editable source escaped")
        found = identity(path)
        require(found == {k: item[k] for k in ("bytes", "sha256")}, "editable source bytes changed: " + name)
        hashes[name] = found
    dist = metadata.distribution(value["distribution"])
    require(dist.version == value["version"] and Path(dist._path).resolve().is_relative_to(Path(sys.prefix).resolve()), "editable active metadata/version changed")
    direct = json.loads(dist.read_text("direct_url.json") or "null")
    require(isinstance(direct, dict) and set(direct) == {"url", "dir_info"} and direct["dir_info"] == {"editable": True} and direct["url"] == source.as_uri(), "same-root editable direct URL changed")
    record = installed_record(dist)
    module = importlib.import_module("smg_grpc_servicer")  # Frozen __init__ is only a docstring.
    require(Path(module.__file__).resolve(strict=True) == source / "smg_grpc_servicer/__init__.py", "editable active package origin changed")
    require([str(Path(p).resolve(strict=True)) for p in module.__path__] == [str(source / "smg_grpc_servicer")], "editable package search path changed")
    return {"manifest_sha256": SERVICER_SOURCE_SHA, "source89": hashes, "metadata_origin": str(dist._path), "version": dist.version, "direct_url": direct, "RECORD_sha256": hashlib.sha256(record.encode()).hexdigest(), "active_module": module.__name__, "active_origin": module.__file__}


def package_proofs(value, root):
    return {name: distribution_proof(name, item, root) for name, item in value["packages"].items()}


def generated_origins():
    dist = metadata.distribution("smg-grpc-proto")
    result = {}
    for path in (ROOT / "crates/grpc_client/proto").glob("*.proto"):
        for suffix in ("_pb2", "_pb2_grpc"):
            name = "smg_grpc_proto.generated." + path.stem + suffix
            module = importlib.import_module(name)
            expected = Path(dist.locate_file(name.replace(".", "/") + ".py")).resolve(strict=True)
            require(Path(module.__file__).resolve(strict=True) == expected, "generated module origin differs")
            result[name] = {"path": str(expected), **identity(expected)}
    require(len(result) == 16, "eight-schema generated module set")
    return result


def cairo_proof(value):
    import cairo

    extension = Path(cairo._cairo.__file__).resolve()
    require(identity(extension)["sha256"] == value["cairo"]["extension_sha256"], "Cairo extension bytes")
    pygobject = metadata.distribution("PyGObject")
    require(pygobject.version == value["cairo"]["pygobject_version"] and str(pygobject._path).startswith("/usr/lib/python3/dist-packages/"), "system PyGObject changed")
    stream = io.BytesIO()
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 2, 2)
    surface.write_to_png(stream)
    raw = stream.getvalue()
    require(raw[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">II", raw[16:24]) == (2, 2), "real Cairo PNG dimensions")
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and Path(fields[5]).name.startswith("libcairo.so"):
            paths.add(Path(fields[5]).resolve(strict=True))
    require(len(paths) == 1, "loaded Cairo library is missing/ambiguous")
    loaded = paths.pop()
    require(loaded == Path("/work/cairo-runtime") / Path(value["cairo"]["seed"]).name and identity(loaded)["sha256"] == value["cairo"]["seed_sha256"], "loaded Cairo seed differs")
    (state() / "cairo-2x2.png").write_bytes(raw)
    return {"extension": str(extension), "extension_sha256": identity(extension)["sha256"], "loaded_library": str(loaded), "loaded_library_sha256": identity(loaded)["sha256"], "PNG_sha256": hashlib.sha256(raw).hexdigest(), "PNG_size": [2, 2], "PyGObject_origin": str(pygobject._path)}


def bootstrap():
    value, root = verify_bundle()
    original = json.loads(state().with_name(state().name + "-original-image-environment.json").read_text())
    original_constraint = original["PIP_CONSTRAINT"]
    expected_constraint = (original_constraint + " " if original_constraint else "") + str(state().with_name(state().name + "-cohort-constraints.txt"))
    require(json.loads((state() / "plan.json").read_text())["original_environment"] == expected_constraint, "original constraints were not preserved before prepare")
    expected_ld = "/work/cairo-runtime:/input/private-five/usr/lib/x86_64-linux-gnu" + (":" + original["LD_LIBRARY_PATH"] if original["LD_LIBRARY_PATH"] else "")
    require(os.environ.get("LD_LIBRARY_PATH") == expected_ld, "original image loader suffix changed before private prepare")
    with (state() / "cohort-original-image-environment.json").open("x") as stream:
        json.dump(original, stream, indent=2)
    for name, version in value["image_versions"].items():
        require(metadata.version(name) == version, "fixed image metadata differs: " + name)
    require(metadata.version("setuptools") == "79.0.1", "stock build frontend changed")
    env = dict(os.environ, PIP_CONFIG_FILE="/dev/null", PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1", PYTHONDONTWRITEBYTECODE="1")
    pip = [sys.executable, "-m", "pip", "install", "--no-compile", "--no-index", "--no-cache-dir", "--find-links=" + str(root / "wheels")]
    commands = [("tools", [*pip, "--ignore-installed", *value["tool_requirements"]]),
                ("cairo", [*pip, str(root / value["packages"]["pycairo"]["wheel"])]),
                ("grpc", [*pip, "--no-build-isolation", "vllm==" + value["image_versions"]["vllm"],
                          "grpcio-health-checking==1.78.0", str(root / value["packages"]["smg-grpc-proto"]["wheel"]), "-e", str(ROOT / "grpc_servicer")])]
    for name, argv in commands:
        with (state() / ("cohort-" + name + ".stdout")).open("xb") as stdout, (state() / ("cohort-" + name + ".stderr")).open("xb") as stderr:
            try:
                subprocess.run(argv, check=True, timeout=120, env=env, stdout=stdout, stderr=stderr)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                stdout.flush()
                stderr.flush()
                for suffix in ("stdout", "stderr"):
                    with (state() / ("cohort-" + name + "." + suffix)).open("rb") as stream:
                        stream.seek(max(0, os.fstat(stream.fileno()).st_size - 65536))
                        raw = stream.read(65536)
                    print(name + ":" + suffix + " (last 64KiB):\n" + raw.decode(errors="replace"), file=sys.stderr)
                raise
    sys.dont_write_bytecode = True
    result = {"spec_sha256": SPEC_SHA, "helper_sha256": identity(Path(__file__))["sha256"], "packages": package_proofs(value, root), "editable_servicer": editable_servicer(), "generated_origins16": generated_origins(), "cairo": cairo_proof(value), "PIP_CONSTRAINT": os.environ.get("PIP_CONSTRAINT"), "bootstrap_LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH")}
    require(not any(x.split(".")[0] in {"torch", "vllm", "sglang", "mooncake"} for x in sys.modules), "Cairo proof imported an engine")
    with (state() / "cohort-whole-pip-before-private.log").open("xb") as output:
        subprocess.run([sys.executable, "-m", "pip", "check"], check=True, timeout=30, env=env, stdout=output, stderr=subprocess.STDOUT)
    with (state() / "cohort-baseline.json").open("x") as output:
        json.dump(result, output, indent=2)


def verify(stage):
    require(stage in {"installed", "final", "unit-before", "unit-after", "pytest-before", "pytest-after"}, "unknown cohort stage")
    value, root = verify_bundle()
    baseline = json.loads((state() / "cohort-baseline.json").read_text())
    current = {"spec_sha256": SPEC_SHA, "helper_sha256": identity(Path(__file__))["sha256"], "packages": package_proofs(value, root), "editable_servicer": editable_servicer(), "PIP_CONSTRAINT": os.environ.get("PIP_CONSTRAINT")}
    require(all(current[k] == baseline[k] for k in current), "auxiliary input provenance changed")
    require(identity(state() / "cairo-2x2.png")["sha256"] == baseline["cairo"]["PNG_sha256"], "Cairo proof bytes changed")
    current["helper_origin"] = str(Path(__file__).resolve())
    with (state() / (stage + "-runtime-inputs.json")).open("x") as output:
        json.dump(current, output, indent=2)


if __name__ == "__main__":
    mode, *rest = sys.argv[1:]
    if mode == "constraints":
        constraints()
    elif mode == "bootstrap":
        bootstrap()
    elif mode == "verify":
        verify(*rest)
    else:
        raise ValueError("unknown MC4 runtime inputs mode")
