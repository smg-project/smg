"""Read-only current-cohort install proof; writes only a fresh evidence directory."""
import argparse, base64, csv, hashlib, importlib, importlib.metadata as md
import importlib.util, io, json, os, stat, sys, traceback, zipfile
from pathlib import Path, PurePosixPath

INPUT_SHA = "c811bad4eb25a0c4cbbc8067b2e21a428b8e115e622d588e3af5dec7459ce315"
ap = argparse.ArgumentParser(description=__doc__)
for name in ("source", "wheel", "inputs", "output"):
    ap.add_argument("--" + name, type=Path, required=True)
args = ap.parse_args()
root, wheel, out = args.source.resolve(), args.wheel.resolve(), args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
report = {"passed": False, "source_basis_head": None, "scope": "NORMAL_INSTALL_PROVENANCE_WITH_AMD_DEVICES_NOT_ENGINE_ACCEPTANCE",
          "context": "initial", "packages": {}, "origins": {}, "descriptors": {}}

def save():
    with (out / "proof.json.tmp").open("w") as f:
        json.dump(report, f, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
    (out / "proof.json.tmp").replace(out / "proof.json")

def need(value, message):
    if not value:
        raise RuntimeError(message)

def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def rawsha(raw):
    return hashlib.sha256(raw).hexdigest()

def origin(name, expected=None):
    module = importlib.import_module(name)
    p = Path(module.__file__).resolve()
    report["origins"][name] = {"path": str(p), "sha256": digest(p)}
    need(expected is None or p == expected.resolve(), "module origin: " + name)
    return module

save()
try:
    need(digest(args.inputs) == INPUT_SHA, "input identity")
    spec = json.loads(args.inputs.read_text()); report["source_basis_head"] = spec["source_head"]
    names = ("smg", "smg-grpc-proto", "smg-grpc-servicer", "smg-client", "smg-e2e-tests",
             "grpcio", "protobuf", "grpcio-health-checking", "grpcio-reflection",
             "apache-tvm-ffi", "vllm", "pycairo")
    dists, missing = {}, []
    for name in names:
        report["context"] = "raw metadata: " + name
        try:
            d = dists[name] = md.distribution(name)
        except md.PackageNotFoundError:
            missing.append(name); continue
        folder = out / name; folder.mkdir()
        for filename in ("RECORD", "METADATA", "direct_url.json"):
            path = Path(d._path) / filename
            if path.exists():
                (folder / filename).write_bytes(path.read_bytes())
        report["packages"][name] = {"version": d.version, "metadata_directory": str(folder)}
    save(); need(not missing, "missing distributions: " + repr(missing))
    need(wheel.stat().st_size == spec["wheel_bytes"] and digest(wheel) == spec["wheel_sha256"], "wheel identity")
    prefix = Path(sys.prefix).resolve(); need(prefix == Path("/usr"), "image interpreter")
    need(Path("/dev/kfd").is_char_device() and any(p.is_char_device() for p in Path("/dev/dri").glob("renderD*")) and not list(Path("/dev").glob("nvidia*")), "real AMD devices required")
    for name, d in dists.items():
        rows = list(csv.reader((out / name / "RECORD").read_text().splitlines()))
        need(all(len(r) == 3 for r in rows), "invalid RECORD row: " + name)
        inventory = {r[0]: r for r in rows}; need(len(inventory) == len(rows), "duplicate RECORD: " + name)
        records = [r[0] for r in rows if r[0].endswith(".dist-info/RECORD")]
        need(len(records) == 1 and Path(d.locate_file(records[0])).resolve() == (Path(d._path) / "RECORD").resolve(), "own RECORD identity")
        checked = set()
        for rel, encoded, size in rows:
            report["context"] = {"package": name, "RECORD_path": rel}
            p = Path(d.locate_file(rel)).resolve()
            need(p.is_relative_to(prefix) and p.is_file(), "installed path outside venv or missing")
            if rel == records[0]:
                need(not encoded and not size, "RECORD self fields"); continue
            if encoded:
                value = base64.urlsafe_b64encode(bytes.fromhex(digest(p))).decode().rstrip("=")
                need(encoded == "sha256=" + value, "RECORD hash")
            if size:
                need(size == str(p.stat().st_size), "RECORD size")
            if encoded and size:
                checked.add(p)
            else:
                need(p.suffix == ".pyc", "blank non-cache RECORD fields")
        for rel, encoded, size in rows:
            if (not encoded or not size) and rel.endswith(".pyc"):
                report["context"] = {"package": name, "cache_path": rel}
                p = Path(d.locate_file(rel)).resolve()
                source = Path(importlib.util.source_from_cache(str(p))).resolve()
                need(source in checked, "cache has no same-RECORD verified source")
        report["packages"][name]["RECORD_rows"] = len(rows)
    save(); report["context"] = "original SMG wheel members"
    smg = dists["smg"]
    with zipfile.ZipFile(wheel) as z:
        members = z.namelist(); need(len(members) == len(set(members)) == 17, "wheel member set")
        rec = next(n for n in members if n.endswith(".dist-info/RECORD"))
        raw = z.read(rec); (out / "original-wheel-RECORD").write_bytes(raw)
        rows = list(csv.reader(io.StringIO(raw.decode())))
        need(len(rows) == 17 and {r[0] for r in rows} == set(members), "wheel RECORD membership")
        for rel, encoded, size in rows:
            report["context"] = {"original_wheel_member": rel}
            need(not PurePosixPath(rel).is_absolute() and ".." not in PurePosixPath(rel).parts, "wheel path")
            need(stat.S_IFMT(z.getinfo(rel).external_attr >> 16) in (0, stat.S_IFREG), "wheel nonregular member")
            if rel == rec:
                need(not encoded and not size, "original RECORD self fields"); continue
            data = z.read(rel); target = Path(smg.locate_file(rel)).resolve()
            value = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
            need(encoded == "sha256=" + value and size == str(len(data)) and digest(target) == rawsha(data), "installed wheel bytes")
    direct = json.loads((out / "smg" / "direct_url.json").read_text())
    need(direct["url"] == wheel.as_uri() and direct.get("archive_info", {}).get("hashes", {}).get("sha256") == spec["wheel_sha256"], "SMG direct URL")
    for name, rel in (("smg-grpc-proto", "crates/grpc_client/python"), ("smg-grpc-servicer", "grpc_servicer")):
        direct = json.loads((out / name / "direct_url.json").read_text())
        need(direct.get("dir_info", {}).get("editable") is True and direct["url"] == (root / rel).as_uri(), "RPC editable source URL")
        package = name.replace("-", "_"); origin(package, root / rel / package / "__init__.py")
    for rel, expected in spec["source_files"].items():
        report["context"] = {"source_file": rel}; need(digest(root / rel) == expected, "source bytes")
    generated = root / "crates/grpc_client/python/smg_grpc_proto/generated"
    for name, expected in spec["protos"].items():
        report["context"] = "RPC descriptor: " + name
        need(digest(generated.parent / "proto" / (name + ".proto")) == expected, "proto source")
        pb = origin("smg_grpc_proto.generated." + name + "_pb2", generated / (name + "_pb2.py"))
        origin("smg_grpc_proto.generated." + name + "_pb2_grpc", generated / (name + "_pb2_grpc.py"))
        need(pb.DESCRIPTOR.name == name + ".proto", "descriptor name")
        (out / (name + ".descriptor.pb")).write_bytes(pb.DESCRIPTOR.serialized_pb)
        report["descriptors"][name] = {"serialized_sha256": rawsha(pb.DESCRIPTOR.serialized_pb), "bytes": len(pb.DESCRIPTOR.serialized_pb)}
    report["context"] = "active native and client"
    native = origin("smg.smg_rs", Path(smg.locate_file("smg/smg_rs.abi3.so")))
    native_path = Path(native.__file__).resolve()
    need(digest(native_path) == spec["native_sha256"] and native_path.stat().st_size == spec["native_bytes"], "native identity")
    need(all(hasattr(native, n) for n in ("Router", "BackendType", "PolicyType")), "native symbols")
    report["native_maps"] = [l for l in Path("/proc/self/maps").read_text().splitlines() if l.endswith(str(native_path))]
    need(report["native_maps"], "native not mapped")
    client = origin("smg_client.types._generated", Path(dists["smg-client"].locate_file("smg_client/types/_generated.py")))
    need(digest(client.__file__) == spec["client_types_sha256"] and Path(client.__file__).stat().st_size == spec["client_types_bytes"], "client types identity")
    direct = json.loads((out / "smg-client" / "direct_url.json").read_text())
    need(direct["url"] == (root / "clients/python").as_uri() and not direct.get("dir_info", {}).get("editable"), "normal client source URL")
    report["passed"] = True; report["context"] = "complete"; save()
except BaseException as exc:
    report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    save(); raise
