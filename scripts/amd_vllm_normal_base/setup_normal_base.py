"""Normal BASE setup inside the admitted original AMD GPU job; no model/test replacement."""
import argparse
import ast
import base64
from collections import Counter
import csv
from email.parser import BytesParser
import hashlib
import importlib.metadata as metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time
import traceback
import zipfile

WORK_SECONDS = 780


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def checked(path, row):
    assert path.is_file() and not path.is_symlink(), str(path)
    assert path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], str(path)


def wheel_data(path):
    with zipfile.ZipFile(path) as z:
        files = {i.filename for i in z.infolist() if not i.is_dir()}
        assert len(files) == len(z.infolist()) and len(files) == len(z.namelist())
        for info in z.infolist():
            p = PurePosixPath(info.filename)
            assert not p.is_absolute() and ".." not in p.parts and "\\" not in info.filename
            assert not stat.S_ISLNK(info.external_attr >> 16)
        assert z.testzip() is None
        records = [n for n in files if n.endswith(".dist-info/RECORD")]
        assert len(records) == 1
        record = records[0]
        rows = list(csv.reader(io.StringIO(z.read(record).decode())))
        assert len({r[0] for r in rows}) == len(rows) and {r[0] for r in rows} == files
        actual = {}
        for name, encoded, size in rows:
            h, count = hashlib.sha256(), 0
            with z.open(name) as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(block); count += len(block)
            value = base64.urlsafe_b64encode(h.digest()).rstrip(b"=").decode()
            if name == record:
                assert encoded == size == ""
            else:
                assert encoded == "sha256=" + value and int(size) == count, name
            actual[name] = {"bytes": count, "sha256": h.hexdigest(),
                            "mode": stat.S_IMODE(z.getinfo(name).external_attr >> 16)}
        m = BytesParser().parsebytes(z.read(record.removesuffix("RECORD") + "METADATA"))
        return {"name": m["Name"], "version": m["Version"], "members": actual,
                "RECORD": record, "Requires-Dist": m.get_all("Requires-Dist", [])}


def installed_wheel(proof, output):
    name = proof["name"]
    canonical = lambda x: re.sub(r"[-_.]+", "-", x).lower()
    matches = [d for d in metadata.distributions() if canonical(d.metadata.get("Name", "")) == canonical(name)]
    assert len(matches) == 1, "duplicate/old visible distribution: " + name
    dist = matches[0]
    assert dist.version == proof["version"] and Path(dist._path).resolve().is_relative_to(Path("/usr"))
    rows = {}
    for rel, expected in proof["members"].items():
        if rel == proof["RECORD"]:
            continue  # Normal installer regenerates RECORD/direct_url/INSTALLER entries.
        p = Path(dist.locate_file(rel)).resolve()
        assert p.is_relative_to(Path("/usr")) and p.is_file(), rel
        assert p.stat().st_size == expected["bytes"] and sha(p) == expected["sha256"], rel
        mode = stat.S_IMODE(p.stat().st_mode)
        wanted_mode = 0o755 if expected["mode"] & 0o111 else 0o644
        assert mode == wanted_mode, (rel, mode, wanted_mode)
        rows[rel] = {"path": str(p), "bytes": p.stat().st_size, "sha256": expected["sha256"], "mode": mode}
    assert Counter(metadata.requires(name) or []) == Counter(proof["Requires-Dist"])
    write(output, {"passed": True, "name": name, "version": dist.version,
          "scope": "installed original wheel files exact; normal RECORD extras separate",
          "metadata_path": str(dist._path), "members_except_installer_RECORD": rows})
    return dist


def generated_stamps(source):
    root = source / "crates/grpc_client/python/smg_grpc_proto/generated"
    names = ("common", "mlx_engine", "sglang_encoder", "sglang_scheduler",
             "tokenspeed_encoder", "tokenspeed_scheduler", "trtllm_service", "vllm_engine")
    rows = {}
    for name in names:
        for suffix in ("_pb2", "_pb2_grpc"):
            p = root / (name + suffix + ".py")
            assert p.is_file() and not p.is_symlink()
            tree = ast.parse(p.read_bytes(), filename=str(p))
            row = {"path": str(p), "sha256": sha(p)}
            if suffix == "_pb2_grpc":
                values = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "GRPC_GENERATED_VERSION" for t in n.targets)]
                assert values == ["1.81.1"], (str(p), values)
                row["GRPC_GENERATED_VERSION"] = values[0]
            else:
                calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Attribute) and n.func.attr == "ValidateProtobufRuntimeVersion"]
                assert len(calls) == 1
                version = tuple(ast.literal_eval(x) for x in calls[0].args[1:4])
                assert version == (6, 33, 5) and version <= (6, 33, 6)
                row["protobuf_generated_version"] = version
            rows[name + suffix] = row
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=Path("/opt/smg-vllm-normal-input"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--admission", type=Path, required=True)
    args = ap.parse_args()
    root = args.source.resolve(strict=True)
    code = Path(__file__).resolve().parent
    admission = json.loads(args.admission.read_text())
    assert admission["admitted_lane"] == "vllm-chat-full-original"
    assert all(admission[k] is True for k in ("ready", "run_budget_admitted", "artifacts_pinned", "normal_validation_authorized"))
    assert admission["runtime_cohort_accepted"] is False and admission["consumer_gpu_ready"] is False  # outcomes are not first-validation preconditions
    binding = admission["normal_base"]
    assert binding["input_ready"] is False and admission["normal_validation_authorized"] is True and binding["root_acceptance_sha256"] == "b4bae63c180d6e6444b5a7fb4465286d827123a51be7282eee9d5e1f3e94a76b"
    assert sha(code / "normal-pins.json") == binding["pins_sha256"]
    assert sha(Path(__file__)) == binding["adapter_sha256"]
    pins = json.loads((code / "normal-pins.json").read_text())
    assert args.input == Path(pins["input_directory"]) and args.input.is_dir() and not args.input.is_symlink()
    assert all(pins["artifact_acceptances"].get(n, {}).get("sha256") for n in ("vllm", "pycairo", "smg"))
    assert os.geteuid() == 0 and sys.prefix == sys.base_prefix == "/usr"
    assert Path("/dev/kfd").is_char_device() and any(p.is_char_device() for p in Path("/dev/dri").glob("renderD*"))
    assert not list(Path("/dev").glob("nvidia*")), "AMD job must not expose NVIDIA devices"
    actual_head = subprocess.check_output(["git", "-c", "safe.directory=" + str(root), "rev-parse", "HEAD"], cwd=root, text=True).strip()
    assert actual_head == os.environ["GITHUB_SHA"] and admission["target_product_commit"]
    def input_path(name):
        if name in (pins["vllm_wheel"], pins["pycairo_wheel"]):
            return args.input / name
        if name == pins["smg_wheel"]:
            return root / "wheel" / name
        if name == "_generated.py":
            return root / "clients/python/smg_client/types/_generated.py"
        return code / name
    args.output.mkdir(exist_ok=True)
    start = time.monotonic(); deadline = start + WORK_SECONDS
    report = {"passed": False, "scope": "normal BASE provenance in an AMD job; original CPU fixture RPC, no model/test outcome",
              "commands": [], "GPU_devices_required": True, "model": False, "actual_checkout": actual_head, "original16_GPU_accepted": False,
              "runtime_accepted": False, "coverage_increment": 0}
    source = root
    os.umask(0o022)
    env = os.environ.copy()
    for name in ("VIRTUAL_ENV", "PYTHONPATH", "UV_PROJECT_ENVIRONMENT", "UV_OFFLINE",
                 "PIP_NO_INDEX", "PIP_FIND_LINKS", "UV_NO_BUILD_ISOLATION"):
        env.pop(name, None)
    constraints = args.output / "runtime-constraints.txt"
    constraints.write_text(pins["runtime_constraints_text"])
    env.update({"PATH": "/usr/bin:/bin:/usr/local/bin", "PIP_CONSTRAINT": str(constraints),
                "UV_INDEX_URL": "https://pypi.org/simple", "UV_NO_MANAGED_PYTHON": "1",
                "UV_CACHE_DIR": "/tmp/final-base-uv-cache", "PIP_CACHE_DIR": "/tmp/final-base-pip-cache",
                "PYTHONDONTWRITEBYTECODE": "1", "DEBIAN_FRONTEND": "noninteractive",
                "PIP_DISABLE_PIP_VERSION_CHECK": "1"})
    def save():
        write(args.output / "result.json", report)
    def command(name, argv, allowed=(0,), cap=180):
        report["stage"] = name; save()
        left = min(cap, deadline - time.monotonic()); assert left > 0, "finite normal BASE work deadline"
        row = {"name": name, "argv": argv, "exit_code": None}
        p = None
        try:
            with (args.output / (name + ".stdout")).open("wb") as so, (args.output / (name + ".stderr")).open("wb") as se:
                p = subprocess.Popen(argv, cwd=source, env=env, stdin=subprocess.DEVNULL,
                                     stdout=so, stderr=se, start_new_session=True)
                row["exit_code"] = p.wait(timeout=left)
        except BaseException as error:
            row["first_error"] = {"type": type(error).__name__, "message": str(error)}
            if p is not None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError as cleanup:
                    row["group_cleanup_error"] = str(cleanup)
                try:
                    row["exit_code"] = p.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    row["reap_unproven"] = True
            raise
        finally:
            row["elapsed_total_seconds"] = time.monotonic() - start
            report["commands"].append(row); save()
        assert row["exit_code"] in allowed, "normal command failed: " + name
    def protected():
        rows = {}
        for name in pins["protected_base_distributions"]:
            d = metadata.distribution(name)
            rows[name] = {"version": d.version, "METADATA_sha256": sha(Path(d._path) / "METADATA"),
                          "RECORD_sha256": sha(Path(d._path) / "RECORD")}
        return rows
    try:
        mount_flags = os.statvfs(args.input).f_flag
        report["normal_input_mount"] = {"path": str(args.input), "f_flag": mount_flags,
            "ST_RDONLY": os.ST_RDONLY, "readonly": bool(mount_flags & os.ST_RDONLY)}
        save()
        assert mount_flags & os.ST_RDONLY, "actual normal input mount must be read-only"
        for name, row in pins["flat_inputs"].items():
            checked(input_path(name), row)
        before = protected(); write(args.output / "protected-base-before.json", before)
        assert not (source / ".venv").exists(), "no shadow venv"
        gpu_identity = r"""import json,os,pathlib,torch
assert torch.version.hip and torch.cuda.is_available()
count=torch.cuda.device_count(); assert count >= 1
rows=[]
for index in range(count):
 p=torch.cuda.get_device_properties(index)
 arch=getattr(p,'gcnArchName',None)
 assert arch is not None and arch.split(':')[0]=='gfx942', (index,arch)
 rows.append({'index':index,'name':p.name,'arch':arch,'total_memory':p.total_memory,
              'uuid':str(getattr(p,'uuid',None)),'pci_bus_id':str(getattr(p,'pci_bus_id',None))})
print(json.dumps({'scope':'actual allocated AMD device properties, not model correctness',
 'HIP':torch.version.hip,'visible_devices':rows,
 'masks':{k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES')},
 'kfd_rdev':pathlib.Path('/dev/kfd').stat().st_rdev,
 'render_rdevs':{str(p):p.stat().st_rdev for p in pathlib.Path('/dev/dri').glob('renderD*')}}))
"""
        command("actual-AMD-device-identity", [sys.executable, "-I", "-B", "-c", gpu_identity], cap=45)
        command("pip-inspect-before", [sys.executable, "-m", "pip", "inspect", "--local"], cap=30)
        command("pip-check-before", [sys.executable, "-m", "pip", "check"], allowed=(0, 1), cap=30)
        apt = ["-o", "Acquire::AllowInsecureRepositories=false", "-o", "Acquire::AllowDowngradeToInsecureRepositories=false",
               "-o", "APT::Get::AllowUnauthenticated=false"]
        source_rows = []
        paths = [Path("/etc/apt/sources.list"), *sorted(Path("/etc/apt/sources.list.d").glob("*.list")),
                 *sorted(Path("/etc/apt/sources.list.d").glob("*.sources"))]
        for path in paths:
            if path.exists():
                assert not re.search(r"(?im)(trusted\s*(?:=|:)\s*yes|allow-insecure\s*=\s*yes|allow-weak\s*=\s*yes)", path.read_text())
                source_rows.append({"path": str(path), "sha256": sha(path)})
        write(args.output / "signed-apt-source-public-paths.json", source_rows)
        command("signed-apt-update", ["apt-get", *apt, "update"])
        command("normal-Cairo-runtime-install", ["apt-get", *apt, "install", "-y", "--no-install-recommends",
                "libcairo2=" + pins["Cairo_runtime_package_version"]])
        command("actual-Cairo-runtime-dpkg", ["dpkg-query", "-W", "-f=${Package}\t${Version}\t${Architecture}\n", "libcairo2"], cap=20)
        vp = wheel_data(input_path(pins["vllm_wheel"]))
        cp = wheel_data(input_path(pins["pycairo_wheel"]))
        assert vp["name"].lower() == "vllm" and vp["version"] == pins["runtime_versions"]["vllm"] and len(vp["members"]) == 3709
        assert cp["name"].lower() == "pycairo" and cp["version"] == "1.29.2" and len(cp["members"]) == 8
        command("normal-base-vllm-pycairo-gRPC-resolution", ["/usr/local/bin/uv", "pip", "install", "--system",
                "--python", sys.executable, "--constraint", str(constraints),
                str(input_path(pins["vllm_wheel"])), str(input_path(pins["pycairo_wheel"])),
                "grpcio==1.81.1", "grpcio-reflection==1.81.1", "grpcio-health-checking==1.81.1",
                "grpcio-tools==1.81.1", "protobuf==6.33.6"], cap=300)
        installed_wheel(vp, args.output / "installed-CDF-vllm-full-files.json")
        installed_wheel(cp, args.output / "installed-e006-pycairo-full-files.json")
        command("normal-source-proto-servicer", [sys.executable, "-m", "pip", "install", "--report",
                str(args.output / "proto-servicer-install.json"), "-e", "crates/grpc_client/python/", "-e", "grpc_servicer[vllm]"])
        command("smg-before", [sys.executable, "-m", "pip", "show", "smg"], allowed=(0, 1), cap=20)
        command("smg-uninstall", [sys.executable, "-m", "pip", "uninstall", "-y", "smg"], cap=30)
        command("normal-same-current9134-smg-install", [sys.executable, "-m", "pip", "install", "--report",
                str(args.output / "smg-install.json"), str(input_path(pins["smg_wheel"]))])
        command("normal-e2e-deps", ["bash", "scripts/ci_install_e2e_deps.sh"], cap=300)
        command("normal-pytest-asyncio", [sys.executable, "-m", "pip", "install", "pytest-asyncio"], cap=60)
        assert not (source / ".venv").exists()
        assert all(metadata.version(n) == v for n, v in pins["runtime_versions"].items())
        write(args.output / "fresh-generated-stamps.json", generated_stamps(source))
        command("whole-BASE-pip-check-final", [sys.executable, "-m", "pip", "check"], cap=30)
        command("whole-BASE-pip-inspect-final", [sys.executable, "-m", "pip", "inspect", "--local"], cap=30)
        # Original same-source postflight/wire/native algorithms are separate
        # fresh processes. Only reviewed prefix/distribution/data literals differ.
        command("normal-source-RECORD-proto-native-postflight", [sys.executable, str(code / "postflight.py"),
                "--source", str(source), "--wheel", str(input_path(pins["smg_wheel"])),
                "--inputs", str(code / "postflight-inputs.json"), "--output", str(args.output / "postflight")], cap=60)
        # Runtime-only admission view; the source template stays defaultoff. No result is prefilled.
        native_pins = json.loads((code / "identity-pins.json").read_text())
        assert native_pins["PREVIEW_ONLY"] is True
        native_pins["PREVIEW_ONLY"] = False
        native_pins["actual_admitted_checkout"] = actual_head
        write(args.output / "admitted-native-pins.json", native_pins)
        command("same-current9134-native-baked-identity", [sys.executable, str(code / "native_identity_adapter.py"),
                "--pins", str(args.output / "admitted-native-pins.json"), "--pins-sha256", sha(args.output / "admitted-native-pins.json"),
                "--output", str(args.output / "native-identity")], cap=30)
        command("original-CPU-fixture-RPC-wire", [sys.executable, str(code / "wire_probe.py")], cap=25)
        smoke = r"""import hashlib, importlib.metadata as md, json, pathlib, sys
import cairo
assert sys.prefix == sys.base_prefix == '/usr' and md.version('pycairo') == '1.29.2'
p=pathlib.Path(sys.modules['cairo._cairo'].__file__).resolve(); assert p.is_relative_to(pathlib.Path('/usr'))
assert hashlib.sha256(p.read_bytes()).hexdigest() == '16ab5a5e6edfe4f277719e1113bb8e5e74571035088247d7cac51f085468b9e0'
s=cairo.ImageSurface(cairo.FORMAT_ARGB32,2,2); c=cairo.Context(s); c.set_source_rgba(1,0,0,1); c.paint(); s.flush()
assert s.get_width()==s.get_height()==2 and s.get_stride()>=8
out=pathlib.Path(sys.argv[1]); s.write_to_png(str(out/'pycairo-real-smoke.png')); raw=(out/'pycairo-real-smoke.png').read_bytes()
assert raw.startswith(b'\x89PNG\r\n\x1a\n')
print(json.dumps({'scope':'genuine Cairo/native/PNG only','cairo_version':cairo.cairo_version_string(),
'native_path':str(p),'native_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'PNG_sha256':hashlib.sha256(raw).hexdigest(),
'native_maps':[x for x in pathlib.Path('/proc/self/maps').read_text().splitlines() if str(p) in x]}))
"""
        command("real-installed-pycairo-PNG-native", [sys.executable, "-I", "-B", "-c", smoke, str(args.output)], cap=15)
        wrapper = r"""import hashlib,importlib,json,pathlib,sys
import torch,vllm,smg_grpc_proto,smg_grpc_servicer
assert sys.prefix==sys.base_prefix=='/usr' and torch.version.hip
root=pathlib.Path(sys.argv[1]).resolve()
assert pathlib.Path(smg_grpc_proto.__file__).resolve()==root/'crates/grpc_client/python/smg_grpc_proto/__init__.py'
assert pathlib.Path(smg_grpc_servicer.__file__).resolve()==root/'grpc_servicer/smg_grpc_servicer/__init__.py'
assert pathlib.Path(vllm.__file__).resolve().is_relative_to(pathlib.Path('/usr'))
assert vllm.__version__=='0.32.1rc1.dev505+g00b6c4c56.rocm100.smggrpc1811'
print(json.dumps({'scope':'normal base imports/origins only, no model execution','python':sys.executable,'prefix':sys.prefix,
'vllm_version':vllm.__version__,'vllm_origin':vllm.__file__,'HIP':torch.version.hip,
'proto_origin':smg_grpc_proto.__file__,'servicer_origin':smg_grpc_servicer.__file__}))
"""
        command("fresh-real-vllm-base-wrapper-imports", [sys.executable, "-I", "-B", "-c", wrapper, str(source)], cap=45)
        # Recheck original bytes after ALL normal installation/source work.
        installed_wheel(vp, args.output / "installed-CDF-vllm-full-files-final.json")
        installed_wheel(cp, args.output / "installed-e006-pycairo-full-files-final.json")
        after = protected(); write(args.output / "protected-base-after.json", after)
        assert before == after, "protected Torch/custom5/ROCm dependency metadata changed"
        with Path(os.environ["GITHUB_ENV"]).open("a") as f:
            f.write("PIP_CONSTRAINT=" + str(constraints.resolve()) + "\n")
        report["masks_preserved"] = {k: env.get(k) for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")}
        report["normal_BASE_proof_passed"] = True
        report["passed"] = True
    except BaseException as error:
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        report["elapsed_seconds"] = time.monotonic() - start
        report["passed"] = report["passed"] and report["elapsed_seconds"] <= WORK_SECONDS
        save()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
