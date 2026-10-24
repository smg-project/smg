"""CPU package identity and wire checks; imports no model or engine."""

import base64
import csv
import hashlib
import importlib
import importlib.metadata as metadata
import io
import json
import pkgutil
import subprocess
import zipfile
from pathlib import Path


def verify_wheel(path):
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate wheel member")
        if path.name.startswith("smg_grpc_servicer-"):
            prefix, suffix = "grpc_servicer/smg_grpc_servicer/", ".py"
            wheel_prefix = "smg_grpc_servicer/"
        else:
            prefix, suffix = "crates/grpc_client/proto/", ".proto"
            wheel_prefix = "smg_grpc_proto/proto/"
        sources = subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", "HEAD", prefix], text=True
        ).splitlines()
        expected = {
            wheel_prefix + name.removeprefix(prefix) for name in sources if name.endswith(suffix)
        }
        actual = {name for name in names if name.startswith(wheel_prefix) and name.endswith(suffix)}
        if not expected or expected != actual:
            raise ValueError("wheel source membership differs")
        records = [name for name in names if name.endswith(".dist-info/RECORD")]
        if len(records) != 1:
            raise ValueError("wheel RECORD missing or duplicated")
        rows = list(csv.reader(io.StringIO(wheel.read(records[0]).decode())))
        if len(rows) != len(names) or {row[0] for row in rows} != set(names):
            raise ValueError("wheel RECORD membership differs")
        for name, digest, size in rows:
            if name == records[0]:
                if digest or size:
                    raise ValueError("RECORD self entry must be unhashed")
                continue
            data = wheel.read(name)
            actual = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
            if digest != "sha256=" + actual or size != str(len(data)):
                raise ValueError("wheel RECORD differs: " + name)
            if name.startswith("smg_grpc_servicer/") and name.endswith(".py"):
                source = "grpc_servicer/" + name
            elif name.startswith("smg_grpc_proto/proto/") and name.endswith(".proto"):
                source = "crates/grpc_client/proto/" + Path(name).name
            elif name == "smg_grpc_proto/__init__.py":
                source = "crates/grpc_client/python/" + name
            else:
                continue
            expected = subprocess.check_output(["git", "show", "HEAD:" + source])
            if data != expected:
                raise ValueError("wheel source differs: " + source)
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    for name, version in {
        "smg-grpc-proto": "0.4.29",
        "smg-grpc-servicer": "0.14.2",
        "grpcio": "1.78.0",
        "protobuf": "6.31.1",
        "grpcio-health-checking": "1.78.0",
        "grpcio-reflection": "1.78.0",
    }.items():
        if metadata.version(name) != version:
            raise ValueError("installed package version differs: " + name)
    wheels = sorted(Path("epd-rpc-wheels").glob("*.whl"))
    expected = {
        "smg_grpc_proto-0.4.29-py3-none-any.whl",
        "smg_grpc_servicer-0.14.2-py3-none-any.whl",
    }
    if {wheel.name for wheel in wheels} != expected or len(wheels) != 2:
        raise ValueError("RPC wheel set differs")
    evidence = [verify_wheel(wheel) for wheel in wheels]
    import smg_grpc_proto.generated as generated
    from smg_grpc_proto import sglang_scheduler_pb2 as pb

    modules = sorted(item.name for item in pkgutil.iter_modules(generated.__path__))
    if len(modules) != 16 or not all(name.endswith(("_pb2", "_pb2_grpc")) for name in modules):
        raise ValueError("generated module set differs")
    for module in modules:
        importlib.import_module(generated.__name__ + "." + module)
    request = pb.GenerateRequest(
        request_id="cpu-producer-29",
        encode_handoff=pb.EncodeHandoff(
            request_id="cpu-producer-29",
            encoder_urls=["grpc://b:5", "grpc://a:6"],
            num_items_assigned=[2, 0],
        ),
    )
    if pb.GenerateRequest.FromString(request.SerializeToString()) != request:
        raise ValueError("handoff wire roundtrip differs")
    if pb.GenerateRequest.DESCRIPTOR.fields_by_name["encode_handoff"].number != 19:
        raise ValueError("handoff field number differs")
    print(
        json.dumps(
            {
                "RPC_wheels": evidence,
                "generated_modules_imported": modules,
                "field19_roundtrip": True,
                "engine_or_EPD_executed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
