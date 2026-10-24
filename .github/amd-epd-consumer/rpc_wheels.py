"""CPU package identity and wire checks; imports no model or engine."""

import base64
import csv
import hashlib
import io
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
            [
                "git",
                "-c",
                f"safe.directory={Path.cwd()}",
                "ls-tree",
                "-r",
                "--name-only",
                "HEAD",
                prefix,
            ],
            text=True,
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
            expected = subprocess.check_output(
                ["git", "-c", f"safe.directory={Path.cwd()}", "show", "HEAD:" + source]
            )
            if data != expected:
                raise ValueError("wheel source differs: " + source)
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
