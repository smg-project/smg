"""Observe RPC imports in the original worker process, then run its original module."""

import builtins
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

from artifact_io import require
from rpc_environment import verify


def observe(directory, expected, role, model, launch):
    seen = set()
    launcher_pid = os.getpid()
    engine = {
        "sglang.srt.disaggregation.encoder.grpc_server": (
            "b071f28e2c204505a4e5efa207c9886a32b157e6329899f1ffdc50bd6816d8c3"
        )
    }
    prefix = ("smg_grpc_proto", "smg_grpc_servicer")
    output = directory / "workers"
    output.mkdir(exist_ok=True)

    def capture():
        pid = os.getpid()
        path = output / f"{pid}.jsonl"
        if (pid, None) not in seen:
            fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
            marker = os.environ["E2E_JOB_MARKER"]
            require(
                ("E2E_JOB_MARKER=" + marker).encode()
                in Path("/proc/self/environ").read_bytes().split(b"\0"),
                "worker marker not visible in proc",
            )
            with path.open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "kind": "process",
                            "pid": pid,
                            "parent_pid": os.getppid(),
                            "start_ticks": fields[19],
                            "role": role,
                            "model_id": model,
                            "job_marker": marker,
                            "original_launcher": launch,
                            "launcher_pid": launcher_pid,
                        }
                    )
                    + "\n"
                )
            seen.add((pid, None))
        for name, module in list(sys.modules.items()):
            if (
                (not name.startswith(prefix) and name not in engine)
                or (pid, name) in seen
                or not getattr(module, "__file__", None)
            ):
                continue
            if getattr(getattr(module, "__spec__", None), "_initializing", False):
                continue
            source = Path(module.__file__).resolve()
            raw = source.read_bytes()
            expected_hash = engine.get(name) or expected.get(str(source), {}).get("sha256")
            require(
                hashlib.sha256(raw).hexdigest() == expected_hash,
                "actual worker imported unexpected source: " + name,
            )
            with path.open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "kind": "import",
                            "pid": pid,
                            "module": name,
                            "path": str(source),
                            "sha256": hashlib.sha256(raw).hexdigest(),
                        }
                    )
                    + "\n"
                )
            seen.add((pid, name))

    original = builtins.__import__

    def checked_import(name, *args, **kwargs):
        result = original(name, *args, **kwargs)
        package = (args[0] or {}).get("__package__", "") if args else ""
        if name.startswith(prefix) or package.startswith(prefix) or name in engine:
            capture()
        return result

    capture()
    builtins.__import__ = checked_import


def main():
    directory = Path(os.environ["E2E_EPD_RPC_PROOF_DIRECTORY"])
    role, model, *launch = sys.argv[1:]
    require(role in {"encode", "prefill", "decode"}, "unsupported EPD worker role")
    require(model == "Qwen/Qwen3.5-9B", "original EPD model changed")
    require(launch[:2] == ["-m", "sglang.launch_server"], "original SGLang module changed")
    report = verify(directory, Path("epd-rpc-wheels"), f"worker-{os.getpid()}")
    expected = {
        p: value for package in report["packages"].values() for p, value in package["files"].items()
    }
    observe(directory, expected, role, model, launch)
    sys.argv = [launch[1], *launch[2:]]
    runpy.run_module(launch[1], run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
