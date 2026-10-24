"""Installed/final source origins; this process is not a substitute for a worker."""
import hashlib
import importlib
import importlib.util
import importlib.metadata as metadata
import json
import os
import sys
from pathlib import Path


def main():
    root = Path(os.environ["GITHUB_WORKSPACE"]).resolve()
    directory = Path(os.environ["SMG_CI_RUNTIME_STATE_DIR"])
    stage = sys.argv[1]
    assert stage in {"installed", "final"}
    contract = json.loads((root / "e2e_test/amd_ts_python173/ts_python173/contract.json").read_text())
    rows = []
    for module, entry in contract["source_origins"].items():
        spec = importlib.util.find_spec(module)
        path = Path(spec.origin).resolve(strict=True)
        expected = (root / entry["path"]).resolve(strict=True)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert path == expected and digest == entry["sha256"], (module, path)
        rows.append({"module": module, "path": str(path), "sha256": digest})
    generated = []
    for name in ("common", "mlx_engine", "sglang_encoder", "sglang_scheduler",
                 "tokenspeed_encoder", "tokenspeed_scheduler", "trtllm_service", "vllm_engine"):
        for suffix in ("_pb2", "_pb2_grpc"):
            module = importlib.import_module("smg_grpc_proto.generated." + name + suffix)
            path = Path(module.__file__).resolve(strict=True)
            assert path.is_relative_to((root / "crates/grpc_client/python").resolve())
            generated.append({"module": module.__name__, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    common = importlib.import_module("smg_grpc_proto.generated.common_pb2")
    assert common.EngineLoad.DESCRIPTOR.fields_by_name["engine_silence_ms"].number == 19
    result = {"stage": stage, "python": sys.executable, "source_origins": rows,
              "generated": generated, "versions": {name: metadata.version(name) for name in
              ("grpcio", "grpcio-reflection", "grpcio-health-checking", "protobuf", "vllm", "torch")},
              "scope": "Actual auditor interpreter origins; same-case worker provenance separately requires argv/env/PID/start/constructor and quality oracles"}
    (directory / (stage + "-python173-origins.json")).write_text(json.dumps(result, indent=2) + "\n")
    if stage == "final":
        before = json.loads((directory / "installed-python173-origins.json").read_text())
        for name in ("source_origins", "generated", "versions"):
            assert result[name] == before[name], name + " changed"


if __name__ == "__main__":
    main()
