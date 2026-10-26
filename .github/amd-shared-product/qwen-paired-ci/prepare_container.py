"""Exact CI setup inside the one GHA-owned4420 container; never start a model."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pwd
import grp
import subprocess
import time


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=Path, required=True)
    args = parser.parse_args()
    assert __debug__ and os.getuid() == 0
    context = json.loads(args.context.read_text())
    out, source = Path("/evidence"), Path("/source")
    packet = source / ".github/amd-shared-product/qwen-paired-ci/paired"
    inputs = json.loads((packet / "inputs.json").read_text())
    deadline = float(context["job_started_monotonic"]) + 180
    env = dict(os.environ, GITHUB_WORKSPACE=str(source), PYTHONDONTWRITEBYTECODE="1", SPT_NOENV="1",
               SMG_SGLANG_ROCM_IMAGE=inputs["image"],
               E2E_ENGINE="sglang", E2E_RUNTIME="sglang", E2E_GPU_TIER="1", PIP_DISABLE_PIP_VERSION_CHECK="1",
               PIP_NO_CACHE_DIR="1", GITHUB_ENV="/evidence/github-env")
    for key in ("SMG_QWEN_MROPE_TRACE", "SMG_QWEN_MM_TRACE", "SMG_QWEN_MM_MODEL", "SGLANG_GPT_OSS_MXFP4_BF16"):
        env.pop(key, None)
    result = {"source_parent": inputs["source_parent"], "source_worktree": str(source), "window_id": context["window_id"],
              "CID": context["CID"], "actual_success": False, "same_for_both_lanes": True,
              "original_CI_setup_used": True, "commands": [], "original_PIP_CONSTRAINT": os.environ.get("PIP_CONSTRAINT")}
    uid, gid = context["runner_UID"], context["runner_GID"]
    # A scheduled runner need not be root. Give that real UID a valid container
    # identity without changing the host account or the immutable base image.
    try:
        pwd.getpwuid(uid)
    except KeyError:
        with Path("/etc/passwd").open("a") as stream:
            stream.write("gha-paired:x:" + str(uid) + ":" + str(gid) + ":GHA paired:/evidence/home:/usr/sbin/nologin\n")
    try:
        grp.getgrgid(gid)
    except KeyError:
        with Path("/etc/group").open("a") as stream:
            stream.write("gha-paired:x:" + str(gid) + ":\n")
    constraint = out / "paired-runtime-constraints.txt"
    constraint.write_text("".join(name + "==" + version + "\n" for name, version in sorted(inputs["runtime_versions"].items())))
    env["PIP_CONSTRAINT"] = (env.get("PIP_CONSTRAINT", "") + " " + str(constraint)).strip()

    def run(name, argv, required=True):
        remaining = deadline - time.monotonic()
        assert remaining > 0, "original combined preparation180 exhausted"
        row = {"name": name, "argv": argv, "started_epoch": time.time()}
        result["commands"].append(row)
        with (out / (name + ".stdout")).open("xb") as stdout, (out / (name + ".stderr")).open("xb") as stderr:
            completed = subprocess.run(argv, cwd=source, env=env, stdout=stdout, stderr=stderr,
                                       timeout=min(90, remaining), check=False)
        row.update(exit_code=completed.returncode, finished_epoch=time.time())
        assert not required or completed.returncode == 0, name + " failed"
        return completed.returncode

    try:
        # These are the actual frozen AMD shared-CI setup/install/overlay steps.
        run("CI-SGLang-setup", ["bash", "scripts/ci_setup_sglang_rocm_e2e.sh"])
        run("CI-remove-previous-SMG", ["/opt/venv/bin/python", "-m", "pip", "uninstall", "-y", "smg"], required=False)
        wheels = list((source / "wheel").glob("*.whl"))
        assert len(wheels) == 1
        run("CI-install-SMG-b7", ["/opt/venv/bin/python", "-m", "pip", "install", str(wheels[0])])
        run("CI-full-e2e-client-dependencies", ["bash", "scripts/ci_install_e2e_deps.sh"])
        run("CI-installed-dependencies", ["/opt/venv/bin/python", "-m", "pip", "check"])
        run("CI-normal-pytest-asyncio", ["/opt/venv/bin/python", "-m", "pip", "install", "pytest-asyncio"])
        run("CI-fresh-current-RPC", ["/opt/venv/bin/python", "-c", 'import importlib\nimport json\nfrom pathlib import Path\nimport sys\n\nimport mooncake\nimport sglang\nimport smg_grpc_proto\nimport smg_grpc_servicer\nimport torch\n\nroot = Path(sys.argv[1]).resolve()\nassert Path(sys.prefix).resolve() == Path("/opt/venv"), "expected image interpreter"\nassert torch.version.hip, "expected ROCm Torch"\nexpected = {\n    smg_grpc_proto: root / "crates/grpc_client/python/smg_grpc_proto/__init__.py",\n    smg_grpc_servicer: root / "grpc_servicer/smg_grpc_servicer/__init__.py",\n    sglang: Path("/sgl-workspace/sglang/python/sglang/__init__.py"),\n}\nfor module, path in expected.items():\n    assert Path(module.__file__).resolve() == path, f"unexpected source: {module.__name__}"\n# Import all generated companions against the normally resolved runtime.\nmodules = {}\nfor name in ("common", "mlx_engine", "sglang_encoder", "sglang_scheduler",\n             "tokenspeed_encoder", "tokenspeed_scheduler", "trtllm_service", "vllm_engine"):\n    for suffix in ("_pb2", "_pb2_grpc"):\n        module = importlib.import_module(f"smg_grpc_proto.generated.{name}{suffix}")\n        path = Path(module.__file__).resolve()\n        assert path.is_relative_to(expected[smg_grpc_proto].parent / "generated")\n        modules[module.__name__] = str(path)\nprint(json.dumps({"python": sys.executable, "prefix": sys.prefix,\n                  "sglang": sglang.__version__, "HIP": torch.version.hip,\n                  "editable_sources": {m.__name__: str(p) for m, p in expected.items()},\n                  "generated_modules": modules, "GPU_or_model_accepted": False}))\n', str(source)])
        run("CI-current-engine-sources", ["/opt/venv/bin/python", "-c", 'import hashlib\nimport importlib.metadata as md\nimport json\nfrom pathlib import Path\nimport sglang\n\npacket = Path("/source/.github/amd-shared-product/qwen-paired-ci/paired")\ninputs = json.loads((packet / "inputs.json").read_text())\nroot = Path(sglang.__file__).resolve().parent\nassert root == Path("/sgl-workspace/sglang/python/sglang")\nrows = []\nfor suffix in inputs["required_engine_source_suffixes"]:\n    path = root.parent / suffix.lstrip("/")\n    assert path.is_file() and not path.is_symlink()\n    data = path.read_bytes()\n    assert len(data) <= 2 * 1024**2\n    assert all(token not in data for token in (b"SMG_QWEN_MROPE_TRACE", b"SMG_QWEN_MM_TRACE", b"SMG_QWEN_MM_MODEL"))\n    rows.append({"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})\ndist = md.distribution("sglang")\nmetadata = {name: dist.read_text(name) for name in ("METADATA", "RECORD", "direct_url.json")}\nPath("/evidence/engine-distribution-metadata.json").write_text(json.dumps(metadata, indent=2) + "\\n")\nPath("/evidence/engine-source-evidence.json").write_text(json.dumps({"engine_source_files": rows, "sglang_origin": str(root / "__init__.py"), "version": dist.version, "source_RECORD_scope": "editable sources are recorded by path/hash; no claim that the editable distribution RECORD hashes these engine source files", "no_private_hook_markers_in_required_files": True}, indent=2) + "\\n")\n'])
        run("paired-runtime-identity", ["/opt/venv/bin/python", str(packet / "runtime_check.py"), "--inputs", str(packet / "inputs.json"), "--out", "/evidence/runtime-prepared.json"])
        result["whole_pip"] = run("whole-pip-prepared", ["/opt/venv/bin/python", "-m", "pip", "check"])
        run("CI-installed-native", ["/opt/venv/bin/python", "scripts/ci_repair_sglang_rocm_native.py", "--verify-only", "/evidence/dependencies/installed-native.json"])
        from importlib.metadata import distribution
        dist = distribution("smg")
        native = Path(dist.locate_file("smg/smg_rs.abi3.so"))
        assert sha(native) == inputs["gateway_native"]["sha256"]
        result["native_sha256"] = sha(native)
        engine = json.loads((out / "engine-source-evidence.json").read_text())
        result["engine_source_files"] = engine["engine_source_files"]
        result["native_download_common_sha256"] = next(
            row["sha256"] for row in engine["engine_source_files"]
            if row["path"].endswith("/sglang/srt/utils/common.py")
        )
        result["old_private_MM_or_MRoPE_hooks_installed"] = False
        result["old_private_probe_scope"] = "No old overlay is applied; all required engine files passed the explicit private-hook marker absence check; actual source hashes/metadata are retained."
        result["python_resolved"] = str(Path("/opt/venv/bin/python").resolve())
        result["effective_PIP_CONSTRAINT"] = env["PIP_CONSTRAINT"]
        result["installed_source_files"] = []
        for item in inputs["installed_source_files"]:
            path = source / item["source_relative"] if "source_relative" in item else Path(item["path"])
            assert path.is_file() and not path.is_symlink() and sha(path) == item["sha256"]
            result["installed_source_files"].append({"path": str(path), "sha256": sha(path)})
        result["actual_success"] = True
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)[:2000]}
    finally:
        result["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
        (out / "preparation-proof.json").write_text(json.dumps(result, indent=2) + "\n")
        # The container's default user is the actual runner UID/GID. Root is used
        # only for original image setup; return only this owned evidence tree.
        seen = 0
        for parent, dirs, files in os.walk(out, followlinks=False):
            assert all(not (Path(parent) / name).is_symlink() for name in dirs + files)
            for path in [Path(parent), *[Path(parent) / name for name in files]]:
                seen += 1
                assert seen <= 4096 and path.resolve().is_relative_to(out)
                os.chown(path, uid, gid)
        # Normal editable/build steps can create root-owned source metadata.
        # Return only this exported nonce workspace so its runner can remove it.
        seen = 0
        for parent, dirs, files in os.walk(source, followlinks=False):
            for name in list(dirs):
                if (Path(parent) / name).is_symlink():
                    dirs.remove(name)
            for path in [Path(parent), *[Path(parent) / name for name in files]]:
                seen += 1
                assert seen <= 8192 and (path == source or path.parent.resolve().is_relative_to(source))
                if not path.is_symlink():
                    os.chown(path, uid, gid)
    return 0 if result["actual_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
