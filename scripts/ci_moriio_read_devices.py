"""A32 exact metadata/cache helpers, selected view cardinality two to four only."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

DEVICE_MARKER = "A27_DEVICE_METADATA="

DEVICE_PROBE = r"""
import json, torch
assert torch.version.hip and torch.cuda.is_available()
count = torch.cuda.device_count()
assert 1 <= count <= 8
items = []
for index in range(count):
    device = torch.cuda.get_device_properties(index)
    items.append(dict(index=index, uuid=str(device.uuid),
        pci_domain=int(device.pci_domain_id), pci_bus=int(device.pci_bus_id),
        pci_device=int(device.pci_device_id), gfx=str(device.gcnArchName).split(":")[0]))
RAY_PROBE
print("A27_DEVICE_METADATA=" + json.dumps(items))
"""


def device_identity(device):
    return tuple(device[key] for key in ("uuid", "pci_domain", "pci_bus", "pci_device", "gfx"))


def device_probe(env, out, name, *, ray, timeout=60):
    source = DEVICE_PROBE.replace(
        "RAY_PROBE", "import ray; assert not ray.is_initialized()" if ray else ""
    )
    run = subprocess.run(
        [sys.executable, "-I", "-B", "-c", source],
        env=env,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    for suffix, content in (("stdout", run.stdout), ("stderr", run.stderr)):
        assert len(content) <= 1024**2
        fd = os.open(
            out / (name + "." + suffix),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
    assert run.returncode == 0, "Fresh device/Ray metadata probe failed; retain logs"
    records = [
        line[len(DEVICE_MARKER) :]
        for line in run.stdout.decode().splitlines()
        if line.startswith(DEVICE_MARKER)
    ]
    assert len(records) == 1
    items = json.loads(records[0])
    assert isinstance(items, list) and 1 <= len(items) <= 8
    assert [item["index"] for item in items] == list(range(len(items)))
    assert len({item["uuid"] for item in items}) == len(items)
    assert all(isinstance(item["uuid"], str) and item["uuid"] for item in items)
    return items


def pytest_device_env(env, expected, out):
    # Officialbf88Worker copiesparentenv thenoverridesCUDA for its ownrange.
    # ParentHIPfour would supersede thatCUDA subset and aborttheROCmagent.
    assert len(expected) == 4 and len({device_identity(item) for item in expected}) == 4
    assert all(item["gfx"] == "gfx942" for item in expected)
    assert env.get("E2E_GPU_OFFSET", "0") in ("", "0"), "unreviewed worker GPU offset"
    clean = dict(env)
    for key in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        clean.pop(key, None)
    deadline = time.monotonic() + 300

    def probe(view, name, *, ray):
        left = deadline - time.monotonic()
        assert left > 0, "Bounded worker visibility preflight exhausted"
        return device_probe(view, out, name, ray=ray, timeout=min(30, left))

    available = probe(clean, "hip-index-discovery", ray=False)
    selected = []
    for wanted in expected:
        matches = [item for item in available if device_identity(item) == device_identity(wanted)]
        assert len(matches) == 1, "Original selected identity missing/ambiguous in HIP index view"
        selected.append(matches[0]["index"])
    assert len(set(selected)) == 4
    mask = ",".join(str(index) for index in selected)
    clean["CUDA_VISIBLE_DEVICES"] = mask
    actual = probe(clean, "cuda-parent-four-ray", ray=True)
    assert [device_identity(item) for item in actual] == [
        device_identity(item) for item in expected
    ], "Parent CUDA view changed selected four physical identities"
    worker_views = []
    # start_workers assigns list(range(gpu_offset,gpu_offset+gpus_per_worker)).
    # These ten nonemptycontiguousranges cover everywithin4 originalTP role.
    for start in range(4):
        for stop in range(start + 1, 5):
            indices = list(range(start, stop))
            worker = dict(clean)
            worker["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, indices))
            assert "HIP_VISIBLE_DEVICES" not in worker and "ROCR_VISIBLE_DEVICES" not in worker
            observed = probe(worker, "cuda-worker-" + "-".join(map(str, indices)), ray=False)
            assert [device_identity(item) for item in observed] == [
                device_identity(expected[i]) for i in indices
            ], "Original worker CUDA subset lost/reordered selected physical identities"
            worker_views.append(
                {
                    "logical_gpu_ids": indices,
                    "CUDA_VISIBLE_DEVICES": worker["CUDA_VISIBLE_DEVICES"],
                    "HIP_VISIBLE_DEVICES": None,
                    "observed": observed,
                }
            )
    assert time.monotonic() < deadline
    return clean, {
        "original_ROCR": env.get("ROCR_VISIBLE_DEVICES"),
        "HIP_VISIBLE_DEVICES": None,
        "CUDA_VISIBLE_DEVICES": mask,
        "expected": expected,
        "available": available,
        "selected": actual,
        "worker_preflight": worker_views,
        "worker_preflight_count": 10,
        "metadata_preflight_wall_seconds": 300,
        "ray_import_in_fresh_selected_process": True,
        "claim": (
            "Actual metadata-only parent and all original within4 worker CUDA range views; "
            "no model/native-case success"
        ),
    }


def cache_files(root):
    files = {}
    for path in sorted(root.rglob("*")):
        assert not path.is_symlink(), "Unexpected link in installed AITER JIT tree"
        if path.is_dir():
            continue
        assert path.is_file(), "Nonregular AITER cache material"
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024**2), b""):
                digest.update(chunk)
        files[str(path.relative_to(root))] = {
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
    return files


def writable_cache_env(env, out, jit_source):
    # AITER's explicit JIT_DIR imports prebuilt modules by top-level name.
    # Copy the same installed JIT tree, like its existing read-only fallback,
    # rather than presenting an empty cache and silently forcing recompilation.
    expected = cache_files(jit_source)
    assert "core.py" in expected
    assert (
        shutil.disk_usage(out).free
        >= sum(item["bytes"] for item in expected.values()) + 20 * 1024**3
    )
    root = out / "cache"
    root.mkdir(mode=0o700)
    aiter = root / "aiter"
    shutil.copytree(jit_source, aiter)
    assert cache_files(aiter) == expected, "Copied AITER JIT source/prebuilt bytes changed"
    paths = {
        "AITER_JIT_DIR": aiter,
        "AITER_ROOT_DIR": root / "aiter-cpp",
        "TORCH_EXTENSIONS_DIR": root / "torch-extensions",
        "VLLM_CACHE_ROOT": root / "vllm",
        "TRITON_HOME": root / "triton-home",
    }
    for path in paths.values():
        path.mkdir(mode=0o700, exist_ok=True)
    result = dict(env)
    # vLLM assigns these inside each original role's VLLM_CACHE_ROOT. Keep
    # its model/role-local behavior, not a shared global compiler-cache override.
    assert not result.get("TRITON_CACHE_DIR") and not result.get("TORCHINDUCTOR_CACHE_DIR")
    result.update({key: str(path) for key, path in paths.items()})
    pytest_cache = root / "pytest"
    pytest_cache.mkdir(mode=0o700)
    return result, {
        "environment": {key: str(path) for key, path in paths.items()},
        "pytest_cache": str(pytest_cache),
        "aiter_installed_source": str(jit_source),
        "aiter_copied_files": expected,
        "scope": "Same installed JIT bytes copied to writable operation cache; not a kernel/ABI or case-success claim",
    }
