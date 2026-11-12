"""The PD topology sweep's restart skips: which sole-worker restart names an engine-side reason.

A vLLM decode over NIXL on CUDA forgets a dead prefill only while it serves a
request, so the sweep skips the sole prefill's restart on that lane alone;
every other lane, backend, role and vendor keeps the restart.
"""

from __future__ import annotations

import importlib
import shutil

import pytest
from infra import constants


def _sweep(monkeypatch, runtime: str, kv_backend: str | None, vendor: str = "nvidia"):
    monkeypatch.setattr(constants, "_RUNTIME_CACHE", None)
    monkeypatch.setenv(constants.ENV_RUNTIME, runtime)
    monkeypatch.setenv(constants.ENV_GPU_VENDOR, vendor)
    monkeypatch.delenv(constants.ENV_KV_BACKEND, raising=False)
    if kv_backend is None:
        monkeypatch.delenv(constants.ENV_VLLM_KV_BACKEND, raising=False)
    else:
        monkeypatch.setenv(constants.ENV_VLLM_KV_BACKEND, kv_backend)
    return importlib.import_module("router.test_pd_topologies")


@pytest.mark.parametrize(
    ("runtime", "kv_backend", "role", "vendor", "skipped"),
    [
        ("vllm", "nixl", "prefill", "nvidia", True),
        ("vllm", None, "prefill", "nvidia", True),  # NIXL is the vLLM lane's default backend
        ("vllm", "nixl", "decode", "nvidia", False),
        ("vllm", "nixl", "prefill", "amd", False),  # the ROCm lane releases the region
        ("vllm", "mooncake", "prefill", "nvidia", False),
        ("vllm", "moriio", "prefill", "nvidia", False),
        ("sglang", None, "prefill", "nvidia", False),
        ("tokenspeed", None, "prefill", "nvidia", False),
    ],
)
def test_only_the_cuda_vllm_nixl_sole_prefill_restart_is_skipped(
    monkeypatch, runtime, kv_backend, role, vendor, skipped
):
    sweep = _sweep(monkeypatch, runtime, kv_backend, vendor)
    assert sweep._nixl_decodes_keep_a_dead_prefills_memory(role) is skipped


def test_the_lane_wide_backend_setting_counts_too(monkeypatch):
    sweep = _sweep(monkeypatch, "vllm", "nixl")
    monkeypatch.setenv(constants.ENV_KV_BACKEND, "mooncake")
    assert sweep._nixl_decodes_keep_a_dead_prefills_memory("prefill") is False


def test_the_vendor_is_read_from_the_job_or_from_the_smi_on_the_path(monkeypatch):
    monkeypatch.setenv(constants.ENV_GPU_VENDOR, " AMD ")
    assert constants.get_gpu_vendor() == "amd"
    monkeypatch.setenv(constants.ENV_GPU_VENDOR, "tpu")
    with pytest.raises(ValueError, match="E2E_GPU_VENDOR"):
        constants.get_gpu_vendor()
    monkeypatch.delenv(constants.ENV_GPU_VENDOR)
    found = {"amd-smi"}
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}" if name in found else None)
    assert constants.get_gpu_vendor() == "amd"
    found.add("nvidia-smi")
    assert constants.get_gpu_vendor() == "nvidia"
    found.clear()
    assert constants.get_gpu_vendor() == "nvidia"


def test_the_skip_reason_names_the_engine_side_cause_and_when_to_lift_it(monkeypatch):
    sweep = _sweep(monkeypatch, "vllm", "nixl")
    assert "startup memory check" in sweep._NIXL_HELD_MEMORY
    assert "while it serves a request" in sweep._NIXL_HELD_MEMORY
    assert "0.31.0" in sweep._NIXL_HELD_MEMORY
    assert "lift the skip" in sweep._NIXL_HELD_MEMORY


@pytest.mark.parametrize(
    ("runtime", "backend", "vendor", "mode", "transport", "supported"),
    [
        ("vllm", "moriio", "amd", "read", "rdma", True),
        ("vllm", "moriio", "amd", "", "", True),
        ("vllm", "moriio", "amd", "write", "rdma", False),
        ("vllm", "moriio", "amd", "read", "xgmi", False),
        ("vllm", "moriio", "nvidia", "read", "rdma", False),
        ("vllm", "nixl", "amd", "read", "rdma", False),
        ("sglang", "moriio", "amd", "read", "rdma", False),
    ],
)
def test_asymmetric_read_selection_requires_the_source_verifier_only_in_its_lane(
    monkeypatch, runtime, backend, vendor, mode, transport, supported
):
    sweep = _sweep(monkeypatch, runtime, backend, vendor)
    monkeypatch.setenv(constants.ENV_VLLM_MORIIO_MODE, mode)
    monkeypatch.setenv(constants.ENV_VLLM_MORIIO_BACKEND, transport)
    calls = []

    class VerifiedSource:
        def verify_installed(self):
            calls.append("verified")

    # This isolates lane selection; the real verifier is exercised below and
    # its exact-source positive is also bound in the offline qualification log.
    monkeypatch.setattr(sweep, "_moriio_read_image_verifier", VerifiedSource)
    assert sweep._moriio_asymmetric_tp_supported() is supported
    assert calls == (["verified"] if supported else [])


@pytest.mark.parametrize("source", ["absent", "missing", "legacy", "shadow"])
def test_real_installed_source_verifier_cannot_unskip_an_unqualified_package(
    monkeypatch, tmp_path, source
):
    import importlib.machinery
    import sys

    sweep = _sweep(monkeypatch, "vllm", "moriio", "amd")
    monkeypatch.setenv(constants.ENV_VLLM_MORIIO_MODE, "read")
    monkeypatch.setenv(constants.ENV_VLLM_MORIIO_BACKEND, "rdma")
    verifier = sweep._moriio_read_image_verifier()
    package = tmp_path / "vllm"
    package.mkdir()
    (package / "__init__.py").write_text("raise RuntimeError('must not import GPU package')\n")
    spec = importlib.machinery.ModuleSpec("vllm", loader=None, is_package=True)
    spec.origin = str(package / "__init__.py")
    spec.submodule_search_locations = [str(package)]
    monkeypatch.delitem(sys.modules, "vllm", raising=False)
    original_find = verifier.importlib.util.find_spec
    monkeypatch.setattr(
        verifier.importlib.util,
        "find_spec",
        lambda name: (
            (None if source == "absent" else spec) if name == "vllm" else original_find(name)
        ),
    )

    class Distribution:
        def locate_file(self, relative):
            return (tmp_path / "other" if source == "shadow" else tmp_path) / relative

    monkeypatch.setattr(verifier.importlib.metadata, "distribution", lambda name: Distribution())
    if source == "legacy":
        for relative in verifier.SOURCE_SHA256:
            path = package / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# unreviewed legacy implementation\n")
    monkeypatch.setattr(sweep, "_moriio_read_image_verifier", lambda: verifier)
    assert sweep._moriio_asymmetric_tp_supported() is False
    assert "vllm" not in sys.modules


def test_original_two_asymmetric_geometry_and_ids_remain(monkeypatch):
    sweep = _sweep(monkeypatch, "vllm", "moriio", "amd")
    rows = {p.id: p for p in sweep._TOPOLOGIES if "-ptp" in p.id}
    assert {name: p.values for name, p in rows.items()} == {
        "1p2d-ptp2-dtp1": (("pd_grpc", (1, 2, 2, 1)),),
        "2p1d-ptp1-dtp2": (("pd_grpc", (2, 1, 1, 2)),),
    }
    assert all(p.marks[0].name == "skipif" for p in rows.values())
