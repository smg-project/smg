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


@pytest.mark.parametrize("verified", [True, False], ids=["reviewed-source", "legacy-source"])
def test_both_original_asymmetric_param_markers_require_verified_read_source(monkeypatch, verified):
    from importlib.machinery import ModuleSpec

    sweep = _sweep(monkeypatch, "vllm", "moriio", "amd")
    monkeypatch.setenv(constants.ENV_VLLM_MORIIO_MODE, "read")
    monkeypatch.setenv(constants.ENV_VLLM_MORIIO_BACKEND, "rdma")
    original_spec = sweep.importlib.util.spec_from_file_location

    class SourceVerifierLoader:
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            def verify_installed():
                if not verified:
                    raise ValueError("installed source does not match the reviewed cohort")

            module.verify_installed = verify_installed

    def verifier_spec(name, path, *args, **kwargs):
        if name == "_pd_moriio_read_image":
            return ModuleSpec(name, SourceVerifierLoader())
        return original_spec(name, path, *args, **kwargs)

    # Install the verifier boundary before the real module constructs its
    # parameter marks. Checking the helper alone would miss a blanket skip.
    monkeypatch.setattr(sweep.importlib.util, "spec_from_file_location", verifier_spec)
    sweep = importlib.reload(sweep)
    rows = {p.id: p for p in sweep._TOPOLOGIES if "-ptp" in p.id}
    assert set(rows) == {"1p2d-ptp2-dtp1", "2p1d-ptp1-dtp2"}
    for row in rows.values():
        assert row.marks[0].name == "skipif"
        assert row.marks[0].args == (not verified,)


def _first_read_attempt(monkeypatch):
    import os

    sweep = _sweep(monkeypatch, "vllm", "moriio", "amd")
    monkeypatch.setenv(constants.ENV_VLLM_MORIIO_MODE, "read")
    monkeypatch.setenv(constants.ENV_VLLM_MORIIO_BACKEND, "rdma")
    verifier = sweep._moriio_read_image_verifier()
    exports = verifier._qualification_exports(
        "example.invalid/read@sha256:" + "a" * 64,
        dict(verifier.SOURCE_SHA256),
        dict(os.environ),
    )
    for name, value in exports.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(verifier, "verify_installed", lambda: dict(verifier.SOURCE_SHA256))
    monkeypatch.setattr(sweep, "_moriio_read_image_verifier", lambda: verifier)
    return sweep, verifier


def test_first_verified_strict_read_attempt_reaches_original_restart_oracles(monkeypatch):
    sweep, _ = _first_read_attempt(monkeypatch)
    assert sweep._moriio_holds_memory_of("prefill") is False
    assert sweep._moriio_holds_memory_of("decode") is False


@pytest.mark.parametrize(
    "bad", ["no_intent", "bare_flag", "old_image", "source", "auto_xgmi", "xgmi", "write", "vendor"]
)
def test_first_attempt_does_not_bypass_legacy_restart_protection(monkeypatch, bad):
    sweep, verifier = _first_read_attempt(monkeypatch)
    if bad == "no_intent":
        monkeypatch.delenv(verifier.QUALIFICATION_ENV)
    elif bad == "bare_flag":
        monkeypatch.setenv(verifier.QUALIFICATION_ENV, "1")
    elif bad == "old_image":
        monkeypatch.setenv(
            "MORIIO_READ_IMAGE", "example.invalid/old@sha256:" + verifier.OLD_IMAGE_DIGEST
        )
    elif bad == "source":

        def wrong_source():
            raise ValueError("source mismatch")

        monkeypatch.setattr(verifier, "verify_installed", wrong_source)
    elif bad == "auto_xgmi":
        monkeypatch.setenv("MORI_DISABLE_AUTO_XGMI", "0")
    elif bad == "xgmi":
        monkeypatch.setenv(constants.ENV_VLLM_MORIIO_BACKEND, "xgmi")
    elif bad == "write":
        monkeypatch.setenv(constants.ENV_VLLM_MORIIO_MODE, "write")
    else:
        monkeypatch.setenv(constants.ENV_GPU_VENDOR, "nvidia")
    assert sweep._moriio_holds_memory_of("prefill") is (bad != "write")
    assert sweep._moriio_holds_memory_of("decode") is (bad == "write")


def _reclaim_workers():
    from types import SimpleNamespace

    victim = SimpleNamespace(gpu_ids=[2, 3], _gpu_mem_baseline={2: 100, 3: 200}, process=None)
    mover = SimpleNamespace(gpu_ids=[4], process=SimpleNamespace(pid=123), is_alive=lambda: True)
    return victim, mover


def test_strict_reclaim_waits_for_every_victim_gpu_with_one_total_budget(monkeypatch):
    sweep, _ = _first_read_attempt(monkeypatch)
    victim, mover = _reclaim_workers()
    clock = [0.0]
    calls = []
    readings = iter([{2: 164, 3: 265}, {2: 164, 3: 264}])
    monkeypatch.setattr(sweep.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        sweep.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )

    def read(gpu_ids, timeout):
        calls.append((list(gpu_ids), timeout))
        return next(readings)

    monkeypatch.setattr(sweep.process_utils, "gpu_memory_used_mib", read)
    sweep._assert_strict_read_gpu_reclaimed(victim, "prefill", [mover])
    assert calls == [([2, 3], 30.0), ([2, 3], 29.5)]


@pytest.mark.parametrize(
    "reading",
    [
        None,
        {},
        {2: 100},
        {2: 100, 3: None},
        {2: 100, 3: True},
        {2: 100, 3: -1},
        {2: 100, 3: 200, 4: 0},
    ],
)
@pytest.mark.parametrize("stage", ["baseline", "final"])
def test_strict_reclaim_rejects_missing_partial_or_invalid_all_gpu_readings(
    monkeypatch, reading, stage
):
    sweep, _ = _first_read_attempt(monkeypatch)
    victim, mover = _reclaim_workers()
    if stage == "baseline":
        victim._gpu_mem_baseline = reading
    monkeypatch.setattr(sweep.process_utils, "gpu_memory_used_mib", lambda *args, **kwargs: reading)
    with pytest.raises(AssertionError, match="baseline|readback"):
        sweep._assert_strict_read_gpu_reclaimed(victim, "prefill", [mover])


def test_strict_reclaim_held_allocation_expires_without_reloading_model(monkeypatch):
    sweep, _ = _first_read_attempt(monkeypatch)
    victim, mover = _reclaim_workers()
    clock = [0.0]
    monkeypatch.setattr(sweep.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        sweep.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    monkeypatch.setattr(
        sweep.process_utils, "gpu_memory_used_mib", lambda *args, **kwargs: {2: 165, 3: 200}
    )
    with pytest.raises(AssertionError, match="30 seconds"):
        sweep._assert_strict_read_gpu_reclaimed(victim, "prefill", [mover])
    assert clock[0] == 30.0


@pytest.mark.parametrize("changed", ["dead", "pid", "gpu"])
def test_strict_reclaim_requires_same_live_mover_identity(monkeypatch, changed):
    sweep, _ = _first_read_attempt(monkeypatch)
    victim, mover = _reclaim_workers()

    def read(*args, **kwargs):
        if changed == "dead":
            mover.is_alive = lambda: False
        elif changed == "pid":
            mover.process.pid += 1
        else:
            mover.gpu_ids = [5]
        return {2: 100, 3: 200}

    monkeypatch.setattr(sweep.process_utils, "gpu_memory_used_mib", read)
    with pytest.raises(AssertionError, match="mover"):
        sweep._assert_strict_read_gpu_reclaimed(victim, "prefill", [mover])


@pytest.mark.parametrize("role,intent", [("decode", True), ("prefill", False)])
def test_reclaim_measurement_does_not_touch_legacy_or_decode_paths(monkeypatch, role, intent):
    sweep, verifier = _first_read_attempt(monkeypatch)
    victim, mover = _reclaim_workers()
    if not intent:
        monkeypatch.delenv(verifier.QUALIFICATION_ENV)

    def unexpected_read(*args, **kwargs):
        pytest.fail("non-qualifying path must not read GPU memory")

    monkeypatch.setattr(sweep.process_utils, "gpu_memory_used_mib", unexpected_read)
    sweep._assert_strict_read_gpu_reclaimed(victim, role, [mover])


@pytest.mark.parametrize("raw", [True, 12.5])
def test_reclaim_consumer_rejects_invalid_raw_smi_before_integer_conversion(monkeypatch, raw):
    import json
    import subprocess

    sweep, _ = _first_read_attempt(monkeypatch)
    victim, mover = _reclaim_workers()
    victim.gpu_ids = [0]
    victim._gpu_mem_baseline = {0: 100}

    def fake_smi(cmd, **kwargs):
        if cmd[0] == "nvidia-smi":
            raise FileNotFoundError("fake AMD-only tool")
        data = (
            [{"gpu": 0, "node_id": 2}]
            if cmd[1] == "list"
            else [{"gpu": 0, "mem_usage": {"used_vram": raw}}]
        )
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(data))

    monkeypatch.setattr(sweep.process_utils.subprocess, "run", fake_smi)
    with pytest.raises(AssertionError, match="readback"):
        sweep._assert_strict_read_gpu_reclaimed(victim, "prefill", [mover])
