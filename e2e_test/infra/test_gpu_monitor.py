"""CPU controls for selected-device GPU monitoring; no runtime SDK imports."""

import ast
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "monitor_under_test", Path(__file__).with_name("gpu_monitor.py")
)
assert _SPEC is not None and _SPEC.loader is not None
monitor = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(monitor)


@pytest.fixture(autouse=True)
def unmasked(monkeypatch):
    for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(key, raising=False)


def fake_torch(props=None):
    props = props or [
        SimpleNamespace(pci_domain_id=0, pci_bus_id=bus, pci_device_id=0)
        for bus in (0x83, 0x8B, 0x93, 0x9B)
    ]
    return SimpleNamespace(
        version=SimpleNamespace(hip="7.15"),
        cuda=SimpleNamespace(
            device_count=lambda: len(props), get_device_properties=lambda i: props[i]
        ),
    )


class FakeSMI:
    __version__ = "27.0.0+6b0e43f3"
    __file__ = "/exact/amdsmi/__init__.py"

    def __init__(self):
        self.handles = ["other", "d", "b", "a", "c"]
        self.bdfs = dict(
            zip(
                self.handles,
                ["0000:bb:00.0", "0000:9b:00.0", "0000:8b:00.0", "0000:83:00.0", "0000:93:00.0"],
            )
        )
        self.values = dict(zip(self.handles, [0, 100, 20, 40, 80]))
        self.init_calls = self.shutdown_calls = 0

    def amdsmi_init(self):
        self.init_calls += 1

    def amdsmi_shut_down(self):
        self.shutdown_calls += 1

    def amdsmi_get_lib_version(self):
        return {"major": 27, "minor": 0}

    def amdsmi_get_processor_handles(self):
        return self.handles

    def amdsmi_get_gpu_device_bdf(self, handle):
        return self.bdfs[handle]

    def amdsmi_get_processor_handle_from_bdf(self, bdf):
        return next(h for h in self.handles if self.bdfs[h] == bdf)

    def amdsmi_get_processor_type(self, handle):
        return {"processor_type": "AMD_GPU"}

    def amdsmi_get_gpu_busy_percent(self, handle):
        value = self.values[handle]
        if isinstance(value, Exception):
            raise value
        return value


def test_permuted_handles_select_actual_four_not_idle_fifth():
    selected = monitor._amd_worker_bdfs(fake_torch(), [0, 1, 2, 3])
    smi = FakeSMI()
    handles = monitor._amd_handles(smi, selected)
    assert list(handles.values()) == ["a", "b", "c", "d"]
    values = monitor._amd_sample(smi, handles)
    assert values == {"0": 40.0, "1": 20.0, "2": 80.0, "3": 100.0}
    assert sum(values.values()) / len(values) == 60


@pytest.mark.parametrize(
    "value", ["N/A", 0xFFFFFFFF, True, False, -1, 101, None, 5.5, RuntimeError("driver error")]
)
def test_invalid_activity_is_unavailable_not_zero(value):
    smi = FakeSMI()
    smi.values["a"] = value
    with pytest.raises((ValueError, RuntimeError)):
        monitor._amd_sample(smi, {0: "a"})


@pytest.mark.parametrize("value", [0, 100])
def test_real_zero_and_hundred_are_valid(value):
    smi = FakeSMI()
    smi.values["a"] = value
    assert monitor._amd_sample(smi, {0: "a"}) == {"0": float(value)}


@pytest.mark.parametrize("ids", [[], [0, 0], [4], [-1], [True]])
def test_invalid_worker_selection_rejected(ids):
    with pytest.raises(ValueError):
        monitor._amd_worker_bdfs(fake_torch(), ids)


@pytest.mark.parametrize(
    "props",
    [
        SimpleNamespace(),
        SimpleNamespace(pci_domain_id=65536, pci_bus_id=0, pci_device_id=0),
        SimpleNamespace(pci_domain_id=0, pci_bus_id=True, pci_device_id=0),
    ],
)
def test_missing_or_invalid_physical_identity_rejected(props):
    with pytest.raises((AttributeError, ValueError)):
        monitor._amd_worker_bdfs(fake_torch([props]), [0])


def test_duplicate_physical_identity_rejected():
    p = SimpleNamespace(pci_domain_id=0, pci_bus_id=0x83, pci_device_id=0)
    with pytest.raises(ValueError):
        monitor._amd_worker_bdfs(fake_torch([p, p]), [0, 1])


@pytest.mark.parametrize(
    "mask", ["CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"]
)
def test_ambiguous_parent_mask_rejected(monkeypatch, mask):
    monkeypatch.setenv(mask, "0,1")
    with pytest.raises(ValueError, match="unmasked"):
        monitor._amd_worker_bdfs(fake_torch(), [0])


@pytest.mark.parametrize("defect", ["missing", "duplicate", "foreign", "wrong_type", "roundtrip"])
def test_smi_mapping_defects_rejected(defect):
    smi = FakeSMI()
    selected = {0: "0000:83:00.0"}
    if defect == "missing":
        smi.handles.remove("a")
    elif defect == "duplicate":
        smi.bdfs["other"] = smi.bdfs["a"]
    elif defect == "foreign":
        selected = {0: "0000:ff:00.0"}
    elif defect == "wrong_type":
        smi.amdsmi_get_processor_type = lambda _: {"processor_type": "AMD_CPU"}
    else:
        smi.amdsmi_get_processor_handle_from_bdf = lambda _: "b"
    with pytest.raises((ValueError, KeyError)):
        monitor._amd_handles(smi, selected)


def run_fake_loop(monkeypatch, tmp_path, smi, ticks=(True, False)):
    monkeypatch.setitem(sys.modules, "amdsmi", smi)
    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    alive = iter(ticks)
    monkeypatch.setattr(monitor, "_process_alive", lambda _: next(alive))
    monkeypatch.setattr(monitor.time, "sleep", lambda _: None)
    path = tmp_path / "gpu_utilization.json"
    monitor._amd_monitor_loop(123, str(path), 2.0, [0, 1, 2, 3])
    return json.loads(path.read_text())


def test_loop_success_and_shutdown(monkeypatch, tmp_path):
    smi = FakeSMI()
    result = run_fake_loop(monkeypatch, tmp_path, smi)
    assert result["status"] == "available" and result["count"] == 1
    assert result["overall"]["mean"] == 60
    assert result["selected_bdfs"]["0"] == "0000:83:00.0"
    assert smi.init_calls == smi.shutdown_calls == 1


@pytest.mark.parametrize("phase", ["init", "read", "shutdown", "no_ticks"])
def test_loop_error_persists_unavailable_and_cleans(monkeypatch, tmp_path, phase):
    smi = FakeSMI()

    def fail():
        raise RuntimeError("explicit failure")

    if phase == "init":
        smi.amdsmi_init = fail
    elif phase == "read":
        smi.values["a"] = RuntimeError("read failed")
    elif phase == "shutdown":
        smi.amdsmi_shut_down = fail
    result = run_fake_loop(
        monkeypatch, tmp_path, smi, (False,) if phase == "no_ticks" else (True, False)
    )
    assert result["status"] == "unavailable" and result["overall"] is None
    assert result["reason"]
    if phase in ("read", "no_ticks"):
        assert smi.shutdown_calls == 1
    if phase == "init":
        assert smi.shutdown_calls == 0


def test_unavailable_is_threshold_miss_not_clean(tmp_path):
    path = tmp_path / "gpu_utilization.json"
    path.write_text(
        json.dumps(
            {"status": "unavailable", "reason": "missing driver", "count": 0, "overall": None}
        )
    )
    m = monitor.GPUMonitor(tmp_path, backend="amdsmi", gpu_ids=[0])
    m._output_path = str(path)
    assert m.threshold_misses({"gpu_util_mean_min": 30}) == [
        "GPU utilization unavailable: missing driver"
    ]
    path.unlink()
    assert m.threshold_misses({"gpu_util_mean_min": 30})


def test_owned_sampler_is_reaped_after_termination(tmp_path):
    events = []
    process = SimpleNamespace(
        join=lambda **_: events.append("join"),
        is_alive=lambda: len(events) < 3,
        terminate=lambda: events.append("terminate"),
        kill=lambda: events.append("kill"),
    )
    m = monitor.GPUMonitor(tmp_path, backend="amdsmi")
    m._process = process
    m.stop(timeout=0.01)
    assert events == ["join", "terminate", "join"] and m._process is None


def test_nvml_original_sampling_path(monkeypatch, tmp_path):
    calls = []
    nvml = SimpleNamespace(
        nvmlInit=lambda: calls.append("init"),
        nvmlDeviceGetCount=lambda: 2,
        nvmlDeviceGetHandleByIndex=lambda i: i,
        nvmlDeviceGetUtilizationRates=lambda h: SimpleNamespace(gpu=20 + h * 40),
        nvmlShutdown=lambda: calls.append("shutdown"),
    )
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    ticks = iter([True, False])
    monkeypatch.setattr(monitor, "_process_alive", lambda _: next(ticks))
    monkeypatch.setattr(monitor.time, "sleep", lambda _: None)
    monkeypatch.setattr(monitor.os, "nice", lambda _: None)
    path = tmp_path / "nvml.json"
    monitor._monitor_loop(123, str(path), 2)
    r = json.loads(path.read_text())
    assert r["overall"]["mean"] == 40 and r["count"] == 1
    assert calls == ["init", "shutdown"] and "backend" not in r


def test_fixture_failure_still_stops_sampler(monkeypatch, tmp_path):
    # Execute the actual fixture with fake process/report seams, not a model.
    import logging
    import subprocess

    source = Path(__file__).parents[1] / "benchmarks" / "conftest.py"
    node = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "genai_bench_runner"
    )
    node.decorator_list = []
    events = []
    proc = SimpleNamespace(pid=123, returncode=1, wait=lambda **_: None, poll=lambda: 1)

    class Sampler:
        def __init__(self, **kwargs):
            pass

        def start(self, **kwargs):
            events.append("start")

        def stop(self, **kwargs):
            events.append("stop")

    ns = {
        "Path": Path,
        "os": os,
        "shutil": __import__("shutil"),
        "subprocess": SimpleNamespace(
            Popen=lambda *a, **k: proc, TimeoutExpired=subprocess.TimeoutExpired
        ),
        "pytest": pytest,
        "logger": logging.getLogger(__name__),
        "GPUMonitor": Sampler,
        "should_monitor_gpu": lambda _: True,
        "_build_command": lambda *a, **k: ["fake-bench"],
        "_cleanup_procs": lambda *a: events.append("cleanup"),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), ns)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(pytest.fail.Exception, match="exit code 1"):
        ns["genai_bench_runner"]()(
            router_url="unused",
            model_path="original",
            experiment_folder="owned",
            thresholds={"gpu_util_mean_min": 30},
        )
    assert events == ["start", "cleanup", "stop"]
