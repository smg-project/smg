"""Engine-free tests for SGLang scheduler process lifecycle."""

import importlib.util
import inspect
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _module(name, **attrs):
    module = ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    return module


def _run_scheduler_process(
    server_args,
    port_args,
    gpu_id,
    tp_rank,
    pp_rank,
    dp_rank,
    pipe_writer,
    display_tp_rank=None,
    display_dp_rank=None,
    display_moe_ep_rank=None,
):
    """SGLang 0.5.21's `run_scheduler_process` signature."""


@pytest.fixture
def scheduler_launcher_mod(monkeypatch):
    """Load scheduler_launcher.py without importing the SGLang engine."""

    class TorchMemorySaverAdapter:
        @staticmethod
        def create(**_kwargs):
            return SimpleNamespace(configure_subprocess=nullcontext)

    stubs = {
        "sglang": _module("sglang"),
        "sglang.srt": _module("sglang.srt"),
        "sglang.srt.managers": _module("sglang.srt.managers"),
        "sglang.srt.managers.data_parallel_controller": _module(
            "sglang.srt.managers.data_parallel_controller",
            run_data_parallel_controller_process=lambda *_args: None,
        ),
        "sglang.srt.managers.scheduler": _module(
            "sglang.srt.managers.scheduler",
            run_scheduler_process=_run_scheduler_process,
        ),
        "sglang.srt.server_args": _module(
            "sglang.srt.server_args",
            PortArgs=object,
            ServerArgs=object,
        ),
        "sglang.srt.utils": _module(
            "sglang.srt.utils",
            configure_logger=lambda _args: None,
            numa_utils=SimpleNamespace(
                configure_subprocess=lambda *_args: nullcontext(),
            ),
        ),
        "sglang.srt.utils.torch_memory_saver_adapter": _module(
            "sglang.srt.utils.torch_memory_saver_adapter",
            TorchMemorySaverAdapter=TorchMemorySaverAdapter,
        ),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_path = (
        Path(__file__).parents[1] / "smg_grpc_servicer" / "sglang" / "scheduler_launcher.py"
    )
    spec = importlib.util.spec_from_file_location("test_sglang_scheduler_launcher", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ready_failure_terminates_every_started_scheduler(monkeypatch, scheduler_launcher_mod):
    class Reader:
        def __init__(self, result):
            self.result = result

        def recv(self):
            return self.result

    class Process:
        instances = []

        def __init__(self, **_kwargs):
            self.alive = False
            self.terminated = False
            self.killed = False
            self.join_timeouts = []
            self.exitcode = None
            self.instances.append(self)

        def start(self):
            self.alive = True

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminated = True
            self.alive = False

        def kill(self):
            self.killed = True
            self.alive = False

        def join(self, timeout=None):
            self.join_timeouts.append(timeout)

    readers = iter(
        [
            Reader({"status": "ready"}),
            Reader({"status": "error", "error": "rank failed"}),
        ]
    )
    monkeypatch.setattr(scheduler_launcher_mod.mp, "set_start_method", lambda *_args, **_kw: None)
    monkeypatch.setattr(
        scheduler_launcher_mod.mp, "Pipe", lambda **_kwargs: (next(readers), object())
    )
    monkeypatch.setattr(scheduler_launcher_mod.mp, "Process", Process)

    server_args = SimpleNamespace(
        check_server_args=lambda: None,
        dp_size=1,
        enable_memory_saver=False,
        nnodes=1,
        pp_size=1,
        tp_size=2,
        node_rank=0,
        base_gpu_id=0,
        gpu_id_step=1,
    )

    with pytest.raises(RuntimeError, match="rank 1 initialization failed: rank failed"):
        scheduler_launcher_mod.launch_scheduler_process_only(
            server_args,
            port_args=object(),
        )

    assert len(Process.instances) == 2
    assert all(proc.terminated for proc in Process.instances)
    assert all(not proc.is_alive() for proc in Process.instances)
    assert all(proc.join_timeouts == [2.0] for proc in Process.instances)


def test_schedulers_start_with_the_run_scheduler_process_arguments(
    monkeypatch, scheduler_launcher_mod
):
    class Reader:
        def recv(self):
            return {"status": "ready"}

    class Process:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.instances.append(self)

        def start(self):
            pass

    writers = [object(), object()]
    pipes = iter([(Reader(), writer) for writer in writers])
    monkeypatch.setattr(scheduler_launcher_mod.mp, "set_start_method", lambda *_args, **_kw: None)
    monkeypatch.setattr(scheduler_launcher_mod.mp, "Pipe", lambda **_kwargs: next(pipes))
    monkeypatch.setattr(scheduler_launcher_mod.mp, "Process", Process)

    server_args = SimpleNamespace(
        check_server_args=lambda: None,
        dp_size=1,
        enable_memory_saver=False,
        nnodes=1,
        pp_size=1,
        tp_size=2,
        node_rank=0,
        base_gpu_id=4,
        gpu_id_step=2,
    )
    port_args = object()

    scheduler_launcher_mod.launch_scheduler_process_only(server_args, port_args=port_args)

    signature = inspect.signature(scheduler_launcher_mod.run_scheduler_process)
    assert len(Process.instances) == 2
    for tp_rank, proc in enumerate(Process.instances):
        # By name: an extra positional argument still binds to a defaulted trailing
        # parameter, so a successful bind alone proves nothing.
        bound = signature.bind(*proc.kwargs["args"]).arguments
        assert bound["server_args"] is server_args
        assert bound["port_args"] is port_args
        assert bound["gpu_id"] == 4 + 2 * tp_rank
        assert bound["tp_rank"] == tp_rank
        assert bound["pp_rank"] == 0
        assert bound["dp_rank"] is None
        assert bound["pipe_writer"] is writers[tp_rank]
        assert not {"display_tp_rank", "display_dp_rank", "display_moe_ep_rank"} & bound.keys()
