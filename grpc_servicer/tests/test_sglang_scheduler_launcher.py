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
        "sglang.srt.arg_groups": _module("sglang.srt.arg_groups"),
        "sglang.srt.arg_groups.model_override_base": _module(
            "sglang.srt.arg_groups.model_override_base",
            resolved_view=lambda args: getattr(args, "resolved", args),
        ),
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

        def poll(self, _timeout=None):
            return True

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
        scheduler_launcher_mod.mp,
        "Pipe",
        lambda **_kwargs: (next(readers), SimpleNamespace(close=lambda: None)),
    )
    monkeypatch.setattr(scheduler_launcher_mod.mp, "Process", Process)

    server_args = SimpleNamespace(
        resolve_once=lambda: None,
        check_server_args=lambda: None,
        dp_size=1,
        ep_join_mode=None,
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


@pytest.mark.parametrize(
    "overrides,raw_dp_size,placements",
    [
        ({}, 1, [(0, 0, 4), (0, 1, 6)]),
        # Resolved DP selects direct launch; TP spans two nodes.
        ({"dp_size": 1, "tp_size": 4, "nnodes": 2, "node_rank": 1}, 2, [(0, 2, 4), (0, 3, 6)]),
        # Resolved PP selects the second pipeline stage on this node.
        (
            {"pp_size": 2, "nnodes": 2, "node_rank": 1, "base_gpu_id": 1, "gpu_id_step": 3},
            1,
            [(1, 0, 1), (1, 1, 4)],
        ),
        ({"pp_size": 2}, 1, [(0, 0, 4), (0, 1, 6), (1, 0, 8), (1, 1, 10)]),
        ({"dp_size": 2}, 1, None),
        ({"ep_join_mode": "scale", "node_rank": 1}, 1, None),
    ],
    ids=[
        "unchanged",
        "resolved-tp",
        "resolved-pp",
        "local-pp-stride",
        "resolved-dp",
        "elastic-ep-scale",
    ],
)
def test_schedulers_start_with_the_run_scheduler_process_arguments(
    monkeypatch, scheduler_launcher_mod, overrides, raw_dp_size, placements
):
    class Reader:
        def poll(self, _timeout=None):
            return True

        def recv(self):
            return {"status": "ready"}

    class Process:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.instances.append(self)

        def start(self):
            pass

    class Writer:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    writers = [Writer() for _ in range(len(placements) if placements else 1)]
    pipes = iter([(Reader(), writer) for writer in writers])
    monkeypatch.setattr(scheduler_launcher_mod.mp, "set_start_method", lambda *_args, **_kw: None)
    monkeypatch.setattr(scheduler_launcher_mod.mp, "Pipe", lambda **_kwargs: next(pipes))
    monkeypatch.setattr(scheduler_launcher_mod.mp, "Process", Process)

    server_args = SimpleNamespace(
        resolve_once=lambda: None,
        check_server_args=lambda: None,
        dp_size=1,
        ep_join_mode=None,
        enable_memory_saver=False,
        nnodes=1,
        pp_size=1,
        tp_size=2,
        node_rank=0,
        base_gpu_id=4,
        gpu_id_step=2,
    )
    events = []
    raw = vars(server_args).copy()

    def resolve():
        events.append("resolve")
        server_args.resolved = SimpleNamespace(**(raw | overrides))

    server_args.dp_size = raw_dp_size
    server_args.resolve_once = resolve
    server_args.check_server_args = lambda: events.append("validate")
    port_args = object()

    scheduler_launcher_mod.launch_scheduler_process_only(server_args, port_args=port_args)

    assert events == ["resolve", "validate"]
    assert server_args.dp_size == raw_dp_size
    assert server_args.tp_size == 2
    assert len(Process.instances) == len(writers)
    assert all(writer.closed for writer in writers)
    if placements is None:
        assert Process.instances[0].kwargs == {
            "target": scheduler_launcher_mod.run_data_parallel_controller_process,
            "args": (server_args, port_args, writers[0]),
        }
        return

    signature = inspect.signature(scheduler_launcher_mod.run_scheduler_process)
    for index, (proc, (pp_rank, tp_rank, gpu_id)) in enumerate(zip(Process.instances, placements)):
        assert proc.kwargs["target"] is scheduler_launcher_mod.run_scheduler_with_signal_handling
        # By name: an extra positional argument still binds to a defaulted trailing
        # parameter, so a successful bind alone proves nothing.
        bound = signature.bind(*proc.kwargs["args"]).arguments
        assert bound["server_args"] is server_args
        assert bound["port_args"] is port_args
        assert bound["gpu_id"] == gpu_id
        assert bound["tp_rank"] == tp_rank
        assert bound["pp_rank"] == pp_rank
        assert bound["dp_rank"] is None
        assert bound["pipe_writer"] is writers[index]
        assert not {"display_tp_rank", "display_dp_rank", "display_moe_ep_rank"} & bound.keys()
    # smg-lab #292: the parent's copy of each ready pipe's write end is closed
    # once the rank has it, so a rank that dies before reporting shows as EOF.
    assert all(writer.closed for writer in writers)


def test_a_scheduler_process_that_dies_before_its_ready_report_fails_the_launch(
    monkeypatch, scheduler_launcher_mod
):
    """smg-lab #292: the wait for the ready reports polls, so a scheduler
    process that exits during its start (the servicer kept running on a
    report that never came) ends the launch with a named error and every
    started process terminated."""

    class Reader:
        def poll(self, _timeout=None):
            return False

        def recv(self):
            raise AssertionError("nothing to receive from a dead rank")

    class Process:
        instances = []

        def __init__(self, **_kwargs):
            self.alive = False
            self.exitcode = None
            self.pid = 4242
            self.terminated = False
            self.join_timeouts = []
            self.instances.append(self)

        def start(self):
            self.alive = True

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminated = True
            self.alive = False

        def kill(self):
            self.alive = False

        def join(self, timeout=None):
            self.join_timeouts.append(timeout)

    monkeypatch.setattr(scheduler_launcher_mod, "READY_POLL_SECS", 0.0)
    monkeypatch.setattr(scheduler_launcher_mod.mp, "set_start_method", lambda *_args, **_kw: None)
    monkeypatch.setattr(
        scheduler_launcher_mod.mp,
        "Pipe",
        lambda **_kwargs: (Reader(), SimpleNamespace(close=lambda: None)),
    )
    monkeypatch.setattr(scheduler_launcher_mod.mp, "Process", Process)
    server_args = SimpleNamespace(
        resolve_once=lambda: None,
        check_server_args=lambda: None,
        dp_size=1,
        ep_join_mode=None,
        enable_memory_saver=False,
        nnodes=1,
        pp_size=1,
        tp_size=1,
        node_rank=0,
        base_gpu_id=0,
        gpu_id_step=1,
    )
    # The rank starts, then dies on its own before it reports.
    original_start = Process.start

    def start_then_die(self):
        original_start(self)
        self.alive = False
        self.exitcode = 1

    monkeypatch.setattr(Process, "start", start_then_die)

    with pytest.raises(RuntimeError, match="exited with code 1 before it was ready"):
        scheduler_launcher_mod.launch_scheduler_process_only(server_args, port_args=object())

    assert len(Process.instances) == 1
    assert Process.instances[0].terminated or not Process.instances[0].is_alive()
