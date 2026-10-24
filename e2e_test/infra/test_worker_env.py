"""The worker environment's servicer-implementation switch, per engine."""

from __future__ import annotations

import pytest
from infra import worker as worker_module
from infra.constants import (
    ENV_SGLANG_SERVICER_IMPL,
    ENV_TOKENSPEED_SERVICER_IMPL,
    ENV_VLLM_SERVICER_IMPL,
    ConnectionMode,
)
from infra.worker import Worker

ENGINES = [
    ("vllm", ENV_VLLM_SERVICER_IMPL, "SMG_VLLM_SERVICER_IMPL", True),
    ("tokenspeed", ENV_TOKENSPEED_SERVICER_IMPL, "SMG_TOKENSPEED_SERVICER_IMPL", False),
    ("sglang", ENV_SGLANG_SERVICER_IMPL, "SMG_SGLANG_SERVICER_IMPL", False),
]


def _worker(engine: str, mode: ConnectionMode) -> Worker:
    return Worker(model_id="m", engine=engine, port=50111, gpu_ids=[0], mode=mode)


@pytest.mark.parametrize(("engine", "lane_setting", "variable", "needs_hook"), ENGINES)
def test_a_grpc_worker_carries_the_lanes_servicer_choice(
    monkeypatch, engine, lane_setting, variable, needs_hook
):
    hook_checks: list[bool] = []
    monkeypatch.setattr(
        worker_module, "_require_rust_servicer_hook", lambda: hook_checks.append(True)
    )
    monkeypatch.setenv(variable, "rust")  # inherited: the lane's choice wins over it
    monkeypatch.delenv(lane_setting, raising=False)
    # The harness's default is the Python servicer, spelled out: the servicer
    # package's own default is rust.
    assert _worker(engine, ConnectionMode.GRPC)._build_env()[variable] == "python"
    assert hook_checks == []

    monkeypatch.setenv(lane_setting, "rust")
    assert _worker(engine, ConnectionMode.GRPC)._build_env()[variable] == "rust"
    assert hook_checks == ([True] if needs_hook else [])


@pytest.mark.parametrize(("engine", "lane_setting", "variable", "_needs_hook"), ENGINES)
def test_a_worker_off_the_grpc_path_keeps_the_inherited_setting(
    monkeypatch, engine, lane_setting, variable, _needs_hook
):
    monkeypatch.setenv(variable, "rust")
    monkeypatch.delenv(lane_setting, raising=False)
    assert _worker(engine, ConnectionMode.HTTP)._build_env()[variable] == "rust"
