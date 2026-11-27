"""CPU-only unit tests for external worker setup_backend cache flushing and pool clustering."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
from fixtures import hooks
from infra.constants import DEFAULT_MODEL, ENV_MODEL, ConnectionMode
from infra.worker import Worker

setup = importlib.import_module("fixtures.setup_backend")


class _FakeGateway:
    """Minimal stand-in for ``infra.gateway.Gateway`` tracking lifecycle calls."""

    def __init__(self) -> None:
        self.base_url = "http://127.0.0.1:39999"
        self.started_kwargs: dict | None = None
        self.flushed = 0
        self.shut_down = 0

    def start(self, **kwargs) -> None:
        """Record start arguments."""
        self.started_kwargs = kwargs

    def flush_cache(self, timeout: float = 30.0) -> bool:
        """Record a cache flush call."""
        self.flushed += 1
        return True

    def shutdown(self) -> None:
        """Record shutdown."""
        self.shut_down += 1


class _SortItem:
    """Minimal stand-in for a pytest item exercised against ``hooks._pool_sort_key``."""

    cls = None

    def __init__(
        self,
        nodeid: str,
        *,
        setup_backend: str | None = None,
        model: str | None = None,
        workers: dict | None = None,
    ) -> None:
        self.nodeid = nodeid
        self.callspec = (
            SimpleNamespace(params={"setup_backend": setup_backend})
            if setup_backend is not None
            else None
        )
        self._markers: dict = {}
        if model is not None:
            self._markers["model"] = pytest.mark.model(model).mark
        if workers is not None:
            self._markers["workers"] = pytest.mark.workers(**workers).mark

    def get_closest_marker(self, name: str):
        """Return marker by name."""
        return self._markers.get(name)


@pytest.mark.parametrize(
    ("external", "flush_cache", "expected_flushes"),
    [
        (True, True, 1),
        (True, False, 0),
        (False, False, 0),
    ],
)
def test_setup_local_flushes_cache_only_for_opt_in_external_workers(
    monkeypatch: pytest.MonkeyPatch,
    external: bool,
    flush_cache: bool,
    expected_flushes: int,
) -> None:
    """``_setup_local`` flushes KV cache after gateway start only for external workers that request it."""
    worker = Worker(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        engine="vllm",
        port=30000,
        gpu_ids=[],
        mode=ConnectionMode.GRPC,
        host="10.0.0.5" if external else "127.0.0.1",
        external=external,
        external_url="grpc://10.0.0.5:30000" if external else None,
        flush_cache=flush_cache,
    )
    stopped: list[object] = []
    gateway = _FakeGateway()

    monkeypatch.setattr(setup, "_start_workers_tracked", lambda **kwargs: [worker])
    monkeypatch.setattr(setup, "stop_workers", lambda ws: stopped.extend(ws))
    monkeypatch.setattr(setup, "_make_openai_client", lambda gw: "client")

    gen = setup._setup_local(
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "vllm",
        ConnectionMode.GRPC,
        {"count": 1, "gpus": None, "extra_engine_args": None},
        dict(setup._GW_DEFAULTS),
        gateway,
        "grpc",
        None,
    )
    backend_name, model_path, client, gw = next(gen)
    assert (backend_name, model_path, client, gw) == (
        "grpc",
        "meta-llama/Llama-3.1-8B-Instruct",
        "client",
        gateway,
    )
    assert gateway.started_kwargs is not None
    assert gateway.started_kwargs["worker_urls"] == [worker.base_url]
    assert gateway.flushed == expected_flushes

    with pytest.raises(StopIteration):
        next(gen)
    assert gateway.shut_down == 1
    assert stopped == []


def test_backend_router_flushes_cache_for_external_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``backend_router`` flushes KV cache on attach and does not stop external workers."""
    worker = Worker(
        model_id=DEFAULT_MODEL,
        engine="vllm",
        port=30001,
        gpu_ids=[],
        mode=ConnectionMode.GRPC,
        host="10.0.0.5",
        external=True,
        external_url="grpc://10.0.0.5:30001",
        flush_cache=True,
    )
    gateway = _FakeGateway()
    stopped: list[object] = []

    monkeypatch.delenv("E2E_CONNECTION_MODE", raising=False)
    monkeypatch.setattr(setup, "get_runtime", lambda: "vllm")
    monkeypatch.setattr(setup, "get_model_spec", lambda model_id: {"model": model_id})
    monkeypatch.setattr(
        setup, "get_pool", lambda: SimpleNamespace(acquire=lambda **kwargs: [worker])
    )
    monkeypatch.setattr(setup, "Gateway", lambda: gateway)
    monkeypatch.setattr(setup, "stop_workers", lambda ws: stopped.extend(ws))

    request = SimpleNamespace(param="grpc")
    gen = setup.backend_router.__wrapped__(request)
    assert next(gen) is gateway
    assert gateway.flushed == 1
    with pytest.raises(StopIteration):
        next(gen)
    assert gateway.shut_down == 1
    assert stopped == []


def test_pool_sort_key_clusters_unmarked_default_model_with_explicit_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unmarked local classes resolve to DEFAULT_MODEL so another model does not split them."""
    monkeypatch.delenv("E2E_CONNECTION_MODE", raising=False)
    monkeypatch.delenv(ENV_MODEL, raising=False)
    unmarked = _SortItem(
        "router/test_admin_ops.py::TestAdminOps::test_a[grpc]", setup_backend="grpc"
    )
    other = _SortItem(
        "chat/test_qwen.py::TestQwen::test_b[grpc]",
        setup_backend="grpc",
        model="Qwen/Qwen2.5-7B-Instruct",
    )
    explicit = _SortItem(
        "router/test_loads.py::TestLoads::test_c[grpc]",
        setup_backend="grpc",
        model=DEFAULT_MODEL,
    )
    ordered = sorted([unmarked, other, explicit], key=hooks._pool_sort_key)
    assert abs(ordered.index(unmarked) - ordered.index(explicit)) == 1


def test_pool_sort_key_keeps_custom_worker_topology_from_splitting_default_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Classes with ``extra_engine_args`` cluster separately instead of evicting default workers."""
    monkeypatch.delenv("E2E_CONNECTION_MODE", raising=False)
    default_a = _SortItem("a.py::TestA::test_1[grpc]", setup_backend="grpc", model=DEFAULT_MODEL)
    custom_b = _SortItem(
        "b.py::TestB::test_2[grpc]",
        setup_backend="grpc",
        model=DEFAULT_MODEL,
        workers={"extra_engine_args": ["--max-model-len", "2048"]},
    )
    default_c = _SortItem("c.py::TestC::test_3[grpc]", setup_backend="grpc", model=DEFAULT_MODEL)
    warmup = _SortItem(
        "chat/test_servicer_warmup.py::TestFullWarmupBeforeServing::test_full",
        model="meta-llama/Llama-3.2-1B-Instruct",
    )
    ordered = sorted([default_c, custom_b, default_a, warmup], key=hooks._pool_sort_key)
    assert ordered[0] is warmup
    assert abs(ordered.index(default_a) - ordered.index(default_c)) == 1
