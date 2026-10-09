"""CPU-only checks for reuse opt-in, ordering and class failure invalidation."""

import importlib
from types import SimpleNamespace

import pytest
from fixtures import hooks
from fixtures import setup_backend as setup_backend_fixture

# Import the module rather than fixtures.__init__'s exported fixture function.
setup = importlib.import_module("fixtures.setup_backend")
pytest_plugins = ["pytester"]


class _Item:
    cls = None

    def __init__(self, name, *, reuse=True, parser="llama", gpus=2):
        self.nodeid = f"test.py::{name}::test_case[grpc]"
        self.callspec = SimpleNamespace(params={"setup_backend": "grpc"})
        self.markers = {
            "model": pytest.mark.model("meta-llama/Llama-3.2-1B-Instruct").mark,
            "gateway": pytest.mark.gateway(
                reuse=reuse, extra_args=["--tool-call-parser", parser]
            ).mark,
            "workers": pytest.mark.workers(gpus=gpus).mark,
        }

    def get_closest_marker(self, name):
        return self.markers.get(name)


def test_collection_groups_complete_config_and_preserves_items(monkeypatch):
    monkeypatch.setenv("E2E_CONNECTION_MODE", "zmq")
    first = _Item("A", parser="llama")
    different = _Item("B", parser="pythonic")
    matching = _Item("C", parser="llama")
    topology = _Item("D", parser="llama", gpus=1)
    items = [first, different, matching, topology]
    ordered = sorted(items, key=hooks._pool_sort_key)
    assert set(ordered) == set(items)
    assert abs(ordered.index(first) - ordered.index(matching)) == 1


def test_non_zmq_order_is_unchanged(monkeypatch):
    monkeypatch.delenv("E2E_CONNECTION_MODE", raising=False)
    first = _Item("A", parser="pythonic")
    second = _Item("B", parser="llama")
    assert sorted([second, first], key=hooks._pool_sort_key) == [first, second]


@pytest.fixture
def pooled_setup(monkeypatch):
    gateway = SimpleNamespace(base_url="http://gateway")
    acquired, discarded = [], []
    pool = SimpleNamespace(
        acquire_zmq=lambda **kw: acquired.append(kw) or gateway,
        discard_zmq=discarded.append,
    )
    monkeypatch.setattr(setup, "get_pool", lambda: pool)
    monkeypatch.setattr(setup, "_make_openai_client", lambda gw: "client")
    monkeypatch.setattr(setup, "_gateway_readiness_timeout", lambda *args: 600)
    session = SimpleNamespace(testsfailed=0)

    def start():
        return setup._setup_pooled_zmq(
            session,
            "model-id",
            "model-path",
            "vllm",
            {"count": 1, "gpus": 2, "extra_engine_args": ["--data-parallel-size", "2"]},
            {**setup._GW_DEFAULTS, "reuse": True},
            "grpc",
            None,
        )

    return start, session, gateway, acquired, discarded


def test_successful_class_retains_pair_and_preserves_topology(pooled_setup):
    start, _, gateway, acquired, discarded = pooled_setup
    fixture = start()
    assert next(fixture) == ("grpc", "model-path", "client", gateway)
    with pytest.raises(StopIteration):
        next(fixture)
    assert discarded == []
    assert acquired[0]["gpus"] == 2
    assert acquired[0]["extra_engine_args"] == ["--data-parallel-size", "2"]
    assert acquired[0]["gateway_config"]["timeout"] == 600


def test_failed_class_releases_pair_at_teardown(pooled_setup):
    start, session, gateway, _, discarded = pooled_setup
    fixture = start()
    next(fixture)
    session.testsfailed += 1
    assert discarded == []
    with pytest.raises(StopIteration):
        next(fixture)
    assert discarded == [gateway]


def test_real_pytest_class_failure_invalidates_before_next_class(pytester, pooled_setup):
    _, _, gateway, acquired, discarded = pooled_setup
    pytester.makeconftest(
        """
        import pytest
        from fixtures.setup_backend import _setup_pooled_zmq, _GW_DEFAULTS

        @pytest.fixture(scope="class")
        def serving(request):
            yield from _setup_pooled_zmq(
                request.session, "model-id", "model-path", "vllm",
                {"count": 1, "gpus": 2}, _GW_DEFAULTS, "grpc", None,
            )
        """
    )
    pytester.makepyfile(
        """
        class TestA:
            def test_pass(self, serving):
                assert serving[0] == "grpc"

        class TestB:
            def test_fail(self, serving):
                assert False, "intentional failure"

        class TestC:
            def test_after_failure(self, serving):
                assert serving[0] == "grpc"
        """
    )
    result = pytester.runpytest("-q")
    result.assert_outcomes(passed=2, failed=1)
    assert len(acquired) == 3
    assert discarded == [gateway]


@pytest.mark.parametrize("reuse", [False, True])
def test_only_opted_in_zmq_classes_use_paired_pool(monkeypatch, reuse):
    item = _Item("Class", reuse=reuse)
    request = SimpleNamespace(param="grpc", node=item, session=SimpleNamespace(testsfailed=0))
    monkeypatch.setenv("E2E_CONNECTION_MODE", "zmq")
    monkeypatch.delenv("E2E_SKIP_BACKEND_SETUP", raising=False)
    monkeypatch.setattr(setup, "Gateway", lambda: SimpleNamespace(shutdown=lambda: None))
    paths = []

    def local(*args):
        paths.append("fresh")
        yield "local"

    def pooled(*args):
        paths.append("pooled")
        yield "pooled"

    monkeypatch.setattr(setup, "_setup_local", local)
    monkeypatch.setattr(setup, "_setup_pooled_zmq", pooled)
    fixture = setup_backend_fixture.__wrapped__(request)
    next(fixture)
    fixture.close()
    assert paths == ["pooled" if reuse else "fresh"]
