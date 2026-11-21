"""HTTP fault outcomes and fleet lifecycle controls, without a GPU/runtime."""

import ast
import importlib.util
import logging
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).with_name("http_pd_stream_fault.py")
spec = importlib.util.spec_from_file_location("http_pd_stream_fault", SCRIPT)
assert spec is not None and spec.loader is not None
fault = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fault)


def outcome(*, finish=None, done=False, error=None, exception=None, status=200):
    return SimpleNamespace(
        status=status,
        exception=exception,
        end=SimpleNamespace(
            finish_reason=finish,
            done=done,
            error=error,
        ),
    )


@pytest.mark.parametrize(
    "value,kind",
    [
        (outcome(finish="stop", done=True), "completed"),
        (outcome(finish="length", done=True), "completed"),
        (outcome(), "eof_interruption"),
        (outcome(finish="abort"), "engine_abort"),
        (outcome(error={"message": "engine died"}), "error_frame"),
        (
            outcome(exception="RemoteProtocolError('incomplete chunked read')"),
            "transport_interruption",
        ),
        (outcome(exception="ReadError('connection reset')"), "transport_interruption"),
    ],
)
def test_http_outcome_classification(value, kind):
    assert fault.classify_http_fault_outcome(value) == kind


@pytest.mark.parametrize(
    "value",
    [
        outcome(finish="stop"),
        outcome(done=True),
        outcome(finish="unexpected"),
        outcome(error={}),
        outcome(status=503),
        outcome(exception="ReadTimeout('timed out')"),
        outcome(exception="ValueError('invalid JSON')"),
    ],
)
def test_invalid_framing_or_timeout_is_not_fault_success(value):
    with pytest.raises(AssertionError):
        fault.classify_http_fault_outcome(value)


def test_complete_only_does_not_prove_fault_observation():
    with pytest.raises(AssertionError, match="all streams completed"):
        fault.assert_http_fault_outcomes([outcome(finish="stop", done=True)])
    assert fault.assert_http_fault_outcomes([outcome(), outcome(finish="length", done=True)]) == [
        "eof_interruption",
        "completed",
    ]


def test_deadline_rejects_live_thread():
    class Thread:
        def join(self, timeout):
            assert 0 <= timeout <= 0.01

        def is_alive(self):
            return True

    with pytest.raises(AssertionError, match="did not terminate"):
        fault.wait_for_fault_streams([Thread(), Thread()], timeout=0.01)


def _test_function(namespace):
    source = SCRIPT.parents[1] / "router" / "test_pd_topologies.py"
    tree = ast.parse(source.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TestHTTPMoRIDecodeStreamFault"
    )
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef))
    method.decorator_list = []
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[method.name]


@pytest.mark.parametrize("fail_survivor", [False, True])
def test_injected_fault_observation_and_owned_teardown(fail_survivor):
    """Execute the actual new test method against concurrent fake workers.

    Streams only end after the victim kill. Fresh requests must use the peer.
    Both a passing verdict and a failed survivor verdict stop the complete fleet.
    """
    import time

    killed = threading.Event()
    pairs, stopped, fresh = [], [], []

    class Worker:
        def __init__(self, url):
            self.base_url = url

        def kill_engine(self):
            killed.set()
            return [123]

    prefill, peer, victim = Worker("prefill"), Worker("peer"), Worker("victim")
    gateway = SimpleNamespace(
        prefill_workers=[prefill],
        decode_workers=[peer, victim],
        shutdown=lambda: stopped.append("gateway"),
    )

    def stream(gateway, model, *, max_tokens, first_token):
        with lock:
            target = peer if len(pairs) % 2 else victim
            pairs.append((prefill.base_url, target.base_url))
        first_token.set()
        assert killed.wait(3)
        return outcome() if target is victim else outcome(finish="stop", done=True)

    def ask(client, model, prompt, *, max_tokens):
        assert killed.is_set()
        fresh.append(prompt)
        pairs.append((prefill.base_url, peer.base_url))
        return {"content": "" if fail_survivor else "answer"}

    def status(gateway, url, status, *, timeout):
        assert killed.is_set() and url == victim.base_url and status == "unhealthy"

    lock = threading.Lock()
    namespace = dict(
        threading=threading,
        time=time,
        _StreamOutcome=SimpleNamespace,
        _stream_chat=stream,
        _pairs_logged=lambda: list(pairs),
        _wait_for_pairs=lambda count: list(pairs),
        _wait_for_status=status,
        _ask=ask,
        _answer_text=lambda response: response["content"],
        assert_http_fault_outcomes=fault.assert_http_fault_outcomes,
        wait_for_fault_streams=fault.wait_for_fault_streams,
        stop_workers=lambda workers: stopped.extend(workers),
        logger=logging.getLogger(__name__),
    )
    test = _test_function(namespace)
    if fail_survivor:
        with pytest.raises(AssertionError, match="empty answer"):
            test(None, ("pd_http", "model", None, gateway))
    else:
        test(None, ("pd_http", "model", None, gateway))
        assert len(fresh) == 6
    assert stopped == ["gateway", prefill, peer, victim]
