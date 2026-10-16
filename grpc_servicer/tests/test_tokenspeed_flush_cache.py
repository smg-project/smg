"""The TokenSpeed Python servicer's ``FlushCache`` outlives the engine's
retirement of a finished request.

The engine refuses a flush while any request still pins cached blocks, and it
retires a finished request a scheduler round after its last token was
streamed. A flush issued right after a completion returns (the gateway's
``POST /flush_cache`` asks for an immediate flush, ``timeout_s=0``) landed in
that gap and was reported as failed from a single attempt. The servicer now
repeats the non-mutating attempt until the engine accepts it, for the
request's ``timeout_s`` and never less than a short retirement grace."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import grpc
import pytest

pytest.importorskip("msgspec")
pytest.importorskip("zmq")
pytest.importorskip("numpy")
pytest.importorskip("smg_grpc_proto")
from smg_grpc_proto.generated import common_pb2  # noqa: E402


class _Context:
    """A servicer context whose ``abort`` is a test failure: these cases end
    in a response, never in a status."""

    async def abort(self, code, details):
        raise AssertionError(f"FlushCache aborted: {code} {details}")


def _servicer(tokenspeed_servicer_mod, outcomes):
    """A servicer whose engine answers ``flush_cache`` with ``outcomes`` in
    turn (the last one repeats); returns it with the list of attempts made."""
    cls = tokenspeed_servicer_mod.TokenSpeedSchedulerServicer
    servicer = cls.__new__(cls)
    attempts = []

    async def flush_cache():
        attempts.append(len(attempts) + 1)
        return SimpleNamespace(success=outcomes[min(len(attempts), len(outcomes)) - 1])

    servicer.async_llm = SimpleNamespace(flush_cache=flush_cache)
    return servicer, attempts


def _flush(servicer, timeout_s=0.0):
    request = common_pb2.FlushCacheRequest(timeout_s=timeout_s)
    return asyncio.run(servicer.FlushCache(request, _Context()))


def _fast(tokenspeed_servicer_mod, monkeypatch, *, grace_s):
    # Never-defined before the retry existed: ``raising=False`` lets the
    # fails-before run reach the assertions instead of erroring on setattr.
    monkeypatch.setattr(tokenspeed_servicer_mod, "FLUSH_RETRY_INTERVAL_S", 0.0, raising=False)
    monkeypatch.setattr(tokenspeed_servicer_mod, "FLUSH_RETIRE_GRACE_S", grace_s, raising=False)


def test_an_immediate_flush_waits_for_the_engine_to_retire_a_finished_request(
    tokenspeed_servicer_mod, monkeypatch
):
    """The gateway's shape: ``timeout_s=0`` right after a completion. The
    engine still pins the finished request's blocks on the first two attempts
    and accepts the third; the caller sees one successful flush."""
    _fast(tokenspeed_servicer_mod, monkeypatch, grace_s=5.0)
    servicer, attempts = _servicer(tokenspeed_servicer_mod, [False, False, True])

    response = _flush(servicer)

    assert response.success is True
    assert response.message == "Cache flushed successfully"
    assert attempts == [1, 2, 3]


def test_a_request_still_in_flight_fails_the_flush_once_the_grace_is_spent(
    tokenspeed_servicer_mod, monkeypatch
):
    """An engine that keeps refusing (a request really in flight) still fails
    the immediate flush, after the grace instead of at once."""
    _fast(tokenspeed_servicer_mod, monkeypatch, grace_s=0.02)
    servicer, attempts = _servicer(tokenspeed_servicer_mod, [False])

    response = _flush(servicer)

    assert response.success is False
    assert "refused" in response.message
    assert len(attempts) >= 2


def test_a_positive_timeout_extends_the_wait_beyond_the_grace(tokenspeed_servicer_mod, monkeypatch):
    """``timeout_s`` is the request's wait-for-idle budget: with no grace at
    all the servicer keeps trying for it and returns the engine's acceptance."""
    _fast(tokenspeed_servicer_mod, monkeypatch, grace_s=0.0)
    servicer, attempts = _servicer(tokenspeed_servicer_mod, [False, False, False, True])

    response = _flush(servicer, timeout_s=5.0)

    assert response.success is True
    assert attempts == [1, 2, 3, 4]


def test_the_engines_own_message_is_reported(tokenspeed_servicer_mod, monkeypatch):
    """A ``FlushCacheReqOutput`` that carries a message keeps it."""
    _fast(tokenspeed_servicer_mod, monkeypatch, grace_s=0.0)
    servicer, _ = _servicer(tokenspeed_servicer_mod, [True])

    async def flush_cache():
        return SimpleNamespace(success=True, message="cleared 12 pages")

    servicer.async_llm = SimpleNamespace(flush_cache=flush_cache)

    assert _flush(servicer).message == "cleared 12 pages"


class _Abort(Exception):
    pass


class _RecordingContext:
    """A servicer context that records ``abort`` and ends the handler the way
    grpc's does (``abort`` never returns)."""

    def __init__(self):
        self.aborts = []

    async def abort(self, code, details):
        self.aborts.append((code, details))
        raise _Abort()


@pytest.mark.parametrize("timeout_s", [-1.0, float("inf"), float("nan")])
def test_a_non_finite_or_negative_timeout_is_refused_before_any_attempt(
    tokenspeed_servicer_mod, monkeypatch, timeout_s
):
    """``timeout_s`` sizes the wait and the RPC bound; a value that cannot is
    an INVALID_ARGUMENT, as the vLLM servicer answers it."""
    _fast(tokenspeed_servicer_mod, monkeypatch, grace_s=0.0)
    servicer, attempts = _servicer(tokenspeed_servicer_mod, [True])
    context = _RecordingContext()

    with pytest.raises(_Abort):
        asyncio.run(servicer.FlushCache(common_pb2.FlushCacheRequest(timeout_s=timeout_s), context))

    assert [code for code, _ in context.aborts] == [grpc.StatusCode.INVALID_ARGUMENT]
    assert attempts == []


def test_the_whole_rpc_stays_within_the_scheduler_bound(tokenspeed_servicer_mod, monkeypatch):
    """The retry window never outlives the scheduler round-trip bound: an
    attempt that hangs is cut at what is left of it."""
    _fast(tokenspeed_servicer_mod, monkeypatch, grace_s=60.0)
    monkeypatch.setattr(tokenspeed_servicer_mod, "FLUSH_SCHEDULER_TIMEOUT_S", 0.05, raising=False)
    monkeypatch.setattr(tokenspeed_servicer_mod, "FLUSH_SCHEDULER_MARGIN_S", 0.05, raising=False)
    servicer, _ = _servicer(tokenspeed_servicer_mod, [False])

    async def flush_cache():
        await asyncio.Event().wait()

    servicer.async_llm = SimpleNamespace(flush_cache=flush_cache)
    context = _RecordingContext()

    started = time.monotonic()
    with pytest.raises(_Abort):
        asyncio.run(servicer.FlushCache(common_pb2.FlushCacheRequest(), context))

    assert time.monotonic() - started < 5.0
    assert [code for code, _ in context.aborts] == [grpc.StatusCode.DEADLINE_EXCEEDED]


def test_a_refusing_engine_is_cut_at_the_scheduler_bound_not_the_grace(
    tokenspeed_servicer_mod, monkeypatch
):
    """The attempts share one deadline: an engine that refuses every attempt
    is cut at the scheduler bound with ``DEADLINE_EXCEEDED`` while the grace
    would still allow attempts (a per-attempt timeout would retry for the whole
    grace and answer a refusal instead)."""
    _fast(tokenspeed_servicer_mod, monkeypatch, grace_s=3.0)
    monkeypatch.setattr(tokenspeed_servicer_mod, "FLUSH_SCHEDULER_TIMEOUT_S", 0.05, raising=False)
    monkeypatch.setattr(tokenspeed_servicer_mod, "FLUSH_SCHEDULER_MARGIN_S", 0.05, raising=False)
    servicer, attempts = _servicer(tokenspeed_servicer_mod, [False])
    context = _RecordingContext()

    started = time.monotonic()
    with pytest.raises(_Abort):
        asyncio.run(servicer.FlushCache(common_pb2.FlushCacheRequest(), context))

    assert time.monotonic() - started < 1.0
    assert len(attempts) >= 2
    assert [code for code, _ in context.aborts] == [grpc.StatusCode.DEADLINE_EXCEEDED]
