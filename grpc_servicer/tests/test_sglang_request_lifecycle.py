import asyncio
import importlib.util
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

_MODULE_PATH = Path(__file__).parents[1] / "smg_grpc_servicer" / "sglang" / "request_lifecycle.py"
_SPEC = importlib.util.spec_from_file_location("sglang_request_lifecycle", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
request_lifecycle = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(request_lifecycle)
RequestLifecycle = request_lifecycle.RequestLifecycle


@dataclass
class FakeRequestState:
    request_id: str = "request"
    finished: bool = False
    stream_finished: bool = False


def test_active_request_is_claimed_and_marked_aborted():
    state = FakeRequestState()
    lifecycle = RequestLifecycle({state.request_id: state})

    assert lifecycle.claim_abort(state.request_id) is state
    assert state.finished is True
    assert state.stream_finished is True


def test_unknown_request_is_not_claimed():
    lifecycle = RequestLifecycle({})

    async def fail_if_forwarded():
        raise AssertionError("unknown request was forwarded")

    assert asyncio.run(lifecycle.abort_if_active("unknown", fail_if_forwarded)) is False


def test_completed_request_is_not_claimed_or_mutated():
    state = FakeRequestState(finished=True)
    lifecycle = RequestLifecycle({state.request_id: state})

    async def fail_if_forwarded():
        raise AssertionError("completed request was forwarded")

    result = asyncio.run(lifecycle.abort_if_active(state.request_id, fail_if_forwarded))

    assert result is False
    assert state.stream_finished is False


def test_duplicate_abort_is_claimed_only_once():
    state = FakeRequestState()
    lifecycle = RequestLifecycle({state.request_id: state})

    assert lifecycle.claim_abort(state.request_id) is state
    assert lifecycle.claim_abort(state.request_id) is None


def test_concurrent_active_aborts_are_forwarded_only_once():
    state = FakeRequestState()
    lifecycle = RequestLifecycle({state.request_id: state})
    forwarded = []

    async def run_aborts():
        async def forward():
            forwarded.append(state.request_id)
            await asyncio.sleep(0)

        return await asyncio.gather(
            *(lifecycle.abort_if_active(state.request_id, forward) for _ in range(8))
        )

    results = asyncio.run(run_aborts())

    assert results.count(True) == 1
    assert results.count(False) == 7
    assert forwarded == [state.request_id]


def test_forward_failure_releases_abort_claim_for_retry():
    state = FakeRequestState()
    lifecycle = RequestLifecycle({state.request_id: state})
    forwarded = []

    async def run_aborts():
        async def fail():
            raise RuntimeError("scheduler send failed")

        try:
            await lifecycle.abort_if_active(state.request_id, fail)
        except RuntimeError:
            pass
        else:
            raise AssertionError("forward failure was not propagated")

        assert state.finished is False
        assert state.stream_finished is False

        async def succeed():
            forwarded.append(state.request_id)

        return await lifecycle.abort_if_active(state.request_id, succeed)

    assert asyncio.run(run_aborts()) is True
    assert forwarded == [state.request_id]


def test_completed_request_ignores_scheduler_abort():
    state = FakeRequestState(finished=True)
    lifecycle = RequestLifecycle({state.request_id: state})

    assert lifecycle.accept_scheduler_abort(state.request_id) is None
    assert state.stream_finished is False


def test_local_abort_accepts_one_scheduler_ack():
    state = FakeRequestState()
    lifecycle = RequestLifecycle({state.request_id: state})

    assert lifecycle.claim_abort(state.request_id) is state
    assert lifecycle.accept_scheduler_abort(state.request_id) is state
    assert lifecycle.accept_scheduler_abort(state.request_id) is None


def test_scheduler_can_abort_an_active_request():
    state = FakeRequestState()
    lifecycle = RequestLifecycle({state.request_id: state})

    assert lifecycle.accept_scheduler_abort(state.request_id) is state
    assert state.finished is True
    assert state.stream_finished is True


def test_scheduler_abort_ack_targets_original_claim_after_id_reuse():
    states = {}
    lifecycle = RequestLifecycle(states)
    old_state = FakeRequestState()
    new_state = FakeRequestState()

    lifecycle.register(old_state)
    assert lifecycle.claim_abort(old_state.request_id) is old_state
    lifecycle.register(new_state)

    assert lifecycle.accept_scheduler_abort(old_state.request_id) is old_state
    assert new_state.finished is False
    assert new_state.stream_finished is False


def test_scheduler_abort_ack_survives_cleanup_before_id_reuse():
    states = {}
    lifecycle = RequestLifecycle(states)
    old_state = FakeRequestState()
    new_state = FakeRequestState()

    lifecycle.register(old_state)
    assert lifecycle.claim_abort(old_state.request_id) is old_state
    assert lifecycle.remove(old_state.request_id, old_state) is old_state
    lifecycle.register(new_state)

    assert lifecycle.accept_scheduler_abort(old_state.request_id) is old_state
    assert new_state.finished is False
    assert new_state.stream_finished is False


def test_concurrent_aborts_are_claimed_only_once():
    state = FakeRequestState()
    lifecycle = RequestLifecycle({state.request_id: state})
    barrier = threading.Barrier(8)

    def claim_abort():
        barrier.wait()
        return lifecycle.claim_abort(state.request_id)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: claim_abort(), range(8)))

    assert results.count(state) == 1
    assert results.count(None) == 7


def test_completion_and_abort_are_mutually_exclusive():
    for _ in range(20):
        state = FakeRequestState()
        lifecycle = RequestLifecycle({state.request_id: state})
        barrier = threading.Barrier(2)

        def claim_abort():
            barrier.wait()
            return lifecycle.claim_abort(state.request_id) is state

        def finish():
            barrier.wait()
            return lifecycle.finish(state.request_id, state, stream_finished=True)

        with ThreadPoolExecutor(max_workers=2) as executor:
            abort_result = executor.submit(claim_abort)
            finish_result = executor.submit(finish)

        assert abort_result.result() != finish_result.result()
        assert state.finished is True
        assert state.stream_finished is True


def test_stale_cleanup_does_not_remove_reused_request_id():
    states = {}
    lifecycle = RequestLifecycle(states)
    old_state = FakeRequestState()
    new_state = FakeRequestState()

    lifecycle.register(old_state)
    lifecycle.register(new_state)

    assert lifecycle.remove(old_state.request_id, old_state) is None
    assert lifecycle.get(new_state.request_id) is new_state
