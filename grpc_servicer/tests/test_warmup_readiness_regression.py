"""Warm-up readiness regressions against the project's current modules.

The failure case originally reproduced unsafe readiness on 14f7; both cases
now guard the repair and its healthy partial-budget behavior. Only the clock
and wire/server/task dependencies are fake. Neither tested function is
extracted, copied, replaced, wrapped, or marked xfail. These CPU tests make
no claim about historical runtime identity or the underlying GPU stall.
"""

import json
from types import SimpleNamespace

from smg_grpc_servicer import rust_lifecycle as lifecycle
from smg_grpc_servicer import warmup


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.jump_after_read = None

    def monotonic(self):
        value = self.now
        if self.jump_after_read is not None:
            self.now = self.jump_after_read
            self.jump_after_read = None
        return value


class FakeWire:
    def __init__(self, clock, *, fail_second):
        self.clock = clock
        self.fail_second = fail_second
        self.calls = []
        self.pending_backend = set()
        self.flush_calls = 0
        self.closed = False

    def generate(self, request, timeout):
        self.calls.append((request.request_id, timeout))
        self.pending_backend.add(request.request_id)
        if self.fail_second and len(self.calls) == 2:
            self.clock.now += timeout + 1.0
            raise TimeoutError(
                "fake Generate DEADLINE_EXCEEDED; backend cancellation unacknowledged"
            )
        self.clock.now += 1.0
        self.pending_backend.remove(request.request_id)
        if not self.fail_second:
            # This successful call completed within its deadline. Time passes
            # between its elapsed-time sample and the next step's budget check.
            self.clock.jump_after_read = 11.0
        return object()

    def flush_cache(self):
        self.flush_calls += 1
        if self.fail_second:
            raise TimeoutError("fake FlushCache deadline; no backend drain ACK")
        return True

    def close(self):
        # No sockets, threads or backend processes exist in this fake.
        # In particular, close does not invent a backend cancellation ACK.
        self.closed = True


class FakeServer:
    def __init__(self):
        self.ready = None
        self.events = []

    def set_ready(self, ready):
        self.ready = ready
        self.events.append(ready)


class FakeCompletedTask:
    def __init__(self, report):
        self.report = report

    def result(self):
        return self.report


def make_plan():
    inputs = warmup.WarmupInputs(
        max_num_seqs=1,
        max_num_batched_tokens=8,
        max_model_len=16,
        vocab_size=32,
        source="CPU fake-wire policy reproducer",
    )
    requests = tuple(warmup.WarmupRequest(f"smg-warmup-fake-{n}", 8, 1, n, 32) for n in range(3))
    steps = tuple(
        warmup.WarmupStep(f"fake-prefill-{n}", "prefill", (request,))
        for n, request in enumerate(requests)
    )
    return warmup.WarmupPlan("full", inputs, steps)


def run_case(monkeypatch, *, fail_second):
    clock = FakeClock()
    # Replace this module's external clock reference, not time.monotonic
    # globally, and leave run_warmup/_Pass/_flush entirely unchanged.
    monkeypatch.setattr(warmup, "time", SimpleNamespace(monotonic=clock.monotonic))
    wire = FakeWire(clock, fail_second=fail_second)
    server = FakeServer()
    lifecycle.hold_readiness(server, False)
    try:
        report = warmup.run_warmup(make_plan(), wire, budget_secs=10.0)
        lifecycle._warmup_over(server, FakeCompletedTask(report))
        print(
            "FAKEWIRE_TRACE "
            + json.dumps(
                {
                    "scenario": "timeout_and_failed_flush"
                    if fail_second
                    else "healthy_partial_budget",
                    "covered": report.covered,
                    "errors": report.errors,
                    "budget_hit": report.budget_hit,
                    "flushed": report.flushed,
                    "flush_refusal": report.flush_refusal,
                    "generate_calls": wire.calls,
                    "pending_backend": sorted(wire.pending_backend),
                    "readiness_events": server.events,
                    "wire_resources": "no real network, model, threads or backend",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return report, wire, server
    finally:
        wire.close()


def test_timeout_and_failed_flush_do_not_release_readiness(monkeypatch):
    report, wire, server = run_case(monkeypatch, fail_second=True)
    assert report.covered == 1
    assert report.errors == 1
    assert report.budget_hit is True
    assert report.flushed is False
    assert "FlushCache deadline" in report.flush_refusal
    assert len(wire.calls) == 2
    assert wire.flush_calls == 1
    assert wire.pending_backend == {"smg-warmup-fake-1"}
    assert wire.closed is True
    # This assertion failed on the original implementation and must pass
    # after the repair. Keep the safety contract, including the positive case.
    assert server.ready is False, (
        "readiness was released after Generate deadline + FlushCache timeout "
        "while the fake backend request remained unresolved and no recovery ran"
    )


def test_completed_partial_budget_releases_readiness(monkeypatch):
    report, wire, server = run_case(monkeypatch, fail_second=False)
    assert report.covered == 1 < len(report.plan.steps)
    assert report.errors == 0
    assert report.budget_hit is True
    assert report.flushed is True
    assert report.flush_refusal is None
    assert len(wire.calls) == 1
    assert wire.flush_calls == 1
    assert wire.pending_backend == set()
    assert wire.closed is True
    assert server.ready is True
    assert server.events == [False, True]
