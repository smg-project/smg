"""The PD sweep's idle check judges a load report taken after the clients left.

``/loads`` serves the gateway's latest load report, refreshed once per
load-monitor interval and published only once the tick's whole poll has
answered, which can take the gateway's poll deadline. Read the moment the
clients leave, it can be the report taken before their requests arrived, and
say idle about engines that still hold the abandoned work. The check waits
until the first report taken after the clients left has reached ``/loads`` at
the latest (the poll under way when they left running to its deadline, the next
tick an interval later, its poll at the deadline again) before it judges, and
its deadline still counts from the moment the clients left. The MoRI-IO lane
reads the engines' own metrics, which are live, and does not wait.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field

import pytest
from infra import constants


def _import_sweep(monkeypatch, runtime: str, kv_backend: str | None):
    monkeypatch.setattr(constants, "_RUNTIME_CACHE", None)
    monkeypatch.setenv(constants.ENV_RUNTIME, runtime)
    monkeypatch.delenv(constants.ENV_KV_BACKEND, raising=False)
    if kv_backend is None:
        monkeypatch.delenv(constants.ENV_VLLM_KV_BACKEND, raising=False)
    else:
        monkeypatch.setenv(constants.ENV_VLLM_KV_BACKEND, kv_backend)
    return importlib.import_module("router.test_pd_topologies")


@pytest.fixture
def sweep(monkeypatch):
    """The sweep on a lane whose idle check reads ``/loads``."""
    return _import_sweep(monkeypatch, "tokenspeed", None)


@pytest.fixture
def moriio_sweep(monkeypatch):
    """The sweep on the MoRI-IO lane, whose idle check reads the engines' metrics."""
    return _import_sweep(monkeypatch, "vllm", "moriio")


class _Clock:
    """A monotonic clock that advances only when the check sleeps."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class _Worker:
    base_url: str
    http_url: str


class _Gateway:
    base_url = "http://gateway"
    prefill_workers = [_Worker("grpc://127.0.0.1:30001", "http://127.0.0.1:31001")]
    decode_workers = [
        _Worker("grpc://127.0.0.1:30002", "http://127.0.0.1:31002"),
        _Worker("grpc://127.0.0.1:30003", "http://127.0.0.1:31003"),
    ]


_WORKERS = _Gateway.prefill_workers + _Gateway.decode_workers
# The gateway's LOAD_POLL_DEADLINE (worker/monitor.rs): a tick's report reaches /loads up to
# this long after the tick. A test-side fact, so the suite pins the sweep's wait to the
# gateway's bound instead of reading the wait back.
_POLL_DEADLINE_S = 3.0
# A hair inside a bound: the latest a report can still appear.
_LATE = 0.05
_IDLE = {w.base_url: 0 for w in _WORKERS}
_HELD = {_WORKERS[0].base_url: 8, _WORKERS[1].base_url: 4, _WORKERS[2].base_url: 4}


@dataclass(frozen=True)
class _Report:
    """A load report: the engines' running counts ``taken`` seconds after the clients left,
    on ``/loads`` from ``published`` seconds after they left (the tick's poll may take the
    gateway's whole deadline before the report is published)."""

    taken: float
    counts: dict[str, int]
    published: float = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.published is None:
            object.__setattr__(self, "published", self.taken)


class _Response:
    status_code = 200

    def __init__(self, loads: list[dict] | None = None, text: str = "") -> None:
        self._loads = loads or []
        self.text = text

    def json(self) -> dict:
        return {"loads": self._loads}

    def raise_for_status(self) -> None:
        pass


class _Engines:
    """The engines' running counts over time, as ``/loads`` reports them and as their metrics show them.

    ``/loads`` serves the latest report published by now; a worker's ``/metrics``
    serves its count now (the latest report taken by now). Every ``/loads`` read
    records when the report it served was taken, so a test can tell which
    reports the check judged.
    """

    def __init__(self, clock: _Clock, left: float, reports: list[_Report]) -> None:
        self._clock = clock
        self._left = left
        self._reports = reports
        self.judged: list[float] = []

    def _latest(self, by: str) -> _Report:
        now = self._clock.now - self._left
        return max((r for r in self._reports if getattr(r, by) <= now), key=lambda r: r.taken)

    def get(self, url: str, timeout: float) -> _Response:
        if url == f"{_Gateway.base_url}/loads":
            report = self._latest("published")
            self.judged.append(report.taken)
            return _Response(
                loads=[
                    {"worker": worker, "num_running_reqs": running, "num_waiting_reqs": 0}
                    for worker, running in report.counts.items()
                ]
            )
        worker = next(w for w in _WORKERS if url == f"{w.http_url}/metrics")
        running = self._latest("taken").counts[worker.base_url]
        return _Response(text=f"vllm:num_requests_running {running}\nvllm:num_requests_waiting 0\n")


class _Bed:
    """The idle check on a fake clock over a timeline of reports."""

    def __init__(self, sweep, monkeypatch, reports: list[_Report]) -> None:
        self._sweep = sweep
        self.clock = _Clock()
        self.left = self.clock.now
        self.engines = _Engines(self.clock, self.left, reports)
        monkeypatch.setattr(sweep, "time", self.clock)
        monkeypatch.setattr(sweep, "httpx", self.engines)

    def check(self, timeout: float = 30.0) -> None:
        self._sweep._assert_fleet_idle_within(_Gateway(), timeout)

    @property
    def elapsed(self) -> float:
        return self.clock.now - self.left


def _before_departure(sweep) -> list[_Report]:
    """The reports from before the clients left, each at the latest it can reach ``/loads``.

    The last poll that ended before they left (published as the next tick
    started), and the poll under way when they left: its samples predate their
    requests by a hair and read idle, and with one engine slow it is published
    the gateway's whole deadline later, after the clients left.
    """
    interval = sweep._LOAD_MONITOR_INTERVAL_S
    return [
        _Report(taken=-_LATE - interval, counts=_IDLE, published=-_LATE),
        _Report(taken=-_LATE, counts=_IDLE, published=_POLL_DEADLINE_S - _LATE),
    ]


def _slow_ticks(sweep, counts_by_tick: list[dict[str, int]]) -> list[_Report]:
    """The reports of the ticks after the clients left, at the latest they can appear.

    Each tick starts an interval after the previous poll ended (the gateway's
    timer fires a missed tick at once, which is sooner) and every poll takes
    the whole deadline (the engines are busy with the abandoned prompts), so
    the first report taken after the clients left is published an interval and
    two deadlines after they left: the case the check's wait exists for.
    """
    interval = sweep._LOAD_MONITOR_INTERVAL_S
    reports: list[_Report] = []
    poll_ended = _before_departure(sweep)[-1].published
    for counts in counts_by_tick:
        taken = poll_ended + interval
        poll_ended = taken + _POLL_DEADLINE_S
        reports.append(_Report(taken=taken, counts=counts, published=poll_ended))
    return reports


def _held_throughout(sweep) -> list[_Report]:
    """Reports taken before the requests arrived, then one per tick showing them held."""
    return [*_before_departure(sweep), *_slow_ticks(sweep, [_HELD] * 20)]


def test_the_check_judges_a_report_taken_after_the_clients_left(sweep, monkeypatch):
    """A report that predates the abandoned requests reads idle and proves nothing,
    however late the poll that took it publishes."""
    fresh = _slow_ticks(sweep, [_HELD, _HELD, _IDLE])  # held, then the engines freed the pairs
    bed = _Bed(sweep, monkeypatch, [*_before_departure(sweep), *fresh])
    bed.check()
    judged = bed.engines.judged
    assert all(offset > 0 for offset in judged), (
        f"the check judged a report taken before the clients left: {judged}"
    )
    assert sorted(set(judged)) == [report.taken for report in fresh], (
        f"the check skipped a report taken after the clients left: {judged}"
    )


def test_the_first_read_lands_when_the_first_fresh_report_must_have(sweep, monkeypatch):
    """The wait is that bound and no longer: when the first poll taken after the clients
    left reads idle, the check returns the moment its report can be on ``/loads``."""
    fresh = _slow_ticks(sweep, [_IDLE])
    bed = _Bed(sweep, monkeypatch, [*_before_departure(sweep), *fresh])
    bed.check()
    assert bed.engines.judged == [fresh[0].taken], f"the check judged {bed.engines.judged}"
    assert bed.elapsed == sweep._LOAD_MONITOR_INTERVAL_S + 2 * _POLL_DEADLINE_S, (
        f"the check returned {bed.elapsed:.2f}s after the clients left"
    )


def test_held_work_fails_the_check_when_its_budget_ends(sweep, monkeypatch):
    """Engines that keep the abandoned work fail the check at its deadline, counted
    from the moment the clients left: the wait for a fresh report is spent inside
    the budget, not added to it."""
    bed = _Bed(sweep, monkeypatch, _held_throughout(sweep))
    with pytest.raises(AssertionError, match="still hold work after 30s"):
        bed.check(timeout=30.0)
    assert 30.0 <= bed.elapsed < 32.0, f"the check ran {bed.elapsed:.1f}s"


def test_the_moriio_lane_reads_the_engines_live_and_does_not_wait(moriio_sweep, monkeypatch):
    """Engine metrics are read live, so that lane has no stale report to wait out."""
    bed = _Bed(
        moriio_sweep,
        monkeypatch,
        [_Report(taken=0.0, counts=_HELD), _Report(taken=2.0, counts=_IDLE)],
    )
    bed.check()
    assert bed.engines.judged == [], "the MoRI-IO lane read /loads"
    assert bed.elapsed == 2.0, f"the check waited {bed.elapsed:.1f}s on live metrics"
