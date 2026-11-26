import importlib.util
import multiprocessing
import queue
import sys
from dataclasses import replace
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "tau2_budget", Path(__file__).resolve().parents[1] / "tau2" / "budget.py"
)
budget = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = budget
SPEC.loader.exec_module(budget)
POLICY = budget.Policy("paid/model", "1", "1", 4000, 2, 1, 10, 10)


def reserve(ledger, request_id="one", arm="vllm", **kwargs):
    values = dict(
        model="paid/model",
        input_tokens=1,
        output_tokens=1,
        num_retries=0,
        output_limit_verified=True,
    )
    ledger.reserve(request_id, arm, **(values | kwargs))


@pytest.mark.parametrize(
    "change",
    [
        {"model": ""},
        {"input_per_million": "0"},
        {"output_per_million": "NaN"},
        {"input_per_million": "bad"},
        {"input_per_million": 1.0},
        {"cap_nusd": "4000"},
        {"calls": True},
        {"output_limit": -1},
    ],
)
def test_invalid_policy_is_rejected(change):
    with pytest.raises(budget.BudgetError):
        replace(POLICY, **change)


@pytest.mark.parametrize(
    "change",
    [
        {"model": "unknown/evaluator"},
        {"arm": "reviewer"},
        {"input_tokens": None},
        {"input_tokens": True},
        {"input_tokens": 11},
        {"output_tokens": 11},
        {"num_retries": 1},
        {"output_limit_verified": False},
    ],
)
def test_rejected_requests_never_reach_provider(tmp_path, change):
    ledger = budget.Ledger(tmp_path / "budget.db", POLICY)
    sent = []
    with pytest.raises(budget.BudgetError):
        reserve(ledger, **change)
        sent.append("sent")
    assert sent == [] and ledger.summary()["calls"] == 0


def test_omitted_retry_policy_cannot_reserve(tmp_path):
    ledger = budget.Ledger(tmp_path / "budget.db", POLICY)
    with pytest.raises(budget.BudgetError):
        ledger.reserve("one", "vllm", "paid/model", 1, 1, output_limit_verified=True)
    assert ledger.summary()["calls"] == 0


def test_first_request_above_cap_sends_nothing(tmp_path):
    ledger = budget.Ledger(tmp_path / "budget.db", replace(POLICY, cap_nusd=1999))
    with pytest.raises(budget.BudgetError):
        reserve(ledger)
        pytest.fail("provider must not be called")
    assert ledger.summary()["allocated_nusd"] == 0


def test_policy_identity_and_retries_survive_reopening(tmp_path):
    path = tmp_path / "budget.db"
    reserve(budget.Ledger(path, POLICY))
    with pytest.raises(budget.BudgetError):
        budget.Ledger(path, replace(POLICY, cap_nusd=8000))
    with pytest.raises(budget.BudgetError):
        reserve(budget.Ledger(path, POLICY))


def test_completed_calls_keep_reservations_and_call_limits(tmp_path):
    ledger = budget.Ledger(tmp_path / "budget.db", POLICY)
    reserve(ledger)
    ledger.finish("one", 0, 0)
    reserve(ledger, "two", "smg")
    with pytest.raises(budget.BudgetError):
        reserve(ledger, "third-vllm")
    with pytest.raises(budget.BudgetError):
        reserve(ledger, "three", "smg")
    ledger.finish("two", 0, 0)
    assert ledger.summary() == {"calls": 2, "allocated_nusd": 4000, "halted": True}
    assert ledger.audit()["pending"] == 0


def test_call_cap_is_shared_even_when_money_and_arm_allow_more(tmp_path):
    ledger = budget.Ledger(
        tmp_path / "budget.db", replace(POLICY, cap_nusd=8000, calls=1, calls_per_arm=2)
    )
    reserve(ledger)
    with pytest.raises(budget.BudgetError):
        reserve(ledger, "two", "smg")


def test_timeout_after_send_keeps_reservation_and_stops_next_arm(tmp_path):
    ledger = budget.Ledger(tmp_path / "budget.db", POLICY)
    sent = []

    def provider():
        sent.append("one")
        raise TimeoutError("response lost after send")

    reserve(ledger)
    with pytest.raises(TimeoutError):
        provider()
    with pytest.raises(budget.BudgetError):
        ledger.finish("one")
    with pytest.raises(budget.BudgetError):
        reserve(ledger, "two", "smg")
        sent.append("two")
    assert sent == ["one"] and ledger.summary()["allocated_nusd"] == 2000


@pytest.mark.parametrize("usage", [(None, None), (2, 1), (1, -1), (True, 1)])
def test_unknown_or_excess_usage_halts_both_arms_without_refund(tmp_path, usage):
    path = tmp_path / "budget.db"
    ledger = budget.Ledger(path, POLICY)
    reserve(ledger)
    with pytest.raises(budget.BudgetError):
        ledger.finish("one", *usage)
    reopened = budget.Ledger(path, POLICY)
    with pytest.raises(budget.BudgetError):
        reserve(reopened, "two", "smg")
    assert reopened.summary() == {"calls": 1, "allocated_nusd": 2000, "halted": True}


def test_fractional_prices_round_up_without_decimal_context_loss():
    p = replace(POLICY, input_per_million="0.0000001", output_per_million="0.0000001")
    assert p.cost(1, 1) == 2
    p = replace(POLICY, input_per_million="1.123456789123456789")
    assert p.cost(10**12, 0) == 1123456789123457


def contender(path, arm, start, sent):
    ledger = budget.Ledger(path, replace(POLICY, cap_nusd=2000))
    start.wait(timeout=5)
    try:
        reserve(ledger, arm, arm)
    except budget.BudgetError:
        return
    sent.put(arm)
    ledger.finish(arm, 1, 1)


def test_two_processes_share_the_last_dollar_slot(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    start, sent = ctx.Barrier(2), ctx.Queue()
    path = tmp_path / "budget.db"
    workers = [
        ctx.Process(target=contender, args=(path, arm, start, sent)) for arm in ("vllm", "smg")
    ]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(10)
            assert worker.exitcode == 0
        assert sent.get(timeout=1) in {"vllm", "smg"}
        with pytest.raises(queue.Empty):
            sent.get(timeout=0.1)
        assert budget.Ledger(path, replace(POLICY, cap_nusd=2000)).summary()["calls"] == 1
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()
        sent.close()


@pytest.mark.parametrize("phase", ["connect", "rollback", "close"])
def test_parent98_regression_sqlite_failure_stays_budget_error(monkeypatch, tmp_path, phase):
    primary = budget.sqlite3.OperationalError("primary SQLite failure")

    class BrokenDB:
        def execute(self, statement):
            if phase == "rollback":
                raise primary

        def commit(self):
            pass

        def rollback(self):
            raise budget.sqlite3.OperationalError("secondary rollback failure")

        def close(self):
            if phase == "close":
                raise primary

    def connect(*args, **kwargs):
        if phase == "connect":
            raise primary
        return BrokenDB()

    monkeypatch.setattr(budget.sqlite3, "connect", connect)
    ledger = object.__new__(budget.Ledger)
    ledger.path = tmp_path / "ledger.db"
    ledger.create = True
    with pytest.raises(budget.BudgetError) as error:
        with ledger._transaction():
            pass
    assert error.value.__cause__ is primary


def test_exhaustion_halts_inside_the_reservation_transaction(tmp_path):
    ledger = budget.Ledger(tmp_path / "atomic-cap.db", replace(POLICY, cap_nusd=3000))
    reserve(ledger)
    with pytest.raises(budget.BudgetError, match="exhausted"):
        reserve(ledger, "too-large", "smg")
    # A new connection after the refusal cannot use the remaining allowance.
    other = budget.Ledger(ledger.path, ledger.policy, create=False)
    assert other.summary()["halted"] is True
    with pytest.raises(budget.BudgetError, match="halted"):
        other.reserve("small", "smg", POLICY.model, 1, 1, num_retries=0, output_limit_verified=True)
    ledger.finish("one", 1, 1)  # Previously admitted calls can still settle.
    assert ledger.audit()["pending"] == 0


def test_halt_attempts_db_even_when_sidecar_fails(tmp_path, monkeypatch):
    ledger = budget.Ledger(tmp_path / "budget.db", POLICY)
    original = budget.os.open

    def sidecar_failure(path, flags, *args):
        if Path(path) == ledger.fail_stop_path:
            raise PermissionError("owned injected sidecar failure")
        return original(path, flags, *args)

    with monkeypatch.context() as patch:
        patch.setattr(budget.os, "open", sidecar_failure)
        with pytest.raises(budget.BudgetError, match="halt stores"):
            ledger.halt()
    assert not ledger.fail_stop_path.exists()
    assert budget.Ledger(ledger.path, POLICY, create=False).summary()["halted"]
    with pytest.raises(budget.BudgetError):
        reserve(ledger)


def test_halt_keeps_sidecar_when_sqlite_fails(tmp_path, monkeypatch):
    from contextlib import contextmanager

    ledger = budget.Ledger(tmp_path / "budget.db", POLICY)

    @contextmanager
    def unavailable():
        raise budget.BudgetError("owned injected DB failure")
        yield

    with monkeypatch.context() as patch:
        patch.setattr(ledger, "_transaction", unavailable)
        with pytest.raises(budget.BudgetError, match="halt stores"):
            ledger.halt()
    assert ledger.fail_stop_path.exists()
    with pytest.raises(budget.BudgetError):
        reserve(budget.Ledger(ledger.path, POLICY, create=False))


def _paused_reservation(path, ready, release, outcome):
    from contextlib import contextmanager

    ledger = budget.Ledger(path, POLICY, create=False)
    original = ledger._transaction

    @contextmanager
    def paused():
        with original() as db:
            yield db
            ready.set()
            assert release.wait(8)

    ledger._transaction = paused
    reserve(ledger)
    outcome.put("admitted")


def _publish_halt(ledger, entered):
    entered.set()
    ledger.halt()


def test_real_process_halt_and_commit_share_ordering_fence(tmp_path):
    import time

    path = tmp_path / "budget.db"
    ledger = budget.Ledger(path, POLICY)
    ctx = multiprocessing.get_context("fork")
    ready, release, entered, outcome = ctx.Event(), ctx.Event(), ctx.Event(), ctx.Queue()
    reserve_child = ctx.Process(target=_paused_reservation, args=(path, ready, release, outcome))
    halt_child = ctx.Process(target=_publish_halt, args=(ledger, entered))
    try:
        reserve_child.start()
        assert ready.wait(8)
        halt_child.start()
        assert entered.wait(8)
        time.sleep(0.1)
        # The prepublication reservation still owns the commit/admission fence.
        assert halt_child.is_alive() and not ledger.fail_stop_path.exists()
        release.set()
        reserve_child.join(8)
        halt_child.join(8)
        assert reserve_child.exitcode == halt_child.exitcode == 0
        assert outcome.get(timeout=1) == "admitted"
        assert ledger.fail_stop_path.exists()
        # Already admitted usage may settle after halt; a new paid call may not.
        ledger.finish("one", 0, 0)
        with pytest.raises(budget.BudgetError):
            reserve(ledger, "later", "smg")
    finally:
        release.set()
        for child in [reserve_child, halt_child]:
            if child.pid is not None:
                if child.is_alive():
                    child.terminate()
                child.join(8)
        outcome.close()


def test_protocol_v1_ledger_cannot_mix_with_new_admission(tmp_path):
    import json
    import sqlite3
    from dataclasses import asdict

    path = tmp_path / "old.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE policy (id INTEGER PRIMARY KEY, config TEXT, halted INTEGER)")
        db.execute(
            "INSERT INTO policy VALUES (1, ?, 0)", (json.dumps(asdict(POLICY), sort_keys=True),)
        )
    with pytest.raises(budget.BudgetError, match="different policy"):
        budget.Ledger(path, POLICY, create=False)
