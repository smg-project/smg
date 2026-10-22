"""Shared pre-request reservations; amounts are integer billionths of a USD.

Callers must verify complete token bounds and provider limits before reserving.
This module is not yet connected to tau2's LLM request path.
"""

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation


class BudgetError(ValueError):
    pass


def _integer(value, minimum=1):
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        raise BudgetError("Expected a bounded integer")


def _rate(value):
    if not isinstance(value, str) or len(value) > 64:
        raise BudgetError("Pricing must be an explicit decimal string")
    try:
        rate = Decimal(value)
        if not rate.is_finite() or rate <= 0 or not -18 <= rate.adjusted() <= 12:
            raise BudgetError("Missing or invalid pricing")
        return rate.as_integer_ratio()
    except InvalidOperation as exc:
        raise BudgetError("Invalid pricing") from exc


@dataclass(frozen=True)
class Policy:
    model: str
    input_per_million: str
    output_per_million: str
    cap_nusd: int
    calls: int
    calls_per_arm: int
    input_limit: int
    output_limit: int

    def __post_init__(self):
        if not isinstance(self.model, str) or not self.model.strip():
            raise BudgetError("An approved model is required")
        _rate(self.input_per_million)
        _rate(self.output_per_million)
        for name in ("cap_nusd", "calls", "calls_per_arm", "input_limit", "output_limit"):
            _integer(getattr(self, name))

    def cost(self, input_tokens, output_tokens):
        _integer(input_tokens, 0)
        _integer(output_tokens, 0)
        amount = 0
        for tokens, price in (
            (input_tokens, self.input_per_million),
            (output_tokens, self.output_per_million),
        ):
            numerator, denominator = _rate(price)
            amount += (tokens * numerator * 1000 + denominator - 1) // denominator
        return amount


class Ledger:
    def __init__(self, path, policy):
        self.path, self.policy = path, policy
        self.config = json.dumps(asdict(policy), sort_keys=True)
        with self._transaction() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS policy "
                "(id INTEGER PRIMARY KEY CHECK(id=1), config TEXT, halted INTEGER)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS requests "
                "(id TEXT PRIMARY KEY, arm TEXT, amount INTEGER, "
                "input_cap INTEGER, output_cap INTEGER, state TEXT)"
            )
            db.execute("INSERT OR IGNORE INTO policy VALUES (1, ?, 0)", (self.config,))
            self._halted(db)

    @contextmanager
    def _transaction(self):
        db = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _halted(self, db):
        config, halted = db.execute("SELECT config, halted FROM policy WHERE id=1").fetchone()
        if config != self.config:
            raise BudgetError("The shared ledger has a different policy")
        return halted

    def reserve(
        self,
        request_id,
        arm,
        model,
        input_tokens,
        output_tokens,
        *,
        num_retries=None,
        output_limit_verified=False,
    ):
        if (
            not isinstance(request_id, str)
            or not 1 <= len(request_id) <= 128
            or arm not in {"vllm", "smg"}
            or model != self.policy.model
        ):
            raise BudgetError("Unknown request, arm or model")
        if type(num_retries) is not int or num_retries != 0 or output_limit_verified is not True:
            raise BudgetError("Retries or an unverified output limit are forbidden")
        _integer(input_tokens)
        _integer(output_tokens)
        if input_tokens > self.policy.input_limit or output_tokens > self.policy.output_limit:
            raise BudgetError("Token bounds exceed the policy")
        amount = self.policy.cost(input_tokens, output_tokens)
        with self._transaction() as db:
            if self._halted(db):
                raise BudgetError("The shared budget is halted")
            if db.execute("SELECT 1 FROM requests WHERE id=?", (request_id,)).fetchone():
                raise BudgetError("Request ID already reserved")
            calls, allocated, arm_calls = db.execute(
                "SELECT COUNT(*), COALESCE(SUM(amount), 0), COALESCE(SUM(arm=?), 0) FROM requests",
                (arm,),
            ).fetchone()
            if (
                calls >= self.policy.calls
                or arm_calls >= self.policy.calls_per_arm
                or allocated + amount > self.policy.cap_nusd
            ):
                raise BudgetError("Shared budget exhausted")
            db.execute(
                "INSERT INTO requests VALUES (?, ?, ?, ?, ?, 'pending')",
                (request_id, arm, amount, input_tokens, output_tokens),
            )
        return amount

    def finish(self, request_id, input_tokens=None, output_tokens=None):
        problem = None
        with self._transaction() as db:
            self._halted(db)
            row = db.execute(
                "SELECT input_cap, output_cap, state FROM requests WHERE id=?", (request_id,)
            ).fetchone()
            if row is None or row[2] != "pending":
                raise BudgetError("No pending reservation")
            valid = all(
                type(n) is int and 0 <= n <= cap
                for n, cap in zip((input_tokens, output_tokens), row[:2])
            )
            db.execute(
                "UPDATE requests SET state=? WHERE id=?",
                ("complete" if valid else "unknown", request_id),
            )
            if not valid:
                db.execute("UPDATE policy SET halted=1 WHERE id=1")
                problem = "Unknown usage or a violated token bound; reservation retained"
        # Successful calls also keep their worst-case reservation: no refunds.
        if problem:
            raise BudgetError(problem)

    def summary(self):
        with self._transaction() as db:
            halted = bool(self._halted(db))
            calls, amount = db.execute(
                "SELECT COUNT(*), COALESCE(SUM(amount), 0) FROM requests"
            ).fetchone()
        return {"calls": calls, "allocated_nusd": amount, "halted": halted}
