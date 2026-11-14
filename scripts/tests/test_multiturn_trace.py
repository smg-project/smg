"""The synthetic multi-turn trace: its sessions and turns, its caps, its order and its
determinism."""

from __future__ import annotations

import hashlib
import itertools
import json

import pytest

from scripts import multiturn_trace as trace

SMALL = ("--duration-s", "120", "--sessions-per-s", "2", "--seed", "3")
# The SHA-256 of the trace the script this one was ported from wrote for SMALL:
# the port keeps its draws in order, so the same flags and seed keep writing the
# same trace.
SMALL_SHA256 = "97821b1961a579ba24f464364f8e65401dbdce6a8c35caec6374ca3033952a69"


def rows_for(*flags: str) -> list[dict]:
    rows, _ = trace.generate(trace.parse_args(["--out", "unused", *flags]))
    return rows


def test_the_same_flags_and_seed_write_the_same_trace(tmp_path):
    out = tmp_path / "trace.jsonl"
    assert trace.main(["--out", str(out), *SMALL]) == 0
    assert hashlib.sha256(out.read_bytes()).hexdigest() == SMALL_SHA256
    assert rows_for(*SMALL) != rows_for(*SMALL[:-1], "4")


def sessions_of(rows: list[dict]) -> dict[int, list[dict]]:
    """Rows by session, each session's in order: a session's own hash ids say which."""
    sessions: dict[int, list[dict]] = {}
    for row in rows:
        own = next(h for h in row["hash_ids"] if h >= trace.SESSION_BASE)
        sessions.setdefault((own - trace.SESSION_BASE) // trace.SESSION_SPAN, []).append(row)
    return sessions


def test_a_later_turn_resends_the_conversation_with_the_answer_and_a_new_message():
    sessions = sessions_of(rows_for(*SMALL))
    later = [turns for turns in sessions.values() if len(turns) > 1]
    assert later, "some session has a second turn"
    for turns in later:
        for before, after in itertools.pairwise(turns):
            old, new = before["hash_ids"], after["hash_ids"][len(before["hash_ids"]) :]
            assert after["hash_ids"][: len(old)] == old
            # The previous answer's 3 blocks, then a new message of 2 to 10, all
            # blocks the session had not used.
            assert 3 + 2 <= len(new) <= 3 + 10
            assert new == list(range(old[-1] + 1, old[-1] + 1 + len(new)))
            assert after["timestamp"] - before["timestamp"] >= 3000


def test_every_request_opens_with_its_tenants_prompt_within_the_cap_and_the_trace():
    rows = rows_for(*SMALL, "--tenants", "4", "--max-blocks", "60")
    assert max(len(row["hash_ids"]) for row in rows) == 60, "the cap is reached, not passed"
    tenants = [turns[0]["hash_ids"][0] // trace.TENANT_SPAN for turns in sessions_of(rows).values()]
    assert sorted(set(tenants)) == [0, 1, 2, 3]
    assert tenants.count(0) > tenants.count(3), "the Zipf draw favours the first tenant"
    for row in rows:
        ids = row["hash_ids"]
        tenant = ids[0] // trace.TENANT_SPAN
        assert 0 <= tenant < 4
        assert ids[:32] == [tenant * trace.TENANT_SPAN + i for i in range(32)]
        assert len(ids) <= 60
        assert row["input_length"] == len(ids) * trace.BLOCK_TOKENS
        assert row["output_length"] == 200
        assert 0 <= row["timestamp"] <= 120_000
    timestamps = [row["timestamp"] for row in rows]
    assert timestamps == sorted(timestamps)


def test_fewer_mean_turns_give_shorter_sessions():
    def turns_per_session(mean_turns: str) -> float:
        rows, sessions = trace.generate(
            trace.parse_args(["--out", "unused", *SMALL, "--mean-turns", mean_turns])
        )
        return len(rows) / sessions

    assert turns_per_session("2") < turns_per_session("8") / 2


def test_main_writes_one_json_row_per_request(tmp_path, capsys):
    out = tmp_path / "trace.jsonl"
    assert trace.main(["--out", str(out), *SMALL]) == 0
    lines = out.read_text().splitlines()
    assert [json.loads(line) for line in lines] == rows_for(*SMALL)
    assert f"requests={len(lines)} " in capsys.readouterr().out


@pytest.mark.parametrize(
    ("flags", "reason"),
    [
        (["--first-blocks", "9,8"], "want 0 <= lo <= hi"),
        (["--turn-blocks", "2"], "want lo,hi block counts"),
        (["--sessions-per-s", "0"], "must be finite and above 0"),
        (["--duration-s", "inf"], "must be finite and above 0"),
        (["--sessions-per-s", "nan"], "must be finite and above 0"),
        (["--zipf", "101"], "--zipf must be from -100 to 100"),
        (["--tenants", "1001"], "--tenants must be 1 to 1000"),
        (["--sys-blocks", "1001", "--max-blocks", "1100"], "--sys-blocks must be 0 to 1000"),
        (["--sys-blocks", "0", "--first-blocks", "0,4"], "must give a first turn a block"),
        (["--max-blocks", "39"], "under the shortest first turn"),
        (["--max-blocks", "10033"], "could hold 10001 of its own blocks"),
        (["--sys-blocks", "0", "--max-blocks", "10001"], "could hold 10001 of its own blocks"),
    ],
)
def test_a_trace_the_flags_or_the_hash_id_spans_cannot_hold_is_refused(flags, reason, capsys):
    with pytest.raises(SystemExit):
        trace.parse_args(["--out", "unused", *flags])
    assert reason in capsys.readouterr().err


def test_the_widest_spans_the_hash_ids_hold_are_accepted():
    for flags in (
        ["--tenants", "1000", "--sys-blocks", "1000", "--max-blocks", "1100"],
        ["--max-blocks", "10032"],
        ["--sys-blocks", "0", "--max-blocks", "10000"],
    ):
        trace.parse_args(["--out", "unused", *flags])
