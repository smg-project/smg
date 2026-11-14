#!/usr/bin/env python3
"""Synthetic multi-turn trace for the mock worker's `replay` tool.

Writes the Mooncake format `replay` reads (one JSON row per request: `timestamp`
in ms, `input_length`, `output_length`, `hash_ids`; see "Replaying a trace" in
crates/mock_worker/README.md), shaped like chat or agent sessions:

- sessions arrive as a Poisson process (`--sessions-per-s`);
- a session opens with one of `--tenants` system prompts, the tenant drawn by
  Zipf (`--zipf`), and a first message (`--first-blocks`);
- every later turn comes a think time after the one before (3 s for the
  previous answer, plus an exponential draw of mean `--think-s`) and re-sends
  the whole conversation, with `--answer-blocks` standing in for the previous
  answer and a new message (`--turn-blocks`) appended;
- a session has 1 + floor(Exp(mean `--mean-turns` - 1)) turns, and ends earlier
  once its context would pass `--max-blocks` or the trace ends.

One hash id is one block of 64 tokens: replay the trace with
`--words-per-block 64`. The same flags and seed write the same trace. Stdlib
only.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections.abc import Sequence

# Tokens one hash id stands for.
BLOCK_TOKENS = 64
# Hash ids: tenant t's system prompt holds t * TENANT_SPAN + i, and session s
# takes its own blocks in order from SESSION_BASE + s * SESSION_SPAN.
TENANT_SPAN = 1_000
SESSION_BASE = 1_000_000
SESSION_SPAN = 10_000
# What a later turn waits beyond its think time: the previous answer's latency.
ANSWER_LATENCY_S = 3.0


def block_range(raw: str) -> tuple[int, int]:
    """`lo,hi`: an inclusive range of block counts."""
    try:
        lo, hi = (int(v) for v in raw.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"want lo,hi block counts, got {raw!r}") from None
    if not 0 <= lo <= hi:
        raise argparse.ArgumentTypeError(f"want 0 <= lo <= hi, got {raw!r}")
    return lo, hi


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", required=True, help="the trace file to write (JSON lines)")
    parser.add_argument("--duration-s", type=float, default=720, help="trace length, seconds")
    parser.add_argument("--sessions-per-s", type=float, default=10, help="session arrival rate")
    parser.add_argument("--tenants", type=int, default=16, help="distinct system prompts")
    parser.add_argument("--zipf", type=float, default=1.0, help="Zipf exponent of the tenant draw")
    parser.add_argument(
        "--sys-blocks", type=int, default=32, help="system prompt blocks (32: 2,048 tokens)"
    )
    parser.add_argument(
        "--first-blocks",
        type=block_range,
        default="8,48",
        help="first message blocks, drawn uniformly from lo,hi (8,48: 512 to 3,072 tokens)",
    )
    parser.add_argument(
        "--turn-blocks",
        type=block_range,
        default="2,10",
        help="new message blocks of every later turn, drawn uniformly from lo,hi",
    )
    parser.add_argument(
        "--answer-blocks",
        type=int,
        default=3,
        help="blocks standing in for the previous answer in every later turn",
    )
    parser.add_argument(
        "--mean-turns",
        type=float,
        default=8,
        help="turn count scale N: a session has 1 + floor(Exp(mean N - 1)) turns, about N - 0.5 "
        "on average before the caps",
    )
    parser.add_argument(
        "--max-blocks", type=int, default=400, help="context cap, blocks (400: 25,600 tokens)"
    )
    parser.add_argument("--think-s", type=float, default=8, help="mean think time, seconds")
    parser.add_argument("--output", type=int, default=200, help="output tokens of every request")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    args = parser.parse_args(argv)
    positive = (args.duration_s, args.sessions_per_s, args.think_s)
    if not all(math.isfinite(v) and v > 0 for v in positive):
        parser.error("--duration-s, --sessions-per-s and --think-s must be finite and above 0")
    if not (math.isfinite(args.mean_turns) and args.mean_turns >= 1):
        parser.error("--mean-turns must be finite and at least 1")
    if not abs(args.zipf) <= 100:
        parser.error("--zipf must be from -100 to 100: the tenant weights are powers of it")
    if not 1 <= args.tenants <= SESSION_BASE // TENANT_SPAN:
        parser.error(
            f"--tenants must be 1 to {SESSION_BASE // TENANT_SPAN}: the tenants' hash ids end "
            f"where the sessions' begin"
        )
    if not 0 <= args.sys_blocks <= TENANT_SPAN:
        parser.error(f"--sys-blocks must be 0 to {TENANT_SPAN}: a tenant's hash ids span that many")
    if args.answer_blocks < 0 or args.output < 0:
        parser.error("--answer-blocks and --output must be at least 0")
    shortest = args.sys_blocks + args.first_blocks[0]
    if shortest == 0:
        parser.error("--sys-blocks or --first-blocks must give a first turn a block")
    if args.max_blocks < shortest:
        parser.error("--max-blocks is under the shortest first turn: no session would send one")
    # A written row holds at most the cap; the system prompt's blocks are the
    # tenant's, the rest the session's own.
    if args.max_blocks - args.sys_blocks > SESSION_SPAN:
        parser.error(
            f"a session's rows could hold {args.max_blocks - args.sys_blocks} of its own blocks, "
            f"more than its {SESSION_SPAN} hash ids"
        )
    return args


def generate(args: argparse.Namespace) -> tuple[list[dict], int]:
    """The trace's rows in timestamp order, and how many sessions arrived."""
    rng = random.Random(args.seed)
    weights = [1.0 / (k + 1) ** args.zipf for k in range(args.tenants)]
    rows: list[dict] = []
    start = 0.0
    session = 0
    while True:
        start += rng.expovariate(args.sessions_per_s)
        if start > args.duration_s:
            break
        tenant = rng.choices(range(args.tenants), weights)[0]
        ids = [tenant * TENANT_SPAN + i for i in range(args.sys_blocks)]
        own = itertools.count(SESSION_BASE + session * SESSION_SPAN)
        ids.extend(itertools.islice(own, rng.randint(*args.first_blocks)))
        turns = 1 + int(rng.expovariate(1.0 / max(args.mean_turns - 1, 1e-9)))
        at = start
        for turn in range(turns):
            if turn:
                ids.extend(itertools.islice(own, args.answer_blocks))
                ids.extend(itertools.islice(own, rng.randint(*args.turn_blocks)))
                at += rng.expovariate(1.0 / args.think_s) + ANSWER_LATENCY_S
            if len(ids) > args.max_blocks or at > args.duration_s:
                break
            rows.append(
                {
                    "timestamp": int(at * 1000),
                    "input_length": len(ids) * BLOCK_TOKENS,
                    "output_length": args.output,
                    "hash_ids": list(ids),
                }
            )
        session += 1
    rows.sort(key=lambda row: row["timestamp"])
    return rows, session


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows, sessions = generate(args)
    with open(args.out, "w") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")
    span_s = rows[-1]["timestamp"] / 1000 if rows else 0.0
    rate = len(rows) / span_s if span_s else 0.0
    mean_in = sum(row["input_length"] for row in rows) / len(rows) if rows else 0.0
    distinct = len({h for row in rows for h in row["hash_ids"]})
    print(
        f"sessions={sessions} requests={len(rows)} rate={rate:.1f}/s mean_in={mean_in:.0f} "
        f"distinct_blocks={distinct}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
