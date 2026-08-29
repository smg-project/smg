#!/usr/bin/env python3
"""Named comparison scenarios on top of sim.py.

Each scenario is a list of legs; a leg is a profile-override set (dotted keys)
plus, for policy A/B, which prebuilt gateway binary to use. `compare` runs the
legs sequentially against a fresh fleet each and emits a side-by-side markdown
table built from every leg's report.json. Per-turn rows are always included,
so `turn-ab` (t2_ratio 1.0) needs only one leg.

Usage:
  scripts/generate_sim/scenarios.py list
  scripts/generate_sim/scenarios.py compare --scenario smg1-vs-smg8 \
      --profile scripts/generate_sim/profiles/local-small.json
  scripts/generate_sim/scenarios.py compare --scenario policy-ab \
      --profile ... --smg-bin-a /path/A/smg --smg-bin-b /path/B/smg
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim  # noqa: E402

# Multi-turn conversational traffic at the SAME total request rate as the
# 1.5-turn baseline (305 sessions/s x ~1.5 turns ~= 110 x ~4.15 turns):
# turn-mix comparisons must hold request RPS constant, not session RPS.
MULTITURN = {
    "loadgen.session_rps": 110,
    "loadgen.t2_ratio": 0.85,
    "loadgen.max_turns": 8,
    "loadgen.t2_suffix_tokens": 256,
}


# 10x-compressed gateway clocks for timing-sensitive legs. The profiles
# compress request time 10x (itl/think/prefill), but the gateway's
# wall-clock timers keep production defaults (load monitor 10 s,
# eviction 120 s, cache TTL 180 s) — uncompressed they distort any leg
# whose behavior depends on load freshness, tree eviction, or TTL.
COMPRESSED_CLOCK_FLAGS = {
    "--load-monitor-interval": "1",
    "--eviction-interval": "12",
    "--cache-ttl-secs": "18",
}

# Shared gateway-flag patch for the radix-replica legs: route on the
# approximate radix tree with no sticky short-circuit. False removes a flag.
RADIX_TREE_FLAGS = {
    "--cache-index": "tree",
    "--routing-key-override": False,
    "--assignment-mode": False,
}

# kv-events legs: gRPC-mode workers streaming KV cache events, tree index,
# no sticky short-circuit. --enable-igw is required because workers are
# registered dynamically: without it the gateway's single router is chosen
# from --worker-urls schemes at startup (HTTP by default) and would never
# route to gRPC workers. Loadgen runs non-streaming: the gRPC router's
# final SSE frame carries only the tail output token, which would starve
# the multi-turn context build.
KV_EVENT_OVERRIDES = {
    "worker_mode": "grpc",
    "loadgen.image_count": 0,
    "loadgen.stream": False,
    "loadgen.model": "mock-model",
    "smg_flag_overrides": {**RADIX_TREE_FLAGS, "--enable-igw": None},
}


def patch_smg_flags(flags, patches):
    """Return a copy of `flags` with each `--flag: value` replaced in place
    (or appended); a value of False removes the flag (and its value when it
    has one). Scenario legs that must differ in exactly one gateway setting
    go through here, so the rest of the flag list is shared by
    construction."""
    out = list(flags)
    for flag, value in patches.items():
        if value is False:
            if flag in out:
                idx = out.index(flag)
                span = 2 if idx + 1 < len(out) and not out[idx + 1].startswith("--") else 1
                del out[idx : idx + span]
            continue
        if flag in out:
            idx = out.index(flag)
            if value is None:
                continue
            out[idx + 1] = str(value)
        else:
            out.append(flag)
            if value is not None:
                out.append(str(value))
    return out


REMOTE_INDEX_FLAGS = {
    "--enable-igw": None,
    "--kv-indexer-url": "http://127.0.0.1:40000",
    "--kv-indexer-block-size": "256",
}


def remote_leg(index_overrides, extra=None):
    """A sprayed remote-index leg; sweep legs vary ONLY index_overrides."""
    leg = {
        **KV_EVENT_OVERRIDES,
        "loadgen.ingress": "random",
        "loadgen.turn2_ingress": "random",
        "index_service": {
            "replicas": 2,
            "bridge": True,
            "inferred_ttl_secs": 18,
            "sweep_interval_secs": 1,
            "default_capacity_blocks": 4688,
            **index_overrides,
        },
        "smg_flag_overrides": {
            **RADIX_TREE_FLAGS,
            **COMPRESSED_CLOCK_FLAGS,
            **REMOTE_INDEX_FLAGS,
        },
    }
    if extra:
        leg.update(extra)
    return leg


# Leg: (label, {dotted override: value}, smg_bin slot: None | "a" | "b").
# The special override key "smg_flag_overrides" patches individual gateway
# flags via patch_smg_flags. smg1-vs-smg8 keeps the same session_rps:
# aggregate rps is a loadgen-side property, so halving the gateway count
# concentrates rather than shrinks load.
SCENARIOS = {
    # Production sticky-session assignment A/B: delegate (pin the worker the
    # policy chose for turn 1) vs min_group (pin the group-least-keys
    # worker). Everything else identical.
    "assignment-mode-ab": [
        ("delegate", {}, None),
        (
            "min-group",
            {"smg_flag_overrides": {"--assignment-mode": "min_group"}},
            None,
        ),
    ],
    # Controlled TTL comparison: traffic identical (think fixed at 6 s
    # compressed = 60 s production), ONLY --cache-ttl-secs differs. 18 s is
    # the production 180 s under 10x time compression; 2 s puts the TTL
    # below the think time.
    "ttl-controlled": [
        (
            "ttl-18s",
            dict(
                MULTITURN,
                **{
                    "loadgen.think_secs": 6,
                    "smg_flag_overrides": {"--cache-ttl-secs": "18"},
                },
            ),
            None,
        ),
        (
            "ttl-2s",
            dict(
                MULTITURN,
                **{
                    "loadgen.think_secs": 6,
                    "smg_flag_overrides": {"--cache-ttl-secs": "2"},
                },
            ),
            None,
        ),
    ],
    "smg1-vs-smg8": [
        ("smg1", {"smg_count": 1}, None),
        ("smg8", {"smg_count": 8}, None),
    ],
    "stable-key-vs-random-ingress": [
        ("hash-ingress", {"loadgen.ingress": "hash"}, None),
        ("random-ingress", {"loadgen.ingress": "random"}, None),
    ],
    # What does it take to sustain >= 0.80 aggregate cached tokens? The
    # aggregate is (1-f)*prefix_share + f*followup_ratio where f is the
    # follow-up share of requests — a traffic property. Request RPS is held
    # constant across legs (see MULTITURN). TTL effects live in the separate
    # ttl-controlled scenario so this one varies traffic shape only.
    "hit-rate-calibration": [
        ("baseline-1p5turn", {}, None),
        ("multiturn", dict(MULTITURN), None),
        (
            "multiturn-prefix4k",
            dict(MULTITURN, **{"loadgen.system_prefix_tokens": 4096}),
            None,
        ),
    ],
    # Gateway-revision comparison: the deployed revision (prebuilt binary,
    # slot a — build it from the actual deployed SHA with an ISOLATED
    # CARGO_TARGET_DIR) vs the branch's ambient latest-main build, plus
    # min_group on latest main. Routing-regime differences show directly in
    # the "body path streamed share" row.
    "revision-ab": [
        ("deployed-rev", {}, "a"),
        ("latest-main", {}, None),
        (
            "latest-main-min-group",
            {"smg_flag_overrides": {"--assignment-mode": "min_group"}},
            None,
        ),
    ],
    # Routing-key stability: stable per-session keys (baseline), a fresh key
    # every turn (sticky pin lost + ingress re-hash each turn), and everyone
    # sharing 32 keys (concurrent same-key pressure on the sticky cap).
    "key-stability": [
        ("stable-key", {}, None),
        (
            "key-per-turn",
            {
                "loadgen.key_per_turn": True,
                "loadgen.turn2_ingress": "hash",
            },
            None,
        ),
        ("shared-keys", {"loadgen.routing_key_reuse": 1.0}, None),
    ],
    # Approximate radix trees (cache_index=tree) are PER-REPLICA state: each
    # gateway learns only from its own placements — over HTTP there are no
    # worker KV events and no gateway-to-gateway sync. These legs drop the
    # sticky override so cache_aware actually consults the tree, and vary
    # (a) the index input — pre-tokenized ids (token tree) vs raw text
    # (string tree), (b) whether a session's turns stay on one SMG (hash
    # ingress, like a consistent-hashing LB) or spray uniformly (random),
    # and (c) the replica count. Prediction accuracy should collapse only
    # when per-replica trees AND sprayed sessions combine. token-hint keeps
    # the body streaming while still feeding the token tree via the
    # x-smg-routing-tokens header. Images off: placeholder expansion is an
    # ids-path feature, so this keeps ids and text legs byte-comparable.
    "radix-replica": [
        (
            "token-affine",
            {
                "smg_flag_overrides": RADIX_TREE_FLAGS,
                "loadgen.image_count": 0,
            },
            None,
        ),
        (
            "token-random",
            {
                "smg_flag_overrides": RADIX_TREE_FLAGS,
                "loadgen.image_count": 0,
                "loadgen.ingress": "random",
                "loadgen.turn2_ingress": "random",
            },
            None,
        ),
        (
            "token-single",
            {
                "smg_flag_overrides": RADIX_TREE_FLAGS,
                "loadgen.image_count": 0,
                "loadgen.ingress": "random",
                "loadgen.turn2_ingress": "random",
                "smg_count": 1,
            },
            None,
        ),
        (
            "text-affine",
            {
                "smg_flag_overrides": RADIX_TREE_FLAGS,
                "loadgen.image_count": 0,
                "loadgen.payload": "text",
            },
            None,
        ),
        (
            "text-random",
            {
                "smg_flag_overrides": RADIX_TREE_FLAGS,
                "loadgen.image_count": 0,
                "loadgen.payload": "text",
                "loadgen.ingress": "random",
                "loadgen.turn2_ingress": "random",
            },
            None,
        ),
        (
            "token-hint-streamed",
            {
                "smg_flag_overrides": RADIX_TREE_FLAGS,
                "loadgen.image_count": 0,
                "loadgen.tokens_hint": True,
            },
            None,
        ),
    ],
    # Event-driven cache-aware routing: gRPC workers broadcast their actual
    # cache contents (SubscribeKvEvents -> PositionalIndexer), so every
    # gateway replica independently converges on ground truth with no
    # gateway-to-gateway sync. The paired approximate-tree legs in
    # radix-replica collapse under sprayed ingress; if worker broadcast
    # works, the sprayed leg here should NOT collapse — that is the whole
    # comparison. buffered-control repeats the approximate token tree under
    # the same gRPC fleet, isolating "events vs approximation" from any
    # HTTP-vs-gRPC pipeline difference... except events cannot be disabled
    # per leg without a mock flag, so the control instead sprays with the
    # sticky override ON (events subscribed but placement sticky), pinning
    # the event machinery's cost while removing its routing influence.
    "kv-events": [
        ("event-affine", dict(KV_EVENT_OVERRIDES), None),
        (
            "event-sprayed",
            {
                **KV_EVENT_OVERRIDES,
                "loadgen.ingress": "random",
                "loadgen.turn2_ingress": "random",
            },
            None,
        ),
        (
            "sticky-control",
            {
                "worker_mode": "grpc",
                "loadgen.image_count": 0,
                "loadgen.stream": False,
                "loadgen.model": "mock-model",
                "smg_flag_overrides": {"--cache-index": "tree", "--enable-igw": None},
            },
            None,
        ),
    ],
    # M2 core matrix (02-experiment-plan.md): sprayed ingress everywhere
    # (the regime that breaks per-replica state), one variable per
    # comparison. local-event = per-gateway event indexers (round-3
    # architecture); remote-event = same feed, ONE shared index (parity =
    # location only); remote-placement = shared index fed ONLY by gateway
    # placements (bridge off — the eventless-engine thesis, the number
    # that does not exist yet); mesh-tree = the in-repo TreeSync
    # alternative (approximate trees synced gateway-to-gateway).
    # Compressed gateway clocks on every leg (held constant).
    "remote-index": [
        (
            "local-event-sprayed",
            {
                **KV_EVENT_OVERRIDES,
                "loadgen.ingress": "random",
                "loadgen.turn2_ingress": "random",
                "smg_flag_overrides": {
                    **RADIX_TREE_FLAGS,
                    "--enable-igw": None,
                    **COMPRESSED_CLOCK_FLAGS,
                },
            },
            None,
        ),
        (
            "remote-event-sprayed",
            {
                **KV_EVENT_OVERRIDES,
                "loadgen.ingress": "random",
                "loadgen.turn2_ingress": "random",
                "index_service": {
                    "replicas": 2,
                    "bridge": True,
                    "inferred_ttl_secs": 18,
                    "sweep_interval_secs": 1,
                    "default_capacity_blocks": 4688,
                },
                "smg_flag_overrides": {
                    **RADIX_TREE_FLAGS,
                    "--enable-igw": None,
                    **COMPRESSED_CLOCK_FLAGS,
                    "--kv-indexer-url": "http://127.0.0.1:40000",
                    "--kv-indexer-block-size": "256",
                },
            },
            None,
        ),
        (
            "remote-placement-sprayed",
            {
                **KV_EVENT_OVERRIDES,
                "loadgen.ingress": "random",
                "loadgen.turn2_ingress": "random",
                "index_service": {
                    "replicas": 2,
                    "bridge": False,
                    "inferred_ttl_secs": 18,
                    "sweep_interval_secs": 1,
                    "default_capacity_blocks": 4688,
                },
                "smg_flag_overrides": {
                    **RADIX_TREE_FLAGS,
                    "--enable-igw": None,
                    **COMPRESSED_CLOCK_FLAGS,
                    "--kv-indexer-url": "http://127.0.0.1:40000",
                    "--kv-indexer-block-size": "256",
                },
            },
            None,
        ),
        (
            "mesh-tree-sprayed",
            {
                **KV_EVENT_OVERRIDES,
                "loadgen.ingress": "random",
                "loadgen.turn2_ingress": "random",
                "mesh_smgs": True,
                "smg_flag_overrides": {
                    **RADIX_TREE_FLAGS,
                    "--enable-igw": None,
                    **COMPRESSED_CLOCK_FLAGS,
                },
            },
            None,
        ),
    ],
    # Mesh TreeSync leg alone (the core matrix's first three legs already
    # completed; peer-url format fix made this rerunnable separately).
    "remote-index-mesh": [
        (
            "mesh-tree-sprayed",
            {
                **KV_EVENT_OVERRIDES,
                "loadgen.ingress": "random",
                "loadgen.turn2_ingress": "random",
                "mesh_smgs": True,
                "smg_flag_overrides": {
                    **RADIX_TREE_FLAGS,
                    "--enable-igw": None,
                    **COMPRESSED_CLOCK_FLAGS,
                },
            },
            None,
        ),
    ],
    # Staleness sweep (event feed; the placement feed has no Removed to
    # delay): constant injected apply lag, reported against the 3 s
    # compressed think time (= 30 s production). Stored and Removed are
    # delayed in SEPARATE legs — they fail in opposite directions.
    "index-staleness": [
        ("stored-30ms", remote_leg({"apply_delay_stored_ms": 30}), None),
        ("stored-300ms", remote_leg({"apply_delay_stored_ms": 300}), None),
        ("stored-3000ms", remote_leg({"apply_delay_stored_ms": 3000}), None),
        ("removed-3000ms", remote_leg({"apply_delay_removed_ms": 3000}), None),
    ],
    # Capacity-model sensitivity for the inferred feed (placement-only):
    # 0.5x / 2x of the 1x=4688 blocks used by the core matrix leg.
    "index-capacity": [
        (
            "capacity-half",
            remote_leg({"bridge": False, "default_capacity_blocks": 2344}),
            None,
        ),
        (
            "capacity-double",
            remote_leg({"bridge": False, "default_capacity_blocks": 9375}),
            None,
        ),
    ],
    # Failover drill: kill replica 0 (the endpoint every gateway and
    # publisher dials) mid-window, relaunch 30 s later bootstrapping from
    # the survivor. Placement-fed (the harder case: no replayable feed).
    # Run with --seeds 6: timing nondeterminism needs the wider n.
    "index-failover": [
        (
            "kill-replica0",
            remote_leg(
                {"bridge": False},
                {
                    "kill_index_replica": {
                        "at_secs": 60,
                        "replica": 0,
                        "relaunch_after_secs": 30,
                    }
                },
            ),
            None,
        ),
    ],
    # Kill and relaunch every SMG mid-window: sticky pins and placements are
    # process state, so affinity must rebuild; errors during the blackout
    # are part of the result.
    "router-restart": [
        (
            "restart-mid-run",
            dict(
                MULTITURN,
                **{"restart_smgs_at_secs": 60},
            ),
            None,
        ),
    ],
    # The hash placement index is LOCAL to each SMG: turn-2 affinity only
    # survives if turn 2 reaches the same SMG. This isolates that effect.
    # Does a smaller SMG fleet raise cache hit rates? With sticky turns the
    # placement index fragmentation shouldn't matter; with scattered turns
    # the chance of landing on the SMG that holds the placement is 1/K.
    "fleet-size-sweep": [
        ("smg2-sticky", {"smg_count": 2}, None),
        ("smg8-sticky", {"smg_count": 8}, None),
        (
            "smg2-t2random",
            {
                "smg_count": 2,
                "loadgen.ingress": "random",
                "loadgen.turn2_ingress": "random",
            },
            None,
        ),
        (
            "smg8-t2random",
            {
                "smg_count": 8,
                "loadgen.ingress": "random",
                "loadgen.turn2_ingress": "random",
            },
            None,
        ),
    ],
    "turn2-same-vs-random-smg": [
        ("t2-same-smg", {"loadgen.turn2_ingress": "same"}, None),
        (
            "t2-random-smg",
            {"loadgen.ingress": "random", "loadgen.turn2_ingress": "random"},
            None,
        ),
    ],
    "cold-vs-warm-prefix": [
        ("cold", {"loadgen.system_prefix_tokens": 0}, None),
        ("warm", {"loadgen.system_prefix_tokens": 2048}, None),
    ],
    "turn-ab": [
        ("turn-ab", {"loadgen.t2_ratio": 1.0}, None),
    ],
    "policy-ab": [
        ("policy-a", {}, "a"),
        ("policy-b", {}, "b"),
    ],
}


def _get(mapping, *keys):
    node = mapping
    for key in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def extract_rows(report):
    """Flatten one report.json into the comparison rows (all guarded)."""
    summary = report.get("loadgen_summary", {})
    req = report.get("requests", {})
    samples = report.get("samples", [])
    branches = report.get("cache_aware_branches", [])

    rss_peaks = [_get(s, "rss_kib", "peak") for s in samples]
    rss_peaks = [v for v in rss_peaks if v is not None]
    cpu_means = [_get(s, "cpu_pct", "mean") for s in samples]
    cpu_means = [v for v in cpu_means if v is not None]
    queue_peaks = [_get(s, "queue_depth", "peak") for s in samples]
    queue_peaks = [v for v in queue_peaks if v is not None]
    rejected = sum(s.get("rejected_total") or 0 for s in samples)

    branch_totals = {}
    for entry in branches:
        for name, count in entry.get("branches", {}).items():
            branch_totals[name] = branch_totals.get(name, 0) + count
    total_decisions = sum(branch_totals.values())
    hash_hit = branch_totals.get("hash_hit", 0)

    rows = {}
    totals = summary.get("totals", {})
    errors = totals.get("errors", {})
    err_count = sum(errors.values()) if isinstance(errors, dict) else errors
    requests_total = totals.get("requests")
    rows["ok"] = (
        requests_total - err_count
        if isinstance(requests_total, int) and isinstance(err_count, int)
        else None
    )
    rows["err"] = err_count
    rows["achieved_rps"] = summary.get("achieved_rps")
    for metric in ("ttft_ms", "e2e_ms"):
        for pct in ("p50", "p90", "p99"):
            rows["%s_%s" % (metric, pct)] = _get(summary, metric, pct)
    # Token-weighted (sum cached / sum prompt) is THE number comparable with
    # backend cached-token telemetry; the per-request mean is a different
    # statistic and is reported under its own name.
    rows["AGG cached tokens (sum/sum)"] = _get(summary, "overall", "cached_token_ratio")
    rows["AGG cached (request mean)"] = _get(
        summary, "overall", "cached_ratio_request_mean"
    )
    for turn in ("turn1", "followup"):
        rows[turn + " cached tokens (sum/sum)"] = _get(
            summary, "turns", turn, "cached_token_ratio"
        )
        rows[turn + " cached (request mean)"] = _get(
            summary, "turns", turn, "cached_ratio_request_mean"
        )
        rows[turn + " prompt tokens sum"] = _get(
            summary, "turns", turn, "prompt_tokens_sum"
        )
        rows[turn + " cached tokens sum"] = _get(
            summary, "turns", turn, "cached_tokens_sum"
        )
    rows["mean turns/session"] = summary.get("mean_turns_per_session")
    rows["t2 same-worker (loadgen)"] = summary.get("turn2_same_worker_rate")
    rows["followup same-worker"] = summary.get("followup_same_worker_rate")
    rows["t1 max worker share"] = _get(summary, "turn1_workers", "max_share")
    rows["t1 entropy (norm)"] = _get(summary, "turn1_workers", "normalized_entropy")
    for turn in ("turn1", "turn2"):
        rows[turn + " cached/prompt"] = _get(req, "turns", turn, "cached_over_prompt")
        rows[turn + " hit rate"] = _get(req, "turns", turn, "hit_rate")
        rows[turn + " CoV (fleet)"] = _get(req, "turns", turn, "imbalance", "cov_fleet")
    rows["t2 same-worker rate"] = req.get("t2_same_worker_rate")
    imb = req.get("overall_imbalance", {})
    rows["overall CoV (fleet)"] = imb.get("cov_fleet")
    rows["distinct workers"] = imb.get("distinct_workers")
    rows["hash_hit share"] = (
        round(hash_hit / total_decisions, 4) if total_decisions else None
    )
    # Sticky-session outcomes (--routing-key-override): occupied_hit is a
    # follow-up landing on its pinned worker; vacant is a fresh key.
    sticky = {}
    for s in samples:
        for name, count in (s.get("sticky_branches") or {}).items():
            sticky[name] = sticky.get(name, 0) + count
    sticky_total = sum(sticky.values())
    rows["sticky occupied_hit share"] = (
        round(sticky.get("occupied_hit", 0) / sticky_total, 4) if sticky_total else None
    )
    rows["sticky cap_respill count"] = sticky.get("cap_respill", 0) if sticky else None
    # Direct verification of the request-body regime: share of requests
    # routed via streaming (header-only selection) vs buffered (typed path).
    paths = {}
    for s in samples:
        for name, count in (s.get("body_paths") or {}).items():
            paths[name] = paths.get(name, 0) + count
    path_total = sum(paths.values())
    streamed = sum(v for k, v in paths.items() if k.startswith("streamed"))
    # Remote-index rows (--kv-indexer-url legs): per-request echo shares
    # and the direct accuracy signal (predicted-vs-actual cached tokens).
    sources = req.get("index_sources") or {}
    total_sourced = sum(sources.values())
    if total_sourced:
        rows["index remote_hit share"] = round(
            sources.get("remote_hit", 0) / total_sourced, 4
        )
        misses = total_sourced - sources.get("remote_hit", 0) - sources.get("remote_empty", 0)
        rows["index degraded share (timeout+disconnect)"] = round(
            misses / total_sourced, 4
        )
    pred = req.get("index_prediction_error_tokens")
    if pred:
        rows["index prediction error mean (tokens)"] = pred.get("mean")
        rows["index prediction error p95 abs (tokens)"] = pred.get("p95_abs")

    rows["body path streamed share"] = (
        round(streamed / path_total, 4) if path_total else None
    )
    rows["offered session rps"] = summary.get("offered_session_rps")
    rows["drain requests (excluded)"] = _get(summary, "drain", "requests")
    rows["rss peak MiB (max smg)"] = (
        round(max(rss_peaks) / 1024, 1) if rss_peaks else None
    )
    rows["cpu mean % (max smg)"] = max(cpu_means) if cpu_means else None
    rows["queue depth peak"] = max(queue_peaks) if queue_peaks else None
    rows["rejected total"] = rejected
    return rows


def write_compare_md(scenario, leg_results, path):
    labels = [label for label, _ in leg_results]
    row_keys = []
    for _, rows in leg_results:
        for key in rows:
            if key not in row_keys:
                row_keys.append(key)
    lines = ["# generate-sim compare — %s" % scenario, ""]
    lines.append("| metric | " + " | ".join(labels) + " |")
    lines.append("|---|" + "---|" * len(labels))
    for key in row_keys:
        cells = [sim._fmt(rows.get(key)) for _, rows in leg_results]
        lines.append("| %s | %s |" % (key, " | ".join(cells)))
    lines.append("")
    for label, _ in leg_results:
        lines.append("- %s: see its run dir for report.md / report.json" % label)
    lines.append("")
    text = "\n".join(lines)
    with open(path, "w") as f:
        f.write(text)
    print(text)


def cmd_compare(args):
    legs = SCENARIOS[args.scenario]
    bins = {"a": args.smg_bin_a, "b": args.smg_bin_b, None: args.smg_bin}
    for _, _, slot in legs:
        if slot is not None and not bins[slot]:
            raise SystemExit(
                "scenario %s needs --smg-bin-%s (a prebuilt gateway binary)"
                % (args.scenario, slot)
            )

    base = sim.load_profile(args.profile)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = Path(args.out_root or sim.REPO_ROOT / "target" / "generate-sim")
    scenario_dir = out_root / ("%s-%s-%s" % (args.scenario, base.get("name", "run"), stamp))

    leg_results = []
    built = args.skip_build
    for idx, (label, overrides, slot) in enumerate(legs):
        seed_rows = []
        for seed_idx in range(args.seeds):
            profile = json.loads(json.dumps(base))  # deep copy: legs must not leak
            flag_patches = None
            for key, val in overrides.items():
                if key == "smg_flag_overrides":
                    flag_patches = val
                else:
                    sim.apply_override(profile, key, val)
            if flag_patches:
                profile["smg_flags"] = patch_smg_flags(profile["smg_flags"], flag_patches)
            for raw in args.override:
                key, val = sim.parse_override_arg(raw)
                sim.apply_override(profile, key, val)
            base_seed = int(profile.get("loadgen", {}).get("seed", 42))
            sim.apply_override(profile, "loadgen.seed", base_seed + seed_idx)
            sim.log(
                "scenario %s leg %d/%d: %s seed %d/%d"
                % (args.scenario, idx + 1, len(legs), label, seed_idx + 1, args.seeds)
            )
            run_dir = sim.run_profile(
                profile,
                scenario_dir / label / ("seed-%d" % (base_seed + seed_idx)),
                smg_bin=bins[slot],
                skip_build=built,
            )
            built = True  # binaries exist after the first run
            with open(Path(run_dir) / "report.json") as f:
                seed_rows.append(extract_rows(json.load(f)))
            time.sleep(2)
        with open(scenario_dir / label / "seed-rows.json", "w") as f:
            json.dump(seed_rows, f, indent=2)
        leg_results.append((label, aggregate_seed_rows(seed_rows)))

    # Leg-to-leg binary identity: when no leg deliberately uses a prebuilt
    # slot (revision A/B does), every leg must have run the SAME gateway
    # binary — otherwise the comparison silently includes a build delta.
    if all(slot is None for _, _, slot in legs):
        smg_shas = set()
        for label, _, _ in legs:
            for meta_path in sorted((scenario_dir / label).glob("seed-*/meta.json")):
                with open(meta_path) as f:
                    smg_shas.add(json.load(f)["binary_sha256"]["smg"])
        if len(smg_shas) > 1:
            raise SystemExit(
                "scenario %s legs ran different gateway binaries: %s"
                % (args.scenario, sorted(smg_shas))
            )

    write_compare_md(args.scenario, leg_results, scenario_dir / "compare.md")
    sim.log("compare: %s" % (scenario_dir / "compare.md"))


# Two-sided 97.5% Student-t quantiles by degrees of freedom; small seed
# counts need these, not the normal 1.96 (n=3 would otherwise report
# intervals ~2.2x too narrow).
STUDENT_T_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
}


def aggregate_seed_rows(seed_rows):
    """Mean ± 95% CI half-width (Student-t) across seeds for numeric rows;
    single-seed runs and non-numeric rows pass through the first value."""
    if len(seed_rows) == 1:
        return seed_rows[0]
    keys = []
    for rows in seed_rows:
        for key in rows:
            if key not in keys:
                keys.append(key)
    out = {}
    for key in keys:
        vals = [r.get(key) for r in seed_rows]
        nums = [v for v in vals if isinstance(v, (int, float))]
        if len(nums) == len(seed_rows):
            n = len(nums)
            m = sum(nums) / n
            var = sum((v - m) ** 2 for v in nums) / (n - 1)
            t = STUDENT_T_975.get(n - 1, 1.96)
            half = t * (var**0.5) / (n**0.5)
            out[key] = "%.4g ±%.2g" % (m, half)
        else:
            out[key] = vals[0]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list scenario names")

    cmp_p = sub.add_parser("compare", help="run a scenario's legs and emit compare.md")
    cmp_p.add_argument("--scenario", required=True, choices=sorted(SCENARIOS))
    cmp_p.add_argument("--profile", required=True, help="base profile JSON")
    cmp_p.add_argument("--skip-build", action="store_true")
    cmp_p.add_argument("--smg-bin", help="gateway binary for non-A/B legs")
    cmp_p.add_argument("--smg-bin-a", help="gateway binary A (policy-ab)")
    cmp_p.add_argument("--smg-bin-b", help="gateway binary B (policy-ab)")
    cmp_p.add_argument("--out-root", help="parent dir for the scenario run dirs")
    cmp_p.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="loadgen seeds per leg; rows report mean ±95%% CI (default 3)",
    )
    cmp_p.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra dotted profile override applied to every leg (repeatable)",
    )

    args = parser.parse_args()
    if args.cmd == "list":
        for name, legs in sorted(SCENARIOS.items()):
            print("%-30s %s" % (name, " vs ".join(label for label, _, _ in legs)))
    elif args.cmd == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()
