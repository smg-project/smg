#!/usr/bin/env python3
"""Focused tests for the generate-sim harness: profile invariants the
benchmark's validity depends on, metric aggregation, scenario construction
(controlled comparisons differ in exactly the intended knob), and the
branch-log parser. Run with:

    python3 -m unittest discover -s scripts/generate_sim -p 'test_*.py'
"""

import json
import tempfile
import unittest
from pathlib import Path

import scenarios
import sim

PROFILES = Path(__file__).resolve().parent / "profiles"
PRODUCTION_PROFILES = ["local-small.json", "local-medium.json", "full.template.json"]


def load(name):
    with open(PROFILES / name) as f:
        return json.load(f)


class ProfileInvariants(unittest.TestCase):
    def test_production_profiles_enable_sticky_override(self):
        # Production runs --routing-key-override; a profile without it
        # measures hash-placement affinity instead of production behavior.
        for name in PRODUCTION_PROFILES + ["smoke.json"]:
            flags = load(name)["smg_flags"]
            self.assertIn("--routing-key-override", flags, name)
            self.assertIn("--assignment-mode", flags, name)

    def test_smg_routing_block_stays_128_while_engine_block_is_256(self):
        for name in PRODUCTION_PROFILES:
            profile = load(name)
            flags = profile["smg_flags"]
            self.assertEqual(flags[flags.index("--block-size") + 1], "128", name)
            self.assertEqual(profile["mock"]["block_size"], 256, name)
            self.assertEqual(profile["mock"]["max_running"], 80, name)

    def test_mock_blocks_use_only_flags_the_mock_worker_accepts(self):
        # The engine is crates/mock_worker's realistic simulator; a key it
        # does not know aborts every worker at launch (the fleet dies before
        # the first request), so the profile schema is pinned to its flags.
        accepted = {
            "engine",
            "prefill_tps",
            "decode_base_ms",
            "decode_per_req_ms",
            "prefill_chunk",
            "max_running",
            "kv_tokens",
            "block_size",
            "prefix_cache",
        }
        for path in sorted(PROFILES.glob("*.json")):
            mock = load(path.name)["mock"]
            self.assertEqual(mock["engine"], "realistic", path.name)
            self.assertTrue(mock.get("prefix_cache"), path.name)
            self.assertLessEqual(set(mock), accepted, path.name)

    def test_full_profile_supports_production_concurrency(self):
        # 6,000 rps x 89 s mean lifetime ~= 534k concurrent requests.
        self.assertGreaterEqual(load("full.template.json")["loadgen"]["max_inflight"], 534_000)

    def test_local_profiles_target_production_worker_pressure(self):
        # ~30-38 concurrent per worker: session_rps x mean_turns x lifetime
        # / workers. Baseline mean turns ~1.5 at t2_ratio 0.5 / max 2.
        for name in ["local-small.json", "local-medium.json"]:
            profile = load(name)
            lg = profile["loadgen"]
            request_rps = lg["session_rps"] * 1.5
            concurrent_per_worker = request_rps * 8.9 / profile["workers_total"]
            self.assertGreater(concurrent_per_worker, 25, name)
            self.assertLess(concurrent_per_worker, 45, name)


class ScenarioConstruction(unittest.TestCase):
    def test_ttl_scenario_differs_only_in_ttl(self):
        base = load("local-small.json")
        rendered = []
        for _, overrides, _ in scenarios.SCENARIOS["ttl-controlled"]:
            profile = json.loads(json.dumps(base))
            patches = None
            for key, val in overrides.items():
                if key == "smg_flag_overrides":
                    patches = val
                else:
                    sim.apply_override(profile, key, val)
            profile["smg_flags"] = scenarios.patch_smg_flags(profile["smg_flags"], patches)
            rendered.append(profile)
        a, b = rendered
        self.assertEqual(a["loadgen"], b["loadgen"], "traffic must be identical")
        diff = [(fa, fb) for fa, fb in zip(a["smg_flags"], b["smg_flags"]) if fa != fb]
        self.assertEqual(len(a["smg_flags"]), len(b["smg_flags"]))
        self.assertEqual(diff, [("18", "2")], "only the TTL value may differ")

    def test_assignment_ab_differs_only_in_mode(self):
        base = load("local-small.json")
        legs = scenarios.SCENARIOS["assignment-mode-ab"]
        flags_a = base["smg_flags"]
        flags_b = scenarios.patch_smg_flags(base["smg_flags"], legs[1][1]["smg_flag_overrides"])
        diff = [(fa, fb) for fa, fb in zip(flags_a, flags_b) if fa != fb]
        self.assertEqual(diff, [("delegate", "min_group")])

    def test_turn_mix_legs_hold_request_rps_constant(self):
        # 305 sessions/s x ~1.5 turns == 110 x ~4.15 turns (within 10%).
        base = load("local-small.json")["loadgen"]["session_rps"]
        multi = scenarios.MULTITURN["loadgen.session_rps"]
        self.assertAlmostEqual(base * 1.5, multi * 4.15, delta=base * 1.5 * 0.10)

    def test_patch_smg_flags_replaces_in_place_and_appends(self):
        flags = ["--assignment-mode", "delegate", "--disable-retries"]
        patched = scenarios.patch_smg_flags(
            flags, {"--assignment-mode": "min_group", "--cache-ttl-secs": "18"}
        )
        self.assertEqual(
            patched,
            [
                "--assignment-mode",
                "min_group",
                "--disable-retries",
                "--cache-ttl-secs",
                "18",
            ],
        )
        self.assertEqual(flags[1], "delegate", "input must not be mutated")

    def test_patch_smg_flags_false_removes_bare_and_valued_flags(self):
        flags = [
            "--cache-index",
            "hash",
            "--routing-key-override",
            "--assignment-mode",
            "delegate",
            "--disable-retries",
        ]
        patched = scenarios.patch_smg_flags(flags, scenarios.RADIX_TREE_FLAGS)
        self.assertEqual(patched, ["--cache-index", "tree", "--disable-retries"])
        self.assertIn("--routing-key-override", flags, "input must not be mutated")

    def test_kv_event_legs_run_grpc_nonstreaming_with_igw(self):
        for label, overrides, _ in scenarios.SCENARIOS["kv-events"]:
            self.assertEqual(overrides["worker_mode"], "grpc", label)
            self.assertIs(
                overrides["loadgen.stream"],
                False,
                f"{label}: gRPC streaming's final frame carries only the tail "
                "token; multi-turn context needs the full output",
            )
            self.assertIn(
                "--enable-igw",
                overrides["smg_flag_overrides"],
                f"{label}: dynamically registered gRPC workers are unreachable without IGW routing",
            )
        # The event legs must drop the sticky short-circuit; the control
        # must keep it (that is what makes it a control).
        by_label = {label: o for label, o, _ in scenarios.SCENARIOS["kv-events"]}
        self.assertIs(
            by_label["event-affine"]["smg_flag_overrides"]["--routing-key-override"],
            False,
        )
        self.assertNotIn(
            "--routing-key-override",
            by_label["sticky-control"]["smg_flag_overrides"],
        )

    def test_radix_legs_share_flag_patch_and_disable_images(self):
        for label, overrides, _ in scenarios.SCENARIOS["radix-replica"]:
            self.assertEqual(
                overrides["smg_flag_overrides"],
                scenarios.RADIX_TREE_FLAGS,
                f"leg {label} must route on the tree without the sticky override",
            )
            self.assertEqual(
                overrides["loadgen.image_count"],
                0,
                f"leg {label}: placeholder expansion is ids-only; images must be off",
            )


class MetricAggregation(unittest.TestCase):
    def _analyze(self, records):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "requests.jsonl"
            with open(path, "w") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")
            return sim.analyze_requests(path, workers_total=4)

    def test_cached_over_prompt_is_token_weighted(self):
        # A fully-cached short prompt plus an uncached long prompt: the
        # token-weighted ratio is 0.1, NOT the 0.5 request mean.
        records = [
            {
                "turn": 1,
                "session": 1,
                "worker_port": 9001,
                "prompt_tokens": 100,
                "cached_tokens": 100,
                "status": 200,
            },
            {
                "turn": 1,
                "session": 2,
                "worker_port": 9002,
                "prompt_tokens": 900,
                "cached_tokens": 0,
                "status": 200,
            },
        ]
        report = self._analyze(records)
        turn1 = report["turns"]["turn1"]
        self.assertEqual(turn1["prompt_tokens_sum"], 1000)
        self.assertEqual(turn1["cached_tokens_sum"], 100)
        self.assertAlmostEqual(turn1["cached_over_prompt"], 0.1)

    def test_seed_aggregation_reports_mean_and_ci(self):
        rows = [{"x": 1.0, "label": "a"}, {"x": 2.0, "label": "a"}, {"x": 3.0, "label": "a"}]
        agg = scenarios.aggregate_seed_rows(rows)
        self.assertTrue(agg["x"].startswith("2 ±"), agg["x"])
        self.assertEqual(agg["label"], "a")


class ArtifactHygiene(unittest.TestCase):
    def test_committed_results_carry_no_absolute_local_paths(self):
        # OSS hygiene: committed artifacts must not leak local usernames or
        # machine paths (repo-relative provenance only).
        results = Path(__file__).resolve().parent / "results"
        if not results.exists():
            self.skipTest("no committed results")
        offenders = []
        for f in results.rglob("*.json"):
            text = f.read_text(errors="replace")
            if "/Users/" in text or "/home/" in text:
                offenders.append(str(f.relative_to(results)))
        self.assertEqual(offenders, [], "absolute local paths in artifacts")


class BranchParsing(unittest.TestCase):
    def test_branch_counts_strip_ansi_color(self):
        line = (
            "\x1b[2m2026-08-27\x1b[0m \x1b[34mDEBUG\x1b[0m Cache-aware selection "
            '\x1b[3mbranch\x1b[0m\x1b[2m=\x1b[0m"hash_hit" worker="http://w"\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "smg.log"
            with open(path, "w") as f:
                f.write(line * 3)
                f.write("unrelated line\n")
            counts = sim.branch_counts(path)
        self.assertEqual(counts["hash_hit"], 3)


class DrillValidation(unittest.TestCase):
    def test_unknown_or_unfireable_drill_keys_fail_loudly(self):
        # A knob the harness would silently ignore must fail at run start.
        for bad in [
            {"index_service": {"replicas": 2}, "explode_index_replica": {"at_secs": 1}},
            {"index_service": {"replicas": 2, "bogus_knob": 1}},
            {"kill_index_replica": {"at_secs": 1}},  # no index_service to kill
            {"index_service": {"replicas": 2}, "partition_drill": {"at_secs": 1}},
            {"index_service": {"replicas": 2, "deferred_replicas": [0]}},
        ]:
            with self.assertRaises(SystemExit, msg=str(bad)):
                sim.validate_profile(bad)
        sim.validate_profile(
            {
                "index_service": {"replicas": 2, "partitionable": True},
                "partition_drill": {"at_secs": 1, "heal_after_secs": 1},
            }
        )
        sim.validate_profile({"restart_smgs_at_secs": 60})

    def test_every_committed_scenario_leg_validates(self):
        # Every leg's knobs must be ones the harness implements, so no
        # scenario can produce a comparison with no independent variable.
        base = load("local-small.json")
        for name, legs in scenarios.SCENARIOS.items():
            for label, overrides, _ in legs:
                profile = json.loads(json.dumps(base))
                for key, val in overrides.items():
                    if key != "smg_flag_overrides":
                        sim.apply_override(profile, key, val)
                try:
                    sim.validate_profile(profile)
                except SystemExit as e:
                    self.fail(f"{name}/{label}: {e}")

    def test_index_replica_cmd_forwards_delays_and_proxied_peers(self):
        profile = {
            "index_service": {"replicas": 3, "apply_delay_stored_ms": 300, "partitionable": True}
        }
        cmd = sim.index_replica_cmd(profile, "/bin/index", 1, bootstrap_from=0)
        self.assertEqual(cmd[cmd.index("--apply-delay-stored-ms") + 1], "300")
        self.assertEqual(
            cmd[cmd.index("--peers") + 1],
            f"http://127.0.0.1:{sim.INDEX_PROXY_BASE},http://127.0.0.1:{sim.INDEX_PROXY_BASE + 2}",
        )
        self.assertEqual(
            cmd[cmd.index("--bootstrap-from") + 1], f"http://127.0.0.1:{sim.INDEX_BASE_PORT}"
        )
        plain = sim.index_replica_cmd({"index_service": {"replicas": 2}}, "/bin/index", 0)
        self.assertEqual(
            plain[plain.index("--peers") + 1], f"http://127.0.0.1:{sim.INDEX_BASE_PORT + 1}"
        )
        self.assertNotIn("--bootstrap-from", plain)


class PartitionProxy(unittest.TestCase):
    def test_sever_cuts_live_and_new_connections_and_heal_restores(self):
        import socket
        import threading

        server = socket.create_server(("127.0.0.1", 0))
        server.settimeout(10)

        def echo():
            while True:
                try:
                    conn, _ = server.accept()
                except OSError:
                    return

                def pump(c):
                    with c:
                        while True:
                            data = c.recv(4096)
                            if not data:
                                return
                            c.sendall(data)

                threading.Thread(target=pump, args=(conn,), daemon=True).start()

        threading.Thread(target=echo, daemon=True).start()
        proxy = sim.TcpProxy(0, server.getsockname()[1])

        def read_eof(sock):
            sock.settimeout(5)
            try:
                return sock.recv(4)
            except OSError:
                return b""

        try:
            with socket.create_connection(("127.0.0.1", proxy.listen_port), timeout=5) as c:
                c.sendall(b"ping")
                self.assertEqual(c.recv(4), b"ping")
                proxy.sever()
                self.assertEqual(read_eof(c), b"", "a severed link must drop the live connection")
            with socket.create_connection(("127.0.0.1", proxy.listen_port), timeout=5) as c2:
                self.assertEqual(read_eof(c2), b"", "a severed link must refuse new connections")
            proxy.heal()
            with socket.create_connection(("127.0.0.1", proxy.listen_port), timeout=5) as c3:
                c3.sendall(b"pong")
                self.assertEqual(c3.recv(4), b"pong")
        finally:
            proxy.close()
            server.close()


class SampleWindow(unittest.TestCase):
    def test_steady_state_window_starts_at_zero(self):
        # The sampler starts after the warmup sleep, so elapsed_s == 0 is
        # already steady state; the first sample must count.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "samples.jsonl"
            with open(path, "w") as f:
                for elapsed, rss in ((0.0, 100), (5.0, 200), (400.0, 999)):
                    rec = {"elapsed_s": elapsed, "smg": [{"idx": 0, "rss_kib": rss, "metrics": {}}]}
                    f.write(json.dumps(rec) + "\n")
            out = sim.summarize_samples(path, 1, window=(0, 150))
            self.assertEqual(out[0]["rss_kib"]["peak"], 200)
            self.assertEqual(out[0]["rss_kib"]["mean"], 150)


class FailoverBins(unittest.TestCase):
    def _run(self, base):
        import subprocess
        import sys

        script = Path(__file__).resolve().parent / "failover_bins.py"
        return subprocess.run(
            [sys.executable, str(script), str(base)], capture_output=True, text=True, check=False
        )

    def test_empty_analysis_exits_non_zero_and_recorded_kill_bins(self):
        with tempfile.TemporaryDirectory() as tmp:
            seed = Path(tmp) / "leg" / "seed-42"
            seed.mkdir(parents=True)
            (seed / "meta.json").write_text(json.dumps({}))
            (seed / "requests.jsonl").write_text("")
            self.assertNotEqual(self._run(tmp).returncode, 0, "no kill observed must fail")
            (seed / "meta.json").write_text(json.dumps({"index_killed_at_ms": 1_000_000}))
            rows = [
                {
                    "turn": 2,
                    "status": 200,
                    "prompt_tokens": 100,
                    "cached_tokens": 80,
                    "start_ms": 1_000_000 - 5_000,
                    "index_source": "remote_hit",
                },
                {
                    "turn": 2,
                    "status": 200,
                    "prompt_tokens": 100,
                    "cached_tokens": 20,
                    "start_ms": 1_000_000 + 5_000,
                    "index_source": "remote_timeout",
                },
            ]
            (seed / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            done = self._run(tmp)
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertIn("kill observed in 1/1 seeds", done.stdout)
            self.assertIn("-10s", done.stdout)
            self.assertIn("+0s", done.stdout)


class SeedAggregation(unittest.TestCase):
    def test_partial_coverage_is_marked_not_passed_through(self):
        rows = [{"x": 1.0}, {"x": None}, {"x": 3.0}]
        agg = scenarios.aggregate_seed_rows(rows)
        self.assertEqual(agg["x"], "2 (2/3 seeds)")


class SelectLegsTest(unittest.TestCase):
    LEGS = [("a", {}, None), ("b", {}, None), ("c", {}, None)]

    def test_empty_filter_runs_every_leg_in_order(self):
        self.assertEqual(scenarios.select_legs(self.LEGS, []), self.LEGS)

    def test_filter_keeps_scenario_order_not_flag_order(self):
        self.assertEqual(
            [leg[0] for leg in scenarios.select_legs(self.LEGS, ["c", "a"])], ["a", "c"]
        )

    def test_unknown_label_is_an_error(self):
        with self.assertRaises(SystemExit):
            scenarios.select_legs(self.LEGS, ["a", "nope"])


if __name__ == "__main__":
    unittest.main()


class IndexAnalysis(unittest.TestCase):
    def test_histogram_percentiles_interpolate_and_floor_at_inf(self):
        # 100 samples: 50 in (0, 1 ms], 40 in (1, 2 ms], 10 beyond 50 ms.
        buckets = {0.001: 50.0, 0.002: 90.0, 0.05: 90.0, float("inf"): 100.0}
        pct = sim.histogram_percentiles(buckets, quantiles=(0.5, 0.9, 0.99))
        self.assertAlmostEqual(pct["p50"], 1.0, places=3)  # exactly at the first edge
        self.assertAlmostEqual(pct["p90"], 2.0, places=3)
        # p99 lands in +Inf: reported as the last finite bound (a floor).
        self.assertEqual(pct["p99"], 50.0)
        self.assertEqual(sim.histogram_percentiles({}), {})

    def test_histogram_deltas_subtract_and_sum_labels(self):
        first = {'h_bucket{le="0.001",gw="a"}': 5.0, 'h_bucket{le="+Inf",gw="a"}': 7.0}
        last = {
            'h_bucket{le="0.001",gw="a"}': 15.0,
            'h_bucket{le="+Inf",gw="a"}': 20.0,
            'h_bucket{le="0.001",gw="b"}': 2.0,
            'h_bucket{le="+Inf",gw="b"}': 3.0,
            'other_bucket{le="0.001"}': 99.0,
        }
        deltas = sim.histogram_deltas(first, last, "h")
        self.assertEqual(deltas, {0.001: 12.0, float("inf"): 16.0})

    def test_prediction_error_summary_reports_bias_exactness_and_tails(self):
        errors = [0] * 90 + [256] * 5 + [-4096] * 5
        s = sim.prediction_error_summary(errors)
        self.assertEqual(s["requests"], 100)
        self.assertEqual(s["exact_share"], 0.9)
        self.assertEqual(s["p50_abs"], 0)
        self.assertEqual(s["p90_abs"], 0)
        self.assertEqual(s["p95_abs"], 256)
        self.assertEqual(s["max_abs"], 4096)
        self.assertLess(s["mean"], 0)
        self.assertIsNone(sim.prediction_error_summary([]))

    def test_evaluation_scenarios_validate_and_differ_only_by_regime(self):
        base = load("local-medium.json")
        for name in ("eval-matrix", "io-shapes", "chaos", "soak"):
            for label, overrides, _ in scenarios.SCENARIOS[name]:
                profile = json.loads(json.dumps(base))
                for key, val in overrides.items():
                    if key == "smg_flag_overrides":
                        profile["smg_flags"] = scenarios.patch_smg_flags(profile["smg_flags"], val)
                    else:
                        sim.apply_override(profile, key, val)
                sim.validate_profile(profile)  # raises SystemExit on a bad leg
                self.assertEqual(profile["loadgen"]["image_count"], 0, label)
        # Same gateway count and rate across the four regimes of one cell.
        cell = [
            leg for leg in scenarios.SCENARIOS["eval-matrix"] if leg[0].endswith("-smg8-rps611")
        ]
        self.assertEqual(len(cell), 4)
        self.assertEqual({leg[1]["smg_count"] for leg in cell}, {8})
        self.assertEqual({leg[1]["loadgen.session_rps"] for leg in cell}, {611})

    def test_gateway_proxy_rewrites_each_gateways_index_url(self):
        profile = load("local-small.json")
        profile["index_service"] = {"replicas": 1, "gateway_proxy": True}
        profile["smg_flags"] = profile["smg_flags"] + ["--kv-indexer-url", "http://127.0.0.1:40000"]
        cmd0 = sim.smg_cmd(profile, "/bin/smg", 0)
        cmd1 = sim.smg_cmd(profile, "/bin/smg", 1)
        self.assertEqual(
            cmd0[cmd0.index("--kv-indexer-url") + 1], f"http://127.0.0.1:{sim.INDEX_GW_PROXY_BASE}"
        )
        self.assertEqual(
            cmd1[cmd1.index("--kv-indexer-url") + 1],
            f"http://127.0.0.1:{sim.INDEX_GW_PROXY_BASE + 1}",
        )
