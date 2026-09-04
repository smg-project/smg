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

    def test_full_profile_supports_production_concurrency(self):
        # 6,000 rps x 89 s mean lifetime ~= 534k concurrent requests.
        self.assertGreaterEqual(
            load("full.template.json")["loadgen"]["max_inflight"], 534_000
        )

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
            profile["smg_flags"] = scenarios.patch_smg_flags(
                profile["smg_flags"], patches
            )
            rendered.append(profile)
        a, b = rendered
        self.assertEqual(a["loadgen"], b["loadgen"], "traffic must be identical")
        diff = [
            (fa, fb)
            for fa, fb in zip(a["smg_flags"], b["smg_flags"])
            if fa != fb
        ]
        self.assertEqual(len(a["smg_flags"]), len(b["smg_flags"]))
        self.assertEqual(diff, [("18", "2")], "only the TTL value may differ")

    def test_assignment_ab_differs_only_in_mode(self):
        base = load("local-small.json")
        legs = scenarios.SCENARIOS["assignment-mode-ab"]
        flags_a = base["smg_flags"]
        flags_b = scenarios.patch_smg_flags(
            base["smg_flags"], legs[1][1]["smg_flag_overrides"]
        )
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
        self.assertEqual(
            patched, ["--cache-index", "tree", "--disable-retries"]
        )
        self.assertIn("--routing-key-override", flags, "input must not be mutated")

    def test_kv_event_legs_run_grpc_nonstreaming_with_igw(self):
        for label, overrides, _ in scenarios.SCENARIOS["kv-events"]:
            self.assertEqual(overrides["worker_mode"], "grpc", label)
            self.assertIs(
                overrides["loadgen.stream"],
                False,
                "%s: gRPC streaming's final frame carries only the tail "
                "token; multi-turn context needs the full output" % label,
            )
            self.assertIn(
                "--enable-igw",
                overrides["smg_flag_overrides"],
                "%s: dynamically registered gRPC workers are unreachable "
                "without IGW routing" % label,
            )
        # The event legs must drop the sticky short-circuit; the control
        # must keep it (that is what makes it a control).
        by_label = {l: o for l, o, _ in scenarios.SCENARIOS["kv-events"]}
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
                "leg %s must route on the tree without the sticky override" % label,
            )
            self.assertEqual(
                overrides["loadgen.image_count"],
                0,
                "leg %s: placeholder expansion is ids-only; images must be off" % label,
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
            {"turn": 1, "session": 1, "worker_port": 9001, "prompt_tokens": 100,
             "cached_tokens": 100, "status": 200},
            {"turn": 1, "session": 2, "worker_port": 9002, "prompt_tokens": 900,
             "cached_tokens": 0, "status": 200},
        ]
        report = self._analyze(records)
        turn1 = report["turns"]["turn1"]
        self.assertEqual(turn1["prompt_tokens_sum"], 1000)
        self.assertEqual(turn1["cached_tokens_sum"], 100)
        self.assertAlmostEqual(turn1["cached_over_prompt"], 0.1)

    def test_seed_aggregation_reports_mean_and_ci(self):
        rows = [{"x": 1.0, "label": "a"}, {"x": 2.0, "label": "a"},
                {"x": 3.0, "label": "a"}]
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
            "\x1b[3mbranch\x1b[0m\x1b[2m=\x1b[0m\"hash_hit\" worker=\"http://w\"\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "smg.log"
            with open(path, "w") as f:
                f.write(line * 3)
                f.write("unrelated line\n")
            counts = sim.branch_counts(path)
        self.assertEqual(counts["hash_hit"], 3)


if __name__ == "__main__":
    unittest.main()
