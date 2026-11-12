//! The engine's cost per decision: the view build and the decision at 8, 64
//! and 512 candidates, per preset, the least-load decision with a credit
//! held per row, and the spec's parse-and-validate time at apply. Ids:
//! `policy_engine/view_build/<n>`, `policy_engine/decide/<preset>/<n>`,
//! `policy_engine/decide/least_load_credited/<n>` and
//! `policy_engine/spec/parse_validate`. The view is built from the synthetic
//! per-worker state of `common` the way a host reads its routing state; the
//! decision runs with a fresh seed per iteration so the drawing pickers draw.
//! This file is the pull-request gate's (`.github/bench-gates/policy-engine.toml`
//! runs the whole binary three times a side): the `epd` plan's encode
//! decisions and the ring's membership costs are the `epd` bench target.
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer; a preset that fails to validate is a bug the unit tests catch"
)]

mod common;

use std::hint::black_box;

use common::{build_view, engine_of, fleet};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use smg_policy_controller::{
    Constraints, Lease, Lifecycle, PolicySpec, PoolFacts, RequestFacts, Scratch, Side, Target,
    PRESETS,
};

const SIZES: [usize; 3] = [8, 64, 512];

fn view_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_engine/view_build");
    for &n in &SIZES {
        let fleet = fleet(n);
        let mut out = Vec::with_capacity(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                build_view(black_box(&fleet), &mut out);
                black_box(&out);
                out.len()
            });
        });
    }
    group.finish();
}

fn decide(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_engine/decide");
    for preset in PRESETS {
        let engine = engine_of(preset);
        for &n in &SIZES {
            let fleet = fleet(n);
            let mut view = Vec::with_capacity(n);
            build_view(&fleet, &mut view);
            let mut scratch = Scratch::new();
            let pool = PoolFacts {
                pool_size: n as u32,
                generation: 1,
                ..PoolFacts::default()
            };
            let cx = Constraints::default();
            group.bench_with_input(BenchmarkId::new(preset, n), &n, |b, _| {
                let mut seed = 0u64;
                b.iter(|| {
                    seed += 1;
                    let facts = RequestFacts {
                        model: "m",
                        prompt_tokens: 512,
                        seed,
                        now_micros: 1_000_000,
                        ..RequestFacts::default()
                    };
                    engine.decide(&facts, &pool, black_box(&view), &cx, &mut scratch)
                });
            });
        }
    }
    // The least-load decision with one dispatch credit per row in the tally,
    // as between two reports under load: every decision reads and ages each
    // reporting row's entry. The credits date after every sample instant, so
    // no decision releases them.
    let engine = engine_of("least_load");
    for &n in &SIZES {
        let fleet = fleet(n);
        let mut view = Vec::with_capacity(n);
        build_view(&fleet, &mut view);
        let lease = Lease {
            issued_micros: 1_000_000,
            tokens: 512,
            ..Lease::default()
        };
        for c in &view {
            let target = Target {
                position: c.position,
                id: c.id,
                dp_rank: c.dp_rank,
                pair: None,
            };
            engine.on_dispatched(&lease, &target, Side::Single, 0);
        }
        let mut scratch = Scratch::new();
        let pool = PoolFacts {
            pool_size: n as u32,
            generation: 1,
            ..PoolFacts::default()
        };
        let cx = Constraints::default();
        group.bench_with_input(BenchmarkId::new("least_load_credited", n), &n, |b, _| {
            let mut seed = 0u64;
            b.iter(|| {
                seed += 1;
                let facts = RequestFacts {
                    model: "m",
                    prompt_tokens: 512,
                    seed,
                    now_micros: 1_000_000,
                    ..RequestFacts::default()
                };
                engine.decide(&facts, &pool, black_box(&view), &cx, &mut scratch)
            });
        });
    }
    group.finish();
}

/// A full document of this build's stages, the size of the largest
/// acceptance-corpus item: the view, two index entries, three profiles,
/// the plan rules and two bindings with their overload defaults.
const SPEC_DOCUMENT: &str = r#"{
  "apiVersion": "smg.policy/v1",
  "generation": 4,
  "view": { "headers": ["x-session-id", "x-smg-routing-key"], "block_size": 128, "boundaries": [256, 1024, 4096], "max_block_hashes": 256 },
  "index": { "*": { "kind": "hash", "ttl_s": 300, "max_size": 67108864, "sweep_s": 120 }, "chat-*": { "max_size": 1048576 } },
  "profiles": {
    "prefill": { "filter": ["healthy", "not-overloaded", "has-capacity", "not-tried"], "score": [{ "least-load": { "kv_pressure_weight": 0.3, "mean_prefill_tokens": 2048, "default_throughput": 1500 } }, "session-affinity"], "pick": "best" },
    "decode": { "filter": ["healthy", "not-overloaded", "has-room", "not-tried"], "score": ["least-load"], "pick": "power-of-two" },
    "default": { "filter": ["healthy", "not-overloaded", "not-tried"], "score": [{ "least-load": { "weight": 0.3 } }, { "least-requests": { "weight": 0.7 } }, "prefix-affinity"], "pick": "best" }
  },
  "plans": { "pd-pair": { "pairing": "strict" } },
  "bindings": [
    { "match": { "model": "pd/*", "phase": ["prefill", "decode"] }, "plan": "pd-pair", "profiles": { "prefill": "prefill", "decode": "decode" },
      "overload": { "enabled": true, "waiting_requests": 32, "token_usage": 0.9, "shed_when_all_vetoed": true } },
    { "match": { "model": "*" }, "profile": "default", "overload": { "enabled": false }, "admission": { "enabled": false } }
  ]
}"#;

fn spec_parse_validate(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_engine/spec");
    group.throughput(Throughput::Bytes(SPEC_DOCUMENT.len() as u64));
    group.bench_function("parse_validate", |b| {
        b.iter(|| {
            let spec: PolicySpec =
                serde_json::from_str(black_box(SPEC_DOCUMENT)).expect("the document parses");
            spec.validate().expect("the document validates")
        });
    });
    group.finish();
}

criterion_group!(benches, view_build, decide, spec_parse_validate);
criterion_main!(benches);
