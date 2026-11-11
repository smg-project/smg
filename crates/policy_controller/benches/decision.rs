//! The engine's cost per decision: the view build and the decision at 8, 64
//! and 512 candidates, per preset, the least-load decision with a credit
//! held per row, and the spec's parse-and-validate time at apply. Ids:
//! `policy_engine/view_build/<n>`, `policy_engine/decide/<preset>/<n>`,
//! `policy_engine/decide/least_load_credited/<n>`,
//! `policy_engine/epd/<preset>/<items>/<n>` (the encode items of one request
//! decided one by one over an encode pool of `n`, each by its own key, under
//! the `epd` plan), `policy_engine/ring/learn/<n>` (a ring learning `n`
//! members in one merge, the cold start of a pool), `policy_engine/ring/
//! learn_one_at_a_time/<n>` (the same pool added one member at a time, the
//! cost the one merge replaces), `policy_engine/ring/
//! add_remove/<n>` (one member added to a ring of `n` and removed again) and
//! `policy_engine/spec/parse_validate`. The view is built from the synthetic
//! per-worker state of `common` the way a host reads its routing state; the
//! decision runs with a fresh seed per iteration so the drawing pickers draw.
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer; a preset that fails to validate is a bug the unit tests catch"
)]

mod common;

use std::{hint::black_box, time::Instant};

use common::{build_view, engine_of, fleet, WorkerState};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use smg_policy_controller::{
    Candidate, Constraints, Identity, Lease, Lifecycle, Phase, PolicySpec, PoolFacts, RequestFacts,
    Ring, Role, Scratch, Side, Target, WorkerId, PRESETS,
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

/// The encode pool as the host fills it: every encoder compatible with the
/// chosen prefill side and taking images.
fn build_encode_view(fleet: &[WorkerState], out: &mut Vec<Candidate>) {
    build_view(fleet, out);
    for row in out.iter_mut() {
        row.role = Role::Encode;
        row.ec_compatible = true;
        row.vision = true;
    }
}

/// The encode items of one request decided one by one, each by its own
/// content hash as the decision's key: the `epd` plan's encode half under
/// the three encode presets. The ring's members are learned at the first
/// decision, before the measurement.
fn epd(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_engine/epd");
    for preset in ["consistent_hashing", "least_load", "round_robin"] {
        let engine = PolicySpec::epd_presets(preset, "round_robin", "round_robin")
            .expect("an epd spec of this build")
            .validate()
            .expect("the epd spec validates");
        for &n in &SIZES {
            let fleet = fleet(n);
            let mut view = Vec::with_capacity(n);
            build_encode_view(&fleet, &mut view);
            let mut scratch = Scratch::new();
            let pool = PoolFacts {
                pool_size: n as u32,
                generation: 1,
                ..PoolFacts::default()
            };
            let cx = Constraints::default();
            for items in [1u64, 4, 16] {
                group.throughput(Throughput::Elements(items));
                group.bench_with_input(
                    BenchmarkId::new(format!("{preset}/{items}"), n),
                    &n,
                    |b, _| {
                        let mut seed = 0u64;
                        b.iter(|| {
                            seed += 1;
                            let mut picked = 0u32;
                            for item in 0..items {
                                let key = (seed * 16 + item).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                                let facts = RequestFacts {
                                    model: "m",
                                    phase: Phase::Encode,
                                    session_key: Identity::hashed(key | 1),
                                    seed: seed ^ item,
                                    now_micros: 1_000_000,
                                    ..RequestFacts::default()
                                };
                                let decision = engine.decide(
                                    &facts,
                                    &pool,
                                    black_box(&view),
                                    &cx,
                                    &mut scratch,
                                );
                                picked += decision.pick().map_or(0, |p| p.targets.len() as u32);
                            }
                            picked
                        });
                    },
                );
            }
        }
    }
    group.finish();
}

/// A membership change on the ring: one member added and removed again on a
/// ring of `n` members (the rebuild cost a registration or a removal pays).
fn ring(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_engine/ring");
    for &n in &SIZES {
        let members: Vec<(WorkerId, u32)> = (0..n as u64)
            .map(|i| (WorkerId(1_000 + i), 1_000 + i as u32))
            .collect();
        let now = Instant::now();
        group.bench_with_input(BenchmarkId::new("learn", n), &n, |b, _| {
            b.iter(|| {
                let mut ring = Ring::new();
                ring.learn(black_box(&members).iter().copied(), now);
                ring.len()
            });
        });
        group.bench_with_input(BenchmarkId::new("learn_one_at_a_time", n), &n, |b, _| {
            b.iter(|| {
                let mut ring = Ring::new();
                for &(id, key) in black_box(&members) {
                    ring.add(id, key);
                }
                ring.len()
            });
        });
        let mut ring = Ring::new();
        ring.learn(members.iter().copied(), now);
        let newcomer = WorkerId(1);
        group.bench_with_input(BenchmarkId::new("add_remove", n), &n, |b, _| {
            b.iter(|| {
                ring.add(black_box(newcomer), 1);
                ring.remove(newcomer, now);
                ring.len()
            });
        });
    }
    group.finish();
}

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

criterion_group!(benches, view_build, decide, epd, ring, spec_parse_validate);
criterion_main!(benches);
