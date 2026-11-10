//! The engine's cost per decision: the view build and the decision at 8, 64
//! and 512 candidates, per preset, the least-load decision with a credit
//! held per row, and the spec's parse-and-validate time at apply. Ids:
//! `policy_engine/view_build/<n>`, `policy_engine/decide/<preset>/<n>`,
//! `policy_engine/decide/least_load_credited/<n>` and
//! `policy_engine/spec/parse_validate`. The view is built from a
//! synthetic per-worker state the way a host reads its routing state; the
//! decision runs with a fresh seed per iteration so the drawing pickers draw.
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer; a preset that fails to validate is a bug the unit tests catch"
)]

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use smg_policy_controller::{
    Candidate, Constraints, Engine, Health, Lease, Lifecycle, LoadRow, PolicySpec, PoolFacts,
    RequestFacts, Scratch, Side, SignalSource, Target, WorkerId, PRESETS,
};

/// What a host reads per worker before it writes a row.
struct WorkerState {
    id: u64,
    healthy: bool,
    in_flight: u32,
    running: u32,
    queued: u32,
    kv_permille: u16,
    live_rate: u32,
    sampled_at_micros: u64,
}

fn fleet(n: usize) -> Vec<WorkerState> {
    let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
    (0..n)
        .map(|i| {
            x = x
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            WorkerState {
                id: 1_000 + i as u64,
                healthy: !x.is_multiple_of(17),
                in_flight: (x >> 8) as u32 % 16,
                running: (x >> 16) as u32 % 12,
                queued: (x >> 24) as u32 % 4,
                kv_permille: ((x >> 32) % 900) as u16,
                live_rate: 1_000 + ((x >> 40) % 3_000) as u32,
                sampled_at_micros: 990_000 + (x >> 48) % 9_000,
            }
        })
        .collect()
}

fn build_view(fleet: &[WorkerState], out: &mut Vec<Candidate>) {
    out.clear();
    for (i, w) in fleet.iter().enumerate() {
        out.push(Candidate {
            position: i as u32,
            id: WorkerId(w.id),
            health: Health {
                healthy: w.healthy,
                ..Health::default()
            },
            in_flight: w.in_flight,
            load: Some(LoadRow {
                running: w.running,
                queued: w.queued,
                kv_usage_permille: w.kv_permille,
                live_rate: w.live_rate,
                report_age_ms: 5,
                source: SignalSource::Events,
                ..LoadRow::default()
            }),
            load_sampled_at_micros: w.sampled_at_micros,
            ..Candidate::default()
        });
    }
}

fn engine_of(preset: &str) -> Engine {
    PolicySpec::preset(preset)
        .expect("a preset of this build")
        .validate()
        .expect("a preset validates")
}

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
