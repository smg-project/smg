//! The engine's cost per decision: the view build and the decision at 8, 64
//! and 512 candidates, per preset, the least-load decision with a credit
//! held per row, the steer of a pool over its thresholds on every row, a
//! keyed decision on its pin and off it, and the spec's parse-and-validate
//! time at apply. Ids: `policy_engine/view_build/<n>`,
//! `policy_engine/decide/<preset>/<n>`,
//! `policy_engine/decide/least_load_credited/<n>`,
//! `policy_engine/decide/all_vetoed/<n>`,
//! `policy_engine/decide/manual_keyed_hit/8`,
//! `policy_engine/decide/manual_keyed_miss/8` and
//! `policy_engine/spec/parse_validate`. The view is built from the synthetic
//! per-worker state of `common` the way a host reads its routing state; the
//! decision runs with a fresh seed per iteration so the drawing pickers draw.
//! This file is the pull-request gate's (`.github/bench-gates/policy-engine.toml`
//! runs the whole binary three times a side): the `epd` plan's encode
//! decisions and the ring's membership costs are the `epd` bench target, and
//! the `bucket` preset's rows the `bucket` bench target.
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer; a preset that fails to validate is a bug the unit tests catch"
)]

mod common;
#[path = "common/placed.rs"]
mod placed;

use std::hint::black_box;

use common::{build_view, engine_of, fleet};
use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion,
    Throughput,
};
use placed::page_start_buffer;
use smg_policy_controller::{
    Constraints, Decision, Identity, Lease, Lifecycle, Lifted, ModelFingerprint, PolicySpec,
    PoolFacts, RequestFacts, Scratch, Side, Target, PRESETS,
};

const SIZES: [usize; 3] = [8, 64, 512];

fn view_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_engine/view_build");
    for &n in &SIZES {
        let fleet = fleet(n);
        // Where a page boundary falls in the rows' buffer is an input of
        // the build's time (`placed`): the timed buffer stands on a page's
        // start, on both sides of the gate.
        let mut out = page_start_buffer(n);
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
    // The `bucket` preset's rows stand in the `bucket` bench target with the
    // rest of its cells. This binary runs at its gate's bound, and a preset
    // more here is three rows more in every pull request's gate.
    for &preset in PRESETS.iter().filter(|&&preset| preset != "bucket") {
        let engine = engine_of(preset);
        for &n in &SIZES {
            let fleet = fleet(n);
            let mut view = Vec::with_capacity(n);
            build_view(&fleet, &mut view);
            let mut scratch = Scratch::new();
            let pool = PoolFacts {
                pool_size: n as u32,
                generation: 1,
                // Every row of the bench's fleet reports a rate.
                rated: true,
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
    // The all-vetoed rule's own path: every row over its thresholds, so the
    // profile's filters leave nobody, run once more with the overload veto
    // lifted, and the decision steers to the fewest requests in flight.
    //
    // The ids stand only on an engine that steers such a pool. The bench
    // gate builds its base side from the base's engine and these sources;
    // an engine without the rule sheds the pool, and a shed is no base to
    // time a steer against. That side lists no such id, and the gate reads
    // an id with no base as new.
    let engine = engine_of("least_load");
    for &n in &SIZES {
        let fleet = fleet(n);
        let mut view = Vec::with_capacity(n);
        build_view(&fleet, &mut view);
        for row in &mut view {
            row.health.overloaded = true;
        }
        let mut scratch = Scratch::new();
        let pool = PoolFacts {
            pool_size: n as u32,
            generation: 1,
            ..PoolFacts::default()
        };
        let cx = Constraints::default();
        let asked = RequestFacts {
            model: "m",
            prompt_tokens: 512,
            now_micros: 1_000_000,
            ..RequestFacts::default()
        };
        let steers = matches!(
            engine.decide(&asked, &pool, &view, &cx, &mut scratch),
            Decision::Pick(pick) if pick.trace.lifted == Lifted::Overload
        );
        if !steers {
            continue;
        }
        group.bench_with_input(BenchmarkId::new("all_vetoed", n), &n, |b, _| {
            let mut seed = 0u64;
            b.iter(|| {
                seed += 1;
                let facts = RequestFacts { seed, ..asked };
                engine.decide(&facts, &pool, black_box(&view), &cx, &mut scratch)
            });
        });
    }
    keyed(&mut group);
    group.finish();
}

/// A model name of an ordinary length; every other cell's name is one byte.
/// Both keyed cells carry its fingerprint, as a host that keeps its models
/// does, so neither reads the name's bytes. A host that carries none pays a
/// pass over the name at each keyed decision, which no cell here times.
const KEYED_MODEL: &str = "acme/assistant-large-instruct-v2";

/// A keyed request under the manual preset's sessions rule, at 8 candidates.
/// The hit decides for one key on the pin its dispatch wrote. The miss
/// decides for 64 keys in turn under an idle time of one second with the
/// clock 20 ms on at each decision, so a key's pins have expired by its next
/// turn: the rule's assignment places it and the table records the claim. A
/// sweep is due once a day of that clock and is no part of the cell. Both
/// carry the model's fingerprint, as a host that keeps its models does.
fn keyed(group: &mut BenchmarkGroup<'_, WallTime>) {
    let n = 8;
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

    let model_fingerprint = ModelFingerprint::of(KEYED_MODEL);
    let engine = engine_of("manual");
    let hit = RequestFacts {
        model: KEYED_MODEL,
        model_fingerprint,
        prompt_tokens: 512,
        session_key: Identity::hashed(0x5e55_10f1),
        now_micros: 1_000_000,
        ..RequestFacts::default()
    };
    let decision = engine.decide(&hit, &pool, &view, &cx, &mut scratch);
    let pick = decision.pick().expect("a keyed pick");
    let target = *pick.targets.first().expect("a target");
    let lease = Lease {
        session: hit.session(),
        session_verdict: pick.session_verdict,
        issued_micros: hit.now_micros,
        tokens: hit.prompt_tokens,
        ..Lease::default()
    };
    engine.on_dispatched(&lease, &target, Side::Single, 0);
    group.bench_with_input(BenchmarkId::new("manual_keyed_hit", n), &n, |b, _| {
        let mut seed = 0u64;
        b.iter(|| {
            seed += 1;
            let facts = RequestFacts { seed, ..hit };
            engine.decide(&facts, &pool, black_box(&view), &cx, &mut scratch)
        });
    });

    let mut spec = PolicySpec::preset("manual").expect("the preset");
    let rule = spec
        .bindings
        .first_mut()
        .and_then(|binding| binding.sessions.as_mut())
        .expect("the preset's sessions rule");
    rule.ttl_s = 1;
    rule.sweep_s = 86_400;
    let engine = spec.validate().expect("the preset validates");
    // The clock runs on over the whole cell, the warm-up included: a turn
    // that began again at its first instant would find every key pinned.
    let mut turn = 0u64;
    group.bench_with_input(BenchmarkId::new("manual_keyed_miss", n), &n, |b, _| {
        b.iter(|| {
            turn += 1;
            let facts = RequestFacts {
                model: KEYED_MODEL,
                model_fingerprint,
                prompt_tokens: 512,
                session_key: Identity::hashed(1 + (turn & 63)),
                seed: turn,
                now_micros: 1_000_000 + turn * 20_000,
                ..RequestFacts::default()
            };
            engine.decide(&facts, &pool, black_box(&view), &cx, &mut scratch)
        });
    });
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
