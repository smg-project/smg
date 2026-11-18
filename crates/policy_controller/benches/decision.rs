//! The engine's cost per decision: the view build and the decision at 8, 64
//! and 512 candidates, per preset, the least-load decision with a credit
//! held per row, the steer of a pool over its thresholds on every row, a
//! prefill placement under the bucket preset with its
//! booking, a keyed decision on its pin and off it, and the spec's
//! parse-and-validate time at apply. Ids: `policy_engine/view_build/<n>`,
//! `policy_engine/decide/<preset>/<n>`,
//! `policy_engine/decide/least_load_credited/<n>`,
//! `policy_engine/decide/all_vetoed/<n>`,
//! `policy_engine/decide/bucket_prefill/<n>`,
//! `policy_engine/decide/bucket_prefill_64_models/8`,
//! `policy_engine/decide/bucket_prefill_64_models_shared/8`,
//! `policy_engine/decide/bucket_prefill_joined/512`,
//! `policy_engine/decide/bucket_prefill_two_rows/256`,
//! `policy_engine/decide/manual_keyed_hit/8`,
//! `policy_engine/decide/manual_keyed_miss/8` and
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
    Constraints, Decision, Identity, Lease, Lifecycle, ModelFingerprint, Phase, PolicySpec,
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
            Decision::Pick(_)
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
    bucket_prefill(&mut group);
    bucket_prefill_models(&mut group);
    bucket_prefill_parted(&mut group);
    keyed(&mut group);
    group.finish();
}

/// A prefill placement under the bucket preset, booked as a host books it.
/// Each iteration decides for one request and issues the decision with the
/// clock one millisecond on, so the preset's five-second window holds about
/// five thousand bookings: each decision ages one out, and every
/// five-thousandth reaches an interval's end, where a cut falls due. The
/// first is made from the window, and none after it while each worker's
/// share stays within a factor of two of the last cut's. The sizes step
/// through the span, so the ranges' owners take turns. The preset's own
/// rows decide a request that is no prefill placement, which the stage
/// leaves to the picker. The pull-request gate builds its base side with
/// this file: a build before the preset times no cell here.
fn bucket_prefill(group: &mut BenchmarkGroup<'_, WallTime>) {
    for &n in &SIZES {
        let Ok(spec) = PolicySpec::preset("bucket") else {
            return;
        };
        let engine = spec.validate().expect("the preset validates");
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
        let mut seed = 0u64;
        group.bench_with_input(BenchmarkId::new("bucket_prefill", n), &n, |b, _| {
            b.iter(|| {
                seed += 1;
                let now_micros = 1_000_000 + seed * 1_000;
                let tokens = (seed.wrapping_mul(2_654_435_761) % 4_096) as u32;
                let facts = RequestFacts {
                    model: "m",
                    phase: Phase::Prefill,
                    prompt_tokens: tokens,
                    seed,
                    now_micros,
                    ..RequestFacts::default()
                };
                let decision = engine.decide(&facts, &pool, black_box(&view), &cx, &mut scratch);
                let lease = Lease {
                    tokens,
                    issued_micros: now_micros,
                    ..Lease::default()
                };
                engine.on_decision_issued(&lease, &decision);
                decision
            });
        });
    }
}

/// The same placement where 64 models take turns: each over 8 workers of its
/// own, then every model over the same 8. A decision finds its model's
/// partition by the name and a booking its worker's, so the first cell
/// reads what the count of models adds to a placement. A lease names no
/// model: where the models list the same workers, the second cell books
/// each issued decision in the 64 partitions that list its worker, one
/// guard after the other. A build before the preset times no cell here.
fn bucket_prefill_models(group: &mut BenchmarkGroup<'_, WallTime>) {
    const MODELS: usize = 64;
    let n = 8;
    for (name, shared) in [
        ("bucket_prefill_64_models", false),
        ("bucket_prefill_64_models_shared", true),
    ] {
        let Ok(spec) = PolicySpec::preset("bucket") else {
            return;
        };
        let engine = spec.validate().expect("the preset validates");
        let fleet = fleet(MODELS * n);
        let mut rows = Vec::with_capacity(MODELS * n);
        build_view(&fleet, &mut rows);
        // A model's view is 8 rows, each at its place in that view: its own
        // 8, or the fleet's first 8 where the models list the same workers.
        let views: Vec<(String, Vec<_>)> = rows
            .chunks(n)
            .enumerate()
            .map(|(at, own)| {
                let mut view = if shared {
                    rows[..n].to_vec()
                } else {
                    own.to_vec()
                };
                for (position, row) in view.iter_mut().enumerate() {
                    row.position = position as u32;
                }
                (format!("model-{at:02}"), view)
            })
            .collect();
        let mut scratch = Scratch::new();
        let pool = PoolFacts {
            pool_size: n as u32,
            generation: 1,
            ..PoolFacts::default()
        };
        let cx = Constraints::default();
        let mut seed = 0u64;
        group.bench_with_input(BenchmarkId::new(name, n), &n, |b, _| {
            b.iter(|| {
                seed += 1;
                let now_micros = 1_000_000 + seed * 1_000;
                let tokens = (seed.wrapping_mul(2_654_435_761) % 4_096) as u32;
                let (model, view) = &views[seed as usize % MODELS];
                let facts = RequestFacts {
                    model: model.as_str(),
                    phase: Phase::Prefill,
                    prompt_tokens: tokens,
                    seed,
                    now_micros,
                    ..RequestFacts::default()
                };
                let decision = engine.decide(&facts, &pool, black_box(view), &cx, &mut scratch);
                let lease = Lease {
                    tokens,
                    issued_micros: now_micros,
                    ..Lease::default()
                };
                engine.on_decision_issued(&lease, &decision);
                decision
            });
        });
    }
}

/// Two views that part from the listing's order, which a placement must not
/// pay for by a search of the listing: at 512 workers one that joined in the
/// middle of the view, and at 256 workers two rows a worker. Every row past
/// the parting is found by its worker's id. A build before the preset times
/// no cell here.
fn bucket_prefill_parted(group: &mut BenchmarkGroup<'_, WallTime>) {
    let fleet = fleet(512);
    let mut whole = Vec::with_capacity(512);
    build_view(&fleet, &mut whole);
    // The view before the join: every row but the middle one.
    let mut before = whole.clone();
    before.remove(256);
    // Two rows a worker: each of the first 256 workers as two ranks.
    let mut twice = Vec::with_capacity(512);
    for row in &whole[..256] {
        for rank in 0..2 {
            let mut row = *row;
            row.dp_rank = rank;
            row.dp_size = 2;
            twice.push(row);
        }
    }
    for view in [&mut before, &mut twice] {
        for (position, row) in view.iter_mut().enumerate() {
            row.position = position as u32;
        }
    }
    for (name, workers, listed, view) in [
        ("bucket_prefill_joined", 512usize, &before, &whole),
        ("bucket_prefill_two_rows", 256usize, &twice, &twice),
    ] {
        let Ok(spec) = PolicySpec::preset("bucket") else {
            return;
        };
        let engine = spec.validate().expect("the preset validates");
        let mut scratch = Scratch::new();
        let cx = Constraints::default();
        let facts_of = |tokens: u32, seed: u64, now_micros: u64| RequestFacts {
            model: "m",
            phase: Phase::Prefill,
            prompt_tokens: tokens,
            seed,
            now_micros,
            ..RequestFacts::default()
        };
        let pool_of = |rows: usize| PoolFacts {
            pool_size: rows as u32,
            generation: 1,
            ..PoolFacts::default()
        };
        // One decision lists the model's workers in the order of `listed`.
        black_box(engine.decide(
            &facts_of(0, 0, 1_000_000),
            &pool_of(listed.len()),
            listed,
            &cx,
            &mut scratch,
        ));
        let pool = pool_of(view.len());
        let mut seed = 0u64;
        group.bench_with_input(BenchmarkId::new(name, workers), &workers, |b, _| {
            b.iter(|| {
                seed += 1;
                let now_micros = 1_000_000 + seed * 1_000;
                let tokens = (seed.wrapping_mul(2_654_435_761) % 4_096) as u32;
                let facts = facts_of(tokens, seed, now_micros);
                let decision = engine.decide(&facts, &pool, black_box(view), &cx, &mut scratch);
                let lease = Lease {
                    tokens,
                    issued_micros: now_micros,
                    ..Lease::default()
                };
                engine.on_decision_issued(&lease, &decision);
                decision
            });
        });
    }
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
