//! The cost of scoring with a profile's send tally over a whole pool. Each
//! cell reads rows with no sends, a send held per row, or a send per row
//! that reaches its time limit at the decision. The `policy_sent` cells read
//! the requests and the stated tokens from a registered scorer, the
//! `policy_sent_whole` cells all three numbers, the
//! `policy_engine/least_cost_sent` cells through `least-cost` terms. The
//! `policy_sent/dispatch` cells read
//! what 64 dispatch reports cost a profile of `least-load`, alone and
//! beside a registered scorer that reads the same sends.

#![expect(
    clippy::expect_used,
    reason = "the benchmark requires a valid policy to measure a decision"
)]

use std::{hint::black_box, sync::Arc};

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use smg_policy_controller::{
    document::{PolicySpec, API_VERSION},
    engine::{
        Candidate, Constraints, Engine, PoolFacts, RequestFacts, Scorer, Scratch, Target, TieBand,
        WorkerId,
    },
    stages::{Lifecycle, StageRegistry},
    tracking::{DispatchCredit, Lease, Side},
};

const TTL_MS: u32 = 53;
const SENT_AT: u64 = 234_100;

#[derive(Debug)]
struct SentScore(Arc<DispatchCredit>);

impl Lifecycle for SentScore {}

impl Scorer for SentScore {
    fn name(&self) -> &'static str {
        "sent-score"
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        let reads = self.0.pass(facts);
        for (&index, score) in kept.iter().zip(out) {
            let sent = reads.counts(&candidates[index as usize]);
            *score = 1.0 / (1.0 + sent.requests as f32 + sent.tokens as f32 / 1024.0);
        }
        TieBand::NORMALIZED
    }
}

/// A registered scorer that prices a send of no stated size too: it reads
/// all three numbers.
#[derive(Debug)]
struct SentWhole(Arc<DispatchCredit>);

impl Lifecycle for SentWhole {}

impl Scorer for SentWhole {
    fn name(&self) -> &'static str {
        "sent-whole"
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        let reads = self.0.pass(facts);
        for (&index, score) in kept.iter().zip(out) {
            let sent = reads.of(&candidates[index as usize]);
            let tokens = sent.tokens as f32 / 1024.0 + sent.unsized_requests as f32;
            *score = 1.0 / (1.0 + sent.requests as f32 + tokens);
        }
        TieBand::NORMALIZED
    }
}

/// A profile of the registered scorer `name` alone, under the time limit.
fn registered(name: &str) -> Engine {
    let mut catalog = StageRegistry::builtin();
    catalog
        .scorer_with("sent-score", |build| {
            Ok(Box::new(SentScore(build.dispatch_credit())))
        })
        .expect("a free name")
        .scorer_with("sent-whole", |build| {
            Ok(Box::new(SentWhole(build.dispatch_credit())))
        })
        .expect("a free name");
    let document: PolicySpec = serde_json::from_value(serde_json::json!({
        "apiVersion": API_VERSION,
        "profiles": {
            "work": { "sent_ttl_ms": TTL_MS, "score": [name], "pick": "best" }
        }
    }))
    .expect("a document");
    document.validate_in(&catalog, &[]).expect("validates")
}

fn engine() -> Engine {
    registered("sent-score")
}

fn whole() -> Engine {
    registered("sent-whole")
}

fn least_cost() -> Engine {
    let document: PolicySpec = serde_json::from_value(serde_json::json!({
        "apiVersion": API_VERSION,
        "profiles": {
            "work": {
                "sent_ttl_ms": TTL_MS,
                "score": [{ "least-cost": { "terms": {
                    "candidate.in_flight": 177, "sent.requests": 1000, "sent.tokens": 1
                } } }],
                "pick": "best"
            }
        }
    }))
    .expect("a document");
    document.validate().expect("validates")
}

fn fill(engine: &Engine, rows: &[Candidate]) {
    let lease = Lease {
        issued_micros: SENT_AT,
        tokens: 379,
        ..Lease::default()
    };
    for row in rows {
        let target = Target {
            id: row.id,
            dp_rank: row.dp_rank,
            ..Target::default()
        };
        engine.on_dispatched(&lease, &target, Side::Single, 0);
    }
}

/// The three cells of one group. The rows' counts in flight repeat every
/// `in_flight_cycle` rows: a cycle of one leaves every row at zero.
fn cells(c: &mut Criterion, group: &str, engine: fn() -> Engine, in_flight_cycle: u32) {
    let mut group = c.benchmark_group(group);
    for n in [8, 64, 512] {
        let rows: Vec<_> = (0..n)
            .map(|position| Candidate {
                position,
                id: WorkerId(1003 + u64::from(position)),
                in_flight: position % in_flight_cycle,
                ..Candidate::default()
            })
            .collect();
        let pool = PoolFacts {
            pool_size: n,
            ..PoolFacts::default()
        };
        let facts = RequestFacts {
            now_micros: SENT_AT + 911,
            ..RequestFacts::default()
        };
        let constraints = Constraints::default();
        let engine = engine();
        let mut scratch = Scratch::new();
        group.bench_function(BenchmarkId::new("no_sends", n), |b| {
            b.iter(|| engine.decide(&facts, &pool, black_box(&rows), &constraints, &mut scratch));
        });
        fill(&engine, &rows);
        group.bench_function(BenchmarkId::new("held_sends", n), |b| {
            b.iter(|| engine.decide(&facts, &pool, black_box(&rows), &constraints, &mut scratch));
        });

        let facts = RequestFacts {
            now_micros: SENT_AT + u64::from(TTL_MS) * 1000,
            ..facts
        };
        // The cutoff empties each row, so put the send back before timing
        // the next decision.
        let _ = engine.decide(&facts, &pool, &rows, &constraints, &mut scratch);
        group.bench_function(BenchmarkId::new("expired_sends", n), |b| {
            b.iter_batched(
                || fill(&engine, &rows),
                |()| engine.decide(&facts, &pool, black_box(&rows), &constraints, &mut scratch),
                BatchSize::PerIteration,
            );
        });
    }
    group.finish();
}

/// A profile of `least-load`, with the registered reader beside it where
/// `reader` says so.
fn least_load(reader: bool) -> Engine {
    let mut catalog = StageRegistry::builtin();
    catalog
        .scorer_with("sent-score", |build| {
            Ok(Box::new(SentScore(build.dispatch_credit())))
        })
        .expect("a free name");
    let score = if reader {
        serde_json::json!(["least-load", "sent-score"])
    } else {
        serde_json::json!(["least-load"])
    };
    let document: PolicySpec = serde_json::from_value(serde_json::json!({
        "apiVersion": API_VERSION,
        "profiles": { "work": { "score": score, "pick": "best" } }
    }))
    .expect("a document");
    document.validate_in(&catalog, &[]).expect("validates")
}

/// One dispatch report to each target. Out of line, so a count by
/// function can name the timed part.
#[inline(never)]
fn tell_each(engine: &Engine, lease: &Lease, targets: &[Target]) {
    for target in targets {
        engine.on_dispatched(lease, target, Side::Single, 0);
    }
}

/// A timed pass tells 64 dispatch reports, one a rank. Its engine is new
/// and heard one report a rank ahead of the pass, so no timed report
/// enters a rank and each rank holds two sends at the pass's end.
fn dispatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_sent/dispatch");
    let lease = Lease {
        issued_micros: SENT_AT,
        tokens: 379,
        ..Lease::default()
    };
    let targets: Vec<_> = (0..64)
        .map(|position| Target {
            id: WorkerId(1003 + position),
            ..Target::default()
        })
        .collect();
    for (name, reader) in [("least_load", false), ("least_load_and_reader", true)] {
        group.bench_function(name, |b| {
            b.iter_batched_ref(
                || {
                    let engine = least_load(reader);
                    for target in &targets {
                        engine.on_dispatched(&lease, target, Side::Single, 0);
                    }
                    engine
                },
                |engine| tell_each(engine, black_box(&lease), &targets),
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

fn sent(c: &mut Criterion) {
    cells(c, "policy_sent", engine, 1);
    cells(c, "policy_sent_whole", whole, 1);
    cells(c, "policy_engine/least_cost_sent", least_cost, 17);
    dispatch(c);
}

criterion_group!(benches, sent);
criterion_main!(benches);
