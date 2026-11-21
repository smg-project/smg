//! The cost of reading each row's load report from a registered scorer,
//! over a whole pool. A registered stage lives in another crate, as this
//! bench does, so these cells count the read across that boundary. The
//! `from_row` cells read `row.load` as it stands. The `reader` cells read
//! through the view's reading ([`LoadReportReader`]) under a maximum report
//! age, and the `reader_any_age` cells under a view that sets none.

#![expect(
    clippy::expect_used,
    reason = "the benchmark requires a valid policy to measure a decision"
)]

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use smg_policy_controller::{
    document::{PolicySpec, API_VERSION},
    engine::{
        Candidate, Constraints, Engine, LoadRow, PoolFacts, RequestFacts, Scorer, Scratch, TieBand,
        WorkerId,
    },
    load::LoadReportReader,
    stages::{Lifecycle, StageRegistry},
};

const MAX_AGE_MS: u64 = 250;
const NOW: u64 = 7_400_000;
/// Every row's report is younger than the maximum, so each read finds one.
const SAMPLED_AT: u64 = NOW - 100_000;

fn score_of(report: Option<&LoadRow>) -> f32 {
    1.0 / (1.0 + report.map_or(0, |report| report.running) as f32)
}

/// Scores a row by its report's running requests, as the row holds it.
#[derive(Debug)]
struct FromRow;

impl Lifecycle for FromRow {}

impl Scorer for FromRow {
    fn name(&self) -> &'static str {
        "from-row"
    }

    fn score(
        &self,
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        for (&index, score) in kept.iter().zip(out) {
            *score = score_of(candidates[index as usize].load.as_ref());
        }
        TieBand::NORMALIZED
    }
}

/// The same score, the report read through the view's reading.
#[derive(Debug)]
struct ThroughReader(LoadReportReader);

impl Lifecycle for ThroughReader {}

impl Scorer for ThroughReader {
    fn name(&self) -> &'static str {
        "through-reader"
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        let reports = self.0.pass(facts);
        for (&index, score) in kept.iter().zip(out) {
            *score = score_of(reports.of(&candidates[index as usize]));
        }
        TieBand::NORMALIZED
    }
}

/// A profile of the registered scorer `name` alone, under a view with
/// `max_age_ms`.
fn engine(name: &str, max_age_ms: Option<u64>) -> Engine {
    let mut catalog = StageRegistry::builtin();
    catalog
        .scorer("from-row", || Box::new(FromRow))
        .expect("a free name")
        .scorer_with("through-reader", |build| {
            Ok(Box::new(ThroughReader(build.load_report_reader())))
        })
        .expect("a free name");
    let document: PolicySpec = serde_json::from_value(serde_json::json!({
        "apiVersion": API_VERSION,
        "view": { "max_report_age_ms": max_age_ms },
        "profiles": { "work": { "score": [name], "pick": "best" } }
    }))
    .expect("a document");
    document.validate_in(&catalog, &[]).expect("validates")
}

fn reports(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_reports");
    let cells = [
        ("from_row", "from-row", Some(MAX_AGE_MS)),
        ("reader", "through-reader", Some(MAX_AGE_MS)),
        ("reader_any_age", "through-reader", None),
    ];
    for n in [8, 64, 512] {
        let rows: Vec<_> = (0..n)
            .map(|position| Candidate {
                position,
                id: WorkerId(2003 + u64::from(position)),
                load: Some(LoadRow {
                    running: position % 7,
                    ..LoadRow::default()
                }),
                load_sampled_at_micros: SAMPLED_AT,
                ..Candidate::default()
            })
            .collect();
        let pool = PoolFacts {
            pool_size: n,
            ..PoolFacts::default()
        };
        let facts = RequestFacts {
            now_micros: NOW,
            ..RequestFacts::default()
        };
        let constraints = Constraints::default();
        for (cell, name, max_age_ms) in cells {
            let engine = engine(name, max_age_ms);
            let mut scratch = Scratch::new();
            group.bench_function(BenchmarkId::new(cell, n), |b| {
                b.iter(|| {
                    engine.decide(&facts, &pool, black_box(&rows), &constraints, &mut scratch)
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, reports);
criterion_main!(benches);
