//! The cost of reading signals beside the rows: one decision under a
//! scorer a host registers, which reads two signals of every row, at 8, 64
//! and 512 rows. Ids: `policy_engine/signals/read_two/<n>`. The cells are
//! handed by hand, as a host that keeps no store hands them.

#![expect(
    clippy::expect_used,
    reason = "the benchmark requires a valid policy to measure a decision"
)]

#[path = "common/reads_two.rs"]
mod reads_two;

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use smg_policy_controller::{
    document::{PolicySpec, API_VERSION},
    engine::{Candidate, Constraints, Engine, PoolFacts, RequestFacts, Scratch, WorkerId},
    load::{SignalCell, SignalCells},
    stages::StageRegistry,
};

use crate::reads_two::ReadsTwo;

/// The decisions' instant on the host's clock.
const NOW: u64 = 1_000_000;

fn engine() -> Engine {
    let mut catalog = StageRegistry::builtin();
    catalog
        .scorer_with("reads-two", |build| {
            let used = build.signal_column("used")?;
            let pending = build.signal_column("pending")?;
            Ok(Box::new(ReadsTwo(used, pending)))
        })
        .expect("a free name");
    let document: PolicySpec = serde_json::from_value(serde_json::json!({
        "apiVersion": API_VERSION,
        "signals": {
            "used": { "unit": "tokens", "max_age_ms": 4000 },
            "pending": { "unit": "tokens" }
        },
        "profiles": {
            "work": { "score": ["reads-two"], "pick": "best" }
        }
    }))
    .expect("a document");
    document.validate_in(&catalog, &[]).expect("validates")
}

fn signals(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_engine/signals");
    for n in [8u32, 64, 512] {
        let rows: Vec<_> = (0..n)
            .map(|position| Candidate {
                position,
                id: WorkerId(1003 + u64::from(position)),
                ..Candidate::default()
            })
            .collect();
        // Two cells a row, in the names' order: `pending`, then `used`.
        let cell = |value: u32| SignalCell {
            value: f64::from(value),
            sampled_at_micros: NOW - 1_000,
        };
        let cells: Vec<_> = (0..n)
            .flat_map(|row| [cell(row % 7), cell(row % 11)])
            .collect();
        let cells = SignalCells::new(cells, 2);
        let pool = PoolFacts {
            pool_size: n,
            ..PoolFacts::default()
        };
        let constraints = Constraints::default();
        let engine = engine();
        let mut scratch = Scratch::new();
        group.bench_function(BenchmarkId::new("read_two", n), |b| {
            let mut seed = 0u64;
            b.iter(|| {
                seed += 1;
                let facts = RequestFacts {
                    seed,
                    now_micros: NOW,
                    signal_cells: Some(&cells),
                    ..RequestFacts::default()
                };
                engine.decide(&facts, &pool, black_box(&rows), &constraints, &mut scratch)
            });
        });
    }
    group.finish();
}

criterion_group!(benches, signals);
criterion_main!(benches);
