//! The row every preset's decision is timed by: the preset's own decision
//! over the synthetic fleet at 8, 64 and 512 candidates, with a fresh seed
//! an iteration so the drawing pickers draw. One body for two binaries: the
//! `decision` bench times every preset but `bucket` by it and the `bucket`
//! bench times that one, so a preset's row is built alike wherever it
//! stands.

use std::hint::black_box;

use criterion::{measurement::WallTime, BenchmarkGroup, BenchmarkId};
use smg_policy_controller::engine::{Constraints, PoolFacts, RequestFacts, Scratch};

use crate::common::{build_view, engine_of, fleet};

/// The candidates a row decides over.
pub const SIZES: [usize; 3] = [8, 64, 512];

/// `policy_engine/decide/<preset>/<n>` at each of [`SIZES`], in `group`.
pub fn rows(group: &mut BenchmarkGroup<'_, WallTime>, preset: &str) {
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
            reported: true,
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
