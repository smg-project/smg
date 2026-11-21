//! The cost of one decision under the presets whose pickers draw first
//! (`passthrough`, `random`, `round_robin`, `power_of_two`), at 4, 8, 16, 32,
//! 64 and 512 candidates over three fleets: every row ready (`ready`), one row
//! in seventeen not ready (`seventeenth`) and every second row not ready
//! (`half`). Ids: `policy_engine/draw/<preset>/<fleet>/<n>`. A pool of
//! sixteen rows or more is decided by the rows its picker draws, and a
//! smaller one by the whole pass: the cells at 4 and 8 hold the pass, the
//! `ready` and `seventeenth` cells from 16 hold the draw as a fleet in
//! service meets it, and the `half` cells hold what the draw pays where
//! every second row it asks about is refused. The loads are the `decision`
//! bench's fleet; the decision runs with a fresh seed per iteration so the
//! drawing pickers draw.
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer; a preset that fails to validate is a bug the unit tests catch"
)]

mod common;

use std::hint::black_box;

use common::{build_view, engine_of, fleet};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use smg_policy_controller::engine::{Candidate, Constraints, PoolFacts, RequestFacts, Scratch};

const SIZES: [usize; 6] = [4, 8, 16, 32, 64, 512];

const DRAWING: [&str; 4] = ["passthrough", "random", "round_robin", "power_of_two"];

/// Whether the row at a position is not ready.
type Refused = fn(usize) -> bool;

/// A fleet's name and the rows of it that are not ready.
const FLEETS: [(&str, Refused); 3] = [
    ("ready", |_| false),
    ("seventeenth", |row| row % 17 == 5),
    ("half", |row| row % 2 == 1),
];

fn view_of(n: usize, refused: Refused) -> Vec<Candidate> {
    let mut view = Vec::with_capacity(n);
    build_view(&fleet(n), &mut view);
    for (row, candidate) in view.iter_mut().enumerate() {
        candidate.health.healthy = !refused(row);
    }
    view
}

fn draw(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_engine/draw");
    for preset in DRAWING {
        let engine = engine_of(preset);
        for (name, refused) in FLEETS {
            for &n in &SIZES {
                let view = view_of(n, refused);
                let mut scratch = Scratch::new();
                let pool = PoolFacts {
                    pool_size: n as u32,
                    generation: 1,
                    ..PoolFacts::default()
                };
                let cx = Constraints::default();
                let id = BenchmarkId::new(format!("{preset}/{name}"), n);
                group.bench_with_input(id, &n, |b, _| {
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
    }
    group.finish();
}

criterion_group!(benches, draw);
criterion_main!(benches);
