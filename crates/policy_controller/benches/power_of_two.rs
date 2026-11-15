//! The cost of one decision under each form of the `power_of_two` preset, at
//! 8, 64 and 512 candidates: the expected wait of two drawn workers, the
//! fewer in flight of two drawn workers, the fewest in flight over every
//! worker, and the least resident work over every worker, that last one
//! also with a dispatch credit held per row, as between two reports under
//! load. Ids: `policy_engine/power_of_two/<form>/<n>`. The fleet is the one
//! the `decision` bench decides over, with the resident tokens a report of
//! that fleet would carry; the decision runs with a fresh seed per iteration
//! so the drawing forms draw.
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer; a form that fails to validate is a bug the unit tests catch"
)]

mod common;

use std::hint::black_box;

use common::{build_view, engine_of, fleet};
use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion,
};
use smg_policy_controller::{
    Candidate, Constraints, Engine, Lease, Lifecycle, PolicySpec, PoolFacts, PowerOfTwoMetric,
    RequestFacts, Scratch, Side, Target,
};

const SIZES: [usize; 3] = [8, 64, 512];

/// What the resident form prices a request at, in context tokens.
const RESIDENT: PowerOfTwoMetric = PowerOfTwoMetric::ResidentTokens {
    request_tokens: 2_048,
};

fn form_of(metric: PowerOfTwoMetric) -> Engine {
    PolicySpec::power_of_two(metric)
        .expect("a form of the preset")
        .validate()
        .expect("the form validates")
}

/// The fleet's view of engines that count resident tokens: every row counts
/// them, a context of a few thousand tokens a request, different on each
/// worker.
fn view_of(n: usize) -> Vec<Candidate> {
    let fleet = fleet(n);
    let mut view = Vec::with_capacity(n);
    build_view(&fleet, &mut view);
    for c in &mut view {
        if let Some(row) = &mut c.load {
            row.resident_tokens = row.running * (3_000 + c.key % 2_000);
            row.resident_counted = true;
        }
    }
    view
}

fn decide_over(
    group: &mut BenchmarkGroup<'_, WallTime>,
    form: &str,
    engine: &Engine,
    view: &[Candidate],
) {
    let n = view.len();
    let mut scratch = Scratch::new();
    let pool = PoolFacts {
        pool_size: n as u32,
        generation: 1,
        ..PoolFacts::default()
    };
    let cx = Constraints::default();
    group.bench_with_input(BenchmarkId::new(form, n), &n, |b, _| {
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
            engine.decide(&facts, &pool, black_box(view), &cx, &mut scratch)
        });
    });
}

fn forms(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_engine/power_of_two");
    for (form, engine) in [
        // The preset under its bare name is the expected-wait form.
        ("expected_wait", engine_of("power_of_two")),
        ("requests", form_of(PowerOfTwoMetric::Requests)),
        ("least_requests", form_of(PowerOfTwoMetric::LeastRequests)),
        ("resident_tokens", form_of(RESIDENT)),
    ] {
        for &n in &SIZES {
            decide_over(&mut group, form, &engine, &view_of(n));
        }
    }
    // The resident form with one dispatch credit per row in the tally: every
    // decision reads each reporting row's entry. The credits date after
    // every sample instant, so no decision releases them.
    let engine = form_of(RESIDENT);
    for &n in &SIZES {
        let view = view_of(n);
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
        decide_over(&mut group, "resident_tokens_credited", &engine, &view);
    }
    group.finish();
}

criterion_group!(benches, forms);
criterion_main!(benches);
