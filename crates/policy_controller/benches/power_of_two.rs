//! The cost of one decision under each form of the `power_of_two` preset, at
//! 8, 64 and 512 candidates: the expected wait of two drawn workers, the
//! fewer in flight of two drawn workers, the fewest in flight over every
//! worker, and the least resident work over every worker, that last one
//! also with a dispatch credit held per row, as between two reports under
//! load, and the expected wait once more over reports that state no drain
//! rate. Ids: `policy_engine/power_of_two/<form>/<n>`. The fleet is the one
//! the `decision` bench decides over, with the resident tokens a report of
//! that fleet would carry; the decision runs with a fresh seed per iteration
//! so the drawing forms draw.
//! The same fleet supplies the `policy_engine/least_cost/<form>/<n>` cells.
//! They measure a sum of resident tokens, queued requests and requests in
//! flight, the sum with a wide margin, and the sum with every report missing.
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
    document::PolicySpec,
    engine::{Candidate, Constraints, Engine, PoolFacts, RequestFacts, Scratch, Target},
    policies::power_of_two::PowerOfTwoMetric,
    stages::{Lifecycle, StageRegistry},
    tracking::{Lease, Side},
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
    // every sample instant, so no decision releases them. Each size takes
    // an engine of its own: the fleet's ids repeat from size to size, and
    // one engine would keep a smaller size's credits.
    for &n in &SIZES {
        let engine = form_of(RESIDENT);
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
    // The expected-wait form over reports that state no drain rate: no
    // drawn row's own rate prices it, so every decision reads the mean
    // rate of the kept rows.
    let engine = engine_of("power_of_two");
    for &n in &SIZES {
        let mut view = view_of(n);
        for c in &mut view {
            if let Some(row) = &mut c.load {
                row.live_rate = 0;
            }
        }
        decide_over(&mut group, "expected_wait_unrated", &engine, &view);
    }
    group.finish();
}

fn least_cost(c: &mut Criterion) {
    if !StageRegistry::builtin()
        .scorers()
        .any(|name| name == "least-cost")
    {
        return;
    }
    let mut group = c.benchmark_group("policy_engine/least_cost");
    for (form, margin, reports) in [
        ("sum", 0, true),
        ("wide_tie", 83_400, true),
        ("unreported", 0, false),
    ] {
        let mut spec = PolicySpec::power_of_two(RESIDENT).expect("a form of the preset");
        let profile = spec.profiles.values_mut().next().expect("one profile");
        profile.score = serde_yaml::from_str(&format!(
            "- least-cost:\n    terms: {{load.resident_tokens: 0.75, load.queued: 43, candidate.in_flight: 177}}\n    margin: {{absolute: {margin}}}\n    unreported: {{over-least: {{request_cost: 211}}}}"
        )).expect("a least-cost scorer");
        let engine = spec.validate().expect("the form validates");
        for &n in &SIZES {
            let mut view = view_of(n);
            if !reports {
                for row in &mut view {
                    row.load = None;
                }
            }
            decide_over(&mut group, form, &engine, &view);
        }
    }
    group.finish();
}

criterion_group!(benches, forms, least_cost);
criterion_main!(benches);
