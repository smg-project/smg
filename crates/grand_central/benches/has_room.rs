//! The cost of one decision under a `has-room` rule the document writes, at
//! 8, 64 and 512 candidates, beside the same profile without the rule. A
//! profile that names the rule folds into no single test of a row: its
//! filters are asked one at a time. Two profiles say what that costs.
//! `scored` ranks the kept rows by `least-load` and picks the best, so its
//! filters pass over the whole pool. `drawn` picks by `power-of-two`, which
//! draws first and asks the filters of the rows it drew.
//!
//! Each stands `without` the rule, `with_rule`, and `with_sends`: the rule
//! over rows that each hold a send its count has not let go.
//! `with_signals` and `with_signal_sends` are the same two under a rule
//! that names signals: one in its sum, one as its capacity. Each row's
//! cells state what its report states, so both rules keep the same rows.
//! Ids: `engine/has_room/<profile>_<form>/<n>`.
//!
//! The fleet is the `decision` bench's with a resident count in every
//! report, so that about one row in six is over the rule's bound. The
//! decision runs with a fresh seed per iteration. A bench target of its
//! own: the `decision` binary takes no new id.

#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer; a document that fails to validate is a bug the unit tests catch"
)]

#[expect(
    dead_code,
    reason = "this bench builds its engine from a document; `common::engine_of` serves the decision and contention benches"
)]
mod common;

use std::hint::black_box;

use common::{build_view, fleet};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use smg_grand_central::{
    document::{PolicySpec, API_VERSION},
    engine::{Candidate, Constraints, Engine, PoolFacts, RequestFacts, Scratch, Target},
    load::{SignalCell, SignalCells},
    stages::Lifecycle,
    tracking::{Lease, Side},
};

const SIZES: [usize; 3] = [8, 64, 512];

/// The rule a document writes: a row holds its resident tokens, the tokens
/// sent to it since its report and the request, and can hold two hundred
/// thousand less a twentieth.
const RULE: &str = r#"{"has-room":{"used":{"load.resident_tokens":1,"sent.tokens":1,"request.prompt_tokens":1},"capacity":200000,"headroom":0.05}}"#;

/// The rule under signals. A row holds what its signal says is used, the
/// tokens sent to it since its report, and the request. It can hold its
/// signal's capacity less a twentieth.
const SIGNAL_RULE: &str = r#"{"has-room":{"used":{"signal.kv_tokens_used":1,"sent.tokens":1,"request.prompt_tokens":1},"capacity":"signal.kv_tokens_capacity","headroom":0.05}}"#;

/// The two signals [`SIGNAL_RULE`] names, as a policy spec declares them.
/// In name order the capacity is column 0 and what is used column 1.
const SIGNALS: &str =
    r#""signals":{"kv_tokens_capacity":{"unit":"tokens"},"kv_tokens_used":{"unit":"tokens"}},"#;

/// The engine of a document with one profile: three stock filters, the rule
/// among them where `with_rule`, the `least-load` score, and the picker:
/// `power-of-two` where `drawn`, `best` where not.
fn policy(with_rule: bool, drawn: bool) -> Engine {
    policy_of("", if with_rule { RULE } else { "" }, drawn)
}

/// [`policy`] with `rule` among the filters, or none where it is empty,
/// under a policy spec whose head holds `signals`.
fn policy_of(signals: &str, rule: &str, drawn: bool) -> Engine {
    let rule = if rule.is_empty() {
        String::new()
    } else {
        format!(",{rule}")
    };
    let pick = if drawn { "power-of-two" } else { "best" };
    let document: PolicySpec = serde_json::from_str(&format!(
        r#"{{"apiVersion":"{API_VERSION}",{signals}"profiles":{{"p":{{"filter":["healthy","not-overloaded"{rule},"not-tried"],"score":["least-load"],"pick":"{pick}"}}}}}}"#
    ))
    .expect("a policy document");
    document.validate().expect("a valid policy")
}

/// The bench's fleet with a resident count in every report, a share of the
/// row's cache use: a row past three quarters of that use is over the
/// rule's bound.
fn rows(n: usize) -> Vec<Candidate> {
    let mut view = Vec::with_capacity(n);
    build_view(&fleet(n), &mut view);
    for row in &mut view {
        if let Some(load) = &mut row.load {
            load.resident_tokens = u32::from(load.kv_usage_permille) * 250;
            load.resident_counted = true;
        }
    }
    view
}

/// Each row's two cells: a capacity of two hundred thousand, and in use
/// what the row's report states as resident. So [`SIGNAL_RULE`] sums what
/// [`RULE`] sums for every row, under the same bound.
fn signal_cells(rows: &[Candidate]) -> SignalCells {
    let mut cells = Vec::with_capacity(rows.len() * 2);
    for row in rows {
        let used = row.load.as_ref().map_or(0, |load| load.resident_tokens);
        let sampled_at_micros = row.load_sampled_at_micros;
        for value in [200_000.0, f64::from(used)] {
            cells.push(SignalCell {
                value,
                sampled_at_micros,
            });
        }
    }
    SignalCells::new(cells, 2)
}

/// Tells the engine of one send to each row, stated after every report of
/// the fleet was sampled: the rule's count holds it at each decision.
fn send_each(engine: &Engine, rows: &[Candidate]) {
    let mut lease = Lease::default();
    lease.issued_micros = 999_500;
    lease.tokens = 379;
    for row in rows {
        let mut target = Target::default();
        target.id = row.id;
        target.dp_rank = row.dp_rank;
        engine.on_dispatched(&lease, &target, Side::Single, 0);
    }
}

fn paths(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine/has_room");
    // Each path: whether the profile holds the rule, whether it draws, and
    // whether every row holds a send.
    let cells = [
        ("scored_without", false, false, false),
        ("scored_with_rule", true, false, false),
        ("scored_with_sends", true, false, true),
        ("drawn_without", false, true, false),
        ("drawn_with_rule", true, true, false),
        ("drawn_with_sends", true, true, true),
    ];
    for &n in &SIZES {
        let view = rows(n);
        let mut pool = PoolFacts::default();
        pool.pool_size = n as u32;
        pool.generation = 1;
        // Every row of the bench's fleet reports a rate.
        pool.reported = true;
        let cx = Constraints::default();
        for (path, with_rule, drawn, sends) in cells {
            let engine = policy(with_rule, drawn);
            if sends {
                send_each(&engine, &view);
            }
            let mut scratch = Scratch::new();
            group.bench_with_input(BenchmarkId::new(path, n), &n, |b, _| {
                let mut seed = 0u64;
                b.iter(|| {
                    seed += 1;
                    let mut facts = RequestFacts::default();
                    facts.model = "m";
                    facts.prompt_tokens = 512;
                    facts.seed = seed;
                    facts.now_micros = 1_000_000;
                    engine.decide(&facts, &pool, black_box(&view), &cx, &mut scratch)
                });
            });
        }
        // The rule under signals, over the same rows with their cells.
        let stated = signal_cells(&view);
        for (path, drawn, sends) in [
            ("scored_with_signals", false, false),
            ("scored_with_signal_sends", false, true),
            ("drawn_with_signals", true, false),
            ("drawn_with_signal_sends", true, true),
        ] {
            let engine = policy_of(SIGNALS, SIGNAL_RULE, drawn);
            if sends {
                send_each(&engine, &view);
            }
            let mut scratch = Scratch::new();
            group.bench_with_input(BenchmarkId::new(path, n), &n, |b, _| {
                let mut seed = 0u64;
                b.iter(|| {
                    seed += 1;
                    let mut facts = RequestFacts::default();
                    facts.model = "m";
                    facts.prompt_tokens = 512;
                    facts.seed = seed;
                    facts.now_micros = 1_000_000;
                    facts.signal_cells = Some(&stated);
                    engine.decide(&facts, &pool, black_box(&view), &cx, &mut scratch)
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, paths);
criterion_main!(benches);
