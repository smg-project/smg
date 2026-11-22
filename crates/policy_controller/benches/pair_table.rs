//! The cost of one pair decision by what its pair table holds, at 128
//! prefills and 384 decodes: the question of which prefill classes have a
//! partner, then the two profile runs. Ids:
//! `policy_engine/pair_table/<fleet>/<state>`.
//!
//! The fleets differ in which classes pair, never in their rows: every
//! decode stands once, so the decode pool is 384 rows in each.
//! - `number`: one class a side and no table; classes pair by number.
//! - `number_apart`: 128 classes a side and no table.
//! - `one`: one class a side under a table of one entry.
//! - `four`: four prefill classes of two decode classes each, one of the
//!   two shared with another prefill class.
//! - `apart`: 128 prefill classes, each with a decode class of its own.
//! - `overlap`: 128 prefill classes, each with a decode class of its own
//!   and one class of 256 decodes that pairs with them all.
//! - `widest`: every decode a class of its own, and 128 prefill classes
//!   that each list every one of them but one: the table at its largest.
//!
//! The states differ in which decodes a decision may take:
//! - `open`: the decodes as a host reads them, most of them routable.
//! - `last`: one decode routable, the pool's last: a prefill class reads its
//!   list up to the class of that decode.
//! - `closed`: no decode routable: the decision sheds, and names the decode
//!   stage from the lists of the routable prefills.
//!
//! A fleet's table is printed once, as
//! `policy_engine/pair_table/sizes/<fleet>`.
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer; a preset that fails to validate is a bug the unit tests catch"
)]

#[expect(
    dead_code,
    reason = "this bench builds its engine from the pair presets; `common::engine_of` serves the decision and contention benches"
)]
mod common;

use std::hint::black_box;

use common::{build_view, fleet};
use criterion::{criterion_group, criterion_main, Criterion};
use smg_policy_controller::{
    document::PolicySpec,
    engine::{Candidate, Constraints, PairPools, PairTable, PoolFacts, RequestFacts, Scratch},
};

const PREFILLS: u32 = 128;
const DECODES: u32 = 384;

/// Which classes pair in one fleet: each row's class by its position, and
/// each prefill class's list; no list is no table.
struct Fleet {
    name: &'static str,
    prefill_class: fn(u32) -> u32,
    decode_class: fn(u32) -> u32,
    lists: Vec<Vec<u32>>,
}

fn fleets() -> Vec<Fleet> {
    let own = |position: u32| position + 1;
    vec![
        Fleet {
            name: "number",
            prefill_class: |_| 1,
            decode_class: |_| 1,
            lists: Vec::new(),
        },
        Fleet {
            name: "number_apart",
            prefill_class: own,
            decode_class: |position| position % PREFILLS + 1,
            lists: Vec::new(),
        },
        Fleet {
            name: "one",
            prefill_class: |_| 1,
            decode_class: |_| 1,
            lists: vec![vec![1]],
        },
        Fleet {
            name: "four",
            prefill_class: |position| position % 4 + 1,
            decode_class: |position| match position % 4 {
                3 => 5 + position / 4 % 2,
                _ => position / 4 % 4 + 1,
            },
            lists: (1..=4)
                .map(|class| vec![class, 5 + (class - 1) / 2])
                .collect(),
        },
        Fleet {
            name: "apart",
            prefill_class: own,
            decode_class: |position| position % PREFILLS + 1,
            lists: (1..=PREFILLS).map(|class| vec![class]).collect(),
        },
        Fleet {
            name: "overlap",
            prefill_class: own,
            decode_class: |position| (position + 1).min(PREFILLS + 1),
            lists: (1..=PREFILLS)
                .map(|class| vec![class, PREFILLS + 1])
                .collect(),
        },
        Fleet {
            name: "widest",
            prefill_class: own,
            decode_class: own,
            lists: (1..=PREFILLS)
                .map(|class| (1..=DECODES).filter(|&listed| listed != class).collect())
                .collect(),
        },
    ]
}

/// `n` rows of the benches' fleet, classed by position.
fn rows(n: u32, class: fn(u32) -> u32) -> Vec<Candidate> {
    let mut rows = Vec::new();
    build_view(&fleet(n as usize), &mut rows);
    for row in &mut rows {
        row.pair_class = class(row.position);
        row.has_capacity = true;
        row.has_room = true;
    }
    rows
}

fn facts(seed: u64) -> RequestFacts<'static> {
    RequestFacts {
        model: "m",
        prompt_tokens: 512,
        seed,
        now_micros: 1_000_000,
        ..RequestFacts::default()
    }
}

#[expect(clippy::print_stdout, reason = "the sizes are the bench's report")]
fn pair_table(c: &mut Criterion) {
    let engine = PolicySpec::pair_presets("least_load", "least_load")
        .expect("the pair spec")
        .validate()
        .expect("the engine");
    let pool = PoolFacts {
        pool_size: PREFILLS + DECODES,
        generation: 1,
        ..PoolFacts::default()
    };
    let cx = Constraints::default();
    let mut group = c.benchmark_group("policy_engine/pair_table");
    for fleet in fleets() {
        let ends: Vec<u32> = fleet
            .lists
            .iter()
            .scan(0u32, |end, list| {
                *end += list.len() as u32;
                Some(*end)
            })
            .collect();
        let classes = fleet.lists.concat();
        // Every prefill routable: the states are the decode side's.
        let mut prefills = rows(PREFILLS, fleet.prefill_class);
        for prefill in &mut prefills {
            prefill.health.healthy = true;
        }
        let open = rows(DECODES, fleet.decode_class);
        println!(
            "policy_engine/pair_table/sizes/{}: prefill_classes={} decode_classes={} entries={} table_bytes={} decode_rows={}",
            fleet.name,
            prefills.iter().map(|row| row.pair_class).max().unwrap_or(0),
            open.iter().map(|row| row.pair_class).max().unwrap_or(0),
            classes.len(),
            4 * (ends.len() + classes.len()),
            open.len(),
        );
        let mut closed = open.clone();
        for decode in &mut closed {
            decode.health.healthy = false;
        }
        let mut last = closed.clone();
        if let Some(decode) = last.last_mut() {
            decode.health.healthy = true;
        }
        for (state, decodes) in [("open", &open), ("last", &last), ("closed", &closed)] {
            let pools = PairPools {
                prefills: &prefills,
                decodes,
                table: PairTable::new(&ends, &classes),
                ..PairPools::default()
            };
            let mut scratch = Scratch::new();
            let probe = engine.decide_pair(&facts(0), &pool, pools, &cx, &cx, &mut scratch);
            assert_eq!(
                probe.pick().is_some(),
                state != "closed",
                "{}/{state}: {probe:?}",
                fleet.name
            );
            group.bench_function(format!("{}/{state}", fleet.name), |b| {
                let mut seed = 0u64;
                b.iter(|| {
                    seed += 1;
                    engine.decide_pair(
                        &facts(seed),
                        &pool,
                        black_box(pools),
                        &cx,
                        &cx,
                        &mut scratch,
                    )
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, pair_table);
criterion_main!(benches);
