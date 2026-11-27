use std::sync::mpsc;

use super::*;
use crate::{
    document::{PolicySpec, SpecError},
    engine::{
        decision::{Pair, Pick, Target, Targets},
        view::{self, Health},
        Constraints, PoolFacts, Scratch,
    },
    stages::{retired_workers::RETIRED_WORKER_TTL, StageRegistry},
};

mod bookings;
mod cuts;
mod dispatches;
mod fences;
mod membership;
mod placement;
mod retirement;

const SECOND: u64 = 1_000_000;
/// An absolute threshold no booking of a case reaches: the balance rule
/// never spills under it.
const NEVER: u32 = u32::MAX;

fn scorer(abs_threshold: u32, rel_threshold: f32, interval_s: u32) -> Bucket {
    Bucket::new(&BucketParams {
        abs_threshold: Some(abs_threshold),
        rel_threshold: Some(rel_threshold),
        adjust_interval_s: Some(interval_s),
        ..BucketParams::default()
    })
}

fn pool(workers: u32) -> Vec<Candidate> {
    (0..workers)
        .map(|row| Candidate {
            position: row,
            id: WorkerId(100 + u64::from(row)),
            health: Health {
                healthy: true,
                ..Health::default()
            },
            ..Candidate::default()
        })
        .collect()
}

fn all(pool: &[Candidate]) -> Vec<u32> {
    (0..pool.len() as u32).collect()
}

fn prefill(model: &str, tokens: u32, now_micros: u64) -> RequestFacts<'_> {
    RequestFacts {
        model,
        phase: Phase::Prefill,
        prompt_tokens: tokens,
        now_micros,
        ..RequestFacts::default()
    }
}

/// The first row of the worker the scorer leads with among `kept`; `None`
/// when every survivor scores equal. Every row of one worker at most
/// leads, at the top of the scale, and every other stands at zero.
fn led_by(
    scorer: &Bucket,
    facts: &RequestFacts<'_>,
    pool: &[Candidate],
    kept: &[u32],
) -> Option<u32> {
    let mut out = vec![0.5; kept.len()];
    assert_eq!(
        scorer.score(facts, pool, kept, &mut out),
        TieBand::NORMALIZED
    );
    let leads: Vec<u32> = kept
        .iter()
        .zip(&out)
        .filter(|&(_, &score)| score > 0.5)
        .map(|(&row, _)| row)
        .collect();
    if let Some(&first) = leads.first() {
        assert!(
            leads
                .iter()
                .all(|&row| pool[row as usize].id == pool[first as usize].id),
            "{out:?}"
        );
    }
    assert!(
        out.iter()
            .all(|&score| score.to_bits() == 0f32.to_bits() || score.to_bits() == 1f32.to_bits()),
        "{out:?}"
    );
    leads.first().copied()
}

/// The row a prefill request of `tokens` leads with over the whole pool.
fn led(scorer: &Bucket, pool: &[Candidate], tokens: u32, now_micros: u64) -> Option<u32> {
    led_by(scorer, &prefill("m", tokens, now_micros), pool, &all(pool))
}

/// Book `tokens` on the worker of `row` at `at_micros`, as an issued
/// decision that names it first.
fn book(scorer: &Bucket, pool: &[Candidate], row: u32, tokens: u32, at_micros: u64) {
    let lease = Lease {
        tokens,
        issued_micros: at_micros,
        ..Lease::default()
    };
    issue(scorer, pool, row, &lease);
}

/// Tell the scorer the issue of `lease` on the worker of `row`, as a
/// decision that names it first.
fn issue(scorer: &Bucket, pool: &[Candidate], row: u32, lease: &Lease) {
    let mut targets = Targets::new();
    targets.push(Target {
        position: row,
        id: pool[row as usize].id,
        dp_rank: 0,
        pair: None,
    });
    let decision = Decision::Pick(Pick {
        targets,
        ..Pick::default()
    });
    scorer.on_decision_issued(lease, &decision);
}

/// The partition of `model`, which a decision listed.
fn of(scorer: &Bucket, model: &str) -> Partition {
    Arc::clone(
        scorer
            .partitions
            .load()
            .get(model)
            .expect("the model's partition"),
    )
}

/// The ranges of model `m`: each as its member, its lower and its upper
/// end.
fn ranges(scorer: &Bucket) -> Vec<(u32, u32, u32)> {
    of(scorer, "m")
        .lock()
        .ranges
        .iter()
        .map(|range| (range.member, range.lower, range.upper))
        .collect()
}

/// The booked tokens of model `m`'s members, in the listing's order, and
/// the window's total.
fn loads(scorer: &Bucket) -> (Vec<u64>, u64) {
    let partition = of(scorer, "m");
    let pool = partition.lock();
    (
        pool.members.iter().map(|member| member.load).collect(),
        pool.total,
    )
}

/// A scorer over a listed pool with `tokens[row]` booked on each row at
/// the listing's instant, one second on the clock.
fn booked(abs_threshold: u32, rel_threshold: f32, tokens: &[u32]) -> (Bucket, Vec<Candidate>) {
    let scorer = scorer(abs_threshold, rel_threshold, 5);
    let pool = pool(tokens.len() as u32);
    led(&scorer, &pool, 0, SECOND);
    for (row, &tokens) in tokens.iter().enumerate() {
        if tokens > 0 {
            book(&scorer, &pool, row as u32, tokens, SECOND);
        }
    }
    (scorer, pool)
}

/// Three workers listed at ten seconds, then ten bookings of 100 tokens,
/// five of 400 and two of 1,500 at eleven: 6,000 tokens, 2,000 a worker.
fn three_classes() -> (Bucket, Vec<Candidate>) {
    let scorer = scorer(NEVER, 1.5, 5);
    let pool = pool(3);
    led(&scorer, &pool, 0, 10 * SECOND);
    for (row, tokens, count) in [(0, 100, 10), (1, 400, 5), (2, 1_500, 2)] {
        for _ in 0..count {
            book(&scorer, &pool, row, tokens, 11 * SECOND);
        }
    }
    (scorer, pool)
}
