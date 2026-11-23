//! A form estimates over the set it compares. The power-of-two picker draws
//! two workers, the count-first picker takes the fewest in flight, and the
//! least-load scorer under either estimates from the workers it is handed:
//! the rate that stands in for a missing one, and the start a worker with no
//! report takes from its best reporting peer. The gateway's policies narrow
//! first and estimate after, so a worker they passed over moves neither.
//!
//! Each cell here is a pool of two to four workers under one form of the
//! `power_of_two` preset as the spec builds it: the expected wait or the
//! requests over two draws, or the least requests. A cell is decided 6,000
//! times from the starting state, one seed a decision. The expected counts
//! were recorded by running the gateway's policy code at 058a0c016 over the
//! same pool, 6,000 single decisions each from a fresh policy. A drawn cell
//! is one sample, so the two are compared as the recorded rows' replay
//! compares them: share for share.
//!
//! The pools of one count hold the tie band. The gateway draws among waits
//! within a microsecond of each other, so a picker that compares scores
//! reads the band the scorers hand it.
//!
//! Two rules of the gateway's are left, each with its reason.
//!
//! Under its count form the tie follows the pool's order: the least wait is
//! a running one, and a worker listed before the least stays in the draw
//! when it lay within a
//! microsecond of the least so far, however far behind the last. This crate
//! draws among the workers within a microsecond of the least alone, so an
//! outcome does not depend on the order a host lists its workers in. The two
//! part where three or more waits chain across more than a microsecond. The
//! last pool is such a chain: its count-first cell is held to the rule of
//! this crate, and the gateway's counts for it are named beside.
//!
//! They also part at the score's own resolution, and no pool here is that
//! slow. A score is a 32-bit float: where a wait compared, or the span of
//! the set compared, passes about 16 s, it holds no microsecond and its
//! rounding decides the tie, drawn where two waits round together and the
//! lesser where they round apart.
#![expect(clippy::expect_used, reason = "tests that stop on a broken setup")]

use smg_policy_controller::{
    document::PolicySpec,
    engine::{
        Candidate, Constraints, Engine, Health, LoadRow, PoolFacts, RequestFacts, Scratch, WorkerId,
    },
    policies::power_of_two::PowerOfTwoMetric::{self, ExpectedWait, LeastRequests, Requests},
};

const DRAWS: u32 = 6_000;

/// A worker's report: the requests it runs and the tokens a second it
/// drains (`0`: it sends no rate).
#[derive(Clone, Copy)]
struct Report {
    running: u32,
    throughput: u32,
}

/// One worker: the requests the host holds on it and its report, if any.
#[derive(Clone, Copy)]
struct Worker {
    in_flight: u32,
    report: Option<Report>,
}

const fn reporting(in_flight: u32, throughput: u32) -> Worker {
    Worker {
        in_flight,
        report: Some(Report {
            running: in_flight,
            throughput,
        }),
    }
}

/// The preset's form under `metric`, as the spec builds it: the least-load
/// scorer under the two-draw picker, or under one of the two count pickers.
fn form(metric: PowerOfTwoMetric) -> Engine {
    PolicySpec::power_of_two(metric)
        .expect("the preset's form")
        .validate()
        .expect("the form validates")
}

/// How often each worker leads over [`DRAWS`] decisions for a prompt of
/// `prompt_tokens` from the starting state: nothing is dispatched, so the
/// engine holds no credit between them.
fn counts(metric: PowerOfTwoMetric, workers: &[Worker], prompt_tokens: u32) -> Vec<u32> {
    let engine = form(metric);
    let clock = 1_000;
    let pool: Vec<Candidate> = workers
        .iter()
        .enumerate()
        .map(|(i, worker)| Candidate {
            position: i as u32,
            id: WorkerId(1_000 + i as u64),
            key: 1_000 + i as u32,
            health: Health {
                healthy: true,
                ..Health::default()
            },
            in_flight: worker.in_flight,
            load: worker.report.map(|report| LoadRow {
                running: report.running,
                live_rate: report.throughput,
                ..LoadRow::default()
            }),
            load_sampled_at_micros: clock,
            ..Candidate::default()
        })
        .collect();
    let pool_facts = PoolFacts {
        pool_size: pool.len() as u32,
        generation: 1,
        ..PoolFacts::default()
    };
    let mut scratch = Scratch::new();
    let mut counts = vec![0u32; pool.len()];
    for seed in 1..=u64::from(DRAWS) {
        let facts = RequestFacts {
            model: "m",
            prompt_tokens,
            seed,
            now_micros: clock + seed * 1_000,
            ..RequestFacts::default()
        };
        let decision = engine.decide(
            &facts,
            &pool_facts,
            &pool,
            &Constraints::default(),
            &mut scratch,
        );
        let pick = decision.pick().expect("a pick");
        counts[pick.targets.first().expect("a target").position as usize] += 1;
    }
    counts
}

/// Whether two count lists are one distribution, by the rule of the recorded
/// rows' replay: a worker never picked on one side is never picked on the
/// other, and any other pair of shares lies within four standard deviations
/// of the difference of two samples of [`DRAWS`].
fn one_distribution(here: &[u32], recorded: &[u32]) -> bool {
    let share = |count: u32| f64::from(count) / f64::from(DRAWS);
    here.len() == recorded.len()
        && here.iter().zip(recorded).all(|(&a, &b)| {
            if a == 0 || b == 0 {
                return a == b;
            }
            let (a, b) = (share(a), share(b));
            let p = (a + b) / 2.0;
            (a - b).abs() <= 4.0 * (2.0 * p * (1.0 - p) / f64::from(DRAWS)).sqrt()
        })
}

/// The cells of one pool: the form and the counts it is held to, the
/// gateway's recorded ones wherever the pool names no rule left.
fn hold(
    pool: &str,
    workers: &[Worker],
    prompt_tokens: u32,
    recorded: [(PowerOfTwoMetric, &[u32]); 3],
) {
    let mut parted = Vec::new();
    for (metric, recorded) in recorded {
        let here = counts(metric, workers, prompt_tokens);
        if !one_distribution(&here, recorded) {
            parted.push(format!("{metric:?}: recorded {recorded:?}, here {here:?}"));
        }
    }
    assert!(parted.is_empty(), "{pool}:\n{}", parted.join("\n"));
}

/// Two workers hold one request each, one of them with no report; a third
/// holds five and drains a thousand times faster. The fast worker is no
/// peer of the two when they are compared: the worker with no report starts
/// from the reporting worker beside it and never wins that pair.
#[test]
fn a_fast_worker_passed_over_lifts_no_worker_without_a_report() {
    let workers = [
        reporting(1, 100),
        Worker {
            in_flight: 1,
            report: None,
        },
        reporting(5, 100_000),
    ];
    hold(
        "a fast peer of a higher count",
        &workers,
        0,
        [
            (LeastRequests, &[6000, 0, 0]),
            (Requests, &[3968, 2032, 0]),
            (ExpectedWait, &[1985, 0, 4015]),
        ],
    );
}

/// The same pool with the third worker as slow as the first: the worker
/// with no report still never wins a comparison of waits.
#[test]
fn a_slow_worker_passed_over_changes_nothing_either() {
    let workers = [
        reporting(1, 100),
        Worker {
            in_flight: 1,
            report: None,
        },
        reporting(5, 100),
    ];
    hold(
        "a slow peer of a higher count",
        &workers,
        0,
        [
            (LeastRequests, &[6000, 0, 0]),
            (Requests, &[3976, 2024, 0]),
            (ExpectedWait, &[3966, 0, 2034]),
        ],
    );
}

/// A worker that sends no rate takes the rate of the workers it is compared
/// with: beside a worker of its own count and rate unknown to it, the two
/// tie and are drawn; a fast worker that was passed over lends it nothing.
#[test]
fn a_missing_rate_stands_in_from_the_workers_compared() {
    let workers = [reporting(1, 0), reporting(1, 100), reporting(5, 100_000)];
    hold(
        "a worker with no rate beside a rated one and a fast peer",
        &workers,
        0,
        [
            (LeastRequests, &[2953, 3047, 0]),
            (Requests, &[2957, 3043, 0]),
            (ExpectedWait, &[3034, 951, 2015]),
        ],
    );
}

/// Two workers of one count whose waits lie a tenth of a microsecond apart:
/// each runs one request, and their rates differ by one token a second in a
/// hundred thousand. The gateway cannot tell them apart and draws, under
/// each form; a picker that compared their scores exactly would hand the
/// faster one every decision.
#[test]
fn waits_within_a_microsecond_are_one_tie_and_drawn() {
    let workers = [reporting(1, 100_000), reporting(1, 100_001)];
    hold(
        "two workers a tenth of a microsecond apart",
        &workers,
        512,
        [
            (LeastRequests, &[3062, 2938]),
            (Requests, &[2968, 3032]),
            (ExpectedWait, &[2971, 3029]),
        ],
    );
}

/// The same two with rates a tenth apart: their waits differ by hundreds
/// of microseconds, and the faster worker takes every decision.
#[test]
fn waits_further_apart_are_no_tie() {
    let workers = [reporting(1, 100_000), reporting(1, 110_000)];
    hold(
        "two workers hundreds of microseconds apart",
        &workers,
        512,
        [
            (LeastRequests, &[0, 6000]),
            (Requests, &[0, 6000]),
            (ExpectedWait, &[0, 6000]),
        ],
    );
}

/// Four workers of one count whose waits step four tenths of a microsecond
/// apart, listed with the slowest second and the least third: a chain across
/// more than a microsecond. The two-draw forms compare a pair and replay: a
/// pair within a microsecond is drawn, and the one pair further apart goes
/// to the faster. The count form is the rule left. The gateway's tie follows
/// this order and gives each worker a quarter (it recorded
/// `[1511, 1459, 1524, 1506]`). Here the slowest, more than a microsecond
/// behind the least, takes no decision, and the three within a microsecond
/// of the least share them.
#[test]
fn a_chain_of_waits_is_drawn_within_a_microsecond_of_the_least() {
    let workers = [
        reporting(1, 100_008),
        reporting(1, 100_000),
        reporting(1, 100_012),
        reporting(1, 100_004),
    ];
    hold(
        "four workers four tenths of a microsecond apart",
        &workers,
        512,
        [
            // The shares of this crate's rule, not a recorded row.
            (LeastRequests, &[2000, 0, 2000, 2000]),
            (Requests, &[1444, 992, 2031, 1533]),
            (ExpectedWait, &[1469, 1026, 2039, 1466]),
        ],
    );
}
