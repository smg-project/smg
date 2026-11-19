//! The recorder of the gateway's bucket policy: each row is a pool of
//! prefill workers the policy lists, its configuration, requests of given
//! sizes on a clock, and the worker each one was placed on. The policy
//! controller crate replays the rows through its own decision
//! (`crates/policy_controller/tests/classic/bucket.rs`).
//!
//! A request's size is what the policy reads: the count of its token ids
//! when it carries them (`tokens`), else the characters of its text
//! (`text`), else zero. A step is a request (`pick`), whether each worker is
//! ready from here on (`ready`), a prefill worker that registers (`join`:
//! it takes the next number, and the policy is handed every listed worker
//! again, as the registration step does), or a worker that leaves (`leave`).
//! An answer is the worker's number: its place in the row's first listing,
//! or the number it joined under.
//!
//! The policy reads two clocks, and this recorder stands for both. A
//! request carries `at_ms`, the row's clock at its placement. The policy's
//! window dates each booking by the system clock, so before a step the
//! recorder moves every queued booking back by the time the row's clock
//! advanced, which is what that much time passing does to the window. The
//! policy's adjustment runs on a thread once an interval; the recorder runs
//! the same adjustment itself at every whole interval of the row's clock,
//! counted from the row's first request, before the request that reaches or
//! passes it. The thread's own first run is an interval of real time away,
//! and a row is over in milliseconds. A step with no clock (`ready`, `join`,
//! `leave`) happens at the instant of the step before it.
//!
//! The system clock still runs between two steps, by microseconds. No row
//! places a request where a booking's age stands within a second under the
//! window's length, so those microseconds decide nothing.
//!
//! This module is a child of the policy's own module and stands in its
//! directory: the adjustment and the window are the policy's private
//! state, and a recorder beside the module could reach neither.
//!
//! Run it from the repository's root:
//!
//! ```text
//! CLASSIC_ROWS_OUT=<dir> CLASSIC_ROWS_COMMIT=<commit> \
//!   cargo test -p smg --lib policies::bucket::classic_bucket_rows -- --ignored --nocapture
//! ```
//!
//! A sequence row is recorded twenty times from a fresh policy and must come
//! out the same each time. The one drawn row places one request 6,000 times,
//! each from a fresh policy, and counts what each worker took.

use std::{fs, path::PathBuf, sync::Arc, time::Duration};

use openai_protocol::worker::WorkerStatus;
use serde::Serialize;
use smg_worker::identity::basic::{Worker, WorkerType};

use super::BucketPolicy;
use crate::{
    policies::{BucketConfig, LoadBalancingPolicy, SelectWorkerInfo},
    routers::common::snapshot::test_support::worker_of,
};

/// The policy's configuration, in the gateway's own spelling.
#[derive(Clone, Copy, Serialize)]
struct Config {
    balance_abs_threshold: usize,
    balance_rel_threshold: f32,
    bucket_adjust_interval_secs: usize,
}

/// What the command line's flags default to.
const FLAGS: Config = Config {
    balance_abs_threshold: 64,
    balance_rel_threshold: 1.5,
    bucket_adjust_interval_secs: 5,
};

/// What the policy takes when it is built by its name alone.
const NAMED: Config = Config {
    balance_abs_threshold: 32,
    balance_rel_threshold: 1.0001,
    bucket_adjust_interval_secs: 5,
};

/// An absolute threshold no row's bookings reach: the balance rule never
/// spills, and a size stays with the owner of its range.
const STRICT: Config = Config {
    balance_abs_threshold: 1_000_000_000,
    ..FLAGS
};

impl Config {
    fn policy(self) -> BucketConfig {
        BucketConfig {
            balance_abs_threshold: self.balance_abs_threshold,
            balance_rel_threshold: self.balance_rel_threshold,
            bucket_adjust_interval_secs: self.bucket_adjust_interval_secs,
        }
    }
}

/// One request: the row's clock at its placement and what it is sized by.
#[derive(Clone, Copy, Serialize)]
struct Request {
    at_ms: u64,
    /// How many token ids the request carries.
    #[serde(skip_serializing_if = "Option::is_none")]
    tokens: Option<usize>,
    /// How many characters its text holds.
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<usize>,
    /// The workers an earlier attempt of the request already tried.
    #[serde(skip_serializing_if = "<[usize]>::is_empty")]
    tried: &'static [usize],
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "snake_case")]
enum Step {
    /// One request.
    Pick(Request),
    /// Whether each worker is ready, from here on.
    Ready(Vec<bool>),
    /// A prefill worker registers, under the next number.
    Join,
    /// The worker of this number leaves.
    Leave(usize),
}

/// A request of `tokens` token ids at `at_ms`.
const fn sized(at_ms: u64, tokens: usize) -> Step {
    Step::Pick(Request {
        at_ms,
        tokens: Some(tokens),
        text: None,
        tried: &[],
    })
}

/// A request of `tokens` token ids at `at_ms` whose earlier attempts tried
/// `tried`.
const fn retried(at_ms: u64, tokens: usize, tried: &'static [usize]) -> Step {
    Step::Pick(Request {
        at_ms,
        tokens: Some(tokens),
        text: None,
        tried,
    })
}

/// A request at `at_ms` that carries a text of `text` characters and
/// `tokens` token ids, where it carries any.
const fn worded(at_ms: u64, tokens: Option<usize>, text: Option<usize>) -> Step {
    Step::Pick(Request {
        at_ms,
        tokens,
        text,
        tried: &[],
    })
}

/// `count` requests of `tokens` token ids at `at_ms`.
fn many(count: usize, at_ms: u64, tokens: usize) -> Vec<Step> {
    vec![sized(at_ms, tokens); count]
}

#[derive(PartialEq, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum Outcome {
    /// The answer to each `Pick` step, in order.
    Sequence(Vec<usize>),
    /// A drawn row: over `draws` placements of its one request, what each
    /// worker took.
    Counts { draws: u32, counts: Vec<u32> },
}

/// Which pool a row's workers are: prefill workers the policy lists at
/// their registration, or regular workers nobody lists.
#[derive(Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Pool {
    Prefill,
    Regular,
}

#[derive(Serialize)]
struct Row {
    name: &'static str,
    config: Config,
    pool: Pool,
    /// How many workers the row starts with, numbered from zero.
    workers: usize,
    steps: Vec<Step>,
    outcome: Outcome,
}

#[derive(Serialize)]
struct Sheet {
    recorded_from: String,
    policy: &'static str,
    rows: Vec<Row>,
}

fn url(number: usize) -> String {
    format!("http://w{number}:8000")
}

/// A fresh policy over a fresh pool, with the row's clock.
struct Bed {
    policy: BucketPolicy,
    pool: Pool,
    /// Every worker the row ever numbered, and whether it left.
    workers: Vec<Arc<dyn Worker>>,
    gone: Vec<bool>,
    interval_ms: u64,
    /// The row's clock, and its first request's instant once it came.
    now_ms: u64,
    first_ms: Option<u64>,
    /// The adjustments run so far.
    adjusted: u64,
}

impl Bed {
    fn new(config: Config, pool: Pool, workers: usize) -> Self {
        let kind = match pool {
            Pool::Prefill => WorkerType::Prefill,
            Pool::Regular => WorkerType::Regular,
        };
        let bed = Self {
            policy: BucketPolicy::with_config(config.policy()),
            pool,
            workers: (0..workers).map(|i| worker_of(&url(i), kind)).collect(),
            gone: vec![false; workers],
            interval_ms: config.bucket_adjust_interval_secs as u64 * 1_000,
            now_ms: 0,
            first_ms: None,
            adjusted: 0,
        };
        bed.list();
        bed
    }

    /// The workers that have not left, each with its number.
    fn listed(&self) -> impl Iterator<Item = (usize, &Arc<dyn Worker>)> {
        self.workers
            .iter()
            .enumerate()
            .filter(|(number, _)| !self.gone[*number])
    }

    /// Hand the policy every prefill worker, as the registration step does
    /// whenever one joins. A regular pool is never listed.
    fn list(&self) {
        if self.pool == Pool::Prefill {
            let workers: Vec<Arc<dyn Worker>> = self
                .listed()
                .map(|(_, worker)| Arc::clone(worker))
                .collect();
            self.policy.init_prefill_worker_urls(&workers);
        }
    }

    /// `ms` of the row's clock pass: every queued booking is that much
    /// older.
    fn elapse(&self, ms: u64) {
        if ms == 0 {
            return;
        }
        for entry in self.policy.buckets.iter() {
            let mut bucket = entry.value().write();
            for request in &mut bucket.request_list {
                request.timestamp -= Duration::from_millis(ms);
            }
        }
    }

    /// Bring the row's clock to a request's instant: each whole interval
    /// since the row's first request that the clock reaches or passes runs
    /// the adjustment the policy's thread runs, at its own instant.
    fn advance_to(&mut self, at_ms: u64) {
        assert!(at_ms >= self.now_ms, "the row's clock does not go back");
        let first = *self.first_ms.get_or_insert(at_ms);
        loop {
            let due = first + self.interval_ms * (self.adjusted + 1);
            if due > at_ms {
                break;
            }
            self.elapse(due - self.now_ms);
            self.now_ms = due;
            for entry in self.policy.buckets.iter() {
                entry.value().write().adjust_boundary();
            }
            self.adjusted += 1;
        }
        self.elapse(at_ms - self.now_ms);
        self.now_ms = at_ms;
    }

    fn ready(&self, ready: &[bool]) {
        assert_eq!(ready.len(), self.workers.len(), "a word on every worker");
        for (worker, &ready) in self.workers.iter().zip(ready) {
            worker.set_status(if ready {
                WorkerStatus::Ready
            } else {
                WorkerStatus::NotReady
            });
        }
    }

    fn join(&mut self) {
        let kind = match self.pool {
            Pool::Prefill => WorkerType::Prefill,
            Pool::Regular => WorkerType::Regular,
        };
        self.workers.push(worker_of(&url(self.workers.len()), kind));
        self.gone.push(false);
        self.list();
    }

    fn leave(&mut self, number: usize) {
        assert!(!self.gone[number], "a worker leaves once");
        self.gone[number] = true;
        LoadBalancingPolicy::remove_worker(&self.policy, &url(number));
    }

    /// One request through the policy's selection, over the workers a
    /// caller offers it: the ones that have not left, less the ones an
    /// earlier attempt tried. The number of the worker it chose.
    fn pick(&mut self, request: Request) -> usize {
        self.advance_to(request.at_ms);
        let (numbers, offered): (Vec<usize>, Vec<Arc<dyn Worker>>) = self
            .listed()
            .filter(|(number, _)| !request.tried.contains(number))
            .map(|(number, worker)| (number, Arc::clone(worker)))
            .unzip();
        let tokens = request.tokens.map(|count| vec![0u32; count]);
        let text = request.text.map(|characters| "a".repeat(characters));
        let info = SelectWorkerInfo {
            tokens: tokens.as_deref(),
            request_text: text.as_deref(),
            ..Default::default()
        };
        let chosen = self
            .policy
            .select_worker(&offered, &info)
            .expect("the request is placed");
        numbers[chosen]
    }

    fn run(config: Config, pool: Pool, workers: usize, steps: &[Step]) -> Vec<usize> {
        let mut bed = Self::new(config, pool, workers);
        let mut answers = Vec::new();
        for step in steps {
            match step {
                Step::Pick(request) => answers.push(bed.pick(*request)),
                Step::Ready(ready) => bed.ready(ready),
                Step::Join => bed.join(),
                Step::Leave(number) => bed.leave(*number),
            }
        }
        answers
    }
}

/// A row of steps over a pool of listed prefill workers, recorded twenty
/// times from a fresh policy.
fn row(name: &'static str, config: Config, workers: usize, steps: Vec<Step>) -> Row {
    let first = Bed::run(config, Pool::Prefill, workers, &steps);
    for _ in 1..20 {
        assert_eq!(
            Bed::run(config, Pool::Prefill, workers, &steps),
            first,
            "{name}: a row must not depend on a draw"
        );
    }
    Row {
        name,
        config,
        pool: Pool::Prefill,
        workers,
        steps,
        outcome: Outcome::Sequence(first),
    }
}

/// How often each worker of a pool nobody lists takes one request, over
/// 6,000 placements, each from a fresh policy.
fn drawn_row(name: &'static str, config: Config, workers: usize, request: Step) -> Row {
    const DRAWS: u32 = 6_000;
    let steps = vec![request];
    let mut counts = vec![0u32; workers];
    for _ in 0..DRAWS {
        let answers = Bed::run(config, Pool::Regular, workers, &steps);
        counts[answers[0]] += 1;
    }
    Row {
        name,
        config,
        pool: Pool::Regular,
        workers,
        steps,
        outcome: Outcome::Counts {
            draws: DRAWS,
            counts,
        },
    }
}

/// Ten requests of 100 tokens, five of 400 and two of 1,500 at the row's
/// start: 6,000 tokens, a third of them 2,000.
fn three_sizes() -> Vec<Step> {
    let mut steps = many(10, 0, 100);
    steps.extend(many(5, 0, 400));
    steps.extend(many(2, 0, 1_500));
    steps
}

/// A size's owner: the even split of a fresh listing, a worker that is not
/// offered, and what a request is sized by.
fn range_rows() -> Vec<Row> {
    vec![
        row(
            "a-size-goes-to-the-owner-of-its-range-in-an-even-split-of-four",
            STRICT,
            4,
            [0, 1_023, 1_024, 2_047, 2_048, 3_071, 3_072, 4_096, 100_000]
                .into_iter()
                .map(|tokens| sized(0, tokens))
                .collect(),
        ),
        row(
            "an-even-split-of-three-leaves-the-rest-of-the-span-to-the-last",
            STRICT,
            3,
            [1_364, 1_365, 2_729, 2_730]
                .into_iter()
                .map(|tokens| sized(0, tokens))
                .collect(),
        ),
        row(
            "a-request-is-sized-by-its-tokens-then-by-its-text-then-at-zero",
            STRICT,
            4,
            vec![
                worded(0, None, Some(1_023)),
                worded(0, None, Some(1_024)),
                worded(0, None, Some(3_072)),
                worded(0, Some(100), Some(3_500)),
                worded(0, None, None),
            ],
        ),
        row(
            "an-owner-that-is-not-ready-hands-its-size-to-the-worker-below-then-above",
            STRICT,
            4,
            vec![
                Step::Ready(vec![true, false, true, true]),
                sized(0, 1_500),
                Step::Ready(vec![false, false, true, true]),
                sized(0, 1_500),
                sized(0, 100),
                Step::Ready(vec![true, true, true, false]),
                sized(0, 3_500),
                Step::Ready(vec![false, false, false, true]),
                sized(0, 100),
            ],
        ),
        row(
            "a-worker-tried-already-hands-its-size-on-as-one-not-ready-does",
            STRICT,
            4,
            vec![
                retried(0, 1_500, &[1]),
                retried(0, 1_500, &[0, 1]),
                retried(0, 3_500, &[3]),
                sized(0, 1_500),
            ],
        ),
    ]
}

/// The balance rule: both of its parts, under the flags' defaults and under
/// the defaults of the policy built by name, and whose bookings it weighs.
fn balance_rows() -> Vec<Row> {
    let sevens = || many(7, 0, 100);
    let edge = || vec![sized(0, 64), sized(0, 0), sized(0, 1), sized(0, 0)];
    let held_back = || {
        [2_048, 2_048, 3_000, 2_100, 2_100, 2_100]
            .into_iter()
            .map(|tokens| sized(0, tokens))
            .collect::<Vec<_>>()
    };
    vec![
        row(
            "the-balance-rule-spills-a-size-to-the-least-booked-worker",
            FLAGS,
            3,
            sevens(),
        ),
        row(
            "the-balance-rule-spills-under-the-defaults-of-the-policy-by-name",
            NAMED,
            3,
            sevens(),
        ),
        row(
            "a-difference-at-the-absolute-threshold-does-not-spill",
            FLAGS,
            2,
            edge(),
        ),
        row(
            "the-relative-part-holds-a-spill-back-under-a-ratio-of-1-5",
            FLAGS,
            2,
            held_back(),
        ),
        row(
            "the-relative-part-lets-the-same-bookings-spill-under-a-ratio-of-1-0001",
            NAMED,
            2,
            held_back(),
        ),
        row(
            "the-balance-rule-weighs-the-workers-offered-and-no-other",
            FLAGS,
            3,
            vec![
                sized(0, 100),
                Step::Ready(vec![true, false, true]),
                sized(0, 100),
                Step::Ready(vec![true, true, true]),
                sized(0, 100),
            ],
        ),
        row(
            "the-first-request-after-a-quiet-interval-is-weighed-by-the-bookings-it-ends",
            FLAGS,
            2,
            vec![
                sized(0, 100),
                sized(1_000, 100),
                sized(2_000, 100),
                sized(12_000, 100),
                sized(12_100, 100),
            ],
        ),
    ]
}

/// The adjustment: the ranges cut again from the sizes seen, when it is made
/// and when it waits.
fn adjustment_rows() -> Vec<Row> {
    let mut cut = three_sizes();
    cut.extend([
        sized(4_000, 400),
        sized(5_001, 400),
        sized(5_001, 100),
        sized(5_001, 401),
        sized(5_001, 1_500),
    ]);
    let mut waits = three_sizes();
    waits.extend(many(10, 7_000, 500));
    waits.extend([
        sized(10_001, 300),
        sized(13_000, 20_000),
        sized(15_001, 300),
        sized(15_001, 301),
        sized(15_001, 20_001),
    ]);
    let mut few = many(3, 0, 200);
    few.extend([
        sized(5_001, 150),
        Step::Ready(vec![false, true, true]),
        sized(5_001, 150),
        Step::Ready(vec![true, true, false]),
        sized(5_001, 5_000),
        Step::Ready(vec![false, true, false]),
        sized(5_001, 5_000),
    ]);
    vec![
        row(
            "an-interval-s-end-cuts-the-ranges-again-by-the-sizes-seen",
            STRICT,
            3,
            cut,
        ),
        row(
            "a-cut-waits-until-a-worker-s-share-moved-by-more-than-a-factor-of-two",
            STRICT,
            3,
            waits,
        ),
        row(
            "fewer-sizes-than-workers-leave-a-worker-without-a-range",
            STRICT,
            3,
            few,
        ),
    ]
}

/// The listing: a worker that joins, a worker that leaves.
fn listing_rows() -> Vec<Row> {
    vec![
        row(
            "a-worker-that-joins-splits-the-span-again-until-the-next-cut",
            STRICT,
            2,
            vec![
                sized(0, 100),
                sized(0, 3_000),
                sized(5_001, 1_500),
                Step::Join,
                sized(5_001, 1_500),
                sized(5_001, 3_000),
                sized(5_001, 100),
                sized(10_001, 1_500),
                sized(10_001, 1_501),
            ],
        ),
        row(
            "a-worker-that-leaves-hands-its-range-to-the-worker-below",
            STRICT,
            3,
            vec![
                sized(0, 1_500),
                Step::Leave(1),
                sized(0, 1_500),
                sized(0, 2_730),
                sized(5_001, 1_500),
                sized(5_001, 1_501),
            ],
        ),
        row(
            "the-first-worker-s-range-goes-to-the-worker-above-when-it-leaves",
            STRICT,
            3,
            vec![
                sized(0, 100),
                Step::Leave(0),
                sized(0, 100),
                sized(0, 2_730),
            ],
        ),
    ]
}

fn rows() -> Vec<Row> {
    let mut rows = range_rows();
    rows.extend(balance_rows());
    rows.extend(adjustment_rows());
    rows.extend(listing_rows());
    rows.push(drawn_row(
        "a-pool-nobody-listed-is-drawn-from",
        FLAGS,
        4,
        sized(0, 1_500),
    ));
    rows
}

#[test]
#[ignore = "records the classic rows into CLASSIC_ROWS_OUT"]
fn record_the_classic_bucket_rows() {
    let out = PathBuf::from(
        std::env::var_os("CLASSIC_ROWS_OUT").expect("CLASSIC_ROWS_OUT names the output directory"),
    );
    let commit =
        std::env::var("CLASSIC_ROWS_COMMIT").expect("CLASSIC_ROWS_COMMIT names the commit");
    let sheet = Sheet {
        recorded_from: commit,
        policy: "bucket",
        rows: rows(),
    };
    fs::create_dir_all(&out).expect("the output directory");
    let path = out.join("bucket.json");
    let text = serde_json::to_string_pretty(&sheet).expect("the sheet serializes");
    fs::write(&path, text + "\n").expect("the sheet is written");
    println!("{} rows -> {}", sheet.rows.len(), path.display());
}
