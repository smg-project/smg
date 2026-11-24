//! The recorder of the gateway's minimum-tokens rank: each row is one worker
//! with its data-parallel ranks, a list of steps and the rank each request
//! took. A step is what the rank cache is handed: a poll's answer
//! (a row a rank), a poll that failed, the worker's loads dropped as it
//! leaves the ready set, or a request, which asks the rank cache's select
//! for a rank and books its estimated tokens on the rank it takes. The
//! gateway's minimum-tokens policy forwards a request to that select and
//! adds nothing. A request that finds no rank records `null`. The policy
//! controller crate replays the rows through the rank cache of the load
//! monitor a host builds
//! (`crates/policy_controller/tests/classic/minimum_tokens.rs`).
//!
//! This file is an ignored test of the worker crate, kept beside the code
//! it runs. It records from the checkout it is built in and writes the sheet
//! under `CLASSIC_ROWS_OUT`, naming `CLASSIC_ROWS_COMMIT` as the commit it
//! recorded from. The sheets the policy controller crate replays are in
//! `crates/policy_controller/tests/classic/sheets`; to record this one again:
//!
//! ```text
//! CLASSIC_ROWS_OUT=<dir> CLASSIC_ROWS_COMMIT=<commit> \
//!   cargo test -p smg-worker --all-features --lib load::monitor::classic_minimum_tokens_rows -- --ignored --nocapture
//! ```
//!
//! Nothing in a row draws. The recorder runs each row twice, each time from
//! a fresh monitor and a fresh worker, and stops if the two runs differ.

use std::{fs, path::PathBuf};

use serde::Serialize;

use super::*;
use crate::{
    identity::basic::WorkerType,
    model_card::ModelCard,
    test_support::{BasicWorkerBuilder, LoadReaders, WorkerRegistry},
    worker::{HealthCheckConfig, WorkerStatus},
};

const URL: &str = "http://w0:8000";
const MODEL: &str = "m";

/// One rank's row in a poll's answer: the tokens its cache uses, which is
/// the count the rank cache reads, its token capacity, and the
/// context tokens of the requests it holds where the rank reports them.
#[derive(Clone, Copy, Serialize)]
struct Figures {
    used: u32,
    capacity: u32,
    resident: Option<u32>,
}

/// A rank with token capacity whose requests hold every token it uses.
const fn rank(used: u32) -> Option<Figures> {
    held(used, used)
}

/// A rank with token capacity that uses `used` tokens while its requests
/// hold `resident` context tokens.
#[expect(
    clippy::unnecessary_wraps,
    reason = "an entry of a poll's answer: `None` is a rank the answer omits"
)]
const fn held(used: u32, resident: u32) -> Option<Figures> {
    Some(Figures {
        used,
        capacity: 8_192,
        resident: Some(resident),
    })
}

/// A rank whose row carries no token capacity.
#[expect(
    clippy::unnecessary_wraps,
    reason = "an entry of a poll's answer: `None` is a rank the answer omits"
)]
const fn bare(used: u32) -> Option<Figures> {
    Some(Figures {
        used,
        capacity: 0,
        resident: Some(used),
    })
}

/// A rank with token capacity that reports no context tokens.
#[expect(
    clippy::unnecessary_wraps,
    reason = "an entry of a poll's answer: `None` is a rank the answer omits"
)]
const fn silent(used: u32) -> Option<Figures> {
    Some(Figures {
        used,
        capacity: 8_192,
        resident: None,
    })
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "snake_case")]
enum Step {
    /// A poll's answer: a row a rank, in rank order, `null` where the answer
    /// omits the rank.
    Poll(Vec<Option<Figures>>),
    /// A poll that failed.
    Failed,
    /// The worker leaves the ready set and its loads are dropped.
    Evict,
    /// A request of `cost` estimated tokens asks for a rank.
    Pick { cost: u32 },
}

const fn pick(cost: u32) -> Step {
    Step::Pick { cost }
}

/// The rank each request took, in order: `null` where it found none.
#[derive(Serialize)]
struct Outcome {
    sequence: Vec<Option<usize>>,
}

#[derive(Serialize)]
struct Row {
    name: &'static str,
    /// The worker's data-parallel ranks.
    ranks: usize,
    /// The worker's own rank where it stands for one rank of its engine;
    /// `null` where it stands for every rank.
    pinned: Option<usize>,
    steps: Vec<Step>,
    outcome: Outcome,
}

#[derive(Serialize)]
struct Sheet {
    recorded_from: String,
    policy: &'static str,
    rows: Vec<Row>,
}

fn count(n: u32) -> i32 {
    i32::try_from(n).expect("a row's count fits the report's field")
}

/// A poll's answer as an engine sends it: one entry for each rank it names.
fn answer(ranks: usize, figures: &[Option<Figures>]) -> WorkerLoadResponse {
    WorkerLoadResponse {
        dp_rank_count: i32::try_from(ranks).expect("a row's ranks fit the report's field"),
        loads: figures
            .iter()
            .enumerate()
            .filter_map(|(at, figures)| {
                figures.map(|figures| SchedulerLoadSnapshot {
                    dp_rank: i32::try_from(at).expect("a rank fits the report's field"),
                    num_used_tokens: count(figures.used),
                    max_total_num_tokens: count(figures.capacity),
                    num_resident_tokens: figures.resident.map(i64::from),
                    ..SchedulerLoadSnapshot::default()
                })
            })
            .collect(),
        ..WorkerLoadResponse::default()
    }
}

/// One run of a row from a fresh monitor and a fresh worker: the rank each
/// request took. Each step reaches the rank cache as the monitor's poll loop
/// hands it over, under the worker's own URL, which names its rank where it
/// stands for one: an answer with no row drops the worker's ranks, or its own
/// rank where the worker stands for one; an answer with absolute token data
/// whose rows are ranks replaces the worker's ranks whole; any other answer
/// drops them; a failed poll changes nothing; the eviction goes through the
/// monitor's own; a request goes through the rank cache's select.
fn run(ranks: usize, pinned: Option<usize>, steps: &[Step]) -> Vec<Option<usize>> {
    let registry = Arc::new(WorkerRegistry::new());
    let monitor = Arc::new(WorkerMonitor::new(
        Arc::clone(&registry),
        Arc::new(LoadReaders::Blind),
        1,
        false,
        false,
    ));
    let builder = BasicWorkerBuilder::new(URL)
        .worker_type(WorkerType::Regular)
        .model(ModelCard::new(MODEL))
        .health_config(HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        });
    let builder = match pinned {
        Some(rank) => builder.dp_config(rank, ranks),
        None => builder,
    };
    let worker: Arc<dyn Worker> = Arc::new(builder.build());
    worker.set_status(WorkerStatus::Ready);
    registry
        .register(Arc::clone(&worker))
        .expect("the worker registers");
    let cache = &monitor.worker_load_manager;
    let url = worker.url().to_string();
    steps
        .iter()
        .filter_map(|step| {
            match step {
                Step::Poll(figures) => {
                    let load = answer(ranks, figures);
                    if load.loads.is_empty() {
                        match worker.dp_rank() {
                            Some(rank) => cache.remove_rank(&url, rank),
                            None => cache.remove_worker(&url),
                        }
                    } else if load.has_absolute_token_data() && load.ranks_are_dp_ranks() {
                        cache.update_dp_loads(&HashMap::from([(
                            url.clone(),
                            dp_rank_token_loads(&load),
                        )]));
                    } else {
                        cache.remove_workers(std::slice::from_ref(&url));
                    }
                }
                Step::Failed => {}
                Step::Evict => monitor.evict_worker_loads(&worker),
                Step::Pick { cost } => {
                    let cost = isize::try_from(*cost).expect("a cost fits the select's count");
                    let rank = cache.select_and_increment_lowest_dp_load(worker.as_ref(), cost);
                    return Some(
                        rank.map(|rank| {
                            usize::try_from(rank).expect("a rank is no negative number")
                        }),
                    );
                }
            }
            None
        })
        .collect()
}

fn row(name: &'static str, ranks: usize, pinned: Option<usize>, steps: Vec<Step>) -> Row {
    let sequence = run(ranks, pinned, &steps);
    assert_eq!(
        sequence,
        run(ranks, pinned, &steps),
        "{name}: two runs of one row differ"
    );
    Row {
        name,
        ranks,
        pinned,
        steps,
        outcome: Outcome { sequence },
    }
}

fn rows() -> Vec<Row> {
    let two = || Step::Poll(vec![rank(100), rank(200)]);
    let omitted = || Step::Poll(vec![None, rank(200), rank(300)]);
    vec![
        row(
            "the-rank-with-the-fewest-used-tokens-takes-the-request",
            3,
            None,
            vec![Step::Poll(vec![rank(300), rank(100), rank(200)]), pick(10)],
        ),
        row(
            "a-pick-books-its-cost-on-its-rank",
            2,
            None,
            vec![
                Step::Poll(vec![rank(100), rank(150)]),
                pick(60),
                pick(60),
                pick(60),
                pick(60),
            ],
        ),
        row(
            "eight-ranks-take-one-large-request-each-from-the-least-used-up",
            8,
            None,
            [Step::Poll(vec![
                rank(800),
                rank(100),
                rank(700),
                rank(200),
                rank(600),
                rank(300),
                rank(500),
                rank(400),
            ])]
            .into_iter()
            .chain((0..8).map(|_| pick(1_000)))
            .collect(),
        ),
        row(
            "equal-ranks-go-to-the-lowest-rank",
            3,
            None,
            vec![
                Step::Poll(vec![rank(100), rank(100), rank(100)]),
                pick(10),
                pick(10),
                pick(10),
                pick(10),
            ],
        ),
        row(
            "a-taken-answer-replaces-the-ranks-and-their-bookings",
            2,
            None,
            vec![two(), pick(500), two(), pick(500), pick(1)],
        ),
        row(
            "a-rank-an-answer-omits-is-no-rank-at-once",
            3,
            None,
            vec![
                Step::Poll(vec![rank(100), rank(200), rank(300)]),
                pick(1),
                Step::Poll(vec![None, rank(200), rank(300)]),
                pick(1),
                pick(1),
            ],
        ),
        row(
            "a-rank-five-answers-omit-is-no-rank-after-any-of-them",
            3,
            None,
            vec![
                Step::Poll(vec![rank(100), rank(200), rank(300)]),
                pick(1),
                omitted(),
                pick(1),
                omitted(),
                pick(1),
                omitted(),
                pick(1),
                omitted(),
                pick(1),
                omitted(),
                pick(1),
            ],
        ),
        row(
            "a-row-with-no-token-capacity-is-no-rank",
            2,
            None,
            vec![Step::Poll(vec![bare(10), rank(500)]), pick(1), pick(1)],
        ),
        row(
            "an-answer-with-no-token-capacity-drops-the-worker-s-ranks",
            2,
            None,
            vec![
                two(),
                pick(1),
                Step::Poll(vec![bare(100), bare(200)]),
                pick(1),
                two(),
                pick(1),
            ],
        ),
        row(
            "an-answer-with-no-row-drops-the-worker-s-ranks",
            2,
            None,
            vec![two(), pick(1), Step::Poll(vec![None, None]), pick(1)],
        ),
        row(
            "a-failed-poll-keeps-the-last-answer-and-its-bookings",
            2,
            None,
            vec![two(), pick(150), Step::Failed, pick(1), pick(1)],
        ),
        row(
            "a-worker-never-polled-has-no-rank",
            2,
            None,
            vec![pick(1), Step::Failed, pick(1)],
        ),
        row(
            "an-evicted-worker-has-no-rank-until-its-next-answer",
            2,
            None,
            vec![two(), pick(1), Step::Evict, pick(1), two(), pick(1)],
        ),
        row(
            "a-worker-of-one-rank-takes-its-own-rank-alone",
            2,
            Some(1),
            vec![Step::Poll(vec![rank(10), rank(500)]), pick(1), pick(1)],
        ),
        row(
            "a-worker-of-one-rank-the-answer-omits-has-none",
            2,
            Some(1),
            vec![Step::Poll(vec![rank(10), None]), pick(1)],
        ),
        row(
            "an-answer-with-no-row-drops-the-one-rank-of-its-worker",
            2,
            Some(1),
            vec![
                Step::Poll(vec![rank(10), rank(500)]),
                pick(1),
                Step::Poll(vec![None, None]),
                pick(1),
            ],
        ),
        row(
            "the-used-tokens-decide-where-a-rank-holds-fewer-context-tokens",
            2,
            None,
            vec![
                Step::Poll(vec![held(100, 900), held(200, 50)]),
                pick(1),
                pick(1),
            ],
        ),
        row(
            "a-rank-that-reports-no-context-tokens-is-a-rank-by-its-used-tokens",
            2,
            None,
            vec![
                Step::Poll(vec![silent(100), silent(200)]),
                pick(1),
                pick(200),
                pick(1),
            ],
        ),
    ]
}

#[tokio::test]
#[ignore = "records the classic rows into CLASSIC_ROWS_OUT"]
async fn record_the_classic_minimum_tokens_rows() {
    let out = PathBuf::from(
        std::env::var_os("CLASSIC_ROWS_OUT").expect("CLASSIC_ROWS_OUT names the output directory"),
    );
    let commit =
        std::env::var("CLASSIC_ROWS_COMMIT").expect("CLASSIC_ROWS_COMMIT names the commit");
    let sheet = Sheet {
        recorded_from: commit,
        policy: "minimum_tokens",
        rows: rows(),
    };
    fs::create_dir_all(&out).expect("the output directory");
    let path = out.join("minimum_tokens.json");
    let text = serde_json::to_string_pretty(&sheet).expect("the sheet serializes");
    fs::write(&path, text + "\n").expect("the sheet is written");
    println!("{} rows -> {}", sheet.rows.len(), path.display());
}
