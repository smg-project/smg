//! The recorder of the gateway's overload verdict: each row is one worker
//! with its data-parallel ranks, the thresholds the gateway holds for every
//! worker and the ones the worker names for itself, a list of steps and
//! whether the worker reads as overloaded after each. A step is what the
//! load monitor is handed: a poll's answer (a row a rank), a poll
//! that brought no report, a load record one rank pushed on its event
//! stream, or the worker's loads dropped as it leaves the ready set. The
//! monitor judges a poll's answer by the answer, and a pushed record by the
//! worker's pushed report: the ranks that have pushed, each at its last
//! record or at a later poll's figures. The
//! verdict is the monitor's own, read off the worker after each step. The
//! policy controller crate replays the rows through its store
//! (`crates/policy_controller/tests/classic/overload.rs`).
//!
//! This file is an ignored test of the worker crate, kept beside the code
//! it runs. It records from the checkout it is built in and writes the sheet
//! under `CLASSIC_ROWS_OUT`, naming `CLASSIC_ROWS_COMMIT` as the commit it
//! recorded from. The sheets the policy controller crate replays are in
//! `crates/policy_controller/tests/classic/sheets`; to record this one again:
//!
//! ```text
//! CLASSIC_ROWS_OUT=<dir> CLASSIC_ROWS_COMMIT=<commit> \
//!   cargo test -p smg-worker --all-features --lib load::monitor::classic_overload_rows -- --ignored --nocapture
//! ```
//!
//! Nothing in a row draws. The recorder runs each row twice, each time from
//! a fresh monitor and a fresh worker, and stops if the two runs differ.

use std::{fs, path::PathBuf};

use serde::Serialize;

use super::{tests::PushedRecord, *};
use crate::{
    health::overload::OverloadThresholds,
    identity::basic::WorkerType,
    model_card::ModelCard,
    test_support::{BasicWorkerBuilder, LoadReaders, WorkerMonitor, WorkerRegistry},
    worker::{HealthCheckConfig, OverloadUpdate, WorkerStatus},
};

const URL: &str = "http://w0:8000";
const MODEL: &str = "m";

/// A pair of thresholds: the queued requests summed over a worker's ranks
/// and the mean KV usage over them, each at or above which the worker is
/// vetoed. `null` leaves a signal off.
#[derive(Clone, Copy, Serialize)]
struct Thresholds {
    waiting_requests: Option<usize>,
    token_usage: Option<f64>,
}

const fn thresholds(waiting_requests: usize, token_usage: f64) -> Thresholds {
    Thresholds {
        waiting_requests: Some(waiting_requests),
        token_usage: Some(token_usage),
    }
}

const OFF: Thresholds = Thresholds {
    waiting_requests: None,
    token_usage: None,
};

/// One rank's figures, in a poll's answer or a pushed record: its queued
/// requests, its KV usage, and the share of that usage its running requests
/// hold where the rank reports one.
#[derive(Clone, Copy, Serialize)]
struct Figures {
    waiting: u32,
    usage: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    active: Option<f64>,
}

#[expect(
    clippy::unnecessary_wraps,
    reason = "an entry of a poll's answer: `None` is a rank the answer omits"
)]
const fn rank(waiting: u32, usage: f64) -> Option<Figures> {
    Some(Figures {
        waiting,
        usage,
        active: None,
    })
}

#[expect(
    clippy::unnecessary_wraps,
    reason = "an entry of a poll's answer: `None` is a rank the answer omits"
)]
const fn rank_active(waiting: u32, usage: f64, active: f64) -> Option<Figures> {
    Some(Figures {
        waiting,
        usage,
        active: Some(active),
    })
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "snake_case")]
enum Step {
    /// A poll's answer: a row a rank, in rank order, `null` where the answer
    /// omits the rank. An answer with no row at all is what the monitor
    /// takes for a poll that brought no report, and the engine's own word
    /// that the ranks it pushed for before are gone.
    Poll(Vec<Option<Figures>>),
    /// A poll that brought no report.
    NoReport,
    /// A load record the rank pushed on its event stream.
    Record { rank: usize, figures: Figures },
    /// The worker leaves the ready set and its loads are dropped.
    Evict,
}

fn record(rank: usize, waiting: u32, usage: f64) -> Step {
    Step::Record {
        rank,
        figures: Figures {
            waiting,
            usage,
            active: None,
        },
    }
}

/// How many times a row repeats the step that leaves a rank unheard: more
/// report periods than a reader that keeps a silent rank's last figures
/// for a while would keep them.
const AGAIN: usize = 5;

/// `first`, then `again` [`AGAIN`] times, then `last`.
fn again(first: Vec<Step>, again: Step, last: Vec<Step>) -> Vec<Step> {
    first
        .into_iter()
        .chain(std::iter::repeat_n(again, AGAIN))
        .chain(last)
        .collect()
}

/// Whether the worker reads as overloaded after each step, in order.
#[derive(Serialize)]
struct Outcome {
    verdicts: Vec<bool>,
}

#[derive(Serialize)]
struct Row {
    name: &'static str,
    /// The thresholds the gateway holds for every worker.
    gateway: Thresholds,
    /// The thresholds the worker names for itself: a value it names stands
    /// over the gateway's.
    worker: Thresholds,
    /// The worker's data-parallel ranks.
    ranks: usize,
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
                    num_waiting_reqs: count(figures.waiting),
                    token_usage: figures.usage,
                    active_token_usage: figures.active,
                    ..SchedulerLoadSnapshot::default()
                })
            })
            .collect(),
        ..WorkerLoadResponse::default()
    }
}

/// One run of a row from a fresh monitor and a fresh worker: the worker's
/// verdict after each step. Each step enters the monitor as the gateway's
/// own loop hands it over: a poll's answer goes to the reports on file and
/// then to the verdict, an answer with no row and a failed poll go to the
/// verdict as no report, the answer with no row then takes the ranks that
/// pushed before it off the worker's pushed report, a pushed record goes
/// through the record's own entry, and the eviction through the monitor's
/// own.
fn run(gateway: Thresholds, own: Thresholds, ranks: usize, steps: &[Step]) -> Vec<bool> {
    let registry = Arc::new(WorkerRegistry::new());
    let monitor = Arc::new(WorkerMonitor::new(
        Arc::clone(&registry),
        Arc::new(LoadReaders::Blind),
        1,
        false,
        false,
    ));
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(URL)
            .worker_type(WorkerType::Regular)
            .model(ModelCard::new(MODEL))
            .overload(OverloadUpdate {
                waiting_requests: own.waiting_requests,
                token_usage: own.token_usage,
            })
            .overload_defaults(OverloadThresholds {
                waiting_requests: gateway.waiting_requests,
                token_usage: gateway.token_usage,
            })
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build(),
    );
    worker.set_status(WorkerStatus::Ready);
    registry
        .register(Arc::clone(&worker))
        .expect("the worker registers");
    let start = Instant::now();
    steps
        .iter()
        .enumerate()
        .map(|(at, step)| {
            // A step a second: no record is newer than the poll after it.
            let now = start + Duration::from_secs(at as u64 + 1);
            match step {
                Step::Poll(figures) => {
                    let load = answer(ranks, figures);
                    let asked = format!("{:?}", load.loads);
                    // The engine's answer as the loop hands it over: to the
                    // sink and the reports on file under one hold, then to
                    // the tick's end, where an answer with rows is judged.
                    // An answer with no row is the engine's own word that it
                    // has no rank: the verdict reads no report, and the
                    // ranks that pushed before the poll leave the worker's
                    // pushed report.
                    let mut engine = vec![(Arc::clone(&worker), Some(load))];
                    let carries_rank_capacity = monitor.reconcile_engine(&mut engine, now);
                    let reports: Vec<_> = engine
                        .into_iter()
                        .zip(carries_rank_capacity)
                        .filter_map(|((worker, response), carries_rank_capacity)| {
                            let load = response.filter(|load| !load.loads.is_empty())?;
                            assert_eq!(
                                format!("{:?}", load.loads),
                                asked,
                                "no rank pushed since the poll began"
                            );
                            Some((worker, load, carries_rank_capacity))
                        })
                        .collect();
                    monitor.end_tick(std::slice::from_ref(&worker), &reports);
                }
                Step::NoReport => {
                    let mut engine = vec![(Arc::clone(&worker), None)];
                    monitor.reconcile_engine(&mut engine, now);
                }
                Step::Record { rank, figures } => {
                    let mut record = PushedRecord::requests(0, count(figures.waiting))
                        .token_usage(figures.usage);
                    record.snapshot.active_token_usage = figures.active;
                    monitor.apply_pushed_load(
                        &worker,
                        i32::try_from(*rank).expect("a rank fits the record's field"),
                        &record,
                        now,
                    );
                }
                Step::Evict => monitor.evict_worker_loads(&worker),
            }
            worker.is_overloaded()
        })
        .collect()
}

fn row(
    name: &'static str,
    gateway: Thresholds,
    worker: Thresholds,
    ranks: usize,
    steps: Vec<Step>,
) -> Row {
    let verdicts = run(gateway, worker, ranks, &steps);
    assert_eq!(
        verdicts,
        run(gateway, worker, ranks, &steps),
        "{name}: two runs of one row differ"
    );
    Row {
        name,
        gateway,
        worker,
        ranks,
        steps,
        outcome: Outcome { verdicts },
    }
}

fn rows() -> Vec<Row> {
    let default = thresholds(8, 0.8);
    // A usage threshold a report's ratio reaches exactly, whatever the
    // arithmetic: a half, and ratios that are sums of powers of two.
    let half = thresholds(8, 0.5);
    vec![
        row(
            "the-queue-vetoes-at-its-count-and-above-summed-over-two-ranks",
            default,
            OFF,
            2,
            vec![
                Step::Poll(vec![rank(3, 0.125), rank(4, 0.125)]),
                Step::Poll(vec![rank(4, 0.125), rank(4, 0.125)]),
                Step::Poll(vec![rank(9, 0.125), rank(0, 0.125)]),
                Step::Poll(vec![rank(0, 0.125), rank(7, 0.125)]),
            ],
        ),
        row(
            "the-usage-vetoes-at-its-mean-and-above-over-two-ranks",
            half,
            OFF,
            2,
            vec![
                Step::Poll(vec![rank(0, 0.25), rank(0, 0.5)]),
                Step::Poll(vec![rank(0, 0.25), rank(0, 0.75)]),
                Step::Poll(vec![rank(0, 0.5), rank(0, 0.5)]),
                Step::Poll(vec![rank(0, 1.0), rank(0, 0.0)]),
                Step::Poll(vec![rank(0, 0.375), rank(0, 0.5)]),
            ],
        ),
        row(
            "either-signal-alone-vetoes",
            half,
            OFF,
            1,
            vec![
                Step::Poll(vec![rank(8, 0.0)]),
                Step::Poll(vec![rank(7, 0.25)]),
                Step::Poll(vec![rank(0, 0.5)]),
                Step::Poll(vec![rank(0, 0.0)]),
            ],
        ),
        row(
            "a-count-of-one-vetoes-any-queue-and-a-usage-of-one-only-a-full-rank",
            thresholds(1, 1.0),
            OFF,
            1,
            vec![
                Step::Poll(vec![rank(0, 0.75)]),
                Step::Poll(vec![rank(1, 0.0)]),
                Step::Poll(vec![rank(0, 1.0)]),
                Step::Poll(vec![rank(0, 0.0)]),
            ],
        ),
        row(
            "two-ranks-that-trade-their-queues-in-one-answer-move-no-verdict",
            default,
            OFF,
            2,
            vec![
                Step::Poll(vec![rank(6, 0.125), rank(1, 0.125)]),
                Step::Poll(vec![rank(1, 0.125), rank(6, 0.125)]),
                Step::Poll(vec![rank(6, 0.125), rank(1, 0.125)]),
            ],
        ),
        row(
            "two-ranks-that-trade-their-usage-in-one-answer-move-no-verdict",
            half,
            OFF,
            2,
            vec![
                Step::Poll(vec![rank(0, 0.75), rank(0, 0.125)]),
                Step::Poll(vec![rank(0, 0.125), rank(0, 0.75)]),
                Step::Poll(vec![rank(0, 0.75), rank(0, 0.125)]),
            ],
        ),
        row(
            "a-poll-that-brought-no-report-clears-the-veto",
            default,
            OFF,
            1,
            vec![
                Step::Poll(vec![rank(9, 0.125)]),
                Step::NoReport,
                Step::NoReport,
                Step::Poll(vec![rank(9, 0.125)]),
                Step::Poll(vec![rank(1, 0.125)]),
            ],
        ),
        row(
            "a-record-after-a-poll-that-brought-no-report-is-judged-again",
            default,
            OFF,
            1,
            vec![
                record(0, 9, 0.125),
                Step::NoReport,
                record(0, 9, 0.125),
                Step::NoReport,
                Step::Poll(vec![rank(9, 0.125)]),
            ],
        ),
        row(
            "an-answer-that-names-no-rank-reads-as-no-report",
            default,
            OFF,
            2,
            vec![
                Step::Poll(vec![rank(9, 0.125), rank(0, 0.125)]),
                Step::Poll(vec![None, None]),
                Step::Poll(vec![rank(9, 0.125), rank(0, 0.125)]),
            ],
        ),
        row(
            "both-signals-off-writes-no-verdict",
            OFF,
            OFF,
            2,
            vec![
                Step::Poll(vec![rank(1000, 1.0), rank(1000, 1.0)]),
                record(0, 1000, 1.0),
                Step::NoReport,
            ],
        ),
        row(
            "a-worker-s-own-count-stands-over-the-gateway-s-and-the-usage-stays-the-gateway-s",
            half,
            Thresholds {
                waiting_requests: Some(3),
                token_usage: None,
            },
            1,
            vec![
                Step::Poll(vec![rank(2, 0.0)]),
                Step::Poll(vec![rank(3, 0.0)]),
                Step::Poll(vec![rank(0, 0.25)]),
                Step::Poll(vec![rank(0, 0.5)]),
            ],
        ),
        row(
            "a-worker-s-own-values-veto-under-a-gateway-with-both-signals-off",
            OFF,
            thresholds(3, 0.5),
            1,
            vec![
                Step::Poll(vec![rank(2, 0.25)]),
                Step::Poll(vec![rank(3, 0.0)]),
                Step::Poll(vec![rank(0, 0.5)]),
                Step::Poll(vec![rank(0, 0.0)]),
            ],
        ),
        row(
            "a-worker-that-names-its-count-alone-under-a-gateway-with-both-signals-off-reads-no-usage",
            OFF,
            Thresholds {
                waiting_requests: Some(3),
                token_usage: None,
            },
            1,
            vec![
                Step::Poll(vec![rank(2, 1.0)]),
                Step::Poll(vec![rank(3, 0.0)]),
                Step::Poll(vec![rank(0, 1.0)]),
            ],
        ),
        row(
            "a-worker-that-names-its-usage-alone-under-a-gateway-with-both-signals-off-reads-no-queue",
            OFF,
            Thresholds {
                waiting_requests: None,
                token_usage: Some(0.5),
            },
            1,
            vec![
                Step::Poll(vec![rank(1000, 0.25)]),
                Step::Poll(vec![rank(0, 0.5)]),
                Step::Poll(vec![rank(1000, 0.0)]),
            ],
        ),
        row(
            "the-usage-is-the-running-requests-share-where-a-rank-reports-one",
            half,
            OFF,
            1,
            vec![
                Step::Poll(vec![rank_active(0, 1.0, 0.25)]),
                Step::Poll(vec![rank_active(0, 0.25, 0.75)]),
                Step::Poll(vec![rank_active(0, 0.75, 0.5)]),
            ],
        ),
        row(
            "a-pushed-record-moves-the-verdict-by-its-own-rank",
            default,
            OFF,
            2,
            vec![
                record(0, 5, 0.125),
                record(1, 5, 0.125),
                record(1, 0, 0.125),
                record(0, 8, 0.125),
                record(0, 0, 0.125),
            ],
        ),
        row(
            "a-record-after-a-poll-is-judged-beside-the-poll-s-figures-once-every-rank-has-pushed",
            default,
            OFF,
            2,
            vec![
                record(0, 1, 0.125),
                record(1, 1, 0.125),
                Step::Poll(vec![rank(4, 0.125), rank(3, 0.125)]),
                record(0, 5, 0.125),
                record(1, 0, 0.125),
            ],
        ),
        row(
            "a-poll-after-a-record-is-judged-by-the-poll-s-own-figures",
            default,
            OFF,
            2,
            vec![
                record(0, 9, 0.125),
                Step::Poll(vec![rank(1, 0.125), rank(1, 0.125)]),
                Step::Poll(vec![rank(4, 0.125), rank(4, 0.125)]),
            ],
        ),
        row(
            "the-worker-s-loads-dropped-clears-the-veto",
            default,
            OFF,
            1,
            vec![
                Step::Poll(vec![rank(9, 0.125)]),
                Step::Evict,
                Step::Poll(vec![rank(9, 0.125)]),
            ],
        ),
        // The six rows the policy controller crate decides differently,
        // each by a rule of its own that its replay names: a usage inside
        // half a permille of the threshold; a rank an answer omits, by the
        // queue's sum and by the usage's mean; and the three that show what
        // a pushed record is judged over, the ranks that have pushed and no
        // rank a poll alone reported, less the ranks an answer with no row
        // took off the report. Each but the first repeats the step that
        // leaves a rank unheard, so the row shows what the gateway reads
        // of that rank answer after answer.
        row(
            "a-usage-under-the-threshold-by-less-than-half-a-permille-is-under-it",
            half,
            OFF,
            1,
            vec![
                Step::Poll(vec![rank(0, 0.4996)]),
                Step::Poll(vec![rank(0, 0.4994)]),
            ],
        ),
        row(
            "a-rank-an-answer-omits-leaves-the-sum-at-once",
            default,
            OFF,
            2,
            again(
                vec![Step::Poll(vec![rank(1, 0.125), rank(9, 0.125)])],
                Step::Poll(vec![rank(1, 0.125), None]),
                Vec::new(),
            ),
        ),
        row(
            "a-rank-an-answer-omits-leaves-the-mean-at-once",
            half,
            OFF,
            2,
            again(
                vec![Step::Poll(vec![rank(0, 0.75), rank(0, 0.125)])],
                Step::Poll(vec![rank(0, 0.75), None]),
                Vec::new(),
            ),
        ),
        row(
            "a-record-after-a-poll-is-judged-over-the-ranks-that-have-pushed",
            default,
            OFF,
            2,
            again(
                vec![Step::Poll(vec![rank(4, 0.125), rank(3, 0.125)])],
                record(0, 5, 0.125),
                vec![record(1, 0, 0.125)],
            ),
        ),
        row(
            "a-record-s-usage-is-judged-over-the-ranks-that-have-pushed",
            half,
            OFF,
            2,
            again(
                vec![Step::Poll(vec![rank(0, 0.125), rank(0, 0.125)])],
                record(0, 0, 0.75),
                vec![record(1, 0, 0.75)],
            ),
        ),
        row(
            "an-answer-that-names-no-rank-takes-the-ranks-that-pushed-before-it-off-the-report",
            default,
            OFF,
            2,
            again(
                vec![
                    record(0, 5, 0.125),
                    record(1, 5, 0.125),
                    Step::Poll(vec![None, None]),
                ],
                record(0, 5, 0.125),
                vec![record(1, 5, 0.125)],
            ),
        ),
    ]
}

#[tokio::test]
#[ignore = "records the classic rows into CLASSIC_ROWS_OUT"]
async fn record_the_classic_overload_rows() {
    let out = PathBuf::from(
        std::env::var_os("CLASSIC_ROWS_OUT").expect("CLASSIC_ROWS_OUT names the output directory"),
    );
    let commit =
        std::env::var("CLASSIC_ROWS_COMMIT").expect("CLASSIC_ROWS_COMMIT names the commit");
    let sheet = Sheet {
        recorded_from: commit,
        policy: "overload",
        rows: rows(),
    };
    fs::create_dir_all(&out).expect("the output directory");
    let path = out.join("overload.json");
    let text = serde_json::to_string_pretty(&sheet).expect("the sheet serializes");
    fs::write(&path, text + "\n").expect("the sheet is written");
    println!("{} rows -> {}", sheet.rows.len(), path.display());
}
