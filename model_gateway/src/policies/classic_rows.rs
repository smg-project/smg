//! The recorder of the classic power-of-two outcomes: each row is a pool of
//! workers, the reports the policy has of them, one request shape and what
//! the classic policy picked for it. The policy controller crate replays the
//! rows through its own decision (`crates/policy_controller/tests/classic_rows.rs`).
//!
//! This file is an ignored test of the gateway crate, kept beside the code
//! it runs. It records from the checkout it is built in and writes the sheet
//! under `CLASSIC_ROWS_OUT`, naming `CLASSIC_ROWS_COMMIT` as the commit it
//! recorded from. The sheets the policy controller crate replays are in
//! `crates/policy_controller/tests/classic/sheets`; to record this one again:
//!
//! ```text
//! CLASSIC_ROWS_OUT=<dir> CLASSIC_ROWS_COMMIT=<commit> \
//!   cargo test -p smg --lib policies::classic_rows -- --ignored --nocapture
//! ```
//!
//! A sequence row is recorded twenty times from a fresh policy and must
//! come out the same each time; a drawn row records the count of each
//! worker over its draws, each draw from a fresh policy.

use std::{collections::HashMap, fs, path::PathBuf, sync::Arc, thread, time::Duration};

use openai_protocol::worker::{SchedulerLoadSnapshot, WorkerLoadResponse};
use serde::Serialize;
use smg_worker::{identity::basic::Worker, load::state::TestFeed};

use super::{LoadBalancingPolicy, PowerOfTwoPolicy, SelectWorkerInfo};
use crate::{config::PowerOfTwoLoadMetric, policies::test_support::make_worker};

/// What a worker's report says: running and waiting requests, the resident
/// context tokens when the engine counts them, the token-work of its queue,
/// the tokens per second it drains (`0`: unknown) and its KV-cache usage.
#[derive(Clone, Copy, Serialize)]
struct Report {
    running: u32,
    queued: u32,
    resident_tokens: Option<u32>,
    waiting_tokens: u32,
    throughput: u32,
    kv_usage_permille: u16,
}

/// One worker of a row: the requests the router holds on it and its report,
/// if the policy has one.
#[derive(Clone, Copy, Serialize)]
struct WorkerState {
    in_flight: u32,
    report: Option<Report>,
}

/// One step of a sequence row.
#[derive(Clone, Serialize)]
#[serde(rename_all = "snake_case")]
enum Step {
    /// One decision; the picked worker then holds one more request.
    Pick,
    /// A new report of every worker (`None`: no report of it), sampled
    /// after the picks before it.
    Report(Vec<Option<Report>>),
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum Outcome {
    /// The worker each `Pick` step chose, in order.
    Sequence(Vec<usize>),
    /// How often each worker was chosen over `draws` single decisions.
    Counts { draws: u32, counts: Vec<u32> },
}

#[derive(Serialize)]
struct Row {
    name: &'static str,
    /// The load metric in its command-line spelling.
    metric: &'static str,
    workers: Vec<WorkerState>,
    /// The request's prompt size; `None` is a request of unknown size.
    prompt_tokens: Option<u32>,
    /// The steps of a sequence row; empty for a drawn row.
    steps: Vec<Step>,
    outcome: Outcome,
}

#[derive(Serialize)]
struct Sheet {
    recorded_from: String,
    policy: &'static str,
    rows: Vec<Row>,
}

fn url(i: usize) -> String {
    format!("http://w{i}:8000")
}

fn snapshot(report: Report) -> WorkerLoadResponse {
    WorkerLoadResponse {
        dp_rank_count: 1,
        loads: vec![SchedulerLoadSnapshot {
            dp_rank: 0,
            num_running_reqs: report.running as i32,
            num_waiting_reqs: report.queued as i32,
            num_resident_tokens: report.resident_tokens.map(i64::from),
            num_waiting_uncached_tokens: report.waiting_tokens as i32,
            gen_throughput: f64::from(report.throughput),
            token_usage: f64::from(report.kv_usage_permille) / 1000.0,
            ..Default::default()
        }],
        ..Default::default()
    }
}

fn loads_of(reports: impl Iterator<Item = Option<Report>>) -> HashMap<String, WorkerLoadResponse> {
    reports
        .enumerate()
        .filter_map(|(i, report)| Some((url(i), snapshot(report?))))
        .collect()
}

/// A fresh policy over a fresh pool in the row's starting state.
struct Bed {
    policy: PowerOfTwoPolicy,
    feed: TestFeed,
    workers: Vec<Arc<dyn Worker>>,
}

impl Bed {
    fn new(metric: &str, workers: &[WorkerState]) -> Self {
        let metric = PowerOfTwoLoadMetric::parse(metric).expect("a load metric");
        let policy = PowerOfTwoPolicy::with_load_metric(metric);
        let feed = TestFeed::attached_to(|rx| policy.set_load_receiver(rx));
        let pool: Vec<Arc<dyn Worker>> = workers
            .iter()
            .enumerate()
            .map(|(i, state)| {
                let worker = make_worker(&url(i));
                (0..state.in_flight).for_each(|_| worker.increment_load());
                worker
            })
            .collect();
        feed.publish(&loads_of(workers.iter().map(|w| w.report)));
        Self {
            policy,
            feed,
            workers: pool,
        }
    }

    /// One decision and its dispatch: the picked worker holds one more.
    fn pick(&self, prompt: Option<&[u32]>) -> usize {
        let info = SelectWorkerInfo {
            tokens: prompt,
            ..SelectWorkerInfo::default()
        };
        let idx = self
            .policy
            .select_worker(&self.workers, &info)
            .expect("a pick");
        self.workers[idx].increment_load();
        idx
    }
}

fn sequence(
    metric: &'static str,
    workers: &[WorkerState],
    prompt: Option<&[u32]>,
    steps: &[Step],
) -> Vec<usize> {
    let run = || {
        let bed = Bed::new(metric, workers);
        let mut picked = Vec::new();
        for step in steps {
            match step {
                Step::Pick => picked.push(bed.pick(prompt)),
                Step::Report(reports) => {
                    // A report is sampled after the dispatches before it.
                    thread::sleep(Duration::from_millis(2));
                    bed.feed.publish(&loads_of(reports.iter().copied()));
                }
            }
        }
        picked
    };
    let first = run();
    for _ in 0..19 {
        assert_eq!(run(), first, "a sequence row must not depend on a draw");
    }
    first
}

fn counts(
    metric: &'static str,
    workers: &[WorkerState],
    prompt: Option<&[u32]>,
    draws: u32,
) -> Vec<u32> {
    let mut counts = vec![0u32; workers.len()];
    for _ in 0..draws {
        counts[Bed::new(metric, workers).pick(prompt)] += 1;
    }
    counts
}

fn reported(in_flight: u32, running: u32, resident_tokens: u32) -> WorkerState {
    WorkerState {
        in_flight,
        report: Some(Report {
            running,
            queued: 0,
            resident_tokens: Some(resident_tokens),
            waiting_tokens: 0,
            throughput: 0,
            kv_usage_permille: 0,
        }),
    }
}

/// A worker whose report holds queued requests beside the running ones.
fn queued(in_flight: u32, running: u32, queued: u32, resident_tokens: u32) -> WorkerState {
    WorkerState {
        in_flight,
        report: Some(Report {
            running,
            queued,
            resident_tokens: Some(resident_tokens),
            waiting_tokens: 0,
            throughput: 0,
            kv_usage_permille: 0,
        }),
    }
}

/// A worker whose report does not count resident tokens.
fn uncounted(in_flight: u32, running: u32) -> WorkerState {
    WorkerState {
        in_flight,
        report: Some(Report {
            running,
            queued: 0,
            resident_tokens: None,
            waiting_tokens: 0,
            throughput: 0,
            kv_usage_permille: 0,
        }),
    }
}

/// A worker that runs one request, drains 100 tokens a second and reports
/// `waiting_tokens` of queued token-work: its expected wait is its queue.
fn queueing(in_flight: u32, waiting_tokens: u32) -> WorkerState {
    WorkerState {
        in_flight,
        report: Some(Report {
            running: 1,
            queued: 0,
            resident_tokens: None,
            waiting_tokens,
            throughput: 100,
            kv_usage_permille: 100,
        }),
    }
}

/// A worker whose report runs `running` requests and drains `throughput`
/// tokens a second (`0`: no rate), with no queue and no KV pressure.
fn rated(in_flight: u32, running: u32, throughput: u32) -> WorkerState {
    WorkerState {
        in_flight,
        report: Some(Report {
            running,
            queued: 0,
            resident_tokens: None,
            waiting_tokens: 0,
            throughput,
            kv_usage_permille: 0,
        }),
    }
}

fn dark(in_flight: u32) -> WorkerState {
    WorkerState {
        in_flight,
        report: None,
    }
}

fn sequence_row(
    name: &'static str,
    metric: &'static str,
    workers: Vec<WorkerState>,
    prompt_tokens: Option<u32>,
    steps: Vec<Step>,
) -> Row {
    let prompt = prompt_tokens.map(|n| vec![0u32; n as usize]);
    let picked = sequence(metric, &workers, prompt.as_deref(), &steps);
    Row {
        name,
        metric,
        workers,
        prompt_tokens,
        steps,
        outcome: Outcome::Sequence(picked),
    }
}

fn drawn_row(
    name: &'static str,
    metric: &'static str,
    workers: Vec<WorkerState>,
    prompt_tokens: Option<u32>,
) -> Row {
    const DRAWS: u32 = 6_000;
    let prompt = prompt_tokens.map(|n| vec![0u32; n as usize]);
    let counts = counts(metric, &workers, prompt.as_deref(), DRAWS);
    Row {
        name,
        metric,
        workers,
        prompt_tokens,
        steps: Vec::new(),
        outcome: Outcome::Counts {
            draws: DRAWS,
            counts,
        },
    }
}

fn picks(n: usize) -> Vec<Step> {
    vec![Step::Pick; n]
}

fn rows() -> Vec<Row> {
    let fleet = || {
        vec![
            reported(4, 4, 900_000),
            reported(12, 12, 130_000),
            reported(5, 5, 810_000),
            reported(2, 2, 950_000),
        ]
    };
    let same_reports = |workers: &[WorkerState]| -> Step {
        Step::Report(workers.iter().map(|w| w.report).collect())
    };
    let pair = vec![reported(1, 1, 2_000), reported(1, 1, 4_000)];
    let mut released = picks(3);
    released.push(same_reports(&pair));
    released.extend(picks(3));
    let mut rows = vec![
        // The resident-token form: the least resident work over every worker.
        sequence_row(
            "resident-tokens-alone-the-fewest-tokens-win",
            "resident_tokens:0",
            fleet(),
            Some(20_000),
            picks(12),
        ),
        sequence_row(
            "resident-tokens-each-request-priced-at-100k",
            "resident_tokens:100000",
            fleet(),
            Some(20_000),
            picks(12),
        ),
        sequence_row(
            "resident-tokens-a-request-of-unknown-size-is-credited-at-the-mean-prefill",
            "resident_tokens:500",
            vec![reported(1, 1, 3_000), reported(1, 1, 3_600), reported(1, 1, 4_500)],
            None,
            picks(9),
        ),
        sequence_row(
            "resident-tokens-a-new-report-releases-the-credit",
            "resident_tokens:1000",
            pair,
            Some(1_500),
            released,
        ),
        sequence_row(
            "resident-tokens-requests-in-transit-are-priced-at-the-mean-context",
            "resident_tokens:1000",
            vec![reported(5, 2, 8_000), reported(3, 3, 21_500), reported(1, 1, 26_500)],
            Some(1_000),
            picks(6),
        ),
        // 8,000 tokens over two running and two queued price the one request
        // in transit at 2,000: 15,000 against 16,000. Over the running alone
        // it would be 17,000.
        sequence_row(
            "resident-tokens-the-mean-context-counts-the-queued-requests-with-the-running",
            "resident_tokens:1000",
            vec![queued(5, 2, 2, 8_000), reported(1, 1, 15_000)],
            Some(1_000),
            picks(1),
        ),
        sequence_row(
            "resident-tokens-a-worker-that-does-not-count-them-scores-from-its-best-peer",
            "resident_tokens:0",
            vec![reported(2, 2, 500_000), uncounted(2, 1)],
            Some(1_000),
            picks(5),
        ),
        sequence_row(
            "resident-tokens-a-worker-with-no-report-scores-from-its-best-peer",
            "resident_tokens:1000",
            vec![reported(1, 1, 5_000), dark(2), reported(3, 3, 30_000)],
            Some(1_000),
            picks(4),
        ),
        sequence_row(
            "resident-tokens-nobody-counts-them-the-in-flight-count-decides",
            "resident_tokens:1000",
            vec![uncounted(5, 5), dark(1), uncounted(4, 4)],
            Some(1_000),
            picks(3),
        ),
        drawn_row(
            "resident-tokens-equal-work-is-drawn",
            "resident_tokens:1000",
            vec![reported(2, 2, 6_000), reported(2, 2, 6_000), reported(2, 2, 6_001)],
            Some(1_000),
        ),
        drawn_row(
            "resident-tokens-an-idle-worker-with-no-report-ties-with-its-best-peer",
            "resident_tokens:1000",
            vec![reported(2, 2, 6_000), dark(0)],
            Some(1_000),
        ),
        // Two reporting peers of different work: the peer an unreported
        // worker is priced at is the best one.
        drawn_row(
            "resident-tokens-an-idle-worker-with-no-report-ties-with-the-best-of-two-peers",
            "resident_tokens:1000",
            vec![reported(2, 2, 6_000), reported(2, 2, 9_000), dark(0)],
            Some(1_000),
        ),
        drawn_row(
            "resident-tokens-an-idle-worker-that-does-not-count-them-ties-with-its-best-peer",
            "resident_tokens:1000",
            vec![reported(2, 2, 6_000), uncounted(0, 0)],
            Some(1_000),
        ),
        // The count is present and zero while a request runs: a report
        // sampled before the request's tokens are resident.
        drawn_row(
            "resident-tokens-a-running-worker-that-reports-zero-of-them-takes-the-request",
            "resident_tokens:1000",
            vec![reported(2, 2, 6_000), reported(1, 1, 0)],
            Some(1_000),
        ),
        // A worker whose report read idle while the router holds requests
        // on it, in each kind of fleet: every engine counts resident
        // tokens, none does, one does.
        drawn_row(
            "resident-tokens-every-worker-counts-them-and-the-idle-one-takes-the-request",
            "resident_tokens:1000",
            vec![reported(2, 2, 6_000), reported(0, 0, 0)],
            Some(1_000),
        ),
        sequence_row(
            "resident-tokens-every-worker-counts-them-and-the-worker-idle-at-its-report-holds-three-in-transit",
            "resident_tokens:1000",
            vec![reported(2, 2, 6_000), reported(3, 0, 0)],
            Some(1_000),
            picks(4),
        ),
        sequence_row(
            "resident-tokens-nobody-counts-them-and-the-worker-idle-at-its-report-holds-four-in-transit",
            "resident_tokens:1000",
            vec![uncounted(4, 0), uncounted(1, 1)],
            Some(1_000),
            picks(2),
        ),
        sequence_row(
            "resident-tokens-nobody-counts-them-and-the-worker-idle-at-its-report-holds-the-most-of-four",
            "resident_tokens:1000",
            vec![uncounted(9, 0), uncounted(1, 1), uncounted(3, 3), uncounted(5, 5)],
            Some(1_000),
            picks(2),
        ),
        sequence_row(
            "resident-tokens-nobody-counts-them-and-the-worker-idle-at-its-report-holds-thirty-over-twenty-picks",
            "resident_tokens:1000",
            vec![uncounted(30, 0), uncounted(1, 1)],
            Some(1_000),
            picks(20),
        ),
        sequence_row(
            "resident-tokens-nobody-counts-them-the-tokens-alone-and-the-worker-idle-at-its-report-holds-four",
            "resident_tokens:0",
            vec![uncounted(4, 0), uncounted(1, 1)],
            Some(1_000),
            picks(2),
        ),
        drawn_row(
            "resident-tokens-nobody-counts-them-a-peer-the-router-holds-nothing-on-beside-the-idle-report-that-holds-five",
            "resident_tokens:1000",
            vec![uncounted(5, 0), uncounted(0, 1)],
            Some(1_000),
        ),
        sequence_row(
            "resident-tokens-nobody-counts-them-every-report-busy-the-in-flight-count-decides",
            "resident_tokens:1000",
            vec![uncounted(4, 4), uncounted(1, 1)],
            Some(1_000),
            picks(2),
        ),
        sequence_row(
            "resident-tokens-nobody-counts-them-a-peer-with-no-report-beside-the-idle-report-that-holds-four",
            "resident_tokens:1000",
            vec![uncounted(4, 0), dark(1)],
            Some(1_000),
            picks(2),
        ),
        sequence_row(
            "resident-tokens-nobody-counts-them-a-queueing-peer-beside-the-idle-report-that-holds-three",
            "resident_tokens:1000",
            vec![uncounted(3, 0), queueing(1, 50_000)],
            Some(1_000),
            picks(1),
        ),
        sequence_row(
            "resident-tokens-one-worker-counts-them-and-of-the-two-that-do-not-the-idle-report-holds-six",
            "resident_tokens:1000",
            vec![reported(40, 40, 900_000), uncounted(6, 0), uncounted(1, 1)],
            Some(1_000),
            picks(3),
        ),
        // The count forms and the default, on a pool with no report: the
        // in-flight count decides.
        sequence_row(
            "least-requests-the-fewest-in-flight-over-every-worker",
            "least_requests",
            vec![dark(6), dark(7), dark(1), dark(5)],
            None,
            picks(4),
        ),
        drawn_row(
            "least-requests-the-fewest-tie-and-are-drawn",
            "least_requests",
            vec![dark(2), dark(2), dark(5), dark(2)],
            None,
        ),
        drawn_row(
            "requests-the-fewer-in-flight-of-two-drawn-workers",
            "requests",
            vec![dark(0), dark(1), dark(2), dark(3)],
            None,
        ),
        drawn_row(
            "expected-wait-the-better-of-two-drawn-workers",
            "expected_wait",
            vec![dark(0), dark(1), dark(2), dark(3)],
            None,
        ),
        drawn_row(
            "expected-wait-a-pool-of-equals-is-drawn-evenly",
            "expected_wait",
            vec![dark(1), dark(1), dark(1), dark(1), dark(1)],
            None,
        ),
        // The count forms against reports that say otherwise: the count
        // comes first, the expected wait decides between equal counts.
        sequence_row(
            "expected-wait-the-shorter-queue-beats-fewer-in-flight",
            "expected_wait",
            vec![queueing(5, 0), queueing(2, 50_000)],
            None,
            picks(2),
        ),
        sequence_row(
            "requests-fewer-in-flight-beats-the-shorter-queue",
            "requests",
            vec![queueing(5, 0), queueing(2, 50_000)],
            None,
            picks(2),
        ),
        sequence_row(
            "requests-equal-counts-go-to-the-shorter-queue",
            "requests",
            vec![queueing(3, 50_000), queueing(3, 0)],
            None,
            picks(1),
        ),
        sequence_row(
            "least-requests-the-fewest-in-flight-beats-the-shorter-queue",
            "least_requests",
            vec![
                queueing(4, 0),
                queueing(5, 0),
                queueing(1, 50_000),
                queueing(3, 0),
            ],
            None,
            picks(2),
        ),
        sequence_row(
            "least-requests-equal-counts-go-to-the-shorter-queue",
            "least_requests",
            vec![queueing(2, 50_000), queueing(2, 0), queueing(5, 0)],
            None,
            picks(1),
        ),
    ];
    rows.extend(count_form_rows());
    rows
}

/// The count forms' rows: three pools, each under the three forms that
/// compare a count or a wait. Every row is drawn.
fn count_form_rows() -> Vec<Row> {
    // A holds one request and reports one running at 100 tokens a second; B
    // holds one and has no report; C holds five and reports five running.
    let fast_peer = || vec![rated(1, 1, 100), dark(1), rated(5, 5, 100_000)];
    let slow_peer = || vec![rated(1, 1, 100), dark(1), rated(5, 5, 100)];
    // A reports no rate, B reports 100 a second, C holds five and drains fast.
    let no_rate = || vec![rated(1, 1, 0), rated(1, 1, 100), rated(5, 5, 100_000)];
    vec![
        drawn_row(
            "least-requests-two-fewest-and-a-fast-peer-of-a-higher-count",
            "least_requests",
            fast_peer(),
            None,
        ),
        drawn_row(
            "requests-two-fewest-and-a-fast-peer-of-a-higher-count",
            "requests",
            fast_peer(),
            None,
        ),
        drawn_row(
            "expected-wait-two-fewest-and-a-fast-peer-of-a-higher-count",
            "expected_wait",
            fast_peer(),
            None,
        ),
        drawn_row(
            "least-requests-two-fewest-and-a-slow-peer-of-a-higher-count",
            "least_requests",
            slow_peer(),
            None,
        ),
        drawn_row(
            "requests-two-fewest-and-a-slow-peer-of-a-higher-count",
            "requests",
            slow_peer(),
            None,
        ),
        drawn_row(
            "expected-wait-two-fewest-and-a-slow-peer-of-a-higher-count",
            "expected_wait",
            slow_peer(),
            None,
        ),
        drawn_row(
            "least-requests-a-worker-with-no-rate-beside-a-rated-one-and-a-fast-peer",
            "least_requests",
            no_rate(),
            None,
        ),
        drawn_row(
            "requests-a-worker-with-no-rate-beside-a-rated-one-and-a-fast-peer",
            "requests",
            no_rate(),
            None,
        ),
        drawn_row(
            "expected-wait-a-worker-with-no-rate-beside-a-rated-one-and-a-fast-peer",
            "expected_wait",
            no_rate(),
            None,
        ),
    ]
}

#[test]
#[ignore = "records the classic rows into CLASSIC_ROWS_OUT"]
fn record_the_classic_power_of_two_rows() {
    let out = PathBuf::from(
        std::env::var_os("CLASSIC_ROWS_OUT").expect("CLASSIC_ROWS_OUT names the output directory"),
    );
    let commit =
        std::env::var("CLASSIC_ROWS_COMMIT").expect("CLASSIC_ROWS_COMMIT names the commit");
    let sheet = Sheet {
        recorded_from: commit,
        policy: "power_of_two",
        rows: rows(),
    };
    fs::create_dir_all(&out).expect("the output directory");
    let path = out.join("power_of_two.json");
    let text = serde_json::to_string_pretty(&sheet).expect("the sheet serializes");
    fs::write(&path, text + "\n").expect("the sheet is written");
    println!("{} rows -> {}", sheet.rows.len(), path.display());
}
