//! The recorder of the classic manual policy's outcomes: each row is a pool
//! of workers, what the router holds of them (the requests in flight, the
//! routing keys with a request in flight), the policy's assignment mode, a
//! list of steps and what the classic policy picked at each request. The
//! policy controller crate replays the rows through its own decision
//! (`crates/policy_controller/tests/classic/manual.rs`).
//!
//! This file is an ignored test of the gateway crate, kept beside the code
//! it runs. It records from the checkout it is built in and writes the sheet
//! under `CLASSIC_ROWS_OUT`, naming `CLASSIC_ROWS_COMMIT` as the commit it
//! recorded from. The sheets the policy controller crate replays are in
//! `crates/policy_controller/tests/classic/sheets`; to record this one again:
//!
//! ```text
//! CLASSIC_ROWS_OUT=<dir> CLASSIC_ROWS_COMMIT=<commit> \
//!   cargo test -p smg --lib policies::classic_manual_rows -- --ignored --nocapture
//! ```
//!
//! A sequence row is recorded twenty times from a fresh policy and must
//! come out the same each time; a drawn row records how often each worker
//! took the row's last request over its draws, each draw from a fresh
//! policy.

use std::{fs, path::PathBuf, sync::Arc};

use openai_protocol::worker::WorkerStatus;
use serde::Serialize;
use smg_worker::identity::basic::Worker;

use super::{LoadBalancingPolicy, ManualConfig, ManualPolicy, SelectWorkerInfo, WorkerRole};
use crate::{config::ManualAssignmentMode, policies::test_support::make_worker};

/// One worker of a row: the requests the router holds on it and the routing
/// keys that have a request in flight on it.
#[derive(Clone, Copy, Serialize)]
struct WorkerState {
    in_flight: u32,
    keys: u32,
}

/// The role a request is placed under: the single-worker path, or one side
/// of a prefill and decode pair.
#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum Role {
    Single,
    Prefill,
    Decode,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "snake_case")]
enum Step {
    /// One request: its routing key, if it carries one, and its role.
    Pick {
        key: Option<&'static str>,
        role: Role,
    },
    /// A worker stops being routable.
    Down(usize),
    /// The worker is routable again.
    Up(usize),
    /// The requests the router holds on each worker, from here on.
    Loads(Vec<u32>),
    /// The routing keys with a request in flight on each worker, from here
    /// on; a count only grows within a row.
    Keys(Vec<u32>),
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum Outcome {
    /// The worker each `Pick` step chose, in order.
    Sequence(Vec<usize>),
    /// How often each worker took the row's last request over `draws` runs
    /// of the row.
    Counts { draws: u32, counts: Vec<u32> },
}

#[derive(Serialize)]
struct Row {
    name: &'static str,
    /// The assignment mode in its configuration spelling.
    assignment: &'static str,
    workers: Vec<WorkerState>,
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

/// A fresh policy over a fresh pool in the row's starting state.
struct Bed {
    policy: ManualPolicy,
    workers: Vec<Arc<dyn Worker>>,
    keys: Vec<u32>,
}

impl Bed {
    fn new(assignment: &str, workers: &[WorkerState]) -> Self {
        let assignment_mode = match assignment {
            "random" => ManualAssignmentMode::Random,
            "min_load" => ManualAssignmentMode::MinLoad,
            "min_group" => ManualAssignmentMode::MinGroup,
            "delegate" => ManualAssignmentMode::Delegate,
            other => panic!("no assignment mode {other}"),
        };
        // No eviction task: a row runs far inside any idle time.
        let policy = ManualPolicy::with_config(ManualConfig {
            eviction_interval_secs: 0,
            max_idle_secs: 0,
            assignment_mode,
        });
        let pool: Vec<Arc<dyn Worker>> = workers
            .iter()
            .enumerate()
            .map(|(i, state)| {
                let worker = make_worker(&url(i));
                (0..state.in_flight).for_each(|_| worker.increment_load());
                (0..state.keys)
                    .for_each(|k| worker.increment_routing_key_load(&format!("held-{i}-{k}")));
                worker
            })
            .collect();
        Self {
            policy,
            workers: pool,
            keys: workers.iter().map(|w| w.keys).collect(),
        }
    }

    /// One step; a request answers the worker it was placed on.
    fn step(&mut self, step: &Step) -> Option<usize> {
        match step {
            Step::Pick { key, role } => {
                let info = SelectWorkerInfo {
                    routing_key: *key,
                    role: match role {
                        Role::Single => WorkerRole::Single,
                        Role::Prefill => WorkerRole::Prefill,
                        Role::Decode => WorkerRole::Decode,
                    },
                    ..SelectWorkerInfo::default()
                };
                Some(
                    self.policy
                        .select_worker(&self.workers, &info)
                        .expect("a pick"),
                )
            }
            Step::Down(i) => {
                self.workers[*i].set_status(WorkerStatus::NotReady);
                None
            }
            Step::Up(i) => {
                self.workers[*i].set_status(WorkerStatus::Ready);
                None
            }
            Step::Loads(loads) => {
                for (worker, &want) in self.workers.iter().zip(loads) {
                    while worker.load() < want as usize {
                        worker.increment_load();
                    }
                    while worker.load() > want as usize {
                        worker.decrement_load();
                    }
                }
                None
            }
            Step::Keys(keys) => {
                for (i, (worker, &want)) in self.workers.iter().zip(keys).enumerate() {
                    assert!(want >= self.keys[i], "a row's key counts only grow");
                    for k in self.keys[i]..want {
                        worker.increment_routing_key_load(&format!("held-{i}-{k}"));
                    }
                    self.keys[i] = want;
                    assert_eq!(worker.routing_key_load(), want as usize);
                }
                None
            }
        }
    }

    fn run(assignment: &str, workers: &[WorkerState], steps: &[Step]) -> Vec<usize> {
        let mut bed = Self::new(assignment, workers);
        steps.iter().filter_map(|step| bed.step(step)).collect()
    }
}

fn sequence_row(
    name: &'static str,
    assignment: &'static str,
    workers: Vec<WorkerState>,
    steps: Vec<Step>,
) -> Row {
    let first = Bed::run(assignment, &workers, &steps);
    for _ in 0..19 {
        assert_eq!(
            Bed::run(assignment, &workers, &steps),
            first,
            "{name}: a sequence row must not depend on a draw"
        );
    }
    Row {
        name,
        assignment,
        workers,
        steps,
        outcome: Outcome::Sequence(first),
    }
}

fn drawn_row(
    name: &'static str,
    assignment: &'static str,
    workers: Vec<WorkerState>,
    steps: Vec<Step>,
) -> Row {
    const DRAWS: u32 = 6_000;
    let mut counts = vec![0u32; workers.len()];
    for _ in 0..DRAWS {
        let picked = Bed::run(assignment, &workers, &steps);
        counts[*picked.last().expect("a drawn row ends with a request")] += 1;
    }
    Row {
        name,
        assignment,
        workers,
        steps,
        outcome: Outcome::Counts {
            draws: DRAWS,
            counts,
        },
    }
}

/// Workers by their requests in flight, holding no routing key.
fn loaded(in_flight: &[u32]) -> Vec<WorkerState> {
    in_flight
        .iter()
        .map(|&in_flight| WorkerState { in_flight, keys: 0 })
        .collect()
}

fn keyed(key: &'static str) -> Step {
    Step::Pick {
        key: Some(key),
        role: Role::Single,
    }
}

fn keyless() -> Step {
    Step::Pick {
        key: None,
        role: Role::Single,
    }
}

fn under(key: &'static str, role: Role) -> Step {
    Step::Pick {
        key: Some(key),
        role,
    }
}

fn rows() -> Vec<Row> {
    vec![
        // The assignment of a key seen first, by mode, and the pin after it.
        sequence_row(
            "manual-min-load-a-key-takes-the-fewest-in-flight-and-stays-past-the-load",
            "min_load",
            loaded(&[3, 1, 2]),
            vec![
                keyed("k1"),
                Step::Loads(vec![0, 5, 2]),
                keyed("k1"),
                keyed("k2"),
                keyed("k1"),
                keyed("k2"),
            ],
        ),
        sequence_row(
            "manual-min-group-a-key-takes-the-fewest-keys-whatever-the-load",
            "min_group",
            vec![
                WorkerState {
                    in_flight: 0,
                    keys: 3,
                },
                WorkerState {
                    in_flight: 5,
                    keys: 0,
                },
                WorkerState {
                    in_flight: 0,
                    keys: 2,
                },
            ],
            vec![
                keyed("k1"),
                Step::Keys(vec![3, 1, 2]),
                keyed("k2"),
                Step::Keys(vec![3, 3, 2]),
                keyed("k3"),
                keyed("k1"),
            ],
        ),
        sequence_row(
            "manual-delegate-with-no-policy-under-it-takes-the-fewest-in-flight",
            "delegate",
            loaded(&[4, 0, 2]),
            vec![keyed("k1"), keyless()],
        ),
        drawn_row(
            "manual-random-a-new-key-is-drawn-over-every-worker",
            "random",
            loaded(&[9, 0, 3]),
            vec![keyed("k1")],
        ),
        drawn_row(
            "manual-random-a-request-with-no-key-is-drawn-over-every-worker",
            "random",
            loaded(&[9, 0, 3]),
            vec![keyless()],
        ),
        drawn_row(
            "manual-min-load-equal-loads-are-drawn",
            "min_load",
            loaded(&[2, 2, 5]),
            vec![keyed("k1")],
        ),
        drawn_row(
            "manual-min-group-equal-key-counts-are-drawn",
            "min_group",
            vec![
                WorkerState {
                    in_flight: 0,
                    keys: 1,
                },
                WorkerState {
                    in_flight: 9,
                    keys: 1,
                },
                WorkerState {
                    in_flight: 0,
                    keys: 4,
                },
            ],
            vec![keyed("k1")],
        ),
        // A request with no key is placed by the mode and pins nothing.
        sequence_row(
            "manual-min-load-a-request-with-no-key-follows-the-load-and-pins-nothing",
            "min_load",
            loaded(&[3, 1, 2]),
            vec![
                keyless(),
                Step::Loads(vec![0, 5, 2]),
                keyless(),
                Step::Loads(vec![4, 5, 2]),
                keyless(),
            ],
        ),
        // A request with no key never goes to a worker that is down.
        sequence_row(
            "manual-min-load-a-request-with-no-key-passes-over-a-worker-that-is-down",
            "min_load",
            loaded(&[0, 3, 2]),
            vec![Step::Down(0), keyless(), Step::Up(0), keyless()],
        ),
        drawn_row(
            "manual-random-a-request-with-no-key-is-never-drawn-on-a-worker-that-is-down",
            "random",
            loaded(&[9, 0, 3]),
            vec![Step::Down(1), keyless()],
        ),
        // The pin's two workers: failover, recovery, the oldest dropped.
        sequence_row(
            "manual-a-key-moves-off-a-worker-that-is-down-and-the-first-owner-leads-again",
            "min_load",
            loaded(&[1, 2, 3]),
            vec![
                keyed("k1"),
                Step::Down(0),
                keyed("k1"),
                keyed("k1"),
                Step::Up(0),
                keyed("k1"),
            ],
        ),
        sequence_row(
            "manual-a-key-keeps-two-workers-and-drops-the-oldest-at-the-third",
            "min_load",
            loaded(&[1, 2, 3, 4]),
            vec![
                keyed("k1"),
                Step::Down(0),
                keyed("k1"),
                Step::Down(1),
                keyed("k1"),
                Step::Up(0),
                Step::Up(1),
                keyed("k1"),
                Step::Down(1),
                keyed("k1"),
            ],
        ),
        sequence_row(
            "manual-two-keys-pin-apart",
            "min_load",
            loaded(&[1, 2, 3]),
            vec![
                keyed("ka"),
                Step::Loads(vec![5, 2, 3]),
                keyed("kb"),
                Step::Loads(vec![9, 9, 0]),
                keyed("ka"),
                keyed("kb"),
            ],
        ),
        // One key under the two roles of a pair, and under none: three pins.
        sequence_row(
            "manual-one-key-pins-apart-under-the-prefill-and-the-decode-role",
            "min_load",
            loaded(&[1, 2, 3]),
            vec![
                under("k1", Role::Prefill),
                Step::Loads(vec![5, 2, 3]),
                under("k1", Role::Decode),
                Step::Loads(vec![9, 9, 0]),
                under("k1", Role::Prefill),
                under("k1", Role::Decode),
                keyed("k1"),
            ],
        ),
    ]
}

#[test]
#[ignore = "records the classic rows into CLASSIC_ROWS_OUT"]
fn record_the_classic_manual_rows() {
    let out = PathBuf::from(
        std::env::var_os("CLASSIC_ROWS_OUT").expect("CLASSIC_ROWS_OUT names the output directory"),
    );
    let commit =
        std::env::var("CLASSIC_ROWS_COMMIT").expect("CLASSIC_ROWS_COMMIT names the commit");
    let sheet = Sheet {
        recorded_from: commit,
        policy: "manual",
        rows: rows(),
    };
    fs::create_dir_all(&out).expect("the output directory");
    let path = out.join("manual.json");
    let text = serde_json::to_string_pretty(&sheet).expect("the sheet serializes");
    fs::write(&path, text + "\n").expect("the sheet is written");
    println!("{} rows -> {}", sheet.rows.len(), path.display());
}
