//! The recorder of the classic routing-key override's outcomes: each row is
//! a pool of workers, what the router holds of them (the requests in
//! flight, the routing keys with a request in flight, the requests of one
//! key), the policy under the override, the sticky map's assignment mode, a
//! list of steps and the worker the gateway picked at each request. The
//! policy controller crate replays the rows through its own decision
//! (`crates/policy_controller/tests/classic/sticky_override.rs`).
//!
//! This file is an ignored test of the gateway crate, kept beside the code
//! it runs. It records from the checkout it is built in and writes the sheet
//! under `CLASSIC_ROWS_OUT`, naming `CLASSIC_ROWS_COMMIT` as the commit it
//! recorded from. The sheets the policy controller crate replays are in
//! `crates/policy_controller/tests/classic/sheets`; to record this one again:
//!
//! ```text
//! CLASSIC_ROWS_OUT=<dir> CLASSIC_ROWS_COMMIT=<commit> \
//!   cargo test -p smg --lib policies::classic_sticky_override_rows -- --ignored --nocapture
//! ```
//!
//! A sequence row is recorded twenty times from a fresh registry and must
//! come out the same each time; a drawn row records how often each worker
//! took the row's last request over its draws, each draw from a fresh
//! registry.

use std::{fs, path::PathBuf, sync::Arc};

use openai_protocol::worker::WorkerStatus;
use serde::Serialize;
use smg_worker::identity::basic::Worker;

use super::{LoadBalancingPolicy, PolicyRegistry, SelectWorkerInfo};
use crate::{
    config::{ManualAssignmentMode, PolicyConfig, PowerOfTwoLoadMetric, RoutingKeyOverrideConfig},
    policies::test_support::make_worker,
};

/// One worker of a row: the requests the router holds on it and the routing
/// keys that have a request in flight on it.
#[derive(Clone, Copy, Serialize)]
struct WorkerState {
    in_flight: u32,
    keys: u32,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "snake_case")]
enum Step {
    /// One request: its routing key, if it carries one, and its model.
    Pick {
        key: Option<&'static str>,
        model: &'static str,
    },
    /// A worker stops being routable.
    Down(usize),
    /// The worker is routable again.
    Up(usize),
    /// The requests the router holds on each worker, from here on.
    Loads(Vec<u32>),
    /// The requests of one routing key in flight on each worker, from here
    /// on: the count the override's cap reads.
    Held { key: &'static str, on: Vec<u32> },
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
    /// The policy under the override, in its configuration spelling.
    policy: &'static str,
    /// The sticky map's assignment mode, in its configuration spelling.
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

/// A fresh registry with the override on, over a fresh pool in the row's
/// starting state.
struct Bed {
    registry: PolicyRegistry,
    policy: Arc<dyn LoadBalancingPolicy>,
    workers: Vec<Arc<dyn Worker>>,
}

impl Bed {
    fn new(policy: &str, assignment: &str, workers: &[WorkerState]) -> Self {
        let config = match policy {
            "random" => PolicyConfig::Random,
            "power_of_two" => PolicyConfig::PowerOfTwo {
                load_metric: PowerOfTwoLoadMetric::default(),
            },
            other => panic!("no policy {other}"),
        };
        let assignment_mode = match assignment {
            "random" => ManualAssignmentMode::Random,
            "min_load" => ManualAssignmentMode::MinLoad,
            "min_group" => ManualAssignmentMode::MinGroup,
            "delegate" => ManualAssignmentMode::Delegate,
            other => panic!("no assignment mode {other}"),
        };
        // No eviction task: a row runs far inside any idle time.
        let registry = PolicyRegistry::with_override(
            config,
            RoutingKeyOverrideConfig {
                enabled: true,
                eviction_interval_secs: 0,
                max_idle_secs: 0,
                assignment_mode,
                ..RoutingKeyOverrideConfig::default()
            },
        );
        let policy = registry.get_default_policy();
        let pool = workers
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
            registry,
            policy,
            workers: pool,
        }
    }

    /// One step; a request answers the worker it was placed on.
    fn step(&self, step: &Step) -> Option<usize> {
        match step {
            Step::Pick { key, model } => {
                let info = SelectWorkerInfo {
                    routing_key: *key,
                    ..SelectWorkerInfo::default()
                };
                Some(
                    self.registry
                        .select_worker_for_model(&self.policy, model, &self.workers, &info)
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
            Step::Held { key, on } => {
                for (worker, &want) in self.workers.iter().zip(on) {
                    while worker.routing_key_inflight(key) < want as usize {
                        worker.increment_routing_key_load(key);
                    }
                    while worker.routing_key_inflight(key) > want as usize {
                        worker.decrement_routing_key_load(key);
                    }
                }
                None
            }
        }
    }

    fn run(policy: &str, assignment: &str, workers: &[WorkerState], steps: &[Step]) -> Vec<usize> {
        let bed = Self::new(policy, assignment, workers);
        steps.iter().filter_map(|step| bed.step(step)).collect()
    }
}

fn sequence_row(
    name: &'static str,
    policy: &'static str,
    assignment: &'static str,
    workers: Vec<WorkerState>,
    steps: Vec<Step>,
) -> Row {
    let first = Bed::run(policy, assignment, &workers, &steps);
    for _ in 0..19 {
        assert_eq!(
            Bed::run(policy, assignment, &workers, &steps),
            first,
            "{name}: a sequence row must not depend on a draw"
        );
    }
    Row {
        name,
        policy,
        assignment,
        workers,
        steps,
        outcome: Outcome::Sequence(first),
    }
}

fn drawn_row(
    name: &'static str,
    policy: &'static str,
    assignment: &'static str,
    workers: Vec<WorkerState>,
    steps: Vec<Step>,
) -> Row {
    const DRAWS: u32 = 6_000;
    let mut counts = vec![0u32; workers.len()];
    for _ in 0..DRAWS {
        let picked = Bed::run(policy, assignment, &workers, &steps);
        counts[*picked.last().expect("a drawn row ends with a request")] += 1;
    }
    Row {
        name,
        policy,
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

/// A request of `key` for the model every row but one asks for.
fn keyed(key: &'static str) -> Step {
    under(key, "m")
}

fn keyless() -> Step {
    Step::Pick {
        key: None,
        model: "m",
    }
}

fn under(key: &'static str, model: &'static str) -> Step {
    Step::Pick {
        key: Some(key),
        model,
    }
}

fn held(key: &'static str, on: &[u32]) -> Step {
    Step::Held {
        key,
        on: on.to_vec(),
    }
}

fn rows() -> Vec<Row> {
    vec![
        // The pin: a key keeps its worker under the cap, under its model.
        sequence_row(
            "sticky-min-load-a-key-keeps-its-worker-under-the-cap-past-the-load",
            "random",
            "min_load",
            loaded(&[3, 1, 2]),
            vec![
                keyed("k1"),
                held("k1", &[0, 1, 0]),
                Step::Loads(vec![0, 5, 2]),
                keyed("k1"),
                keyed("k2"),
                keyed("k1"),
            ],
        ),
        sequence_row(
            "sticky-one-key-pins-apart-under-two-models",
            "random",
            "min_load",
            loaded(&[3, 1, 2]),
            vec![
                under("k1", "m1"),
                Step::Loads(vec![0, 5, 2]),
                under("k1", "m2"),
                Step::Loads(vec![9, 9, 0]),
                under("k1", "m1"),
                under("k1", "m2"),
            ],
        ),
        // A request with no key is the policy's, whatever the map's mode.
        drawn_row(
            "sticky-a-request-with-no-key-is-the-policy-s-draw-and-pins-nothing",
            "random",
            "min_load",
            loaded(&[9, 0, 3]),
            vec![keyless(), keyless()],
        ),
        // The map's own modes place a key seen first.
        sequence_row(
            "sticky-min-group-a-new-key-takes-the-fewest-keys-whatever-the-load",
            "random",
            "min_group",
            vec![
                WorkerState {
                    in_flight: 0,
                    keys: 2,
                },
                WorkerState {
                    in_flight: 5,
                    keys: 0,
                },
                WorkerState {
                    in_flight: 0,
                    keys: 1,
                },
            ],
            vec![
                keyed("k1"),
                held("k1", &[0, 1, 0]),
                held("x1", &[0, 1, 0]),
                held("x2", &[0, 1, 0]),
                keyed("k2"),
                keyed("k1"),
            ],
        ),
        drawn_row(
            "sticky-random-a-new-key-is-drawn-over-every-worker",
            "random",
            "random",
            loaded(&[9, 0, 3]),
            vec![keyed("k1")],
        ),
        // The pin's two workers under the map's own modes: the first owner
        // leads again.
        sequence_row(
            "sticky-min-load-a-key-moves-off-a-worker-that-is-down-and-the-first-owner-leads-again",
            "random",
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
        // `delegate`, the default: the policy places a key seen first, and
        // the latest worker leads.
        sequence_row(
            "sticky-delegate-a-new-key-is-the-policy-s-pick-and-stays-past-the-load",
            "power_of_two",
            "delegate",
            loaded(&[5, 0]),
            vec![
                keyed("k1"),
                Step::Loads(vec![0, 5]),
                keyed("k1"),
                keyless(),
                keyed("k2"),
                keyed("k1"),
            ],
        ),
        sequence_row(
            "sticky-delegate-the-latest-worker-leads-after-a-failover",
            "power_of_two",
            "delegate",
            loaded(&[0, 5]),
            vec![
                keyed("k1"),
                Step::Down(0),
                keyed("k1"),
                Step::Up(0),
                keyed("k1"),
                Step::Down(1),
                keyed("k1"),
                Step::Up(1),
                keyed("k1"),
            ],
        ),
        drawn_row(
            "sticky-delegate-a-new-key-is-the-policy-s-draw",
            "random",
            "delegate",
            loaded(&[9, 0, 3]),
            vec![keyed("k1")],
        ),
        // The cap: with two requests of the key on its worker the request is
        // placed again, and the pin moves only to another worker under the
        // cap.
        sequence_row(
            "sticky-the-cap-counts-the-request-s-own-key-alone",
            "random",
            "min_load",
            loaded(&[3, 1, 2]),
            vec![
                keyed("ka"),
                held("kb", &[0, 2, 0]),
                Step::Loads(vec![0, 5, 2]),
                keyed("ka"),
                held("ka", &[0, 2, 0]),
                keyed("ka"),
            ],
        ),
        sequence_row(
            "sticky-min-load-at-the-cap-the-request-goes-to-the-least-loaded-and-the-pin-follows",
            "random",
            "min_load",
            loaded(&[3, 1, 2]),
            vec![
                keyed("k1"),
                held("k1", &[0, 2, 0]),
                Step::Loads(vec![0, 5, 2]),
                keyed("k1"),
                held("k1", &[1, 2, 0]),
                Step::Loads(vec![5, 0, 2]),
                keyed("k1"),
            ],
        ),
        sequence_row(
            "sticky-min-load-at-the-cap-the-least-loaded-is-the-pinned-worker-again",
            "random",
            "min_load",
            loaded(&[3, 1, 2]),
            vec![
                keyed("k1"),
                held("k1", &[0, 2, 0]),
                keyed("k1"),
                held("k1", &[0, 3, 0]),
                keyed("k1"),
            ],
        ),
        sequence_row(
            "sticky-min-load-at-the-cap-the-pin-does-not-move-to-a-worker-at-the-cap",
            "random",
            "min_load",
            loaded(&[3, 1, 2]),
            vec![
                keyed("k1"),
                held("k1", &[2, 2, 0]),
                Step::Loads(vec![0, 5, 2]),
                keyed("k1"),
                held("k1", &[2, 1, 0]),
                keyed("k1"),
            ],
        ),
        sequence_row(
            "sticky-min-load-every-worker-at-the-cap",
            "random",
            "min_load",
            loaded(&[3, 1, 2]),
            vec![
                keyed("k1"),
                held("k1", &[2, 2, 2]),
                Step::Loads(vec![0, 5, 2]),
                keyed("k1"),
                held("k1", &[2, 1, 2]),
                keyed("k1"),
            ],
        ),
        sequence_row(
            "sticky-min-load-at-the-cap-with-no-other-worker-routable-the-key-stays",
            "random",
            "min_load",
            loaded(&[1, 2]),
            vec![
                keyed("k1"),
                held("k1", &[2, 0]),
                Step::Down(1),
                keyed("k1"),
                Step::Up(1),
                Step::Loads(vec![5, 0]),
                keyed("k1"),
            ],
        ),
        sequence_row(
            "sticky-delegate-at-the-cap-the-policy-places-the-request-and-the-pin-follows",
            "power_of_two",
            "delegate",
            loaded(&[0, 5]),
            vec![
                keyed("k1"),
                held("k1", &[2, 0]),
                Step::Loads(vec![5, 0]),
                keyed("k1"),
                held("k1", &[2, 1]),
                Step::Loads(vec![0, 5]),
                keyed("k1"),
            ],
        ),
        sequence_row(
            "sticky-delegate-at-the-cap-the-policy-picks-the-pinned-worker-again",
            "power_of_two",
            "delegate",
            loaded(&[0, 5]),
            vec![keyed("k1"), held("k1", &[2, 0]), keyed("k1")],
        ),
        drawn_row(
            "sticky-delegate-at-the-cap-the-policy-draws-over-every-worker",
            "random",
            "delegate",
            loaded(&[0, 0, 0]),
            vec![
                Step::Down(1),
                Step::Down(2),
                keyed("k1"),
                Step::Up(1),
                Step::Up(2),
                held("k1", &[2, 0, 0]),
                keyed("k1"),
            ],
        ),
    ]
}

#[test]
#[ignore = "records the classic rows into CLASSIC_ROWS_OUT"]
fn record_the_classic_sticky_override_rows() {
    let out = PathBuf::from(
        std::env::var_os("CLASSIC_ROWS_OUT").expect("CLASSIC_ROWS_OUT names the output directory"),
    );
    let commit =
        std::env::var("CLASSIC_ROWS_COMMIT").expect("CLASSIC_ROWS_COMMIT names the commit");
    let sheet = Sheet {
        recorded_from: commit,
        policy: "sticky_override",
        rows: rows(),
    };
    fs::create_dir_all(&out).expect("the output directory");
    let path = out.join("sticky_override.json");
    let text = serde_json::to_string_pretty(&sheet).expect("the sheet serializes");
    fs::write(&path, text + "\n").expect("the sheet is written");
    println!("{} rows -> {}", sheet.rows.len(), path.display());
}
