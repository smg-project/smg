//! The recorder of the gateway's answer to a pool whose every usable worker
//! is over its overload thresholds: each row is a pool of workers, what the
//! router holds of each one (the requests in flight, and whether it is
//! ready, over its thresholds, doubted by the liveness tracker or behind an
//! open circuit breaker), the
//! policy the pool runs under, whether shedding is on, a list of steps and
//! what the gateway's single-worker placement answered at each request. The
//! policy controller crate replays the rows through its own decision
//! (`crates/policy_controller/tests/classic_rows.rs`).
//!
//! A request's answer is the worker it was placed on, by its place in the
//! row's list. A request that is placed nowhere is recorded as a number past
//! the last worker: the count of the workers when the gateway sheds it as an
//! overload shed, the count plus one when it finds no worker to use.
//!
//! This file is an ignored test of the gateway crate, kept beside the code
//! it runs. It records from the checkout it is built in and writes the sheet
//! under `CLASSIC_ROWS_OUT`, naming `CLASSIC_ROWS_COMMIT` as the commit it
//! recorded from. The sheets the policy controller crate replays are in
//! `crates/policy_controller/tests/classic/sheets`; to record this one again:
//!
//! ```text
//! CLASSIC_ROWS_OUT=<dir> CLASSIC_ROWS_COMMIT=<commit> \
//!   cargo test -p smg --lib policies::classic_all_vetoed_rows -- --ignored --nocapture
//! ```
//!
//! A sequence row is recorded twenty times from a fresh pool and a fresh
//! policy and must come out the same each time; a drawn row records how
//! often each worker took the row's last request over its draws, each draw
//! from a fresh pool and a fresh policy.
//!
//! A ring row (`ring_key`) records a request placed by its routing key on
//! the hash ring. Two rings put one key on different workers, so such a row
//! lists its workers in the key's ring order: the worker the key lands on
//! first, then the one it passes to when that worker is down, and so on.
//! The recorder finds that order on the gateway's ring by placing the key
//! with every worker free and then with the members found so far down. A
//! ring row's loads, states, tried lists and answers all count in that
//! order, so an answer of 0 is the key's own worker and 1 the next member
//! of the ring, on whatever ring replays the row.

use std::{fs, path::PathBuf, sync::Arc};

use http::HeaderMap;
use openai_protocol::worker::WorkerStatus;
use serde::Serialize;
use smg_worker::{identity::basic::Worker, runtime::StallReason, test_support::ready_worker};

use super::PolicyRegistry;
use crate::{
    config::types::PolicyConfig,
    routers::common::{
        placement::{self, PlacementFailure, PlacementInputs},
        snapshot::{RoutingPool, WorkerRegistry},
    },
};

const MODEL: &str = "m";

/// What the router holds of a worker besides its requests.
#[derive(Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum State {
    /// Ready and under its overload thresholds.
    Free,
    /// Ready and over its overload thresholds.
    Over,
    /// Ready, under its thresholds, doubted by the liveness tracker.
    Stalled,
    /// Ready, over its thresholds and doubted by the liveness tracker.
    StalledOver,
    /// Ready and under its thresholds, behind an open circuit breaker.
    Breaker,
    /// Not ready.
    Down,
}

/// One worker of a row: the requests the router holds on it and its state.
#[derive(Clone, Copy, Serialize)]
struct WorkerState {
    in_flight: u32,
    state: State,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "snake_case")]
enum Step {
    /// One request: its routing key, if it carries one, and the workers an
    /// earlier attempt of it already tried.
    Pick {
        key: Option<&'static str>,
        tried: Vec<usize>,
    },
    /// The requests the router holds on each worker, from here on.
    Loads(Vec<u32>),
    /// Each worker's state, from here on.
    States(Vec<State>),
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum Outcome {
    /// The answer to each `Pick` step, in order.
    Sequence(Vec<usize>),
    /// How often each worker took the row's last request over `draws` runs
    /// of the row.
    Counts { draws: u32, counts: Vec<u32> },
}

#[derive(Serialize)]
struct Row {
    name: &'static str,
    /// The policy, in its configuration spelling.
    policy: &'static str,
    /// Whether an all-overloaded pool is shed (`--worker-overload-shed`).
    shed: bool,
    /// A ring row's routing key: the row's workers are listed in this key's
    /// ring order.
    #[serde(skip_serializing_if = "Option::is_none")]
    ring_key: Option<&'static str>,
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

/// The policy's configuration, as the gateway's own configuration file
/// spells it. The manual policy places a new key on the fewest requests in
/// flight, so that a row does not start with a draw.
fn config_of(policy: &str) -> PolicyConfig {
    let text = match policy {
        "manual" => r#"{"type":"manual","assignment_mode":"min_load"}"#.to_string(),
        "bucket" => concat!(
            r#"{"type":"bucket","balance_abs_threshold":32,"#,
            r#""balance_rel_threshold":1.1,"bucket_adjust_interval_secs":5}"#
        )
        .to_string(),
        "cache_aware" => return PolicyConfig::cache_aware(0.5, 32, 1.1, 0, 4096),
        other => format!(r#"{{"type":"{other}"}}"#),
    };
    serde_json::from_str(&text).expect("the policy's configuration")
}

/// A fresh registry of ready workers under a fresh policy, in the row's
/// starting state.
struct Bed {
    registry: WorkerRegistry,
    policies: PolicyRegistry,
    workers: Vec<Arc<dyn Worker>>,
    /// Where each of the row's workers stands in the registered pool: the
    /// row's own order, or the key's ring order for a ring row.
    places: Vec<usize>,
}

impl Bed {
    fn new(policy: &str, shed: bool, workers: &[WorkerState], places: &[usize]) -> Self {
        assert_eq!(places.len(), workers.len(), "a place of every worker");
        let registry = WorkerRegistry::new();
        let pool: Vec<Arc<dyn Worker>> = (0..workers.len())
            .map(|i| {
                let worker = ready_worker(&url(i), MODEL);
                registry
                    .register(Arc::clone(&worker))
                    .expect("the worker registers");
                worker
            })
            .collect();
        registry.set_overload_shed(shed);
        let bed = Self {
            registry,
            policies: PolicyRegistry::new(config_of(policy)),
            workers: pool,
            places: places.to_vec(),
        };
        bed.loads(&workers.iter().map(|w| w.in_flight).collect::<Vec<_>>());
        bed.states(&workers.iter().map(|w| w.state).collect::<Vec<_>>());
        bed
    }

    fn loads(&self, loads: &[u32]) {
        assert_eq!(loads.len(), self.workers.len(), "a load of every worker");
        for (&place, &want) in self.places.iter().zip(loads) {
            let worker = &self.workers[place];
            while worker.load() < want as usize {
                worker.increment_load();
            }
            while worker.load() > want as usize {
                worker.decrement_load();
            }
        }
    }

    fn states(&self, states: &[State]) {
        assert_eq!(states.len(), self.workers.len(), "a state of every worker");
        for (&place, &state) in self.places.iter().zip(states) {
            let worker = &self.workers[place];
            worker.set_status(if state == State::Down {
                WorkerStatus::NotReady
            } else {
                WorkerStatus::Ready
            });
            self.registry
                .set_worker_overloaded(worker, matches!(state, State::Over | State::StalledOver));
            worker.set_stall(
                matches!(state, State::Stalled | State::StalledOver).then_some(StallReason::Wedged),
            );
            // An open breaker refuses execution: failures are recorded
            // until it does. Every other state starts from a closed one.
            worker.reset_circuit_breaker();
            if state == State::Breaker {
                for _ in 0..1_000 {
                    if !worker.circuit_breaker_can_execute() {
                        break;
                    }
                    worker.record_circuit_breaker_outcome(false);
                }
                assert!(!worker.circuit_breaker_can_execute(), "the breaker opens");
            }
        }
    }

    /// One request through the gateway's single-worker placement: the
    /// worker's place, or a number past the last worker for a request that
    /// is placed nowhere.
    fn pick(&self, key: Option<&str>, tried: &[usize]) -> usize {
        let mut headers = HeaderMap::new();
        if let Some(key) = key {
            headers.insert("x-smg-routing-key", key.parse().expect("a header value"));
        }
        let tried: Vec<String> = tried.iter().map(|&i| url(self.places[i])).collect();
        let inputs = PlacementInputs {
            headers: Some(&headers),
            tried: &tried,
            ..Default::default()
        };
        let picked = placement::select_single(
            &self.registry,
            &self.policies,
            MODEL,
            RoutingPool::HttpRegular,
            None,
            inputs,
            placement::Stated::nothing,
        );
        match picked {
            Some(worker) => {
                let place = self
                    .workers
                    .iter()
                    .position(|w| w.url() == worker.url())
                    .expect("a worker of the pool");
                self.places
                    .iter()
                    .position(|&p| p == place)
                    .expect("a worker of the row")
            }
            None => {
                let failure = placement::single_failure(
                    &self.registry,
                    MODEL,
                    RoutingPool::HttpRegular,
                    None,
                );
                match failure {
                    PlacementFailure::AllOverloaded(_) => self.workers.len(),
                    _ => self.workers.len() + 1,
                }
            }
        }
    }

    /// One step; a request answers where it was placed.
    fn step(&self, step: &Step) -> Option<usize> {
        match step {
            Step::Pick { key, tried } => return Some(self.pick(*key, tried)),
            Step::Loads(loads) => self.loads(loads),
            Step::States(states) => self.states(states),
        }
        None
    }

    fn run(
        policy: &str,
        shed: bool,
        workers: &[WorkerState],
        steps: &[Step],
        places: &[usize],
    ) -> Vec<usize> {
        let bed = Self::new(policy, shed, workers, places);
        steps.iter().filter_map(|step| bed.step(step)).collect()
    }
}

/// The row's own order: worker `i` of the row is worker `i` of the pool.
fn in_order(workers: usize) -> Vec<usize> {
    (0..workers).collect()
}

/// The ring order of `key` over a pool of `workers` under `policy`: the
/// worker the key lands on with every worker free and idle, then with the
/// members found so far down, until every worker is placed.
fn ring_order(policy: &str, workers: usize, key: &str) -> Vec<usize> {
    let mut order: Vec<usize> = Vec::with_capacity(workers);
    while order.len() < workers {
        let pool: Vec<WorkerState> = (0..workers)
            .map(|place| WorkerState {
                in_flight: 0,
                state: if order.contains(&place) {
                    State::Down
                } else {
                    State::Free
                },
            })
            .collect();
        let next = Bed::new(policy, false, &pool, &in_order(workers)).pick(Some(key), &[]);
        assert!(
            next < workers && !order.contains(&next),
            "{policy}: the key's next member is a worker not yet placed, got {next}"
        );
        order.push(next);
    }
    order
}

fn sequence_row(
    name: &'static str,
    policy: &'static str,
    shed: bool,
    workers: Vec<WorkerState>,
    steps: Vec<Step>,
) -> Row {
    let places = in_order(workers.len());
    let first = Bed::run(policy, shed, &workers, &steps, &places);
    for _ in 0..19 {
        assert_eq!(
            Bed::run(policy, shed, &workers, &steps, &places),
            first,
            "{name}: a sequence row must not depend on a draw"
        );
    }
    Row {
        name,
        policy,
        shed,
        ring_key: None,
        workers,
        steps,
        outcome: Outcome::Sequence(first),
    }
}

/// A sequence row whose workers are listed in the ring order of `key`: every
/// request of the row carries that key.
fn ring_row(
    name: &'static str,
    policy: &'static str,
    shed: bool,
    key: &'static str,
    workers: Vec<WorkerState>,
    steps: Vec<Step>,
) -> Row {
    let places = ring_order(policy, workers.len(), key);
    assert_eq!(
        ring_order(policy, workers.len(), key),
        places,
        "{name}: the ring's order must not depend on a draw"
    );
    let first = Bed::run(policy, shed, &workers, &steps, &places);
    for _ in 0..19 {
        assert_eq!(
            Bed::run(policy, shed, &workers, &steps, &places),
            first,
            "{name}: a ring row must not depend on a draw"
        );
    }
    Row {
        name,
        policy,
        shed,
        ring_key: Some(key),
        workers,
        steps,
        outcome: Outcome::Sequence(first),
    }
}

fn drawn_row(
    name: &'static str,
    policy: &'static str,
    shed: bool,
    workers: Vec<WorkerState>,
    steps: Vec<Step>,
) -> Row {
    const DRAWS: u32 = 6_000;
    let mut counts = vec![0u32; workers.len() + 2];
    let places = in_order(workers.len());
    for _ in 0..DRAWS {
        let picked = Bed::run(policy, shed, &workers, &steps, &places);
        counts[*picked.last().expect("a drawn row ends with a request")] += 1;
    }
    assert_eq!(
        counts.split_off(workers.len()),
        [0, 0],
        "{name}: a drawn row places every request"
    );
    Row {
        name,
        policy,
        shed,
        ring_key: None,
        workers,
        steps,
        outcome: Outcome::Counts {
            draws: DRAWS,
            counts,
        },
    }
}

/// Workers in one state, by their requests in flight.
fn all(state: State, in_flight: &[u32]) -> Vec<WorkerState> {
    in_flight
        .iter()
        .map(|&in_flight| WorkerState { in_flight, state })
        .collect()
}

/// Workers by their state and their requests in flight.
fn mixed(workers: &[(State, u32)]) -> Vec<WorkerState> {
    workers
        .iter()
        .map(|&(state, in_flight)| WorkerState { in_flight, state })
        .collect()
}

fn pick() -> Step {
    Step::Pick {
        key: None,
        tried: vec![],
    }
}

fn keyed(key: &'static str) -> Step {
    Step::Pick {
        key: Some(key),
        tried: vec![],
    }
}

fn retry(tried: &[usize]) -> Step {
    Step::Pick {
        key: None,
        tried: tried.to_vec(),
    }
}

fn keyed_retry(key: &'static str, tried: &[usize]) -> Step {
    Step::Pick {
        key: Some(key),
        tried: tried.to_vec(),
    }
}

/// The policies that answer a pool with no available worker themselves, and
/// the ones the placement filters for.
const POLICIES: [&str; 10] = [
    "random",
    "round_robin",
    "passthrough",
    "power_of_two",
    "least_load",
    "cache_aware",
    "manual",
    "consistent_hashing",
    "prefix_hash",
    "bucket",
];

/// `name` for the policy at `at`: the rows of one cell are named alike.
fn named(names: &[&'static str; 10], at: usize) -> &'static str {
    names[at]
}

const STEERED: [&str; 10] = [
    "random-a-pool-over-on-every-worker-goes-to-the-fewest-in-flight",
    "round-robin-a-pool-over-on-every-worker-goes-to-the-fewest-in-flight",
    "passthrough-a-pool-over-on-every-worker-goes-to-the-fewest-in-flight",
    "power-of-two-a-pool-over-on-every-worker-goes-to-the-fewest-in-flight",
    "least-load-a-pool-over-on-every-worker-goes-to-the-fewest-in-flight",
    "cache-aware-a-pool-over-on-every-worker-goes-to-the-fewest-in-flight",
    "manual-a-pool-over-on-every-worker-goes-to-the-fewest-in-flight",
    "consistent-hashing-a-pool-over-on-every-worker-goes-to-the-fewest-in-flight",
    "prefix-hash-a-pool-over-on-every-worker-goes-to-the-fewest-in-flight",
    "bucket-a-pool-over-on-every-worker-goes-to-the-fewest-in-flight",
];

const SHED: [&str; 10] = [
    "random-a-pool-over-on-every-worker-is-shed-under-shedding",
    "round-robin-a-pool-over-on-every-worker-is-shed-under-shedding",
    "passthrough-a-pool-over-on-every-worker-is-shed-under-shedding",
    "power-of-two-a-pool-over-on-every-worker-is-shed-under-shedding",
    "least-load-a-pool-over-on-every-worker-is-shed-under-shedding",
    "cache-aware-a-pool-over-on-every-worker-is-shed-under-shedding",
    "manual-a-pool-over-on-every-worker-is-shed-under-shedding",
    "consistent-hashing-a-pool-over-on-every-worker-is-shed-under-shedding",
    "prefix-hash-a-pool-over-on-every-worker-is-shed-under-shedding",
    "bucket-a-pool-over-on-every-worker-is-shed-under-shedding",
];

fn rows() -> Vec<Row> {
    let mut rows = Vec::new();
    // Under every policy: the fewest in flight takes the request, and the
    // answer follows the load from request to request.
    for (at, policy) in POLICIES.into_iter().enumerate() {
        rows.push(sequence_row(
            named(&STEERED, at),
            policy,
            false,
            all(State::Over, &[3, 1, 2]),
            vec![
                pick(),
                Step::Loads(vec![0, 5, 2]),
                pick(),
                Step::Loads(vec![4, 5, 2]),
                pick(),
            ],
        ));
    }
    // Under every policy: with shedding on the same pool is shed.
    for (at, policy) in POLICIES.into_iter().enumerate() {
        rows.push(sequence_row(
            named(&SHED, at),
            policy,
            true,
            all(State::Over, &[3, 1, 2]),
            vec![pick()],
        ));
    }
    // Equal loads are drawn, over the tied workers alone.
    for (name, policy) in [
        (
            "round-robin-equal-loads-over-on-every-worker-are-drawn",
            "round_robin",
        ),
        (
            "least-load-equal-loads-over-on-every-worker-are-drawn",
            "least_load",
        ),
        (
            "manual-equal-loads-over-on-every-worker-are-drawn",
            "manual",
        ),
        (
            "consistent-hashing-equal-loads-over-on-every-worker-are-drawn",
            "consistent_hashing",
        ),
    ] {
        rows.push(drawn_row(
            name,
            policy,
            false,
            all(State::Over, &[2, 2, 5]),
            vec![pick()],
        ));
    }
    // One worker under its thresholds: the policy places on it, whatever it
    // holds, and the rule above does not run.
    rows.push(sequence_row(
        "round-robin-one-worker-under-its-thresholds-takes-every-request",
        "round_robin",
        false,
        mixed(&[(State::Over, 0), (State::Free, 9), (State::Over, 0)]),
        vec![pick(), pick(), pick()],
    ));
    rows.push(sequence_row(
        "round-robin-one-worker-under-its-thresholds-takes-every-request-under-shedding",
        "round_robin",
        true,
        mixed(&[(State::Over, 0), (State::Free, 9), (State::Over, 0)]),
        vec![pick(), pick(), pick()],
    ));
    // A worker that is not ready is never the answer, however few requests
    // it holds; neither is one the liveness tracker doubts while another
    // worker is over its thresholds alone.
    rows.push(sequence_row(
        "round-robin-a-worker-that-is-down-is-never-the-fewest-in-flight",
        "round_robin",
        false,
        mixed(&[(State::Down, 0), (State::Over, 3), (State::Over, 1)]),
        vec![pick(), Step::Loads(vec![0, 1, 3]), pick()],
    ));
    rows.push(sequence_row(
        "round-robin-a-doubted-worker-is-passed-over-for-one-over-its-thresholds",
        "round_robin",
        false,
        mixed(&[(State::Stalled, 0), (State::Over, 3), (State::Over, 1)]),
        vec![pick(), Step::Loads(vec![0, 1, 3]), pick()],
    ));
    rows.push(sequence_row(
        "round-robin-a-pool-with-nothing-ready-finds-no-worker",
        "round_robin",
        false,
        all(State::Down, &[0, 0]),
        vec![pick()],
    ));
    // A retry: the worker an earlier attempt tried is taken again when it
    // holds the fewest requests.
    rows.push(sequence_row(
        "round-robin-a-tried-worker-is-taken-again-when-it-holds-the-fewest",
        "round_robin",
        false,
        all(State::Over, &[3, 1, 2]),
        vec![retry(&[1]), retry(&[1, 2])],
    ));
    rows.push(sequence_row(
        "least-load-a-tried-worker-is-taken-again-when-it-holds-the-fewest",
        "least_load",
        false,
        all(State::Over, &[3, 1, 2]),
        vec![retry(&[1])],
    ));
    // A key under the manual policy: its worker goes over with every other,
    // the request goes to the fewest in flight, and the key is back on its
    // worker when the pool is under its thresholds again.
    rows.push(sequence_row(
        "manual-a-key-goes-to-the-fewest-in-flight-and-returns-to-its-worker",
        "manual",
        false,
        all(State::Free, &[1, 2, 3]),
        vec![
            keyed("k1"),
            Step::States(vec![State::Over, State::Over, State::Over]),
            Step::Loads(vec![9, 2, 3]),
            keyed("k1"),
            keyed("k1"),
            Step::States(vec![State::Free, State::Free, State::Free]),
            keyed("k1"),
        ],
    ));
    rows.push(sequence_row(
        "manual-a-key-is-shed-with-every-worker-over-under-shedding",
        "manual",
        true,
        all(State::Free, &[1, 2, 3]),
        vec![
            keyed("k1"),
            Step::States(vec![State::Over, State::Over, State::Over]),
            keyed("k1"),
            Step::States(vec![State::Free, State::Free, State::Free]),
            keyed("k1"),
        ],
    ));
    // A routing key under consistent hashing, the workers in the key's ring
    // order: its own worker over its thresholds beside free workers, and
    // with every worker over; the same two pools under shedding; its worker
    // down; a retry that tried its worker; its worker and the next member
    // both over; its worker down and the next member over.
    let ring = "consistent_hashing";
    let beside_free = [
        (State::Over, 5),
        (State::Free, 4),
        (State::Free, 0),
        (State::Free, 2),
    ];
    rows.push(ring_row(
        "consistent-hashing-a-key-keeps-its-worker-over-its-thresholds-beside-free-workers",
        ring,
        false,
        "k1",
        mixed(&beside_free),
        vec![keyed("k1")],
    ));
    rows.push(ring_row(
        "consistent-hashing-a-key-keeps-its-worker-with-every-worker-over",
        ring,
        false,
        "k1",
        all(State::Over, &[5, 4, 0, 2]),
        vec![keyed("k1")],
    ));
    rows.push(ring_row(
        "consistent-hashing-a-key-leaves-its-worker-over-its-thresholds-under-shedding",
        ring,
        true,
        "k1",
        mixed(&beside_free),
        vec![keyed("k1")],
    ));
    rows.push(ring_row(
        "consistent-hashing-a-key-is-shed-with-every-worker-over-under-shedding",
        ring,
        true,
        "k1",
        all(State::Over, &[5, 4, 0, 2]),
        vec![keyed("k1")],
    ));
    rows.push(ring_row(
        "consistent-hashing-a-key-passes-to-the-next-member-when-its-worker-is-down",
        ring,
        false,
        "k1",
        mixed(&[
            (State::Down, 0),
            (State::Free, 4),
            (State::Free, 0),
            (State::Free, 2),
        ]),
        vec![keyed("k1")],
    ));
    rows.push(ring_row(
        "consistent-hashing-a-retry-that-tried-the-keys-worker-passes-the-key-on",
        ring,
        false,
        "k1",
        mixed(&beside_free),
        vec![keyed_retry("k1", &[0])],
    ));
    rows.push(ring_row(
        "consistent-hashing-a-key-keeps-its-worker-when-the-next-member-is-over-too",
        ring,
        false,
        "k1",
        mixed(&[
            (State::Over, 5),
            (State::Over, 4),
            (State::Free, 0),
            (State::Free, 2),
        ]),
        vec![keyed("k1")],
    ));
    rows.push(ring_row(
        "consistent-hashing-a-key-passes-to-a-next-member-over-its-thresholds-when-its-worker-is-down",
        ring,
        false,
        "k1",
        mixed(&[
            (State::Down, 0),
            (State::Over, 4),
            (State::Free, 0),
            (State::Free, 2),
        ]),
        vec![keyed("k1")],
    ));
    // The lift is the ring's alone: under round robin a keyed request leaves
    // a worker over its thresholds for the free one.
    rows.push(sequence_row(
        "round-robin-a-keyed-request-leaves-a-worker-over-its-thresholds",
        "round_robin",
        false,
        mixed(&[(State::Over, 0), (State::Free, 3)]),
        vec![keyed("k1"), keyed("k1")],
    ));
    liveness_rows(&mut rows);
    breaker_rows(&mut rows);
    rows
}

#[test]
#[ignore = "records the classic rows into CLASSIC_ROWS_OUT"]
fn record_the_classic_all_vetoed_rows() {
    let out = PathBuf::from(
        std::env::var_os("CLASSIC_ROWS_OUT").expect("CLASSIC_ROWS_OUT names the output directory"),
    );
    let commit =
        std::env::var("CLASSIC_ROWS_COMMIT").expect("CLASSIC_ROWS_COMMIT names the commit");
    let sheet = Sheet {
        recorded_from: commit,
        policy: "all_vetoed",
        rows: rows(),
    };
    fs::create_dir_all(&out).expect("the output directory");
    let path = out.join("all_vetoed.json");
    let text = serde_json::to_string_pretty(&sheet).expect("the sheet serializes");
    fs::write(&path, text + "\n").expect("the sheet is written");
    println!("{} rows -> {}", sheet.rows.len(), path.display());
}

/// A row's name under `policy`: the policy's name, then what the row shows.
fn row_name(policy: &str, shows: &str) -> &'static str {
    Box::leak(format!("{}-{shows}", policy.replace('_', "-")).into_boxed_str())
}

/// The rows of the liveness veto: a pool whose ready workers are each
/// doubted by the liveness tracker, over their thresholds, or both.
fn liveness_rows(rows: &mut Vec<Row>) {
    use State::Stalled;
    // Under every policy: a pool doubted on every worker goes to the fewest
    // in flight, with shedding off and on.
    for policy in POLICIES {
        rows.push(sequence_row(
            row_name(
                policy,
                "a-pool-doubted-on-every-worker-goes-to-the-fewest-in-flight",
            ),
            policy,
            false,
            all(Stalled, &[3, 1, 2]),
            vec![
                pick(),
                Step::Loads(vec![0, 5, 2]),
                pick(),
                Step::Loads(vec![4, 5, 2]),
                pick(),
            ],
        ));
        rows.push(sequence_row(
            row_name(
                policy,
                "a-pool-doubted-on-every-worker-is-placed-under-shedding-too",
            ),
            policy,
            true,
            all(Stalled, &[3, 1, 2]),
            vec![pick()],
        ));
    }
    // Equal loads are drawn, over the tied workers alone.
    for policy in ["round_robin", "consistent_hashing"] {
        rows.push(drawn_row(
            row_name(policy, "equal-loads-doubted-on-every-worker-are-drawn"),
            policy,
            false,
            all(Stalled, &[2, 2, 5]),
            vec![pick()],
        ));
    }
    liveness_cells(rows);
    liveness_keys(rows);
}

/// One cell of a policy under the two vetoes: what it shows, whether the
/// binding sheds, each worker's state with its requests in flight, and the
/// steps.
type PolicyCell = (&'static str, bool, Vec<(State, u32)>, Vec<Step>);

/// One cell of the ring under the two vetoes: what it shows, whether the
/// binding sheds, and four workers' states with their requests in flight.
type RingCell = (&'static str, bool, [(State, u32); 4]);

/// The cells of one policy: the two vetoes beside each other.
fn liveness_cells(rows: &mut Vec<Row>) {
    use State::{Down, Free, Over, Stalled, StalledOver};
    let rr = "round_robin";
    let follow = || vec![pick(), Step::Loads(vec![0, 1, 3]), pick()];
    let cells: Vec<PolicyCell> = vec![
        (
            "a-doubted-worker-under-its-thresholds-beside-one-over-under-shedding",
            true,
            vec![(Over, 0), (Stalled, 3), (Stalled, 1)],
            follow(),
        ),
        (
            "a-pool-over-on-every-worker-some-doubted-too-under-shedding",
            true,
            vec![(Over, 3), (StalledOver, 1)],
            vec![pick()],
        ),
        (
            "a-pool-doubted-and-over-on-every-worker-under-shedding",
            true,
            all_of(StalledOver),
            vec![pick()],
        ),
        (
            "a-pool-doubted-and-over-on-every-worker",
            false,
            all_of(StalledOver),
            vec![pick(), Step::Loads(vec![0, 5, 2]), pick()],
        ),
        (
            "a-worker-over-alone-beside-the-doubted-ones",
            false,
            vec![(StalledOver, 0), (Stalled, 3), (Over, 5)],
            vec![pick()],
        ),
        (
            "doubted-workers-over-their-thresholds-or-not",
            false,
            vec![(StalledOver, 2), (Stalled, 1), (Down, 0)],
            vec![pick(), Step::Loads(vec![0, 5, 0]), pick()],
        ),
        (
            "a-doubted-worker-over-its-thresholds-beside-one-under-them-under-shedding",
            true,
            vec![(StalledOver, 0), (Stalled, 3)],
            vec![pick()],
        ),
        (
            "a-worker-that-is-down-beside-the-doubted-ones",
            false,
            vec![(Down, 0), (Stalled, 3), (Stalled, 1)],
            follow(),
        ),
        (
            "a-worker-free-of-both-vetoes-beside-the-doubted-ones",
            false,
            vec![(Stalled, 0), (Free, 9), (StalledOver, 0)],
            vec![pick(), pick(), pick()],
        ),
        (
            "a-worker-free-of-both-vetoes-beside-the-doubted-ones-under-shedding",
            true,
            vec![(Stalled, 0), (Free, 9), (StalledOver, 0)],
            vec![pick(), pick(), pick()],
        ),
        (
            "a-tried-worker-among-the-doubted",
            false,
            all_of(Stalled),
            vec![retry(&[1]), retry(&[1, 2])],
        ),
    ];
    for (shows, shed, workers, steps) in cells {
        rows.push(sequence_row(
            row_name(rr, shows),
            rr,
            shed,
            mixed(&workers),
            steps,
        ));
    }
}

/// Three workers in one state, holding 3, 1 and 2 requests.
fn all_of(state: State) -> Vec<(State, u32)> {
    vec![(state, 3), (state, 1), (state, 2)]
}

/// The rows of the breaker: a worker behind an open circuit breaker, in a
/// pool of such workers and beside a worker a soft veto stands on.
fn breaker_rows(rows: &mut Vec<Row>) {
    use State::{Breaker, Over, Stalled};
    // Under every policy, with shedding off and on: open breakers on every
    // worker.
    for policy in POLICIES {
        for (shed, shows) in [
            (false, "a-pool-behind-open-breakers-on-every-worker"),
            (
                true,
                "a-pool-behind-open-breakers-on-every-worker-under-shedding",
            ),
        ] {
            rows.push(sequence_row(
                row_name(policy, shows),
                policy,
                shed,
                all(Breaker, &[3, 1, 2]),
                vec![pick()],
            ));
        }
    }
    // One worker a soft veto stands on, the fullest of the pool, between
    // two behind open breakers.
    let rr = "round_robin";
    let cells: Vec<(&str, bool, State)> = vec![
        ("a-doubted-worker-among-open-breakers", false, Stalled),
        (
            "a-doubted-worker-among-open-breakers-under-shedding",
            true,
            Stalled,
        ),
        (
            "a-worker-over-its-thresholds-among-open-breakers",
            false,
            Over,
        ),
        (
            "a-worker-over-its-thresholds-among-open-breakers-under-shedding",
            true,
            Over,
        ),
    ];
    for (shows, shed, soft) in cells {
        rows.push(sequence_row(
            row_name(rr, shows),
            rr,
            shed,
            mixed(&[(Breaker, 0), (soft, 9), (Breaker, 1)]),
            vec![pick(), pick()],
        ));
    }
}

/// Keyed requests under the liveness veto: a pinned key of the manual
/// policy, and a routing key on the ring.
fn liveness_keys(rows: &mut Vec<Row>) {
    use State::{Free, Over, Stalled};
    rows.push(sequence_row(
        "manual-a-key-while-every-worker-is-doubted",
        "manual",
        false,
        all(Free, &[1, 2, 3]),
        vec![
            keyed("k1"),
            Step::States(vec![Stalled, Stalled, Stalled]),
            Step::Loads(vec![9, 2, 3]),
            keyed("k1"),
            keyed("k1"),
            Step::States(vec![Free, Free, Free]),
            keyed("k1"),
        ],
    ));
    let ring = "consistent_hashing";
    let beside_over = [(Stalled, 5), (Over, 4), (Over, 0), (Over, 2)];
    let cells: [RingCell; 4] = [
        (
            "a-key-whose-worker-is-doubted-beside-free-members",
            false,
            [(Stalled, 5), (Free, 4), (Free, 0), (Free, 2)],
        ),
        (
            "a-key-with-every-worker-doubted",
            false,
            [(Stalled, 5), (Stalled, 4), (Stalled, 0), (Stalled, 2)],
        ),
        (
            "a-key-whose-worker-is-doubted-beside-members-over-their-thresholds",
            false,
            beside_over,
        ),
        (
            "a-key-whose-worker-is-doubted-beside-members-over-their-thresholds-under-shedding",
            true,
            beside_over,
        ),
    ];
    for (shows, shed, workers) in cells {
        rows.push(ring_row(
            row_name(ring, shows),
            ring,
            shed,
            "k1",
            mixed(&workers),
            vec![keyed("k1")],
        ));
    }
}
