//! The recorder of the gateway's placement of a prefill and decode pair:
//! each row is a prefill pool and a decode pool, what the router holds of
//! each worker (the requests in flight, and whether it is ready, over its
//! overload thresholds or doubted by the liveness tracker), the KV transport
//! each worker names, the policy each pool runs under, whether shedding is
//! on, and what the gateway's pair placement answered at each request. The
//! policy controller crate replays the rows through its own pair decision
//! (`crates/policy_controller/tests/classic_rows.rs`).
//!
//! A request's answer is the pair it was placed on, each worker by its place
//! in its pool's list, or no pair with the role the placement named and why.
//! Which decodes a prefill may pair with is the gateway's own pair index
//! under the lenient pairing mode on the plain wire: two workers pair when
//! their transports are equal or one of them names none.
//!
//! A pool's policy is named in its configuration spelling; `power_of_two`
//! carries its load metric after a colon. Under `least_requests` the fewest
//! requests in flight decides with no draw and no state, so a row replays
//! pick for pick. A rotation does not: the gateway starts a new candidate
//! set's rotation at a shared count, so no row runs under `round_robin`.
//! `random` stands on the side where a row must show whether the policy was
//! asked: where it was not, one pair takes every draw.
//!
//! This file is an ignored test of the gateway crate, kept beside the code
//! it runs. It records from the checkout it is built in and writes the sheet
//! under `CLASSIC_ROWS_OUT`, naming `CLASSIC_ROWS_COMMIT` as the commit it
//! recorded from. The sheets the policy controller crate replays are in
//! `crates/policy_controller/tests/classic/sheets`; to record this one again:
//!
//! ```text
//! CLASSIC_ROWS_OUT=<dir> CLASSIC_ROWS_COMMIT=<commit> \
//!   cargo test -p smg --lib routers::common::classic_pair_select_rows -- --ignored --nocapture
//! ```
//!
//! That filter also runs this module's two child recorders
//! ([`classic_pair_bound_rows`], [`classic_pair_runtime_rows`]), each of
//! which writes a sheet of its own beside this one: their rows draw
//! nothing. A row of the second is placed on the wire that carries one
//! runtime's request, and names the runtime of a worker that is not on the
//! sheet's own.
//!
//! A sequence row is recorded twenty times from fresh pools and fresh
//! policies and must come out the same each time; a drawn row records how
//! often each pair took the row's one request over its draws, each draw from
//! fresh pools and fresh policies.

use std::{collections::HashMap, fs, path::PathBuf, sync::Arc, time::Duration};

use openai_protocol::{
    model_card::ModelCard,
    worker::{WorkerSpec, WorkerStatus},
};
use serde::Serialize;
use smg_worker::{
    identity::basic::{ConnectionMode, RuntimeType, Worker, WorkerType},
    pd::pairing::PdPairingMode,
    runtime::StallReason,
    test_support::no_health_check,
};

use super::{
    placement::{select_pair, PairFailure, PlacedPair, PlacementFailure, PlacementInputs, Stated},
    prefill_admission::{PrefillAdmission, PrefillAdmissionAttempt, PrefillSelectionContext},
};
use crate::{
    config::types::PolicyConfig,
    policies::{PolicyFactory, PolicyRegistry},
    routers::{
        common::{
            pair_index::{PdPairIndex, PdWire},
            snapshot::WorkerRegistry,
        },
        grpc::backend_client::BasicWorkerBuilder,
    },
};

const MODEL: &str = "m";
/// Two KV transports, as a worker's registration names one.
const TX_A: &str = "NixlConnector";
const TX_B: &str = "MooncakeConnector";

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
    /// Not ready.
    Down,
}

/// A decode's running window as its registration reports it, and the rooms
/// of it the gateway's decode admission has claimed.
#[derive(Clone, Copy, Serialize)]
struct Rooms {
    window: u16,
    admitted: u16,
}

/// One worker of a row: the requests the router holds on it, its state, the
/// transport its registration names (`a`, `b` or none) and, for a decode
/// that reports a running window, its rooms. A worker that is not on the
/// sheet's one runtime names its own: `other`, a second known runtime, or
/// `unknown`, none.
#[derive(Clone, Copy, Serialize)]
struct Member {
    in_flight: u32,
    state: State,
    transport: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rooms: Option<Rooms>,
    #[serde(skip_serializing_if = "Option::is_none")]
    runtime: Option<&'static str>,
}

const fn member(in_flight: u32, state: State, transport: Option<&'static str>) -> Member {
    Member {
        in_flight,
        state,
        transport,
        rooms: None,
        runtime: None,
    }
}

impl Member {
    /// The member on `runtime`: `other` or `unknown`.
    const fn on(mut self, runtime: &'static str) -> Self {
        self.runtime = Some(runtime);
        self
    }
}

/// The answer to one request.
#[derive(Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Answer {
    /// The prefill's place in its pool, then the decode's in its own.
    Pair(usize, usize),
    /// No pair: the role the placement named, and why.
    None { role: String, why: &'static str },
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum Outcome {
    /// The answer to each request, in order.
    Sequence(Vec<Answer>),
    /// How often each pair took the row's one request over `draws` runs of
    /// the row: prefill, decode, count; a pair never placed is not listed.
    Counts {
        draws: u32,
        pairs: Vec<(usize, usize, u32)>,
    },
}

#[derive(Serialize)]
struct Row {
    name: &'static str,
    /// Each pool's policy, in its configuration spelling.
    prefill_policy: &'static str,
    decode_policy: &'static str,
    /// Whether an all-overloaded pool is shed (`--worker-overload-shed`).
    shed: bool,
    /// The most requests the prefill admission lets one prefill hold, when
    /// the row runs behind it; a prefill holding that many has no capacity.
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_bound: Option<u32>,
    /// The wire the row's pair is placed for, where it is not the plain
    /// one: `grpc`, which carries one runtime's request.
    #[serde(skip_serializing_if = "Option::is_none")]
    wire: Option<&'static str>,
    prefills: Vec<Member>,
    decodes: Vec<Member>,
    /// How many requests the row places, one after another.
    requests: u32,
    outcome: Outcome,
}

#[derive(Serialize)]
struct Sheet {
    recorded_from: String,
    policy: &'static str,
    rows: Vec<Row>,
}

/// The policy's configuration, as the gateway's own configuration file
/// spells it; a load metric follows `power_of_two` after a colon.
fn config_of(policy: &str) -> PolicyConfig {
    if policy == "cache_aware" {
        return PolicyConfig::cache_aware(0.5, 32, 1.1, 0, 4096);
    }
    let text = match policy.split_once(':') {
        Some((kind, metric)) => format!(r#"{{"type":"{kind}","load_metric":"{metric}"}}"#),
        None => format!(r#"{{"type":"{policy}"}}"#),
    };
    serde_json::from_str(&text).expect("the policy's configuration")
}

fn worker(
    side: &str,
    place: usize,
    role: WorkerType,
    member: Member,
    wire: PdWire,
) -> Arc<dyn Worker> {
    let (scheme, mode) = match wire {
        PdWire::Http => ("http", ConnectionMode::Http),
        PdWire::Grpc => ("grpc", ConnectionMode::Grpc),
    };
    let mut spec = WorkerSpec::new(format!("{scheme}://{side}{place}:8000"));
    spec.runtime_type = match member.runtime {
        None => RuntimeType::Vllm,
        Some("other") => RuntimeType::Sglang,
        Some("unknown") => RuntimeType::Unspecified,
        Some(runtime) => panic!("a row names no runtime {runtime}"),
    };
    spec.kv_connector = member.transport.map(|transport| {
        if transport == "a" {
            TX_A.to_string()
        } else {
            TX_B.to_string()
        }
    });
    let mut labels = HashMap::new();
    if let Some(rooms) = member.rooms {
        labels.insert("max_running_requests".to_string(), rooms.window.to_string());
    }
    let worker = BasicWorkerBuilder::from_spec(spec)
        .model(ModelCard::new(MODEL))
        .worker_type(role)
        .connection_mode(mode)
        .labels(labels)
        .health_config(no_health_check())
        .build();
    worker.set_status(WorkerStatus::Ready);
    Arc::new(worker)
}

/// Fresh pools of ready workers under fresh policies, in the row's state,
/// with the pair index the gateway builds of them.
struct Bed {
    registry: WorkerRegistry,
    policies: PolicyRegistry,
    prefills: Vec<Arc<dyn Worker>>,
    decodes: Vec<Arc<dyn Worker>>,
    pairs: PdPairIndex,
    /// The prefill admission the row runs behind, if it names a bound: no
    /// queue, so a request that finds no capacity is answered at once.
    admission: Option<PrefillAdmission>,
}

impl Bed {
    fn new(cell: &Cell) -> Self {
        let registry = WorkerRegistry::new();
        let wire = match cell.wire {
            None => PdWire::Http,
            Some("grpc") => PdWire::Grpc,
            Some(wire) => panic!("{}: a row names no wire {wire}", cell.name),
        };
        let pool = |side: &str, role: WorkerType, members: &[Member]| -> Vec<Arc<dyn Worker>> {
            members
                .iter()
                .enumerate()
                .map(|(place, &member)| {
                    let worker = worker(side, place, role, member, wire);
                    registry
                        .register(Arc::clone(&worker))
                        .expect("the worker registers");
                    worker
                })
                .collect()
        };
        let prefill_pool = pool("p", WorkerType::Prefill, &cell.prefills);
        let decode_pool = pool("d", WorkerType::Decode, &cell.decodes);
        registry.set_overload_shed(cell.shed);
        let policies = PolicyRegistry::new(config_of("random"));
        policies.set_prefill_policy(PolicyFactory::create_from_config(&config_of(
            cell.prefill_policy,
        )));
        policies.set_decode_policy(PolicyFactory::create_from_config(&config_of(
            cell.decode_policy,
        )));
        // The index is judged from membership alone, before any state.
        let pairs = PdPairIndex::build(
            Arc::from(prefill_pool.clone()),
            Arc::from(decode_pool.clone()),
            wire,
            PdPairingMode::Lenient,
        );
        for (workers, members) in [
            (&prefill_pool, &cell.prefills),
            (&decode_pool, &cell.decodes),
        ] {
            for (worker, member) in workers.iter().zip(members) {
                while worker.load() < member.in_flight as usize {
                    worker.increment_load();
                }
                if let Some(rooms) = member.rooms {
                    assert!(
                        worker.try_admit_pd(usize::from(rooms.admitted), usize::from(rooms.window)),
                        "{}: the rooms claimed fit the window",
                        cell.name
                    );
                }
                worker.set_status(if member.state == State::Down {
                    WorkerStatus::NotReady
                } else {
                    WorkerStatus::Ready
                });
                registry.set_worker_overloaded(worker, member.state == State::Over);
                worker.set_stall((member.state == State::Stalled).then_some(StallReason::Wedged));
            }
        }
        let admission = cell
            .prefill_bound
            .map(|bound| PrefillAdmission::new(bound as usize, 0, Duration::ZERO));
        Self {
            registry,
            policies,
            prefills: prefill_pool,
            decodes: decode_pool,
            pairs,
            admission,
        }
    }

    fn answer(&self, placed: &Result<PlacedPair, Box<PairFailure>>) -> Answer {
        let place = |pool: &[Arc<dyn Worker>], worker: &Arc<dyn Worker>| {
            pool.iter()
                .position(|w| w.url() == worker.url())
                .expect("a worker of the row")
        };
        match placed {
            Ok(pair) => Answer::Pair(
                place(&self.prefills, pair.prefill.worker()),
                place(&self.decodes, pair.decode.worker()),
            ),
            Err(failure) => Answer::None {
                role: format!("{:?}", failure.role).to_lowercase(),
                why: match failure.verdict {
                    PlacementFailure::NoCandidates => "no_candidates",
                    PlacementFailure::AllOverloaded(_) => "overloaded",
                    PlacementFailure::Unavailable => "unavailable",
                    PlacementFailure::PrefillAtCapacity => "at_capacity",
                    PlacementFailure::PolicyDeclined(_) => "declined",
                    PlacementFailure::NoCompatiblePair { .. } => "other",
                },
            },
        }
    }

    /// One request through the gateway's pair placement, behind the prefill
    /// admission when the row names its bound.
    fn pick(&self) -> Answer {
        let Some(admission) = &self.admission else {
            return self.answer(&select_pair(
                &self.registry,
                &self.policies,
                MODEL,
                &self.pairs,
                None,
                None,
                PlacementInputs::default(),
                Stated::nothing,
            ));
        };
        let mut answer = None;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .expect("a runtime");
        let admitted = runtime.block_on(admission.admit(
            None,
            |capacity: &PrefillSelectionContext<'_>| {
                let placed = select_pair(
                    &self.registry,
                    &self.policies,
                    MODEL,
                    &self.pairs,
                    None,
                    Some(capacity),
                    PlacementInputs::default(),
                    Stated::nothing,
                );
                answer = Some(self.answer(&placed));
                match placed {
                    Ok(pair) => capacity.select(pair.prefill.into_worker(), ()),
                    Err(failure)
                        if matches!(failure.verdict, PlacementFailure::PrefillAtCapacity) =>
                    {
                        PrefillAdmissionAttempt::AtCapacity
                    }
                    Err(_) => PrefillAdmissionAttempt::Unavailable,
                }
            },
        ));
        // The admitted request is let go at once: the row's next request
        // meets the pools as the row states them.
        drop(admitted);
        answer.expect("the admission asked the placement")
    }
}

/// The row's shape before its outcome is recorded.
struct Cell {
    name: &'static str,
    prefill_policy: &'static str,
    decode_policy: &'static str,
    shed: bool,
    prefill_bound: Option<u32>,
    wire: Option<&'static str>,
    prefills: Vec<Member>,
    decodes: Vec<Member>,
}

impl Cell {
    /// The row behind a prefill admission of `bound` requests a prefill.
    fn bound(mut self, bound: u32) -> Self {
        self.prefill_bound = Some(bound);
        self
    }

    /// Every decode reports a window of two rooms; `admitted` says how many
    /// of each decode's the admission has claimed.
    fn rooms(mut self, admitted: [u16; 3]) -> Self {
        for (decode, admitted) in self.decodes.iter_mut().zip(admitted) {
            decode.rooms = Some(Rooms {
                window: 2,
                admitted,
            });
        }
        self
    }

    fn run(&self, requests: u32) -> Vec<Answer> {
        let bed = Bed::new(self);
        (0..requests).map(|_| bed.pick()).collect()
    }

    fn row(self, requests: u32, outcome: Outcome) -> Row {
        Row {
            name: self.name,
            prefill_policy: self.prefill_policy,
            decode_policy: self.decode_policy,
            shed: self.shed,
            prefill_bound: self.prefill_bound,
            wire: self.wire,
            prefills: self.prefills,
            decodes: self.decodes,
            requests,
            outcome,
        }
    }

    /// `requests` requests, one after another: the same answers in each of
    /// twenty recordings.
    fn sequence(self, requests: u32) -> Row {
        let first = self.run(requests);
        for _ in 0..19 {
            assert!(
                self.run(requests) == first,
                "{}: a sequence row must not depend on a draw",
                self.name
            );
        }
        self.row(requests, Outcome::Sequence(first))
    }

    /// One request, drawn 6,000 times: how often each pair took it.
    fn drawn(self) -> Row {
        const DRAWS: u32 = 6_000;
        let mut counts = vec![vec![0u32; self.decodes.len()]; self.prefills.len()];
        for _ in 0..DRAWS {
            match self.run(1).pop() {
                Some(Answer::Pair(prefill, decode)) => counts[prefill][decode] += 1,
                _ => panic!("{}: a drawn row places its request", self.name),
            }
        }
        let pairs = counts
            .iter()
            .enumerate()
            .flat_map(|(prefill, row)| {
                row.iter()
                    .enumerate()
                    .filter(|(_, &count)| count > 0)
                    .map(move |(decode, &count)| (prefill, decode, count))
            })
            .collect();
        self.row(
            1,
            Outcome::Counts {
                draws: DRAWS,
                pairs,
            },
        )
    }
}

/// Two prefills of two transports and three decodes: one of each transport
/// and one that names none. The first prefill pairs with decodes 0 and 2,
/// the second with decodes 1 and 2.
fn fleet(
    name: &'static str,
    policies: (&'static str, &'static str),
    shed: bool,
    prefills: [(u32, State); 2],
    decodes: [(u32, State); 3],
) -> Cell {
    let [(p0, s0), (p1, s1)] = prefills;
    let [(d0, t0), (d1, t1), (d2, t2)] = decodes;
    Cell {
        name,
        prefill_policy: policies.0,
        decode_policy: policies.1,
        shed,
        prefill_bound: None,
        wire: None,
        prefills: vec![member(p0, s0, Some("a")), member(p1, s1, Some("b"))],
        decodes: vec![
            member(d0, t0, Some("a")),
            member(d1, t1, Some("b")),
            member(d2, t2, None),
        ],
    }
}

fn rows() -> Vec<Row> {
    use State::{Down, Free, Over, Stalled};
    // The fewest requests in flight decides: no draw, no state.
    const BY_COUNT: &str = "power_of_two:least_requests";
    // A policy that weighs load against what a worker holds.
    const WEIGHING: &str = "cache_aware";
    // A uniform draw: a row under it shows whether the policy was asked.
    const DRAWN: &str = "random";
    const COUNTED: (&str, &str) = (BY_COUNT, BY_COUNT);
    let mut rows = Vec::new();

    // Every worker open: each pool's policy picks, the decode among the
    // chosen prefill's partners.
    let open2 = [(1, Free), (0, Free)];
    let open3 = [(2, Free), (1, Free), (0, Free)];
    let idle3 = [(0, Free), (0, Free), (0, Free)];
    rows.push(fleet("pair-every-worker-open", COUNTED, false, open2, open3).sequence(2));
    rows.push(
        fleet(
            "pair-the-decode-policy-draws-among-the-chosen-prefills-partners",
            (BY_COUNT, DRAWN),
            false,
            open2,
            idle3,
        )
        .drawn(),
    );

    // ONE prefill over its thresholds beside one under them.
    let one_over = [(0, Over), (5, Free)];
    rows.push(
        fleet(
            "pair-one-prefill-over-is-passed-by",
            COUNTED,
            false,
            one_over,
            open3,
        )
        .sequence(2),
    );
    rows.push(
        fleet(
            "pair-one-prefill-over-under-a-weighing-policy",
            (WEIGHING, BY_COUNT),
            false,
            one_over,
            open3,
        )
        .sequence(2),
    );
    rows.push(
        fleet(
            "pair-one-prefill-over-under-a-weighing-policy-and-shedding",
            (WEIGHING, BY_COUNT),
            true,
            one_over,
            open3,
        )
        .sequence(2),
    );

    // Every prefill over, under each kind of policy and under shedding. The
    // decode policy draws: where it is not asked, one pair takes every draw.
    let both_over = [(4, Over), (2, Over)];
    let decode_loads = [(1, Free), (0, Free), (5, Free)];
    rows.push(
        fleet(
            "pair-every-prefill-over-pairs-by-load",
            (BY_COUNT, DRAWN),
            false,
            both_over,
            decode_loads,
        )
        .drawn(),
    );
    rows.push(
        fleet(
            "pair-every-prefill-over-under-a-weighing-policy",
            (WEIGHING, DRAWN),
            false,
            both_over,
            decode_loads,
        )
        .drawn(),
    );
    rows.push(
        fleet(
            "pair-every-prefill-over-is-shed-under-shedding",
            COUNTED,
            true,
            both_over,
            decode_loads,
        )
        .sequence(2),
    );
    rows.push(
        fleet(
            "pair-every-prefill-over-under-a-weighing-policy-and-shedding",
            (WEIGHING, BY_COUNT),
            true,
            both_over,
            decode_loads,
        )
        .sequence(2),
    );

    // The decode pool over its thresholds, whole and in part.
    let decodes_over = [(3, Over), (1, Over), (2, Over)];
    let prefills = [(0, Free), (1, Free)];
    rows.push(
        fleet(
            "pair-every-decode-over-takes-the-least-loaded-partner",
            (BY_COUNT, DRAWN),
            false,
            prefills,
            decodes_over,
        )
        .drawn(),
    );
    rows.push(
        fleet(
            "pair-every-decode-over-is-shed-under-shedding",
            COUNTED,
            true,
            prefills,
            decodes_over,
        )
        .sequence(2),
    );
    rows.push(
        fleet(
            "pair-one-prefills-partners-over-beside-a-free-decode",
            COUNTED,
            false,
            prefills,
            [(0, Over), (0, Free), (0, Over)],
        )
        .sequence(2),
    );
    rows.push(
        fleet(
            "pair-every-decode-over-a-tie-of-two-partners",
            COUNTED,
            false,
            [(0, Free), (0, Down)],
            [(2, Over), (2, Over), (2, Over)],
        )
        .drawn(),
    );
    // The second prefill is chosen; its least loaded partner is doubted by
    // the liveness tracker, which the overload veto's lift does not cross.
    rows.push(
        fleet(
            "pair-two-decodes-over-and-one-stalled",
            (BY_COUNT, DRAWN),
            false,
            open2,
            [(3, Over), (1, Stalled), (2, Over)],
        )
        .drawn(),
    );

    // A side doubted by the liveness tracker, whole and in part.
    let stalled2 = [(4, Stalled), (2, Stalled)];
    rows.push(
        fleet(
            "pair-every-prefill-stalled-pairs-by-load",
            (BY_COUNT, DRAWN),
            false,
            stalled2,
            decode_loads,
        )
        .drawn(),
    );
    rows.push(
        fleet(
            "pair-every-prefill-stalled-under-a-weighing-policy",
            (WEIGHING, DRAWN),
            false,
            stalled2,
            decode_loads,
        )
        .drawn(),
    );
    rows.push(
        fleet(
            "pair-every-prefill-stalled-under-shedding",
            COUNTED,
            true,
            stalled2,
            decode_loads,
        )
        .sequence(2),
    );
    rows.push(
        fleet(
            "pair-every-decode-stalled-pairs-by-load",
            COUNTED,
            false,
            [(3, Free), (1, Free)],
            [(2, Stalled), (4, Stalled), (0, Stalled)],
        )
        .sequence(2),
    );
    let over_and_stalled = [(5, Over), (1, Stalled)];
    rows.push(
        fleet(
            "pair-one-prefill-over-and-one-stalled",
            (BY_COUNT, DRAWN),
            false,
            over_and_stalled,
            decode_loads,
        )
        .drawn(),
    );
    rows.push(
        fleet(
            "pair-one-prefill-over-and-one-stalled-under-a-weighing-policy",
            (WEIGHING, DRAWN),
            false,
            over_and_stalled,
            decode_loads,
        )
        .drawn(),
    );
    rows.push(
        fleet(
            "pair-one-prefill-over-and-one-stalled-under-shedding",
            COUNTED,
            true,
            over_and_stalled,
            decode_loads,
        )
        .sequence(2),
    );

    // The default's ties: a pair is placed prefill first, so the unit of a
    // tie is the prefill, then its partners.
    rows.push(
        fleet(
            "pair-the-default-draws-over-two-tied-prefills",
            COUNTED,
            false,
            [(2, Over), (2, Over)],
            [(0, Free), (0, Free), (9, Free)],
        )
        .drawn(),
    );
    rows.push(
        fleet(
            "pair-the-default-draws-a-prefill-then-its-tied-partners",
            COUNTED,
            false,
            [(2, Over), (2, Over)],
            [(5, Free), (0, Free), (0, Free)],
        )
        .drawn(),
    );

    // A decode's rooms: every decode reports a window of two.
    rows.push(
        fleet(
            "pair-a-decode-without-a-free-room-is-passed-by",
            COUNTED,
            false,
            prefills,
            [(0, Free), (0, Free), (5, Free)],
        )
        .rooms([2, 0, 0])
        .sequence(2),
    );
    rows.push(
        fleet(
            "pair-no-partner-has-a-free-room",
            COUNTED,
            false,
            prefills,
            [(0, Free), (0, Free), (5, Free)],
        )
        .rooms([2, 0, 2])
        .sequence(2),
    );
    rows.push(
        fleet(
            "pair-every-decode-over-takes-a-free-room-first",
            (BY_COUNT, DRAWN),
            false,
            prefills,
            decodes_over,
        )
        .rooms([0, 0, 2])
        .drawn(),
    );
    rows.push(
        fleet(
            "pair-the-default-reads-no-room",
            (BY_COUNT, DRAWN),
            false,
            both_over,
            decode_loads,
        )
        .rooms([0, 2, 0])
        .drawn(),
    );

    // The prefill admission's bound: four requests a prefill.
    rows.push(
        fleet(
            "pair-a-prefill-at-capacity-is-passed-by",
            (DRAWN, BY_COUNT),
            false,
            [(4, Free), (1, Free)],
            open3,
        )
        .bound(4)
        .drawn(),
    );
    rows.push(
        fleet(
            "pair-every-prefill-at-capacity",
            COUNTED,
            false,
            [(4, Free), (5, Free)],
            open3,
        )
        .bound(4)
        .sequence(2),
    );
    rows.push(
        fleet(
            "pair-the-default-sheds-when-its-prefill-is-at-capacity",
            COUNTED,
            false,
            [(4, Over), (6, Over)],
            open3,
        )
        .bound(4)
        .sequence(2),
    );

    // A side with no worker ready.
    rows.push(
        fleet(
            "pair-every-prefill-down",
            COUNTED,
            false,
            [(0, Down), (0, Down)],
            open3,
        )
        .sequence(2),
    );
    rows.push(
        fleet(
            "pair-every-decode-down",
            COUNTED,
            false,
            prefills,
            [(0, Down), (0, Down), (0, Down)],
        )
        .sequence(2),
    );
    rows.push(
        fleet(
            "pair-one-prefills-partners-down",
            COUNTED,
            false,
            prefills,
            [(0, Down), (0, Free), (0, Down)],
        )
        .sequence(2),
    );

    // The least-loaded default's own rules, a row each. It runs only over a
    // pool whose every routable worker is vetoed: one free decode keeps it
    // off, though no partner of the one ready prefill is open.
    rows.push(
        fleet(
            "pair-the-only-ready-prefills-partners-over-beside-a-free-decode",
            COUNTED,
            false,
            [(0, Free), (0, Down)],
            [(0, Over), (0, Free), (0, Over)],
        )
        .sequence(2),
    );
    // It reads a worker's health: the least loaded prefill's least loaded
    // partner is down.
    rows.push(
        fleet(
            "pair-every-prefill-over-passes-by-a-down-partner",
            (BY_COUNT, DRAWN),
            false,
            both_over,
            [(1, Free), (0, Down), (5, Free)],
        )
        .drawn(),
    );
    // Between prefills of one load it weighs the partner's load.
    rows.push(
        fleet(
            "pair-the-default-weighs-the-partners-load-between-tied-prefills",
            COUNTED,
            false,
            [(2, Over), (2, Over)],
            [(3, Free), (0, Free), (9, Free)],
        )
        .drawn(),
    );
    // Under shedding it passes by a worker over its thresholds for a
    // stalled one that holds more.
    rows.push(
        fleet(
            "pair-one-prefill-over-and-one-stalled-with-more-under-shedding",
            COUNTED,
            true,
            [(0, Over), (5, Stalled)],
            decode_loads,
        )
        .sequence(2),
    );
    // A weighing policy's pool, every worker over, with no open partner:
    // the placement names the prefill.
    rows.push(
        fleet(
            "pair-every-prefill-over-under-a-weighing-policy-and-every-decode-down",
            (WEIGHING, BY_COUNT),
            false,
            both_over,
            [(0, Down), (0, Down), (0, Down)],
        )
        .sequence(2),
    );

    // A role with no worker at all names itself.
    let one_role = |name: &'static str, prefills: Vec<Member>, decodes: Vec<Member>| Cell {
        name,
        prefill_policy: BY_COUNT,
        decode_policy: BY_COUNT,
        shed: false,
        prefill_bound: None,
        wire: None,
        prefills,
        decodes,
    };
    let lone = vec![member(0, Free, Some("a"))];
    rows.push(one_role("pair-no-prefill-of-the-model", Vec::new(), lone.clone()).sequence(2));
    rows.push(one_role("pair-no-decode-of-the-model", lone, Vec::new()).sequence(2));
    rows
}

/// Writes `rows` as the sheet `file` of `policy` under `CLASSIC_ROWS_OUT`,
/// naming `CLASSIC_ROWS_COMMIT` as the commit they were recorded from.
fn write(file: &str, policy: &'static str, rows: Vec<Row>) {
    let out = PathBuf::from(
        std::env::var_os("CLASSIC_ROWS_OUT").expect("CLASSIC_ROWS_OUT names the output directory"),
    );
    let commit =
        std::env::var("CLASSIC_ROWS_COMMIT").expect("CLASSIC_ROWS_COMMIT names the commit");
    let sheet = Sheet {
        recorded_from: commit,
        policy,
        rows,
    };
    fs::create_dir_all(&out).expect("the output directory");
    let path = out.join(file);
    let text = serde_json::to_string_pretty(&sheet).expect("the sheet serializes");
    fs::write(&path, text + "\n").expect("the sheet is written");
    println!("{} rows -> {}", sheet.rows.len(), path.display());
}

#[test]
#[ignore = "records the classic rows into CLASSIC_ROWS_OUT"]
fn record_the_classic_pair_select_rows() {
    write("pair_select.json", "pair-select", rows());
}

/// The rows behind the prefill admission's bound where a prefill that holds
/// it has an open partner or none, on a sheet of their own.
mod classic_pair_bound_rows;

/// The rows of a pair on one known runtime beside workers of another or of
/// none, on the wire that carries one runtime's request and on a sheet of
/// their own.
mod classic_pair_runtime_rows;
