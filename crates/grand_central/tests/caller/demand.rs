//! A retry stays on its wire.
//!
//! The program states each worker's wire beside its lists: one number for
//! the runtime the worker runs and the transport it is dialed over. A retry
//! states the number of the worker its request went to first as its demand,
//! since the request a route keeps for a retry was built for that wire, and
//! the engine holds the retry to the workers that state the same number,
//! ahead of every stage. A first attempt demands nothing. Each case serves
//! requests over workers on more than one wire and reads where a retry was
//! sent.

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, OnceLock, Weak,
};

use smg_grand_central::{
    document::{DecodePick, LoadMetricSetting, PolicySettings},
    engine::{ShedReason, Unplaced},
};
use smg_worker::{
    identity::{basic::Worker, builder::BasicWorkerBuilder},
    model_card::ModelCard,
    test_support::no_health_check,
    worker::{ConnectionMode, RuntimeType, WorkerStatus, WorkerType},
};

use crate::{
    pair::{decode, in_turn, picked, prefill, PLAIN},
    program::{Answer, Program, ATTEMPTS, NO_WORKER},
    under, under_pairs, url, MODEL,
};

/// A wire: a runtime, and the transport a worker that runs it is dialed
/// over.
pub(crate) type Wire = (RuntimeType, ConnectionMode);

/// The wire of a worker that states no runtime, dialed over gRPC.
pub(crate) const GRPC: Wire = (RuntimeType::Unspecified, ConnectionMode::Grpc);

/// The wire of a worker that states no runtime, dialed over HTTP: the wire
/// of the other transport than [`GRPC`].
pub(crate) const HTTP: Wire = (RuntimeType::Unspecified, ConnectionMode::Http);

/// The wire of a worker of another runtime than [`GRPC`], dialed over gRPC
/// too.
const OTHER: Wire = (RuntimeType::Generic, ConnectionMode::Grpc);

/// The wire of a worker of [`OTHER`]'s runtime, dialed over HTTP. The two
/// differ by their transport alone, under a runtime whose number, laid over
/// a transport's, would hide which transport it is.
const OTHER_HTTP: Wire = (RuntimeType::Generic, ConnectionMode::Http);

/// A ready worker of `kind` that serves [`MODEL`] at `address`, on `wire`.
fn dialed(address: &str, kind: WorkerType, wire: Wire) -> Arc<dyn Worker> {
    let (runtime, transport) = wire;
    let worker = <BasicWorkerBuilder>::new(address)
        .model(ModelCard::new(MODEL))
        .worker_type(kind)
        .runtime_type(runtime)
        .connection_mode(transport)
        .health_config(no_health_check())
        .build();
    worker.set_status(WorkerStatus::Ready);
    Arc::new(worker)
}

/// Single workers at [`url`]`(0)` and on, each on the wire `wires` names at
/// its place.
pub(crate) fn singles(wires: &[Wire]) -> Vec<Arc<dyn Worker>> {
    let at_place = wires.iter().enumerate();
    let single = |(at, wire): (usize, &Wire)| dialed(&url(at), WorkerType::Regular, *wire);
    at_place.map(single).collect()
}

/// A send that the worker at `failing` answers a 502, and every other
/// worker a 200.
fn failing_at(failing: String) -> impl Fn(&dyn Worker) -> u16 + Send + Sync + 'static {
    move |worker: &dyn Worker| if worker.url() == failing { 502 } else { 200 }
}

/// A retry stays on the wire of its first worker. Three workers take their
/// turns: the first and the third on one wire, the second on another,
/// which differs by its transport or by its runtime. The first fails, and
/// each request that meets it is asked again at the third, though the turn
/// is the second's. A first attempt demands nothing: the requests that meet
/// no failure go to the workers of both wires. It holds for the wire that
/// stands first in the worker crate's lists too, a worker over HTTP that
/// states no runtime: its number is not zero, which would state no wire.
/// And it holds for two transports of one runtime ([`OTHER_HTTP`]): a
/// wire's number keeps its runtime and its transport in bits of their own.
///
/// The gateway states a wire's number of each worker and of a retry pinned
/// to its plan's wire (`stated_facts` and the request's demand,
/// `model_gateway/src/routers/common/place_request.rs:113-123`, `:192-195`):
/// its case `the_engine_keeps_a_pinned_retry_on_its_runtime_and_transport`
/// (`model_gateway/src/routers/common/placement.rs:7930-7972`).
#[test]
fn a_retry_stays_on_the_wire_of_its_first_worker() {
    for (wire, another) in [
        (GRPC, HTTP),
        (GRPC, OTHER),
        (HTTP, GRPC),
        (OTHER, OTHER_HTTP),
    ] {
        let fleet = singles(&[wire, another, wire]);
        let sends = failing_at(url(0));
        let (program, _loads, _task) = under(PolicySettings::RoundRobin, &fleet, sends);
        let answers: Vec<Answer> = (0..6).map(|_| program.serve(&PLAIN)).collect();
        assert!(answers.iter().all(|answer| answer.status == 200));
        let retried = [url(0), url(2)];
        let mut met = 0;
        for answer in &answers {
            if answer.sent[0] == url(0) {
                assert_eq!(answer.sent, retried, "beside {another:?}: {answers:?}");
                met += 1;
            } else {
                assert_eq!(answer.sent.len(), 1, "beside {another:?}: {answers:?}");
            }
        }
        assert!(met > 0, "no request met the failing worker: {answers:?}");
        let firsts = |address: String| answers.iter().any(|answer| answer.sent[0] == address);
        assert!(firsts(url(1)) && firsts(url(2)), "{answers:?}");
    }
}

/// A wire that holds one worker keeps it for the retry: the workers a
/// request tried step aside only while another is left, and the workers of
/// the other wires are no such workers. Three workers, each alone on its
/// wire, none of which serves: each request is sent to the worker its first
/// attempt went to, up to the program's bound of sends, and to no other.
///
/// The gateway narrows a pinned request's pool to its wire, and lets the
/// workers it tried step aside while another is left: its case
/// `a_pinned_wire_narrows_to_its_runtime_and_transport`
/// (`model_gateway/src/routers/common/placement.rs:4774-4796`, and
/// `:1728-1748`).
#[test]
fn a_wire_with_one_worker_keeps_it_for_the_retry() {
    let fleet = singles(&[GRPC, HTTP, OTHER]);
    let (program, _loads, _task) = under(PolicySettings::RoundRobin, &fleet, |_| 502);
    for _ in 0..3 {
        let answer = program.serve(&PLAIN);
        let expected = Answer {
            sent: vec![answer.sent[0].clone(); ATTEMPTS],
            status: 502,
            unplaced: None,
        };
        assert_eq!(answer, expected);
    }
}

/// A wire that holds no worker any more leaves the retry no endpoint.
/// Three workers, each alone on its wire: the one a request is sent to
/// leaves the model while its send runs, and the retry is placed over the
/// generation that stands then, whose two workers are on other wires. The
/// client reads the program's answer for a request the engine named no
/// worker for, after one send.
///
/// The gateway's pinned request has no answer where no worker of its pool
/// is on the wire: its case
/// `the_engine_keeps_a_pinned_retry_on_its_runtime_and_transport`
/// (`model_gateway/src/routers/common/placement.rs:7930-7972`).
#[test]
fn a_wire_with_no_worker_left_leaves_the_retry_no_endpoint() {
    let fleet = singles(&[GRPC, HTTP, OTHER]);
    // The program, for the send that makes its worker leave.
    let started: Arc<OnceLock<Program>> = Arc::new(OnceLock::new());
    let leaves: Weak<OnceLock<Program>> = Arc::downgrade(&started);
    let sends = move |worker: &dyn Worker| {
        let started = leaves.upgrade().expect("the case holds the program");
        let program = started.get().expect("the program is started");
        assert!(program.leave(worker.url()), "a worker that joined leaves");
        502
    };
    let (program, _loads, _task) = under(PolicySettings::RoundRobin, &fleet, sends);
    assert!(started.set(program).is_ok(), "the program is set once");
    let program = started.get().expect("the program is started");
    let answer = program.serve(&PLAIN);
    let no_endpoint = Unplaced::Shed {
        reason: ShedReason::NoEndpoints,
        emptied_by: "wires",
        retry_after_secs: 1,
    };
    assert_eq!(answer.sent.len(), 1, "{answer:?}");
    assert_eq!(answer.status, NO_WORKER);
    assert_eq!(answer.unplaced, Some(no_endpoint));
}

/// A pair's retry holds both sides to its wire, the wire of its first
/// attempt's prefill. Two prefills over gRPC, the first of which fails, and
/// two decodes: one over HTTP, one over gRPC. A first attempt takes either
/// decode, and a request meets the failing prefill beside the decode over
/// HTTP: the wire it keeps is its prefill's, and not that decode's. Each
/// retry goes to the second prefill and to the decode over gRPC, and never
/// to the one over HTTP.
///
/// The gateway holds a pinned pair's two sides to the retained wire: its
/// case `a_pinned_pair_keeps_the_retained_runtime_and_transport`
/// (`model_gateway/src/routers/common/placement.rs:4847-4912`).
#[test]
fn a_pairs_retry_holds_both_sides_to_its_wire() {
    let fleet = [
        dialed(&prefill(0), WorkerType::Prefill, GRPC),
        dialed(&prefill(1), WorkerType::Prefill, GRPC),
        dialed(&decode(0), WorkerType::Decode, HTTP),
        dialed(&decode(1), WorkerType::Decode, GRPC),
    ];
    let sends = failing_at(prefill(0));
    let (program, _loads, _task) = under_pairs(PolicySettings::RoundRobin, &fleet, sends);
    let answers: Vec<Answer> = (0..4).map(|_| program.serve(&PLAIN)).collect();
    assert!(answers.iter().all(|answer| answer.status == 200));
    let mut met = 0;
    for answer in &answers {
        if answer.sent[0] == prefill(0) {
            assert_eq!(answer.sent[2..], [prefill(1), decode(1)], "{answers:?}");
            met += 1;
        } else {
            assert_eq!(answer.sent.len(), 2, "{answers:?}");
        }
    }
    assert!(met > 0, "no request met the failing prefill: {answers:?}");
    // The wire kept is the prefill's: a first attempt failed at its prefill
    // beside the decode over HTTP, and its retry stayed on gRPC.
    let beside_http = |answer: &Answer| answer.sent[..2] == [prefill(0), decode(0)];
    assert!(answers.iter().any(beside_http), "{answers:?}");
}

/// A pair's retry in turn holds its second ask to its wire too. One
/// prefill and two decodes, the decode picked behind the prefill by the
/// fewest requests in flight: one decode on the prefill's wire, and one on
/// another wire that holds a request. The first attempt takes the idle
/// decode and fails at its prefill, so the retry is pinned. While the
/// retry's prefill runs, the decode on the wire takes three requests: the
/// second ask still names it, though the other decode is the lighter one
/// and was not tried.
///
/// The gateway states a pinned request's wire again at its second ask
/// (`RequestAgain`,
/// `model_gateway/src/routers/grpc/sequential_pd.rs:702-710`).
#[test]
fn a_pairs_retry_in_turn_holds_its_second_ask_to_its_wire() {
    let fleet = [
        dialed(&prefill(0), WorkerType::Prefill, GRPC),
        dialed(&decode(0), WorkerType::Decode, GRPC),
        dialed(&decode(1), WorkerType::Decode, HTTP),
    ];
    fleet[2].increment_load();
    let on_wire = Arc::clone(&fleet[1]);
    let prefills = AtomicUsize::new(0);
    let sends = move |worker: &dyn Worker| {
        if *worker.worker_type() != WorkerType::Prefill {
            return 200;
        }
        if prefills.fetch_add(1, Ordering::Relaxed) == 0 {
            return 502;
        }
        (0..3).for_each(|_| on_wire.increment_load());
        200
    };
    let fewest = PolicySettings::PowerOfTwo {
        load_metric: LoadMetricSetting::LeastRequests,
    };
    let document = picked(DecodePick::AfterPrefill, PolicySettings::RoundRobin, fewest);
    let (program, _loads, _task) = in_turn(&document, &fleet, sends);
    let answer = program.serve(&PLAIN);
    let expected = Answer {
        sent: vec![prefill(0), prefill(0), decode(0)],
        status: 200,
        unplaced: None,
    };
    assert_eq!(answer, expected);
}
