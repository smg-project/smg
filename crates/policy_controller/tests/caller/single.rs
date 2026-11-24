//! A single request, served by the program over one worker list.
//!
//! Each case serves requests as a route would ([`Program::serve`]) and reads
//! what the client was answered and what the workers saw. Where the classic
//! router has an answer of its own for the case, the case names the lines
//! that give it.
//!
//! [`Program::serve`]: crate::program::Program::serve

use std::{
    collections::{BTreeMap, HashMap},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Barrier, Mutex,
    },
    thread,
    time::Duration,
};

use http::{HeaderMap, HeaderValue};
use smg_policy_controller::{
    document::{AppliedSpec, Assignment, PolicySettings, ScoreSpec},
    engine::{
        Candidate, Decision, RequestFacts, Scorer, ShedReason, Target, TieBand, Unplaced, WorkerId,
    },
    stages::{Lifecycle, StageRegistry},
    tracking::{EventSet, Lease, Membership, Outcome, ReleaseReason, Side},
};
use smg_worker::{
    identity::basic::Worker,
    load::state::one_rank_load,
    worker::{WorkerLoadResponse, WorkerStatus},
};
use tokio::{task::JoinHandle, time::sleep};

use crate::{
    program::{settings, Answer, Asked, Program, ATTEMPTS},
    served, started, under, url, worker, workers, MODEL,
};

/// A request for [`MODEL`] that carries no header.
const PLAIN: Asked<'static> = Asked {
    model: MODEL,
    headers: None,
};

/// How often each worker was sent to over `answers`, by its address.
fn sends<'a>(answers: impl IntoIterator<Item = &'a Answer>) -> BTreeMap<&'a str, u32> {
    let mut times = BTreeMap::new();
    for address in answers.into_iter().flat_map(|answer| &answer.sent) {
        *times.entry(address.as_str()).or_insert(0) += 1;
    }
    times
}

/// Plain requests under `round_robin` take the workers of their list in
/// turn: nine requests over three workers, three each, one send a request.
///
/// The classic router rotates the list it is given the same way
/// (`model_gateway/src/routers/common/placement.rs:4044-4071`).
#[test]
fn plain_requests_take_each_worker_in_turn() {
    let fleet = workers(3);
    let (program, _loads, _task) = under(PolicySettings::RoundRobin, &fleet, served);
    let answers: Vec<Answer> = (0..9).map(|_| program.serve(&PLAIN)).collect();
    for answer in &answers {
        assert_eq!((answer.status, answer.sent.len()), (200, 1));
    }
    let times = sends(&answers);
    assert_eq!(times.into_values().collect::<Vec<_>>(), [3, 3, 3]);
}

/// Callers that ask at the same moment do not pile on one worker: four
/// callers ask together, a barrier a round, over 256 idle workers, 64
/// rounds, and every worker takes exactly one request.
///
/// The classic router answers so
/// (`model_gateway/src/routers/common/placement.rs:4284-4321`).
#[test]
fn callers_that_ask_at_once_take_each_idle_worker_once() {
    const CALLERS: usize = 4;
    const ROUNDS: usize = 64;
    let fleet = workers(CALLERS * ROUNDS);
    let (program, _loads, _task) = under(PolicySettings::RoundRobin, &fleet, served);
    let round = Barrier::new(CALLERS);
    let answers: Vec<Answer> = thread::scope(|scope| {
        let caller = || {
            let serve = |_| {
                round.wait();
                program.serve(&PLAIN)
            };
            (0..ROUNDS).map(serve).collect::<Vec<_>>()
        };
        let callers: Vec<_> = (0..CALLERS).map(|_| scope.spawn(caller)).collect();
        let joined = callers.into_iter().map(|caller| caller.join());
        joined
            .flat_map(|answers| answers.expect("a caller ends"))
            .collect()
    });
    let times = sends(&answers);
    let piled = times.values().filter(|&&times| times != 1).count();
    assert_eq!(
        (times.len(), piled),
        (fleet.len(), 0),
        "workers that took a request, and those that took more than one"
    );
}

/// A send that a worker answers with a server error is asked again, and the
/// next attempt names that worker as tried: it goes to the other worker,
/// every time, and the rotation of first attempts goes on behind it. A list
/// of one worker keeps its worker for the retry, up to the program's bound
/// of sends. A send answered with a client error is the client's answer and
/// is not asked again.
///
/// The classic router lets the workers a request tried step aside while
/// another is left, and uses the list whole otherwise
/// (`model_gateway/src/routers/common/placement.rs:556-576`, its case
/// `:4218-4252`).
#[test]
fn a_retry_names_the_worker_it_tried_and_goes_to_another() {
    let fleet = workers(2);
    let failing = url(0);
    let sends = move |worker: &dyn Worker| if worker.url() == failing { 502 } else { 200 };
    let (program, _loads, _task) = under(PolicySettings::RoundRobin, &fleet, sends);
    let answers: Vec<Answer> = (0..4).map(|_| program.serve(&PLAIN)).collect();
    assert!(answers.iter().all(|answer| answer.status == 200));
    let sent: Vec<&[String]> = answers.iter().map(|answer| &answer.sent[..]).collect();
    // Every other first attempt meets the failing worker, and its retry
    // goes to the other one.
    let (served, retried) = ([url(1)], [url(0), url(1)]);
    assert_eq!(sent, [&served[..], &retried[..], &served[..], &retried[..]]);

    let lone = workers(1);
    let (program, _loads, _task) = under(PolicySettings::RoundRobin, &lone, |_| 502);
    let answer = program.serve(&PLAIN);
    assert_eq!(
        answer,
        Answer {
            sent: vec![url(0); ATTEMPTS],
            status: 502,
            unplaced: None,
        },
        "nothing else is left: the one worker takes the retries"
    );

    let (program, _loads, _task) = under(PolicySettings::RoundRobin, &workers(2), |_| 404);
    let answer = program.serve(&PLAIN);
    assert_eq!((answer.status, answer.sent.len()), (404, 1));
}

/// A send is asked again for the statuses a gateway's retry layer asks
/// again for, a 408 and a 429 among them, and for no other: a 501 or a 505
/// is the client's answer after one send.
///
/// The gateway's set is `is_retryable_status`
/// (`crates/external_router/src/retry.rs:10-19`).
#[test]
fn a_send_is_asked_again_for_the_statuses_a_gateway_asks_again_for() {
    let again = [408, 429, 500, 502, 503, 504].map(|status| (status, ATTEMPTS));
    let once = [400, 404, 501, 505].map(|status| (status, 1));
    for (status, sends) in again.into_iter().chain(once) {
        let fleet = workers(2);
        let (program, _loads, _task) = under(PolicySettings::RoundRobin, &fleet, move |_| status);
        let answer = program.serve(&PLAIN);
        assert_eq!(
            (answer.status, answer.sent.len()),
            (status, sends),
            "a send answered {status}"
        );
    }
}

/// A worker that cannot be sent to takes no request: one that is not ready,
/// and one whose circuit breaker is open. Where no worker of the list can
/// be sent to, the engine names none: it says the `healthy` filter left
/// nobody, with a hint to come back in a second, and the program answers a
/// 503.
///
/// The classic router keeps a worker that is available
/// (`model_gateway/src/routers/common/placement.rs:536-539`), and answers a
/// list with none a 503 with the same hint (`:577-580`, `:682`, `:703`).
#[test]
fn a_worker_that_cannot_be_sent_to_takes_no_request() {
    let fleet = workers(4);
    let (program, _loads, _task) = under(PolicySettings::RoundRobin, &fleet, served);
    fleet[1].set_status(WorkerStatus::NotReady);
    for _ in 0..64 {
        if !fleet[2].circuit_breaker_can_execute() {
            break;
        }
        fleet[2].record_circuit_breaker_outcome(false);
    }
    assert!(
        !fleet[2].circuit_breaker_can_execute(),
        "the breaker opened"
    );
    let answers: Vec<Answer> = (0..8).map(|_| program.serve(&PLAIN)).collect();
    let times = sends(&answers);
    let (first, last) = (url(0), url(3));
    let expected = BTreeMap::from([(first.as_str(), 4), (last.as_str(), 4)]);
    assert_eq!(times, expected, "the two workers that can be sent to");

    fleet[0].set_status(WorkerStatus::NotReady);
    fleet[3].set_status(WorkerStatus::NotReady);
    let answer = program.serve(&PLAIN);
    let no_worker = Unplaced::Shed {
        reason: ShedReason::NoEndpoints,
        emptied_by: "healthy",
        retry_after_secs: 1,
    };
    assert_eq!(
        answer,
        Answer {
            sent: Vec::new(),
            status: 503,
            unplaced: Some(no_worker),
        }
    );
}

/// Under `manual` the requests of one routing key stay on the worker the
/// key's first request took, while a lighter worker stands beside it, and
/// another key's first request goes to a lighter worker. The placement
/// names the key, so the worker counts a running request under it.
///
/// The classic router's `manual` policy pins a key to the worker its first
/// request took, and places a key it has not seen by the configured
/// assignment (`model_gateway/src/policies/manual.rs:292-326`, `:208-216`).
#[test]
fn the_requests_of_one_key_stay_on_its_worker() {
    const KEY: &str = "session-7";
    let fleet = workers(3);
    let manual = PolicySettings::Manual {
        eviction_interval_secs: 60,
        max_idle_secs: 14_400,
        assignment_mode: Assignment::MinLoad,
    };
    // What a worker counts under the key while a send to it runs.
    let counted = Arc::new(Mutex::new(Vec::new()));
    let counts = Arc::clone(&counted);
    let sends = move |worker: &dyn Worker| {
        let running = worker.routing_key_inflight(KEY);
        counts.lock().expect("no case panicked").push(running);
        200
    };
    let (program, _loads, _task) = under(manual, &fleet, sends);
    let keyed = |key: &str| {
        let mut headers = HeaderMap::new();
        let value = HeaderValue::from_str(key).expect("a header's value");
        headers.insert("x-smg-routing-key", value);
        headers
    };
    let (session, other) = (keyed(KEY), keyed("session-8"));
    let turn = Asked {
        headers: Some(&session),
        ..PLAIN
    };

    let first = program.serve(&turn);
    assert_eq!((first.status, first.sent.len()), (200, 1));
    let owner = fleet.iter().find(|worker| worker.url() == first.sent[0]);
    let owner = owner.expect("a worker of the fleet");
    // A key not seen yet would go to a lighter worker.
    owner.increment_load();
    for _ in 0..4 {
        assert_eq!(program.serve(&turn).sent, first.sent, "the key's worker");
    }
    let stranger = program.serve(&Asked {
        headers: Some(&other),
        ..PLAIN
    });
    assert_ne!(
        stranger.sent, first.sent,
        "a new key takes a lighter worker"
    );
    owner.decrement_load();

    let counted = counted.lock().expect("no case panicked");
    assert_eq!(counted.as_slice(), [1, 1, 1, 1, 1, 0]);
    assert_eq!(owner.routing_key_inflight(KEY), 0, "every request ended");
}

/// What a stage heard of the leases of its requests, in order.
type Heard = Arc<Mutex<Vec<String>>>;

/// A scorer added in code that scores every worker alike and takes every
/// word of a lease.
#[derive(Debug)]
struct Hears(Heard);

impl Hears {
    fn heard(&self, word: String) {
        self.0.lock().expect("no case panicked").push(word);
    }
}

impl Lifecycle for Hears {
    fn hears(&self) -> EventSet {
        EventSet::LEASE
    }

    fn on_decision_issued(&self, _: &Lease, _: &Decision) {
        self.heard("issued".to_owned());
    }

    fn on_dispatched(&self, _: &Lease, _: &Target, _: Side, _: u32) {
        self.heard("dispatched".to_owned());
    }

    fn on_completed(&self, _: &Lease, outcome: &Outcome) {
        self.heard(format!("completed {} {:?}", outcome.status, outcome.class));
    }

    fn on_released(&self, _: &Lease, reason: ReleaseReason) {
        self.heard(format!("released {reason:?}"));
    }

    fn on_membership(&self, change: &Membership<'_>) {
        match change {
            Membership::Added(worker) => self.heard(joined(*worker)),
            Membership::Removed(worker) => self.heard(left(*worker)),
            _ => {}
        }
    }
}

/// The word [`Hears`] keeps of a worker that joined.
fn joined(worker: WorkerId) -> String {
    format!("joined {}", worker.0)
}

/// The word [`Hears`] keeps of a worker that left.
fn left(worker: WorkerId) -> String {
    format!("left {}", worker.0)
}

impl Scorer for Hears {
    fn name(&self) -> &'static str {
        "hears"
    }

    fn score(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &[u32], out: &mut [f32]) -> TieBand {
        out.fill(1.0);
        TieBand::NORMALIZED
    }
}

/// A program under `round_robin` with [`Hears`] among its scorers, over two
/// workers: the program, what that stage heard, and the word that makes the
/// next send fail. A send is answered a 200, and a 502 once where the word
/// is set, and says `went out` among the stage's words as it goes.
///
/// The engine's task is spawned, and its handle is the last of the four.
/// The program told the engine of no worker: at its first step the task
/// read the program's registry, and the stage heard each of the two
/// workers join, under the id every decision names that worker by: the
/// fingerprint of the address the worker states.
async fn hearing() -> (Program, Heard, Arc<AtomicBool>, JoinHandle<()>) {
    let heard = Heard::default();
    let kept = Arc::clone(&heard);
    let mut stages = StageRegistry::builtin();
    let added = stages.scorer("hears", move || Box::new(Hears(Arc::clone(&kept))));
    added.expect("a free name");
    let document = settings(PolicySettings::RoundRobin).policy_document();
    let mut document = document.expect("the settings have a document");
    let profile = document.profiles.get_mut("round_robin");
    let profile = profile.expect("the preset's profile");
    profile.score.push(ScoreSpec::from("hears"));
    let spec = AppliedSpec::validate_in(&document, &stages, &[]);
    let spec = spec.expect("the stages hold the scorer");
    let fails = Arc::new(AtomicBool::new(false));
    let (next, went) = (Arc::clone(&fails), Arc::clone(&heard));
    let sends = move |_: &dyn Worker| {
        let mut words = went.lock().expect("no case panicked");
        words.push("went out".to_owned());
        if next.swap(false, Ordering::Relaxed) {
            502
        } else {
            200
        }
    };
    let (program, _loads, task) = started(spec, &workers(2), sends);
    #[expect(
        clippy::disallowed_methods,
        reason = "the case holds the task's handle and ends the task"
    )]
    let task = tokio::spawn(task);
    // The task's first step reads the registry whole.
    sleep(Duration::from_millis(1)).await;
    let mut fleet = [0, 1].map(|at| joined(WorkerId::from_address(&url(at))));
    fleet.sort_unstable();
    let mut told = taken(&heard);
    told.sort_unstable();
    assert_eq!(told, fleet, "each worker's join");
    (program, heard, fails, task)
}

/// The words `heard` holds, taken out of it.
fn taken(heard: &Heard) -> Vec<String> {
    std::mem::take(&mut *heard.lock().expect("no case panicked"))
}

/// The engine hears each request's end from the placement its program
/// keeps: a served request is issued, dispatched and completed, once each
/// and in that order, and the program says the dispatch before the send
/// goes out; a send answered a server error completes so, and its retry is
/// a request of its own to the stages. A send answered a client error
/// completes as one, and a 408 or a 504 as a timeout. A placement dropped
/// before its send is a release, one dropped behind it an end of class
/// cancelled, and a placement kept ends on another thread than the one
/// that placed it.
#[tokio::test(start_paused = true)]
async fn the_engine_hears_each_requests_end() {
    let (program, heard, fails, task) = hearing().await;
    assert_eq!(program.serve(&PLAIN).status, 200);
    let served = ["issued", "dispatched", "went out", "completed 200 Ok"];
    assert_eq!(taken(&heard), served);

    fails.store(true, Ordering::Relaxed);
    let answer = program.serve(&PLAIN);
    assert_eq!((answer.status, answer.sent.len()), (200, 2));
    assert_ne!(answer.sent[0], answer.sent[1], "the retry left the worker");
    let failed = [
        "issued",
        "dispatched",
        "went out",
        "completed 502 ServerError",
    ];
    assert_eq!(taken(&heard), [failed, served].concat());

    let unsent = program.place(&PLAIN, &[]).expect("a placement");
    drop(unsent);
    assert_eq!(taken(&heard), ["issued", "released Abandoned"]);

    let mut cut = program.place(&PLAIN, &[]).expect("a placement");
    cut.sent();
    drop(cut);
    let cancelled = taken(&heard);
    assert_eq!(cancelled[..2], ["issued", "dispatched"]);
    assert!(cancelled[2].ends_with("Cancelled"), "{cancelled:?}");
    assert_eq!(cancelled.len(), 3);

    let mut streamed = program.place(&PLAIN, &[]).expect("a placement");
    streamed.sent();
    let ended = thread::spawn(move || streamed.ended(200));
    ended.join().expect("the stream's thread ends");
    assert_eq!(taken(&heard), ["issued", "dispatched", "completed 200 Ok"]);

    for (status, class) in [(404, "ClientError"), (408, "Timeout"), (504, "Timeout")] {
        let mut ended = program.place(&PLAIN, &[]).expect("a placement");
        ended.sent();
        ended.ended(status);
        let completed = format!("completed {status} {class}");
        assert_eq!(taken(&heard), ["issued", "dispatched", completed.as_str()]);
    }
    task.abort();
}

/// A worker that leaves is forgotten under the address it states. The
/// registry finds a worker by an address written without its scheme, and
/// the stages know the worker by the whole address. The program tells the
/// engine nothing: the engine's task hears the removal from the registry
/// with the worker's own object, and tells the stages the worker under the
/// address that object states. A leave that finds no worker tells nothing.
///
/// The classic router's registry resolves such an address the same way
/// (`crates/worker/src/registry.rs:2032-2035`).
#[tokio::test(start_paused = true)]
async fn a_worker_that_leaves_is_forgotten_under_the_address_it_states() {
    let (program, heard, _fails, task) = hearing().await;
    let bare = url(1);
    let bare = bare.strip_prefix("http://").expect("an http address");
    assert!(program.leave(bare), "the registry finds the worker");
    sleep(Duration::from_millis(1)).await;
    assert_eq!(taken(&heard), [left(WorkerId::from_address(&url(1)))]);
    assert!(!program.leave(&url(1)), "it left already");
    sleep(Duration::from_millis(1)).await;
    assert!(taken(&heard).is_empty(), "nothing left, nothing told");
    task.abort();
}

/// A worker that joins later is told to the stages by the engine's task.
/// The program tells the engine no join: it handed its registry at its
/// start, and the join itself tells nothing until the task takes its
/// turn. Once the task has ended nothing is told any more.
#[tokio::test(start_paused = true)]
async fn a_worker_that_joins_later_is_told_by_the_engines_task() {
    let (program, heard, _fails, task) = hearing().await;
    assert!(program.join(worker(2)), "a free address joins");
    assert!(taken(&heard).is_empty(), "the join itself tells nothing");
    sleep(Duration::from_millis(1)).await;
    assert_eq!(taken(&heard), [joined(WorkerId::from_address(&url(2)))]);

    task.abort();
    assert!(task.await.is_err_and(|ended| ended.is_cancelled()));
    assert!(program.leave(&url(0)), "a worker that joined leaves");
    sleep(Duration::from_millis(1)).await;
    assert!(taken(&heard).is_empty(), "no task, nothing told");
}

/// One report a worker: the worker at `short` has no queue, every other a
/// long one.
fn reports(workers: usize, short: usize) -> HashMap<String, WorkerLoadResponse> {
    (0..workers)
        .map(|at| {
            let queued = if at == short { 0 } else { 90_000 };
            (url(at), one_rank_load(queued, 0.1, 1_000.0))
        })
        .collect()
}

/// A program lends no load snapshot with a request: it hands its load
/// monitor's channel at its start and spawns the engine's task. Under
/// `least_load` a request then goes to the worker the newest report shows
/// with no queue, and to another worker once the monitor publishes that
/// one so.
///
/// The classic router's `least_load` reads its load monitor's snapshot at
/// each decision too (`model_gateway/src/policies/least_load.rs:918`).
#[tokio::test(start_paused = true)]
async fn a_request_goes_by_the_newest_load_report() {
    let least_load = PolicySettings::LeastLoad {
        kv_pressure_weight: 0.15,
        mean_prefill_tokens: 1024,
        default_throughput: 2000.0,
    };
    let fleet = workers(4);
    let (program, loads, task) = under(least_load, &fleet, served);
    loads.publish(&reports(4, 2));
    #[expect(
        clippy::disallowed_methods,
        reason = "the case holds the task's handle and ends the task"
    )]
    let task = tokio::spawn(task);
    // The task's first step takes the channel and keeps its snapshot.
    sleep(Duration::from_millis(1)).await;
    for turn in 0..8 {
        assert_eq!(program.serve(&PLAIN).sent, [url(2)], "turn {turn}");
    }

    loads.publish(&reports(4, 0));
    sleep(Duration::from_millis(1)).await;
    for turn in 0..8 {
        assert_eq!(program.serve(&PLAIN).sent, [url(0)], "turn {turn}");
    }
    task.abort();
}

/// A worker that left takes no later request, and a worker that joined
/// takes its turn: each request is placed over the list of the generation
/// that stands, lent with that generation's own cell. A request under way
/// on a worker that leaves ends as any other.
#[test]
fn a_request_is_placed_over_the_generation_that_stands() {
    let fleet = workers(3);
    let (program, _loads, _task) = under(PolicySettings::RoundRobin, &fleet, served);
    let before = program.published(MODEL).expect("the model has workers");
    let answers: Vec<Answer> = (0..3).map(|_| program.serve(&PLAIN)).collect();
    assert_eq!(sends(&answers).len(), 3, "every worker took a request");

    // A request is under way on the worker that leaves.
    let leaving = url(1);
    let mut under_way = loop {
        let attempt = program.place(&PLAIN, &[]).expect("a placement");
        if attempt.placement.worker.url() == leaving {
            break attempt;
        }
    };
    under_way.sent();
    assert!(program.leave(&leaving), "a worker that joined leaves");
    let after = program.published(MODEL).expect("the model has workers");
    assert!(!Arc::ptr_eq(&before, &after), "the next generation stands");
    let answers: Vec<Answer> = (0..6).map(|_| program.serve(&PLAIN)).collect();
    let (first, last) = (url(0), url(2));
    let expected = BTreeMap::from([(first.as_str(), 3), (last.as_str(), 3)]);
    assert_eq!(sends(&answers), expected, "the worker that left takes none");
    under_way.ended(200);
    drop(under_way);
    assert_eq!(fleet[1].load(), 0, "its request ended");

    assert!(program.join(worker(3)), "a free address joins");
    let answers: Vec<Answer> = (0..9).map(|_| program.serve(&PLAIN)).collect();
    let joined = url(3);
    let expected = BTreeMap::from([
        (first.as_str(), 3),
        (last.as_str(), 3),
        (joined.as_str(), 3),
    ]);
    assert_eq!(
        sends(&answers),
        expected,
        "the worker that joined takes its turn"
    );
}
