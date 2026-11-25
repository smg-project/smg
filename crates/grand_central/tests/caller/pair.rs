//! A pair, served by the program over its two lists.
//!
//! The program routes to pairs of a prefill worker and a decode worker
//! ([`Plan::Pair`]): a request lends the two lists its generation holds, and
//! the one call names both of its workers. Each case serves requests as a
//! route would ([`Program::serve`]) and reads what the client was answered
//! and what the workers saw. Where the classic router has a case of its own
//! for the answer, the case here names it and its lines.
//!
//! [`Plan::Pair`]: crate::program::Plan::Pair
//! [`Program::serve`]: crate::program::Program::serve

use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use smg_grand_central::{
    document::{AppliedSpec, PolicySettings, ScoreSpec},
    engine::{ShedReason, Unplaced},
    stages::StageRegistry,
};
use smg_worker::{
    identity::basic::Worker,
    test_support::worker_with_model_aliases,
    worker::{WorkerStatus, WorkerType},
};
use tokio::time::sleep;

use crate::{
    program::{pair_settings, Answer, Asked, Plan, ATTEMPTS},
    served,
    single::{taken, Heard, Hears},
    started_as, under_pairs, MODEL,
};

/// A request for [`MODEL`] that carries no header and no item.
pub(crate) const PLAIN: Asked<'static> = Asked {
    model: MODEL,
    headers: None,
    items: &[],
};

/// The address of the prefill worker at `at`.
pub(crate) fn prefill(at: usize) -> String {
    format!("http://prefill-{at}:8000")
}

/// The address of the decode worker at `at`.
pub(crate) fn decode(at: usize) -> String {
    format!("http://decode-{at}:8000")
}

/// A ready worker of `kind` that serves [`MODEL`], at `address`.
pub(crate) fn typed(address: &str, kind: WorkerType) -> Arc<dyn Worker> {
    let worker = worker_with_model_aliases(address, MODEL, &[], kind);
    worker.set_status(WorkerStatus::Ready);
    worker
}

/// `prefills` prefill workers and then `decodes` decode workers, each kind
/// in the order of its addresses.
pub(crate) fn fleet(prefills: usize, decodes: usize) -> Vec<Arc<dyn Worker>> {
    let prefills = (0..prefills).map(|at| typed(&prefill(at), WorkerType::Prefill));
    let decodes = (0..decodes).map(|at| typed(&decode(at), WorkerType::Decode));
    prefills.chain(decodes).collect()
}

/// How often each worker was sent to over `answers`, by its address.
pub(crate) fn sends<'a>(answers: impl IntoIterator<Item = &'a Answer>) -> BTreeMap<&'a str, u32> {
    let mut times = BTreeMap::new();
    for address in answers.into_iter().flat_map(|answer| &answer.sent) {
        *times.entry(address.as_str()).or_insert(0) += 1;
    }
    times
}

/// A pair's request is answered by one call that names a prefill and a
/// decode: one attempt is two sends, the prefill's first. Under
/// `round_robin` each side takes the workers of its own list in turn: six
/// requests over two prefills and three decodes, three a prefill and two a
/// decode.
///
/// The classic router names both sides of a pair with their places: its
/// case `a_pair_placement_names_both_sides_with_their_positions`
/// (`model_gateway/src/routers/common/placement.rs:3472-3519`).
#[test]
fn a_pair_names_a_prefill_and_a_decode_and_each_side_takes_its_turn() {
    let fleet = fleet(2, 3);
    let (program, _loads, _task) = under_pairs(PolicySettings::RoundRobin, &fleet, served);
    let answers: Vec<Answer> = (0..6).map(|_| program.serve(&PLAIN)).collect();
    for answer in &answers {
        assert_eq!((answer.status, answer.sent.len()), (200, 2), "{answer:?}");
        assert!(answer.sent[0].contains("prefill"), "{answer:?}");
        assert!(answer.sent[1].contains("decode"), "{answer:?}");
    }
    let times = sends(&answers);
    let (first, second) = (prefill(0), prefill(1));
    let decodes = [decode(0), decode(1), decode(2)];
    let mut expected = BTreeMap::from([(first.as_str(), 3), (second.as_str(), 3)]);
    expected.extend(decodes.iter().map(|decode| (decode.as_str(), 2)));
    assert_eq!(times, expected);
}

/// An attempt that a worker answers with a server error is asked again, and
/// the next attempt names both of its workers as tried: each side steps
/// aside from its own tried worker while another is left. A pair that
/// stands alone keeps its two workers for the retries, up to the program's
/// bound of attempts. A prefill's failure is the attempt's answer whatever
/// its decode answered.
///
/// The classic router lets a pair's tried workers step aside a side at a
/// time, and goes back to a tried worker that is the only one open: its
/// cases `a_pairs_retry_steps_aside_from_the_tried_engines_a_side_at_a_time`
/// and `a_pairs_retry_goes_back_to_a_tried_worker_that_is_the_only_one_open`
/// (`model_gateway/src/routers/common/placement.rs:3703-3751`, `:3757-3789`).
#[test]
fn a_pairs_retry_names_both_workers_it_tried_and_steps_aside_from_each() {
    let fleet = fleet(2, 2);
    let failing = decode(0);
    let sends = move |worker: &dyn Worker| if worker.url() == failing { 502 } else { 200 };
    let (program, _loads, _task) = under_pairs(PolicySettings::RoundRobin, &fleet, sends);
    let answers: Vec<Answer> = (0..4).map(|_| program.serve(&PLAIN)).collect();
    assert!(answers.iter().all(|answer| answer.status == 200));
    let mut retried = 0;
    for answer in &answers {
        if answer.sent[1] != decode(0) {
            assert_eq!(answer.sent.len(), 2, "{answer:?}");
            continue;
        }
        retried += 1;
        assert_eq!(answer.sent.len(), 4, "{answer:?}");
        assert_eq!(answer.sent[3], decode(1), "the other decode");
        assert_ne!(answer.sent[2], answer.sent[0], "the other prefill");
    }
    assert_eq!(retried, 2, "every other first attempt meets the decode");

    let lone = self::fleet(1, 1);
    let (program, _loads, _task) = under_pairs(PolicySettings::RoundRobin, &lone, |_| 502);
    let answer = program.serve(&PLAIN);
    let pair = [prefill(0), decode(0)];
    assert_eq!(
        answer,
        Answer {
            sent: (0..ATTEMPTS).flat_map(|_| pair.clone()).collect(),
            status: 502,
            unplaced: None,
        },
        "nothing else is left: the one pair takes the retries"
    );

    let failing = prefill(0);
    let sends = move |worker: &dyn Worker| if worker.url() == failing { 500 } else { 200 };
    let (program, _loads, _task) =
        under_pairs(PolicySettings::RoundRobin, &self::fleet(2, 1), sends);
    let answers: Vec<Answer> = (0..2).map(|_| program.serve(&PLAIN)).collect();
    assert!(answers.iter().all(|answer| answer.status == 200));
    let asked_again: Vec<&Answer> = answers
        .iter()
        .filter(|answer| answer.sent.len() == 4)
        .collect();
    let [answer] = asked_again[..] else {
        panic!("one of the two requests meets the failing prefill: {answers:?}");
    };
    let sent = [prefill(0), decode(0), prefill(1), decode(0)];
    assert_eq!(
        answer.sent, sent,
        "the one decode stays, the prefill steps aside"
    );
}

/// A worker that cannot be sent to takes no request on its side: a decode
/// that is not ready, and a prefill whose circuit breaker is open. Where no
/// decode can be sent to, the engine names no pair under the `healthy`
/// filter's name, and the program answers a 503. So it does for a model
/// that has prefills and no decode at all, where no filter emptied a list
/// and the engine names none.
///
/// The classic router answers a pair whose only decode is not ready as the
/// decode role's unavailability: its case
/// `a_pair_whose_only_decode_is_not_ready_is_the_decode_roles_unavailability`
/// (`model_gateway/src/routers/common/placement.rs:4512-4524`).
#[test]
fn a_side_that_cannot_be_sent_to_takes_no_request() {
    let fleet = fleet(2, 3);
    let (program, _loads, _task) = under_pairs(PolicySettings::RoundRobin, &fleet, served);
    let (broken, down) = (&fleet[1], &fleet[3]);
    down.set_status(WorkerStatus::NotReady);
    for _ in 0..64 {
        if !broken.circuit_breaker_can_execute() {
            break;
        }
        broken.record_circuit_breaker_outcome(false);
    }
    assert!(!broken.circuit_breaker_can_execute(), "the breaker opened");
    let answers: Vec<Answer> = (0..8).map(|_| program.serve(&PLAIN)).collect();
    let times = sends(&answers);
    let (only, first, last) = (prefill(0), decode(0), decode(2));
    let expected = BTreeMap::from([(only.as_str(), 8), (first.as_str(), 4), (last.as_str(), 4)]);
    assert_eq!(times, expected, "the workers that can be sent to");

    fleet[2].set_status(WorkerStatus::NotReady);
    fleet[4].set_status(WorkerStatus::NotReady);
    let answer = program.serve(&PLAIN);
    let no_decode = Unplaced::Shed {
        reason: ShedReason::NoEndpoints,
        emptied_by: "healthy",
        retry_after_secs: 1,
    };
    assert_eq!(
        answer,
        Answer {
            sent: Vec::new(),
            status: 503,
            unplaced: Some(no_decode),
        }
    );

    let prefills = self::fleet(2, 0);
    let (program, _loads, _task) = under_pairs(PolicySettings::RoundRobin, &prefills, served);
    let answer = program.serve(&PLAIN);
    let no_list = Unplaced::Shed {
        reason: ShedReason::NoEndpoints,
        emptied_by: "none",
        retry_after_secs: 1,
    };
    assert_eq!(
        answer,
        Answer {
            sent: Vec::new(),
            status: 503,
            unplaced: Some(no_list),
        }
    );
}

/// The engine hears both sides of a pair from the placement its program
/// keeps: a served pair is issued once, each side is dispatched before a
/// send goes out, and each side's end is said as its worker answered. An
/// attempt whose decode answers a server error ends so on that side, and
/// its retry is a request of its own to the stages.
///
/// The classic router's engine arm says each side by that side's guard: its
/// case `the_engine_hears_each_side_of_a_placed_pair_by_that_sides_guard`
/// (`model_gateway/src/routers/common/placement.rs:9367-9402`).
#[tokio::test(start_paused = true)]
async fn the_engine_hears_both_sides_of_a_pair() {
    let heard = Heard::default();
    let kept = Arc::clone(&heard);
    let mut stages = StageRegistry::builtin();
    let added = stages.scorer("hears", move || Box::new(Hears(Arc::clone(&kept))));
    added.expect("a free name");
    let settings = pair_settings(PolicySettings::RoundRobin);
    let document = settings.policy_document();
    let mut document = document.expect("the settings have a document");
    // One stage of the document hears: the decode side's profile names it.
    let profile = document.profiles.get_mut("decode");
    let profile = profile.expect("the decode side's profile");
    profile.score.push(ScoreSpec::from("hears"));
    let spec = AppliedSpec::validate_in(&document, &stages, &[]);
    let spec = spec.expect("the stages hold the scorer");
    let fails = Arc::new(AtomicBool::new(false));
    let (next, went) = (Arc::clone(&fails), Arc::clone(&heard));
    let sends = move |worker: &dyn Worker| {
        let mut words = went.lock().expect("no case panicked");
        words.push("went out".to_owned());
        let decode = *worker.worker_type() == WorkerType::Decode;
        if decode && next.swap(false, Ordering::Relaxed) {
            502
        } else {
            200
        }
    };
    let (program, _loads, task) = started_as(Plan::Pair, spec, &fleet(2, 2), sends);
    #[expect(
        clippy::disallowed_methods,
        reason = "the case holds the task's handle and ends the task"
    )]
    let task = tokio::spawn(task);
    // The task's first step reads the registry whole.
    sleep(Duration::from_millis(1)).await;
    let told = taken(&heard);
    assert_eq!(told.len(), 4, "each worker's join: {told:?}");

    assert_eq!(program.serve(&PLAIN).status, 200);
    let said = |decode: &'static str| {
        [
            "issued Prefill",
            "issued Decode",
            "dispatched Prefill",
            "dispatched Decode",
            "went out",
            "completed Prefill 200 Ok",
            "went out",
            decode,
        ]
    };
    let served = said("completed Decode 200 Ok");
    assert_eq!(taken(&heard), served);

    fails.store(true, Ordering::Relaxed);
    let answer = program.serve(&PLAIN);
    assert_eq!((answer.status, answer.sent.len()), (200, 4));
    let failed = said("completed Decode 502 ServerError");
    assert_eq!(taken(&heard), [failed, served].concat());
    task.abort();
}

/// A worker that left takes no later request on its side, and one that
/// joined takes them: each membership change publishes the next generation,
/// whose two lists the program cuts anew.
#[test]
fn a_pair_is_placed_over_the_generation_that_stands() {
    let fleet = fleet(1, 2);
    let (program, _loads, _task) = under_pairs(PolicySettings::RoundRobin, &fleet, served);
    let before = program.published(MODEL).expect("the model's generation");
    let answers: Vec<Answer> = (0..4).map(|_| program.serve(&PLAIN)).collect();
    let (first, second) = (decode(0), decode(1));
    let only = prefill(0);
    let expected = BTreeMap::from([
        (only.as_str(), 4),
        (first.as_str(), 2),
        (second.as_str(), 2),
    ]);
    assert_eq!(sends(&answers), expected);

    assert!(program.leave(&decode(0)), "a decode that joined leaves");
    let after = program.published(MODEL).expect("the model's generation");
    assert!(!Arc::ptr_eq(&before, &after), "the next generation");
    let answers: Vec<Answer> = (0..4).map(|_| program.serve(&PLAIN)).collect();
    let expected = BTreeMap::from([(only.as_str(), 4), (second.as_str(), 4)]);
    assert_eq!(sends(&answers), expected, "the decode that is left");

    let joined = prefill(1);
    assert!(program.join(typed(&joined, WorkerType::Prefill)));
    let answers: Vec<Answer> = (0..4).map(|_| program.serve(&PLAIN)).collect();
    let times = sends(&answers);
    assert_eq!(times.get(joined.as_str()), Some(&2), "{times:?}");
}
