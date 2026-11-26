//! The route's own rule for a retry.
//!
//! A worker that answered a request itself takes no further attempt of it
//! ([`OWN_ANSWER`]). Behind such an answer the program states its rule with
//! each attempt ([`Program::serve`]), and the engine sends the request to no
//! worker the rule refuses.
//!
//! Each case serves requests over workers that fail each in its own way. It
//! reads where each attempt was sent and what the client read.
//!
//! A gateway keeps such a rule in its ledger of one request's attempts:
//! `AttemptLedger` (`model_gateway/src/routers/common/attempt_ledger.rs`).
//!
//! [`Program::serve`]: crate::program::Program::serve

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use smg_grand_central::{
    document::{DecodePick, PolicySettings},
    engine::{ShedReason, Unplaced},
};
use smg_worker::{
    identity::{basic::Worker, builder::BasicWorkerBuilder},
    model_card::ModelCard,
    test_support::no_health_check,
    worker::{WorkerStatus, WorkerType},
};

use crate::{
    demand::{singles, GRPC, HTTP},
    pair::{decode, fleet, in_turn, picked, prefill, PLAIN},
    program::{Answer, ATTEMPTS, NOT_REACHED, NO_WORKER, OWN_ANSWER, UNANSWERED},
    under, under_pairs, url, worker, workers, MODEL,
};

/// How many sends of `answer` went to an address that starts with `engine`.
fn times(answer: &Answer, engine: &str) -> usize {
    let sent = answer.sent.iter();
    sent.filter(|sent| sent.starts_with(engine)).count()
}

/// A send that the worker at `own` answers itself ([`OWN_ANSWER`]), the
/// worker at `idle` a 502, and every other worker a 200.
fn answering(own: String, idle: String) -> impl Fn(&dyn Worker) -> u16 + Send + Sync + 'static {
    move |worker: &dyn Worker| match worker.url() {
        address if address == own => OWN_ANSWER,
        address if address == idle => 502,
        _ => 200,
    }
}

/// A worker that answered a request itself takes no further attempt of it. Two
/// workers stand: the first answers every request itself, and the second is
/// not serving. Each request is sent to the first once and takes its other two
/// attempts at the second.
///
/// By the third attempt both workers were tried, so the engine picks among the
/// whole list. Where its pick is the first, the rule refuses it, and the call
/// decides again, inside itself, for the second.
///
/// The gateway's ledger refuses a retry the workers that answered the request
/// definitively: `AttemptLedger::admits`
/// (`model_gateway/src/routers/common/attempt_ledger.rs`). It hands its rule
/// to the selection behind such an answer alone:
/// `AttemptLedger::with_candidate_filter`. Its case is
/// `a_candidate_filter_is_handed_only_after_a_definitive_answer`.
#[test]
fn a_worker_that_answered_a_request_itself_takes_no_retry_of_it() {
    let fleet = workers(2);
    let sends = answering(url(0), url(1));
    let (program, _loads, _task) = under(PolicySettings::RoundRobin, &fleet, sends);
    for _ in 0..4 {
        let answer = program.serve(&PLAIN);
        let (status, sends) = (answer.status, answer.sent.len());
        assert_eq!((status, sends), (502, ATTEMPTS), "{answer:?}");
        assert_eq!(times(&answer, &url(0)), 1, "{answer:?}");
        assert_eq!(answer.unplaced, None);
    }
}

/// A request no other worker can take ends with the worker's own answer. A
/// lone worker answers the request itself. The rule of the next attempt then
/// takes no worker, and the engine says so by its shed under `admits`. The
/// client reads the worker's answer after one send. Two workers that both
/// answer so take one send each.
///
/// The gateway ends such a request with the answer as it is, where no
/// available worker that its ledger admits is left: `AttemptLedger::settle`
/// (`model_gateway/src/routers/common/attempt_ledger.rs`). Its case is
/// `a_definitive_answer_excludes_its_worker_and_ends_the_window_when_alone`.
///
/// The gateway ends the request ahead of another call. The program asks the
/// engine once more and reads its word.
#[test]
fn a_request_no_other_worker_can_take_ends_with_the_workers_own_answer() {
    let ruled_out = Unplaced::Shed {
        reason: ShedReason::NoEndpoints,
        emptied_by: "admits",
        retry_after_secs: 1,
        side: None,
    };
    for size in [1, 2] {
        let fleet = workers(size);
        let (program, _loads, _task) = under(PolicySettings::RoundRobin, &fleet, |_| OWN_ANSWER);
        let mut answer = program.serve(&PLAIN);
        answer.sent.sort();
        let expected = Answer {
            sent: (0..size).map(url).collect(),
            status: OWN_ANSWER,
            unplaced: Some(ruled_out),
        };
        assert_eq!(answer, expected, "{size} workers, each sent to once");
    }
}

/// A send no worker answered refuses nobody. The rule reads what a worker
/// answered, and a worker that was not reached answered nothing. So a lone
/// worker that is never reached takes each attempt of its request, and the
/// client reads the program's own status for such a send. So it is for a lone
/// worker that ran out of time at each attempt.
///
/// The gateway's ledger takes a worker's own answer alone as definitive:
/// `is_definitive_failure`
/// (`model_gateway/src/routers/common/attempt_ledger.rs`). A connect failure
/// or a timeout stays replayable, as that file's opening comment says. Its
/// case is `only_the_workers_own_500_is_definitive`.
#[test]
fn a_send_no_worker_answered_refuses_nobody() {
    for (answered, read) in [(UNANSWERED, NOT_REACHED), (504, 504)] {
        let lone = workers(1);
        let sends = move |_: &dyn Worker| answered;
        let (program, _loads, _task) = under(PolicySettings::RoundRobin, &lone, sends);
        let expected = Answer {
            sent: vec![url(0); ATTEMPTS],
            status: read,
            unplaced: None,
        };
        let answer = program.serve(&PLAIN);
        assert_eq!(answer, expected, "a send that answered {answered}");
    }
}

/// A shed for another cause at a retry is answered as at a first attempt. Two
/// workers stand: the first answers the request itself, and the second cannot
/// be sent to. The rule takes the second, so the engine does not shed the
/// retry under `admits`. It names the filter that left the request no worker.
/// The client reads the program's own status for a request no worker was named
/// for, and not the first worker's answer.
///
/// A retry held to its wire is answered as at a first attempt too, where the
/// one worker on that wire answered the request itself. The rule is asked of
/// the model's workers whole and takes the worker on the other wire, so the
/// engine sheds the retry under the wire's word.
///
/// A gateway differs here. It ends the request with the worker's own answer
/// where no available worker that its ledger admits is left, ahead of another
/// call: `AttemptLedger::settle`
/// (`model_gateway/src/routers/common/attempt_ledger.rs`).
#[test]
fn a_shed_for_another_cause_at_a_retry_is_answered_as_at_a_first_attempt() {
    let not_ready = workers(2);
    not_ready[1].set_status(WorkerStatus::NotReady);
    let causes = [(not_ready, "healthy"), (singles(&[GRPC, HTTP]), "wires")];
    for (fleet, emptied_by) in causes {
        let (program, _loads, _task) = under(PolicySettings::RoundRobin, &fleet, |_| OWN_ANSWER);
        let no_worker = Unplaced::Shed {
            reason: ShedReason::NoEndpoints,
            emptied_by,
            retry_after_secs: 1,
            side: None,
        };
        let answer = program.serve(&PLAIN);
        assert_eq!(answer.sent.len(), 1, "under {emptied_by}: {answer:?}");
        assert_eq!(answer.status, NO_WORKER, "under {emptied_by}");
        assert_eq!(answer.unplaced, Some(no_worker));
    }
}

/// One rank's answer stands for its engine. Two ranks of one engine stand
/// beside a worker of another, which is not serving. A rank answers each
/// request itself, and the request's other attempts go to the other engine. No
/// attempt goes to the engine's other rank, though the request never tried it
/// and the engine picks among the whole list by the third attempt.
///
/// The gateway's ledger keeps an engine by its base address: the `answered`
/// list of its `Ledger`
/// (`model_gateway/src/routers/common/attempt_ledger.rs`). Its case is
/// `a_ranks_answer_stands_for_its_engine`.
#[test]
fn a_ranks_answer_stands_for_its_engine() {
    const ENGINE: &str = "http://engine-0:8000";
    let rank = |rank: usize| -> Arc<dyn Worker> {
        let worker = <BasicWorkerBuilder>::new(ENGINE)
            .model(ModelCard::new(MODEL))
            .dp_config(rank, 2)
            .health_config(no_health_check())
            .build();
        worker.set_status(WorkerStatus::Ready);
        Arc::new(worker)
    };
    let fleet = [rank(0), rank(1), worker(2)];
    let sends = |worker: &dyn Worker| {
        if worker.base_url() == ENGINE {
            OWN_ANSWER
        } else {
            502
        }
    };
    let (program, _loads, _task) = under(PolicySettings::RoundRobin, &fleet, sends);
    for _ in 0..6 {
        let answer = program.serve(&PLAIN);
        let (status, sends) = (answer.status, answer.sent.len());
        assert_eq!((status, sends), (502, ATTEMPTS), "{answer:?}");
        assert_eq!(times(&answer, ENGINE), 1, "{answer:?}");
    }
}

/// For a pair the rule is the plain one: the side whose own answer it was is
/// refused, and its partner is not. Two prefills and one decode stand, sent at
/// once. The first prefill answers each request itself, the second is not
/// serving, and the decode serves.
///
/// Each request is sent to the first prefill once and takes its other attempts
/// at the second. Its decode goes out with every attempt: the partner of the
/// prefill that answered is refused nothing.
///
/// The gateway's ledger closes a request's window on the side that answered
/// and leaves its partner selectable. Its case is
/// `a_failed_side_excludes_its_worker_alone_and_ends_the_window_without_another_of_its_kind`
/// (`model_gateway/src/routers/common/attempt_ledger.rs`).
///
/// The gateway's pair route goes further where no pair is left without the
/// worker that answered, and refuses the partner instead: `settle_attempt`
/// (`model_gateway/src/routers/http/pd_router.rs`). The program keeps the
/// plain rule alone.
#[test]
fn a_pairs_side_that_answered_itself_is_refused_and_its_partner_is_not() {
    let fleet = fleet(2, 1);
    let sends = answering(prefill(0), prefill(1));
    let (program, _loads, _task) = under_pairs(PolicySettings::RoundRobin, &fleet, sends);
    for _ in 0..4 {
        let answer = program.serve(&PLAIN);
        let (status, sends) = (answer.status, answer.sent.len());
        assert_eq!((status, sends), (502, 2 * ATTEMPTS), "{answer:?}");
        assert_eq!(times(&answer, &prefill(0)), 1, "{answer:?}");
        assert_eq!(times(&answer, &decode(0)), ATTEMPTS, "{answer:?}");
    }
}

/// The rule holds a pair's decode as it holds its prefill. One prefill and two
/// decodes stand, sent at once. The prefill serves, the first decode answers
/// each request itself, and the second is not serving.
///
/// A decode's own answer behind a prefill that served is the decode's word on
/// the request. So each request is sent to the first decode once and takes its
/// other attempts at the second. Its prefill goes out with every attempt.
///
/// The gateway's pair route marks a decode's refusal as the decode's own
/// failure, and settles that side: `side_refused` and `decode_refused`
/// (`model_gateway/src/routers/http/pd_router.rs`).
#[test]
fn a_pairs_decode_that_answered_itself_is_refused_and_its_prefill_is_not() {
    let fleet = fleet(1, 2);
    let sends = answering(decode(0), decode(1));
    let (program, _loads, _task) = under_pairs(PolicySettings::RoundRobin, &fleet, sends);
    for _ in 0..4 {
        let answer = program.serve(&PLAIN);
        let (status, sends) = (answer.status, answer.sent.len());
        assert_eq!((status, sends), (502, 2 * ATTEMPTS), "{answer:?}");
        assert_eq!(times(&answer, &decode(0)), 1, "{answer:?}");
        assert_eq!(times(&answer, &prefill(0)), ATTEMPTS, "{answer:?}");
    }
}

/// An attempt reads one side's answer: the side that gave it its status, the
/// first that failed in the order of its sends.
///
/// Two prefills and one decode stand, sent at once. The first prefill is not
/// serving. The decode answers a 500 behind a prefill that failed, and serves
/// behind one that served.
///
/// A request that meets the first prefill is asked again, since its status is
/// that prefill's. The decode's 500 behind it is not the decode's own word on
/// the request. So the retry is sent to the second prefill and to the same
/// decode, which serves it. Read as its own answer, that 500 would leave the
/// retry no decode, and the client would read it.
///
/// The gateway's pair route settles the one side its response names, since a
/// side's status can carry its partner's part of the work: `own_failure` in
/// `settle_attempt` (`model_gateway/src/routers/http/pd_router.rs`).
#[test]
fn a_decodes_answer_behind_a_failed_prefill_refuses_nobody() {
    let fleet = fleet(2, 1);
    // Whether the prefill of the attempt under way failed.
    let failed = AtomicBool::new(false);
    let sends = move |worker: &dyn Worker| {
        if *worker.worker_type() == WorkerType::Prefill {
            let fails = worker.url() == prefill(0);
            failed.store(fails, Ordering::Relaxed);
            return if fails { 502 } else { 200 };
        }
        if failed.load(Ordering::Relaxed) {
            OWN_ANSWER
        } else {
            200
        }
    };
    let (program, _loads, _task) = under_pairs(PolicySettings::RoundRobin, &fleet, sends);
    let answers: Vec<Answer> = (0..4).map(|_| program.serve(&PLAIN)).collect();
    let retried = [prefill(0), decode(0), prefill(1), decode(0)];
    let mut met = 0;
    for answer in &answers {
        assert_eq!(answer.status, 200, "{answers:?}");
        if answer.sent[0] == prefill(0) {
            assert_eq!(answer.sent, retried, "{answers:?}");
            met += 1;
        } else {
            assert_eq!(answer.sent.len(), 2, "{answers:?}");
        }
    }
    assert!(met > 0, "no request met the failing prefill: {answers:?}");
}

/// A pair sent in turn keeps the rule at both of its calls, by the answer of
/// the side that gave the attempt its status.
///
/// One prefill and two decodes stand, the decode picked behind the prefill.
/// The prefill serves, the first decode answers each request itself, and the
/// second is not serving.
///
/// An attempt in turn is answered its decode's status where its prefill
/// served, so each request is asked again. It is sent to the first decode
/// once, to the second at its other attempts, and to its one prefill at every
/// attempt.
///
/// The gateway states a request's ledger again at a pair's second call:
/// `RequestAgain`, as `execute_sequential_pd` builds it
/// (`model_gateway/src/routers/grpc/sequential_pd.rs`).
#[test]
fn a_pair_in_turn_refuses_the_decode_that_answered_itself() {
    let fleet = fleet(1, 2);
    let sends = answering(decode(0), decode(1));
    let turns = PolicySettings::RoundRobin;
    let document = picked(DecodePick::AfterPrefill, turns.clone(), turns);
    let (program, _loads, _task) = in_turn(&document, &fleet, sends);
    for _ in 0..4 {
        let answer = program.serve(&PLAIN);
        let (status, sends) = (answer.status, answer.sent.len());
        assert_eq!((status, sends), (502, 2 * ATTEMPTS), "{answer:?}");
        assert_eq!(times(&answer, &decode(0)), 1, "{answer:?}");
        assert_eq!(times(&answer, &prefill(0)), ATTEMPTS, "{answer:?}");
    }
}

/// A prefill that answered itself in turn is refused too: where the prefill
/// fails, it is the side that gave the attempt its status.
///
/// Two prefills and one decode stand, the decode picked behind the prefill.
/// The first prefill answers each request itself, and the second is not
/// serving.
///
/// Each request is sent to the first prefill once and to the second at its
/// other attempts. Its decode is sent nothing, since a prefill that failed
/// sends no decode, and is refused nothing by a prefill's answer.
///
/// The gateway's ledger closes a request's window on the side that answered.
/// Its case is
/// `a_failed_side_excludes_its_worker_alone_and_ends_the_window_without_another_of_its_kind`
/// (`model_gateway/src/routers/common/attempt_ledger.rs`).
#[test]
fn a_pair_in_turn_refuses_the_prefill_that_answered_itself() {
    let fleet = fleet(2, 1);
    let sends = answering(prefill(0), prefill(1));
    let turns = PolicySettings::RoundRobin;
    let document = picked(DecodePick::AfterPrefill, turns.clone(), turns);
    let (program, _loads, _task) = in_turn(&document, &fleet, sends);
    for _ in 0..4 {
        let answer = program.serve(&PLAIN);
        let (status, sends) = (answer.status, answer.sent.len());
        assert_eq!((status, sends), (502, ATTEMPTS), "{answer:?}");
        assert_eq!(times(&answer, &prefill(0)), 1, "{answer:?}");
        assert_eq!(times(&answer, &decode(0)), 0, "{answer:?}");
    }
}
