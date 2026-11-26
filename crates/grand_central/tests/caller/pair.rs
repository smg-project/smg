//! A pair, served by the program over its two lists.
//!
//! The program routes to pairs of a prefill worker and a decode worker
//! ([`Plan::Pair`]): a request lends the two lists its generation holds, and
//! the one call names both of its workers. Each case serves requests as a
//! route would ([`Program::serve`]) and reads what the client was answered
//! and what the workers saw. Where the classic router has a case of its own
//! for the answer, the case here names it and its lines.
//!
//! A route sends a pair's decode at once, or behind its prefill's answer
//! ([`Program::in_turn`]). The last cases serve in turn: the program asks
//! the engine for the decode again behind that answer.
//!
//! [`Program::in_turn`]: crate::program::Program::in_turn
//!
//! [`Plan::Pair`]: crate::program::Plan::Pair
//! [`Program::serve`]: crate::program::Program::serve

use std::{
    collections::BTreeMap,
    future::Future,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use http::{HeaderMap, HeaderValue};
use smg_grand_central::{
    document::{
        AppliedSpec, Assignment, DecodePick, LoadMetricSetting, ModeSettings, PolicySettings,
        PolicySpec, RoutingSettings, ScoreSpec,
    },
    engine::{PairSide, ShedReason, Unplaced},
    stages::StageRegistry,
};
use smg_worker::{
    identity::basic::Worker,
    load::state::TestFeed,
    test_support::worker_with_model_aliases,
    worker::{WorkerStatus, WorkerType},
};
use tokio::time::sleep;

use crate::{
    program::{pair_settings, settings, Answer, Asked, Plan, Program, ATTEMPTS},
    served,
    single::{taken, Heard, Hears},
    started_as, under_pairs, MODEL,
};

/// A request for [`MODEL`] that carries no header and no item.
pub(crate) const PLAIN: Asked<'static> = Asked {
    model: MODEL,
    headers: None,
    items: &[],
    tokens: None,
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
/// (`model_gateway/src/routers/common/placement.rs:3598-3645`).
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
/// (`model_gateway/src/routers/common/placement.rs:3829-3877`, `:3883-3915`).
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

    // A status that says the request was not served now: the prefill steps
    // aside as a tried worker, where its own answer would state the rule.
    let failing = prefill(0);
    let sends = move |worker: &dyn Worker| if worker.url() == failing { 502 } else { 200 };
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
/// and the engine names none. Each refusal names the decode side, which
/// was left with nobody.
///
/// The classic router answers a pair whose only decode is not ready as the
/// decode role's unavailability: its case
/// `a_pair_whose_only_decode_is_not_ready_is_the_decode_roles_unavailability`
/// (`model_gateway/src/routers/common/placement.rs`). It answers a decode
/// pool with no worker for the decode's role too (`pair_pass` in that
/// file: its test of `decode_pool` for a role with no worker).
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
        side: Some(PairSide::Decode),
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
        side: Some(PairSide::Decode),
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
/// its retry is a request of its own to the stages. A program that sends
/// both sides at once says of no request that its decode is asked again,
/// so the engine books both sides at the one call under either word of the
/// document on when a pair's decode is picked.
///
/// The classic router's engine arm says each side by that side's guard: its
/// case `the_engine_hears_each_side_of_a_placed_pair_by_that_sides_guard`
/// (`model_gateway/src/routers/common/placement.rs:9718-9753`).
#[tokio::test(start_paused = true)]
async fn the_engine_hears_both_sides_of_a_pair() {
    for when in [DecodePick::WithPrefill, DecodePick::AfterPrefill] {
        let heard = Heard::default();
        let kept = Arc::clone(&heard);
        let mut stages = StageRegistry::builtin();
        let added = stages.scorer("hears", move || Box::new(Hears(Arc::clone(&kept))));
        added.expect("a free name");
        let settings = pair_settings(PolicySettings::RoundRobin);
        let document = settings.policy_document();
        let mut document = document.expect("the settings have a document");
        document.plans.pd_pair.decode = when;
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
        assert_eq!(taken(&heard), served, "the decode picked {when:?}");

        fails.store(true, Ordering::Relaxed);
        let answer = program.serve(&PLAIN);
        assert_eq!((answer.status, answer.sent.len()), (200, 4));
        let failed = said("completed Decode 502 ServerError");
        assert_eq!(taken(&heard), [failed, served].concat());
        task.abort();
    }
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

/// The document a gateway means that routes to pairs, its prefill side
/// under `prefill` and its decode side under `decode`, with the decode
/// picked `when`. The settings hold no word for when a pair's decode is
/// picked: it is the document's own (`plans.pd-pair.decode`).
pub(crate) fn picked(
    when: DecodePick,
    prefill: PolicySettings,
    decode: PolicySettings,
) -> PolicySpec {
    let settings = RoutingSettings {
        mode: ModeSettings::PrefillDecode {
            prefill_policy: None,
            decode_policy: Some(decode),
        },
        ..settings(prefill)
    };
    let document = settings.policy_document();
    let mut document = document.expect("the settings have a document");
    document.plans.pd_pair.decode = when;
    document
}

/// The program under `document` over `fleet`, with a route that sends a
/// pair's decode behind its prefill's answer.
pub(crate) fn in_turn(
    document: &PolicySpec,
    fleet: &[Arc<dyn Worker>],
    sends: impl Fn(&dyn Worker) -> u16 + Send + Sync + 'static,
) -> (Program, TestFeed, impl Future<Output = ()> + Send + 'static) {
    let spec = AppliedSpec::validate(document).expect("the document validates");
    let (program, loads, task) = started_as(Plan::Pair, spec, fleet, sends);
    (program.in_turn(), loads, task)
}

/// A decode asked again is picked as the decodes stand behind the prefill's
/// answer, and the request's count follows it. The first decision names the
/// decode that holds no request, and the program counts the request there
/// from the placement, under the request's routing key. While the prefill
/// runs that decode takes three more requests, and the second ask names the
/// other one: at the decode's send the request is counted on that one,
/// under its key still, and on the first no more. Under a document that
/// books both sides at once the decode never stands chosen: the program
/// makes no second ask, and the first decision's decode is sent, under the
/// guard it has.
///
/// The gateway asks so where the engine placed the pair: its cases
/// `a_decode_asked_again_moves_the_request_and_its_guard_to_the_decode_the_engine_takes`
/// and `a_decode_asked_again_that_takes_no_other_worker_leaves_the_first_standing`
/// (`model_gateway/src/routers/common/placement.rs:10288-10314`, `:10323-10379`).
/// It makes no ask for a decode booked with its prefill either
/// (`:1090-1096`): its case
/// `a_decode_booked_with_its_prefill_answers_the_ask_as_the_first_decode_standing`
/// (`:10529-10575`).
/// A pair the classic policies picked is sent the decode its one selection
/// named, and no decode is picked again (`DecodeBehindPrefill`,
/// `model_gateway/src/routers/grpc/sequential_pd.rs:595-603`).
#[test]
fn a_decode_asked_again_is_picked_as_the_decodes_stand_behind_the_prefills_answer() {
    const KEY: &str = "session-7";
    let fewest = PolicySettings::PowerOfTwo {
        load_metric: LoadMetricSetting::LeastRequests,
    };
    // When the document picks the decode, the decode that is sent, and what
    // the two decodes count at its send: each one's requests in flight, and
    // those of them under the request's key.
    let turns = [
        (DecodePick::AfterPrefill, 1, [(3, 0), (2, 1)]),
        (DecodePick::WithPrefill, 0, [(4, 1), (1, 0)]),
    ];
    for (when, sent_to, at_decode) in turns {
        let fleet = fleet(1, 2);
        let (idle, busy) = (Arc::clone(&fleet[1]), Arc::clone(&fleet[2]));
        busy.increment_load();
        // What the two decodes count at each send, read ahead of the three
        // requests the first one takes while the prefill runs.
        let counted = Arc::new(Mutex::new(Vec::new()));
        let counts = Arc::clone(&counted);
        let sends = move |worker: &dyn Worker| {
            let running = [&idle, &busy].map(|decode| {
                let under_key = decode.routing_key_inflight(KEY);
                (decode.load(), under_key)
            });
            counts.lock().expect("no case panicked").push(running);
            if *worker.worker_type() == WorkerType::Prefill {
                (0..3).for_each(|_| idle.increment_load());
            }
            200
        };
        // The prefill side is under a policy that reads a routing key, and
        // the document's view lists the key's header: the placement names
        // the key, and the program's guards count under it.
        let hashed = PolicySettings::ConsistentHashing;
        let mut document = picked(when, hashed, fewest.clone());
        document.view.headers.push("x-smg-routing-key".to_owned());
        let (program, _loads, _task) = in_turn(&document, &fleet, sends);
        let mut headers = HeaderMap::new();
        headers.insert("x-smg-routing-key", HeaderValue::from_static(KEY));
        let asked = Asked {
            headers: Some(&headers),
            ..PLAIN
        };
        let answer = program.serve(&asked);
        assert_eq!(
            answer,
            Answer {
                sent: vec![prefill(0), decode(sent_to)],
                status: 200,
                unplaced: None,
            },
            "the decode picked {when:?}"
        );
        let counted = counted.lock().expect("no case panicked");
        assert_eq!(
            counted.as_slice(),
            [[(1, 1), (1, 0)], at_decode],
            "at the prefill's send and at the decode's, picked {when:?}"
        );
    }
}

/// The request a second ask states says that its first decode counts it,
/// since the program's guard stands there across the ask: the engine then
/// reads that decode without the request. Under a rule that lets a worker
/// hold two requests of one key, one other request of the key on the first
/// decode leaves the request there. Read with the request's own count, that
/// decode would stand at the rule's bound and the request would move.
///
/// The gateway states the same word behind the guard it makes ahead of the
/// decode's send (`model_gateway/src/routers/common/placement.rs:1108-1112`,
/// over `LoadGuards::behind_prefill`,
/// `model_gateway/src/routers/grpc/context.rs:1116-1130`).
#[test]
fn a_decode_asked_again_is_read_without_the_request_its_guard_counts_there() {
    const KEY: &str = "session-7";
    let manual = PolicySettings::Manual {
        eviction_interval_secs: 60,
        max_idle_secs: 14_400,
        assignment_mode: Assignment::MinLoad,
    };
    let document = pair_settings(manual).policy_document();
    let mut document = document.expect("the settings have a document");
    document.plans.pd_pair.decode = DecodePick::AfterPrefill;
    for binding in &mut document.bindings {
        if let Some(rule) = &mut binding.sessions {
            rule.inflight_cap = 2;
        }
    }
    let fleet = fleet(1, 2);
    let (first, other) = (Arc::clone(&fleet[1]), &fleet[2]);
    // The key's first request takes the lighter decode.
    other.increment_load();
    let sends = move |worker: &dyn Worker| {
        if *worker.worker_type() == WorkerType::Prefill {
            // One other request of the key lands on the first decode while
            // the prefill runs.
            first.increment_load();
            first.increment_routing_key_load(KEY);
        }
        200
    };
    let (program, _loads, _task) = in_turn(&document, &fleet, sends);
    let mut headers = HeaderMap::new();
    headers.insert("x-smg-routing-key", HeaderValue::from_static(KEY));
    let asked = Asked {
        headers: Some(&headers),
        ..PLAIN
    };
    let answer = program.serve(&asked);
    assert_eq!(
        answer,
        Answer {
            sent: vec![prefill(0), decode(0)],
            status: 200,
            unplaced: None,
        },
        "the request stays on its first decode"
    );
}

/// A pair served in turn under the document the settings mean, which books
/// both sides at once, is served as a pair sent at once is: two sends a
/// request, the prefill's first, and each side takes its turn. Its decode
/// never stands chosen, so the program asks for none again.
///
/// The classic router names both sides of a pair in its one selection: its
/// case `a_pair_placement_names_both_sides_with_their_positions`
/// (`model_gateway/src/routers/common/placement.rs:3598-3645`).
#[test]
fn a_pair_in_turn_under_a_document_that_books_both_sides_takes_its_turns() {
    let fleet = fleet(2, 3);
    let turns = PolicySettings::RoundRobin;
    let document = picked(DecodePick::WithPrefill, turns.clone(), turns);
    let (program, _loads, _task) = in_turn(&document, &fleet, served);
    let answers: Vec<Answer> = (0..6).map(|_| program.serve(&PLAIN)).collect();
    for answer in &answers {
        assert_eq!((answer.status, answer.sent.len()), (200, 2), "{answer:?}");
        assert!(answer.sent[0].contains("prefill"), "{answer:?}");
        assert!(answer.sent[1].contains("decode"), "{answer:?}");
    }
    let (first, second) = (prefill(0), prefill(1));
    let decodes = [decode(0), decode(1), decode(2)];
    let mut expected = BTreeMap::from([(first.as_str(), 3), (second.as_str(), 3)]);
    expected.extend(decodes.iter().map(|decode| (decode.as_str(), 2)));
    assert_eq!(sends(&answers), expected);
}

/// A prefill that answered in turn holds the request no more: the program
/// lets its load guard go where it says the prefill's end, so the prefill
/// counts the request at its own send and no more at the decode's, whether
/// the decode was asked again or booked with it. A pair sent at once is
/// counted on its prefill through both sends.
///
/// The gateway's sequential dispatch frees the prefill once it is drained,
/// ahead of its decode
/// (`model_gateway/src/routers/grpc/sequential_pd.rs:919-921`).
#[test]
fn a_prefill_that_answered_in_turn_holds_the_request_no_more() {
    let turns = PolicySettings::RoundRobin;
    // Whether the route sends the pair in turn, when the document picks the
    // decode, and the prefill's requests in flight at the request's two
    // sends.
    let served = [
        (true, DecodePick::AfterPrefill, [1, 0]),
        (true, DecodePick::WithPrefill, [1, 0]),
        (false, DecodePick::WithPrefill, [1, 1]),
    ];
    for (by_turn, when, expected) in served {
        let fleet = fleet(1, 1);
        let one = Arc::clone(&fleet[0]);
        let counted = Arc::new(Mutex::new(Vec::new()));
        let counts = Arc::clone(&counted);
        let sends = move |_: &dyn Worker| {
            counts.lock().expect("no case panicked").push(one.load());
            200
        };
        let document = picked(when, turns.clone(), turns.clone());
        let spec = AppliedSpec::validate(&document).expect("the document validates");
        let (program, _loads, _task) = started_as(Plan::Pair, spec, &fleet, sends);
        let program = if by_turn { program.in_turn() } else { program };
        assert_eq!(program.serve(&PLAIN).sent, [prefill(0), decode(0)]);
        let counted = counted.lock().expect("no case panicked");
        assert_eq!(
            counted.as_slice(),
            expected,
            "in turn: {by_turn}, the decode picked {when:?}"
        );
    }
}

/// A prefill that fails in turn sends no decode: its attempt is one send.
/// The failure here is a 429: any status from 400 up is a prefill's
/// failure, and a decode goes out behind none of them.
/// The decode is counted from the placement, so it holds the request while
/// the prefill runs, and no more once the attempt is over. The retry states
/// both workers of the failed attempt as tried, so the prefill steps aside
/// and the one decode stays.
///
/// The gateway's retry names the engines of both sides of the failed
/// attempt's selection, whichever side failed and whether or not its
/// decode was sent: it reads the selection, and not what went out
/// (`WorkerSelection::engines`,
/// `model_gateway/src/routers/grpc/context.rs:1717-1721`, read at
/// `model_gateway/src/routers/grpc/pipeline.rs:849-861`).
#[test]
fn a_prefill_that_fails_in_turn_sends_no_decode() {
    let fleet = fleet(2, 1);
    let (failing, one) = (prefill(0), Arc::clone(&fleet[2]));
    // The decode's requests in flight at each send.
    let counted = Arc::new(Mutex::new(Vec::new()));
    let counts = Arc::clone(&counted);
    let sends = move |worker: &dyn Worker| {
        counts.lock().expect("no case panicked").push(one.load());
        if worker.url() == failing {
            429
        } else {
            200
        }
    };
    let turns = PolicySettings::RoundRobin;
    let document = picked(DecodePick::AfterPrefill, turns.clone(), turns);
    let (program, _loads, _task) = in_turn(&document, &fleet, sends);
    let answers: Vec<Answer> = (0..2).map(|_| program.serve(&PLAIN)).collect();
    assert!(answers.iter().all(|answer| answer.status == 200));
    let asked_again: Vec<&Answer> = answers
        .iter()
        .filter(|answer| answer.sent.len() == 3)
        .collect();
    let [answer] = asked_again[..] else {
        panic!("one of the two requests meets the failing prefill: {answers:?}");
    };
    assert_eq!(answer.sent, [prefill(0), prefill(1), decode(0)]);
    // Five sends, and the decode holds one request at each: the attempt's
    // own, the failed attempt's no more.
    let counted = counted.lock().expect("no case panicked");
    assert_eq!(counted.as_slice(), [1; 5]);
    assert_eq!(fleet[2].load(), 0, "every guard ended with its attempt");
}

/// A retry in turn states its tried workers at its second ask too. One
/// prefill and two decodes under the policy that takes the lighter worker:
/// the first decode is idle and the second holds a request, so the first
/// attempt goes to the first decode, which fails. The retry's second ask
/// names the second decode, since the first was tried: an ask that stated
/// no tried worker would take the lighter first one again.
///
/// The gateway states a request's tried workers again at its second ask
/// (`RequestAgain`,
/// `model_gateway/src/routers/grpc/sequential_pd.rs:702-710`).
#[test]
fn a_retry_in_turn_names_the_decode_it_did_not_try_at_its_second_ask() {
    let fewest = PolicySettings::PowerOfTwo {
        load_metric: LoadMetricSetting::LeastRequests,
    };
    let document = picked(DecodePick::AfterPrefill, PolicySettings::RoundRobin, fewest);
    let fleet = fleet(1, 2);
    fleet[2].increment_load();
    let failing = decode(0);
    let sends = move |worker: &dyn Worker| if worker.url() == failing { 502 } else { 200 };
    let (program, _loads, _task) = in_turn(&document, &fleet, sends);
    let answer = program.serve(&PLAIN);
    assert_eq!(
        answer,
        Answer {
            sent: vec![prefill(0), decode(0), prefill(0), decode(1)],
            status: 200,
            unplaced: None,
        }
    );
}

/// The engine hears a decode asked again at its ask, its send and its end:
/// the prefill is issued alone and dispatched, the decode is issued at the
/// second ask, ahead of the prefill's end, and dispatched where its send is
/// said, behind that end. The guard the program made at the placement told
/// the engine nothing. An ask whose own decision places nobody, since the
/// one decode is not ready then, is answered that decision's shed and
/// issues nothing: the first decision's decode is sent, and the engine
/// hears its send ahead of the prefill's end. Under a document that books
/// both sides at once, a prefill that fails leaves a decode that was never
/// sent: the engine hears it given up.
///
/// The gateway's stages hear a pair sent prefill first in this order: its
/// case `a_decode_decided_behind_its_prefill_is_heard_behind_the_prefills_answer`
/// (`model_gateway/src/routers/http/pd_router/tests/grand_central.rs:460-543`).
#[tokio::test(start_paused = true)]
async fn the_engine_hears_a_decode_asked_again_at_its_ask_its_send_and_its_end() {
    for when in [DecodePick::AfterPrefill, DecodePick::WithPrefill] {
        let heard = Heard::default();
        let kept = Arc::clone(&heard);
        let mut stages = StageRegistry::builtin();
        let added = stages.scorer("hears", move || Box::new(Hears(Arc::clone(&kept))));
        added.expect("a free name");
        let turns = PolicySettings::RoundRobin;
        let mut document = picked(when, turns.clone(), turns);
        // One stage of the document hears: the decode side's profile names it.
        let profile = document.profiles.get_mut("decode");
        let profile = profile.expect("the decode side's profile");
        profile.score.push(ScoreSpec::from("hears"));
        let spec = AppliedSpec::validate_in(&document, &stages, &[]);
        let spec = spec.expect("the stages hold the scorer");
        let fleet = fleet(1, 1);
        let fails = Arc::new(AtomicBool::new(false));
        // Whether the next prefill's send leaves the decode not ready.
        let stalls = Arc::new(AtomicBool::new(false));
        let (next, went) = (Arc::clone(&fails), Arc::clone(&heard));
        let (stalled, one) = (Arc::clone(&stalls), Arc::clone(&fleet[1]));
        let sends = move |worker: &dyn Worker| {
            let mut words = went.lock().expect("no case panicked");
            words.push("went out".to_owned());
            let prefill = *worker.worker_type() == WorkerType::Prefill;
            if prefill && stalled.swap(false, Ordering::Relaxed) {
                one.set_status(WorkerStatus::NotReady);
            }
            // A status that says the request was not served now: the one
            // prefill takes the retry, where its own answer would end it.
            if prefill && next.swap(false, Ordering::Relaxed) {
                502
            } else {
                200
            }
        };
        let (program, _loads, task) = started_as(Plan::Pair, spec, &fleet, sends);
        let program = program.in_turn();
        #[expect(
            clippy::disallowed_methods,
            reason = "the case holds the task's handle and ends the task"
        )]
        let task = tokio::spawn(task);
        // The task's first step reads the registry whole.
        sleep(Duration::from_millis(1)).await;
        let told = taken(&heard);
        assert_eq!(told.len(), 2, "each worker's join: {told:?}");

        assert_eq!(program.serve(&PLAIN).status, 200);
        let served: &[&str] = match when {
            DecodePick::AfterPrefill => &[
                "issued Prefill",
                "dispatched Prefill",
                "went out",
                "issued Decode",
                "completed Prefill 200 Ok",
                "dispatched Decode",
                "went out",
                "completed Decode 200 Ok",
            ],
            _ => &[
                "issued Prefill",
                "issued Decode",
                "dispatched Prefill",
                "went out",
                "dispatched Decode",
                "completed Prefill 200 Ok",
                "went out",
                "completed Decode 200 Ok",
            ],
        };
        assert_eq!(taken(&heard), served, "picked {when:?}");

        // A second ask that takes no decode: the decode is not ready behind
        // the prefill's send, so its own decision at the ask places nobody.
        // The ask is answered that decision's shed (`Unplaced::Shed`) and
        // issues nothing: an ask that named a decode it could not send to
        // (`Unplaced::NoWorkers`) would issue a side and give it up. The
        // first decision's decode is sent, and its send is heard ahead of
        // the prefill's end.
        if matches!(when, DecodePick::AfterPrefill) {
            stalls.store(true, Ordering::Relaxed);
            assert_eq!(program.serve(&PLAIN).status, 200);
            fleet[1].set_status(WorkerStatus::Ready);
            let none_taken = [
                "issued Prefill",
                "dispatched Prefill",
                "went out",
                "issued Decode",
                "dispatched Decode",
                "completed Prefill 200 Ok",
                "went out",
                "completed Decode 200 Ok",
            ];
            assert_eq!(taken(&heard), none_taken, "an ask that took none");
        }

        // A prefill that fails: no decode is sent in that attempt.
        fails.store(true, Ordering::Relaxed);
        let answer = program.serve(&PLAIN);
        assert_eq!((answer.status, answer.sent.len()), (200, 3));
        let failed: &[&str] = match when {
            DecodePick::AfterPrefill => &[
                "issued Prefill",
                "dispatched Prefill",
                "went out",
                "completed Prefill 502 ServerError",
            ],
            _ => &[
                "issued Prefill",
                "issued Decode",
                "dispatched Prefill",
                "went out",
                "completed Prefill 502 ServerError",
                "released Decode Abandoned",
            ],
        };
        assert_eq!(taken(&heard), [failed, served].concat(), "picked {when:?}");
        task.abort();
    }
}
