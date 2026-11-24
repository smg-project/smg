//! The cases of a pair's decode decided again on its placement
//! ([`Placement::decode_again`]).

use super::*;
use crate::{
    engine::{decision_engine::DecodeRefused, policy_engine::encode::EncodeLists, view::Session},
    tracking::lease::TransportClass,
};

/// A pair's request whose caller asks its decode again.
fn later(pair: PairLists<'_>) -> PlaceRequest<'_> {
    PlaceRequest {
        decode_again: true,
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    }
}

/// The end of a side that was served, as its caller says it: the placement
/// writes the side and the worker.
const SERVED: Outcome = Outcome {
    side: Side::Single,
    served: WorkerId(0),
    status: 200,
    class: TransportClass::Ok,
    ttft_micros: 0,
    tokens_in: 0,
    tokens_out: 0,
    duration_micros: 0,
};

/// One word a stage heard of a lease, as [`InOrder`] writes it.
fn word(heard: &str, worker: WorkerId) -> (String, WorkerId) {
    (heard.to_owned(), worker)
}

/// The stock pair document with [`InOrder`] for both sides' pick, picking
/// a pair's decode after its prefill, and what each picker heard.
fn asked_again() -> (&'static PolicyEngine, HeardBy) {
    paired_picking(false, DecodePick::AfterPrefill)
}

/// A decode call that takes no decode, answered. What its caller reads of
/// the placement is read ahead of the call and behind it, and stands: its
/// two workers, their ids, whether an event is owed, and all the placement
/// shows of the decision it holds.
fn untaken(lists: &Lists, placement: &mut Placement<'_>, request: &PlaceRequest<'_>) -> Unplaced {
    let read = |placement: &Placement<'_>| {
        let ids = (placement.id(), placement.decode_id());
        let shown = format!("{placement:?}");
        (lists.places(placement), ids, placement.owes_events(), shown)
    };
    let before = read(placement);
    let answer = placement.decode_again(request);
    assert_eq!(read(placement), before, "the placement stands as it stood");
    answer.expect_err("no decode is taken")
}

/// A picker that names every worker the filters keep, in the list's order,
/// and keeps what each issue and each release of a lease states, as its
/// reader takes it from the lease: a size, or a side with its session.
#[derive(Debug)]
struct Stated<T>(Arc<Mutex<Vec<(&'static str, T)>>>, fn(&Lease) -> T);

impl<T: fmt::Debug + Send> Lifecycle for Stated<T> {
    fn hears(&self) -> EventSet {
        EventSet::LEASE
    }

    fn on_decision_issued(&self, lease: &Lease, _: &Decision) {
        self.0.lock().push(("issued", (self.1)(lease)));
    }

    fn on_released(&self, lease: &Lease, _: ReleaseReason) {
        self.0.lock().push(("released", (self.1)(lease)));
    }
}

impl<T: fmt::Debug + Send> Picker for Stated<T> {
    fn name(&self) -> &'static str {
        "stated"
    }

    fn pick(
        &self,
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        _: &[f32],
        _: TieBand,
        out: &mut Targets,
    ) {
        push_drawn(candidates, kept, out);
    }
}

/// A decode decided again is picked on the records of its own call: under
/// a decode side that reads load, the decode that holds the fewest
/// requests then, whatever the first decision named. The placement names
/// it by its worker and by its id, and the prefill that was sent stays.
///
/// The classic router's sequential dispatch sends a pair's decode once the
/// prefill's answer is on its request, to the decode worker its one
/// selection named ahead of the prefill
/// (`model_gateway/src/routers/grpc/sequential_pd.rs:948-957`): it picks no
/// decode again.
#[test]
fn a_decode_decided_again_is_picked_on_the_records_of_its_own_call() {
    let mut document = PolicySpec::pair_presets("round_robin", "least_load").expect("presets");
    document.generation = 1;
    document.plans.pd_pair.decode = DecodePick::AfterPrefill;
    let engine = hosted(engine_of(&document));
    let lists = Lists::of(2, 2);
    let classes = lists.every();
    let request = later(lists.under(&classes, &Open));
    for first in [0, 1] {
        // At the first call the other decode holds a request. Behind the
        // prefill's send the two change places.
        let (named, other) = (&lists.decodes[first], &lists.decodes[1 - first]);
        other.increment_load();
        let mut placement = engine.place(&request).expect("a placement");
        let (prefill, decode) = lists.places(&placement);
        assert_eq!(decode, first, "the decode that holds no request");
        placement.sent(Side::Prefill);
        other.decrement_load();
        named.increment_load();
        assert_eq!(placement.decode_again(&request), Ok(()));
        assert_eq!(lists.places(&placement), (prefill, 1 - first));
        let ids = (lists.prefill_id(prefill), Some(lists.decode_id(1 - first)));
        assert_eq!((placement.id(), placement.decode_id()), ids);
        named.decrement_load();
    }
}

/// The decodes a second decision names are asked best first, and the first
/// that can be sent to is taken: one that is not ready, or that its
/// circuit breaker holds back, is passed over. The stages hear the side's
/// issue at the call, on the decision's first decode, and its dispatch on
/// the decode taken.
#[test]
fn a_decode_decided_again_passes_over_a_decode_that_cannot_be_sent_to() {
    for by_breaker in [false, true] {
        let (engine, heard) = asked_again();
        let lists = Lists::of(1, 3);
        let classes = lists.every();
        let request = later(lists.under(&classes, &Open));
        let mut placement = engine.place(&request).expect("a placement");
        assert_eq!(lists.places(&placement), (0, 0));
        placement.sent(Side::Prefill);

        let first = &lists.decodes[0];
        if by_breaker {
            for _ in 0..64 {
                if !first.circuit_breaker_can_execute() {
                    break;
                }
                first.record_circuit_breaker_outcome(false);
            }
            assert!(first.routing_state().healthy && !ready(first.as_ref()));
        } else {
            first.set_status(WorkerStatus::NotReady);
        }
        assert_eq!(placement.decode_again(&request), Ok(()));
        assert_eq!(lists.places(&placement), (0, 1));
        assert_eq!(placement.decode_id(), Some(lists.decode_id(1)));
        placement.sent(Side::Decode);
        let (prefill, passed, taken) =
            (lists.prefill_id(0), lists.decode_id(0), lists.decode_id(1));
        assert_eq!(
            said(&heard),
            [
                word("issued at 0", prefill),
                word("dispatched at 0", prefill),
                word("issued at 0", passed),
                word("dispatched at 1", taken),
            ],
            "by the breaker: {by_breaker}"
        );
    }
}

/// A second decision none of whose decodes can be sent to answers that the
/// request has no worker. The side the call issued is given up by name,
/// once, and the placement stands as it stood: on the first decision's
/// decode, still chosen. Each way on is then open to its caller: it asks
/// again and a decode is taken, it sends the first decision's decode, or
/// it says the prefill's end and lets the decode go.
#[test]
fn a_decode_call_none_of_whose_decodes_holds_gives_its_side_up_and_may_be_asked_again() {
    let (engine, heard) = asked_again();
    let lists = Lists::of(1, 2);
    let classes = lists.every();
    let request = later(lists.under(&classes, &Open));
    let (prefill, lead, chosen) = (lists.prefill_id(0), lists.decode_id(0), lists.decode_id(1));
    // A placement on the second decode, whose decode call found nobody to
    // send to, and the words the stages heard of it from its first ask.
    let missed = || {
        let from = said(&heard).len();
        lists.decodes[0].set_status(WorkerStatus::NotReady);
        lists.decodes[1].set_status(WorkerStatus::Ready);
        let mut placement = engine.place(&request).expect("a placement");
        assert_eq!(lists.places(&placement), (0, 1));
        placement.sent(Side::Prefill);
        lists.decodes[1].set_status(WorkerStatus::NotReady);
        let answer = untaken(&lists, &mut placement, &request);
        assert_eq!(answer, Unplaced::NoWorkers);
        assert_eq!(
            said(&heard)[from..],
            [
                word("issued at 0", prefill),
                word("dispatched at 1", prefill),
                word("issued at 0", lead),
                word("released CommitFailed", lead),
            ]
        );
        placement
    };

    // Asked again, once a decode can be sent to: the call is not refused.
    let mut asked = missed();
    lists.decodes[0].set_status(WorkerStatus::Ready);
    let from = said(&heard).len();
    assert_eq!(asked.decode_again(&request), Ok(()));
    assert_eq!(lists.places(&asked), (0, 0));
    asked.sent(Side::Decode);
    assert_eq!(
        said(&heard)[from..],
        [word("issued at 0", lead), word("dispatched at 0", lead)]
    );
    drop(asked);

    // The first decision's decode, sent: it is issued at its send.
    let mut sent = missed();
    lists.decodes[1].set_status(WorkerStatus::Ready);
    let from = said(&heard).len();
    sent.sent(Side::Decode);
    assert_eq!(lists.places(&sent), (0, 1));
    assert_eq!(
        said(&heard)[from..],
        [word("issued at 1", chosen), word("dispatched at 1", chosen)]
    );
    drop(sent);

    // The prefill's end, said: the chosen decode leaves with it, and
    // nothing is said of a decode that was booked nowhere.
    let mut ended = missed();
    let from = said(&heard).len();
    ended.ended(Side::Prefill, SERVED);
    assert!(!ended.owes_events());
    drop(ended);
    assert_eq!(said(&heard)[from..], [word("completed Ok", prefill)]);
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A decode call that takes no decode puts back the size the pair's words
/// state. The call issues its side under the size its own request states,
/// and gives it up under that size. The first decision's decode, sent
/// behind it, is issued under the first call's size again.
#[test]
fn a_decode_call_that_takes_no_decode_leaves_the_first_calls_size_stated() {
    let sizes = Arc::new(Mutex::new(Vec::new()));
    let kept = Arc::clone(&sizes);
    let mut stages = StageRegistry::builtin();
    let size = |lease: &Lease| lease.tokens;
    let added = stages.picker("stated", move || Box::new(Stated(Arc::clone(&kept), size)));
    added.expect("a free name");
    let mut document = PolicySpec::pair_presets("round_robin", "round_robin").expect("presets");
    document.generation = 1;
    document.plans.pd_pair.decode = DecodePick::AfterPrefill;
    // A decode that is not ready stays a candidate of the decision. The
    // decode side's picker hears every word of the pair's lease.
    for (name, profile) in &mut document.profiles {
        let healthy = FilterSpec::Name("healthy".to_owned());
        profile.filter.retain(|filter| *filter != healthy);
        if name == "decode" {
            profile.pick = "stated".into();
        }
    }
    let spec = AppliedSpec::validate_in(&document, &stages, &[]);
    let engine = hosted(PolicyEngine::new(spec.expect("the stages hold the picker")));

    let lists = Lists::of(1, 1);
    let classes = lists.every();
    let sized = |tokens: &'static [u32]| PlaceRequest {
        prompt: Prompt {
            tokens: Some(tokens),
            ..Prompt::NONE
        },
        ..later(lists.under(&classes, &Open))
    };
    let mut placement = engine.place(&sized(&[0; 8])).expect("a placement");
    placement.sent(Side::Prefill);
    lists.decodes[0].set_status(WorkerStatus::NotReady);
    let answer = untaken(&lists, &mut placement, &sized(&[0; 64]));
    assert_eq!(answer, Unplaced::NoWorkers);
    lists.decodes[0].set_status(WorkerStatus::Ready);
    placement.sent(Side::Decode);
    assert_eq!(
        *sizes.lock(),
        [
            ("issued", 8),
            ("issued", 64),
            ("released", 64),
            ("issued", 8)
        ]
    );
}

/// A decode call that takes no decode puts back the session the decode
/// side's words state, as it puts back their size. A request's own id is
/// its routing value here, so the call issues its side under the session
/// its own request states, and gives it up under that session. The first
/// decision's decode, sent behind it, is issued under the first call's
/// session again.
#[test]
fn a_decode_call_that_takes_no_decode_leaves_the_first_calls_session_stated() {
    let sessions = Arc::new(Mutex::new(Vec::new()));
    let kept = Arc::clone(&sessions);
    let mut stages = StageRegistry::builtin();
    let session = |lease: &Lease| (lease.side, lease.session);
    let added = stages.picker("stated", move || {
        Box::new(Stated(Arc::clone(&kept), session))
    });
    added.expect("a free name");
    let mut document = PolicySpec::pair_presets("round_robin", "round_robin").expect("presets");
    document.generation = 1;
    document.plans.pd_pair.decode = DecodePick::AfterPrefill;
    let rid = KeySourceSpec {
        rid: true,
        ..KeySourceSpec::default()
    };
    document.bindings[0].routing_key = Some(RoutingKeySpec {
        sources: vec![rid],
        strip_lineage: false,
    });
    // A decode that is not ready stays a candidate of the decision. The
    // decode side's picker hears every word of the pair's lease.
    for (name, profile) in &mut document.profiles {
        let healthy = FilterSpec::Name("healthy".to_owned());
        profile.filter.retain(|filter| *filter != healthy);
        if name == "decode" {
            profile.pick = "stated".into();
        }
    }
    let spec = AppliedSpec::validate_in(&document, &stages, &[]);
    let engine = hosted(PolicyEngine::new(spec.expect("the stages hold the picker")));

    let lists = Lists::of(1, 1);
    let classes = lists.every();
    let keyed = |id: &'static str| PlaceRequest {
        request_id: Some(id),
        ..later(lists.under(&classes, &Open))
    };
    // The decode side's words among the ones heard from `from` on.
    let of_the_decode = |from: usize| -> Vec<(&'static str, Session)> {
        let heard = sessions.lock();
        let decode = heard[from..]
            .iter()
            .filter(|(_, stated)| stated.0 == Side::Decode);
        decode
            .map(|&(word, (_, session))| (word, session))
            .collect()
    };
    // The session a request's decode side states where its first
    // decision's decode is sent with no decode call.
    let stated = |id: &'static str| {
        let from = sessions.lock().len();
        let mut placement = engine.place(&keyed(id)).expect("a placement");
        placement.sent(Side::Prefill);
        placement.sent(Side::Decode);
        let words = of_the_decode(from);
        let [("issued", session)] = words[..] else {
            panic!("one issue of the decode side: {words:?}");
        };
        session
    };
    let (first, second) = (stated("first"), stated("second"));
    assert_ne!(first, Session::NONE, "a keyed request states a session");
    assert_ne!(first, second, "each request its own session");

    let from = sessions.lock().len();
    let mut placement = engine.place(&keyed("first")).expect("a placement");
    placement.sent(Side::Prefill);
    lists.decodes[0].set_status(WorkerStatus::NotReady);
    let answer = untaken(&lists, &mut placement, &keyed("second"));
    assert_eq!(answer, Unplaced::NoWorkers);
    lists.decodes[0].set_status(WorkerStatus::Ready);
    placement.sent(Side::Decode);
    assert_eq!(
        of_the_decode(from),
        [("issued", second), ("released", second), ("issued", first)]
    );
}

/// A route's rule that panics as a decode decided again asks it of a
/// decode passes its panic to the call's caller, and the side the call
/// issued is given up by name as the panic passes, once. The placement
/// stands as it stood before the call, on the first decision's decode,
/// still chosen. Asked again under a rule that holds, it takes a decode.
/// Dropped behind the panic, it says its prefill's end and nothing of a
/// decode that was booked nowhere.
#[test]
fn a_rule_that_panics_inside_a_decode_decided_again_loses_no_lease() {
    let (engine, heard) = asked_again();
    let lists = Lists::of(1, 1);
    let classes = lists.every();
    let fails = Cell::new(false);
    let admits = |_: &dyn Worker| -> bool {
        assert!(!fails.get(), "the route's rule failed");
        true
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..later(lists.under(&classes, &Open))
    };
    let (prefill, decode) = (lists.prefill_id(0), lists.decode_id(0));
    // A placement whose decode call met the rule's panic. What its caller
    // reads of it is read ahead of the call and behind it, and stands.
    let met = || {
        let from = said(&heard).len();
        let mut placement = engine.place(&request).expect("a placement");
        placement.sent(Side::Prefill);
        let read = |placement: &Placement<'_>| {
            let ids = (placement.id(), placement.decode_id());
            let shown = format!("{placement:?}");
            (lists.places(placement), ids, placement.owes_events(), shown)
        };
        let before = read(&placement);
        fails.set(true);
        let asked = catch_unwind(AssertUnwindSafe(|| {
            placement.decode_again(&request).is_ok()
        }));
        fails.set(false);
        assert!(asked.is_err(), "the rule's panic reaches the caller");
        assert_eq!(read(&placement), before, "the placement stands as it stood");
        assert_eq!(
            said(&heard)[from..],
            [
                word("issued at 0", prefill),
                word("dispatched at 0", prefill),
                word("issued at 0", decode),
                word("released CommitFailed", decode),
            ]
        );
        placement
    };

    // Asked again under a rule that holds: the call is not refused.
    let mut asked = met();
    let from = said(&heard).len();
    assert_eq!(asked.decode_again(&request), Ok(()));
    asked.sent(Side::Decode);
    asked.ended(Side::Prefill, SERVED);
    asked.ended(Side::Decode, SERVED);
    assert!(!asked.owes_events());
    assert_eq!(
        said(&heard)[from..],
        [
            word("issued at 0", decode),
            word("dispatched at 0", decode),
            word("completed Ok", prefill),
            word("completed Ok", decode),
        ]
    );
    drop(asked);

    // Dropped behind the panic: the prefill's end as cancelled, and
    // nothing of a decode that stands chosen.
    let dropped = met();
    let from = said(&heard).len();
    drop(dropped);
    assert_eq!(said(&heard)[from..], [word("completed Cancelled", prefill)]);
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A decode's decision that panics does not reach the call's caller: the
/// engine holds the panic, counts it and refuses the call. Nothing was
/// issued for it, so the stages hear nothing, and the first decision's
/// decode still stands chosen.
#[test]
fn a_decode_decision_that_panics_is_refused_and_leaves_the_first_decode_chosen() {
    let (engine, heard) = asked_again();
    let lists = Lists::of(1, 1);
    let classes = lists.every();
    let request = later(lists.under(&classes, &Open));
    let mut placement = engine.place(&request).expect("a placement");
    placement.sent(Side::Prefill);
    let at_send = said(&heard);

    let failing = Gates(|_| true, |_| panic!("the caller's admission failed"));
    let again = later(lists.under(&classes, &failing));
    let answer = untaken(&lists, &mut placement, &again);
    assert_eq!(answer, Unplaced::Refused(DecisionRefused::Faulted));
    assert_eq!(engine.faults(), 1);
    assert_eq!(said(&heard), at_send, "a refused call issues nothing");

    placement.sent(Side::Decode);
    let decode = lists.decode_id(0);
    assert_eq!(
        said(&heard)[at_send.len()..],
        [word("issued at 0", decode), word("dispatched at 0", decode)]
    );
}

/// A second decision that sheds is answered its shed by name: the reason,
/// and the stage of the decode's own decision that left nobody. Nothing
/// was issued for it, so the stages hear nothing. The first decision's
/// decode still stands chosen, and its caller sends it ahead of the
/// prefill's end.
#[test]
fn a_decode_call_that_sheds_is_answered_by_name_and_leaves_the_first_decode_chosen() {
    let (engine, heard) = asked_again();
    let lists = Lists::of(1, 1);
    let classes = lists.every();
    let request = later(lists.under(&classes, &Open));
    let mut placement = engine.place(&request).expect("a placement");
    placement.sent(Side::Prefill);
    let (prefill, decode) = (lists.prefill_id(0), lists.decode_id(0));
    let at_send = [
        word("issued at 0", prefill),
        word("dispatched at 0", prefill),
    ];

    // At the second call the caller's classes pair the prefill with no
    // decode: the call reads the lists of its own request.
    let alone = PartnerClasses::number(1, [&[0_u32; 0][..]]);
    let again = later(lists.under(&alone, &Open));
    let answer = untaken(&lists, &mut placement, &again);
    let by_name = matches!(
        answer,
        Unplaced::Shed {
            reason: ShedReason::NoEndpoints,
            emptied_by: "partners",
            ..
        }
    );
    assert!(by_name, "{answer:?}");
    assert_eq!(said(&heard), at_send, "a shed issues nothing");

    placement.sent(Side::Decode);
    placement.ended(Side::Prefill, SERVED);
    placement.ended(Side::Decode, SERVED);
    assert!(!placement.owes_events());
    assert_eq!(
        said(&heard)[at_send.len()..],
        [
            word("issued at 0", decode),
            word("dispatched at 0", decode),
            word("completed Ok", prefill),
            word("completed Ok", decode),
        ]
    );
}

/// A decode call out of turn is refused by name, with no decision made and
/// nothing said: ahead of the prefill's send, behind a call that took a
/// decode, behind that decode's end beside a prefill still owed, behind the
/// end of both sides, behind the send of the first decision's decode, and
/// behind a prefill's end said with the decode still chosen.
#[test]
fn a_decode_call_out_of_turn_is_refused_by_name_and_says_nothing() {
    let (engine, heard) = asked_again();
    let lists = Lists::of(1, 2);
    let classes = lists.every();
    let request = later(lists.under(&classes, &Open));
    let refused = |placement: &mut Placement<'_>, why: DecodeRefused| {
        let before = said(&heard);
        let answer = untaken(&lists, placement, &request);
        assert_eq!(answer, Unplaced::Refused(DecisionRefused::Decode(why)));
        assert_eq!(said(&heard), before, "{why:?}");
    };

    let mut placement = engine.place(&request).expect("a placement");
    refused(&mut placement, DecodeRefused::PrefillNotHeld);
    placement.sent(Side::Prefill);
    assert_eq!(placement.decode_again(&request), Ok(()));
    refused(&mut placement, DecodeRefused::DecodeStands);
    placement.sent(Side::Decode);
    placement.ended(Side::Decode, SERVED);
    refused(&mut placement, DecodeRefused::DecodeStands);
    placement.ended(Side::Prefill, SERVED);
    refused(&mut placement, DecodeRefused::NoLease);
    drop(placement);

    let mut sent = engine.place(&request).expect("a placement");
    sent.sent(Side::Prefill);
    sent.sent(Side::Decode);
    refused(&mut sent, DecodeRefused::DecodeStands);
    drop(sent);

    let mut ended = engine.place(&request).expect("a placement");
    ended.sent(Side::Prefill);
    ended.ended(Side::Prefill, SERVED);
    refused(&mut ended, DecodeRefused::NoLease);
}

/// The order a caller keeps behind a decode its call took: the call, then
/// the prefill's end when the prefill answers, then the decode's send at
/// its own later instant, then the decode's end. The stages hear each word
/// at its call: the decode's issue at the decode call, the prefill's end
/// ahead of the decode's dispatch, and the decode's end on the decode
/// taken.
///
/// A placement dropped between its prefill's send and its decode call
/// says the prefill's end as cancelled, and nothing of a decode that was
/// never issued. Dropped behind a decode call it gives the issued decode
/// up as well.
#[test]
fn the_stages_hear_a_decode_decided_again_at_its_call_its_send_and_its_end() {
    let (engine, heard) = asked_again();
    let lists = Lists::of(1, 2);
    let classes = lists.every();
    let request = later(lists.under(&classes, &Open));
    let (prefill, lead, taken) = (lists.prefill_id(0), lists.decode_id(0), lists.decode_id(1));
    let at_send = [
        word("issued at 0", prefill),
        word("dispatched at 0", prefill),
    ];

    let mut dropped = engine.place(&request).expect("a placement");
    dropped.sent(Side::Prefill);
    drop(dropped);
    let mut expected = at_send.to_vec();
    expected.push(word("completed Cancelled", prefill));
    assert_eq!(said(&heard), expected);

    let mut dropped = engine.place(&request).expect("a placement");
    dropped.sent(Side::Prefill);
    assert_eq!(dropped.decode_again(&request), Ok(()));
    drop(dropped);
    expected.extend(at_send.iter().cloned());
    expected.extend([
        word("issued at 0", lead),
        word("completed Cancelled", prefill),
        word("released Abandoned", lead),
    ]);
    assert_eq!(said(&heard), expected);

    // The first decode stops being ready behind the prefill's send, so
    // the call takes the second.
    let mut placement = engine.place(&request).expect("a placement");
    placement.sent(Side::Prefill);
    lists.decodes[0].set_status(WorkerStatus::NotReady);
    expected.extend(at_send.iter().cloned());
    assert_eq!(said(&heard), expected);

    assert_eq!(placement.decode_again(&request), Ok(()));
    expected.push(word("issued at 0", lead));
    assert_eq!(said(&heard), expected, "the issue, at the call");

    placement.ended(Side::Prefill, SERVED);
    expected.push(word("completed Ok", prefill));
    assert_eq!(said(&heard), expected, "the prefill's end, at its answer");
    assert!(placement.owes_events(), "the decode is still owed");

    placement.sent(Side::Decode);
    expected.push(word("dispatched at 1", taken));
    assert_eq!(said(&heard), expected, "the dispatch, at the send");

    placement.ended(Side::Decode, SERVED);
    expected.push(word("completed Ok", taken));
    assert_eq!(said(&heard), expected, "the end, on the decode taken");
    assert!(!placement.owes_events());
    drop(placement);
    assert_eq!(said(&heard), expected, "a drop says nothing more");
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A decode that never stood chosen takes no decode call: a pair whose
/// first call asked no decode again, a pair under a document that picks
/// both sides at once, and a single request, placed under a lease or under
/// none. Each is refused as not chosen, and the decode the first decision
/// booked is sent as it stands.
#[test]
fn a_decode_that_never_stood_chosen_takes_no_decode_call() {
    let not_chosen = Unplaced::Refused(DecisionRefused::Decode(DecodeRefused::NotChosen));
    for (pick, asks) in [
        (DecodePick::AfterPrefill, false),
        (DecodePick::WithPrefill, true),
    ] {
        let (engine, heard) = paired_picking(false, pick);
        let lists = Lists::of(1, 2);
        let classes = lists.every();
        let pair = lists.under(&classes, &Open);
        let first = PlaceRequest {
            decode_again: asks,
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
        };
        let mut placement = engine.place(&first).expect("a placement");
        placement.sent(Side::Prefill);
        let answer = untaken(&lists, &mut placement, &later(pair));
        assert_eq!(answer, not_chosen, "{pick:?}");
        placement.sent(Side::Decode);
        let (prefill, decode) = (lists.prefill_id(0), lists.decode_id(0));
        assert_eq!(
            said(&heard),
            [
                word("issued at 0", prefill),
                word("issued at 0", decode),
                word("dispatched at 0", prefill),
                word("dispatched at 0", decode),
            ],
            "{pick:?}"
        );
    }

    let workers = pool(1);
    let engines = [
        (in_order(false).0, true),
        (hosted(preset("round_robin")), false),
    ];
    for (engine, leased) in engines {
        let members = PublishedList::new(Arc::clone(&workers));
        let single = PlaceRequest {
            decode_again: true,
            ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
        };
        let mut placement = engine.place(&single).expect("a placement");
        placement.sent(Side::Single);
        assert_eq!(placement.owes_events(), leased);
        assert_eq!(placement.decode_again(&single).err(), Some(not_chosen));
        assert_eq!(placement.owes_events(), leased);
        assert!(placement.decode.is_none());
    }
}

/// The workers a second request states as tried are read by its decision:
/// its soft filter keeps the pick to the decodes the request has not tried
/// while one is left. With nothing stated the decode the first decision
/// named leads again.
#[test]
fn a_decode_decided_again_reads_what_its_request_tried() {
    for tries in [false, true] {
        let (engine, _) = asked_again();
        let lists = Lists::of(1, 2);
        let classes = lists.every();
        let request = later(lists.under(&classes, &Open));
        let mut placement = engine.place(&request).expect("a placement");
        assert_eq!(lists.places(&placement), (0, 0));
        placement.sent(Side::Prefill);
        let tried = [lists.decode_id(0)];
        let again = PlaceRequest {
            tried: if tries { &tried[..] } else { &[] },
            attempt: u8::from(tries),
            ..later(lists.under(&classes, &Open))
        };
        assert_eq!(placement.decode_again(&again), Ok(()));
        assert_eq!(lists.places(&placement), (0, usize::from(tries)));
    }
}

/// An encode request's three lists are its pair's two for a decode decided
/// again: the call reads the decode list among them, as the engine's decode
/// entry does, and takes the first of its decodes that can be sent to.
#[test]
fn a_decode_decided_again_over_an_encode_requests_lists_reads_its_pairs_decodes() {
    let (engine, _) = asked_again();
    let lists = Lists::of(1, 2);
    let classes = lists.every();
    let pair = lists.under(&classes, &Open);
    let mut placement = engine.place(&later(pair)).expect("a placement");
    assert_eq!(lists.places(&placement), (0, 0));
    placement.sent(Side::Prefill);
    lists.decodes[0].set_status(WorkerStatus::NotReady);
    let encoders = PublishedList::new(pool(0));
    let three = EncodeLists {
        pair,
        encoders: encoders.lent(),
        encoders_places: None,
        encoders_stalled: &[],
        encoders_held: &[],
        facts: &[],
        prefill_classes: &[],
    };
    let again = PlaceRequest {
        decode_again: true,
        ..PlaceRequest::of(MODEL, WorkerLists::Encode(three))
    };
    assert_eq!(placement.decode_again(&again), Ok(()));
    assert_eq!(lists.places(&placement), (0, 1));
}

/// A caller that says the prefill's end with the decode still chosen lets
/// the decode go with it: the send of that decode behind the end is out of
/// order, and a debug build stops on it by the order's name.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "sent behind the end of both its sides")]
fn a_chosen_decode_sent_behind_its_prefills_end_stops_a_debug_build() {
    let (engine, _) = asked_again();
    let lists = Lists::of(1, 1);
    let classes = lists.every();
    let request = later(lists.under(&classes, &Open));
    let mut placement = engine.place(&request).expect("a placement");
    placement.sent(Side::Prefill);
    placement.ended(Side::Prefill, SERVED);
    placement.sent(Side::Decode);
}

/// A decode's send said a second time says nothing, as its guard says a
/// word once. Behind the end of both sides it stops no debug build where
/// the pair's decode was booked with its prefill: no chosen decode was lost
/// there.
#[test]
fn a_decode_send_said_again_behind_both_ends_of_a_pair_booked_whole_says_nothing() {
    for (pick, asks) in [
        (DecodePick::AfterPrefill, false),
        (DecodePick::WithPrefill, true),
    ] {
        let (engine, heard) = paired_picking(false, pick);
        let lists = Lists::of(1, 1);
        let classes = lists.every();
        let request = PlaceRequest {
            decode_again: asks,
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(lists.under(&classes, &Open)))
        };
        let mut placement = engine.place(&request).expect("a placement");
        placement.sent(Side::Prefill);
        placement.sent(Side::Decode);
        placement.ended(Side::Prefill, SERVED);
        placement.ended(Side::Decode, SERVED);
        let ended = said(&heard);
        assert_eq!(ended.len(), 6, "{pick:?}");
        placement.sent(Side::Decode);
        assert_eq!(said(&heard), ended, "{pick:?}");
        assert!(!placement.owes_events());
    }
}

/// A caller of `place` states no worker id of its own, at a decode call as
/// at its first call: a decode list whose records name its workers
/// otherwise is told so in a debug build, at the decode the call reads.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "the decode stands at the target's position")]
fn a_decode_list_that_states_worker_ids_is_told_at_its_decode_call_in_a_debug_build() {
    let (engine, _) = asked_again();
    let lists = Lists::of(1, 1);
    let classes = lists.every();
    let request = later(lists.under(&classes, &Open));
    let mut placement = engine.place(&request).expect("a placement");
    placement.sent(Side::Prefill);
    let stated = PublishedList::with_facts(Arc::clone(&lists.decodes), |_| FixedFacts {
        id: Some(WorkerId(7)),
        ..FixedFacts::default()
    });
    let again = later(PairLists {
        decode: stated.lent(),
        ..lists.under(&classes, &Open)
    });
    let _ = placement.decode_again(&again);
}

/// A decode decided again keeps its prompt for the decode the request is
/// sent to. Where the route's rule refuses the decode the first decision
/// chose, the walk takes a later decode. Over the lists without the decode
/// first chosen, the prompt's next requests go to the decode it was sent
/// to, ahead of one that was sent less of it. A rule that panics where the
/// decode's decision asks it passes its panic to the call's caller, and
/// the placement stands on the decode first chosen.
#[test]
fn a_prompt_whose_decode_was_decided_past_a_refused_decode_goes_to_it_again() {
    let mut document = PolicySpec::pair_presets("cache_aware", "cache_aware").expect("presets");
    document.generation = 1;
    document.plans.pd_pair.decode = DecodePick::AfterPrefill;
    let spec = AppliedSpec::validate(&document).expect("a valid document");
    let engine = hosted(PolicyEngine::new(spec));
    let lists = Lists::of(1, 3);
    let classes = lists.every();
    let pair = lists.under(&classes, &Open);
    let tokens: Vec<u32> = (1_000..5_096).collect();
    let prompt = Prompt {
        tokens: Some(&tokens),
        ..Prompt::NONE
    };
    let first = PlaceRequest {
        prompt,
        decode_again: true,
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    let mut placement = engine.place(&first).expect("a placement");
    let chosen = lists.places(&placement);
    placement.sent(Side::Prefill);
    // The rule refuses the decode the first decision chose.
    let refused = lists.decodes[chosen.1].url().to_owned();
    let rule = |worker: &dyn Worker| worker.url() != refused;
    let again = PlaceRequest {
        prompt,
        decode_again: true,
        admits: Some(&rule),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    assert_eq!(placement.decode_again(&again), Ok(()));
    let to = lists.places(&placement);
    assert_ne!(to.1, chosen.1, "past the refused decode");
    placement.sent(Side::Decode);
    placement.ended(Side::Prefill, SERVED);
    placement.ended(Side::Decode, SERVED);
    drop(placement);
    // The decode first chosen keeps the entry its first decision wrote, so
    // the lists go on without it: among their decodes a read of the record
    // names the one sent most of the prompt.
    let sent = Arc::clone(&lists.decodes[to.1]);
    let stays = |decode: &&Arc<dyn Worker>| decode.url() != refused;
    let decodes: Arc<[Arc<dyn Worker>]> = lists.decodes.iter().filter(stays).cloned().collect();
    let rest = Lists {
        published: [&lists.prefills, &decodes].map(|list| PublishedList::new(Arc::clone(list))),
        prefills: Arc::clone(&lists.prefills),
        decodes,
    };
    let partners = rest.every();
    let two = rest.under(&partners, &Open);
    // The other decode is sent the prompt's leading three quarters.
    let other = |worker: &dyn Worker| worker.url() != sent.url();
    let leading = PlaceRequest {
        prompt: Prompt {
            tokens: Some(&tokens[..3_072]),
            ..Prompt::NONE
        },
        admits: Some(&other),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(two))
    };
    drop(engine.place(&leading).expect("a placement"));
    // The prompt goes back to the decode it was sent to.
    let stated = PlaceRequest {
        prompt,
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(two))
    };
    let went: Vec<_> = (0..16)
        .map(|_| rest.places(&engine.place(&stated).expect("a placement")))
        .collect();
    let to = (0, usize::from(!Arc::ptr_eq(&rest.decodes[0], &sent)));
    assert_eq!(went, [to; 16], "the decode the prompt was sent to");

    // A rule that panics at its first question alone, inside the decode's
    // decision: asked again, it would admit.
    let mut placement = engine.place(&first).expect("a placement");
    let stood = lists.places(&placement);
    placement.sent(Side::Prefill);
    let asked = Cell::new(false);
    let once = |_: &dyn Worker| -> bool {
        if asked.replace(true) {
            return true;
        }
        panic!("the route's rule failed")
    };
    let failing = PlaceRequest {
        prompt,
        decode_again: true,
        admits: Some(&once),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    let raised = catch_unwind(AssertUnwindSafe(|| {
        placement.decode_again(&failing).is_ok()
    }));
    assert!(raised.is_err(), "the rule's panic reaches the caller");
    assert_eq!(
        lists.places(&placement),
        stood,
        "on the decode first chosen"
    );
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}
