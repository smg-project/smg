//! The cases of the decisions an engine keeps under `explain.sample`.

use std::{
    sync::{atomic::Ordering, Arc},
    thread,
};

use smg_worker::identity::basic::Worker;

use super::{
    super::{
        place::PlaceRequest,
        single::tests::{pool, request, Asks, MODEL},
    },
    *,
};
use crate::{
    document::spec::{AppliedSpec, PolicySpec, ShadowSpec},
    engine::{
        decision_engine::{Decided, DecisionEngine, WorkerLists},
        framework::Place,
        list_cell::PublishedList,
        records::worker_id,
    },
    tracking::lease::EventSet,
};

/// The engine of `document`, as its host holds it: for the life of the
/// process.
fn hosted(document: &PolicySpec) -> &'static PolicyEngine {
    let spec = AppliedSpec::validate(document).expect("the document validates");
    Box::leak(Box::new(PolicyEngine::new(spec)))
}

/// The preset `name` under a document that keeps every `sample`th decision,
/// and none where `sample` is zero.
fn sampling(name: &str, sample: u32) -> PolicySpec {
    let mut document = PolicySpec::preset(name).expect("a built-in preset");
    document.generation = 1;
    document.explain.sample = sample;
    document
}

/// One request that states `id`, placed over `members`, the published form
/// of `list`: the place in the list of the worker its placement names.
fn placed(
    engine: &'static PolicyEngine,
    list: &[Arc<dyn Worker>],
    members: &PublishedList,
    id: Option<&str>,
) -> usize {
    let request = PlaceRequest {
        request_id: id,
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    let placement = engine.place(&request).expect("a placement");
    let at = list
        .iter()
        .position(|worker| Arc::ptr_eq(worker, &placement.worker));
    at.expect("a worker of the list")
}

/// A sample changes no answer. Under `round_robin` the workers picked over
/// six rounds of a list are the ones a document that keeps no sample picks,
/// in the same order, whether the document keeps every request, every
/// third or every seventh: a kept request takes one turn of its pool, as
/// every other does.
#[test]
fn a_sample_changes_no_answer() {
    let turns = |sample: u32| {
        let engine = hosted(&sampling("round_robin", sample));
        let list = pool(5);
        let members = PublishedList::new(Arc::clone(&list));
        let ids: Vec<String> = (0..30).map(|n| format!("r-{n}")).collect();
        let places: Vec<usize> = ids
            .iter()
            .map(|id| placed(engine, &list, &members, Some(id.as_str())))
            .collect();
        let kept = ids
            .iter()
            .filter(|id| engine.explained(id.as_bytes()).is_some());
        (places, kept.count())
    };
    let (plain, kept) = turns(0);
    assert_eq!(kept, 0, "a document that states no sample keeps none");
    for turn in plain.windows(2) {
        assert_eq!(turn[1], (turn[0] + 1) % 5, "one turn a request");
    }
    for (sample, kept) in [(1, 30), (3, 10), (7, 4)] {
        assert_eq!(turns(sample), (plain.clone(), kept), "a sample of {sample}");
    }
}

/// Every Nth plain request of a place one pick decides is kept as the pick
/// it was, and found by the id its caller stated: the decision that names
/// the worker the request was placed on alone, a drawn pick's trace with no
/// survivor counted, under the binding and the profile that picked, dated
/// when it was kept. The answer is still a place: no lease takes an id.
#[test]
fn a_kept_place_is_the_pick_as_it_was_taken() {
    let engine = hosted(&sampling("round_robin", 2));
    let handle = engine.spec();
    let alone = matches!(handle.place_for(MODEL, Phase::Single), Place::Alone(_));
    assert!(alone, "one pick decides the place");
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let before = Instant::now();
    for n in 0..8 {
        let id = format!("r-{n}");
        let place = placed(engine, &list, &members, Some(id.as_str()));
        let record = engine.explained(id.as_bytes());
        assert_eq!(record.is_some(), n % 2 == 1, "request {n}");
        let Some(record) = record else { continue };
        let pick = record.decision.pick().expect("a pick");
        assert_eq!(pick.targets.len(), 1);
        let target = pick.targets.first().expect("the target");
        let named = (target.position as usize, target.id);
        assert_eq!(named, (place, worker_id(list[place].as_ref())));
        assert!(pick.trace.drew_first(), "a drawn pick's trace");
        assert_eq!(pick.trace.survivors.get(), None, "no survivor counted");
        assert_eq!(pick.trace.generation, 1);
        assert_eq!((record.binding, record.shadow), (Some(0), None));
        assert!(before <= record.at && record.at <= Instant::now());
        let names = record.names(handle.engine()).expect("its names");
        let named = (names.plan, names.profile.as_str(), names.picker.as_str());
        assert_eq!(named, ("single", "round_robin", "round-robin"));
    }
    assert_eq!(engine.leases.0.load(Ordering::Relaxed), 0, "no id taken");
}

/// A request that states no id is not counted and not kept: the requests
/// that state one are counted among themselves, and no record is found
/// under no id.
#[test]
fn a_request_that_states_no_id_is_not_counted() {
    let engine = hosted(&sampling("round_robin", 2));
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    for id in [Some("a"), None, None, Some("b"), None, Some("c"), Some("d")] {
        let _ = placed(engine, &list, &members, id);
    }
    let kept = ["a", "b", "c", "d"].map(|id| engine.explained(id.as_bytes()).is_some());
    assert_eq!(kept, [false, true, false, true]);
    assert_eq!(engine.explained(b""), None);
}

/// A whole decision is kept as it was answered, a pick and a shed alike,
/// dated at the decision's instant. The newest record under an id is the
/// one read.
#[test]
fn a_whole_decision_is_kept_as_it_was_answered() {
    let engine = hosted(&sampling("power_of_two", 1));
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let asks = Asks::of(request(), Some(&members));
    let whole = |asks: &Asks<'_>, now| {
        let answered = engine.decide_under(GatewayId::LOCAL, asks, now);
        let (decided, _) = answered.expect("an answer").whole().expect("no place");
        decided.decision
    };
    let now = Instant::now();
    let decision = whole(&asks, now);
    assert!(decision.pick().is_some());
    let record = engine.explained(b"request").expect("the pick is kept");
    assert_eq!((record.decision, record.at), (decision, now));
    assert_eq!((record.binding, record.shadow), (Some(0), None));
    assert_eq!(asks.made.get(), 1, "the request is made once");

    // A request whose host hands no list is shed, and its shed is kept
    // under the same id: the newer of the two records is read.
    let none = Asks::of(request(), None);
    let later = Instant::now();
    let shed = whole(&none, later);
    assert!(matches!(shed, Decision::Shed(_)));
    let record = engine.explained(b"request").expect("the shed is kept");
    assert_eq!((record.decision, record.at), (shed, later));
}

/// A single's record holds what its binding's shadow said beside the
/// decision: the word the engine counted for the binding.
#[test]
fn a_singles_record_holds_what_its_shadow_said() {
    let mut document = sampling("power_of_two", 1);
    let live = document.bindings[0].profile.clone();
    let twin = document.profiles[&live.expect("a binding of one profile")].clone();
    document.profiles.insert("twin".into(), twin);
    document.bindings[0].shadow = Some(ShadowSpec::Profile("twin".into()));
    let engine = hosted(&document);
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let asks = Asks::of(request(), Some(&members));
    let answered = engine.decide_under(GatewayId::LOCAL, &asks, Instant::now());
    assert!(matches!(answered, Ok(Decided::Whole(..))));
    let record = engine.explained(b"request").expect("the pick is kept");
    let said = record.shadow.expect("the shadow's word");
    let counted = engine.shadow_kinds(0)[said.index()];
    assert_eq!(counted, 1, "the word the engine counted");
}

/// Under a document that takes a word on a lease and keeps a sample, a
/// place one pick decides is noted, and kept as it was noted: the record
/// is the decision its lease was issued for, dated as the lease is. Each
/// request takes one id and one turn of its pool.
#[test]
fn a_noted_place_is_kept_as_it_was_noted() {
    // Another model's binding names a profile that takes a lease's words:
    // the document takes them, and this model's place stays one pick's.
    let mut document = sampling("round_robin", 1);
    let loads = PolicySpec::preset("least_load").expect("a built-in preset");
    document.profiles.extend(loads.profiles);
    let mut other = document.bindings[0].clone();
    other.matcher.model = "another-model".into();
    other.profile = Some("least_load".into());
    document.bindings.insert(0, other);
    let engine = hosted(&document);
    assert!(engine.in_force().hears.intersects(EventSet::LEASE));
    let handle = engine.spec();
    let alone = matches!(handle.place_for(MODEL, Phase::Single), Place::Alone(_));
    assert!(alone, "one pick decides the place");

    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    let asks = Asks::of(request(), Some(&members));
    let mut turns = Vec::new();
    for n in 1..=8_u64 {
        let now = Instant::now();
        let answered = engine.decide_under(GatewayId::LOCAL, &asks, now);
        let (decided, _) = answered.expect("an answer").whole().expect("noted");
        let lease = decided.lease.expect("a lease").lease;
        assert_eq!(lease.id.0, n, "one id a request");
        let record = engine.explained(b"request").expect("the pick is kept");
        assert_eq!((record.decision, record.at), (decided.decision, now));
        assert_eq!(record.binding, Some(1), "this model's binding");
        let pick = decided.decision.pick().expect("a pick");
        assert!(pick.trace.drew_first(), "a drawn pick's trace");
        turns.push(pick.targets.first().expect("the target").position);
    }
    for turn in turns.windows(2) {
        assert_eq!(turn[1], (turn[0] + 1) % 4, "one turn a request");
    }
}

/// A plain request that is never made is asked its id through its maker,
/// under a document that keeps a sample and under no other. A maker that
/// does not answer is made for its id, once a request, and its place is
/// still the answer. A request stated ahead answers with its own id.
#[test]
fn a_request_that_is_never_made_is_asked_its_id_under_a_sample_alone() {
    let placed_under = |sample: u32, requests: u32| {
        let engine = hosted(&sampling("round_robin", sample));
        let list = pool(4);
        let members = PublishedList::new(Arc::clone(&list));
        let asks = Asks::of(request(), Some(&members));
        for _ in 0..requests {
            let answered = engine.decide_under(GatewayId::LOCAL, &asks, Instant::now());
            assert!(matches!(answered, Ok(Decided::Alone(_))), "{answered:?}");
        }
        (asks.made.get(), engine.explained(b"request").is_some())
    };
    assert_eq!(placed_under(0, 4), (0, false), "no sample: no id is asked");
    assert_eq!(placed_under(2, 4), (4, true), "made for its id alone");

    let engine = hosted(&sampling("round_robin", 2));
    assert_eq!(request().request_id(&engine.spec()), b"request");
}

/// The count is the calling thread's own, and it starts over under another
/// document. Two threads that each place one request short of a sample
/// keep none between them. A thread that places under a second engine
/// starts over there, and starts over again when it comes back.
#[test]
fn the_count_is_each_threads_own_and_starts_over_under_another_document() {
    let engine = hosted(&sampling("round_robin", 3));
    let place = |engine: &'static PolicyEngine, ids: &[&str]| {
        let list = pool(4);
        let members = PublishedList::new(Arc::clone(&list));
        for id in ids {
            let _ = placed(engine, &list, &members, Some(*id));
        }
    };
    let kept = |engine: &PolicyEngine, ids: &[&str]| -> Vec<bool> {
        let found = |id: &&str| engine.explained(id.as_bytes()).is_some();
        ids.iter().map(found).collect()
    };
    thread::scope(|threads| {
        threads.spawn(|| place(engine, &["x-1", "x-2"]));
        threads.spawn(|| place(engine, &["y-1", "y-2"]));
    });
    assert_eq!(kept(engine, &["x-1", "x-2", "y-1", "y-2"]), [false; 4]);

    // This thread's third request is its sample.
    place(engine, &["z-1", "z-2", "z-3", "z-4", "z-5"]);
    let third = [false, false, true, false, false];
    assert_eq!(kept(engine, &["z-1", "z-2", "z-3", "z-4", "z-5"]), third);

    // Two requests stand counted under the first engine. Under a second
    // one the thread starts over, and back under the first it starts over
    // again: the two it had counted there are gone.
    let other = hosted(&sampling("round_robin", 3));
    place(other, &["o-1", "o-2"]);
    assert_eq!(kept(other, &["o-1", "o-2"]), [false; 2]);
    place(engine, &["b-1", "b-2", "b-3"]);
    assert_eq!(kept(engine, &["b-1", "b-2", "b-3"]), [false, false, true]);
}

/// The ring is the document's. A document put in force on the running
/// engine starts with an empty ring: a decision kept under the document
/// that stood is found no more. The calling thread's count starts over
/// too: one request stood counted, and the second request under the new
/// document is its sample all the same.
#[test]
fn a_document_put_in_force_starts_with_an_empty_ring() {
    let engine = hosted(&sampling("round_robin", 2));
    let list = pool(4);
    let members = PublishedList::new(Arc::clone(&list));
    for id in ["a-1", "a-2", "a-3"] {
        let _ = placed(engine, &list, &members, Some(id));
    }
    let kept = ["a-1", "a-2", "a-3"].map(|id| engine.explained(id.as_bytes()).is_some());
    assert_eq!(kept, [false, true, false]);

    let mut next = sampling("round_robin", 2);
    next.generation = 2;
    let next = AppliedSpec::validate(&next).expect("the document validates");
    let applied = engine.apply(next);
    applied.expect("a number above the standing one");
    assert_eq!(engine.explained(b"a-2"), None, "the ring that stood");
    for id in ["b-1", "b-2"] {
        let _ = placed(engine, &list, &members, Some(id));
    }
    let kept = ["b-1", "b-2"].map(|id| engine.explained(id.as_bytes()).is_some());
    assert_eq!(kept, [false, true]);
}
