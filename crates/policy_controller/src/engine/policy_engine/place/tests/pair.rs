//! The cases of a request with a pair's two lists.

mod decode_again;

use std::panic::{catch_unwind, AssertUnwindSafe};

use smg_worker::{
    identity::builder::BasicWorkerBuilder,
    test_support::no_health_check,
    worker::{WorkerSpec, WorkerType},
};

use super::{super::pair::ask, *};

/// The caller's admission as two tests of a place: the prefills', then the
/// decodes'.
#[derive(Debug)]
struct Gates(fn(usize) -> bool, fn(usize) -> bool);

impl PairAdmission for Gates {
    fn has_capacity(&self, prefill: usize) -> bool {
        (self.0)(prefill)
    }

    fn has_room(&self, decode: usize) -> bool {
        (self.1)(decode)
    }
}

/// What the pickers of a document heard, a list a picker.
type HeardBy = Arc<Mutex<Vec<Heard>>>;

/// The stock pair document with [`InOrder`] in the place of both sides'
/// pick and without their `healthy` filter, so that a worker that is not
/// ready stays a candidate of the decision, and what each picker heard.
/// `keyed` makes the request's own id its routing value.
fn paired(keyed: bool) -> (&'static PolicyEngine, HeardBy) {
    paired_picking(keyed, DecodePick::default())
}

/// [`paired`] under a document that picks a pair's decode when `decode`
/// says.
fn paired_picking(keyed: bool, decode: DecodePick) -> (&'static PolicyEngine, HeardBy) {
    let heard = HeardBy::default();
    let kept = Arc::clone(&heard);
    let mut stages = StageRegistry::builtin();
    let added = stages.picker("in-order", move || {
        let own = Heard::default();
        kept.lock().push(Arc::clone(&own));
        Box::new(InOrder {
            heard: own,
            budgets: Budgets::default(),
            panics: false,
        })
    });
    added.expect("a free name");
    let mut document = PolicySpec::pair_presets("round_robin", "round_robin").expect("presets");
    document.generation = 1;
    document.plans.pd_pair.decode = decode;
    for profile in document.profiles.values_mut() {
        profile.pick = "in-order".into();
        profile
            .filter
            .retain(|filter| *filter != FilterSpec::Name("healthy".to_owned()));
    }
    if keyed {
        let rid = KeySourceSpec {
            rid: true,
            ..KeySourceSpec::default()
        };
        document.bindings[0].routing_key = Some(RoutingKeySpec {
            sources: vec![rid],
            strip_lineage: false,
        });
    }
    let spec = AppliedSpec::validate_in(&document, &stages, &[]);
    let engine = PolicyEngine::new(spec.expect("the stages hold the picker"));
    (hosted(engine), heard)
}

/// What the stages heard. The document's two pickers both hear every word
/// of a pair's lease: each heard the words answered here, once each and in
/// this order. A picker that heard nothing at all is not counted.
fn said(heard: &HeardBy) -> Vec<(String, WorkerId)> {
    let pickers = heard.lock();
    let each = pickers.iter().map(|one| one.lock().clone());
    let mut each = each.filter(|one| !one.is_empty());
    let Some(said) = each.next() else {
        return Vec::new();
    };
    let others: Vec<_> = each.collect();
    assert_eq!(
        others,
        std::slice::from_ref(&said),
        "both pickers, the same words"
    );
    said
}

/// A caller's two lists of ready workers as it publishes them, each with
/// its cell.
struct Lists {
    prefills: Arc<[Arc<dyn Worker>]>,
    decodes: Arc<[Arc<dyn Worker>]>,
    published: [PublishedList; 2],
}

impl Lists {
    fn of(prefills: usize, decodes: usize) -> Self {
        let side = |name: &str, worker_type, count| -> Arc<[Arc<dyn Worker>]> {
            let ready = |at: usize| -> Arc<dyn Worker> {
                let worker = <BasicWorkerBuilder>::from_spec(WorkerSpec {
                    worker_type,
                    ..WorkerSpec::new(format!("grpc://{name}-{at}:9000"))
                })
                .health_config(no_health_check())
                .build();
                worker.set_status(WorkerStatus::Ready);
                Arc::new(worker)
            };
            (0..count).map(ready).collect()
        };
        let prefills = side("prefill", WorkerType::Prefill, prefills);
        let decodes = side("decode", WorkerType::Decode, decodes);
        let published = [&prefills, &decodes].map(|list| PublishedList::new(Arc::clone(list)));
        Self {
            prefills,
            decodes,
            published,
        }
    }

    /// The lists as their caller hands them for one request.
    fn under<'a>(
        &'a self,
        classes: &'a PartnerClasses,
        admission: &'a dyn PairAdmission,
    ) -> PairLists<'a> {
        let [prefill, decode] = &self.published;
        PairLists {
            decode: decode.lent(),
            admission,
            ..pair_of(prefill, classes)
        }
    }

    /// Classes under which every prefill pairs with every decode.
    fn every(&self) -> PartnerClasses {
        let all: Vec<u32> = (0..self.decodes.len() as u32).collect();
        PartnerClasses::number(self.decodes.len(), self.prefills.iter().map(|_| &all[..]))
    }

    /// The places in the two lists of the workers `placement` names.
    fn places(&self, placement: &Placement<'_>) -> (usize, usize) {
        let at = |list: &[Arc<dyn Worker>], worker: &Arc<dyn Worker>| {
            let place = list.iter().position(|listed| Arc::ptr_eq(listed, worker));
            place.expect("a worker of the list")
        };
        let decode = placement.decode.as_ref().expect("a pair's decode");
        (
            at(&self.prefills, &placement.worker),
            at(&self.decodes, decode),
        )
    }

    fn prefill_id(&self, at: usize) -> WorkerId {
        worker_id(self.prefills[at].as_ref())
    }

    fn decode_id(&self, at: usize) -> WorkerId {
        worker_id(self.decodes[at].as_ref())
    }
}

/// A pair's targets are asked best first, and the first both of whose
/// workers can be sent to is taken: a target whose decode cannot is passed
/// over whole. The placement names the prefill as its worker and the decode
/// beside it, each by its own id and each side's profile by its label, and
/// a side's dispatch is said on the target taken.
///
/// The classic router hands no pair with a side that does not hold either:
/// a decode counts as a prefill's partner only while it is eligible
/// (`model_gateway/src/routers/common/placement.rs:1389-1394`, over
/// `:1342-1345`).
#[test]
fn a_pair_is_placed_on_the_first_target_both_of_whose_workers_can_be_sent_to() {
    // The first decode cannot be sent to, then the first two, then the
    // first three: the walk goes on to the last target a decision names.
    // The second is held back by its circuit breaker alone.
    for down in 1..MAX_TARGETS {
        let (engine, heard) = paired(false);
        let lists = Lists::of(2, MAX_TARGETS);
        let classes = lists.every();
        for (at, decode) in lists.decodes[..down].iter().enumerate() {
            if at == 1 {
                trip(decode);
            } else {
                decode.set_status(WorkerStatus::NotReady);
            }
        }
        let pair = lists.under(&classes, &Open);
        let request = PlaceRequest::of(MODEL, WorkerLists::Pair(pair));
        let mut placement = engine.place(&request).expect("a placement");
        assert_eq!(lists.places(&placement), (0, down));
        assert_eq!(
            (placement.id(), placement.decode_id()),
            (lists.prefill_id(0), Some(lists.decode_id(down)))
        );
        let labels = (placement.profile, placement.decode_profile);
        assert_eq!(labels, ("prefill", Some("decode")));
        assert!(placement.owes_events());
        placement.sent(Side::Prefill);
        placement.sent(Side::Decode);
        assert_eq!(
            said(&heard),
            [
                ("issued at 0".to_owned(), lists.prefill_id(0)),
                ("issued at 0".to_owned(), lists.decode_id(0)),
                (format!("dispatched at {down}"), lists.prefill_id(0)),
                (format!("dispatched at {down}"), lists.decode_id(down)),
            ]
        );
        drop(placement);
        assert_eq!((engine.faults(), engine.unheard()), (0, 0));
    }

    // A prefill held back by its circuit breaker alone holds no target.
    let (engine, _) = paired(false);
    let lists = Lists::of(1, 1);
    let classes = lists.every();
    trip(&lists.prefills[0]);
    let pair = lists.under(&classes, &Open);
    let request = PlaceRequest::of(MODEL, WorkerLists::Pair(pair));
    assert_eq!(engine.place(&request).err(), Some(Unplaced::NoWorkers));
}

/// A side that cannot be sent to is left out of the next ask by its own
/// id, and the side that held stays a candidate of it. A pair's targets
/// share one prefill, so a prefill that cannot be sent to fails every
/// target of its decision: the decision is given up by name, once, and the
/// prefill's id alone joins the ids to leave out. The next ask names
/// another prefill with the decode the first one named. One call of
/// `place` makes both asks.
///
/// The classic router counts a prefill only while it is available and one
/// of its partners is
/// (`model_gateway/src/routers/common/placement.rs:1342-1345`,
/// `:1392-1394`).
#[test]
fn a_side_that_cannot_be_sent_to_is_left_out_of_the_next_ask_by_its_own_id() {
    let (engine, heard) = paired(false);
    let lists = Lists::of(2, 2);
    let classes = lists.every();
    lists.prefills[0].set_status(WorkerStatus::NotReady);
    let pair = lists.under(&classes, &Open);
    let request = PlaceRequest::of(MODEL, WorkerLists::Pair(pair));
    let mut call = Call::default();
    let first = ask(engine.loaded(), &request, &pair, &mut call);
    assert!(matches!(first, Ok(None)), "no target holds");
    assert_eq!(
        call.missed,
        [lists.prefill_id(0)],
        "the prefill alone, once"
    );
    assert_eq!(
        said(&heard),
        [
            ("issued at 0".to_owned(), lists.prefill_id(0)),
            ("issued at 0".to_owned(), lists.decode_id(0)),
            ("released CommitFailed".to_owned(), lists.prefill_id(0)),
            ("released CommitFailed".to_owned(), lists.decode_id(0)),
        ]
    );
    let second = ask(engine.loaded(), &request, &pair, &mut call);
    let second = second.expect("an answer").expect("a placement");
    assert_eq!(lists.places(&second), (1, 0), "the decode is named again");
    drop(second);

    let placed = engine.place(&request).expect("the call's second ask");
    assert_eq!(lists.places(&placed), (1, 0));
    drop(placed);

    // Decodes that cannot be sent to are named, each by its own id, and
    // the prefill that held is not.
    lists.prefills[0].set_status(WorkerStatus::Ready);
    for decode in lists.decodes.iter() {
        decode.set_status(WorkerStatus::NotReady);
    }
    let mut call = Call::default();
    let third = ask(engine.loaded(), &request, &pair, &mut call);
    assert!(matches!(third, Ok(None)));
    assert_eq!(call.missed, [lists.decode_id(0), lists.decode_id(1)]);

    // A target neither of whose workers can be sent to names both, and
    // the prefill every target of the decision names joins once.
    lists.prefills[0].set_status(WorkerStatus::NotReady);
    let mut call = Call::default();
    let fourth = ask(engine.loaded(), &request, &pair, &mut call);
    assert!(matches!(fourth, Ok(None)));
    assert_eq!(
        call.missed,
        [lists.prefill_id(0), lists.decode_id(0), lists.decode_id(1)]
    );

    // An ask that fails again adds to the ids it was handed and drops none:
    // told to leave decode 1 out, it names the prefill and decode 0.
    let mut call = Call::default();
    call.missed.push(lists.decode_id(1));
    let fifth = ask(engine.loaded(), &request, &pair, &mut call);
    assert!(matches!(fifth, Ok(None)));
    assert_eq!(
        call.missed,
        [lists.decode_id(1), lists.prefill_id(0), lists.decode_id(0)]
    );
}

/// Which workers pair is the caller's word: a decision names no pair
/// across the classes its lists carry, and the walk takes none across
/// them. Under the stock pair document every placement is a pair of one
/// class, and a prefill whose one partner is not ready is passed over by
/// the decision for the prefill of the other class with its own decode.
/// Under a document that keeps a decode that is not ready, the prefill
/// holds no target: the walk does not turn to a decode of another class.
///
/// The classic router's lists keep two known runtimes apart on the wire
/// that cannot carry such a pair
/// (`model_gateway/src/routers/common/pair_index.rs:66-76`), and its
/// selection pairs a prefill with its own partners alone
/// (`model_gateway/src/routers/common/placement.rs:1392-1394`).
#[test]
fn no_pair_crosses_the_classes_its_caller_states() {
    let mut document = PolicySpec::pair_presets("round_robin", "round_robin").expect("presets");
    document.generation = 1;
    let engine = hosted(engine_of(&document));
    let lists = Lists::of(2, 2);
    // Prefill 0 pairs with decode 0 alone, and prefill 1 with decode 1.
    let classes = PartnerClasses::number(2, [&[0_u32][..], &[1][..]]);
    let pair = lists.under(&classes, &Open);
    let request = PlaceRequest::of(MODEL, WorkerLists::Pair(pair));
    let mut taken = [0_u32; 2];
    for _ in 0..8 {
        let placement = engine.place(&request).expect("a placement");
        let (prefill, decode) = lists.places(&placement);
        assert_eq!(prefill, decode, "a pair of one class");
        taken[prefill] += 1;
    }
    assert!(taken[0] > 0 && taken[1] > 0, "{taken:?}");
    lists.decodes[0].set_status(WorkerStatus::NotReady);
    for _ in 0..4 {
        let placement = engine.place(&request).expect("a placement");
        assert_eq!(
            lists.places(&placement),
            (1, 1),
            "never prefill 0 with decode 1"
        );
    }

    let (keeps, _) = paired(false);
    let lists = Lists::of(2, 2);
    let pair = lists.under(&classes, &Open);
    let request = PlaceRequest::of(MODEL, WorkerLists::Pair(pair));
    lists.decodes[0].set_status(WorkerStatus::NotReady);
    let mut call = Call::default();
    let held = ask(keeps.loaded(), &request, &pair, &mut call);
    assert!(matches!(held, Ok(None)), "prefill 0 has no other partner");
    assert_eq!(call.missed, [lists.decode_id(0)]);
}

/// A caller that states first partners is placed on a pair of them while
/// one can be sent to, whatever else its lists hold: the engine hands the
/// pair's decision the first partners its caller numbered
/// ([`PartnerClasses::number_first`]). Where no first partner is ready,
/// every pair of the classes stands.
///
/// The classic router places a pair on one known runtime ahead of a pair
/// with a worker whose runtime is unknown
/// (`model_gateway/src/routers/common/placement.rs:2479-2502` on main
/// `30f636706`), and a gateway states that order as first partners.
#[test]
fn a_pair_of_first_partners_is_placed_ahead_of_every_other() {
    let mut document = PolicySpec::pair_presets("round_robin", "round_robin").expect("presets");
    document.generation = 1;
    let engine = hosted(engine_of(&document));
    let lists = Lists::of(2, 2);
    // Both prefills pair with both decodes, and prefill 1 with decode 1
    // first.
    let both: &[u32] = &[0, 1];
    let classes = PartnerClasses::number_first(2, [(both, &[][..]), (both, &[1][..])]);
    let pair = lists.under(&classes, &Open);
    let request = PlaceRequest::of(MODEL, WorkerLists::Pair(pair));
    for _ in 0..8 {
        let placement = engine.place(&request).expect("a placement");
        assert_eq!(lists.places(&placement), (1, 1), "the pair that goes first");
    }
    lists.decodes[1].set_status(WorkerStatus::NotReady);
    let mut taken = [0_u32; 2];
    for _ in 0..8 {
        let placement = engine.place(&request).expect("a placement");
        let (prefill, decode) = lists.places(&placement);
        assert_eq!(decode, 0, "the one ready decode");
        taken[prefill] += 1;
    }
    assert!(taken[0] > 0 && taken[1] > 0, "{taken:?}");
}

/// Prefills that are all at their caller's admission limit answer the word
/// on which the caller waits for room, and no refusal: no lease was issued
/// and nothing is said. With one prefill under its limit the request is
/// placed on it.
///
/// The classic router answers the same way where each prefill at its limit
/// has a partner that can be sent to: "Selection was not run; the request
/// waits for a slot"
/// (`model_gateway/src/routers/common/placement.rs:332-334`, `:2532-2545`).
/// Where a prefill at its limit has none, neither waits for it
/// ([`a_pairs_shed_is_answered_by_its_name`]).
#[test]
fn prefills_all_at_their_limit_answer_the_word_their_caller_waits_on() {
    let (engine, heard) = paired(false);
    let lists = Lists::of(2, 2);
    let classes = lists.every();
    let full = Gates(|_| false, |_| true);
    let pair = lists.under(&classes, &full);
    let request = PlaceRequest::of(MODEL, WorkerLists::Pair(pair));
    assert_eq!(engine.place(&request).err(), Some(Unplaced::PrefillsFull));
    assert!(said(&heard).is_empty(), "a shed issues no lease");

    let second = Gates(|prefill| prefill == 1, |_| true);
    let pair = lists.under(&classes, &second);
    let request = PlaceRequest::of(MODEL, WorkerLists::Pair(pair));
    let placement = engine.place(&request).expect("a placement");
    assert_eq!(lists.places(&placement), (1, 0));
}

/// A pair's decision that sheds is answered its shed by name: the reason,
/// and the stage that left nobody. Under the stock pair document a pair
/// none of whose decodes is ready has no worker for the request, by its
/// `healthy` filter.
///
/// The classic router answers such lists that no pair is open
/// (`model_gateway/src/routers/common/placement.rs:2503-2530` on main
/// `30f636706`). It asks a prefill's limit of the prefills that still have
/// a partner to send to, and of no other (`:2479-2502`, `:2532-2545`
/// there), and so does the pair's decision. One prefill at its limit whose
/// one partner is not ready has no pair, and no request waits for it. With
/// two classes, one prefill at its limit beside its ready decode and the
/// other's decode not ready, the request waits for the full prefill,
/// whichever of the two it is.
#[test]
fn a_pairs_shed_is_answered_by_its_name() {
    let mut document = PolicySpec::pair_presets("round_robin", "round_robin").expect("presets");
    document.generation = 1;
    let stock = || hosted(engine_of(&document));
    let no_worker = |answer: Option<Unplaced>, stage: &str| {
        let by = |emptied_by: &str| emptied_by == stage;
        matches!(
            answer,
            Some(Unplaced::Shed { reason: ShedReason::NoEndpoints, emptied_by, .. }) if by(emptied_by)
        )
    };

    // No decode is ready.
    let engine = stock();
    let lists = Lists::of(2, 2);
    let classes = lists.every();
    for decode in lists.decodes.iter() {
        decode.set_status(WorkerStatus::NotReady);
    }
    let pair = lists.under(&classes, &Open);
    let request = PlaceRequest::of(MODEL, WorkerLists::Pair(pair));
    let answer = engine.place(&request).err();
    assert!(no_worker(answer, "healthy"), "{answer:?}");

    // One prefill at its limit, whose one partner is not ready: no pair,
    // ahead of the limit.
    let engine = stock();
    let lists = Lists::of(1, 1);
    let classes = lists.every();
    lists.decodes[0].set_status(WorkerStatus::NotReady);
    let full = Gates(|_| false, |_| true);
    let pair = lists.under(&classes, &full);
    let request = PlaceRequest::of(MODEL, WorkerLists::Pair(pair));
    let answer = engine.place(&request).err();
    assert!(no_worker(answer, "healthy"), "{answer:?}");

    // Two classes: one prefill is at its limit beside its ready decode, and
    // the other's decode is not ready. The request waits for the full
    // prefill, asked both ways round.
    let one_full = [
        Gates(|prefill| prefill != 0, |_| true),
        Gates(|prefill| prefill != 1, |_| true),
    ];
    for (full, gates) in one_full.iter().enumerate() {
        let engine = stock();
        let lists = Lists::of(2, 2);
        let classes = PartnerClasses::number(2, [&[0_u32][..], &[1][..]]);
        lists.decodes[1 - full].set_status(WorkerStatus::NotReady);
        let pair = lists.under(&classes, gates);
        let request = PlaceRequest::of(MODEL, WorkerLists::Pair(pair));
        let answer = engine.place(&request).err();
        assert_eq!(answer, Some(Unplaced::PrefillsFull), "prefill {full}");
    }
}

/// A pair's placement names the request's routing value where its document
/// derived one that leaves in the clear, as a single request's does.
#[test]
fn a_pairs_placement_names_the_routing_value_its_document_derived() {
    let (engine, _) = paired(true);
    let lists = Lists::of(1, 1);
    let classes = lists.every();
    let pair = lists.under(&classes, &Open);
    let request = PlaceRequest {
        request_id: Some("request-1"),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    let placement = engine.place(&request).expect("a placement");
    assert_eq!(placement.routing_key, Some("request-1"));

    // Under a document that derives none the request's id is not one.
    let (engine, _) = paired(false);
    let lists = Lists::of(1, 1);
    let classes = lists.every();
    let pair = lists.under(&classes, &Open);
    let request = PlaceRequest {
        request_id: Some("request-1"),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    let placement = engine.place(&request).expect("a placement");
    assert_eq!(placement.routing_key, None);
}

/// A pair whose caller asks its decode again is placed with its prefill
/// booked alone where the document picks the decode after the prefill: the
/// decode the decision named stands chosen, and it is issued when its
/// caller sends it. Where the document picks both sides at once the same
/// request books both at its decision. Such a request is not plain: the
/// word is a pair's.
///
/// The classic router picks a pair's decode with its prefill, one worker a
/// role in one selection, and asks no decode again
/// (`model_gateway/src/routers/common/placement.rs:1181-1182`,
/// `:1227-1249`).
#[test]
fn a_pair_whose_caller_asks_its_decode_again_books_its_prefill_alone() {
    let issued = |worker| ("issued at 0".to_owned(), worker);
    let sent = |worker| ("dispatched at 0".to_owned(), worker);
    for (pick, later) in [
        (DecodePick::AfterPrefill, true),
        (DecodePick::WithPrefill, false),
    ] {
        let (engine, heard) = paired_picking(false, pick);
        let lists = Lists::of(1, 2);
        let classes = lists.every();
        let pair = lists.under(&classes, &Open);
        let request = PlaceRequest {
            decode_again: true,
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
        };
        assert_eq!(Made::new(engine, &request, &[]).plain(), None);
        let mut placement = engine.place(&request).expect("a placement");
        assert_eq!(lists.places(&placement), (0, 0), "{pick:?}");
        let (prefill, decode) = (lists.prefill_id(0), lists.decode_id(0));
        let at_decision = if later {
            vec![issued(prefill)]
        } else {
            vec![issued(prefill), issued(decode)]
        };
        assert_eq!(said(&heard), at_decision, "{pick:?}");
        placement.sent(Side::Prefill);
        placement.sent(Side::Decode);
        let at_sends = if later {
            vec![issued(prefill), sent(prefill), issued(decode), sent(decode)]
        } else {
            vec![issued(prefill), issued(decode), sent(prefill), sent(decode)]
        };
        assert_eq!(said(&heard), at_sends, "{pick:?}");
        drop(placement);
        assert_eq!((engine.faults(), engine.unheard()), (0, 0));

        // Where no target holds, what a decision booked is released and
        // no more: the prefill alone where the decode stood chosen. The
        // call makes three such decisions over lists that hold no other
        // worker.
        let (engine, heard) = paired_picking(false, pick);
        let lists = Lists::of(1, 2);
        let classes = lists.every();
        for decode in lists.decodes.iter() {
            decode.set_status(WorkerStatus::NotReady);
        }
        let pair = lists.under(&classes, &Open);
        let request = PlaceRequest {
            decode_again: true,
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
        };
        assert_eq!(engine.place(&request).err(), Some(Unplaced::NoWorkers));
        let (prefill, decode) = (lists.prefill_id(0), lists.decode_id(0));
        let released = |worker| ("released CommitFailed".to_owned(), worker);
        let given_up = if later {
            vec![issued(prefill), released(prefill)]
        } else {
            vec![
                issued(prefill),
                issued(decode),
                released(prefill),
                released(decode),
            ]
        };
        let thrice = [given_up.clone(), given_up.clone(), given_up].concat();
        assert_eq!(said(&heard), thrice, "{pick:?}");
        assert_eq!((engine.faults(), engine.unheard()), (0, 0));
    }
}

/// A pair's decision that panics is refused, as a single request's is, and
/// costs its request alone: the next request over the same lists is placed.
#[test]
fn a_pairs_decision_that_panics_is_refused() {
    let (engine, _) = paired(false);
    let lists = Lists::of(1, 1);
    let classes = lists.every();
    let failing = Gates(|_| panic!("the caller's admission failed"), |_| true);
    let pair = lists.under(&classes, &failing);
    let request = PlaceRequest::of(MODEL, WorkerLists::Pair(pair));
    assert_eq!(
        engine.place(&request).err(),
        Some(Unplaced::Refused(DecisionRefused::Faulted))
    );
    assert_eq!(engine.faults(), 1);

    let pair = lists.under(&classes, &Open);
    let request = PlaceRequest::of(MODEL, WorkerLists::Pair(pair));
    let placement = engine.place(&request).expect("a placement");
    assert_eq!(lists.places(&placement), (0, 0));
}

/// The route's own rule is asked of both workers of a pair's target, of
/// each one that is ready and of no other: a target one of whose workers
/// it refuses is passed over whole, like one with a worker that is not
/// ready. A decode it refuses leaves the decision's next target. A prefill
/// it refuses fails every target of the decision, and the call decides
/// again past it.
///
/// The classic router narrows both of a pair's lists by its request's rule
/// ahead of its selection
/// (`model_gateway/src/routers/common/placement.rs:1342-1345`). Both send
/// to no worker the rule refuses.
#[test]
fn the_routes_rule_is_asked_of_both_workers_of_a_pairs_target() {
    let (engine, _) = paired(false);
    let lists = Lists::of(2, 3);
    let classes = lists.every();
    let pair = lists.under(&classes, &Open);
    // The first decode is not ready, and the rule refuses the second: the
    // decision's third target holds.
    lists.decodes[0].set_status(WorkerStatus::NotReady);
    let asked = RefCell::new(Vec::new());
    let refused = Cell::new(lists.decode_id(1));
    let admits = |worker: &dyn Worker| {
        asked.borrow_mut().push(worker_id(worker));
        worker_id(worker) != refused.get()
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    let placement = engine.place(&request).expect("a placement");
    assert_eq!(lists.places(&placement), (0, 2), "the third target");
    let prefill = lists.prefill_id(0);
    assert_eq!(
        *asked.borrow(),
        [
            prefill,
            prefill,
            lists.decode_id(1),
            prefill,
            lists.decode_id(2)
        ],
        "a target's two workers, each one that is ready"
    );
    drop(placement);

    // The rule refuses the first prefill, which every target of the first
    // decision names: the call's second decision names the other.
    lists.decodes[0].set_status(WorkerStatus::Ready);
    refused.set(prefill);
    let placement = engine.place(&request).expect("the call's second ask");
    assert_eq!(lists.places(&placement), (1, 0), "past the refused prefill");
    drop(placement);
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A pair's call makes three decisions at most: none of them is held to
/// the workers a route's rule takes. Over lists whose first three prefills
/// cannot be sent to, each decision names the next prefill, each is given
/// up by name, and the request has no worker though a fourth prefill could
/// be sent to.
#[test]
fn a_pairs_call_makes_three_decisions_at_most() {
    let (engine, heard) = paired(false);
    let lists = Lists::of(4, 1);
    let classes = lists.every();
    for prefill in &lists.prefills[..3] {
        prefill.set_status(WorkerStatus::NotReady);
    }
    let pair = lists.under(&classes, &Open);
    let request = PlaceRequest::of(MODEL, WorkerLists::Pair(pair));
    assert_eq!(engine.place(&request).err(), Some(Unplaced::NoWorkers));
    let decode = lists.decode_id(0);
    let given_up = |prefill: usize| {
        let prefill = lists.prefill_id(prefill);
        [
            ("issued at 0".to_owned(), prefill),
            ("issued at 0".to_owned(), decode),
            ("released CommitFailed".to_owned(), prefill),
            ("released CommitFailed".to_owned(), decode),
        ]
    };
    assert_eq!(
        said(&heard),
        [given_up(0), given_up(1), given_up(2)].concat()
    );
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A route's rule that panics passes its panic to the call's caller, and
/// the decision it was asked under is given up by name as the panic
/// passes: its lease is released once, and the engine is owed nothing. The
/// same holds for a single request's whole decision, and for a rule that
/// panics where a pair's decision asks it.
#[test]
fn a_rule_that_panics_loses_no_lease() {
    let failing = |_: &dyn Worker| -> bool { panic!("the route's rule failed") };
    let passed = |placed: &dyn Fn() -> bool| catch_unwind(AssertUnwindSafe(placed)).is_err();

    let (engine, heard) = paired(false);
    let lists = Lists::of(1, 1);
    let classes = lists.every();
    let pair = lists.under(&classes, &Open);
    let request = PlaceRequest {
        admits: Some(&failing),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    assert!(passed(&|| engine.place(&request).is_ok()));
    let (prefill, decode) = (lists.prefill_id(0), lists.decode_id(0));
    assert_eq!(
        said(&heard),
        [
            ("issued at 0".to_owned(), prefill),
            ("issued at 0".to_owned(), decode),
            ("released CommitFailed".to_owned(), prefill),
            ("released CommitFailed".to_owned(), decode),
        ]
    );
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));

    let (engine, heard) = in_order(false);
    let list = pool(2);
    let members = PublishedList::new(Arc::clone(&list));
    let request = PlaceRequest {
        admits: Some(&failing),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(members.lent()))
    };
    assert!(passed(&|| engine.place(&request).is_ok()));
    let first = worker_id(list[0].as_ref());
    assert_eq!(heard.lock().as_slice(), given_up(first));
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));

    // A pair whose sides read their record of sent prefixes asks the rule
    // inside its decision. This rule panics at its first question alone:
    // a panic swallowed there would leave the walk a rule that admits.
    let mut document = PolicySpec::pair_presets("cache_aware", "cache_aware").expect("presets");
    document.generation = 1;
    let spec = AppliedSpec::validate(&document).expect("a valid document");
    let engine = hosted(PolicyEngine::new(spec));
    let lists = Lists::of(1, 1);
    let classes = lists.every();
    let tokens: Vec<u32> = (1_000..5_096).collect();
    let asked = Cell::new(false);
    let once = |_: &dyn Worker| -> bool {
        if asked.replace(true) {
            return true;
        }
        panic!("the route's rule failed")
    };
    let request = PlaceRequest {
        prompt: Prompt {
            tokens: Some(&tokens),
            ..Prompt::NONE
        },
        admits: Some(&once),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(lists.under(&classes, &Open)))
    };
    assert!(passed(&|| engine.place(&request).is_ok()));
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A pair's decode decided again is asked the route's own rule too, as its
/// caller states it at that call: a decode the rule refuses is passed over
/// like one that is not ready.
///
/// The classic router picks no decode again: its one selection narrows the
/// decodes by the request's rule ahead of the pick
/// (`model_gateway/src/routers/common/placement.rs:1342-1345`).
#[test]
fn a_decode_decided_again_is_asked_the_routes_rule() {
    let (engine, _) = paired_picking(false, DecodePick::AfterPrefill);
    let lists = Lists::of(1, 3);
    let classes = lists.every();
    let refused = Cell::new(None);
    let admits = |worker: &dyn Worker| Some(worker_id(worker)) != refused.get();
    let request = PlaceRequest {
        decode_again: true,
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(lists.under(&classes, &Open)))
    };
    let mut placement = engine.place(&request).expect("a placement");
    assert_eq!(lists.places(&placement), (0, 0));
    placement.sent(Side::Prefill);
    refused.set(Some(lists.decode_id(0)));
    assert_eq!(placement.decode_again(&request), Ok(()));
    assert_eq!(lists.places(&placement), (0, 1), "past the refused decode");
}

/// A pair whose caller lends its ring has each side placed on the first
/// worker of its own list that the ring's walk reaches from where the
/// routing value stands, as the gateway's own policy takes each side from
/// the pair's one ring (`model_gateway/src/routers/common/placement.rs`,
/// lines 1498 to 1583 at 882be6f4f).
#[test]
fn a_pair_that_lends_its_ring_has_each_side_placed_where_its_walk_reaches() {
    let document = PolicySpec::pair_presets("consistent_hashing", "consistent_hashing");
    let engine = hosted(engine_of(&document.expect("the pair presets")));
    let lists = Lists::of(3, 3);
    let classes = lists.every();
    // The ring names the six workers, the two lists mixed: a prefill or a
    // decode, by its place in its list.
    let order = [
        (false, 1),
        (true, 2),
        (false, 0),
        (true, 0),
        (true, 1),
        (false, 2),
    ];
    let address = |&(prefill, at): &(bool, usize)| {
        let list = if prefill {
            &lists.prefills
        } else {
            &lists.decodes
        };
        list[at].url().to_owned()
    };
    let ring = Lent(order.iter().map(address).collect());
    let headers = [("x-smg-routing-key", "conv-1")];
    let request = PlaceRequest {
        headers: &headers,
        ring: Some(&ring),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(lists.under(&classes, &Open)))
    };
    let placement = engine.place(&request).expect("a placement");
    // Each side's first worker along the ring from where the key stands.
    let start = (ring.position("conv-1") % 6) as usize;
    let first = |prefill: bool| {
        let mut along = order.iter().cycle().skip(start).take(6);
        along.find(|side| side.0 == prefill).expect("a worker").1
    };
    assert_eq!(lists.places(&placement), (first(true), first(false)));
}

/// A pair whose request states a hard demand is placed on the workers of
/// its two lists that meet it, whatever the pick's order names first, and
/// its decode decided again is held to the decodes that meet it. A demand
/// one side leaves nobody for leaves the request no worker: it is shed
/// with no endpoint under that demand's stage, and nothing is issued.
///
/// The classic router holds both of a pair's pools to the request's own
/// rule (`model_gateway/src/routers/common/placement.rs:1227`, the rule's
/// read at `:1262-1266`, at main `b8c15d40a`).
#[test]
fn a_pair_that_states_a_demand_is_placed_on_the_workers_that_meet_it() {
    let (engine, heard) = paired_picking(false, DecodePick::AfterPrefill);
    let mut lists = Lists::of(2, 2);
    // The second worker of each list takes images.
    let seeing = |list: &Arc<[Arc<dyn Worker>]>| {
        let second = Arc::clone(&list[1]);
        PublishedList::with_facts(Arc::clone(list), |worker| FixedFacts {
            vision: Arc::ptr_eq(worker, &second),
            ..FixedFacts::default()
        })
    };
    lists.published = [seeing(&lists.prefills), seeing(&lists.decodes)];
    let classes = lists.every();
    let stating = |demands: Demands| PlaceRequest {
        demands,
        decode_again: true,
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(lists.under(&classes, &Open)))
    };
    // With no demand the picker's order names the first worker of each.
    let plain = stating(Demands::default());
    let placement = engine.place(&plain).expect("a placement");
    assert_eq!(lists.places(&placement), (0, 0));
    drop(placement);

    let pixels = stating(Demands {
        vision: true,
        ..Demands::default()
    });
    let mut placement = engine.place(&pixels).expect("a placement");
    assert_eq!(lists.places(&placement), (1, 1));
    placement.sent(Side::Prefill);
    assert_eq!(placement.decode_again(&pixels), Ok(()));
    assert_eq!(lists.places(&placement), (1, 1));
    drop(placement);

    // No decode states a way to decide a constraint: the decodes are left
    // nobody, and a prefill is not held to it.
    let from = said(&heard).len();
    let decided = stating(Demands {
        decoding: 3,
        ..Demands::default()
    });
    let none = Unplaced::Shed {
        reason: ShedReason::NoEndpoints,
        emptied_by: "decoding",
        retry_after_secs: 1,
    };
    assert_eq!(engine.place(&decided).err(), Some(none));
    assert_eq!(said(&heard).len(), from, "a shed issues nothing");
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A pair's record of sent prefixes names the workers its request went to.
/// Where the route's rule refuses the decode of the pick's first target,
/// the walk takes a later target or the call's next decision, and the
/// prompt's next requests go to the pair that one was sent to. A decision
/// asks the rule of a worker once: a decode refused at its first question
/// alone is passed by the request and by the record.
#[test]
fn a_prompt_sent_past_a_refused_decode_goes_to_its_pair_again() {
    let mut document = PolicySpec::pair_presets("cache_aware", "cache_aware").expect("presets");
    document.generation = 1;
    let spec = AppliedSpec::validate(&document).expect("a valid document");
    let engine = hosted(PolicyEngine::new(spec));
    let lists = Lists::of(2, 4);
    let classes = lists.every();
    let pair = lists.under(&classes, &Open);
    let tokens: Vec<u32> = (1_000..5_096).collect();
    let prompt = Prompt {
        tokens: Some(&tokens),
        ..Prompt::NONE
    };
    // The rule refuses the first decode it is asked of, at that question
    // alone: asked of it again, it would take it.
    let refused = RefCell::new(None::<String>);
    let rule = |worker: &dyn Worker| {
        if *worker.worker_type() != WorkerType::Decode {
            return true;
        }
        let mut refused = refused.borrow_mut();
        if refused.is_some() {
            return true;
        }
        *refused = Some(worker.url().to_owned());
        false
    };
    let refusing = PlaceRequest {
        prompt,
        admits: Some(&rule),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    let sent = engine.place(&refusing).expect("a placement");
    let to = lists.places(&sent);
    drop(sent);
    let passed = refused.borrow().clone().expect("the rule refused a decode");
    assert_ne!(lists.decodes[to.1].url(), passed, "past the refused decode");
    let stated = PlaceRequest {
        prompt,
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    let mut went = Vec::new();
    for _ in 0..16 {
        let again = engine.place(&stated).expect("a placement");
        went.push(lists.places(&again));
        drop(again);
    }
    assert_eq!(went, [to; 16], "the pair the prompt was sent to");
}
