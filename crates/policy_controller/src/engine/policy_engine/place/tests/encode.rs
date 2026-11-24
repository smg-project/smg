//! The cases of a request with an encode request's three lists.

use std::panic::{catch_unwind, AssertUnwindSafe};

use super::{
    super::encode::{ask, PLACED},
    *,
};
use crate::{
    engine::{decision_engine::DecodeRefused, EncoderFacts},
    tracking::lease::TransportClass,
};

/// The caller's admission as one test of a prefill's place: every decode
/// has room.
#[derive(Debug)]
struct Capacity(fn(usize) -> bool);

impl PairAdmission for Capacity {
    fn has_capacity(&self, prefill: usize) -> bool {
        (self.0)(prefill)
    }

    fn has_room(&self, _: usize) -> bool {
        true
    }
}

/// A picker of the encode side that places an item by its value: the item
/// `last` on the last worker the filters left, every other on the first.
#[derive(Debug)]
struct ByItem;

impl Lifecycle for ByItem {}

impl Picker for ByItem {
    fn name(&self) -> &'static str {
        "by-item"
    }

    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        _: &[f32],
        _: TieBand,
        out: &mut Targets,
    ) {
        let last = facts.session_key.plain == Some("last");
        let at = if last { kept.last() } else { kept.first() };
        push_drawn(candidates, at.map_or(&[], std::slice::from_ref), out);
    }
}

/// The encode document of the classic router's recorded lines: the
/// `power_of_two` preset on each of the three sides, each picking the
/// worker that holds the fewest requests, with a pair's decode picked when
/// `decode` says.
fn fewest(decode: DecodePick) -> PolicySpec {
    let preset = "power_of_two";
    let mut document = PolicySpec::epd_presets(preset, preset, preset).expect("the presets");
    document.plans.pd_pair.decode = decode;
    for profile in document.profiles.values_mut() {
        profile.pick = "count-first".into();
    }
    document
}

/// [`fewest`] without the `healthy` filter of any side, so that a worker
/// that cannot be sent to stays a candidate of the decision.
fn keeping() -> PolicySpec {
    let mut document = fewest(DecodePick::default());
    for profile in document.profiles.values_mut() {
        profile
            .filter
            .retain(|filter| *filter != FilterSpec::Name("healthy".to_owned()));
    }
    document
}

/// An engine under `document`, and what its stages heard of each lease.
/// One more profile, which no binding names, picks by [`InOrder`]: every
/// profile's stages hear every lease, so that picker hears each word once
/// and picks for no request. The encode side's [`ByItem`] is at hand under
/// its name.
fn hearing(mut document: PolicySpec) -> (&'static PolicyEngine, Heard) {
    let heard = Heard::default();
    let kept = Arc::clone(&heard);
    let mut stages = StageRegistry::builtin();
    let added = stages.picker("in-order", move || {
        Box::new(InOrder {
            heard: Arc::clone(&kept),
            budgets: Budgets::default(),
            panics: false,
        })
    });
    added.expect("a free name");
    let added = stages.picker("by-item", || Box::new(ByItem));
    added.expect("a free name");
    let mut random = PolicySpec::preset("random").expect("a built-in preset");
    let listener = random.profiles.remove("random");
    let mut listener = listener.expect("the preset's profile");
    listener.pick = "in-order".into();
    document.profiles.insert("listener".into(), listener);
    document.generation = 1;
    let spec = AppliedSpec::validate_in(&document, &stages, &[]);
    let engine = PolicyEngine::new(spec.expect("the stages hold the pickers"));
    (hosted(engine), heard)
}

/// What a caller says of its encoders, one entry an encoder, and the encode
/// class of each of its prefills.
type Said<'a> = (&'a [EncoderFacts], &'a [u32]);

/// What an encoder of the one encode class states: no push, no references.
const PLAIN: EncoderFacts = EncoderFacts {
    class: 1,
    pushes: false,
    references: false,
};

/// A caller's three lists of ready workers as it publishes them, each with
/// its cell: the prefills (list `0`), the decodes (`1`) and the encoders
/// (`2`). Every prefill pairs with every decode.
struct Three {
    workers: [Arc<[Arc<dyn Worker>]>; 3],
    published: [PublishedList; 3],
    partners: PartnerClasses,
}

impl Three {
    /// Lists of `counts` workers. `sees` says of a worker, by its list and
    /// its place there, whether it takes images.
    fn of(counts: [usize; 3], sees: fn(usize, usize) -> bool) -> Self {
        let named = ["prefill", "decode", "encoder"];
        let workers = [0, 1, 2].map(|list| -> Arc<[Arc<dyn Worker>]> {
            let ready = |at| ready_worker(&format!("http://{}-{at}:8000", named[list]), MODEL);
            (0..counts[list]).map(ready).collect()
        });
        let published = [0, 1, 2].map(|list| {
            // Asked once a worker, in the list's order.
            let mut places = 0..;
            let facts_of = |_: &Arc<dyn Worker>| FixedFacts {
                vision: places.next().is_some_and(|place| sees(list, place)),
                ..FixedFacts::default()
            };
            PublishedList::with_facts(Arc::clone(&workers[list]), facts_of)
        });
        let all: Vec<u32> = (0..counts[1] as u32).collect();
        let partners = PartnerClasses::number(counts[1], (0..counts[0]).map(|_| &all[..]));
        Self {
            workers,
            published,
            partners,
        }
    }

    /// The lists as their caller hands them for one request, with what it
    /// says of them, under an admission that admits every worker.
    fn lists<'a>(&'a self, said: Said<'a>) -> EncodeLists<'a> {
        let [prefill, decode, encoders] = &self.published;
        let pair = PairLists {
            decode: decode.lent(),
            ..pair_of(prefill, &self.partners)
        };
        EncodeLists {
            pair,
            encoders: encoders.lent(),
            encoders_places: None,
            encoders_stalled: &[],
            encoders_held: &[],
            facts: said.0,
            prefill_classes: said.1,
        }
    }

    /// The id of the worker at `place` of list `list`.
    fn id(&self, list: usize, place: usize) -> WorkerId {
        worker_id(self.workers[list][place].as_ref())
    }

    /// The places in the three lists of the workers `placement` names: its
    /// prefill, its decode and each item's encoder.
    fn places(&self, placement: &Placement<'_>) -> (usize, usize, Vec<usize>) {
        let at = |list: usize, worker: &Arc<dyn Worker>| {
            let mut listed = self.workers[list].iter();
            let place = listed.position(|listed| Arc::ptr_eq(listed, worker));
            place.expect("a worker of the list")
        };
        let decode = placement.decode.as_ref().expect("a pair's decode");
        let encoders = placement.encoders.workers().iter();
        (
            at(0, &placement.worker),
            at(1, decode),
            encoders.map(|encoder| at(2, encoder)).collect(),
        )
    }
}

/// A request of `items` over `lists`.
fn asking<'a>(lists: EncodeLists<'a>, items: EncodeItems<'a>) -> PlaceRequest<'a> {
    PlaceRequest {
        encode: Some(items),
        ..PlaceRequest::of(MODEL, WorkerLists::Encode(lists))
    }
}

/// The word a stage hears where a side's lease is issued on `worker`.
fn issued(worker: WorkerId) -> (String, WorkerId) {
    ("issued at 0".to_owned(), worker)
}

/// The stage a request with no worker for it was shed under.
fn gap(answer: Result<Placement<'_>, Unplaced>) -> Option<&'static str> {
    match answer.err() {
        Some(Unplaced::Shed {
            reason: ShedReason::NoEndpoints,
            emptied_by,
            ..
        }) => Some(emptied_by),
        _ => None,
    }
}

/// A recorded line of the classic router through the one call: one encoder,
/// one prefill, and two decodes of which the second takes images and holds
/// four requests. A request whose items are references and that states an
/// image is placed on the second decode. It keeps one decision, which books
/// both its sides, though its caller asks the decode again under a document
/// that picks the decode after the prefill: a decode asked again reads no
/// image fact. So that decode never stood chosen, and the placement refuses
/// to decide it again. The same items with no image stated take the decode
/// that holds fewer requests. That request's prefill alone is booked, and
/// its decode is decided again on the placement, over the pair's two of the
/// three lists, with the item's encoder kept. References beside a video and
/// no image are booked as those beside an image.
///
/// The classic router prefers a decode that takes images for such a
/// request where a pair has one
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2208-2212`
/// on main 3d0315f3d). Its answers are recorded in the line
/// `references-with-images-prefer-a-decode-that-takes-images`
/// (`crates/policy_controller/tests/classic/sheets/encode_pair.json:3131-3199`):
/// prefill 0 with decode 1 and encoder 0, then prefill 0 with decode 0.
#[test]
fn references_beside_an_image_keep_one_decision_on_a_decode_that_takes_images() {
    let (engine, heard) = hearing(fewest(DecodePick::AfterPrefill));
    let three = Three::of([1, 2, 1], |list, place| {
        list == 2 || (list, place) == (1, 1)
    });
    for _ in 0..4 {
        three.workers[1][1].increment_load();
    }
    // The one encoder pushes, takes references and hands to the prefill.
    let pushes = [EncoderFacts {
        pushes: true,
        references: true,
        ..PLAIN
    }];
    let keys = [Identity::plain("01")];
    let items = EncodeItems {
        keys: &keys,
        references: true,
        ..EncodeItems::default()
    };
    let with_an_image = PlaceRequest {
        media: MediaCounts {
            images: 1,
            ..MediaCounts::NONE
        },
        decode_again: true,
        ..asking(three.lists((&pushes, &[1])), items)
    };
    let mut placement = engine.place(&with_an_image).expect("a placement");
    assert_eq!(three.places(&placement), (0, 1, vec![0]));
    assert!(!placement.encoders.one());
    let (prefill, takes_images) = (three.id(0, 0), three.id(1, 1));
    assert_eq!(*heard.lock(), [issued(prefill), issued(takes_images)]);
    placement.sent(Side::Prefill);
    let never_chosen = Unplaced::Refused(DecisionRefused::Decode(DecodeRefused::NotChosen));
    assert_eq!(placement.decode_again(&with_an_image), Err(never_chosen));
    drop(placement);
    heard.lock().clear();

    let with_none = PlaceRequest {
        media: MediaCounts::NONE,
        ..with_an_image
    };
    let mut placement = engine.place(&with_none).expect("a placement");
    assert_eq!(three.places(&placement), (0, 0, vec![0]));
    assert_eq!(*heard.lock(), [issued(prefill)]);
    placement.sent(Side::Prefill);
    assert_eq!(placement.decode_again(&with_none), Ok(()));
    assert_eq!(three.places(&placement), (0, 0, vec![0]));
    drop(placement);
    heard.lock().clear();

    // No request of the recorded line: one that asks images of its decode
    // is booked as the first is.
    let demanding = PlaceRequest {
        encode: Some(EncodeItems {
            references: false,
            decode_images: true,
            ..items
        }),
        ..with_none
    };
    let placement = engine.place(&demanding).expect("a placement");
    assert_eq!(three.places(&placement), (0, 1, vec![0]));
    assert_eq!(*heard.lock(), [issued(prefill), issued(takes_images)]);
    drop(placement);
    heard.lock().clear();

    // Nor is this one: the same references beside a video and no image. A
    // video holds the decode as an image does, so the request keeps one
    // decision that books both sides, and its decode never stood chosen.
    // The classic router asks vision for an image or a video alike
    // (`model_gateway/src/routers/grpc/multimodal/capability.rs:325-344`
    // on main d6517a4fe).
    let with_a_video = PlaceRequest {
        media: MediaCounts {
            video: 1,
            ..MediaCounts::NONE
        },
        ..with_an_image
    };
    let mut placement = engine.place(&with_a_video).expect("a placement");
    assert_eq!(three.places(&placement), (0, 1, vec![0]));
    assert_eq!(*heard.lock(), [issued(prefill), issued(takes_images)]);
    placement.sent(Side::Prefill);
    let never_chosen = Unplaced::Refused(DecisionRefused::Decode(DecodeRefused::NotChosen));
    assert_eq!(placement.decode_again(&with_a_video), Err(never_chosen));
}

/// An encode request is placed by one call: the placement names its
/// prefill and its decode, as a pair's does, and each item's encoder in
/// the items' order, each the list's own worker object. It names the
/// request's routing value and each side's profile as a pair's placement
/// does, and what the request keeps to its end names the encoders still.
/// The ids a next attempt states as tried are the pair's two, and no word
/// of the lease names an encoder. A placement that ended holds no encoder.
///
/// The classic router picks the pair, then one encoder an item among those
/// that hand to the chosen prefill
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2217-2229`,
/// `:2252-2265` on main 3d0315f3d). Its retry leaves out the workers of the
/// pair alone (`model_gateway/src/routers/grpc/pipeline.rs:848-859`, over
/// `model_gateway/src/routers/grpc/context.rs:1577-1581`).
#[test]
fn an_encode_request_is_placed_on_its_pair_and_on_each_items_encoder() {
    let mut document = fewest(DecodePick::default());
    let encode = document.profiles.get_mut("encode");
    encode.expect("the encode profile").pick = "by-item".into();
    // The request's own id is its routing value.
    let rid = KeySourceSpec {
        rid: true,
        ..KeySourceSpec::default()
    };
    document.bindings[0].routing_key = Some(RoutingKeySpec {
        sources: vec![rid],
        strip_lineage: false,
    });
    let (engine, heard) = hearing(document);
    let three = Three::of([1, 1, 3], |_, _| false);
    let (prefill, decode) = (three.id(0, 0), three.id(1, 0));
    // Each encoder's holders while no placement names it.
    let holders = || -> Vec<usize> { three.workers[2].iter().map(Arc::strong_count).collect() };
    let unnamed = holders();
    let orders = [
        (["first", "last"], vec![0, 2]),
        (["last", "first"], vec![2, 0]),
    ];
    for (values, encoders) in orders {
        let keys = values.map(Identity::plain);
        let items = EncodeItems {
            keys: &keys,
            ..EncodeItems::default()
        };
        let request = PlaceRequest {
            request_id: Some("request-1"),
            ..asking(three.lists((&[PLAIN; 3], &[1])), items)
        };
        let placement = engine.place(&request).expect("a placement");
        assert_eq!(three.places(&placement), (0, 0, encoders.clone()));
        assert!(!placement.encoders.one() && placement.owes_events());
        assert_eq!(placement.routing_key, Some("request-1"));
        let labels = (placement.profile, placement.decode_profile);
        assert_eq!(labels, ("prefill", Some("decode")));
        let tried = (placement.id(), placement.decode_id());
        assert_eq!(tried, (prefill, Some(decode)));
        assert_ne!(holders(), unnamed);
        assert_eq!(three.places(&placement.kept()).2, encoders);
        assert_eq!(holders(), unnamed);
    }
    let heard = heard.lock();
    assert_eq!(heard[..2], [issued(prefill), issued(decode)]);
    assert!(heard
        .iter()
        .all(|(_, worker)| [prefill, decode].contains(worker)));
}

/// Items their caller hands as one are placed on one encoder where the
/// encoders of the prefill's class push, and the placement says so. Where
/// they do not push, each item is placed on its own.
///
/// The classic router places one digest in the items' stead under a push
/// handoff, for a request that states no fan-out
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2238-2251`
/// on main 3d0315f3d).
#[test]
fn items_handed_as_one_are_placed_on_one_encoder_and_the_placement_says_so() {
    let (engine, _) = hearing(fewest(DecodePick::default()));
    let three = Three::of([1, 1, 2], |_, _| false);
    // The second encoder holds fewer requests.
    three.workers[2][0].increment_load();
    let keys = [Identity::plain("a"), Identity::plain("b")];
    let items = EncodeItems {
        keys: &keys,
        one: Some(Identity::plain("a-and-b")),
        ..EncodeItems::default()
    };
    let pushing = [EncoderFacts {
        pushes: true,
        ..PLAIN
    }; 2];
    let request = asking(three.lists((&pushing, &[1])), items);
    let placement = engine.place(&request).expect("a placement");
    assert_eq!(three.places(&placement), (0, 0, vec![1]));
    assert!(placement.encoders.one());

    let request = asking(three.lists((&[PLAIN; 2], &[1])), items);
    let placement = engine.place(&request).expect("a placement");
    assert_eq!(three.places(&placement), (0, 0, vec![1, 1]));
    assert!(!placement.encoders.one());
}

/// An item has one encoder and no second. Where the encoder a decision
/// named cannot be sent to, held back by its circuit breaker, the decision
/// is given up by name at once and the stages hear the release of both the
/// pair's sides. The call then decides again with that encoder left out,
/// and the item is placed on the next one. Where no encoder can be sent
/// to, the request has no worker behind the call's third decision. A side
/// of the pair that cannot be sent to either is left out by the same
/// decision, which takes no target.
///
/// The classic router offers an item the encoders that are available alone
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2132`
/// on main 3d0315f3d), and answers that none is where none is left
/// (`:2150-2153`).
#[test]
fn an_item_whose_encoder_cannot_be_sent_to_is_placed_by_the_calls_next_decision() {
    let (engine, heard) = hearing(keeping());
    let three = Three::of([1, 1, 2], |_, _| false);
    // The first encoder holds fewer requests, and its breaker is open.
    three.workers[2][1].increment_load();
    trip(&three.workers[2][0]);
    let keys = [Identity::plain("item")];
    let items = EncodeItems {
        keys: &keys,
        ..EncodeItems::default()
    };
    let request = asking(three.lists((&[PLAIN; 2], &[1])), items);
    let placement = engine.place(&request).expect("a placement");
    assert_eq!(three.places(&placement), (0, 0, vec![1]));
    let (prefill, decode) = (three.id(0, 0), three.id(1, 0));
    let released = |worker| ("released CommitFailed".to_owned(), worker);
    let given_up = [
        issued(prefill),
        issued(decode),
        released(prefill),
        released(decode),
    ];
    assert_eq!(heard.lock()[..4], given_up);
    assert_eq!(heard.lock()[4..], [issued(prefill), issued(decode)]);
    drop(placement);
    heard.lock().clear();

    trip(&three.workers[2][1]);
    assert_eq!(engine.place(&request).err(), Some(Unplaced::NoWorkers));
    let thrice: Vec<_> = (0..3).flat_map(|_| given_up.clone()).collect();
    assert_eq!(*heard.lock(), thrice);
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));

    // One ask: an encoder two items cannot be sent to joins the workers the
    // call missed once.
    let two = [Identity::plain("a"), Identity::plain("b")];
    let both = EncodeItems {
        keys: &two,
        ..EncodeItems::default()
    };
    let lists = three.lists((&[PLAIN; 2], &[1]));
    let mut call = Call::default();
    let asked = ask(
        engine.loaded(),
        &asking(lists, both),
        &lists,
        both,
        &mut call,
    );
    assert!(matches!(asked, Ok(None)), "no placement");
    assert_eq!(call.missed, [three.id(2, 0)]);

    // The same ask notes the prefill beside the encoder where neither can
    // be sent to.
    trip(&three.workers[0][0]);
    let mut call = Call::default();
    let asked = ask(
        engine.loaded(),
        &asking(lists, both),
        &lists,
        both,
        &mut call,
    );
    assert!(matches!(asked, Ok(None)), "no placement");
    assert_eq!(call.missed, [three.id(0, 0), three.id(2, 0)]);
}

/// The route's rule is asked of the pair's workers behind an encoder that
/// cannot be sent to as well, under the pair's lease. A rule that panics
/// there passes its panic to the call's caller, and the decision is given
/// up by name as the panic passes: its lease is released once, and the
/// engine is owed nothing.
#[test]
fn a_rule_that_panics_behind_an_encoder_that_cannot_be_sent_to_loses_no_lease() {
    let failing = |_: &dyn Worker| -> bool { panic!("the route's rule failed") };
    let (engine, heard) = hearing(keeping());
    let three = Three::of([1, 1, 1], |_, _| false);
    trip(&three.workers[2][0]);
    let keys = [Identity::plain("item")];
    let items = EncodeItems {
        keys: &keys,
        ..EncodeItems::default()
    };
    let request = PlaceRequest {
        admits: Some(&failing),
        ..asking(three.lists((&[PLAIN], &[1])), items)
    };
    let placed = catch_unwind(AssertUnwindSafe(|| engine.place(&request).is_ok()));
    assert!(placed.is_err(), "the panic passes");
    let (prefill, decode) = (three.id(0, 0), three.id(1, 0));
    let released = |worker| ("released CommitFailed".to_owned(), worker);
    let given_up = [
        issued(prefill),
        issued(decode),
        released(prefill),
        released(decode),
    ];
    assert_eq!(*heard.lock(), given_up);
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// The pair's targets of an encode request are walked as a pair's are. A
/// target whose decode cannot be sent to is passed over and the next is
/// taken, with the items' encoders beside it: every target of one decision
/// shares its prefill, which the encoders hand to. A prefill that cannot be
/// sent to holds no target.
///
/// The classic router hands no pair with a side that does not hold either
/// (`model_gateway/src/routers/common/placement.rs:1389-1394` on main
/// 3d0315f3d).
#[test]
fn an_encode_requests_targets_are_walked_with_its_encoders_kept() {
    let (engine, heard) = hearing(keeping());
    let three = Three::of([1, 2, 1], |_, _| false);
    // The first decode holds fewer requests, and its breaker is open.
    three.workers[1][1].increment_load();
    trip(&three.workers[1][0]);
    let keys = [Identity::plain("item")];
    let items = EncodeItems {
        keys: &keys,
        ..EncodeItems::default()
    };
    let request = asking(three.lists((&[PLAIN], &[1])), items);
    let mut placement = engine.place(&request).expect("a placement");
    assert_eq!(three.places(&placement), (0, 1, vec![0]));
    assert_eq!(placement.decode_id(), Some(three.id(1, 1)));
    placement.sent(Side::Decode);
    assert_eq!(
        *heard.lock(),
        [
            issued(three.id(0, 0)),
            issued(three.id(1, 0)),
            ("dispatched at 1".to_owned(), three.id(1, 1))
        ]
    );
    drop(placement);

    trip(&three.workers[0][0]);
    assert_eq!(engine.place(&request).err(), Some(Unplaced::NoWorkers));
}

/// What an encode request reports: its pair's two sides, as a pair does,
/// and nothing of an encode side. No lease stands for an encoder, so a
/// send and an end said under the encode side reach no stage, move no
/// worker's count and end no side. Both sides of the pair stay owed their
/// own words, and the stages hear each as a pair's.
///
/// The classic router tells no policy of an encode call either. The code
/// that sends the call counts it on its worker while it is in flight
/// (`model_gateway/src/routers/grpc/common/stages/encode.rs:190-196` on
/// main 384b8fd6b), so a placement that booked an encoder would count the
/// call twice.
#[test]
fn an_encode_sides_send_and_end_say_nothing_and_leave_the_pair_owed() {
    let (engine, heard) = hearing(fewest(DecodePick::default()));
    let three = Three::of([1, 1, 1], |_, _| false);
    let (prefill, decode) = (three.id(0, 0), three.id(1, 0));
    let keys = [Identity::plain("item")];
    let items = EncodeItems {
        keys: &keys,
        ..EncodeItems::default()
    };
    let request = asking(three.lists((&[PLAIN], &[1])), items);
    let mut placement = engine.place(&request).expect("a placement");
    assert_eq!(three.places(&placement), (0, 0, vec![0]));
    let said = || heard.lock().clone();
    let counts = || [0, 1, 2].map(|list| three.workers[list][0].load());
    let (at_issue, booked) = (said(), counts());
    assert_eq!(at_issue, [issued(prefill), issued(decode)]);
    assert_eq!(booked[2], 0, "nothing is booked on an encoder");
    // The end of a side that was served, as its caller says it.
    let served = Outcome {
        side: Side::Single,
        served: WorkerId(0),
        status: 200,
        class: TransportClass::Ok,
        ttft_micros: 0,
        tokens_in: 0,
        tokens_out: 0,
        duration_micros: 0,
    };
    placement.sent(Side::Encode);
    placement.ended(Side::Encode, served);
    assert_eq!((said(), counts()), (at_issue, booked));
    assert!(placement.owes_events(), "both sides of the pair are owed");
    placement.sent(Side::Prefill);
    placement.sent(Side::Decode);
    placement.ended(Side::Prefill, served);
    placement.ended(Side::Decode, served);
    assert!(!placement.owes_events());
    let word = |heard: &str, worker| (heard.to_owned(), worker);
    assert_eq!(
        said()[2..],
        [
            word("dispatched at 0", prefill),
            word("dispatched at 0", decode),
            word("completed Ok", prefill),
            word("completed Ok", decode),
        ]
    );
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// An encode request's decision that sheds is answered its shed by name,
/// and no lease is left behind. With no encoder ready the request has no
/// worker under the encode side's stage. With no decode ready, and where it
/// states a demand, it is answered as its pair is: a pair's decision
/// narrows no list by a demand.
///
/// The classic router answers a request no encoder is available for ahead
/// of its pair
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2150-2153`
/// on main 3d0315f3d), and the pair's own failure behind it (`:2217-2229`).
#[test]
fn an_encode_requests_shed_is_answered_by_its_name() {
    let (engine, heard) = hearing(fewest(DecodePick::default()));
    let three = Three::of([1, 1, 1], |_, _| false);
    let keys = [Identity::plain("item")];
    let items = EncodeItems {
        keys: &keys,
        ..EncodeItems::default()
    };
    let request = asking(three.lists((&[PLAIN], &[1])), items);
    let demanding = PlaceRequest {
        demands: Demands {
            vision: true,
            ..Demands::default()
        },
        ..request
    };
    assert_eq!(gap(engine.place(&demanding)), Some("wires"));
    three.workers[2][0].set_status(WorkerStatus::NotReady);
    assert_eq!(gap(engine.place(&request)), Some("encoders"));
    three.workers[2][0].set_status(WorkerStatus::Ready);
    three.workers[1][0].set_status(WorkerStatus::NotReady);
    let as_a_pair = PlaceRequest {
        encode: None,
        ..request
    };
    assert_eq!(gap(engine.place(&as_a_pair)), Some("healthy"));
    assert_eq!(gap(engine.place(&request)), Some("healthy"));
    assert!(heard.lock().is_empty(), "a shed issues no lease");
}

/// Three lists with no item stated are placed as their pair: no encoder is
/// read, the placement names none, and a decode its caller asks again is
/// the pair's word over them.
///
/// The classic router reads no encoder for a request with no item
/// (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2131`,
/// `:2150` on main 3d0315f3d).
#[test]
fn three_lists_with_no_item_are_placed_as_their_pair() {
    let (engine, heard) = hearing(fewest(DecodePick::AfterPrefill));
    // No encoder stands in the third list, and nothing is said of one.
    let three = Three::of([1, 1, 0], |_, _| false);
    let request = PlaceRequest::of(MODEL, WorkerLists::Encode(three.lists((&[], &[]))));
    let placement = engine.place(&request).expect("a placement");
    assert_eq!(three.places(&placement), (0, 0, vec![]));
    assert!(!placement.encoders.one());
    let (prefill, decode) = (three.id(0, 0), three.id(1, 0));
    assert_eq!(*heard.lock(), [issued(prefill), issued(decode)]);
    drop(placement);
    heard.lock().clear();

    let again = PlaceRequest {
        decode_again: true,
        ..request
    };
    assert_eq!(Made::new(engine, &again, &[]).plain(), None);
    let placement = engine.place(&again).expect("a placement");
    assert_eq!(three.places(&placement), (0, 0, vec![]));
    assert_eq!(*heard.lock(), [issued(prefill)]);
}

/// Prefills that are all at their caller's admission limit answer the word
/// on which the caller waits for room, as a pair's do, and nothing is
/// issued. A decision that panics is refused and costs its request alone:
/// the next request over the same lists is placed. So is one placed while
/// this thread's list of encoders is lent, as a stage places a request from
/// inside a decision: it takes a list of its own.
///
/// The classic router answers the same way where a prefill at its limit
/// has a partner that can be sent to
/// (`model_gateway/src/routers/common/placement.rs:1470-1480` on main
/// 3d0315f3d).
#[test]
fn an_encode_request_waits_for_a_prefills_room_and_a_panic_costs_it_alone() {
    let (engine, heard) = hearing(fewest(DecodePick::default()));
    let three = Three::of([1, 1, 1], |_, _| false);
    let keys = [Identity::plain("item")];
    let items = EncodeItems {
        keys: &keys,
        ..EncodeItems::default()
    };
    let open = three.lists((&[PLAIN], &[1]));
    let under = |admission: &dyn PairAdmission| {
        let pair = PairLists {
            admission,
            ..open.pair
        };
        let lists = EncodeLists { pair, ..open };
        engine.place(&asking(lists, items)).err()
    };
    assert_eq!(under(&Capacity(|_| false)), Some(Unplaced::PrefillsFull));
    assert!(heard.lock().is_empty(), "a shed issues no lease");
    let failing = Capacity(|_| panic!("the caller's admission failed"));
    let refused = Unplaced::Refused(DecisionRefused::Faulted);
    assert_eq!(under(&failing), Some(refused));
    assert_eq!(engine.faults(), 1);

    let request = asking(open, items);
    let placement = PLACED.with(|placed| {
        let _lent = placed.borrow_mut();
        engine.place(&request)
    });
    let placement = placement.expect("a placement");
    assert_eq!(three.places(&placement), (0, 0, vec![0]));
}

/// An item is placed by its own key, whatever handed its request's key.
/// Under a ring's prefix form a routing value the request's own id handed
/// leads no prompt ([`RequestFacts::session_key_from_rid`]). That word says
/// how the REQUEST's key came, and an item's facts carry the item's key in
/// its place: so under a prefix ring on the encode side each item of an
/// id-keyed request goes to the encoder its key goes to with no id, beside
/// a prompt and under two ids. The keys have two encoders among them, so
/// items placed by the one prompt would not all be at home.
#[test]
fn an_id_keyed_requests_item_goes_where_its_key_goes_under_a_prefix_ring() {
    let side = "round_robin";
    let mut document = PolicySpec::epd_presets("prefix_hash", side, side).expect("the presets");
    // The request's own id is its routing value.
    let rid = KeySourceSpec {
        rid: true,
        ..KeySourceSpec::default()
    };
    document.bindings[0].routing_key = Some(RoutingKeySpec {
        sources: vec![rid],
        strip_lineage: false,
    });
    let spec = AppliedSpec::validate(&document).expect("a valid document");
    let engine = hosted(PolicyEngine::new(spec));
    let three = Three::of([1, 1, 8], |_, _| false);
    let ids: Vec<u32> = (500..800).collect();
    // The encoder of the one item `key` of a request whose own id is `id`.
    let encoder_of = |key: Identity<'_>, id: Option<&str>| {
        let keys = [key];
        let items = EncodeItems {
            keys: &keys,
            ..EncodeItems::default()
        };
        let request = PlaceRequest {
            prompt: Prompt {
                tokens: Some(&ids),
                ..Prompt::NONE
            },
            request_id: id,
            ..asking(three.lists((&[PLAIN; 8], &[1])), items)
        };
        let placement = engine.place(&request).expect("a placement");
        assert_eq!(placement.routing_key, id);
        three.places(&placement).2[0]
    };
    let keys = ["a", "b", "c", "d", "e", "f"].map(Identity::plain);
    let homes = keys.map(|key| encoder_of(key, None));
    assert!(homes.iter().any(|home| *home != homes[0]), "{homes:?}");
    for id in ["conv-1", "conv-2"] {
        assert_eq!(keys.map(|key| encoder_of(key, Some(id))), homes, "{id}");
    }
}
