//! The cases of an encode request that names its worker: its pair is
//! counted as a pair's is, a side at a time, by the number its request
//! carries under the binding's target source, and its items go to encoders
//! beside the pair the number reaches. The number names no encoder.
//!
//! A case that is read against the classic router names its lines: of
//! its encode stage
//! (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs`,
//! `WS` below), of its pair selection
//! (`model_gateway/src/routers/common/placement.rs`, `P`) and of its hash
//! policy (`model_gateway/src/policies/consistent_hashing.rs`, `CH`), on
//! main `86e4dfb33`. That stage hands the pair selection its rule of the
//! workers an encoder serves as the request's filter (`WS:2671-2674`,
//! `WS:2683`), and the selection counts the workers its filter takes
//! (`P:3208-3213`, `P:3249-3252`). Here the count asks the route's rule
//! ([`PlaceRequest::admits`]) and no encoder. One case says where the
//! engine answers otherwise: under a ring on the decode's side alone. One
//! says where its own device would: a rule that changes the worker it is
//! asked of. The cases of a header reader that panics and of a request
//! that names no worker hold the engine to its own word, and name no line.

use super::{
    super::target::{missed_on, Counted, MISSED, TARGET},
    *,
};
use crate::engine::References;

/// One item, under a key its host holds as a fingerprint.
const KEYS: [Identity<'static>; 1] = [Identity::hashed(1)];

/// The stock encode document over the pair of the presets `prefill` and
/// `decode`, its encoders under the preset `encode`, as `edit` changes
/// it. A side of the pair that is the hash ring brings the binding its
/// target source.
fn stock_of(
    [encode, prefill, decode]: [&str; 3],
    edit: impl FnOnce(&mut PolicySpec),
) -> &'static InProcess {
    let presets = PolicySpec::epd_presets(encode, prefill, decode);
    let mut document = presets.expect("presets");
    document.generation = 1;
    edit(&mut document);
    hosted(engine_of(&document))
}

/// [`stock_of`] with the encoders under the `power_of_two` preset, and the
/// document as its presets make it.
fn stock(prefill: &str, decode: &str) -> &'static InProcess {
    stock_of(["power_of_two", prefill, decode], |_| {})
}

/// `document` with an encode side that keeps a worker that cannot be sent
/// to, and picks the encoder that holds the fewest requests.
fn keeps_its_encoders(document: &mut PolicySpec) {
    let encode = document.profiles.get_mut("encode");
    let encode = encode.expect("the encode profile");
    let healthy = FilterSpec::Name("healthy".to_owned());
    encode.filter.retain(|filter| *filter != healthy);
    encode.pick = "count-first".into();
}

/// The stock encode document over a pair of two hash rings: both sides of
/// the pair read the target source.
fn rings() -> &'static InProcess {
    stock("consistent_hashing", "consistent_hashing")
}

/// An encode request of one item over `lists` that carries `headers`.
fn carrying<'a>(lists: EncodeLists<'a>, headers: &'a dyn HeaderRead) -> PlaceRequest<'a> {
    let items = EncodeItems {
        keys: &KEYS,
        ..EncodeItems::default()
    };
    PlaceRequest {
        headers,
        ..asking(lists, items)
    }
}

/// The places in `three` of the pair `request` is placed on, where each
/// item it carries is placed on an encoder: `items` of them.
fn placed(
    engine: &'static InProcess,
    three: &Three,
    request: &PlaceRequest<'_>,
    items: usize,
) -> Result<(usize, usize), Unplaced> {
    let placement = engine.place(request)?;
    let (prefill, decode, encoders) = three.places(&placement);
    assert_eq!(encoders.len(), items, "an encoder an item");
    Ok((prefill, decode))
}

/// The places of the pair an encode request of one item that names
/// `number` is placed on over `lists`, under one routing value.
fn named(
    engine: &'static InProcess,
    three: &Three,
    lists: EncodeLists<'_>,
    number: &str,
) -> Result<(usize, usize), Unplaced> {
    let headers = [(MANUAL_SESSION_HEADER, "talk-1"), (TARGET, number)];
    placed(engine, three, &carrying(lists, &headers), 1)
}

/// What stands behind the answers that a case of an answer that stands
/// compares, of its three calls: a request that names no worker, one that
/// names the number `1`, and one that names a number past the prefills,
/// each with its reads of the header. Each call looked for the header
/// once, and the number past the prefills was the target's miss: the
/// count ran ahead of the answer.
fn counted_first(answered: [(Option<Unplaced>, u32); 3]) {
    let [.., (past, _)] = answered;
    assert_eq!(past, Some(missed_on(PairSide::Prefill)), "said first");
    let reads = answered.map(|(_, reads)| reads);
    assert_eq!(reads, [1; 3], "the header is looked for once a call");
}

/// A filter of a host's own that keeps no worker under the key of the
/// cases' one item ([`KEYS`]), and every worker under another key: it is
/// asked for an item under the item's key.
#[derive(Debug)]
struct NotTheItem;

impl Filter for NotTheItem {
    fn name(&self) -> &'static str {
        "item-only"
    }

    fn keep(&self, facts: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        facts.session_key != KEYS[0]
    }
}

/// The stock encode document over two hash rings, whose encode side runs
/// [`NotTheItem`] behind its own filters: a stage of a host's that leaves
/// the item no encoder, under a name of its own.
fn keeping_no_encoder() -> &'static InProcess {
    let mut stages = StageRegistry::builtin();
    let added = stages.filter("item-only", || Box::new(NotTheItem));
    added.expect("a free name");
    let ring = "consistent_hashing";
    let presets = PolicySpec::epd_presets("power_of_two", ring, ring);
    let mut document = presets.expect("presets");
    document.generation = 1;
    let encode = document.profiles.get_mut("encode");
    let encode = encode.expect("the encode profile");
    encode.filter.push(FilterSpec::Name("item-only".to_owned()));
    let spec = AppliedSpec::validate_in(&document, &stages, &[]);
    hosted(InProcess::new(spec.expect("the stages hold the filter")))
}

/// An encode request that names its worker goes to the worker at that
/// number on each side of its pair, ahead of its routing value, with its
/// item on an encoder beside them: under the stock encode document over
/// two hash rings, over three prefills and three decodes that all pair
/// and are all available, each number takes the prefill and the decode at
/// that place, whichever workers own the request's routing value, and
/// with no routing value too. The header is read once a call. A request
/// that names no worker goes to its routing value's owners, as before.
///
/// The classic router takes such a request's pair by its pair selection
/// (`WS:2675-2683`), where each side's policy reads the header over its
/// own workers (`P:3427-3428`, `P:3490`; `CH:187-193`). It reads the
/// header for no encoder: an item's encoder is picked under headers that
/// hold the item's content hash alone (`WS:2800-2819`).
#[test]
fn an_encode_request_that_names_its_worker_goes_to_that_worker_on_each_side() {
    let engine = rings();
    assert!(engine.spec().names_target());
    let three = Three::of([3, 3, 2], |_, _| false);
    let asked = |carried: &[(&str, &str)]| {
        let headers = Counted::of(carried);
        let request = carrying(three.lists((&[PLAIN; 2], &[1; 3])), &headers);
        let pair = placed(engine, &three, &request, 1).expect("a placement");
        (headers.reads.get(), pair)
    };
    let mut owners = BTreeSet::new();
    for key in ["talk-1", "talk-2", "talk-3", "talk-4", "talk-5", "talk-6"] {
        let (reads, owned) = asked(&[(MANUAL_SESSION_HEADER, key)]);
        assert_eq!(reads, 1, "{key}: the header is looked for once");
        owners.insert(owned);
        for place in 0..3 {
            let number = place.to_string();
            let named = asked(&[(MANUAL_SESSION_HEADER, key), (TARGET, number.as_str())]);
            assert_eq!(named, (1, (place, place)), "{key}, names {place}");
        }
    }
    assert!(
        owners.len() > 1,
        "the keys have more than one pair of owners"
    );
    for place in 0..3 {
        let number = format!("+{place}");
        let named = asked(&[(TARGET, number.as_str())]);
        assert_eq!(named, (1, (place, place)), "no key, names {place}");
    }
}

/// An encode request whose number reaches no worker is shed with no
/// endpoint under `target-worker`, on the side the worker was missed on,
/// and moves to no other worker. The prefills are counted first: a text
/// that is no number and a number past the prefills counted miss on the
/// prefill's side, and a number that reaches a prefill and none of its
/// partners misses on the decode's. An empty text names no worker. The
/// count runs ahead of the request's decision: over no encoder at all, a
/// number that reaches no worker is said first, and one that reaches a
/// pair is answered by the encoders its request lacks.
///
/// The classic policy misses a text it cannot read as a number and a
/// number outside the workers it is handed, and names no other worker
/// (`CH:150-162`, `CH:187-193`). The pair selection refuses the pair for
/// the role of the policy that answered so: the prefill's, which it asks
/// first (`P:3427-3428`, `P:3446`), or the decode's (`P:3490-3492`).
#[test]
fn an_encode_request_whose_number_reaches_no_worker_is_shed_on_that_side() {
    let engine = rings();
    let three = Three::of([3, 2, 2], |_, _| false);
    let at = |number: &str| named(engine, &three, three.lists((&[PLAIN; 2], &[1; 3])), number);
    assert_eq!(at("1"), Ok((1, 1)));
    assert_eq!(at("one"), Err(missed_on(PairSide::Prefill)));
    assert_eq!(at("3"), Err(missed_on(PairSide::Prefill)), "three prefills");
    assert_eq!(at("2"), Err(missed_on(PairSide::Decode)), "two decodes");
    assert!(at("").is_ok(), "an empty text names no worker");

    let bare = Three::of([3, 2, 0], |_, _| false);
    let at = |number: &str| named(engine, &bare, bare.lists((&[], &[1; 3])), number);
    assert_eq!(at("3"), Err(missed_on(PairSide::Prefill)), "said first");
    let lacking = Unplaced::Shed {
        reason: ShedReason::NoEndpoints,
        emptied_by: "encoders",
        retry_after_secs: 1,
        side: None,
    };
    assert_eq!(at("1"), Err(lacking), "its encoders' answer");
}

/// Where no prefill is counted there is nobody to name: the request leaves
/// its hold and is placed as an encode request that names no worker,
/// whatever its header's text. Over prefills that are all over their
/// thresholds it goes to the one that holds the fewest requests, with its
/// item on an encoder, as the same request does with no named worker. The
/// header is read all the same, once a call: the count is what finds
/// nobody. The request is placed as it states its items: one that holds
/// its decode to images, over decodes that take none, is answered the gap
/// of the decodes with a worker named as with none.
///
/// The classic router finds no prefill its policy could be handed, and
/// pairs by load with no read of the header (`P:3302-3320`).
#[test]
fn where_no_prefill_is_counted_an_encode_request_leaves_its_hold() {
    let engine = rings();
    let three = Three::of([2, 2, 2], |_, _| false);
    for prefill in three.workers[0].iter() {
        prefill.set_overloaded(true);
    }
    three.workers[0][1].increment_load();
    // The header's reads of one call, and the prefill it places on.
    let prefill_of = |carried: &[(&str, &str)]| {
        let headers = Counted::of(carried);
        let request = carrying(three.lists((&[PLAIN; 2], &[1; 2])), &headers);
        let (prefill, _) = placed(engine, &three, &request, 1).expect("a placement");
        (headers.reads.get(), prefill)
    };
    for number in ["1", "7", "one"] {
        let read = prefill_of(&[(TARGET, number)]);
        assert_eq!(
            read,
            (1, 0),
            "names {number}: the prefill of the least load"
        );
    }
    for key in ["talk-1", "talk-2", "talk-3", "talk-4", "talk-5", "talk-6"] {
        let keyed = (MANUAL_SESSION_HEADER, key);
        assert_eq!(prefill_of(&[keyed]), (1, 0), "{key}, no named worker");
        let named = prefill_of(&[keyed, (TARGET, "1")]);
        assert_eq!(named, (1, 0), "{key}, names 1");
    }
    // The gap a request that holds its decode to images is answered.
    let to_images = EncodeItems {
        keys: &KEYS,
        decode_images: true,
        ..EncodeItems::default()
    };
    let gap_of = |carried: &[(&str, &str)]| {
        let headers = Counted::of(carried);
        let request = PlaceRequest {
            headers: &headers,
            ..asking(three.lists((&[PLAIN; 2], &[1; 2])), to_images)
        };
        gap(engine.place(&request))
    };
    assert_eq!(gap_of(&[]), Some("decode-vision"), "no worker named");
    assert_eq!(gap_of(&[(TARGET, "1")]), Some("decode-vision"), "names 1");
}

/// Three lists with no item are held to the worker their request names as
/// a pair's two lists are, by the pair's own rule: each number takes the
/// prefill and the decode at that place, a number that reaches no worker
/// is shed under `target-worker` on its side, and the placement names no
/// encoder. A request that states no items, and one that states an empty
/// list of them and holds its decode to no images, are answered alike:
/// the encode pass is laid for neither.
///
/// The classic router assigns no encoder where a request has no item
/// (`WS:2584`, `WS:2673`), and takes its pair by the same pair selection
/// (`WS:2675-2683`).
#[test]
fn three_lists_with_no_item_are_held_to_the_worker_their_request_names() {
    let engine = rings();
    let three = Three::of([3, 3, 2], |_, _| false);
    for encode in [None, Some(EncodeItems::default())] {
        let at = |number: &str| {
            let headers = [(MANUAL_SESSION_HEADER, "talk-1"), (TARGET, number)];
            let lists = three.lists((&[PLAIN; 2], &[1; 3]));
            let request = PlaceRequest {
                headers: &headers,
                encode,
                ..PlaceRequest::of(MODEL, WorkerLists::Encode(lists))
            };
            placed(engine, &three, &request, 0)
        };
        for place in 0..3 {
            assert_eq!(at(&place.to_string()), Ok((place, place)));
        }
        assert_eq!(at("3"), Err(missed_on(PairSide::Prefill)));
        assert_eq!(at("one"), Err(missed_on(PairSide::Prefill)));
    }
}

/// The count asks no encoder. A request that states no rule is counted as
/// a pair is, so its number can reach a prefill no encoder serves: the
/// second of three prefills here is of an encode class with no encoder.
/// The decision held to that prefill's place names no worker, and the
/// request is shed under `target-worker`, as a pair is where a filter
/// refuses the prefill it is held to. The shed names no side: the encode
/// pass refused the place, and said of neither list that it was left with
/// nobody. The request moves to no other prefill. The numbers that reach
/// a prefill an encoder serves are placed there.
///
/// The classic router's hash policy names nobody where its number reaches
/// no worker it may use, and the request is refused (`CH:150-162`,
/// `CH:187-193`). Its count is narrowed by its stage's rule, which the
/// case `the_number_counts_among_the_workers_the_gateways_rule_takes`
/// states.
#[test]
fn a_number_that_reaches_a_prefill_no_encoder_serves_is_shed_under_the_targets_name() {
    let engine = rings();
    let three = Three::of([3, 3, 2], |_, _| false);
    let lists = || three.lists((&[PLAIN; 2], &[1, 2, 1]));
    assert_eq!(named(engine, &three, lists(), "0"), Ok((0, 0)));
    assert_eq!(named(engine, &three, lists(), "1"), Err(MISSED));
    assert_eq!(named(engine, &three, lists(), "2"), Ok((2, 2)));
    // With no named worker the same request goes to a prefill an encoder
    // serves, under each routing value.
    for key in ["talk-1", "talk-2", "talk-3", "talk-4", "talk-5", "talk-6"] {
        let headers = [(MANUAL_SESSION_HEADER, key)];
        let pair = placed(engine, &three, &carrying(lists(), &headers), 1);
        let (prefill, _) = pair.expect("a placement");
        assert_ne!(prefill, 1, "{key}: no encoder serves the second prefill");
    }
}

/// The number names no encoder, so what a request held to its pair is
/// answered for its encoders stands as it is. Over each fleet here it is
/// what the same request is answered with no worker named. Encoders that
/// are all over their thresholds, under a binding that sheds such a list,
/// shed it as saturated. Encoders that are all held back by their
/// breakers shed it with no endpoint, under the encode side's own name. A
/// stage of the encode side that leaves an item no encoder sheds it under
/// that stage's name, a host's own here. None is the target's miss, and
/// none names a side of the pair. Each call looks for the header once,
/// and the request that names a worker was counted ahead of its answer:
/// over the same encoders, a number past the prefills is the target's
/// miss.
///
/// The classic router answers a request that carries an item and that no
/// available encoder takes ahead of its pair selection (`WS:2584`,
/// `WS:2590`, `WS:2605-2634`).
#[test]
fn a_shed_made_for_the_encoders_of_a_named_pair_stands() {
    let three = Three::of([2, 2, 2], |_, _| false);
    let lists = || three.lists((&[PLAIN; 2], &[1; 2]));
    // What a request is answered with no worker named, naming one, and
    // naming a number past the prefills, each with its reads of the
    // header.
    let answers = |engine: &'static InProcess| {
        let answer = |carried: &[(&str, &str)]| {
            let headers = Counted::of(carried);
            let answer = engine.place(&carrying(lists(), &headers)).err();
            (answer, headers.reads.get())
        };
        let key = (MANUAL_SESSION_HEADER, "talk-1");
        [&[key][..], &[key, (TARGET, "1")], &[key, (TARGET, "7")]].map(answer)
    };
    let ring = "consistent_hashing";
    let shedding = stock_of(["power_of_two", ring, ring], |document| {
        for binding in &mut document.bindings {
            binding.overload = Some(OverloadSpec {
                shed_when_all_vetoed: true,
                ..OverloadSpec::default()
            });
        }
    });
    for encoder in three.workers[2].iter() {
        encoder.set_overloaded(true);
    }
    let answered = answers(shedding);
    let [(unnamed, _), (named, _), _] = answered;
    let saturated = matches!(
        unnamed,
        Some(Unplaced::Shed {
            reason: ShedReason::Saturated,
            side: None,
            ..
        })
    );
    assert!(saturated, "{unnamed:?}");
    assert_eq!(named, unnamed, "as with no worker named");
    counted_first(answered);

    // The shed with no endpoint that names no side, under `stage`.
    let emptied_by = |stage: &'static str| Unplaced::Shed {
        reason: ShedReason::NoEndpoints,
        emptied_by: stage,
        retry_after_secs: 1,
        side: None,
    };
    for encoder in three.workers[2].iter() {
        encoder.set_overloaded(false);
    }
    let answered = answers(keeping_no_encoder());
    let [(unnamed, _), (named, _), _] = answered;
    assert_eq!(unnamed, Some(emptied_by("item-only")), "no worker named");
    assert_eq!(named, unnamed, "as with no worker named");
    counted_first(answered);

    for encoder in three.workers[2].iter() {
        trip(encoder);
    }
    let answered = answers(rings());
    let [(unnamed, _), (named, _), _] = answered;
    assert_eq!(unnamed, Some(emptied_by("encoders")), "no worker named");
    assert_eq!(named, unnamed, "as with no worker named");
    counted_first(answered);
}

/// A gap the encode pass says over the lists whole stands for a request
/// that names its worker, under the pass's own name: no place the pair is
/// held to opens it, and none closes it. So the request is answered as it
/// is with no worker named. The pass's four names beside the encoders' own
/// are shown here: no encoder takes the request's images, none takes its
/// references, none hands to a prefill of a pair, and no decode of a pair
/// an encoder serves takes the images the request holds its decode to.
/// Each call looks for the header once, and the request that names a
/// worker was counted ahead of its answer: over the same lists, a number
/// past the prefills is the target's miss.
///
/// The classic router says each from its registered workers, ahead of its
/// pair selection and of any read of the header: the encoders' three
/// (`WS:2607-2634`) and the pairs' two (`WS:2656-2665`).
#[test]
fn a_gap_the_encode_pass_says_over_the_lists_whole_stands_for_a_named_request() {
    let engine = rings();
    // No worker of the three lists takes images.
    let three = Three::of([2, 2, 2], |_, _| false);
    // What a request is answered with no worker named, naming one, and
    // naming a number past the prefills, each with its reads of the
    // header.
    let answers = |said: Said<'_>, demands: Demands, items: EncodeItems<'_>| {
        let answer = |carried: &[(&str, &str)]| {
            let headers = Counted::of(carried);
            let request = PlaceRequest {
                headers: &headers,
                demands,
                ..asking(three.lists(said), items)
            };
            (engine.place(&request).err(), headers.reads.get())
        };
        let key = (MANUAL_SESSION_HEADER, "talk-1");
        [&[key][..], &[key, (TARGET, "1")], &[key, (TARGET, "7")]].map(answer)
    };
    let item = EncodeItems {
        keys: &KEYS,
        ..EncodeItems::default()
    };
    let references = EncodeItems {
        references: true,
        ..item
    };
    let to_images = EncodeItems {
        decode_images: true,
        ..item
    };
    let none = Demands::default();
    let vision = Demands {
        vision: true,
        ..none
    };
    // Each prefill is of the encoders' class, or of one with no encoder.
    let served: Said<'_> = (&[PLAIN; 2], &[1; 2]);
    let unserved: Said<'_> = (&[PLAIN; 2], &[2; 2]);
    let gaps = [
        ("encode-vision", answers(served, vision, item)),
        ("encode-media", answers(served, none, references)),
        ("ec-compatible", answers(unserved, none, item)),
        ("decode-vision", answers(served, none, to_images)),
    ];
    for (stage, answered) in gaps {
        let [(unnamed, _), (named, _), _] = answered;
        let shed = Unplaced::Shed {
            reason: ShedReason::NoEndpoints,
            emptied_by: stage,
            retry_after_secs: 1,
            side: None,
        };
        assert_eq!(unnamed, Some(shed), "{stage}: no worker named");
        assert_eq!(named, unnamed, "{stage}: as with no worker named");
        counted_first(answered);
    }
}

/// The count does not read the items' hold of the decode to images. A
/// request that holds its decode to the workers that take images, and
/// states no rule, has its decode counted among them all: the number can
/// reach a decode that takes no images. The encode pass refuses that
/// place, and the request is shed under `target-worker`, on no side. The
/// rule here refuses that decode, as the rule the gateway's encode stage
/// states refuses, for such a request, a decode of one serving engine
/// that takes no images (`WS:2650-2654`, `WS:2669-2674`). Under it the
/// number passes over that decode: it reaches the decode the gateway's
/// own selection reaches (`P:3458-3460`, `P:3490`).
///
/// The first decode takes no images, and every other worker does.
#[test]
fn a_decode_that_takes_no_images_takes_a_number_under_no_rule() {
    let engine = rings();
    let three = Three::of([3, 3, 2], |list, place| (list, place) != (1, 0));
    let items = EncodeItems {
        keys: &KEYS,
        decode_images: true,
        ..EncodeItems::default()
    };
    // The rule: it refuses the first decode, the one worker of the bed
    // that takes no images.
    let rule = |worker: &dyn Worker| worker_id(worker) != three.id(1, 0);
    let at = |number: &str, admits: Option<WorkerRule<'_>>| {
        let headers = [(MANUAL_SESSION_HEADER, "talk-1"), (TARGET, number)];
        let request = PlaceRequest {
            headers: &headers,
            admits,
            ..asking(three.lists((&[PLAIN; 2], &[1; 3])), items)
        };
        placed(engine, &three, &request, 1)
    };
    assert_eq!(at("0", None), Err(MISSED), "the first decode takes none");
    assert_eq!(at("1", None), Ok((1, 1)));
    assert_eq!(
        at("0", Some(&rule)),
        Ok((0, 1)),
        "the first that takes them"
    );
    assert_eq!(at("1", Some(&rule)), Ok((1, 2)));
    let third = at("2", Some(&rule));
    assert_eq!(third, Err(missed_on(PairSide::Decode)), "two decodes");
}

/// The number counts among the workers the route's rule takes, and the
/// rule says which workers an encoder serves. The rule here is the one
/// the gateway's encode stage states: a worker that an available encoder
/// is compatible with (`WS:2671-2674`). The second of three prefills is of
/// an encode class with no encoder, so the rule refuses it. The number `1`
/// then reaches the third prefill, the second the rule takes, and the
/// request is placed there with its item on an encoder. The number `2`
/// reaches nobody. The rule is asked of no encoder.
///
/// The gateway's own selection reaches the same worker. Its candidates
/// are the available prefills its filter takes that have an open partner
/// (`P:3208-3213`, `P:3249-3252`, `P:3329-3344`), and its hash policy
/// takes the worker at the number in that list (`P:3391`, `P:3427-3428`;
/// `CH:150-162`). The decode's number counts the open partners of that
/// prefill (`P:3458-3460`, `P:3490`).
#[test]
fn the_number_counts_among_the_workers_the_gateways_rule_takes() {
    let engine = rings();
    let three = Three::of([3, 3, 2], |_, _| false);
    let (facts, classes) = ([PLAIN; 2], [1, 2, 1]);
    let of_encoders = Cell::new(0_u32);
    // The stage's rule: an available encoder hands to the worker. A
    // prefill takes the encoders of its encode class, and every decode is
    // served.
    let rule = |worker: &dyn Worker| {
        let id = worker_id(worker);
        let at = |list: usize| (0..3).find(|&place| three.id(list, place) == id);
        if (0..2).any(|place| three.id(2, place) == id) {
            of_encoders.set(of_encoders.get() + 1);
        }
        at(0).is_none_or(|prefill| {
            let mut encoders = three.workers[2].iter().zip(&facts);
            encoders.any(|(encoder, of)| encoder.is_available() && of.class == classes[prefill])
        })
    };
    let at = |number: &str| {
        let headers = [(MANUAL_SESSION_HEADER, "talk-1"), (TARGET, number)];
        let request = PlaceRequest {
            admits: Some(&rule),
            ..carrying(three.lists((&facts, &classes)), &headers)
        };
        placed(engine, &three, &request, 1)
    };
    assert_eq!(at("0"), Ok((0, 0)));
    assert_eq!(at("1"), Ok((2, 1)), "the second prefill the rule takes");
    assert_eq!(at("2"), Err(missed_on(PairSide::Prefill)), "two prefills");
    assert_eq!(of_encoders.get(), 0, "the rule is asked of no encoder");
}

/// A retry that names its encode request's pair is handed the pair again,
/// though its failed attempt was sent to both workers. A side that is not
/// held steps aside from the worker the request tried.
///
/// The classic router keeps the tried workers in a held side's candidates,
/// and narrows a side under another policy (`P:3399-3401`, `P:3475-3477`).
#[test]
fn a_retry_that_names_its_encode_requests_pair_is_handed_it_again() {
    let three = Three::of([3, 3, 2], |_, _| false);
    let headers = [(MANUAL_SESSION_HEADER, "talk-1"), (TARGET, "1")];
    let retry = |engine: &'static InProcess, tried: &[WorkerId]| {
        let request = PlaceRequest {
            attempt: 1,
            tried,
            ..carrying(three.lists((&[PLAIN; 2], &[1; 3])), &headers)
        };
        placed(engine, &three, &request, 1)
    };
    let tried = [three.id(0, 1), three.id(1, 1)];
    assert_eq!(retry(rings(), &tried), Ok((1, 1)));

    let ring_prefill = stock("consistent_hashing", "round_robin");
    for _ in 0..6 {
        let (prefill, decode) = retry(ring_prefill, &tried).expect("a placement");
        assert_eq!(prefill, 1, "the named prefill, though tried");
        assert_ne!(decode, 1, "the decode steps aside");
    }
}

/// An item's encoder is no part of the pair its request names. Where the
/// encoder a decision named cannot be sent to, held back by its circuit
/// breaker under a document whose encode side keeps such a worker, the
/// call decides again and is told that encoder, as any encode request's
/// call is, and the pair stays the one the request names: behind two such
/// encoders the call's third decision places the item, under each number
/// and whichever pair owns the request's routing value. A rule that takes
/// every worker changes nothing of it: the pair is held on both sides, and
/// the rule is asked of no list behind an encoder. Where the third
/// encoder cannot be sent to either, the request has no worker behind
/// that third decision, though a fourth encoder is ready: a call asks
/// three times. It says so as it does for a request that names no worker:
/// the pair could be sent to at each ask, so the target is not missed.
/// The later decisions keep the decode the number reached where it stands
/// at another place than its prefill: under a rule that passes over the
/// first decode, the number 0 reaches the first prefill and the second
/// decode.
///
/// The classic router offers an item the encoders that are available
/// alone, whatever worker its request names (`WS:2590`, `WS:2605`).
#[test]
fn a_named_pairs_item_is_placed_past_an_encoder_that_cannot_be_sent_to() {
    let ring = "consistent_hashing";
    let engine = stock_of(["power_of_two", ring, ring], keeps_its_encoders);
    let three = Three::of([3, 3, 4], |_, _| false);
    // The encoders hold more requests from the first to the fourth, and
    // the breakers of the first two are open.
    for (held, encoder) in three.workers[2].iter().enumerate() {
        for _ in 0..held {
            encoder.increment_load();
        }
    }
    trip(&three.workers[2][0]);
    trip(&three.workers[2][1]);
    let rule = |_: &dyn Worker| true;
    // Where a request that names `number` is placed, under `admits`.
    let asked = |number: &str, admits: Option<WorkerRule<'_>>| {
        let headers = [(MANUAL_SESSION_HEADER, "talk-1"), (TARGET, number)];
        let request = PlaceRequest {
            admits,
            ..carrying(three.lists((&[PLAIN; 4], &[1; 3])), &headers)
        };
        let placement = engine.place(&request);
        placement.map(|placement| three.places(&placement))
    };
    for place in 0..3 {
        let third = Ok((place, place, vec![2]));
        let number = place.to_string();
        assert_eq!(asked(&number, None), third, "the third encoder");
        assert_eq!(asked(&number, Some(&rule)), third, "under a rule");
    }

    let behind = |worker: &dyn Worker| worker_id(worker) != three.id(1, 0);
    let kept = Ok((0, 1, vec![2]));
    assert_eq!(asked("0", Some(&behind)), kept, "the decode's own place");

    trip(&three.workers[2][2]);
    let no_worker = Err(Unplaced::NoWorkers);
    assert_eq!(asked("1", None), no_worker, "its encoders, not its pair");
    assert_eq!(asked("1", Some(&rule)), no_worker, "under a rule");
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A pair held on both sides stands on one pair of places, in an encode
/// request's call as in a pair's own: behind a side that cannot be sent
/// to it has no worker, and the request is shed under `target-worker`.
/// The route's rule here takes the named prefill at the count and refuses
/// it where the call would send to it. The call makes no second decision,
/// though the rule would take the prefill again: it is asked of that
/// prefill twice.
///
/// The classic router asks its filter of the workers it gathers as
/// candidates, ahead of its policies (`P:3208-3213`, `P:3329-3344`).
#[test]
fn a_named_pair_one_side_of_which_cannot_be_sent_to_is_asked_no_second_time() {
    let engine = rings();
    let three = Three::of([3, 3, 2], |_, _| false);
    let of_prefill = Cell::new(0_u32);
    let rule = |worker: &dyn Worker| {
        if worker_id(worker) != three.id(0, 1) {
            return true;
        }
        of_prefill.set(of_prefill.get() + 1);
        of_prefill.get() != 2
    };
    let headers = [(MANUAL_SESSION_HEADER, "talk-1"), (TARGET, "1")];
    let request = PlaceRequest {
        admits: Some(&rule),
        ..carrying(three.lists((&[PLAIN; 2], &[1; 3])), &headers)
    };
    assert_eq!(placed(engine, &three, &request, 1), Err(MISSED));
    assert_eq!(of_prefill.get(), 2, "the count's ask and one decision's");
}

/// An ask given up for an encoder alone says so of itself, and of no ask
/// behind it. The first encoder cannot be sent to, so the call decides
/// again. The route's rule refuses the named prefill where that second
/// decision would send to it: the call ends there, with no third
/// decision, and the request is shed under `target-worker`. The rule is
/// asked of the named prefill three times: at the count, and where each
/// of the two decisions would send to it.
///
/// The classic router asks its filter of the workers it gathers as
/// candidates, ahead of its policies (`P:3208-3213`, `P:3329-3344`).
#[test]
fn a_side_refused_behind_an_encoder_that_could_not_be_sent_to_ends_the_call() {
    let ring = "consistent_hashing";
    let engine = stock_of(["power_of_two", ring, ring], keeps_its_encoders);
    let three = Three::of([3, 3, 2], |_, _| false);
    // The first encoder holds fewer requests, and its breaker is open.
    three.workers[2][1].increment_load();
    trip(&three.workers[2][0]);
    let of_prefill = Cell::new(0_u32);
    let rule = |worker: &dyn Worker| {
        if worker_id(worker) != three.id(0, 1) {
            return true;
        }
        of_prefill.set(of_prefill.get() + 1);
        of_prefill.get() != 3
    };
    let headers = [(MANUAL_SESSION_HEADER, "talk-1"), (TARGET, "1")];
    let request = PlaceRequest {
        admits: Some(&rule),
        ..carrying(three.lists((&[PLAIN; 2], &[1; 3])), &headers)
    };
    assert_eq!(placed(engine, &three, &request, 1), Err(MISSED));
    assert_eq!(of_prefill.get(), 3, "the count's ask and two decisions'");
}

/// A rule that refuses the place of the prefill a request names, behind
/// an ask given up for an encoder alone, misses the target. The pair is
/// held on its prefill alone, and the first encoder cannot be sent to:
/// the first decision's pair could be sent to, so that ask is given up
/// for the encoder alone. The call asks again as any pair's does, and
/// asks the rule of the one place the prefill is held to. The rule
/// refuses the prefill there, and the request is shed under
/// `target-worker`, behind one decision: the stages hear one pair issued
/// and let go. That refusal is the pair's, whatever the ask before it was
/// given up for: it is no answer of the encoders, and does not stand
/// under the rule's own name.
///
/// The classic router asks its filter of the workers it gathers as
/// candidates, ahead of its policies (`P:3208-3213`, `P:3329-3344`).
#[test]
fn a_rule_that_refuses_the_named_prefill_behind_an_encoder_misses_the_target() {
    let ring = "consistent_hashing";
    let presets = PolicySpec::epd_presets("power_of_two", ring, "round_robin");
    let mut document = presets.expect("presets");
    keeps_its_encoders(&mut document);
    let (engine, heard) = hearing(document);
    // One decode: a decision has one pair to send to.
    let three = Three::of([3, 1, 2], |_, _| false);
    // The first encoder holds fewer requests, and its breaker is open.
    three.workers[2][1].increment_load();
    trip(&three.workers[2][0]);
    let of_prefill = Cell::new(0_u32);
    // The rule takes the named prefill twice: at the count, and where the
    // first decision would send to it.
    let rule = |worker: &dyn Worker| {
        if worker_id(worker) != three.id(0, 1) {
            return true;
        }
        of_prefill.set(of_prefill.get() + 1);
        of_prefill.get() <= 2
    };
    let headers = [(MANUAL_SESSION_HEADER, "talk-1"), (TARGET, "1")];
    let request = PlaceRequest {
        admits: Some(&rule),
        ..carrying(three.lists((&[PLAIN; 2], &[1; 3])), &headers)
    };
    assert_eq!(placed(engine, &three, &request, 1), Err(MISSED));
    assert_eq!(of_prefill.get(), 3, "the count's ask, a send's, the list's");
    // No second decision stands behind the list's refusal.
    let (prefill, decode) = (three.id(0, 1), three.id(1, 0));
    let released = |worker| ("released CommitFailed".to_owned(), worker);
    let one = [
        issued(prefill),
        issued(decode),
        released(prefill),
        released(decode),
    ];
    assert_eq!(*heard.lock(), one, "one decision");
}

/// A named worker that changes between the count that reached it and its
/// request's decision is missed on its own side in each leg here. Each
/// request carries its item, and its pair is held on both sides. The rule
/// is asked of a worker as the count takes it, and the worker it names
/// here changes behind that. One that stops being ready leaves the
/// decision held to the two places nobody on its side, and that shed
/// takes the target's name with the side. A named worker that comes over
/// its thresholds, under a binding that sheds such a list, is missed so
/// too, on either side: a shed as saturated over a list of the pair is
/// the pair's, and no word of the encoders.
///
/// The classic router's pair selection refuses the pair for the role of
/// the policy that found its target nowhere (`P:3446`, `P:3490-3492`).
#[test]
fn a_named_worker_that_cannot_take_an_encode_request_after_all_is_missed_on_its_side() {
    // The answer of a request that names the second worker of each side,
    // where the rule changes the second worker of `list` by `change`.
    let changing = |engine: &'static InProcess, list: usize, change: fn(&dyn Worker)| {
        let three = Three::of([2, 2, 2], |_, _| false);
        let changed = three.id(list, 1);
        let rule = |worker: &dyn Worker| {
            if worker_id(worker) == changed {
                change(worker);
            }
            true
        };
        let headers = [(TARGET, "1")];
        let request = PlaceRequest {
            admits: Some(&rule),
            ..carrying(three.lists((&[PLAIN; 2], &[1; 2])), &headers)
        };
        placed(engine, &three, &request, 1)
    };
    let leaves = |worker: &dyn Worker| {
        worker.set_status(WorkerStatus::NotReady);
    };
    let on_prefills = Err(missed_on(PairSide::Prefill));
    assert_eq!(
        changing(rings(), 0, leaves),
        on_prefills,
        "the prefill left"
    );
    let on_decodes = Err(missed_on(PairSide::Decode));
    assert_eq!(changing(rings(), 1, leaves), on_decodes, "the decode left");

    let ring = "consistent_hashing";
    let shedding = stock_of(["power_of_two", ring, ring], |document| {
        for binding in &mut document.bindings {
            binding.overload = Some(OverloadSpec {
                shed_when_all_vetoed: true,
                ..OverloadSpec::default()
            });
        }
    });
    let fills = |worker: &dyn Worker| {
        worker.set_overloaded(true);
    };
    assert_eq!(
        changing(shedding, 0, fills),
        on_prefills,
        "the prefill is over"
    );
    assert_eq!(
        changing(shedding, 1, fills),
        on_decodes,
        "the decode is over"
    );
}

/// A rule that refuses the prefill a request names, behind the count that
/// took it, misses the target. The pair is held on its prefill alone
/// here, so the call asks again as any pair's does: it asks the rule of
/// that one place, the rule takes nobody there, and the request is shed
/// under `target-worker` and not under the rule's own name. A pair's own
/// call answers so.
///
/// The last two legs make that prefill not ready as the count takes it,
/// so the first decision sheds the request with the prefill's side. The
/// leg with its item is a request the encode pass is laid for: such a
/// shed answers its call ahead of the rule (`pair::goes_to_the_rule`),
/// and the miss names the prefill's side. The leg that states no items is
/// its pair's alone, and states no hard demand: there that shed goes to
/// the rule, the rule takes nobody at the one place, and the miss names
/// no side.
///
/// The classic router asks its filter of the workers it gathers as
/// candidates, ahead of its policies (`P:3208-3213`, `P:3329-3344`). The
/// rule that changes its worker is this case's device for a worker that
/// leaves between the count and the decision. Where that router gathers
/// its candidates it asks its filter ahead of a worker's availability
/// (`P:3249-3252`), and the count reads the worker's state first: under
/// this very rule that router would count past the prefill.
#[test]
fn a_rule_that_refuses_the_named_prefill_behind_the_count_misses_the_target() {
    let engine = stock("consistent_hashing", "round_robin");
    let three = Three::of([3, 3, 2], |_, _| false);
    let of_prefill = Cell::new(0_u32);
    // The rule takes the named prefill once, at the count.
    let rule = |worker: &dyn Worker| {
        if worker_id(worker) != three.id(0, 1) {
            return true;
        }
        of_prefill.set(of_prefill.get() + 1);
        of_prefill.get() == 1
    };
    let headers = [(MANUAL_SESSION_HEADER, "talk-1"), (TARGET, "1")];
    let request = PlaceRequest {
        admits: Some(&rule),
        ..carrying(three.lists((&[PLAIN; 2], &[1; 3])), &headers)
    };
    assert_eq!(placed(engine, &three, &request, 1), Err(MISSED));
    assert!(of_prefill.get() > 2, "asked again at the one place");

    // The rule again, and it makes the prefill not ready as the count
    // takes it: the answer of the request with its item, or with no items
    // stated, and how often the rule was asked of that prefill.
    let left = |item: bool| {
        let three = Three::of([3, 3, 2], |_, _| false);
        let of_prefill = Cell::new(0_u32);
        let rule = |worker: &dyn Worker| {
            if worker_id(worker) != three.id(0, 1) {
                return true;
            }
            worker.set_status(WorkerStatus::NotReady);
            of_prefill.set(of_prefill.get() + 1);
            of_prefill.get() == 1
        };
        let mut request = carrying(three.lists((&[PLAIN; 2], &[1; 3])), &headers);
        request.admits = Some(&rule);
        if !item {
            request.encode = None;
        }
        let answer = placed(engine, &three, &request, usize::from(item));
        (answer, of_prefill.get())
    };
    // The pass is laid for the request with its item: the first decision's
    // shed answers the call with its side, and no rule is asked behind it.
    assert_eq!(left(true), (Err(missed_on(PairSide::Prefill)), 1));
    // This leg states no items and no hard demand: it is its pair's alone,
    // its first decision's shed goes to the rule, which takes nobody at
    // the place, and the miss names no side.
    assert_eq!(left(false), (Err(MISSED), 2));
}

/// A request that names no worker is placed as under a document with no
/// target source, whatever it states of its items: its binding's target
/// source costs it one lookup of the header, and changes nothing else of
/// its answer. Three lists under a binding of the
/// pair's own plan are a host's wrong hand, which the engine's entry for
/// three lists declines by name and a pair's own takes. With the header
/// absent, and with its text empty, a request that states items by an
/// empty list goes to that entry and is declined, and one that states
/// none is placed as its pair: as each is under a document that names no
/// target. The header is looked for once. One that names a worker and
/// that the encode pass is not laid for is its pair's alone, and is
/// answered as a pair that names that worker is, under either statement
/// of its items.
#[test]
fn a_request_that_names_no_worker_is_placed_as_before_under_a_target_source() {
    let ring = "consistent_hashing";
    let mut document = PolicySpec::pair_presets(ring, ring).expect("presets");
    document.generation = 1;
    let engine = hosted(engine_of(&document));
    assert!(engine.spec().names_target());
    let three = Three::of([2, 2, 0], |_, _| false);
    let asked = |carried: &[(&str, &str)], encode: Option<EncodeItems<'_>>| {
        let headers = Counted::of(carried);
        let lists = three.lists((&[], &[1; 2]));
        let request = PlaceRequest {
            headers: &headers,
            encode,
            ..PlaceRequest::of(MODEL, WorkerLists::Encode(lists))
        };
        let answer = placed(engine, &three, &request, 0);
        assert_eq!(headers.reads.get(), 1, "the header is looked for once");
        answer
    };
    let key = (MANUAL_SESSION_HEADER, "talk-1");
    let empty = Some(EncodeItems::default());
    for carried in [&[key][..], &[key, (TARGET, "")]] {
        assert!(asked(carried, None).is_ok(), "no items: as its pair");
        let answer = asked(carried, empty);
        let declined = matches!(
            answer,
            Err(Unplaced::Shed {
                reason: ShedReason::PolicyDeclined,
                emptied_by: "binding",
                ..
            })
        );
        assert!(declined, "an empty list: the entry for three lists");
    }
    let named = [key, (TARGET, "1")];
    assert_eq!(asked(&named, None), Ok((1, 1)), "its pair's alone");
    assert_eq!(asked(&named, empty), Ok((1, 1)), "its pair's alone");
}

/// Places its host states for the encoders stand beside a pair the
/// request names. An item goes to an encoder at them alone, whichever
/// holds fewer requests. Where they hold the request off every encoder
/// that takes it, the request is shed under the places' name, as it is
/// with no worker named: that shed is the encoders', by the decision's own
/// word, and the target is not missed. A place of the encoders that a
/// standing veto of their host masks takes no item either. A pair held
/// on its prefill alone is answered the same way. The second encoder
/// holds more requests, and the first takes no images.
///
/// The classic router states no places of its encoders: the case answers
/// as no line of it.
#[test]
fn places_stated_for_the_encoders_stand_beside_a_named_pair() {
    let engine = rings();
    let three = Three::of([2, 2, 2], |list, place| (list, place) != (2, 0));
    three.workers[2][1].increment_load();
    // Where a request under `demands` is placed, its item held to the
    // encoders at `places`.
    let asked = |carried: &[(&str, &str)], demands: Demands, places: &[u32]| {
        let headers = Counted::of(carried);
        let mut lists = three.lists((&[PLAIN; 2], &[1; 2]));
        lists.encoders_places = Some(places);
        let request = PlaceRequest {
            demands,
            ..carrying(lists, &headers)
        };
        let placement = engine.place(&request);
        placement.map(|placement| three.places(&placement))
    };
    let key = (MANUAL_SESSION_HEADER, "talk-1");
    let named = [key, (TARGET, "1")];
    let none = Demands::default();
    assert_eq!(asked(&named, none, &[0]), Ok((1, 1, vec![0])));
    let second = Ok((1, 1, vec![1]));
    assert_eq!(asked(&named, none, &[1]), second, "the place, not the load");

    // The request demands vision: the places hold it off the one encoder
    // that takes images.
    let vision = Demands {
        vision: true,
        ..none
    };
    let held_off = Err(Unplaced::Shed {
        reason: ShedReason::NoEndpoints,
        emptied_by: "places",
        retry_after_secs: 1,
        side: None,
    });
    assert_eq!(asked(&[key], vision, &[0]), held_off, "no worker named");
    assert_eq!(asked(&named, vision, &[0]), held_off, "as with none named");
    assert_eq!(asked(&named, vision, &[1]), second, "the one that sees");

    // A pair held on its prefill alone: the shed is its encoders' too.
    let alone = stock("consistent_hashing", "round_robin");
    let mut lists = three.lists((&[PLAIN; 2], &[1; 2]));
    lists.encoders_places = Some(&[0]);
    let request = PlaceRequest {
        demands: vision,
        ..carrying(lists, &named)
    };
    let answer = alone
        .place(&request)
        .map(|placement| three.places(&placement));
    assert_eq!(answer, held_off, "held on its prefill alone");

    // The first encoder stands under a standing veto of its host: the
    // item goes past it, to the one that holds more requests.
    let mut lists = three.lists((&[PLAIN; 2], &[1; 2]));
    lists.encoders_stalled = &[0];
    let request = carrying(lists, &named);
    let placement = engine.place(&request);
    let past = placement.map(|placement| three.places(&placement));
    assert_eq!(past, second, "past the masked encoder");
}

/// A place its host masks takes no number in an encode request's call, as
/// in a pair's own: the first decode stands under a standing veto of the
/// lists' host, so the number 0 reaches the second decode, and the number
/// 1 reaches none. The prefills' mask is read so too, and so are the
/// places the lists' host states for each side.
///
/// The mask is the engine's word to a host that lends it lists: the case
/// answers as no line of the classic router.
#[test]
fn a_place_its_host_masks_takes_no_number_in_an_encode_requests_call() {
    let engine = rings();
    let three = Three::of([2, 2, 2], |_, _| false);
    let mut lists = three.lists((&[PLAIN; 2], &[1; 2]));
    lists.pair.decode_stalled = &[0];
    assert_eq!(named(engine, &three, lists, "0"), Ok((0, 1)));
    let missed = Err(missed_on(PairSide::Decode));
    assert_eq!(named(engine, &three, lists, "1"), missed, "one decode");

    let mut lists = three.lists((&[PLAIN; 2], &[1; 2]));
    lists.pair.prefill_stalled = &[0];
    assert_eq!(named(engine, &three, lists, "0"), Ok((1, 0)), "the mask");
    let mut lists = three.lists((&[PLAIN; 2], &[1; 2]));
    lists.pair.prefill_places = Some(&[1]);
    assert_eq!(named(engine, &three, lists, "0"), Ok((1, 0)), "its places");
    let mut lists = three.lists((&[PLAIN; 2], &[1; 2]));
    lists.pair.decode_places = Some(&[1]);
    assert_eq!(
        named(engine, &three, lists, "0"),
        Ok((0, 1)),
        "the decode's"
    );
}

/// A pair held on its prefill alone asks again as any pair's call does,
/// and its rule is asked of the one place the number reached. Behind an
/// encoder that cannot be sent to, under a rule that takes every worker,
/// the call's later decisions stay on the prefill the request names,
/// whichever prefill owns its routing value, and the item goes to the
/// next encoder. Where no encoder can be sent to, the call has no worker
/// behind its third decision, as a request that names none: the prefill
/// it names could be sent to at each ask, so the target is not missed.
///
/// The classic router holds a side under its hash policy to the worker
/// the header names, whatever its filter takes (`P:3427-3428`;
/// `CH:187-193`).
#[test]
fn a_pair_held_on_its_prefill_alone_stays_on_it_behind_an_encoder() {
    let sides = ["power_of_two", "consistent_hashing", "round_robin"];
    let engine = stock_of(sides, keeps_its_encoders);
    let three = Three::of([3, 3, 2], |_, _| false);
    // The first encoder holds fewer requests, and its breaker is open.
    three.workers[2][1].increment_load();
    trip(&three.workers[2][0]);
    let rule = |_: &dyn Worker| true;
    for key in ["talk-1", "talk-2", "talk-3", "talk-4", "talk-5", "talk-6"] {
        let headers = [(MANUAL_SESSION_HEADER, key), (TARGET, "1")];
        let request = PlaceRequest {
            admits: Some(&rule),
            ..carrying(three.lists((&[PLAIN; 2], &[1; 3])), &headers)
        };
        let placement = engine.place(&request).expect("a placement");
        let (prefill, _, encoders) = three.places(&placement);
        assert_eq!((prefill, encoders), (1, vec![1]), "{key}");
    }

    trip(&three.workers[2][1]);
    let headers = [(MANUAL_SESSION_HEADER, "talk-1"), (TARGET, "1")];
    let request = PlaceRequest {
        admits: Some(&rule),
        ..carrying(three.lists((&[PLAIN; 2], &[1; 3])), &headers)
    };
    let answer = engine.place(&request).err();
    assert_eq!(answer, Some(Unplaced::NoWorkers), "its encoders' answer");
}

/// An encode request held to the two workers it names has its decode
/// decided with its prefill. Under a document that picks a pair's decode
/// behind its prefill, its decision books the decode with its prefill, as
/// a pair's own does: a decode asked again is refused as one that never
/// stood chosen, and the placement stands on the decode the request
/// named. The same request with no worker named has its decode decided
/// again, and so has one held to its prefill alone: it names no decode.
///
/// The classic router picks a pair's decode once, with its prefill, by
/// its pair selection (`P:3458-3460`, `P:3490`).
#[test]
fn an_encode_request_held_to_the_decode_it_names_books_it_with_its_prefill() {
    let later = |prefill: &str, decode: &str| {
        stock_of(["power_of_two", prefill, decode], |document| {
            document.plans.pd_pair.decode = DecodePick::AfterPrefill;
        })
    };
    let three = Three::of([3, 3, 2], |_, _| false);
    // The pair a request is placed on, what its decode asked again is
    // answered behind the prefill's dispatch, and the pair it then names.
    let again = |engine: &'static InProcess, carried: &[(&str, &str)]| {
        let headers = Counted::of(carried);
        let request = PlaceRequest {
            decode_again: true,
            ..carrying(three.lists((&[PLAIN; 2], &[1; 3])), &headers)
        };
        let mut placement = engine.place(&request).expect("a placement");
        let (prefill, decode, _) = three.places(&placement);
        placement.sent(Side::Prefill);
        let chosen = placement.decode_chosen();
        let answer = placement.decode_again(&request);
        assert_eq!(chosen, answer.is_ok(), "the placement's word ahead of it");
        let (_, then, _) = three.places(&placement);
        ((prefill, decode), answer, then)
    };
    const BOOKED: Unplaced = Unplaced::Refused(DecisionRefused::Decode(DecodeRefused::NotChosen));
    let ring = "consistent_hashing";
    let key = (MANUAL_SESSION_HEADER, "talk-1");
    let both = later(ring, ring);
    let named = again(both, &[key, (TARGET, "2")]);
    assert_eq!(named, ((2, 2), Err(BOOKED), 2), "the pair it names");
    let (_, answer, _) = again(both, &[key]);
    assert_eq!(
        answer,
        Ok(()),
        "no worker named: its decode is decided again"
    );
    let prefill_alone = later(ring, "round_robin");
    let ((prefill, _), answer, _) = again(prefill_alone, &[key, (TARGET, "2")]);
    assert_eq!((prefill, answer), (2, Ok(())), "no decode named");
}

/// A request that names its worker lends its items the ring it states, as
/// every encode request's call does. An item that carries its text goes to
/// the encoder that ring's walk reaches from where the text stands,
/// whatever ring its caller wrote into the items, beside the pair the
/// number reaches. The call lends the items whole: handed as one, over
/// encoders that push, they take one encoder beside that pair, and the
/// placement says so.
///
/// The classic router places an item by its content hash on the model's
/// ring, under headers that hold no number (`WS:2800-2819`).
#[test]
fn a_named_encode_requests_items_are_lent_the_ring_their_request_states() {
    let ring = "consistent_hashing";
    let engine = stock_of([ring; 3], |_| {});
    let three = Three::of([2, 2, 8], |_, _| false);
    let order = [6usize, 1, 4, 0, 7, 2, 5, 3];
    let reversed = [3usize, 5, 2, 7, 0, 4, 1, 6];
    let of = |order: &[usize]| {
        let url = |&at: &usize| three.workers[2][at].url().to_owned();
        Lent(order.iter().map(url).collect())
    };
    let (stated, turned) = (of(&order), of(&reversed));
    // Six texts at six places of the ring's eight.
    let texts = ["a0", "a1", "a2", "a3", "a4", "a5"];
    let keys = texts.map(Identity::plain);
    let headers = [(TARGET, "1")];
    let items = EncodeItems {
        keys: &keys,
        ring: Some(&turned),
        ..EncodeItems::default()
    };
    let request = PlaceRequest {
        headers: &headers,
        ring: Some(&stated),
        ..asking(three.lists((&[PLAIN; 8], &[1; 2])), items)
    };
    let placement = engine.place(&request).expect("a placement");
    let (prefill, decode, encoders) = three.places(&placement);
    // The encoder a ring of `order` reaches first from each text's place.
    let walked = |order: &[usize]| -> Vec<usize> {
        let first = |text: &&str| order[(stated.position(text) % 8) as usize];
        texts.iter().map(first).collect()
    };
    let by_the_call = walked(&order);
    assert_ne!(
        walked(&reversed),
        by_the_call,
        "the two rings place the items alike"
    );
    assert_eq!(encoders, by_the_call, "the call's ring, not the items' own");
    assert_eq!((prefill, decode), (1, 1), "the pair the number reaches");

    // Items handed as one, over encoders that push.
    let pushing = [EncoderFacts {
        pushes: true,
        ..PLAIN
    }; 8];
    let as_one = EncodeItems {
        one: Some(Identity::plain("a0-to-a5")),
        ..items
    };
    let request = PlaceRequest {
        headers: &headers,
        ring: Some(&stated),
        ..asking(three.lists((&pushing, &[1; 2])), as_one)
    };
    let placement = engine.place(&request).expect("a placement");
    assert!(placement.encoders.one(), "the items are placed as one");
    let (prefill, decode, encoders) = three.places(&placement);
    assert_eq!((prefill, decode, encoders.len()), (1, 1, 1));
}

/// A header reader that panics where it is asked for the target header
/// costs its encode request alone. The call reads that header under the
/// engine's catch, as the request's decision reads every other: it answers
/// a fault, the engine counts it as a decision's, and the next request is
/// placed. That is so whatever the request states of its items: with an
/// item, with no items stated, and with an empty list of them.
#[test]
fn a_header_reader_that_panics_on_the_target_costs_its_encode_request_alone() {
    /// Headers whose read of the target header panics.
    struct Panics;

    impl HeaderRead for Panics {
        fn header(&self, name: &str) -> Option<&[u8]> {
            assert_ne!(name, TARGET, "the case's reader panics here");
            None
        }
    }

    let engine = rings();
    let three = Three::of([2, 2, 2], |_, _| false);
    let lists = || three.lists((&[PLAIN; 2], &[1; 2]));
    let item = carrying(lists(), &Panics).encode;
    for encode in [item, None, Some(EncodeItems::default())] {
        let request = PlaceRequest {
            encode,
            ..carrying(lists(), &Panics)
        };
        let faults = engine.faults();
        let answer = engine.place(&request).err();
        let faulted = Unplaced::Refused(DecisionRefused::Faulted);
        assert_eq!(answer, Some(faulted), "the reader's panic is caught");
        assert_eq!(engine.faults(), faults + 1, "counted as a decision's");
    }
    assert_eq!(named(engine, &three, lists(), "1"), Ok((1, 1)), "the next");
}

/// Under a binding whose decode side alone is the hash ring, an encode
/// request that carries an item and names a worker is refused by name: it
/// is shed under `target-worker` for the decode's side, whatever its
/// number, and placed nowhere. A pair's decision counts such a number
/// behind the prefill it picks, and an encode request's decision counts
/// none. The header is read once. A request that names no worker is placed
/// as any encode request. Three lists with no item are placed by the
/// pair's own rule: on the decode at the number, under each of the
/// prefill's turns. They are so whether the request states no items or an
/// empty list of them. An empty list whose decode is held to the workers
/// that take images is an encode request's, and refused as one: where the
/// request says so of its decode, and where its items are references
/// beside an image.
///
/// Here the engine answers otherwise than the classic router, until an
/// encode request's decision counts a decode's number: classic holds the
/// decode alone there, behind the prefill its policy picked
/// (`P:3458-3460`, `P:3490`; `CH:187-193`).
#[test]
fn an_encode_request_named_on_its_decode_side_alone_is_refused_by_name() {
    let engine = stock("round_robin", "consistent_hashing");
    assert!(engine.spec().names_target());
    let three = Three::of([3, 3, 2], |_, _| false);
    let lists = || three.lists((&[PLAIN; 2], &[1; 3]));
    for number in ["0", "1", "7", "one"] {
        let carried = [(MANUAL_SESSION_HEADER, "talk-1"), (TARGET, number)];
        let headers = Counted::of(&carried);
        let answer = placed(engine, &three, &carrying(lists(), &headers), 1);
        assert_eq!(answer, Err(missed_on(PairSide::Decode)), "names {number}");
        assert_eq!(headers.reads.get(), 1, "names {number}: one read");
    }
    let unnamed = [(MANUAL_SESSION_HEADER, "talk-1"), (TARGET, "")];
    let answer = placed(engine, &three, &carrying(lists(), &unnamed), 1);
    assert!(answer.is_ok(), "an empty text names no worker");

    let headers = [(TARGET, "1")];
    for encode in [None, Some(EncodeItems::default())] {
        let no_item = PlaceRequest {
            headers: &headers,
            encode,
            ..PlaceRequest::of(MODEL, WorkerLists::Encode(lists()))
        };
        let mut prefills = BTreeSet::new();
        for _ in 0..6 {
            let (prefill, decode) = placed(engine, &three, &no_item, 0).expect("a placement");
            assert_eq!(decode, 1, "the named decode");
            prefills.insert(prefill);
        }
        assert_eq!(prefills.len(), 3, "the prefill takes its turns");
    }
    let to_images = EncodeItems {
        decode_images: true,
        ..EncodeItems::default()
    };
    let held_to_images = PlaceRequest {
        headers: &headers,
        ..asking(lists(), to_images)
    };
    let answer = placed(engine, &three, &held_to_images, 0);
    assert_eq!(answer, Err(missed_on(PairSide::Decode)), "the pass is laid");
    let references = EncodeItems {
        references: true,
        ..EncodeItems::default()
    };
    let beside_an_image = PlaceRequest {
        headers: &headers,
        media: MediaCounts {
            images: 1,
            ..MediaCounts::NONE
        },
        ..asking(lists(), references)
    };
    let answer = placed(engine, &three, &beside_an_image, 0);
    assert_eq!(answer, Err(missed_on(PairSide::Decode)), "beside an image");
}

/// The numbers count among the workers that meet each side's part of the
/// request's hard demands, as the request's decision holds that side. A
/// request that carries an item leaves the media's part of its demand to
/// its encoders: a worker of its pair that takes no images takes a number
/// all the same, and one off the request's wire takes none. A request
/// that carries no item takes no encoder, and holds its pair to the whole
/// demand: a worker that takes no images takes no number there. That is
/// so for one the encode pass is not laid for, which is its pair's alone,
/// and for one that holds its decode to images with no item. Each other
/// part of a demand is counted by the side that holds it: the way a
/// constraint is decided is the decodes' alone, with an item as without,
/// and references are the encoders' where the request carries an item.
/// A window stays the pair's beside an item: the input's bound counts
/// both sides, and the completion's the decodes alone.
///
/// The first prefill and the first decode take no images and state a
/// window of 64, the third of each state another runtime than the
/// request's wire, and both encoders take images on that wire. The
/// decodes behind the first decide a constraint the way the request asks,
/// and the encoders alone process references.
///
/// The classic router narrows both pools of a pinned request to its wire
/// ahead of its policies (`P:3208-3213`), and holds an encode request's
/// media to its encoders (`WS:2595-2605`).
#[test]
fn the_numbers_count_among_the_workers_that_meet_the_pairs_part_of_a_demand() {
    let engine = rings();
    let mut three = Three::of([3, 3, 2], |_, _| false);
    three.published = [0, 1, 2].map(|list| {
        let (first, third) = (
            Arc::clone(&three.workers[list][0]),
            three.workers[list].get(2).map(Arc::clone),
        );
        PublishedList::with_facts(Arc::clone(&three.workers[list]), |worker| {
            let apart = third
                .as_ref()
                .is_some_and(|third| Arc::ptr_eq(worker, third));
            let behind = !Arc::ptr_eq(worker, &first);
            let mut facts = FixedFacts {
                vision: list == 2 || behind,
                ..FixedFacts::default()
            };
            facts.demands.runtime = if apart { 1 } else { 2 };
            facts.demands.decoding = if list == 1 && behind { 3 } else { 0 };
            facts.demands.references = list == 2;
            facts.demands.window = if list == 2 || behind { 0 } else { 64 };
            facts
        })
    });
    let asked = |number: &str, demands: Demands, items: EncodeItems<'_>| {
        let headers = [(TARGET, number)];
        let request = PlaceRequest {
            headers: &headers,
            demands,
            ..asking(three.lists((&[PLAIN; 2], &[1; 3])), items)
        };
        placed(engine, &three, &request, items.keys.len())
    };
    let at = |number: &str, demands: Demands, keys: &[Identity<'static>]| {
        let items = EncodeItems {
            keys,
            ..EncodeItems::default()
        };
        asked(number, demands, items)
    };
    let media = Demands {
        vision: true,
        ..Demands::default()
    };
    let wire = Demands {
        runtime: 2,
        ..media
    };
    let missed = Err(missed_on(PairSide::Prefill));

    // An item: its encoders meet the media's part, and the pair the rest.
    let item = &KEYS[..];
    for place in 0..3 {
        let number = place.to_string();
        assert_eq!(at(&number, media, item), Ok((place, place)), "{place}");
    }
    assert_eq!(at("0", wire, item), Ok((0, 0)), "the first on the wire");
    assert_eq!(at("1", wire, item), Ok((1, 1)));
    assert_eq!(at("2", wire, item), missed, "two a side on the wire");

    // No item: the pair meets the whole demand.
    let none: &[Identity<'static>] = &[];
    assert_eq!(at("0", media, none), Ok((1, 1)), "the first that sees");
    assert_eq!(at("1", media, none), Ok((2, 2)));
    assert_eq!(at("2", media, none), missed, "two a side take images");
    assert_eq!(at("0", wire, none), Ok((1, 1)), "the one that meets both");
    assert_eq!(at("1", wire, none), missed, "one a side meets both");

    // No item, and a decode held to images: an encode request's own call,
    // whose pair meets the whole demand too.
    let to_images = EncodeItems {
        decode_images: true,
        ..EncodeItems::default()
    };
    assert_eq!(
        asked("0", media, to_images),
        Ok((1, 1)),
        "the first that sees"
    );
    assert_eq!(asked("1", media, to_images), Ok((2, 2)));
    assert_eq!(
        asked("2", media, to_images),
        missed,
        "two a side take images"
    );

    // The way a constraint is decided holds the decodes alone, beside an
    // item too: the decode's number passes over the first decode, and
    // every prefill takes a number.
    let decided = Demands {
        decoding: 3,
        ..Demands::default()
    };
    assert_eq!(
        at("0", decided, item),
        Ok((0, 1)),
        "the first that decides so"
    );
    assert_eq!(at("1", decided, item), Ok((1, 2)));
    let on_decodes = Err(missed_on(PairSide::Decode));
    assert_eq!(at("2", decided, item), on_decodes, "two decodes decide so");

    // A window is the pair's to hold beside an item too: the input's
    // bound holds both sides, and the completion's the decodes alone. The
    // first prefill and the first decode state a window of 64.
    let input = Demands {
        input_window: 128,
        ..Demands::default()
    };
    assert_eq!(at("0", input, item), Ok((1, 1)), "the first that holds it");
    assert_eq!(at("2", input, item), missed, "two a side hold the input");
    let completion = Demands {
        output_window: 128,
        ..Demands::default()
    };
    assert_eq!(at("0", completion, item), Ok((0, 1)), "the decodes alone");
    assert_eq!(at("2", completion, item), on_decodes, "two decodes hold it");

    // References are the encoders' to process beside an item: no worker
    // of the pair processes them, and each takes its number.
    let by_reference = Demands {
        references: References::Any,
        ..Demands::default()
    };
    for place in 0..3 {
        let number = place.to_string();
        let pair = at(&number, by_reference, item);
        assert_eq!(pair, Ok((place, place)), "by reference, {place}");
    }
}
