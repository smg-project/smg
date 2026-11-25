//! The cases of a pair that names its worker: a side at a time, by the
//! number its request carries under the binding's target source.
//!
//! Every case names the lines of the classic router's pair selection it
//! answers as (`model_gateway/src/routers/common/placement.rs`, `P` below)
//! and of its hash policy (`model_gateway/src/policies/consistent_hashing.rs`,
//! `CH` below), on main `4b41f39a7`.

use super::{
    super::target::{names_target, Counted, MISSED, TARGET},
    *,
};
use crate::engine::DecodeRefused;

mod decode_alone;

/// The stock pair of the presets `prefill` and `decode`. A side that is
/// the hash ring brings the binding its target source.
fn stock(prefill: &str, decode: &str) -> &'static InProcess {
    let mut document = PolicySpec::pair_presets(prefill, decode).expect("presets");
    document.generation = 1;
    hosted(engine_of(&document))
}

/// The stock pair of two hash rings: both sides read the target source.
fn rings() -> &'static InProcess {
    stock("consistent_hashing", "consistent_hashing")
}

/// The places in `lists` of the pair `request` is placed on.
fn placed(
    engine: &'static InProcess,
    lists: &Lists,
    request: &PlaceRequest<'_>,
) -> Result<(usize, usize), Unplaced> {
    let placement = engine.place(request)?;
    Ok(lists.places(&placement))
}

/// The places of the pair a request that names `number` is placed on over
/// `pair`, under one routing value.
fn named(
    engine: &'static InProcess,
    lists: &Lists,
    pair: PairLists<'_>,
    number: &str,
) -> Result<(usize, usize), Unplaced> {
    let headers = [(MANUAL_SESSION_HEADER, "talk-1"), (TARGET, number)];
    let request = PlaceRequest {
        headers: &headers,
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    placed(engine, lists, &request)
}

/// A pair that names its worker goes to the worker at that number on each
/// side, ahead of its routing value: under the stock pair of two hash
/// rings, over three prefills and three decodes that all pair and are all
/// available, each number takes the prefill and the decode at that place,
/// whichever workers own the request's routing value, and with no routing
/// value too. The header is read once a call. A pair that names no worker
/// goes to its routing value's owners, as before.
///
/// The classic router reads the header once a side, each side's policy
/// over its own workers (`P:2854-2855`, `P:2914-2920`; `CH:187-193`), and
/// holds a pair so on both sides (`P:3695`).
#[test]
fn a_pair_that_names_its_worker_goes_to_that_worker_on_each_side() {
    let engine = rings();
    assert!(engine.spec().names_target());
    let lists = Lists::of(3, 3);
    let classes = lists.every();
    let asked = |carried: &[(&str, &str)]| {
        let headers = Counted::of(carried);
        let request = PlaceRequest {
            headers: &headers,
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(lists.under(&classes, &Open)))
        };
        let places = placed(engine, &lists, &request).expect("a placement");
        (headers.reads.get(), places)
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

/// A side reads the name where its profile is the hash ring keyed by no
/// prefix. Under a ring prefill beside a round robin decode the prefill is
/// the one named and the decode takes its turns. Under a round robin
/// prefill beside a ring decode the prefill takes its turns and the decode
/// is the one named, whatever the request's routing value
/// ([`decode_alone`]). A pair none of whose sides is the ring is read for
/// no target, though its binding names a target source. Under a document
/// that names no target source the header is never read.
///
/// The classic router holds the side whose own policy reads the header and
/// no other: the prefill alone (`P:3695`, its second half), the decode
/// alone behind the prefill its policy picked (`P:3477`).
#[test]
fn a_side_is_held_where_its_profile_is_the_plain_ring() {
    let lists = Lists::of(3, 3);
    let classes = lists.every();
    // The pairs six requests that name `1` are placed on, and how often
    // each read the header.
    let six = |engine: &'static InProcess| {
        let mut sides = [BTreeSet::new(), BTreeSet::new()];
        let mut reads = BTreeSet::new();
        for _ in 0..6 {
            let headers = Counted::of(&[(MANUAL_SESSION_HEADER, "talk-1"), (TARGET, "1")]);
            let request = PlaceRequest {
                headers: &headers,
                ..PlaceRequest::of(MODEL, WorkerLists::Pair(lists.under(&classes, &Open)))
            };
            let (prefill, decode) = placed(engine, &lists, &request).expect("a placement");
            sides[0].insert(prefill);
            sides[1].insert(decode);
            reads.insert(headers.reads.get());
        }
        (sides, reads)
    };
    let (named, once) = (BTreeSet::from([1_usize]), BTreeSet::from([1_u32]));

    let ([prefills, decodes], reads) = six(stock("consistent_hashing", "round_robin"));
    assert_eq!(
        (prefills, reads),
        (named.clone(), once.clone()),
        "the prefill is held"
    );
    assert_eq!(decodes.len(), 3, "the decode takes its turns");

    let ring_decode = stock("round_robin", "consistent_hashing");
    assert!(ring_decode.spec().names_target());
    let ([prefills, decodes], reads) = six(ring_decode);
    assert_eq!(prefills.len(), 3, "the prefill takes its turns");
    assert_eq!(
        (decodes, reads),
        (named, once.clone()),
        "the decode is the one named"
    );

    let mut document = PolicySpec::pair_presets("round_robin", "round_robin").expect("presets");
    document.generation = 1;
    names_target(&mut document);
    let no_ring = hosted(engine_of(&document));
    assert!(no_ring.spec().names_target());
    let ([prefills, decodes], reads) = six(no_ring);
    assert_eq!((prefills.len(), decodes.len()), (3, 3), "no side is held");
    assert_eq!(reads, once);

    let unnamed = stock("round_robin", "round_robin");
    assert!(!unnamed.spec().names_target());
    let (_, reads) = six(unnamed);
    let never = BTreeSet::from([0_u32]);
    assert_eq!(reads, never, "no read under a document that names none");
}

/// The numbers count the workers that can be sent to when the call reads
/// them, in each list's order: a prefill that is not ready, a decode over
/// its overload thresholds and a prefill behind an open breaker are each
/// passed over, and the number lands on the next. The route's rule is
/// counted the same way: a worker it refuses stands at no number, and a
/// number past the workers it takes is shed.
///
/// The classic router counts the candidates its policy is handed: the
/// workers the route's rule takes that are available (`P:2635-2640`,
/// `P:2676-2680`, `P:2818`; `CH:150-162`).
#[test]
fn the_numbers_count_the_workers_that_can_be_sent_to() {
    let engine = rings();
    let at = |lists: &Lists, number: &str| {
        let classes = lists.every();
        named(engine, lists, lists.under(&classes, &Open), number)
    };

    let lists = Lists::of(3, 3);
    lists.prefills[1].set_status(WorkerStatus::NotReady);
    assert_eq!(
        at(&lists, "1"),
        Ok((2, 1)),
        "past the prefill that is not ready"
    );
    assert_eq!(at(&lists, "2"), Err(MISSED), "two prefills are counted");

    let lists = Lists::of(3, 3);
    lists.decodes[0].set_overloaded(true);
    assert_eq!(
        at(&lists, "0"),
        Ok((0, 1)),
        "past the decode over its thresholds"
    );

    let lists = Lists::of(3, 3);
    trip(&lists.prefills[0]);
    assert_eq!(
        at(&lists, "0"),
        Ok((1, 0)),
        "past the prefill behind its breaker"
    );

    let lists = Lists::of(3, 3);
    let classes = lists.every();
    let refused = [lists.prefills[0].url(), lists.decodes[2].url()];
    let rule = |worker: &dyn Worker| !refused.contains(&worker.url());
    let ruled = |number: &str| {
        let headers = [(TARGET, number)];
        let request = PlaceRequest {
            headers: &headers,
            admits: Some(&rule),
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(lists.under(&classes, &Open)))
        };
        placed(engine, &lists, &request)
    };
    assert_eq!(ruled("0"), Ok((1, 0)));
    assert_eq!(ruled("1"), Ok((2, 1)));
    assert_eq!(ruled("2"), Err(MISSED), "the rule takes two workers a side");
}

/// A prefill is counted where it has a partner that can be sent to, and a
/// decode where it is such a partner of the prefill the number reached.
/// Two classes pair apart here: the first prefill with the first decode,
/// the second with the other two. The decode's number counts its own
/// prefill's partners, so number 1 is the second prefill with the second
/// of its own decodes. With the first class's decode not ready, number 0
/// is the second prefill with the first of its decodes, and number 1
/// reaches nobody.
///
/// The classic router's candidates are the prefills with an open partner
/// (`P:2723-2728`, `P:2754-2771`), and its decode's number counts the
/// open partners of the prefill picked (`P:2884-2887`).
#[test]
fn a_prefill_is_counted_where_one_of_its_partners_can_be_sent_to() {
    let engine = rings();
    let lists = Lists::of(2, 3);
    let classes = PartnerClasses::number(3, [&[0_u32][..], &[1, 2][..]]);
    let at = |number: &str| named(engine, &lists, lists.under(&classes, &Open), number);
    assert_eq!(
        (at("0"), at("1")),
        (Ok((0, 0)), Ok((1, 2))),
        "each among its own"
    );
    lists.decodes[0].set_status(WorkerStatus::NotReady);
    assert_eq!(at("0"), Ok((1, 1)), "the one prefill with a partner left");
    assert_eq!(at("1"), Err(MISSED));
}

/// Where a prefill that can be sent to has a first partner that can, the
/// numbers count such prefills alone and each one's first partners alone,
/// as a pair's decision shows them alone. Where none has, they count
/// every prefill with a partner and every such partner.
///
/// The classic router's candidates are the prefills with an open partner
/// on their own known runtime while there is one, and the pick's decode is
/// one of its partners on that runtime (`P:2754-2771`, `P:2884-2887`).
#[test]
fn first_partners_are_counted_alone_while_one_can_be_sent_to() {
    let engine = rings();
    let lists = Lists::of(2, 2);
    // Both prefills pair with both decodes, and prefill 1 with decode 1
    // first.
    let both: &[u32] = &[0, 1];
    let classes = PartnerClasses::number_first(2, [(both, &[][..]), (both, &[1][..])]);
    let at = |number: &str| named(engine, &lists, lists.under(&classes, &Open), number);
    assert_eq!(at("0"), Ok((1, 1)), "the one pair that goes first");
    assert_eq!(
        at("1"),
        Err(MISSED),
        "no second prefill has a first partner"
    );
    lists.decodes[1].set_status(WorkerStatus::NotReady);
    assert_eq!(at("0"), Ok((0, 0)), "every prefill with a partner");
    assert_eq!(at("1"), Err(MISSED), "prefill 1 has one partner left");
}

/// A named worker that is full answers as its side's own gate does. A
/// named prefill at its caller's admission limit is waited for: the call
/// answers the word its caller waits on, though another prefill has room.
/// A named decode with no free room is taken, and the request waits at
/// that decode's gate.
///
/// The classic router runs neither admission filter under a named worker
/// (`P:2804-2814`, `P:2895`, `P:2978-2987`): the named prefill at its
/// limit makes the request wait for a slot (`P:2876-2881`), and the named
/// decode is picked full (`P:3477`).
#[test]
fn a_named_prefill_at_its_limit_is_waited_for_and_a_full_decode_is_taken() {
    let engine = rings();
    let lists = Lists::of(2, 2);
    let classes = lists.every();
    let full_prefill = Gates(|prefill| prefill != 1, |_| true);
    let at = |number: &str| named(engine, &lists, lists.under(&classes, &full_prefill), number);
    assert_eq!(at("1"), Err(Unplaced::PrefillsFull));
    assert_eq!(at("0"), Ok((0, 0)));
    let full_decode = Gates(|_| true, |decode| decode != 1);
    let at = |number: &str| named(engine, &lists, lists.under(&classes, &full_decode), number);
    assert_eq!(at("1"), Ok((1, 1)), "the decode with no free room");
}

/// A retry that names its pair is handed the pair again, though its failed
/// attempt was sent to both workers. A side that is not held steps aside
/// from the worker the request tried.
///
/// The classic router keeps the tried workers in a held side's candidates,
/// and narrows a side under another policy (`P:2826`, `P:2902`, `P:3695`).
#[test]
fn a_retry_that_names_its_pair_is_handed_it_again() {
    let lists = Lists::of(3, 3);
    let classes = lists.every();
    let headers = [(MANUAL_SESSION_HEADER, "talk-1"), (TARGET, "1")];
    let retry = |engine: &'static InProcess, tried: &[WorkerId]| {
        let request = PlaceRequest {
            headers: &headers,
            attempt: 1,
            tried,
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(lists.under(&classes, &Open)))
        };
        placed(engine, &lists, &request)
    };
    let tried = [lists.prefill_id(1), lists.decode_id(1)];
    assert_eq!(retry(rings(), &tried), Ok((1, 1)));

    let ring_prefill = stock("consistent_hashing", "round_robin");
    for _ in 0..6 {
        let (prefill, decode) = retry(ring_prefill, &tried).expect("a placement");
        assert_eq!(prefill, 1, "the named prefill, though tried");
        assert_ne!(decode, 1, "the decode steps aside");
    }
}

/// A pair held to its prefill alone asks again as any pair's does, and
/// stays at that prefill. The route's rule takes the last of sixteen
/// decodes alone here, and the decode's profile takes its turns over all
/// of them: a decision names four targets at most, so most first decisions
/// name no decode the rule takes. The call then asks the rule of the named
/// prefill's one place and of the decodes, and its next decision takes the
/// named prefill with the one decode, under every routing value.
///
/// The classic router hands its decode policy the partners the route's
/// rule takes (`P:2635-2640`, `P:2884-2887`), behind a prefill its hash
/// policy holds (`P:2854-2855`).
#[test]
fn a_pair_held_to_its_prefill_alone_asks_again_at_that_prefill() {
    let engine = stock("consistent_hashing", "round_robin");
    let lists = Lists::of(3, 16);
    let classes = lists.every();
    let taken = lists.decodes[15].url();
    let prefills: Vec<_> = lists.prefills.iter().map(|prefill| prefill.url()).collect();
    let rule = |worker: &dyn Worker| worker.url() == taken || prefills.contains(&worker.url());
    for key in ["talk-1", "talk-2", "talk-3", "talk-4", "talk-5", "talk-6"] {
        for number in ["0", "1", "2"] {
            let headers = [(MANUAL_SESSION_HEADER, key), (TARGET, number)];
            let request = PlaceRequest {
                headers: &headers,
                admits: Some(&rule),
                ..PlaceRequest::of(MODEL, WorkerLists::Pair(lists.under(&classes, &Open)))
            };
            let place: usize = number.parse().expect("a number");
            let pair = placed(engine, &lists, &request);
            assert_eq!(pair, Ok((place, 15)), "{key}, names {number}");
        }
    }
}

/// A pair its caller holds to places of its lists, as a retry is held to
/// its wire, is counted among those places.
///
/// The classic router counts the workers on the retry's wire (`P:2635-2640`).
#[test]
fn a_pair_held_to_places_is_counted_among_them() {
    let engine = rings();
    let lists = Lists::of(3, 3);
    let classes = lists.every();
    let at = |number: &str| {
        let pair = PairLists {
            prefill_places: Some(&[1, 2][..]),
            decode_places: Some(&[0, 2][..]),
            ..lists.under(&classes, &Open)
        };
        named(engine, &lists, pair, number)
    };
    assert_eq!((at("0"), at("1")), (Ok((1, 0)), Ok((2, 2))));
    assert_eq!(at("2"), Err(MISSED));
}

/// A pair whose number reaches no worker is shed with no endpoint under
/// `target-worker`, and moves to no other worker: a text that is no
/// number, a number past the workers counted, and a pair the rule takes
/// when it is counted and refuses as the pick. Behind that pick no second
/// decision is made: the rule is asked of the two picks and of no worker
/// more. An empty text names no worker.
///
/// The classic policy misses a text it cannot read as a number and a
/// number outside its workers, and names no other worker (`CH:155-161`,
/// `CH:191`): the pair is refused on that side (`P:2873`, `P:2919`).
#[test]
fn a_number_that_reaches_no_pair_is_shed_under_its_own_name() {
    let engine = rings();
    let lists = Lists::of(2, 2);
    let classes = lists.every();
    let at = |number: &str| named(engine, &lists, lists.under(&classes, &Open), number);
    assert_eq!(at("one"), Err(MISSED));
    assert_eq!(at("2"), Err(MISSED));
    assert!(at("").is_ok(), "an empty text names no worker");

    // The count asks the rule of the two decodes and the two prefills.
    let asked = Cell::new(0_u32);
    let rule = |_: &dyn Worker| {
        asked.set(asked.get() + 1);
        asked.get() <= 4
    };
    let headers = [(TARGET, "1")];
    let request = PlaceRequest {
        headers: &headers,
        admits: Some(&rule),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(lists.under(&classes, &Open)))
    };
    assert_eq!(placed(engine, &lists, &request), Err(MISSED));
    assert_eq!(asked.get(), 6, "one decision's two picks");
}

/// Where no decode can be sent to, a decode over its overload thresholds
/// alone counts as a prefill's partner: the prefill's number is read, and
/// the decode's is not. The decode goes by the binding's rule for decodes
/// that are all over them, to the one that holds the fewest requests.
/// Under a binding that sheds such a pool no decode counts so: no prefill
/// is counted, the header is not read, and the pair is shed as saturated,
/// as the same pair is with no named worker.
///
/// The classic router counts a prefill by a partner over its thresholds
/// alone where no decode at all is available and its shedding is off, and
/// takes the least-loaded of the pick's partners with no read of the
/// header (`P:2648`, `P:2653-2655`, `P:2675`, `P:2909-2912`). With its
/// shedding on it finds no prefill with a partner and refuses the pair for
/// the decode pool's overload (`P:2772-2799`).
#[test]
fn a_decode_pool_all_over_its_thresholds_keeps_the_prefills_number_alone() {
    let engine = rings();
    let lists = Lists::of(2, 2);
    let classes = lists.every();
    for decode in lists.decodes.iter() {
        decode.set_overloaded(true);
    }
    lists.decodes[0].increment_load();
    let at = |number: &str| named(engine, &lists, lists.under(&classes, &Open), number);
    assert_eq!((at("0"), at("1")), (Ok((0, 1)), Ok((1, 1))));
    assert_eq!(at("2"), Err(MISSED), "the prefills are counted");

    let mut document =
        PolicySpec::pair_presets("consistent_hashing", "consistent_hashing").expect("presets");
    document.generation = 1;
    document.bindings[0].overload = Some(OverloadSpec {
        shed_when_all_vetoed: true,
        ..OverloadSpec::default()
    });
    let shedding = hosted(engine_of(&document));
    let reason = |answer: Result<(usize, usize), Unplaced>| match answer {
        Err(Unplaced::Shed { reason, .. }) => Some(reason),
        _ => None,
    };
    let unnamed = PlaceRequest::of(MODEL, WorkerLists::Pair(lists.under(&classes, &Open)));
    let saturated = Some(ShedReason::Saturated);
    assert_eq!(reason(placed(shedding, &lists, &unnamed)), saturated);
    for number in ["0", "1", "2", "one"] {
        let answer = named(shedding, &lists, lists.under(&classes, &Open), number);
        assert_ne!(answer, Err(MISSED), "names {number}");
        assert_eq!(reason(answer), saturated, "names {number}");
    }
}

/// Where no prefill is counted there is nobody to name: the pair leaves
/// its hold and is placed as one that names no worker, whatever its
/// header's text. Over prefills that are all over their thresholds it goes
/// to the one that holds the fewest requests, under each routing value,
/// as the same pair does with no named worker: a pair's routing value
/// keeps no prefill over a veto, so the pair is made with its own.
///
/// The classic router finds no prefill its policy could be handed, and
/// pairs by load with no read of the header (`P:2729-2747`).
#[test]
fn where_no_prefill_is_counted_the_pair_leaves_its_hold() {
    let engine = rings();
    let lists = Lists::of(2, 2);
    let classes = lists.every();
    for prefill in lists.prefills.iter() {
        prefill.set_overloaded(true);
    }
    lists.prefills[1].increment_load();
    let at = |number: &str| named(engine, &lists, lists.under(&classes, &Open), number);
    for number in ["1", "7", "one"] {
        let (prefill, _) = at(number).expect("a placement");
        assert_eq!(prefill, 0, "names {number}: the prefill of the least load");
    }
    let prefill_of = |carried: &[(&str, &str)]| {
        let headers = Counted::of(carried);
        let request = PlaceRequest {
            headers: &headers,
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(lists.under(&classes, &Open)))
        };
        placed(engine, &lists, &request).expect("a placement").0
    };
    for key in ["talk-1", "talk-2", "talk-3", "talk-4", "talk-5", "talk-6"] {
        let keyed = (MANUAL_SESSION_HEADER, key);
        assert_eq!(prefill_of(&[keyed]), 0, "{key}, no named worker");
        assert_eq!(prefill_of(&[keyed, (TARGET, "1")]), 0, "{key}, names 1");
    }
}

/// The numbers count among the workers that meet what each side's role is
/// held to of the request's hard demands: a worker that fails one takes
/// no number on its side, as none takes a single request's. A prefill is
/// not held to the way a constraint is decided, so that demand leaves the
/// prefills' count as it is. A number past the workers that meet the
/// demand sheds the pair under `target-worker`. Where a demand leaves a
/// side nobody there is none to count, and the pair is shed under the
/// demand's own stage, as any pair is.
///
/// The classic router narrows both pools of a pinned pair to its wire
/// ahead of its policies, which count in what is left (`P:2635-2640`, read
/// at `P:2677`).
#[test]
fn the_numbers_count_among_the_workers_that_meet_the_requests_demands() {
    let engine = rings();
    let mut lists = Lists::of(3, 3);
    // The first worker of each list states the runtime 1 and the others
    // the runtime 2. The third decode alone states a way to decide a
    // constraint.
    let stating = |list: &Arc<[Arc<dyn Worker>]>, decoding: u8| {
        let (first, third) = (Arc::clone(&list[0]), Arc::clone(&list[2]));
        PublishedList::with_facts(Arc::clone(list), |worker| {
            let mut facts = FixedFacts::default();
            facts.demands.runtime = if Arc::ptr_eq(worker, &first) { 1 } else { 2 };
            if Arc::ptr_eq(worker, &third) {
                facts.demands.decoding = decoding;
            }
            facts
        })
    };
    lists.published = [stating(&lists.prefills, 0), stating(&lists.decodes, 3)];
    let classes = lists.every();
    let placed = |number: &str, demands: Demands| {
        let headers = [(TARGET, number)];
        let request = PlaceRequest {
            headers: &headers,
            demands,
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(lists.under(&classes, &Open)))
        };
        let placement = engine.place(&request);
        placement.map(|placement| lists.places(&placement))
    };
    let none = Demands::default();
    assert_eq!(placed("0", none), Ok((0, 0)), "no demand: every worker");

    // A pinned pair's wire: two workers a side state its runtime.
    let wire = Demands { runtime: 2, ..none };
    assert_eq!(placed("0", wire), Ok((1, 1)), "the first on the wire");
    assert_eq!(placed("1", wire), Ok((2, 2)));
    assert_eq!(placed("2", wire), Err(MISSED), "two a side");
    let other = Demands { runtime: 1, ..none };
    assert_eq!(placed("0", other), Ok((0, 0)), "the other wire's one");
    assert_eq!(placed("1", other), Err(MISSED));

    // The prefills are not held to the way: all three are counted, and
    // the one decode that states it is the reached prefill's one partner.
    let decided = Demands {
        decoding: 3,
        ..none
    };
    assert_eq!(placed("0", decided), Ok((0, 2)), "any prefill");
    assert_eq!(placed("1", decided), Err(MISSED), "one decode");

    // No decode states the way 4: there is no partner to count a prefill
    // by, and the pair is shed as one that names no worker is.
    let unmet = Demands {
        decoding: 4,
        ..none
    };
    let shed = Unplaced::Shed {
        reason: ShedReason::NoEndpoints,
        emptied_by: "decoding",
        retry_after_secs: 1,
    };
    assert_eq!(placed("0", unmet), Err(shed));
}

/// A place a standing veto of the lists' host masks takes no number, on
/// either side: a host whose worker objects do not carry its vetoes states
/// them by place beside each list, and the numbers count as they do over
/// worker objects that carry them. A decode over its thresholds alone is a
/// partner at no masked place, so a prefill whose one partner stands there
/// is not counted. Where a side's every place is masked there is nobody to
/// count, and the pair is placed as one that names no worker, whatever its
/// header's text: that decision lifts a mask no hold stands on as its last
/// resort, and takes the pair of the least load.
///
/// Every worker the classic router counts on a side passes the worker's
/// own test, or is over its thresholds alone under the lift
/// (`P:2676-2679`): its worker objects carry each veto of its own.
#[test]
fn a_place_its_host_masks_takes_no_number() {
    let engine = rings();
    let lists = Lists::of(3, 3);
    let classes = lists.every();
    let masking = |prefills: &'static [u32], decodes: &'static [u32]| PairLists {
        prefill_stalled: prefills,
        decode_stalled: decodes,
        ..lists.under(&classes, &Open)
    };
    let at = |pair: PairLists<'_>, number: &str| named(engine, &lists, pair, number);

    let pair = masking(&[0], &[]);
    assert_eq!(at(pair, "0"), Ok((1, 0)), "past the masked prefill");
    assert_eq!(at(pair, "1"), Ok((2, 1)));
    assert_eq!(at(pair, "2"), Err(MISSED), "two prefills are counted");
    let pair = masking(&[], &[0]);
    assert_eq!(at(pair, "0"), Ok((0, 1)), "past the masked decode");
    assert_eq!(at(pair, "1"), Ok((1, 2)));
    assert_eq!(at(pair, "2"), Err(MISSED), "two decodes are counted");

    // Every decode is masked and none is held: nobody is counted, and the
    // pair goes where the same pair goes with no named worker, to the
    // pair of the least load.
    for list in [&lists.prefills, &lists.decodes] {
        list[0].increment_load();
        list[1].increment_load();
    }
    let pair = masking(&[], &[0, 1, 2]);
    let talk = [(MANUAL_SESSION_HEADER, "talk-1")];
    let unnamed = PlaceRequest {
        headers: &talk,
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    let unnamed = placed(engine, &lists, &unnamed);
    assert_eq!(unnamed, Ok((2, 2)), "no named worker");
    for number in ["0", "1", "2", "one"] {
        assert_eq!(at(pair, number), Ok((2, 2)), "names {number}");
    }

    // Decodes all over their thresholds: the first prefill's one partner
    // is masked, so the second prefill alone is counted.
    let lists = Lists::of(2, 3);
    let classes = PartnerClasses::number(3, [&[0_u32][..], &[1, 2][..]]);
    for decode in lists.decodes.iter() {
        decode.set_overloaded(true);
    }
    lists.decodes[1].increment_load();
    let at = |number: &str| {
        let pair = PairLists {
            decode_stalled: &[0],
            ..lists.under(&classes, &Open)
        };
        named(engine, &lists, pair, number)
    };
    assert_eq!(at("0"), Ok((1, 2)), "its partners are not masked");
    assert_eq!(at("1"), Err(MISSED), "one prefill is counted");
}

/// A pair that names its worker under a capped sessions rule goes to that
/// worker on each side, or to none. The rule takes the first pin the
/// decision's filters kept, and a decision held to the named places keeps
/// no other worker: a session pinned on another pair follows its request
/// to the pair it names, a pin at the rule's cap on a named worker has no
/// worker to hand the session on to, and a number past the workers sheds
/// the pair under `target-worker` whatever the session's pins.
///
/// The classic router's sticky override never runs on a side under
/// consistent hashing: a policy that reads the routing value itself is
/// not overridden (`sticky_for` in `model_gateway/src/policies/registry.rs`),
/// so its hash policy reads the header on each side as with no override.
#[test]
fn a_pair_that_names_its_worker_under_a_capped_sessions_rule_goes_there_or_nowhere() {
    let mut document =
        PolicySpec::pair_presets("consistent_hashing", "consistent_hashing").expect("presets");
    document.generation = 1;
    // The stock rule with a cap of one request a session on a worker. Its
    // own header names are left out: the binding's sources name the key.
    let stock = PolicySpec::preset("manual").expect("a built-in preset");
    let mut rule = stock.bindings[0].sessions.clone().expect("the stock rule");
    rule.sticky_header.clear();
    rule.headers.clear();
    rule.inflight_cap = 1;
    document.bindings[0].sessions = Some(rule);
    let engine = hosted(engine_of(&document));
    let lists = Lists::of(3, 3);
    let classes = lists.every();
    // The pair a request of the session is sent to, and its placement.
    let sent = |number: &str| {
        let headers = [(MANUAL_SESSION_HEADER, "talk-1"), (TARGET, number)];
        let request = PlaceRequest {
            headers: &headers,
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(lists.under(&classes, &Open)))
        };
        engine.place(&request).map(|mut placement| {
            placement.sent(Side::Prefill);
            placement.sent(Side::Decode);
            (lists.places(&placement), placement.kept())
        })
    };
    let pair_of = |number: &str| sent(number).map(|(pair, _)| pair);
    // The session's first pair is sent and stays in flight: its pins, and
    // one request of the session on each of its workers, the rule's cap.
    let (first, _in_flight) = sent("0").expect("a placement");
    assert_eq!(first, (0, 0), "the session's first pair");
    lists.prefills[0].increment_routing_key_load("talk-1");
    lists.decodes[0].increment_routing_key_load("talk-1");
    // At the cap the session has no other worker among the named places.
    assert_eq!(pair_of("0"), Ok((0, 0)), "at the rule's cap");
    // Pinned on the first pair, it goes to the pair it names all the same.
    assert_eq!(pair_of("1"), Ok((1, 1)), "past the session's pins");
    assert_eq!(pair_of("3"), Err(MISSED), "three workers a side");
}

/// A pair that names its decode is decided once. Under a document that
/// picks a pair's decode behind its prefill, such a pair's decision books
/// the decode with its prefill, whatever its caller states: a decode asked
/// again is refused as one that never stood chosen, with no decision made,
/// and the placement stands on the decode the request named. A pair that
/// names no worker is decided again as before, and so is one held to its
/// prefill alone: it names no decode. A pair whose decode alone is named,
/// behind a prefill of another profile, books it so as well. The placement
/// says which it is ahead of each call ([`Placement::decode_chosen`]).
///
/// The classic router picks a pair's decode once, with its prefill
/// (`P:2884-2920`), and sends that decode behind the prefill's answer.
#[test]
fn a_pair_held_to_the_decode_it_names_books_it_with_its_prefill() {
    let later = |prefill: &str, decode: &str| {
        let mut document = PolicySpec::pair_presets(prefill, decode).expect("presets");
        document.generation = 1;
        document.plans.pd_pair.decode = DecodePick::AfterPrefill;
        hosted(engine_of(&document))
    };
    let engine = later("consistent_hashing", "consistent_hashing");
    let lists = Lists::of(3, 3);
    let classes = lists.every();
    let again_under = |engine: &'static InProcess, carried: &[(&str, &str)]| {
        let headers = Counted::of(carried);
        let request = PlaceRequest {
            headers: &headers,
            decode_again: true,
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(lists.under(&classes, &Open)))
        };
        let mut placement = engine.place(&request).expect("a placement");
        let first = lists.places(&placement);
        placement.sent(Side::Prefill);
        let chosen = placement.decode_chosen();
        let answer = placement.decode_again(&request);
        assert_eq!(chosen, answer.is_ok(), "the placement's word ahead of it");
        (first, answer, lists.places(&placement))
    };
    let again = |carried: &[(&str, &str)]| again_under(engine, carried);
    const BOOKED: Unplaced = Unplaced::Refused(DecisionRefused::Decode(DecodeRefused::NotChosen));
    let key = (MANUAL_SESSION_HEADER, "talk-1");
    assert_eq!(again(&[key, (TARGET, "2")]), ((2, 2), Err(BOOKED), (2, 2)));
    let (first, answer, then) = again(&[key]);
    assert_eq!(
        (answer, then),
        (Ok(()), first),
        "the routing value's owners"
    );
    let ring_prefill = later("consistent_hashing", "round_robin");
    let (first, answer, then) = again_under(ring_prefill, &[key, (TARGET, "2")]);
    assert_eq!((first.0, answer, then.0), (2, Ok(()), 2), "no decode named");
    let ring_decode = later("round_robin", "consistent_hashing");
    let (first, answer, then) = again_under(ring_decode, &[key, (TARGET, "2")]);
    assert_eq!(
        (first.1, answer, then.1),
        (2, Err(BOOKED), 2),
        "the decode named behind a prefill of another profile"
    );
}
