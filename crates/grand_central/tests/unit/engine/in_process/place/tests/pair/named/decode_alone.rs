//! The cases of a pair whose decode side alone reads the name its request
//! carries: the prefill is its own profile's pick, and each decision of
//! the call counts the decode's number behind the prefill it picks.
//!
//! `P` and `CH` are the classic router's files, as in the parent's cases,
//! on main `4b41f39a7`.

use smg_worker::runtime::StallReason;

use super::*;

/// The stock pair of a round robin prefill beside a hash ring decode: the
/// decode's side alone reads the target source.
fn ring_decode() -> &'static InProcess {
    stock("round_robin", "consistent_hashing")
}

/// The pairs that `rounds` requests naming `number` are placed on over
/// `pair`, and how many of the requests were shed under the target's name.
/// Any other answer fails the case.
fn asked(
    engine: &'static InProcess,
    lists: &Lists,
    pair: PairLists<'_>,
    number: &str,
    rounds: usize,
) -> (BTreeSet<(usize, usize)>, usize) {
    let mut pairs = BTreeSet::new();
    let mut missed = 0;
    for _ in 0..rounds {
        match named(engine, lists, pair, number) {
            Ok(placed) => {
                pairs.insert(placed);
            }
            Err(unplaced) => {
                assert_eq!(unplaced, missed_on(PairSide::Decode), "names {number}");
                missed += 1;
            }
        }
    }
    (pairs, missed)
}

/// The decodes of [`asked`]'s pairs.
fn decodes_of(pairs: &BTreeSet<(usize, usize)>) -> BTreeSet<usize> {
    pairs.iter().map(|&(_, decode)| decode).collect()
}

/// The answer to one request over `pair` whose route states a rule that
/// takes every worker, and the workers that rule was asked of, each by
/// its address, in the order of its asks. The request names `number`, and
/// nobody for `None`.
fn rule_asks(
    engine: &'static InProcess,
    lists: &Lists,
    pair: PairLists<'_>,
    number: Option<&str>,
) -> (Result<(usize, usize), Unplaced>, Vec<String>) {
    let asked = RefCell::new(Vec::new());
    let rule = |worker: &dyn Worker| {
        asked.borrow_mut().push(worker.url().to_owned());
        true
    };
    let stated = number.map(|number| [(TARGET, number)]);
    let headers: &dyn HeaderRead = match &stated {
        Some(stated) => stated,
        None => &(),
    };
    let request = PlaceRequest {
        headers,
        admits: Some(&rule),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    let answer = placed(engine, lists, &request);
    (answer, asked.take())
}

/// The decode a pair names is counted behind the prefill its own profile
/// picks. The prefill takes its turns, and the decode is the one the
/// number reaches among that prefill's partners, in the decode list's
/// order, whatever the request's routing value. Two classes pair apart
/// here: prefill 0 with decodes 0 and 1, prefill 1 with decodes 1 and 2.
/// The header is read once a call.
///
/// The classic router hands its decode policy the open partners of the
/// prefill its prefill policy picked, and that policy reads the header
/// over them (`P:2884-2887`, `P:2914-2920`; `CH:150-162`; its case at
/// `P:3477`).
#[test]
fn a_decode_named_alone_is_counted_behind_the_prefill_its_profile_picks() {
    let engine = ring_decode();
    assert!(engine.spec().names_target());
    let lists = Lists::of(2, 3);
    let classes = PartnerClasses::number(3, [&[0_u32, 1][..], &[1, 2][..]]);
    let pair = lists.under(&classes, &Open);
    let both = |first, second| (BTreeSet::from([first, second]), 0);
    assert_eq!(asked(engine, &lists, pair, "0", 4), both((0, 0), (1, 1)));
    assert_eq!(asked(engine, &lists, pair, "1", 4), both((0, 1), (1, 2)));
    let headers = Counted::of(&[(MANUAL_SESSION_HEADER, "talk-2"), (TARGET, "+1")]);
    let request = PlaceRequest {
        headers: &headers,
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    let (prefill, decode) = placed(engine, &lists, &request).expect("a placement");
    assert_eq!(decode, prefill + 1, "the second partner of the prefill");
    assert_eq!(headers.reads.get(), 1, "the header is looked for once");
}

/// The number counts the partners of the one prefill picked. Where it
/// reaches none of them the pair is shed under `target-worker`, though
/// another prefill has a partner at that number: no other prefill's
/// partners are asked. A text that is no number reaches no decode behind
/// any prefill.
///
/// The classic policy misses a number outside the workers it is handed,
/// and a text it cannot read as one, and the pair is refused on its decode
/// side (`CH:155-161`; `P:2919`).
#[test]
fn a_number_past_the_picks_partners_sheds_the_pair_behind_that_pick() {
    let engine = ring_decode();
    let lists = Lists::of(2, 3);
    // Prefill 0 pairs with decode 0 alone, prefill 1 with decodes 1 and 2.
    let classes = PartnerClasses::number(3, [&[0_u32][..], &[1, 2][..]]);
    let pair = lists.under(&classes, &Open);
    let (pairs, missed) = asked(engine, &lists, pair, "1", 6);
    assert_eq!(pairs, BTreeSet::from([(1, 2)]), "behind prefill 1 alone");
    assert!(missed > 0, "prefill 0 has no second partner");
    assert_eq!(asked(engine, &lists, pair, "2", 4), (BTreeSet::new(), 4));
    assert_eq!(asked(engine, &lists, pair, "one", 4), (BTreeSet::new(), 4));
}

/// The decode's number counts the decodes that can be sent to when the
/// call reads them, in the decode list's order, at the places the caller
/// states for the decodes: a decode that is not ready and one over its
/// thresholds are passed over, and so is one the route's rule refuses.
/// The rule is asked there of each decode that is available, of the
/// prefills that are as far as the first that opens the pair, then of
/// each prefill ready or not, and behind the decision of its pick's two
/// workers alone.
///
/// The classic router's decode policy is handed the partners the route's
/// rule takes that are available (`P:2635-2640`, `P:2676-2680`,
/// `P:2884-2887`).
#[test]
fn the_decodes_number_counts_the_decodes_that_can_be_sent_to() {
    let engine = ring_decode();
    let decodes = |lists: &Lists, pair: PairLists<'_>, number: &str| {
        let (pairs, missed) = asked(engine, lists, pair, number, 2);
        (decodes_of(&pairs), missed)
    };
    let one = |decode: usize| (BTreeSet::from([decode]), 0);
    let none = (BTreeSet::new(), 2);

    let lists = Lists::of(2, 3);
    let classes = lists.every();
    lists.decodes[0].set_status(WorkerStatus::NotReady);
    let pair = lists.under(&classes, &Open);
    assert_eq!(decodes(&lists, pair, "0"), one(1), "past the one not ready");
    assert_eq!(decodes(&lists, pair, "1"), one(2));
    assert_eq!(decodes(&lists, pair, "2"), none, "two decodes are counted");

    let lists = Lists::of(2, 3);
    let classes = lists.every();
    lists.decodes[1].set_overloaded(true);
    let pair = lists.under(&classes, &Open);
    assert_eq!(decodes(&lists, pair, "1"), one(2), "past the one over");

    let lists = Lists::of(2, 3);
    let classes = lists.every();
    let held = PairLists {
        decode_places: Some(&[0, 2][..]),
        ..lists.under(&classes, &Open)
    };
    assert_eq!(decodes(&lists, held, "1"), one(2), "among the places");
    assert_eq!(decodes(&lists, held, "2"), none);

    let refused = lists.decodes[0].url();
    let rule_asked = Cell::new(0_u32);
    let rule = |worker: &dyn Worker| {
        rule_asked.set(rule_asked.get() + 1);
        worker.url() != refused
    };
    let headers = [(TARGET, "0")];
    let request = PlaceRequest {
        headers: &headers,
        admits: Some(&rule),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(lists.under(&classes, &Open)))
    };
    let (_, decode) = placed(engine, &lists, &request).expect("a placement");
    assert_eq!(decode, 1, "past the decode the rule refuses");
    let asks = "three decodes, the prefill that opens the pair, two held, the pick's two";
    assert_eq!(rule_asked.get(), 8, "{asks}");
}

/// A rule the request states is asked of the prefills ahead of the call's
/// first ask, and that ask is held to the prefills it takes: the rule
/// takes the last of four prefills here, and every request is placed on
/// it at once, with the decode it names.
///
/// The classic router hands its prefill policy the prefills the route's
/// rule takes (`P:2635-2640`).
#[test]
fn a_decode_named_alone_is_placed_behind_a_prefill_the_rule_takes() {
    let engine = ring_decode();
    let lists = Lists::of(4, 2);
    let classes = lists.every();
    let taken = lists.prefills[3].url();
    let decodes: Vec<_> = lists.decodes.iter().map(|decode| decode.url()).collect();
    let rule = |worker: &dyn Worker| worker.url() == taken || decodes.contains(&worker.url());
    let headers = [(TARGET, "1")];
    for _ in 0..4 {
        let request = PlaceRequest {
            headers: &headers,
            admits: Some(&rule),
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(lists.under(&classes, &Open)))
        };
        assert_eq!(placed(engine, &lists, &request), Ok((3, 1)));
    }
}

/// Where the pair is made among the prefills with a first partner that
/// can be sent to, the decode's number counts the pick's first partners
/// alone. Where no first partner can be sent to, it counts every partner
/// of the prefill picked.
///
/// The classic router's pick is a prefill with an open partner on its own
/// known runtime while there is one, and its decode policy is handed the
/// pick's partners on that runtime (`P:2754-2771`, `P:2884-2887`).
#[test]
fn a_decode_named_alone_is_counted_among_the_picks_first_partners() {
    let engine = ring_decode();
    let lists = Lists::of(2, 3);
    // Both prefills pair with every decode, and prefill 1 with decodes 1
    // and 2 first.
    let all: &[u32] = &[0, 1, 2];
    let classes = PartnerClasses::number_first(3, [(all, &[][..]), (all, &[1, 2][..])]);
    let pair = lists.under(&classes, &Open);
    let only = |pair| (BTreeSet::from([pair]), 0);
    assert_eq!(asked(engine, &lists, pair, "0", 3), only((1, 1)));
    assert_eq!(asked(engine, &lists, pair, "1", 3), only((1, 2)));
    assert_eq!(asked(engine, &lists, pair, "2", 3), (BTreeSet::new(), 3));
    lists.decodes[1].set_status(WorkerStatus::NotReady);
    lists.decodes[2].set_status(WorkerStatus::NotReady);
    let either = (BTreeSet::from([(0, 0), (1, 0)]), 0);
    assert_eq!(asked(engine, &lists, pair, "0", 4), either);
}

/// Where no decode can be sent to, the decode's number is not read: the
/// pair is placed as one that names no worker. Over decodes that are all
/// over their thresholds the binding's rule takes the one that holds the
/// fewest requests, whatever the header's text.
///
/// The classic router takes the least-loaded of the pick's partners over
/// a decode pool all over its thresholds, with no read of the header
/// (`P:2653-2655`, `P:2909-2912`).
#[test]
fn where_no_decode_can_be_sent_to_the_decodes_number_is_not_read() {
    let engine = ring_decode();
    let lists = Lists::of(2, 3);
    let classes = lists.every();
    for decode in lists.decodes.iter() {
        decode.set_overloaded(true);
    }
    lists.decodes[0].increment_load();
    lists.decodes[2].increment_load();
    let pair = lists.under(&classes, &Open);
    for number in ["0", "2", "7", "one"] {
        let (pairs, missed) = asked(engine, &lists, pair, number, 2);
        let least = BTreeSet::from([1_usize]);
        assert_eq!((decodes_of(&pairs), missed), (least, 0), "names {number}");
    }
}

/// A named decode with no free room is taken, and a retry is handed the
/// decode it names again, though its failed attempt was sent there. The
/// prefill's side is not held: it steps aside from the prefill the
/// request tried.
///
/// The classic router runs no room filter and no retry's narrowing over
/// the decodes of a pair whose decode policy reads the header (`P:2895`,
/// `P:2902`, `P:2978-2987`), and narrows the prefills of another policy
/// (`P:2826`).
#[test]
fn a_decode_named_alone_is_taken_full_and_handed_to_a_retry_again() {
    let engine = ring_decode();
    let lists = Lists::of(2, 2);
    let classes = lists.every();
    let full = Gates(|_| true, |decode| decode != 1);
    let (pairs, missed) = asked(engine, &lists, lists.under(&classes, &full), "1", 2);
    assert_eq!((decodes_of(&pairs), missed), (BTreeSet::from([1]), 0));

    let headers = [(TARGET, "1")];
    let tried = [lists.prefill_id(0), lists.decode_id(1)];
    for _ in 0..2 {
        let request = PlaceRequest {
            headers: &headers,
            attempt: 1,
            tried: &tried,
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(lists.under(&classes, &Open)))
        };
        let pair = placed(engine, &lists, &request);
        assert_eq!(pair, Ok((1, 1)), "the other prefill, the decode named");
    }
}

/// A named decode that cannot take the request after all is the pair's
/// one decode still: the request moves to no other. One the route's rule
/// takes when it is counted and refuses as a pick is named again by the
/// call's second ask, which counts as its first did, and the pair is shed
/// under `target-worker` behind it. One that stops being ready behind the
/// count sheds the pair under `target-worker`, whatever stage refused it.
/// One that goes over its thresholds behind the count
/// is answered by the binding's rule for decodes that are all over them,
/// which sees that one decode: it is taken where the rule steers, and the
/// pair is shed under `target-worker` where the rule sheds. The other
/// decode, which stands, is never taken.
///
/// The classic policy reads the named worker once more behind the count
/// and misses it there, and the pair is refused (`CH:156-160`; `P:2919`).
#[test]
fn a_decode_named_alone_that_cannot_take_the_request_after_all_is_never_replaced() {
    // The route's rule is asked of the named decode when it is counted:
    // what `then` does to the worker there happens behind the count.
    let racing = |engine: &'static InProcess, then: fn(&dyn Worker)| {
        let lists = Lists::of(2, 2);
        let classes = lists.every();
        let named = lists.decodes[0].url();
        let rule = |worker: &dyn Worker| {
            if worker.url() == named {
                then(worker);
            }
            true
        };
        let headers = [(TARGET, "0")];
        let request = PlaceRequest {
            headers: &headers,
            admits: Some(&rule),
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(lists.under(&classes, &Open)))
        };
        placed(engine, &lists, &request).map(|(_, decode)| decode)
    };
    let leaves = |worker: &dyn Worker| worker.set_status(WorkerStatus::NotReady);
    let overloads = |worker: &dyn Worker| {
        worker.set_overloaded(true);
    };
    let steering = ring_decode();
    assert_eq!(racing(steering, leaves), Err(missed_on(PairSide::Decode)));
    assert_eq!(
        racing(steering, overloads),
        Ok(0),
        "taken over its thresholds"
    );

    let mut document =
        PolicySpec::pair_presets("round_robin", "consistent_hashing").expect("presets");
    document.generation = 1;
    document.bindings[0].overload = Some(OverloadSpec {
        shed_when_all_vetoed: true,
        ..OverloadSpec::default()
    });
    let shedding = hosted(engine_of(&document));
    assert_eq!(
        racing(shedding, overloads),
        Err(missed_on(PairSide::Decode))
    );

    // The rule takes every worker while it is counted: two decodes, and
    // the first prefill, which opens the pair. Behind that it refuses the
    // named decode.
    let lists = Lists::of(2, 2);
    let classes = lists.every();
    let named = lists.decodes[0].url();
    let rule_asked = Cell::new(0_u32);
    let rule = |worker: &dyn Worker| {
        rule_asked.set(rule_asked.get() + 1);
        rule_asked.get() <= 3 || worker.url() != named
    };
    let headers = [(TARGET, "0")];
    let request = PlaceRequest {
        headers: &headers,
        admits: Some(&rule),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(lists.under(&classes, &Open)))
    };
    assert_eq!(placed(steering, &lists, &request), Err(MISSED));
    let asks = "the count's three, the two prefills held, two decisions' picks";
    assert_eq!(rule_asked.get(), 9, "{asks}");
}

/// The decodes' count passes over a decode at a place its host masks and
/// over one that fails what a decode is held to of the request's hard
/// demands, as the count of a pair named on both sides does: neither takes
/// a number, and a number past the decodes left sheds the pair under
/// `target-worker`. Where none of the decodes left can be sent to, the
/// header is not read. The prefills are the decision's to judge, as for
/// any pair.
///
/// The classic router narrows both pools of a pinned pair to its wire
/// ahead of its policies (`P:2635-2640`), and every worker it counts
/// passes the worker's own test (`P:2676-2679`).
#[test]
fn a_decode_named_alone_takes_no_number_behind_a_mask_or_a_demand() {
    let engine = ring_decode();
    let lists = Lists::of(2, 3);
    let classes = lists.every();
    let masked = PairLists {
        decode_stalled: &[0],
        ..lists.under(&classes, &Open)
    };
    let (pairs, missed) = asked(engine, &lists, masked, "0", 4);
    assert_eq!((decodes_of(&pairs), missed), (BTreeSet::from([1]), 0));
    let (pairs, missed) = asked(engine, &lists, masked, "1", 4);
    assert_eq!((decodes_of(&pairs), missed), (BTreeSet::from([2]), 0));
    let (pairs, missed) = asked(engine, &lists, masked, "2", 4);
    assert_eq!((pairs.len(), missed), (0, 4), "two decodes are counted");

    // The first decode states the runtime 1, every other worker the
    // runtime 2: a pair pinned to the second wire counts two decodes.
    let mut lists = Lists::of(2, 3);
    let first = Arc::clone(&lists.decodes[0]);
    let stating = |list: &Arc<[Arc<dyn Worker>]>| {
        PublishedList::with_facts(Arc::clone(list), |worker| {
            let mut facts = FixedFacts::default();
            facts.demands.runtime = if Arc::ptr_eq(worker, &first) { 1 } else { 2 };
            facts
        })
    };
    lists.published = [stating(&lists.prefills), stating(&lists.decodes)];
    let classes = lists.every();
    let wire = Demands {
        runtime: 2,
        ..Demands::default()
    };
    let decode_of = |number: &str| {
        let headers = [(TARGET, number)];
        let request = PlaceRequest {
            headers: &headers,
            demands: wire,
            ..PlaceRequest::of(MODEL, WorkerLists::Pair(lists.under(&classes, &Open)))
        };
        let placement = engine.place(&request);
        placement.map(|placement| lists.places(&placement).1)
    };
    assert_eq!(decode_of("0"), Ok(1), "the first decode on the wire");
    assert_eq!(decode_of("1"), Ok(2));
    assert_eq!(
        decode_of("2"),
        Err(missed_on(PairSide::Decode)),
        "two decodes meet the demand"
    );

    // Where neither decode on the wire can be sent to, no decode is
    // counted, though the first decode is ready: the header is not read,
    // and the pair is refused as one that names no worker is.
    lists.decodes[1].set_status(WorkerStatus::NotReady);
    lists.decodes[2].set_status(WorkerStatus::NotReady);
    let unnamed = PlaceRequest {
        demands: wire,
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(lists.under(&classes, &Open)))
    };
    let refused = placed(engine, &lists, &unnamed).expect_err("no decode on the wire is ready");
    assert_ne!(refused, MISSED, "no header, no target's name");
    let headers = [(TARGET, "0")];
    let request = PlaceRequest {
        headers: &headers,
        ..unnamed
    };
    assert_eq!(placed(engine, &lists, &request), Err(refused));
}

/// Where no prefill opens the pair, the decode's number is not read: the
/// pair is placed as one that names no worker. A prefill opens it where
/// it can be sent to and a decode the call counts partners it. Here the
/// first prefill and its one partner are over their thresholds, and the
/// second prefill is not ready beside the one decode that can be sent to:
/// the binding's default pairs the first prefill with its partner by
/// load, whatever the header's text, as for a request that names nobody.
/// Lent the one counted decode's place alone, that default would find no
/// pair.
///
/// The classic router reads no header where no prefill is open, and pairs
/// such a pool by load (`P:2729-2747`, `P:2772-2798`, its `relaxed_pair`
/// at `P:1959`). The sheet's line
/// `named-decode-no-open-prefill-pairs-by-load` records this fleet.
#[test]
fn where_no_prefill_opens_the_pair_the_decodes_number_is_not_read() {
    let engine = ring_decode();
    let lists = Lists::of(2, 2);
    // Prefill 0 pairs with decode 0 alone, prefill 1 with decode 1 alone.
    let classes = PartnerClasses::number(2, [&[0_u32][..], &[1][..]]);
    lists.prefills[0].set_overloaded(true);
    lists.decodes[0].set_overloaded(true);
    lists.prefills[1].set_status(WorkerStatus::NotReady);
    let pair = lists.under(&classes, &Open);
    let unnamed = PlaceRequest::of(MODEL, WorkerLists::Pair(pair));
    assert_eq!(placed(engine, &lists, &unnamed), Ok((0, 0)), "by load");
    for number in ["0", "1", "one"] {
        let by_load = (BTreeSet::from([(0, 0)]), 0);
        assert_eq!(asked(engine, &lists, pair, number, 2), by_load, "{number}");
    }
}

/// A prefill opens the pair by a partner the call counts, and by no other
/// decode. Here the first prefill can be sent to and its one partner is
/// over its thresholds. The second prefill is not ready, and of its two
/// partners the first is under a liveness doubt and the second can be
/// sent to. No prefill opens the pair: the header's number is not acted
/// on, and the
/// request is answered as one that names nobody is, whatever the header's
/// text. That answer is the pair's shed with no endpoint under `healthy`:
/// a decode pool that is not over its thresholds whole is steered nowhere
/// and shed over by nobody, and the pool's first decode with room is the
/// one under the doubt, which the health stage refuses. Lent the one
/// counted decode's place, a decision would find the first prefill no
/// partner there, and shed the pair under `partnered`: the stage tells a
/// number that was acted on from one that was not.
///
/// The classic router counts a prefill as open by an open partner of its
/// own (`P:2723-2728`, `P:2756-2771`). With none open it never asks its
/// decode policy, the header's reader: it pairs by load a pool whose
/// every worker is vetoed, and refuses the pair where one is not, as here
/// (`P:2772-2798`, `P:1969`).
#[test]
fn a_prefill_with_no_counted_partner_opens_no_pair() {
    let engine = ring_decode();
    let lists = Lists::of(2, 3);
    // Prefill 0 pairs with decode 0 alone, prefill 1 with the two others.
    let classes = PartnerClasses::number(3, [&[0_u32][..], &[1, 2][..]]);
    lists.decodes[0].set_overloaded(true);
    lists.decodes[1].set_stall(Some(StallReason::Wedged));
    lists.prefills[1].set_status(WorkerStatus::NotReady);
    let pair = lists.under(&classes, &Open);
    let unnamed = PlaceRequest::of(MODEL, WorkerLists::Pair(pair));
    let nobody = placed(engine, &lists, &unnamed);
    // The decode side: the prefill that can be sent to has no partner
    // left. The classic router refuses for the decode's role where an
    // available prefill has no open partner (`P:2943-2952`, on main
    // `d43c7b2a9`).
    let doubted = Err(Unplaced::Shed {
        reason: ShedReason::NoEndpoints,
        emptied_by: "healthy",
        retry_after_secs: 1,
        side: Some(PairSide::Decode),
    });
    assert_eq!(nobody, doubted, "a request that names nobody");
    for number in ["0", "1", "one"] {
        assert_eq!(named(engine, &lists, pair, number), nobody, "{number}");
    }
}

/// A prefill over its thresholds alone opens the pair where its profile
/// places among such prefills: one that weighs load against affinity by a
/// gate of its own, under a binding that steers. The number is read then,
/// and counts the partners of the prefill that profile picks.
///
/// The classic router lifts the overload veto off a prefill pool whose
/// policy weighs load itself: its prefills over their thresholds are
/// open, and the decode's policy reads the header behind the pick
/// (`P:2672-2680`). No line of the sheet holds it: the sheet's replay
/// states its lines' two presets, and neither weighs.
#[test]
fn a_prefill_over_its_thresholds_opens_the_pair_under_a_profile_that_weighs() {
    let engine = stock("cache_aware", "consistent_hashing");
    let lists = Lists::of(2, 3);
    let classes = lists.every();
    for prefill in lists.prefills.iter() {
        prefill.set_overloaded(true);
    }
    let pair = lists.under(&classes, &Open);
    for (number, decode) in [("0", 0), ("2", 2)] {
        let (pairs, missed) = asked(engine, &lists, pair, number, 2);
        let named = (BTreeSet::from([decode]), 0);
        assert_eq!((decodes_of(&pairs), missed), named, "names {number}");
    }
}

/// Under a profile that does not weigh load itself, a prefill over its
/// thresholds opens no pair, though a decode the call counts partners it.
/// Here every prefill is over them. The second one's partner can be sent
/// to, and the first one's is over its thresholds too. The header is not
/// read: the binding's default pairs the first prefill, the less loaded,
/// with its partner by load, as for a request that names nobody. Lent the
/// one counted decode's place, that default would pair the second.
///
/// The classic router lifts the overload veto off a prefill pool for a
/// policy that weighs load itself and for no other (`P:2672-2674`). Under
/// any other no prefill over its thresholds is open: the header is not
/// read, and the pool is paired by load (`P:2729-2747`).
#[test]
fn a_prefill_over_its_thresholds_opens_no_pair_under_a_profile_that_does_not_weigh() {
    let engine = ring_decode();
    let lists = Lists::of(2, 2);
    // Prefill 0 pairs with decode 0 alone, prefill 1 with decode 1 alone.
    let classes = PartnerClasses::number(2, [&[0_u32][..], &[1][..]]);
    for prefill in lists.prefills.iter() {
        prefill.set_overloaded(true);
    }
    lists.decodes[0].set_overloaded(true);
    for _ in 0..8 {
        lists.prefills[1].increment_load();
    }
    let pair = lists.under(&classes, &Open);
    let unnamed = PlaceRequest::of(MODEL, WorkerLists::Pair(pair));
    assert_eq!(placed(engine, &lists, &unnamed), Ok((0, 0)), "by load");
    for number in ["0", "1", "one"] {
        let by_load = (BTreeSet::from([(0, 0)]), 0);
        assert_eq!(asked(engine, &lists, pair, number, 2), by_load, "{number}");
    }
}

/// Under a binding that sheds a pool its overload veto empties, a prefill
/// over its thresholds opens no pair, whatever its profile. Here the
/// profile weighs load itself. The first prefill can be sent to beside a
/// partner that is over its thresholds. The second prefill is over its
/// thresholds, and of its two partners the first is under a liveness
/// doubt and the second can be sent to. No prefill opens the pair: the
/// header's number is not acted on, and the request is answered as one
/// that names
/// nobody is, whatever the header's text: by the pair's shed with no
/// endpoint under `healthy`, since the decode pool is not over its
/// thresholds whole, and its first decode with room is the one under the
/// doubt. Read by the second prefill's leave, the call would lend the one
/// counted decode's place: the decision would show its prefill profile
/// the second prefill alone, refuse it over its thresholds, and find the
/// first prefill no partner there, under `partnered`. The stage tells a
/// number that was acted on from one that was not.
///
/// The classic router lifts no veto with its shedding on (`P:2672`,
/// `P:2653-2655`): no prefill is open, and it refuses the pair, whose
/// decode pool is not vetoed whole (`P:2772-2798`, `P:1969`).
#[test]
fn under_a_binding_that_sheds_a_prefill_over_its_thresholds_opens_no_pair() {
    let mut document =
        PolicySpec::pair_presets("cache_aware", "consistent_hashing").expect("presets");
    document.generation = 1;
    document.bindings[0].overload = Some(OverloadSpec {
        shed_when_all_vetoed: true,
        ..OverloadSpec::default()
    });
    let shedding = hosted(engine_of(&document));
    let lists = Lists::of(2, 3);
    // Prefill 0 pairs with decode 0 alone, prefill 1 with the two others.
    let classes = PartnerClasses::number(3, [&[0_u32][..], &[1, 2][..]]);
    lists.decodes[0].set_overloaded(true);
    lists.decodes[1].set_stall(Some(StallReason::Wedged));
    lists.prefills[1].set_overloaded(true);
    let pair = lists.under(&classes, &Open);
    let unnamed = PlaceRequest::of(MODEL, WorkerLists::Pair(pair));
    let nobody = placed(shedding, &lists, &unnamed);
    // The decode side, as above: with its shedding on the classic router
    // lifts no veto, so the first prefill is its one available prefill and
    // has no open partner (`P:2943-2952`, on main `d43c7b2a9`).
    let doubted = Err(Unplaced::Shed {
        reason: ShedReason::NoEndpoints,
        emptied_by: "healthy",
        retry_after_secs: 1,
        side: Some(PairSide::Decode),
    });
    assert_eq!(nobody, doubted, "a request that names nobody");
    for number in ["0", "1", "one"] {
        assert_eq!(named(shedding, &lists, pair, number), nobody, "{number}");
    }
}

/// The prefills that open the pair are read as the count of a pair named
/// on both sides reads them. One at a place its host masks opens none,
/// and neither does one that fails what a prefill is held to of the
/// request's hard demands, though each is ready beside a decode that can
/// be sent to. The other prefill and its one partner are over their
/// thresholds: the default pairs those two by load, whatever the header's
/// text, as for a request that names nobody.
///
/// The classic router narrows both pools of a pinned pair to its wire
/// ahead of its policies (`P:2635-2640`), and a prefill is open only
/// where it passes the worker's own test (`P:2676-2680`).
#[test]
fn a_prefill_behind_a_mask_or_a_demand_opens_no_pair() {
    let engine = ring_decode();
    // Prefill 0 pairs with decode 0 alone, prefill 1 with decode 1 alone.
    let classes = PartnerClasses::number(2, [&[0_u32][..], &[1][..]]);
    let over = |lists: &Lists| {
        lists.prefills[1].set_overloaded(true);
        lists.decodes[1].set_overloaded(true);
    };
    let by_load = (BTreeSet::from([(1, 1)]), 0);

    let lists = Lists::of(2, 2);
    over(&lists);
    let masked = PairLists {
        prefill_stalled: &[0],
        ..lists.under(&classes, &Open)
    };
    let unnamed = PlaceRequest::of(MODEL, WorkerLists::Pair(masked));
    assert_eq!(placed(engine, &lists, &unnamed), Ok((1, 1)), "by load");
    assert_eq!(asked(engine, &lists, masked, "0", 2), by_load, "a mask");

    // The first prefill states the runtime 1, every other worker the
    // runtime 2, and the pair is pinned to the second wire.
    let mut lists = Lists::of(2, 2);
    over(&lists);
    let first = Arc::clone(&lists.prefills[0]);
    let stating = |list: &Arc<[Arc<dyn Worker>]>| {
        PublishedList::with_facts(Arc::clone(list), |worker| {
            let mut facts = FixedFacts::default();
            facts.demands.runtime = if Arc::ptr_eq(worker, &first) { 1 } else { 2 };
            facts
        })
    };
    lists.published = [stating(&lists.prefills), stating(&lists.decodes)];
    let wire = Demands {
        runtime: 2,
        ..Demands::default()
    };
    let pinned = || PlaceRequest {
        demands: wire,
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(lists.under(&classes, &Open)))
    };
    assert_eq!(placed(engine, &lists, &pinned()), Ok((1, 1)), "by load");
    let headers = [(TARGET, "0")];
    let request = PlaceRequest {
        headers: &headers,
        ..pinned()
    };
    assert_eq!(placed(engine, &lists, &request), Ok((1, 1)), "a demand");
}

/// The prefills are read in their list's order as far as the first that
/// opens the pair, and the route's rule is asked of no prefill behind it:
/// of the first of four here, ahead of its own read of every prefill,
/// ready or not. Where no prefill opens the pair, every prefill that can
/// be sent to is read and asked. Here three prefills pair with a decode
/// over its thresholds, and the fourth, the one partner of the decode
/// that can be sent to, is not ready: the header is not read.
///
/// The classic router asks its route's rule of every candidate of both
/// pools ahead of its policies (`P:2635-2640`), once a pass
/// (`P:196-200`). The answers are the same: a rule there answers by
/// reading alone.
#[test]
fn the_prefills_are_read_as_far_as_the_first_that_opens_the_pair() {
    let engine = ring_decode();
    let lists = Lists::of(4, 2);
    let urls = |workers: &[Arc<dyn Worker>]| -> Vec<String> {
        let url = |worker: &Arc<dyn Worker>| worker.url().to_owned();
        workers.iter().map(url).collect()
    };
    let (prefills, decodes) = (urls(&lists.prefills), urls(&lists.decodes));

    let every = lists.every();
    let (pair, asks) = rule_asks(engine, &lists, lists.under(&every, &Open), Some("0"));
    assert_eq!(pair.map(|(_, decode)| decode), Ok(0), "the decode named");
    // Both decodes, the first prefill, the rule's own read of the four
    // prefills, and behind them the pick's two workers.
    let counted = [&decodes[..], &prefills[..1], &prefills[..]].concat();
    assert_eq!(asks[..counted.len()], counted, "one prefill opens the pair");
    assert_eq!(asks.len(), counted.len() + 2, "the pick's two");

    // The first three prefills pair with decode 0 alone, the fourth with
    // decode 1 alone.
    let (zero, one): (&[u32], &[u32]) = (&[0], &[1]);
    let apart = PartnerClasses::number(2, [zero, zero, zero, one]);
    lists.decodes[0].set_overloaded(true);
    lists.prefills[3].set_status(WorkerStatus::NotReady);
    let pair = lists.under(&apart, &Open);
    let (unnamed, _) = rule_asks(engine, &lists, pair, None);
    let (named, asks) = rule_asks(engine, &lists, pair, Some("0"));
    assert_eq!(named, unnamed, "the header is not read");
    // The one decode that can be sent to, then each prefill that can.
    let read = [&decodes[1..], &prefills[..3]].concat();
    assert_eq!(asks[..read.len()], read, "no prefill opens the pair");
}
