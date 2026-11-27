//! The cases of an encode request that carries items held to the workers
//! its route's rule takes: behind the first decision that places nothing,
//! the rule is asked of the pair's two lists, and the call's later
//! decisions are made among the workers it takes on each, as a pair's are
//! and by the pair's own routine
//! ([`pair::again`](super::super::super::pair::again)). The rule is asked
//! of no encoder.
//!
//! The classic router takes an encode request's pair by the pair's own
//! selection, under the filter it makes for such a request
//! (`model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2599-2619`
//! on main `bd55beae5`). That selection's pass counts a worker of either
//! list only where the filter takes it, and judges what is available and
//! what is over its thresholds over those workers alone
//! (`model_gateway/src/routers/common/placement.rs:2819-2839`, `:2860-2864`
//! there). It asks the filter of no encoder: an item goes to an available
//! encoder that pairs with the chosen prefill, by the encode policy
//! (`worker_selection.rs:2518-2533`, `:2620-2655`).

use super::*;

/// One item, under a key its host holds as a fingerprint.
const KEYS: [Identity<'static>; 1] = [Identity::hashed(1)];

/// An encode request of one item over `lists` whose route states `admits`.
fn ruled<'a>(lists: EncodeLists<'a>, admits: WorkerRule<'a>) -> PlaceRequest<'a> {
    let items = EncodeItems {
        keys: &KEYS,
        ..EncodeItems::default()
    };
    PlaceRequest {
        admits: Some(admits),
        ..asking(lists, items)
    }
}

/// What the stages hear of an encode request's decision that is given up:
/// both sides of its pair issued, then both released. Nothing is said of an
/// encoder.
fn given_up(prefill: WorkerId, decode: WorkerId) -> [(String, WorkerId); 4] {
    let released = |worker| ("released CommitFailed".to_owned(), worker);
    [
        issued(prefill),
        issued(decode),
        released(prefill),
        released(decode),
    ]
}

/// The pair of the first decision `heard` tells of, which was given up: a
/// side's pick among workers that hold equally many requests is drawn, so a
/// case names that pair by what its stages heard.
fn first_given_up(heard: &[(String, WorkerId)]) -> (WorkerId, WorkerId) {
    assert!(heard.len() >= 4, "a decision given up: {heard:?}");
    let pair = (heard[0].1, heard[1].1);
    assert_eq!(heard[..4], given_up(pair.0, pair.1), "a decision given up");
    pair
}

/// The one pair of two long lists the rule takes is found by the second
/// decision of an encode request that carries an item, with the item on its
/// encoder. Each worker of that pair holds one request more than the
/// others of its list, so no decision over the whole lists names it while
/// another stands. The first decision's targets are refused. The rule is
/// then asked of the thirteen workers of each of the pair's lists and of
/// no encoder, and the second decision is held to the prefill and the
/// decode it takes. Over the whole lists three decisions name three
/// prefills and twelve decodes, never the thirteenth of either, and the
/// request has no worker.
///
/// The classic router places such a request: its pass pairs among the
/// workers the filter takes (`placement.rs:2907-2955`, `:3038-3039`,
/// `:3069-3071`, `:3101-3102` on main `bd55beae5`).
#[test]
fn the_one_pair_a_rule_takes_in_two_long_lists_is_found_for_a_request_with_items() {
    let (engine, heard) = hearing(fewest(DecodePick::default()));
    let three = Three::of([13, 13, 2], |_, _| false);
    three.workers[0][12].increment_load();
    three.workers[1][12].increment_load();
    let (prefill, decode) = (three.id(0, 12), three.id(1, 12));
    let encoders = [three.id(2, 0), three.id(2, 1)];
    let (calls, of_encoders) = (Cell::new(0), Cell::new(0));
    let admits = |worker: &dyn Worker| {
        calls.set(calls.get() + 1);
        let id = worker_id(worker);
        of_encoders.set(of_encoders.get() + usize::from(encoders.contains(&id)));
        id == prefill || id == decode
    };
    let request = ruled(three.lists((&[PLAIN; 2], &[1; 13])), &admits);
    let placement = engine.place(&request).expect("a placement");
    let (at_prefill, at_decode, at_encoders) = three.places(&placement);
    assert_eq!((at_prefill, at_decode), (12, 12), "the rule's one pair");
    assert_eq!(at_encoders.len(), 1, "the item's encoder");
    assert_eq!(
        calls.get(),
        2 * MAX_TARGETS + 26 + 2,
        "each target's two sides, the two lists, the pick's two sides"
    );
    assert_eq!(of_encoders.get(), 0, "the rule is asked of no encoder");
    let heard = heard.lock().clone();
    let first = first_given_up(&heard);
    assert!(
        first.0 != prefill && first.1 != decode,
        "another pair first"
    );
    assert_eq!(
        heard[4..],
        [issued(prefill), issued(decode)],
        "two decisions"
    );
    drop(placement);
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A prefill the rule takes that is over its overload thresholds is placed
/// beside a prefill the rule refuses that is under them. Over the whole
/// list the overload filter leaves the first out while the second stands,
/// so every decision names the refused prefill. Held to the rule's
/// prefill, the decision is over a list wholly over the thresholds, and
/// the binding's rule for such a list answers: it falls back, and the
/// prefill is taken.
///
/// The classic router steers such a pool the same way: no prefill the
/// filter takes is available, so the pool is judged over the ones it takes
/// and placed by the fallback for a pool wholly vetoed
/// (`placement.rs:2837-2839`, `:2913-2925`, over `admitted_pool`,
/// `:1878-1900`, on main `bd55beae5`).
#[test]
fn an_encode_request_is_placed_on_the_rules_prefill_though_it_is_overloaded() {
    let (engine, heard) = hearing(fewest(DecodePick::default()));
    let three = Three::of([2, 1, 1], |_, _| false);
    three.workers[0][1].set_overloaded(true);
    let (refused, decode) = (three.id(0, 0), three.id(1, 0));
    let admits = |worker: &dyn Worker| worker_id(worker) != refused;
    let request = ruled(three.lists((&[PLAIN], &[1; 2])), &admits);
    let placement = engine.place(&request).expect("the binding's fallback");
    assert_eq!(three.places(&placement), (1, 0, vec![0]));
    let taken = [issued(three.id(0, 1)), issued(decode)];
    assert_eq!(
        *heard.lock(),
        [&given_up(refused, decode)[..], &taken].concat(),
        "two decisions"
    );
    drop(placement);
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A prefill the rule takes that an earlier attempt tried is handed again
/// beside a prefill the rule refuses. Over the whole list the filter of
/// tried workers passes it over while the refused prefill stands. Held to
/// the rule's prefill, the filter would leave nobody, and steps aside. The
/// tried prefill holds one request more than the refused one: behind the
/// first decision the filter passes over both and steps aside, and a
/// decision over the whole list then names the refused prefill by its
/// count, never by a draw.
///
/// The classic router steps a retry aside from the prefills it tried only
/// while another open prefill the filter takes is left
/// (`placement.rs:3010-3012`, over `Narrowed::narrow_to`, `:328-344`, on
/// main `bd55beae5`).
#[test]
fn an_encode_request_is_placed_on_the_rules_prefill_though_an_earlier_attempt_tried_it() {
    let (engine, heard) = hearing(fewest(DecodePick::default()));
    let three = Three::of([2, 1, 1], |_, _| false);
    three.workers[0][1].increment_load();
    let (refused, decode) = (three.id(0, 0), three.id(1, 0));
    let admits = |worker: &dyn Worker| worker_id(worker) != refused;
    let tried = [three.id(0, 1)];
    let retry = PlaceRequest {
        attempt: 1,
        tried: &tried,
        ..ruled(three.lists((&[PLAIN], &[1; 2])), &admits)
    };
    let placement = engine
        .place(&retry)
        .expect("tried, and the rule's one prefill");
    assert_eq!(three.places(&placement), (1, 0, vec![0]));
    let taken = [issued(three.id(0, 1)), issued(decode)];
    assert_eq!(
        *heard.lock(),
        [&given_up(refused, decode)[..], &taken].concat(),
        "two decisions"
    );
    drop(placement);
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// An encode request whose rule takes no worker of one of its pair's lists
/// is shed with no endpoint under the rule's name, behind one decision
/// that picked. The rule is asked of that decision's picks, then of every
/// prefill, then of every decode, ready or not, and no second decision is
/// made: the pass is laid for the request, which is asked no decision more
/// over its rule's prefills. Behind a first decision that sheds the request
/// as saturated, that shed stands in two cases ([`rows`](super::rows)).
///
/// The classic router answers such a request that it has no worker: its
/// pass finds nobody open on a side none of whose workers the filter takes
/// (`placement.rs:2956-2982` on main `bd55beae5`), and no second pass is
/// made while no worker the filter takes came free (`:2411-2425`).
#[test]
fn an_encode_request_whose_rule_takes_no_worker_of_a_list_is_shed_under_its_name() {
    let (engine, heard) = hearing(fewest(DecodePick::default()));
    let three = Three::of([3, 2, 1], |_, _| false);
    three.workers[1][1].set_status(WorkerStatus::NotReady);
    let prefills = [0, 1, 2].map(|at| three.id(0, at));
    let decodes = [0, 1].map(|at| three.id(1, at));
    let asked = RefCell::new(Vec::new());
    let admits = |worker: &dyn Worker| {
        let id = worker_id(worker);
        asked.borrow_mut().push(id);
        prefills.contains(&id)
    };
    let request = ruled(three.lists((&[PLAIN], &[1; 3])), &admits);
    let shed = Unplaced::Shed {
        reason: ShedReason::NoEndpoints,
        emptied_by: "admits",
        retry_after_secs: 1,
        side: None,
    };
    assert_eq!(engine.place(&request).err(), Some(shed));
    let asked = asked.borrow();
    let lists = [&prefills[..], &decodes].concat();
    assert!(
        asked.ends_with(&lists),
        "the prefills, then the decodes: {asked:?}"
    );
    let picks = &asked[..asked.len() - lists.len()];
    assert!(
        !picks.is_empty() && picks.iter().all(|id| *id != decodes[1]),
        "a decode that is not ready is asked as no pick: {picks:?}"
    );
    let heard = heard.lock().clone();
    let first = first_given_up(&heard);
    assert!(
        prefills.contains(&first.0) && first.1 == decodes[0] && heard.len() == 4,
        "one decision"
    );
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A rule that takes a worker of each of the pair's lists is the
/// decision's to answer where those workers do not pair: held to them, the
/// decision finds no prefill with a partner, and its shed names what left
/// nobody. The rule sheds nothing itself here.
///
/// The classic router answers such a request that it has no worker: a
/// prefill counts only while one of its partners the filter takes is open
/// (`placement.rs:2907-2912`, `:2941-2982` on main `bd55beae5`).
#[test]
fn an_encode_requests_rule_whose_workers_do_not_pair_is_answered_by_the_decision() {
    let (engine, _) = hearing(fewest(DecodePick::default()));
    let mut three = Three::of([2, 2, 1], |_, _| false);
    // Prefill 0 pairs with decode 0 alone, and prefill 1 with decode 1.
    three.partners = PartnerClasses::number(2, [&[0_u32][..], &[1][..]]);
    let (prefill, decode) = (three.id(0, 0), three.id(1, 1));
    let admits = |worker: &dyn Worker| {
        let id = worker_id(worker);
        id == prefill || id == decode
    };
    let request = ruled(three.lists((&[PLAIN], &[1; 2])), &admits);
    let answer = engine.place(&request).err();
    let Some(Unplaced::Shed {
        reason, emptied_by, ..
    }) = answer
    else {
        panic!("a shed: {answer:?}");
    };
    assert_eq!((reason, emptied_by), (ShedReason::NoEndpoints, "partnered"));
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// The rule's one prefill hands to no encoder, and the prefill the rule
/// refuses does. The first decision names the refused prefill, the one the
/// items admit. Held to the rule's prefill, the decision is held off every
/// prefill that takes the request, and sheds it under the places' name.
///
/// The classic router has no such request. The one filter it makes for an
/// encode request's pair is the hand-off itself: it refuses a worker no
/// available encoder is compatible with, takes one an encoder is
/// compatible with, and stands in the place of any filter its caller
/// stated (`worker_selection.rs:2599-2613` on main `bd55beae5`). A rule
/// that refuses a prefill an encoder hands to is a host's own.
#[test]
fn a_rules_prefill_that_hands_to_no_encoder_sheds_an_encode_request_by_its_decision() {
    let (engine, heard) = hearing(fewest(DecodePick::default()));
    let three = Three::of([2, 1, 1], |_, _| false);
    let (refused, decode) = (three.id(0, 0), three.id(1, 0));
    let admits = |worker: &dyn Worker| worker_id(worker) != refused;
    // The second prefill names an encode class no encoder is of.
    let request = ruled(three.lists((&[PLAIN], &[1, 2])), &admits);
    let answer = engine.place(&request).err();
    let Some(Unplaced::Shed {
        reason, emptied_by, ..
    }) = answer
    else {
        panic!("a shed: {answer:?}");
    };
    assert_eq!((reason, emptied_by), (ShedReason::NoEndpoints, "places"));
    assert_eq!(
        *heard.lock(),
        given_up(refused, decode),
        "one decision issued"
    );
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// An encode request held to the rule's workers answers the word its
/// caller waits on where each prefill the rule takes is at the caller's
/// admission limit beside an open partner, as one over whole lists does.
/// The decision that names a prefill with room is given up first, and
/// nothing is issued for the one that sheds.
///
/// The classic router asks a prefill's limit of the open prefills the
/// filter takes, and answers that they are at capacity
/// (`placement.rs:2988-2997` on main `bd55beae5`).
#[test]
fn a_held_encode_request_whose_prefills_are_at_their_limit_answers_the_word_its_caller_waits_on() {
    let (engine, heard) = hearing(fewest(DecodePick::default()));
    let three = Three::of([2, 1, 1], |_, _| false);
    // The second prefill is at its limit.
    let gates = Capacity(|prefill| prefill != 1);
    let whole = three.lists((&[PLAIN], &[1; 2]));
    let lists = EncodeLists {
        pair: PairLists {
            admission: &gates,
            ..whole.pair
        },
        ..whole
    };
    let (refused, decode) = (three.id(0, 0), three.id(1, 0));
    let admits = |worker: &dyn Worker| worker_id(worker) != refused;
    let request = ruled(lists, &admits);
    assert_eq!(engine.place(&request).err(), Some(Unplaced::PrefillsFull));
    assert_eq!(
        *heard.lock(),
        given_up(refused, decode),
        "one decision issued"
    );
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// The rule is asked among the places an encode request's caller states
/// for a list of its pair, and of no worker outside them: the call's later
/// decisions are held to places of the caller's that the rule takes. The
/// first decision names a stated prefill the rule refuses, and the second
/// the one stated prefill it takes.
#[test]
fn the_rule_is_asked_among_the_places_an_encode_requests_caller_states() {
    let (engine, _) = hearing(fewest(DecodePick::default()));
    let three = Three::of([5, 2, 1], |_, _| false);
    // The caller holds the request to the first three prefills, and the
    // third holds one request more: the first decision names another.
    three.workers[0][2].increment_load();
    let stated = [0, 1, 2];
    let whole = three.lists((&[PLAIN], &[1; 5]));
    let lists = EncodeLists {
        pair: PairLists {
            prefill_places: Some(&stated),
            ..whole.pair
        },
        ..whole
    };
    // The rule takes the third prefill and the fifth, and every decode.
    let refused = [0, 1, 3].map(|at| three.id(0, at));
    let asked = RefCell::new(Vec::new());
    let admits = |worker: &dyn Worker| {
        let id = worker_id(worker);
        asked.borrow_mut().push(id);
        !refused.contains(&id)
    };
    let request = ruled(lists, &admits);
    let placement = engine.place(&request).expect("a placement");
    assert_eq!(three.places(&placement).0, 2, "the caller's and the rule's");
    drop(placement);
    let asked = asked.borrow();
    let outside = [three.id(0, 3), three.id(0, 4)];
    assert!(
        asked.iter().all(|id| !outside.contains(id)),
        "no prefill outside the caller's places is asked"
    );
    let of_list = stated.map(|at| three.id(0, at as usize));
    assert!(
        asked.windows(3).any(|asked| asked == of_list),
        "the caller's three prefills, in the list's order: {asked:?}"
    );
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// The rule is asked of an encode request's lists once a call, behind its
/// first decision: a later decision none of whose targets holds asks it of
/// no list again. The rule's one prefill is not ready. Under a profile
/// with no `healthy` filter the two decisions held to it name it and are
/// given up, so the request has no worker behind three decisions, and the
/// rule was asked of that prefill once: with the list, and at no pick,
/// where a worker that is not ready is not asked of it.
#[test]
fn an_encode_requests_lists_are_asked_of_the_rule_once_a_call() {
    let (engine, _) = hearing(keeping());
    let three = Three::of([2, 1, 1], |_, _| false);
    three.workers[0][1].set_status(WorkerStatus::NotReady);
    let (held, decode) = (three.id(0, 1), three.id(1, 0));
    let asked = Cell::new(0);
    let admits = |worker: &dyn Worker| {
        let id = worker_id(worker);
        asked.set(asked.get() + usize::from(id == held));
        id == held || id == decode
    };
    let request = ruled(three.lists((&[PLAIN], &[1; 2])), &admits);
    assert_eq!(engine.place(&request).err(), Some(Unplaced::NoWorkers));
    assert_eq!(asked.get(), 1, "the list is asked once");
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A held encode request's third decision is held as its second was: each
/// later ask is lent the lists with the places the rule takes. The last
/// prefill of six holds one request more than the others, so the first
/// decision names another. The rule takes the last where it is asked of
/// the list, refuses it at the second decision's pick, and takes it at the
/// third's. Over the whole lists the third decision would name a prefill
/// the rule refuses, and the request would have no worker.
#[test]
fn a_held_encode_requests_third_decision_is_held_to_the_rules_workers_too() {
    let (engine, _) = hearing(fewest(DecodePick::default()));
    let three = Three::of([6, 1, 1], |_, _| false);
    three.workers[0][5].increment_load();
    let (last, decode) = (three.id(0, 5), three.id(1, 0));
    let asked = Cell::new(0);
    let admits = |worker: &dyn Worker| {
        let id = worker_id(worker);
        asked.set(asked.get() + usize::from(id == last));
        id == decode || (id == last && asked.get() != 2)
    };
    let request = ruled(three.lists((&[PLAIN], &[1; 6])), &admits);
    let placement = engine.place(&request).expect("the call's third decision");
    assert_eq!(three.places(&placement), (5, 0, vec![0]));
    assert_eq!(asked.get(), 3, "the list, and two picks");
    drop(placement);
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// The rule is asked of an encode request's lists while the call holds no
/// decision: where it panics there, the decision it refused was given up
/// by name already, both its sides, and no lease is left behind.
#[test]
fn an_encode_requests_rule_that_panics_at_a_list_leaves_no_lease_behind() {
    let (engine, heard) = hearing(fewest(DecodePick::default()));
    let three = Three::of([2, 1, 1], |_, _| false);
    let calls = Cell::new(0);
    // The decision's prefill and its decode are refused, and the first
    // prefill of the list panics.
    let admits = |_: &dyn Worker| {
        calls.set(calls.get() + 1);
        assert!(calls.get() <= 2, "this rule panics");
        false
    };
    let request = ruled(three.lists((&[PLAIN], &[1; 2])), &admits);
    let placed = catch_unwind(AssertUnwindSafe(|| engine.place(&request).is_ok()));
    assert!(placed.is_err(), "the rule's panic reaches the caller");
    assert_eq!(calls.get(), 3, "the two picks, then the first prefill");
    let heard = heard.lock().clone();
    let first = first_given_up(&heard);
    assert!(
        first.1 == three.id(1, 0) && heard.len() == 4,
        "one decision"
    );
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// An encoder that cannot be sent to holds the pair to the rule's workers
/// too: the rule is asked of the two lists behind the first decision that
/// takes no placement, whatever kept it from one. The rule takes both
/// sides of the first decision, whose one encoder cannot be sent to. Under
/// a profile with no `healthy` filter the next decision names the other
/// encoder, and its pair is decided among the workers the rule takes. The
/// rule takes every worker here, so the case holds the question of the
/// two lists and the second decision; that a later ask is lent the places
/// is the routine's one path, held by the cases above.
///
/// The classic router assigns an item among the available encoders alone,
/// and its pair is among the workers its filter takes whichever encoder
/// serves (`worker_selection.rs:2272-2282`, `:2518`, `:2642-2655` on main
/// `bd55beae5`).
#[test]
fn an_encoder_that_cannot_be_sent_to_holds_the_pair_to_the_rules_workers_too() {
    let (engine, heard) = hearing(keeping());
    let three = Three::of([2, 2, 2], |_, _| false);
    // The first encoder holds fewer requests, and its breaker is open.
    three.workers[2][1].increment_load();
    trip(&three.workers[2][0]);
    let asked = RefCell::new(Vec::new());
    let admits = |worker: &dyn Worker| {
        asked.borrow_mut().push(worker_id(worker));
        true
    };
    let request = ruled(three.lists((&[PLAIN; 2], &[1; 2])), &admits);
    let placement = engine.place(&request).expect("the call's second decision");
    assert_eq!(three.places(&placement).2, vec![1], "the other encoder");
    let lists = [0, 1].map(|list| [0, 1].map(|at| three.id(list, at)));
    let asked = asked.borrow();
    let of_lists = [&lists[0][..], &lists[1]].concat();
    assert!(
        asked.windows(4).any(|asked| asked == of_lists),
        "the prefills, then the decodes: {asked:?}"
    );
    assert_eq!(heard.lock().len(), 6, "one decision given up, one taken");
    drop(placement);
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// An encode request whose first decision sheds it as saturated has its
/// rule asked of the pair's two lists too. The decode the rule takes is
/// over its thresholds, and an idle decode stands that the rule refuses
/// and that no prefill pairs with: no decision names it, and it keeps the
/// decode list from being wholly over its thresholds, so the first
/// decision sheds the request. Held to the rule's workers, the second
/// decision judges the overload over that decode alone, and the binding
/// falls back to it.
///
/// The classic router lifts the thresholds of a pool where no worker the
/// filter takes is under them (`placement.rs:2837-2839`, `:2859`,
/// `:3093-3096` on main `bd55beae5`).
#[test]
fn an_encode_request_shed_as_saturated_is_decided_again_among_the_rules_workers() {
    let (engine, heard) = hearing(fewest(DecodePick::default()));
    let mut three = Three::of([1, 2, 1], |_, _| false);
    // The prefill pairs with the first decode alone.
    three.partners = PartnerClasses::number(2, [&[0_u32][..]]);
    three.workers[1][0].set_overloaded(true);
    let refused = three.id(1, 1);
    let admits = |worker: &dyn Worker| worker_id(worker) != refused;
    let request = ruled(three.lists((&[PLAIN], &[1])), &admits);
    let placement = engine.place(&request).expect("the binding's fallback");
    assert_eq!(three.places(&placement), (0, 0, vec![0]));
    let taken = [issued(three.id(0, 0)), issued(three.id(1, 0))];
    assert_eq!(*heard.lock(), taken, "the first decision issued nothing");
    drop(placement);
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// An encode request that states no rule keeps the saturated shed of its
/// first decision: no rule narrows the lists that decision judged, and its
/// shed is the call's answer. A call that decided on would end with no
/// worker behind its third decision: a shed issues nothing, so the answer
/// alone tells the two apart.
#[test]
fn an_encode_request_shed_as_saturated_under_no_rule_keeps_its_shed() {
    let (engine, heard) = hearing(fewest(DecodePick::default()));
    let mut three = Three::of([1, 2, 1], |_, _| false);
    // The prefill pairs with the first decode alone.
    three.partners = PartnerClasses::number(2, [&[0_u32][..]]);
    three.workers[1][0].set_overloaded(true);
    let items = EncodeItems {
        keys: &KEYS,
        ..EncodeItems::default()
    };
    let request = asking(three.lists((&[PLAIN], &[1])), items);
    let answer = engine.place(&request).err();
    let Some(Unplaced::Shed {
        reason, emptied_by, ..
    }) = answer
    else {
        panic!("a shed: {answer:?}");
    };
    assert_eq!(
        (reason, emptied_by),
        (ShedReason::Saturated, "not-overloaded")
    );
    assert!(heard.lock().is_empty(), "no decision issued");
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// An encode request shed as saturated by its first decision, whose rule
/// takes no worker of one list, is shed with no endpoint under the rule's
/// name. The rule is asked of the prefill and then of the two decodes, it
/// takes neither decode, and no second decision is made. The saturated
/// shed does not stand here. The binding steers, and an idle decode that
/// pairs with no prefill keeps the decode list from being wholly over its
/// thresholds: the decision says no list of its shed. Where it says one,
/// the shed stands where the rule's prefills let it
/// ([`rows`](super::rows)).
///
/// The classic router answers such a request that it has no worker: its
/// pass finds nobody open on a side none of whose workers the filter takes
/// (`placement.rs:2956-2982` on main `bd55beae5`), and judges that side
/// with the filter left out (`admitted_pool`, `:1878-1900`; `failure_from`,
/// `:1757-1771`): a decode of it is under its thresholds.
#[test]
fn an_encode_request_shed_as_saturated_whose_rule_takes_no_worker_of_a_list_is_shed_under_its_name()
{
    let (engine, heard) = hearing(fewest(DecodePick::default()));
    let mut three = Three::of([1, 2, 1], |_, _| false);
    // The prefill pairs with the first decode alone.
    three.partners = PartnerClasses::number(2, [&[0_u32][..]]);
    three.workers[1][0].set_overloaded(true);
    let prefill = three.id(0, 0);
    let asked = RefCell::new(Vec::new());
    let admits = |worker: &dyn Worker| {
        let id = worker_id(worker);
        asked.borrow_mut().push(id);
        id == prefill
    };
    let request = ruled(three.lists((&[PLAIN], &[1])), &admits);
    let shed = Unplaced::Shed {
        reason: ShedReason::NoEndpoints,
        emptied_by: "admits",
        retry_after_secs: 1,
        side: None,
    };
    assert_eq!(engine.place(&request).err(), Some(shed));
    let decodes = [0, 1].map(|at| three.id(1, at));
    assert_eq!(
        *asked.borrow(),
        [&[prefill][..], &decodes].concat(),
        "the prefill, then the decodes"
    );
    assert!(heard.lock().is_empty(), "no decision issued");
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A first decision that sheds an encode request as saturated over its
/// ENCODERS stands, and no rule is asked behind it: the rule is asked of
/// no encoder, so it narrows nothing that decision judged. Every encoder
/// is over its thresholds under a binding that sheds a list wholly over
/// them. The rule takes every worker and is asked of none, and the call's
/// answer is the one a request with no rule is given. The decision says
/// that it shed over its encoders, and the call reads that word.
///
/// The classic router has no encoder available for such a request, and
/// answers by the encode role's pool: under a registry that sheds a pool
/// wholly over its thresholds, its overload shed, as here
/// (`worker_selection.rs:2518`, `:2535-2538`, `:1154-1159`, `:1699-1704`
/// on main `bd55beae5`).
#[test]
fn an_encode_request_shed_as_saturated_over_its_encoders_keeps_its_shed() {
    let mut document = fewest(DecodePick::default());
    for binding in &mut document.bindings {
        binding.overload = Some(OverloadSpec {
            shed_when_all_vetoed: true,
            ..OverloadSpec::default()
        });
    }
    let (engine, heard) = hearing(document);
    let three = Three::of([2, 2, 1], |_, _| false);
    three.workers[2][0].set_overloaded(true);
    let items = EncodeItems {
        keys: &KEYS,
        ..EncodeItems::default()
    };
    let unruled = asking(three.lists((&[PLAIN], &[1; 2])), items);
    let under_no_rule = engine.place(&unruled).err();
    let saturated = |answer: &Option<Unplaced>| {
        matches!(
            answer,
            Some(Unplaced::Shed {
                reason: ShedReason::Saturated,
                ..
            })
        )
    };
    assert!(saturated(&under_no_rule), "{under_no_rule:?}");

    let calls = Cell::new(0);
    let admits = |_: &dyn Worker| {
        calls.set(calls.get() + 1);
        true
    };
    let request = ruled(three.lists((&[PLAIN], &[1; 2])), &admits);
    assert_eq!(engine.place(&request).err(), under_no_rule, "the same shed");
    assert_eq!(calls.get(), 0, "no rule is asked behind such a shed");
    assert!(heard.lock().is_empty(), "no decision issued");
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// An encoders' shed stands whatever the rule takes of the pair's lists.
/// The one encoder is over its thresholds under a binding that sheds, and
/// the rule takes no prefill: the request keeps the saturated shed a
/// request with no rule is given, and the rule is asked of no worker. So
/// it is where the decode's list is over its thresholds too: the decision
/// judges the encoders ahead of the pair, and sheds over them first.
///
/// The classic router judges the encoders ahead of the pair as well,
/// whatever its filter takes: no encoder is available, and under a
/// registry that sheds a pool wholly over its thresholds it answers its
/// overload shed (`worker_selection.rs:2518`, `:2535-2538`, `:1699-1704`
/// on main `bd55beae5`).
#[test]
fn an_encode_request_shed_over_its_encoders_keeps_its_shed_whatever_its_rule_takes() {
    for decode_over in [false, true] {
        let mut document = fewest(DecodePick::default());
        for binding in &mut document.bindings {
            binding.overload = Some(OverloadSpec {
                shed_when_all_vetoed: true,
                ..OverloadSpec::default()
            });
        }
        let (engine, heard) = hearing(document);
        let three = Three::of([1, 1, 1], |_, _| false);
        three.workers[2][0].set_overloaded(true);
        three.workers[1][0].set_overloaded(decode_over);
        let items = EncodeItems {
            keys: &KEYS,
            ..EncodeItems::default()
        };
        let unruled = asking(three.lists((&[PLAIN], &[1])), items);
        let under_no_rule = engine.place(&unruled).err();
        assert!(
            matches!(
                under_no_rule,
                Some(Unplaced::Shed {
                    reason: ShedReason::Saturated,
                    ..
                })
            ),
            "{under_no_rule:?}"
        );

        let decode = three.id(1, 0);
        let calls = Cell::new(0);
        let admits = |worker: &dyn Worker| {
            calls.set(calls.get() + 1);
            worker_id(worker) == decode
        };
        let request = ruled(three.lists((&[PLAIN], &[1])), &admits);
        let answer = engine.place(&request).err();
        assert_eq!(answer, under_no_rule, "the decode over: {decode_over}");
        assert_eq!(calls.get(), 0, "no rule is asked behind such a shed");
        assert!(heard.lock().is_empty(), "no decision issued");
        assert_eq!((engine.faults(), engine.unheard()), (0, 0));
    }
}

/// The decision's own word tells an encoders' shed from a pair's, and no
/// profile's place does: under a document that names one profile for the
/// encode side and the prefill side, both sheds are made under that
/// profile. An encoders' shed stands there with no rule asked, under a
/// binding that sheds a list wholly over its thresholds: the encoders are
/// shed over under such a binding alone. A shed made over the decode's
/// list has the rule asked of the two lists, and the call's next decision
/// is made among the workers the rule takes.
#[test]
fn an_encoders_shed_is_told_from_a_pairs_under_one_profile_for_both_sides() {
    let mut document = fewest(DecodePick::default());
    for binding in &mut document.bindings {
        let prefill = binding.profiles.get("prefill").cloned();
        let prefill = prefill.expect("the prefill side's profile");
        binding.profiles.insert("encode".to_owned(), prefill);
    }
    let mut shedding = document.clone();
    for binding in &mut shedding.bindings {
        binding.overload = Some(OverloadSpec {
            shed_when_all_vetoed: true,
            ..OverloadSpec::default()
        });
    }
    let calls = Cell::new(0);

    // Every encoder is over its thresholds, under the binding that sheds:
    // the shed stands.
    let (engine, heard) = hearing(shedding);
    let three = Three::of([2, 2, 1], |_, _| false);
    three.workers[2][0].set_overloaded(true);
    let admits = |_: &dyn Worker| {
        calls.set(calls.get() + 1);
        true
    };
    let request = ruled(three.lists((&[PLAIN], &[1; 2])), &admits);
    let answer = engine.place(&request).err();
    assert!(
        matches!(
            answer,
            Some(Unplaced::Shed {
                reason: ShedReason::Saturated,
                ..
            })
        ),
        "{answer:?}"
    );
    assert_eq!(calls.get(), 0, "no rule is asked behind such a shed");
    assert!(heard.lock().is_empty(), "no decision issued");
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));

    // The decode the rule takes is over its thresholds beside an idle one
    // that the rule refuses and no prefill pairs with: the pair's shed has
    // the rule asked, and the binding, which steers, falls back to the
    // rule's decode.
    let (engine, _) = hearing(document);
    let mut three = Three::of([1, 2, 1], |_, _| false);
    three.partners = PartnerClasses::number(2, [&[0_u32][..]]);
    three.workers[1][0].set_overloaded(true);
    let refused = three.id(1, 1);
    let admits = |worker: &dyn Worker| {
        calls.set(calls.get() + 1);
        worker_id(worker) != refused
    };
    let request = ruled(three.lists((&[PLAIN], &[1])), &admits);
    let placement = engine.place(&request).expect("the binding's fallback");
    assert_eq!(three.places(&placement), (0, 0, vec![0]));
    assert!(calls.get() >= 3, "the two lists are asked: {}", calls.get());
    drop(placement);
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}
