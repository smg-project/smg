//! The cases of a pair's call held to the workers its route's rule takes
//! ([`hold_to_rule`]): behind the first decision no target of which can be
//! sent to, the rule is asked of the pair's two lists, and the call's
//! later decisions are made among the workers it takes on each.
//!
//! The classic router asks the rule ahead of its selection. Its pair pass
//! counts a worker of either list only where the rule takes it
//! (`model_gateway/src/routers/common/placement.rs:2495-2500` on main
//! `77d6a42d3`), as its selection for a single request narrows the list by
//! the rule first (`accepted_by`, `:474-483` there).

use super::*;

/// What the stages hear of a pair's decision that is given up: both sides
/// issued, then both released.
fn both_given_up(prefill: WorkerId, decode: WorkerId) -> [(String, WorkerId); 4] {
    [
        ("issued at 0".to_owned(), prefill),
        ("issued at 0".to_owned(), decode),
        ("released CommitFailed".to_owned(), prefill),
        ("released CommitFailed".to_owned(), decode),
    ]
}

/// What the stages hear of a pair's decision that is taken.
fn both_issued(prefill: WorkerId, decode: WorkerId) -> [(String, WorkerId); 2] {
    [
        ("issued at 0".to_owned(), prefill),
        ("issued at 0".to_owned(), decode),
    ]
}

/// The one pair of two long lists the rule takes is found by the call's
/// second decision. The first names one prefill and four decodes, and the
/// rule refuses them. The rule is then asked of the thirteen workers of
/// each list, and the second decision is held to the prefill and the
/// decode it takes. Over the whole lists three decisions name three
/// prefills and twelve decodes, never the thirteenth of either, and the
/// request has no worker.
///
/// The classic router places such a request: its pair pass pairs among the
/// workers the rule takes
/// (`model_gateway/src/routers/common/placement.rs:2495-2500` on main
/// `77d6a42d3`).
#[test]
fn the_one_pair_a_rule_takes_in_two_long_lists_is_found_by_the_second_decision() {
    let (engine, heard) = paired(false);
    let lists = Lists::of(13, 13);
    let classes = lists.every();
    let pair = lists.under(&classes, &Open);
    let (prefill, decode) = (lists.prefill_id(12), lists.decode_id(12));
    let calls = Cell::new(0);
    let admits = |worker: &dyn Worker| {
        calls.set(calls.get() + 1);
        let id = worker_id(worker);
        id == prefill || id == decode
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    let placement = engine.place(&request).expect("a placement");
    assert_eq!(lists.places(&placement), (12, 12), "the rule's one pair");
    assert_eq!(
        calls.get(),
        2 * MAX_TARGETS + 26 + 2,
        "each target's two sides, the two lists, the pick's two sides"
    );
    let first = both_given_up(lists.prefill_id(0), lists.decode_id(0));
    assert_eq!(
        said(&heard),
        [&first[..], &both_issued(prefill, decode)].concat(),
        "two decisions"
    );
    drop(placement);
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A prefill the rule takes that is over its overload thresholds is placed
/// beside a prefill the rule refuses that is under them. Over the whole
/// list the overload filter leaves the first out while the second stands,
/// so every decision names the refused prefill and the request has no
/// worker. Held to the rule's prefill, the decision is over a list wholly
/// over the thresholds, and the binding's rule for such a list answers:
/// it falls back, and the prefill is taken.
///
/// The classic router lifts the thresholds the same way, where no worker
/// the rule takes is under them
/// (`model_gateway/src/routers/common/placement.rs:2501-2515` on main
/// `77d6a42d3`, over the rule's test at `:2495-2500`).
#[test]
fn a_pair_is_placed_on_the_rules_prefill_though_it_is_overloaded() {
    let (engine, heard) = paired(false);
    let lists = Lists::of(2, 1);
    let classes = lists.every();
    let pair = lists.under(&classes, &Open);
    lists.prefills[1].set_overloaded(true);
    let (refused, decode) = (lists.prefill_id(0), lists.decode_id(0));
    let admits = |worker: &dyn Worker| worker_id(worker) != refused;
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    let placement = engine.place(&request).expect("the binding's fallback");
    assert_eq!(lists.places(&placement), (1, 0), "the overloaded prefill");
    let taken = both_issued(lists.prefill_id(1), decode);
    assert_eq!(
        said(&heard),
        [&both_given_up(refused, decode)[..], &taken].concat(),
        "two decisions"
    );
    drop(placement);
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A prefill the rule takes that an earlier attempt tried is handed again
/// beside a prefill the rule refuses. Over the whole list the filter of
/// tried workers passes it over while the refused prefill stands, and
/// keeps both once the call missed that one: every decision names the
/// refused prefill, and the request has no worker. Held to the rule's
/// prefill, the filter would leave nobody, and steps aside.
///
/// The classic router steps a retry aside from the prefills it tried while
/// another open prefill is left, among the workers the rule takes
/// (`model_gateway/src/routers/common/placement.rs:2679-2688` on main
/// `77d6a42d3`, over the rule's test at `:2495-2500`).
#[test]
fn a_pair_is_placed_on_the_rules_prefill_though_an_earlier_attempt_tried_it() {
    let (engine, heard) = paired(false);
    let lists = Lists::of(2, 1);
    let classes = lists.every();
    let pair = lists.under(&classes, &Open);
    let (refused, decode) = (lists.prefill_id(0), lists.decode_id(0));
    let admits = |worker: &dyn Worker| worker_id(worker) != refused;
    let tried = [lists.prefill_id(1)];
    let retry = PlaceRequest {
        attempt: 1,
        tried: &tried,
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    let placement = engine
        .place(&retry)
        .expect("tried, and the rule's one prefill");
    assert_eq!(lists.places(&placement), (1, 0), "the tried prefill");
    let taken = both_issued(lists.prefill_id(1), decode);
    assert_eq!(
        said(&heard),
        [&both_given_up(refused, decode)[..], &taken].concat(),
        "two decisions"
    );
    drop(placement);
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A pair whose rule takes no worker of one list is shed with no endpoint
/// under the rule's name, behind one decision. The rule is asked of that
/// decision's picks, then of every prefill, then of every decode, ready or
/// not, and no second decision is made.
///
/// The classic router's pair pass finds nobody open on a side none of
/// whose workers the rule takes
/// (`model_gateway/src/routers/common/placement.rs:2495-2500` on main
/// `77d6a42d3`).
#[test]
fn a_pair_whose_rule_takes_no_worker_of_a_list_is_shed_under_its_name() {
    let (engine, heard) = paired(false);
    let lists = Lists::of(3, 2);
    lists.decodes[1].set_status(WorkerStatus::NotReady);
    let classes = lists.every();
    let pair = lists.under(&classes, &Open);
    let prefills = [0, 1, 2].map(|at| lists.prefill_id(at));
    let asked = RefCell::new(Vec::new());
    let admits = |worker: &dyn Worker| {
        let id = worker_id(worker);
        asked.borrow_mut().push(id);
        prefills.contains(&id)
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    let shed = Unplaced::Shed {
        reason: ShedReason::NoEndpoints,
        emptied_by: "admits",
        retry_after_secs: 1,
    };
    assert_eq!(engine.place(&request).err(), Some(shed));
    let (first, decodes) = (prefills[0], [0, 1].map(|at| lists.decode_id(at)));
    // The decision's two targets name the first prefill with each decode,
    // and the decode that is not ready is not asked as a pick.
    let picks = [first, decodes[0], first];
    assert_eq!(
        *asked.borrow(),
        [&picks[..], &prefills, &decodes].concat(),
        "the picks, the prefills, the decodes"
    );
    let given_up = both_given_up(first, decodes[0]);
    assert_eq!(said(&heard), given_up, "one decision");
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A rule that takes a worker of each list is the decision's to answer
/// where those workers do not pair: held to them, the decision finds no
/// prefill with a partner, and its shed names what left nobody. The rule
/// sheds nothing itself here.
#[test]
fn a_rule_whose_workers_do_not_pair_is_answered_by_the_decision() {
    let (engine, _) = paired(false);
    let lists = Lists::of(2, 2);
    // Prefill 0 pairs with decode 0 alone, and prefill 1 with decode 1.
    let classes = PartnerClasses::number(2, [&[0_u32][..], &[1][..]]);
    let pair = lists.under(&classes, &Open);
    let (prefill, decode) = (lists.prefill_id(0), lists.decode_id(1));
    let admits = |worker: &dyn Worker| {
        let id = worker_id(worker);
        id == prefill || id == decode
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    let Some(Unplaced::Shed {
        reason, emptied_by, ..
    }) = engine.place(&request).err()
    else {
        panic!("a shed");
    };
    assert_eq!((reason, emptied_by), (ShedReason::NoEndpoints, "partnered"));
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A pair whose first decision names a prefill that is not ready has its
/// rule asked of the two lists behind that decision, though the rule
/// refused nobody: a side that is not ready is not asked of it, so such a
/// decision learns nothing of the rule. Under a profile with no health
/// filter, three decisions over the whole lists name the three prefills
/// that are not ready, and the request had no worker though a ready
/// prefill and a ready decode stood that the rule takes. Held to them,
/// the second decision names that pair.
///
/// The classic router places it: its pair pass counts a prefill only while
/// it is available and the rule takes it
/// (`model_gateway/src/routers/common/placement.rs:2495-2500` on main
/// `77d6a42d3`).
#[test]
fn a_pair_whose_first_picks_are_not_ready_is_held_to_the_rules_workers() {
    let (engine, heard) = paired(false);
    let lists = Lists::of(6, 1);
    for prefill in &lists.prefills[..3] {
        prefill.set_status(WorkerStatus::NotReady);
    }
    let classes = lists.every();
    let pair = lists.under(&classes, &Open);
    let (last, decode) = (lists.prefill_id(5), lists.decode_id(0));
    let admits = |worker: &dyn Worker| {
        let id = worker_id(worker);
        id == last || id == decode
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    let placement = engine.place(&request).expect("the call's second decision");
    assert_eq!(lists.places(&placement), (5, 0));
    let given_up = both_given_up(lists.prefill_id(0), decode).to_vec();
    assert_eq!(
        said(&heard),
        [given_up, both_issued(last, decode).to_vec()].concat(),
        "two decisions"
    );
    drop(placement);
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A pair held to the rule's workers answers the word its caller waits on
/// where each prefill the rule takes is at the caller's admission limit
/// beside an open partner, as a pair over whole lists does. The decision
/// that names a prefill with room is given up first, and nothing is issued
/// for the one that sheds.
///
/// The classic router asks a prefill's limit of the open prefills the rule
/// takes, and answers that they are at capacity
/// (`model_gateway/src/routers/common/placement.rs:2664-2674` on main
/// `77d6a42d3`, over the rule's test at `:2495-2500`).
#[test]
fn a_held_pair_whose_prefills_are_at_their_limit_answers_the_word_its_caller_waits_on() {
    let (engine, heard) = paired(false);
    let lists = Lists::of(2, 1);
    let classes = lists.every();
    // The second prefill is at its limit.
    let gates = Gates(|prefill| prefill != 1, |_| true);
    let pair = lists.under(&classes, &gates);
    let (refused, decode) = (lists.prefill_id(0), lists.decode_id(0));
    let admits = |worker: &dyn Worker| worker_id(worker) != refused;
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    assert_eq!(engine.place(&request).err(), Some(Unplaced::PrefillsFull));
    assert_eq!(
        said(&heard),
        both_given_up(refused, decode),
        "one decision issued"
    );
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// Under a binding that sheds a list wholly over its overload thresholds,
/// a pair held to a decode the rule takes that is over them is shed as
/// saturated, at once. The prefill the rule takes is at its limit, and the
/// request does not wait for it beside a partner that cannot be sent to.
///
/// The classic router counts a prefill only while one of its partners is
/// open, ahead of the prefill's limit, and sheds such a request by its
/// overload protection
/// (`model_gateway/src/routers/common/placement.rs:2583-2588`,
/// `:2632-2659` on main `77d6a42d3`).
#[test]
fn a_held_pair_whose_decode_is_overloaded_is_shed_ahead_of_its_prefills_limit() {
    let mut document = PolicySpec::pair_presets("round_robin", "round_robin").expect("presets");
    document.generation = 1;
    document.bindings[0].overload = Some(OverloadSpec {
        shed_when_all_vetoed: true,
        ..OverloadSpec::default()
    });
    let engine = hosted(engine_of(&document));
    let lists = Lists::of(2, 2);
    let classes = lists.every();
    // The second prefill is at its limit, and the second decode over its
    // thresholds.
    let gates = Gates(|prefill| prefill != 1, |_| true);
    lists.decodes[1].set_overloaded(true);
    let pair = lists.under(&classes, &gates);
    let taken = [lists.prefill_id(1), lists.decode_id(1)];
    let admits = |worker: &dyn Worker| taken.contains(&worker_id(worker));
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    let answer = engine.place(&request).err();
    let saturated = matches!(
        answer,
        Some(Unplaced::Shed {
            reason: ShedReason::Saturated,
            ..
        })
    );
    assert!(saturated, "{answer:?}");
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A pair's decision that stores where its prompt went asks the rule
/// itself, of its targets' workers, and the walk takes its answers. A
/// refusal it was given there holds the call's later decisions to the
/// rule's workers, as one the walk is given does: under the `cache_aware`
/// presets a prompt lands on the one pair of two long lists its rule
/// takes.
#[test]
fn a_pair_whose_decision_asks_the_rule_itself_is_held_to_the_rules_workers() {
    let mut document = PolicySpec::pair_presets("cache_aware", "cache_aware").expect("presets");
    document.generation = 1;
    let engine = hosted(engine_of(&document));
    let lists = Lists::of(13, 13);
    let classes = lists.every();
    let pair = lists.under(&classes, &Open);
    let tokens: Vec<u32> = (1_000..5_096).collect();
    let taken = [lists.prefill_id(12), lists.decode_id(12)];
    let admits = |worker: &dyn Worker| taken.contains(&worker_id(worker));
    let request = PlaceRequest {
        prompt: Prompt {
            tokens: Some(&tokens),
            ..Prompt::NONE
        },
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    let placement = engine.place(&request).expect("a placement");
    assert_eq!(lists.places(&placement), (12, 12), "the rule's one pair");
    drop(placement);
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// The rule is asked among the places a pair's caller states for a list,
/// and of no worker outside them: the call's later decisions are held to
/// places of the caller's that the rule takes. A place past a list's end
/// names no worker.
#[test]
fn the_rule_is_asked_among_the_places_a_pairs_caller_states() {
    let (engine, _) = paired(false);
    let lists = Lists::of(5, 2);
    let classes = lists.every();
    // The caller holds the request to the first three prefills.
    let stated = [0, 1, 2];
    let pair = PairLists {
        prefill_places: Some(&stated),
        ..lists.under(&classes, &Open)
    };
    // The rule takes the third prefill and the fifth, and every decode.
    let refused = [0, 1, 3].map(|at| lists.prefill_id(at));
    let asked = RefCell::new(Vec::new());
    let admits = |worker: &dyn Worker| {
        let id = worker_id(worker);
        asked.borrow_mut().push(id);
        !refused.contains(&id)
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    let placement = engine.place(&request).expect("a placement");
    assert_eq!(lists.places(&placement).0, 2, "the caller's and the rule's");
    let outside = [lists.prefill_id(3), lists.prefill_id(4)];
    assert!(
        asked.borrow().iter().all(|id| !outside.contains(id)),
        "no prefill outside the caller's places is asked"
    );
    drop(placement);

    // The places kept for a list are the caller's that the rule takes.
    let older = [1, 2, 9];
    let mut kept = None;
    let held = hold_to_rule(&request, Some(pair.prefill), Some(&older), &mut kept);
    held.expect("the rule takes a stated prefill");
    assert_eq!(kept.as_deref(), Some(&[2][..]));
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// Three lists with no item stated are placed as their pair, and held to
/// the workers the rule takes as a pair is.
#[test]
fn three_lists_with_no_item_are_held_to_the_rules_workers_as_their_pair() {
    let (engine, heard) = paired(false);
    let lists = Lists::of(13, 13);
    let classes = lists.every();
    let pair = lists.under(&classes, &Open);
    let (prefill, decode) = (lists.prefill_id(12), lists.decode_id(12));
    let admits = |worker: &dyn Worker| {
        let id = worker_id(worker);
        id == prefill || id == decode
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, three_of(pair))
    };
    let placement = engine.place(&request).expect("a placement");
    assert_eq!(lists.places(&placement), (12, 12));
    let first = both_given_up(lists.prefill_id(0), lists.decode_id(0));
    assert_eq!(
        said(&heard),
        [&first[..], &both_issued(prefill, decode)].concat(),
        "two decisions"
    );
    drop(placement);
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A pair that states no rule keeps no places: the hold asks nothing of a
/// list, and the call's later decisions are made over both lists whole.
/// Without that test such a pair would be held to every place of each
/// list behind its first decision that takes no target.
#[test]
fn a_pair_that_states_no_rule_keeps_no_places() {
    let lists = Lists::of(3, 2);
    let classes = lists.every();
    let pair = lists.under(&classes, &Open);
    let request = PlaceRequest::of(MODEL, WorkerLists::Pair(pair));
    let (mut prefills, mut decodes) = (None, None);
    let asked = hold_to_rule(&request, Some(pair.prefill), None, &mut prefills);
    asked.expect("no rule sheds nothing");
    let asked = hold_to_rule(&request, Some(pair.decode), None, &mut decodes);
    asked.expect("no rule sheds nothing");
    assert_eq!((prefills, decodes), (None, None));
}

/// The rule is asked of a pair's lists once a call, behind its first
/// decision: a later decision none of whose targets holds asks it of no
/// list again. The rule's one prefill is not ready. Under a profile with
/// no `healthy` filter the two decisions held to it name it and are given
/// up, so the request has no worker behind three decisions, and the rule
/// was asked of that prefill once: with the list, and at no pick, where
/// a worker that is not ready is not asked of it.
#[test]
fn a_pairs_lists_are_asked_of_the_rule_once_a_call() {
    let (engine, _heard) = paired(false);
    let lists = Lists::of(2, 1);
    lists.prefills[1].set_status(WorkerStatus::NotReady);
    let classes = lists.every();
    let pair = lists.under(&classes, &Open);
    let (held, decode) = (lists.prefill_id(1), lists.decode_id(0));
    let asked = Cell::new(0);
    let admits = |worker: &dyn Worker| {
        let id = worker_id(worker);
        asked.set(asked.get() + usize::from(id == held));
        id == held || id == decode
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    assert_eq!(engine.place(&request).err(), Some(Unplaced::NoWorkers));
    assert_eq!(asked.get(), 1, "the list is asked once");
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A held pair's third decision is held as its second was: each later
/// ask is lent the lists with the places the rule takes. The rule takes
/// the last prefill of six where it is asked of the list, refuses it at
/// the second decision's pick, and takes it at the third's. Over the
/// whole lists the third decision would name the one prefill no decision
/// tried, which the rule refuses, and the request would have no worker.
#[test]
fn a_held_pairs_third_decision_is_held_to_the_rules_workers_too() {
    let (engine, _heard) = paired(false);
    let lists = Lists::of(6, 1);
    let classes = lists.every();
    let pair = lists.under(&classes, &Open);
    let (last, decode) = (lists.prefill_id(5), lists.decode_id(0));
    let asked = Cell::new(0);
    let admits = |worker: &dyn Worker| {
        let id = worker_id(worker);
        asked.set(asked.get() + usize::from(id == last));
        id == decode || (id == last && asked.get() != 2)
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    let placement = engine.place(&request).expect("the call's third decision");
    assert_eq!(lists.places(&placement), (5, 0));
    assert_eq!(asked.get(), 3, "the list, and two picks");
    drop(placement);
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// The rule is asked of a pair's lists while the call holds no decision:
/// where it panics there, the decision it refused was given up by name
/// already, both its sides, and no lease is left behind. A single
/// request's list is held the same way
/// (`a_rule_that_panics_at_the_list_leaves_no_lease_behind`).
#[test]
fn a_pairs_rule_that_panics_at_a_list_leaves_no_lease_behind() {
    let (engine, heard) = paired(false);
    let lists = Lists::of(2, 1);
    let classes = lists.every();
    let pair = lists.under(&classes, &Open);
    let calls = Cell::new(0);
    // The decision's prefill and its decode are refused, and the first
    // prefill of the list panics.
    let admits = |_: &dyn Worker| {
        calls.set(calls.get() + 1);
        assert!(calls.get() <= 2, "this rule panics");
        false
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    let placed = catch_unwind(AssertUnwindSafe(|| engine.place(&request).is_ok()));
    assert!(placed.is_err(), "the rule's panic reaches the caller");
    assert_eq!(calls.get(), 3, "the two picks, then the first prefill");
    let given_up = both_given_up(lists.prefill_id(0), lists.decode_id(0));
    assert_eq!(said(&heard), given_up, "one decision");
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A pair whose first decision sheds the request as saturated has its rule
/// asked of the two lists too. The decode the rule takes is over its
/// thresholds, and an idle decode stands that the rule refuses and that no
/// prefill pairs with: no decision names it, and it keeps the decode list
/// from being wholly over its thresholds, so the first decision sheds the
/// request. Held to the rule's workers, the second decision judges the
/// overload over that decode alone, and the binding falls back to it.
///
/// The classic router lifts the thresholds of a pool where no worker the
/// rule takes is under them
/// (`model_gateway/src/routers/common/placement.rs:2663-2677` on main
/// `cf6f0c15e`).
#[test]
fn a_pair_shed_as_saturated_is_decided_again_among_the_rules_workers() {
    let (engine, heard) = paired(false);
    let lists = Lists::of(1, 2);
    // The prefill pairs with the first decode alone.
    let classes = PartnerClasses::number(2, [&[0_u32][..]]);
    let pair = lists.under(&classes, &Open);
    lists.decodes[0].set_overloaded(true);
    let refused = lists.decode_id(1);
    let admits = |worker: &dyn Worker| worker_id(worker) != refused;
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    let placement = engine.place(&request).expect("the binding's fallback");
    assert_eq!(lists.places(&placement), (0, 0), "the overloaded decode");
    let taken = both_issued(lists.prefill_id(0), lists.decode_id(0));
    assert_eq!(said(&heard), taken, "the first decision issued nothing");
    drop(placement);
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A pair that states no rule keeps the saturated shed of its first
/// decision: no rule narrows the lists that decision judged, and the call
/// decides no more.
#[test]
fn a_pair_shed_as_saturated_under_no_rule_keeps_its_shed() {
    let (engine, heard) = paired(false);
    let lists = Lists::of(1, 2);
    // The prefill pairs with the first decode alone.
    let classes = PartnerClasses::number(2, [&[0_u32][..]]);
    let pair = lists.under(&classes, &Open);
    lists.decodes[0].set_overloaded(true);
    let request = PlaceRequest::of(MODEL, WorkerLists::Pair(pair));
    let Some(Unplaced::Shed {
        reason, emptied_by, ..
    }) = engine.place(&request).err()
    else {
        panic!("a shed");
    };
    assert_eq!(
        (reason, emptied_by),
        (ShedReason::Saturated, "not-overloaded")
    );
    assert!(said(&heard).is_empty(), "no decision issued");
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A saturated shed stands where the decision among the rule's workers
/// sheds the request too. The rule takes the idle decode no prefill pairs
/// with, so its decodes are not wholly over their thresholds, and it
/// refuses the one prefill under its limit. The request is shed as
/// saturated, and does not wait for the prefill at its limit: no decode
/// would pair with that prefill.
///
/// The classic router counts a prefill only while one of its partners is
/// open, and judges such a miss from the decode side, ahead of the
/// prefills' limit
/// (`model_gateway/src/routers/common/placement.rs:2778-2822` on main
/// `cf6f0c15e`).
#[test]
fn a_pair_shed_as_saturated_twice_does_not_wait_for_its_prefills_limit() {
    let (engine, heard) = paired(false);
    let lists = Lists::of(2, 2);
    // Both prefills pair with the first decode alone, and the first
    // prefill is at its limit.
    let classes = PartnerClasses::number(2, [&[0_u32][..], &[0][..]]);
    let gates = Gates(|prefill| prefill != 0, |_| true);
    let pair = lists.under(&classes, &gates);
    lists.decodes[0].set_overloaded(true);
    let refused = lists.prefill_id(1);
    let asked = Cell::new(0);
    let admits = |worker: &dyn Worker| {
        asked.set(asked.get() + 1);
        worker_id(worker) != refused
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    let Some(Unplaced::Shed {
        reason, emptied_by, ..
    }) = engine.place(&request).err()
    else {
        panic!("a shed");
    };
    assert_eq!(
        (reason, emptied_by),
        (ShedReason::Saturated, "not-overloaded")
    );
    assert_eq!(asked.get(), 4, "the two lists, once");
    assert!(said(&heard).is_empty(), "no decision issued");
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A pair shed as saturated by its first decision, whose rule takes no
/// worker of one list, is shed with no endpoint under the rule's name. The
/// rule is asked of the prefill and then of the two decodes, it takes
/// neither decode, and no second decision is made. The saturated shed does
/// not stand: a rule that takes no worker of a list is told first.
///
/// The classic router's pair pass finds nobody open on a side none of
/// whose workers the rule takes
/// (`model_gateway/src/routers/common/placement.rs:2495-2500` on main
/// `77d6a42d3`).
#[test]
fn a_pair_shed_as_saturated_whose_rule_takes_no_worker_of_a_list_is_shed_under_its_name() {
    let (engine, heard) = paired(false);
    let lists = Lists::of(1, 2);
    // The prefill pairs with the first decode alone.
    let classes = PartnerClasses::number(2, [&[0_u32][..]]);
    let pair = lists.under(&classes, &Open);
    lists.decodes[0].set_overloaded(true);
    let prefill = lists.prefill_id(0);
    let asked = RefCell::new(Vec::new());
    let admits = |worker: &dyn Worker| {
        let id = worker_id(worker);
        asked.borrow_mut().push(id);
        id == prefill
    };
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    let shed = Unplaced::Shed {
        reason: ShedReason::NoEndpoints,
        emptied_by: "admits",
        retry_after_secs: 1,
    };
    assert_eq!(engine.place(&request).err(), Some(shed));
    let decodes = [0, 1].map(|at| lists.decode_id(at));
    assert_eq!(
        *asked.borrow(),
        [&[prefill][..], &decodes].concat(),
        "the prefill, then the decodes"
    );
    assert!(said(&heard).is_empty(), "no decision issued");
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A pair shed as saturated by its first decision, whose rule takes a
/// worker of each list that do not pair, is the held decision's to answer.
/// The rule refuses the overloaded decode and takes the idle one no prefill
/// pairs with: the decision among the rule's workers finds no prefill with
/// a partner, and its shed names what left nobody. The saturated shed does
/// not stand: the rule was asked here.
///
/// The classic router counts a prefill only while one of its partners is
/// open (`model_gateway/src/routers/common/placement.rs:2778-2822` on main
/// `cf6f0c15e`).
#[test]
fn a_pair_shed_as_saturated_whose_rules_workers_do_not_pair_is_answered_by_the_decision() {
    let (engine, _) = paired(false);
    let lists = Lists::of(1, 2);
    // The prefill pairs with the first decode alone.
    let classes = PartnerClasses::number(2, [&[0_u32][..]]);
    let pair = lists.under(&classes, &Open);
    lists.decodes[0].set_overloaded(true);
    let refused = lists.decode_id(0);
    let admits = |worker: &dyn Worker| worker_id(worker) != refused;
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    let Some(Unplaced::Shed {
        reason, emptied_by, ..
    }) = engine.place(&request).err()
    else {
        panic!("a shed");
    };
    assert_eq!((reason, emptied_by), (ShedReason::NoEndpoints, "partnered"));
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}

/// A held pair's call makes three decisions, as any call does: the two
/// behind its first are made among the rule's workers, and no fourth
/// follows them. Under a profile with no `healthy` filter the rule takes
/// three prefills of four, two of them not ready. The second decision and
/// the third each name one of those two and are given up, and the request
/// has no worker though the rule's third prefill is ready.
#[test]
fn a_held_pairs_call_makes_three_decisions() {
    let (engine, heard) = paired(false);
    let lists = Lists::of(4, 1);
    lists.prefills[1].set_status(WorkerStatus::NotReady);
    lists.prefills[2].set_status(WorkerStatus::NotReady);
    let classes = lists.every();
    let pair = lists.under(&classes, &Open);
    let refused = lists.prefill_id(0);
    let admits = |worker: &dyn Worker| worker_id(worker) != refused;
    let request = PlaceRequest {
        admits: Some(&admits),
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(pair))
    };
    assert_eq!(engine.place(&request).err(), Some(Unplaced::NoWorkers));
    let decode = lists.decode_id(0);
    let given_up = [0, 1, 2].map(|at| both_given_up(lists.prefill_id(at), decode));
    assert_eq!(said(&heard), given_up.concat(), "three decisions");
    assert_eq!((engine.faults(), engine.unheard()), (0, 0));
}
