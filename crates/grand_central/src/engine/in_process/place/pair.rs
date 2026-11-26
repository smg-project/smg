//! A pair's request placed: a decision over its caller's two lists, and
//! the first of its targets both of whose workers can be sent to. Where no
//! target holds, the call decides again as a single request's does
//! ([`place`]).
//!
//! A pair's target names a prefill by its place in the prefill list and a
//! decode by its place in the decode list ([`PairLists`]). The targets are
//! walked as a single request's are ([`walk`]), over two workers a target:
//! each side is read once, by the one read a single request's worker gets
//! ([`sendable`]): ready, and admitted by the route's own rule. A target
//! one of whose sides cannot be sent to is passed over whole, and the next
//! is asked.
//!
//! Where no target of a call's first decision holds, or that decision
//! sheds the request as saturated, and the request states a rule, the
//! rule is asked of the two lists once, the prefills' first, and the
//! call's later decisions are held to the workers it takes on each
//! ([`again`]): they are lent the same two lists with each side's places
//! beside it ([`PairLists::prefill_places`],
//! [`PairLists::decode_places`]). A list the rule takes no worker of ends
//! the call there: behind such prefills the decodes are not asked. The
//! request is then shed with no endpoint under the rule's name, or by its
//! first decision's shed as saturated where that shed stands ([`again`]).
//! An encode request that carries items is held the same way, by the same
//! routine, over the pair's two of its three lists ([`super::encode`]).
//!
//! Which workers pair is the caller's word: the classes its lists carry
//! ([`PartnerClasses`](crate::engine::PartnerClasses)). A decision names no
//! pair across them, and this walk tests no pair again.

use std::{sync::Arc, time::Instant};

use super::{
    decide, hold_to_rule, raise, ready_by, sendable, walk, Call, Made, PlaceRequest, Sent, Sides,
    Unplaced,
};
use crate::{
    engine::{
        decision::{PairSide, ShedReason, Target},
        decision_engine::DecisionRefused,
        framework::ShedOver,
        in_process::{pair::shed_over, Loaded},
        list_cell::LentList,
        prefills_full,
        view::{Phase, WorkerId},
        EncodeLists, PairLists, WorkerLists,
    },
    tracking::Placement,
};

/// The label of the profile that picks a pair's decode side under the
/// decision's document: the decode profile of the binding `model` resolves
/// to, and that binding's one profile where it names no other. The empty
/// label where no binding matches.
pub(super) fn decode_label(engine: Loaded<'_, '_>, model: &str) -> &'static str {
    let binding = engine.document.spec.engine().binding(model, Phase::Prefill);
    let position = binding.map(|binding| binding.decode_profile.unwrap_or(binding.profile));
    position.map_or("", |position| engine.label_at(position))
}

/// One ask of `call` for a pair over `lists`, the two lists its caller
/// handed with their pair classes: one decision told to leave out the
/// workers the call missed so far ([`Call::missed`]), and the first of its
/// targets both of whose workers can be sent to. The decision is lent
/// `lent`: the request's own lists, or those lists with each side's
/// places beside it, the workers a rule takes ([`again`]) or the worker a
/// request names ([`target::pair`](super::target::pair)).
///
/// The placement names the request's routing value, and the profile of its
/// decode side by its label ([`decode_label`]).
///
/// `None` where no target holds. The decision is then given up by name,
/// and each side that could not be sent to joins the call's missed
/// workers by its own id: a prefill's or a decode's, never the pair as a
/// whole. The side that held stays a candidate of the next ask.
///
/// # Errors
///
/// The decision's shed, its queue or its refusal. Prefills with an open
/// partner that are all at the caller's admission limit shed the decision
/// under the capacity filter's name ([`prefills_full`]): that is no
/// refusal, and the answer is [`Unplaced::PrefillsFull`], on which the
/// caller waits for room. A place answered for a pair is the engine's
/// fault ([`DecisionRefused::Faulted`]): no ask after it would leave
/// anything new out.
pub(super) fn ask<'a>(
    engine: Loaded<'static, '_>,
    request: &PlaceRequest<'a>,
    lists: &PairLists<'_>,
    lent: &WorkerLists<'_>,
    call: &mut Call,
) -> Result<Option<Placement<'a>>, Unplaced> {
    let mut made = Made {
        lists: lent,
        named_decode: call.named_decode,
        ..Made::asked(engine, request, call)
    };
    let mut whole = None;
    let decided = decide(engine, &made, &mut whole);
    // The route's rule panicked inside the decision, at the question of
    // where the prompt goes: the panic is the caller's, as a single's is.
    if let Some(panic) = made.panicked.take() {
        raise(engine.object, panic, whole, decided);
    }
    let (alone, dated) = decided.map_err(Unplaced::Refused)?;
    let (key, kept) = (made.key(), [made.sent.get(), made.decodes.get()]);
    // A pair's decision is whole: no one pick decides a place of two lists.
    let (None, Some((decided, later))) = (alone, whole) else {
        return Err(Unplaced::Refused(DecisionRefused::Faulted));
    };
    if prefills_full(&decided.decision) {
        return Err(Unplaced::PrefillsFull);
    }
    let now = dated.unwrap_or_else(Instant::now);
    let held = |target: &Target| sides(request, lists, target, &kept);
    let placement = walk(engine, decided, later, now, &mut call.missed, held)?;
    Ok(placement.map(|mut placement| {
        placement.routing_key = key;
        placement.decode_profile = Some(decode_label(engine, request.model));
        placement
    }))
}

/// The two workers of `target` in `lists`, where `request` can be sent to
/// both ([`sendable`]): what a walk of a pair's targets asks of each
/// ([`walk`]), over a pair's two lists and over the pair's two of an encode
/// request's three. Else the ids of the sides it cannot be sent to.
///
/// A worker the decision asked of is taken by the answer it was given
/// there (`kept`: the prefills', then the decodes'), and read no second
/// time: the request goes to the pair its prompt was stored for.
pub(super) fn sides(
    request: &PlaceRequest<'_>,
    lists: &PairLists<'_>,
    target: &Target,
    kept: &[Sent; 2],
) -> Result<Sides, [Option<WorkerId>; 2]> {
    let side = |list: LentList<'_>, place: u32, kept: &Sent| {
        let worker = list.workers.get(place as usize)?;
        let sends = kept.answer(place);
        let sends = sends.unwrap_or_else(|| sendable(request, worker.as_ref()));
        sends.then(|| Arc::clone(worker))
    };
    // A target with no decode side is no pair: it names nobody.
    let pair = target.pair.ok_or([None; 2])?;
    let prefill = side(lists.prefill, target.position, &kept[0]);
    let decode = side(lists.decode, pair.decode_position, &kept[1]);
    match (prefill, decode) {
        (Some(prefill), Some(decode)) => Ok((prefill, Some(decode))),
        (prefill, decode) => Err([
            prefill.is_none().then_some(target.id),
            decode.is_none().then_some(pair.decode_id),
        ]),
    }
}

/// The asks of one call for a pair over `lists` ([`asks`]): the first is
/// lent the request's own lists.
///
/// # Errors
///
/// What an ask answers, and [`Unplaced::NoWorkers`] where no target of the
/// call's last ask holds. A rule that takes no worker of one of the two
/// lists sheds the request with no endpoint, under `admits`
/// ([`hold_to_rule`]), or leaves it its first decision's shed as saturated
/// ([`again`]).
pub(super) fn place<'a>(
    engine: Loaded<'static, '_>,
    request: &PlaceRequest<'a>,
    lists: &PairLists<'_>,
    call: &mut Call,
) -> Result<Placement<'a>, Unplaced> {
    asks(engine, request, lists, &request.lists, call)
}

/// An ask of a call for a pair over `lists`, lent `lent` ([`ask`]), and
/// the asks of the call behind it: where no target of an ask holds, the
/// call's next ask answers in its place ([`again`]), three asks a call at
/// most ([`Call::again`]), and no worker for the request behind the last.
/// An ask that sheds the request as saturated goes there too: it judged
/// the overload over the lists it was lent, which a rule may narrow.
///
/// A pair held to the prefill its request names is asked so over its lists
/// with that prefill's place beside them
/// ([`target::pair`](super::target::pair)): a rule the request states is
/// then asked of that place alone. One held to the two workers it names
/// is the call's one decision: no ask follows it ([`again`]).
pub(super) fn asks<'a>(
    engine: Loaded<'static, '_>,
    request: &PlaceRequest<'a>,
    lists: &PairLists<'_>,
    lent: &WorkerLists<'_>,
    call: &mut Call,
) -> Result<Placement<'a>, Unplaced> {
    let shed = match ask(engine, request, lists, lent, call) {
        Ok(Some(placement)) => return Ok(placement),
        Ok(None) => None,
        Err(
            shed @ Unplaced::Shed {
                reason: ShedReason::Saturated,
                ..
            },
        ) => Some(shed),
        Err(unplaced) => return Err(unplaced),
    };
    let next = |lent: &WorkerLists<'_>, call: &mut Call| asks(engine, request, lists, lent, call);
    again(request, lists, lent, call, shed, &next)
}

/// The side a first decision's shed as saturated stands for behind a rule
/// that takes the prefills at the places `taken` and no decode, the
/// decision having said the list `word` ([`again`]): the list the gateway's
/// own pair selection answers its overload shed over there. `None` where
/// that selection answers no overload shed, and the shed does not stand.
///
/// The rule's prefills here are the ones that pair with a decode: that
/// selection holds no other prefill, so a rule that takes only prefills
/// which pair with no decode takes none of its prefills
/// (`model_gateway/src/routers/common/pair_index.rs:158-161`, on main
/// `56773d28d`). The shed then stands as it does behind a rule that takes
/// no prefill: where the decision shed over the prefills.
///
/// Else the rule's prefills are read in the list's order, each once and
/// none behind the first that can take the request, with the host's
/// vetoes beside the list. One is ready where it can be sent to ([`ready_by`])
/// and no hold of its host stands on it. One can take the request where
/// it is ready, under its thresholds and under no liveness doubt.
///
/// Where one of them can take the request, that selection judges the
/// decode list, whole behind a rule that takes no decode
/// (`model_gateway/src/routers/common/placement.rs:2909-2936`, `:1831-1852`):
/// the shed stands where the decision said the decodes. Where none can,
/// it judges the rule's prefills (`:2866-2884`): the shed stands where one
/// of them is ready and every ready one is over its thresholds, whichever
/// list the decision said, and it stands for the prefill side: that
/// selection names the prefill's role there. A ready prefill under a doubt
/// is one of that
/// count: under its thresholds it keeps the rule's prefills from being
/// wholly over theirs, and over them it does not
/// (`model_gateway/src/routers/common/overload.rs:58-74`).
#[cold]
#[inline(never)]
fn stands(lists: &PairLists<'_>, taken: &[u32], word: ShedOver) -> Option<PairSide> {
    let classes = lists.classes.prefills();
    let pairs = |place: &&u32| {
        classes
            .get(**place as usize)
            .is_some_and(|&class| class != 0)
    };
    let mut ruled = taken.iter().filter(pairs).peekable();
    if ruled.peek().is_none() {
        return (word == ShedOver::Prefills).then_some(PairSide::Prefill);
    }
    let stated = |places: &[u32], place: u32| places.binary_search(&place).is_ok();
    let (mut one_ready, mut all_over) = (false, true);
    for &place in ruled {
        let Some(worker) = lists.prefill.workers.get(place as usize) else {
            continue;
        };
        let state = worker.routing_state();
        if !ready_by(state) || stated(lists.prefill_held, place) {
            continue;
        }
        one_ready = true;
        if !state.overloaded {
            // This one can take the request: no prefill behind it is read.
            if !state.stalled && !stated(lists.prefill_stalled, place) {
                return (word == ShedOver::Decodes).then_some(PairSide::Decode);
            }
            all_over = false;
        }
    }
    (one_ready && all_over).then_some(PairSide::Prefill)
}

/// The call's next ask (`next`) behind one none of whose targets held, or
/// that shed the request as saturated (`shed`), which was lent `lent`
/// ([`asks`]). Behind the call's first ask, a rule the request
/// states is asked of the two lists first, while the call holds no
/// decision ([`hold_to_rule`]): of the prefills, then of the decodes,
/// among the places the caller states for each. It is asked whether it
/// refused a side of the first decision or the sides were not ready: a
/// side that is not ready is not asked of it, so a decision none of whose
/// sides is ready learns nothing of the rule.
///
/// The asks behind that one are lent the request's own lists with each
/// side's places set to the workers the rule takes
/// ([`PairLists::prefill_places`], [`PairLists::decode_places`]): the
/// lists stay the caller's, with their cells and their classes. The
/// places live here, beside those asks: a call keeps none of a pair's. A
/// request that states no rule keeps no places, and each of its asks is
/// lent its lists as they stand. Out of line: a pair whose first ask
/// takes a target runs none of it.
///
/// One routine for a pair and for an encode request that carries items
/// ([`super::encode`]): `lists` are the pair's two, of a pair's lists or
/// of an encode request's three, and `next` is the caller's own ask, which
/// this routine is handed and does not name. The rule is asked of the
/// pair's two lists and of no encoder, and the encoders' list is lent as
/// it stands. An encode request comes here too where an encoder could not
/// be sent to: its pair is then decided among the rule's workers as well.
/// A shed made over its encoders comes here and stands, ahead of the rule:
/// the rule is asked of no encoder, so it narrows nothing that decision
/// judged, and a decision among its workers could only shed again.
///
/// A decision judges the overload over the lists it is lent: a worker
/// under its thresholds that the rule refuses, and that no decision
/// names, keeps a list from being wholly over them, and the decision
/// sheds the request. Held to the rule's workers, the next decision
/// judges it over those alone. Under a document that sheds a list
/// wholly over its thresholds that decision sheds again, and its shed
/// stands. A shed stands where the rule is not asked here: behind a
/// later ask, and under no rule.
///
/// Where the rule takes no worker of a list, no decision can be made
/// among its workers. The first decision's shed as saturated is then the
/// call's answer in two cases, and in both that decision said a list: its
/// own word of the list it shed over ([`shed_over`]), which it says under
/// a document that sheds a list wholly over its thresholds, where the
/// gateway's own selection would read that list so too. The rule takes no
/// prefill, and the decision shed over the prefills. Or the rule takes a
/// prefill and no decode, and its prefills let the shed stand
/// ([`stands`]): one of them can take the request and the decision shed
/// over the decodes, or none can, one is ready and every ready one is over
/// its thresholds. The shed that stands names the side of the list that
/// selection judges: the decision's own side, but the prefill side where
/// none of the rule's prefills can take the request, whichever list the
/// decision said. In every other case the request is shed with no
/// endpoint under `admits`, as it is behind a first decision that picked
/// and behind one that said no list.
///
/// A request held to a part of a list keeps the shed under `admits` in
/// both cases: one that states a hard demand
/// ([`PlaceRequest::demands`]), and one whose host states places of the
/// prefills or of the decodes ([`PairLists::prefill_places`],
/// [`PairLists::decode_places`]). The demand hands the decision the
/// workers that meet it, and the places the workers at them, so a list
/// the decision counted may be a part. The call acts on no word of the
/// prefills or the decodes for such a request: a word of the list that
/// stayed whole goes too, and so does one of a list whose places name
/// every worker. The gateway's own selection states no places. It holds
/// such a need in its filter, and judges the list on the request's wire
/// whole where its filter takes nobody (`admitted_pool`).
///
/// A shed made over the encoders stands for such a request as for any,
/// ahead of this test. Places of the encoders alone
/// ([`EncodeLists::encoders_places`]) hold an encode request to no part
/// of its pair's two lists. Where the encoders at them leave a prefill or
/// a decode out, the encode pass takes the word back itself. So where the
/// word stays, the decision judged both lists whole, and its shed stands.
///
/// The gateway's own pair selection answers its overload shed in both
/// cases, and over the lists as the decision counted them the shed stands
/// in no case where it answers that it has no worker. It judges one list:
/// the prefills where no prefill the rule takes can take the request, and
/// else the decodes. It judges that list over the rule's workers of it, and
/// whole where the rule takes none of it
/// (`model_gateway/src/routers/common/placement.rs:2866-2884`, `:2909-2936`
/// and `:1831-1852`, on main `56773d28d`). A worker that is not ready is no
/// part of a list's overload there, and one under a liveness doubt is
/// (`model_gateway/src/routers/common/overload.rs:58-74`).
///
/// One read behind the rule is fresh, in the second case: the state of
/// the prefills the rule takes that pair with a decode, in the list's
/// order and up to the first that can take the request ([`stands`]),
/// whether the decision counted them or not. One of them that is ready,
/// under no hold, and came under its thresholds since the decision keeps
/// no shed over the prefills. The decision's count of each list is not
/// read again: a decode, or a prefill the rule does not take, whose state
/// changes behind the decision leaves the answer as it is, though that
/// selection asks its filter first and judges the list behind it
/// (`:2866-2884`). A second read here would only move that instant.
///
/// A pair held to the two workers its request names
/// ([`target::pair`](super::target::pair)) is asked no second time: behind
/// a worker it cannot send to it has no worker, and its shed stands.
#[cold]
#[inline(never)]
pub(super) fn again<'a>(
    request: &PlaceRequest<'a>,
    lists: &PairLists<'_>,
    lent: &WorkerLists<'_>,
    call: &mut Call,
    shed: Option<Unplaced>,
    next: &dyn Fn(&WorkerLists<'_>, &mut Call) -> Result<Placement<'a>, Unplaced>,
) -> Result<Placement<'a>, Unplaced> {
    /// `pair` with each side's places beside its list.
    fn held<'w>(pair: PairLists<'w>, prefills: &'w [u32], decodes: &'w [u32]) -> PairLists<'w> {
        PairLists {
            prefill_places: Some(prefills),
            decode_places: Some(decodes),
            ..pair
        }
    }
    if call.named_pair {
        return Err(shed.unwrap_or(Unplaced::NoWorkers));
    }
    let (mut prefills, mut decodes) = (None, None);
    // The rule is asked of a list once a call: behind the first ask.
    if call.spent == 0 {
        // The list the first decision shed the request over, by that
        // decision's own word. Read ahead of the rule: a rule may place a
        // request on this thread, and that decision writes its own word.
        let over = shed.and_then(|shed| Some((shed, shed_over()?)));
        // The rule is asked of no encoder: a shed made over the encoders
        // stands, and no rule is asked behind it.
        if let Some((shed, ShedOver::Encoders)) = over {
            return Err(shed);
        }
        // A hard demand hands the decision a part of each list, the
        // workers that meet it, and the places its host states for the
        // prefills or the decodes hand it the workers at them. A list the
        // decision counted may then be a part: no word of the prefills or
        // the decodes is acted on for such a request.
        let whole = request.demands.is_none()
            && lists.prefill_places.is_none()
            && lists.decode_places.is_none();
        let over = over.filter(|_| whole);
        let prefill = lists.prefill;
        let asked = hold_to_rule(request, Some(prefill), lists.prefill_places, &mut prefills);
        if let Err(none) = asked {
            // The rule takes no prefill: a shed made over the prefills
            // stands.
            return Err(match over {
                Some((shed, ShedOver::Prefills)) => shed,
                _ => none,
            });
        }
        let asked = hold_to_rule(
            request,
            Some(lists.decode),
            lists.decode_places,
            &mut decodes,
        );
        if let Err(none) = asked {
            // The rule takes a prefill and no decode: the shed stands by
            // what the rule's prefills are, for the side they say
            // ([`stands`]).
            let taken = prefills.as_deref().unwrap_or_default();
            let side = over.and_then(|(_, word)| stands(lists, taken, word));
            return Err(match (over, side) {
                (
                    Some((
                        Unplaced::Shed {
                            reason,
                            emptied_by,
                            retry_after_secs,
                            ..
                        },
                        _,
                    )),
                    Some(_),
                ) => Unplaced::Shed {
                    reason,
                    emptied_by,
                    retry_after_secs,
                    side,
                },
                _ => none,
            });
        }
    }
    // A shed stands where the rule was not asked here: behind a later ask,
    // and under no rule.
    if let (Some(shed), None) = (shed, &prefills) {
        return Err(shed);
    }
    let lent = match (request.lists, &prefills, &decodes) {
        (WorkerLists::Pair(pair), Some(prefills), Some(decodes)) => {
            WorkerLists::Pair(held(pair, prefills, decodes))
        }
        // An encode request's three lists: the pair's two with their
        // places, the encoders' as it stands. With no item they are
        // decided as their pair, and with items by the request's own ask.
        (WorkerLists::Encode(three), Some(prefills), Some(decodes)) => {
            WorkerLists::Encode(EncodeLists {
                pair: held(three.pair, prefills, decodes),
                ..three
            })
        }
        // No places were found here: the next ask is lent what this one was.
        _ => *lent,
    };
    call.again(|call| next(&lent, call))
}
