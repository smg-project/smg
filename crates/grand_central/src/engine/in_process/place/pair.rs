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
//! the call there: behind such prefills the decodes are not asked. An
//! encode request that carries items is held the same way, by the same
//! routine, over the pair's two of its three lists ([`super::encode`]).
//!
//! Which workers pair is the caller's word: the classes its lists carry
//! ([`PartnerClasses`](crate::engine::PartnerClasses)). A decision names no
//! pair across them, and this walk tests no pair again.

use std::{sync::Arc, time::Instant};

use super::{
    decide, hold_to_rule, raise, sendable, walk, Call, Made, PlaceRequest, Sent, Sides, Unplaced,
};
use crate::{
    engine::{
        decision::{ShedReason, Target},
        decision_engine::DecisionRefused,
        in_process::Loaded,
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
/// ([`hold_to_rule`]).
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
/// A shed made over its encoders does not come here: it stands with the
/// ask that made it.
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
        hold_to_rule(
            request,
            Some(lists.prefill),
            lists.prefill_places,
            &mut prefills,
        )?;
        hold_to_rule(
            request,
            Some(lists.decode),
            lists.decode_places,
            &mut decodes,
        )?;
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
