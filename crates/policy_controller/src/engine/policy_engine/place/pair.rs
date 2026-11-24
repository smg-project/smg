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
//! Which workers pair is the caller's word: the classes its lists carry
//! ([`PartnerClasses`](crate::engine::PartnerClasses)). A decision names no
//! pair across them, and this walk tests no pair again.

use std::{sync::Arc, time::Instant};

use super::{decide, raise, sendable, walk, Call, Made, PlaceRequest, Sent, Sides, Unplaced};
use crate::{
    engine::{
        decision::Target,
        decision_engine::DecisionRefused,
        list_cell::LentList,
        policy_engine::Loaded,
        prefills_full,
        view::{Phase, WorkerId},
        PairLists,
    },
    tracking::Placement,
};

/// The label of the profile that picks a pair's decode side under the
/// engine's document: the decode profile of the binding `model` resolves
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
/// targets both of whose workers can be sent to.
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
    call: &mut Call,
) -> Result<Option<Placement<'a>>, Unplaced> {
    let mut made = Made::asked(engine.object, request, call);
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

/// The asks of one call for a pair over `lists` ([`ask`]): where no target
/// of an ask holds, the call's next ask answers in its place, three asks a
/// call at most ([`Call::again`]), and no worker for the request behind the
/// last.
///
/// # Errors
///
/// What an ask answers, and [`Unplaced::NoWorkers`] where no target of the
/// call's last ask holds.
pub(super) fn place<'a>(
    engine: Loaded<'static, '_>,
    request: &PlaceRequest<'a>,
    lists: &PairLists<'_>,
    call: &mut Call,
) -> Result<Placement<'a>, Unplaced> {
    match ask(engine, request, lists, call)? {
        Some(placement) => Ok(placement),
        None => call.again(|call| place(engine, request, lists, call)),
    }
}
