//! A pair's request placed: one decision over its caller's two lists, and
//! the first of its targets both of whose workers can be sent to.
//!
//! A pair's target names a prefill by its place in the prefill list and a
//! decode by its place in the decode list ([`PairLists`]). The targets are
//! walked as a single request's are ([`walk`]), over two workers a target:
//! each side is read once, by the one read a single request's worker gets
//! ([`ready`]). A target one of whose sides cannot be sent to is passed
//! over whole, and the next is asked.
//!
//! Which workers pair is the caller's word: the classes its lists carry
//! ([`PartnerClasses`](crate::engine::PartnerClasses)). A decision names no
//! pair across them, and this walk tests no pair again.

use std::{sync::Arc, time::Instant};

use super::{decide, ready, walk, Made, PlaceRequest, Unplaced};
use crate::{
    engine::{
        decision_engine::Decided,
        policy_engine::{PolicyEngine, PoolMembers},
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
fn decode_label(engine: &PolicyEngine, model: &str) -> &'static str {
    let binding = engine.spec().engine().binding(model, Phase::Prefill);
    let position = binding.map(|binding| binding.decode_profile.unwrap_or(binding.profile));
    position.map_or("", |position| engine.label_at(position))
}

/// One ask for a pair over `lists`, the two lists its caller handed with
/// their pair classes: one decision told to leave `missed` out, and the
/// first of its targets both of whose workers can be sent to.
///
/// The placement names the request's routing value, and the profile of its
/// decode side by its label ([`decode_label`]).
///
/// `None` where no target holds. The decision is then given up by name,
/// and each side that could not be sent to joins `missed` by its own id: a
/// prefill's or a decode's, never the pair as a whole. The side that held
/// stays a candidate of the next ask.
///
/// # Errors
///
/// The decision's shed, its queue or its refusal. Prefills that are all at
/// the caller's admission limit shed the decision under the capacity
/// filter's name ([`prefills_full`]): that is no refusal, and the answer is
/// [`Unplaced::PrefillsFull`], on which the caller waits for room.
pub(super) fn ask<'a>(
    engine: &'static PolicyEngine,
    request: &PlaceRequest<'a>,
    lists: &PairLists<'_>,
    missed: &mut Vec<WorkerId>,
) -> Result<Option<Placement<'a>>, Unplaced> {
    let made = Made::new(engine, request, missed);
    let (decided, dated) = decide(engine, &made).map_err(Unplaced::Refused)?;
    let key = made.key();
    // A pair's decision is whole: no one pick decides a place of two lists.
    let Decided::Whole(decided, later) = decided else {
        return Ok(None);
    };
    if prefills_full(&decided.decision) {
        return Err(Unplaced::PrefillsFull);
    }
    let now = dated.unwrap_or_else(Instant::now);
    let side = |list: &PoolMembers, place: u32| {
        let worker = list.workers().get(place as usize)?;
        ready(worker.as_ref()).then(|| Arc::clone(worker))
    };
    let placement = walk(engine, decided, later, now, missed, |target| {
        // A target with no decode side is no pair: it names nobody.
        let pair = target.pair.ok_or([None; 2])?;
        let prefill = side(lists.prefill, target.position);
        let decode = side(lists.decode, pair.decode_position);
        match (prefill, decode) {
            (Some(prefill), Some(decode)) => Ok((prefill, Some(decode))),
            (prefill, decode) => Err([
                prefill.is_none().then_some(target.id),
                decode.is_none().then_some(pair.decode_id),
            ]),
        }
    })?;
    Ok(placement.map(|mut placement| {
        placement.routing_key = key;
        placement.decode_profile = Some(decode_label(engine, request.model));
        placement
    }))
}
