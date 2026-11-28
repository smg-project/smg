//! A pair's shadows by side: a pair's decision made with room for what its
//! shadows say, and the count of it.
//!
//! [`decide_shadowed`] makes a pair's first decision and
//! [`decide_decode_shadowed`] its decode decided again, each by the pair's
//! own function. Each leaves what the shadows said beside that decision in
//! the room its entry lent ([`side_shadows`], [`later_shadow`]), and
//! [`count_sides`] counts it.
//!
//! This module is a child of the pair's because it reads private items of
//! that module as they stand: [`Beside`] and its method `of`,
//! [`decide_records`], [`decide_decode_records`] and [`first_decode`].

use super::{decide_decode_records, decide_records, first_decode, Beside};
use crate::engine::{
    decision::{Decision, PairSide, Pick, ShadowKind, Target},
    framework::{PairPools, Scratch, ShadowRows},
    in_process::{with_shadow, Loaded},
    records::with_shadow::Lends,
    shadow::{budget_left, ShadowSide, Shadowed, SHADOW_FLOOR},
    view::{Constraints, Phase, PoolFacts, RequestFacts, WorkerId},
};
// `Engine`, `decided` and `InProcess` are named only by doc links of this module.
#[cfg(doc)]
use crate::engine::{
    framework::Engine,
    in_process::{pair::decided, InProcess},
};

/// What each side's shadow said beside a pair's first decision, the
/// prefill's then the decode's ([`side_shadows`]): `None` for a side that
/// was due none, and for one whose shadow panicked. Beside a decode
/// decided again the decode's place holds its shadow's word
/// ([`later_shadow`]) and the prefill's holds none. The room an entry
/// lends its decision for those words: an entry that counts no side's
/// shadow lends none, and its decision asks none.
pub(super) type SidesSaid = [Option<Shadowed>; 2];

/// The room an entry lends its decision for its sides' shadows
/// ([`SidesSaid`]), with what the request lends a side's shadow profile
/// alone ([`RequestMaker::shadow_ring`](crate::engine::RequestMaker::shadow_ring)).
/// A side's shadow asks it inside its run, under its catch, so a lender
/// that panics is the shadow's fault, and behind the run's filters, so a
/// shadow whose set they emptied asks nothing ([`Engine::shadow_side`]).
/// A shadow profile one of whose filters declares the ring need asks
/// ahead of them.
pub(super) type Room<'s, 'l> = (&'s mut SidesSaid, Lends<'l>);

/// A pair's first decision by `own` over the records of its two lists
/// ([`decide_records`]), with what each side's shadow said beside it left
/// in `said` ([`side_shadows`]): the shadows run on the scratch as that
/// decision left it, ahead of anything else that writes it.
///
/// Out of line: the path of a decision lent no room holds no part of it.
/// The shadows read the live decision by reference, and a decision lent
/// so inside that path is copied once more on its way out, for every
/// pair.
#[inline(never)]
pub(super) fn decide_shadowed(
    own: Loaded<'_, '_>,
    facts: &RequestFacts<'_>,
    sides: (&Constraints<'_>, [Beside<'_>; 2]),
    pool: &PoolFacts<'_>,
    pools: PairPools<'_>,
    scratch: &mut Scratch,
    (said, lends): Room<'_, '_>,
) -> Decision {
    let engine = own.document.spec.engine();
    let decision = decide_records(engine, facts, sides, pool, pools, scratch);
    *said = side_shadows(own, (facts, lends), sides, pool, pools, &decision, scratch);
    decision
}

/// What each side's shadow said beside `live`, a pair's first decision
/// over the records of its two lists ([`decide_records`]), which read the
/// request's constraints with each side's places and mask (`sides`).
///
/// A side's shadow is due where the pair's binding names one for it that
/// is free to run ([`Engine::shadows_side`]) and a profile made the live
/// decision. A request refused ahead of the pair's profiles has no shadow,
/// as a single's has none: no profile decided it
/// ([`Trace::made_under`](crate::engine::decision::Trace::made_under)).
///
/// A side that is due is decided once more by its shadow profile
/// ([`Engine::shadow_side`]): over the records the live decision read and
/// the scratch it left, under that side's own places and mask. A shadow
/// profile that reads a ring where neither live side does is stated its
/// own place on the ring the request lends, asked of `lends` inside the
/// run, behind its filters or, where one of them declares the ring need,
/// ahead of them. The run writes the scratch and the shadow
/// profile's stages. It writes no record of sent prefixes, issues nothing
/// and tells no stage.
///
/// A side that is due and not asked is said as skipped
/// ([`ShadowKind::Skipped`]): beside a pair that was shed, under less than
/// [`SHADOW_FLOOR`] of the request's stated budget, and where the engine's
/// run says none. That run says none where the side's shadow profile would
/// be handed another column than the side's pool carries
/// ([`Engine::side_apart`]), and beside a side its binding's sessions rule
/// or its picker's assignment placed.
///
/// A prefill's shadow that would ask the pair's default says none too,
/// under a live decode profile with a hard filter that holds a scope or
/// reads more than a row ([`Engine::shadow_side`]).
///
/// Beside a pair whose decode side the binding's all-vetoed rule steered,
/// the decode's shadow is put to that rule and compared. Beside a pair the
/// pair's default made, each side's shadow follows the default as its live
/// side does. Beside a pair a weighing prefill profile's lift alone placed,
/// each runs as beside any pair of the profiles. Both are compared.
/// So the kinds a side counts add up to the decisions its pair's profiles
/// made.
///
/// Each side's run stands under the engine's catch
/// ([`InProcess::caught`]): a shadow that panics costs its request
/// nothing, is counted with the engine's faults and says nothing, and the
/// other side is asked all the same.
fn side_shadows(
    own: Loaded<'_, '_>,
    (facts, lends): (&RequestFacts<'_>, Lends<'_>),
    (cx, sides): (&Constraints<'_>, [Beside<'_>; 2]),
    pool: &PoolFacts<'_>,
    pools: PairPools<'_>,
    live: &Decision,
    scratch: &mut Scratch,
) -> SidesSaid {
    let engine = own.document.spec.engine();
    let made = live.trace().made_under();
    let word = made.and_then(|_| engine.shadow_for(facts.model, Phase::Prefill));
    let Some((at, ShadowRows::Sides)) = word else {
        return [None; 2];
    };
    let [prefills, decodes] = sides;
    let (prefill_cx, decode_cx) = (prefills.of(cx), decodes.of(cx));
    let left = budget_left(facts);
    PairSide::BOTH.map(|side| {
        if !engine.shadows_side(at, side) {
            return None;
        }
        let said = |kind, first| {
            Some(Shadowed {
                binding: at,
                kind,
                first,
            })
        };
        // A live pick and a budget over the floor: else the side is due
        // and not asked.
        let Some(pick) = live.pick().filter(|_| left >= SHADOW_FLOOR) else {
            return said(ShadowKind::Skipped, None);
        };
        let ran = own.caught(|| {
            let (cx, decode_cx) = (&prefill_cx, &decode_cx);
            Ok(engine.shadow_side(
                at, side, facts, pool, pools, cx, decode_cx, pick, scratch, lends,
            ))
        });
        match ran {
            Ok(Some(shadow)) => {
                let (kind, first) = beside(side, pick, &shadow);
                said(kind, first)
            }
            // The engine's run says none for a profile of another column,
            // beside a side its rule or its picker's assignment placed,
            // and for a prefill's shadow that may not ask the default.
            Ok(None) => said(ShadowKind::Skipped, None),
            // The shadow panicked: counted where it was caught.
            Err(_) => None,
        }
    })
}

/// How `shadow`, the decision of a side's shadow, stands beside the live
/// `pick` of its pair, and the worker the shadow took. A pair's first
/// target names its prefill, with the decode beside it, at a first
/// decision and at a decode decided again: a side is compared by its own
/// row of that target.
fn beside(side: PairSide, pick: &Pick, shadow: &Decision) -> (ShadowKind, Option<WorkerId>) {
    let first = shadow.pick().and_then(|pick| pick.targets.first());
    let taken = pick.targets.first().and_then(|target| match side {
        PairSide::Prefill => Some(target.position),
        PairSide::Decode => target.pair.map(|pair| pair.decode_position),
    });
    let kind = match first {
        Some(first) if Some(first.position) == taken => ShadowKind::Agreed,
        Some(_) => ShadowKind::OtherTarget,
        None => ShadowKind::OtherOutcome,
    };
    (kind, first.map(|first| first.id))
}

/// The decode decided again by `own` over the records of the two lists
/// ([`decide_decode_records`]), with what the decode side's shadow said
/// beside it left in the decode's place of `said` ([`later_shadow`]).
///
/// Out of line, as a first decision that is lent room is
/// ([`decide_shadowed`]): the path of a call lent no room holds no part
/// of it.
#[inline(never)]
#[expect(
    clippy::too_many_arguments,
    reason = "the decode call's inputs, beside the room its shadow's word is left in"
)]
pub(super) fn decide_decode_shadowed(
    own: Loaded<'_, '_>,
    facts: &RequestFacts<'_>,
    side: (&Constraints<'_>, Beside<'_>),
    pool: &PoolFacts<'_>,
    pools: PairPools<'_>,
    held: &Target,
    scratch: &mut Scratch,
    (said, lends): Room<'_, '_>,
) -> Decision {
    let engine = own.document.spec.engine();
    let decision = decide_decode_records(engine, facts, side, pool, pools, held, scratch);
    let (again, facts) = ((held, &decision), (facts, lends));
    said[PairSide::Decode.index()] = later_shadow(own, facts, side, pool, pools, again, scratch);
    decision
}

/// What the decode side's shadow said beside the decode decided again,
/// `again`: the target the pair holds and the live decision behind its
/// prefill ([`decide_decode_records`]), which read the request's
/// constraints with the decode list's places and mask (`side`).
///
/// The shadow is due where the pair's binding names one for its decode
/// side that is free to run ([`Engine::shadows_side`]) and the call was
/// decided under the decode profile's name
/// ([`Trace::made_under`](crate::engine::decision::Trace::made_under)): a
/// call refused ahead of it has no shadow.
///
/// A shadow that is due is asked once ([`Engine::shadow_decode_later`]):
/// over the partners of the held prefill the live call read, under the
/// decode list's places and mask, on the call's own draw sequence. It is
/// handed the decode the pair's first decision named, as the live call is
/// ([`first_decode`]): a shadow whose picker takes turns keeps that decode
/// and takes no turn, as its live side takes none. One that reads a ring
/// where neither live side does is stated its own place on the ring the
/// request lends, as beside a first decision ([`side_shadows`]). It is
/// said as skipped ([`ShadowKind::Skipped`]) beside a call that picked no
/// decode, under less than [`SHADOW_FLOOR`] of the request's stated
/// budget, and where the engine's run says none, as a side's is beside a
/// pair's first decision ([`side_shadows`]). The run stands under the
/// engine's catch ([`InProcess::caught`]): a shadow that panics costs
/// its call nothing and is counted with the engine's faults.
fn later_shadow(
    own: Loaded<'_, '_>,
    (facts, lends): (&RequestFacts<'_>, Lends<'_>),
    (cx, side): (&Constraints<'_>, Beside<'_>),
    pool: &PoolFacts<'_>,
    pools: PairPools<'_>,
    (held, live): (&Target, &Decision),
    scratch: &mut Scratch,
) -> Option<Shadowed> {
    let engine = own.document.spec.engine();
    let made = live.trace().made_under();
    let word = made.and_then(|_| engine.shadow_for(facts.model, Phase::Prefill));
    let Some((at, ShadowRows::Sides)) = word else {
        return None;
    };
    if !engine.shadows_side(at, PairSide::Decode) {
        return None;
    }
    let said = |kind, first| {
        Some(Shadowed {
            binding: at,
            kind,
            first,
        })
    };
    let Some(pick) = live.pick().filter(|_| budget_left(facts) >= SHADOW_FLOOR) else {
        return said(ShadowKind::Skipped, None);
    };
    // The constraints of the live call. The run states the decode phase
    // itself, where it takes the decodes' side of the pools.
    let decode_cx = side.of(cx);
    let first = first_decode(held);
    let ran = own.caught(|| {
        let cx = &decode_cx;
        Ok(engine.shadow_decode_later(at, facts, pool, pools, held.id, first, cx, scratch, lends))
    });
    match ran {
        Ok(Some(shadow)) => {
            let (kind, first) = beside(PairSide::Decode, pick, &shadow);
            said(kind, first)
        }
        Ok(None) => said(ShadowKind::Skipped, None),
        // The shadow panicked: counted where it was caught.
        Err(_) => None,
    }
}

/// What each shadow said beside a pair's decision by `own` ([`decided`]),
/// counted in its own table ([`with_shadow::count`]): a first decision's
/// under its side, and the decode decided `again` apart from the decode
/// its pair named first ([`ShadowSide::DecodeAgain`]).
///
/// Out of line: the path of a document that names no shadow holds no part
/// of it.
#[inline(never)]
pub(super) fn count_sides(own: Loaded<'_, '_>, said: SidesSaid, again: bool) {
    for (side, said) in PairSide::BOTH.into_iter().zip(said) {
        if let Some(said) = said {
            let asked = if again {
                ShadowSide::DecodeAgain
            } else {
                ShadowSide::of_pair(side)
            };
            with_shadow::count(own, said, asked);
        }
    }
}
