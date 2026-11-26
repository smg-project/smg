//! A placement's sides: one placed request as the values its host keeps
//! with each send, from that send to that side's end.
//!
//! A host that sends a request keeps something with the send until the
//! response ends: a gateway keeps the worker's load guard. A kept
//! placement gives such a host one owned value a side
//! ([`Placement::sides`]): a single request's one, a pair's prefill and
//! decode. A side says its dispatch by one call ([`PlacementSide::sent`])
//! and its end at its drop. With no outcome reported, that end is said in
//! the words a dropped [`Placement`] says: a release where the side never
//! went out, an end of class cancelled where it did. Its host writes no
//! drop of its own and shares nothing itself. A side also answers the
//! label of the profile that picked it ([`PlacementSide::profile`]), so a
//! host counts each side by the value it keeps.
//!
//! **A side's outcome.** The value a host keeps with a send is often not
//! where the answer is read: a streamed response's guard rides in the
//! response's body, and another task reads the worker's answer. So a side
//! hands out handles on its end ([`PlacementSide::side_end`], a
//! [`SideEnd`]), and whoever reads the answer reports its outcome through
//! one ([`SideEnd::reported`]). The side's end is then said once the side
//! and every handle on it are gone: with the outcome last reported, or in
//! the words above where none was. A host that asks no handle changes
//! nothing: the end is said at the side's drop.
//!
//! A handle reaches a stage only under a document one of whose stages
//! takes an end
//! ([`EventKind::Completed`](super::lease::EventKind::Completed)). Under
//! any other document a side keeps its placement by itself, as it does
//! with no handle: the handle it hands out is empty, allocates nothing and
//! takes no lock, and a report through it says nothing. Under a document
//! that takes an end, a side shares its placement with its handles: one
//! allocation a side, and a lock of the side's own at its send, at each
//! report and at its drop. The document asked is the one the placement
//! was decided under, whichever is in force when its host splits it.
//!
//! **A pair's two sides** share the pair's one placement behind a lock.
//! The lock is one request's own: its two sides alone take it, each for
//! the time of one call or of its end, and no other request ever waits
//! on it. The decode side holds it through its second decision
//! ([`PlacementSide::decode_again`]), so whatever the request lends that
//! decision is asked under it: its route's rule
//! ([`PlaceRequest::admits`]), its headers ([`PlaceRequest::headers`]),
//! its ring ([`PlaceRequest::ring`]), its lists' admission
//! ([`PairLists::admission`](crate::engine::PairLists::admission)) and its
//! lists' workers, by their own calls ([`Worker::routing_state`] among
//! them). None of them may send or drop a side of the pair, or report
//! through or drop a handle on a side's end: it would wait on a lock it is
//! asked under and never return. A single request's side owns its
//! placement and takes no lock, but the one it shares the placement with
//! its handles behind.
//!
//! **The order of a pair's words** is kept here, in whatever order its
//! host lets the two sides go. A decode that still stands chosen leaves
//! with its prefill's last word: the prefill's end
//! ([`DecisionGuard::completed`](super::DecisionGuard::completed)), or its
//! release where it never went out. The decode's send would then say
//! nothing. So a prefill side let go while the decode still stands chosen
//! holds its word back until the decode side speaks: at its dispatch, at
//! the call that takes its decode again, or at its own end. The word is
//! said then as it stood when it was held. An end is said with the outcome
//! a handle reported, or with the instant the prefill side was dropped at,
//! so the span a stage reads is the prefill's own. A prefill that never
//! went out is released then: a host that sends a pair's decode and no
//! prefill lets the prefill side go when it will.
//!
//! **A decode decided again** is asked of the pair's decode side
//! ([`PlacementSide::decode_again`]), behind its prefill's answer. Its host
//! may let the prefill side go before that call or behind it.

use std::{fmt, sync::Arc, time::Instant};

use parking_lot::Mutex;
use smg_worker::identity::basic::Worker;

use super::{
    lease::{Outcome, ReleaseReason, Side},
    placement::Placement,
};
use crate::engine::{
    decision_engine::{DecisionRefused, DecodeRefused},
    PlaceRequest, Unplaced,
};

/// One side of a placed request, kept by its host from that side's send to
/// its end ([`Placement::sides`]).
///
/// Dropped, it says the side's end, once, where no handle on that end is
/// left ([`Self::side_end`]): the last of them says it otherwise. With no
/// outcome reported, a side that never went out is released as abandoned,
/// and a side that went out ends as class cancelled, with the span from
/// its dispatch to its drop. A side of a placement that owes no event says
/// nothing and reads no clock.
///
/// A pair's sides say two of those words otherwise (the module's words on
/// the order). A prefill side let go while the decode still stands chosen
/// says its word later, behind the decode side's: its end where it went
/// out, its release where it never did. A decode side that stands chosen
/// and never went out says nothing at its drop: nothing was booked for it.
pub struct PlacementSide {
    side: Side,
    /// The label of the profile that picked this side.
    profile: &'static str,
    kept: Kept,
}

/// How a side keeps the placement it speaks of.
#[expect(
    clippy::large_enum_variant,
    reason = "a single request's side is its placement: its host boxes the side once, and a box here would be a second one"
)]
enum Kept {
    /// By itself: no stage takes an end, so no handle reports one.
    Own(Of),
    /// With the handles on its end ([`SideEnd`]), behind the side's lock.
    Told(Arc<Mutex<Told>>),
}

/// The placement a side speaks of.
#[expect(
    clippy::large_enum_variant,
    reason = "a single request's side is its placement: its host boxes the side once, and a box here would be a second one"
)]
enum Of {
    /// A single request's, which its one side owns.
    Whole(Placement<'static>),
    /// A pair's, which its two sides share.
    Shared(Arc<Mutex<Pair>>),
}

/// A pair's one placement, as its two sides share it.
struct Pair {
    placement: Placement<'static>,
    /// The prefill side's last word, where it is held back for the decode
    /// side's: its end, or the release of a prefill that never went out.
    held_end: Option<End>,
}

/// A side's end, as it is said: the outcome a handle reported and the
/// instant of that report, or no outcome and the instant the side was
/// dropped at.
#[derive(Clone, Copy)]
struct End {
    at: Instant,
    outcome: Option<Outcome>,
}

/// A side as it shares its placement with the handles on its end. Its
/// drop, when the side and every handle are gone, says the side's end.
struct Told {
    side: Side,
    of: Of,
    /// What a handle last reported.
    reported: Option<End>,
    /// The instant the side was dropped at, where its placement still owed
    /// an event then.
    dropped: Option<Instant>,
}

/// Says `side`'s end of `placement` as `end` states it. An outcome goes
/// the way [`Placement::ended`] takes. With none the side is given up in
/// the words of a dropped placement, a single request's one side as a
/// pair's prefill ([`DecisionGuard::released_prefill`](super::DecisionGuard)).
fn say(placement: &mut Placement<'static>, side: Side, end: End) {
    if let Some(outcome) = end.outcome {
        placement.reported(side, outcome, end.at);
        return;
    }
    let Some(held) = &mut placement.held else {
        return;
    };
    match side {
        Side::Decode => held.released_decode(ReleaseReason::Abandoned, end.at),
        _ => held.released_prefill(ReleaseReason::Abandoned, end.at),
    }
}

impl Pair {
    /// Says the prefill's word where it is held back, as it stood when it
    /// was held.
    fn end_held_prefill(&mut self) {
        if let Some(end) = self.held_end.take() {
            say(&mut self.placement, Side::Prefill, end);
        }
    }

    /// `side`'s end. A prefill's is held back where saying it now would
    /// take the chosen decode with it, and one that waited is said behind
    /// the decode's. The release of a prefill that never went out takes a
    /// chosen decode as an end does, and is held back the same.
    ///
    /// An outcome says that its side went out. A prefill's send never said
    /// is said here first, at the report's instant, as a placement's own
    /// end says it ([`Placement::ended`]). So a prefill reported with no
    /// send is held back like any other while its decode stands chosen. A
    /// decode needs no such word: its end says its own send, at the same
    /// instant.
    fn ended(&mut self, side: Side, end: End) {
        if side != Side::Decode {
            if let (Some(_), Some(held)) = (end.outcome, &mut self.placement.held) {
                held.dispatched(end.at);
            }
        }
        let lease = self.placement.held.as_ref();
        if side != Side::Decode && lease.is_some_and(|held| held.end_takes_decode()) {
            self.held_end = Some(end);
            return;
        }
        say(&mut self.placement, side, end);
        if side == Side::Decode {
            self.end_held_prefill();
        }
    }
}

impl Of {
    /// [`PlacementSide::sent`], for `side`.
    fn sent(&mut self, side: Side) {
        match self {
            Self::Whole(placement) => placement.sent(side),
            Self::Shared(pair) => {
                let mut pair = pair.lock();
                pair.placement.sent(side);
                // The decode no longer stands chosen: a prefill's end or
                // release that waited for this word is said behind it.
                if side == Side::Decode {
                    pair.end_held_prefill();
                }
            }
        }
    }

    /// [`PlacementSide::decode_again`], for `side`.
    fn decode_again(
        &mut self,
        side: Side,
        request: &PlaceRequest<'_>,
    ) -> Result<Arc<dyn Worker>, Unplaced> {
        let (Side::Decode, Self::Shared(pair)) = (side, &*self) else {
            let refused = DecisionRefused::Decode(DecodeRefused::NotChosen);
            return Err(Unplaced::Refused(refused));
        };
        let mut pair = pair.lock();
        let decode = Arc::clone(pair.placement.decode_taken(request)?);
        // The decode no longer stands chosen: a prefill's end that waited
        // for this side's word is said behind the decode the call took.
        pair.end_held_prefill();
        Ok(decode)
    }

    /// Whether the placement still owes an event.
    fn owes(&self) -> bool {
        match self {
            Self::Whole(placement) => placement.owes_events(),
            Self::Shared(pair) => pair.lock().placement.owes_events(),
        }
    }

    /// [`PlacementSide::decode_chosen`].
    fn decode_chosen(&self) -> bool {
        match self {
            Self::Whole(placement) => placement.decode_chosen(),
            Self::Shared(pair) => pair.lock().placement.decode_chosen(),
        }
    }

    /// `side`'s end, as `end` states it.
    fn ended(&mut self, side: Side, end: End) {
        match self {
            Self::Whole(placement) => say(placement, side, end),
            Self::Shared(pair) => pair.lock().ended(side, end),
        }
    }
}

impl Drop for Told {
    /// The side's end, once the side and every handle on it are gone: with
    /// the outcome last reported, or with none at the instant the side was
    /// dropped at. Nothing is said where the placement owed no event at
    /// that drop.
    fn drop(&mut self) {
        let Some(at) = self.dropped else {
            return;
        };
        let end = self.reported.unwrap_or(End { at, outcome: None });
        self.of.ended(self.side, end);
    }
}

impl Placement<'static> {
    /// The placement as the sides its host sends, each an owned value to
    /// keep with that side's send: the side of [`Placement::worker`], which
    /// is a single request's one side or a pair's prefill, and a pair's
    /// decode side. A placement is a pair's where its lease is a pair's: a
    /// pair always holds one.
    ///
    /// Its host reads the placement's workers before this call and leaves
    /// them as placed: a side names none. A debug build stops on a pair's
    /// placement whose decode worker was taken out, and on any other that
    /// a decode worker was written into.
    ///
    /// Where a stage of the document the placement was decided under
    /// takes an end, each side shares the placement with the handles on
    /// its end ([`PlacementSide::side_end`]): one allocation a side. Under
    /// any other document a side keeps it by itself. A document put in
    /// force since the decision changes neither: a word of the placement's
    /// lease goes to the stages that stood at its issue.
    #[must_use]
    pub fn sides(self) -> (PlacementSide, Option<PlacementSide>) {
        let profile = self.profile;
        let lease = self.held.as_ref();
        let pair = lease.is_some_and(|held| held.target().pair.is_some());
        debug_assert_eq!(
            pair,
            self.decode.is_some(),
            "a placement names a decode worker where it is a pair's, and nowhere else"
        );
        let told = lease.is_some_and(|held| held.takes_end());
        if !pair {
            if !told {
                // One expression, so that a build can write the side where
                // it is returned. Handed through a call, the placement is
                // copied at each step on the way.
                let whole = PlacementSide {
                    side: Side::Single,
                    profile,
                    kept: Kept::Own(Of::Whole(self)),
                };
                return (whole, None);
            }
            let whole = PlacementSide::told(Side::Single, profile, Of::Whole(self));
            return (whole, None);
        }
        let decode_profile = self.decode_profile.unwrap_or(profile);
        let pair = Arc::new(Mutex::new(Pair {
            placement: self,
            held_end: None,
        }));
        let shared = Arc::clone(&pair);
        if !told {
            // As a single request's side above: each side is written where
            // the pair is returned. Handed through a call, a side is copied
            // whole on the way: one shared pointer in a placement's room.
            return (
                PlacementSide {
                    side: Side::Prefill,
                    profile,
                    kept: Kept::Own(Of::Shared(shared)),
                },
                Some(PlacementSide {
                    side: Side::Decode,
                    profile: decode_profile,
                    kept: Kept::Own(Of::Shared(pair)),
                }),
            );
        }
        let prefill = PlacementSide::told(Side::Prefill, profile, Of::Shared(shared));
        let decode = PlacementSide::told(Side::Decode, decode_profile, Of::Shared(pair));
        (prefill, Some(decode))
    }
}

impl PlacementSide {
    /// `side` of the placement `of`, shared with the handles on its end: a
    /// side under a document one of whose stages takes an end.
    ///
    /// A side kept by itself is built in [`Placement::sides`] directly, so
    /// that neither a placement nor a side is moved through here.
    fn told(side: Side, profile: &'static str, of: Of) -> Self {
        let handles = Arc::new(Mutex::new(Told {
            side,
            of,
            reported: None,
            dropped: None,
        }));
        Self {
            side,
            profile,
            kept: Kept::Told(handles),
        }
    }

    /// The label of the profile that picked this side, as its placement
    /// names it: [`Placement::profile`], and for a pair's decode side
    /// [`Placement::decode_profile`]. What a host counts the side under.
    #[must_use]
    pub fn profile(&self) -> &'static str {
        self.profile
    }

    /// A handle on this side's end, for whoever reads the side's answer
    /// ([`SideEnd::reported`]). A host asks one for each party that may
    /// learn how the side ended, or clones one. The side's end waits for
    /// every handle: it is said when the side and the last of them are
    /// gone.
    ///
    /// The handle is empty under a document none of whose stages takes an
    /// end, and for a side of a placement that holds no lease: it
    /// allocates nothing and takes no lock.
    #[must_use]
    pub fn side_end(&self) -> SideEnd {
        match &self.kept {
            Kept::Own(_) => SideEnd(None),
            Kept::Told(told) => SideEnd(Some(Arc::clone(told))),
        }
    }

    /// This side of the request went out: its dispatch, said once
    /// ([`Placement::sent`]).
    pub fn sent(&mut self) {
        match &mut self.kept {
            Kept::Own(of) => of.sent(self.side),
            Kept::Told(told) => told.lock().of.sent(self.side),
        }
    }

    /// Whether the pair's decode stands chosen
    /// ([`Placement::decode_chosen`]): the answer of the pair's one
    /// placement, the same on either of its sides, and `false` on a single
    /// request's one side. The host of a pair's decode side asks it ahead
    /// of [`Self::decode_again`]: where it answers `false` that call is
    /// refused by name with no decision made, so the host states no request
    /// again and sends the decode it holds. It is read under the pair's
    /// lock, as that call is.
    #[must_use]
    pub fn decode_chosen(&self) -> bool {
        match &self.kept {
            Kept::Own(of) => of.decode_chosen(),
            Kept::Told(told) => told.lock().of.decode_chosen(),
        }
    }

    /// Decides the pair's decode again behind its prefill's answer
    /// ([`Placement::decode_again`]), for a pair whose first decision
    /// booked its prefill alone: the call of the pair's decode side. It
    /// answers the decode the request now goes to. Its host sends the
    /// decode there, and says this side's [`Self::sent`] at that send.
    ///
    /// `request` states the request again, as [`Placement::decode_again`]
    /// reads it. What it lends the decision (its rule, its headers, its
    /// ring, its lists' admission and their workers' own calls) is asked
    /// under the pair's lock and must leave the pair's sides and the
    /// handles on their ends alone (the module's words on the lock).
    ///
    /// **The order.** The call is made once the prefill side's send was
    /// said. Its host may let the prefill side go before the call or
    /// behind it: a prefill's end that waits for this side's word is said
    /// behind the decode the call took, as it stood when it was held.
    /// Behind a call that took no decode nothing more is said, and that
    /// end still waits for this side's word.
    ///
    /// # Errors
    ///
    /// As [`Placement::decode_again`]: the placement stands as it stood
    /// before the call. A side that is not a pair's decode side, which is
    /// a pair's prefill side or a single request's one side, is refused
    /// with no decision made: [`Unplaced::Refused`] with
    /// [`DecisionRefused::Decode`] and [`DecodeRefused::NotChosen`].
    pub fn decode_again(
        &mut self,
        request: &PlaceRequest<'_>,
    ) -> Result<Arc<dyn Worker>, Unplaced> {
        match &mut self.kept {
            Kept::Own(of) => of.decode_again(self.side, request),
            Kept::Told(told) => told.lock().of.decode_again(self.side, request),
        }
    }
}

impl Drop for PlacementSide {
    /// The side's end, with no outcome, of a side that keeps its placement
    /// by itself. A single request's placement says its own at its drop.
    /// A side that shares its placement with handles leaves the instant of
    /// this drop, and its end is said when the last of them is gone: here,
    /// where none is left.
    fn drop(&mut self) {
        match &mut self.kept {
            Kept::Own(Of::Whole(_)) => {}
            Kept::Own(Of::Shared(pair)) => {
                let mut pair = pair.lock();
                // The common drop of a side that ended: nothing is owed on
                // either side, so no word is said and no clock is read.
                if pair.placement.owes_events() {
                    let at = Instant::now();
                    pair.ended(self.side, End { at, outcome: None });
                }
            }
            Kept::Told(told) => {
                let mut told = told.lock();
                if told.of.owes() {
                    told.dropped = Some(Instant::now());
                }
            }
        }
    }
}

impl fmt::Debug for PlacementSide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PlacementSide")
            .field("side", &self.side)
            .field("profile", &self.profile)
            .finish_non_exhaustive()
    }
}

/// A handle on one side's end ([`PlacementSide::side_end`]): whoever reads
/// the side's answer reports through it how the side ended.
///
/// A handle is cheap to clone and travels with its holder: to the task
/// that reads a streamed answer, and past the side itself. The side's end
/// is said once the side and every handle on it are gone. A handle dropped
/// with no report changes nothing: with none reported the end is said in
/// the words of a side with no handle, with the instant that side was
/// dropped at.
///
/// The default handle is empty, as is the one a side hands out where no
/// stage takes an end: a report through it says nothing, and it holds
/// nothing.
#[derive(Clone, Default)]
pub struct SideEnd(Option<Arc<Mutex<Told>>>);

impl SideEnd {
    /// Whether the handle is empty: a report through it reaches nobody.
    /// Its holder asks before it makes an outcome for the handle alone.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_none()
    }

    /// The side ended with `outcome`, as this handle's holder read it:
    /// the status, the class and what it measured
    /// ([`Outcome::reported`]). The clock is read here.
    ///
    /// The outcome is said when the side and every handle on it are gone,
    /// the way [`Placement::ended`] says one. The side writes three things
    /// into it, whatever its holder wrote there: the side, the worker that
    /// served, and the span from the side's dispatch to this report. An
    /// outcome reported of a side whose send was never said says the send
    /// first, as [`Placement::ended`] does. A pair's prefill reported so is
    /// held back for its decode's word as one that was sent: the decode
    /// that stands chosen stays, and may still be sent or asked again.
    ///
    /// The last report of a side's handles stands: a later one writes the
    /// earlier over, with its own instant.
    pub fn reported(&self, outcome: Outcome) {
        let Some(told) = &self.0 else {
            return;
        };
        let at = Instant::now();
        told.lock().reported = Some(End {
            at,
            outcome: Some(outcome),
        });
    }
}

impl fmt::Debug for SideEnd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SideEnd")
            .field("empty", &self.is_empty())
            .finish()
    }
}

#[cfg(test)]
#[path = "../../tests/unit/tracking/placement_side/tests.rs"]
mod tests;
