//! One `Decide` frame as the engine over worker lists takes a request.
//!
//! That engine asks its caller for a request's maker ([`RequestMaker`]) and
//! makes the request itself, under the document that decides.
//! [`FrameRequest`] is the maker of a controller: a checked frame
//! ([`Checked`]), the worker lists the controller hands for the frame's
//! model, and what a controller alone knows of the decision.
//!
//! **Nothing is derived here.** A frame's routing key, its seed and the
//! block hashes of its prompt were made once, by the gateway that asks, and
//! the frame carries them. They are read by the one reader of a frame
//! ([`Checked::facts`], [`Checked::constraints`]), as they are read for the
//! engine over stored records.
//!
//! **What a controller adds.** The block hashes are cut at the depth the
//! deciding document reads of them. The budget is what is left of the
//! frame's at the decision. The places of its list the asking gateway may
//! take, and the positions its standing vetoes hold back, are stated by
//! position ([`FrameRequest::held_to`]).
//!
//! **What the maker does not state.**
//!
//! - The asking gateway's wires. The engine declines a list that is not
//!   dialed whole on the wires a request states. The maker's caller states
//!   them as places instead ([`FrameRequest::held_to`]): a worker that
//!   shares no wire with the gateway is no place.
//! - A subset hint. No stage reads a positive one, and the engine would
//!   decline the request that states it. A hint of 0 is a decision over no
//!   candidates: its answer is the caller's to give, before it asks the
//!   engine.
//! - A plain request ([`RequestMaker::plain`]). Every decision of a
//!   controller is a whole one and names its alternates. A gateway's
//!   dispatch names a target of the answer by its index, and a gateway
//!   cannot ask again for the price of a call.

use std::time::Duration;

use super::wire::{Checked, FrameLists};
use crate::{
    document::spec::AppliedSpec,
    engine::{
        decision_engine::{Request, RequestMaker, WorkerLists},
        view::{Constraints, WireModes},
    },
};

/// One checked `Decide` frame as a request of the engine over worker
/// lists: the frame, its lists as its decider read them, what is left of
/// its budget, the worker lists of its model, and the places of a single
/// request's list its gateway may take.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "a deciding thread makes one of each frame once it decides by the engine over worker lists"
    )
)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct FrameRequest<'a> {
    frame: &'a Checked,
    lists: &'a FrameLists<'a>,
    /// What is left of the frame's budget at the decision, in microseconds.
    budget_micros: u32,
    workers: WorkerLists<'a>,
    /// The places of its list the request may take; `None` for every place.
    places: Option<&'a [u32]>,
    /// The positions a standing veto of the asking gateway holds back.
    stalled: &'a [u32],
    /// Those of `stalled` whose veto no rule lifts.
    held: &'a [u32],
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "a deciding thread makes one of each frame once it decides by the engine over worker lists"
    )
)]
impl<'a> FrameRequest<'a> {
    /// `frame` as a request over `workers`, the lists its controller hands
    /// for the frame's model, with `left` of its budget at the decision.
    /// `lists` are the frame's own lists as its decider read them
    /// ([`Lists::of`](super::wire::Lists::of)).
    ///
    /// The request may take every place of its list, and no position is
    /// held back ([`Self::held_to`]).
    pub(crate) fn new(
        frame: &'a Checked,
        lists: &'a FrameLists<'a>,
        left: Duration,
        workers: WorkerLists<'a>,
    ) -> Self {
        Self {
            frame,
            lists,
            // The engine is told what is left in whole microseconds. More
            // than the field holds reads as its widest.
            budget_micros: u32::try_from(left.as_micros()).unwrap_or(u32::MAX),
            workers,
            places: None,
            stalled: &[],
            held: &[],
        }
    }

    /// The request held to what its gateway may take of a single request's
    /// list: `places`, the places of the list it may take, ascending and
    /// each once, and `None` for every place; `stalled`, the positions a
    /// standing veto of the gateway holds back; and `held`, those of them
    /// whose veto no rule lifts. Each is stated as the engine takes it
    /// ([`Constraints::places`], [`Constraints::stalled`],
    /// [`Constraints::held`]).
    #[must_use]
    pub(crate) fn held_to(
        self,
        places: Option<&'a [u32]>,
        stalled: &'a [u32],
        held: &'a [u32],
    ) -> Self {
        Self {
            places,
            stalled,
            held,
            ..self
        }
    }
}

// `plain` keeps the trait's answer: a controller's request is never plain.
// `loads` keeps it too: the engine reads the snapshot it holds itself.
impl RequestMaker for FrameRequest<'_> {
    fn model(&self) -> &str {
        self.frame.model()
    }

    fn make<'w>(&'w self, spec: &'w AppliedSpec) -> Request<'w> {
        let frame = self.frame;
        Request {
            id: frame.id(),
            facts: frame.facts(self.lists, frame.hashes_read(spec), self.budget_micros),
            constraints: Constraints {
                // The gateway's wires are the caller's to state, as places.
                wire_modes: WireModes::NONE,
                // No stage reads a positive hint, and a hint of 0 is the
                // caller's to answer before the engine is asked.
                subset_hint: None,
                places: self.places,
                stalled: self.stalled,
                held: self.held,
                ..frame.constraints(self.lists)
            },
        }
    }

    fn workers(&self) -> WorkerLists<'_> {
        self.workers
    }
}

#[cfg(test)]
mod tests;
