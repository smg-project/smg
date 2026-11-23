//! One `Decide` frame as the engine over worker lists takes a request.
//!
//! That engine asks its caller for a request's maker ([`RequestMaker`]) and
//! makes the request itself, under the document that decides.
//! [`FrameRequest`] is the maker of a controller: a checked frame
//! ([`Checked`]), the worker lists the controller hands for the frame's
//! model, and what a controller alone knows of the decision.
//!
//! **No seed and no block hash is made here.** A frame's routing key, its
//! seed and the block hashes of its prompt were made once, by the gateway
//! that asks, and the frame carries them. They are read by the one reader
//! of a frame ([`Checked::facts`], [`Checked::constraints`]). That reader
//! fingerprints a tenant or a session key the frame carries in the clear,
//! at each read.
//!
//! **What a controller adds.** The block hashes are cut at the depth the
//! deciding document reads of them. The budget is what is left of the
//! frame's at the decision. The places of its list the asking gateway may
//! take, and the positions its standing vetoes hold back, are stated by
//! position ([`FrameRequest::held_to`]). A keyed frame's request says that
//! its key's requests are counted on each worker, and the maker says under
//! what name ([`RequestMaker::session_counted_under`]): the session's
//! number, by the decisions the controller holds on file.
//!
//! **What the maker does not state.**
//!
//! - The asking gateway's wires. The engine declines a list that is not
//!   dialed whole on the wires a request states. The maker's caller states
//!   them as places instead ([`FrameRequest::held_to`]): a worker that
//!   shares no wire with the gateway is no place.
//! - A positive subset hint. No stage of the crate reads one, and the
//!   engine would decline the request that states it. A hint of 0 is a
//!   decision over no candidates, and the maker keeps it. Its answer is
//!   the caller's to give before it asks the engine
//!   (`GatewayPlaces::taken`), and the engine refuses a frame whose caller
//!   gave none: it declines a single request, and it sheds a pair.
//! - A plain request ([`RequestMaker::plain`]). Every decision of a
//!   controller is a whole one, with the alternates its pass finds. A
//!   gateway's dispatch names a target of the answer by its index, and a
//!   gateway cannot ask again for the price of a call.
//!
//! **What stays the caller's.** The lists are the ones read from the frame
//! itself ([`Lists::of`](super::wire::Lists::of)): the two are handed
//! apart. A frame that carries its prompt as text
//! ([`Checked::has_text`]) is refused before its request is made: the
//! maker makes a request of any checked frame. And the worker lists state
//! the store's ids: a frame names the workers it tried and the ones it
//! excludes by them.

use std::{sync::Arc, time::Duration};

use smg_worker::load::state::LoadSnapshot;

use super::wire::{Checked, FrameLists};
use crate::{
    document::spec::AppliedSpec,
    engine::{
        decision_engine::{Request, RequestMaker, WorkerLists},
        view::{Constraints, RequestFacts, SessionCountedUnder, WireModes},
    },
};

/// One checked `Decide` frame as a request of the engine over worker
/// lists: the frame, its lists as its decider read them, what is left of
/// its budget, the worker lists of its model, and the places of a single
/// request's list its gateway may take.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FrameRequest<'a> {
    frame: &'a Checked,
    lists: &'a FrameLists<'a>,
    /// What is left of the frame's budget at the decision, in microseconds.
    budget_micros: u32,
    workers: WorkerLists<'a>,
    /// The load snapshot the request's decision reads; `None` for the one
    /// its engine keeps.
    loads: Option<&'a Arc<LoadSnapshot>>,
    /// The places of its list the request may take; `None` for every place.
    places: Option<&'a [u32]>,
    /// The positions a standing veto of the asking gateway holds back.
    stalled: &'a [u32],
    /// Those of `stalled` whose veto no rule lifts.
    held: &'a [u32],
}

impl<'a> FrameRequest<'a> {
    /// `frame` as a request over `workers`, the lists its controller hands
    /// for the frame's model, with `left` of its budget at the decision.
    /// `lists` are the lists its decider read from this frame
    /// ([`Lists::of`](super::wire::Lists::of)) and from no other: the two
    /// are handed apart, and a request made of one frame and another
    /// frame's lists bears that frame's headers and exclusions.
    ///
    /// The request may take every place of its list, and no position is
    /// held back ([`Self::held_to`]). Its decision reads the load snapshot
    /// its engine keeps ([`Self::reading`]).
    ///
    /// `left` is a microsecond or more. A frame with less left is answered
    /// where its decider takes it, and no request is made of it: a request
    /// that states a budget of 0 says that its frame states none.
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
            loads: None,
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

    /// The request with the load snapshot its decision reads: the newest
    /// its deciding thread holds of the replica's load monitor
    /// ([`FrameLoads::newest`](super::frame_loads::FrameLoads::newest)).
    #[must_use]
    pub(crate) fn reading(self, loads: &'a Arc<LoadSnapshot>) -> Self {
        Self {
            loads: Some(loads),
            ..self
        }
    }
}

// `plain` keeps the trait's answer: a controller's request is never plain.
impl RequestMaker for FrameRequest<'_> {
    fn model(&self) -> &str {
        self.frame.model()
    }

    fn make<'w>(&'w self, spec: &'w AppliedSpec) -> Request<'w> {
        let frame = self.frame;
        let facts = frame.facts(self.lists, frame.hashes_read(spec), self.budget_micros);
        Request {
            id: frame.id(),
            facts: RequestFacts {
                // The decisions a controller holds on file count a key's
                // requests on each worker: a keyed frame's count stands.
                session_counted: !facts.session_key.is_none(),
                ..facts
            },
            constraints: Constraints {
                // The gateway's wires are the caller's to state, as places.
                wire_modes: WireModes::NONE,
                // No stage of the crate reads a positive hint. A hint of 0
                // is kept: the engine refuses a frame whose caller did not
                // answer it.
                subset_hint: frame.subset().filter(|&hint| hint == 0),
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

    fn loads(&self) -> Option<&Arc<LoadSnapshot>> {
        self.loads
    }

    // A controller serves no request, so no guard of its own counts one
    // under its key's text. The decisions it holds on file count a key's
    // requests on each worker, under the session's number.
    fn session_counted_under(&self) -> SessionCountedUnder {
        SessionCountedUnder::SessionNumber
    }
}

#[cfg(test)]
mod tests;
