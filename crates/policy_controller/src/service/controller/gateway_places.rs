//! What one gateway may take of one worker list.
//!
//! The engine over worker lists decides over a list every gateway shares.
//! A frame is decided for one gateway, and that gateway's view of the list
//! is its own: it cannot dial another gateway's host-local worker, it
//! dials a worker only on a wire they share, and its standing vetoes hold
//! workers back for it alone. [`GatewayPlaces::taken`] states that view as
//! the engine takes it: the places of the list the frame's request may
//! take, and the positions held back (`FrameRequest::held_to`).
//!
//! - **Reach.** A worker one gateway alone dials is a place for that
//!   gateway and for no other.
//! - **Wires.** A gateway that states wires takes the workers that state
//!   one of them. A worker that states none shares none. A gateway that
//!   states none is held to none. The stated set of a worker decides, not
//!   the one wire its object speaks ([`ListReach`]). The workers a
//!   gateway's wires take are counted ([`GatewayPlaces::undialed`]).
//! - **Vetoes.** A standing veto of the gateway on a worker it may take
//!   holds that worker's position back, and holds it for good while the
//!   veto's hold stands ([`StandingVeto`]). A veto on a worker the list
//!   does not hold, or that is no place, says nothing.
//! - **Nothing left.** Where the gateway dials a worker and its wires
//!   take every one, or it hints a subset of 0, the frame is shed under
//!   that stage and the engine is asked nothing ([`Taken::Emptied`]).
//!   Where it dials no worker of the list the engine decides over no
//!   worker ([`Taken::Unreached`]).
//!
//! A veto names its worker by the store's id, and a position is found by
//! it: the list's records state the store's ids
//! ([`FixedFacts::id`](crate::engine::view::FixedFacts::id)). Over records
//! that carry an address's fingerprint no veto finds its worker.
//!
//! Where nothing narrows, a frame pays two tests and no place is stated.
//! Else the places are written once a frame: a cost that grows with the
//! list.

use super::wire::Checked;
use crate::{
    engine::{
        decision::{SUBSET_HINT, WIRES},
        records::PoolRecords,
        view::WireModes,
    },
    store::{snapshot::StandingVeto, GatewayId, Reachability},
    worker_host::reach::ListReach,
};

/// What a frame says of its gateway: who asks, the wires it states, and
/// its subset hint.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Asking {
    pub(crate) gateway: GatewayId,
    /// `None` where the gateway states no wires.
    pub(crate) wires: Option<WireModes>,
    pub(crate) subset: Option<u32>,
}

impl Asking {
    /// What `frame` says of `gateway`, the gateway whose stream carried it.
    pub(crate) fn of(gateway: GatewayId, frame: &Checked) -> Self {
        Self {
            gateway,
            wires: frame.wires(),
            subset: frame.subset(),
        }
    }
}

/// What a gateway may take of one list.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Taken<'a> {
    /// The frame's request is held to these: what its maker's `held_to`
    /// takes, in its order.
    Held(Held<'a>),
    /// The gateway dials a worker of the list and none is left to take:
    /// the frame is shed `NoEndpoints` under this stage,
    /// [`WIRES`] or [`SUBSET_HINT`].
    Emptied(&'static str),
    /// The gateway dials no worker of the list: its decision is one over
    /// no worker, and no stage emptied anything.
    Unreached,
}

/// The places a frame's request may take and the positions held back, each
/// ascending and each position once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Held<'a> {
    /// The places of the list the request may take: `None` for every
    /// place.
    pub(crate) places: Option<&'a [u32]>,
    /// The places a standing veto of the gateway holds back.
    pub(crate) stalled: &'a [u32],
    /// Those of `stalled` whose veto no rule lifts.
    pub(crate) held: &'a [u32],
}

/// One deciding thread's buffers for the places of its frames: kept from
/// frame to frame, so a frame that narrows allocates nothing once they
/// have grown.
#[derive(Debug, Default)]
pub(crate) struct GatewayPlaces {
    places: Vec<u32>,
    stalled: Vec<u32>,
    held: Vec<u32>,
    /// What [`Self::undialed`] answers: written by each [`Self::taken`].
    undialed: u64,
}

impl GatewayPlaces {
    /// What `asking` may take of the list whose workers' reach is `reach`
    /// and whose records are `records`, under the gateway's standing
    /// `vetoes`, read by worker id. The workers its wires take are counted
    /// for [`Self::undialed`].
    pub(crate) fn taken<'a>(
        &'a mut self,
        reach: &ListReach,
        records: &PoolRecords,
        asking: Asking,
        vetoes: impl Iterator<Item = StandingVeto>,
    ) -> Taken<'a> {
        debug_assert_eq!(reach.workers().len(), records.len(), "one list");
        self.places.clear();
        self.stalled.clear();
        self.held.clear();
        self.undialed = 0;
        let mut narrowed = false;
        if asking.subset == Some(0) || !reach.whole(asking.wires) {
            let mut dialed = 0_usize;
            for (place, worker) in (0_u32..).zip(reach.workers()) {
                let dials = match worker.from {
                    Reachability::Any => true,
                    Reachability::Gateway(only) => only == asking.gateway,
                };
                if !dials {
                    continue;
                }
                dialed += 1;
                if asking
                    .wires
                    .is_none_or(|wires| worker.wires.intersects(wires))
                {
                    self.places.push(place);
                }
            }
            if dialed == 0 {
                return Taken::Unreached;
            }
            // A hint of 0 takes every worker ahead of the wires: they
            // have none to take.
            if asking.subset == Some(0) {
                return Taken::Emptied(SUBSET_HINT);
            }
            self.undialed = (dialed - self.places.len()) as u64;
            if self.places.is_empty() {
                return Taken::Emptied(WIRES);
            }
            // The list's host-local workers can all be the asking
            // gateway's own: it then takes every place, and states none.
            narrowed = self.places.len() != reach.workers().len();
        }
        for veto in vetoes {
            let Some(place) = records.place_of(veto.id) else {
                continue;
            };
            if narrowed && self.places.binary_search(&place).is_err() {
                continue;
            }
            self.stalled.push(place);
            if veto.held {
                self.held.push(place);
            }
        }
        // The vetoes come by worker id, the positions go by place.
        self.stalled.sort_unstable();
        self.held.sort_unstable();
        Taken::Held(Held {
            places: narrowed.then_some(self.places.as_slice()),
            stalled: &self.stalled,
            held: &self.held,
        })
    }

    /// How many workers the wires of the last [`Self::taken`]'s gateway
    /// took: the workers it dials that share no wire with it. None under a
    /// hint of 0, and none for a gateway that states no wires.
    pub(crate) fn undialed(&self) -> u64 {
        self.undialed
    }
}

#[cfg(test)]
mod tests;
