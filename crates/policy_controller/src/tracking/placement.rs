//! The placement: one placed request as its host keeps it, from the
//! engine's answer to the request's end.
//!
//! The engine of a gateway's own process answers a request in two shapes
//! ([`Decided`](crate::engine::Decided)): a place of which nothing is owed,
//! or a whole decision, whose pick is leased where a stage of the document
//! takes a word of a request, and always for a pair. A [`Placement`] is the
//! one value for both. It names the worker the request goes to, and a
//! pair's decode worker, as the host's own worker objects. Where the pick
//! is leased it holds the lease behind a [`DecisionGuard`], and its host
//! says two things of the request: [`Placement::sent`] and
//! [`Placement::ended`]. Where no lease stands the two calls say nothing
//! and read no clock.
//!
//! **Who makes one.** The host contract has one call for a request, which
//! answers a placement ([`PolicyEngine::place`], the design's section 8.2).
//! No host makes one itself. The call places a single request today.
//!
//! **The instants.** A host reads no clock for the engine: the call that
//! says a word reads it.
//!
//! **The drop.** A placement dropped with a side not ended says what its
//! guard says there: a release before the side went out, an end of class
//! cancelled after it. Its host states no reason.
//!
//! **The ids.** A placement keeps no id of its own. Under a lease it
//! answers the ids of the target its guard holds, and so follows a target
//! taken later. With no lease it takes a worker's id from the worker when
//! asked: a request that is never asked again pays nothing for it.

use std::{sync::Arc, time::Instant};

use smg_worker::identity::basic::Worker;

use super::{
    decision_guard::DecisionGuard,
    lease::{Outcome, Side},
};
use crate::{
    engine::{
        decision_engine::DecideOutcome, policy_engine::PolicyEngine, records::worker_id,
        view::WorkerId,
    },
    store::GatewayId,
};

/// One placed request, kept by its host until the request ends.
///
/// `'a` is the life of what the request lends: the text its routing key
/// borrows. [`Placement::kept`] gives that up, for a value the request
/// keeps to its end.
#[derive(Debug)]
pub struct Placement<'a> {
    /// Where the request goes, as the host's own worker object: a pair's
    /// prefill.
    pub worker: Arc<dyn Worker>,
    /// A pair's decode worker. Its host reads it and leaves it in place:
    /// [`Placement::decode_id`] answers `None` once it is taken out.
    pub decode: Option<Arc<dyn Worker>>,
    /// The routing key the document derived for the request, for its
    /// host's in-flight counters. `None` where none was derived. Neither
    /// constructor derives one: the call that places the request writes it.
    pub routing_key: Option<&'a str>,
    /// The profile whose picker picked, by its label: the profile's name as
    /// the document writes it, as a string that lives as long as the
    /// process. What a host counts the decision under. The call that
    /// decided writes it, so it names the profile of the document that
    /// decided, whatever document is in force later. The empty label for a
    /// decision made ahead of every profile.
    pub profile: &'static str,
    /// The pick's lease: a pair's always, a single request's where a stage
    /// of the document takes a word of a request. `None` for an answer of
    /// which nothing is owed.
    held: Option<DecisionGuard<&'static PolicyEngine>>,
}

impl Placement<'_> {
    /// The placement of a place
    /// ([`AlonePick`](crate::engine::decision_engine::AlonePick)): `worker`
    /// is the worker at the pick's place in the list its host handed, and
    /// `profile` the label of the pick's profile, which the call that
    /// decided read ([`PolicyEngine::label_at`]). Nothing is owed of it, so
    /// it holds no lease, takes no id and reads no clock. The one call for
    /// a request makes it ([`PolicyEngine::place`]).
    #[must_use]
    pub(crate) fn alone(worker: Arc<dyn Worker>, profile: &'static str) -> Self {
        Self {
            worker,
            decode: None,
            routing_key: None,
            profile,
            held: None,
        }
    }

    /// The placement of a whole decision, with the target at `index` of
    /// its pick taken, best first: `decided` and `later` as `engine`
    /// answered them at `now`
    /// ([`Decided::Whole`](crate::engine::Decided::Whole)), and the workers
    /// its host read at that target's positions in its lists, `decode` for
    /// a pair.
    ///
    /// A lease the answer carries is held from here, under the gateway of
    /// the engine's own process, and its dispatch names `index`
    /// ([`DecisionGuard::take`]).
    ///
    /// `None` where the decision picked no target at `index`. A lease the
    /// answer carried is then given up as a guard gives one up at its
    /// drop. The call that places a request makes it
    /// ([`PolicyEngine::place`]). Where none of the targets can be sent
    /// to, that call holds the answer in a guard itself and releases it
    /// under its reason ([`DecisionGuard::released`]).
    #[must_use]
    pub(crate) fn taken(
        engine: &'static PolicyEngine,
        decided: DecideOutcome,
        later: bool,
        now: Instant,
        index: usize,
        worker: Arc<dyn Worker>,
        decode: Option<Arc<dyn Worker>>,
    ) -> Option<Self> {
        let pick = decided.decision.pick();
        // Read here, in the call that decided: the trace's number is a
        // position in the document that decided.
        let profile = pick.map(|pick| engine.label_at(pick.trace.profile));
        let target = pick.and_then(|pick| pick.targets.as_slice().get(index).copied());
        // Held ahead of every refusal below: a lease ends at this guard's
        // drop where the placement is not made.
        let leased = decided.lease.is_some();
        let mut held =
            leased.then(|| DecisionGuard::holding(engine, GatewayId::LOCAL, decided, later, now));
        let (profile, target) = (profile?, target?);
        if let Some(held) = &mut held {
            held.take(index)?;
        }
        debug_assert_eq!(
            (worker_id(worker.as_ref()), decode.as_deref().map(worker_id)),
            (target.id, target.pair.map(|pair| pair.decode_id)),
            "the workers stand at the target's positions"
        );
        Some(Self {
            worker,
            decode,
            routing_key: None,
            profile,
            held,
        })
    }

    /// The placement without what it borrowed of its request, the routing
    /// key: a value that owns all it holds, which the request keeps to its
    /// end, across threads and past the request's own text. Its host reads
    /// the key first.
    #[must_use]
    pub fn kept(self) -> Placement<'static> {
        Placement {
            worker: self.worker,
            decode: self.decode,
            routing_key: None,
            profile: self.profile,
            held: self.held,
        }
    }

    /// The id of the worker the request goes to, for the workers a next
    /// attempt states as tried.
    #[must_use]
    pub fn id(&self) -> WorkerId {
        match &self.held {
            Some(held) => held.target().id,
            None => worker_id(self.worker.as_ref()),
        }
    }

    /// The id of a pair's decode worker. `None` for a single request.
    #[must_use]
    pub fn decode_id(&self) -> Option<WorkerId> {
        let decode = self.decode.as_deref()?;
        let pair = self.held.as_ref().and_then(|held| held.target().pair);
        Some(pair.map_or_else(|| worker_id(decode), |pair| pair.decode_id))
    }

    /// Whether the engine is still owed an event of this request: a side
    /// placed under a lease that has not ended. `false` for an answer of
    /// which nothing was owed, and once every side ended. A host that asks
    /// whether the engine placed its request under a lease asks this
    /// before the request ends, and not whether a placement stands.
    #[must_use]
    pub fn owes_events(&self) -> bool {
        self.held.as_ref().is_some_and(DecisionGuard::owes)
    }

    /// One side of the request went out: its dispatch, said once
    /// ([`DecisionGuard::dispatched`], and
    /// [`DecisionGuard::dispatched_decode`] for a pair's decode). No lease
    /// stands for an encode side, so its word says nothing. A single
    /// request's lease goes out under any other side its host names, as
    /// [`Self::ended`] ends it under any.
    pub fn sent(&mut self, side: Side) {
        let Some(held) = &mut self.held else {
            return;
        };
        let pair = held.target().pair.is_some();
        match side {
            Side::Encode => {}
            Side::Decode if pair => held.dispatched_decode(Instant::now()),
            Side::Single | Side::Prefill | Side::Decode => held.dispatched(Instant::now()),
        }
    }

    /// One side of the request ended with `outcome`
    /// ([`DecisionGuard::completed`]). The placement writes into it the
    /// side and the worker that served, whatever its host wrote there: the
    /// worker of the target taken, or that target's decode worker for a
    /// pair's decode side. No lease stands for an encode side, so its end
    /// says nothing, as its send says nothing: it ends no side of the
    /// request.
    ///
    /// An end said of a side whose send was never said says the send
    /// first, at the same instant: the stages hear a request's events in
    /// their order, and one that times a request from its dispatch reads a
    /// span of zero for it.
    pub fn ended(&mut self, side: Side, outcome: Outcome) {
        let Some(held) = &mut self.held else {
            return;
        };
        let target = *held.target();
        let served = match (side, target.pair) {
            (Side::Encode, _) => return,
            (Side::Decode, Some(pair)) => pair.decode_id,
            _ => target.id,
        };
        let outcome = Outcome {
            side,
            served,
            ..outcome
        };
        held.completed(&outcome, Instant::now());
    }
}

#[cfg(test)]
mod tests {
    use super::Placement;

    /// A kept placement travels with its request: to the task that streams
    /// the response, and behind a pointer several threads share.
    #[test]
    fn a_kept_placement_crosses_threads_and_borrows_nothing() {
        fn travels<T: Send + Sync + 'static>() {}
        travels::<Placement<'static>>();
    }
}
