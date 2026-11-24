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
//! No host makes one itself. The call places a single request, a pair
//! over its two lists and an encode request over its three today.
//!
//! **A decode decided again.** A pair whose first decision booked its
//! prefill alone names a decode that stands chosen
//! ([`PlaceRequest::decode_again`]). When the prefill has answered, its host
//! asks the decode again on the same placement
//! ([`Placement::decode_again`]), and the decode the request goes to is the
//! one that call takes.
//!
//! **The instants.** A host reads no clock for the engine: the call that
//! says a word reads it.
//!
//! **The drop.** A placement dropped with a side not ended says what its
//! guard says there: a release before the side went out, an end of class
//! cancelled after it. Its host states no reason.
//!
//! **Its sides.** A host that keeps a value with each send, as a gateway
//! keeps a worker's load guard, takes a kept placement apart
//! ([`Placement::sides`]): each side then says its own dispatch, and its
//! end at its drop.
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
        decision_engine::{DecideOutcome, DecisionRefused, DecodeRefused},
        policy_engine::{Loaded, PolicyEngine},
        records::worker_id,
        view::WorkerId,
        PlaceRequest, Unplaced,
    },
    store::GatewayId,
};

/// The encoders of an encode request's items, as the host's own worker
/// objects. Nothing is owed of an encoder: the placement's lease is the
/// pair's, and no word of an encode side reaches a stage.
///
/// The value is one pointer wide, and names nothing for a request with no
/// item: a placement of such a request allocates nothing for it.
#[derive(Debug, Default)]
pub struct Encoders(Option<Box<Named>>);

/// What an encode request's placement names of its encoders.
#[derive(Debug)]
struct Named {
    workers: Vec<Arc<dyn Worker>>,
    one: bool,
}

impl Encoders {
    /// The encoders the call that placed a request read for it: `workers`
    /// in the items' order, and whether the items were placed as `one`.
    #[must_use]
    pub(crate) fn named(workers: Vec<Arc<dyn Worker>>, one: bool) -> Self {
        Self(Some(Box::new(Named { workers, one })))
    }

    /// Each item's encoder, in the items' order, read at the place the
    /// decision names in the host's encoder list. One worker where the
    /// items were placed as one. Empty for a request with no item.
    #[must_use]
    pub fn workers(&self) -> &[Arc<dyn Worker>] {
        self.0.as_deref().map_or(&[], |named| &named.workers)
    }

    /// Whether the items were placed as one, on the one encoder
    /// ([`EncodePlaced::one`](crate::engine::EncodePlaced::one)).
    #[must_use]
    pub fn one(&self) -> bool {
        self.0.as_deref().is_some_and(|named| named.one)
    }
}

impl Drop for Encoders {
    /// One test where no encoder is named, which is the end of every
    /// placement but an encode request's: what is named is released out of
    /// line.
    #[inline]
    fn drop(&mut self) {
        if let Some(named) = self.0.take() {
            release(named);
        }
    }
}

/// Releases an encode request's named encoders. It stands out of line, so
/// that the end of a placement that names none holds one test and no call.
#[inline(never)]
fn release(named: Box<Named>) {
    drop(named);
}

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
    /// An encode request's encoders ([`Encoders`]). It names none for every
    /// other request, and allocates nothing then. A host that splits the
    /// placement into its sides ([`Placement::sides`]) reads the encoders
    /// first: a side names none, and they stay in the placement the sides
    /// share until the request ends.
    pub encoders: Encoders,
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
    /// The label of the profile that picks a pair's decode side: the
    /// binding's decode profile, and its one profile where it names no
    /// other. What a host counts the decode side under. `None` for a single
    /// request. Neither constructor reads one: the call that places a pair
    /// writes it, under the document it decided with.
    pub decode_profile: Option<&'static str>,
    /// The pick's lease: a pair's always, a single request's where a stage
    /// of the document takes a word of a request. `None` for an answer of
    /// which nothing is owed.
    pub(super) held: Option<DecisionGuard<&'static PolicyEngine>>,
}

impl Placement<'_> {
    /// The placement of a place
    /// ([`AlonePick`](crate::engine::decision_engine::AlonePick)): `worker`
    /// is the worker at the pick's place in the list its host handed, and
    /// `profile` the label of the pick's profile, which the call that
    /// decided read ([`Loaded::label_at`]). Nothing is owed of it, so
    /// it holds no lease, takes no id and reads no clock. The one call for
    /// a request makes it ([`PolicyEngine::place`]).
    #[must_use]
    pub(crate) fn alone(worker: Arc<dyn Worker>, profile: &'static str) -> Self {
        Self {
            worker,
            decode: None,
            encoders: Encoders::default(),
            routing_key: None,
            profile,
            decode_profile: None,
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
        engine: Loaded<'static, '_>,
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
        let mut held = leased
            .then(|| DecisionGuard::holding(engine.object, GatewayId::LOCAL, decided, later, now));
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
            encoders: Encoders::default(),
            routing_key: None,
            profile,
            decode_profile: None,
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
            encoders: self.encoders,
            routing_key: None,
            profile: self.profile,
            decode_profile: self.decode_profile,
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

    /// Decides a pair's decode again, behind its prefill's answer, for a
    /// pair whose first decision booked the prefill alone
    /// ([`PlaceRequest::decode_again`] under a document that picks the
    /// decode after the prefill). The engine decides the decode on the
    /// records of this instant, behind the prefill that was sent, and the
    /// first of the decodes it names that can be sent to is taken: ready,
    /// and let through by its circuit breaker. [`Placement::decode`] is
    /// then that worker, and [`Placement::decode_id`] follows. The prefill
    /// never moves.
    ///
    /// `request` states the request again, as its first call stated it: the
    /// same model, and [`PlaceRequest::decode_again`]. The engine makes the
    /// request of it anew, as it makes a first call's, and decides under
    /// the binding of the model it states. Every fact of the decode's
    /// decision is this request's: the pair's two lists as their caller
    /// holds them now, with their pair classes, the request's size, its
    /// routing value and the workers it tried among them. The prefill is
    /// found again in its list by its id. [`Placement::routing_key`],
    /// [`Placement::profile`] and [`Placement::decode_profile`] stay as
    /// the first call wrote them.
    ///
    /// **The order.** The call is made once the prefill's send was said
    /// ([`Placement::sent`]) and before the prefill's end is said
    /// ([`Placement::ended`]). A caller asks it, or says the send of the
    /// first decision's decode, before it says the prefill's end: a decode
    /// that still stands chosen at that end leaves with it. Behind a
    /// decode this call took, the prefill's end is said when the prefill
    /// answers and the decode's send at its own later instant.
    ///
    /// **A panic.** A route's own rule ([`PlaceRequest::admits`]) that
    /// panics as it is asked of a decode passes its panic to this call's
    /// caller. The decode side the call issued is given up by name first,
    /// and the placement stands as it stood before the call. A stage that
    /// panics inside the decode's decision reaches no caller: the engine
    /// holds the panic, counts it and refuses the call
    /// ([`DecisionRefused::Faulted`]), and nothing was issued.
    ///
    /// # Errors
    ///
    /// Why no decode was taken. The placement stands as it stood before
    /// the call.
    ///
    /// - [`Unplaced::Shed`]: the decode's own decision placed nobody, under
    ///   the reason and the stage it names. [`Unplaced::Queued`]: it queued
    ///   the request. The first decision's decode still stands chosen and
    ///   is booked nowhere: the caller may send it, or answer its client.
    /// - [`Unplaced::NoWorkers`]: the decision named decodes and none can
    ///   be sent to now. The decode side the call issued is given up at
    ///   once, and the stages that take a release hear it. The first
    ///   decision's decode stands chosen as before the call, and the
    ///   caller may also ask again, stating what it tried.
    /// - [`Unplaced::Refused`] with [`DecisionRefused::Decode`]: the call
    ///   came out of turn, and no decision was made. The guard refuses by
    ///   name, in this order ([`DecisionGuard::decide_decode`]). A single
    ///   request has no decode side ([`DecodeRefused::NotChosen`]). A pair
    ///   neither of whose sides is owed anything is gone: both ended, or
    ///   the prefill's end was said with the decode still chosen
    ///   ([`DecodeRefused::NoLease`]). A decode side that ended beside a
    ///   prefill still owed, that a call took, or whose first decode's
    ///   send was said, stands ([`DecodeRefused::DecodeStands`]). A decode
    ///   booked with its prefill never stood chosen
    ///   ([`DecodeRefused::NotChosen`]). A decode that stands chosen is
    ///   asked once its prefill's send was said
    ///   ([`DecodeRefused::PrefillNotHeld`]).
    /// - [`Unplaced::Refused`] with any other reason: the engine made no
    ///   decision, as at a first call.
    pub fn decode_again(&mut self, request: &PlaceRequest<'_>) -> Result<(), Unplaced> {
        self.decode_taken(request).map(drop)
    }

    /// [`Self::decode_again`], answering the decode it took and wrote: what
    /// a pair's decode side hands its host.
    pub(super) fn decode_taken(
        &mut self,
        request: &PlaceRequest<'_>,
    ) -> Result<Arc<dyn Worker>, Unplaced> {
        // An answer of which nothing is owed is a single request's.
        let Some(held) = &mut self.held else {
            let refused = DecisionRefused::Decode(DecodeRefused::NotChosen);
            return Err(Unplaced::Refused(refused));
        };
        let decode = PolicyEngine::decode_placed(held, request)?;
        self.decode = Some(Arc::clone(&decode));
        Ok(decode)
    }

    /// One side of the request went out: its dispatch, said once
    /// ([`DecisionGuard::dispatched`], and
    /// [`DecisionGuard::dispatched_decode`] for a pair's decode). No lease
    /// stands for an encode side, so its word says nothing. A single
    /// request's lease goes out under any other side its host names, as
    /// [`Self::ended`] ends it under any.
    ///
    /// A pair's decode that stood chosen when its prefill's end was said
    /// left with that end ([`Placement::decode_again`], the order): its
    /// send then says nothing and the decode runs unbooked. A debug build
    /// stops on a decode's send said behind the end of both sides of a
    /// pair that asked its decode later.
    pub fn sent(&mut self, side: Side) {
        let Some(held) = &mut self.held else {
            return;
        };
        let pair = held.target().pair.is_some();
        match side {
            Side::Encode => {}
            Side::Decode if pair => {
                debug_assert!(
                    !held.later_ended(),
                    "a pair's decode is sent behind the end of both its sides: where it stood \
                     chosen, its send or its second ask is said before the prefill's end"
                );
                held.dispatched_decode(Instant::now());
            }
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
        self.end(side, outcome, Instant::now);
    }

    /// One side of the request ended with the `outcome` a handle on that
    /// side's end reported at `at` ([`SideEnd`](super::SideEnd)):
    /// [`Self::ended`], said under that instant. The placement writes one
    /// thing more into the outcome, whatever the handle's holder wrote
    /// there: the span from the side's dispatch to the report, as an end
    /// with no outcome states the span to its side's drop. Zero for a side
    /// whose send was never said.
    pub(super) fn reported(&mut self, side: Side, outcome: Outcome, at: Instant) {
        let Some(held) = &self.held else {
            return;
        };
        let outcome = Outcome {
            duration_micros: held.since_sent(side, at),
            ..outcome
        };
        self.end(side, outcome, || at);
    }

    /// The one way an outcome reaches the lease: [`Self::ended`] and
    /// [`Self::reported`] by it. The clock is asked for `at` where the end
    /// is said, and not for an end that says nothing.
    fn end(&mut self, side: Side, outcome: Outcome, at: impl FnOnce() -> Instant) {
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
        held.completed(&outcome, at());
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
