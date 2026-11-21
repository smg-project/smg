//! The decision guard: a decision as its host holds it, from the answer to
//! the end of the request.
//!
//! A host asks its door for a decision ([`DecisionGuard::decide`]) or, with the
//! `client` feature, awaits one over a link (`DecisionGuard::asked`). It keeps
//! the handle while it commits a target, sends the request and reads its end.
//! Each report goes through the door ([`DecisionEngine::say`]) or onto
//! the link's session. The thread that says it is never held for a decision.
//! Whoever applies a report tells the engine that issued the lease.
//!
//! **Once.** A lease is reported once on every path. A handle knows what
//! it still owes its lease: a dispatch, a completion, or nothing. A report
//! that is not owed says nothing, and a handle dropped with a report owed
//! says it.
//!
//! **Before a dispatch** a lease ends by a release, under its host's
//! reason: the commit refused every target
//! ([`ReleaseReason::CommitFailed`]), or the client went away
//! ([`ReleaseReason::ClientGone`]). A handle dropped there knows no
//! reason: a client left, a timeout fired or an error path returned
//! early. It gives the lease up as abandoned
//! ([`ReleaseReason::Abandoned`]), the word for a lease that ends before
//! a dispatch with no cause given. A host that knows its client left
//! says so before it lets the handle go.
//!
//! **After a dispatch** a lease lives until its completion. A client that
//! leaves then is a completion of class cancelled
//! ([`TransportClass::Cancelled`]), and so are a release said then and a
//! handle dropped with no outcome.
//!
//! **The instants.** A dispatch, a completion and a release each take the
//! instant their host read, and a span is counted between two of them.
//! Reports over a link carry spans, and the controller records when it
//! reads them. A handle dropped with a report owed reads the clock for it.
//! A host can supply its own instant by reporting before drop.
//!
//! **The targets not taken.** A pick's lease books one of its targets,
//! and a dispatch that names another moves the booking there
//! ([`DecisionGuard::take`]). A target the host passes over holds nothing on
//! file: its release is the commit itself, and no report.
//!
//! A decision that picks no worker has no lease, and its handle owes
//! nothing.
//!
//! **A pair.** A pair's prefill and decode share a lease id, and the
//! handle reports each side. Before a dispatch, [`DecisionGuard::take`] takes a
//! target for both sides. After the prefill's dispatch,
//! [`DecisionGuard::take_decode`] can take another decode of that prefill.
//! [`DecisionGuard::dispatched`] says the prefill's dispatch, and
//! [`DecisionGuard::dispatched_decode`] the decode's. A completion names the
//! side that ended. A release or a drop gives up both sides in one word
//! while both are issued and still owe dispatch. A dispatched side
//! completes as cancelled, and an issued side is released by name.
//! [`DecisionGuard::released_decode`] gives up the decode alone.
//!
//! When the request sets [`Constraints::decode_later`](crate::engine::view::Constraints::decode_later),
//! its first pick names the decode targets and books the prefill alone.
//! After reporting the prefill dispatch and before reporting its end,
//! the host can call [`DecisionGuard::decide_decode`] for a fresh pick or send
//! a decode from the first pick. A decode still chosen leaves with its prefill.
//! That holds for a handle over a door in process. A handle over a link sends
//! the field as false: the controller issues both sides at the first ask, and
//! [`DecisionGuard::decide_decode`] answers that the side was not chosen.

use std::{
    fmt,
    ops::Deref,
    time::{Duration, Instant},
};

#[cfg(feature = "client")]
use crate::service::client::{wire, DecisionUnavailable, Error, Link};
use crate::{
    engine::{
        decision::{Decision, Target},
        decision_engine::{DecisionEngine, HeldDecision, RequestMaker},
        framework::{EncodeItems, EncodePlaced},
        store_backed_engine::{DecideOutcome, DecisionRefused, Leased, Request},
        view::WorkerId,
    },
    store::{DecodeRefused, Dispatch, GatewayId, Said},
    tracking::lease::{LeaseId, Outcome, ReleaseReason, Side, TransportClass},
};

/// Where a handle's decision came from, and where its reports go.
enum Arm<D> {
    /// The engine of this process, and the gateway the decision was made for.
    Engine { door: D, gateway: GatewayId },
    /// A controller, over the host's link. The link rides behind a
    /// pointer: a host embeds the handle in each request's context, and a
    /// handle of its door is the size it has in a build with no link.
    #[cfg(feature = "client")]
    Link(Box<Link>),
}

/// What a handle still owes its lease.
#[derive(Clone, Copy, Debug)]
enum Owed {
    /// Nothing: the lease's end was said, or the decision picked no
    /// worker.
    Nothing,
    /// A chosen decode with no booking. Its original pick can still dispatch.
    Later { lease: LeaseId, decided: Instant },
    /// A dispatch, or a release in its place: the lease stands as issued
    /// since `decided`, the decision's instant.
    Dispatch {
        lease: LeaseId,
        side: Side,
        decided: Instant,
    },
    /// A completion: the lease was dispatched to `worker` at `sent`.
    Completion {
        lease: LeaseId,
        side: Side,
        sent: Instant,
        worker: WorkerId,
    },
}

impl Owed {
    fn waits(self) -> bool {
        matches!(self, Self::Dispatch { .. } | Self::Later { .. })
    }

    /// A side that stood chosen, once its dispatch is owed: the span its
    /// dispatch says counts from the first decision. The side's lease
    /// keeps that decision's issue instant, one record a lease, and a
    /// stage dates a dispatch by the two together. Any other state stands.
    fn sent_later(self) -> Self {
        match self {
            Self::Later { lease, decided } => Self::Dispatch {
                lease,
                side: Side::Decode,
                decided,
            },
            owed => owed,
        }
    }
}

/// A span in whole microseconds, saturating.
fn micros(span: Duration) -> u32 {
    u32::try_from(span.as_micros()).unwrap_or(u32::MAX)
}

/// One decision held by its host from the answer to the end of the request.
///
/// `D` is how the host holds its door: a borrow where the handle stays in
/// its caller's frame, a shared pointer where it travels with the request.
/// A handle that asks over a link has the same type and keeps the link
/// for its reports and for its drop.
pub struct DecisionGuard<D: Deref<Target: DecisionEngine>> {
    arm: Arm<D>,
    decided: DecideOutcome,
    /// The index to name at dispatch: the first until the host takes
    /// another. After the prefill dispatch it can move for the decode.
    index: u8,
    /// The first decision stated the later decode: its decode side stood
    /// chosen. A decode that owes a dispatch or an end was then picked by
    /// a decode call, and was issued with its prefill otherwise.
    later: bool,
    /// The target at that index.
    target: Target,
    owed: Owed,
    /// What the handle still owes the decode side's lease of a pair.
    /// Nothing for a single.
    decode: Owed,
}

impl<D: Deref<Target: DecisionEngine>> DecisionGuard<D> {
    /// One decision of `door` for `gateway`'s `request` at `now`, on the
    /// calling thread ([`DecisionEngine::decide`]), held.
    ///
    /// # Errors
    ///
    /// [`DecisionRefused`]: the door made no decision for the request.
    pub fn decide(
        door: D,
        gateway: GatewayId,
        request: &Request<'_>,
        now: Instant,
    ) -> Result<Self, DecisionRefused> {
        let decided = door.decide(gateway, request, now)?;
        let later = request.constraints.decode_later;
        Ok(Self::hold(
            Arm::Engine { door, gateway },
            decided,
            later,
            now,
        ))
    }

    /// [`DecisionGuard::decide`] for a request its host makes from the world
    /// the door decides under ([`DecisionEngine::decide_under`]).
    ///
    /// # Errors
    ///
    /// [`DecisionRefused`]: the door made no decision for the request.
    pub fn decide_under(
        door: D,
        gateway: GatewayId,
        maker: &impl RequestMaker,
        now: Instant,
    ) -> Result<Self, DecisionRefused> {
        let (decided, later) = door.decide_under(gateway, maker, now)?;
        Ok(Self::hold(
            Arm::Engine { door, gateway },
            decided,
            later,
            now,
        ))
    }

    /// Holds the decision made with the request's encode items
    /// ([`DecisionEngine::decide_encode_pair`]). For a pair, the handle
    /// reports the prefill and decode.
    ///
    /// # Errors
    ///
    /// [`DecisionRefused`]: the door made no decision for the request.
    pub fn decide_encode_pair(
        door: D,
        gateway: GatewayId,
        request: &Request<'_>,
        items: EncodeItems<'_>,
        placed: &mut EncodePlaced,
        now: Instant,
    ) -> Result<Self, DecisionRefused> {
        let decided = door.decide_encode_pair(gateway, request, items, placed, now)?;
        let later = request.constraints.decode_later;
        Ok(Self::hold(
            Arm::Engine { door, gateway },
            decided,
            later,
            now,
        ))
    }

    fn hold(arm: Arm<D>, decided: DecideOutcome, later: bool, now: Instant) -> Self {
        let first = decided
            .decision
            .pick()
            .and_then(|pick| pick.targets.first());
        let target = first.copied().unwrap_or_default();
        let owed = decided
            .lease
            .map_or(Owed::Nothing, |leased| Owed::Dispatch {
                lease: leased.lease.id,
                side: leased.lease.side,
                decided: now,
            });
        // A pair's decode side owes a dispatch of its own under the lease's
        // id, or stands chosen where the request sends its decode later.
        let decode = if let (true, Owed::Dispatch { lease, .. }) = (decided.pair(), owed) {
            if later {
                Owed::Later {
                    lease,
                    decided: now,
                }
            } else {
                Owed::Dispatch {
                    lease,
                    side: Side::Decode,
                    decided: now,
                }
            }
        } else {
            Owed::Nothing
        };
        Self {
            arm,
            decided,
            index: 0,
            later,
            target,
            owed,
            decode,
        }
    }

    /// The handle's view of its decision, as it lends it to its engine
    /// with a word and with a decode call: read before the word moves the
    /// handle. `None` for a decision that picked no worker.
    fn held(&self) -> Option<HeldDecision<'_>> {
        let leased = self.decided.lease.as_ref()?;
        Some(HeldDecision {
            decision: &self.decided.decision,
            lease: &leased.lease,
            decode_session: self.decided.decode_session,
            index: self.index,
            target: &self.target,
            decode_chosen: matches!(self.decode, Owed::Later { .. }),
        })
    }

    /// The engine's decision.
    #[must_use]
    pub fn decision(&self) -> &Decision {
        &self.decided.decision
    }

    /// Decides the pair's decode side while its prefill is dispatched and
    /// held. A pick replaces the held targets and takes the new first
    /// target. If the door sheds, the handle answers the decision it holds,
    /// the first pick's: this answer does not tell a shed from a pick. The
    /// lease's `due` does ([`Self::lease`]): a pick moves it to the call's
    /// deadline and a shed leaves the first.
    ///
    /// The decode's dispatch still says its span since the first decision:
    /// the side's lease keeps that decision's issue instant.
    ///
    /// # Errors
    ///
    /// [`DecisionRefused`] gives the reason the call made no decision. The
    /// handle keeps its current pick. The handle refuses by name, and asks
    /// no door, all that a file of leases would refuse: a single's lease
    /// and a decode issued with its prefill (not chosen), a decode side
    /// that a call picked already, that ended or that the host gave up
    /// ([`Self::released_decode`]) (the decode stands), a prefill not yet
    /// dispatched (the prefill is not held), and a pair whose two sides
    /// are owed nothing (a lease that is gone). A handle over a link is
    /// refused as not chosen: no link asks a decode later.
    pub fn decide_decode(
        &mut self,
        request: &Request<'_>,
        now: Instant,
    ) -> Result<&Decision, DecisionRefused> {
        self.decode_decided(|door, gateway, held| door.decide_decode(gateway, request, held, now))
    }

    /// [`Self::decide_decode`] for a request its host makes from the world
    /// the call decides under ([`DecisionEngine::decide_decode_under`]).
    ///
    /// # Errors
    ///
    /// As [`Self::decide_decode`].
    pub fn decide_decode_under(
        &mut self,
        maker: &impl RequestMaker,
        now: Instant,
    ) -> Result<&Decision, DecisionRefused> {
        self.decode_decided(|door, gateway, held| {
            door.decide_decode_under(gateway, maker, held, now)
        })
    }

    /// Takes the answer `call` gets of the handle's door for the decode
    /// side of its pair's lease.
    fn decode_decided(
        &mut self,
        call: impl FnOnce(
            &D::Target,
            GatewayId,
            &HeldDecision<'_>,
        ) -> Result<DecideOutcome, DecisionRefused>,
    ) -> Result<&Decision, DecisionRefused> {
        let refused = |refused| Err(DecisionRefused::Decode(refused));
        if self.decided.lease.is_none() {
            return refused(DecodeRefused::NoLease);
        }
        // The handle's own word comes first: it is the note of the pair
        // where its engine keeps no file of leases, so it answers all that
        // a file answers of a decode call, in the file's order. A single's
        // lease has no decode side. A pair whose two sides are owed
        // nothing has left the file. A decode side owed nothing beside a
        // prefill still owed has ended, or was given up while it stood
        // chosen: that one was owed no report, so a file still holds it
        // chosen and would issue it, a booking nobody reports again. A
        // decode that owes a dispatch or an end does not stand chosen: a
        // decode call picked it already, or it was issued with its
        // prefill. A decode that stands chosen is decided once its prefill
        // is held: dispatched, and not ended. A file also counts a side's
        // issues and refuses a call past the count's end. A handle sees
        // one issue of a side at the most, and leaves that count to the
        // file.
        match (self.decided.pair(), self.owed, self.decode) {
            (false, ..) => return refused(DecodeRefused::NotChosen),
            (true, Owed::Nothing, Owed::Nothing | Owed::Later { .. }) => {
                return refused(DecodeRefused::NoLease);
            }
            (true, _, Owed::Nothing) => return refused(DecodeRefused::DecodeStands),
            (true, _, Owed::Dispatch { .. } | Owed::Completion { .. }) => {
                return refused(if self.later {
                    DecodeRefused::DecodeStands
                } else {
                    DecodeRefused::NotChosen
                });
            }
            (true, Owed::Dispatch { .. } | Owed::Later { .. }, Owed::Later { .. }) => {
                return refused(DecodeRefused::PrefillNotHeld);
            }
            (true, Owed::Completion { .. }, Owed::Later { .. }) => {}
        }
        let answer = match (&self.arm, self.held()) {
            (Arm::Engine { door, gateway }, Some(held)) => call(door, *gateway, &held)?,
            // A controller over a link issued both sides at the first ask.
            #[cfg(feature = "client")]
            (Arm::Link(_), _) => return refused(DecodeRefused::NotChosen),
            (_, None) => return refused(DecodeRefused::NoLease),
        };
        let first = answer.decision.pick().and_then(|pick| pick.targets.first());
        if let Some((leased, target)) = answer.lease.zip(first) {
            self.decided.decision = answer.decision;
            self.decided.decode_session = answer.decode_session;
            (self.index, self.target) = (0, *target);
            self.decode = self.decode.sent_later();
            if let Some(held) = &mut self.decided.lease {
                held.due = leased.due;
            }
        }
        Ok(self.decision())
    }

    /// Whether the pair's decode side stands chosen: its request stated a
    /// later decode ([`Constraints::decode_later`](crate::engine::view::Constraints::decode_later)),
    /// the first pick named the side and booked nothing for it, and
    /// [`Self::decide_decode`] may pick it again. `false` from that call's
    /// pick, from the side's dispatch, from the end of its prefill, and
    /// for every other handle.
    #[must_use]
    pub fn decode_chosen(&self) -> bool {
        matches!(self.decode, Owed::Later { .. })
    }

    /// The lease filed for a pick, the prefill side for a pair.
    /// `None` for a decision that picks no worker. A lease has one dispatch
    /// deadline: behind a decode call that picked, `due` is the call's.
    #[must_use]
    pub fn lease(&self) -> Option<&Leased> {
        self.decided.lease.as_ref()
    }

    /// Takes the target at `index` of the pick, best first: the one the
    /// host's commit holds, which its dispatch then names. A host that
    /// takes none dispatches to the first. A take says nothing to the
    /// store: the lease holds credit on the target it booked until the
    /// dispatch names the index taken and the file moves the record.
    /// Nothing stands on a target not taken, so there is nothing
    /// to release. `None` where the pick has no target at `index`, and
    /// where no dispatch is owed any more: what was taken before stands.
    /// Before any dispatch a pair takes the target for both sides. Where
    /// its decode went out first, the prefill stays the one the decode
    /// was sent for: a target of another prefill is not taken. After the
    /// prefill's dispatch, see [`Self::take_decode`].
    pub fn take(&mut self, index: usize) -> Option<Target> {
        let Owed::Dispatch { .. } = self.owed else {
            return None;
        };
        let (taken, target) = self.target_at(index)?;
        // A decode that no longer waits names its prefill: the prefill
        // goes there.
        let settled = self.decided.pair() && !self.decode.waits();
        if settled && target.id != self.target.id {
            return None;
        }
        (self.index, self.target) = (taken, target);
        Some(target)
    }

    /// Takes the decode of target `index` after the prefill dispatch.
    /// The target must name the prefill already sent. The decode's
    /// dispatch books that target's decode worker.
    /// `None` if the target cannot be taken or the decode owes no dispatch.
    pub fn take_decode(&mut self, index: usize) -> Option<Target> {
        if matches!(self.owed, Owed::Dispatch { .. }) || !self.decode.waits() {
            return None;
        }
        let (taken, target) = self.target_at(index)?;
        if target.id != self.target.id {
            return None;
        }
        (self.index, self.target) = (taken, target);
        Some(target)
    }

    fn target_at(&self, index: usize) -> Option<(u8, Target)> {
        let taken = u8::try_from(index).ok()?;
        let pick = self.decided.decision.pick()?;
        let target = *pick.targets.as_slice().get(index)?;
        Some((taken, target))
    }

    /// Reports what the host says of `lease` where the decision was made.
    /// The door reads the side from what is said. A frame on the link
    /// names it.
    #[inline(always)]
    #[cfg_attr(
        not(feature = "client"),
        expect(unused_variables, reason = "the link's frame alone names the side")
    )]
    fn say(&self, lease: LeaseId, side: Side, said: Said, now: Instant) {
        match &self.arm {
            // The engine is lent the handle's view of the decision, as it
            // stands before the word moves the handle.
            Arm::Engine { door, gateway } => {
                if let Some(held) = self.held() {
                    debug_assert_eq!(held.lease.id, lease, "a handle says of its own lease");
                    door.say(*gateway, &held, said, now);
                }
            }
            #[cfg(feature = "client")]
            Arm::Link(link) => {
                // With no session standing the report waits in the link's line.
                link.queue_frame(wire::frame(lease, side, said));
            }
        }
    }

    /// The host sent the request to the target it took, at `now`: the
    /// lease is dispatched on that target, and no dispatch deadline ends
    /// it from here. A dispatch said after that deadline already ended
    /// the lease meets no lease: the request runs unbooked, and its host
    /// is told nothing. Said once: a second call says nothing.
    /// For a pair this says the prefill's dispatch.
    pub fn dispatched(&mut self, now: Instant) {
        self.owed = self.dispatch(self.owed, self.target.id, now);
    }

    /// The host sent the decode to the target it took, at `now`. Said
    /// once, for a pair's decode side alone. A chosen side books its
    /// decode at this dispatch.
    pub fn dispatched_decode(&mut self, now: Instant) {
        if let Some(pair) = self.target.pair {
            self.decode = self.dispatch(self.decode.sent_later(), pair.decode_id, now);
        }
    }

    /// Says a dispatch and keeps the worker it named for a cancelled end.
    fn dispatch(&self, owed: Owed, worker: WorkerId, now: Instant) -> Owed {
        let Owed::Dispatch {
            lease,
            side,
            decided,
        } = owed
        else {
            return owed;
        };
        let dispatch = Dispatch {
            target_index: self.index,
            side,
            since_decision_micros: micros(now.saturating_duration_since(decided)),
        };
        self.say(lease, side, Said::Dispatched(dispatch, now), now);
        Owed::Completion {
            lease,
            side,
            sent: now,
            worker,
        }
    }

    /// The request ended with `outcome`, as its host read it at `now`: the
    /// lease ends, and the completion is written on the served worker for
    /// the health precedence by the door or controller that decided it.
    /// A completion said with no dispatch says the dispatch first, at
    /// `now`: a request whose send failed was sent, and the engine hears a
    /// lease's events in their order. That dispatch carries the
    /// completion's instant, so a stage that times a request from its
    /// dispatch reads a span of zero for it. On a pair's handle a decode
    /// outcome ends the decode side. An encode outcome is said whole
    /// there and ends no side, since the lease holds none such: the
    /// served worker's health takes it, and what the handle owes stands.
    /// Any other outcome ends the handle's own side, and a single's lease
    /// ends under any side its host names. A side's end is said once. An
    /// encode outcome on a pair's handle is said at each call.
    pub fn completed(&mut self, outcome: &Outcome, now: Instant) {
        let pair = self.decided.pair();
        let owed = match outcome.side {
            Side::Decode if pair => {
                self.dispatched_decode(now);
                &mut self.decode
            }
            // A side the pair's lease does not hold: the completion is said
            // whole, as a served controller says such a frame. The file ends
            // no side of the lease, and the served worker's health takes it.
            Side::Encode if pair => {
                if let Some(leased) = self.decided.lease {
                    self.say(
                        leased.lease.id,
                        outcome.side,
                        Said::Completed(*outcome),
                        now,
                    );
                }
                return;
            }
            Side::Single | Side::Prefill | Side::Decode | Side::Encode => {
                self.dispatched(now);
                &mut self.owed
            }
        };
        let Owed::Completion { lease, side, .. } = *owed else {
            return;
        };
        *owed = Owed::Nothing;
        let outcome = Outcome {
            side: if side == Side::Prefill {
                side
            } else {
                outcome.side
            },
            ..*outcome
        };
        self.say(lease, side, Said::Completed(outcome), now);
        if side == Side::Prefill && matches!(self.decode, Owed::Later { .. }) {
            self.decode = Owed::Nothing;
        }
    }

    /// Gives up every side at `now`, under `reason`, in one release if
    /// both sides of a pair are issued and still owe dispatch. A dispatched
    /// side completes as cancelled on the worker it was sent to, with the
    /// span from its dispatch to `now`. An issued side is released.
    pub fn released(&mut self, reason: ReleaseReason, now: Instant) {
        if let (Owed::Dispatch { lease, side, .. }, Owed::Dispatch { .. }) =
            (self.owed, self.decode)
        {
            self.say(lease, side, Said::Released(None, reason), now);
            self.owed = Owed::Nothing;
            self.decode = Owed::Nothing;
            return;
        }
        self.owed = self.release(self.owed, reason, now);
        self.released_decode(reason, now);
    }

    /// Gives up the decode side alone. A sent decode completes as
    /// cancelled. An issued decode is released by name under `reason`.
    /// A chosen decode has no booking and needs no report: the store still
    /// holds it chosen, and the handle takes no decode call for it from
    /// here ([`Self::decide_decode`]).
    pub fn released_decode(&mut self, reason: ReleaseReason, now: Instant) {
        self.decode = self.release(self.decode, reason, now);
    }

    fn release(&self, owed: Owed, reason: ReleaseReason, now: Instant) -> Owed {
        match owed {
            Owed::Nothing | Owed::Later { .. } => {}
            Owed::Dispatch { lease, side, .. } => {
                let released = (side != Side::Single).then_some(side);
                self.say(lease, side, Said::Released(released, reason), now);
            }
            Owed::Completion {
                lease,
                side,
                sent,
                worker,
            } => {
                let cancelled = Outcome {
                    side,
                    served: worker,
                    status: 0,
                    class: TransportClass::Cancelled,
                    ttft_micros: 0,
                    tokens_in: 0,
                    tokens_out: 0,
                    duration_micros: micros(now.saturating_duration_since(sent)),
                };
                self.say(lease, side, Said::Completed(cancelled), now);
            }
        }
        Owed::Nothing
    }
}

#[cfg(feature = "client")]
impl<D: Deref<Target: DecisionEngine>> DecisionGuard<D> {
    /// One decision of the controller behind `link` for `request`, asked at
    /// `now` and held. The one call of a handle that waits.
    ///
    /// The request's `budget_micros` gives the controller time to decide.
    /// The link waits up to `budget` for the answer ([`Link::ask`]). The
    /// handle counts time to dispatch and the lease's span from `now`.
    /// A linked target names its worker and rank. Its position in the
    /// controller's candidate slice stays with the controller.
    ///
    /// # Errors
    ///
    /// [`DecisionUnavailable`] if the link has no answer or the handle cannot
    /// read it. The link releases the leases of an unread answer as late
    /// decisions.
    ///
    /// A request whose `budget_micros` is 0 is answered
    /// [`DecisionUnavailable::Late`] before anything is sent. Over a link that
    /// budget says the request's time is spent, and the controller would
    /// answer busy at its receipt. A request built for the door in process
    /// carries that budget unless its host states one.
    ///
    /// A request that states a hard demand is answered
    /// [`DecisionUnavailable::Unsent`] with [`Error::OutOfRange`] before
    /// anything is sent. This protocol version states no demand, and a
    /// controller that reads none would pick among workers the demand rules
    /// out. Its host decides such a request itself.
    pub async fn asked(
        link: &Link,
        request: &Request<'_>,
        now: Instant,
        budget: Duration,
    ) -> Result<Self, DecisionUnavailable> {
        if request.facts.budget_micros == 0 {
            return Err(DecisionUnavailable::Late);
        }
        if !request.constraints.demands.is_none() {
            return Err(DecisionUnavailable::Unsent(Error::OutOfRange(
                "constraints.demands",
            )));
        }
        let (asked, left_out) = wire::request(request);
        let told = link.ask_cut(asked, left_out, budget).await?;
        match wire::decided(&told, now) {
            // No link's request asks its decode later: both sides are issued.
            Ok(decided) => Ok(Self::hold(
                Arm::Link(Box::new(link.clone())),
                decided,
                false,
                now,
            )),
            Err(unread) => {
                link.release_late(told);
                Err(DecisionUnavailable::Unsent(Error::NoDecision(
                    unread.to_string(),
                )))
            }
        }
    }
}

impl<D: Deref<Target: DecisionEngine>> Drop for DecisionGuard<D> {
    /// A handle dropped with a report owed speaks for a lease nobody
    /// spoke for, in the word of its phase that names no cause: a release
    /// as abandoned before any dispatch, a completion of class cancelled
    /// after one. A drop is handed no instant and reads the clock. A
    /// handle that owes nothing says nothing and reads no clock.
    fn drop(&mut self) {
        // The common drop: the request completed, on every side.
        if matches!((self.owed, self.decode), (Owed::Nothing, Owed::Nothing)) {
            return;
        }
        self.released(ReleaseReason::Abandoned, Instant::now());
    }
}

impl<D: Deref<Target: DecisionEngine>> fmt::Debug for DecisionGuard<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecisionGuard")
            .field("decided", &self.decided)
            .field("index", &self.index)
            .field("owed", &self.owed)
            .field("decode", &self.decode)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        cell::RefCell,
        error::Error,
        mem,
        sync::{mpsc, Arc, Barrier},
        thread,
        time::{Duration, Instant},
    };

    use parking_lot::Mutex;

    use super::DecisionGuard;
    use crate::{
        document::spec::AppliedSpec,
        engine::{
            decision::{Decision, ShadowKind, Target, Targets},
            decision_engine::{DecisionEngine, HeldDecision, RequestMaker},
            framework::{EncodeItems, EncodePlaced, Picker, TieBand},
            prompt_hashes::{HashSeeds, PromptHashes},
            store_backed_engine::{
                tests::{
                    counted, fleet, held as deciding, later_request, load_decodes, pair_counts,
                    pair_fleet, pair_request, picking, push_fleet, request, Ends, GATEWAY,
                    PATIENCE,
                },
                DecideOutcome, DecisionRefused, Request, StoreBackedEngine, SweepOutcome,
            },
            view::{Candidate, Identity, RequestFacts, WireModes, WorkerId},
        },
        policies::random::Random,
        stages::lifecycle::Lifecycle,
        store::{
            pairing::Refusal, DecodeRefused, Dispatch, GatewayId, HealthSink, LeaseState,
            LeasesStore, MemoryStore, Said, WorkersStore,
        },
        tracking::lease::{Lease, LeaseId, Outcome, ReleaseReason, Side, TransportClass},
    };

    /// What a stage heard of a lease.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Call {
        Issued,
        /// The target the dispatch named, its index on the lease told, and
        /// the microseconds since the decision.
        Dispatched(WorkerId, u8, u32),
        /// The served worker, the class and the microseconds the dispatch
        /// took.
        Completed(WorkerId, TransportClass, u32),
        Released(ReleaseReason),
    }

    type Heard = Arc<Mutex<Vec<Call>>>;

    /// The `random` picker, keeping what the engine tells it of leases.
    #[derive(Debug)]
    struct Hears(Heard);

    impl Lifecycle for Hears {
        fn on_decision_issued(&self, _: &Lease, _: &Decision) {
            self.0.lock().push(Call::Issued);
        }

        fn on_dispatched(&self, lease: &Lease, target: &Target, _: Side, since: u32) {
            let call = Call::Dispatched(target.id, lease.target_index, since);
            self.0.lock().push(call);
        }

        fn on_completed(&self, _: &Lease, outcome: &Outcome) {
            let call = Call::Completed(outcome.served, outcome.class, outcome.duration_micros);
            self.0.lock().push(call);
        }

        fn on_released(&self, _: &Lease, reason: ReleaseReason) {
            self.0.lock().push(Call::Released(reason));
        }
    }

    impl Picker for Hears {
        fn name(&self) -> &'static str {
            "hears"
        }

        fn pick(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            scores: &[f32],
            band: TieBand,
            out: &mut Targets,
        ) {
            Random.pick(facts, candidates, kept, scores, band, out);
        }
    }

    /// A door over the fleet's three ready members, whose picker keeps
    /// what the engine tells it of leases, and what it heard.
    fn bed() -> (StoreBackedEngine, Heard) {
        let store = fleet(1);
        let heard = Heard::default();
        let kept = Arc::clone(&heard);
        picking(&store, 2, "hears", move || {
            Box::new(Hears(Arc::clone(&kept)))
        });
        (StoreBackedEngine::new(store), heard)
    }

    /// One decision of `door` for the request `id`, held.
    fn held<'a>(
        door: &'a StoreBackedEngine,
        id: &[u8],
        now: Instant,
    ) -> DecisionGuard<&'a StoreBackedEngine> {
        let decided = DecisionGuard::decide(door, GATEWAY, &request(id, 7), now);
        decided.expect("a decision")
    }

    /// An outcome of `class` on `served`, as a host reads one.
    fn outcome(served: WorkerId, class: TransportClass) -> Outcome {
        Outcome {
            side: Side::Single,
            served,
            status: 200,
            class,
            ttft_micros: 1_000,
            tokens_in: 12,
            tokens_out: 34,
            duration_micros: 5_000,
        }
    }

    /// What was heard, with a cancelled completion's time read as 0: a
    /// clock made it, and it is `least` microseconds at least.
    fn clocked(heard: &Heard, least: u32) -> Vec<Call> {
        let heard = heard.lock();
        let calls = heard.iter().map(|call| match *call {
            Call::Completed(served, TransportClass::Cancelled, took) => {
                assert!(took >= least, "a cancelled request took {took}");
                Call::Completed(served, TransportClass::Cancelled, 0)
            }
            said => said,
        });
        calls.collect()
    }

    /// A host embeds the handle in the context it allocates once a request,
    /// and fits that context to its allocator's class. A handle is the size
    /// it has in a build with no link: the link's arm rides behind a
    /// pointer, and the door's own pointer tells the arms apart. A field
    /// that grows the handle grows each such context, so its hosts are
    /// read before this number moves.
    #[test]
    fn a_handle_is_the_size_it_has_in_a_build_with_no_link() {
        const HELD: usize = 488;
        assert_eq!(size_of::<DecisionGuard<Arc<StoreBackedEngine>>>(), HELD);
        assert_eq!(size_of::<DecisionGuard<&StoreBackedEngine>>(), HELD);
    }

    /// A request that is served: its lease is booked at the issue, moves
    /// to dispatched at the host's word and ends at its completion, which
    /// is written on the served worker. Each is told once: what the host
    /// says after the end, and the drop, say nothing.
    #[test]
    fn a_served_request_tells_its_dispatch_and_its_completion_once_each() {
        let (door, heard) = bed();
        let store = door.store();
        let now = Instant::now();
        let mut handle = held(&door, b"r-1", now);
        let lease = handle.lease().expect("a pick's lease").lease.id;
        let first = handle.take(0).expect("the pick's first target");
        assert_eq!(counted(store), [(first.id, 1)], "booked at the issue");
        let sent = now + Duration::from_micros(250);
        handle.dispatched(sent);
        let on_file = LeasesStore::get(&**store, GATEWAY, lease);
        let state = on_file.expect("the lease is on file").state;
        assert_eq!(state, LeaseState::Dispatched { at: sent });
        let ended = now + Duration::from_millis(5);
        handle.completed(&outcome(first.id, TransportClass::Ok), ended);
        assert_eq!(counted(store), []);
        assert_eq!(LeasesStore::live(&**store, GATEWAY), 0);
        assert!(store.completed_ok_since(GATEWAY, first.id, now));
        // Nothing more is owed: what its host says behind the end is told
        // to nobody and written nowhere.
        let later = ended + Duration::from_secs(1);
        handle.dispatched(later);
        handle.completed(&outcome(first.id, TransportClass::Ok), later);
        handle.released(ReleaseReason::ClientGone, Instant::now());
        drop(handle);
        let again = store.completed_ok_since(GATEWAY, first.id, later);
        assert!(!again, "one completion is written");
        let told = [
            Call::Issued,
            Call::Dispatched(first.id, 0, 250),
            Call::Completed(first.id, TransportClass::Ok, 5_000),
        ];
        assert_eq!(*heard.lock(), told);
    }

    /// A host whose commit passes over the first target takes another: a
    /// take says nothing to the store, and the dispatch moves the lease's
    /// booking to the target taken. A take of no target of the pick, and
    /// one behind the dispatch, change nothing.
    #[test]
    fn a_dispatch_on_an_alternate_moves_the_leases_booking_there() {
        let (door, heard) = bed();
        let store = door.store();
        let now = Instant::now();
        let mut handle = held(&door, b"r-1", now);
        let first = handle.take(0).expect("the pick's first target");
        let second = handle.take(1).expect("an alternate of the pick");
        assert_ne!(first.id, second.id);
        assert_eq!(handle.take(9), None, "no such target");
        assert_eq!(counted(store), [(first.id, 1)], "a take books nothing");
        // Two hours behind the decision: more microseconds than the span's
        // field holds, so it says its most.
        handle.dispatched(now + Duration::from_secs(7_200));
        assert_eq!(counted(store), [(second.id, 1)]);
        assert_eq!(handle.take(0), None, "the dispatch named its target");
        handle.completed(&outcome(second.id, TransportClass::Ok), now);
        assert_eq!(counted(store), []);
        let told = [
            Call::Issued,
            Call::Dispatched(second.id, 1, u32::MAX),
            Call::Completed(second.id, TransportClass::Ok, 5_000),
        ];
        assert_eq!(*heard.lock(), told);
    }

    /// A host whose commit refuses every target releases the lease under
    /// that reason, once: nothing it says behind the release is told.
    #[test]
    fn a_commit_that_refuses_every_target_releases_the_lease_once() {
        let (door, heard) = bed();
        let store = door.store();
        let now = Instant::now();
        let mut handle = held(&door, b"r-1", now);
        let first = handle.take(0).expect("the pick's first target");
        handle.released(ReleaseReason::CommitFailed, Instant::now());
        assert_eq!(counted(store), []);
        assert_eq!(LeasesStore::live(&**store, GATEWAY), 0);
        handle.released(ReleaseReason::ClientGone, Instant::now());
        handle.completed(&outcome(first.id, TransportClass::Ok), now);
        drop(handle);
        let written = store.completed_ok_since(GATEWAY, first.id, now);
        assert!(!written, "no completion is written behind a release");
        let told = [Call::Issued, Call::Released(ReleaseReason::CommitFailed)];
        assert_eq!(*heard.lock(), told);
    }

    /// A handle dropped with no call knows no reason for its drop: before
    /// any dispatch its lease is given up as abandoned, the word for a
    /// lease nobody spoke for.
    #[test]
    fn a_handle_dropped_with_no_call_gives_its_lease_up_as_abandoned() {
        let (door, heard) = bed();
        let store = door.store();
        let handle = held(&door, b"r-1", Instant::now());
        assert_eq!(LeasesStore::live(&**store, GATEWAY), 1);
        drop(handle);
        assert_eq!(counted(store), []);
        assert_eq!(LeasesStore::live(&**store, GATEWAY), 0);
        let told = [Call::Issued, Call::Released(ReleaseReason::Abandoned)];
        assert_eq!(*heard.lock(), told);
    }

    /// A host that knows its client left before any dispatch says so, and
    /// the handle's drop behind it says nothing more.
    #[test]
    fn a_host_that_knows_its_client_left_says_so_once() {
        let (door, heard) = bed();
        let store = door.store();
        let mut handle = held(&door, b"r-1", Instant::now());
        handle.released(ReleaseReason::ClientGone, Instant::now());
        assert_eq!(LeasesStore::live(&**store, GATEWAY), 0);
        drop(handle);
        let told = [Call::Issued, Call::Released(ReleaseReason::ClientGone)];
        assert_eq!(*heard.lock(), told);
    }

    /// After a dispatch a lease lives until its completion: a handle
    /// dropped with no outcome, and a release said there, are told as one
    /// completion of class cancelled on the dispatched target, which says
    /// the time since the dispatch. It is no completion of class ok for
    /// the health precedence.
    #[test]
    fn behind_a_dispatch_a_drop_and_a_release_are_a_cancelled_completion() {
        for released in [false, true] {
            let (door, heard) = bed();
            let store = door.store();
            let now = Instant::now();
            let mut handle = held(&door, b"r-1", now);
            let second = handle.take(1).expect("an alternate of the pick");
            handle.dispatched(now);
            thread::sleep(Duration::from_millis(1));
            if released {
                handle.released(ReleaseReason::ClientGone, Instant::now());
                assert_eq!(counted(store), [], "told where it was said");
            }
            drop(handle);
            assert_eq!(counted(store), []);
            assert_eq!(LeasesStore::live(&**store, GATEWAY), 0);
            assert!(!store.completed_ok_since(GATEWAY, second.id, now));
            let told = [
                Call::Issued,
                Call::Dispatched(second.id, 1, 0),
                Call::Completed(second.id, TransportClass::Cancelled, 0),
            ];
            assert_eq!(clocked(&heard, 1_000), told, "released: {released}");
        }
    }

    /// A request whose send failed was sent: a completion said with no
    /// dispatch says the dispatch first, on the target taken and at the
    /// completion's instant, so the engine hears the lease's events in
    /// their order.
    #[test]
    fn a_completion_said_with_no_dispatch_says_the_dispatch_first() {
        let (door, heard) = bed();
        let now = Instant::now();
        let mut handle = held(&door, b"r-1", now);
        let second = handle.take(1).expect("an alternate of the pick");
        let failed = outcome(second.id, TransportClass::ConnectFailed);
        handle.completed(&failed, now + Duration::from_micros(400));
        assert_eq!(counted(door.store()), []);
        let told = [
            Call::Issued,
            Call::Dispatched(second.id, 1, 400),
            Call::Completed(second.id, TransportClass::ConnectFailed, 5_000),
        ];
        assert_eq!(*heard.lock(), told);
    }

    /// A decision that picks no worker has no lease: its handle takes no
    /// target and tells nothing, whatever its host says.
    #[test]
    fn a_decision_that_picks_no_worker_owes_no_report() {
        let (door, heard) = bed();
        let store = door.store();
        let now = Instant::now();
        // A member of the fleet, by a decision that picks one: a
        // completion on a worker the store holds no row of is written
        // for nobody.
        let mut picked = held(&door, b"r-0", now);
        let member = picked.take(0).expect("the pick's first target").id;
        picked.released(ReleaseReason::CommitFailed, Instant::now());
        heard.lock().clear();
        let mut asked = request(b"r-1", 7);
        asked.facts.model = "model-none";
        let decided = DecisionGuard::decide(&door, GATEWAY, &asked, now);
        let mut handle = decided.expect("a decision");
        assert!(handle.decision().pick().is_none(), "no pool, no pick");
        assert!(handle.lease().is_none());
        assert_eq!(handle.take(0), None);
        handle.dispatched(now);
        handle.completed(&outcome(member, TransportClass::Ok), now);
        handle.released(ReleaseReason::CommitFailed, Instant::now());
        drop(handle);
        assert_eq!(LeasesStore::live(&**store, GATEWAY), 0);
        assert!(!store.completed_ok_since(GATEWAY, member, now));
        assert_eq!(*heard.lock(), []);
    }

    /// A host that takes no target dispatches to the pick's first, and a
    /// handle let go behind that dispatch completes the lease as cancelled
    /// on that target. The completion's span counts from the dispatch: one
    /// said ahead of the clock has no span yet.
    #[test]
    fn a_host_that_takes_no_target_dispatches_to_the_picks_first() {
        let (door, heard) = bed();
        let store = door.store();
        let now = Instant::now();
        let mut handle = held(&door, b"r-1", now);
        let pick = handle.decision().pick();
        let first = pick.and_then(|pick| pick.targets.first().copied());
        let first = first.expect("the pick's first target");
        handle.dispatched(now + Duration::from_secs(3_600));
        assert_eq!(counted(store), [(first.id, 1)]);
        thread::sleep(Duration::from_millis(1));
        drop(handle);
        assert_eq!(counted(store), []);
        let told = [
            Call::Issued,
            Call::Dispatched(first.id, 0, 3_600_000_000),
            Call::Completed(first.id, TransportClass::Cancelled, 0),
        ];
        assert_eq!(*heard.lock(), told);
    }

    /// A handle's report holds no thread for a decision. While one
    /// decision stands, more handles than a file's ring takes are let go:
    /// every one returns, and their leases stay on file, the deciding
    /// thread's to end. As that caller leaves, every lease is ended.
    #[test]
    fn handles_let_go_at_a_full_ring_hold_no_thread_for_a_decision() {
        let store = fleet(1);
        let door = Arc::new(StoreBackedEngine::new(Arc::clone(&store)));
        let more = MemoryStore::HANDOVER + 8;
        let now = Instant::now();
        let asks = (0..more).map(|n| {
            let id = n.to_le_bytes();
            let asked = request(&id, 7);
            let decided = DecisionGuard::decide(Arc::clone(&door), GATEWAY, &asked, now);
            decided.expect("a decision")
        });
        let handles: Vec<_> = asks.collect();
        let ends = Ends::over(&store);
        let held = deciding(&door);
        ends.stands("pick");
        let (done, returned) = mpsc::channel();
        thread::spawn(move || {
            drop(handles);
            done.send(()).ok();
        });
        let returned = returned.recv_timeout(PATIENCE).is_ok();
        let on_file = LeasesStore::live(&*store, GATEWAY);
        ends.go();
        ends.stands("issue");
        ends.go();
        let lease = held.join().expect("the held decision's thread");
        assert!(returned, "every handle was let go while the decision stood");
        assert_eq!(on_file, more, "a report is the deciding thread's to apply");
        assert!(lease.is_some(), "the held request has its lease");
        assert_eq!(LeasesStore::live(&*store, GATEWAY), 1, "its lease alone");
    }

    /// A handle that owns its door travels with its request: dropped on
    /// another thread behind its dispatch, it ends the lease from there.
    #[test]
    fn a_handle_that_owns_its_door_ends_its_lease_from_another_thread() {
        let (door, heard) = bed();
        let door = Arc::new(door);
        let now = Instant::now();
        let asked = request(b"r-1", 7);
        let decided = DecisionGuard::decide(Arc::clone(&door), GATEWAY, &asked, now);
        let mut handle = decided.expect("a decision");
        let first = handle.take(0).expect("the pick's first target");
        handle.dispatched(now);
        let ended = thread::spawn(move || drop(handle));
        ended.join().expect("the request's thread");
        assert_eq!(counted(door.store()), []);
        let told = [
            Call::Issued,
            Call::Dispatched(first.id, 0, 0),
            Call::Completed(first.id, TransportClass::Cancelled, 0),
        ];
        assert_eq!(clocked(&heard, 0), told);
    }

    /// Sixty-four hosts at once on one model, each holding its requests
    /// from the decision to the end, by turns served, dropped behind the
    /// dispatch, refused at the commit and dropped before any dispatch:
    /// every lease is ended once, by its own path's word, none is left on
    /// file and no count is left on a member.
    #[test]
    fn sixty_four_hosts_at_once_end_every_lease_once() {
        let (door, heard) = bed();
        let (hosts, each) = (64_u8, 32_u8);
        let start = Barrier::new(usize::from(hosts));
        thread::scope(|scope| {
            for host in 0..hosts {
                let (door, start) = (&door, &start);
                scope.spawn(move || {
                    start.wait();
                    for turn in 0..each {
                        let now = Instant::now();
                        let mut handle = held(door, &[host, turn], now);
                        match turn % 4 {
                            0 => {
                                let first = handle.take(0).expect("the first target");
                                handle.dispatched(now);
                                handle.completed(&outcome(first.id, TransportClass::Ok), now);
                            }
                            1 => handle.dispatched(now),
                            2 => handle.released(ReleaseReason::CommitFailed, Instant::now()),
                            _ => {}
                        }
                    }
                });
            }
        });
        let store = door.store();
        assert_eq!(counted(store), []);
        assert_eq!(LeasesStore::live(&**store, GATEWAY), 0);
        let heard = heard.lock();
        let told = |want: fn(&Call) -> bool| heard.iter().filter(|call| want(call)).count();
        let quarter = usize::from(hosts) * usize::from(each) / 4;
        let ok = |call: &Call| matches!(call, Call::Completed(_, TransportClass::Ok, _));
        let cancelled =
            |call: &Call| matches!(call, Call::Completed(_, TransportClass::Cancelled, _));
        let refused = |call: &Call| matches!(call, Call::Released(ReleaseReason::CommitFailed));
        let dropped = |call: &Call| matches!(call, Call::Released(ReleaseReason::Abandoned));
        assert_eq!(told(|call| matches!(call, Call::Issued)), 4 * quarter);
        assert_eq!(
            told(|call| matches!(call, Call::Dispatched(..))),
            2 * quarter
        );
        assert_eq!(
            (told(ok), told(cancelled), told(refused), told(dropped)),
            (quarter, quarter, quarter, quarter)
        );
    }

    /// A completion is written on the worker its host names, which need
    /// not be the target the handle took: the health precedence reads the
    /// worker that served.
    #[test]
    fn a_completion_is_written_on_the_worker_its_host_names() {
        let (door, _) = bed();
        let store = door.store();
        let now = Instant::now();
        let mut handle = held(&door, b"r-1", now);
        let other = handle.take(1).expect("the pick's second target");
        let taken = handle.take(0).expect("the pick's first target");
        handle.dispatched(now);
        handle.completed(&outcome(other.id, TransportClass::Ok), now);
        assert!(store.completed_ok_since(GATEWAY, other.id, now));
        assert!(!store.completed_ok_since(GATEWAY, taken.id, now));
    }

    /// A completion is written at the instant its host read: a host whose
    /// instants run ahead of the clock writes there, as its dispatch did.
    #[test]
    fn a_completion_is_written_at_its_hosts_instant() {
        let (door, _) = bed();
        let store = door.store();
        let now = Instant::now();
        let mut handle = held(&door, b"r-1", now);
        let first = handle.take(0).expect("the pick's first target");
        let ahead = now + Duration::from_secs(2);
        handle.completed(&outcome(first.id, TransportClass::Ok), ahead);
        assert!(store.completed_ok_since(GATEWAY, first.id, ahead));
    }

    /// A release behind a dispatch spans its host's two instants, the
    /// dispatch's and its own: no clock is read.
    #[test]
    fn a_release_behind_a_dispatch_spans_its_hosts_instants() {
        let (door, heard) = bed();
        let now = Instant::now();
        let mut handle = held(&door, b"r-1", now);
        let first = handle.take(0).expect("the pick's first target");
        let sent = now + Duration::from_secs(3_600);
        handle.dispatched(sent);
        handle.released(ReleaseReason::ClientGone, sent + Duration::from_millis(7));
        let told = [
            Call::Issued,
            Call::Dispatched(first.id, 0, 3_600_000_000),
            Call::Completed(first.id, TransportClass::Cancelled, 7_000),
        ];
        assert_eq!(*heard.lock(), told);
    }

    thread_local! {
        static WORDS: RefCell<Vec<Said>> = const { RefCell::new(Vec::new()) };
    }

    fn words() -> Vec<Said> {
        WORDS.with_borrow_mut(mem::take)
    }

    fn pair_door(prefills: usize, decodes: usize) -> Result<StoreBackedEngine, Box<dyn Error>> {
        let store = pair_fleet(prefills, decodes)?;
        store.tell_through(|_, _, _, said, _| WORDS.with_borrow_mut(|words| words.push(said)));
        Ok(StoreBackedEngine::new(store))
    }

    fn paired(
        door: &StoreBackedEngine,
        now: Instant,
    ) -> Result<DecisionGuard<&StoreBackedEngine>, DecisionRefused> {
        DecisionGuard::decide(door, GATEWAY, &pair_request(b"pair"), now)
    }

    fn states(store: &MemoryStore, lease: LeaseId) -> Option<[LeaseState; 2]> {
        let filed = store.sides(GATEWAY, lease)?;
        Some([filed.record.state, filed.decode?.state])
    }

    #[test]
    fn a_handle_of_an_encode_pair_holds_the_pairs_two_sides() -> Result<(), Box<dyn Error>> {
        let (door, encoder) = push_fleet()?;
        let store = door.store();
        let now = Instant::now();
        let items = EncodeItems {
            keys: &[Identity::hashed(11)],
            ..EncodeItems::default()
        };
        let mut placed = EncodePlaced::new();
        let mut handle = DecisionGuard::decide_encode_pair(
            &door,
            GATEWAY,
            &pair_request(b"pair"),
            items,
            &mut placed,
            now,
        )
        .expect("a decision");
        let lease = handle.lease().expect("the prefill's lease");
        let id = lease.lease.id;
        assert_eq!(lease.lease.side, Side::Prefill);
        let target = handle.take(0).expect("the first pair");
        let decode = target.pair.expect("the decode's target");
        assert_eq!(placed.encoders().len(), 1);
        assert_eq!(
            placed.encoders().first().map(|target| target.id),
            Some(encoder)
        );
        assert_eq!(states(store, id), Some([LeaseState::ISSUED; 2]));
        handle.dispatched_decode(now);
        assert_eq!(
            states(store, id),
            Some([LeaseState::ISSUED, LeaseState::Dispatched { at: now }])
        );
        drop(handle);
        assert!(store.sides(GATEWAY, id).is_none());
        assert_eq!(pair_counts(store, [target.id, decode.decode_id]), [0, 0]);
        assert_eq!(LeasesStore::live(&**store, GATEWAY), 0);
        Ok(())
    }

    #[test]
    fn each_side_of_a_pair_is_dispatched_and_ended_by_its_own_word() {
        for explicit in [true, false] {
            let door = pair_door(1, 2).expect("a pair door");
            let store = door.store();
            let now = Instant::now();
            let mut handle = paired(&door, now).expect("a pair's handle");
            let lease = handle.lease().expect("the pair's lease").lease.id;
            let first = handle.take(0).expect("the first target");
            let second = handle.take(1).expect("the alternate target");
            let d0 = first.pair.expect("the first decode").decode_id;
            let d1 = second.pair.expect("the alternate decode").decode_id;
            assert_ne!(d0, d1);
            let workers = [first.id, d0, d1];
            let (prefill_at, decode_at) = (
                now + Duration::from_micros(100),
                now + Duration::from_micros(200),
            );
            let prefill = LeaseState::Dispatched { at: prefill_at };
            let decode = LeaseState::Dispatched { at: decode_at };
            let issued = LeaseState::ISSUED;
            assert_eq!(states(store, lease), Some([issued, issued]));
            assert_eq!(pair_counts(store, workers), [1, 1, 0]);

            handle.dispatched(prefill_at);
            assert_eq!(states(store, lease), Some([prefill, issued]));
            assert_eq!(pair_counts(store, workers), [1, 1, 0]);
            if explicit {
                handle.dispatched_decode(decode_at);
                handle.dispatched_decode(decode_at);
                assert_eq!(states(store, lease), Some([prefill, decode]));
                assert_eq!(pair_counts(store, workers), [1, 0, 1]);
            }

            let decoded = Outcome {
                side: Side::Decode,
                ..outcome(d1, TransportClass::Ok)
            };
            handle.completed(&decoded, decode_at);
            handle.completed(&decoded, decode_at);
            assert_eq!(states(store, lease), Some([prefill, LeaseState::Completed]));
            assert_eq!(pair_counts(store, workers), [1, 0, 0]);
            let prefilled = Outcome {
                side: Side::Prefill,
                ..outcome(first.id, TransportClass::Ok)
            };
            let said = if explicit {
                prefilled
            } else {
                outcome(first.id, TransportClass::Ok)
            };
            handle.completed(&said, decode_at);
            drop(handle);
            assert!(store.sides(GATEWAY, lease).is_none());
            assert_eq!(pair_counts(store, workers), [0, 0, 0]);
            assert_eq!(
                words(),
                [
                    Said::Dispatched(
                        Dispatch {
                            target_index: 1,
                            side: Side::Prefill,
                            since_decision_micros: 100,
                        },
                        prefill_at
                    ),
                    Said::Dispatched(
                        Dispatch {
                            target_index: 1,
                            side: Side::Decode,
                            since_decision_micros: 200,
                        },
                        decode_at
                    ),
                    Said::Completed(decoded),
                    Said::Completed(prefilled),
                ]
            );
        }
    }

    #[test]
    fn a_handle_asks_for_its_decode_once_the_prefill_went_out() {
        let door = pair_door(1, 2).expect("a pair door");
        let store = door.store();
        let now = Instant::now();
        let asked = later_request(b"pair");
        let mut handle =
            DecisionGuard::decide(&door, GATEWAY, &asked, now).expect("a pair's handle");
        let lease = handle.lease().expect("the prefill lease").lease.id;
        let first = handle.take(0).expect("the first target");
        let other = handle.take(1).expect("the alternate target");
        let d0 = first.pair.expect("the first decode").decode_id;
        let d1 = other.pair.expect("the alternate decode").decode_id;
        let workers = [first.id, d0, d1];
        assert_eq!(
            states(store, lease),
            Some([LeaseState::ISSUED, LeaseState::Chosen])
        );
        assert_eq!(pair_counts(store, workers), [1, 0, 0]);
        assert!(handle.decode_chosen());
        let original = *handle.decision();
        assert_eq!(
            handle.decide_decode(&asked, now).err(),
            Some(DecisionRefused::Decode(DecodeRefused::PrefillNotHeld))
        );
        assert_eq!(*handle.decision(), original);
        assert_eq!(handle.index, 1);
        assert!(
            handle.decode_chosen(),
            "a refused call leaves the side chosen"
        );

        let sent = now + Duration::from_micros(100);
        handle.dispatched(sent);
        let prefill = LeaseState::Dispatched { at: sent };
        assert_eq!(states(store, lease), Some([prefill, LeaseState::Chosen]));
        assert_eq!(pair_counts(store, workers), [1, 0, 0]);
        let call = load_decodes(store, [d0, d1], sent + Duration::from_micros(50));
        let decision = handle
            .decide_decode(&asked, call)
            .expect("the decode decision");
        let new = decision
            .pick()
            .and_then(|pick| pick.targets.first())
            .expect("the new first target");
        assert_eq!(new.pair, other.pair);
        assert_eq!((handle.index, handle.target), (0, other));
        assert!(!handle.decode_chosen(), "the call's pick issued the side");
        assert_eq!(
            handle
                .lease()
                .map(|leased| (leased.lease.id, leased.lease.side)),
            Some((lease, Side::Prefill))
        );
        assert_eq!(states(store, lease), Some([prefill, LeaseState::ISSUED]));
        assert_eq!(pair_counts(store, workers), [1, 0, 1]);
        assert_eq!(
            handle.decide_decode(&asked, call).err(),
            Some(DecisionRefused::Decode(DecodeRefused::DecodeStands))
        );
        assert_eq!(handle.take_decode(1), Some(first));
        assert_eq!(states(store, lease), Some([prefill, LeaseState::ISSUED]));
        assert_eq!(pair_counts(store, workers), [1, 0, 1]);
        let decode_at = call + Duration::from_micros(60);
        handle.dispatched_decode(decode_at);
        assert_eq!(
            states(store, lease),
            Some([prefill, LeaseState::Dispatched { at: decode_at }])
        );
        assert_eq!(pair_counts(store, workers), [1, 1, 0]);
        let decoded = Outcome {
            side: Side::Decode,
            ..outcome(d0, TransportClass::Ok)
        };
        let prefilled = Outcome {
            side: Side::Prefill,
            ..outcome(first.id, TransportClass::Ok)
        };
        handle.completed(&decoded, decode_at);
        assert_eq!(states(store, lease), Some([prefill, LeaseState::Completed]));
        assert_eq!(pair_counts(store, workers), [1, 0, 0]);
        handle.completed(&prefilled, decode_at);
        drop(handle);
        assert_eq!(states(store, lease), None);
        assert_eq!(pair_counts(store, workers), [0, 0, 0]);
        assert_eq!(
            words(),
            [
                Said::Dispatched(
                    Dispatch {
                        side: Side::Prefill,
                        target_index: 1,
                        since_decision_micros: 100
                    },
                    sent
                ),
                // The span since the first decision: the side's lease is
                // dated by that decision's issue.
                Said::Dispatched(
                    Dispatch {
                        side: Side::Decode,
                        target_index: 1,
                        since_decision_micros: super::micros(decode_at - now)
                    },
                    decode_at
                ),
                Said::Completed(decoded),
                Said::Completed(prefilled),
            ]
        );
    }

    #[test]
    fn a_handle_keeps_its_first_pick_when_the_decode_call_sheds() {
        let door = pair_door(1, 2).expect("a pair door");
        let store = door.store();
        let now = Instant::now();
        let mut asked = later_request(b"pair");
        let mut handle =
            DecisionGuard::decide(&door, GATEWAY, &asked, now).expect("a pair's handle");
        let lease = handle.lease().expect("the prefill lease").lease.id;
        let picked = handle.take(1).expect("an alternate target");
        let original = *handle.decision();
        handle.dispatched(now);
        let before = store.sides(GATEWAY, lease).expect("both sides");
        asked.constraints.subset_hint = Some(0);
        assert_eq!(handle.decide_decode(&asked, now).ok(), Some(&original));
        assert_eq!(*handle.decision(), original);
        assert_eq!((handle.index, handle.target), (1, picked));
        assert_eq!(store.sides(GATEWAY, lease), Some(before));
        assert!(handle.decode_chosen(), "a shed leaves the side chosen");
        handle.dispatched_decode(now);
        assert!(!handle.decode_chosen(), "the dispatch booked the side");
        let worker = picked.pair.expect("the alternate decode").decode_id;
        let decode = store
            .sides(GATEWAY, lease)
            .and_then(|sides| sides.decode)
            .expect("the dispatched decode");
        assert_eq!(
            (decode.state, decode.target),
            (LeaseState::Dispatched { at: now }, worker)
        );
        drop(handle);
        let _ = words();
    }

    /// A host that makes a pair's request with a later decode.
    struct Later;

    impl RequestMaker for Later {
        fn model(&self) -> &str {
            later_request(b"pair").facts.model
        }

        fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
            later_request(b"pair")
        }
    }

    /// A handle asks its decode again for a request its host makes, as for
    /// one it stated: the call's pick issues the side.
    #[test]
    fn a_handle_asks_its_decode_again_for_a_request_its_host_makes() {
        let door = pair_door(1, 1).expect("a pair door");
        let now = Instant::now();
        let mut handle =
            DecisionGuard::decide_under(&door, GATEWAY, &Later, now).expect("a pair's handle");
        handle.dispatched(now);
        assert!(handle.decode_chosen());
        let decided = handle.decide_decode_under(&Later, now);
        assert!(decided.is_ok_and(|decision| decision.pick().is_some()));
        assert!(!handle.decode_chosen(), "the call's pick issued the side");
        // An issued side takes no second call.
        assert_eq!(
            handle.decide_decode_under(&Later, now).err(),
            Some(DecisionRefused::Decode(DecodeRefused::DecodeStands))
        );
        drop(handle);
        let _ = words();
    }

    #[test]
    fn a_handle_decided_under_a_maker_holds_the_decode_as_the_made_request_said() {
        let door = pair_door(1, 1).expect("a pair door");
        let handle = DecisionGuard::decide_under(&door, GATEWAY, &Later, Instant::now())
            .expect("a pair's handle");
        let _ = words();
        drop(handle);
        // A chosen decode is owed no word: the prefill's release alone.
        assert_eq!(
            words(),
            [Said::Released(
                Some(Side::Prefill),
                ReleaseReason::Abandoned
            )]
        );
    }

    /// The decode a call issued is dated by its own dispatch. The side's
    /// lease keeps the first decision's issue instant, and the handle
    /// counts the dispatch's span from that decision: a stage that dates a
    /// credit by the two keeps it past a report sampled while the prefill
    /// ran. The handle's lease shows the call's deadline, the lease's one.
    #[test]
    fn a_decode_asked_later_is_dated_by_its_own_dispatch() {
        let door = pair_door(1, 1).expect("a pair door");
        let now = Instant::now();
        let asked = later_request(b"pair");
        let mut handle =
            DecisionGuard::decide(&door, GATEWAY, &asked, now).expect("a pair's handle");
        handle.dispatched(now + Duration::from_micros(100));
        // The prefill ran for eight seconds.
        let call = now + Duration::from_secs(8);
        let decided = handle.decide_decode(&asked, call);
        assert!(decided.is_ok_and(|decision| decision.pick().is_some()));
        let due = handle.lease().map(|leased| leased.due);
        assert_eq!(due, Some(call + Duration::from_secs(5)));
        let sent = call + Duration::from_micros(60);
        handle.dispatched_decode(sent);
        let decode = words().into_iter().find_map(|said| match said {
            Said::Dispatched(dispatch, at) if dispatch.side == Side::Decode => {
                Some((dispatch.since_decision_micros, at))
            }
            _ => None,
        });
        assert_eq!(decode, Some((8_000_060, sent)));
    }

    /// What a handle lent with a word or a decode call: the word, the
    /// index of the target taken, and whether the decode stood chosen.
    type Lent = Mutex<Vec<(&'static str, u8, bool)>>;

    /// A door that keeps what each word and each decode call was lent.
    struct Lends<'a>(&'a StoreBackedEngine, Lent);

    impl DecisionEngine for Lends<'_> {
        fn decide_under(
            &self,
            gateway: GatewayId,
            maker: &impl RequestMaker,
            now: Instant,
        ) -> Result<(DecideOutcome, bool), DecisionRefused> {
            DecisionEngine::decide_under(self.0, gateway, maker, now)
        }

        fn decide_decode_under(
            &self,
            gateway: GatewayId,
            maker: &impl RequestMaker,
            held: &HeldDecision<'_>,
            now: Instant,
        ) -> Result<DecideOutcome, DecisionRefused> {
            let lent = ("decode call", held.index, held.decode_chosen);
            self.1.lock().push(lent);
            DecisionEngine::decide_decode_under(self.0, gateway, maker, held, now)
        }

        fn decide_encode_pair(
            &self,
            gateway: GatewayId,
            request: &Request<'_>,
            items: EncodeItems<'_>,
            placed: &mut EncodePlaced,
            now: Instant,
        ) -> Result<DecideOutcome, DecisionRefused> {
            DecisionEngine::decide_encode_pair(self.0, gateway, request, items, placed, now)
        }

        fn say(&self, gateway: GatewayId, held: &HeldDecision<'_>, said: Said, now: Instant) {
            let word = match said {
                Said::Dispatched(dispatch, _) if dispatch.side == Side::Decode => "decode sent",
                Said::Dispatched(..) => "sent",
                Said::Completed(outcome) if outcome.side == Side::Decode => "decode ended",
                Said::Completed(_) => "ended",
                Said::Released(..) => "released",
            };
            self.1.lock().push((word, held.index, held.decode_chosen));
            DecisionEngine::say(self.0, gateway, held, said, now);
        }

        fn sweep(&self, now: Instant) -> SweepOutcome {
            DecisionEngine::sweep(self.0, now)
        }

        fn with_spec<R>(&self, read: impl FnOnce(&AppliedSpec) -> R) -> R {
            DecisionEngine::with_spec(self.0, read)
        }

        fn prompt_hashes(
            &self,
            model: &str,
            tokens: Option<&[u32]>,
            text: Option<&str>,
            seeds: HashSeeds,
        ) -> PromptHashes {
            DecisionEngine::prompt_hashes(self.0, model, tokens, text, seeds)
        }

        fn pair_refusal(&self, model: &str, wires: WireModes) -> Option<Refusal> {
            DecisionEngine::pair_refusal(self.0, model, wires)
        }

        fn faults(&self) -> u64 {
            DecisionEngine::faults(self.0)
        }

        fn shadowed(&self, binding: u32, kind: ShadowKind) -> u64 {
            DecisionEngine::shadowed(self.0, binding, kind)
        }
    }

    /// What a handle lends its engine is read before the word moves the
    /// handle. An engine with no file of its leases tells its stages from
    /// it: the index names the target taken at an end too, where the word
    /// carries none, and a decode that stood chosen is known at its
    /// dispatch, which is its issue.
    #[test]
    fn a_handle_lends_the_index_taken_and_whether_the_decode_stood_chosen() {
        let door = pair_door(1, 2).expect("a pair door");
        let lends = Lends(&door, Lent::default());
        let now = Instant::now();
        let asked = later_request(b"pair");
        // No decode call: the first pick's decode goes out, a side that
        // stood chosen until its dispatch.
        let mut handle =
            DecisionGuard::decide(&lends, GATEWAY, &asked, now).expect("a pair's handle");
        let taken = handle.take(1).expect("the alternate target");
        let decode = taken.pair.expect("the alternate decode").decode_id;
        handle.dispatched(now);
        handle.dispatched_decode(now);
        let decoded = Outcome {
            side: Side::Decode,
            ..outcome(decode, TransportClass::Ok)
        };
        let prefilled = Outcome {
            side: Side::Prefill,
            ..outcome(taken.id, TransportClass::Ok)
        };
        handle.completed(&decoded, now);
        handle.completed(&prefilled, now);
        drop(handle);
        assert_eq!(
            mem::take(&mut *lends.1.lock()),
            [
                ("sent", 1, true),
                ("decode sent", 1, true),
                ("decode ended", 1, false),
                ("ended", 1, false),
            ]
        );
        // A decode call's pick issues the side and takes the call's first
        // target: the side's dispatch is lent a side that stands issued.
        // The handle is dropped with both sides sent, and each ends.
        let mut handle =
            DecisionGuard::decide(&lends, GATEWAY, &asked, now).expect("a pair's handle");
        handle.take(1).expect("the alternate target");
        handle.dispatched(now);
        let decided = handle.decide_decode(&asked, now);
        assert!(decided.is_ok_and(|decision| decision.pick().is_some()));
        handle.dispatched_decode(now);
        drop(handle);
        assert_eq!(
            mem::take(&mut *lends.1.lock()),
            [
                ("sent", 1, true),
                ("decode call", 1, true),
                ("decode sent", 0, false),
                ("ended", 0, false),
                ("decode ended", 0, false),
            ]
        );
        let _ = words();
    }

    /// A decode side given up while it stood chosen takes no call. The
    /// handle owed no report of it, so the store still holds it chosen:
    /// the handle refuses the call by name and asks no door. Nothing is
    /// issued, booked or said.
    #[test]
    fn a_decode_given_up_while_chosen_takes_no_call() {
        let door = pair_door(1, 1).expect("a pair door");
        let store = door.store();
        let now = Instant::now();
        let asked = later_request(b"pair");
        let mut handle =
            DecisionGuard::decide(&door, GATEWAY, &asked, now).expect("a pair's handle");
        let lease = handle.lease().expect("the prefill lease").lease.id;
        let first = handle.take(0).expect("the first target");
        let decode = first.pair.expect("the first decode").decode_id;
        handle.dispatched(now);
        handle.released_decode(ReleaseReason::ClientGone, now);
        let before = store.sides(GATEWAY, lease);
        let chosen = before.and_then(|sides| sides.decode).map(|side| side.state);
        assert_eq!(chosen, Some(LeaseState::Chosen));
        assert_eq!(
            handle.decide_decode(&asked, now).err(),
            Some(DecisionRefused::Decode(DecodeRefused::DecodeStands))
        );
        assert_eq!(store.sides(GATEWAY, lease), before);
        assert_eq!(pair_counts(store, [first.id, decode]), [1, 0]);
        let said = words();
        assert!(
            matches!(said.as_slice(), [Said::Dispatched(..)]),
            "{said:?}"
        );
        // A single's handle answers by name too: its lease has no decode
        // side.
        let (single, _) = bed();
        let mut alone = held(&single, b"r-1", now);
        assert_eq!(
            alone.decide_decode(&request(b"r-1", 7), now).err(),
            Some(DecisionRefused::Decode(DecodeRefused::NotChosen))
        );
    }

    /// A decode that left with its prefill is refused as the store refuses
    /// it. The prefill's end took the chosen side off the file with it, so
    /// the handle owes neither side a word and the lease is gone.
    #[test]
    fn a_decode_that_left_with_its_prefill_is_a_lease_that_is_gone() {
        let door = pair_door(1, 1).expect("a pair door");
        let store = door.store();
        let now = Instant::now();
        let asked = later_request(b"pair");
        let mut handle =
            DecisionGuard::decide(&door, GATEWAY, &asked, now).expect("a pair's handle");
        let lease = handle.lease().expect("the prefill lease").lease.id;
        let first = handle.take(0).expect("the first target");
        handle.dispatched(now);
        handle.completed(&outcome(first.id, TransportClass::Ok), now);
        assert_eq!(store.sides(GATEWAY, lease), None);
        assert_eq!(
            handle.decide_decode(&asked, now).err(),
            Some(DecisionRefused::Decode(DecodeRefused::NoLease))
        );
        assert_eq!(store.sides(GATEWAY, lease), None);
    }

    #[test]
    fn a_handle_that_never_asks_sends_the_first_picks_decode() {
        let door = pair_door(1, 1).expect("a pair door");
        let store = door.store();
        let now = Instant::now();
        let asked = later_request(b"pair");
        let mut handle =
            DecisionGuard::decide(&door, GATEWAY, &asked, now).expect("a pair's handle");
        let lease = handle.lease().expect("the prefill lease").lease.id;
        let first = handle.take(0).expect("the first target");
        let decode = first.pair.expect("the first decode").decode_id;
        assert_eq!(handle.take_decode(0), None);
        handle.dispatched(now);
        assert_eq!(handle.take_decode(0), Some(first));
        assert_eq!(pair_counts(store, [first.id, decode]), [1, 0]);
        let at = now + Duration::from_micros(80);
        handle.dispatched_decode(at);
        handle.dispatched_decode(at);
        assert_eq!(
            states(store, lease),
            Some([
                LeaseState::Dispatched { at: now },
                LeaseState::Dispatched { at }
            ])
        );
        assert_eq!(pair_counts(store, [first.id, decode]), [1, 1]);
        assert_eq!(
            words(),
            [
                Said::Dispatched(
                    Dispatch {
                        side: Side::Prefill,
                        target_index: 0,
                        since_decision_micros: 0
                    },
                    now
                ),
                Said::Dispatched(
                    Dispatch {
                        side: Side::Decode,
                        target_index: 0,
                        since_decision_micros: 80
                    },
                    at
                ),
            ]
        );
        drop(handle);
        let _ = words();
    }

    #[test]
    fn a_handle_dropped_with_its_decode_still_chosen_says_nothing_of_it() {
        for sent in [false, true] {
            let door = pair_door(1, 1).expect("a pair door");
            let store = door.store();
            let now = Instant::now();
            let asked = later_request(b"pair");
            let mut handle =
                DecisionGuard::decide(&door, GATEWAY, &asked, now).expect("a pair's handle");
            let lease = handle.lease().expect("the prefill lease").lease.id;
            let first = handle.take(0).expect("the first target");
            let decode = first.pair.expect("the first decode").decode_id;
            assert_eq!(
                states(store, lease),
                Some([LeaseState::ISSUED, LeaseState::Chosen])
            );
            if sent {
                handle.dispatched(now);
                assert_eq!(words().len(), 1);
            }
            drop(handle);
            let [report]: [Said; 1] = words().try_into().expect("one prefill report");
            if sent {
                let Said::Completed(cancelled) = report else {
                    panic!("the prefill completed");
                };
                assert_eq!(
                    (cancelled.side, cancelled.served, cancelled.class),
                    (Side::Prefill, first.id, TransportClass::Cancelled)
                );
            } else {
                assert_eq!(
                    report,
                    Said::Released(Some(Side::Prefill), ReleaseReason::Abandoned)
                );
            }
            assert_eq!(states(store, lease), None);
            assert_eq!(pair_counts(store, [first.id, decode]), [0, 0]);
        }

        let door = pair_door(1, 1).expect("a pair door");
        let store = door.store();
        let now = Instant::now();
        let asked = later_request(b"pair");
        let mut handle =
            DecisionGuard::decide(&door, GATEWAY, &asked, now).expect("a pair's handle");
        let lease = handle.lease().expect("the prefill lease").lease.id;
        handle.dispatched(now);
        let _ = words();
        // A named release would be visible while the prefill is still held.
        let before = store.sides(GATEWAY, lease);
        handle.released_decode(ReleaseReason::ClientGone, now);
        assert!(words().is_empty());
        assert_eq!(store.sides(GATEWAY, lease), before);
        assert_eq!(handle.take_decode(0), None);
        drop(handle);
        assert_eq!(words().len(), 1, "the prefill's end alone");

        let mut handle =
            DecisionGuard::decide(&door, GATEWAY, &asked, now).expect("a pair's handle");
        let lease = handle.lease().expect("the prefill lease").lease.id;
        let prefill = handle.take(0).expect("the first target").id;
        handle.dispatched(now);
        let done = Outcome {
            side: Side::Prefill,
            ..outcome(prefill, TransportClass::Ok)
        };
        handle.completed(&done, now);
        assert_eq!(handle.take_decode(0), None);
        assert_eq!(states(store, lease), None);
        assert_eq!(words().len(), 2);
    }

    #[test]
    fn a_decode_taken_after_the_prefill_went_out_moves_the_decode_alone() {
        let door = pair_door(2, 2).expect("a pair door");
        let store = door.store();
        let now = Instant::now();
        let mut handle = paired(&door, now).expect("a pair's handle");
        let lease = handle.lease().expect("the pair's lease").lease.id;
        let first = handle.take(0).expect("the first target");
        let alternate = handle.take(1).expect("the alternate target");
        assert_eq!(handle.take(0), Some(first));
        assert_eq!(
            handle.take_decode(1),
            None,
            "the prefill still owes a dispatch"
        );
        let other = ["http://p0:8000", "http://p1:8000"]
            .into_iter()
            .filter_map(|key| store.by_key(key))
            .find(|worker| worker.id != first.id)
            .expect("another prefill");
        // The picker lists one prefill. Add a target to exercise the handle's check.
        let Decision::Pick(pick) = &mut handle.decided.decision else {
            panic!("a pair pick");
        };
        let different = pick.targets.as_slice().len();
        pick.targets.push(Target {
            id: other.id,
            ..alternate
        });

        handle.dispatched(now);
        assert_eq!(handle.take(1), None, "the prefill has dispatched");
        assert_eq!(handle.take_decode(different), None);
        assert_eq!(handle.take_decode(usize::MAX), None);
        assert_eq!(handle.take_decode(1), Some(alternate));
        let d0 = first.pair.expect("the first decode").decode_id;
        let d1 = alternate.pair.expect("the alternate decode").decode_id;
        let workers = [first.id, other.id, d0, d1];
        assert_eq!(
            pair_counts(store, workers),
            [1, 0, 1, 0],
            "a take books nothing"
        );
        handle.dispatched_decode(now);
        assert_eq!(handle.take_decode(0), None);
        assert_eq!(pair_counts(store, workers), [1, 0, 0, 1]);
        let filed = store.sides(GATEWAY, lease).expect("both sides on file");
        let decode = filed.decode.expect("the decode side");
        assert_eq!(filed.record.target, first.id);
        assert_eq!((decode.target, decode.target_index), (d1, 1));
        drop(handle);
        let _ = words();
    }

    #[test]
    fn a_pair_given_up_before_any_send_ends_both_sides_in_one_word() {
        let door = pair_door(1, 1).expect("a pair door");
        let store = door.store();
        let now = Instant::now();
        let mut handle = paired(&door, now).expect("a pair's handle");
        assert!(!handle.decode_chosen(), "a pair decided whole issues both");
        let first = handle.take(0).expect("the first target");
        let decode = first.pair.expect("the first decode").decode_id;
        let reason = ReleaseReason::CommitFailed;
        handle.released(reason, now);
        assert_eq!(pair_counts(store, [first.id, decode]), [0, 0]);
        assert_eq!(LeasesStore::live(&**store, GATEWAY), 0);
        drop(handle);
        assert_eq!(words(), [Said::Released(None, reason)]);
    }

    #[test]
    fn a_pair_dropped_with_one_side_sent_completes_it_and_gives_the_other_up() {
        for sent in [Side::Prefill, Side::Decode] {
            let door = pair_door(1, 2).expect("a pair door");
            let store = door.store();
            let now = Instant::now();
            let mut handle = paired(&door, now).expect("a pair's handle");
            let taken = handle.take(1).expect("the alternate target");
            let decode = taken.pair.expect("the alternate decode").decode_id;
            let (worker, waiting) = if sent == Side::Prefill {
                handle.dispatched(now);
                (taken.id, Side::Decode)
            } else {
                handle.dispatched_decode(now);
                assert!(handle.take(0).is_some());
                (decode, Side::Prefill)
            };
            assert_eq!(words().len(), 1);
            drop(handle);
            let [first, second]: [Said; 2] = words().try_into().expect("two reports");
            let (completed, released) = if sent == Side::Prefill {
                (first, second)
            } else {
                (second, first)
            };
            let Said::Completed(cancelled) = completed else {
                panic!("the sent side completed");
            };
            assert_eq!(
                (cancelled.side, cancelled.served, cancelled.class),
                (sent, worker, TransportClass::Cancelled)
            );
            assert_eq!(
                released,
                Said::Released(Some(waiting), ReleaseReason::Abandoned)
            );
            assert_eq!(LeasesStore::live(&**store, GATEWAY), 0);
            assert_eq!(pair_counts(store, [taken.id, decode]), [0, 0]);

            let mut handle = paired(&door, now).expect("another pair's handle");
            let lease = handle.lease().expect("the pair's lease").lease.id;
            let first = handle.take(0).expect("the first target");
            let decode = first.pair.expect("the first decode").decode_id;
            let reason = ReleaseReason::ClientGone;
            let (state, report) = if sent == Side::Decode {
                handle.dispatched_decode(now);
                assert_eq!(words().len(), 1);
                let cancelled = Outcome {
                    side: Side::Decode,
                    served: decode,
                    class: TransportClass::Cancelled,
                    status: 0,
                    ttft_micros: 0,
                    tokens_in: 0,
                    tokens_out: 0,
                    duration_micros: 0,
                };
                (LeaseState::Completed, Said::Completed(cancelled))
            } else {
                (
                    LeaseState::Released(reason),
                    Said::Released(Some(Side::Decode), reason),
                )
            };
            handle.released_decode(reason, now);
            handle.released_decode(reason, now);
            assert_eq!(states(store, lease), Some([LeaseState::ISSUED, state]));
            assert_eq!(pair_counts(store, [first.id, decode]), [1, 0]);
            assert_eq!(words(), [report]);
            drop(handle);
            assert_eq!(
                words(),
                [Said::Released(
                    Some(Side::Prefill),
                    ReleaseReason::Abandoned
                )]
            );
        }
    }

    /// A single's handle holds no decode side: a decode's take, dispatch
    /// and release say nothing. Its lease ends under any side its host
    /// names, a decode outcome among them: a decode phase can stand under
    /// a single plan, and the outcome its host read reaches the store.
    #[test]
    fn a_single_handle_ends_under_a_decode_outcome_and_says_nothing_else_of_a_decode_side() {
        let (door, heard) = bed();
        let store = door.store();
        let now = Instant::now();
        let mut handle = held(&door, b"single", now);
        let lease = handle.lease().expect("the single's lease").lease.id;
        let first = handle.take(0).expect("the first target");
        handle.dispatched_decode(now);
        assert_eq!(handle.take_decode(1), None);
        handle.released_decode(ReleaseReason::ClientGone, now);
        let state = LeasesStore::get(&**store, GATEWAY, lease).map(|filed| filed.state);
        assert_eq!(state, Some(LeaseState::ISSUED));
        assert_eq!(counted(store), [(first.id, 1)]);
        assert_eq!(*heard.lock(), [Call::Issued]);
        let decoded = Outcome {
            side: Side::Decode,
            ..outcome(first.id, TransportClass::Ok)
        };
        handle.completed(&decoded, now + Duration::from_micros(400));
        assert!(LeasesStore::get(&**store, GATEWAY, lease).is_none());
        assert_eq!(counted(store), []);
        let told = [
            Call::Issued,
            Call::Dispatched(first.id, 0, 400),
            Call::Completed(first.id, TransportClass::Ok, 5_000),
        ];
        assert_eq!(*heard.lock(), told);
        drop(handle);
        assert_eq!(*heard.lock(), told, "the drop says nothing more");
    }

    /// A pair's lease holds no encode side: an encode outcome ends no
    /// side and says no word of the lease, and the prefill's lease stands
    /// for its own end. The served worker's health takes the completion,
    /// as a served controller writes it for such a frame.
    #[test]
    fn an_encode_outcome_on_a_pairs_handle_writes_its_workers_health_alone() {
        let door = pair_door(1, 1).expect("a pair door");
        let store = door.store();
        let now = Instant::now();
        let mut handle = paired(&door, now).expect("a pair's handle");
        let lease = handle.lease().expect("the pair's lease").lease.id;
        let first = handle.take(0).expect("the first target");
        let decode = first.pair.expect("the first decode").decode_id;
        let before = states(store, lease);
        let encoded = Outcome {
            side: Side::Encode,
            ..outcome(decode, TransportClass::Ok)
        };
        assert!(!HealthSink::completed_ok_since(
            &**store, GATEWAY, decode, now
        ));
        handle.completed(&encoded, now);
        assert_eq!(states(store, lease), before, "neither side moved");
        assert_eq!(pair_counts(store, [first.id, decode]), [1, 1]);
        // The file hears the completion whole, under its own side, and a
        // frame on that side ends no side: no stage is told of it.
        assert_eq!(words(), [Said::Completed(encoded)]);
        let sides = store.sides(GATEWAY, lease).expect("the lease on file");
        assert_eq!(sides.ending(Some(Side::Encode)), [None, None]);
        assert!(
            HealthSink::completed_ok_since(&**store, GATEWAY, decode, now),
            "the served worker's health takes the completion"
        );
        handle.completed(&outcome(first.id, TransportClass::Ok), now);
        let ended = states(store, lease).map(|[prefill, _]| prefill);
        assert_eq!(ended, Some(LeaseState::Completed));
        drop(handle);
        let _ = words();
    }

    /// A decode already sent names its prefill: a take behind it keeps
    /// the prefill the decode was sent for.
    #[test]
    fn a_prefill_taken_behind_a_sent_decode_stays_on_its_prefill() {
        let door = pair_door(2, 2).expect("a pair door");
        let store = door.store();
        let now = Instant::now();
        let mut handle = paired(&door, now).expect("a pair's handle");
        let first = handle.take(0).expect("the first target");
        let alternate = handle.take(1).expect("the alternate target");
        assert_eq!(handle.take(0), Some(first));
        let other = ["http://p0:8000", "http://p1:8000"]
            .into_iter()
            .filter_map(|key| store.by_key(key))
            .find(|worker| worker.id != first.id)
            .expect("another prefill");
        // The picker lists one prefill. Add a target to exercise the handle's check.
        let Decision::Pick(pick) = &mut handle.decided.decision else {
            panic!("a pair pick");
        };
        let different = pick.targets.as_slice().len();
        pick.targets.push(Target {
            id: other.id,
            ..alternate
        });
        handle.dispatched_decode(now);
        assert_eq!(handle.take(different), None, "another prefill");
        assert_eq!(handle.take(1), Some(alternate), "the same prefill");
        drop(handle);
        let _ = words();
    }

    #[cfg(feature = "client")]
    mod linked {
        use std::pin::pin;

        use tokio::time::timeout;
        use tokio_stream::StreamExt as _;

        use super::{
            outcome, request, DecisionGuard, DecisionRefused, DecodeRefused, Duration, Instant,
            Request, Side, StoreBackedEngine, TransportClass, WorkerId, PATIENCE,
        };
        use crate::{
            engine::Demands,
            service::{
                client::{poll, DecisionUnavailable, Error, Peer},
                proto::{
                    self,
                    completed::Class,
                    controller_frame,
                    decide_answer::Answer,
                    decision::Outcome,
                    gateway_frame::Frame,
                    released::{Reason, What},
                },
            },
            tracking::ReleaseReason,
        };

        fn picked() -> proto::Decision {
            proto::Decision {
                leases: vec![proto::Lease {
                    id: 73,
                    kind: proto::LeaseKind::Credit.into(),
                    side: proto::Side::Single.into(),
                    dispatch_ttl_ms: 70,
                    ..proto::Lease::default()
                }],
                outcome: Some(Outcome::Pick(proto::Pick {
                    targets: [28, 19]
                        .into_iter()
                        .map(|id| proto::Target {
                            id,
                            dp_rank: 2,
                            pair: None,
                        })
                        .collect(),
                })),
                trace: None,
            }
        }

        async fn answered(
            peer: &mut Peer,
            told: proto::Decision,
            now: Instant,
        ) -> Result<DecisionGuard<&'static StoreBackedEngine>, DecisionUnavailable> {
            let mut ask = request(&[1], 7);
            ask.facts.budget_micros = 200_000;
            answered_to(peer, &ask, told, now).await
        }

        /// [`answered`] for the request `ask`, of id 1 and a budget of 200 ms.
        async fn answered_to(
            peer: &mut Peer,
            ask: &Request<'_>,
            told: proto::Decision,
            now: Instant,
        ) -> Result<DecisionGuard<&'static StoreBackedEngine>, DecisionUnavailable> {
            peer.read_hello().await;
            let link = peer.link.clone();
            let exchange = async {
                let sent = peer.requests.next().await.expect("an ask went out");
                assert_eq!(sent.request_id, [1]);
                assert_eq!(sent.facts.expect("request facts").budget_micros, 200_000);
                peer.answer(1, link.term_floor(), Answer::Decision(told));
            };
            let both = async {
                let held = pin!(DecisionGuard::asked(&link, ask, now, PATIENCE));
                let (handle, ()) = tokio::join!(held, exchange);
                handle
            };
            timeout(PATIENCE, both).await.expect("an answer came back")
        }

        async fn reports(peer: &mut Peer, expected: impl IntoIterator<Item = Frame>) {
            for (index, frame) in expected.into_iter().enumerate() {
                let report = timeout(PATIENCE, peer.reports.next())
                    .await
                    .expect("a report was queued")
                    .expect("the session stands");
                assert_eq!(
                    report.seq,
                    u64::try_from(index).expect("the sequence fits") + 2
                );
                assert_eq!(report.frame, Some(frame));
            }
            assert!(
                poll(pin!(peer.reports.next())).is_pending(),
                "no further report is owed"
            );
        }

        fn dispatch(index: u32, since: u32) -> Frame {
            Frame::Dispatched(proto::Dispatched {
                lease_id: 73,
                target_index: index,
                side: proto::Side::Single.into(),
                since_decision_micros: since,
            })
        }

        fn released(side: proto::Side, reason: Reason) -> Frame {
            Frame::Released(proto::Released {
                what: Some(What::LeaseId(73)),
                reason: reason.into(),
                side: side.into(),
            })
        }

        #[tokio::test(start_paused = true)]
        async fn a_served_link_request_reports_its_dispatch_and_completion_once() {
            let mut peer = Peer::new(1);
            let now = Instant::now();
            let mut handle = answered(&mut peer, picked(), now)
                .await
                .expect("the pick is held");
            assert_eq!(
                handle.lease().expect("a lease").due,
                now + Duration::from_millis(70)
            );
            let target = handle.take(1).expect("the pick's second target");
            assert_eq!(target.id, WorkerId(19));
            assert!(handle.take(2).is_none());
            let dispatched = now + Duration::from_micros(250);
            handle.dispatched(dispatched);
            assert!(handle.take(0).is_none());
            let complete = outcome(WorkerId(28), TransportClass::Ok);
            handle.completed(&complete, now + Duration::from_millis(5));
            handle.dispatched(now);
            handle.completed(&complete, now);
            handle.released(ReleaseReason::CommitFailed, now);
            drop(handle);
            reports(
                &mut peer,
                [
                    dispatch(1, 250),
                    Frame::Completed(proto::Completed {
                        lease_id: 73,
                        side: proto::Side::Single.into(),
                        served: 28,
                        status: 200,
                        class: Class::Ok.into(),
                        ttft_micros: 1_000,
                        tokens_in: 12,
                        tokens_out: 34,
                        duration_micros: 5_000,
                    }),
                ],
            )
            .await;
        }

        /// A request can outlast the session its decision came over. Its
        /// handle holds the gateway's link, so what it reports then waits
        /// for the next session and rides it.
        #[tokio::test(start_paused = true)]
        async fn a_request_that_outlasts_its_session_reports_on_the_next_one() {
            let mut peer = Peer::patient(1);
            let now = Instant::now();
            let mut handle = answered(&mut peer, picked(), now)
                .await
                .expect("the pick is held");
            let target = handle.take(1).expect("the pick's second target");
            assert_eq!(target.id, WorkerId(19));
            peer.end_session().await;
            handle.dispatched(now + Duration::from_micros(250));

            drop(peer.redial(1));
            peer.read_hello().await;
            let complete = outcome(WorkerId(28), TransportClass::Ok);
            handle.completed(&complete, now + Duration::from_millis(5));
            drop(handle);
            reports(
                &mut peer,
                [
                    dispatch(1, 250),
                    Frame::Completed(proto::Completed {
                        lease_id: 73,
                        side: proto::Side::Single.into(),
                        served: 28,
                        status: 200,
                        class: Class::Ok.into(),
                        ttft_micros: 1_000,
                        tokens_in: 12,
                        tokens_out: 34,
                        duration_micros: 5_000,
                    }),
                ],
            )
            .await;
        }

        #[tokio::test(start_paused = true)]
        async fn a_link_reports_when_every_commit_is_refused_or_a_client_leaves() {
            let now = Instant::now();
            for (reason, wire) in [
                (ReleaseReason::CommitFailed, Reason::CommitFailed),
                (ReleaseReason::ClientGone, Reason::ClientGone),
            ] {
                let mut peer = Peer::new(1);
                let mut handle = answered(&mut peer, picked(), now)
                    .await
                    .expect("the pick is held");
                handle.released(reason, now);
                handle.completed(&outcome(WorkerId(28), TransportClass::Ok), now);
                drop(handle);
                reports(&mut peer, [released(proto::Side::Single, wire)]).await;
            }
        }

        #[tokio::test(start_paused = true)]
        async fn dropping_a_link_handle_before_dispatch_reports_abandonment() {
            let mut peer = Peer::new(1);
            let handle = answered(&mut peer, picked(), Instant::now())
                .await
                .expect("the pick is held");
            drop(handle);
            reports(
                &mut peer,
                [released(proto::Side::Single, Reason::Abandoned)],
            )
            .await;
        }

        #[tokio::test(start_paused = true)]
        async fn dropping_or_releasing_a_link_handle_after_dispatch_reports_cancellation() {
            for release in [false, true] {
                let mut peer = Peer::new(1);
                let now = Instant::now();
                let mut handle = answered(&mut peer, picked(), now)
                    .await
                    .expect("the pick is held");
                let sent = now + Duration::from_secs(3_600);
                handle.dispatched(sent);
                if release {
                    handle.released(ReleaseReason::ClientGone, sent + Duration::from_millis(7));
                }
                drop(handle);
                reports(
                    &mut peer,
                    [
                        dispatch(0, 3_600_000_000),
                        Frame::Completed(proto::Completed {
                            lease_id: 73,
                            side: proto::Side::Single.into(),
                            served: 28,
                            status: 0,
                            class: Class::Cancelled.into(),
                            ttft_micros: 0,
                            tokens_in: 0,
                            tokens_out: 0,
                            duration_micros: if release { 7_000 } else { 0 },
                        }),
                    ],
                )
                .await;
            }
        }

        #[tokio::test(start_paused = true)]
        async fn a_link_completion_supplies_the_dispatch_when_the_send_failed() {
            let mut peer = Peer::new(1);
            let now = Instant::now();
            let mut handle = answered(&mut peer, picked(), now)
                .await
                .expect("the pick is held");
            let target = handle.take(1).expect("the pick's second target");
            handle.completed(
                &outcome(target.id, TransportClass::ConnectFailed),
                now + Duration::from_micros(400),
            );
            drop(handle);
            reports(
                &mut peer,
                [
                    dispatch(1, 400),
                    Frame::Completed(proto::Completed {
                        lease_id: 73,
                        side: proto::Side::Single.into(),
                        served: 19,
                        status: 200,
                        class: Class::ConnectFailed.into(),
                        ttft_micros: 1_000,
                        tokens_in: 12,
                        tokens_out: 34,
                        duration_micros: 5_000,
                    }),
                ],
            )
            .await;
        }

        #[tokio::test(start_paused = true)]
        async fn a_link_decision_without_a_lease_reports_nothing() {
            let mut peer = Peer::new(1);
            let now = Instant::now();
            let told = proto::Decision {
                outcome: Some(Outcome::Shed(proto::Shed {
                    reason: proto::ShedReason::NoEndpoints.into(),
                    ..proto::Shed::default()
                })),
                ..proto::Decision::default()
            };
            let mut handle = answered(&mut peer, told, now)
                .await
                .expect("the shed is held");
            assert!(handle.lease().is_none());
            assert!(handle.take(0).is_none());
            handle.dispatched(now);
            handle.completed(&outcome(WorkerId(28), TransportClass::Ok), now);
            handle.released(ReleaseReason::CommitFailed, now);
            drop(handle);
            reports(&mut peer, []).await;
        }

        #[tokio::test(start_paused = true)]
        async fn an_unreadable_answer_is_unsent_and_its_lease_is_released_late() {
            let missing = proto::Decision {
                outcome: None,
                ..picked()
            };
            let mut unknown = picked();
            unknown.leases[0].side = i32::MAX;
            for (told, path, side) in [
                (missing, "outcome:", proto::Side::Single),
                (unknown, "leases.side:", proto::Side::Unspecified),
            ] {
                let mut peer = Peer::new(1);
                let answer = answered(&mut peer, told, Instant::now()).await;
                assert!(
                    matches!(&answer, Err(DecisionUnavailable::Unsent(Error::NoDecision(why))) if why.starts_with(path)),
                    "{answer:?}"
                );
                reports(&mut peer, [released(side, Reason::LateDecision)]).await;
            }
        }

        fn paired() -> proto::Decision {
            let mut told = picked();
            told.leases[0].side = proto::Side::Prefill.into();
            told.leases.push(proto::Lease {
                side: proto::Side::Decode.into(),
                ..told.leases[0]
            });
            let Some(Outcome::Pick(pick)) = &mut told.outcome else {
                panic!("a pick");
            };
            for (index, target) in pick.targets.iter_mut().enumerate() {
                target.id = 28;
                target.pair = Some(proto::Pair {
                    decode_id: [47, 49][index],
                    decode_dp_rank: 3,
                });
            }
            told
        }

        fn pair_dispatch(side: proto::Side, index: u32, since: u32) -> Frame {
            let Frame::Dispatched(mut said) = dispatch(index, since) else {
                panic!("a dispatch");
            };
            said.side = side.into();
            Frame::Dispatched(said)
        }

        fn pair_completion(side: proto::Side, served: u64) -> Frame {
            Frame::Completed(proto::Completed {
                lease_id: 73,
                side: side.into(),
                served,
                status: 200,
                class: Class::Ok.into(),
                ttft_micros: 1_000,
                tokens_in: 12,
                tokens_out: 34,
                duration_micros: 5_000,
            })
        }

        #[tokio::test(start_paused = true)]
        async fn a_linked_pair_reports_each_side_and_keeps_its_encode_outcome() {
            let mut peer = Peer::new(1);
            let now = Instant::now();
            let mut handle = answered(&mut peer, paired(), now)
                .await
                .expect("the pair is held");
            assert_eq!(
                handle.lease().expect("prefill lease").lease.side,
                Side::Prefill
            );
            let first = handle.take(0).expect("the first pair");
            assert_eq!(first.pair.expect("a decode").decode_id, WorkerId(47));
            let sent = now + Duration::from_micros(100);
            handle.dispatched(sent);
            let alternate = handle
                .take_decode(1)
                .expect("another decode of that prefill");
            let decode = alternate.pair.expect("a decode").decode_id;
            assert_eq!(decode, WorkerId(49));
            handle.completed(
                &crate::tracking::Outcome {
                    side: Side::Encode,
                    ..outcome(WorkerId(47), TransportClass::Ok)
                },
                sent,
            );
            let prefill = outcome(first.id, TransportClass::Ok);
            handle.completed(&prefill, sent);
            handle.completed(&prefill, sent);
            let ended = now + Duration::from_micros(200);
            let decoded = crate::tracking::Outcome {
                side: Side::Decode,
                ..outcome(decode, TransportClass::Ok)
            };
            handle.completed(&decoded, ended);
            handle.completed(&decoded, ended);
            drop(handle);
            reports(
                &mut peer,
                [
                    pair_dispatch(proto::Side::Prefill, 0, 100),
                    pair_completion(proto::Side::Encode, 47),
                    pair_completion(proto::Side::Prefill, 28),
                    pair_dispatch(proto::Side::Decode, 1, 200),
                    pair_completion(proto::Side::Decode, 49),
                ],
            )
            .await;
        }

        #[tokio::test(start_paused = true)]
        async fn a_linked_pair_releases_both_waiting_sides_or_just_its_waiting_decode() {
            let now = Instant::now();
            let mut peer = Peer::new(1);
            let mut handle = answered(&mut peer, paired(), now)
                .await
                .expect("the pair is held");
            handle.released(ReleaseReason::CommitFailed, now);
            drop(handle);
            reports(
                &mut peer,
                [released(proto::Side::Unspecified, Reason::CommitFailed)],
            )
            .await;

            let mut peer = Peer::new(1);
            let mut handle = answered(&mut peer, paired(), now)
                .await
                .expect("another pair");
            handle.completed(&outcome(WorkerId(28), TransportClass::Ok), now);
            drop(handle);
            reports(
                &mut peer,
                [
                    pair_dispatch(proto::Side::Prefill, 0, 0),
                    pair_completion(proto::Side::Prefill, 28),
                    released(proto::Side::Decode, Reason::Abandoned),
                ],
            )
            .await;
        }

        /// No link asks a decode later: the link's request drops the word,
        /// so its controller issues both sides, and the handle owes the
        /// decode its dispatch or its release. A decode call on such a
        /// handle is refused as not chosen, and nothing crosses the link
        /// for it.
        #[tokio::test(start_paused = true)]
        async fn a_linked_pair_asked_with_its_decode_later_is_held_as_issued_and_takes_no_call() {
            let now = Instant::now();
            let mut peer = Peer::new(1);
            let mut ask = request(&[1], 7);
            ask.facts.budget_micros = 200_000;
            ask.constraints.decode_later = true;
            let mut handle = answered_to(&mut peer, &ask, paired(), now)
                .await
                .expect("the pair is held");
            handle.dispatched(now);
            assert_eq!(
                handle.decide_decode(&ask, now).err(),
                Some(DecisionRefused::Decode(DecodeRefused::NotChosen))
            );
            handle.completed(&outcome(WorkerId(28), TransportClass::Ok), now);
            drop(handle);
            reports(
                &mut peer,
                [
                    pair_dispatch(proto::Side::Prefill, 0, 0),
                    pair_completion(proto::Side::Prefill, 28),
                    released(proto::Side::Decode, Reason::Abandoned),
                ],
            )
            .await;
        }

        #[tokio::test(start_paused = true)]
        async fn an_unreadable_pair_is_released_whole_whatever_its_labels() {
            let mut index = paired();
            index.leases[1].target_index = 2;
            // The prefill lease under the decode side's label: a release by
            // the labels would give the decode side back twice and keep the
            // prefill booking.
            let mut label = paired();
            label.leases[0].side = proto::Side::Decode.into();
            for (told, path) in [(index, "leases.target_index:"), (label, "leases")] {
                let mut peer = Peer::new(1);
                let answer = answered(&mut peer, told, Instant::now()).await;
                assert!(
                    matches!(&answer, Err(DecisionUnavailable::Unsent(Error::NoDecision(why))) if why.starts_with(path)),
                    "{answer:?}"
                );
                reports(
                    &mut peer,
                    [released(proto::Side::Unspecified, Reason::LateDecision)],
                )
                .await;
                assert!(peer.link.stands());
            }
        }

        #[tokio::test(start_paused = true)]
        async fn a_handle_can_report_after_its_link_ends() {
            let mut peer = Peer::new(1);
            let now = Instant::now();
            let mut handle = answered(&mut peer, picked(), now)
                .await
                .expect("the pick is held");
            peer.controller(
                peer.link.term_floor(),
                Some(controller_frame::Frame::NotLeader(
                    proto::NotLeader::default(),
                )),
            );
            assert!(timeout(PATIENCE, peer.reports.next())
                .await
                .expect("the session ended")
                .is_none());
            handle.dispatched(now);
            handle.completed(&outcome(WorkerId(28), TransportClass::Ok), now);
            assert!(!peer.link.stands());
        }

        #[tokio::test(start_paused = true)]
        async fn a_handle_passes_the_wait_budget_to_the_link() {
            let peer = Peer::new(1);
            let mut ask = request(&[1], 7);
            ask.facts.budget_micros = 200_000;
            let before = tokio::time::Instant::now();
            let answer = timeout(
                PATIENCE,
                pin!(DecisionGuard::<&StoreBackedEngine>::asked(
                    &peer.link,
                    &ask,
                    Instant::now(),
                    Duration::ZERO
                )),
            )
            .await
            .expect("the ask returns");
            assert!(
                matches!(answer, Err(DecisionUnavailable::Late)),
                "{answer:?}"
            );
            assert_eq!(tokio::time::Instant::now(), before);
        }

        /// Over a link a budget of 0 says the request's time is spent, and
        /// the controller answers busy at its receipt. The handle answers
        /// such a request late before anything is sent: no credit goes, and
        /// no time passes.
        #[tokio::test(start_paused = true)]
        async fn a_request_with_no_budget_is_late_before_anything_is_sent() {
            let mut peer = Peer::new(1);
            let ask = request(&[1], 7);
            assert_eq!(ask.facts.budget_micros, 0);
            let before = tokio::time::Instant::now();
            let answer = timeout(
                PATIENCE,
                pin!(DecisionGuard::<&StoreBackedEngine>::asked(
                    &peer.link,
                    &ask,
                    Instant::now(),
                    PATIENCE
                )),
            )
            .await
            .expect("the ask returns");
            assert!(
                matches!(answer, Err(DecisionUnavailable::Late)),
                "{answer:?}"
            );
            assert_eq!(tokio::time::Instant::now(), before);
            assert!(
                poll(pin!(peer.requests.next())).is_pending(),
                "nothing went out"
            );
        }

        /// This protocol version states no hard demand, and a controller
        /// that reads none picks among workers the demand rules out. The
        /// handle refuses a request that states one before anything is
        /// sent: no credit goes, and no time passes.
        #[tokio::test(start_paused = true)]
        async fn a_request_that_states_a_demand_is_refused_before_anything_is_sent() {
            let mut peer = Peer::new(1);
            let mut ask = request(&[1], 7);
            ask.facts.budget_micros = 200_000;
            ask.constraints.demands = Demands {
                runtime: 7,
                ..Demands::default()
            };
            let before = tokio::time::Instant::now();
            let answer = timeout(
                PATIENCE,
                pin!(DecisionGuard::<&StoreBackedEngine>::asked(
                    &peer.link,
                    &ask,
                    Instant::now(),
                    PATIENCE
                )),
            )
            .await
            .expect("the ask returns");
            assert!(
                matches!(
                    answer,
                    Err(DecisionUnavailable::Unsent(Error::OutOfRange(
                        "constraints.demands"
                    )))
                ),
                "{answer:?}"
            );
            assert_eq!(tokio::time::Instant::now(), before);
            assert!(
                poll(pin!(peer.requests.next())).is_pending(),
                "nothing went out"
            );
        }
    }
}
