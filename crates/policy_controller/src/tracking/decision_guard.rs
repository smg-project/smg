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

#[cfg(feature = "in-process")]
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
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
        decision_engine::{
            DecideOutcome, DecisionEngine, DecisionRefused, DecodeRefused, HeldDecision,
            HeldDecode, Leased, Request, RequestMaker,
        },
        view::WorkerId,
    },
    store::GatewayId,
    tracking::lease::{
        Dispatch, Lease, LeaseId, Outcome, ReleaseReason, Said, Side, TransportClass,
    },
};

/// The three sizes a side's words state: its lease's token count and its
/// text's two lengths ([`Lease::tokens`], [`Lease::chars`] and
/// [`Lease::bytes`]).
#[derive(Clone, Copy, Default)]
struct Stated {
    tokens: u32,
    chars: u32,
    bytes: u32,
}

impl Stated {
    const fn of(lease: &Lease) -> Self {
        Self {
            tokens: lease.tokens,
            chars: lease.chars,
            bytes: lease.bytes,
        }
    }
}

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
    /// The three sizes the decode side's words state: the lease's own, and
    /// a decode call's from the call that picks the side. Nothing reads a
    /// single's.
    stated: Stated,
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
    /// the door decides under ([`DecisionEngine::decide_under`]), where the
    /// host hands no single list of its own: the door's answer is then
    /// whole. A host that hands one asks its door itself, since a place
    /// may answer it with nothing to hold, and holds a whole answer by
    /// [`DecisionGuard::holding`].
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
        let decided = door.decide_under(gateway, maker, now)?;
        let (decided, later) = decided.whole().ok_or(DecisionRefused::Faulted)?;
        Ok(Self::holding(door, gateway, decided, later, now))
    }

    /// Holds `decided`, a whole answer its host asked of `door` itself at
    /// `now` ([`Decided::Whole`](crate::engine::Decided::Whole)), with what
    /// the answer said of the later decode.
    pub fn holding(
        door: D,
        gateway: GatewayId,
        decided: DecideOutcome,
        later: bool,
        now: Instant,
    ) -> Self {
        Self::hold(Arm::Engine { door, gateway }, decided, later, now)
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
        // The sizes the decode side's words state until a decode call picks
        // the side: the lease's own.
        let stated = decided.lease.map(|leased| Stated::of(&leased.lease));
        let stated = stated.unwrap_or_default();
        Self {
            arm,
            decided,
            index: 0,
            later,
            stated,
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
            index: self.index,
            target: &self.target,
            // Lent as the handle keeps them, with no test of the lease's
            // side: a single's answer names no decode session, its sizes are
            // the lease's own and its decode side never stands chosen.
            decode: HeldDecode {
                session: self.decided.decode_session,
                tokens: self.stated.tokens,
                chars: self.stated.chars,
                bytes: self.stated.bytes,
                chosen: matches!(self.decode, Owed::Later { .. }),
            },
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
        self.decode_asked(call)?;
        Ok(self.decision())
    }

    /// Asks the handle's door by `call` for the decode side of its pair's
    /// lease. A pick is taken, its first target with it, and the answer is
    /// `None`. Where the door picked nothing the handle stands as it was,
    /// and the answer is the door's own: its shed, or its queue.
    fn decode_asked(
        &mut self,
        call: impl FnOnce(
            &D::Target,
            GatewayId,
            &HeldDecision<'_>,
        ) -> Result<DecideOutcome, DecisionRefused>,
    ) -> Result<Option<Decision>, DecisionRefused> {
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
        // issues and refuses a call past the count's end. A handle holds
        // one issue of a side at a time: a placement's call that takes no
        // decode releases the side it issued before the handle asks again
        // ([`Self::decode_placed_under`]). It leaves that count to the
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
            // The call issued the side under the three sizes it stated.
            self.stated = Stated::of(&leased.lease);
            (self.index, self.target) = (0, *target);
            self.decode = self.decode.sent_later();
            if let Some(held) = &mut self.decided.lease {
                held.due = leased.due;
            }
            return Ok(None);
        }
        Ok(Some(answer.decision))
    }

    /// [`Self::decide_decode_under`] for a placement, which takes a decode
    /// itself and answers its host a shed by name
    /// ([`Placement::decode_again`](super::Placement::decode_again)).
    ///
    /// `sendable` is asked of the targets of the call's pick, best first,
    /// and the first it admits is taken ([`Self::take_decode`]): the
    /// answer is then `None`. Otherwise the answer is the decision the
    /// call made, and the handle stands as it stood before the call, on
    /// the first decision's target with its decode chosen and booked
    /// nowhere, so its host may ask again. That decision is the door's own
    /// shed or queue, or a pick none of whose targets `sendable` admits:
    /// the side such a pick issued is released by name
    /// ([`ReleaseReason::CommitFailed`]).
    ///
    /// What stands again is the handle's own note of the pair. The engine
    /// of a gateway's own process keeps no other, and a placement is held
    /// by no other door: a door that keeps a file of leases answers a
    /// second ask by its file.
    ///
    /// What `sendable` asks can panic, a route's own rule among it. The
    /// side the call issued is then released by the same name, the handle
    /// stands as it stood before the call, and the panic goes on to the
    /// caller: a walk leaves no lease behind, and its host may still send
    /// the first decision's decode or ask again.
    ///
    /// # Errors
    ///
    /// As [`Self::decide_decode`].
    #[cfg(feature = "in-process")]
    pub(crate) fn decode_placed_under(
        &mut self,
        maker: &impl RequestMaker,
        now: Instant,
        mut sendable: impl FnMut(&Target) -> bool,
    ) -> Result<Option<Decision>, DecisionRefused> {
        let (decision, lease) = (self.decided.decision, self.decided.lease);
        let before = (self.decided.decode_session, self.stated, self.index);
        let (target, chosen) = (self.target, self.decode);
        let asked = self.decode_asked(|door, gateway, held| {
            door.decide_decode_under(gateway, maker, held, now)
        })?;
        if asked.is_some() {
            return Ok(asked);
        }
        let pick = self.decided.decision.pick();
        let walked = catch_unwind(AssertUnwindSafe(|| {
            pick.and_then(|pick| pick.targets.iter().position(&mut sendable))
        }));
        let first = walked.as_ref().map_or(None, |first| *first);
        if first.and_then(|index| self.take_decode(index)).is_some() {
            return Ok(None);
        }
        self.released_decode(ReleaseReason::CommitFailed, now);
        let unsent = self.decided.decision;
        (self.decided.decision, self.decided.lease) = (decision, lease);
        (self.decided.decode_session, self.stated, self.index) = before;
        (self.target, self.decode) = (target, chosen);
        match walked {
            Ok(_) => Ok(Some(unsent)),
            Err(panic) => resume_unwind(panic),
        }
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

    /// Whether the handle still owes its lease a word on some side: a
    /// dispatch, an end, or the release in their place. A handle that
    /// owes nothing says nothing at its drop.
    #[must_use]
    pub(crate) fn owes(&self) -> bool {
        !matches!((self.owed, self.decode), (Owed::Nothing, Owed::Nothing))
    }

    /// Whether the handle is of a pair that asked its decode later and owes
    /// nothing on either side any more: both sides ended, or the prefill
    /// ended with the decode still chosen, which left with it. A decode's
    /// send said then meets no side.
    #[cfg(feature = "in-process")]
    #[must_use]
    pub(crate) fn later_ended(&self) -> bool {
        self.later && !self.owes()
    }

    /// The engine of this process the handle decides by. `None` for a
    /// handle over a link.
    #[cfg(feature = "in-process")]
    #[cfg_attr(
        not(feature = "client"),
        expect(
            clippy::unnecessary_wraps,
            reason = "a handle over a link alone has no engine"
        )
    )]
    #[must_use]
    pub(crate) fn door(&self) -> Option<&D> {
        match &self.arm {
            Arm::Engine { door, .. } => Some(door),
            #[cfg(feature = "client")]
            Arm::Link(_) => None,
        }
    }

    /// The target the handle names at its dispatch: the pick's first, or
    /// the one a take moved to. A pair's decode is in it.
    #[cfg(feature = "in-process")]
    #[must_use]
    pub(crate) fn target(&self) -> &Target {
        &self.target
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
    /// It stands in line in its two callers: a host says a dispatch once
    /// a request.
    #[inline]
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

    /// Gives up the prefill side alone, a single's one side, as
    /// [`Self::released_decode`] gives up the decode's. A sent prefill
    /// completes as cancelled on the worker it was sent to, with the span
    /// from its dispatch to `now`. An issued prefill is released by name
    /// under `reason`. A decode that still stands chosen leaves with it,
    /// as it leaves with the prefill's end ([`Self::completed`]).
    #[cfg(feature = "in-process")]
    pub(crate) fn released_prefill(&mut self, reason: ReleaseReason, now: Instant) {
        self.owed = self.release(self.owed, reason, now);
        if self.decode_chosen() {
            self.decode = Owed::Nothing;
        }
    }

    /// Whether the prefill's last word would take the pair's decode with
    /// it: the prefill is still owed one, its end where it went out and
    /// its release where it never did, and the decode still stands chosen
    /// ([`Self::decode_chosen`]).
    #[cfg(feature = "in-process")]
    #[must_use]
    pub(crate) fn end_takes_decode(&self) -> bool {
        !matches!(self.owed, Owed::Nothing) && self.decode_chosen()
    }

    /// The span from `side`'s dispatch to `at`, in whole microseconds: what
    /// an end of class cancelled states of a side given up at `at`
    /// ([`Self::released`]). Zero for a side whose dispatch was not said,
    /// and for one whose end was.
    #[cfg(feature = "in-process")]
    #[must_use]
    pub(crate) fn since_sent(&self, side: Side, at: Instant) -> u32 {
        let owed = match side {
            Side::Decode if self.decided.pair() => self.decode,
            _ => self.owed,
        };
        if let Owed::Completion { sent, .. } = owed {
            micros(at.saturating_duration_since(sent))
        } else {
            0
        }
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
                    duration_micros: micros(now.saturating_duration_since(sent)),
                    ..Outcome::reported(0, TransportClass::Cancelled)
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
        if !self.owes() {
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

// The engine over worker lists stands under the feature `in-process`: the
// cases that hold a guard over it need the feature, and so does every
// fixture they share. The feature's gate is an attribute of its own: the
// lints read a module of cases by its plain `cfg(test)`.
#[cfg(test)]
#[cfg(feature = "in-process")]
mod tests {
    use std::{
        sync::{Arc, Barrier},
        thread,
        time::{Duration, Instant},
    };

    use parking_lot::Mutex;
    use smg_worker::{identity::basic::Worker, test_support::ready_worker};

    use super::DecisionGuard;
    #[cfg(feature = "client")]
    use crate::engine::decision_engine::{DecisionRefused, DecodeRefused};
    use crate::{
        document::spec::{AppliedSpec, PolicySpec},
        engine::{
            decision::{Decision, Target, Targets},
            decision_engine::{Request, RequestMaker, WorkerLists},
            framework::{Picker, TieBand},
            list_cell::tests::PublishedList,
            policy_engine::PolicyEngine,
            view::{Candidate, Constraints, RequestFacts, WorkerId},
        },
        policies::random::Random,
        stages::{lifecycle::Lifecycle, registry::StageRegistry},
        store::GatewayId,
        tracking::lease::{EventSet, Lease, Outcome, ReleaseReason, Side, TransportClass},
    };

    /// The model of the list's workers.
    const MODEL: &str = "model-a";

    /// How long a case waits for an answer it expects back. The cases of the
    /// client's link alone read it, so it stands under their feature.
    #[cfg(feature = "client")]
    const PATIENCE: Duration = Duration::from_secs(20);

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
        fn hears(&self) -> EventSet {
            EventSet::LEASE
        }

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

    /// The engine over worker lists under the `random` preset with
    /// [`Hears`] in its pick's place, the list of four ready workers its
    /// host hands it, and what the picker heard.
    struct Host {
        engine: PolicyEngine,
        members: PublishedList,
        heard: Heard,
    }

    impl Host {
        /// One decision over the host's list for the request `id`, held.
        fn held(&self, id: &[u8], now: Instant) -> DecisionGuard<&PolicyEngine> {
            let asks = Listed(stated(id), Some(&self.members));
            let decided = DecisionGuard::decide_under(&self.engine, GatewayId::LOCAL, &asks, now);
            decided.expect("a decision")
        }
    }

    /// A host with an engine, a list and a hearing picker of its own.
    fn host() -> Host {
        let heard = Heard::default();
        let kept = Arc::clone(&heard);
        let mut stages = StageRegistry::builtin();
        let added = stages.picker("hears", move || Box::new(Hears(Arc::clone(&kept))));
        added.expect("a free name");
        let mut document = PolicySpec::preset("random").expect("a built-in preset");
        document.generation = 1;
        let profile = document.profiles.get_mut("random");
        profile.expect("the preset's profile").pick = "hears".into();
        let spec = AppliedSpec::validate_in(&document, &stages, &[]);
        let engine = PolicyEngine::new(spec.expect("the stages hold the picker"));
        let list: Arc<[Arc<dyn Worker>]> = (0..4)
            .map(|at| ready_worker(&format!("http://worker-{at}:8000"), MODEL))
            .collect();
        let members = PublishedList::new(list);
        Host {
            engine,
            members,
            heard,
        }
    }

    /// A request for [`MODEL`] under `id`, stated ahead.
    fn stated(id: &[u8]) -> Request<'_> {
        Request {
            id,
            facts: RequestFacts {
                model: MODEL,
                seed: 7,
                ..RequestFacts::default()
            },
            constraints: Constraints::default(),
        }
    }

    /// A request stated ahead, with the list its host hands for it.
    struct Listed<'a>(Request<'a>, Option<&'a PublishedList>);

    impl RequestMaker for Listed<'_> {
        fn model(&self) -> &str {
            self.0.facts.model
        }

        fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
            self.0
        }

        fn workers(&self) -> WorkerLists<'_> {
            self.1
                .map_or(WorkerLists::None, |list| WorkerLists::Single(list.lent()))
        }
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
        const HELD: usize = 504;
        assert_eq!(size_of::<DecisionGuard<Arc<PolicyEngine>>>(), HELD);
        assert_eq!(size_of::<DecisionGuard<&PolicyEngine>>(), HELD);
        // A placement is the handle and what its host reads of the answer:
        // two worker pointers, the routing key, the profile's label, the
        // label of a pair's decode side and one pointer to an encode
        // request's encoders.
        assert_eq!(size_of::<crate::tracking::Placement<'_>>(), HELD + 88);
        // The lease the handle keeps states its request's text's two
        // lengths beside its token count: eight bytes more, the handle's
        // too. The second length sits in the padding the first left. The
        // handle keeps the two lengths the decode side's words state
        // beside that side's token count: eight bytes more.
        assert_eq!(size_of::<Lease>(), 72);
    }

    /// A request that is served: its lease is issued on the pick's first
    /// target, dispatched at the host's word and ended at its completion.
    /// Each is told once: what the host says after the end, and the drop,
    /// say nothing.
    #[test]
    fn a_served_request_tells_its_dispatch_and_its_completion_once_each() {
        let host = host();
        let now = Instant::now();
        let mut handle = host.held(b"r-1", now);
        let issued = handle.lease().map(|leased| leased.lease.booked);
        let first = handle.take(0).expect("the pick's first target");
        assert_eq!(issued, Some(first.id), "issued on the pick's first target");
        handle.dispatched(now + Duration::from_micros(250));
        let ended = now + Duration::from_millis(5);
        handle.completed(&outcome(first.id, TransportClass::Ok), ended);
        let told = [
            Call::Issued,
            Call::Dispatched(first.id, 0, 250),
            Call::Completed(first.id, TransportClass::Ok, 5_000),
        ];
        assert_eq!(*host.heard.lock(), told);
        // Nothing more is owed: what its host says behind the end is told
        // to nobody.
        let later = ended + Duration::from_secs(1);
        handle.dispatched(later);
        handle.completed(&outcome(first.id, TransportClass::Ok), later);
        handle.released(ReleaseReason::ClientGone, Instant::now());
        drop(handle);
        assert_eq!(*host.heard.lock(), told, "one completion is told");
    }

    /// A host whose commit passes over the first target takes another: a
    /// take says nothing, and the dispatch names the target taken under its
    /// index in the pick. A take of no target of the pick, and one behind
    /// the dispatch, change nothing.
    #[test]
    fn a_dispatch_on_an_alternate_names_the_target_taken() {
        let host = host();
        let now = Instant::now();
        let mut handle = host.held(b"r-1", now);
        let first = handle.take(0).expect("the pick's first target");
        let second = handle.take(1).expect("an alternate of the pick");
        assert_ne!(first.id, second.id);
        assert_eq!(handle.take(9), None, "no such target");
        assert_eq!(*host.heard.lock(), [Call::Issued], "a take says nothing");
        // Two hours behind the decision: more microseconds than the span's
        // field holds, so it says its most.
        handle.dispatched(now + Duration::from_secs(7_200));
        assert_eq!(handle.take(0), None, "the dispatch named its target");
        handle.completed(&outcome(second.id, TransportClass::Ok), now);
        let told = [
            Call::Issued,
            Call::Dispatched(second.id, 1, u32::MAX),
            Call::Completed(second.id, TransportClass::Ok, 5_000),
        ];
        assert_eq!(*host.heard.lock(), told);
    }

    /// A host whose commit refuses every target releases the lease under
    /// that reason, once: nothing it says behind the release is told.
    #[test]
    fn a_commit_that_refuses_every_target_releases_the_lease_once() {
        let host = host();
        let now = Instant::now();
        let mut handle = host.held(b"r-1", now);
        let first = handle.take(0).expect("the pick's first target");
        handle.released(ReleaseReason::CommitFailed, Instant::now());
        let told = [Call::Issued, Call::Released(ReleaseReason::CommitFailed)];
        assert_eq!(*host.heard.lock(), told);
        handle.released(ReleaseReason::ClientGone, Instant::now());
        handle.completed(&outcome(first.id, TransportClass::Ok), now);
        drop(handle);
        assert_eq!(*host.heard.lock(), told, "no completion behind a release");
    }

    /// A handle dropped with no call knows no reason for its drop: before
    /// any dispatch its lease is given up as abandoned, the word for a
    /// lease nobody spoke for.
    #[test]
    fn a_handle_dropped_with_no_call_gives_its_lease_up_as_abandoned() {
        let host = host();
        let handle = host.held(b"r-1", Instant::now());
        assert_eq!(*host.heard.lock(), [Call::Issued]);
        drop(handle);
        let told = [Call::Issued, Call::Released(ReleaseReason::Abandoned)];
        assert_eq!(*host.heard.lock(), told);
    }

    /// A host that knows its client left before any dispatch says so, and
    /// the handle's drop behind it says nothing more.
    #[test]
    fn a_host_that_knows_its_client_left_says_so_once() {
        let host = host();
        let mut handle = host.held(b"r-1", Instant::now());
        handle.released(ReleaseReason::ClientGone, Instant::now());
        let told = [Call::Issued, Call::Released(ReleaseReason::ClientGone)];
        assert_eq!(*host.heard.lock(), told);
        drop(handle);
        assert_eq!(*host.heard.lock(), told);
    }

    /// After a dispatch a lease lives until its completion: a handle
    /// dropped with no outcome, and a release said there, are told as one
    /// completion of class cancelled on the dispatched target, which says
    /// the time since the dispatch.
    #[test]
    fn behind_a_dispatch_a_drop_and_a_release_are_a_cancelled_completion() {
        for released in [false, true] {
            let host = host();
            let now = Instant::now();
            let mut handle = host.held(b"r-1", now);
            let second = handle.take(1).expect("an alternate of the pick");
            handle.dispatched(now);
            thread::sleep(Duration::from_millis(1));
            let told = [
                Call::Issued,
                Call::Dispatched(second.id, 1, 0),
                Call::Completed(second.id, TransportClass::Cancelled, 0),
            ];
            if released {
                handle.released(ReleaseReason::ClientGone, Instant::now());
                assert_eq!(clocked(&host.heard, 1_000), told, "told where it was said");
            }
            drop(handle);
            assert_eq!(clocked(&host.heard, 1_000), told, "released: {released}");
        }
    }

    /// A request whose send failed was sent: a completion said with no
    /// dispatch says the dispatch first, on the target taken and at the
    /// completion's instant, so the engine hears the lease's events in
    /// their order.
    #[test]
    fn a_completion_said_with_no_dispatch_says_the_dispatch_first() {
        let host = host();
        let now = Instant::now();
        let mut handle = host.held(b"r-1", now);
        let second = handle.take(1).expect("an alternate of the pick");
        let failed = outcome(second.id, TransportClass::ConnectFailed);
        handle.completed(&failed, now + Duration::from_micros(400));
        let told = [
            Call::Issued,
            Call::Dispatched(second.id, 1, 400),
            Call::Completed(second.id, TransportClass::ConnectFailed, 5_000),
        ];
        assert_eq!(*host.heard.lock(), told);
    }

    /// A decision that picks no worker has no lease: its handle takes no
    /// target and tells nothing, whatever its host says.
    #[test]
    fn a_decision_that_picks_no_worker_owes_no_report() {
        let host = host();
        let now = Instant::now();
        // A worker of the list, by a decision that picks one.
        let mut picked = host.held(b"r-0", now);
        let member = picked.take(0).expect("the pick's first target").id;
        picked.released(ReleaseReason::CommitFailed, Instant::now());
        host.heard.lock().clear();
        // A request whose host hands no list is decided over no worker.
        let asks = Listed(stated(b"r-1"), None);
        let decided = DecisionGuard::decide_under(&host.engine, GatewayId::LOCAL, &asks, now);
        let mut handle = decided.expect("a decision");
        assert!(handle.decision().pick().is_none(), "no list, no pick");
        assert!(handle.lease().is_none());
        assert_eq!(handle.take(0), None);
        handle.dispatched(now);
        handle.completed(&outcome(member, TransportClass::Ok), now);
        handle.released(ReleaseReason::CommitFailed, Instant::now());
        drop(handle);
        assert_eq!(*host.heard.lock(), []);
    }

    /// A host that takes no target dispatches to the pick's first, and a
    /// handle let go behind that dispatch completes the lease as cancelled
    /// on that target. The completion's span counts from the dispatch: one
    /// said ahead of the clock has no span yet.
    #[test]
    fn a_host_that_takes_no_target_dispatches_to_the_picks_first() {
        let host = host();
        let now = Instant::now();
        let mut handle = host.held(b"r-1", now);
        let pick = handle.decision().pick();
        let first = pick.and_then(|pick| pick.targets.first().copied());
        let first = first.expect("the pick's first target");
        handle.dispatched(now + Duration::from_secs(3_600));
        thread::sleep(Duration::from_millis(1));
        drop(handle);
        let told = [
            Call::Issued,
            Call::Dispatched(first.id, 0, 3_600_000_000),
            Call::Completed(first.id, TransportClass::Cancelled, 0),
        ];
        assert_eq!(*host.heard.lock(), told);
    }

    /// A handle that owns its engine travels with its request: dropped on
    /// another thread behind its dispatch, it ends the lease from there.
    #[test]
    fn a_handle_that_owns_its_engine_ends_its_lease_from_another_thread() {
        let Host {
            engine,
            members,
            heard,
        } = host();
        let engine = Arc::new(engine);
        let now = Instant::now();
        let asks = Listed(stated(b"r-1"), Some(&members));
        let decided =
            DecisionGuard::decide_under(Arc::clone(&engine), GatewayId::LOCAL, &asks, now);
        let mut handle = decided.expect("a decision");
        let first = handle.take(0).expect("the pick's first target");
        handle.dispatched(now);
        let ended = thread::spawn(move || drop(handle));
        ended.join().expect("the request's thread");
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
    /// every lease is ended once, by its own path's word.
    #[test]
    fn sixty_four_hosts_at_once_end_every_lease_once() {
        let shared = host();
        let (hosts, each) = (64_u8, 32_u8);
        let start = Barrier::new(usize::from(hosts));
        thread::scope(|scope| {
            for host in 0..hosts {
                let (shared, start) = (&shared, &start);
                scope.spawn(move || {
                    start.wait();
                    for turn in 0..each {
                        let now = Instant::now();
                        let mut handle = shared.held(&[host, turn], now);
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
        let heard = shared.heard.lock();
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

    /// A completion names the worker its host names, which need not be the
    /// target the handle took: a stage reads the worker that served.
    #[test]
    fn a_completion_names_the_worker_its_host_names() {
        let host = host();
        let now = Instant::now();
        let mut handle = host.held(b"r-1", now);
        let other = handle.take(1).expect("the pick's second target");
        let taken = handle.take(0).expect("the pick's first target");
        assert_ne!(other.id, taken.id);
        handle.dispatched(now);
        handle.completed(&outcome(other.id, TransportClass::Ok), now);
        let told = [
            Call::Issued,
            Call::Dispatched(taken.id, 0, 0),
            Call::Completed(other.id, TransportClass::Ok, 5_000),
        ];
        assert_eq!(*host.heard.lock(), told);
    }

    /// A release behind a dispatch spans its host's two instants, the
    /// dispatch's and its own: no clock is read.
    #[test]
    fn a_release_behind_a_dispatch_spans_its_hosts_instants() {
        let host = host();
        let now = Instant::now();
        let mut handle = host.held(b"r-1", now);
        let first = handle.take(0).expect("the pick's first target");
        let sent = now + Duration::from_secs(3_600);
        handle.dispatched(sent);
        handle.released(ReleaseReason::ClientGone, sent + Duration::from_millis(7));
        let told = [
            Call::Issued,
            Call::Dispatched(first.id, 0, 3_600_000_000),
            Call::Completed(first.id, TransportClass::Cancelled, 7_000),
        ];
        assert_eq!(*host.heard.lock(), told);
    }

    mod pairs;

    /// A single's handle holds no decode side: a decode's take, dispatch
    /// and release say nothing. Its lease ends under any side its host
    /// names, a decode outcome among them: a decode phase can stand under
    /// a single plan, and the outcome its host read reaches the engine.
    #[test]
    fn a_single_handle_ends_under_a_decode_outcome_and_says_nothing_else_of_a_decode_side() {
        let host = host();
        let now = Instant::now();
        let mut handle = host.held(b"single", now);
        let first = handle.take(0).expect("the first target");
        handle.dispatched_decode(now);
        assert_eq!(handle.take_decode(1), None);
        handle.released_decode(ReleaseReason::ClientGone, now);
        assert!(handle.lease().is_some(), "the single's lease stands");
        assert_eq!(*host.heard.lock(), [Call::Issued]);
        let decoded = Outcome {
            side: Side::Decode,
            ..outcome(first.id, TransportClass::Ok)
        };
        handle.completed(&decoded, now + Duration::from_micros(400));
        let told = [
            Call::Issued,
            Call::Dispatched(first.id, 0, 400),
            Call::Completed(first.id, TransportClass::Ok, 5_000),
        ];
        assert_eq!(*host.heard.lock(), told);
        drop(handle);
        assert_eq!(*host.heard.lock(), told, "the drop says nothing more");
    }

    #[cfg(feature = "client")]
    mod linked {
        use std::pin::pin;

        use tokio::time::timeout;
        use tokio_stream::StreamExt as _;

        use super::{
            outcome, stated, DecisionGuard, DecisionRefused, DecodeRefused, Duration, Instant,
            PolicyEngine, Request, Side, TransportClass, WorkerId, PATIENCE,
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
        ) -> Result<DecisionGuard<&'static PolicyEngine>, DecisionUnavailable> {
            let mut ask = stated(&[1]);
            ask.facts.budget_micros = 200_000;
            answered_to(peer, &ask, told, now).await
        }

        /// [`answered`] for the request `ask`, of id 1 and a budget of 200 ms.
        async fn answered_to(
            peer: &mut Peer,
            ask: &Request<'_>,
            told: proto::Decision,
            now: Instant,
        ) -> Result<DecisionGuard<&'static PolicyEngine>, DecisionUnavailable> {
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
            let mut ask = stated(&[1]);
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
            let mut ask = stated(&[1]);
            ask.facts.budget_micros = 200_000;
            let before = tokio::time::Instant::now();
            let answer = timeout(
                PATIENCE,
                pin!(DecisionGuard::<&PolicyEngine>::asked(
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
            let ask = stated(&[1]);
            assert_eq!(ask.facts.budget_micros, 0);
            let before = tokio::time::Instant::now();
            let answer = timeout(
                PATIENCE,
                pin!(DecisionGuard::<&PolicyEngine>::asked(
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
            let mut ask = stated(&[1]);
            ask.facts.budget_micros = 200_000;
            ask.constraints.demands = Demands {
                runtime: 7,
                ..Demands::default()
            };
            let before = tokio::time::Instant::now();
            let answer = timeout(
                PATIENCE,
                pin!(DecisionGuard::<&PolicyEngine>::asked(
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
