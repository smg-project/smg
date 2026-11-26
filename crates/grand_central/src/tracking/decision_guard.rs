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
#[cfg(feature = "client")]
use crate::{
    engine::{
        framework::{EncodeItems, EncodePlaced},
        view::Phase,
    },
    service::{
        client::{reads_demands, wire, DecisionUnavailable, Error, Link},
        proto,
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
    /// Whether a stage of the document the decision was made under takes
    /// an end, as the call that placed the request stated it
    /// ([`Self::set_takes_end`]). `false` where no call stated it.
    #[cfg(feature = "in-process")]
    takes_end: bool,
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
            #[cfg(feature = "in-process")]
            takes_end: false,
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

    /// States whether a stage of the document the decision was made under
    /// takes an end. The call that places a request states it of the
    /// document it decided under, and the split of the placement reads it
    /// ([`Self::takes_end`]).
    #[cfg(feature = "in-process")]
    pub(crate) fn set_takes_end(&mut self, takes: bool) {
        self.takes_end = takes;
    }

    /// Whether a stage of the document the decision was made under takes
    /// an end ([`Self::set_takes_end`]). A document put in force since the
    /// decision changes nothing here: a word of the handle's lease goes to
    /// the stages that stood at its issue.
    #[cfg(feature = "in-process")]
    #[must_use]
    pub(crate) fn takes_end(&self) -> bool {
        self.takes_end
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
    /// For a pair this says the prefill's dispatch. A decode side issued
    /// with that prefill stands on a controller's file from here while the
    /// prefill runs, and has the deadline's span from the prefill's end.
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
    /// lease ends.
    /// A completion said with no dispatch says the dispatch first, at
    /// `now`: a request whose send failed was sent, and the engine hears a
    /// lease's events in their order. That dispatch carries the
    /// completion's instant, so a stage that times a request from its
    /// dispatch reads a span of zero for it. On a pair's handle a decode
    /// outcome ends the decode side. An encode outcome there is said to
    /// nobody: the pair's lease holds no such side and no stage hears
    /// one, so the handle says nothing and what it owes stands.
    /// Any other outcome ends the handle's own side, and a single's lease
    /// ends under any side its host names. A side's end is said once.
    pub fn completed(&mut self, outcome: &Outcome, now: Instant) {
        let pair = self.decided.pair();
        let owed = match outcome.side {
            Side::Decode if pair => {
                self.dispatched_decode(now);
                &mut self.decode
            }
            // A side the pair's lease does not hold: no file ends a side
            // for it and no stage hears it, so nothing is said.
            Side::Encode if pair => return,
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
    /// A single request that states a hard demand states it in its frame
    /// (`DecideRequest.demands`), and the controller keeps the workers that
    /// meet it. That takes a session whose welcome chose protocol 2, or a
    /// later version the link's hello offered. Any other request that
    /// states one is answered
    /// [`DecisionUnavailable::Unsent`] with [`Error::OutOfRange`] before
    /// anything is sent, and its host decides it itself:
    ///
    /// - a request over a session on protocol 1, whose controller may read no
    ///   demand and would then pick among workers the demand rules out, or
    ///   on a version the hello did not offer
    /// - a request at a side's phase, an encode request and one that names
    ///   its worker, which a controller holds to no demand yet
    /// - a request that demands vision, by that part's name, since no
    ///   registration over the wire states whether a worker takes images
    ///
    /// A shed keeps the stage of the demand that left nobody
    /// (`Shed::emptied_by`), as the door in process names it. Any other
    /// stage of a controller's shed reads `controller`.
    ///
    /// A request's places and its stalled and held rows are positions of
    /// its host's own list. They cross with no request, with a demand or
    /// without one (`Constraints::places`).
    ///
    /// Where the link's session changes under an ask, the link refuses the
    /// frame itself, by the frame's field `demands`.
    pub async fn asked(
        link: &Link,
        request: &Request<'_>,
        now: Instant,
        budget: Duration,
    ) -> Result<Self, DecisionUnavailable> {
        Self::asked_naming(link, request, None, now, budget).await
    }

    /// [`Self::asked`] for a request that may name its worker. `named` is
    /// what the target source of the request's binding answered for it
    /// ([`KeySources::target`](crate::document::KeySources::target)), which
    /// its host reads once, where it derives the request's keys: the
    /// number, `Some(None)` for a text that is no number, and `None` for a
    /// request that names no worker, which is asked as [`Self::asked`]
    /// asks it.
    ///
    /// The frame states the number (`DecideRequest.named_worker`). The
    /// controller counts it over its own list, from 0 and in that list's
    /// order, among the workers this gateway can be sent to: by its reach
    /// and wires, its standing vetoes, and each worker's own test of being
    /// available. The decision is held to the worker the number reaches,
    /// and the answer names that worker or sheds the request with no
    /// endpoint. The frame names `target-worker` for that shed, and this
    /// handle reads it as `controller`, as it reads every stage but a
    /// demand's ([`Self::asked`]).
    /// Where this gateway can be sent to no worker of the list, the request
    /// is decided as one that names no worker, with neither its session
    /// key nor its client's key, and the answer is that decision's. A
    /// controller older than the field reads none, and places the request
    /// by the rest of its document.
    ///
    /// A pair's frame states the one number for its two sides. The
    /// controller counts it as this process's one call counts a pair's
    /// ([`InProcess::place`](crate::engine::InProcess::place)),
    /// by those tests of its own, on each side whose profile is the plain
    /// hash ring: the prefills this gateway can be sent to that pair with a
    /// decode it can be sent to, those with such a decode among their first
    /// partners alone where there is one, then the partners so counted of
    /// the prefill reached. The decision is held to the two workers, or to
    /// the prefill alone where the decode's side does not read the number.
    /// Where the decode's side alone is that ring, the decision counts the
    /// number among the partners of the prefill it picks. A number that
    /// reaches no worker on a side sheds the pair with no endpoint, and so
    /// does a named worker that cannot take the request after all. Where
    /// no prefill is counted, or none opens the pair, the pair is decided
    /// as one that names no worker, and keeps its keys. Where every decode
    /// is over its thresholds and the binding steers such a pool, the
    /// decode goes by load and its number is not read. An encode request's
    /// frame is read for no number.
    ///
    /// The host's own rule for a worker is not asked there, and the
    /// controller's list need not stand in the order of the host's own. A
    /// host that decides by itself for lack of an answer counts over its
    /// own list, by its own rule.
    ///
    /// # Errors
    ///
    /// Those of [`Self::asked`].
    pub async fn asked_naming(
        link: &Link,
        request: &Request<'_>,
        named: Option<Option<u32>>,
        now: Instant,
        budget: Duration,
    ) -> Result<Self, DecisionUnavailable> {
        Self::sendable(link, request, named.is_some())?;
        Self::answered(link, wire::request(request, named), now, budget, |_, _| {
            Ok(())
        })
        .await
    }

    /// [`Self::asked`] for an encode request, its pair's one request: the
    /// frame states its `items`, and `placed` takes each item's encoder
    /// from the answer, in the keys' order. The two are the arguments the
    /// engine's entry takes in process
    /// ([`InProcess::decide_encode_at`](crate::engine::InProcess::decide_encode_at)),
    /// and the handle is a pair's: no lease is an encoder's.
    ///
    /// `request` is at the prefill phase, as that entry asks. `placed` is
    /// emptied at each ask and stays empty unless a pick is read. A linked
    /// encoder names its worker and its rank
    /// ([`EncodePlaced::encoders`]).
    ///
    /// A ring lent for the items ([`EncodeItems::ring`]) does not cross:
    /// a frame states none, as it states none of the request's own, and
    /// the controller places each key by the ring its own stage keeps.
    ///
    /// # Errors
    ///
    /// As [`Self::asked`]. A request of another phase is answered
    /// [`DecisionUnavailable::Unsent`] with [`Error::OutOfRange`] before
    /// anything is sent, as the door in process refuses a side sent on
    /// its own. An answer whose encoders are not one where it says so,
    /// else one a key, or that names a pair on an encoder, is one the
    /// handle cannot read.
    pub async fn asked_encode(
        link: &Link,
        request: &Request<'_>,
        items: EncodeItems<'_>,
        placed: &mut EncodePlaced,
        now: Instant,
        budget: Duration,
    ) -> Result<Self, DecisionUnavailable> {
        placed.clear();
        Self::sendable(link, request, false)?;
        if request.facts.phase != Phase::Prefill {
            return Err(DecisionUnavailable::Unsent(Error::OutOfRange(
                "facts.phase",
            )));
        }
        let asked = wire::encode_request(request, items);
        let beside = |told: Option<&proto::EncodePlaced>, decision: &Decision| {
            wire::placed(told, decision, &items, placed)
        };
        Self::answered(link, asked, now, budget, beside).await
    }

    /// What a handle refuses of a request before anything is sent: a
    /// budget of 0, and a hard demand its controller would not hold
    /// ([`Self::demands_sendable`]). `named` is whether the request names
    /// its worker.
    fn sendable(
        link: &Link,
        request: &Request<'_>,
        named: bool,
    ) -> Result<(), DecisionUnavailable> {
        if request.facts.budget_micros == 0 {
            return Err(DecisionUnavailable::Late);
        }
        if request.constraints.demands.is_none() {
            return Ok(());
        }
        Self::demands_sendable(link, request, named)
    }

    /// Whether a request's hard demands go out in its frame. They do for a
    /// single request that names no worker, over a session whose welcome
    /// chose a protocol that states them. The part that demands vision
    /// never does. Out of line: a request that demands nothing reads none
    /// of it.
    #[cold]
    #[inline(never)]
    fn demands_sendable(
        link: &Link,
        request: &Request<'_>,
        named: bool,
    ) -> Result<(), DecisionUnavailable> {
        let unsent = |field| Err(DecisionUnavailable::Unsent(Error::OutOfRange(field)));
        // A controller on protocol 1 reads no demand, and a welcome on a
        // version the hello did not offer says nothing of it. One on
        // protocol 2 holds it on a single request's frame that names no
        // worker. So no demand is dropped on the way.
        if !reads_demands(link.welcome().protocol) || request.facts.phase != Phase::Single || named
        {
            return unsent("constraints.demands");
        }
        // No registration over the wire states whether a worker takes
        // images, so a controller sheds a request that demands it unless
        // its own host laid such a worker.
        if request.constraints.demands.vision {
            return unsent("constraints.demands.vision");
        }
        Ok(())
    }

    /// The frame `asked` sent over `link` and its answer held. `beside`
    /// reads what the answer carries beside its decision. An answer that
    /// it, or the handle, cannot read has its leases released as late.
    async fn answered(
        link: &Link,
        (asked, left_out): (proto::DecideRequest, usize),
        now: Instant,
        budget: Duration,
        beside: impl FnOnce(Option<&proto::EncodePlaced>, &Decision) -> Result<(), wire::Unread>,
    ) -> Result<Self, DecisionUnavailable> {
        let (told, encode) = link.ask_cut(asked, left_out, budget).await?;
        let read = wire::decided(&told, now)
            .and_then(|decided| beside(encode.as_ref(), &decided.decision).map(|()| decided));
        match read {
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
#[path = "../../tests/unit/tracking/decision_guard/tests/mod.rs"]
mod tests;
