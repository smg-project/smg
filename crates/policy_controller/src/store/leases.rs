//! The leases store: every issued decision, per gateway, with the term of
//! the leader that issued it (design 4.3, 5.1 the leases row, 5.3 fencing).

use std::time::Instant;

/// The table of counts the leases on file book and a member's place in
/// it, named here for the store's files that write it: their home is the
/// counts' own module.
pub(super) use crate::tracking::in_flight::{HostCount, InFlightCounts, WorkerSlot};
use crate::{
    engine::{
        decision::{
            Decision, LiftedVeto, Pick, Survivors, Target, Targets, Trace, MAX_TARGETS, TRACE_TOP,
        },
        decision_engine::{DecodeRefused, HeldDecision, HeldDecode},
        sticky_sessions::SessionVerdict,
        view::{Session, WorkerId},
    },
    store::GatewayId,
    tracking::lease::{Dispatch, Lease, LeaseId, Outcome, ReleaseReason, Said, Side},
};

/// The state of a single's lease or of one side of a pair's lease.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeaseState {
    /// Decided, not yet dispatched; the dispatch deadline bounds this.
    Issued,
    /// Dispatched to its target at `at`.
    Dispatched {
        at: Instant,
    },
    Completed,
    Released(ReleaseReason),
    /// A pair's decode targets before issue. The side books no count and
    /// has no dispatch deadline. A lease record's own state is never chosen.
    Chosen,
}

impl LeaseState {
    /// Whether the decode side holds a choice that has yet to be issued or
    /// dispatched.
    pub const fn is_chosen(self) -> bool {
        matches!(self, Self::Chosen)
    }

    /// Whether a frame still moves a lease that stands so: issued or
    /// dispatched.
    pub const fn is_live(self) -> bool {
        matches!(self, Self::Issued | Self::Dispatched { .. })
    }

    /// The issued state, by the name its callers write.
    pub const ISSUED: Self = Self::Issued;

    /// Whether the lease still waits for its dispatch.
    pub fn is_issued(self) -> bool {
        matches!(self, Self::Issued)
    }
}

/// What a lease's record keeps of its pick beside the target it books: the
/// pick's other targets and every target's rank, so a `Dispatched` that
/// names a target by its index finds the worker. Kept inline: a record is
/// filed inside a decision's clock. The default is a pick of one target at
/// rank 0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PickRecord {
    /// The pick's targets behind its first, in the pick's order;
    /// `WorkerId(0)` is no target.
    rest: [WorkerId; MAX_TARGETS - 1],
    /// Each target's rank, by its index in the pick.
    ranks: [u16; MAX_TARGETS],
}

impl PickRecord {
    /// What a record keeps of the pick `targets`.
    pub fn of(targets: &Targets) -> Self {
        let mut picked = Self::default();
        for (target, rank) in targets.iter().zip(picked.ranks.iter_mut()) {
            *rank = target.dp_rank;
        }
        for (target, id) in targets.iter().skip(1).zip(picked.rest.iter_mut()) {
            *id = target.id;
        }
        picked
    }

    /// What a pair's decode side keeps of the pick `targets`: the decode
    /// of each target behind the first and every target's decode rank. A
    /// target that names no pair keeps none.
    pub fn of_decodes(targets: &Targets) -> Self {
        let mut picked = Self::default();
        let decodes = || targets.iter().map(|target| target.pair.unwrap_or_default());
        for (decode, rank) in decodes().zip(picked.ranks.iter_mut()) {
            *rank = decode.decode_dp_rank;
        }
        for (decode, id) in decodes().skip(1).zip(picked.rest.iter_mut()) {
            *id = decode.decode_id;
        }
        picked
    }

    /// The worker and the rank at `index` of the pick, for the side that
    /// books `booked` at the index `own`; `None` for an index the pick
    /// holds no target at.
    fn at(&self, booked: WorkerId, own: u8, index: u8) -> Option<(WorkerId, u16)> {
        let place = usize::from(index);
        let rank = *self.ranks.get(place)?;
        if index == own {
            return Some((booked, rank));
        }
        // A side books the pick's first target at its issue: the others
        // stand behind it.
        let id = *self.rest.get(place.checked_sub(1)?)?;
        (id != WorkerId(0)).then_some((id, rank))
    }
}

/// The decode side of a pair's lease, as the store keeps it beside the
/// lease's record ([`Sides`]). The sides share the lease's id and generation.
/// Each has its own target, state, and, when live, booking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeSide {
    /// The first decode target of the pick, or the worker the side moved
    /// to at dispatch.
    pub target: WorkerId,
    /// The decodes of the pick's other targets and every target's decode
    /// rank, for that `Dispatched`.
    pub picked: PickRecord,
    /// The request's session under the decode phase.
    pub session: Session,
    /// Where the side stands.
    pub state: LeaseState,
    /// The token-work the side carries when issued.
    pub tokens: u32,
    /// The index of the side's target in the pick.
    pub target_index: u8,
}

impl DecodeSide {
    /// The worker and the rank a `Dispatched` on the side names by its
    /// target `index`: the side's target at its own index,
    /// else the decode of the pick's target at that index
    /// ([`LeaseRecord::target_at`], for this side).
    pub fn target_at(&self, index: u8) -> Option<(WorkerId, u16)> {
        self.picked.at(self.target, self.target_index, index)
    }

    /// Builds the side's lease from the shared fields on `prefill` and
    /// the decode side's own fields. [`Lease::booked`] names the decode
    /// target at the read, including any move at dispatch. A dispatch
    /// hook for an issued side reads the lease as it stood at issue.
    pub fn lease(&self, prefill: &Lease) -> Lease {
        Lease {
            side: Side::Decode,
            target_index: self.target_index,
            booked: self.target,
            session: self.session,
            tokens: self.tokens,
            ..*prefill
        }
    }
}

/// The mark a pair's record carries: its lease has a decode side, which
/// stands beside the record in the file's list of places. It names no
/// worker: the side's own target is the one place its worker is read
/// ([`Sides`]), and a dispatch that moves the side moves it there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeMark;

/// One lease as the store holds it: a single's whole, or the prefill part
/// of a pair's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LeaseRecord {
    pub lease: Lease,
    /// The gateway the lease was issued to; the pair (gateway, lease id) is
    /// the key.
    pub gateway: GatewayId,
    /// The request id's fingerprint, for `Explain`.
    pub request: u64,
    /// The worker the lease books: the pick's first target, until a
    /// `Dispatched` names another one ([`LeaseRecord::target_at`]).
    pub target: WorkerId,
    /// The pick's other targets and the ranks, for that `Dispatched`.
    pub picked: PickRecord,
    /// The mark of a pair's lease. `None` marks a single's lease and no
    /// pair's: a pair is filed with its decode side
    /// ([`LeasesStore::issue_pair`]), and the door that files it sets the
    /// mark, as a door that files a record alone clears it. The mark says
    /// that a decode side stands beside the record and nothing of that
    /// side: its worker, its state and its booking are the side's own
    /// word ([`Sides`]).
    pub decode: Option<DecodeMark>,
    /// The lease's request is kept from its profile: the binding's sessions
    /// rule placed it alone, and the rule says such a request is none of
    /// the profile's own ([`RulePlaced::Untold`]). The decider that files
    /// the lease writes the word from the decision in hand
    /// (`lease_events::stages_hear`), and tells no stage the issue. The
    /// store's teller reads it at the dispatch: the session table hears,
    /// and no stage and no count of sends does. An end and a release are
    /// told as any lease's. `false` on every other record, a pair's among
    /// them: each side of a pair keeps a verdict of its own, and a pick's
    /// one word does not say which side the rule placed. The door that
    /// files a pair clears the word, as it writes the pair's mark.
    ///
    /// No replication entry of a lease's record exists yet, so none is
    /// written for the word: the entry that carries a record will carry
    /// it. A record a gateway's `Hello` page lists is filed with the word
    /// its caller wrote ([`LeasesStore::restore`]).
    ///
    /// [`RulePlaced::Untold`]: crate::document::spec::RulePlaced::Untold
    pub untold: bool,
    /// The term of the leader that issued the lease.
    pub term: u64,
    pub issued_at: Instant,
    /// Past it, an issued side is released as abandoned.
    pub dispatch_deadline: Instant,
    /// The state of a single's lease or a pair's prefill side.
    pub state: LeaseState,
}

impl LeaseRecord {
    /// The worker and the rank a `Dispatched` names by its target `index`:
    /// the target the lease books at the lease's own index, else the
    /// pick's target at that index. `None` for an index the pick holds no
    /// target at, and for the pick's first target once a dispatch moved
    /// the lease: the record keeps the target it books in that one's
    /// place. No caller asks then, since a lease moves once.
    pub fn target_at(&self, index: u8) -> Option<(WorkerId, u16)> {
        self.picked.at(self.target, self.lease.target_index, index)
    }

    /// The record's own lease, a single's or a pair's prefill side, as the
    /// store tells of it at a dispatch and at an end: it names the worker
    /// the record books ([`Lease::booked`]), whatever the lease on file
    /// names. The record's target is the one place the store reads it
    /// from, and a dispatch that moves the lease moves it there.
    fn own(&self) -> Lease {
        Lease {
            booked: self.target,
            ..self.lease
        }
    }
}

/// A lease on file, both its parts as one hold of the file reads them:
/// the record and, where the record carries the mark of a pair's lease,
/// its decode side. The prefill side of a pair is the record's own lease,
/// target and state. What a frame on a side names is read here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sides {
    pub record: LeaseRecord,
    /// The decode side of a pair's lease; `None` for a single's.
    pub decode: Option<DecodeSide>,
}

impl Sides {
    /// The worker and the rank a `Dispatched` on `side` names by its
    /// target `index`: on a pair's decode side the decode of that target,
    /// else [`LeaseRecord::target_at`].
    pub fn target_of(&self, side: Side, index: u8) -> Option<(WorkerId, u16)> {
        match self.decode {
            Some(decode) if side == Side::Decode => decode.target_at(index),
            _ => self.record.target_at(index),
        }
    }

    /// The lease `side` names and where that side stands. A pair holds a
    /// prefill side, the record's own lease and state, and a decode side;
    /// `None` for another side. A single's lease reads no side: it is its
    /// one lease under any. Each side's lease names its own target. The
    /// state says whether the side is live.
    pub fn side(&self, side: Side) -> Option<(Lease, LeaseState)> {
        let record = &self.record;
        let Some(decode) = self.decode else {
            return Some((record.own(), record.state));
        };
        match side {
            Side::Prefill => Some((record.own(), record.state)),
            Side::Decode => Some((decode.lease(&record.lease), decode.state)),
            Side::Single | Side::Encode => None,
        }
    }

    /// The lease of each side a frame on `side` ends: the side named
    /// where it is issued or dispatched, and with `None` every such side.
    /// A single's lease ends whole under any side.
    pub fn ending(&self, side: Option<Side>) -> [Option<Lease>; 2] {
        let live = |side| {
            self.side(side)
                .and_then(|(lease, state)| state.is_live().then_some(lease))
        };
        match (self.decode, side) {
            (None, _) => [Some(self.record.own()), None],
            (Some(_), Some(side)) => [live(side), None],
            (Some(_), None) => [live(Side::Prefill), live(Side::Decode)],
        }
    }

    /// Calls `told` with the lease of each side a frame on `side` ends
    /// ([`Self::ending`]). A single's lease is told off its record: it
    /// reads no side and builds no list.
    pub fn each_ending(&self, side: Option<Side>, mut told: impl FnMut(&Lease)) {
        if self.decode.is_none() {
            return told(&self.record.own());
        }
        for lease in self.ending(side).iter().flatten() {
            told(lease);
        }
    }

    /// Calls `told` with a single's lease as a host that held its decision
    /// would lend it with `said` ([`HeldDecision`]): the view a request's
    /// guard lends its engine, made from the record on file. `None`, and
    /// no call, for a pair's lease, which is no single's, and for a
    /// dispatch that names no target of the pick: it tells nothing of the
    /// lease.
    ///
    /// The parts are read as they stood before the word:
    ///
    /// - **The lease** is the record's own as it stands: the lease as
    ///   issued until a dispatch, and behind one the lease on the index and
    ///   the worker the dispatch took, which every word behind a dispatch
    ///   is told under.
    /// - **The target** is the one the word is about. A dispatch names it
    ///   by its frame's index. An end and a release name the one the lease
    ///   books.
    /// - **The pick** is one of no target: no word of a single carries its
    ///   pick to a stage. It says what the record says of who is told
    ///   ([`LeaseRecord::untold`]), the one word of a pick a single's
    ///   telling reads.
    /// - **The decode side** is a single's, as a guard lends it: no
    ///   session, the lease's own three sizes, never chosen.
    ///
    /// The view is made in this call's frame and lent there: no part of it
    /// is moved, and no pick is built.
    #[inline]
    pub fn with_held_single<R>(
        &self,
        said: Said,
        told: impl FnOnce(&HeldDecision<'_>) -> R,
    ) -> Option<R> {
        if self.decode.is_some() {
            return None;
        }
        let record = &self.record;
        let own = record.lease.target_index;
        let (index, (id, dp_rank)) = match said {
            Said::Dispatched(dispatch, _) => {
                let index = dispatch.target_index;
                (index, record.target_at(index)?)
            }
            // An end is of the worker the lease books, whatever its index.
            Said::Completed(_) | Said::Released(..) => {
                let booked = record.target_at(own);
                (own, booked.unwrap_or((record.target, 0)))
            }
        };
        let lease = record.own();
        let target = Target {
            id,
            dp_rank,
            ..Target::default()
        };
        Some(told(&HeldDecision {
            decision: if record.untold {
                &UNTOLD_PICK
            } else {
                &EMPTY_PICK
            },
            lease: &lease,
            index,
            target: &target,
            decode: HeldDecode {
                session: Session::NONE,
                tokens: lease.tokens,
                chars: lease.chars,
                bytes: lease.bytes,
                chosen: false,
            },
        }))
    }
}

/// The pick lent with a single's lease ([`Sides::with_held_single`]): no
/// target, under no trace. A file keeps of a single's pick the workers a
/// dispatch names by its index, and lists none of them for a word: no word
/// of a single carries its pick to a stage.
static EMPTY_PICK: Decision = lent_pick(Targets::new());

/// [`EMPTY_PICK`] for a lease whose request is kept from its profile
/// ([`LeaseRecord::untold`]): the pick says so, as the one a host that held
/// the decision lends.
static UNTOLD_PICK: Decision = lent_pick(Targets::untold());

/// A pick of `targets` under no trace, with no verdict on a session key.
const fn lent_pick(targets: Targets) -> Decision {
    Decision::Pick(Pick {
        targets,
        trace: Trace {
            profile: Trace::NO_PROFILE,
            survivors: Survivors::counted(0),
            top: [(WorkerId(0), 0.0); TRACE_TOP],
            top_len: 0,
            seed: 0,
            generation: 0,
            lifted: LiftedVeto::None,
        },
        session_verdict: SessionVerdict::None,
    })
}

/// Why a lease was not recorded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeaseRefused {
    /// The record's term is above the store's: the issuer has not noticed it
    /// lost, or this replica has not noticed it won.
    TermAhead { lease: u64, store: u64 },
    /// The (gateway, lease id) pair is on file already.
    Duplicate,
    /// The store mints no id under the record's term: the term is not the
    /// store's own any more, is past the width an id carries it in, or its
    /// counter is spent (the contract's bounds: a term below 2^31 over a
    /// counter of 1 to 2^32 - 1). Equal terms say the term itself mints no
    /// more.
    Unminted { lease: u64, store: u64 },
    /// The file of the record's id is not settled yet: the id is of the
    /// store's term from a block of the counter that no file claimed, or
    /// of a term the store can still take. No decider minted it, and the
    /// block's claim would name its file anew with the lease on file. The
    /// same id is taken once a claim settles its file. A store of one
    /// file refuses none by it: there every id names that file.
    Unclaimed,
}

/// Where leases are recorded.
///
/// A lease carries the term of the leader that issued it; a record whose term
/// is above the store's own is refused, one below is kept (a lease the old
/// leader issued keeps its lifetime on the new one, learned from the gateway's
/// `Hello` pages, which restore exactly what they list and release what they
/// omit). A gateway gone past its grace period releases every lease of its
/// own. Reservations and pins are replicated; credits are rebuilt from
/// `Hello`.
pub trait LeasesStore: Send + Sync {
    /// The store's term: the leader's, from the elector; `0` before any.
    fn term(&self) -> u64;

    /// The elector's new term; never below the current one.
    fn set_term(&self, term: u64);

    /// Files `record`, a single's lease. A record alone is a single's: the
    /// mark of a pair's lease is cleared on it.
    fn issue(&self, record: LeaseRecord) -> Result<(), LeaseRefused>;

    /// Files a pair's lease: `record`, its prefill part, with `decode`,
    /// its decode side, under one hold. The record takes the mark of a
    /// pair's lease. A chosen decode side is filed without a booking or
    /// a dispatch deadline of its own. A pair that is handed to another
    /// store travels by this door, with its decode side.
    fn issue_pair(&self, record: LeaseRecord, decode: DecodeSide) -> Result<(), LeaseRefused>;

    /// Issues the pair's chosen decode side once, while its prefill is
    /// dispatched and still held. The side takes `decode`'s targets,
    /// session, and tokens, books its first target, and has a new dispatch
    /// deadline starting at `now`.
    ///
    /// If `decode` carries an issued state's session TTL, the new side
    /// keeps it.
    fn issue_decode(
        &self,
        gateway: GatewayId,
        lease: LeaseId,
        decode: DecodeSide,
        now: Instant,
    ) -> Result<(), DecodeRefused>;

    /// The gateway dispatched the lease as `dispatch` says, at `at`;
    /// `false` when the lease is unknown. A first dispatch that names
    /// another target of the pick moves the lease to it: the record's
    /// index, and where that target is another worker the record's target
    /// and the count the lease books. On a pair's lease the dispatch moves
    /// the side it names and no other: a decode's on another target moves
    /// the decode's booking alone, and a side the lease does not hold
    /// changes nothing. A dispatch on a chosen decode side books the
    /// target it names. At an index the pick holds no target at, that
    /// side stays chosen and books none.
    fn dispatched(
        &self,
        gateway: GatewayId,
        lease: LeaseId,
        dispatch: Dispatch,
        at: Instant,
    ) -> bool;

    /// A side completed. Returns `false` when the lease is unknown. A
    /// pair's live side ends alone and gives its count back. The lease
    /// and any chosen side leave the file at the last live side's end.
    fn completed(&self, gateway: GatewayId, lease: LeaseId, outcome: &Outcome) -> bool;

    /// The gateway gave the lease up, on `side` or, with `None`, on every
    /// side (a pair's leases share an id); `false` when the lease is
    /// unknown. A second release of one lease changes nothing.
    fn released(
        &self,
        gateway: GatewayId,
        lease: LeaseId,
        side: Option<Side>,
        reason: ReleaseReason,
    ) -> bool;

    /// The record of the lease: a single's whole, or the prefill part of
    /// a pair's ([`Self::sides`] reads both parts).
    fn get(&self, gateway: GatewayId, lease: LeaseId) -> Option<LeaseRecord>;

    /// Both parts of the lease, read under one hold: its record and, for
    /// a pair's, its decode side.
    fn sides(&self, gateway: GatewayId, lease: LeaseId) -> Option<Sides>;

    /// Releases issued sides past their dispatch deadline at `now` as
    /// abandoned and returns both parts of each lease as the call left
    /// them. A file whose decider is at work is left to its decider, who
    /// ends due leases at its issues. The store tells the lifecycle of
    /// each end itself.
    fn expire(&self, now: Instant) -> Vec<Sides>;

    /// Restore what a gateway's `Hello` pages list, releasing what they
    /// omit; a record above the store's term is skipped, and so is one
    /// whose id names no file yet ([`LeaseRefused::Unclaimed`]).
    /// Returns how many stand after it. A record alone is a single's: a
    /// page lists no decode side, and the mark of a pair's lease is
    /// cleared on what it lists.
    fn restore(&self, gateway: GatewayId, records: &[LeaseRecord]) -> usize;

    /// The gateways are gone past their grace: every lease of each released
    /// as client gone, and with them what the store holds of each gateway's
    /// own about a worker (its vetoes, its completed dispatches). One write
    /// a table for them all: one pass over the leases, one over the vetoes
    /// and one publication. Returns how many leases each held, in the order
    /// of `gone`. It is the store's one door for a gone gateway.
    fn drop_gateways(&self, gone: &[GatewayId]) -> Vec<usize>;

    /// The leases of a gateway that are issued or dispatched.
    fn live(&self, gateway: GatewayId) -> usize;
}
