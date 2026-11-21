//! The lease of a decision, and what the calls of a stage's lifecycle carry:
//! the outcome, a membership change, a worker's signal. The trait a stage
//! implements to hear them is [`Lifecycle`](crate::stages::Lifecycle).
//!
//! A lease records that a decision was issued and is expected to be
//! dispatched and completed. A `credit` is soft load accounting; a
//! `reservation` is a prefill slot or a decode room the host holds
//! exclusively. A pair decision carries one lease per side, and the host
//! calls the lifecycle once per side. Every stateful built-in keeps its state
//! behind these events alone.

use std::fmt;

use crate::{
    engine::{
        sticky_sessions::SessionVerdict,
        view::{Health, LoadRow, Session, WorkerId},
    },
    tracking::in_flight::WorkerSlot,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct LeaseId(pub u64);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LeaseKind {
    /// Soft load accounting. The built-in least-load scorer releases it at the
    /// next report sampled after the dispatch; release at completion and at
    /// the dispatch deadline arrive with the lifecycle step.
    #[default]
    Credit,
    /// A prefill slot or a decode room, released by its side's completion.
    Reservation,
}

/// Which side of a dispatch a lease or an event is about.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Side {
    #[default]
    Single,
    Prefill,
    Decode,
    Encode,
}

/// The record of one issued decision, or of one side of a pair decision.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Lease {
    pub id: LeaseId,
    pub kind: LeaseKind,
    /// The side this lease books; `Single` outside a pair.
    pub side: Side,
    /// Which target of the pick this lease books (a reservation on a pair
    /// side names its own).
    pub target_index: u8,
    /// The worker the lease books. The decider's mint names the worker its
    /// issue books, and the lease as issued, and as given up where its
    /// answer does not leave, carries that word. Where the served store
    /// tells of a lease on its file, at a dispatch and at an end, it names
    /// its record's target: the same worker until a dispatch moves the
    /// lease, and the worker it moved to behind one. A stage that books at
    /// the issue reads it at the dispatch ([`Lifecycle::on_dispatched`]),
    /// where `target_index` names the target the dispatch took.
    /// `WorkerId(0)` where a host names none. A controller answer gives
    /// the worker by `target_index` into its pick.
    ///
    /// [`Lifecycle::on_dispatched`]: crate::stages::Lifecycle::on_dispatched
    pub booked: WorkerId,
    pub generation: u64,
    /// The request's session ([`crate::engine::RequestFacts::session`]: its key
    /// within its model and phase), which the host fills for every keyed
    /// request, whatever rule governed it, so the commit moves that
    /// session's pins and a sticky in-flight cap counts per session per
    /// worker from leases; [`Session::NONE`] when the request carried no
    /// key. The wire carries [`Session::get`] in its `session_key` field.
    pub session: Session,
    /// What the decision did with the session key, copied from its pick, so
    /// the commit moves the key's pins by its own decision alone.
    pub session_verdict: SessionVerdict,
    /// The host's clock at issue, microseconds since its origin. In-process
    /// only; the wire carries the dispatch deadline as a duration.
    pub issued_micros: u64,
    /// The prompt size the request stated, in tokens: 0 where it stated
    /// none. The `least-load` and `resident-work` stages credit a dispatch
    /// of no stated size at their own mean prefill; every other reader
    /// takes the 0 as stated.
    pub tokens: u32,
}

/// The transport outcome of a dispatch, beside its status: what a health
/// rule reads when it weighs a prober's verdict against the gateways'
/// reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransportClass {
    Ok,
    ClientError,
    ServerError,
    ConnectFailed,
    Timeout,
    Cancelled,
}

/// What a completed dispatch reported.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Outcome {
    pub side: Side,
    pub served: WorkerId,
    /// The HTTP status or the gRPC code the request ended with.
    pub status: u16,
    pub class: TransportClass,
    pub ttft_micros: u32,
    pub tokens_in: u32,
    pub tokens_out: u32,
    /// Microseconds the dispatch took, saturating at `u32::MAX`, never wrapping.
    pub duration_micros: u32,
}

/// Why a lease was released before its completion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReleaseReason {
    /// The lease ended before a dispatch: its host gave up, no dispatch
    /// was named by its deadline, or its answer was stopped for the fence
    /// (its controller no longer led the term that minted it).
    Abandoned,
    /// A late decision: it arrived after a fallback dispatch, or was cut
    /// before it left, for its request's budget or its dispatch span.
    LateDecision,
    /// The client went away.
    ClientGone,
    /// The commit refused every target (a removed worker, a lost room).
    CommitFailed,
    /// The stream rule's grace passed without a completion.
    Expired,
    /// An immediate deregistration released the worker's leases.
    WorkerRemoved,
}

/// A membership change the host reports. A host tells a join when it takes
/// it, and a view that lists the row may follow: a stage that lists by the
/// feed treats a member its candidates do not hold as one a filter took.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Membership<'a> {
    Added(WorkerId),
    Removed(WorkerId),
    /// The same registration key with a new incarnation.
    Replaced {
        old: WorkerId,
        new: WorkerId,
    },
    Relabelled(WorkerId),
    /// The worker's data-parallel width changed.
    Resized(WorkerId),
    /// Every worker of every pool the engine decides over, after a resync;
    /// a host with a prefill and a decode pool sends one snapshot of both.
    Snapshot(&'a [WorkerId]),
    /// The worker left its pool of each of `models`: its role moved, or it
    /// serves them no longer. `id` stands when the word is said. A
    /// registration that keeps its id says it ahead of its own word, the
    /// [`Membership::Relabelled`] or [`Membership::Resized`]. A new
    /// incarnation says its [`Membership::Replaced`] first and this word
    /// behind it, for the new id. A stage that lists workers by model
    /// unlists the worker under these alone. Both plugin contracts carry
    /// the word: the guest's world as the case `left`, the remote one as a
    /// kind a stream of protocol 2 is sent.
    Left {
        id: WorkerId,
        models: &'a [String],
    },
}

/// A signal about one worker. The named values a document declares under
/// `signals` are the other thing of that name: numbers read beside the
/// rows ([`SignalCells`](crate::load::SignalCells)).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Signal {
    /// A load row for one rank of a worker, as the host shows it to its
    /// decisions.
    Load {
        worker: WorkerId,
        dp_rank: u16,
        row: LoadRow,
        /// The place of the rank's count in the host's counts: a stage that
        /// keeps a number a member finds the member's cell by it.
        /// In-process only; [`WorkerSlot::NONE`] from a host that keeps no
        /// counts.
        slot: WorkerSlot,
        /// The sample instant of `row` on the host's clock, in microseconds
        /// since the host's origin, as the rank's row names it to a
        /// decision. In-process only; the wire carries the row's age.
        sampled_at_micros: u64,
    },
    Health {
        worker: WorkerId,
        health: Health,
    },
    /// A gateway's standing veto on a worker, until the given instant on the
    /// host's clock. In-process only; the wire carries a duration.
    GatewayVeto {
        worker: WorkerId,
        until_micros: u64,
    },
}

/// One kind of lifecycle event: one of the six calls of [`Lifecycle`], as
/// both plugin contracts name them.
///
/// [`Lifecycle`]: crate::stages::Lifecycle
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventKind {
    Issued,
    Dispatched,
    Completed,
    Released,
    Membership,
    Signal,
}

impl EventKind {
    /// Every kind, in the order of its bit.
    pub const ALL: [Self; 6] = [
        Self::Issued,
        Self::Dispatched,
        Self::Completed,
        Self::Released,
        Self::Membership,
        Self::Signal,
    ];

    const fn bit(self) -> u8 {
        1 << self as u8
    }
}

/// A set of kinds of lifecycle event, as bits over [`EventKind`]. Empty by
/// default. A plugin's row lists in one the kinds its plugin hears, and a
/// plugin whose row lists nothing is told nothing. It prints the kinds it
/// holds by name, as `{Completed, Signal}`.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct EventSet(u8);

impl EventSet {
    /// No event.
    pub const NONE: Self = Self(0);

    /// Every kind.
    pub const ALL: Self = {
        let mut set = Self::NONE;
        let mut at = 0;
        while at < EventKind::ALL.len() {
            set.0 |= EventKind::ALL[at].bit();
            at += 1;
        }
        set
    };

    /// The four kinds told of one lease: its issue, its dispatch, its
    /// completion and its release.
    pub const LEASE: Self = Self::NONE
        .with(EventKind::Issued)
        .with(EventKind::Dispatched)
        .with(EventKind::Completed)
        .with(EventKind::Released);

    /// The set with `kind` in it.
    #[must_use]
    pub const fn with(self, kind: EventKind) -> Self {
        Self(self.0 | kind.bit())
    }

    /// The kinds either set holds.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether the two sets hold a kind in common.
    #[must_use]
    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    /// The set of these kinds.
    pub fn of(kinds: impl IntoIterator<Item = EventKind>) -> Self {
        let mut set = Self::NONE;
        for kind in kinds {
            set.0 |= kind.bit();
        }
        set
    }

    pub const fn contains(self, kind: EventKind) -> bool {
        self.0 & kind.bit() != 0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl fmt::Debug for EventSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set()
            .entries(EventKind::ALL.iter().filter(|kind| self.contains(**kind)))
            .finish()
    }
}

#[cfg(feature = "controller")]
pub(crate) mod wire;

#[cfg(test)]
mod tests {

    use super::*;

    /// The type's list of every kind, held to the order of the bits. The
    /// match names each, so a new kind does not compile until it has its
    /// place here, which is its place in [`EventKind::ALL`]: the one list
    /// that the set of every kind, the set's print and the cases read.
    fn kinds() -> [EventKind; 6] {
        let kinds = EventKind::ALL;
        for (at, kind) in kinds.into_iter().enumerate() {
            let place = match kind {
                EventKind::Issued => 0,
                EventKind::Dispatched => 1,
                EventKind::Completed => 2,
                EventKind::Released => 3,
                EventKind::Membership => 4,
                EventKind::Signal => 5,
            };
            // Its place in the list is its bit.
            assert_eq!((place, EventSet::of([kind]).0), (at, 1 << at), "{kind:?}");
        }
        kinds
    }

    #[test]
    fn an_event_set_holds_the_kinds_it_was_made_of_and_no_other() {
        let kinds = kinds();
        assert_eq!(EventSet::default(), EventSet::NONE);
        assert!(EventSet::NONE.is_empty() && !EventSet::ALL.is_empty());
        assert_eq!(EventSet::of(kinds), EventSet::ALL);
        for kind in kinds {
            let alone = EventSet::of([kind]);
            assert!(EventSet::ALL.contains(kind) && !EventSet::NONE.contains(kind));
            for other in kinds {
                assert_eq!(alone.contains(other), other == kind, "{kind:?} {other:?}");
            }
        }
    }

    /// A set prints the kinds it holds by name, in the order of the bits: a
    /// row in a log line or a failing case names what the row lists.
    #[test]
    fn an_event_set_prints_the_kinds_it_holds_by_name() {
        let two = EventSet::of([EventKind::Signal, EventKind::Completed]);
        assert_eq!(format!("{two:?}"), "{Completed, Signal}");
        assert_eq!(format!("{:?}", EventSet::NONE), "{}");
        assert_eq!(
            format!("{:?}", EventSet::ALL),
            "{Issued, Dispatched, Completed, Released, Membership, Signal}"
        );
    }

    #[test]
    fn a_pair_books_one_lease_per_side_on_its_own_target() {
        let prefill = Lease {
            side: Side::Prefill,
            target_index: 0,
            kind: LeaseKind::Reservation,
            ..Lease::default()
        };
        let decode = Lease {
            side: Side::Decode,
            target_index: 0,
            ..prefill
        };
        assert_ne!(prefill.side, decode.side);
        assert_eq!(prefill.target_index, decode.target_index);
        assert_eq!(decode.kind, LeaseKind::Reservation);
    }
}
