//! The lifecycle (design 4.3): how a stateful stage learns what became of a
//! decision.
//!
//! A lease records that a decision was issued and is expected to be
//! dispatched and completed. A `credit` is soft load accounting; a
//! `reservation` is a prefill slot or a decode room the host holds
//! exclusively. A pair decision carries one lease per side, and the host
//! calls the lifecycle once per side. Every stateful built-in keeps its state
//! behind these events alone.

use std::fmt;

use crate::{
    counts::Slot,
    decision::{Decision, Target},
    sessions::SessionVerdict,
    view::{Health, LoadRow, Session, WorkerId},
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
    pub booked: WorkerId,
    pub generation: u64,
    /// The request's session ([`crate::RequestFacts::session`]: its key
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
/// rows ([`SignalCells`](crate::SignalCells)).
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
        /// In-process only; [`Slot::NONE`] from a host that keeps no
        /// counts.
        slot: Slot,
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

/// The lifecycle a stateful stage implements. Every hook defaults to a
/// no-op, so a stateless stage implements nothing.
///
/// A served controller tells a lease's issue, its dispatch, its completion
/// and its release while it holds the turn of the lease's file, and the
/// release of a lease its decision gives up right after that turn: those
/// hooks are bounded, and call no door of the leases' store. A hook that
/// panics at a dispatch, a completion or a release that a frame brings,
/// or at a dispatch deadline's end, is held there: the lease's
/// event stands as applied, and so does one that panics at the issue a
/// dispatch brings. One that panics at a decision's issue, or at the
/// release of a lease its decision gives up, costs that request its
/// decision.
pub trait Lifecycle: Send + Sync {
    /// The engine issued `decision` under `lease`. Its host filed the
    /// lease as it issued it, in process as behind a served controller:
    /// the lease books its first target from here.
    ///
    /// A pair's decode side whose request asked for it later is not issued
    /// with the pair, and no call here names it then. A decode call issues
    /// it and is told here, where the lease's generation still stands. A
    /// side no call issued goes out as the first pick's, and its dispatch
    /// is its issue: a store that files leases tells it here just ahead of
    /// that dispatch's own word. The decision told then is the pair's pick
    /// as the lease's record keeps it, every target's workers and ranks
    /// with no position and no trace. The lease names the target the
    /// dispatch took, and its instant is the dispatch's.
    fn on_decision_issued(&self, _lease: &Lease, _decision: &Decision) {}

    /// The host dispatched to `target` (one of the decision's targets), on
    /// `side`, `since_decision_micros` after the decision on the host's clock
    /// (the wire's own field): a credit dates from the dispatch, not from a
    /// decision a queue may have held. The untaken targets are released by
    /// the host's commit. The lease told names the target the dispatch
    /// took: its `target_index` is that target's index in the pick, and its
    /// [`Lease::booked`] is still the worker its issue booked. A decode
    /// side no call issued was told issued just ahead of this word, on the
    /// target it names ([`Self::on_decision_issued`]).
    fn on_dispatched(
        &self,
        _lease: &Lease,
        _target: &Target,
        _side: Side,
        _since_decision_micros: u32,
    ) {
    }

    /// A dispatched side finished, whatever the status.
    fn on_completed(&self, _lease: &Lease, _outcome: &Outcome) {}

    /// The host gave the lease up before its completion.
    fn on_released(&self, _lease: &Lease, _reason: ReleaseReason) {}

    /// The host's membership changed. A host may call this under its own
    /// membership lock, as the policy store does: a stage that calls from
    /// here a door of the host that takes that lock waits on itself. A read
    /// that takes no lock does not.
    fn on_membership(&self, _change: &Membership<'_>) {}

    fn on_signal(&self, _signal: &Signal) {}

    /// Whether the stage keeps a number of each rank's report, which a host
    /// tells it before it shows a decision that report ([`Signal::Load`]
    /// with the rank's slot and the report's sample instant). `false` by
    /// default: a host tells no report to an engine none of whose stages
    /// keeps one, and such a stage decides the same untold. The engine asks
    /// once, at its build: the answer is the stage's for the engine's life.
    /// A host tells such a stage under its own locks, as the policy store
    /// does under its membership and load locks: a stage that calls from
    /// that word a door of the host that takes one of them waits on itself.
    fn keeps_loads(&self) -> bool {
        false
    }

    /// Whether the stage decides wrong under a host that makes none of
    /// these calls: `true` for one whose state the calls keep, as the
    /// workers it lists or the bookings it learns from. The standalone
    /// controller makes the membership call and tells a stage of a lease's
    /// issue, of its dispatch, completion and release at its gateway's word
    /// and of its end at the dispatch deadline or where its answer does not
    /// leave, and not yet of the end of a lease whose gateway was let go: it
    /// refuses a document that names such a stage as not in this build
    /// ([`crate::SpecError::NotInBuild`]). A host that embeds the engine
    /// makes every call. `false` by default: a stage that takes a call to
    /// shed state early decides the same without it.
    fn needs_lifecycle(&self) -> bool {
        false
    }
}

/// A boxed stage forwards to the stage it holds.
impl<T: Lifecycle + ?Sized> Lifecycle for Box<T> {
    fn on_decision_issued(&self, lease: &Lease, decision: &Decision) {
        (**self).on_decision_issued(lease, decision);
    }

    fn on_dispatched(
        &self,
        lease: &Lease,
        target: &Target,
        side: Side,
        since_decision_micros: u32,
    ) {
        (**self).on_dispatched(lease, target, side, since_decision_micros);
    }

    fn on_completed(&self, lease: &Lease, outcome: &Outcome) {
        (**self).on_completed(lease, outcome);
    }

    fn on_released(&self, lease: &Lease, reason: ReleaseReason) {
        (**self).on_released(lease, reason);
    }

    fn on_membership(&self, change: &Membership<'_>) {
        (**self).on_membership(change);
    }

    fn on_signal(&self, signal: &Signal) {
        (**self).on_signal(signal);
    }

    fn keeps_loads(&self) -> bool {
        (**self).keeps_loads()
    }

    fn needs_lifecycle(&self) -> bool {
        (**self).needs_lifecycle()
    }
}

#[cfg(feature = "controller")]
pub(crate) mod wire;

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

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

    /// A stage that counts what reached it, to prove the box forwards every
    /// hook and a stateless stage needs none.
    #[derive(Default)]
    struct Counting(AtomicUsize);

    impl Lifecycle for Counting {
        fn on_released(&self, _: &Lease, _: ReleaseReason) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }

        fn on_signal(&self, _: &Signal) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    struct Stateless;

    impl Lifecycle for Stateless {}

    /// A stage whose state the calls keep.
    struct Kept;

    impl Lifecycle for Kept {
        fn needs_lifecycle(&self) -> bool {
            true
        }
    }

    #[test]
    fn a_boxed_stage_says_what_its_stage_says_of_the_calls() {
        let kept: Box<dyn Lifecycle> = Box::new(Kept);
        assert!(Lifecycle::needs_lifecycle(&kept));
        let stateless: Box<dyn Lifecycle> = Box::new(Stateless);
        assert!(!Lifecycle::needs_lifecycle(&stateless));
    }

    #[test]
    fn a_boxed_stage_forwards_its_hooks_and_a_stateless_one_ignores_them() {
        let counting = Box::new(Counting::default());
        let lease = Lease::default();
        Lifecycle::on_released(&counting, &lease, ReleaseReason::ClientGone);
        Lifecycle::on_signal(
            &counting,
            &Signal::Health {
                worker: WorkerId(1),
                health: Health::default(),
            },
        );
        Lifecycle::on_membership(&counting, &Membership::Resized(WorkerId(1)));
        assert_eq!(
            counting.0.load(Ordering::Relaxed),
            2,
            "the two implemented hooks reached the stage through the box; the membership hook is its default no-op"
        );
        let stateless: Box<dyn Lifecycle> = Box::new(Stateless);
        stateless.on_released(&lease, ReleaseReason::Abandoned);
        stateless.on_dispatched(&lease, &Target::default(), Side::Prefill, 0);
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
