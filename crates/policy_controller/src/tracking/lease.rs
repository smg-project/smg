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

use std::{fmt, time::Instant};

use crate::{
    engine::{
        sticky_sessions::SessionVerdict,
        view::{tokens_or_text, Health, LoadRow, Session, WorkerId},
    },
    load::ReportSignals,
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
///
/// The record gains fields: code outside this crate that builds one writes the
/// rest as `..Lease::default()`, and a pattern of it ends with `..`.
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
    /// none, as a request of text alone does. The `least-load` and
    /// `resident-work` stages credit a dispatch of no stated size at their
    /// own mean prefill. A profile's count of sends keeps such a dispatch
    /// by the estimate of its text ([`Self::prompt_size`]), which a
    /// document's `sent.tokens` term adds. Every other reader takes the 0
    /// as stated.
    pub tokens: u32,
    /// The length of the request's text in characters
    /// ([`crate::engine::RequestFacts::prompt_chars`]): 0 where the request
    /// stated token ids, or no text, or no stage of its document reads a
    /// request's size. The `bucket` stage books a request of no stated
    /// tokens by it, as it places one, and it has no other reader.
    /// In-process only: the wire carries [`Self::tokens`].
    pub chars: u32,
    /// The length of the request's text in bytes
    /// ([`crate::engine::RequestFacts::prompt_bytes`]): 0 where the request
    /// stated token ids, or no text. [`Self::prompt_size`] makes a count
    /// of it, under any document. In-process only: the wire carries
    /// [`Self::tokens`].
    pub bytes: u32,
}

impl Lease {
    /// The size the request stated in tokens, for a reader that prices a
    /// request of text by its length: [`Self::tokens`], and where the
    /// request stated none its text's bytes over four ([`Self::bytes`]).
    /// It is the number
    /// [`RequestFacts::prompt_size`](crate::engine::RequestFacts::prompt_size)
    /// reads of the request: a plugin's view of a held decision shows
    /// this where its view of the request showed that, and the two are
    /// one number for every text, under every document. The decode side
    /// of a pair that a decode call picked states that call's request.
    #[inline]
    #[must_use]
    pub const fn prompt_size(&self) -> u32 {
        tokens_or_text(self.tokens, self.bytes)
    }
}

/// The transport outcome of a dispatch, beside its status: what a health
/// rule reads when it weighs a prober's verdict against the gateways'
/// reports. Further classes add their cases: a `match` outside this crate
/// keeps an arm for the rest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TransportClass {
    Ok,
    ClientError,
    ServerError,
    ConnectFailed,
    Timeout,
    Cancelled,
}

/// What a completed dispatch reported. The record gains fields: code
/// outside this crate makes it by [`Outcome::reported`], sets what it
/// measured a field each, and matches it with `..`.
///
/// ```
/// use smg_policy_controller::{
///     engine::WorkerId,
///     tracking::{Outcome, Side, TransportClass},
/// };
/// let mut outcome =
///     Outcome::reported(200, TransportClass::Ok).with_served(Side::Single, WorkerId(7));
/// outcome.tokens_out = 12;
/// assert_eq!((outcome.served, outcome.tokens_out), (WorkerId(7), 12));
/// ```
///
/// ```compile_fail
/// use smg_policy_controller::tracking::{Outcome, TransportClass};
/// let _ = Outcome { status: 500, ..Outcome::reported(200, TransportClass::Ok) };
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
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

impl Outcome {
    /// An outcome that states how its request ended and nothing it
    /// measured: the status and the class, with every measure zero, under
    /// no side and no served worker. Its caller sets what it measured, a
    /// field each.
    ///
    /// A placement that is told an end (`Placement::ended`, with the feature
    /// `in-process`) writes the side and the served worker over: its caller
    /// leaves both as made here. A guard's caller states both
    /// ([`Outcome::with_served`]).
    ///
    /// The class has no default: it feeds a worker's health, and an end
    /// whose caller forgot it must not read as [`TransportClass::Ok`].
    #[must_use]
    pub const fn reported(status: u16, class: TransportClass) -> Self {
        Self {
            side: Side::Single,
            served: WorkerId(0),
            status,
            class,
            ttft_micros: 0,
            tokens_in: 0,
            tokens_out: 0,
            duration_micros: 0,
        }
    }

    /// The outcome as the end of `side`'s dispatch to `served`: what a
    /// caller that holds a guard states, where no placement writes the two
    /// for it.
    #[must_use]
    pub const fn with_served(mut self, side: Side, served: WorkerId) -> Self {
        self.side = side;
        self.served = served;
        self
    }
}

/// Why a lease was released before its completion. Further reasons add
/// their cases: a `match` outside this crate keeps an arm for the rest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
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

/// The time a gateway has to name a lease's dispatch: the design's default.
/// A lease still issued past it is released as abandoned.
pub(crate) const DISPATCH_TTL: std::time::Duration = std::time::Duration::from_secs(5);

/// What a gateway's `Dispatched` report says beside its lease.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dispatch {
    /// Which target of the pick the gateway dispatched to.
    pub target_index: u8,
    pub side: Side,
    /// Decision to dispatch on the gateway's clock, in microseconds.
    pub since_decision_micros: u32,
}

/// What a host says of a lease
/// ([`DecisionEngine::say`](crate::engine::DecisionEngine::say)): a side's
/// dispatch, a side's end, or a release.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Said {
    /// The dispatch the frame names, and the instant the frame was read.
    Dispatched(Dispatch, Instant),
    /// The outcome of the side that ended.
    Completed(Outcome),
    /// The side given up, where the frame names one, and the reason.
    Released(Option<Side>, ReleaseReason),
}

/// A membership change the host reports. A host tells a join when it takes
/// it, and a view that lists the row may follow: a stage that lists by the
/// feed treats a member its candidates do not hold as one a filter took.
///
/// Further changes add their cases, and a case with named fields may gain
/// one: a `match` outside this crate keeps an arm for the rest and ends
/// such a case's pattern with `..`. It makes those two cases by their
/// calls ([`Membership::replaced`], [`Membership::left`]).
///
/// ```
/// use smg_policy_controller::{engine::WorkerId, tracking::Membership};
/// let models = ["a-model".to_owned()];
/// let replaced = Membership::replaced(WorkerId(1), WorkerId(2));
/// let left = Membership::left(WorkerId(2), &models);
/// assert!(matches!(replaced, Membership::Replaced { old: WorkerId(1), .. }));
/// assert!(matches!(left, Membership::Left { models: [_], .. }));
/// ```
///
/// ```compile_fail
/// use smg_policy_controller::{engine::WorkerId, tracking::Membership};
/// let _ = Membership::Replaced { old: WorkerId(1), new: WorkerId(2) };
/// ```
///
/// ```compile_fail
/// use smg_policy_controller::{engine::WorkerId, tracking::Membership};
/// let _ = Membership::Left { id: WorkerId(2), models: &[] };
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Membership<'a> {
    Added(WorkerId),
    Removed(WorkerId),
    /// The same registration key with a new incarnation.
    #[non_exhaustive]
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
    #[non_exhaustive]
    Left {
        id: WorkerId,
        models: &'a [String],
    },
}

impl<'a> Membership<'a> {
    /// The registration key of `old` under its new incarnation `new`
    /// ([`Membership::Replaced`]).
    #[must_use]
    pub const fn replaced(old: WorkerId, new: WorkerId) -> Self {
        Self::Replaced { old, new }
    }

    /// The worker `id` left its pool of each of `models`
    /// ([`Membership::Left`]).
    #[must_use]
    pub const fn left(id: WorkerId, models: &'a [String]) -> Self {
        Self::Left { id, models }
    }
}

/// A signal about one worker. The named values a document declares under
/// `signals` are the other thing of that name: numbers read beside the
/// rows ([`SignalCells`](crate::load::SignalCells)). A load row states its
/// report's named values too, and borrows them: the word lives as long as
/// they are lent.
///
/// Further signals add their cases, and each case may gain a field: a
/// `match` outside this crate keeps an arm for the rest and ends a case's
/// pattern with `..`. It makes a case by its call ([`Signal::load`],
/// [`Signal::health`], [`Signal::gateway_veto`]).
///
/// ```
/// use smg_policy_controller::{
///     engine::{Health, WorkerId},
///     tracking::Signal,
/// };
/// let health = Signal::health(WorkerId(1), Health::default());
/// let veto = Signal::gateway_veto(WorkerId(1), 500_000);
/// assert!(matches!(health, Signal::Health { worker: WorkerId(1), .. }));
/// assert!(matches!(veto, Signal::GatewayVeto { until_micros: 500_000, .. }));
/// ```
///
/// ```compile_fail
/// use smg_policy_controller::{
///     engine::{Health, WorkerId},
///     tracking::Signal,
/// };
/// let _ = Signal::Health { worker: WorkerId(1), health: Health::default() };
/// ```
///
/// ```compile_fail
/// use smg_policy_controller::{engine::WorkerId, tracking::Signal};
/// let _ = Signal::GatewayVeto { worker: WorkerId(1), until_micros: 500_000 };
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Signal<'a> {
    /// A load row for one rank of a worker, as the host shows it to its
    /// decisions. The in-process engine tells one row a worker object: the
    /// sum of the ranks its report holds, under the rank the object states.
    ///
    /// A host makes the word by [`Signal::load`] and puts its report's
    /// named values on it by [`Signal::with_signals`]. The variant may
    /// gain a field: code outside this crate matches it with `..` and
    /// writes it by those two calls.
    #[non_exhaustive]
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
        /// decision. In-process only; the wire carries the row's age. Each
        /// host dates a polled report by its own rule. The policy store
        /// states the instant it sent the poll less the age the row states
        /// ([`LoadRow::report_age_ms`]), or the stamp the answer carries
        /// where it carries one. The worker crate's load monitor, which the
        /// in-process engine reads, states the instant it read the answers
        /// of the poll's round, behind the last of them, and takes off
        /// nothing. It dates a pushed record at its receipt less the age
        /// the record states and a one-way margin, and a report that holds
        /// both at the earlier.
        sampled_at_micros: u64,
        /// The named values the report states, for a stage that keeps a
        /// signal's column
        /// ([`SignalColumn::told`](crate::load::SignalColumn::told)): one
        /// cell a signal the document declares, each value under its own
        /// sample instant on the clock of `sampled_at_micros`. They are
        /// lent for the call that tells the word
        /// ([`Lifecycle::on_signal`](crate::stages::Lifecycle::on_signal)).
        /// In-process only: no wire carries them, and a plugin is told
        /// the row alone. None ([`ReportSignals::NONE`]) from a host that
        /// states none, as the policy store does. The in-process engine
        /// states them under a document a stage of which keeps a signal's
        /// column, as its lists' cells state them, and none of a report
        /// that is not the whole of its worker object.
        signals: ReportSignals<'a>,
    },
    #[non_exhaustive]
    Health { worker: WorkerId, health: Health },
    /// A gateway's standing veto on a worker, until the given instant on the
    /// host's clock. In-process only; the wire carries a duration.
    #[non_exhaustive]
    GatewayVeto { worker: WorkerId, until_micros: u64 },
}

impl<'a> Signal<'a> {
    /// A load row of `worker`'s rank `dp_rank`, sampled at
    /// `sampled_at_micros`, as a host that keeps no counts tells it: with
    /// no slot ([`WorkerSlot::NONE`]) and no named value
    /// ([`ReportSignals::NONE`]).
    #[must_use]
    pub const fn load(
        worker: WorkerId,
        dp_rank: u16,
        row: LoadRow,
        sampled_at_micros: u64,
    ) -> Self {
        Self::Load {
            worker,
            dp_rank,
            row,
            slot: WorkerSlot::NONE,
            sampled_at_micros,
            signals: ReportSignals::NONE,
        }
    }

    /// The health of `worker` ([`Signal::Health`]).
    #[must_use]
    pub const fn health(worker: WorkerId, health: Health) -> Self {
        Self::Health { worker, health }
    }

    /// A gateway's veto on `worker` until `until_micros` on the host's
    /// clock ([`Signal::GatewayVeto`]).
    #[must_use]
    pub const fn gateway_veto(worker: WorkerId, until_micros: u64) -> Self {
        Self::GatewayVeto {
            worker,
            until_micros,
        }
    }

    /// The word with `signals` as the named values its report states. A
    /// word that is no load row is answered as it is.
    #[must_use]
    pub const fn with_signals(self, signals: ReportSignals<'a>) -> Self {
        match self {
            Self::Load {
                worker,
                dp_rank,
                row,
                slot,
                sampled_at_micros,
                ..
            } => Self::Load {
                worker,
                dp_rank,
                row,
                slot,
                sampled_at_micros,
                signals,
            },
            other => other,
        }
    }
}

/// One kind of lifecycle event: one of the six calls of [`Lifecycle`], as
/// both plugin contracts name them. Further calls add their kinds: a
/// `match` outside this crate keeps an arm for the rest.
///
/// [`Lifecycle`]: crate::stages::Lifecycle
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EventKind {
    Issued,
    Dispatched,
    Completed,
    Released,
    Membership,
    Signal,
}

impl EventKind {
    /// Every kind, in the order of its bit. A slice: a further kind changes
    /// no type that code outside this crate names.
    pub const ALL: &'static [Self] = &Self::EVERY;

    /// The list behind [`Self::ALL`] with its length in its type, for the
    /// crate's own code that maps it.
    pub(crate) const EVERY: [Self; 6] = [
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
        let kinds = EventKind::EVERY;
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

    /// A load word made by its call names no slot and no named value. It
    /// takes a report's values whole and keeps every other field, its slot
    /// too. Two words of one row are two where their values are. A word of
    /// another kind takes no value.
    #[test]
    fn a_load_word_is_made_with_no_slot_and_takes_its_reports_values() {
        use crate::{load::SignalCell, tracking::in_flight::InFlightCounts};
        let row = LoadRow {
            waiting_uncached_tokens: 7,
            ..LoadRow::default()
        };
        let load = |slot, signals| Signal::Load {
            worker: WorkerId(3),
            dp_rank: 1,
            row,
            slot,
            sampled_at_micros: 9,
            signals,
        };
        let made = Signal::load(WorkerId(3), 1, row, 9);
        assert_eq!(made, load(WorkerSlot::NONE, ReportSignals::NONE));
        let cells = [
            SignalCell {
                value: 7.0,
                sampled_at_micros: 5,
            },
            SignalCell::NONE,
        ];
        let stated = ReportSignals::new(&cells);
        assert_eq!(made.with_signals(stated), load(WorkerSlot::NONE, stated));
        assert_ne!(made.with_signals(stated), made, "one row, two words");
        let slot = InFlightCounts::default().claim();
        assert_ne!(slot, WorkerSlot::NONE, "a table's first slot");
        let counted = load(slot, ReportSignals::NONE);
        assert_eq!(counted.with_signals(stated), load(slot, stated));
        let health = Signal::Health {
            worker: WorkerId(3),
            health: Health::default(),
        };
        assert_eq!(health.with_signals(stated), health);
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

    /// An outcome made by its constructor states the status and the class
    /// it was given and nothing else: every measure is zero, and it names
    /// no side and no served worker. It is the literal it stands for.
    #[test]
    fn an_outcome_reported_states_its_end_and_no_measure() {
        let reported = Outcome::reported(503, TransportClass::ServerError);
        let literal = Outcome {
            side: Side::Single,
            served: WorkerId(0),
            status: 503,
            class: TransportClass::ServerError,
            ttft_micros: 0,
            tokens_in: 0,
            tokens_out: 0,
            duration_micros: 0,
        };
        assert_eq!(reported, literal);
    }
}
