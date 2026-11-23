//! Leader election (design 10): one [`Elector`] over one [`LeaseBackend`],
//! so a Kubernetes `Lease` and a lease file on one host elect the same way,
//! and the [`Fence`] a shard thread reads before every decision it sends.
//!
//! The lease is one record ([`LeaseRecord`]: holder, term, the acquire and
//! renew wall timestamps, the holder's duration) at one place: a file on a
//! local filesystem ([`FileLease`]), a record in this process
//! ([`MemoryLease`], the one-replica controller), or a `Lease` object of
//! `coordination.k8s.io/v1` (`KubernetesLease`, behind the `kubernetes`
//! feature). A backend reads the
//! record and writes it when the stored record is still the one the writer
//! read: a compare-and-swap under the file's advisory lock, the object's
//! resource version on the API server. Everything else is the elector's and
//! the same over every backend:
//!
//! - **Acquire.** A candidate that never saw the lease held takes what the
//!   store shows at once when it is absent or a leader's release, which says
//!   no holder and no duration. One that saw it held takes a leader's release
//!   at once as well (a release never lowers the term, so the record's term
//!   is at or above the one last seen held), and a record that stood
//!   unchanged for its duration *by this candidate's own clock*, its own
//!   identity's record after a restart included: it times the record from the
//!   instant it first saw it in its present shape, never from the wall
//!   timestamp the record carries, so no second clock enters the decision. A
//!   record that is gone after one was seen held (a deleted object, a lost or
//!   emptied file, a `Lease` replaced by an empty one, which reads as
//!   released under term zero) is no holder's release, and its holder may
//!   still lead: the candidate waits the duration of the record it saw held
//!   last, from its own first sight of the change, as for a record that
//!   stands still, and follows nobody meanwhile ([`TickOutcome::Gone`]). A
//!   record with no holder that still says a duration is no leader's release
//!   either, which writes none: a hand cleared the holder in place, or a taker
//!   gave the lease up while its own take still waited to lead and says the
//!   renew deadline, the whole wait of that take. Every candidate waits the
//!   duration the record says, from its first sight of it. A take writes the
//!   next term (terms only grow, one per change of holder, and a candidate
//!   counts from the highest term it saw, so a record that vanished under it is
//!   taken at the term after the one it held). A term is taken once: a run that
//!   finds its own identity's record from an earlier run takes above it as any
//!   candidate would, so a run leads under no term below the highest it saw and
//!   under no term it saw another run lead.
//! - **Renew.** The holder rewrites the record every renew period with a
//!   later wall timestamp under its own term; a record that moved under it (a
//!   candidate took the expired lease) refuses the renewal and the holder has
//!   lost.
//! - **Step down.** A holder told to leave ([`Elector::step_down`]: a planned
//!   hand-over, or a term above its own that a caller says it saw) closes its
//!   fence, releases the lease and takes nothing for the duration its record
//!   said: a standby takes at its next look, the replica that left takes
//!   after that hold and answers [`TickOutcome::Waiting`] meanwhile. An elector
//!   alone holds nothing back: no other candidate can take its lease. The
//!   release writes the highest term its run saw, so the next holder,
//!   whoever it is, leads above a term the holder was told. A term said to
//!   a replica that holds no lease is counted among the terms its run saw
//!   and holds nothing back. No step leads under a term below the highest
//!   its run saw: a take in doubt that landed under one is left at the
//!   step that finds it. Every term is below [`TERM_CEILING`].
//! - **Fence.** The [`Fence`] holds the window its holder leads in: from
//!   the lead's opening to one renew deadline past the last successful
//!   renewal. Past the renew deadline without a renewal the fence closes by
//!   itself: [`Fence::leader_term`] answers `None`, the shard thread answers
//!   `NotLeader` with an empty address and releases nothing, whether the
//!   election thread is stalled on its backend or gone. The deadline is
//!   shorter than the duration, so a holder stops serving before any
//!   candidate that saw its record may take its lease; the residual window
//!   is the renew deadline plus clock-rate skew, bounded by one decision's
//!   send. A candidate that takes a record with no holder to wait for
//!   cannot have seen a holder that may still lead: it takes at once and
//!   its lead opens one renew deadline after the take's write returned. The
//!   fence opens at that instant by itself, on every backend and under
//!   every setting, where a renewal inside the deadline stands behind it.
//!   Such a record is absent, gone and waited out, or released under term
//!   zero, which is no holder's word: no release lowers its term and a take
//!   writes one or more. The take's write landed over a record with no
//!   holder, so a holder still alive renewed before that write, whether its
//!   record went before this candidate's read or between that read and the
//!   write: its own fence is closed by then, and a holder stops serving
//!   before any taker of its lease serves, for replicas that share their
//!   settings. A taker that gives the lease up inside that wait hands on
//!   what it still owed: its release says the time left to its opening as
//!   the record's duration, rounded up.
//!   An elector alone over a store it made itself ([`Elector::alone`]) has
//!   no other holder to wait for: its take leads at once.
//!   The controller hands its candidate a step for each take. It runs
//!   behind the take's write and behind the fence's stamp, so the step
//!   reads the fence as any reader does: the take's term where its window
//!   is open, and none where the window is past or the lead still waits.
//!
//! The elector keeps five private instants, each with one meaning. Before
//! the first it writes no record that names it: it takes nothing, and a
//! holder told to leave whose release has not landed tries that release
//! again in place of a renewal. A step-down sets it one lease duration
//! ahead. The second is the instant its lead opens: the take's own
//! instant, or one renew deadline after the write of a take of a record
//! with no holder to wait for. The fence reads closed before it, and a
//! release before it hands on the wait left to it. The third is the end it
//! stamped its fence with last: a renewal no later than it keeps the opening.
//! The fourth is the latest instant a step or a word was handed: a call
//! handed no instant reads the later of it and the clock. The fifth is
//! this run's first sight of a record that is gone after one was seen
//! held: the wait for that record counts from it. One more date of that
//! kind is kept with the record this run read last: its first sight of
//! the record as it stands, at a step's read or at the read after a write
//! that came back moved. The wait for a record that stands still counts
//! from it.
//!
//! The settings ([`ElectionSettings`]): duration 3 s, renew period 1 s, renew
//! deadline 2 s by default; a deployment that prefers fewer leadership
//! changes runs a duration of 15 s, a renew deadline of 10 s and a renew
//! period of 2 s, and lets the fallback profile cover the longer gap. Every
//! instant the elector times by is the candidate's own
//! `Instant`; the record's wall timestamps are for an operator reading the
//! file and for the API object.
//!
//! The election thread's loop is short: [`Elector::tick`] every renew period,
//! the returned [`TickOutcome`] logged, the term handed to the store
//! (`LeasesStore::set_term`) on `Acquired`, [`Elector::release`] on a planned
//! stop, [`Elector::step_down`] on a word to leave. A backend call blocks
//! that thread alone; the shard threads read the fence's snapshot, never the
//! backend. The identity is `controller-id@host:port`, dialable without a
//! Service and unique per instance. A restart under the same identity counts
//! as a change of holder: it takes its earlier run's record under the next
//! term once the record stood still for its duration, the wait of every
//! candidate (the earlier run's fence, stalled or gone, is closed by then). A
//! term is taken once, and the restart reads to a gateway as a hand-over to a
//! new leader, which a new run over an empty store is. That needs a lease
//! that outlives the process, a file or a `Lease`: the record in this process
//! is empty at every start, and a controller over it takes term one again. A
//! second instance under a duplicated identity sees the record move under its
//! name and refuses to take it for as long as it moves, so no second fence
//! opens beside the first.
//! A write that landed but reported an error (a sync that failed after the
//! rename, an update the server applied before the call timed out) is told
//! from a stranger's by the record the elector attempted, and is adopted as
//! its own.

mod file;
#[cfg(feature = "kubernetes")]
mod kubernetes;

use std::{
    fmt,
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use arc_swap::ArcSwap;
pub use file::FileLease;
#[cfg(feature = "kubernetes")]
pub use kubernetes::{KubernetesLease, TERM_ANNOTATION};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

/// The election's timing, shared by every backend (design 10).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ElectionSettings {
    /// How long the record may stand unchanged before a candidate takes it,
    /// timed by each candidate from its own first sight of the record.
    pub duration: Duration,
    /// How often the holder renews and the candidates look.
    pub renew_period: Duration,
    /// How long after its last renewal a holder still counts itself the
    /// leader; past it the fence closes on its own.
    pub renew_deadline: Duration,
}

impl Default for ElectionSettings {
    /// Duration 3 s, renew period 1 s, renew deadline 2 s.
    fn default() -> Self {
        Self {
            duration: Duration::from_secs(3),
            renew_period: Duration::from_secs(1),
            renew_deadline: Duration::from_secs(2),
        }
    }
}

impl ElectionSettings {
    /// The bounds and order the fence rests on: a renew period above zero,
    /// a renew deadline longer than it, and a record-sized duration longer
    /// than the deadline.
    pub fn validate(&self) -> Result<(), ElectError> {
        if self.renew_period.is_zero() {
            return Err(ElectError::Settings("the renew period is zero"));
        }
        if self.renew_deadline <= self.renew_period {
            return Err(ElectError::Settings(
                "the renew deadline is not longer than the renew period",
            ));
        }
        if self.duration <= self.renew_deadline {
            return Err(ElectError::Settings(
                "the duration is not longer than the renew deadline",
            ));
        }
        whole_millis_up(self.duration)?;
        Ok(())
    }
}

/// The lease as its backend stores it: the file's one JSON line, the
/// `Lease` object's spec.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeaseRecord {
    /// The holder's identity; empty once released.
    pub holder: String,
    /// The fencing term: one more at every change of holder, the same across
    /// a holder's renewals. A restart under the same identity counts as a
    /// change of holder.
    pub term: u64,
    /// Wall time of the take, microseconds since the Unix epoch.
    pub acquired_at_micros: u64,
    /// Wall time of the last renewal, microseconds since the Unix epoch;
    /// later than the record's last at every write, so an observer sees
    /// every renewal as a change.
    pub renewed_at_micros: u64,
    /// The holder's duration in milliseconds: how long the record may stand
    /// unchanged before another candidate takes it. A leader's release says
    /// `0`; a release inside its take's wait says the wait still owed.
    pub duration_ms: u64,
}

impl LeaseRecord {
    /// No holder is named: released, or never held.
    pub fn is_released(&self) -> bool {
        self.holder.is_empty()
    }

    /// The holder's duration.
    pub fn duration(&self) -> Duration {
        Duration::from_millis(self.duration_ms)
    }
}

/// What a backend did with a conditional write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConditionalWriteOutcome {
    /// The record was replaced.
    Done,
    /// The stored record was no longer the expected one: another candidate
    /// wrote first. Nothing was written.
    Moved,
}

/// Where one lease record lives: the contract the lease file, the in-process
/// record and the Kubernetes `Lease` implement.
///
/// Both calls may block the election thread (a file lock held for
/// microseconds, a round trip to an API server); neither is on a decision
/// path. A backend stores a record so that `read` returns it equal, field
/// for field: the elector's renewals and releases are conditional on the
/// record it wrote last. A backend keeps whatever makes its `write`
/// conditional (the object's resource version); the elector hands it the
/// record it last read.
pub trait LeaseBackend: Send {
    /// The record as stored now; `None` when no lease exists yet.
    fn read(&mut self) -> Result<Option<LeaseRecord>, ElectError>;

    /// Store `record` if the record stored now is still `expected` (`None`:
    /// none exists); otherwise write nothing and answer
    /// [`ConditionalWriteOutcome::Moved`].
    fn write(
        &mut self,
        expected: Option<&LeaseRecord>,
        record: &LeaseRecord,
    ) -> Result<ConditionalWriteOutcome, ElectError>;

    /// Where the record lives, for logs: the file's path, the object's
    /// namespace and name.
    fn place(&self) -> String;
}

/// The ceiling of a term: every term a record carries is below it. Two
/// readers keep a term in a narrower field than the record's, a membership
/// life's number in twenty-four bits and a lease's id in thirty-one, and the
/// narrower is the ceiling: a term a second lasts half a year under it. A
/// word may not spend the last term: a term said whose next is not below
/// the ceiling is refused ([`term_said`]), by the elector and by every door
/// in front of it. A take that would reach the ceiling is an error.
pub const TERM_CEILING: u64 = 1 << 24;

/// Why an election step failed.
#[derive(Debug, thiserror::Error)]
pub enum ElectError {
    #[error("election settings: {0}")]
    Settings(&'static str),
    #[error("election identity: {0}")]
    Identity(&'static str),
    #[error("the lease at {place}: {source}")]
    Io {
        place: String,
        #[source]
        source: std::io::Error,
    },
    #[error("the lease at {place} holds a record this build cannot read: {detail}")]
    Malformed { place: String, detail: String },
    #[error("election term: the next term {0} is not below the ceiling {ceiling}", ceiling = TERM_CEILING)]
    Term(u64),
}

/// What one [`Elector::tick`] did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TickOutcome {
    /// This candidate took the lease under `term`. It leads from now, or
    /// from one renew deadline after this take's write where the record had
    /// no holder to wait for (absent, gone and waited out, or released under
    /// term zero). An elector alone ([`Elector::alone`]) leads from now
    /// whatever the record.
    Acquired { term: u64 },
    /// The lease is renewed under `term`; the fence is open for another
    /// renew deadline, or still closed after a take of a record with no
    /// holder to wait for: it opens one renew deadline after that take's
    /// write, by itself where this renewal's deadline is past that instant.
    Renewed { term: u64 },
    /// The lease moved under this holder: a candidate took it after it stood
    /// unchanged for the duration, or the record was rewritten by hand. The
    /// fence is closed. `holder` is the record's holder now, empty when the
    /// lease is released or gone.
    Lost { term: u64, holder: String },
    /// Another candidate holds the lease and renews it under `term`.
    Following { holder: String, term: u64 },
    /// The record this candidate saw held under `term` is gone: absent, or
    /// released under a lower term, a shape no holder's release leaves. Its
    /// holder may still lead, so nobody is followed, and the lease is taken
    /// once that record's duration has passed since this candidate first saw
    /// the change. Also the answer at a record under `term` with no holder
    /// and a duration: a hand's clearing leaves one, and so does a release
    /// inside its take's wait. A candidate inside its own hold answers
    /// [`TickOutcome::Waiting`] there.
    Gone { term: u64 },
    /// The lease stands free, released or unrenewed past its duration, and
    /// this candidate waits its turn: it left on a word and takes nothing
    /// until its hold ends ([`Elector::step_down`]). Nobody is followed, and
    /// the lease is taken at the hold's end unless another candidate takes it
    /// first. `term` is the highest this candidate saw: its take is above it.
    Waiting { term: u64 },
    /// The lease looked free, but another candidate wrote first; the next
    /// tick reads what it wrote.
    Raced,
}

/// What the shard threads read: the term this process leads under and the
/// window it leads in, replaced whole at every renewal and read lock-free
/// before every decision.
#[derive(Debug)]
pub struct Fence {
    lead: ArcSwap<Lead>,
}

#[derive(Debug, Default)]
struct Lead {
    term: u64,
    /// The window this process leads in: from its opening to its end, one
    /// renew deadline past its last renewal. `None` while no lease is held,
    /// and while a take's opening is at or past that end.
    window: Option<(Instant, Instant)>,
}

impl Lead {
    /// The term led under at `now`, where the window holds that instant.
    fn term_at(&self, now: Instant) -> Option<u64> {
        let (opens, until) = self.window?;
        (opens <= now && now <= until).then_some(self.term)
    }

    /// The instant the clock alone next turns [`Lead::term_at`].
    fn turns_at(&self, now: Instant) -> Option<Instant> {
        let (opens, until) = self.window?;
        if now < opens {
            Some(opens)
        } else {
            (now <= until).then_some(until)
        }
    }
}

impl Fence {
    fn new() -> Self {
        Self {
            lead: ArcSwap::from_pointee(Lead::default()),
        }
    }

    /// The term this process leads under at `now`, or `None`: no lease held,
    /// `now` before the lead's opening, or the last renewal past the renew
    /// deadline (the self-fence). The shard thread's read before every
    /// decision it sends; past the deadline it answers `NotLeader` and
    /// releases nothing.
    pub fn leader_term(&self, now: Instant) -> Option<u64> {
        self.lead.load().term_at(now)
    }

    /// The term of the last lease this process held, `0` before any: what
    /// its frames carry, leading or not.
    pub fn term(&self) -> u64 {
        self.lead.load().term
    }

    /// The instant the clock alone next turns the answer of
    /// [`Fence::leader_term`], as the fence stands at `now`: the opening of
    /// a take's window still ahead, or the end of the window this process
    /// leads in, past which the fence is closed. `None` where no window
    /// stands or it has ended: only a write of the elector turns the answer
    /// then. A reader that waits on the role sleeps to this instant.
    pub fn turns_at(&self, now: Instant) -> Option<Instant> {
        self.lead.load().turns_at(now)
    }

    /// [`Fence::leader_term`] and [`Fence::turns_at`] at `now`, by one read
    /// of the fence. A reader that holds work to the lead takes the pair
    /// from here. Two reads can stand astride a write of the elector: a
    /// lead read before a release, beside a turn read after it, has no
    /// instant to sleep to. A term from this read comes with its window's
    /// end.
    pub fn term_and_turn(&self, now: Instant) -> (Option<u64>, Option<Instant>) {
        let lead = self.lead.load();
        (lead.term_at(now), lead.turns_at(now))
    }

    fn set(&self, term: u64, window: Option<(Instant, Instant)>) {
        self.lead.store(Arc::new(Lead { term, window }));
    }
}

/// The record as this candidate last read it, and when it first saw it in
/// that shape.
struct Observation {
    record: LeaseRecord,
    since: Instant,
}

/// One candidate of one election: acquire, renew, release over a backend,
/// and the fence its shard threads read.
pub struct Elector {
    backend: Box<dyn LeaseBackend>,
    identity: String,
    settings: ElectionSettings,
    fence: Arc<Fence>,
    observed: Option<Observation>,
    /// The record this process wrote last while holding the lease.
    held: Option<LeaseRecord>,
    /// The record of the last write that reported an error: it may have
    /// landed (a sync that failed after the rename, an update that timed out
    /// after the server applied it), now or later, and a stored record equal
    /// to it is this process's own. Kept until a write of this process lands
    /// with its answer or another record supersedes it.
    attempted: Option<LeaseRecord>,
    /// New shapes of the record under this identity that this run did not
    /// write: the first is the earlier run's last write, a second is another
    /// candidate under a duplicated identity. Cleared when the record stands
    /// still for its duration (this run takes above it), is released, or
    /// passes to another identity: the refusal lasts as long as the writing
    /// does.
    strangers: u32,
    /// The highest term this run saw stored or wrote: a record that vanished
    /// (a deleted object, a lost file) is taken at the term after it, so
    /// terms only grow while any candidate that saw them is alive.
    highest_term: u64,
    /// The term and the duration of the record this run saw held last, its
    /// own writes included: what a record without a holder is judged
    /// against. Released at this term or above and saying no duration, it is
    /// that holder's release; absent, or released under a lower term, it is
    /// gone by another hand.
    last_held: Option<(u64, Duration)>,
    /// This run's first sight of a record that is gone after one was seen
    /// held: the wait for it counts from here. Cleared by a held record and
    /// by a holder's release.
    gone_since: Option<Instant>,
    /// The instant this run's lead opens, set by its take: the take's own
    /// instant, or one renew deadline after the take's write returned where
    /// the record had no holder to wait for. That write landed over a
    /// record with no holder, so any holder that still leads renewed before
    /// it and its own fence is closed by then. `None` while this run holds
    /// no lease.
    opens: Option<Instant>,
    /// The end this run stamped its fence with last: one renew deadline
    /// past its last write. A renewal no later than it finds the fence
    /// open and keeps the lead's opening, and reads no shared state to
    /// know it.
    until: Option<Instant>,
    /// The take this run attempted last was of a record with no holder to
    /// wait for. An attempt in doubt keeps it for the step that finds the
    /// attempt landed: that step's write dates the opening, since the
    /// record may have landed any time before it.
    waits: bool,
    /// The latest instant this elector was stepped at. A call that is
    /// handed no instant reads the later of it and the clock.
    stepped: Option<Instant>,
    /// This elector made its store itself and handed it to nobody
    /// ([`Elector::alone`]): no other holder can be, so no take of its
    /// waits to open its fence.
    alone: bool,
    /// The instant before which this run writes no record that names it:
    /// it takes nothing, and a holder under it, whose release has not
    /// landed, tries the release again in place of a renewal. Set when the
    /// run leaves on a word ([`Elector::step_down`]), one lease duration
    /// ahead, and cleared by its next take.
    hold: Option<Instant>,
    /// What this candidate's holder does at each take, behind the take's
    /// write and behind the fence's stamp ([`Elector::at_take`]).
    at_take: Option<Box<dyn Fn() + Send>>,
}

impl fmt::Debug for Elector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Elector")
            .field("place", &self.backend.place())
            .field("identity", &self.identity)
            .field("settings", &self.settings)
            .field("held", &self.held)
            .finish_non_exhaustive()
    }
}

impl Elector {
    /// A candidate named `identity` over `backend`; the settings are
    /// validated here, so a fence never runs on an inverted order, and the
    /// identity must not be empty: an empty holder is a released record.
    pub fn new(
        backend: Box<dyn LeaseBackend>,
        identity: impl Into<String>,
        settings: ElectionSettings,
    ) -> Result<Self, ElectError> {
        settings.validate()?;
        let identity = identity.into();
        if identity.is_empty() {
            return Err(ElectError::Identity("the identity is empty"));
        }
        Ok(Self {
            backend,
            identity,
            settings,
            fence: Arc::new(Fence::new()),
            observed: None,
            held: None,
            attempted: None,
            strangers: 0,
            highest_term: 0,
            last_held: None,
            gone_since: None,
            opens: None,
            until: None,
            waits: false,
            stepped: None,
            alone: false,
            hold: None,
            at_take: None,
        })
    }

    /// A candidate named `identity` alone over a lease of its own: the
    /// record lives in a store made here and handed to nobody, so no other
    /// holder can be, and a take opens its fence at once. One replica runs
    /// so. A store that is handed in ([`Elector::new`]) keeps the wait of a
    /// take with no holder to wait for, a memory one included: its clones
    /// share it.
    pub fn alone(
        identity: impl Into<String>,
        settings: ElectionSettings,
    ) -> Result<Self, ElectError> {
        let mut elector = Self::new(Box::new(MemoryLease::new()), identity, settings)?;
        elector.alone = true;
        Ok(elector)
    }

    /// The fence the shard threads read; one per elector, shared by handle.
    pub fn fence(&self) -> Arc<Fence> {
        Arc::clone(&self.fence)
    }

    /// Hands this candidate a step it runs at each take, behind the take's
    /// write and behind the fence's stamp. The step reads the fence as any
    /// reader does. It finds the take's term where the take's window is
    /// open, and no term where the write returned past that window or the
    /// lead still waits to open: what a step does for its term it does
    /// under the fence, so a take that never leads does nothing by it. The
    /// step runs inside [`Elector::tick`], on its caller's thread; a
    /// renewal is no take and runs no step.
    /// Built with the controller, its one caller.
    #[cfg(any(test, feature = "controller"))]
    pub(crate) fn at_take(&mut self, step: impl Fn() + Send + 'static) {
        self.at_take = Some(Box::new(step));
    }

    /// This run's take still waits to lead: it took a record with no holder
    /// to wait for, and the instant its lead opens is ahead. The wait is a
    /// holder's: a loss and a release end it, and a run that holds nothing
    /// answers no.
    pub fn waits_to_lead(&self) -> bool {
        !self.owed().is_zero()
    }

    /// The wait this run's take still owes: the time from the latest
    /// instant this elector knows to the instant its lead opens, none once
    /// that instant has come or where no take of this run waits.
    fn owed(&self) -> Duration {
        let latest = self.latest();
        self.opens.map_or(Duration::ZERO, |opens| {
            opens.saturating_duration_since(latest)
        })
    }

    /// The latest instant this elector knows: the clock, or its last step's
    /// instant where that is ahead of the clock.
    fn latest(&self) -> Instant {
        let clock = Instant::now();
        self.stepped.map_or(clock, |at| at.max(clock))
    }

    /// Keeps the latest instant a step or a word was handed: an earlier one
    /// moves nothing back.
    fn step_at(&mut self, now: Instant) {
        self.stepped = Some(self.stepped.map_or(now, |at| at.max(now)));
    }

    /// This elector made its store itself ([`Elector::alone`]): no other
    /// candidate can take its lease.
    pub fn is_alone(&self) -> bool {
        self.alone
    }

    pub fn identity(&self) -> &str {
        &self.identity
    }

    pub fn settings(&self) -> ElectionSettings {
        self.settings
    }

    /// This process wrote the record last and has not learned otherwise. The
    /// fence, not this, says whether it may serve.
    pub fn is_holder(&self) -> bool {
        self.held.is_some()
    }

    /// The record as this candidate last read it: who holds the lease and
    /// under which term, for `Leader`.
    pub fn observed(&self) -> Option<&LeaseRecord> {
        self.observed.as_ref().map(|o| &o.record)
    }

    /// The highest term this run saw in a record or was told: its next take
    /// is above it, and the standing names no holder under a lower one.
    pub fn highest_term(&self) -> u64 {
        self.highest_term
    }

    /// One step of the election at `now`: a holder renews, a candidate takes
    /// a free lease or keeps following. Call it every renew period. An error
    /// changes nothing: a holder keeps its record and its fence closes by
    /// itself at the deadline; the next successful step settles it.
    pub fn tick(&mut self, now: Instant) -> Result<TickOutcome, ElectError> {
        self.step_at(now);
        let current = self.backend.read()?;
        // A first sight is dated when the read returned, never before it: a
        // slow read may return a record the holder wrote during the read, and
        // the follower's wait starts from that record's shape at the
        // earliest. The holder's side keeps `now`, the instant before its own
        // read and write, which is the conservative stamp for the fence.
        let seen = Instant::now().max(now);
        self.observe(current.as_ref(), seen);
        // `held` stays as it is until a write or a read settles it: an error
        // on the way leaves the holder its record.
        match self.held.clone() {
            // Told to leave and no release landed yet: the release again,
            // never a renewal.
            Some(mine) if self.hold.is_some() => self.leave(current, &mine),
            Some(mine) => self.renew(current, mine, now),
            None => self.acquire(current, now),
        }
    }

    /// Give the lease up: the record is rewritten released, under the
    /// highest term this run saw, so the next candidate to look takes it at
    /// once and above that term. The fence closes now and stays closed.
    /// `true` when the lease is released now, by this call's write or by an
    /// earlier one of this process that landed and reported an error;
    /// `false` when it held nothing, or when another candidate holds the
    /// lease. On an error the elector keeps the record, so a second call
    /// retries the write; a caller that ticks instead renews, or answers
    /// [`TickOutcome::Lost`] when the release landed after all.
    ///
    /// A release while this run's own take still waits to lead hands on
    /// what that wait still owed: the record says the time left to the
    /// take's opening as its duration, rounded up, and the next candidate
    /// waits it out before it takes. It leads no earlier than this run
    /// would have, and within two looks after: one to see the release, one
    /// past its wait.
    pub fn release(&mut self) -> Result<bool, ElectError> {
        let Some(mine) = self.held.clone() else {
            return Ok(false);
        };
        self.fence.set(mine.term, None);
        // No duration: it tells a holder's release, taken at once, from a
        // holder that another hand cleared in place. A release while this
        // run's own take still waits to lead says the wait it still owed
        // instead: a holder this run never saw may lead until that take's
        // opening, this run's wait was all that covered it, and the next
        // candidate waits the release out as it waits out a cleared record.
        // It times the wait from its own sight of the release, after the
        // write, so it takes no earlier than that opening. The term
        // is the highest this run saw, its own unless it was told one above:
        // the next holder leads above it, whoever it is.
        let released = LeaseRecord {
            holder: String::new(),
            term: self.highest_term.max(mine.term),
            duration_ms: whole_millis_up(self.owed())?,
            renewed_at_micros: next_wall_micros(mine.renewed_at_micros),
            ..mine.clone()
        };
        let written = match self.backend.write(Some(&mine), &released) {
            Ok(written) => written,
            Err(err) => {
                // A renewal attempt that may have landed stays the record to
                // adopt on the retry; a release that landed is read by its
                // released state and term, and needs no attempt kept.
                if self.attempted.is_none() {
                    self.attempted = Some(released);
                }
                return Err(err);
            }
        };
        match written {
            ConditionalWriteOutcome::Done => {
                self.held = None;
                self.attempted = None;
                self.opens = None;
                if let Some(observation) = &mut self.observed {
                    // This run's own sight of its release is dated at the
                    // write's return, never before its take's read returned:
                    // its own next take waits out what it handed on, as
                    // every candidate does.
                    observation.since = observation.since.max(Instant::now());
                    observation.record = released;
                }
                Ok(true)
            }
            ConditionalWriteOutcome::Moved => {
                // The record is kept until the re-read settles what moved:
                // a read error leaves a second call its retry.
                let stored = self.backend.read()?;
                // A sight as any other: the run counts its term.
                self.observe_again(stored.as_ref());
                self.held = None;
                match (stored, self.attempted.take()) {
                    // This process's own renewal that landed and reported an
                    // error: the lease to give up is that record.
                    (Some(stored), Some(attempted))
                        if stored == attempted && !stored.is_released() =>
                    {
                        self.held = Some(stored);
                        self.release()
                    }
                    // Another candidate's write, or this process's own
                    // release that landed and reported an error: released
                    // under this term, or under one above it that this run
                    // was told.
                    (stored, _) => {
                        self.opens = None;
                        Ok(stored
                            .is_some_and(|stored| stored.is_released() && stored.term >= mine.term))
                    }
                }
            }
        }
    }

    /// Leave on a word from outside the election: a planned hand-over
    /// (`None`), or a term `above` that a caller says it saw. The term is
    /// counted among the terms this run saw, so its next take is above it;
    /// a term said must leave one to take, so one whose next is not below
    /// [`TERM_CEILING`] is the term error, before it is counted. A term said
    /// while a take is in doubt, an attempt that reported an error and may
    /// have landed, stands: no step leads under a term below the highest
    /// its run saw, so the step that finds the attempt landed leaves the
    /// lease as this word would have. A word of no term finds no holder
    /// then and moves nothing. A holder that leads under a lower term, or
    /// is told none, closes its fence, releases the lease as
    /// [`Elector::release`] does and takes nothing for the duration its
    /// record said, from `now`: a candidate that followed it takes at its
    /// next look. A holder that leads at or above the term said keeps on as
    /// it is. A candidate keeps on too, and where it saw the lease released
    /// under a term below the highest its run saw, the term said counted,
    /// it writes that release again under that highest, an empty record
    /// and one gone for this run apart: whoever takes next takes above it.
    /// An elector alone ([`Elector::alone`])
    /// holds nothing back: its next step takes its own lease again, above
    /// the term said.
    ///
    /// `true` when the lease is released now. On a holder's error the fence
    /// is closed, and every later [`Elector::tick`] tries the release again
    /// in place of a renewal, until it lands or the record moves. A
    /// candidate's error is its write over a release: the term is counted
    /// and the record stays under the lower one.
    pub fn step_down(&mut self, above: Option<u64>, now: Instant) -> Result<bool, ElectError> {
        term_said(above)?;
        self.step_at(now);
        self.highest_term = self.highest_term.max(above.unwrap_or(0));
        let duration = match &self.held {
            Some(mine) if above.is_none_or(|term| term > mine.term) => mine.duration(),
            Some(_) => return Ok(false),
            None => return self.raise(above, now),
        };
        // An elector alone holds nothing back: no other candidate can take
        // its lease, so its next step takes it again, above the term said.
        self.hold = (!self.alone).then_some(now + duration);
        self.release()
    }

    /// A term said to a run that holds no lease, over a release it saw
    /// under a lower term: the release is written again under the highest
    /// term this run saw, conditional on the record seen, so the candidate
    /// that takes the lease next takes above that term. The term said is
    /// counted before this call, so the highest is at or above it, and a
    /// term whose write failed goes into the lease at the next word. A
    /// holder's release carries the highest term its run saw at that write
    /// ([`Elector::release`]), and a term said after it reached the lease by
    /// no record before this. A record that is held, absent, or released at
    /// or above the highest stays as it is, and so does one that moved
    /// since it was seen: its writer took the lease, and is told the term by
    /// the gateway that holds it. A record gone for this run, released under
    /// a term below the one it last saw held, stays as it is too: no
    /// holder's release made it so, that holder may still lead, and written
    /// under a higher term the record would read as its release. An empty
    /// record, released under term zero, stays as it is for the same
    /// reason: no holder wrote it, one this run never saw may still lead,
    /// and the take over it opens its fence a renew deadline late, which a
    /// take over a release under a higher term does not. Never `true`:
    /// this call releases no lease.
    fn raise(&mut self, above: Option<u64>, now: Instant) -> Result<bool, ElectError> {
        let gone = self
            .gone(self.observed.as_ref().map(|seen| &seen.record))
            .is_some();
        let (Some(_), Some(observation)) = (above, self.observed.as_mut()) else {
            return Ok(false);
        };
        let term = self.highest_term;
        let seen = &observation.record;
        if gone || !seen.is_released() || seen.term == 0 || seen.term >= term {
            return Ok(false);
        }
        let raised = LeaseRecord {
            term,
            renewed_at_micros: next_wall_micros(seen.renewed_at_micros),
            ..seen.clone()
        };
        if self.backend.write(Some(seen), &raised)? == ConditionalWriteOutcome::Done {
            // The record's wait, where it says one, is timed from this
            // sight of it, as every candidate times it from its own.
            observation.since = observation.since.max(now);
            observation.record = raised;
        }
        Ok(false)
    }

    /// A step of a holder that was told to leave and whose release has not
    /// landed: the release again.
    fn leave(
        &mut self,
        current: Option<LeaseRecord>,
        mine: &LeaseRecord,
    ) -> Result<TickOutcome, ElectError> {
        if self.release()? {
            return Ok(TickOutcome::Waiting {
                term: self.highest_term,
            });
        }
        // The record moved before a release landed: another candidate took
        // the lease, or a hand rewrote it.
        Ok(self.lose(mine.term, current.filter(|record| record != mine)))
    }

    /// The term and the duration of the record seen held last, when
    /// `current` is a shape no holder's release leaves after it: absent, or
    /// released under a lower term.
    fn gone(&self, current: Option<&LeaseRecord>) -> Option<(u64, Duration)> {
        let (term, duration) = self.last_held?;
        match current {
            Some(record) if !record.is_released() || record.term >= term => None,
            _ => Some((term, duration)),
        }
    }

    /// A sight of the lease at `now`: a step's read, or the read after a
    /// write that came back moved. The run counts the record's term and
    /// keeps the record with the date of its first sight in that shape.
    /// Inlined into the step, as it was while the step was its one caller;
    /// a record read again goes through [`Elector::observe_again`].
    #[inline(always)]
    fn observe(&mut self, current: Option<&LeaseRecord>, now: Instant) {
        // The first sight of a record that is gone dates the wait for it;
        // any other sight ends it.
        if self.gone(current).is_some() {
            self.gone_since.get_or_insert(now);
        } else {
            self.gone_since = None;
        }
        if let Some(record) = current.filter(|record| !record.is_released()) {
            self.last_held = Some((record.term, record.duration()));
        }
        // An absent record keeps the attempted write: an initial take whose
        // answer was lost may still land, and the record it leaves is this
        // run's own.
        let Some(record) = current else {
            self.observed = None;
            self.strangers = 0;
            return;
        };
        self.highest_term = self.highest_term.max(record.term);
        if self.observed.as_ref().is_some_and(|o| o.record == *record) {
            return;
        }
        // A new shape. The write this run attempted and lost sight of is its
        // own and stays for the adoption; any other supersedes the attempt,
        // and one under this identity was written by another run under this
        // name: the earlier run's last write at the first sight, or a
        // candidate under a duplicated identity. A record released or under
        // another identity ends that count.
        if self.attempted.as_ref() != Some(record) {
            self.attempted = None;
            self.strangers = if record.holder == self.identity {
                self.strangers.saturating_add(1)
            } else {
                0
            };
        }
        self.observed = Some(Observation {
            record: record.clone(),
            since: now,
        });
    }

    /// The record read again after a write came back moved: a sight as any
    /// other, dated at the latest instant this elector knows. Out of line:
    /// a moved write is the rare end of a step.
    #[cold]
    #[inline(never)]
    fn observe_again(&mut self, current: Option<&LeaseRecord>) {
        self.observe(current, self.latest());
    }

    fn renew(
        &mut self,
        current: Option<LeaseRecord>,
        mut mine: LeaseRecord,
        now: Instant,
    ) -> Result<TickOutcome, ElectError> {
        // An attempt that reported an error and has not landed may still
        // land late (a server applies a timed-out update after this tick's
        // read), so it is kept until a write of this process lands with its
        // answer.
        if current.as_ref() != Some(&mine) {
            match (current, self.attempted.take()) {
                // The attempt landed after all: the stored record is this
                // process's own, renewed from here. A released one is this
                // process's own release, landed: the lease is given up, the
                // fence stays closed, and the next tick is a candidate's.
                (Some(stored), Some(attempted)) if stored == attempted && !stored.is_released() => {
                    mine = stored;
                    self.held = Some(mine.clone());
                }
                (current, _) => return Ok(self.lose(mine.term, current)),
            }
        }
        let renewed = LeaseRecord {
            renewed_at_micros: next_wall_micros(mine.renewed_at_micros),
            ..mine.clone()
        };
        let written = match self.backend.write(Some(&mine), &renewed) {
            Ok(written) => written,
            Err(err) => {
                self.attempted = Some(renewed);
                return Err(err);
            }
        };
        match written {
            ConditionalWriteOutcome::Done => {
                self.attempted = None;
                self.stamp(renewed.term, now);
                self.observed = Some(Observation {
                    record: renewed.clone(),
                    since: now,
                });
                self.held = Some(renewed);
                Ok(TickOutcome::Renewed { term: mine.term })
            }
            ConditionalWriteOutcome::Moved => {
                let current = self.backend.read()?;
                // The record read again is a sight as any other: the run
                // counts its term and the standing reads it.
                self.observe_again(current.as_ref());
                match (current, self.attempted.take()) {
                    // This process's own earlier attempt landed between this
                    // tick's read and its write: the record is its own,
                    // renewed over in one more round.
                    (Some(stored), Some(attempted))
                        if stored == attempted && !stored.is_released() =>
                    {
                        self.held = Some(stored.clone());
                        self.renew(Some(stored.clone()), stored, now)
                    }
                    (current, _) => Ok(self.lose(mine.term, current)),
                }
            }
        }
    }

    /// Stamps the fence at a write of this run's own record at `now`: the
    /// window runs from the lead's opening to one renew deadline past `now`.
    /// A write past the end stamped last opens a new window at `now`, since
    /// the fence had closed by itself, unless the lead's opening is still
    /// ahead of `now`: a take's wait is kept however late its first renewal
    /// lands. An opening at or past the new end is no window: a take whose
    /// renewals do not land never leads.
    fn stamp(&mut self, term: u64, now: Instant) {
        let opens = match self.opens {
            Some(opens) if opens > now || self.until.is_some_and(|until| now <= until) => opens,
            _ => now,
        };
        self.opens = Some(opens);
        let until = later(now, self.settings.renew_deadline);
        self.until = Some(until);
        self.fence
            .set(term, (opens < until).then_some((opens, until)));
    }

    /// The instant the lead of a take made at `now` opens: `now` itself, or
    /// one renew deadline after the take's write returned where the record
    /// had no holder to wait for.
    fn opening(&self, waits: bool, now: Instant) -> Instant {
        if waits {
            later(Instant::now().max(now), self.settings.renew_deadline)
        } else {
            now
        }
    }

    fn lose(&mut self, term: u64, current: Option<LeaseRecord>) -> TickOutcome {
        self.held = None;
        self.opens = None;
        self.fence.set(term, None);
        TickOutcome::Lost {
            term,
            holder: current.map(|record| record.holder).unwrap_or_default(),
        }
    }

    fn acquire(
        &mut self,
        current: Option<LeaseRecord>,
        now: Instant,
    ) -> Result<TickOutcome, ElectError> {
        // A new term is the one after the highest this run saw, whatever the
        // record in front of it says: a record that vanished and came back
        // lower (a deleted object recreated, a lost file) never lowers it.
        let next_term = self.highest_term.saturating_add(1);
        // A record that is gone after one was seen held: no release made it
        // so, and the holder of the record seen last may lead to its renew
        // deadline. This run waits that record's duration from its first
        // sight of the change, the wait of a record that stands still, and
        // takes after it. The old holder waits as every candidate does.
        if let Some((term, duration)) = self.gone(current.as_ref()) {
            let since = self.gone_since.unwrap_or(now);
            if now.saturating_duration_since(since) < duration {
                return Ok(TickOutcome::Gone { term });
            }
        }
        // A record with no holder to wait for: absent, gone by no holder's
        // hand and waited out, or released under term zero, which no holder
        // wrote: no release lowers its term and a take writes one or more. A
        // holder this run never saw may still lead to its renew deadline and
        // no wait before the take covers it, so the take is made at once and
        // the fence opens one renew deadline later. An elector alone has no
        // such holder: its store is its own.
        let absent = current.is_none()
            || matches!(&current, Some(record) if record.is_released() && record.term == 0)
            || self.gone(current.as_ref()).is_some();
        let landed = matches!(&current, Some(record) if !record.is_released() && self.attempted.as_ref() == Some(record));
        // A term this run was told while its take was in doubt stands above
        // the take that landed: no step leads under a term below the highest
        // its run saw, so the lease is left as the word would have had it.
        if let Some(mine) = current
            .clone()
            .filter(|record| landed && record.term < self.highest_term)
        {
            self.attempted = None;
            self.held = Some(mine.clone());
            self.hold = (!self.alone).then_some(now + mine.duration());
            // The take never led: where it had no holder to wait for, its
            // release hands that wait on.
            self.opens = Some(self.opening(self.waits, now));
            return self.leave(current, &mine);
        }
        let (term, acquired_at_micros) = match &current {
            // Never seen held, or waited out.
            None => (next_term, 0),
            // A record with no holder is waited out for the duration it
            // says, from this run's first sight of it. A holder's release
            // says none and is taken at once, unless its own take still
            // waited to lead: then it says the wait that take still owed.
            // A record whose holder another hand cleared in place
            // keeps its duration, and that holder may lead until its own
            // next step reads the lease.
            Some(record) if record.is_released() => {
                let since = self.observed.as_ref().map_or(now, |o| o.since);
                if now.saturating_duration_since(since) < record.duration() {
                    // Inside its hold this run waits its turn: the record
                    // is its own release inside a take's wait, or one a
                    // later holder wrote, and neither is a record gone.
                    if self.hold.is_some_and(|until| now < until) {
                        return Ok(TickOutcome::Waiting {
                            term: self.highest_term,
                        });
                    }
                    return Ok(TickOutcome::Gone { term: record.term });
                }
                (next_term, 0)
            }
            // The write this run attempted landed after all: its own lease.
            Some(record) if self.attempted.as_ref() == Some(record) => {
                (record.term, record.acquired_at_micros)
            }
            // A record under this identity that this run did not write: the
            // earlier run's last write, or another candidate's under a
            // duplicated identity. Either writer's fence may be open to its
            // renew deadline, so this run waits for the record's duration of
            // stillness, as every candidate does. Then it takes the lease as
            // any other candidate would, above the highest term it saw and
            // new in every field: the earlier run led under the record's
            // term, and a term is taken once, so what a reader holds under
            // that term came from that run. A record that moves again under
            // this name has another writer and is refused for as long as it
            // keeps moving, since a take would open a second fence beside
            // that writer's.
            Some(record) if record.holder == self.identity => {
                let since = self.observed.as_ref().map_or(now, |o| o.since);
                if now.saturating_duration_since(since) < record.duration() {
                    if self.strangers > 1 {
                        return Err(ElectError::Identity(
                            "another candidate writes under this identity",
                        ));
                    }
                    return Ok(TickOutcome::Following {
                        holder: record.holder.clone(),
                        term: record.term,
                    });
                }
                self.strangers = 0;
                (next_term, 0)
            }
            Some(record) => {
                let since = self.observed.as_ref().map_or(now, |o| o.since);
                if now.saturating_duration_since(since) < record.duration() {
                    return Ok(TickOutcome::Following {
                        holder: record.holder.clone(),
                        term: record.term,
                    });
                }
                (next_term, 0)
            }
        };
        // This run's turn has not come: it left on a word and takes nothing
        // until its hold ends. A write of its own that landed is no take.
        if !landed && self.hold.is_some_and(|until| now < until) {
            return Ok(TickOutcome::Waiting {
                term: self.highest_term,
            });
        }
        if term >= TERM_CEILING {
            return Err(ElectError::Term(term));
        }
        let wall = next_wall_micros(current.as_ref().map_or(0, |r| r.renewed_at_micros));
        let record = LeaseRecord {
            holder: self.identity.clone(),
            term,
            acquired_at_micros: if acquired_at_micros == 0 {
                wall
            } else {
                acquired_at_micros
            },
            renewed_at_micros: wall,
            duration_ms: whole_millis_up(self.settings.duration)?,
        };
        // The take says here whether it had a holder to wait for. Its
        // lead's opening is dated once the write has returned, never from
        // the read before it: a holder may take, renew and lose its record
        // while this step is out, and its fence runs from its last renewal.
        // An attempt that reports an error may land at any time after, so
        // it dates nothing and keeps the word: the step that finds it landed
        // takes it by a write of its own, and that write dates the opening.
        if !landed {
            self.waits = absent && !self.alone;
        }
        let written = match self.backend.write(current.as_ref(), &record) {
            Ok(written) => written,
            Err(err) => {
                self.attempted = Some(record);
                return Err(err);
            }
        };
        match written {
            ConditionalWriteOutcome::Done => {
                self.opens = Some(self.opening(self.waits, now));
                self.attempted = None;
                self.hold = None;
                self.highest_term = self.highest_term.max(term);
                // This run's own record is one it saw held: if it vanishes
                // before the next step reads it, it is waited out as well.
                self.last_held = Some((term, record.duration()));
                self.gone_since = None;
                self.stamp(term, now);
                self.observed = Some(Observation {
                    record: record.clone(),
                    since: now,
                });
                self.held = Some(record);
                // The holder's step, behind the stamp: it reads the fence
                // as any reader does, and a take whose window is past or
                // still ahead shows it no term.
                if let Some(step) = &self.at_take {
                    step();
                }
                Ok(TickOutcome::Acquired { term })
            }
            // Another write landed first and nothing is taken.
            ConditionalWriteOutcome::Moved => Ok(TickOutcome::Raced),
        }
    }
}

/// A term a caller says it saw must leave one to take: the term after it
/// stays below [`TERM_CEILING`]. The elector refuses one that does not
/// before it counts or writes it, and so does every door in front of it.
pub fn term_said(above: Option<u64>) -> Result<(), ElectError> {
    match above.map(|term| term.saturating_add(1)) {
        Some(next) if next >= TERM_CEILING => Err(ElectError::Term(next)),
        _ => Ok(()),
    }
}

/// `at` plus `by`, or `at` plus the largest of `by`'s halves the clock can
/// tell where the sum is past it: an instant that far never comes.
fn later(at: Instant, by: Duration) -> Instant {
    let mut by = by;
    loop {
        if let Some(sum) = at.checked_add(by) {
            return sum;
        }
        by /= 2;
    }
}

/// A duration in whole milliseconds, rounded up: the record's duration is
/// never shorter than the settings' and so never shorter than the renew
/// deadline the fence runs on, and a wait handed on is never shorter than
/// what was owed.
fn whole_millis_up(duration: Duration) -> Result<u64, ElectError> {
    let whole = duration.as_millis();
    let rounded = if duration.subsec_nanos().is_multiple_of(1_000_000) {
        whole
    } else {
        whole + 1
    };
    u64::try_from(rounded)
        .map_err(|_| ElectError::Settings("the duration is too large for the lease record"))
}

/// The wall clock in microseconds since the Unix epoch, and at least one
/// past `previous`: every write moves the record.
fn next_wall_micros(previous: u64) -> u64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|since| u64::try_from(since.as_micros()).unwrap_or(u64::MAX))
        .unwrap_or(0);
    now.max(previous.saturating_add(1))
}

/// The one-replica backend: a record in this process, shared by every clone
/// of the handle. A controller that runs alone elects itself through it; the
/// tests run several candidates over one.
#[derive(Clone, Debug, Default)]
pub struct MemoryLease {
    record: Arc<Mutex<Option<LeaseRecord>>>,
}

impl MemoryLease {
    pub fn new() -> Self {
        Self::default()
    }
}

impl LeaseBackend for MemoryLease {
    fn read(&mut self) -> Result<Option<LeaseRecord>, ElectError> {
        Ok(self.record.lock().clone())
    }

    fn write(
        &mut self,
        expected: Option<&LeaseRecord>,
        record: &LeaseRecord,
    ) -> Result<ConditionalWriteOutcome, ElectError> {
        let mut slot = self.record.lock();
        if slot.as_ref() != expected {
            return Ok(ConditionalWriteOutcome::Moved);
        }
        *slot = Some(record.clone());
        Ok(ConditionalWriteOutcome::Done)
    }

    fn place(&self) -> String {
        "memory".to_string()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            atomic::{AtomicBool, AtomicU32, Ordering},
            mpsc,
        },
        thread,
        time::Duration,
    };

    use super::*;

    /// One second: the step between the tests' instants.
    const S: Duration = Duration::from_secs(1);

    /// How far ahead of the clock the tests' instants start. The elector
    /// dates a first sight at the later of the caller's instant and its own
    /// clock after the read, so a synthetic instant must never be overtaken
    /// by the real clock. One second ahead was overtaken where a host held
    /// the test or the lease file's writes for longer; an hour is not.
    const AHEAD: Duration = S.saturating_mul(3_600);

    fn settings() -> ElectionSettings {
        ElectionSettings::default()
    }

    fn candidate(lease: &MemoryLease, identity: &str) -> Elector {
        Elector::new(Box::new(lease.clone()), identity, settings()).expect("valid settings")
    }

    /// A held record no candidate wrote.
    fn held_by(holder: &str, term: u64, duration_ms: u64) -> LeaseRecord {
        LeaseRecord {
            holder: holder.to_string(),
            term,
            acquired_at_micros: 1,
            renewed_at_micros: 1,
            duration_ms,
        }
    }

    impl Elector {
        /// The take of a lease where no record stood and the renewals after
        /// it, a period apart, up to `at`: the fence of such a take opens
        /// one renew deadline after it, a renewal behind it, and the last
        /// renewal here is the one at `at`. The wait is dated by the clock
        /// where the clock is past the instant handed, so `at` is a renew
        /// deadline ahead of the clock or more. An elector alone takes at
        /// `at` and leads at once. For the cases of this crate.
        pub(crate) fn lead_at(&mut self, at: Instant) {
            if self.alone {
                let take = self.tick(at).expect("the take");
                assert!(matches!(take, TickOutcome::Acquired { .. }), "{take:?}");
                assert!(self.fence.leader_term(at).is_some());
                return;
            }
            let (period, deadline) = (self.settings.renew_period, self.settings.renew_deadline);
            let steps = u32::try_from(deadline.as_nanos().div_ceil(period.as_nanos()))
                .expect("a count of periods");
            let took = at - period * steps;
            let take = self.tick(took).expect("the take");
            assert!(matches!(take, TickOutcome::Acquired { .. }), "{take:?}");
            for step in 1..=steps {
                assert_eq!(self.fence.leader_term(took + period * (step - 1)), None);
                let renewal = self.tick(took + period * step).expect("a renewal");
                assert!(
                    matches!(renewal, TickOutcome::Renewed { .. }),
                    "{renewal:?}"
                );
            }
            assert!(self.fence.leader_term(at).is_some());
        }
    }

    impl MemoryLease {
        /// The record gone by no holder's hand, as a deleted object or a
        /// deleted file: for the cases of this crate.
        pub(crate) fn lose(&self) {
            *self.record.lock() = None;
        }
    }

    #[test]
    fn an_elector_alone_over_its_own_lease_leads_at_its_take() {
        // Nothing can hold a store that was handed to nobody: no holder
        // this run never saw may lead, and the fence opens at the take.
        let mut alone = Elector::alone("a@host:1", settings()).expect("valid settings");
        let t0 = Instant::now() + AHEAD;
        assert_eq!(
            alone.tick(t0).expect("tick"),
            TickOutcome::Acquired { term: 1 }
        );
        assert_eq!(alone.fence().leader_term(t0), Some(1));
        // A store that is handed in keeps the wait, a memory one included:
        // its clones share it.
        let mut handed = candidate(&MemoryLease::new(), "b@host:2");
        assert_eq!(
            handed.tick(t0).expect("tick"),
            TickOutcome::Acquired { term: 1 }
        );
        assert_eq!(handed.fence().leader_term(t0), None);
    }

    /// A take's step runs behind the take's stamp and reads the fence as
    /// any reader does: the take's term where its window is open, and no
    /// term where the lead still waits to open. A renewal is no take and
    /// runs no step.
    #[test]
    fn a_takes_step_runs_behind_its_stamp() {
        let t0 = Instant::now() + AHEAD;
        // The steps a candidate ran, and whether the last was shown the
        // first term.
        let stepping = |elector: &mut Elector| {
            let fence = elector.fence();
            let (steps, shown) = (
                Arc::new(AtomicU32::new(0)),
                Arc::new(AtomicBool::new(false)),
            );
            let step = {
                let (steps, shown) = (Arc::clone(&steps), Arc::clone(&shown));
                move || {
                    steps.fetch_add(1, Ordering::Relaxed);
                    shown.store(fence.leader_term(t0) == Some(1), Ordering::Relaxed);
                }
            };
            elector.at_take(step);
            (steps, shown)
        };
        let mut alone = Elector::alone("a@host:1", settings()).expect("valid settings");
        let (steps, shown) = stepping(&mut alone);
        assert_eq!(
            alone.tick(t0).expect("tick"),
            TickOutcome::Acquired { term: 1 }
        );
        assert_eq!(steps.load(Ordering::Relaxed), 1);
        assert!(
            shown.load(Ordering::Relaxed),
            "the fence showed the take's term at the step"
        );
        assert_eq!(
            alone.tick(t0 + S).expect("renew"),
            TickOutcome::Renewed { term: 1 }
        );
        assert_eq!(steps.load(Ordering::Relaxed), 1, "a renewal is no take");

        // A take with no holder to wait for leads one renew deadline
        // later: its step is shown no term.
        let lease = MemoryLease::new();
        let mut waiting = candidate(&lease, "b@host:1");
        let (steps, shown) = stepping(&mut waiting);
        assert_eq!(
            waiting.tick(t0).expect("tick"),
            TickOutcome::Acquired { term: 1 }
        );
        assert_eq!(steps.load(Ordering::Relaxed), 1);
        assert!(
            !shown.load(Ordering::Relaxed),
            "the lead still waits to open"
        );
    }

    #[test]
    fn a_candidate_takes_an_absent_lease_at_term_one_and_leads_a_renew_deadline_later() {
        let lease = MemoryLease::new();
        let mut a = candidate(&lease, "a@host:1");
        let t0 = Instant::now() + AHEAD;
        assert_eq!(a.tick(t0).expect("tick"), TickOutcome::Acquired { term: 1 });
        assert!(a.is_holder());
        let record = a.observed().expect("the record").clone();
        assert_eq!(record.holder, "a@host:1");
        assert_eq!(record.duration_ms, 3_000);
        assert_eq!(record.acquired_at_micros, record.renewed_at_micros);
        assert!(record.acquired_at_micros > 0);
        // No record stood, so nobody was waited for, and a holder this run
        // never saw may lead to its renew deadline: the fence stays closed.
        let fence = a.fence();
        assert_eq!(fence.leader_term(t0), None);
        assert_eq!(fence.term(), 1, "the term is on the frames from the take");
        // The take alone carries the fence nowhere: its own renew deadline
        // ends where the lead would open.
        assert_eq!(fence.leader_term(t0 + 2 * S), None, "no renewal landed");
        assert_eq!(
            a.tick(t0 + S).expect("renew"),
            TickOutcome::Renewed { term: 1 }
        );
        assert_eq!(fence.leader_term(t0 + S), None, "inside the renew deadline");
        // The clock alone turns that answer twice from here: at the
        // opening still ahead, then at the window's end, and never after.
        assert_eq!(fence.turns_at(t0 + S), Some(t0 + 2 * S));
        assert_eq!(fence.turns_at(t0 + 2 * S), Some(t0 + 3 * S));
        assert_eq!(fence.turns_at(t0 + 3 * S), Some(t0 + 3 * S));
        let past = t0 + 3 * S + Duration::from_micros(1);
        assert_eq!(fence.turns_at(past), None);
        // The pair by one read is the two reads at one instant.
        for at in [t0 + S, t0 + 2 * S, t0 + 3 * S, past] {
            assert_eq!(
                fence.term_and_turn(at),
                (fence.leader_term(at), fence.turns_at(at))
            );
        }
        // One renew deadline after the take the fence opens by itself, at
        // no step: the renewal behind it carries the window past that
        // instant, to its own renew deadline.
        assert_eq!(
            fence.leader_term(t0 + 2 * S - Duration::from_micros(1)),
            None,
            "the instant before the opening"
        );
        assert_eq!(fence.leader_term(t0 + 2 * S), Some(1));
        assert_eq!(fence.leader_term(t0 + 3 * S), Some(1), "at the deadline");
        assert_eq!(
            fence.leader_term(t0 + 3 * S + Duration::from_micros(1)),
            None,
            "past the renew deadline the fence closes by itself"
        );
        // A renewal moves the end and never the opening: a reader whose
        // instant is behind the renewal's still reads the lead it is under.
        assert_eq!(
            a.tick(t0 + 3 * S).expect("renew"),
            TickOutcome::Renewed { term: 1 }
        );
        assert_eq!(fence.leader_term(t0 + 2 * S), Some(1));
        assert_eq!(fence.leader_term(t0 + 5 * S), Some(1), "at the deadline");
        assert_eq!(
            fence.leader_term(t0 + 5 * S + Duration::from_micros(1)),
            None
        );
    }

    #[test]
    fn a_candidate_takes_an_empty_record_and_leads_a_renew_deadline_later() {
        // A `Lease` made with no record, or replaced by such a manifest
        // under a live holder, reads as released under term zero. No holder
        // wrote that: no release lowers its term and a take writes one or
        // more. A candidate that never saw the lease held takes it at once,
        // and a holder it never saw may lead to its renew deadline: the
        // fence stays closed that long, as over no record.
        let mut lease = MemoryLease::new();
        let empty = LeaseRecord::default();
        assert_eq!(
            lease.write(None, &empty).expect("a write"),
            ConditionalWriteOutcome::Done
        );
        let mut a = candidate(&lease, "a@host:1");
        let t0 = Instant::now() + AHEAD;
        assert_eq!(a.tick(t0).expect("tick"), TickOutcome::Acquired { term: 1 });
        let fence = a.fence();
        assert_eq!(fence.leader_term(t0), None);
        assert_eq!(
            a.tick(t0 + S).expect("renew"),
            TickOutcome::Renewed { term: 1 }
        );
        assert_eq!(fence.leader_term(t0 + S), None, "inside the renew deadline");
        // The first renewal one renew deadline after the take opens it.
        assert_eq!(
            a.tick(t0 + 2 * S).expect("renew"),
            TickOutcome::Renewed { term: 1 }
        );
        assert_eq!(fence.leader_term(t0 + 2 * S), Some(1));
    }

    #[test]
    fn a_take_of_a_record_that_vanished_leads_a_renew_deadline_later() {
        // The wait before the take covers the holder this run saw. A record
        // taken by another and removed again inside that wait had a holder
        // this run never saw, so this take's fence opens late as well.
        let mut lease = MemoryLease::new();
        let held = held_by("x@host:8", 1, 3_000);
        assert_eq!(
            lease.write(None, &held).expect("a write"),
            ConditionalWriteOutcome::Done
        );
        let mut b = candidate(&lease, "b@host:2");
        let t0 = Instant::now() + AHEAD;
        assert!(matches!(
            b.tick(t0).expect("follow"),
            TickOutcome::Following { term: 1, .. }
        ));
        lease.lose();
        assert_eq!(b.tick(t0 + S).expect("look"), TickOutcome::Gone { term: 1 });
        let took = t0 + 4 * S;
        assert_eq!(
            b.tick(took).expect("take"),
            TickOutcome::Acquired { term: 2 }
        );
        let fence = b.fence();
        assert_eq!(fence.leader_term(took), None);
        assert_eq!(
            b.tick(took + S).expect("renew"),
            TickOutcome::Renewed { term: 2 }
        );
        assert_eq!(fence.leader_term(took + S), None);
        assert_eq!(
            b.tick(took + 2 * S).expect("renew"),
            TickOutcome::Renewed { term: 2 }
        );
        assert_eq!(fence.leader_term(took + 2 * S), Some(2));
    }

    #[test]
    fn a_replica_started_while_a_live_holders_record_is_gone_leads_after_that_holders_fence() {
        let lease = MemoryLease::new();
        let mut a = candidate(&lease, "a@host:1");
        let t0 = Instant::now() + AHEAD;
        a.lead_at(t0);
        // The record goes right after the holder's renewal, and the holder
        // stalls: its fence is open to its renew deadline. A replica that
        // starts now never saw the lease held and takes at once.
        lease.lose();
        let mut c = candidate(&lease, "c@host:3");
        let took = t0 + S / 10;
        assert_eq!(
            c.tick(took).expect("take"),
            TickOutcome::Acquired { term: 1 }
        );
        let (holders, takers) = (a.fence(), c.fence());
        assert_eq!(
            c.tick(took + S).expect("renew"),
            TickOutcome::Renewed { term: 1 }
        );
        // The holder's fence at its last open instant, the taker's closed.
        assert_eq!(holders.leader_term(t0 + 2 * S), Some(1));
        assert_eq!(takers.leader_term(t0 + 2 * S), None);
        // One renew deadline after the take: the taker's opens, the holder's
        // closed a tenth of a second before.
        assert_eq!(
            c.tick(took + 2 * S).expect("renew"),
            TickOutcome::Renewed { term: 1 }
        );
        assert_eq!(holders.leader_term(took + 2 * S), None);
        assert_eq!(takers.leader_term(took + 2 * S), Some(1));
    }

    #[test]
    fn a_record_removed_again_before_a_waiter_saw_its_taker_has_one_leader() {
        let mut lease = MemoryLease::new();
        let held = held_by("a@host:1", 1, 3_000);
        assert_eq!(
            lease.write(None, &held).expect("a write"),
            ConditionalWriteOutcome::Done
        );
        let (mut b, mut c) = (candidate(&lease, "b@host:2"), candidate(&lease, "c@host:3"));
        let t0 = Instant::now() + AHEAD;
        for waiter in [&mut b, &mut c] {
            assert!(matches!(
                waiter.tick(t0).expect("follow"),
                TickOutcome::Following { term: 1, .. }
            ));
        }
        lease.lose();
        for waiter in [&mut b, &mut c] {
            assert_eq!(
                waiter.tick(t0 + S).expect("look"),
                TickOutcome::Gone { term: 1 }
            );
        }
        // Both waits end at the same instant. b takes; the record goes again
        // before c looks, so c never saw b's record and takes under the same
        // term: a wait before the take cannot see that holder.
        let took = t0 + 4 * S;
        assert_eq!(
            b.tick(took).expect("take"),
            TickOutcome::Acquired { term: 2 }
        );
        lease.lose();
        let again = took + S / 2;
        assert_eq!(
            c.tick(again).expect("take"),
            TickOutcome::Acquired { term: 2 }
        );
        // Neither take opened a fence, and b's first renewal reads its loss:
        // one leader under the term, a renew deadline after its own take.
        assert_eq!(b.fence().leader_term(again), None);
        assert_eq!(c.fence().leader_term(again), None);
        assert_eq!(
            b.tick(took + S).expect("the renewal"),
            TickOutcome::Lost {
                term: 2,
                holder: "c@host:3".to_string()
            }
        );
        assert_eq!(
            c.tick(again + S).expect("renew"),
            TickOutcome::Renewed { term: 2 }
        );
        assert_eq!(c.fence().leader_term(again + S), None);
        assert_eq!(
            c.tick(again + 2 * S).expect("renew"),
            TickOutcome::Renewed { term: 2 }
        );
        assert_eq!(c.fence().leader_term(again + 2 * S), Some(2));
        assert_eq!(b.fence().leader_term(again + 2 * S), None);
    }

    #[test]
    fn a_renewal_reopens_the_fence_and_keeps_the_term() {
        let lease = MemoryLease::new();
        let mut a = candidate(&lease, "a@host:1");
        let t0 = Instant::now() + AHEAD;
        a.lead_at(t0);
        let first = a.observed().expect("the record").clone();
        assert_eq!(
            a.tick(t0 + S).expect("renew"),
            TickOutcome::Renewed { term: 1 }
        );
        let second = a.observed().expect("the record").clone();
        assert!(
            second.renewed_at_micros > first.renewed_at_micros,
            "every renewal moves the record"
        );
        assert_eq!(second.acquired_at_micros, first.acquired_at_micros);
        assert_eq!(second.term, 1);
        let fence = a.fence();
        assert_eq!(
            fence.leader_term(t0 + 3 * S),
            Some(1),
            "the deadline counts from the renewal"
        );
        assert_eq!(
            fence.leader_term(t0 + 3 * S + Duration::from_micros(1)),
            None
        );
    }

    #[test]
    fn a_follower_times_the_record_from_its_own_first_sight() {
        let lease = MemoryLease::new();
        let mut a = candidate(&lease, "a@host:1");
        let mut b = candidate(&lease, "b@host:2");
        let t0 = Instant::now() + AHEAD;
        a.tick(t0).expect("acquire");
        // b first sees the record half a second after it was written; the
        // record's clock says nothing to b.
        let t1 = t0 + S / 2;
        assert_eq!(
            b.tick(t1).expect("tick"),
            TickOutcome::Following {
                holder: "a@host:1".to_string(),
                term: 1
            }
        );
        assert!(matches!(
            b.tick(t0 + 3 * S).expect("tick"),
            TickOutcome::Following { .. }
        ));
        assert!(matches!(
            b.tick(t1 + 3 * S - Duration::from_micros(1)).expect("tick"),
            TickOutcome::Following { .. }
        ));
        assert_eq!(
            b.tick(t1 + 3 * S).expect("tick"),
            TickOutcome::Acquired { term: 2 }
        );
        assert_eq!(b.fence().leader_term(t1 + 3 * S), Some(2));
    }

    #[test]
    fn a_stalled_holders_renewal_is_refused_by_the_moved_record() {
        let lease = MemoryLease::new();
        let mut a = candidate(&lease, "a@host:1");
        let mut b = candidate(&lease, "b@host:2");
        let t0 = Instant::now() + AHEAD;
        a.tick(t0).expect("acquire");
        b.tick(t0).expect("follow");
        // a stalls past the duration; b takes the lease.
        assert_eq!(
            b.tick(t0 + 3 * S).expect("take"),
            TickOutcome::Acquired { term: 2 }
        );
        assert_eq!(
            a.fence().leader_term(t0 + 3 * S),
            None,
            "a fenced itself at 2 s"
        );
        assert_eq!(
            a.tick(t0 + 3 * S + S / 2).expect("the stalled renewal"),
            TickOutcome::Lost {
                term: 1,
                holder: "b@host:2".to_string()
            }
        );
        assert!(!a.is_holder());
        assert_eq!(a.fence().term(), 1, "the lost term stays on a's frames");
        assert_eq!(
            a.tick(t0 + 4 * S + S / 2).expect("tick"),
            TickOutcome::Following {
                holder: "b@host:2".to_string(),
                term: 2
            }
        );
        assert_eq!(
            b.tick(t0 + 4 * S).expect("renew"),
            TickOutcome::Renewed { term: 2 }
        );
    }

    #[test]
    fn a_renewing_holder_keeps_its_lease_against_a_waiting_candidate() {
        let lease = MemoryLease::new();
        let mut a = candidate(&lease, "a@host:1");
        let mut b = candidate(&lease, "b@host:2");
        let t0 = Instant::now() + AHEAD;
        a.lead_at(t0);
        for i in 1..=10u32 {
            let now = t0 + i * S;
            assert_eq!(
                a.tick(now).expect("renew"),
                TickOutcome::Renewed { term: 1 }
            );
            assert!(matches!(
                b.tick(now).expect("follow"),
                TickOutcome::Following { term: 1, .. }
            ));
            assert_eq!(a.fence().leader_term(now), Some(1));
            assert_eq!(b.fence().leader_term(now), None);
        }
    }

    #[test]
    fn a_release_inside_the_takes_wait_leaves_the_wait_to_the_next_candidate() {
        // b took a record with no holder to wait for and gives it up before
        // its fence opened. A holder b never saw may still lead to its renew
        // deadline, and b's wait was all that covered it: the release says
        // that deadline as its duration, and the next candidate waits it out
        // from its first sight of the release.
        let lease = MemoryLease::new();
        let mut b = candidate(&lease, "b@host:2");
        let mut c = candidate(&lease, "c@host:3");
        let t0 = Instant::now() + AHEAD;
        assert_eq!(b.tick(t0).expect("take"), TickOutcome::Acquired { term: 1 });
        assert!(matches!(
            c.tick(t0).expect("follow"),
            TickOutcome::Following { term: 1, .. }
        ));
        assert!(b.release().expect("release"));
        let released = b.observed().expect("the release").clone();
        assert!(released.is_released());
        assert_eq!(released.duration_ms, 2_000);
        let saw = t0 + Duration::from_millis(1);
        assert_eq!(c.tick(saw).expect("look"), TickOutcome::Gone { term: 1 });
        assert_eq!(
            c.tick(saw + 2 * S - Duration::from_micros(1))
                .expect("look"),
            TickOutcome::Gone { term: 1 }
        );
        assert_eq!(
            c.tick(saw + 2 * S).expect("take"),
            TickOutcome::Acquired { term: 2 }
        );
        assert_eq!(c.fence().leader_term(saw + 2 * S), Some(2));
    }

    #[test]
    fn a_releaser_waits_out_its_own_release_from_the_writes_return() {
        // b's take is handed an instant from before its read, so its own
        // sight of the record is dated there and its wait at the read's
        // return. Its release inside that wait is dated at the write's
        // return: b's own next take waits out what it handed on, as every
        // candidate does, and never the read's length short of it.
        let lease = MemoryLease::new();
        let mut b = candidate(&lease, "b@host:2");
        let before = Instant::now();
        thread::sleep(Duration::from_millis(20));
        assert_eq!(
            b.tick(before).expect("take"),
            TickOutcome::Acquired { term: 1 }
        );
        assert!(b.release().expect("release"));
        let released = Instant::now();
        assert_eq!(
            b.tick(before + 2 * S).expect("look"),
            TickOutcome::Gone { term: 1 }
        );
        assert_eq!(b.fence().leader_term(before + 2 * S), None);
        assert_eq!(
            b.tick(released + 2 * S).expect("take"),
            TickOutcome::Acquired { term: 2 }
        );
        assert_eq!(b.fence().leader_term(released + 2 * S), Some(2));
    }

    #[test]
    fn a_taker_whose_lease_went_waits_for_nothing() {
        // The wait belongs to the take. A loss ends it, and so does a
        // release: the reader says no from then on.
        let mut lease = MemoryLease::new();
        let mut a = candidate(&lease, "a@host:1");
        let t0 = Instant::now() + AHEAD;
        assert_eq!(a.tick(t0).expect("take"), TickOutcome::Acquired { term: 1 });
        assert!(a.waits_to_lead());
        let own = a.observed().expect("the take's record").clone();
        let other = held_by("x@host:8", 2, 3_000);
        assert_eq!(
            lease.write(Some(&own), &other).expect("a write"),
            ConditionalWriteOutcome::Done
        );
        assert!(matches!(
            a.tick(t0 + S).expect("look"),
            TickOutcome::Lost { term: 1, .. }
        ));
        assert!(!a.waits_to_lead());
        let mut b = candidate(&MemoryLease::new(), "b@host:2");
        assert_eq!(b.tick(t0).expect("take"), TickOutcome::Acquired { term: 1 });
        assert!(b.waits_to_lead());
        assert!(b.release().expect("release"));
        assert!(!b.waits_to_lead());
    }

    #[test]
    fn a_released_lease_is_taken_at_once_with_the_next_term() {
        let lease = MemoryLease::new();
        let mut a = candidate(&lease, "a@host:1");
        let mut b = candidate(&lease, "b@host:2");
        // The holder leads before it releases: its release is a leader's
        // word, and the next candidate leads at its take.
        let t0 = Instant::now() + AHEAD;
        a.lead_at(t0);
        assert!(matches!(
            b.tick(t0).expect("follow"),
            TickOutcome::Following { .. }
        ));
        assert!(a.release().expect("release"));
        assert!(!a.is_holder());
        assert_eq!(
            a.fence().leader_term(t0),
            None,
            "released: the fence closes now"
        );
        assert!(!a.release().expect("a second release"), "nothing held");
        let record = b.observed().expect("b's last read");
        assert_eq!(record.holder, "a@host:1", "b has not looked again yet");
        assert_eq!(
            b.tick(t0 + Duration::from_millis(1)).expect("take"),
            TickOutcome::Acquired { term: 2 }
        );
        assert_eq!(
            b.fence().leader_term(t0 + Duration::from_millis(1)),
            Some(2),
            "a holder's release is led at once"
        );
        assert!(
            a.observed().expect("a's last write").is_released(),
            "a's last sight of the record is its own release"
        );
    }

    #[test]
    fn a_restart_under_the_same_identity_takes_the_term_above() {
        let lease = MemoryLease::new();
        let t0 = Instant::now() + AHEAD;
        let acquired_at = {
            let mut a = candidate(&lease, "a@host:1");
            a.tick(t0).expect("acquire");
            a.observed().expect("the record").acquired_at_micros
        };
        // The process restarts: a fresh elector under the same identity. It
        // takes its earlier run's record once the record stood still for its
        // duration from its first sight, the wait of every candidate: a live
        // writer under the same name would have moved it by then. The take
        // is above the record's term: the earlier run led under that one,
        // and a term is taken once.
        let mut again = candidate(&lease, "a@host:1");
        let first_sight = t0 + S / 10;
        assert_eq!(
            again.tick(first_sight).expect("look"),
            TickOutcome::Following {
                holder: "a@host:1".to_string(),
                term: 1
            },
            "not yet: the earlier run may still be writing"
        );
        assert!(matches!(
            again
                .tick(first_sight + 3 * S - Duration::from_micros(1))
                .expect("look"),
            TickOutcome::Following { .. }
        ));
        assert_eq!(
            again.tick(first_sight + 3 * S).expect("take"),
            TickOutcome::Acquired { term: 2 }
        );
        let record = again.observed().expect("the record");
        assert_eq!(record.term, 2);
        assert!(
            record.acquired_at_micros > acquired_at,
            "a take in every field"
        );
        assert_eq!(record.acquired_at_micros, record.renewed_at_micros);
        assert_eq!(again.fence().leader_term(first_sight + 3 * S), Some(2));
    }

    #[test]
    fn a_restart_takes_only_once_the_stalled_earlier_runs_fence_is_closed() {
        let lease = MemoryLease::new();
        let t0 = Instant::now() + AHEAD;
        let mut earlier = candidate(&lease, "a@host:1");
        earlier.lead_at(t0);
        assert_eq!(
            earlier.tick(t0 + S).expect("renew"),
            TickOutcome::Renewed { term: 1 }
        );
        // The earlier run stalls with its fence open to the renew deadline,
        // and a new run under the same identity starts at once. A take
        // after a renew period of stillness would open a second fence beside
        // it; the new run waits for the record's duration instead.
        let mut again = candidate(&lease, "a@host:1");
        let first_sight = t0 + S + S / 10;
        assert!(matches!(
            again.tick(first_sight).expect("look"),
            TickOutcome::Following { .. }
        ));
        let deadline = t0 + S + 2 * S;
        assert_eq!(earlier.fence().leader_term(deadline), Some(1));
        assert!(matches!(
            again.tick(deadline).expect("look"),
            TickOutcome::Following { .. }
        ));
        assert_eq!(again.fence().leader_term(deadline), None);
        assert!(matches!(
            again
                .tick(first_sight + 3 * S - Duration::from_micros(1))
                .expect("look"),
            TickOutcome::Following { .. }
        ));
        assert_eq!(
            again.tick(first_sight + 3 * S).expect("take"),
            TickOutcome::Acquired { term: 2 }
        );
        assert_eq!(earlier.fence().leader_term(first_sight + 3 * S), None);
        assert_eq!(again.fence().leader_term(first_sight + 3 * S), Some(2));
    }

    #[test]
    fn two_electors_under_one_identity_never_both_lead() {
        let lease = MemoryLease::new();
        let mut a = candidate(&lease, "a@host:1");
        let mut twin = candidate(&lease, "a@host:1");
        let t0 = Instant::now() + AHEAD;
        assert_eq!(
            a.tick(t0).expect("acquire"),
            TickOutcome::Acquired { term: 1 }
        );
        // The twin sees a live record under its own name: it waits for the
        // record's duration of stillness, and the holder's renewal turns the
        // wait into a refusal, never a take.
        assert!(matches!(
            twin.tick(t0 + S / 2).expect("look"),
            TickOutcome::Following { .. }
        ));
        assert_eq!(
            a.tick(t0 + S).expect("renew"),
            TickOutcome::Renewed { term: 1 }
        );
        let err = twin
            .tick(t0 + S + S / 2)
            .expect_err("a second writer under this identity");
        assert_eq!(
            err.to_string(),
            "election identity: another candidate writes under this identity"
        );
        assert!(!twin.is_holder());
        for i in 2..6u32 {
            let now = t0 + i * S;
            assert_eq!(
                a.tick(now).expect("renew"),
                TickOutcome::Renewed { term: 1 }
            );
            twin.tick(now + S / 2)
                .expect_err("refused while the record moves");
            assert_eq!(a.fence().leader_term(now), Some(1));
            assert_eq!(twin.fence().leader_term(now + S / 2), None);
        }
        // The holder stalls after its renewal at `last`. Its fence is open to
        // the renew deadline; the twin keeps its distance for the record's
        // duration from its first sight of that renewal, then takes the
        // term above with the earlier fence closed: the refusal was not for
        // life.
        let last = t0 + 5 * S;
        let first_sight = last + S / 2;
        twin.tick(last + 2 * S)
            .expect_err("the record may still move");
        assert_eq!(a.fence().leader_term(last + 2 * S), Some(1));
        assert_eq!(twin.fence().leader_term(last + 2 * S), None);
        twin.tick(first_sight + 3 * S - Duration::from_micros(1))
            .expect_err("not yet");
        assert_eq!(
            twin.tick(first_sight + 3 * S).expect("take"),
            TickOutcome::Acquired { term: 2 }
        );
        assert_eq!(a.fence().leader_term(first_sight + 3 * S), None);
        assert_eq!(twin.fence().leader_term(first_sight + 3 * S), Some(2));
        // The stalled holder wakes to a record that moved under it.
        assert_eq!(
            a.tick(first_sight + 4 * S).expect("look"),
            TickOutcome::Lost {
                term: 1,
                holder: "a@host:1".to_string()
            }
        );
        assert_eq!(a.fence().leader_term(first_sight + 4 * S), None);
    }

    #[test]
    fn terms_only_grow_across_takes_and_releases() {
        let lease = MemoryLease::new();
        let mut a = candidate(&lease, "a@host:1");
        let mut b = candidate(&lease, "b@host:2");
        let mut c = candidate(&lease, "c@host:3");
        let t0 = Instant::now() + AHEAD;
        a.lead_at(t0);
        a.release().expect("release");
        assert_eq!(
            b.tick(t0 + S).expect("tick"),
            TickOutcome::Acquired { term: 2 }
        );
        c.tick(t0 + S).expect("c sees b");
        assert_eq!(
            c.tick(t0 + 4 * S).expect("b expired for c"),
            TickOutcome::Acquired { term: 3 }
        );
        c.release().expect("release");
        assert_eq!(
            a.tick(t0 + 5 * S).expect("tick"),
            TickOutcome::Acquired { term: 4 }
        );
        assert_eq!(
            b.tick(t0 + 5 * S).expect("b's stalled renewal"),
            TickOutcome::Lost {
                term: 2,
                holder: "a@host:1".to_string()
            }
        );
    }

    /// A backend whose first write runs what happens between this
    /// candidate's read and its write.
    struct Stalled {
        lease: MemoryLease,
        between: Option<Box<dyn FnOnce() + Send>>,
    }

    impl LeaseBackend for Stalled {
        fn read(&mut self) -> Result<Option<LeaseRecord>, ElectError> {
            self.lease.read()
        }

        fn write(
            &mut self,
            expected: Option<&LeaseRecord>,
            record: &LeaseRecord,
        ) -> Result<ConditionalWriteOutcome, ElectError> {
            if let Some(between) = self.between.take() {
                between();
            }
            self.lease.write(expected, record)
        }

        fn place(&self) -> String {
            "stalled".to_string()
        }
    }

    #[test]
    fn a_take_that_lands_after_a_later_holders_record_went_opens_after_that_holders_fence() {
        // b reads an absent lease and stalls before its write. c takes the
        // lease meanwhile and leads, and its record is removed: b's write
        // lands over an absent record again. c leads to one renew deadline
        // past its last renewal, which b never saw, so b's lead opens one
        // renew deadline after its own write returned, by the clock: no
        // instant has both fences open.
        let settings = ElectionSettings {
            duration: Duration::from_millis(60),
            renew_period: Duration::from_millis(10),
            renew_deadline: Duration::from_millis(40),
        };
        let lease = MemoryLease::new();
        let mut c = Elector::new(Box::new(lease.clone()), "c@host:3", settings).expect("settings");
        let c_fence = c.fence();
        let removed = lease.clone();
        let (last, renewed) = mpsc::channel();
        let between = move || {
            let take = c.tick(Instant::now()).expect("take");
            assert_eq!(take, TickOutcome::Acquired { term: 1 });
            let mut rounds = 0;
            let led = loop {
                thread::sleep(settings.renew_period);
                let now = Instant::now();
                assert_eq!(
                    c.tick(now).expect("renew"),
                    TickOutcome::Renewed { term: 1 }
                );
                if c.fence().leader_term(now).is_some() {
                    break now;
                }
                rounds += 1;
                assert!(rounds < 100, "the other holder never led");
            };
            last.send(led).expect("the case waits");
            removed.lose();
        };
        let backend = Stalled {
            lease: lease.clone(),
            between: Some(Box::new(between)),
        };
        let mut b = Elector::new(Box::new(backend), "b@host:2", settings).expect("settings");
        let take = b.tick(Instant::now()).expect("take");
        let returned = Instant::now();
        assert_eq!(take, TickOutcome::Acquired { term: 1 });
        let renewed = renewed.recv().expect("the other holder's last renewal");
        // b renews while c may still lead, and reads closed to the end of
        // c's fence.
        let within = renewed + settings.renew_deadline / 2;
        assert_eq!(
            b.tick(within).expect("renew"),
            TickOutcome::Renewed { term: 1 }
        );
        assert_eq!(c_fence.leader_term(within), Some(1));
        assert_eq!(b.fence().leader_term(within), None);
        let closes = renewed + settings.renew_deadline;
        assert_eq!(c_fence.leader_term(closes), Some(1));
        assert_eq!(b.fence().leader_term(closes), None);
        // b's lead opens past it, one renew deadline after its write
        // returned: the step that returned is no earlier than that write.
        let opens = returned + settings.renew_deadline;
        assert!(opens > closes);
        assert_eq!(
            b.tick(opens).expect("renew"),
            TickOutcome::Renewed { term: 1 }
        );
        assert_eq!(b.fence().leader_term(opens), Some(1));
        assert_eq!(c_fence.leader_term(opens), None);
    }

    #[test]
    fn a_release_past_the_takes_opening_hands_on_no_wait() {
        // The wait ends at its instant by the clock, whatever the steps
        // since. One renew deadline after the take's write no holder this
        // run never saw leads, so a release from then on says no duration:
        // the next candidate takes at once and leads at its take.
        let settings = ElectionSettings {
            duration: Duration::from_millis(60),
            renew_period: Duration::from_millis(10),
            renew_deadline: Duration::from_millis(40),
        };
        let lease = MemoryLease::new();
        let mut b = Elector::new(Box::new(lease.clone()), "b@host:2", settings).expect("settings");
        let take = b.tick(Instant::now()).expect("take");
        assert_eq!(take, TickOutcome::Acquired { term: 1 });
        assert!(b.waits_to_lead());
        // No step since the take: the clock alone ends the wait.
        thread::sleep(settings.renew_deadline);
        assert!(!b.waits_to_lead(), "the wait ended at its instant");
        assert!(b.release().expect("release"));
        let released = b.observed().expect("the release").clone();
        assert!(released.is_released());
        assert_eq!(released.duration_ms, 0);
        let mut c = Elector::new(Box::new(lease.clone()), "c@host:3", settings).expect("settings");
        let now = Instant::now();
        assert_eq!(
            c.tick(now).expect("take"),
            TickOutcome::Acquired { term: 2 }
        );
        assert_eq!(c.fence().leader_term(now), Some(2));
    }

    /// A backend that lets another candidate write between this candidate's
    /// read and its write: the race a compare-and-swap exists for. A case
    /// may set the rival after the candidate's take.
    struct Interposed {
        lease: MemoryLease,
        rival: Arc<Mutex<Option<LeaseRecord>>>,
    }

    impl LeaseBackend for Interposed {
        fn read(&mut self) -> Result<Option<LeaseRecord>, ElectError> {
            self.lease.read()
        }

        fn write(
            &mut self,
            expected: Option<&LeaseRecord>,
            record: &LeaseRecord,
        ) -> Result<ConditionalWriteOutcome, ElectError> {
            if let Some(rival) = self.rival.lock().take() {
                assert_eq!(
                    self.lease
                        .write(expected, &rival)
                        .expect("the rival's write"),
                    ConditionalWriteOutcome::Done
                );
            }
            self.lease.write(expected, record)
        }

        fn place(&self) -> String {
            "interposed".to_string()
        }
    }

    #[test]
    fn a_take_the_rival_won_is_a_race_and_the_next_tick_follows() {
        let lease = MemoryLease::new();
        let rival = LeaseRecord {
            holder: "r@host:9".to_string(),
            term: 1,
            acquired_at_micros: 7,
            renewed_at_micros: 7,
            duration_ms: 3_000,
        };
        let backend = Interposed {
            lease: lease.clone(),
            rival: Arc::new(Mutex::new(Some(rival.clone()))),
        };
        let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
        let t0 = Instant::now() + AHEAD;
        assert_eq!(a.tick(t0).expect("tick"), TickOutcome::Raced);
        assert!(!a.is_holder());
        assert!(!a.waits_to_lead(), "a take a rival won waits for no lead");
        assert_eq!(a.fence().leader_term(t0), None);
        assert_eq!(
            a.tick(t0 + S).expect("tick"),
            TickOutcome::Following {
                holder: "r@host:9".to_string(),
                term: 1
            }
        );
        assert_eq!(lease.clone().read().expect("read"), Some(rival));
    }

    /// A holder over a store where another replica's write lands between
    /// its next read and its next write: `theirs` makes that record from
    /// the holder's own, and it is stored one wall instant after it.
    fn led_before_a_rivals_write(
        t0: Instant,
        theirs: impl FnOnce(LeaseRecord) -> LeaseRecord,
    ) -> Elector {
        let lease = MemoryLease::new();
        let rival = Arc::new(Mutex::new(None));
        let backend = Interposed {
            lease: lease.clone(),
            rival: Arc::clone(&rival),
        };
        let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
        a.lead_at(t0);
        let mine = lease.clone().read().expect("read").expect("the record");
        *rival.lock() = Some(LeaseRecord {
            renewed_at_micros: mine.renewed_at_micros + 1,
            ..theirs(mine)
        });
        a
    }

    /// A holder over a store where another replica's release under term
    /// nine lands between its next read and its next write.
    fn led_before_a_rivals_release(t0: Instant) -> Elector {
        led_before_a_rivals_write(t0, |mine| LeaseRecord {
            holder: String::new(),
            term: 9,
            duration_ms: 0,
            ..mine
        })
    }

    #[test]
    fn a_release_that_lands_under_a_renewals_write_is_observed_at_the_read_after_it() {
        // The renewal's write comes back moved and the lease is read again.
        // That record is a sight as any other: the run counts its term, so
        // the standing reads it and the next take is above it.
        let t0 = Instant::now() + AHEAD;
        let mut a = led_before_a_rivals_release(t0);
        assert_eq!(
            a.tick(t0 + S).expect("the look that loses the lease"),
            TickOutcome::Lost {
                term: 1,
                holder: String::new()
            }
        );
        assert_eq!(a.observed().map(|record| record.term), Some(9));
        assert_eq!(a.highest_term(), 9);
        assert_eq!(
            a.tick(t0 + 2 * S).expect("take"),
            TickOutcome::Acquired { term: 10 }
        );
    }

    #[test]
    fn a_release_that_lands_under_a_stops_write_is_observed_at_the_read_after_it() {
        // The same race at a stop: the release's write comes back moved, the
        // lease read again is released under a term above this run's, and
        // the run counts that term.
        let t0 = Instant::now() + AHEAD;
        let mut a = led_before_a_rivals_release(t0);
        assert!(a.release().expect("the release"));
        assert!(!a.is_holder());
        assert_eq!(a.observed().map(|record| record.term), Some(9));
        assert_eq!(a.highest_term(), 9);
    }

    #[test]
    fn a_take_that_lands_under_a_renewals_write_is_observed_at_the_read_after_it() {
        // The rival's write is a take, a held record under the next term.
        // The look that loses the lease names its holder; the run counts
        // its term and keeps the record, and its wait for that record
        // counts from the read that found it.
        let t0 = Instant::now() + AHEAD;
        let mut a = led_before_a_rivals_write(t0, |mine| LeaseRecord {
            holder: "b@host:2".to_owned(),
            term: 2,
            ..mine
        });
        assert_eq!(
            a.tick(t0 + S).expect("the look that loses the lease"),
            TickOutcome::Lost {
                term: 1,
                holder: "b@host:2".to_owned()
            }
        );
        assert_eq!(
            a.observed()
                .map(|record| (record.holder.as_str(), record.term)),
            Some(("b@host:2", 2))
        );
        assert_eq!(a.highest_term(), 2);
        // The rival's record stood still for its duration since that read:
        // a look a microsecond under the wait's end still follows it, and
        // the take at its end is above the term counted.
        assert_eq!(
            a.tick(t0 + 4 * S - Duration::from_micros(1))
                .expect("a look inside the wait"),
            TickOutcome::Following {
                holder: "b@host:2".to_owned(),
                term: 2
            }
        );
        assert_eq!(
            a.tick(t0 + 4 * S).expect("the take after the wait"),
            TickOutcome::Acquired { term: 3 }
        );
    }

    /// A backend whose writes fail while `fails` is above zero, as a stalled
    /// file system or an API server away would.
    struct Failing {
        lease: MemoryLease,
        fails: Arc<AtomicU32>,
    }

    impl LeaseBackend for Failing {
        fn read(&mut self) -> Result<Option<LeaseRecord>, ElectError> {
            self.lease.read()
        }

        fn write(
            &mut self,
            expected: Option<&LeaseRecord>,
            record: &LeaseRecord,
        ) -> Result<ConditionalWriteOutcome, ElectError> {
            if self.fails.load(Ordering::Relaxed) > 0 {
                self.fails.fetch_sub(1, Ordering::Relaxed);
                return Err(ElectError::Io {
                    place: self.place(),
                    source: std::io::Error::other("the backend is away"),
                });
            }
            self.lease.write(expected, record)
        }

        fn place(&self) -> String {
            "failing".to_string()
        }
    }

    #[test]
    fn a_failed_renewal_leaves_the_holder_its_record_and_the_fence_to_its_deadline() {
        let lease = MemoryLease::new();
        let fails = Arc::new(AtomicU32::new(0));
        let backend = Failing {
            lease: lease.clone(),
            fails: Arc::clone(&fails),
        };
        let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
        let t0 = Instant::now() + AHEAD;
        a.lead_at(t0);
        fails.store(1, Ordering::Relaxed);
        let err = a.tick(t0 + S).expect_err("the backend is away");
        assert_eq!(err.to_string(), "the lease at failing: the backend is away");
        assert!(a.is_holder(), "an error changes nothing");
        assert_eq!(a.observed().expect("the record").holder, "a@host:1");
        let fence = a.fence();
        assert_eq!(
            fence.leader_term(t0 + S),
            Some(1),
            "the fence runs to its deadline"
        );
        assert_eq!(
            fence.leader_term(t0 + 2 * S + Duration::from_micros(1)),
            None,
            "and closes on its own past it"
        );
        assert_eq!(
            a.tick(t0 + 2 * S).expect("the backend is back"),
            TickOutcome::Renewed { term: 1 }
        );
        assert_eq!(
            fence.leader_term(t0 + 4 * S),
            Some(1),
            "reopened by the renewal"
        );
        assert_eq!(
            lease
                .clone()
                .read()
                .expect("read")
                .expect("the record")
                .holder,
            "a@host:1"
        );
    }

    #[test]
    fn a_take_whose_renewals_do_not_land_never_leads() {
        // The lead of a take with no holder to wait for opens one renew
        // deadline after the take, and a fence runs one renew deadline past
        // its last renewal: with no renewal landed the two meet, and no
        // instant reads this term as the leader's.
        let lease = MemoryLease::new();
        let fails = Arc::new(AtomicU32::new(0));
        let backend = Failing {
            lease: lease.clone(),
            fails: Arc::clone(&fails),
        };
        let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
        let t0 = Instant::now() + AHEAD;
        assert_eq!(a.tick(t0).expect("take"), TickOutcome::Acquired { term: 1 });
        fails.store(2, Ordering::Relaxed);
        a.tick(t0 + S).expect_err("the backend is away");
        a.tick(t0 + 2 * S).expect_err("the backend is away");
        let fence = a.fence();
        for after in [0, 2_000_000, 2_000_001, 5_000_000] {
            let at = t0 + Duration::from_micros(after);
            assert_eq!(fence.leader_term(at), None, "{after} us after the take");
        }
        assert!(!a.waits_to_lead(), "the wait ended at its instant");
        // A renewal that lands after all leads from its own instant, as a
        // renewal past a fence that closed by itself does.
        let late = t0 + 2 * S + Duration::from_millis(500);
        assert_eq!(
            a.tick(late).expect("renew"),
            TickOutcome::Renewed { term: 1 }
        );
        assert_eq!(fence.leader_term(late - Duration::from_micros(1)), None);
        assert_eq!(fence.leader_term(late), Some(1));
        assert_eq!(fence.leader_term(late + 2 * S), Some(1), "at the deadline");
    }

    #[test]
    fn a_take_whose_write_failed_waits_for_nothing_while_it_follows_a_rival() {
        // The wait belongs to a take that landed with its answer. An attempt
        // in doubt is no take: the reader says no from the failed write on,
        // and through the steps that follow the rival that took the lease.
        let lease = MemoryLease::new();
        let backend = Failing {
            lease: lease.clone(),
            fails: Arc::new(AtomicU32::new(1)),
        };
        let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
        let mut b = candidate(&lease, "b@host:2");
        let t0 = Instant::now() + AHEAD;
        a.tick(t0).expect_err("the backend is away");
        assert!(!a.waits_to_lead(), "an attempt in doubt is no take");
        assert_eq!(b.tick(t0).expect("take"), TickOutcome::Acquired { term: 1 });
        assert!(b.waits_to_lead());
        assert!(matches!(
            a.tick(t0 + S).expect("follow"),
            TickOutcome::Following { term: 1, .. }
        ));
        assert!(!a.waits_to_lead(), "a follower waits for no lead");
    }

    #[test]
    fn a_failed_release_keeps_the_record_for_a_retry_and_the_fence_closed() {
        let lease = MemoryLease::new();
        let fails = Arc::new(AtomicU32::new(0));
        let backend = Failing {
            lease: lease.clone(),
            fails: Arc::clone(&fails),
        };
        let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
        let t0 = Instant::now() + AHEAD;
        a.tick(t0).expect("acquire");
        fails.store(1, Ordering::Relaxed);
        a.release().expect_err("the backend is away");
        assert!(a.is_holder(), "kept for the retry");
        assert_eq!(a.fence().leader_term(t0), None, "closed by the first call");
        assert!(a.release().expect("the retry"), "released");
        assert!(!a.is_holder());
        assert!(lease
            .clone()
            .read()
            .expect("read")
            .expect("the record")
            .is_released());
        assert!(a.observed().expect("the last write").is_released());
    }

    #[test]
    fn a_holder_told_to_leave_releases_and_takes_nothing_for_the_duration_its_record_said() {
        // The hold is one lease duration, the one the holder's own record
        // said, from the word. Nobody else looks here, so the lease stands
        // free through it and the replica that left takes it back under the
        // next term.
        let lease = MemoryLease::new();
        let mut a = candidate(&lease, "a@host:1");
        let t0 = Instant::now() + AHEAD;
        a.lead_at(t0);
        let word = t0 + S / 2;
        assert!(a.step_down(None, word).expect("the release"));
        assert!(!a.is_holder());
        assert_eq!(a.fence().leader_term(word), None, "closed at the word");
        let stored = lease.clone().read().expect("read").expect("the record");
        assert_eq!(
            (stored.holder.as_str(), stored.term, stored.duration_ms),
            ("", 1, 0),
            "a holder's release under its own term"
        );
        for wait in [Duration::ZERO, S, 3 * S - Duration::from_micros(1)] {
            assert_eq!(
                a.tick(word + wait).expect("look"),
                TickOutcome::Waiting { term: 1 },
                "{wait:?}"
            );
        }
        assert_eq!(
            a.tick(word + 3 * S).expect("take"),
            TickOutcome::Acquired { term: 2 }
        );
        assert_eq!(a.fence().leader_term(word + 3 * S), Some(2));
        // The take ends the hold: the holder renews from here.
        assert_eq!(
            a.tick(word + 4 * S).expect("renew"),
            TickOutcome::Renewed { term: 2 }
        );
    }

    #[test]
    fn a_replica_whose_hold_ended_over_a_release_that_still_says_a_duration_reads_it_gone() {
        // The waiting answer ends with the hold. A release another replica
        // wrote inside the hold says a duration that outlasts it: from the
        // hold's end this replica is a candidate as any other, and a record
        // with no holder and a duration is the gone record's answer again.
        let lease = MemoryLease::new();
        let mut a = candidate(&lease, "a@host:1");
        let t0 = Instant::now() + AHEAD;
        a.lead_at(t0);
        let word = t0 + S / 2;
        assert!(a.step_down(None, word).expect("the release"));
        let own = lease.clone().read().expect("read").expect("the record");
        let later = LeaseRecord {
            term: 2,
            renewed_at_micros: own.renewed_at_micros + 1,
            duration_ms: 60_000,
            ..own.clone()
        };
        assert_eq!(
            lease.clone().write(Some(&own), &later).expect("a write"),
            ConditionalWriteOutcome::Done
        );
        assert_eq!(
            a.tick(word + S).expect("look"),
            TickOutcome::Waiting { term: 2 }
        );
        assert_eq!(
            a.tick(word + 3 * S - Duration::from_micros(1))
                .expect("look"),
            TickOutcome::Waiting { term: 2 }
        );
        assert_eq!(
            a.tick(word + 3 * S).expect("look"),
            TickOutcome::Gone { term: 2 }
        );
        assert!(!a.is_holder());
    }

    #[test]
    fn a_standby_takes_a_stepped_down_holders_lease_at_its_next_look() {
        let lease = MemoryLease::new();
        let mut a = candidate(&lease, "a@host:1");
        let t0 = Instant::now() + AHEAD;
        a.lead_at(t0);
        let mut b = candidate(&lease, "b@host:2");
        assert!(matches!(
            b.tick(t0).expect("follow"),
            TickOutcome::Following { term: 1, .. }
        ));
        let word = t0 + S / 2;
        assert!(a.step_down(None, word).expect("the release"));
        // A holder's release: taken at once and led at once.
        assert_eq!(
            b.tick(word).expect("take"),
            TickOutcome::Acquired { term: 2 }
        );
        assert_eq!(b.fence().leader_term(word), Some(2));
        // The replica that left follows the taker, and still holds its own
        // takes back when the taker stops inside the hold.
        assert!(matches!(
            a.tick(word + S).expect("follow"),
            TickOutcome::Following { term: 2, .. }
        ));
        assert!(b.release().expect("the taker's stop"));
        assert_eq!(
            a.tick(word + 2 * S).expect("look"),
            TickOutcome::Waiting { term: 2 }
        );
        assert_eq!(
            a.tick(word + 3 * S).expect("take"),
            TickOutcome::Acquired { term: 3 }
        );
    }

    #[test]
    fn a_term_told_to_a_holder_goes_into_the_record_it_releases() {
        // A holder told a term above its own leaves, and its release carries
        // that term: whoever takes next leads above it. A release never
        // lowers the term, so no candidate reads this record as gone.
        let lease = MemoryLease::new();
        let mut a = candidate(&lease, "a@host:1");
        let t0 = Instant::now() + AHEAD;
        a.lead_at(t0);
        let mut b = candidate(&lease, "b@host:2");
        assert!(matches!(
            b.tick(t0).expect("follow"),
            TickOutcome::Following { term: 1, .. }
        ));
        let word = t0 + S / 2;
        assert!(a.step_down(Some(7), word).expect("the release"));
        let stored = lease.clone().read().expect("read").expect("the record");
        assert_eq!(
            (stored.holder.as_str(), stored.term, stored.duration_ms),
            ("", 7, 0)
        );
        assert_eq!(
            a.tick(word).expect("look"),
            TickOutcome::Waiting { term: 7 }
        );
        assert_eq!(
            b.tick(word).expect("take"),
            TickOutcome::Acquired { term: 8 }
        );
        assert_eq!(b.fence().leader_term(word), Some(8));
    }

    #[test]
    fn an_elector_alone_told_a_term_takes_above_it_at_its_next_step() {
        // No other candidate can take a lease the elector made itself, so a
        // word holds nothing back: the fence closes at the word, and the
        // next step takes the lease again above the term said and leads.
        let mut a = Elector::alone("a@host:1", settings()).expect("settings");
        let t0 = Instant::now() + AHEAD;
        a.lead_at(t0);
        assert!(a.step_down(Some(5), t0 + S).expect("the word"));
        assert_eq!(a.fence().leader_term(t0 + S), None);
        assert_eq!(
            a.tick(t0 + S).expect("take"),
            TickOutcome::Acquired { term: 6 }
        );
        assert_eq!(a.fence().leader_term(t0 + S), Some(6));
    }

    #[test]
    fn a_word_inside_the_takes_wait_hands_the_wait_on_as_a_stop_does() {
        // A taker told to leave before its fence opened never led, so its
        // release pays no wait: the record says the second its wait still
        // owed as its duration, as at a stop inside the wait.
        let lease = MemoryLease::new();
        let mut a = candidate(&lease, "a@host:1");
        let t0 = Instant::now() + AHEAD;
        assert_eq!(a.tick(t0).expect("take"), TickOutcome::Acquired { term: 1 });
        assert!(a.waits_to_lead());
        assert!(a.step_down(None, t0 + S).expect("the word"));
        let released = a.observed().expect("the release").clone();
        assert!(released.is_released());
        assert_eq!(released.duration_ms, 1_000);
        assert!(!a.waits_to_lead());
        // Its own next steps wait their turn: the release it reads is its
        // own, and no record is gone.
        assert_eq!(
            a.tick(t0 + S + Duration::from_millis(1)).expect("a step"),
            TickOutcome::Waiting { term: 1 }
        );
        // A term said while it waits is the term its wait says, the
        // release's own term below it.
        assert!(!a
            .step_down(Some(5), t0 + S + Duration::from_millis(2))
            .expect("a word"));
        assert_eq!(
            a.tick(t0 + S + Duration::from_millis(3)).expect("a step"),
            TickOutcome::Waiting { term: 5 }
        );
        // A word once the opening has passed finds no wait to hand on: no
        // holder this run never saw leads by then, and the release says no
        // duration.
        let mut b = candidate(&MemoryLease::new(), "b@host:2");
        assert_eq!(b.tick(t0).expect("take"), TickOutcome::Acquired { term: 1 });
        assert!(b.step_down(None, t0 + 2 * S).expect("the word"));
        assert_eq!(b.observed().expect("the release").duration_ms, 0);
    }

    #[test]
    fn a_higher_term_said_after_the_release_goes_into_the_released_record() {
        // Three gateways show rising floors to one leader. The first word
        // releases the lease under five; the second, nine, finds no lease
        // held here and is written over the release; the third, twelve, is
        // written over the record the second wrote, which this run's sight
        // of the lease moved to. The candidate that takes next takes above
        // twelve, so no gateway refuses its lead.
        let lease = MemoryLease::new();
        let mut a = candidate(&lease, "a@host:1");
        let t0 = Instant::now() + AHEAD;
        a.lead_at(t0);
        let mut b = candidate(&lease, "b@host:2");
        assert!(matches!(
            b.tick(t0).expect("follow"),
            TickOutcome::Following { term: 1, .. }
        ));
        let word = t0 + S / 2;
        assert!(a.step_down(Some(5), word).expect("the release"));
        assert!(!a.step_down(Some(9), word).expect("the term is written"));
        let stored = lease.clone().read().expect("read").expect("the record");
        assert_eq!(
            (stored.holder.as_str(), stored.term, stored.duration_ms),
            ("", 9, 0)
        );
        assert!(!a.step_down(Some(12), word).expect("the term is written"));
        let stored = lease.clone().read().expect("read").expect("the record");
        assert_eq!(
            (stored.holder.as_str(), stored.term, stored.duration_ms),
            ("", 12, 0)
        );
        // A term at or under the record's writes nothing more.
        assert!(!a.step_down(Some(12), word).expect("no write"));
        assert!(!a.step_down(Some(7), word).expect("no write"));
        assert_eq!(lease.clone().read().expect("read"), Some(stored));
        assert_eq!(
            a.tick(word).expect("look"),
            TickOutcome::Waiting { term: 12 }
        );
        assert_eq!(
            b.tick(word).expect("take"),
            TickOutcome::Acquired { term: 13 }
        );
        assert_eq!(b.fence().leader_term(word), Some(13));
    }

    #[test]
    fn a_term_said_over_a_release_that_moved_is_counted_and_writes_nothing() {
        // The candidate took the released lease before the higher term was
        // said: the record is that holder's and stays as it is. The gateway
        // that holds the term tells it to that holder.
        let lease = MemoryLease::new();
        let mut a = candidate(&lease, "a@host:1");
        let t0 = Instant::now() + AHEAD;
        a.lead_at(t0);
        let mut b = candidate(&lease, "b@host:2");
        assert!(matches!(
            b.tick(t0).expect("follow"),
            TickOutcome::Following { term: 1, .. }
        ));
        let word = t0 + S / 2;
        assert!(a.step_down(Some(5), word).expect("the release"));
        assert_eq!(
            b.tick(word).expect("take"),
            TickOutcome::Acquired { term: 6 }
        );
        assert!(!a.step_down(Some(9), word).expect("the record moved"));
        let stored = lease.clone().read().expect("read").expect("the record");
        assert_eq!((stored.holder.as_str(), stored.term), ("b@host:2", 6));
        assert_eq!(a.highest_term(), 9);
    }

    #[test]
    fn a_term_whose_write_failed_goes_into_the_release_at_the_next_word() {
        // The write of nine over the release fails: the term is counted and
        // the record stays under five. A lower word, seven, then writes the
        // highest term this run saw and not its own: the candidate that
        // takes next takes above nine.
        let lease = MemoryLease::new();
        let fails = Arc::new(AtomicU32::new(0));
        let backend = Failing {
            lease: lease.clone(),
            fails: Arc::clone(&fails),
        };
        let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
        let t0 = Instant::now() + AHEAD;
        a.lead_at(t0);
        let mut b = candidate(&lease, "b@host:2");
        assert!(matches!(
            b.tick(t0).expect("follow"),
            TickOutcome::Following { term: 1, .. }
        ));
        let word = t0 + S / 2;
        assert!(a.step_down(Some(5), word).expect("the release"));
        fails.store(1, Ordering::Relaxed);
        a.step_down(Some(9), word).expect_err("the backend is away");
        let stored = lease.clone().read().expect("read").expect("the record");
        assert_eq!((stored.term, a.highest_term()), (5, 9));
        assert!(!a.step_down(Some(7), word).expect("the highest is written"));
        let stored = lease.clone().read().expect("read").expect("the record");
        assert_eq!(
            (stored.holder.as_str(), stored.term, stored.duration_ms),
            ("", 9, 0)
        );
        assert_eq!(
            b.tick(word).expect("take"),
            TickOutcome::Acquired { term: 10 }
        );
    }

    #[test]
    fn a_term_said_over_a_record_gone_for_the_run_writes_nothing() {
        // A hand writes an old release over a holder's record. For the
        // candidate that saw that holder the record is gone: no holder's
        // release made it so, and the holder may still lead. A term said
        // inside the wait the candidate owes is counted and the record
        // stays as it is: written under the term it would read as the
        // holder's release and be taken at once, beside an open fence.
        let lease = MemoryLease::new();
        let mut w = candidate(&lease, "w@host:1");
        let t0 = Instant::now() + AHEAD;
        w.lead_at(t0);
        let mut x = candidate(&lease, "x@host:2");
        let mut c = candidate(&lease, "c@host:3");
        x.tick(t0).expect("follow");
        assert!(w.step_down(Some(4), t0 + S).expect("the release"));
        let released = lease.clone().read().expect("read").expect("the release");
        let took = x.tick(t0 + S).expect("take");
        assert!(matches!(took, TickOutcome::Acquired { .. }), "{took:?}");
        c.tick(t0 + 2 * S).expect("follow");
        x.tick(t0 + 2 * S).expect("renew");
        let held = lease.clone().read().expect("read").expect("the record");
        assert_eq!(
            lease
                .clone()
                .write(Some(&held), &released)
                .expect("a write"),
            ConditionalWriteOutcome::Done
        );
        let at = t0 + 3 * S;
        let gone = c.tick(at).expect("look");
        assert!(!matches!(gone, TickOutcome::Acquired { .. }), "{gone:?}");
        let said = held.term;
        assert!(!c.step_down(Some(said), at).expect("the word"));
        assert_eq!(lease.clone().read().expect("read"), Some(released));
        assert_eq!(c.highest_term(), said);
        let next = c.tick(at).expect("look");
        assert!(!matches!(next, TickOutcome::Acquired { .. }), "{next:?}");
        assert_eq!(c.fence().leader_term(at), None);
        assert_eq!(x.fence().leader_term(at), Some(said));
    }

    #[test]
    fn a_term_said_over_an_empty_record_writes_nothing() {
        // An empty record, released under term zero, is no holder's
        // release: a manifest put under a live holder reads so, and the
        // take over it opens its fence a renew deadline late for a holder
        // the taker never saw. A candidate whose take over it failed is
        // told a term: the term is counted and the record stays empty.
        // Written under the term it would read as a release, and the next
        // take would lead at once.
        let mut lease = MemoryLease::new();
        let empty = LeaseRecord::default();
        assert_eq!(
            lease.write(None, &empty).expect("a write"),
            ConditionalWriteOutcome::Done
        );
        let fails = Arc::new(AtomicU32::new(1));
        let backend = Failing {
            lease: lease.clone(),
            fails: Arc::clone(&fails),
        };
        let mut c = Elector::new(Box::new(backend), "c@host:3", settings()).expect("settings");
        let t0 = Instant::now() + AHEAD;
        c.tick(t0).expect_err("the backend is away");
        assert!(!c.step_down(Some(5), t0).expect("the word"));
        assert_eq!(lease.clone().read().expect("read"), Some(empty));
        assert_eq!(c.highest_term(), 5);
        // The next take is still a take over an empty record: at once, and
        // its fence a renew deadline late.
        let mut d = candidate(&lease, "d@host:4");
        assert_eq!(d.tick(t0).expect("take"), TickOutcome::Acquired { term: 1 });
        let fence = d.fence();
        assert_eq!(fence.leader_term(t0), None);
        assert_eq!(
            d.tick(t0 + S).expect("renew"),
            TickOutcome::Renewed { term: 1 }
        );
        assert_eq!(fence.leader_term(t0 + S), None, "inside the renew deadline");
        assert_eq!(
            d.tick(t0 + 2 * S).expect("renew"),
            TickOutcome::Renewed { term: 1 }
        );
        assert_eq!(fence.leader_term(t0 + 2 * S), Some(1));
    }

    #[test]
    fn a_term_told_while_a_take_is_in_doubt_stands_over_the_take_that_landed() {
        // The take lands and its answer is lost: the run holds nothing it
        // knows of when a term of seven is said, and the term is counted.
        // The step that finds the take landed under term one leaves the
        // lease as the word would have: no step leads under a term below
        // the highest its run saw.
        let lease = MemoryLease::new();
        let fails = Arc::new(AtomicU32::new(1));
        let backend = LandsThenErrors {
            lease: lease.clone(),
            fails: Arc::clone(&fails),
        };
        let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
        let t0 = Instant::now() + AHEAD;
        a.tick(t0).expect_err("the answer was lost");
        assert!(!a.step_down(Some(7), t0).expect("no holder is known"));
        assert_eq!(
            a.tick(t0 + S).expect("the landed take is left"),
            TickOutcome::Waiting { term: 7 }
        );
        assert!(!a.is_holder());
        assert_eq!(a.fence().leader_term(t0 + S), None);
        // The take had no holder to wait for and never led: its release
        // says the term told and hands the take's wait on.
        let left = lease.clone().read().expect("read").expect("the record");
        assert!(left.is_released());
        assert_eq!((left.term, left.duration_ms), (7, 2_000));
        // The run takes nothing inside its hold, one lease duration from the
        // step that left, and leads above the term said after it.
        assert_eq!(
            a.tick(t0 + 4 * S - Duration::from_micros(1))
                .expect("inside the hold"),
            TickOutcome::Waiting { term: 7 }
        );
        assert_eq!(
            a.tick(t0 + 4 * S).expect("the take above"),
            TickOutcome::Acquired { term: 8 }
        );
        assert_eq!(a.fence().leader_term(t0 + 4 * S), Some(8));
    }

    #[test]
    fn a_term_at_or_under_a_holders_own_and_a_word_to_a_candidate_move_no_lease() {
        let lease = MemoryLease::new();
        let mut a = candidate(&lease, "a@host:1");
        let t0 = Instant::now() + AHEAD;
        a.lead_at(t0);
        let mut b = candidate(&lease, "b@host:2");
        assert!(matches!(
            b.tick(t0).expect("follow"),
            TickOutcome::Following { term: 1, .. }
        ));
        // The holder's own term is no word to leave.
        assert!(!a.step_down(Some(1), t0).expect("no write"));
        assert_eq!(
            a.tick(t0 + S).expect("renew"),
            TickOutcome::Renewed { term: 1 }
        );
        assert_eq!(a.fence().leader_term(t0 + S), Some(1));
        // A candidate holds no lease: the term is counted and nothing is
        // held back, so its take at the holder's stop is at once and above
        // the term it was told.
        assert!(!b.step_down(Some(7), t0 + S).expect("no write"));
        assert!(!b.step_down(None, t0 + S).expect("no write"));
        assert!(a.release().expect("the holder's stop"));
        assert_eq!(
            b.tick(t0 + S).expect("take"),
            TickOutcome::Acquired { term: 8 }
        );
    }

    #[test]
    fn a_release_that_fails_at_the_word_is_tried_again_at_every_step_and_never_renewed_over() {
        let lease = MemoryLease::new();
        let fails = Arc::new(AtomicU32::new(0));
        let backend = Failing {
            lease: lease.clone(),
            fails: Arc::clone(&fails),
        };
        let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
        let t0 = Instant::now() + AHEAD;
        a.lead_at(t0);
        fails.store(2, Ordering::Relaxed);
        let word = t0 + S / 2;
        a.step_down(Some(7), word).expect_err("the backend is away");
        assert!(a.is_holder(), "kept for the retry");
        assert_eq!(a.fence().leader_term(word), None, "closed at the word");
        // The next step is the release again, not a renewal: the fence stays
        // closed and the record is not renewed under the term it left.
        a.tick(t0 + S).expect_err("the backend is still away");
        assert_eq!(a.fence().leader_term(t0 + S), None);
        let stored = lease.clone().read().expect("read").expect("the record");
        assert_eq!(
            stored.renewed_at_micros,
            a.observed().expect("its record").renewed_at_micros
        );
        assert_eq!(
            a.tick(t0 + 2 * S).expect("the release lands"),
            TickOutcome::Waiting { term: 7 }
        );
        let stored = lease.clone().read().expect("read").expect("the record");
        assert_eq!((stored.holder.as_str(), stored.term), ("", 7));
        // The hold counts from the word, not from the release that landed.
        assert_eq!(
            a.tick(word + 3 * S - Duration::from_micros(1))
                .expect("look"),
            TickOutcome::Waiting { term: 7 }
        );
        assert_eq!(
            a.tick(word + 3 * S).expect("take"),
            TickOutcome::Acquired { term: 8 }
        );
    }

    #[test]
    fn a_release_under_a_told_term_that_landed_but_errored_reads_as_released_at_the_next_step() {
        let lease = MemoryLease::new();
        let fails = Arc::new(AtomicU32::new(0));
        let backend = LandsThenErrors {
            lease: lease.clone(),
            fails: Arc::clone(&fails),
        };
        let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
        let t0 = Instant::now() + AHEAD;
        a.lead_at(t0);
        fails.store(1, Ordering::Relaxed);
        let word = t0 + S / 2;
        a.step_down(Some(7), word).expect_err("the answer was lost");
        assert!(a.is_holder(), "kept for the retry");
        let stored = lease.clone().read().expect("read").expect("the record");
        assert_eq!(
            (stored.holder.as_str(), stored.term),
            ("", 7),
            "the release landed"
        );
        // The next step finds its own release under the term it was told:
        // the lease is given up, and it waits its turn.
        assert_eq!(
            a.tick(t0 + S).expect("look"),
            TickOutcome::Waiting { term: 7 }
        );
        assert!(!a.is_holder());
        assert_eq!(
            a.tick(word + 3 * S).expect("take"),
            TickOutcome::Acquired { term: 8 }
        );
    }

    #[test]
    fn a_leaving_holder_whose_lease_was_taken_before_its_release_landed_has_lost_it() {
        let lease = MemoryLease::new();
        let fails = Arc::new(AtomicU32::new(0));
        let backend = Failing {
            lease: lease.clone(),
            fails: Arc::clone(&fails),
        };
        let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
        let t0 = Instant::now() + AHEAD;
        a.lead_at(t0);
        let mut b = candidate(&lease, "b@host:2");
        assert!(matches!(
            b.tick(t0).expect("follow"),
            TickOutcome::Following { term: 1, .. }
        ));
        fails.store(1, Ordering::Relaxed);
        let word = t0 + S / 2;
        a.step_down(None, word).expect_err("the backend is away");
        // The record stands unrenewed for its duration and the standby
        // takes it; the leaving holder reads that as its loss.
        assert_eq!(
            b.tick(t0 + 3 * S).expect("take"),
            TickOutcome::Acquired { term: 2 }
        );
        assert_eq!(
            a.tick(t0 + 3 * S).expect("the loss"),
            TickOutcome::Lost {
                term: 1,
                holder: "b@host:2".to_string()
            }
        );
        assert!(!a.is_holder());
        // Its hold stands: when the taker stops inside it, the replica that
        // left waits its turn.
        assert!(b.release().expect("the taker's stop"));
        assert_eq!(
            a.tick(word + 3 * S - Duration::from_micros(1))
                .expect("look"),
            TickOutcome::Waiting { term: 2 }
        );
        assert_eq!(
            a.tick(word + 3 * S).expect("take"),
            TickOutcome::Acquired { term: 3 }
        );
    }

    #[test]
    fn a_take_that_would_reach_the_term_ceiling_is_an_error_at_every_step() {
        // The last term under the ceiling is taken; above it none is left.
        let free = LeaseRecord {
            holder: String::new(),
            ..held_by("x@host:9", TERM_CEILING - 2, 0)
        };
        let mut lease = MemoryLease::new();
        assert_eq!(
            lease.write(None, &free).expect("a write"),
            ConditionalWriteOutcome::Done
        );
        let mut b = candidate(&lease, "b@host:2");
        let t0 = Instant::now() + AHEAD;
        assert_eq!(
            b.tick(t0).expect("the last term"),
            TickOutcome::Acquired {
                term: TERM_CEILING - 1
            }
        );
        assert!(b.release().expect("the stop"));
        let mut c = candidate(&lease, "c@host:3");
        for step in 1..=2 {
            let err = c.tick(t0 + step * S).expect_err("no term is left");
            assert_eq!(
                err.to_string(),
                format!(
                    "election term: the next term {TERM_CEILING} is not below the ceiling {TERM_CEILING}"
                )
            );
            assert!(!c.is_holder());
        }
        let stored = lease.clone().read().expect("read").expect("the record");
        assert!(stored.is_released(), "nothing was written");
    }

    #[test]
    fn a_take_over_the_largest_term_a_record_can_say_is_the_term_error() {
        // A record no elector wrote says the largest term: the next term's
        // sum stops there and never wraps to a term under the ceiling.
        let free = LeaseRecord {
            holder: String::new(),
            ..held_by("x@host:9", u64::MAX, 0)
        };
        let mut lease = MemoryLease::new();
        assert_eq!(
            lease.write(None, &free).expect("a write"),
            ConditionalWriteOutcome::Done
        );
        let mut b = candidate(&lease, "b@host:2");
        let refused = b
            .tick(Instant::now() + AHEAD)
            .expect_err("no term is left to take");
        assert!(matches!(refused, ElectError::Term(u64::MAX)), "{refused}");
        assert!(!b.is_holder());
        let stored = lease.clone().read().expect("read").expect("the record");
        assert!(stored.is_released(), "nothing was written");
    }

    #[test]
    fn a_step_handed_an_earlier_instant_leaves_the_latest_one_standing() {
        // The loop steps by the clock, which only grows. A caller that
        // hands its own instants out of order takes nothing back: the
        // wait's reader answers by the latest instant a step was handed.
        let lease = MemoryLease::new();
        let mut a = candidate(&lease, "a@host:1");
        let t0 = Instant::now() + AHEAD;
        assert_eq!(a.tick(t0).expect("take"), TickOutcome::Acquired { term: 1 });
        assert_eq!(
            a.tick(t0 + S).expect("renew"),
            TickOutcome::Renewed { term: 1 }
        );
        assert!(a.waits_to_lead());
        assert_eq!(
            a.tick(t0 + 2 * S).expect("renew"),
            TickOutcome::Renewed { term: 1 }
        );
        assert!(!a.waits_to_lead());
        assert_eq!(
            a.tick(t0 + S).expect("renew"),
            TickOutcome::Renewed { term: 1 }
        );
        assert!(!a.waits_to_lead(), "the opening stays passed");
    }

    #[test]
    fn a_taker_stopped_at_half_its_wait_hands_on_the_half_it_still_owed() {
        // The take's lead would open two seconds after it. Stopped a second
        // into that wait, the taker hands on the second it still owed: the
        // next candidate takes a second after its sight of the release and
        // leads at its take, when the taker would have and one look.
        let lease = MemoryLease::new();
        let mut a = candidate(&lease, "a@host:1");
        let mut b = candidate(&lease, "b@host:2");
        let t0 = Instant::now() + AHEAD;
        assert_eq!(a.tick(t0).expect("take"), TickOutcome::Acquired { term: 1 });
        // A microsecond past the second: the wait still owed has a
        // fraction, and the record's millisecond rounds it up.
        let stopped = t0 + S + Duration::from_micros(1);
        assert_eq!(
            a.tick(stopped).expect("renew"),
            TickOutcome::Renewed { term: 1 }
        );
        assert!(a.waits_to_lead());
        assert!(a.release().expect("the stop"));
        let released = lease.clone().read().expect("read").expect("the record");
        assert!(released.is_released());
        assert_eq!(released.duration_ms, 1_000);
        let saw = t0 + S + Duration::from_millis(1);
        assert_eq!(b.tick(saw).expect("look"), TickOutcome::Gone { term: 1 });
        assert_eq!(
            b.tick(saw + S - Duration::from_micros(1)).expect("look"),
            TickOutcome::Gone { term: 1 }
        );
        assert_eq!(
            b.tick(saw + S).expect("take"),
            TickOutcome::Acquired { term: 2 }
        );
        assert_eq!(b.fence().leader_term(saw + S), Some(2));
        assert!(!b.waits_to_lead());
    }

    #[test]
    fn a_release_after_the_lease_moved_answers_false() {
        let lease = MemoryLease::new();
        let mut a = candidate(&lease, "a@host:1");
        let mut b = candidate(&lease, "b@host:2");
        let t0 = Instant::now() + AHEAD;
        a.tick(t0).expect("acquire");
        assert!(a.waits_to_lead());
        b.tick(t0).expect("follow");
        assert_eq!(
            b.tick(t0 + 3 * S).expect("take"),
            TickOutcome::Acquired { term: 2 }
        );
        assert!(
            !a.release().expect("nothing to give up"),
            "the lease is b's"
        );
        assert!(!a.is_holder());
        assert!(!a.waits_to_lead(), "the wait went with the lease");
        assert_eq!(
            lease
                .clone()
                .read()
                .expect("read")
                .expect("the record")
                .holder,
            "b@host:2"
        );
    }

    /// A backend whose first read stalls while the holder renews inside the
    /// stall: the record it returns is newer than the instant the read began.
    struct SlowRead {
        lease: MemoryLease,
        holder: Arc<Mutex<Elector>>,
        stall: Option<Duration>,
        /// The record goes after the holder's renewal, inside the stall.
        loses: bool,
    }

    impl LeaseBackend for SlowRead {
        fn read(&mut self) -> Result<Option<LeaseRecord>, ElectError> {
            if let Some(stall) = self.stall.take() {
                thread::sleep(stall);
                self.holder.lock().tick(Instant::now())?;
                if self.loses {
                    self.lease.lose();
                }
                thread::sleep(Duration::from_millis(5));
            }
            self.lease.read()
        }

        fn write(
            &mut self,
            expected: Option<&LeaseRecord>,
            record: &LeaseRecord,
        ) -> Result<ConditionalWriteOutcome, ElectError> {
            self.lease.write(expected, record)
        }

        fn place(&self) -> String {
            "slow".to_string()
        }
    }

    #[test]
    fn a_slow_read_dates_the_followers_first_sight_when_the_read_returns() {
        // Real instants: the read's own delay is what this is about. A tenth
        // of the design's settings keeps the run under a second.
        let fast = ElectionSettings {
            duration: Duration::from_millis(120),
            renew_period: Duration::from_millis(40),
            renew_deadline: Duration::from_millis(80),
        };
        let lease = MemoryLease::new();
        let holder = Arc::new(Mutex::new(
            Elector::new(Box::new(lease.clone()), "a@host:1", fast).expect("settings"),
        ));
        let a_fence = {
            let mut a = holder.lock();
            assert_eq!(
                a.tick(Instant::now()).expect("acquire"),
                TickOutcome::Acquired { term: 1 }
            );
            a.fence()
        };
        let backend = SlowRead {
            lease,
            holder: Arc::clone(&holder),
            stall: Some(Duration::from_millis(90)),
            loses: false,
        };
        let mut b = Elector::new(Box::new(backend), "b@host:2", fast).expect("settings");
        // b's first look begins now; 90 ms into the read the holder renews,
        // so its fence runs 80 ms from that renewal; then the holder stalls
        // for good. Dated before the read, b's wait would end inside that
        // window.
        assert!(matches!(
            b.tick(Instant::now()).expect("look"),
            TickOutcome::Following { .. }
        ));
        let started = Instant::now();
        let took = loop {
            assert!(
                started.elapsed() < Duration::from_secs(5),
                "b never took the lease"
            );
            let now = Instant::now();
            if b.tick(now).expect("look") == (TickOutcome::Acquired { term: 2 }) {
                break now;
            }
            thread::sleep(Duration::from_millis(10));
        };
        assert_eq!(
            a_fence.leader_term(took),
            None,
            "the holder's fence had closed before the take began"
        );
    }

    #[test]
    fn a_slow_read_dates_the_takers_opening_when_the_read_returns() {
        // Real instants and a tenth of the design's settings, as above. The
        // holder renews 90 ms into b's first read and its record goes before
        // that read returns: b never saw the lease held, reads no record and
        // takes at once. The holder's fence runs 80 ms from that renewal, so
        // an opening dated before the read would end inside it.
        let fast = ElectionSettings {
            duration: Duration::from_millis(120),
            renew_period: Duration::from_millis(40),
            renew_deadline: Duration::from_millis(80),
        };
        let lease = MemoryLease::new();
        let holder = Arc::new(Mutex::new(
            Elector::new(Box::new(lease.clone()), "a@host:1", fast).expect("settings"),
        ));
        let a_fence = {
            let mut a = holder.lock();
            assert_eq!(
                a.tick(Instant::now()).expect("acquire"),
                TickOutcome::Acquired { term: 1 }
            );
            a.fence()
        };
        let backend = SlowRead {
            lease,
            holder: Arc::clone(&holder),
            stall: Some(Duration::from_millis(90)),
            loses: true,
        };
        let mut b = Elector::new(Box::new(backend), "b@host:2", fast).expect("settings");
        assert_eq!(
            b.tick(Instant::now()).expect("take"),
            TickOutcome::Acquired { term: 1 }
        );
        let b_fence = b.fence();
        let started = Instant::now();
        let opened = loop {
            assert!(
                started.elapsed() < Duration::from_secs(5),
                "b's fence never opened"
            );
            let now = Instant::now();
            b.tick(now).expect("renew");
            if b_fence.leader_term(now).is_some() {
                break now;
            }
            thread::sleep(Duration::from_millis(10));
        };
        assert_eq!(
            a_fence.leader_term(opened),
            None,
            "the holder's fence had closed before the taker's opened"
        );
    }

    #[test]
    fn an_empty_identity_is_refused() {
        let err = Elector::new(Box::new(MemoryLease::new()), "", settings()).expect_err("refused");
        assert_eq!(err.to_string(), "election identity: the identity is empty");
    }

    #[test]
    fn the_records_duration_is_whole_milliseconds_rounded_up() {
        let fine = ElectionSettings {
            duration: Duration::from_micros(1_900),
            renew_period: Duration::from_micros(1_000),
            renew_deadline: Duration::from_micros(1_800),
        };
        let lease = MemoryLease::new();
        let mut a = Elector::new(Box::new(lease.clone()), "a@host:1", fine).expect("settings");
        a.tick(Instant::now() + AHEAD).expect("acquire");
        let record = lease.clone().read().expect("read").expect("the record");
        assert_eq!(record.duration_ms, 2, "1.9 ms rounds up");
        assert!(record.duration() >= fine.duration);
        assert!(
            record.duration() > fine.renew_deadline,
            "a follower waits past the fence"
        );
        assert!(matches!(whole_millis_up(Duration::from_secs(3)), Ok(3_000)));
        assert!(matches!(whole_millis_up(Duration::from_millis(1)), Ok(1)));
    }

    /// A backend whose write lands and then reports an error while `fails`
    /// is above zero: a sync that failed after the rename, an update the
    /// server applied before the call timed out.
    struct LandsThenErrors {
        lease: MemoryLease,
        fails: Arc<AtomicU32>,
    }

    impl LeaseBackend for LandsThenErrors {
        fn read(&mut self) -> Result<Option<LeaseRecord>, ElectError> {
            self.lease.read()
        }

        fn write(
            &mut self,
            expected: Option<&LeaseRecord>,
            record: &LeaseRecord,
        ) -> Result<ConditionalWriteOutcome, ElectError> {
            let written = self.lease.write(expected, record)?;
            if self.fails.load(Ordering::Relaxed) > 0 {
                self.fails.fetch_sub(1, Ordering::Relaxed);
                return Err(ElectError::Io {
                    place: self.place(),
                    source: std::io::Error::other("the answer was lost"),
                });
            }
            Ok(written)
        }

        fn place(&self) -> String {
            "lands-then-errors".to_string()
        }
    }

    #[test]
    fn a_write_that_landed_but_reported_an_error_is_this_runs_own() {
        let lease = MemoryLease::new();
        let fails = Arc::new(AtomicU32::new(1));
        let backend = LandsThenErrors {
            lease: lease.clone(),
            fails: Arc::clone(&fails),
        };
        let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
        let t0 = Instant::now() + AHEAD;
        // The take lands and the answer is lost.
        a.tick(t0).expect_err("the answer was lost");
        assert!(!a.is_holder());
        assert_eq!(
            a.fence().leader_term(t0),
            None,
            "no fence on an unanswered take"
        );
        assert_eq!(
            lease
                .clone()
                .read()
                .expect("read")
                .expect("the record")
                .holder,
            "a@host:1"
        );
        // The next look finds this run's own write and takes it at once. No
        // record stood at the attempt, so the fence opens one renew deadline
        // after that attempt: the renewal below, not this look.
        assert_eq!(
            a.tick(t0 + S).expect("own"),
            TickOutcome::Acquired { term: 1 }
        );
        assert_eq!(a.fence().leader_term(t0 + S), None);
        // A renewal lands and the answer is lost: the record is still this
        // run's own, renewed from there, never a loss.
        fails.store(1, Ordering::Relaxed);
        a.tick(t0 + 2 * S).expect_err("the answer was lost");
        assert!(a.is_holder());
        assert_eq!(
            a.tick(t0 + 3 * S).expect("renew"),
            TickOutcome::Renewed { term: 1 }
        );
        assert_eq!(a.fence().leader_term(t0 + 3 * S), Some(1));
        // A standby under another name saw a live record throughout.
        let mut b = candidate(&lease, "b@host:2");
        assert!(matches!(
            b.tick(t0 + 3 * S).expect("follow"),
            TickOutcome::Following { term: 1, .. }
        ));
    }

    #[test]
    fn a_release_that_landed_but_errored_reads_as_released_on_the_retry() {
        let lease = MemoryLease::new();
        let fails = Arc::new(AtomicU32::new(0));
        let backend = LandsThenErrors {
            lease: lease.clone(),
            fails: Arc::clone(&fails),
        };
        let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
        let t0 = Instant::now() + AHEAD;
        a.tick(t0).expect("acquire");
        fails.store(1, Ordering::Relaxed);
        a.release().expect_err("the answer was lost");
        assert!(a.is_holder(), "kept for the retry");
        assert!(
            lease
                .clone()
                .read()
                .expect("read")
                .expect("the record")
                .is_released(),
            "the release landed"
        );
        assert!(a.release().expect("the retry"), "released after all");
        assert!(!a.is_holder());
        assert_eq!(a.fence().leader_term(t0), None);
    }

    #[test]
    fn a_release_that_landed_but_errored_is_not_renewed_by_the_next_tick() {
        let lease = MemoryLease::new();
        let fails = Arc::new(AtomicU32::new(0));
        let backend = LandsThenErrors {
            lease: lease.clone(),
            fails: Arc::clone(&fails),
        };
        let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
        let t0 = Instant::now() + AHEAD;
        a.lead_at(t0);
        fails.store(1, Ordering::Relaxed);
        a.release().expect_err("the answer was lost");
        // The caller ticks instead of retrying: the released record is this
        // run's own release, not a lease to renew. The fence stays closed
        // under the old term while the candidates take the free lease.
        assert_eq!(
            a.tick(t0 + S).expect("look"),
            TickOutcome::Lost {
                term: 1,
                holder: String::new()
            }
        );
        assert!(!a.is_holder());
        assert_eq!(a.fence().leader_term(t0 + S), None);
        assert!(
            lease
                .clone()
                .read()
                .expect("read")
                .expect("the record")
                .is_released(),
            "still released"
        );
        let mut b = candidate(&lease, "b@host:2");
        assert_eq!(
            b.tick(t0 + S + S / 2).expect("take"),
            TickOutcome::Acquired { term: 2 }
        );
        assert!(matches!(
            a.tick(t0 + 2 * S).expect("look"),
            TickOutcome::Following { term: 2, .. }
        ));
    }

    #[test]
    fn a_release_after_a_renewal_that_landed_but_errored_gives_up_the_landed_record() {
        let lease = MemoryLease::new();
        let fails = Arc::new(AtomicU32::new(0));
        let backend = LandsThenErrors {
            lease: lease.clone(),
            fails: Arc::clone(&fails),
        };
        let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
        let t0 = Instant::now() + AHEAD;
        a.lead_at(t0);
        fails.store(1, Ordering::Relaxed);
        a.tick(t0 + S).expect_err("the answer was lost");
        // A planned stop: the release goes over the renewal that landed, not
        // over the record before it, and the lease is free at once.
        assert!(a.release().expect("release"), "released");
        assert!(!a.is_holder());
        assert_eq!(a.fence().leader_term(t0 + S), None);
        let stored = lease.clone().read().expect("read").expect("the record");
        assert!(stored.is_released());
        assert_eq!(stored.term, 1);
        let mut b = candidate(&lease, "b@host:2");
        assert_eq!(
            b.tick(t0 + S + S / 2).expect("take"),
            TickOutcome::Acquired { term: 2 }
        );
    }

    #[test]
    fn a_release_inside_the_wait_over_a_renewal_that_landed_hands_the_wait_on() {
        // This run's renewal landed with its answer lost, inside the take's
        // wait. The release finds that renewal, releases over it, and still
        // says what the take's wait owed as the record's duration.
        let lease = MemoryLease::new();
        let fails = Arc::new(AtomicU32::new(0));
        let backend = LandsThenErrors {
            lease: lease.clone(),
            fails: Arc::clone(&fails),
        };
        let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
        let t0 = Instant::now() + AHEAD;
        assert_eq!(a.tick(t0).expect("take"), TickOutcome::Acquired { term: 1 });
        fails.store(1, Ordering::Relaxed);
        a.tick(t0 + S).expect_err("the answer was lost");
        assert!(a.waits_to_lead());
        assert!(a.release().expect("release"));
        assert!(!a.waits_to_lead(), "a release ends the wait");
        let stored = lease.clone().read().expect("read").expect("the record");
        assert!(stored.is_released());
        assert_eq!(stored.duration_ms, 1_000);
    }

    /// A backend whose reads fail while `read_fails` is above zero, whose
    /// writes land and then report an error while `lost_answers` is above
    /// zero, and whose writes fail before landing while `write_fails` is
    /// above zero: one API server through its blips.
    struct Blips {
        lease: MemoryLease,
        read_fails: Arc<AtomicU32>,
        lost_answers: Arc<AtomicU32>,
        write_fails: Arc<AtomicU32>,
    }

    impl LeaseBackend for Blips {
        fn read(&mut self) -> Result<Option<LeaseRecord>, ElectError> {
            if self.read_fails.load(Ordering::Relaxed) > 0 {
                self.read_fails.fetch_sub(1, Ordering::Relaxed);
                return Err(ElectError::Io {
                    place: self.place(),
                    source: std::io::Error::other("the read was lost"),
                });
            }
            self.lease.read()
        }

        fn write(
            &mut self,
            expected: Option<&LeaseRecord>,
            record: &LeaseRecord,
        ) -> Result<ConditionalWriteOutcome, ElectError> {
            if self.write_fails.load(Ordering::Relaxed) > 0 {
                self.write_fails.fetch_sub(1, Ordering::Relaxed);
                return Err(ElectError::Io {
                    place: self.place(),
                    source: std::io::Error::other("the write was refused"),
                });
            }
            let written = self.lease.write(expected, record)?;
            if self.lost_answers.load(Ordering::Relaxed) > 0 {
                self.lost_answers.fetch_sub(1, Ordering::Relaxed);
                return Err(ElectError::Io {
                    place: self.place(),
                    source: std::io::Error::other("the answer was lost"),
                });
            }
            Ok(written)
        }

        fn place(&self) -> String {
            "blips".to_string()
        }
    }

    #[test]
    fn a_release_whose_re_read_failed_keeps_the_record_for_the_retry() {
        let lease = MemoryLease::new();
        let read_fails = Arc::new(AtomicU32::new(0));
        let lost_answers = Arc::new(AtomicU32::new(0));
        let backend = Blips {
            lease: lease.clone(),
            read_fails: Arc::clone(&read_fails),
            lost_answers: Arc::clone(&lost_answers),
            write_fails: Arc::new(AtomicU32::new(0)),
        };
        let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
        let t0 = Instant::now() + AHEAD;
        a.tick(t0).expect("acquire");
        // A renewal lands and its answer is lost.
        lost_answers.store(1, Ordering::Relaxed);
        a.tick(t0 + S).expect_err("the answer was lost");
        // A planned stop: the release's write is refused over the record
        // before that renewal, and the re-read that settles what moved fails
        // too. The record is kept, so a second call retries, finds its own
        // landed renewal and releases it.
        read_fails.store(1, Ordering::Relaxed);
        a.release().expect_err("the read was lost");
        assert!(a.is_holder(), "kept for the retry");
        assert_eq!(a.fence().leader_term(t0 + S), None);
        assert!(a.release().expect("the retry"), "released after all");
        assert!(!a.is_holder());
        let stored = lease.clone().read().expect("read").expect("the record");
        assert!(stored.is_released());
        assert_eq!(stored.term, 1);
    }

    #[test]
    fn a_release_write_that_fails_after_a_landed_renewal_keeps_that_renewal_for_the_retry() {
        let lease = MemoryLease::new();
        let lost_answers = Arc::new(AtomicU32::new(0));
        let write_fails = Arc::new(AtomicU32::new(0));
        let backend = Blips {
            lease: lease.clone(),
            read_fails: Arc::new(AtomicU32::new(0)),
            lost_answers: Arc::clone(&lost_answers),
            write_fails: Arc::clone(&write_fails),
        };
        let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
        let t0 = Instant::now() + AHEAD;
        a.lead_at(t0);
        // A renewal lands and its answer is lost; then the release's own
        // write fails before it lands. The landed renewal stays the record
        // the retry adopts: the lease is free at once, not at its expiry.
        lost_answers.store(1, Ordering::Relaxed);
        a.tick(t0 + S).expect_err("the answer was lost");
        write_fails.store(1, Ordering::Relaxed);
        a.release().expect_err("the write was refused");
        assert!(a.is_holder(), "kept for the retry");
        assert!(a.release().expect("the retry"), "released after all");
        assert!(!a.is_holder());
        let stored = lease.clone().read().expect("read").expect("the record");
        assert!(stored.is_released());
        assert_eq!(stored.term, 1);
        let mut b = candidate(&lease, "b@host:2");
        assert_eq!(
            b.tick(t0 + 2 * S).expect("take"),
            TickOutcome::Acquired { term: 2 }
        );
    }

    #[test]
    fn a_record_that_vanished_is_waited_out_and_taken_at_the_term_after_the_highest_seen() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let path = dir.path().join("lease");
        let mut a = Elector::new(Box::new(FileLease::new(&path)), "a@host:1", settings())
            .expect("settings");
        let mut b = Elector::new(Box::new(FileLease::new(&path)), "b@host:2", settings())
            .expect("settings");
        let t0 = Instant::now() + AHEAD;
        a.lead_at(t0);
        assert!(a.release().expect("release"));
        assert_eq!(
            b.tick(t0 + S).expect("take"),
            TickOutcome::Acquired { term: 2 }
        );
        assert!(matches!(
            a.tick(t0 + S + S / 2).expect("follow"),
            TickOutcome::Following { term: 2, .. }
        ));
        // The record is deleted under the holder (an operator forcing a
        // re-election). No release made it so, and the holder's fence is
        // open until its own next step reads the lease: a candidate that saw
        // the record held waits that record's duration from its own first
        // sight of the loss, the old holder as well, and follows nobody. The
        // take then counts from the highest term either saw, never from one.
        std::fs::remove_file(&path).expect("the deletion");
        let a_saw = t0 + S + 3 * S / 4;
        assert_eq!(a.tick(a_saw).expect("look"), TickOutcome::Gone { term: 2 });
        assert_eq!(a.observed(), None, "nobody is named");
        assert_eq!(
            b.fence().leader_term(a_saw),
            Some(2),
            "the holder still leads when the candidate first sees the loss"
        );
        assert_eq!(
            b.tick(t0 + 2 * S).expect("look"),
            TickOutcome::Lost {
                term: 2,
                holder: String::new()
            }
        );
        assert_eq!(
            b.tick(t0 + 3 * S).expect("look"),
            TickOutcome::Gone { term: 2 }
        );
        assert_eq!(
            a.tick(a_saw + 3 * S - Duration::from_micros(1))
                .expect("look"),
            TickOutcome::Gone { term: 2 },
            "the record's duration counts from a's first sight of the loss"
        );
        assert_eq!(
            a.tick(a_saw + 3 * S).expect("take"),
            TickOutcome::Acquired { term: 3 }
        );
        assert!(matches!(
            b.tick(t0 + 5 * S).expect("follow"),
            TickOutcome::Following { term: 3, .. }
        ));
    }

    #[test]
    fn an_emptied_lease_file_is_waited_out_as_a_deleted_one() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let path = dir.path().join("lease");
        let mut a = Elector::new(Box::new(FileLease::new(&path)), "a@host:1", settings())
            .expect("settings");
        let mut b = Elector::new(Box::new(FileLease::new(&path)), "b@host:2", settings())
            .expect("settings");
        let t0 = Instant::now() + AHEAD;
        a.lead_at(t0);
        assert!(matches!(
            b.tick(t0 + S / 2).expect("follow"),
            TickOutcome::Following { term: 1, .. }
        ));
        // The file is emptied in place: it reads as no lease, as a deleted
        // one does, and no release wrote it.
        std::fs::write(&path, b"").expect("the emptying");
        let b_saw = t0 + 3 * S / 4;
        assert_eq!(b.tick(b_saw).expect("look"), TickOutcome::Gone { term: 1 });
        assert_eq!(
            a.fence().leader_term(b_saw),
            Some(1),
            "the holder still leads when the candidate first sees the loss"
        );
        assert_eq!(
            b.tick(b_saw + 3 * S - Duration::from_micros(1))
                .expect("look"),
            TickOutcome::Gone { term: 1 }
        );
        assert_eq!(
            a.fence().leader_term(b_saw + 3 * S),
            None,
            "the holder's fence closed inside the wait"
        );
        assert_eq!(
            b.tick(b_saw + 3 * S).expect("take"),
            TickOutcome::Acquired { term: 2 }
        );
    }

    /// The record of `holder` with its holder cleared in place, as a hand's
    /// patch leaves it: the term, the stamps and the duration kept.
    fn cleared(lease: &MemoryLease, holder: &Elector) {
        let held = holder.observed().expect("the holder's record").clone();
        let cleared = LeaseRecord {
            holder: String::new(),
            ..held.clone()
        };
        let mut hand = lease.clone();
        assert_eq!(
            hand.write(Some(&held), &cleared).expect("a write"),
            ConditionalWriteOutcome::Done
        );
    }

    #[test]
    fn a_record_cleared_in_place_under_a_live_holder_is_waited_out() {
        // A holder's release says no duration. A record whose holder another
        // hand cleared keeps its term and its duration, and that holder
        // leads until its next step reads the lease: the record is waited
        // out for the duration it says, from the first sight of it.
        let lease = MemoryLease::new();
        let mut a = candidate(&lease, "a@host:1");
        let t0 = Instant::now() + AHEAD;
        a.lead_at(t0);
        let mut b = candidate(&lease, "b@host:2");
        assert!(matches!(
            b.tick(t0).expect("follow"),
            TickOutcome::Following { term: 1, .. }
        ));
        cleared(&lease, &a);
        let saw = t0 + S / 2;
        assert_eq!(b.tick(saw).expect("look"), TickOutcome::Gone { term: 1 });
        assert_eq!(a.fence().leader_term(saw), Some(1));
        assert_eq!(b.fence().leader_term(saw), None);
        // The holder reads its loss at its next step and waits as every
        // candidate does.
        assert_eq!(
            a.tick(t0 + S).expect("the renewal"),
            TickOutcome::Lost {
                term: 1,
                holder: String::new()
            }
        );
        assert_eq!(a.fence().leader_term(t0 + S), None);
        assert_eq!(
            a.tick(t0 + 2 * S).expect("look"),
            TickOutcome::Gone { term: 1 }
        );
        assert_eq!(
            b.tick(saw + 3 * S - Duration::from_micros(1))
                .expect("look"),
            TickOutcome::Gone { term: 1 }
        );
        assert_eq!(
            b.tick(saw + 3 * S).expect("take"),
            TickOutcome::Acquired { term: 2 }
        );
    }

    #[test]
    fn a_candidate_that_never_saw_the_lease_held_waits_out_a_record_cleared_in_place() {
        let lease = MemoryLease::new();
        let mut a = candidate(&lease, "a@host:1");
        let t0 = Instant::now() + AHEAD;
        a.lead_at(t0);
        cleared(&lease, &a);
        // A replica that starts now never saw the lease held, and the holder
        // stalls with its fence open. The record still says its duration:
        // it is waited out from this first sight, and the holder's fence has
        // closed by itself before the take.
        let mut c = candidate(&lease, "c@host:3");
        let saw = t0 + S / 2;
        assert_eq!(c.tick(saw).expect("look"), TickOutcome::Gone { term: 1 });
        assert_eq!(a.fence().leader_term(saw), Some(1));
        assert_eq!(c.fence().leader_term(saw), None);
        assert_eq!(
            c.tick(saw + 3 * S - Duration::from_micros(1))
                .expect("look"),
            TickOutcome::Gone { term: 1 }
        );
        assert_eq!(
            c.tick(saw + 3 * S).expect("take"),
            TickOutcome::Acquired { term: 2 }
        );
        assert_eq!(a.fence().leader_term(saw + 3 * S), None);
        assert_eq!(c.fence().leader_term(saw + 3 * S), Some(2));
    }

    #[test]
    fn a_record_cleared_in_place_under_a_passed_term_is_waited_out_for_its_own_duration() {
        // The wait is the cleared record's own whatever its term. One under
        // a term this run saw pass is a late writer's, and that writer may
        // lead for the duration its record says: here two seconds past the
        // wait for the holder seen last.
        let held = held_by("x@host:9", 4, 3_000);
        let mut lease = MemoryLease::new();
        assert_eq!(
            lease.write(None, &held).expect("a write"),
            ConditionalWriteOutcome::Done
        );
        let mut b = candidate(&lease, "b@host:2");
        let t0 = Instant::now() + AHEAD;
        assert!(matches!(
            b.tick(t0).expect("follow"),
            TickOutcome::Following { term: 4, .. }
        ));
        let late = LeaseRecord {
            holder: String::new(),
            ..held_by("y@host:8", 2, 5_000)
        };
        assert_eq!(
            lease.write(Some(&held), &late).expect("a write"),
            ConditionalWriteOutcome::Done
        );
        let saw = t0 + S;
        for wait in [Duration::ZERO, 3 * S, 5 * S - Duration::from_micros(1)] {
            let look = b.tick(saw + wait).expect("look");
            assert!(
                matches!(look, TickOutcome::Gone { .. }),
                "{wait:?}: {look:?}"
            );
        }
        assert_eq!(
            b.tick(saw + 5 * S).expect("take"),
            TickOutcome::Acquired { term: 5 }
        );
        // No holder's release leaves a record under a passed term, so this
        // take waits as the take of an absent record does: its fence opens
        // one renew deadline later.
        let took = saw + 5 * S;
        assert_eq!(b.fence().leader_term(took), None);
        assert_eq!(
            b.tick(took + 2 * S).expect("renew"),
            TickOutcome::Renewed { term: 5 }
        );
        assert_eq!(b.fence().leader_term(took + 2 * S), Some(5));
    }

    #[test]
    fn a_lease_replaced_by_an_empty_one_is_waited_out() {
        // A `Lease` replaced by a manifest with no record reads as released
        // under term zero. No release lowers its holder's term, so one under a
        // term below the record last seen held is no holder's release.
        let held = held_by("x@host:9", 4, 3_000);
        let mut lease = MemoryLease::new();
        assert_eq!(
            lease.write(None, &held).expect("a write"),
            ConditionalWriteOutcome::Done
        );
        let mut b = candidate(&lease, "b@host:2");
        let t0 = Instant::now() + AHEAD;
        assert!(matches!(
            b.tick(t0).expect("follow"),
            TickOutcome::Following { term: 4, .. }
        ));
        let empty = LeaseRecord::default();
        assert_eq!(
            lease.write(Some(&held), &empty).expect("a write"),
            ConditionalWriteOutcome::Done
        );
        let saw = t0 + S;
        assert_eq!(b.tick(saw).expect("look"), TickOutcome::Gone { term: 4 });
        assert_eq!(
            b.tick(saw + 3 * S - Duration::from_micros(1))
                .expect("look"),
            TickOutcome::Gone { term: 4 }
        );
        assert_eq!(
            b.tick(saw + 3 * S).expect("take"),
            TickOutcome::Acquired { term: 5 }
        );
        // An empty record has no holder to wait for either: the take opens
        // no fence, as over an absent one.
        assert_eq!(b.fence().leader_term(saw + 3 * S), None);
        // The taker's own record, replaced again before its next look, is
        // one it saw held: the wait counts anew from that look.
        let own = b.observed().expect("the take's record").clone();
        assert_eq!(
            lease.write(Some(&own), &empty).expect("a write"),
            ConditionalWriteOutcome::Done
        );
        let again = saw + 4 * S;
        assert_eq!(
            b.tick(again).expect("look"),
            TickOutcome::Lost {
                term: 5,
                holder: String::new()
            }
        );
        assert_eq!(
            b.tick(again + 3 * S - Duration::from_micros(1))
                .expect("look"),
            TickOutcome::Gone { term: 5 }
        );
        assert_eq!(
            b.tick(again + 3 * S).expect("take"),
            TickOutcome::Acquired { term: 6 }
        );
    }

    #[test]
    fn a_candidate_that_never_saw_the_lease_held_takes_an_empty_one_at_once() {
        // A first start over the empty `Lease` a chart renders, or over a
        // new lease file: there is no record to wait out.
        let mut lease = MemoryLease::new();
        assert_eq!(
            lease.write(None, &LeaseRecord::default()).expect("a write"),
            ConditionalWriteOutcome::Done
        );
        let mut a = candidate(&lease, "a@host:1");
        let t0 = Instant::now() + AHEAD;
        assert_eq!(a.tick(t0).expect("take"), TickOutcome::Acquired { term: 1 });
    }

    #[test]
    fn a_release_at_the_held_term_is_taken_at_once_by_a_run_that_saw_a_higher_term() {
        // The rule is keyed on the record last seen held, not on the highest
        // term: a store put back to a lower term, then released by the holder
        // it names, shows that holder's own release.
        let mut lease = MemoryLease::new();
        let (first, fallen) = (held_by("x@host:8", 5, 3_000), held_by("y@host:9", 3, 3_000));
        assert_eq!(
            lease.write(None, &first).expect("a write"),
            ConditionalWriteOutcome::Done
        );
        let mut b = candidate(&lease, "b@host:2");
        let t0 = Instant::now() + AHEAD;
        assert!(matches!(
            b.tick(t0).expect("follow"),
            TickOutcome::Following { term: 5, .. }
        ));
        assert_eq!(
            lease.write(Some(&first), &fallen).expect("a write"),
            ConditionalWriteOutcome::Done
        );
        assert!(matches!(
            b.tick(t0 + S).expect("follow"),
            TickOutcome::Following { term: 3, .. }
        ));
        let released = LeaseRecord {
            holder: String::new(),
            duration_ms: 0,
            renewed_at_micros: 2,
            ..fallen.clone()
        };
        assert_eq!(
            lease.write(Some(&fallen), &released).expect("a write"),
            ConditionalWriteOutcome::Done
        );
        assert_eq!(
            b.tick(t0 + S + Duration::from_millis(1)).expect("take"),
            TickOutcome::Acquired { term: 6 },
            "at once, above the highest term this run saw"
        );
    }

    #[test]
    fn a_held_record_under_a_lower_term_is_no_loss_and_its_own_loss_is_waited_from_that_sight() {
        // A store put back to a held record under a lower term shows a
        // holder, not a loss: the wait for the record lost after it is dated
        // at the first sight of that loss, never at the sight of the holder.
        let mut lease = MemoryLease::new();
        let (first, fallen) = (held_by("x@host:8", 5, 3_000), held_by("y@host:9", 3, 3_000));
        assert_eq!(
            lease.write(None, &first).expect("a write"),
            ConditionalWriteOutcome::Done
        );
        let mut b = candidate(&lease, "b@host:2");
        let t0 = Instant::now() + AHEAD;
        assert!(matches!(
            b.tick(t0).expect("follow"),
            TickOutcome::Following { term: 5, .. }
        ));
        assert_eq!(
            lease.write(Some(&first), &fallen).expect("a write"),
            ConditionalWriteOutcome::Done
        );
        assert!(matches!(
            b.tick(t0 + S).expect("follow"),
            TickOutcome::Following { term: 3, .. }
        ));
        lease.lose();
        let saw = t0 + 2 * S;
        assert_eq!(b.tick(saw).expect("look"), TickOutcome::Gone { term: 3 });
        assert_eq!(
            b.tick(saw + 3 * S - Duration::from_micros(1))
                .expect("look"),
            TickOutcome::Gone { term: 3 },
            "the duration counts from the first sight of the loss"
        );
        assert_eq!(
            b.tick(saw + 3 * S).expect("take"),
            TickOutcome::Acquired { term: 6 },
            "above the highest term this run saw"
        );
    }

    #[test]
    fn a_lost_record_is_waited_out_for_its_own_duration_not_the_waiters() {
        // Two replicas may run two durations. The wait covers the lost
        // holder's fence, so it is that record's duration, longer or shorter
        // than the three seconds this replica runs.
        let mut lease = MemoryLease::new();
        let long = held_by("x@host:8", 1, 6_000);
        assert_eq!(
            lease.write(None, &long).expect("a write"),
            ConditionalWriteOutcome::Done
        );
        let mut b = candidate(&lease, "b@host:2");
        let t0 = Instant::now() + AHEAD;
        assert!(matches!(
            b.tick(t0).expect("follow"),
            TickOutcome::Following { term: 1, .. }
        ));
        lease.lose();
        let saw = t0 + S;
        assert_eq!(b.tick(saw).expect("look"), TickOutcome::Gone { term: 1 });
        assert_eq!(
            b.tick(saw + 6 * S - Duration::from_micros(1))
                .expect("look"),
            TickOutcome::Gone { term: 1 },
            "six seconds, the lost record's, where this replica runs three"
        );
        assert_eq!(
            b.tick(saw + 6 * S).expect("take"),
            TickOutcome::Acquired { term: 2 }
        );
        // A holder with a one second lease writes over this replica's
        // record, and that record is lost in turn.
        let own = b.observed().expect("the take's record").clone();
        let short = held_by("z@host:9", 3, 1_000);
        assert_eq!(
            lease.write(Some(&own), &short).expect("a write"),
            ConditionalWriteOutcome::Done
        );
        let lost = saw + 7 * S;
        assert!(matches!(
            b.tick(lost).expect("look"),
            TickOutcome::Lost { term: 2, .. }
        ));
        lease.lose();
        let again = lost + S / 2;
        assert_eq!(b.tick(again).expect("look"), TickOutcome::Gone { term: 3 });
        assert_eq!(
            b.tick(again + S - Duration::from_micros(1)).expect("look"),
            TickOutcome::Gone { term: 3 },
            "one second, the lost record's"
        );
        assert_eq!(
            b.tick(again + S).expect("take"),
            TickOutcome::Acquired { term: 4 }
        );
    }

    #[test]
    fn a_record_taken_while_it_was_gone_ends_the_wait_and_is_followed() {
        // A replica that starts while the record is gone never saw it held
        // and takes at once; a candidate waiting the old record out follows
        // the new holder as it follows any.
        let lease = MemoryLease::new();
        let mut a = candidate(&lease, "a@host:1");
        let mut b = candidate(&lease, "b@host:2");
        let t0 = Instant::now() + AHEAD;
        assert_eq!(
            a.tick(t0).expect("acquire"),
            TickOutcome::Acquired { term: 1 }
        );
        assert!(matches!(
            b.tick(t0 + S / 2).expect("follow"),
            TickOutcome::Following { term: 1, .. }
        ));
        lease.lose();
        assert_eq!(b.tick(t0 + S).expect("look"), TickOutcome::Gone { term: 1 });
        let mut c = candidate(&lease, "c@host:3");
        assert_eq!(
            c.tick(t0 + S + S / 2).expect("take"),
            TickOutcome::Acquired { term: 1 }
        );
        assert_eq!(
            b.tick(t0 + 2 * S).expect("follow"),
            TickOutcome::Following {
                holder: "c@host:3".to_string(),
                term: 1
            }
        );
        // The new record is lost in turn: the wait for it counts from b's
        // first sight of this loss, never from the one before it.
        lease.lose();
        let again = t0 + 2 * S + S / 2;
        assert_eq!(b.tick(again).expect("look"), TickOutcome::Gone { term: 1 });
        assert_eq!(
            b.tick(again + 3 * S - Duration::from_micros(1))
                .expect("look"),
            TickOutcome::Gone { term: 1 }
        );
        assert_eq!(
            b.tick(again + 3 * S).expect("take"),
            TickOutcome::Acquired { term: 2 }
        );
    }

    #[test]
    fn a_record_under_this_identity_that_came_back_lower_is_taken_above_the_highest_term() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let path = dir.path().join("lease");
        let mut a = Elector::new(Box::new(FileLease::new(&path)), "a@host:1", settings())
            .expect("settings");
        let mut b = Elector::new(Box::new(FileLease::new(&path)), "b@host:2", settings())
            .expect("settings");
        let t0 = Instant::now() + AHEAD;
        a.lead_at(t0);
        assert!(a.release().expect("release"));
        assert_eq!(
            b.tick(t0 + S).expect("take"),
            TickOutcome::Acquired { term: 2 }
        );
        assert!(matches!(
            a.tick(t0 + S + S / 2).expect("follow"),
            TickOutcome::Following { term: 2, .. }
        ));
        // The record is deleted, and a second process under a's identity (a
        // restart racing the first) takes the empty lease at term one and
        // stalls. The first process saw term two: once that record has been
        // still for its duration it takes above the highest term it saw, a
        // take in every field, never one under the lower term. Its next
        // take after a deletion is above that again, once it waited its own
        // lost record out as every candidate that saw it does.
        std::fs::remove_file(&path).expect("the deletion");
        let mut again = Elector::new(Box::new(FileLease::new(&path)), "a@host:1", settings())
            .expect("settings");
        assert_eq!(
            again.tick(t0 + 2 * S).expect("take"),
            TickOutcome::Acquired { term: 1 }
        );
        let first_sight = t0 + 2 * S + S / 2;
        assert!(matches!(
            a.tick(first_sight).expect("look"),
            TickOutcome::Following { term: 1, .. }
        ));
        let twins = a.observed().expect("the twin's record").clone();
        assert_eq!(
            a.tick(first_sight + 3 * S).expect("take"),
            TickOutcome::Acquired { term: 3 }
        );
        let taken = a.observed().expect("the take's record").clone();
        assert_eq!(taken.acquired_at_micros, taken.renewed_at_micros);
        assert!(taken.acquired_at_micros > twins.acquired_at_micros);
        std::fs::remove_file(&path).expect("the second deletion");
        assert_eq!(
            a.tick(first_sight + 4 * S).expect("look"),
            TickOutcome::Lost {
                term: 3,
                holder: String::new()
            }
        );
        assert_eq!(
            a.tick(first_sight + 7 * S - Duration::from_micros(1))
                .expect("look"),
            TickOutcome::Gone { term: 3 }
        );
        assert_eq!(
            a.tick(first_sight + 7 * S).expect("take"),
            TickOutcome::Acquired { term: 4 }
        );
    }

    /// A backend whose write does not land while `delay` is set: it reports
    /// an error and lands at the backend's next write, before that write is
    /// judged, as a server applies a timed-out update late.
    struct LandsLate {
        lease: MemoryLease,
        delay: Arc<AtomicBool>,
        pending: Option<(Option<LeaseRecord>, LeaseRecord)>,
    }

    impl LeaseBackend for LandsLate {
        fn read(&mut self) -> Result<Option<LeaseRecord>, ElectError> {
            self.lease.read()
        }

        fn write(
            &mut self,
            expected: Option<&LeaseRecord>,
            record: &LeaseRecord,
        ) -> Result<ConditionalWriteOutcome, ElectError> {
            if let Some((expected, record)) = self.pending.take() {
                self.lease.write(expected.as_ref(), &record)?;
            }
            if self.delay.swap(false, Ordering::Relaxed) {
                self.pending = Some((expected.cloned(), record.clone()));
                return Err(ElectError::Io {
                    place: self.place(),
                    source: std::io::Error::other("the answer was lost"),
                });
            }
            self.lease.write(expected, record)
        }

        fn place(&self) -> String {
            "lands-late".to_string()
        }
    }

    #[test]
    fn an_initial_take_that_landed_late_is_this_runs_own_on_the_next_tick() {
        let lease = MemoryLease::new();
        let delay = Arc::new(AtomicBool::new(true));
        let backend = LandsLate {
            lease: lease.clone(),
            delay: Arc::clone(&delay),
            pending: None,
        };
        let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
        let t0 = Instant::now() + AHEAD;
        // The first take reports an error and lands after the next read:
        // that read finds nothing, the retry's write is refused as moved,
        // and the tick after reads this run's own record and takes it.
        a.tick(t0).expect_err("the answer was lost");
        assert_eq!(a.tick(t0 + S).expect("retry"), TickOutcome::Raced);
        assert_eq!(
            a.tick(t0 + 2 * S).expect("own"),
            TickOutcome::Acquired { term: 1 }
        );
        assert!(a.is_holder());
        // No record stood at either attempt, and the record may have landed
        // at any time before this look: the fence opens one renew deadline
        // after the write that took it here, never after an attempt.
        assert_eq!(a.fence().leader_term(t0 + 2 * S), None);
        assert_eq!(
            a.tick(t0 + 3 * S).expect("renew"),
            TickOutcome::Renewed { term: 1 }
        );
        assert_eq!(a.fence().leader_term(t0 + 3 * S), None);
        assert_eq!(a.fence().leader_term(t0 + 4 * S), Some(1));
        assert_eq!(
            lease
                .clone()
                .read()
                .expect("read")
                .expect("the record")
                .holder,
            "a@host:1"
        );
    }

    #[test]
    fn a_renewal_that_landed_late_is_renewed_over_not_lost() {
        let lease = MemoryLease::new();
        let delay = Arc::new(AtomicBool::new(false));
        let backend = LandsLate {
            lease: lease.clone(),
            delay: Arc::clone(&delay),
            pending: None,
        };
        let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
        let t0 = Instant::now() + AHEAD;
        assert_eq!(
            a.tick(t0).expect("acquire"),
            TickOutcome::Acquired { term: 1 }
        );
        delay.store(true, Ordering::Relaxed);
        a.tick(t0 + S).expect_err("the answer was lost");
        assert!(a.is_holder());
        // The next tick reads the record before the lost renewal lands and
        // writes after it did: the record under its write is this run's
        // own, and the renewal goes over it within the tick.
        assert_eq!(
            a.tick(t0 + 2 * S).expect("renew"),
            TickOutcome::Renewed { term: 1 }
        );
        assert!(a.is_holder());
        assert_eq!(a.fence().leader_term(t0 + 2 * S), Some(1));
        assert_eq!(
            lease
                .clone()
                .read()
                .expect("read")
                .expect("the record")
                .holder,
            "a@host:1"
        );
        assert_eq!(
            a.tick(t0 + 3 * S).expect("renew"),
            TickOutcome::Renewed { term: 1 }
        );
        let mut b = candidate(&lease, "b@host:2");
        assert!(matches!(
            b.tick(t0 + 3 * S).expect("follow"),
            TickOutcome::Following { term: 1, .. }
        ));
    }

    /// A holder whose renewal reported an error and lands at the store's
    /// next write, before that write is judged; the lease comes with it.
    fn led_with_a_late_renewal(t0: Instant) -> (Elector, MemoryLease) {
        let lease = MemoryLease::new();
        let delay = Arc::new(AtomicBool::new(false));
        let backend = LandsLate {
            lease: lease.clone(),
            delay: Arc::clone(&delay),
            pending: None,
        };
        let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
        assert_eq!(
            a.tick(t0).expect("acquire"),
            TickOutcome::Acquired { term: 1 }
        );
        delay.store(true, Ordering::Relaxed);
        a.tick(t0 + S).expect_err("the answer was lost");
        (a, lease)
    }

    #[test]
    fn a_renewal_adopted_at_the_read_after_a_moved_write_is_no_stranger() {
        // The late renewal lands under the next renewal's write, which
        // comes back moved. The read after it observes that record with
        // the attempt still kept: the record is this run's own.
        let t0 = Instant::now() + AHEAD;
        let (mut a, lease) = led_with_a_late_renewal(t0);
        assert_eq!(
            a.tick(t0 + 2 * S).expect("renew"),
            TickOutcome::Renewed { term: 1 }
        );
        // Another run under this name writes the record once: the first
        // stranger, followed for the record's duration. A second would be
        // refused.
        let mine = lease.clone().read().expect("read").expect("the record");
        let theirs = LeaseRecord {
            renewed_at_micros: mine.renewed_at_micros + 1,
            ..mine.clone()
        };
        assert_eq!(
            lease.clone().write(Some(&mine), &theirs).expect("write"),
            ConditionalWriteOutcome::Done
        );
        assert_eq!(
            a.tick(t0 + 3 * S).expect("the look that loses the lease"),
            TickOutcome::Lost {
                term: 1,
                holder: "a@host:1".to_owned()
            }
        );
        assert_eq!(
            a.tick(t0 + 4 * S).expect("the first stranger"),
            TickOutcome::Following {
                holder: "a@host:1".to_owned(),
                term: 1
            }
        );
    }

    #[test]
    fn a_renewal_adopted_at_a_releases_read_after_a_moved_write_is_no_stranger() {
        // The same adoption at a word to leave: the release's write comes
        // back moved over the late renewal, the read after it observes
        // that record with the attempt still kept, and the release goes
        // over it.
        let t0 = Instant::now() + AHEAD;
        let (mut a, lease) = led_with_a_late_renewal(t0);
        assert!(a.step_down(None, t0 + S).expect("the release"));
        let released = lease.clone().read().expect("read").expect("the record");
        assert!(released.is_released());
        // Another run under this name takes the lease: the first stranger.
        let theirs = LeaseRecord {
            holder: "a@host:1".to_owned(),
            term: 2,
            duration_ms: 3_000,
            renewed_at_micros: released.renewed_at_micros + 1,
            ..released.clone()
        };
        assert_eq!(
            lease
                .clone()
                .write(Some(&released), &theirs)
                .expect("write"),
            ConditionalWriteOutcome::Done
        );
        assert_eq!(
            a.tick(t0 + 2 * S).expect("the first stranger"),
            TickOutcome::Following {
                holder: "a@host:1".to_owned(),
                term: 2
            }
        );
    }

    #[test]
    fn a_term_said_that_leaves_none_to_take_is_refused_before_it_is_counted() {
        let lease = MemoryLease::new();
        let mut a = candidate(&lease, "a@host:1");
        let t0 = Instant::now() + AHEAD;
        a.lead_at(t0);
        for said in [TERM_CEILING - 1, TERM_CEILING, u64::MAX] {
            let refused = a
                .step_down(Some(said), t0)
                .expect_err("no term is left to take");
            assert!(
                matches!(refused, ElectError::Term(next) if next == said.saturating_add(1)),
                "{refused}"
            );
        }
        // Nothing was counted and nothing released: the holder leads on
        // under its own term.
        assert!(a.is_holder());
        assert_eq!(a.fence().leader_term(t0), Some(1));
        assert_eq!(
            a.tick(t0 + S).expect("renew"),
            TickOutcome::Renewed { term: 1 }
        );
        // The highest term a word may say leaves the last one to take.
        assert!(a
            .step_down(Some(TERM_CEILING - 2), t0 + S)
            .expect("the word"));
        let released = a.observed().expect("the release").clone();
        assert_eq!(released.term, TERM_CEILING - 2);
    }

    #[test]
    fn a_shared_records_duration_outlasts_its_fence() {
        let duration = Duration::from_millis(u64::MAX);
        let settings = ElectionSettings {
            duration,
            renew_period: S,
            renew_deadline: duration - S,
        };
        let mut lease = MemoryLease::new();
        let released = held_by("", 1, 0);
        assert_eq!(
            lease.write(None, &released).expect("seed"),
            ConditionalWriteOutcome::Done
        );
        let mut a = Elector::new(Box::new(lease.clone()), "a@host:1", settings).expect("settings");
        let mut b = Elector::new(Box::new(lease.clone()), "b@host:2", settings).expect("settings");
        let t0 = Instant::now() + AHEAD;
        assert!(
            t0.checked_add(Duration::MAX).is_none(),
            "the greatest duration is past the clock's reach"
        );
        assert!(
            later(t0, Duration::MAX) > t0 + 3600 * S,
            "the clock-overflow fallback stays in the distant future"
        );
        assert_eq!(a.tick(t0).expect("take"), TickOutcome::Acquired { term: 2 });
        assert_eq!(a.fence().leader_term(t0), Some(2));
        assert!(matches!(
            b.tick(t0).expect("follow"),
            TickOutcome::Following { .. }
        ));
        let Some(take) = t0.checked_add(duration) else {
            panic!("the record's greatest duration is inside the clock's reach");
        };
        assert_eq!(a.fence().leader_term(take), None);
        assert_eq!(
            b.tick(take).expect("take"),
            TickOutcome::Acquired { term: 3 }
        );
        assert_eq!(b.fence().leader_term(take), Some(3));
    }

    #[test]
    fn settings_out_of_order_are_refused_before_any_step() {
        let bad = [
            (
                ElectionSettings {
                    renew_period: Duration::ZERO,
                    ..settings()
                },
                "the renew period is zero",
            ),
            (
                ElectionSettings {
                    renew_deadline: Duration::from_secs(1),
                    ..settings()
                },
                "the renew deadline is not longer than the renew period",
            ),
            (
                ElectionSettings {
                    duration: Duration::from_secs(2),
                    ..settings()
                },
                "the duration is not longer than the renew deadline",
            ),
        ];
        for (settings, why) in bad {
            let err =
                Elector::new(Box::new(MemoryLease::new()), "a", settings).expect_err("refused");
            assert_eq!(err.to_string(), format!("election settings: {why}"));
        }
        // The design's profile for fewer leadership changes.
        let long = ElectionSettings {
            duration: Duration::from_secs(15),
            renew_period: Duration::from_secs(2),
            renew_deadline: Duration::from_secs(10),
        };
        assert!(long.validate().is_ok());
    }

    #[test]
    fn a_duration_too_large_for_the_record_is_refused() {
        // The whole milliseconds fit; rounded up they do not.
        let edge = Duration::from_millis(u64::MAX) + Duration::from_nanos(1);
        for duration in [edge, Duration::MAX] {
            let settings = ElectionSettings {
                duration,
                renew_period: S,
                renew_deadline: duration - S,
            };
            let err =
                Elector::new(Box::new(MemoryLease::new()), "a", settings).expect_err("refused");
            assert_eq!(
                err.to_string(),
                "election settings: the duration is too large for the lease record"
            );
        }
    }

    #[test]
    fn a_read_that_finds_a_lead_finds_the_end_of_its_window() {
        // An elector that takes and leaves the lead as fast as it can, under
        // a reader of the pair. One read of the fence never shows a lead
        // beside no turn, as two reads astride a release would.
        let fence = Arc::new(Fence::new());
        let t0 = Instant::now();
        let until = t0 + 3_600 * S;
        let stop = Arc::new(AtomicBool::new(false));
        let writer = {
            let (fence, stop) = (Arc::clone(&fence), Arc::clone(&stop));
            thread::spawn(move || {
                let mut term = 0;
                while !stop.load(Ordering::Relaxed) {
                    term += 1;
                    fence.set(term, Some((t0, until)));
                    fence.set(term, None);
                }
            })
        };
        let mut led = 0;
        while led < 50_000 && t0.elapsed() < 10 * S {
            let (term, turn) = fence.term_and_turn(Instant::now());
            if term.is_some() {
                led += 1;
                assert_eq!(turn, Some(until), "a lead with no instant to sleep to");
            }
        }
        stop.store(true, Ordering::Relaxed);
        writer.join().expect("the writer thread");
        assert!(led > 0, "the reader saw no lead");
    }

    #[test]
    fn the_fence_is_read_from_other_threads_without_the_elector() {
        let lease = MemoryLease::new();
        let mut a = candidate(&lease, "a@host:1");
        let t0 = Instant::now() + AHEAD;
        a.lead_at(t0);
        let fence = a.fence();
        let read = thread::spawn(move || fence.leader_term(t0 + S))
            .join()
            .expect("the reader thread");
        assert_eq!(read, Some(1));
        fn assert_send<T: Send>(_: &T) {}
        assert_send(&a);
    }

    #[test]
    fn four_candidates_never_lead_at_once() {
        // Every round is four seconds later, so every follower's sight of the
        // record is expired and all four write for it; the compare-and-swap
        // lets one through per round, and the fences agree.
        let lease = MemoryLease::new();
        let mut electors: Vec<Elector> = (0..4)
            .map(|i| candidate(&lease, &format!("c{i}@host:{i}")))
            .collect();
        let t0 = Instant::now() + AHEAD;
        let mut last_term = 0;
        for round in 0..24u32 {
            let now = t0 + round * 4 * S;
            for i in 0..4 {
                let elector = &mut electors[(i + round as usize) % 4];
                let tick = elector.tick(now).expect("tick");
                if let TickOutcome::Acquired { term } | TickOutcome::Renewed { term } = tick {
                    assert!(
                        term >= last_term,
                        "terms only grow: {term} after {last_term}"
                    );
                    last_term = term;
                }
            }
            let leaders = electors
                .iter()
                .filter(|e| e.fence().leader_term(now).is_some())
                .count();
            assert!(leaders <= 1, "round {round}: {leaders} fenced leaders");
            let holders = electors.iter().filter(|e| e.is_holder()).count();
            assert_eq!(holders, 1, "round {round}: one record holder");
        }
        assert!(last_term >= 2, "the lease changed hands");
    }

    #[test]
    fn a_record_serializes_as_one_json_object() {
        let record = LeaseRecord {
            holder: "a@host:1".to_string(),
            term: 3,
            acquired_at_micros: 10,
            renewed_at_micros: 20,
            duration_ms: 3_000,
        };
        let json = serde_json::to_string(&record).expect("serialize");
        assert_eq!(
            json,
            r#"{"holder":"a@host:1","term":3,"acquired_at_micros":10,"renewed_at_micros":20,"duration_ms":3000}"#
        );
        let back: LeaseRecord = serde_json::from_str(&json).expect("parse");
        assert_eq!(back, record);
        assert!(!record.is_released());
        assert_eq!(record.duration(), Duration::from_secs(3));
    }

    // The backend contract, over every in-tree backend. The third element
    // keeps the backend's place alive: the file's directory, the Kubernetes
    // stand-in's runtime and server.

    type Backend = (&'static str, Box<dyn LeaseBackend>, Box<dyn std::any::Any>);

    fn backends() -> Vec<Backend> {
        let dir = tempfile::tempdir().expect("a temp dir");
        let file = FileLease::new(dir.path().join("lease"));
        let backends: Vec<Backend> = vec![
            ("memory", Box::new(MemoryLease::new()), Box::new(())),
            ("file", Box::new(file), Box::new(dir)),
        ];
        #[cfg(feature = "kubernetes")]
        let backends = {
            let mut backends = backends;
            let cell = kubernetes::tests::Cell::new();
            backends.push(("kubernetes", Box::new(cell.backend()), Box::new(cell)));
            backends
        };
        backends
    }

    #[test]
    fn a_backend_stores_a_record_equal_and_writes_only_over_the_expected_one() {
        for (name, mut backend, _place) in backends() {
            assert_eq!(
                backend.read().expect("read"),
                None,
                "{name}: empty at first"
            );
            let first = LeaseRecord {
                holder: "a@host:1".to_string(),
                term: 1,
                acquired_at_micros: 5,
                renewed_at_micros: 5,
                duration_ms: 3_000,
            };
            assert_eq!(
                backend.write(Some(&first), &first).expect("write"),
                ConditionalWriteOutcome::Moved,
                "{name}: nothing to replace yet"
            );
            assert_eq!(
                backend.read().expect("read"),
                None,
                "{name}: a refused write stores nothing"
            );
            assert_eq!(
                backend.write(None, &first).expect("write"),
                ConditionalWriteOutcome::Done,
                "{name}"
            );
            assert_eq!(
                backend.read().expect("read"),
                Some(first.clone()),
                "{name}: equal, field for field"
            );
            let second = LeaseRecord {
                renewed_at_micros: 6,
                ..first.clone()
            };
            assert_eq!(
                backend.write(None, &second).expect("write"),
                ConditionalWriteOutcome::Moved,
                "{name}: the record exists"
            );
            assert_eq!(
                backend.write(Some(&second), &second).expect("write"),
                ConditionalWriteOutcome::Moved,
                "{name}: not the stored record"
            );
            assert_eq!(backend.read().expect("read"), Some(first.clone()), "{name}");
            assert_eq!(
                backend.write(Some(&first), &second).expect("write"),
                ConditionalWriteOutcome::Done,
                "{name}"
            );
            assert_eq!(backend.read().expect("read"), Some(second), "{name}");
            assert!(!backend.place().is_empty());
        }
    }
}
