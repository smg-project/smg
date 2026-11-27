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

/// The election's timing, shared by every backend (design 10). The default
/// is the design's: a duration of 3 s, a renew period of 1 s and a renew
/// deadline of 2 s.
///
/// The record gains fields: code outside this crate starts from the default
/// and assigns what it states otherwise, and a pattern of it ends with `..`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
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
/// `Lease` object's spec. The default is a record with no holder, under
/// term zero.
///
/// The record gains fields: code outside this crate starts from the default
/// and assigns what it knows, and a pattern of it ends with `..`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
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
#[non_exhaustive]
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
#[non_exhaustive]
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
///
/// The enum is closed: the program that drives an elector acts on each
/// form, it leads, follows or stops, so a form that joins stops its build
/// until it says what it does there.
#[derive(Clone, Debug, PartialEq, Eq)]
#[expect(
    clippy::exhaustive_enums,
    reason = "closed by its doc: a host acts on each form, and a new one must stop its build"
)]
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
#[path = "../../../tests/unit/service/elector/tests/mod.rs"]
mod tests;
