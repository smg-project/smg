//! Leader election (design 10): one [`Elector`] over one [`LeaseBackend`],
//! so a Kubernetes `Lease` and a lease file on one host elect the same way,
//! and the [`Fence`] a shard thread reads before every decision it sends.
//!
//! The lease is one record ([`LeaseRecord`]: holder, term, the acquire and
//! renew wall timestamps, the holder's duration) at one place: a file on a
//! local filesystem ([`FileLease`]), a record in this process
//! ([`MemoryLease`], the one-replica controller), or a `Lease` object of
//! `coordination.k8s.io/v1` (the next step's backend). A backend reads the
//! record and writes it when the stored record is still the one the writer
//! read: a compare-and-swap under the file's advisory lock, the object's
//! resource version on the API server. Everything else is the elector's and
//! the same over every backend:
//!
//! - **Acquire.** A candidate takes a lease that is absent, released, its own
//!   (the same identity after a restart), or that stood unchanged for the
//!   record's duration *by this candidate's own clock*: it times the record
//!   from the instant it first saw it in its present shape, never from the
//!   wall timestamp the record carries, so no second clock enters the
//!   decision. The take writes the next term: terms only grow, one per change
//!   of holder.
//! - **Renew.** The holder rewrites the record every renew period with a
//!   later wall timestamp under its own term; a record that moved under it (a
//!   candidate took the expired lease) refuses the renewal and the holder has
//!   lost.
//! - **Fence.** The holder records the monotonic instant of its last
//!   successful renewal in the [`Fence`]. Past the renew deadline without a
//!   renewal the fence closes by itself: [`Fence::leader_term`] answers
//!   `None`, the shard thread answers `NotLeader` with an empty address and
//!   releases nothing, whether the election thread is stalled on its backend
//!   or gone. The deadline is shorter than the duration, so a holder stops
//!   serving before any candidate may take its lease; the residual window is
//!   the renew deadline plus clock-rate skew, bounded by one decision's send.
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
//! the returned [`Tick`] logged, the term handed to the store
//! (`LeasesStore::set_term`) on `Acquired`, [`Elector::release`] on a planned
//! stop. A backend call blocks that thread alone; the shard threads read the
//! fence's snapshot, never the backend. The identity is
//! `controller-id@host:port`, dialable without a Service and unique per
//! instance: two instances under one identity would both count the lease as
//! their own.

mod file;

use std::{
    fmt,
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use arc_swap::ArcSwap;
pub use file::FileLease;
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
    /// The order the fence rests on: a renew period above zero, a renew
    /// deadline longer than it, a duration longer than the deadline.
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
    /// a holder's renewals and its restarts.
    pub term: u64,
    /// Wall time of the take, microseconds since the Unix epoch.
    pub acquired_at_micros: u64,
    /// Wall time of the last renewal, microseconds since the Unix epoch;
    /// later than the record's last at every write, so an observer sees
    /// every renewal as a change.
    pub renewed_at_micros: u64,
    /// The holder's duration in milliseconds: how long the record may stand
    /// unchanged before another candidate takes it; `0` once released.
    pub duration_ms: u64,
}

impl LeaseRecord {
    /// Free for the taking: released, or never held.
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
pub enum Written {
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
    /// none exists); otherwise write nothing and answer [`Written::Moved`].
    fn write(
        &mut self,
        expected: Option<&LeaseRecord>,
        record: &LeaseRecord,
    ) -> Result<Written, ElectError>;

    /// Where the record lives, for logs: the file's path, the object's
    /// namespace and name.
    fn place(&self) -> String;
}

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
}

/// What one [`Elector::tick`] did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Tick {
    /// This candidate took the lease and leads under `term` from now.
    Acquired { term: u64 },
    /// The lease is renewed under `term`; the fence stays open for another
    /// renew deadline.
    Renewed { term: u64 },
    /// The lease moved under this holder: a candidate took it after it stood
    /// unchanged for the duration, or the record was rewritten by hand. The
    /// fence is closed. `holder` is the record's holder now, empty when the
    /// lease is released or gone.
    Lost { term: u64, holder: String },
    /// Another candidate holds the lease and renews it under `term`.
    Following { holder: String, term: u64 },
    /// The lease looked free, but another candidate wrote first; the next
    /// tick reads what it wrote.
    Raced,
}

/// What the shard threads read: the term this process leads under and the
/// instant of its last renewal, replaced whole at every renewal and read
/// lock-free before every decision.
#[derive(Debug)]
pub struct Fence {
    renew_deadline: Duration,
    lead: ArcSwap<Lead>,
}

#[derive(Debug, Default)]
struct Lead {
    term: u64,
    /// The instant of the last successful renewal; `None` while no lease is
    /// held.
    renewed: Option<Instant>,
}

impl Fence {
    fn new(renew_deadline: Duration) -> Self {
        Self {
            renew_deadline,
            lead: ArcSwap::from_pointee(Lead::default()),
        }
    }

    /// The term this process leads under at `now`, or `None`: no lease held,
    /// or the last renewal is past the renew deadline (the self-fence). The
    /// shard thread's read before every decision it sends; past the deadline
    /// it answers `NotLeader` and releases nothing.
    pub fn leader_term(&self, now: Instant) -> Option<u64> {
        let lead = self.lead.load();
        let renewed = lead.renewed?;
        (now.saturating_duration_since(renewed) <= self.renew_deadline).then_some(lead.term)
    }

    /// The term of the last lease this process held, `0` before any: what
    /// its frames carry, leading or not.
    pub fn term(&self) -> u64 {
        self.lead.load().term
    }

    fn set(&self, term: u64, renewed: Option<Instant>) {
        self.lead.store(Arc::new(Lead { term, renewed }));
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
            fence: Arc::new(Fence::new(settings.renew_deadline)),
            observed: None,
            held: None,
        })
    }

    /// The fence the shard threads read; one per elector, shared by handle.
    pub fn fence(&self) -> Arc<Fence> {
        Arc::clone(&self.fence)
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

    /// One step of the election at `now`: a holder renews, a candidate takes
    /// a free lease or keeps following. Call it every renew period. An error
    /// changes nothing: a holder keeps its record and its fence closes by
    /// itself at the deadline; the next successful step settles it.
    pub fn tick(&mut self, now: Instant) -> Result<Tick, ElectError> {
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
            Some(mine) => self.renew(current, mine, now),
            None => self.acquire(current, now),
        }
    }

    /// Give the lease up: the record is rewritten released, so the next
    /// candidate to look takes it at once. The fence closes now and stays
    /// closed. `true` when this process rewrote the record; `false` when it
    /// held nothing, or when the lease had moved under it already. On an
    /// error the record stays held by this identity and the elector keeps
    /// it, so a second call retries the write; a caller that ticks instead
    /// renews.
    pub fn release(&mut self) -> Result<bool, ElectError> {
        let Some(mine) = self.held.clone() else {
            return Ok(false);
        };
        self.fence.set(mine.term, None);
        let released = LeaseRecord {
            holder: String::new(),
            duration_ms: 0,
            renewed_at_micros: next_wall_micros(mine.renewed_at_micros),
            ..mine.clone()
        };
        let written = self.backend.write(Some(&mine), &released)?;
        self.held = None;
        if written == Written::Done {
            if let Some(observation) = &mut self.observed {
                observation.record = released;
            }
        }
        Ok(written == Written::Done)
    }

    fn observe(&mut self, current: Option<&LeaseRecord>, now: Instant) {
        match (current, &mut self.observed) {
            (Some(record), Some(observation)) if observation.record == *record => {}
            (Some(record), observation) => {
                *observation = Some(Observation {
                    record: record.clone(),
                    since: now,
                });
            }
            (None, observation) => *observation = None,
        }
    }

    fn renew(
        &mut self,
        current: Option<LeaseRecord>,
        mine: LeaseRecord,
        now: Instant,
    ) -> Result<Tick, ElectError> {
        if current.as_ref() != Some(&mine) {
            return Ok(self.lose(mine.term, current));
        }
        let renewed = LeaseRecord {
            renewed_at_micros: next_wall_micros(mine.renewed_at_micros),
            ..mine.clone()
        };
        match self.backend.write(Some(&mine), &renewed)? {
            Written::Done => {
                self.fence.set(renewed.term, Some(now));
                self.observed = Some(Observation {
                    record: renewed.clone(),
                    since: now,
                });
                self.held = Some(renewed);
                Ok(Tick::Renewed { term: mine.term })
            }
            Written::Moved => {
                let current = self.backend.read()?;
                Ok(self.lose(mine.term, current))
            }
        }
    }

    fn lose(&mut self, term: u64, current: Option<LeaseRecord>) -> Tick {
        self.held = None;
        self.fence.set(term, None);
        Tick::Lost {
            term,
            holder: current.map(|record| record.holder).unwrap_or_default(),
        }
    }

    fn acquire(&mut self, current: Option<LeaseRecord>, now: Instant) -> Result<Tick, ElectError> {
        let (term, acquired_at_micros) = match &current {
            None => (1, 0),
            Some(record) if record.is_released() => (record.term + 1, 0),
            // The same identity again: its own lease, under its own term.
            Some(record) if record.holder == self.identity => {
                (record.term, record.acquired_at_micros)
            }
            Some(record) => {
                let since = self.observed.as_ref().map_or(now, |o| o.since);
                if now.saturating_duration_since(since) < record.duration() {
                    return Ok(Tick::Following {
                        holder: record.holder.clone(),
                        term: record.term,
                    });
                }
                (record.term + 1, 0)
            }
        };
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
            duration_ms: whole_millis_up(self.settings.duration),
        };
        match self.backend.write(current.as_ref(), &record)? {
            Written::Done => {
                self.fence.set(term, Some(now));
                self.observed = Some(Observation {
                    record: record.clone(),
                    since: now,
                });
                self.held = Some(record);
                Ok(Tick::Acquired { term })
            }
            Written::Moved => Ok(Tick::Raced),
        }
    }
}

/// A duration in whole milliseconds, rounded up: the record's duration is
/// never shorter than the settings' and so never shorter than the renew
/// deadline the fence runs on.
fn whole_millis_up(duration: Duration) -> u64 {
    let whole = duration.as_millis();
    let rounded = if duration.subsec_nanos().is_multiple_of(1_000_000) {
        whole
    } else {
        whole + 1
    };
    u64::try_from(rounded).unwrap_or(u64::MAX)
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
    ) -> Result<Written, ElectError> {
        let mut slot = self.record.lock();
        if slot.as_ref() != expected {
            return Ok(Written::Moved);
        }
        *slot = Some(record.clone());
        Ok(Written::Done)
    }

    fn place(&self) -> String {
        "memory".to_string()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::atomic::{AtomicU32, Ordering},
        thread,
        time::Duration,
    };

    use super::*;

    const S: Duration = Duration::from_secs(1);

    fn settings() -> ElectionSettings {
        ElectionSettings::default()
    }

    fn candidate(lease: &MemoryLease, identity: &str) -> Elector {
        Elector::new(Box::new(lease.clone()), identity, settings()).expect("valid settings")
    }

    #[test]
    fn a_candidate_takes_an_absent_lease_at_term_one_and_leads_until_the_deadline() {
        let lease = MemoryLease::new();
        let mut a = candidate(&lease, "a@host:1");
        let t0 = Instant::now() + S;
        assert_eq!(a.tick(t0).expect("tick"), Tick::Acquired { term: 1 });
        assert!(a.is_holder());
        let fence = a.fence();
        assert_eq!(fence.leader_term(t0), Some(1));
        assert_eq!(fence.leader_term(t0 + 2 * S), Some(1), "at the deadline");
        assert_eq!(
            fence.leader_term(t0 + 2 * S + Duration::from_micros(1)),
            None,
            "past the renew deadline the fence closes by itself"
        );
        assert_eq!(fence.term(), 1, "the term stays on the frames");
        let record = a.observed().expect("the record").clone();
        assert_eq!(record.holder, "a@host:1");
        assert_eq!(record.duration_ms, 3_000);
        assert_eq!(record.acquired_at_micros, record.renewed_at_micros);
        assert!(record.acquired_at_micros > 0);
    }

    #[test]
    fn a_renewal_reopens_the_fence_and_keeps_the_term() {
        let lease = MemoryLease::new();
        let mut a = candidate(&lease, "a@host:1");
        let t0 = Instant::now() + S;
        a.tick(t0).expect("acquire");
        let first = a.observed().expect("the record").clone();
        assert_eq!(a.tick(t0 + S).expect("renew"), Tick::Renewed { term: 1 });
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
        let t0 = Instant::now() + S;
        a.tick(t0).expect("acquire");
        // b first sees the record half a second after it was written; the
        // record's clock says nothing to b.
        let t1 = t0 + S / 2;
        assert_eq!(
            b.tick(t1).expect("tick"),
            Tick::Following {
                holder: "a@host:1".to_string(),
                term: 1
            }
        );
        assert!(matches!(
            b.tick(t0 + 3 * S).expect("tick"),
            Tick::Following { .. }
        ));
        assert!(matches!(
            b.tick(t1 + 3 * S - Duration::from_micros(1)).expect("tick"),
            Tick::Following { .. }
        ));
        assert_eq!(
            b.tick(t1 + 3 * S).expect("tick"),
            Tick::Acquired { term: 2 }
        );
        assert_eq!(b.fence().leader_term(t1 + 3 * S), Some(2));
    }

    #[test]
    fn a_stalled_holders_renewal_is_refused_by_the_moved_record() {
        let lease = MemoryLease::new();
        let mut a = candidate(&lease, "a@host:1");
        let mut b = candidate(&lease, "b@host:2");
        let t0 = Instant::now() + S;
        a.tick(t0).expect("acquire");
        b.tick(t0).expect("follow");
        // a stalls past the duration; b takes the lease.
        assert_eq!(
            b.tick(t0 + 3 * S).expect("take"),
            Tick::Acquired { term: 2 }
        );
        assert_eq!(
            a.fence().leader_term(t0 + 3 * S),
            None,
            "a fenced itself at 2 s"
        );
        assert_eq!(
            a.tick(t0 + 3 * S + S / 2).expect("the stalled renewal"),
            Tick::Lost {
                term: 1,
                holder: "b@host:2".to_string()
            }
        );
        assert!(!a.is_holder());
        assert_eq!(a.fence().term(), 1, "the lost term stays on a's frames");
        assert_eq!(
            a.tick(t0 + 4 * S + S / 2).expect("tick"),
            Tick::Following {
                holder: "b@host:2".to_string(),
                term: 2
            }
        );
        assert_eq!(
            b.tick(t0 + 4 * S).expect("renew"),
            Tick::Renewed { term: 2 }
        );
    }

    #[test]
    fn a_renewing_holder_keeps_its_lease_against_a_waiting_candidate() {
        let lease = MemoryLease::new();
        let mut a = candidate(&lease, "a@host:1");
        let mut b = candidate(&lease, "b@host:2");
        let t0 = Instant::now() + S;
        a.tick(t0).expect("acquire");
        for i in 1..=10u32 {
            let now = t0 + i * S;
            assert_eq!(a.tick(now).expect("renew"), Tick::Renewed { term: 1 });
            assert!(matches!(
                b.tick(now).expect("follow"),
                Tick::Following { term: 1, .. }
            ));
            assert_eq!(a.fence().leader_term(now), Some(1));
            assert_eq!(b.fence().leader_term(now), None);
        }
    }

    #[test]
    fn a_released_lease_is_taken_at_once_with_the_next_term() {
        let lease = MemoryLease::new();
        let mut a = candidate(&lease, "a@host:1");
        let mut b = candidate(&lease, "b@host:2");
        let t0 = Instant::now() + S;
        a.tick(t0).expect("acquire");
        assert!(matches!(
            b.tick(t0).expect("follow"),
            Tick::Following { .. }
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
            Tick::Acquired { term: 2 }
        );
        assert!(
            a.observed().expect("a's last write").is_released(),
            "a's last sight of the record is its own release"
        );
    }

    #[test]
    fn the_same_identity_resumes_its_own_lease_under_its_term() {
        let lease = MemoryLease::new();
        let t0 = Instant::now() + S;
        let acquired_at = {
            let mut a = candidate(&lease, "a@host:1");
            a.tick(t0).expect("acquire");
            a.observed().expect("the record").acquired_at_micros
        };
        // The process restarts: a fresh elector under the same identity.
        let mut again = candidate(&lease, "a@host:1");
        assert_eq!(
            again.tick(t0 + S / 10).expect("resume"),
            Tick::Acquired { term: 1 }
        );
        let record = again.observed().expect("the record");
        assert_eq!(record.acquired_at_micros, acquired_at, "the take stands");
        assert!(record.renewed_at_micros > acquired_at);
        assert_eq!(again.fence().leader_term(t0 + S / 10), Some(1));
    }

    #[test]
    fn terms_only_grow_across_takes_and_releases() {
        let lease = MemoryLease::new();
        let mut a = candidate(&lease, "a@host:1");
        let mut b = candidate(&lease, "b@host:2");
        let mut c = candidate(&lease, "c@host:3");
        let t0 = Instant::now() + S;
        assert_eq!(a.tick(t0).expect("tick"), Tick::Acquired { term: 1 });
        a.release().expect("release");
        assert_eq!(b.tick(t0 + S).expect("tick"), Tick::Acquired { term: 2 });
        c.tick(t0 + S).expect("c sees b");
        assert_eq!(
            c.tick(t0 + 4 * S).expect("b expired for c"),
            Tick::Acquired { term: 3 }
        );
        c.release().expect("release");
        assert_eq!(
            a.tick(t0 + 5 * S).expect("tick"),
            Tick::Acquired { term: 4 }
        );
        assert_eq!(
            b.tick(t0 + 5 * S).expect("b's stalled renewal"),
            Tick::Lost {
                term: 2,
                holder: "a@host:1".to_string()
            }
        );
    }

    /// A backend that lets another candidate write between this candidate's
    /// read and its write: the race a compare-and-swap exists for.
    struct Interposed {
        lease: MemoryLease,
        rival: Option<LeaseRecord>,
    }

    impl LeaseBackend for Interposed {
        fn read(&mut self) -> Result<Option<LeaseRecord>, ElectError> {
            self.lease.read()
        }

        fn write(
            &mut self,
            expected: Option<&LeaseRecord>,
            record: &LeaseRecord,
        ) -> Result<Written, ElectError> {
            if let Some(rival) = self.rival.take() {
                assert_eq!(
                    self.lease
                        .write(expected, &rival)
                        .expect("the rival's write"),
                    Written::Done
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
            rival: Some(rival.clone()),
        };
        let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
        let t0 = Instant::now() + S;
        assert_eq!(a.tick(t0).expect("tick"), Tick::Raced);
        assert!(!a.is_holder());
        assert_eq!(a.fence().leader_term(t0), None);
        assert_eq!(
            a.tick(t0 + S).expect("tick"),
            Tick::Following {
                holder: "r@host:9".to_string(),
                term: 1
            }
        );
        assert_eq!(lease.clone().read().expect("read"), Some(rival));
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
        ) -> Result<Written, ElectError> {
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
        let t0 = Instant::now() + S;
        assert_eq!(a.tick(t0).expect("acquire"), Tick::Acquired { term: 1 });
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
            Tick::Renewed { term: 1 }
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
    fn a_failed_release_keeps_the_record_for_a_retry_and_the_fence_closed() {
        let lease = MemoryLease::new();
        let fails = Arc::new(AtomicU32::new(0));
        let backend = Failing {
            lease: lease.clone(),
            fails: Arc::clone(&fails),
        };
        let mut a = Elector::new(Box::new(backend), "a@host:1", settings()).expect("settings");
        let t0 = Instant::now() + S;
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
    fn a_release_after_the_lease_moved_answers_false() {
        let lease = MemoryLease::new();
        let mut a = candidate(&lease, "a@host:1");
        let mut b = candidate(&lease, "b@host:2");
        let t0 = Instant::now() + S;
        a.tick(t0).expect("acquire");
        b.tick(t0).expect("follow");
        assert_eq!(
            b.tick(t0 + 3 * S).expect("take"),
            Tick::Acquired { term: 2 }
        );
        assert!(
            !a.release().expect("nothing to give up"),
            "the lease is b's"
        );
        assert!(!a.is_holder());
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
    }

    impl LeaseBackend for SlowRead {
        fn read(&mut self) -> Result<Option<LeaseRecord>, ElectError> {
            if let Some(stall) = self.stall.take() {
                thread::sleep(stall);
                self.holder.lock().tick(Instant::now())?;
                thread::sleep(Duration::from_millis(5));
            }
            self.lease.read()
        }

        fn write(
            &mut self,
            expected: Option<&LeaseRecord>,
            record: &LeaseRecord,
        ) -> Result<Written, ElectError> {
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
                Tick::Acquired { term: 1 }
            );
            a.fence()
        };
        let backend = SlowRead {
            lease,
            holder: Arc::clone(&holder),
            stall: Some(Duration::from_millis(90)),
        };
        let mut b = Elector::new(Box::new(backend), "b@host:2", fast).expect("settings");
        // b's first look begins now; 90 ms into the read the holder renews,
        // so its fence runs 80 ms from that renewal; then the holder stalls
        // for good. Dated before the read, b's wait would end inside that
        // window.
        assert!(matches!(
            b.tick(Instant::now()).expect("look"),
            Tick::Following { .. }
        ));
        let started = Instant::now();
        let took = loop {
            assert!(
                started.elapsed() < Duration::from_secs(5),
                "b never took the lease"
            );
            let now = Instant::now();
            if b.tick(now).expect("look") == (Tick::Acquired { term: 2 }) {
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
        a.tick(Instant::now() + S).expect("acquire");
        let record = lease.clone().read().expect("read").expect("the record");
        assert_eq!(record.duration_ms, 2, "1.9 ms rounds up");
        assert!(record.duration() >= fine.duration);
        assert!(
            record.duration() > fine.renew_deadline,
            "a follower waits past the fence"
        );
        assert_eq!(whole_millis_up(Duration::from_secs(3)), 3_000);
        assert_eq!(whole_millis_up(Duration::from_millis(1)), 1);
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
    fn the_fence_is_read_from_other_threads_without_the_elector() {
        let lease = MemoryLease::new();
        let mut a = candidate(&lease, "a@host:1");
        let t0 = Instant::now() + S;
        a.tick(t0).expect("acquire");
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
        let t0 = Instant::now() + S;
        let mut last_term = 0;
        for round in 0..24u32 {
            let now = t0 + round * 4 * S;
            for i in 0..4 {
                let elector = &mut electors[(i + round as usize) % 4];
                let tick = elector.tick(now).expect("tick");
                if let Tick::Acquired { term } | Tick::Renewed { term } = tick {
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

    // The backend contract, over every in-tree backend.

    fn backends() -> Vec<(&'static str, Box<dyn LeaseBackend>, tempfile::TempDir)> {
        let dir = tempfile::tempdir().expect("a temp dir");
        let file = FileLease::new(dir.path().join("lease"));
        let dir2 = tempfile::tempdir().expect("a temp dir");
        vec![
            ("memory", Box::new(MemoryLease::new()), dir2),
            ("file", Box::new(file), dir),
        ]
    }

    #[test]
    fn a_backend_stores_a_record_equal_and_writes_only_over_the_expected_one() {
        for (name, mut backend, _dir) in backends() {
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
                Written::Moved,
                "{name}: nothing to replace yet"
            );
            assert_eq!(
                backend.read().expect("read"),
                None,
                "{name}: a refused write stores nothing"
            );
            assert_eq!(
                backend.write(None, &first).expect("write"),
                Written::Done,
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
                Written::Moved,
                "{name}: the record exists"
            );
            assert_eq!(
                backend.write(Some(&second), &second).expect("write"),
                Written::Moved,
                "{name}: not the stored record"
            );
            assert_eq!(backend.read().expect("read"), Some(first.clone()), "{name}");
            assert_eq!(
                backend.write(Some(&first), &second).expect("write"),
                Written::Done,
                "{name}"
            );
            assert_eq!(backend.read().expect("read"), Some(second), "{name}");
            assert!(!backend.place().is_empty());
        }
    }
}
