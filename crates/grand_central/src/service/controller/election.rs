//! The election wired into the run loop (design 5.3): where the lease lives,
//! the identity this replica holds it under, the thread that steps the
//! elector, the standing the `Leader` call answers from on a replica that
//! does not lead, and the reader of who leads ([`Leadership`]).
//!
//! One path in every mode: a controller with no shared lease elects itself
//! over a record in its own process, so the [`Fence`]
//! a shard thread reads is the same object whether one replica runs or
//! three. The fence is the one judge of "leader": readiness, the state gauge
//! and the `Leader` call read it at the instant they answer, and nothing is
//! told its role by a message that can be late.
//!
//! The reader holds that fence, the identity, the standing and the log the
//! replica's store journals to, each a private field. The service's calls
//! ask it through their shared state, each at its own instant, and hand it
//! no term: it derives the leader from its own read of the fence, and it
//! checks the log for a call that needs a led store ([`Leadership::led`]).
//! The source of a watch of the worker rows alone takes the fence itself
//! from it ([`Leadership::fence`]).
//!
//! The thread is a plain thread, never a runtime worker: a backend call
//! blocks it alone. It steps the elector every renew period, hands the term
//! to the store at a take and publishes what it saw. At the shutdown signal
//! it closes the fence and rewrites the record released, in that order, so
//! no instant has two open fences and the next candidate to look takes the
//! lease at once, or after the wait this thread's own take still owed; it
//! then stops and never takes again.
//!
//! A word to leave (`ElectionNotifier::hand_over` for a planned hand-over,
//! `ElectionNotifier::saw` for a term a caller says it saw) is taken by the
//! same thread between two steps: the fence closed, the lease released, no take
//! for one lease duration (a replica alone over a lease in its own process
//! takes it again at its next step, behind the word). The thread goes on
//! stepping. A decider says the planned hand-over itself where its term's lease
//! ids pass their mark: the next take brings a term whose ids start again. The
//! hand-over names that term, and the thread drops one whose term this replica
//! no longer holds the lease under: that lease is left already.
//!
//! The thread leaves one lease with no word: a take that starts no term in
//! the store's log. The log followed a term at or above the take's, so
//! another replica led that term. The thread leaves in the take's own step,
//! ahead of the standing, as at a word of a term seen. The next take stands
//! above the followed term, and the log leads it. Where no term is left to
//! take above the followed term, the thread keeps the lease.
//!
//! Behind either leave of a replica alone over its lease, the wait for the
//! next step is none. The thread first reads the words on its channel, in
//! their order, until it finds none, and works each as at any other time.
//! So a stop read there ends the thread with no further take, as a closed
//! channel does, and a take the next step makes stands above a term
//! counted there. A word behind a stop is never read, and a word that
//! comes behind the last read is read behind that step.

use std::{
    fmt, io,
    panic::{catch_unwind, resume_unwind, AssertUnwindSafe},
    path::{Path, PathBuf},
    sync::{mpsc, Arc, Weak},
    thread,
    time::Instant,
};

use tokio::sync::{oneshot, watch};
use tracing::{debug, info, warn};

#[cfg(feature = "kubernetes")]
use crate::service::elector::KubernetesLease;
use crate::{
    service::{
        elector::{
            term_said, ElectError, ElectionSettings, Elector, Fence, FileLease, LeaseBackend,
            TickOutcome,
        },
        replication::{Log, NotLeading},
    },
    store::{LeasesStore, MemoryStore},
};

/// How many times a planned stop writes the release before it leaves the
/// lease to expire by its duration.
const RELEASE_ATTEMPTS: u32 = 3;

/// Where the lease lives.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum LeasePlace {
    /// A record in this process: one replica, which leads itself. A second
    /// replica under this place leads too; replicas need a shared place.
    #[default]
    Process,
    /// A file on a local filesystem: the replicas of one host.
    File(PathBuf),
    /// A `Lease` object of `coordination.k8s.io/v1`.
    #[cfg(feature = "kubernetes")]
    Kubernetes {
        /// The object's namespace; `None` takes the client configuration's
        /// own, a pod's namespace inside a cluster.
        namespace: Option<String>,
        /// The object's name.
        name: String,
    },
}

impl LeasePlace {
    /// Whether another replica can see the lease.
    pub(super) fn is_shared(&self) -> bool {
        !matches!(self, Self::Process)
    }
}

impl fmt::Display for LeasePlace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Process => f.write_str("this process (one replica)"),
            Self::File(path) => write!(f, "file {}", path.display()),
            #[cfg(feature = "kubernetes")]
            Self::Kubernetes { namespace, name } => match namespace {
                Some(namespace) => write!(f, "Lease {namespace}/{name}"),
                None => write!(f, "Lease {name}"),
            },
        }
    }
}

/// The election's configuration (design 5.3). The default is one replica
/// that leads itself over a lease in its own process.
///
/// The record gains fields: code outside this crate starts from the default
/// and assigns what it states otherwise, and a pattern of it ends with `..`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub struct ElectionConfig {
    /// Where the lease lives.
    pub lease: LeasePlace,
    /// The timing: duration 3 s, renew period 1 s, renew deadline 2 s by
    /// default.
    pub settings: ElectionSettings,
    /// This replica's dialable address, `host` or `host:port`: what the
    /// holder identity and the `Leader` call name, so a client reaches the
    /// leader without a Service. A host alone takes the gateway listener's
    /// bound port. An address is written in its standard form: the wildcard
    /// is refused, and so is any other host that ends in a number. `None`
    /// takes the gateway listener's own address when it binds one; a shared
    /// lease behind a wildcard listener needs it set.
    pub advertise: Option<String>,
}

/// The holder identity (design 5.3): `controller-id@host:port`, dialable
/// without a Service and unique per instance; the id alone when no address
/// is known.
pub(crate) fn identity(controller_id: &str, address: Option<&str>) -> String {
    match address {
        Some(address) => format!("{controller_id}@{address}"),
        None => controller_id.to_owned(),
    }
}

/// A holder identity as its controller id and its address; the address is
/// empty when the identity carries none.
fn split_identity(identity: &str) -> (&str, &str) {
    identity.split_once('@').unwrap_or((identity, ""))
}

/// The lease could not be opened.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum LeaseError {
    /// The lease file's directory is not there: the file is created at the
    /// first take, its directory never.
    #[error("the lease file's directory {}: {source}", .directory.display())]
    Directory {
        /// The directory.
        directory: PathBuf,
        /// Why it could not be read.
        #[source]
        source: io::Error,
    },
    /// The lease file's directory is a file.
    #[error("the lease file's directory {} is not a directory", .0.display())]
    NotADirectory(PathBuf),
    /// No client configuration for the `Lease` object's API server.
    #[error("the Lease's client: {0}")]
    Client(String),
    /// The elector refused its settings or its identity.
    #[error(transparent)]
    Elector(#[from] ElectError),
}

/// The elector of `election` under `identity`. The lease's place is opened
/// here, on the runtime the election thread will block on, so a place that
/// is not there refuses the start before a listener serves.
pub(crate) async fn elector(
    election: &ElectionConfig,
    identity: String,
) -> Result<Elector, LeaseError> {
    let backend: Box<dyn LeaseBackend> = match &election.lease {
        // The elector makes this place's store itself and hands it to
        // nobody: no other holder can be, and its take leads at once.
        LeasePlace::Process => return Ok(Elector::alone(identity, election.settings)?),
        LeasePlace::File(path) => {
            let directory = match path.parent() {
                Some(directory) if !directory.as_os_str().is_empty() => directory,
                _ => Path::new("."),
            };
            let metadata =
                tokio::fs::metadata(directory)
                    .await
                    .map_err(|source| LeaseError::Directory {
                        directory: directory.to_owned(),
                        source,
                    })?;
            if !metadata.is_dir() {
                return Err(LeaseError::NotADirectory(directory.to_owned()));
            }
            Box::new(FileLease::new(path))
        }
        #[cfg(feature = "kubernetes")]
        LeasePlace::Kubernetes { namespace, name } => {
            let inferred = kube::Config::infer()
                .await
                .map_err(|error| LeaseError::Client(error.to_string()))?;
            Box::new(kubernetes_lease(
                inferred,
                namespace.as_deref(),
                name,
                election.settings.renew_deadline,
            )?)
        }
    };
    Ok(Elector::new(backend, identity, election.settings)?)
}

/// The `Lease` object `name` as a lease's backend, on the client
/// configuration `inferred` for this process; with no `namespace`, in the
/// configuration's own. Its calls block on the runtime this runs on.
#[cfg(feature = "kubernetes")]
fn kubernetes_lease(
    inferred: kube::Config,
    namespace: Option<&str>,
    name: &str,
    deadline: std::time::Duration,
) -> Result<KubernetesLease, LeaseError> {
    let namespace = namespace.map_or_else(|| inferred.default_namespace.clone(), str::to_owned);
    let client = kube::Client::try_from(lease_config(inferred, deadline))
        .map_err(|error| LeaseError::Client(error.to_string()))?;
    Ok(KubernetesLease::new(
        client,
        &namespace,
        name,
        tokio::runtime::Handle::current(),
    ))
}

/// `inferred` as a lease's client takes it: every round trip is bounded by
/// the renew `deadline` and made once.
///
/// A stalled round trip returns before the fence it could have renewed
/// closes, so the thread's next step is never behind a call that outlives
/// the lease. The client library's own retry of an answer that says try
/// later (429, 503, 504) would hold one call past that bound, so it is off:
/// the thread's next step is the retry.
#[cfg(feature = "kubernetes")]
fn lease_config(mut inferred: kube::Config, deadline: std::time::Duration) -> kube::Config {
    inferred.connect_timeout = Some(deadline);
    inferred.read_timeout = Some(deadline);
    inferred.write_timeout = Some(deadline);
    inferred.default_retry = false;
    inferred
}

/// What the election's last step saw of the lease.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct ElectionObservation {
    /// The lease's holder, `controller-id@host:port`; empty when the lease
    /// is free, when the last look failed, and when it read a term under
    /// the highest this replica saw: no write stored that holder under
    /// the term beside it.
    pub(crate) holder: String,
    /// The highest term this replica saw, in the lease or said by a caller.
    pub(crate) term: u64,
    /// Whether a look at the lease has succeeded: readiness waits for the
    /// first, so a replica that cannot reach its lease never reads ready.
    pub(crate) looked: bool,
}

/// Who leads, as this replica knows it by its fence's verdict at one
/// instant.
pub(crate) struct Leader {
    /// The leader's id; empty when nobody is named.
    pub(crate) controller_id: String,
    /// Its dialable address; empty when nobody is named or it has none.
    pub(crate) address: String,
    /// The fence's term where this replica leads. Elsewhere the higher of
    /// two: the standing's term, and the term of the last lease this
    /// replica held.
    pub(crate) term: u64,
    /// Whether the leader is this replica.
    pub(crate) is_leader: bool,
}

/// Why a call that needs a led store is not served ([`Leadership::led`]).
pub(crate) enum NotLed {
    /// The fence is closed at the call's instant: this replica does not
    /// lead.
    Elsewhere {
        /// The holder's dialable address; empty when no leader is known or
        /// it has none.
        holder: String,
    },
    /// The fence shows a term the store's log does not lead.
    Store(NotLeading),
}

/// The reader of who leads that the service's calls ask through their
/// shared state. It holds the controller's fence, the identity the replica
/// elects under, the election's last sight of the lease and the log its
/// store journals to, each a private field, with the id and the address it
/// names itself by. No method of it takes a term from its caller: the
/// leader is derived from its own reads.
pub(crate) struct Leadership {
    /// This controller's id.
    controller_id: String,
    /// Its dialable address, `host:port`; empty when no client could be
    /// told one.
    address: String,
    /// The holder identity it elects under.
    identity: String,
    /// The controller's one fence: the judge of "leader" at every answer.
    fence: Arc<Fence>,
    /// The election's last sight of the lease.
    standing: watch::Receiver<ElectionObservation>,
    /// The log the replica's store journals to: it says whether the store
    /// leads a term.
    log: Arc<Log>,
}

impl Leadership {
    /// The reader of the replica `elector` stands for, named by
    /// `controller_id` and `address`. The identity and the fence are the
    /// elector's, `standing` receives its thread's sights of the lease, and
    /// `log` is the one the replica's store journals to.
    pub(crate) fn new(
        controller_id: String,
        address: String,
        elector: &Elector,
        standing: watch::Receiver<ElectionObservation>,
        log: Arc<Log>,
    ) -> Self {
        Self {
            controller_id,
            address,
            identity: elector.identity().to_owned(),
            fence: elector.fence(),
            standing,
            log,
        }
    }

    /// The term this replica leads under at `now`, or `None`: one read of
    /// the fence ([`Fence::leader_term`]).
    pub(crate) fn leading(&self, now: Instant) -> Option<u64> {
        self.fence.leader_term(now)
    }

    /// The term of the last lease this replica held, `0` before any
    /// ([`Fence::term`]).
    pub(crate) fn held_term(&self) -> u64 {
        self.fence.term()
    }

    /// Who leads, by the fence's verdict at `now` ([`Self::leading`]). A
    /// replica whose fence is open names itself under the fence's term. Any
    /// other answers as [`Self::leader_elsewhere`] does, which reads the
    /// fence once more, for the term of the last lease held. Under the
    /// standing the election's own thread publishes, no answer falls in
    /// term: that standing's term is the highest its elector saw or was
    /// told, and that elector takes no term under its highest.
    pub(crate) fn leader(&self, now: Instant) -> Leader {
        match self.leading(now) {
            Some(term) => Leader {
                controller_id: self.controller_id.clone(),
                address: self.address.clone(),
                term,
                is_leader: true,
            },
            None => self.leader_elsewhere(),
        }
    }

    /// Who leads, for a replica whose fence read closed. It names the
    /// holder its last look at the lease saw, or nobody (an empty address:
    /// no leader known) when the lease is free, when that look failed, when
    /// it read a term under the highest this replica saw, when the record
    /// still names this replica behind a closed fence, and when this
    /// replica's own take is newer than that look: a holder is named under
    /// a term one write of the lease stored for it. The term is the higher
    /// of two: the standing's, and the term of the last lease this replica
    /// held ([`Fence::term`]).
    pub(crate) fn leader_elsewhere(&self) -> Leader {
        let standing = self.standing.borrow();
        let fenced = self.fence.term();
        let term = standing.term.max(fenced);
        let (controller_id, address) = if standing.holder.is_empty()
            || standing.holder == self.identity
            || fenced > standing.term
        {
            (String::new(), String::new())
        } else {
            let (id, address) = split_identity(&standing.holder);
            (id.to_owned(), address.to_owned())
        };
        Leader {
            controller_id,
            address,
            term,
            is_leader: false,
        }
    }

    /// The term this replica leads under at `now`, once its store leads it:
    /// the fence's verdict at `now` ([`Self::leading`]), then one read of
    /// the log ([`Log::leads`]), whose refusal says whether it is a moment
    /// or the whole lead. Where the fence is closed, the refusal names the
    /// holder's address by a second verdict of the fence at `now`
    /// ([`Self::leader`]): where that second verdict is open, the address
    /// is this replica's own.
    ///
    /// # Errors
    ///
    /// [`NotLed::Elsewhere`] where the fence is closed at `now`, and
    /// [`NotLed::Store`] where the log does not lead the term it shows.
    pub(crate) fn led(&self, now: Instant) -> Result<u64, NotLed> {
        match self.leading(now) {
            Some(term) => match self.log.leads(term) {
                Ok(()) => Ok(term),
                Err(refused) => Err(NotLed::Store(refused)),
            },
            None => Err(NotLed::Elsewhere {
                holder: self.leader(now).address,
            }),
        }
    }

    /// A receiver of the election's sights of the lease: a copy of this
    /// reader's own.
    pub(crate) fn steps(&self) -> watch::Receiver<ElectionObservation> {
        self.standing.clone()
    }

    /// The holder identity this replica elects under.
    pub(crate) fn identity(&self) -> &str {
        &self.identity
    }

    /// The fence itself, for the source of a watch of the worker rows
    /// alone: the watch asks it whether this replica still leads under the
    /// term it was admitted under.
    pub(crate) fn fence(&self) -> Arc<Fence> {
        Arc::clone(&self.fence)
    }
}

/// A word to the election thread.
enum Word {
    /// The planned stop: the fence closed, the lease released, the thread
    /// ended.
    Stop,
    /// Leave and stay in the election: a term a caller says it saw
    /// ([`ElectionNotifier::saw`]). With no term it is the leave itself, which
    /// the thread makes of a hand-over of the term held: cases alone send it.
    StepDown(Option<u64>),
    /// The planned hand-over of the lease held under this term: a replica
    /// that no longer holds the lease under it keeps on as it is.
    HandOver(u64),
}

/// The running election: its words and its end.
pub(crate) struct Running {
    /// The one strong holder of the words: a door holds them weakly, so
    /// the thread hears its channel close when this handle is dropped.
    words: Arc<mpsc::Sender<Word>>,
    /// Taken once the thread's end was seen.
    end: Option<oneshot::Receiver<()>>,
}

impl Running {
    /// The planned stop: the thread closes the fence, releases the lease
    /// and ends.
    pub(crate) fn stop(&self) {
        // A thread that is gone has nothing to release.
        let _ = self.words.send(Word::Stop);
    }

    /// The door for a word from outside the run loop: what the service's
    /// shared state holds, so a session says a gateway's term floor and a
    /// decider the planned hand-over. The door keeps nothing alive: this
    /// handle dropped, the election ends as before.
    pub(crate) fn door(&self) -> ElectionNotifier {
        ElectionNotifier {
            words: Arc::downgrade(&self.words),
        }
    }

    /// Resolves when the thread is gone, after a stop or on its own; at
    /// once on every call after that.
    pub(crate) async fn ended(&mut self) {
        if let Some(end) = self.end.as_mut() {
            // Sent after cleanup on a normal end or a caught loop panic.
            // Dropped if cleanup itself panics: gone either way.
            let _ = end.await;
            self.end = None;
        }
    }
}

/// A word to leave on `words`: the term `above` a caller says it saw, which
/// must leave one to take ([`term_said`]): one that does not is the term
/// error, and no word is said.
fn leave(words: &mpsc::Sender<Word>, above: u64) -> Result<(), ElectError> {
    term_said(Some(above))?;
    // A thread that is gone holds no lease.
    let _ = words.send(Word::StepDown(Some(above)));
    Ok(())
}

/// Tells the election a term from outside its run loop: one a caller saw,
/// or the planned hand-over. It holds the election's channel weakly, apart
/// from the run loop's handle. The handle alone keeps the channel open,
/// so a host that drops the election without a stop ends it while the
/// service's shared state still holds this notifier.
///
/// The thread counts a term said among the terms this replica saw. Where it
/// leads under a lower term, or no term is said, it closes the fence,
/// releases the lease under the highest term it saw and takes nothing for
/// one lease duration: but a replica alone over its lease takes it again at
/// its next step. A replica that leads at or above the term said keeps on as
/// it is.
/// So does one that holds no lease: where it saw the lease released under a
/// lower term, it writes that release again under the highest term it saw.
#[derive(Clone)]
pub(crate) struct ElectionNotifier {
    words: Weak<mpsc::Sender<Word>>,
}

impl ElectionNotifier {
    /// A term `above` that a caller says it saw, a gateway's floor or a
    /// standby's. A word may not spend the last term: a term whose next is
    /// not below the ceiling ([`crate::service::elector::TERM_CEILING`]) is the term
    /// error, and no word is said. Behind a handle that is gone nobody
    /// leads: the term is judged the same and said to no one.
    pub(crate) fn saw(&self, above: u64) -> Result<(), ElectError> {
        match self.words.upgrade() {
            Some(words) => leave(&words, above),
            None => term_said(Some(above)),
        }
    }

    /// The planned hand-over of the lease held under the term `under`: its
    /// holder leaves, as at a word with no term. A replica that no longer
    /// holds the lease under that term keeps on as it is: a hand-over made
    /// under one term and taken after a later take leaves that take alone.
    /// Behind a handle that is gone nobody leads, and the word is said to
    /// no one.
    pub(crate) fn hand_over(&self, under: u64) {
        if let Some(words) = self.words.upgrade() {
            // A thread that is gone holds no lease.
            let _ = words.send(Word::HandOver(under));
        }
    }
}

/// Starts the election thread over `elector`: the term goes to `store` at
/// every take, each step's sight of the lease to `standing`.
pub(crate) fn start(
    elector: Elector,
    store: Arc<MemoryStore>,
    standing: watch::Sender<ElectionObservation>,
) -> io::Result<Running> {
    let (words, heard) = mpsc::channel();
    let (end, ended) = oneshot::channel();
    thread::Builder::new()
        .name("election".to_owned())
        .spawn(move || {
            let mut elector = elector;
            let result = catch_unwind(AssertUnwindSafe(|| {
                run_loop(&mut elector, &store, &standing, &heard)
            }));
            // The elector is safe to clean up after an unwind: release closes
            // the fence before a backend call, a stale held record only makes
            // its compare-and-swap miss, and its locks do not poison.
            give_up(&mut elector);
            let failure = match result {
                Ok(seen) => {
                    publish_end(&elector, &standing, seen);
                    None
                }
                Err(failure) => Some(failure),
            };
            // The run loop is gone with its receiver when the controller
            // returned first.
            let _ = end.send(());
            if let Some(failure) = failure {
                resume_unwind(failure);
            }
        })?;
    Ok(Running {
        words: Arc::new(words),
        end: Some(ended),
    })
}

fn run_loop(
    elector: &mut Elector,
    store: &MemoryStore,
    standing: &watch::Sender<ElectionObservation>,
    words: &mpsc::Receiver<Word>,
) -> ElectionObservation {
    let period = elector.settings().renew_period;
    let deadline = elector.settings().renew_deadline;
    let mut seen = ElectionObservation::default();
    let mut following = String::new();
    let mut streak = Streak::default();
    let mut leading = false;
    let mut failing = false;
    let mut fallen = false;
    // The start of the step that stamped the fence last, while this replica
    // leads, and whether the renewal before this one landed on a closed fence.
    let mut stamped: Option<Instant> = None;
    let mut late = false;
    loop {
        // One instant a step: it stamps the fence, and the next step begins
        // one renew period after it. The loop adds no time of its own to the
        // fence's margin: a renewal lands before the fence closes while its
        // own time, after the period or after a step that took longer than
        // the period, fits in the deadline.
        let began = Instant::now();
        let ticked = elector.tick(began);
        // The standing's term is the highest this replica saw or was told,
        // read before anybody is named: the loop's one reader of a term. The
        // elector counts the term of every record a step read, whatever the
        // step then answered, an error too, and its own lead's term: no
        // holder restored under a lower term is named.
        seen.term = seen.term.max(elector.highest_term());
        // Whether this step left a lease that no other candidate can take:
        // the next step then follows at once, behind the words that came.
        let mut again = false;
        match ticked {
            Ok(tick) => {
                failing = false;
                seen.looked = true;
                match &tick {
                    TickOutcome::Acquired { .. } => {
                        stamped = Some(began);
                        late = false;
                    }
                    TickOutcome::Renewed { .. } => {
                        // One line a streak: the margin went to the store or
                        // to steps that failed, and nobody leads from the
                        // deadline until a renewal lands under it. The field
                        // is the time since the stamp before this one, in
                        // both shapes.
                        let since = stamped.map(|last| last.elapsed());
                        let closed = since.is_some_and(|since| since > deadline);
                        if closed && !late {
                            warn!(
                                identity = elector.identity(),
                                since_last_renewal = ?since.unwrap_or_default(),
                                "a renewal landed on a fence that had closed; nobody leads from the renew deadline until a renewal under the deadline lands"
                            );
                        }
                        late = closed;
                        stamped = Some(began);
                    }
                    _ => {
                        stamped = None;
                        late = false;
                    }
                }
                // A record that is gone names nobody and has a line of its
                // own. So does a lease lost to nobody, released or emptied
                // under this holder: the next step says which. The standing
                // reads either as no record.
                let nobody = match &tick {
                    TickOutcome::Gone { .. } => true,
                    TickOutcome::Lost { holder, .. } => holder.is_empty(),
                    // A replica that waits its turn follows nobody,
                    // whatever record stands.
                    TickOutcome::Waiting { .. } => true,
                    _ => false,
                };
                let named = if nobody { None } else { elector.observed() };
                match named {
                    // The lease lost its history (a Lease deleted or
                    // replaced, a lease file restored): no write stored
                    // this holder under the highest term this replica
                    // knows, and an answer never falls in term. The term
                    // stays and nobody is named, for every reader of the
                    // standing, until a look reads a term at the highest
                    // or above.
                    Some(record) if record.term < seen.term => {
                        // One line a streak.
                        if !fallen {
                            warn!(
                                identity = elector.identity(),
                                term = record.term,
                                highest = seen.term,
                                "the lease's term is under the highest this replica saw or was told; nobody is named until it is back"
                            );
                        }
                        fallen = true;
                        seen.holder.clear();
                    }
                    Some(record) => {
                        fallen = false;
                        seen.holder.clone_from(&record.holder);
                    }
                    // A step that names nobody still read a term: it was
                    // counted above with every step's.
                    None => seen.holder.clear(),
                }
                let followed = report(
                    elector,
                    &tick,
                    store,
                    &mut following,
                    &mut streak,
                    &mut leading,
                );
                if let (Some(followed), TickOutcome::Acquired { term }) = (followed, &tick) {
                    // The log is one more witness of a term. The replica
                    // leaves as at a word of a term seen, ahead of the
                    // standing: no reader is shown this leader. A followed
                    // term above the take's is said, and the elector counts
                    // it. At the take's own term none is said, as a planned
                    // hand-over says none: a holder told its own term keeps
                    // on.
                    let above = (followed > *term).then_some(followed);
                    // The next take stands above the followed term,
                    // whichever of the two is the higher. Where that
                    // leaves none to take, no lease is left. The elector
                    // is said no term at the take's own, so its rule is
                    // asked here.
                    let left = match term_said(Some(followed)) {
                        Ok(()) => leaves(elector, above, &mut leading, &mut seen),
                        Err(_) => None,
                    };
                    match left {
                        Some(left) => again = left.alone,
                        // A term that leaves none to take: the replica
                        // leads on as it stands.
                        None => warn!(
                            identity = elector.identity(),
                            term,
                            followed,
                            "the election refused the term the log followed: this replica keeps the lease, its writes stand under no term and no standby follows them until a later take"
                        ),
                    }
                }
            }
            Err(error) => {
                // One line a streak: the step repeats every renew period.
                if !failing {
                    warn!(
                        identity = elector.identity(),
                        %error,
                        "an election step failed; the fence closes at the renew deadline unless a later step lands"
                    );
                }
                failing = true;
                seen.holder.clear();
            }
        }
        standing.send_replace(seen.clone());
        // A step that took the period or more is followed at once, never
        // sooner. So is a step that left a lease no other candidate can
        // take: the time with no leader ends at this replica's own take.
        // The wait for the next step is then none: the words on the channel
        // are read in their order until none is found, and the step follows.
        // A stop or the channel's end ends the loop there, with no further
        // take and with the words behind it unread. A take that step makes
        // stands above a term counted there.
        let mut next = if again { began } else { began + period };
        let stop = loop {
            let above = match words.recv_timeout(next.saturating_duration_since(Instant::now())) {
                Err(mpsc::RecvTimeoutError::Timeout) => break false,
                Ok(Word::Stop) | Err(mpsc::RecvTimeoutError::Disconnected) => break true,
                Ok(Word::StepDown(above)) => above,
                // The planned hand-over leaves the lease of its own term, as
                // a word with no term does.
                Ok(Word::HandOver(under)) if holds_under(elector, under) => None,
                // Planned under a term this replica no longer holds the lease
                // under, it asks nothing: that lease is left already, and
                // the take since stands.
                Ok(Word::HandOver(under)) => {
                    debug!(
                        identity = elector.identity(),
                        under,
                        "a planned hand-over of a term this replica no longer holds the lease under is dropped"
                    );
                    continue;
                }
            };
            // A word the elector refused counts nothing here
            // either.
            let Some(left) = leaves(elector, above, &mut leading, &mut seen) else {
                continue;
            };
            if left.moved {
                standing.send_replace(seen.clone());
            }
            // A word to leave is taken when it comes, and the wait for the
            // next step goes on. Where the word left a lease no other
            // candidate can take, the wait ends as behind such a step: the
            // words still queued are read in their order, and the next
            // step follows the last. A stop among them ends the loop with
            // no take and with the words behind it unread, and so does
            // the channel's end.
            if left.alone {
                next = began;
            }
        };
        if stop {
            break;
        }
    }
    seen
}

/// Publishes the standing after the release at the run loop's end.
fn publish_end(
    elector: &Elector,
    standing: &watch::Sender<ElectionObservation>,
    mut seen: ElectionObservation,
) {
    seen.holder.clear();
    // The stop's release may read the lease again, and the elector counts
    // that record's term as at any step: the last standing reads it here.
    seen.term = seen.term.max(elector.highest_term());
    standing.send_replace(seen);
}

/// The steps in a row that follow nobody for one reason: its line is written
/// once a streak.
#[derive(Clone, Copy, Default, Eq, PartialEq)]
enum Streak {
    #[default]
    None,
    /// The record is gone or names no holder.
    Gone,
    /// The lease stands free and this replica waits its turn.
    Waiting,
}

/// One line a change of role, a gone streak or a waiting one; a take's term
/// to the store. Where that take started no term in the store's log: one
/// warning, and the term the log followed. The loop leaves the lease for
/// that term, or keeps it where no term is left to take above it.
fn report(
    elector: &Elector,
    tick: &TickOutcome,
    store: &MemoryStore,
    following: &mut String,
    streak: &mut Streak,
    leading: &mut bool,
) -> Option<u64> {
    let now = match tick {
        TickOutcome::Gone { .. } => Streak::Gone,
        TickOutcome::Waiting { .. } => Streak::Waiting,
        _ => Streak::None,
    };
    let was = std::mem::replace(streak, now);
    let mut followed = None;
    match tick {
        TickOutcome::Acquired { term } => {
            following.clear();
            store.set_term(*term);
            // The take started no term in the store's log.
            followed = store.followed();
            if let Some(followed) = followed {
                warn!(
                    identity = elector.identity(),
                    term,
                    followed,
                    "the take started no term in the log: the log followed a term at or above it, so another replica led that term"
                );
            }
        }
        TickOutcome::Lost { term, holder } => {
            warn!(
                identity = elector.identity(),
                term,
                holder = holder.as_str(),
                "the lease moved; the fence is closed"
            );
        }
        TickOutcome::Following { holder, term } => {
            if following != holder {
                following.clone_from(holder);
                info!(
                    identity = elector.identity(),
                    holder = holder.as_str(),
                    term,
                    "following"
                );
            }
        }
        TickOutcome::Gone { term } => {
            // One line a streak; a holder followed after it is named anew.
            following.clear();
            if was != Streak::Gone {
                warn!(
                    identity = elector.identity(),
                    term,
                    "the lease's record is gone or names no holder; nobody is named and it is taken after that record's duration"
                );
            }
        }
        TickOutcome::Waiting { term } => {
            // One line a streak, at info: a turn waited out is no anomaly.
            following.clear();
            if was != Streak::Waiting {
                info!(
                    identity = elector.identity(),
                    term,
                    "the lease stands free and this replica waits its turn; it takes the lease when its hold ends unless another replica takes it first"
                );
            }
        }
        TickOutcome::Renewed { .. } | TickOutcome::Raced => {
            debug!(identity = elector.identity(), ?tick, "election step");
        }
    }
    // One line when the fence opens: at the take, or at the first step
    // after the opening of a take of a record with no holder to wait for,
    // one renew deadline after that take's write. The elector says which
    // take it made: after a slow step the fence reads closed here as well,
    // with no wait to name, and the next renewal leads.
    match tick {
        // A take that started no term says no line of a lead: the loop
        // leaves it in this step. Where the loop keeps the lease, the next
        // renewal says it.
        TickOutcome::Acquired { .. } if followed.is_some() => {}
        TickOutcome::Acquired { term } | TickOutcome::Renewed { term } => {
            if !*leading {
                *leading = elector.fence().leader_term(Instant::now()).is_some();
                if *leading {
                    info!(identity = elector.identity(), term, "leading");
                } else if matches!(tick, TickOutcome::Acquired { .. }) && elector.waits_to_lead() {
                    info!(
                        identity = elector.identity(),
                        term,
                        "took a lease with no holder to wait for; leading one renew deadline after the take"
                    );
                }
            }
        }
        _ => *leading = false,
    }
    followed
}

/// What a leave of the thread's came to ([`leaves`]).
#[derive(Clone, Copy)]
struct Left {
    /// Whether the standing changed: a lease was left, or a term was counted
    /// above the one its holder is named under.
    moved: bool,
    /// Whether this replica left a lease that no other candidate can take.
    alone: bool,
}

/// A leave the thread makes ([`step_down`]): at a word, or of a take that
/// started no term in the store's log. The fence closed where a lease was
/// left: the take that opens it again says so, as the first take did. A
/// lease this replica left names nobody in `seen`, and so does a term above
/// the one the holder is named under: the elector counted the term said.
/// `None` where the elector refused the term: nothing was counted.
fn leaves(
    elector: &mut Elector,
    above: Option<u64>,
    leading: &mut bool,
    seen: &mut ElectionObservation,
) -> Option<Left> {
    let left = step_down(elector, above)?;
    if left {
        *leading = false;
    }
    let highest = elector.highest_term();
    let moved = left || highest > seen.term;
    if moved {
        seen.holder.clear();
        seen.term = seen.term.max(highest);
    }
    Some(Left {
        moved,
        alone: left && elector.is_alone(),
    })
}

/// Whether this replica holds the lease under the term `under`, the term a
/// planned hand-over was made under.
fn holds_under(elector: &Elector, under: u64) -> bool {
    elector.is_holder() && elector.fence().term() == under
}

/// A word to leave, taken by the election thread. The elector counts the
/// term; a holder under a lower term, or one told no term, closes its fence,
/// releases the lease and holds its takes back. `Some(true)` when this
/// replica held the lease and left it: a release that failed is tried again
/// at every step, in place of a renewal. `Some(false)` when it left none: a
/// replica that saw the lease released under a term below the highest its
/// run saw writes that release again under that highest, and a write that
/// failed is logged and not tried again. `None` when the elector refused
/// the word, a term that leaves none to take: nothing was counted.
fn step_down(elector: &mut Elector, above: Option<u64>) -> Option<bool> {
    match elector.step_down(above, Instant::now()) {
        Ok(true) => {
            if elector.is_alone() {
                info!(
                    identity = elector.identity(),
                    above,
                    "stepped down; the lease is released and this replica, alone over its lease, takes it again at its next step"
                );
            } else {
                info!(
                    identity = elector.identity(),
                    above,
                    "stepped down; the lease is released and this replica takes nothing for one lease duration"
                );
            }
            Some(true)
        }
        Ok(false) => Some(false),
        // The elector's own door: nothing was counted and nothing released.
        Err(error @ ElectError::Term(_)) => {
            warn!(
                identity = elector.identity(),
                above,
                %error,
                "a word to leave said a term that leaves none to take; the word is refused"
            );
            None
        }
        Err(error) if elector.is_holder() => {
            warn!(
                identity = elector.identity(),
                above,
                %error,
                "stepped down; the release failed and is tried again at the next step"
            );
            Some(true)
        }
        // No lease is held here: the term was counted, and its write over
        // a release this replica saw failed.
        Err(error) => {
            warn!(
                identity = elector.identity(),
                above,
                %error,
                "a term said over a released lease was counted and not written; a take below it is left again at its gateway's word"
            );
            Some(false)
        }
    }
}

/// The planned stop: the elector closes the fence, then rewrites the record
/// released, so the next candidate to look takes the lease at once, or
/// after the wait the take still owed. The first
/// release follows the loop's last step at once: a stop does not wait for
/// the period. A release that fails is tried again, a bounded number of
/// times, and no sooner than one renew period after the attempt before it
/// began, as the loop spaces its steps; an attempt that took the period is
/// followed at once. The fence is closed either way and the lease then
/// expires by its duration.
fn give_up(elector: &mut Elector) {
    let period = elector.settings().renew_period;
    // A holder whose release finds the record moved has none to release,
    // and says what a step that read the loss says. It lost the lease after
    // its last step, or an earlier attempt of this stop stored the release
    // with its answer lost and the next candidate took the lease since.
    let held = elector.is_holder().then(|| elector.fence().term());
    for attempt in 1..=RELEASE_ATTEMPTS {
        let began = Instant::now();
        match elector.release() {
            Ok(true) => {
                info!(identity = elector.identity(), "the lease is released");
                return;
            }
            Ok(false) => {
                if let Some(term) = held {
                    warn!(
                        identity = elector.identity(),
                        term, "the lease moved; the fence is closed"
                    );
                }
                return;
            }
            Err(error) => {
                warn!(identity = elector.identity(), attempt, %error, "the release failed");
            }
        }
        if attempt < RELEASE_ATTEMPTS {
            thread::sleep(period.saturating_sub(began.elapsed()));
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/service/controller/election/tests.rs"]
pub(super) mod tests;
