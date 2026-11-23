//! The election wired into the run loop (design 5.3): where the lease lives,
//! the identity this replica holds it under, the thread that steps the
//! elector, and the standing the `Leader` call answers from on a replica
//! that does not lead.
//!
//! One path in every mode: a controller with no shared lease elects itself
//! over a record in its own process, so the [`Fence`](crate::service::elector::Fence)
//! a shard thread reads is the same object whether one replica runs or
//! three. The fence is the one judge of "leader": readiness, the state gauge
//! and the `Leader` call read it at the instant they answer, and nothing is
//! told its role by a message that can be late.
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
//! takes it again at once: its next step follows the word). The thread goes on
//! stepping. A decider says the planned hand-over itself where its term's lease
//! ids pass their mark: the next take brings a term whose ids start again. The
//! hand-over names that term, and the thread drops one whose term this replica
//! no longer holds the lease under: that lease is left already.

use std::{
    fmt, io,
    net::{IpAddr, SocketAddr},
    panic::{catch_unwind, resume_unwind, AssertUnwindSafe},
    path::{Path, PathBuf},
    sync::{mpsc, Arc, Weak},
    thread,
    time::Instant,
};

use tokio::sync::{oneshot, watch};
use tracing::{debug, info, warn};

use super::config::ConfigError;
#[cfg(feature = "kubernetes")]
use crate::service::elector::KubernetesLease;
use crate::{
    service::elector::{
        term_said, ElectError, ElectionSettings, Elector, FileLease, LeaseBackend, TickOutcome,
    },
    store::{LeasesStore, MemoryStore},
};

/// How many times a planned stop writes the release before it leaves the
/// lease to expire by its duration.
const RELEASE_ATTEMPTS: u32 = 3;

/// Where the lease lives.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
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

/// The election's configuration (design 5.3).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
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

impl ElectionConfig {
    /// Refuses what no bind could cure: settings out of order, a lease with
    /// no name or an empty namespace, and an advertised address that is
    /// malformed, that is the wildcard in any spelling (a client that dials
    /// it reaches its own host), or that ends in a number and is no address
    /// in its standard form (a resolver reads such a host as an address).
    ///
    /// A multicast or broadcast address, like a name nobody holds, is left
    /// to the dial; a loopback address is taken as written, right where a
    /// client shares the host.
    pub(crate) fn validate(&self, gateway: SocketAddr) -> Result<(), ConfigError> {
        if let Err(ElectError::Settings(why)) = self.settings.validate() {
            return Err(ConfigError::ElectionSettings(why));
        }
        match &self.advertise {
            // The port is a stand-in: the form alone is judged before the
            // bind names the real one.
            Some(advertise) => {
                dialable(advertise, 1)?;
            }
            None if self.lease.is_shared() && is_wildcard(gateway.ip()) => {
                return Err(ConfigError::AdvertiseAddressRequired);
            }
            None => {}
        }
        match &self.lease {
            LeasePlace::Process => {}
            LeasePlace::File(path) => {
                if path.as_os_str().is_empty() {
                    return Err(ConfigError::EmptyLeaseName);
                }
            }
            #[cfg(feature = "kubernetes")]
            LeasePlace::Kubernetes { namespace, name } => {
                if name.is_empty() {
                    return Err(ConfigError::EmptyLeaseName);
                }
                if namespace.as_ref().is_some_and(String::is_empty) {
                    return Err(ConfigError::EmptyLeaseNamespace);
                }
                // The object stores whole seconds: a fraction would be
                // refused at the first write, after the listeners serve.
                if self.settings.duration.subsec_nanos() != 0 {
                    return Err(ConfigError::LeaseDurationNotWholeSeconds(
                        self.settings.duration,
                    ));
                }
            }
        }
        Ok(())
    }

    /// This replica's dialable address once the gateway listener is bound
    /// at `gateway`; `None` when no client could be told one (a lone
    /// controller behind a wildcard listener).
    pub(crate) fn address(&self, gateway: SocketAddr) -> Result<Option<String>, ConfigError> {
        match &self.advertise {
            Some(advertise) => dialable(advertise, gateway.port()).map(Some),
            None if is_wildcard(gateway.ip()) => Ok(None),
            None => Ok(Some(gateway.to_string())),
        }
    }
}

/// The dialable `host:port` of an advertised address: an address or a name,
/// alone (it takes `port`) or with its own port. An IPv6 address alone is
/// bracketed here, so a deployment passes a pod's address as it reads it.
/// An address has one spelling here, the standard one: the wildcard is
/// refused by name, and so is any other host that ends in a number, which a
/// resolver would read as an address of its own making. Every other name is
/// taken as given.
fn dialable(advertise: &str, port: u16) -> Result<String, ConfigError> {
    let malformed = || ConfigError::MalformedAdvertiseAddress(advertise.to_owned());
    let dialed = |ip: IpAddr| {
        if is_wildcard(ip) {
            Err(ConfigError::WildcardAdvertiseAddress(advertise.to_owned()))
        } else {
            Ok(ip)
        }
    };
    let bare = advertise
        .strip_prefix('[')
        .and_then(|rest| rest.strip_suffix(']'))
        .unwrap_or(advertise);
    if let Ok(ip) = bare.parse::<IpAddr>() {
        return Ok(SocketAddr::new(dialed(ip)?, port).to_string());
    }
    if let Ok(addr) = advertise.parse::<SocketAddr>() {
        dialed(addr.ip())?;
        return if addr.port() == 0 {
            Err(malformed())
        } else {
            Ok(addr.to_string())
        };
    }
    let (host, port) = match advertise.rsplit_once(':') {
        Some((host, given)) => (
            host,
            given
                .parse::<u16>()
                .ok()
                .filter(|port| *port != 0)
                .ok_or_else(malformed)?,
        ),
        None => (advertise, port),
    };
    let named = !host.is_empty()
        && host
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_'));
    if !named {
        return Err(malformed());
    }
    if ends_in_a_number(host) {
        return Err(ConfigError::NumericAdvertiseHost(advertise.to_owned()));
    }
    Ok(format!("{host}:{port}"))
}

/// Whether `ip` is the wildcard: the unspecified address of either family,
/// or the IPv4 one written mapped into IPv6, which a listener binds as the
/// wildcard too. A client that dials it reaches its own host.
fn is_wildcard(ip: IpAddr) -> bool {
    ip.to_canonical().is_unspecified()
}

/// Whether a host's last label is a number: digits alone, or `0x` in either
/// case and then hexadecimal digits, none included; one closing dot is
/// dropped first. This is the URL standard's check of a host that ends in a
/// number. A resolver reads such a host as an address, whatever the labels
/// before it, and not always the one that was meant: `0` is the wildcard,
/// `010.0.0.1` is `8.0.0.1`, `8080` is `0.0.31.144`.
fn ends_in_a_number(host: &str) -> bool {
    let last = host
        .strip_suffix('.')
        .unwrap_or(host)
        .rsplit('.')
        .next()
        .unwrap_or_default();
    let hexadecimal = last
        .strip_prefix("0x")
        .or_else(|| last.strip_prefix("0X"))
        .is_some_and(|digits| digits.bytes().all(|b| b.is_ascii_hexdigit()));
    hexadecimal || (!last.is_empty() && last.bytes().all(|b| b.is_ascii_digit()))
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
pub(crate) fn split_identity(identity: &str) -> (&str, &str) {
    identity.split_once('@').unwrap_or((identity, ""))
}

/// The lease could not be opened.
#[derive(Debug, thiserror::Error)]
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
            let mut config = kube::Config::infer()
                .await
                .map_err(|error| LeaseError::Client(error.to_string()))?;
            // A stalled round trip returns before the fence it could have
            // renewed closes, so the thread's next step is never behind a
            // call that outlives the lease.
            let deadline = election.settings.renew_deadline;
            config.connect_timeout = Some(deadline);
            config.read_timeout = Some(deadline);
            config.write_timeout = Some(deadline);
            let namespace = namespace
                .clone()
                .unwrap_or_else(|| config.default_namespace.clone());
            let client = kube::Client::try_from(config)
                .map_err(|error| LeaseError::Client(error.to_string()))?;
            Box::new(KubernetesLease::new(
                client,
                &namespace,
                name,
                tokio::runtime::Handle::current(),
            ))
        }
    };
    Ok(Elector::new(backend, identity, election.settings)?)
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
/// once. A replica that leads at or above the term said keeps on as it is.
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

/// A door no thread listens behind, for a case of the door's caller. It
/// stands in for the handle too: the door says while this lives.
#[cfg(test)]
pub(crate) struct Said {
    /// The handle's part: the door holds these weakly.
    _words: Arc<mpsc::Sender<Word>>,
    heard: mpsc::Receiver<Word>,
}

#[cfg(test)]
impl Said {
    /// The door, and what is said through it.
    pub(crate) fn door() -> (ElectionNotifier, Self) {
        let (words, heard) = mpsc::channel();
        let words = Arc::new(words);
        let door = ElectionNotifier {
            words: Arc::downgrade(&words),
        };
        let said = Self {
            _words: words,
            heard,
        };
        (door, said)
    }

    /// The terms said since the last call, in order.
    pub(crate) fn terms(&self) -> Vec<u64> {
        self.heard
            .try_iter()
            .filter_map(|word| match word {
                Word::StepDown(above) => above,
                Word::HandOver(_) | Word::Stop => None,
            })
            .collect()
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
                report(
                    elector,
                    &tick,
                    store,
                    &mut following,
                    &mut streak,
                    &mut leading,
                );
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
        // sooner. A word to leave is taken when it comes, and the wait for
        // the next step goes on: but for a replica alone over its lease
        // that the word left with no leader, whose next step comes now.
        let next = began + period;
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
            let Some(left) = step_down(elector, above) else {
                continue;
            };
            // The fence closed at the word: the take that opens it
            // again says so, as the first take did.
            if left {
                leading = false;
            }
            // A lease this replica left names nobody, and so does a
            // term above the one the holder is named under: the
            // elector counted the term said.
            let highest = elector.highest_term();
            if left || highest > seen.term {
                seen.holder.clear();
                seen.term = seen.term.max(highest);
                standing.send_replace(seen.clone());
            }
            // No other candidate can take the lease this replica left: the
            // time with no leader ends at its own take.
            if left && elector.is_alone() {
                break false;
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

/// Runs the loop through its normal cleanup for cases that call it directly.
#[cfg(test)]
fn run(
    mut elector: Elector,
    store: &MemoryStore,
    standing: &watch::Sender<ElectionObservation>,
    words: &mpsc::Receiver<Word>,
) {
    let seen = run_loop(&mut elector, store, standing, words);
    give_up(&mut elector);
    publish_end(&elector, standing, seen);
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
/// to the store, and one warning where that take started no term in the
/// store's log.
fn report(
    elector: &Elector,
    tick: &TickOutcome,
    store: &MemoryStore,
    following: &mut String,
    streak: &mut Streak,
    leading: &mut bool,
) {
    let now = match tick {
        TickOutcome::Gone { .. } => Streak::Gone,
        TickOutcome::Waiting { .. } => Streak::Waiting,
        _ => Streak::None,
    };
    let was = std::mem::replace(streak, now);
    match tick {
        TickOutcome::Acquired { term } => {
            following.clear();
            store.set_term(*term);
            // The take started no term in the store's log.
            if let Some(followed) = store.followed() {
                warn!(
                    identity = elector.identity(),
                    term,
                    followed,
                    "the take started no term in the log: the log followed a term at or above it, so its writes stand under no term and no standby follows them until a later take"
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
                    "stepped down; the lease is released and this replica, alone over its lease, takes it again at once"
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
mod tests {
    use std::{
        collections::VecDeque,
        io::Write,
        net::{Ipv4Addr, Ipv6Addr, SocketAddr},
        path::PathBuf,
        sync::{
            atomic::{AtomicBool, AtomicU32, Ordering},
            mpsc, Arc,
        },
        thread,
        time::{Duration, Instant},
    };

    use parking_lot::Mutex;
    use tokio::sync::watch;

    use super::{
        dialable, elector, identity, run, split_identity, start, ElectionConfig,
        ElectionObservation, LeaseError, LeasePlace, Running, Said, Word,
    };
    use crate::{
        service::{
            controller::{config::ConfigError, replication::journaled_from},
            elector::{
                ConditionalWriteOutcome, ElectError, ElectionSettings, Elector, Fence,
                LeaseBackend, LeaseRecord, MemoryLease, TickOutcome, TERM_CEILING,
            },
        },
        store::{LeasesStore, MemoryStore, Position},
    };

    #[test]
    fn an_advertised_host_takes_the_listeners_port_and_a_host_with_a_port_keeps_its_own() {
        for (advertise, dialed) in [
            ("10.0.0.7", "10.0.0.7:30100"),
            ("10.0.0.7:31000", "10.0.0.7:31000"),
            ("fd00::7", "[fd00::7]:30100"),
            ("[fd00::7]", "[fd00::7]:30100"),
            ("[fd00::7]:31000", "[fd00::7]:31000"),
            ("controller-0.policy.svc", "controller-0.policy.svc:30100"),
            (
                "controller-0.policy.svc:31000",
                "controller-0.policy.svc:31000",
            ),
        ] {
            assert_eq!(
                dialable(advertise, 30100).as_deref(),
                Ok(dialed),
                "{advertise}"
            );
        }
        assert_eq!(
            dialable("10.0.0.7:0", 30100),
            Err(ConfigError::MalformedAdvertiseAddress(
                "10.0.0.7:0".to_owned()
            ))
        );
    }

    #[test]
    fn the_wildcard_is_refused_in_every_spelling() {
        // A client that dials the wildcard reaches its own host: alone or
        // with a port, in either family, bracketed or bare, and as the IPv4
        // one mapped into IPv6.
        for wildcard in [
            "0.0.0.0",
            "0.0.0.0:31000",
            "::",
            "[::]",
            "[::]:31000",
            "::ffff:0.0.0.0",
            "[::ffff:0.0.0.0]",
            "[::ffff:0.0.0.0]:31000",
        ] {
            assert_eq!(
                dialable(wildcard, 30100),
                Err(ConfigError::WildcardAdvertiseAddress(wildcard.to_owned())),
                "{wildcard}"
            );
        }
        // Every other address is taken as written: a loopback one, a mapped
        // one, and a multicast or the broadcast one, the dial's to fail.
        for (advertise, dialed) in [
            ("0.0.0.1", "0.0.0.1:30100"),
            ("127.0.0.1", "127.0.0.1:30100"),
            ("127.0.0.1:31000", "127.0.0.1:31000"),
            ("::1", "[::1]:30100"),
            ("::ffff:10.0.0.7", "[::ffff:10.0.0.7]:30100"),
            ("224.0.0.1", "224.0.0.1:30100"),
            ("224.0.0.1:31000", "224.0.0.1:31000"),
            ("255.255.255.255", "255.255.255.255:30100"),
            ("[ff02::1]:31000", "[ff02::1]:31000"),
        ] {
            assert_eq!(
                dialable(advertise, 30100).as_deref(),
                Ok(dialed),
                "{advertise}"
            );
        }
    }

    #[test]
    fn a_host_that_ends_in_a_number_is_refused_alone_and_with_a_port() {
        // What a resolver reads as an address though the standard parser
        // does not, and not always the one that was meant: `0` and `0x0`
        // are the wildcard, `010.0.0.1` is `8.0.0.1`, `8080` is
        // `0.0.31.144`.
        for numeric in [
            "0",
            "00",
            "0.0",
            "0.0.0",
            "0x0",
            "0X0",
            "0x0.0",
            "00.0.0.0",
            "000.000.000.000",
            "4294967295",
            "0xffffffff",
            "0xe0000001",
            "8080",
            "1.2.3",
            "010.0.0.1",
            "0.",
            "0x",
            "1.",
            "999.1.1.1",
            "example.123",
            "0xcafe",
        ] {
            for advertise in [numeric.to_owned(), format!("{numeric}:31000")] {
                assert_eq!(
                    dialable(&advertise, 30100),
                    Err(ConfigError::NumericAdvertiseHost(advertise.clone())),
                    "{advertise}"
                );
            }
        }
        // A number inside a name, or a last label that only opens like one,
        // is a name's and taken as given; two closing dots leave an empty
        // last label, which is no number.
        for name in [
            "pod-0",
            "a1",
            "10.example",
            "0xcafe.example",
            "0xzz",
            "node7.cluster.local",
            "host.example.",
            "1..",
        ] {
            assert_eq!(dialable(name, 30100), Ok(format!("{name}:30100")), "{name}");
            let with_a_port = format!("{name}:31000");
            assert_eq!(
                dialable(&with_a_port, 30100).as_deref(),
                Ok(with_a_port.as_str()),
                "{with_a_port}"
            );
        }
    }

    #[test]
    fn the_address_defaults_to_a_listener_that_binds_one() {
        let bound = |ip: std::net::IpAddr| SocketAddr::new(ip, 40123);
        let mut election = ElectionConfig::default();
        assert_eq!(
            election.address(bound(Ipv4Addr::LOCALHOST.into())),
            Ok(Some("127.0.0.1:40123".to_owned()))
        );
        assert_eq!(
            election.address(bound(Ipv6Addr::LOCALHOST.into())),
            Ok(Some("[::1]:40123".to_owned()))
        );
        // Behind a wildcard listener no client could be told an address:
        // either family's, and the IPv4 one mapped into IPv6.
        for wildcard in [
            Ipv4Addr::UNSPECIFIED.into(),
            Ipv6Addr::UNSPECIFIED.into(),
            Ipv4Addr::UNSPECIFIED.to_ipv6_mapped().into(),
        ] {
            assert_eq!(election.address(bound(wildcard)), Ok(None), "{wildcard}");
        }
        // A mapped address that is bound is told as the listener reads it.
        assert_eq!(
            election.address(bound(Ipv4Addr::LOCALHOST.to_ipv6_mapped().into())),
            Ok(Some("[::ffff:127.0.0.1]:40123".to_owned()))
        );

        // An advertised host takes the port the listener bound.
        election.advertise = Some("10.0.0.7".to_owned());
        assert_eq!(
            election.address(bound(Ipv4Addr::UNSPECIFIED.into())),
            Ok(Some("10.0.0.7:40123".to_owned()))
        );
    }

    #[test]
    fn the_identity_is_the_id_at_the_address_and_splits_back() {
        let named = identity("controller-0", Some("10.0.0.7:30100"));
        assert_eq!(named, "controller-0@10.0.0.7:30100");
        assert_eq!(split_identity(&named), ("controller-0", "10.0.0.7:30100"));
        let v6 = identity("controller-0", Some("[fd00::7]:30100"));
        assert_eq!(split_identity(&v6), ("controller-0", "[fd00::7]:30100"));
        let alone = identity("controller-0", None);
        assert_eq!(alone, "controller-0");
        assert_eq!(split_identity(&alone), ("controller-0", ""));
    }

    #[test]
    fn a_lease_names_its_place() {
        assert_eq!(
            LeasePlace::Process.to_string(),
            "this process (one replica)"
        );
        assert!(!LeasePlace::Process.is_shared());
        let file = LeasePlace::File(PathBuf::from("/run/smg/lease"));
        assert_eq!(file.to_string(), "file /run/smg/lease");
        assert!(file.is_shared());
        #[cfg(feature = "kubernetes")]
        {
            let object = |namespace: Option<&str>| LeasePlace::Kubernetes {
                namespace: namespace.map(str::to_owned),
                name: "smg-policy".to_owned(),
            };
            assert_eq!(
                object(Some("policy")).to_string(),
                "Lease policy/smg-policy"
            );
            assert_eq!(object(None).to_string(), "Lease smg-policy");
            assert!(object(None).is_shared());
        }
    }

    #[tokio::test]
    async fn the_process_place_makes_an_elector_alone_that_leads_at_its_take() {
        let mut alone = elector(
            &ElectionConfig::default(),
            "controller-0@127.0.0.1:1".to_owned(),
        )
        .await
        .expect("the process place needs nothing");
        let now = Instant::now();
        alone.tick(now).expect("the take");
        assert_eq!(alone.fence().leader_term(now), Some(1));
    }

    #[tokio::test]
    async fn a_lease_file_needs_its_directory_and_writes_nothing_before_the_first_step() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let at = |path: PathBuf| ElectionConfig {
            lease: LeasePlace::File(path),
            ..ElectionConfig::default()
        };
        let candidate = "controller-0@127.0.0.1:1".to_owned();

        let missing = dir.path().join("missing");
        match elector(&at(missing.join("lease")), candidate.clone()).await {
            Err(LeaseError::Directory { directory, .. }) => assert_eq!(directory, missing),
            other => panic!("a missing directory was not refused: {other:?}"),
        }

        let file = dir.path().join("file");
        std::fs::write(&file, b"").expect("a file");
        match elector(&at(file.join("lease")), candidate.clone()).await {
            Err(LeaseError::NotADirectory(directory)) => assert_eq!(directory, file),
            other => panic!("a file for a directory was not refused: {other:?}"),
        }

        let lease = dir.path().join("lease");
        let opened = elector(&at(lease.clone()), candidate.clone())
            .await
            .expect("the directory is there");
        assert!(!opened.is_holder());
        assert!(!lease.exists(), "nothing is written before the first step");

        // A file named without a directory lives where the process runs.
        let alone = PathBuf::from("a-lease-named-without-a-directory");
        let opened = elector(&at(alone.clone()), candidate)
            .await
            .expect("the process runs in a directory");
        assert!(!opened.is_holder());
        assert!(!alone.exists(), "nothing is written before the first step");
    }

    /// A host that runs the election in its own process cancels it by
    /// dropping it: the thread hears its stop channel close, gives the
    /// lease up and ends, as at a stop.
    #[test]
    fn an_election_dropped_without_a_stop_gives_the_lease_up_and_ends() {
        let mut lease = MemoryLease::new();
        let settings = ElectionSettings {
            duration: Duration::from_millis(600),
            renew_period: Duration::from_millis(50),
            renew_deadline: Duration::from_millis(400),
        };
        let candidate = Elector::new(Box::new(lease.clone()), "a@127.0.0.1:1", settings)
            .expect("the settings are in order");
        let fence = candidate.fence();
        let (standing, _stood) = watch::channel(ElectionObservation::default());
        let mut running =
            start(candidate, Arc::new(MemoryStore::new()), standing).expect("a thread");
        until("the take", || fence.leader_term(Instant::now()) == Some(1));

        let mut end = running.end.take().expect("the end was not seen yet");
        drop(running);
        until("the thread's end", || end.try_recv().is_ok());
        assert_eq!(fence.leader_term(Instant::now()), None);
        let stored = lease.read().expect("a read");
        assert!(
            stored.is_some_and(|record| record.holder.is_empty()),
            "the lease is free"
        );
    }

    /// A lease that can be armed to panic on a read after it stores a take.
    struct PanicsAfterTake {
        lease: MemoryLease,
        armed: Arc<AtomicBool>,
        took: bool,
    }

    impl LeaseBackend for PanicsAfterTake {
        fn read(&mut self) -> Result<Option<LeaseRecord>, ElectError> {
            assert!(
                !self.took || !self.armed.load(Ordering::SeqCst),
                "the read after the take panics"
            );
            self.lease.read()
        }

        fn write(
            &mut self,
            expected: Option<&LeaseRecord>,
            record: &LeaseRecord,
        ) -> Result<ConditionalWriteOutcome, ElectError> {
            let written = self.lease.write(expected, record)?;
            if written == ConditionalWriteOutcome::Done && !record.holder.is_empty() {
                self.took = true;
            }
            Ok(written)
        }

        fn place(&self) -> String {
            self.lease.place()
        }
    }

    #[test]
    fn a_panicking_step_closes_its_fence_and_releases_before_the_thread_ends() {
        let mut lease = MemoryLease::new();
        let armed = Arc::new(AtomicBool::new(false));
        let released = LeaseRecord {
            term: 1,
            ..LeaseRecord::default()
        };
        assert_eq!(
            lease.write(None, &released).expect("a write"),
            ConditionalWriteOutcome::Done
        );
        let settings = ElectionSettings {
            duration: Duration::from_millis(600),
            renew_period: Duration::from_millis(50),
            renew_deadline: Duration::from_millis(400),
        };
        let candidate = Elector::new(
            Box::new(PanicsAfterTake {
                lease: lease.clone(),
                armed: armed.clone(),
                took: false,
            }),
            "a@127.0.0.1:1",
            settings,
        )
        .expect("the settings are in order");
        let fence = candidate.fence();
        let (standing, _stood) = watch::channel(ElectionObservation::default());
        let running = start(candidate, Arc::new(MemoryStore::new()), standing).expect("a thread");
        thread::sleep(Duration::from_millis(100));
        until("the take", || fence.leader_term(Instant::now()) == Some(2));
        armed.store(true, Ordering::SeqCst);

        ended(running);
        assert_eq!(fence.leader_term(Instant::now()), None);
        let stored = lease.read().expect("a read");
        assert!(
            stored.is_some_and(|record| record.holder.is_empty()),
            "the lease is free"
        );
    }

    /// A door given out keeps no election alive: the handle alone holds the
    /// words. Dropped while the service's shared state still holds the
    /// door, the election ends as it does with no door, and the door then
    /// judges a term and says it to no one.
    #[test]
    fn an_election_dropped_while_a_door_is_held_gives_the_lease_up_and_ends() {
        let mut lease = MemoryLease::new();
        let settings = ElectionSettings {
            duration: Duration::from_millis(600),
            renew_period: Duration::from_millis(50),
            renew_deadline: Duration::from_millis(400),
        };
        let candidate = Elector::new(Box::new(lease.clone()), "a@127.0.0.1:1", settings)
            .expect("the settings are in order");
        let fence = candidate.fence();
        let (standing, _stood) = watch::channel(ElectionObservation::default());
        let mut running =
            start(candidate, Arc::new(MemoryStore::new()), standing).expect("a thread");
        until("the take", || fence.leader_term(Instant::now()) == Some(1));

        let door = running.door();
        let mut end = running.end.take().expect("the end was not seen yet");
        drop(running);
        until("the thread's end", || end.try_recv().is_ok());
        assert_eq!(fence.leader_term(Instant::now()), None);
        let stored = lease.read().expect("a read");
        assert!(
            stored.is_some_and(|record| record.holder.is_empty()),
            "the lease is free"
        );
        door.saw(7).expect("a term that leaves one to take");
        let refused = door
            .saw(TERM_CEILING - 1)
            .expect_err("no term is left to take");
        assert!(
            matches!(refused, ElectError::Term(next) if next == TERM_CEILING),
            "{refused}"
        );
    }

    /// Polls `met` every few milliseconds. No instant has two open fences:
    /// at every poll the first fence is read, then the second, then the
    /// first again, each at its own instant, and the first open under one
    /// term on both sides of an open second is two open fences at the
    /// second's instant. One instant for both reads would call a hand-over
    /// between them two leaders: the taker's stamp lies after that instant.
    fn wait_for(what: &str, fences: [&Fence; 2], met: impl Fn() -> bool) {
        let started = Instant::now();
        loop {
            let before = fences[0].leader_term(Instant::now());
            let between = fences[1].leader_term(Instant::now());
            let after = fences[0].leader_term(Instant::now());
            assert!(
                before.is_none() || between.is_none() || before != after,
                "two open fences while waiting for {what}"
            );
            if met() {
                return;
            }
            assert!(
                started.elapsed() < Duration::from_secs(20),
                "{what} did not happen in 20 s"
            );
            thread::sleep(Duration::from_millis(2));
        }
    }

    /// Polls until `met`, bounded.
    fn until(what: &str, mut met: impl FnMut() -> bool) {
        let started = Instant::now();
        while !met() {
            assert!(
                started.elapsed() < Duration::from_secs(20),
                "{what} did not happen in 20 s"
            );
            thread::sleep(Duration::from_millis(2));
        }
    }

    /// Waits, bounded, for an election thread to signal that it is gone.
    fn ended(running: Running) {
        let mut end = running.end.expect("the end was not seen yet");
        let started = Instant::now();
        while end.try_recv().is_err() {
            assert!(
                started.elapsed() < Duration::from_secs(20),
                "the election thread did not signal its end in 20 s"
            );
            thread::sleep(Duration::from_millis(2));
        }
    }

    /// How many times a case that bounds the loop on the real clock is made.
    /// A try whose bound fails where its witness says the host held this
    /// process says so and is made again. No witness passes a run: the last
    /// try stands whatever its witness says, so a loop that fails by itself
    /// fails each try, and the case.
    const TRIES: u32 = 3;

    /// A bound on the real clock that did not hold: the one failure a hold
    /// of the host explains, so the one [`unheld`] makes a try again for.
    struct Late(String);

    /// `assert!` for a bound on the real clock: the try unwinds as [`Late`]
    /// with no line of the panic hook's, and [`unheld`] says its words.
    macro_rules! bound {
        ($met:expr, $($said:tt)+) => {
            if !$met {
                let said = format!("{}:{}: {}", file!(), line!(), format_args!($($said)+));
                std::panic::resume_unwind(Box::new(Late(said)));
            }
        };
    }

    /// A witness of the host's holds beside such a case: a thread that naps
    /// 5 ms at a time and keeps the longest it overslept. A host that holds
    /// this process holds its witness too.
    struct Witness {
        end: mpsc::Sender<()>,
        naps: thread::JoinHandle<Duration>,
    }

    impl Witness {
        fn start() -> Self {
            let (end, ended) = mpsc::channel();
            // The first nap begins here, before the case can run, and each
            // nap begins where the one before it ended: no instant of the
            // case is outside a nap, the witness thread's own start with it.
            let mut began = Instant::now();
            let naps = thread::spawn(move || {
                let nap = Duration::from_millis(5);
                let mut longest = Duration::ZERO;
                loop {
                    let over = ended.recv_timeout(nap) != Err(mpsc::RecvTimeoutError::Timeout);
                    let woke = Instant::now();
                    // The nap its end cut short counts too: a case the host
                    // held can end before its witness wakes.
                    longest = longest.max(woke.duration_since(began).saturating_sub(nap));
                    if over {
                        return longest;
                    }
                    began = woke;
                }
            });
            Self { end, naps }
        }

        /// Ends the witness: the longest hold it saw.
        fn held(self) -> Duration {
            drop(self.end);
            self.naps.join().expect("the witness returns")
        }
    }

    /// Runs `case`, whose bounds on the real clock ([`bound!`]) leave the
    /// loop `room`. A host that holds this process longer breaks such a
    /// bound with no fault of the loop's, so a try whose bound fails where
    /// the witness saw a hold of half the room is made again, [`TRIES`] at
    /// most. No hold explains any other failure: it fails the case at once.
    fn unheld(name: &str, room: Duration, case: fn()) {
        for tried in 1..=TRIES {
            let witness = Witness::start();
            let failed = std::panic::catch_unwind(case);
            let held = witness.held();
            let Err(failure) = failed else {
                return;
            };
            let late = match failure.downcast::<Late>() {
                Ok(late) => late,
                Err(failure) => std::panic::resume_unwind(failure),
            };
            assert!(held >= room / 2 && tried < TRIES, "{}", late.0);
            writeln!(
                std::io::stdout(),
                "{name}: made again in this run: the host held the test {held:?}: {}",
                late.0
            )
            .expect("the test's output");
        }
    }

    /// A lease whose reads fail while `away` is set.
    struct Away {
        lease: MemoryLease,
        away: Arc<AtomicBool>,
    }

    impl LeaseBackend for Away {
        fn read(&mut self) -> Result<Option<LeaseRecord>, ElectError> {
            if self.away.load(Ordering::SeqCst) {
                return Err(ElectError::Io {
                    place: self.place(),
                    source: std::io::Error::other("the store is away"),
                });
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
            self.lease.place()
        }
    }

    #[test]
    fn a_look_that_fails_names_nobody_until_one_lands() {
        unheld(
            "a look that fails",
            Duration::from_millis(600),
            a_failed_look_names_nobody,
        );
    }

    /// One try of the case above: 600 ms of room, the lease's duration.
    fn a_failed_look_names_nobody() {
        let lease = MemoryLease::new();
        let settings = ElectionSettings {
            duration: Duration::from_millis(600),
            renew_period: Duration::from_millis(50),
            renew_deadline: Duration::from_millis(400),
        };
        let away = Arc::new(AtomicBool::new(false));
        let leader = Elector::new(Box::new(lease.clone()), "a@127.0.0.1:1", settings)
            .expect("the settings are in order");
        let follower = Elector::new(
            Box::new(Away {
                lease: lease.clone(),
                away: Arc::clone(&away),
            }),
            "b@127.0.0.1:2",
            settings,
        )
        .expect("the settings are in order");
        let (leader_fence, follower_fence) = (leader.fence(), follower.fence());
        let fences = [&*leader_fence, &*follower_fence];
        let (leader_standing, _led) = watch::channel(ElectionObservation::default());
        let (follower_standing, mut followed) = watch::channel(ElectionObservation::default());
        let leading =
            start(leader, Arc::new(MemoryStore::new()), leader_standing).expect("a thread");
        wait_for("the take", fences, || {
            leader_fence.leader_term(Instant::now()) == Some(1)
        });
        let following =
            start(follower, Arc::new(MemoryStore::new()), follower_standing).expect("a thread");
        // The bound on the clock, here and at the last wait: the holder
        // kept its lease. A host that held it past the lease's duration,
        // right after a look of the follower's, handed the lease to the
        // follower: its standing then reads the next term, and the wait
        // ends on that too.
        wait_for("the follower's look", fences, || {
            followed.borrow().holder == "a@127.0.0.1:1" || followed.borrow().term > 1
        });
        bound!(
            followed.borrow().term == 1,
            "the follower took the lease before its first look landed"
        );

        // The follower's store goes away. The second publish from here is
        // of a step that began after it did: that look failed, and it names
        // nobody under the term the looks before it saw.
        away.store(true, Ordering::SeqCst);
        for _ in 0..2 {
            followed.mark_unchanged();
            wait_for("a publish", fences, || {
                followed.has_changed().expect("the thread lives")
            });
        }
        assert_eq!(
            *followed.borrow(),
            ElectionObservation {
                holder: String::new(),
                term: 1,
                looked: true,
            }
        );

        // The store is back: the next look names the holder again.
        away.store(false, Ordering::SeqCst);
        wait_for("a look that lands", fences, || {
            followed.borrow().holder == "a@127.0.0.1:1" || followed.borrow().term > 1
        });
        bound!(
            followed.borrow().term == 1,
            "the follower took the lease before a look landed"
        );
        following.stop();
        ended(following);
        leading.stop();
        ended(leading);
    }

    #[test]
    fn a_lease_whose_term_fell_names_nobody_under_the_highest_term_seen() {
        // A store whose term went back, as no elector writes it: a Lease
        // deleted or replaced, a lease file restored. The look keeps the
        // highest term it saw and names nobody: no write stored the new
        // holder under that term.
        let held = |holder: &str, term: u64| LeaseRecord {
            holder: holder.to_owned(),
            term,
            duration_ms: 60_000,
            ..LeaseRecord::default()
        };
        let mut lease = MemoryLease::new();
        let first = held("x@127.0.0.1:8", 5);
        let (fallen, back) = (held("y@127.0.0.1:9", 3), held("y@127.0.0.1:9", 5));
        assert_eq!(
            lease.write(None, &first).expect("a write"),
            ConditionalWriteOutcome::Done
        );
        let settings = ElectionSettings {
            duration: Duration::from_millis(600),
            renew_period: Duration::from_millis(20),
            renew_deadline: Duration::from_millis(400),
        };
        let follower = Elector::new(Box::new(lease.clone()), "b@127.0.0.1:2", settings)
            .expect("the settings are in order");
        let (standing, stood) = watch::channel(ElectionObservation::default());
        let running = start(follower, Arc::new(MemoryStore::new()), standing).expect("a thread");
        until("the first look", || stood.borrow().holder == first.holder);
        assert_eq!(stood.borrow().term, 5);

        assert_eq!(
            lease.write(Some(&first), &fallen).expect("a write"),
            ConditionalWriteOutcome::Done
        );
        until("the look at the fallen term", || {
            stood.borrow().holder.is_empty()
        });
        assert_eq!(stood.borrow().term, 5);

        // A holder again at the first look that reads a term at the highest.
        assert_eq!(
            lease.write(Some(&fallen), &back).expect("a write"),
            ConditionalWriteOutcome::Done
        );
        until("the look at the term come back", || {
            stood.borrow().holder == back.holder
        });
        assert_eq!(stood.borrow().term, 5);
        running.stop();
        ended(running);
    }

    /// A lease whose first `failing` releases fail, the first of them after
    /// `stalled`; every release is stamped as its write begins.
    struct FailsReleases {
        lease: MemoryLease,
        failing: u32,
        stalled: Duration,
        releases: Arc<Mutex<Vec<Instant>>>,
    }

    impl LeaseBackend for FailsReleases {
        fn read(&mut self) -> Result<Option<LeaseRecord>, ElectError> {
            self.lease.read()
        }

        fn write(
            &mut self,
            expected: Option<&LeaseRecord>,
            record: &LeaseRecord,
        ) -> Result<ConditionalWriteOutcome, ElectError> {
            if !record.holder.is_empty() {
                return self.lease.write(expected, record);
            }
            self.releases.lock().push(Instant::now());
            if self.failing == 0 {
                return self.lease.write(expected, record);
            }
            self.failing -= 1;
            thread::sleep(std::mem::take(&mut self.stalled));
            Err(ElectError::Io {
                place: self.place(),
                source: std::io::Error::other("the store is away"),
            })
        }

        fn place(&self) -> String {
            self.lease.place()
        }
    }

    /// What one stop showed.
    struct Stop {
        /// How long after the stop each release began. The stop's instant
        /// is taken before the stop is sent, so no release began sooner:
        /// a hold of the host only makes these longer.
        releases: Vec<Duration>,
        /// How long after the stop the thread had ended.
        took: Duration,
        /// Whether the lease is free.
        free: bool,
    }

    /// What the loop logged: the writer of a subscriber the case sets on the
    /// thread it runs the loop on.
    #[derive(Clone)]
    struct Logged(Arc<Mutex<Vec<u8>>>);

    impl Write for Logged {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// Waits through `count` more looks at the least: the loop publishes once
    /// a look.
    fn looks(stood: &mut watch::Receiver<ElectionObservation>, count: u32) {
        stood.borrow_and_update();
        let mut seen = 0;
        until("the looks", || {
            if stood.has_changed().expect("the loop runs") {
                stood.borrow_and_update();
                seen += 1;
            }
            seen >= count
        });
    }

    #[test]
    fn a_fallen_term_logs_one_line_a_streak_and_one_more_when_it_falls_again() {
        let held = |holder: &str, term: u64| LeaseRecord {
            holder: holder.to_owned(),
            term,
            duration_ms: 60_000,
            ..LeaseRecord::default()
        };
        let mut lease = MemoryLease::new();
        let first = held("x@127.0.0.1:8", 5);
        let fallen = held("y@127.0.0.1:9", 3);
        let back = held("y@127.0.0.1:9", 5);
        let again = held("y@127.0.0.1:9", 4);
        assert_eq!(
            lease.write(None, &first).expect("a write"),
            ConditionalWriteOutcome::Done
        );
        let settings = ElectionSettings {
            duration: Duration::from_millis(600),
            renew_period: Duration::from_millis(20),
            renew_deadline: Duration::from_millis(400),
        };
        let follower = Elector::new(Box::new(lease.clone()), "b@127.0.0.1:2", settings)
            .expect("the settings are in order");
        let (standing, mut stood) = watch::channel(ElectionObservation::default());
        // The loop itself, on a thread of the case's own under a subscriber
        // that keeps its lines. A second subscriber stays registered beside
        // it: with one alone a line is on or off by the thread that reaches
        // it first, and a case beside this one reaches it under none.
        let lines = Logged(Arc::new(Mutex::new(Vec::new())));
        let writer = lines.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_writer(move || writer.clone())
            .finish();
        let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
        let (stop, stopped) = mpsc::channel();
        let looping = thread::spawn(move || {
            tracing::subscriber::with_default(subscriber, || {
                run(follower, &MemoryStore::new(), &standing, &stopped);
            });
        });
        until("the first look", || stood.borrow().holder == first.holder);

        // A streak of looks under the fallen term: one line.
        assert_eq!(
            lease.write(Some(&first), &fallen).expect("a write"),
            ConditionalWriteOutcome::Done
        );
        until("the look at the fallen term", || {
            stood.borrow().holder.is_empty()
        });
        looks(&mut stood, 5);
        // The term comes back and falls again: one line more.
        assert_eq!(
            lease.write(Some(&fallen), &back).expect("a write"),
            ConditionalWriteOutcome::Done
        );
        until("the look at the term come back", || {
            stood.borrow().holder == back.holder
        });
        assert_eq!(
            lease.write(Some(&back), &again).expect("a write"),
            ConditionalWriteOutcome::Done
        );
        until("the look at the second fall", || {
            stood.borrow().holder.is_empty()
        });
        looks(&mut stood, 5);
        stop.send(Word::Stop).expect("the loop runs");
        // Bounded like every wait here: a loop that ignored its stop fails
        // this case by name.
        until("the loop's end after its stop", || looping.is_finished());
        looping.join().expect("the loop ends");

        let text = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
        let fell = text
            .lines()
            .filter(|line| line.contains("the lease's term is under the highest this replica saw"))
            .count();
        assert_eq!(fell, 2, "{text}");
    }

    #[test]
    fn a_take_that_starts_no_term_in_the_log_is_one_warning_at_the_take() {
        let lines = Logged(Arc::new(Mutex::new(Vec::new())));
        // A second subscriber stays registered beside each loop's own, as in
        // the cases above.
        let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
        // Two candidates, each over a lease and a store of its own: the
        // second's log followed term 5 before its take of term 1.
        for (name, followed) in [("a@127.0.0.1:1", false), ("b@127.0.0.1:2", true)] {
            let settings = ElectionSettings {
                duration: Duration::from_millis(600),
                renew_period: Duration::from_millis(20),
                renew_deadline: Duration::from_millis(400),
            };
            let (store, feed) = journaled_from(&LeasePlace::Process, 0);
            if followed {
                feed.log
                    .restart(Position { seq: 9, term: 5 })
                    .expect("a place the log takes");
            }
            let candidate = Elector::new(Box::new(MemoryLease::new()), name, settings)
                .expect("the settings are in order");
            let (standing, stood) = watch::channel(ElectionObservation::default());
            let writer = lines.clone();
            let subscriber = tracing_subscriber::fmt()
                .with_ansi(false)
                .with_writer(move || writer.clone())
                .finish();
            let (stop, stopped) = mpsc::channel();
            let taking = Arc::clone(&store);
            let looping = thread::spawn(move || {
                tracing::subscriber::with_default(subscriber, || {
                    run(candidate, &taking, &standing, &stopped);
                });
            });
            until("the take", || stood.borrow().holder == name);
            stop.send(Word::Stop).expect("the loop runs");
            until("the loop's end after its stop", || looping.is_finished());
            looping.join().expect("the loop ends");
            assert_eq!(store.term(), 1);
            assert_eq!(store.followed(), followed.then_some(5), "{name}");
        }
        let text = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
        let count = |name: &str, what: &str| {
            text.lines()
                .filter(|line| line.contains(name) && line.contains(what))
                .count()
        };
        assert_eq!(count("a@127.0.0.1:1", "started no term"), 0, "{text}");
        assert_eq!(count("b@127.0.0.1:2", "started no term"), 1, "{text}");
        assert_eq!(count("b@127.0.0.1:2", "followed=5"), 1, "{text}");
    }

    /// A lease whose every write takes `takes` to store while `slow` is set,
    /// and when each began.
    struct WritesSlowly {
        lease: MemoryLease,
        takes: Duration,
        slow: Arc<AtomicBool>,
        began: Arc<Mutex<Vec<Instant>>>,
    }

    impl LeaseBackend for WritesSlowly {
        fn read(&mut self) -> Result<Option<LeaseRecord>, ElectError> {
            self.lease.read()
        }

        fn write(
            &mut self,
            expected: Option<&LeaseRecord>,
            record: &LeaseRecord,
        ) -> Result<ConditionalWriteOutcome, ElectError> {
            self.began.lock().push(Instant::now());
            if self.slow.load(Ordering::SeqCst) {
                thread::sleep(self.takes);
            }
            self.lease.write(expected, record)
        }

        fn place(&self) -> String {
            self.lease.place()
        }
    }

    /// What a leader over a slow store showed, sampled every 5 ms.
    struct Slow {
        samples: u32,
        /// The samples that read the fence closed.
        closed: u32,
        /// The samples whose standing did not name the leader.
        unnamed: u32,
        /// When each write of the loop began.
        began: Vec<Instant>,
    }

    /// One candidate takes the lease over a store whose every write takes
    /// `takes`, leads through `renewals` renewals a period apart, and is
    /// stopped. The deadline is two periods and the duration three, so the
    /// fence's margin is one period by both of its forms.
    fn led_over_a_slow_store(period: Duration, takes: Duration, renewals: u32) -> Slow {
        let identity = "a@127.0.0.1:1";
        let began = Arc::new(Mutex::new(Vec::new()));
        let store = WritesSlowly {
            lease: MemoryLease::new(),
            takes,
            slow: Arc::new(AtomicBool::new(true)),
            began: Arc::clone(&began),
        };
        let settings = ElectionSettings {
            duration: period * 3,
            renew_period: period,
            renew_deadline: period * 2,
        };
        let candidate =
            Elector::new(Box::new(store), identity, settings).expect("the settings are in order");
        let fence = candidate.fence();
        let (standing, stood) = watch::channel(ElectionObservation::default());
        let running = start(candidate, Arc::new(MemoryStore::new()), standing).expect("a thread");
        until("the take", || {
            fence.leader_term(Instant::now()) == Some(1) && stood.borrow().holder == identity
        });
        // Through the renewals the fence reads open and the standing names
        // the holder, at every sample: what readiness and the Leader answer
        // read.
        let through = Instant::now() + period * renewals + takes;
        let (mut samples, mut closed, mut unnamed) = (0_u32, 0_u32, 0_u32);
        while Instant::now() < through {
            samples += 1;
            closed += u32::from(fence.leader_term(Instant::now()).is_none());
            unnamed += u32::from(stood.borrow().holder != identity);
            thread::sleep(Duration::from_millis(5));
        }
        // The loop's writes alone: the release at the stop follows the last
        // step at once.
        let began = began.lock().clone();
        running.stop();
        ended(running);
        Slow {
            samples,
            closed,
            unnamed,
            began,
        }
    }

    #[test]
    fn a_leader_whose_renewals_take_under_the_margin_never_closes_its_fence() {
        unheld(
            "a slow store under the margin",
            Duration::from_secs(1),
            renewals_under_the_margin,
        );
    }

    /// One try of the case above: a second of room.
    fn renewals_under_the_margin() {
        // The fence's margin is the smaller of the deadline less the period
        // and half the deadline, both 2.5 s here. Every write takes 1.5 s.
        // That is over half the margin: steps that began a period after the
        // step before them RETURNED would land a renewal 5.5 s after the
        // stamp before it and close the fence for half a second in every
        // cycle. It is under the margin by a second: a held host has that
        // room.
        let period = Duration::from_millis(2500);
        let slow = led_over_a_slow_store(period, Duration::from_millis(1500), 3);
        let (samples, closed, unnamed) = (slow.samples, slow.closed, slow.unnamed);
        // The page's row reads this line.
        writeln!(
            std::io::stdout(),
            "SLOW_STORE samples={samples} closed={closed} unnamed={unnamed}"
        )
        .expect("the test's output");
        // No step met the store sooner than a period after the step before
        // it began. A hold only parts two writes further, so no hold
        // excuses this floor: it is read before the bounds.
        for pair in slow.began.windows(2) {
            let apart = pair[1].duration_since(pair[0]);
            assert!(apart >= period * 9 / 10, "two writes {apart:?} apart");
        }
        bound!(
            (closed, unnamed) == (0, 0),
            "{closed} closed and {unnamed} unnamed of {samples} samples"
        );
        bound!(slow.began.len() >= 4, "the take and three renewals");
    }

    #[test]
    fn the_loop_holds_the_fence_just_under_the_margin_and_follows_a_slow_step_at_once() {
        unheld(
            "a slow store just under the margin",
            Duration::from_millis(500),
            just_under_the_margin,
        );
    }

    /// One try of the case above: half a second of room.
    fn just_under_the_margin() {
        // A write of three quarters of the margin, 1.5 s under 2 s: half a
        // second of room for a held host. A wait that put any share of the
        // step's own time on top of the period would land the renewal after
        // the fence closed: half that time, here, a quarter of a second
        // after it.
        let slow =
            led_over_a_slow_store(Duration::from_millis(2000), Duration::from_millis(1500), 1);
        bound!(
            (slow.closed, slow.unnamed) == (0, 0),
            "{} closed and {} unnamed of {} samples",
            slow.closed,
            slow.unnamed,
            slow.samples
        );

        // A step that took the period or more is followed at once: the next
        // write begins when the one before it returned, never a period
        // after that. A hold only parts two writes further, so the closest
        // pair tells.
        let period = Duration::from_millis(200);
        let takes = Duration::from_millis(300);
        let began = Arc::new(Mutex::new(Vec::new()));
        let store = WritesSlowly {
            lease: MemoryLease::new(),
            takes,
            slow: Arc::new(AtomicBool::new(true)),
            began: Arc::clone(&began),
        };
        let settings = ElectionSettings {
            duration: period * 3,
            renew_period: period,
            renew_deadline: period * 2,
        };
        let candidate = Elector::new(Box::new(store), "a@127.0.0.1:1", settings)
            .expect("the settings are in order");
        let (standing, _stood) = watch::channel(ElectionObservation::default());
        let running = start(candidate, Arc::new(MemoryStore::new()), standing).expect("a thread");
        until("the take and four renewals", || began.lock().len() >= 5);
        let began = began.lock().clone();
        running.stop();
        ended(running);
        let closest = began
            .windows(2)
            .map(|pair| pair[1].duration_since(pair[0]))
            .min()
            .expect("five writes began");
        assert!(
            closest >= takes,
            "a write began {closest:?} after the one before it"
        );
        assert!(
            closest < takes + period / 2,
            "the closest two writes began {closest:?} apart: a step of {takes:?} was not followed at once"
        );
    }

    /// The loop over `elector` on a thread of the case's own, under a
    /// subscriber that keeps its lines. A second subscriber stays registered
    /// beside it, for the reason the fallen term's case gives.
    struct Looping {
        lines: Logged,
        stop: mpsc::Sender<Word>,
        thread: thread::JoinHandle<()>,
        _beside: tracing::Dispatch,
    }

    fn looping(elector: Elector, standing: watch::Sender<ElectionObservation>) -> Looping {
        let lines = Logged(Arc::new(Mutex::new(Vec::new())));
        let writer = lines.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_writer(move || writer.clone())
            .finish();
        let beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
        let (stop, stopped) = mpsc::channel();
        let thread = thread::spawn(move || {
            tracing::subscriber::with_default(subscriber, || {
                run(elector, &MemoryStore::new(), &standing, &stopped);
            });
        });
        Looping {
            lines,
            stop,
            thread,
            _beside: beside,
        }
    }

    impl Looping {
        /// The lines so far hold `words`.
        fn said(&self, words: &str) -> bool {
            String::from_utf8_lossy(&self.lines.0.lock()).contains(words)
        }

        /// Stops the loop, waits its end out, bounded, and counts the lines
        /// that hold `words`; the whole text comes back for a failure's
        /// message.
        fn stopped_with(self, words: &str) -> (usize, String) {
            self.stop.send(Word::Stop).expect("the loop runs");
            until("the loop's end after its stop", || {
                self.thread.is_finished()
            });
            self.thread.join().expect("the loop ends");
            let text = String::from_utf8(self.lines.0.lock().clone()).expect("the lines are text");
            let count = text.lines().filter(|line| line.contains(words)).count();
            (count, text)
        }
    }

    /// A leader whose lease moved after its last step finds it at its
    /// stop's release: it releases nothing and says the lease moved under
    /// the term it held, as a step that read the loss would. A replica that
    /// held nothing at its stop says nothing of a move, and neither does a
    /// holder whose release lands.
    #[test]
    fn a_stop_that_finds_the_lease_moved_says_so_and_releases_nothing() {
        const MOVED: &str = "the lease moved; the fence is closed";
        // A renew period of twenty seconds: the stop's release is the
        // first to read the lease after the take.
        let settings = ElectionSettings {
            duration: Duration::from_secs(60),
            renew_period: Duration::from_secs(20),
            renew_deadline: Duration::from_secs(40),
        };
        let store = Losable::default();
        let holder = Elector::new(Box::new(store.clone()), "a@127.0.0.1:1", settings)
            .expect("the settings are in order");
        let (standing, stood) = watch::channel(ElectionObservation::default());
        let leading = looping(holder, standing);
        until("the take", || stood.borrow().holder == "a@127.0.0.1:1");
        // Another candidate's record is stored over the holder's.
        let theirs = LeaseRecord {
            holder: "b@127.0.0.1:2".to_owned(),
            term: 2,
            duration_ms: 60_000,
            ..LeaseRecord::default()
        };
        store.put(&theirs);
        let (moved, text) = leading.stopped_with(MOVED);
        assert_eq!(moved, 1, "{text}");
        // The line carries the term this replica held, not the record's.
        let line = text
            .lines()
            .find(|line| line.contains(MOVED))
            .expect("the line");
        assert!(
            line.split_whitespace().any(|field| field == "term=1"),
            "{line}"
        );
        assert!(!text.contains("the lease is released"), "{text}");
        let mut lease = store.clone();
        assert_eq!(lease.read().expect("a read"), Some(theirs));

        let follower = Elector::new(Box::new(store.clone()), "c@127.0.0.1:3", settings)
            .expect("the settings are in order");
        let (standing, stood) = watch::channel(ElectionObservation::default());
        let following = looping(follower, standing);
        until("the first look", || stood.borrow().looked);
        let (moved, text) = following.stopped_with(MOVED);
        assert_eq!(moved, 0, "{text}");

        // A holder that stops over a quiet store releases, and says nothing
        // of a move.
        let quiet = Elector::new(Box::new(MemoryLease::new()), "d@127.0.0.1:4", settings)
            .expect("the settings are in order");
        let (standing, stood) = watch::channel(ElectionObservation::default());
        let released = looping(quiet, standing);
        until("the take", || stood.borrow().holder == "d@127.0.0.1:4");
        let (moved, text) = released.stopped_with(MOVED);
        assert_eq!(moved, 0, "{text}");
        assert!(text.contains("the lease is released"), "{text}");
    }

    /// A stop whose release comes back moved reads the lease again, and the
    /// elector counts that record's term as at any step. The last standing
    /// a stopping replica leaves counts it too: another replica's release
    /// under term nine, stored after the holder's last step, is read by
    /// the stop alone.
    #[test]
    fn a_release_read_again_at_a_stop_is_counted_by_the_last_standing() {
        let store = Losable::default();
        let holder = Elector::new(Box::new(store.clone()), "a@127.0.0.1:1", unhurried())
            .expect("the settings are in order");
        let (standing, stood) = watch::channel(ElectionObservation::default());
        let leading = looping(holder, standing);
        until("the take", || stood.borrow().holder == "a@127.0.0.1:1");
        assert_eq!(stood.borrow().term, 1);
        // The release under term nine is stored before the stop's own
        // write, which comes back moved; the read after it finds it.
        store.ahead([Ahead::Write(LeaseRecord {
            term: 9,
            ..LeaseRecord::default()
        })]);
        let (released, text) = leading.stopped_with("the lease is released");
        assert_eq!(released, 1, "{text}");
        assert_eq!(
            *stood.borrow(),
            ElectionObservation {
                holder: String::new(),
                term: 9,
                looked: true,
            },
            "{text}"
        );
    }

    #[test]
    fn a_late_renewal_logs_one_line_a_streak_and_one_more_when_renewals_turn_late_again() {
        unheld(
            "a late renewal's lines",
            Duration::from_millis(300),
            late_renewals_log_one_line_a_streak,
        );
    }

    /// One try of the case above: 300 ms of room, three renew periods.
    fn late_renewals_log_one_line_a_streak() {
        // A slow write takes four periods under a deadline of four: every
        // renewal of a slow streak lands on a fence that had closed. A fast
        // one lands a period after the stamp before it, three periods inside
        // the deadline: a held host has that room.
        let period = Duration::from_millis(100);
        let slow = Arc::new(AtomicBool::new(false));
        let began = Arc::new(Mutex::new(Vec::new()));
        let store = WritesSlowly {
            lease: MemoryLease::new(),
            takes: period * 4,
            slow: Arc::clone(&slow),
            began: Arc::clone(&began),
        };
        let settings = ElectionSettings {
            duration: period * 6,
            renew_period: period,
            renew_deadline: period * 4,
        };
        let leader = Elector::new(Box::new(store), "a@127.0.0.1:1", settings)
            .expect("the settings are in order");
        let (standing, _stood) = watch::channel(ElectionObservation::default());
        let looping = looping(leader, standing);
        // Waits until `more` more writes began: all but the last of them
        // have landed by then.
        let writes = |more: usize| {
            let from = began.lock().len();
            until("the writes", || began.lock().len() >= from + more);
        };
        writes(3);
        // A slow streak, two late renewals at the least: one line.
        slow.store(true, Ordering::SeqCst);
        writes(4);
        // Fast again: the streak ends at the first renewal that lands
        // inside the deadline.
        slow.store(false, Ordering::SeqCst);
        writes(4);
        // Late again: one line more.
        slow.store(true, Ordering::SeqCst);
        writes(3);
        slow.store(false, Ordering::SeqCst);
        let (late, text) = looping.stopped_with("a renewal landed on a fence that had closed");
        // The bound on the clock: every fast renewal landed inside the
        // deadline. A host that held one past it made one more late line.
        bound!(late == 2, "{late} late lines:\n{text}");
    }

    /// A lease in this process that a case can lose: after `lose` it holds
    /// no record, as a deleted object or a deleted file. Every write that
    /// lands is counted. A case can also say what the next looks find: one
    /// record stored before each read.
    #[derive(Clone, Default)]
    struct Losable {
        lease: MemoryLease,
        landed: Arc<AtomicU32>,
        ahead: Arc<Mutex<VecDeque<Ahead>>>,
        fails: Arc<AtomicBool>,
    }

    /// What another hand does to a [`Losable`] store ahead of the
    /// candidate's next calls, in order: the first waits for its call.
    enum Ahead {
        /// Before a read: a record stored, or nothing at this read.
        Read(Option<LeaseRecord>),
        /// Before a write: a record stored, so the write comes back moved.
        Write(LeaseRecord),
    }

    impl Losable {
        fn lose(&self) {
            self.lease.lose();
        }

        /// A record no candidate wrote, stored over the one there. A live
        /// holder's renewal can land between the read and the write, and
        /// the write is then turned away: both are made again.
        fn put(&self, record: &LeaseRecord) {
            let mut lease = self.lease.clone();
            until("a record stored over the one there", || {
                let stored = lease.read().expect("a read");
                lease.write(stored.as_ref(), record).expect("a write")
                    == ConditionalWriteOutcome::Done
            });
        }

        /// Records no candidate wrote, stored one before each of the next
        /// reads, in order.
        fn put_before_the_next_looks(&self, records: impl IntoIterator<Item = LeaseRecord>) {
            self.ahead(records.into_iter().map(|record| Ahead::Read(Some(record))));
        }

        /// What another hand does ahead of the next calls, in order.
        fn ahead(&self, steps: impl IntoIterator<Item = Ahead>) {
            self.ahead.lock().extend(steps);
        }

        /// Every write fails from here as a store that is away, and stores
        /// nothing, until a case says otherwise: but a write the script
        /// names ([`Ahead::Write`]) is answered by the store it left.
        fn fail_writes(&self, fail: bool) {
            self.fails.store(fail, Ordering::SeqCst);
        }

        fn landed(&self) -> u32 {
            self.landed.load(Ordering::SeqCst)
        }
    }

    impl LeaseBackend for Losable {
        fn read(&mut self) -> Result<Option<LeaseRecord>, ElectError> {
            let next = {
                let mut ahead = self.ahead.lock();
                match ahead.front() {
                    Some(Ahead::Read(_)) => ahead.pop_front(),
                    _ => None,
                }
            };
            if let Some(Ahead::Read(Some(record))) = next {
                self.put(&record);
            }
            self.lease.read()
        }

        fn write(
            &mut self,
            expected: Option<&LeaseRecord>,
            record: &LeaseRecord,
        ) -> Result<ConditionalWriteOutcome, ElectError> {
            let next = {
                let mut ahead = self.ahead.lock();
                match ahead.front() {
                    Some(Ahead::Write(_)) => ahead.pop_front(),
                    _ => None,
                }
            };
            // A write the script names is answered by the store it left,
            // whatever the failing mode: a case scripts a step in one call.
            if let Some(Ahead::Write(rival)) = next {
                self.put(&rival);
            } else if self.fails.load(Ordering::SeqCst) {
                return Err(ElectError::Io {
                    place: self.place(),
                    source: std::io::Error::other("the store is away"),
                });
            }
            let written = self.lease.write(expected, record)?;
            if written == ConditionalWriteOutcome::Done {
                self.landed.fetch_add(1, Ordering::SeqCst);
            }
            Ok(written)
        }

        fn place(&self) -> String {
            "a store that can lose its record".to_owned()
        }
    }

    #[test]
    fn a_record_that_is_gone_logs_one_line_a_streak_and_one_more_when_it_goes_again() {
        // The record's duration is a minute: the case ends inside the wait.
        let held = LeaseRecord {
            holder: "x@127.0.0.1:8".to_owned(),
            term: 5,
            duration_ms: 60_000,
            ..LeaseRecord::default()
        };
        let store = Losable::default();
        store.put(&held);
        let settings = ElectionSettings {
            duration: Duration::from_millis(600),
            renew_period: Duration::from_millis(20),
            renew_deadline: Duration::from_millis(400),
        };
        let follower = Elector::new(Box::new(store.clone()), "b@127.0.0.1:2", settings)
            .expect("the settings are in order");
        let (standing, mut stood) = watch::channel(ElectionObservation::default());
        let looping = looping(follower, standing);
        until("the first look", || stood.borrow().holder == held.holder);

        // A streak of looks at a record that is gone: one line, nobody
        // named, the term kept.
        store.lose();
        until("the look at the loss", || stood.borrow().holder.is_empty());
        looks(&mut stood, 5);
        assert_eq!(stood.borrow().term, 5);
        // The record is back and goes again, replaced by an empty one this
        // time: one line more, and no line of a fallen term beside it.
        store.put(&held);
        until("the look at the record come back", || {
            stood.borrow().holder == held.holder
        });
        store.put(&LeaseRecord::default());
        until("the look at the second loss", || {
            stood.borrow().holder.is_empty()
        });
        looks(&mut stood, 5);
        let (gone, text) = looping.stopped_with("the lease's record is gone or names no holder");
        assert_eq!(gone, 2, "{text}");
        assert!(!text.contains("the lease's term fell"), "{text}");
        // The holder followed again after the first streak is named anew.
        assert_eq!(text.matches(": following").count(), 2, "{text}");
    }

    #[test]
    fn a_take_whose_slow_step_closed_its_fence_logs_no_wait() {
        const TOOK: &str = "took a lease with no holder to wait for";
        let settings = ElectionSettings {
            duration: Duration::from_millis(600),
            renew_period: Duration::from_millis(20),
            renew_deadline: Duration::from_millis(400),
        };
        // A holder's release is taken at once and waits for nobody. This
        // take's step began a second before its report, as over a store that
        // answers slowly: the fence reads closed at the report, past its
        // renew deadline, and no line names a wait that was not made.
        let mut lease = MemoryLease::new();
        let released = LeaseRecord {
            term: 1,
            ..LeaseRecord::default()
        };
        assert_eq!(
            lease.write(None, &released).expect("a write"),
            ConditionalWriteOutcome::Done
        );
        let mut taker = Elector::new(Box::new(lease.clone()), "a@127.0.0.1:1", settings)
            .expect("the settings are in order");
        let began = Instant::now()
            .checked_sub(Duration::from_secs(1))
            .expect("the clock is past its first second");
        let tick = taker.tick(began).expect("the take");
        assert_eq!(tick, TickOutcome::Acquired { term: 2 });
        assert_eq!(taker.fence().leader_term(Instant::now()), None);
        let lines = Logged(Arc::new(Mutex::new(Vec::new())));
        let writer = lines.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_writer(move || writer.clone())
            .finish();
        let mut leading = false;
        tracing::subscriber::with_default(subscriber, || {
            super::report(
                &taker,
                &tick,
                &MemoryStore::new(),
                &mut String::new(),
                &mut super::Streak::default(),
                &mut leading,
            );
        });
        let text = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
        assert_eq!(text.matches(TOOK).count(), 0, "{text}");
        assert!(!leading, "{text}");
    }

    #[test]
    fn a_take_with_no_holder_to_wait_for_logs_its_take_and_leading_once_the_fence_opens() {
        const TOOK: &str = "took a lease with no holder to wait for";
        let settings = ElectionSettings {
            duration: Duration::from_millis(600),
            renew_period: Duration::from_millis(20),
            renew_deadline: Duration::from_millis(400),
        };
        // Stopped inside the renew deadline of a take over no record: the
        // take's line alone. The deadline is half a minute: the case ends
        // inside it.
        let patient = ElectionSettings {
            duration: Duration::from_secs(60),
            renew_deadline: Duration::from_secs(30),
            ..settings
        };
        let lease = MemoryLease::new();
        let taker = Elector::new(Box::new(lease.clone()), "a@127.0.0.1:1", patient)
            .expect("the settings are in order");
        let fence = taker.fence();
        let (standing, mut stood) = watch::channel(ElectionObservation::default());
        let inside = looping(taker, standing);
        until("the take", || stood.borrow().holder == "a@127.0.0.1:1");
        looks(&mut stood, 5);
        assert_eq!(fence.leader_term(Instant::now()), None);
        let (leading, text) = inside.stopped_with(": leading");
        assert_eq!((leading, text.matches(TOOK).count()), (0, 1), "{text}");

        // The same take under a deadline the case outlives: one line of
        // leading, after the take's, at the first step after the opening.
        let mut lease = MemoryLease::new();
        let taker = Elector::new(Box::new(lease.clone()), "a@127.0.0.1:1", settings)
            .expect("the settings are in order");
        let fence = taker.fence();
        let (standing, _stood) = watch::channel(ElectionObservation::default());
        let outlived = looping(taker, standing);
        until("the fence's opening", || {
            fence.leader_term(Instant::now()) == Some(1)
        });
        // The fence opens between two steps: the next one writes the line.
        until("the line of leading", || outlived.said(": leading"));
        // Another holder takes the lease away and stalls. Its record is
        // waited out and taken above: a holder was waited for, so that take
        // leads at once, under the line of leading and no line of the take.
        let other = LeaseRecord {
            holder: "x@127.0.0.1:8".to_owned(),
            term: 2,
            duration_ms: 600,
            ..LeaseRecord::default()
        };
        until("the other holder's take", || {
            let current = lease.read().expect("a read");
            lease.write(current.as_ref(), &other).expect("a write") == ConditionalWriteOutcome::Done
        });
        until("the take above", || {
            fence.leader_term(Instant::now()) == Some(3)
        });
        let (leading, text) = outlived.stopped_with(": leading");
        assert_eq!((leading, text.matches(TOOK).count()), (2, 1), "{text}");
        assert!(text.find(TOOK) < text.find(": leading"), "{text}");
    }

    #[test]
    fn a_holder_whose_record_is_emptied_logs_that_it_is_gone_and_no_fallen_term() {
        unheld(
            "an emptied record",
            Duration::from_millis(500),
            an_emptied_record_is_gone,
        );
    }

    /// One try of the case above, inside the wait: half a second of room.
    fn an_emptied_record_is_gone() {
        let settings = ElectionSettings {
            duration: Duration::from_millis(600),
            renew_period: Duration::from_millis(20),
            renew_deadline: Duration::from_millis(400),
        };
        let store = Losable::default();
        let holder = Elector::new(Box::new(store.clone()), "a@127.0.0.1:1", settings)
            .expect("the settings are in order");
        let fence = holder.fence();
        let (standing, mut stood) = watch::channel(ElectionObservation::default());
        let looping = looping(holder, standing);
        until("the fence's opening", || {
            fence.leader_term(Instant::now()) == Some(1)
        });
        // The record is replaced by an empty one under its holder. The
        // holder's step reads a loss to nobody, the next one a record that
        // is gone: one line, and none of a fallen term, which never comes
        // back. The case ends inside the wait.
        let emptied = Instant::now();
        store.put(&LeaseRecord::default());
        // The look at the loss names nobody under the term it held. Past
        // the wait the holder takes again, under the next term: the wait
        // ends on that too, and the bound below tells the two apart.
        until("the look at the loss", || {
            let look = stood.borrow();
            look.holder.is_empty() || look.term > 1
        });
        looks(&mut stood, 5);
        let term = stood.borrow().term;
        let (gone, text) = looping.stopped_with("the lease's record is gone or names no holder");
        // No hold explains a fallen term, which never comes back: its line
        // is read before the bound.
        assert!(!text.contains("the lease's term fell"), "{text}");
        // The bound on the clock: the case ended inside the wait. A host
        // that held it past the wait's end let the holder take again.
        let took = emptied.elapsed();
        bound!(
            took < settings.duration,
            "the case ended {took:?} after the record was emptied, outside the wait"
        );
        assert_eq!(term, 1);
        assert_eq!(gone, 1, "{text}");
    }

    /// A lease lost to another holder names that holder on the standing at
    /// the look that reads the loss: only a loss to nobody reads as no
    /// record.
    #[test]
    fn a_lease_lost_to_another_holder_names_that_holder_at_the_look_at_the_loss() {
        let settings = ElectionSettings {
            duration: Duration::from_millis(600),
            renew_period: Duration::from_millis(20),
            renew_deadline: Duration::from_millis(400),
        };
        let store = Losable::default();
        let holder = Elector::new(Box::new(store.clone()), "a@127.0.0.1:1", settings)
            .expect("the settings are in order");
        let fence = holder.fence();
        let (standing, stood) = watch::channel(ElectionObservation::default());
        let looping = looping(holder, standing);
        until("the fence's opening", || {
            fence.leader_term(Instant::now()) == Some(1)
        });
        // Another candidate's record is stored over the holder's, under the
        // next term. The first look that no longer names the holder read
        // the loss, and names the lease's holder.
        store.put(&LeaseRecord {
            holder: "b@127.0.0.1:2".to_owned(),
            term: 2,
            duration_ms: 60_000,
            ..LeaseRecord::default()
        });
        until("the look at the loss", || {
            stood.borrow().holder != "a@127.0.0.1:1"
        });
        let at_the_loss = stood.borrow().clone();
        let (_, text) = looping.stopped_with("the lease");
        assert_eq!(at_the_loss.holder, "b@127.0.0.1:2", "{text}");
        assert_eq!(at_the_loss.term, 2, "{text}");
    }

    /// A lease lost to nobody under a higher term. The tick of the look
    /// that loses it says the term this replica held; the record it read,
    /// a release another replica wrote, says the release's own. The
    /// standing counts the record's, so a holder restored under a term
    /// between the two is not named.
    #[test]
    fn a_term_read_at_the_look_that_loses_the_lease_stands_over_a_holder_restored_under_it() {
        let store = Losable::default();
        let holder = Elector::new(Box::new(store.clone()), "a@127.0.0.1:1", brisk())
            .expect("the settings are in order");
        let fence = holder.fence();
        let (standing, stood) = watch::channel(ElectionObservation::default());
        let looping = looping(holder, standing);
        until("the first lead", || {
            fence.leader_term(Instant::now()) == Some(1)
        });
        // Between two looks another replica takes the lease and gives it up
        // under term nine: the next look loses the lease and names nobody.
        // Before the look after it a holder is restored under term seven.
        store.put_before_the_next_looks([
            LeaseRecord {
                term: 9,
                ..LeaseRecord::default()
            },
            LeaseRecord {
                holder: "x@127.0.0.1:9".to_owned(),
                term: 7,
                duration_ms: 60_000,
                ..LeaseRecord::default()
            },
        ]);
        until("the look at the restored holder", || {
            stood.borrow().holder == "x@127.0.0.1:9"
                || looping.said("the lease's term is under the highest")
        });
        assert_eq!(
            *stood.borrow(),
            ElectionObservation {
                holder: String::new(),
                term: 9,
                looked: true,
            }
        );
        let (fallen, text) = looping.stopped_with("the lease's term is under the highest");
        assert_eq!(fallen, 1, "{text}");
    }

    /// The same release seen only where a renewal's write came back moved.
    /// It lands between the holder's read and its write, and the lease is
    /// read again: the elector observes that record, so the standing counts
    /// its term as at a plain read.
    #[test]
    fn a_release_read_again_after_a_moved_write_stands_over_a_holder_restored_under_it() {
        let store = Losable::default();
        let holder = Elector::new(Box::new(store.clone()), "a@127.0.0.1:1", brisk())
            .expect("the settings are in order");
        let fence = holder.fence();
        let (standing, stood) = watch::channel(ElectionObservation::default());
        let looping = looping(holder, standing);
        until("the first lead", || {
            fence.leader_term(Instant::now()) == Some(1)
        });
        // The release under term nine is stored before the renewal's write;
        // the read after that write finds it, and the look after that one
        // finds a holder restored under term seven.
        store.ahead([
            Ahead::Write(LeaseRecord {
                term: 9,
                ..LeaseRecord::default()
            }),
            Ahead::Read(None),
            Ahead::Read(Some(LeaseRecord {
                holder: "x@127.0.0.1:9".to_owned(),
                term: 7,
                duration_ms: 60_000,
                ..LeaseRecord::default()
            })),
        ]);
        until("the look at the restored holder", || {
            stood.borrow().holder == "x@127.0.0.1:9"
                || looping.said("the lease's term is under the highest")
        });
        assert_eq!(
            *stood.borrow(),
            ElectionObservation {
                holder: String::new(),
                term: 9,
                looked: true,
            }
        );
        let (fallen, text) = looping.stopped_with("the lease's term is under the highest");
        assert_eq!(fallen, 1, "{text}");
    }

    /// A step that read a term and then failed. A follower reads a release
    /// under term nine and its take's write fails, step after step: each
    /// step is an error, and the elector counted the term it read. The
    /// standing counts it with every step's, while the steps fail.
    #[test]
    fn a_term_read_at_a_step_that_then_failed_is_counted_while_the_steps_fail() {
        let store = Losable::default();
        store.put(&LeaseRecord {
            holder: "x@127.0.0.1:9".to_owned(),
            term: 3,
            duration_ms: 60_000,
            ..LeaseRecord::default()
        });
        let follower = Elector::new(Box::new(store.clone()), "b@127.0.0.1:2", brisk())
            .expect("the settings are in order");
        let (standing, mut stood) = watch::channel(ElectionObservation::default());
        let looping = looping(follower, standing);
        until("the holder is followed", || stood.borrow().term == 3);
        store.fail_writes(true);
        store.put_before_the_next_looks([LeaseRecord {
            term: 9,
            ..LeaseRecord::default()
        }]);
        until("the failing streak", || {
            looping.said("an election step failed")
        });
        looks(&mut stood, 2);
        assert_eq!(
            *stood.borrow(),
            ElectionObservation {
                holder: String::new(),
                term: 9,
                looked: true,
            }
        );
        let (failed, text) = looping.stopped_with("an election step failed");
        assert_eq!(failed, 1, "{text}");
    }

    /// A step that reads the lease twice. A holder told to leave whose
    /// release is still owed reads another holder under term nine; its
    /// release's write comes back moved over a holder restored under term
    /// seven, which the read after it finds. The term is read before
    /// anybody is named: that step names nobody, under term nine.
    #[test]
    fn a_step_that_reads_a_term_and_then_a_holder_restored_under_it_names_nobody() {
        const FALLEN: &str = "the lease's term is under the highest";
        let held = |holder: &str, term: u64| LeaseRecord {
            holder: holder.to_owned(),
            term,
            duration_ms: 60_000,
            ..LeaseRecord::default()
        };
        let store = Losable::default();
        let holder = Elector::new(Box::new(store.clone()), "a@127.0.0.1:1", brisk())
            .expect("the settings are in order");
        let fence = holder.fence();
        let (standing, stood) = watch::channel(ElectionObservation::default());
        let looping = looping(holder, standing);
        until("the first lead", || {
            fence.leader_term(Instant::now()) == Some(1)
        });
        // The word's release fails and stays owed: every step tries it
        // again in place of a renewal.
        store.fail_writes(true);
        looping
            .stop
            .send(Word::StepDown(None))
            .expect("the loop runs");
        until("the failed release", || {
            looping.said("the release failed and is tried again")
        });
        // The next step's read finds the holder under term nine, and its
        // release's write the holder restored under term seven. The script
        // is one call and its write lands under the failing mode: no step
        // of the loop reads the first with its own write still failing.
        store.ahead([
            Ahead::Read(Some(held("z@127.0.0.1:9", 9))),
            Ahead::Write(held("y@127.0.0.1:7", 7)),
        ]);
        until("the step that read both", || stood.borrow().term == 9);
        assert_eq!(
            *stood.borrow(),
            ElectionObservation {
                holder: String::new(),
                term: 9,
                looked: true,
            }
        );
        until("the line of the fallen term", || looping.said(FALLEN));
        let (fallen, text) = looping.stopped_with(FALLEN);
        assert_eq!(fallen, 1, "{text}");
    }

    /// A record lost under a live holder, by no release. The holder's fence
    /// is open until its own next step reads the lease; a replica that saw
    /// the record held waits that record's duration from its first sight of
    /// the loss, so no instant has two open fences, and it names nobody
    /// while it waits. Two election threads over one store, the standby's
    /// steps half a period behind the holder's. Both wait the lost record
    /// out and the first whose wait ends takes it: the standby on an idle
    /// host, the old holder where the standby's thread is held. The case
    /// reads the first take by either.
    #[test]
    fn a_record_lost_under_a_live_holder_is_taken_only_after_its_duration() {
        unheld(
            "a lost record",
            Duration::from_millis(400),
            a_lost_record_is_waited_out,
        );
    }

    /// One try of the case above: 400 ms of room, the renew deadline, the
    /// shortest stop that breaks one of its bounds.
    fn a_lost_record_is_waited_out() {
        let store = Losable::default();
        let settings = ElectionSettings {
            duration: Duration::from_millis(800),
            renew_period: Duration::from_millis(200),
            renew_deadline: Duration::from_millis(400),
        };
        let (holder, standby) = ("a@127.0.0.1:1", "b@127.0.0.1:2");
        let leader = Elector::new(Box::new(store.clone()), holder, settings)
            .expect("the settings are in order");
        let follower = Elector::new(Box::new(store.clone()), standby, settings)
            .expect("the settings are in order");
        let (leader_fence, follower_fence) = (leader.fence(), follower.fence());
        let (leader_standing, led) = watch::channel(ElectionObservation::default());
        let (follower_standing, followed) = watch::channel(ElectionObservation::default());
        let leading =
            start(leader, Arc::new(MemoryStore::new()), leader_standing).expect("a thread");
        until("the take", || {
            leader_fence.leader_term(Instant::now()) == Some(1)
        });
        let renewals = store.landed();
        until("a renewal", || store.landed() > renewals);
        thread::sleep(settings.renew_period / 2);
        let following =
            start(follower, Arc::new(MemoryStore::new()), follower_standing).expect("a thread");
        until("the standby's look", || followed.borrow().holder == holder);

        // The loss, right after one of the holder's steps.
        let renewals = store.landed();
        until("a renewal", || store.landed() > renewals);
        // The bound on the clock: the holder led to the loss. A host that
        // held it a renew deadline closed its fence, and one that held it
        // past the lease's duration handed the lease on: the record lost
        // is then of a later term.
        bound!(
            leader_fence.leader_term(Instant::now()) == Some(1),
            "the holder did not lead under term 1 at the loss"
        );
        let lost = Instant::now();
        store.lose();
        // Both fences through the loss, the first read on both sides of the
        // second as `wait_for` reads them, until a renew deadline after the
        // first take under the next term: a holder's fence closes by itself
        // inside that. `unnamed` is each replica seen with its fence closed
        // and nobody named: the old holder first, the standby second.
        let (mut samples, mut both, mut unnamed) = (0_u32, 0_u32, (false, false));
        // The first take, under whatever later term: how long after the
        // loss its read was asked for, its term, whether the standby's, and
        // that instant. A fence answers a term only inside its window, so
        // that instant is no sooner than the take, and a stop between a
        // read and the line that dates it lengthens nothing.
        let mut taken: Option<(Duration, u64, bool, Instant)> = None;
        while taken.is_none_or(|(_, _, _, seen)| seen.elapsed() < settings.renew_deadline) {
            let asked_before = Instant::now();
            let before = leader_fence.leader_term(asked_before);
            let asked_between = Instant::now();
            let between = follower_fence.leader_term(asked_between);
            let asked_after = Instant::now();
            let after = leader_fence.leader_term(asked_after);
            samples += 1;
            both += u32::from(before.is_some() && between.is_some() && before == after);
            unnamed.0 |= before.is_none() && led.borrow().holder.is_empty();
            unnamed.1 |= between.is_none() && followed.borrow().holder.is_empty();
            // The loop runs to its end whatever that term: the bound on it is
            // read after the count of two open fences, never ahead of it.
            let later = [
                (before, asked_before),
                (between, asked_between),
                (after, asked_after),
            ]
            .into_iter()
            .find_map(|(term, asked)| term.filter(|term| *term > 1).map(|term| (term, asked)));
            if let (None, Some((term, asked))) = (taken, later) {
                taken = Some((
                    asked.duration_since(lost),
                    term,
                    between == Some(term),
                    asked,
                ));
            }
            assert!(
                lost.elapsed() < Duration::from_secs(20),
                "neither replica took the lost record in 20 s"
            );
            thread::sleep(Duration::from_micros(200));
        }
        let (took, term, by_standby, _) = taken.expect("a replica leads under a later term");
        let (taker, named_nobody, other) = if by_standby {
            (standby, unnamed.1, &led)
        } else {
            (holder, unnamed.0, &followed)
        };
        // No hold excuses two open fences or a take sooner than the lost
        // record's duration: both are read before the bound.
        assert_eq!(
            both, 0,
            "two open fences in {both} of {samples} samples after the loss"
        );
        assert!(
            took >= settings.duration,
            "{taker} led {took:?} after the loss, sooner than the lost record's duration"
        );
        // The other bound on the clock: the first take read is the next
        // term's. A host that held the test past the lease's duration
        // handed the lease on once more before a fence was read under
        // that term, and the first take read is then a later one.
        bound!(
            term == 2,
            "the first take read after the loss leads under term {term}"
        );
        // The page's row reads this line.
        writeln!(
            std::io::stdout(),
            "LOST_RECORD samples={samples} both_open={both} took_ms={} by={taker}",
            took.as_millis()
        )
        .expect("stdout");
        assert!(named_nobody, "{taker} named a holder through its wait");
        // The bound on the clock once more: the other replica follows the
        // taker from its first look after the take, under the take's term:
        // a look from before the loss names the old holder under term 1
        // and ends nothing. A host that held the test past the lease's
        // duration handed the lease on, and the other replica then reads a
        // later term: the wait ends on that too.
        until("the other replica follows the taker", || {
            let look = other.borrow();
            (look.holder == taker && look.term == 2) || look.term > 2
        });
        let read = other.borrow().term;
        bound!(
            read == 2,
            "the other replica reads term {read}, past the taker's"
        );
        following.stop();
        ended(following);
        leading.stop();
        ended(leading);
    }

    /// One candidate takes the free lease over a store that fails its first
    /// `failing` releases, the first of them after `stalled`, and is
    /// stopped.
    fn stopped_over(failing: u32, stalled: Duration, renew_period: Duration) -> Stop {
        let mut lease = MemoryLease::new();
        let releases = Arc::new(Mutex::new(Vec::new()));
        let store = FailsReleases {
            lease: lease.clone(),
            failing,
            stalled,
            releases: Arc::clone(&releases),
        };
        let settings = ElectionSettings {
            duration: renew_period * 6,
            renew_period,
            renew_deadline: renew_period * 4,
        };
        let candidate = Elector::new(Box::new(store), "a@127.0.0.1:1", settings)
            .expect("the settings are in order");
        let fence = candidate.fence();
        let (standing, _stood) = watch::channel(ElectionObservation::default());
        let running = start(candidate, Arc::new(MemoryStore::new()), standing).expect("a thread");
        until("the take", || fence.leader_term(Instant::now()) == Some(1));
        let stopped = Instant::now();
        running.stop();
        ended(running);
        let took = stopped.elapsed();
        let free = lease
            .read()
            .expect("a read")
            .is_some_and(|record| record.holder.is_empty());
        let releases = releases
            .lock()
            .iter()
            .map(|began| began.duration_since(stopped))
            .collect();
        Stop {
            releases,
            took,
            free,
        }
    }

    #[test]
    fn a_release_that_fails_is_tried_again_a_renew_period_after_it_began() {
        unheld(
            "a release that fails",
            Duration::from_secs(1),
            failed_releases_tried_again,
        );
    }

    /// One try of the case above: a second of room on each upper bound.
    fn failed_releases_tried_again() {
        // Every write fails, and the first takes the whole period: three
        // writes and no more, and the lease is left to expire by its
        // duration.
        let period = Duration::from_secs(2);
        let stop = stopped_over(u32::MAX, period, period);
        assert_eq!(stop.releases.len(), 3);
        assert!(!stop.free);
        // THE FLOOR, which no hold of the host breaks: a write begins no
        // sooner than a renew period after the one before began. The first
        // began after the stop, so the second is a period past the stop
        // and the third two.
        assert!(
            stop.releases[1] >= period && stop.releases[2] >= period * 2,
            "the releases began {:?} after the stop",
            stop.releases
        );
        // THE MOST, each with a second of room on this period. The write
        // that took the period is followed at once: the pause never adds
        // to a stall. And no pause follows the last attempt.
        bound!(
            stop.releases[1] < period + period / 2,
            "the second release began {:?} after the stop",
            stop.releases[1]
        );
        bound!(
            stop.took < period * 2 + period / 2,
            "the thread ended {:?} after its stop",
            stop.took
        );

        // Two failures: the third write releases.
        let stop = stopped_over(2, Duration::ZERO, Duration::from_millis(20));
        assert_eq!(stop.releases.len(), 3);
        assert!(stop.free, "the third write releases");

        // No failure: one write, and the thread ends with no pause.
        let stop = stopped_over(0, Duration::ZERO, period);
        assert_eq!(stop.releases.len(), 1);
        assert!(stop.free);
        bound!(
            stop.took < period,
            "the thread ended {:?} after its stop",
            stop.took
        );
    }

    #[test]
    fn two_candidates_over_one_lease_lead_follow_and_hand_over_at_a_stop() {
        unheld(
            "two candidates over one lease",
            Duration::from_millis(600),
            two_candidates_hand_over_at_a_stop,
        );
    }

    /// One try of the case above: 600 ms of room, the lease's duration.
    fn two_candidates_hand_over_at_a_stop() {
        let lease = MemoryLease::new();
        // The order of the design's settings at a fifth of their size.
        let settings = ElectionSettings {
            duration: Duration::from_millis(600),
            renew_period: Duration::from_millis(50),
            renew_deadline: Duration::from_millis(400),
        };
        let candidate = |identity: &str| {
            Elector::new(Box::new(lease.clone()), identity, settings)
                .expect("the settings are in order")
        };
        let (first, second) = (candidate("a@127.0.0.1:1"), candidate("b@127.0.0.1:2"));
        let (first_fence, second_fence) = (first.fence(), second.fence());
        let fences = [&*first_fence, &*second_fence];
        let (first_store, second_store) =
            (Arc::new(MemoryStore::new()), Arc::new(MemoryStore::new()));
        let (first_standing, first_stood) = watch::channel(ElectionObservation::default());
        let (second_standing, second_stood) = watch::channel(ElectionObservation::default());

        // The first takes the free lease at its first step: the term is in
        // its store and its standing names itself.
        let first_running =
            start(first, Arc::clone(&first_store), first_standing).expect("a thread");
        wait_for("the first take", fences, || {
            first_stood.borrow().looked && first_fence.leader_term(Instant::now()) == Some(1)
        });
        assert_eq!(first_store.term(), 1);
        assert_eq!(first_stood.borrow().holder, "a@127.0.0.1:1");

        // The second follows: its standing names the first under its term.
        let second_running =
            start(second, Arc::clone(&second_store), second_standing).expect("a thread");
        wait_for("the second's first look", fences, || {
            second_stood.borrow().looked
        });
        // The bound on the clock: the first kept its lease to the second's
        // look. A host that held it past the lease's duration handed the
        // lease to the second, under the next term.
        let (holder, term) = {
            let look = second_stood.borrow();
            (look.holder.clone(), look.term)
        };
        bound!(
            holder == "a@127.0.0.1:1" && term == 1,
            "the second's first look names {holder:?} under term {term}"
        );
        assert_eq!(second_store.term(), 0, "a follower holds no term");

        // The planned stop: the first closes its fence, releases and ends;
        // the second takes the released lease under the next term.
        first_running.stop();
        // The thread opens its fence inside the take and writes its store's
        // term after it: the wait asks both.
        wait_for("the hand-over", fences, || {
            second_fence.leader_term(Instant::now()) == Some(2) && second_store.term() == 2
        });
        ended(first_running);
        assert_eq!(first_fence.leader_term(Instant::now()), None);
        assert_eq!(
            *first_stood.borrow(),
            ElectionObservation {
                holder: String::new(),
                term: 1,
                looked: true,
            },
            "a stopped replica names nobody"
        );

        // A stopped thread never takes again: the second keeps the lease
        // until its own stop, and the lease is then free.
        second_running.stop();
        ended(second_running);
        assert_eq!(second_fence.leader_term(Instant::now()), None);
        let mut reader = candidate("c@127.0.0.1:3");
        reader.tick(Instant::now()).expect("a look");
        assert_eq!(
            reader.observed().map(|record| record.term),
            Some(3),
            "the free lease is taken at once under the next term"
        );
    }

    /// The order of the design's settings at a fifth of their size, with a
    /// short period.
    fn brisk() -> ElectionSettings {
        ElectionSettings {
            duration: Duration::from_millis(600),
            renew_period: Duration::from_millis(20),
            renew_deadline: Duration::from_millis(400),
        }
    }

    /// A renew period of a minute, past the bound of every wait of a case:
    /// after its first step nothing reads the lease but the case's own
    /// word or stop.
    fn unhurried() -> ElectionSettings {
        ElectionSettings {
            duration: Duration::from_secs(180),
            renew_period: Duration::from_secs(60),
            renew_deadline: Duration::from_secs(120),
        }
    }

    #[test]
    fn a_leader_told_a_term_above_its_own_leaves_and_the_standby_leads_above_that_term() {
        let lease = MemoryLease::new();
        let candidate = |identity: &str| {
            Elector::new(Box::new(lease.clone()), identity, brisk())
                .expect("the settings are in order")
        };
        let (first, second) = (candidate("a@127.0.0.1:1"), candidate("b@127.0.0.1:2"));
        let (first_fence, second_fence) = (first.fence(), second.fence());
        let fences = [&*first_fence, &*second_fence];
        let second_store = Arc::new(MemoryStore::new());
        let (first_standing, first_stood) = watch::channel(ElectionObservation::default());
        let (second_standing, second_stood) = watch::channel(ElectionObservation::default());
        let first_running =
            start(first, Arc::new(MemoryStore::new()), first_standing).expect("a thread");
        wait_for("the first lead", fences, || {
            first_fence.leader_term(Instant::now()) == Some(1)
        });
        let second_running =
            start(second, Arc::clone(&second_store), second_standing).expect("a thread");
        wait_for("the second's first look", fences, || {
            second_stood.borrow().holder == "a@127.0.0.1:1"
        });

        // A term of seven is said to the leader, as a gateway's floor is:
        // it leaves, and the standby takes the released lease at its next
        // look, above the term the leader was told. No instant has two open
        // fences: every wait here asks both.
        first_running
            .door()
            .saw(7)
            .expect("a term that leaves one to take");
        wait_for("the standby's take above the term said", fences, || {
            second_fence.leader_term(Instant::now()) == Some(8) && second_store.term() == 8
        });
        wait_for("the replica that left follows the taker", fences, || {
            *first_stood.borrow()
                == ElectionObservation {
                    holder: "b@127.0.0.1:2".to_owned(),
                    term: 8,
                    looked: true,
                }
        });
        assert_eq!(first_fence.leader_term(Instant::now()), None);
        first_running.stop();
        ended(first_running);
        second_running.stop();
        ended(second_running);
    }

    #[test]
    fn a_replica_alone_over_its_own_lease_that_is_told_to_leave_leads_again_at_the_word() {
        // No other candidate can take a lease the elector made itself: a
        // word holds nothing back and the take follows it at once, a renew
        // period of a minute ahead of the next step by the clock. The line
        // at the word says so.
        let elector =
            Elector::alone("a@127.0.0.1:1", unhurried()).expect("the settings are in order");
        let fence = elector.fence();
        let (standing, stood) = watch::channel(ElectionObservation::default());
        let looping = looping(elector, standing);
        until("the first lead", || {
            fence.leader_term(Instant::now()) == Some(1)
        });
        looping
            .stop
            .send(Word::StepDown(Some(7)))
            .expect("the loop runs");
        until("the retake above the term said", || {
            fence.leader_term(Instant::now()) == Some(8)
        });
        until("the standing after the retake", || {
            stood.borrow().holder == "a@127.0.0.1:1"
        });
        assert_eq!(stood.borrow().term, 8);
        let (said, text) = looping.stopped_with("alone over its lease, takes it again at once");
        assert_eq!(said, 1, "{text}");
        // The fence closed at the word, and the retake says it leads again.
        assert_eq!(text.matches(": leading").count(), 2, "{text}");
        assert!(!text.contains("waits its turn"), "{text}");
        assert!(
            !text.contains("takes nothing for one lease duration"),
            "{text}"
        );
    }

    /// A replica another candidate can follow takes no step at the word: a
    /// standby's next look takes what it left, and its own next step comes
    /// with the renew period, a minute away here. A step would write the
    /// line of the turn it waits or of the record with no holder.
    #[test]
    fn a_replica_another_can_follow_takes_no_step_at_the_word_to_leave() {
        let elector = Elector::new(Box::new(MemoryLease::new()), "a@127.0.0.1:1", unhurried())
            .expect("the settings are in order");
        let (standing, stood) = watch::channel(ElectionObservation::default());
        let looping = looping(elector, standing);
        until("the first take", || {
            stood.borrow().holder == "a@127.0.0.1:1"
        });
        looping
            .stop
            .send(Word::StepDown(None))
            .expect("the loop runs");
        until("the leave", || {
            looping.said("takes nothing for one lease duration")
        });
        // The loop is past the word: a step that followed it is written
        // by now, and the stop writes none.
        thread::sleep(Duration::from_millis(200));
        let (_, text) = looping.stopped_with("stepped down");
        assert!(!text.contains("waits its turn"), "{text}");
        assert!(!text.contains("names no holder"), "{text}");
    }

    #[test]
    fn a_leader_told_its_own_term_keeps_on_and_logs_no_second_line_of_leading() {
        // A word that moves no lease leaves the report's flag as it stood:
        // the leader renews on under its term, and no renewal says again
        // that it leads.
        let elector = Elector::alone("a@127.0.0.1:1", brisk()).expect("the settings are in order");
        let fence = elector.fence();
        let (standing, mut stood) = watch::channel(ElectionObservation::default());
        let looping = looping(elector, standing);
        until("the first lead", || {
            fence.leader_term(Instant::now()) == Some(1)
        });
        until("the line of leading", || looping.said(": leading"));
        looping
            .stop
            .send(Word::StepDown(Some(1)))
            .expect("the loop runs");
        looks(&mut stood, 3);
        assert_eq!(fence.leader_term(Instant::now()), Some(1));
        let (leading, text) = looping.stopped_with(": leading");
        assert_eq!(leading, 1, "{text}");
    }

    /// A planned hand-over is of one term's lease. Made under term one and
    /// taken after a gateway's floor moved this replica to term three, it
    /// leaves that take alone; made under the term held, it is taken.
    #[test]
    fn a_hand_over_planned_under_a_term_the_replica_left_releases_nothing() {
        let elector = Elector::alone("a@127.0.0.1:1", brisk()).expect("the settings are in order");
        let fence = elector.fence();
        let (standing, mut stood) = watch::channel(ElectionObservation::default());
        let running = start(elector, Arc::new(MemoryStore::new()), standing).expect("a thread");
        until("the first lead", || {
            fence.leader_term(Instant::now()) == Some(1)
        });
        let door = running.door();
        door.saw(2).expect("a term that leaves one to take");
        until("the take above the floor", || {
            fence.leader_term(Instant::now()) == Some(3)
        });
        // The hand-over of term one comes late: three steps on, the lease
        // of term three is still held.
        door.hand_over(1);
        looks(&mut stood, 3);
        assert_eq!(fence.leader_term(Instant::now()), Some(3));
        door.hand_over(3);
        until("the take after the hand-over of the term held", || {
            fence.leader_term(Instant::now()) == Some(4)
        });
        running.stop();
        ended(running);
    }

    #[test]
    fn a_replica_with_no_standby_that_is_told_to_leave_waits_one_lease_duration_and_leads_again_above(
    ) {
        let lease = MemoryLease::new();
        let settings = brisk();
        let elector = Elector::new(Box::new(lease.clone()), "a@127.0.0.1:1", settings)
            .expect("the settings are in order");
        let fence = elector.fence();
        let (standing, stood) = watch::channel(ElectionObservation::default());
        let looping = looping(elector, standing);
        until("the first lead", || {
            fence.leader_term(Instant::now()) == Some(1)
        });
        // The fence opens between two steps: the next one writes the line.
        until("the line of leading", || looping.said(": leading"));

        // A term of seven is said: the replica leaves, and it names nobody
        // under that term while it waits its turn.
        let word = Instant::now();
        looping
            .stop
            .send(Word::StepDown(Some(7)))
            .expect("the loop runs");
        until("the leave", || {
            *stood.borrow()
                == ElectionObservation {
                    holder: String::new(),
                    term: 7,
                    looked: true,
                }
        });
        assert_eq!(fence.leader_term(Instant::now()), None);
        // With no standby over a lease it shares, it takes the lease back
        // when its hold ends, one lease duration after the word and above
        // the term it was told: the price of the door where no standby
        // looks.
        until("the retake", || {
            fence.leader_term(Instant::now()) == Some(8)
        });
        let out = word.elapsed();
        assert!(
            out >= settings.duration,
            "led again {out:?} after the word, inside its hold"
        );
        until("the standing after the retake", || {
            stood.borrow().holder == "a@127.0.0.1:1"
        });
        assert_eq!(stood.borrow().term, 8);
        let (waits, text) =
            looping.stopped_with("the lease stands free and this replica waits its turn");
        assert_eq!(waits, 1, "{text}");
        let line = text
            .lines()
            .find(|line| line.contains("waits its turn"))
            .expect("the waiting line");
        assert!(
            line.contains(" INFO "),
            "a turn waited out is no warning: {line}"
        );
        assert_eq!(
            text.matches("stepped down; the lease is released").count(),
            1,
            "{text}"
        );
        assert_eq!(text.matches(": leading").count(), 2, "{text}");
        assert!(!text.contains("the lease's record is gone"), "{text}");
        assert!(!text.contains("the lease's term is under"), "{text}");
    }

    #[test]
    fn a_replica_that_waits_its_turn_names_nobody_over_a_record_that_ran_out() {
        // A record that names a holder and ran out by this replica's clock
        // is free to take. A replica inside its hold takes nothing and
        // follows nobody: its standing names no holder, whatever the record
        // says.
        let mut lease = MemoryLease::new();
        let elector = Elector::new(Box::new(lease.clone()), "a@127.0.0.1:1", brisk())
            .expect("the settings are in order");
        let fence = elector.fence();
        let (standing, stood) = watch::channel(ElectionObservation::default());
        let looping = looping(elector, standing);
        until("the first lead", || {
            fence.leader_term(Instant::now()) == Some(1)
        });
        looping
            .stop
            .send(Word::StepDown(Some(7)))
            .expect("the loop runs");
        until("the leave", || stood.borrow().term == 7);
        // Another holder's record lands over the release, says five renew
        // periods and is never renewed: the replica follows it while it
        // lasts, and its own hold outlasts it.
        let released = lease.read().expect("a read").expect("the release");
        assert!(released.is_released());
        let ran_out = LeaseRecord {
            holder: "x@127.0.0.1:9".to_owned(),
            term: 8,
            acquired_at_micros: released.renewed_at_micros + 1,
            renewed_at_micros: released.renewed_at_micros + 1,
            duration_ms: 100,
        };
        assert_eq!(
            lease.write(Some(&released), &ran_out).expect("a write"),
            ConditionalWriteOutcome::Done
        );
        until("the record that ran out names nobody", || {
            *stood.borrow()
                == ElectionObservation {
                    holder: String::new(),
                    term: 8,
                    looked: true,
                }
        });
        assert_eq!(fence.leader_term(Instant::now()), None);
        until("the retake above the record that ran out", || {
            fence.leader_term(Instant::now()) == Some(9)
        });
        let (waits, text) =
            looping.stopped_with("the lease stands free and this replica waits its turn");
        assert!(waits >= 1, "{text}");
    }

    #[test]
    fn a_replica_that_waits_its_turn_counts_a_term_it_read_in_a_release() {
        // Inside this replica's hold a standby takes the lease and gives
        // it up between two looks: the replica reads only the release,
        // under term eight. The standing counts that term, so a holder
        // restored under term seven is not named.
        let mut lease = MemoryLease::new();
        let elector = Elector::new(Box::new(lease.clone()), "a@127.0.0.1:1", brisk())
            .expect("the settings are in order");
        let fence = elector.fence();
        let (standing, stood) = watch::channel(ElectionObservation::default());
        let looping = looping(elector, standing);
        until("the first lead", || {
            fence.leader_term(Instant::now()) == Some(1)
        });
        looping
            .stop
            .send(Word::StepDown(Some(7)))
            .expect("the loop runs");
        until("the leave", || stood.borrow().term == 7);
        let released = lease.read().expect("a read").expect("the release");
        assert!(released.is_released());
        let later = LeaseRecord {
            term: 8,
            renewed_at_micros: released.renewed_at_micros + 1,
            duration_ms: 0,
            ..released.clone()
        };
        assert_eq!(
            lease.write(Some(&released), &later).expect("a write"),
            ConditionalWriteOutcome::Done
        );
        until("the term read in the release is counted", || {
            *stood.borrow()
                == ElectionObservation {
                    holder: String::new(),
                    term: 8,
                    looked: true,
                }
        });
        let restored = LeaseRecord {
            holder: "x@127.0.0.1:9".to_owned(),
            term: 7,
            acquired_at_micros: later.renewed_at_micros + 1,
            renewed_at_micros: later.renewed_at_micros + 1,
            duration_ms: 60_000,
        };
        assert_eq!(
            lease.write(Some(&later), &restored).expect("a write"),
            ConditionalWriteOutcome::Done
        );
        until("the line of the fallen term", || {
            looping.said("the lease's term is under the highest")
        });
        assert_eq!(
            *stood.borrow(),
            ElectionObservation {
                holder: String::new(),
                term: 8,
                looked: true,
            }
        );
        let (fallen, text) = looping.stopped_with("the lease's term is under the highest");
        assert_eq!(fallen, 1, "{text}");
    }

    #[test]
    fn a_term_told_stands_over_a_release_read_under_a_lower_term() {
        // A release under term three stands for a minute. A follower told
        // term nine reads it look after look, and the standing keeps the
        // term told.
        let free = LeaseRecord {
            term: 3,
            duration_ms: 60_000,
            ..LeaseRecord::default()
        };
        let mut lease = MemoryLease::new();
        assert_eq!(
            lease.write(None, &free).expect("a write"),
            ConditionalWriteOutcome::Done
        );
        let follower = Elector::new(Box::new(lease.clone()), "b@127.0.0.1:2", brisk())
            .expect("the settings are in order");
        let (standing, mut stood) = watch::channel(ElectionObservation::default());
        let looping = looping(follower, standing);
        until("the term read in the release", || stood.borrow().term == 3);
        looping
            .stop
            .send(Word::StepDown(Some(9)))
            .expect("the loop runs");
        until("the term counted", || stood.borrow().term == 9);
        looks(&mut stood, 5);
        assert_eq!(
            *stood.borrow(),
            ElectionObservation {
                holder: String::new(),
                term: 9,
                looked: true
            }
        );
        let (gone, text) = looping.stopped_with("the lease's record is gone or names no holder");
        assert_eq!(gone, 1, "{text}");
    }

    #[test]
    fn a_replica_told_to_leave_inside_its_takes_wait_says_only_that_it_waits() {
        // The release such a replica reads at its next steps is its own:
        // it waits its turn, and no line says a record is gone.
        let elector = Elector::new(Box::new(MemoryLease::new()), "a@127.0.0.1:1", brisk())
            .expect("the settings are in order");
        let (standing, _stood) = watch::channel(ElectionObservation::default());
        let looping = looping(elector, standing);
        until("the line of the take", || {
            looping.said("took a lease with no holder to wait for")
        });
        looping
            .stop
            .send(Word::StepDown(None))
            .expect("the loop runs");
        until("the line of the wait", || looping.said("waits its turn"));
        let (gone, text) = looping.stopped_with("the lease's record is gone");
        assert_eq!(gone, 0, "{text}");
    }

    #[test]
    fn a_word_past_the_door_that_leaves_no_term_to_take_counts_nothing() {
        // The controller's door refuses such a term before a word is said.
        // One that reaches the loop all the same is refused by the elector:
        // the holder leads on, and the standing counts nothing.
        let elector = Elector::new(Box::new(MemoryLease::new()), "a@127.0.0.1:1", brisk())
            .expect("the settings are in order");
        let fence = elector.fence();
        let (standing, mut stood) = watch::channel(ElectionObservation::default());
        let looping = looping(elector, standing);
        until("the first lead", || {
            fence.leader_term(Instant::now()) == Some(1)
        });
        looping
            .stop
            .send(Word::StepDown(Some(TERM_CEILING - 1)))
            .expect("the loop runs");
        until("the line of the refusal", || {
            looping.said("the word is refused")
        });
        looks(&mut stood, 3);
        assert_eq!(fence.leader_term(Instant::now()), Some(1));
        assert_eq!(
            *stood.borrow(),
            ElectionObservation {
                holder: "a@127.0.0.1:1".to_owned(),
                term: 1,
                looked: true
            }
        );
        let (refused, text) = looping.stopped_with("the word is refused");
        assert_eq!(refused, 1, "{text}");
        assert!(!text.contains("stepped down;"), "{text}");
    }

    #[test]
    fn a_term_said_to_a_follower_is_counted_and_its_take_is_above_it() {
        // The holder's duration is a minute: the follower takes only at its
        // release.
        let held = LeaseRecord {
            holder: "x@127.0.0.1:8".to_owned(),
            term: 5,
            duration_ms: 60_000,
            ..LeaseRecord::default()
        };
        let mut lease = MemoryLease::new();
        assert_eq!(
            lease.write(None, &held).expect("a write"),
            ConditionalWriteOutcome::Done
        );
        let follower = Elector::new(Box::new(lease.clone()), "b@127.0.0.1:2", brisk())
            .expect("the settings are in order");
        let fence = follower.fence();
        let (standing, mut stood) = watch::channel(ElectionObservation::default());
        let looping = looping(follower, standing);
        until("the first look", || stood.borrow().holder == held.holder);

        // A caller says it saw term nine. No lease moves; the holder under
        // term five is named no longer, since no write stored it under nine.
        looping
            .stop
            .send(Word::StepDown(Some(9)))
            .expect("the loop runs");
        until("the term counted", || {
            *stood.borrow()
                == ElectionObservation {
                    holder: String::new(),
                    term: 9,
                    looked: true,
                }
        });
        // Looks at the holder's record under the lower term: one line a
        // streak, nobody named, no write.
        looks(&mut stood, 5);
        assert!(stood.borrow().holder.is_empty());
        assert_eq!(stood.borrow().term, 9);
        assert_eq!(lease.read().expect("a read"), Some(held.clone()));
        // The holder releases: the follower takes at once, above the term it
        // was told, and it was never held back.
        let released = LeaseRecord {
            holder: String::new(),
            duration_ms: 0,
            ..held.clone()
        };
        assert_eq!(
            lease.write(Some(&held), &released).expect("a write"),
            ConditionalWriteOutcome::Done
        );
        until("the take", || fence.leader_term(Instant::now()) == Some(10));
        let (under, text) = looping
            .stopped_with("the lease's term is under the highest this replica saw or was told");
        assert_eq!(under, 1, "{text}");
        assert!(!text.contains("stepped down"), "{text}");
        assert!(!text.contains("waits its turn"), "{text}");
    }

    /// A term said reaches the standing at the word, before the next step
    /// could read it: a follower told term nine names the holder it
    /// followed under term three no longer, a renew period ahead of its
    /// next look.
    #[test]
    fn a_term_said_to_a_follower_stands_before_its_next_step() {
        let held = LeaseRecord {
            holder: "x@127.0.0.1:8".to_owned(),
            term: 3,
            duration_ms: 60_000,
            ..LeaseRecord::default()
        };
        let mut lease = MemoryLease::new();
        assert_eq!(
            lease.write(None, &held).expect("a write"),
            ConditionalWriteOutcome::Done
        );
        let follower = Elector::new(Box::new(lease), "b@127.0.0.1:2", unhurried())
            .expect("the settings are in order");
        let (standing, stood) = watch::channel(ElectionObservation::default());
        let looping = looping(follower, standing);
        until("the first look", || stood.borrow().holder == held.holder);
        looping
            .stop
            .send(Word::StepDown(Some(9)))
            .expect("the loop runs");
        until("the term said", || stood.borrow().term == 9);
        assert_eq!(
            *stood.borrow(),
            ElectionObservation {
                holder: String::new(),
                term: 9,
                looked: true,
            }
        );
        let (stepped, text) = looping.stopped_with("stepped down");
        assert_eq!(stepped, 0, "{text}");
    }

    /// A word's release that comes back moved reads the lease again, and
    /// the standing counts that record's term at the word. A holder told to
    /// leave under no term finds another replica's release under term
    /// nine: nobody is named under term nine, before the next step.
    #[test]
    fn a_release_read_again_at_a_word_is_counted_before_the_next_step() {
        let store = Losable::default();
        let holder = Elector::new(Box::new(store.clone()), "a@127.0.0.1:1", unhurried())
            .expect("the settings are in order");
        let (standing, stood) = watch::channel(ElectionObservation::default());
        let looping = looping(holder, standing);
        until("the take", || stood.borrow().holder == "a@127.0.0.1:1");
        store.ahead([Ahead::Write(LeaseRecord {
            term: 9,
            ..LeaseRecord::default()
        })]);
        looping
            .stop
            .send(Word::StepDown(None))
            .expect("the loop runs");
        until("the word taken", || stood.borrow().holder.is_empty());
        assert_eq!(
            *stood.borrow(),
            ElectionObservation {
                holder: String::new(),
                term: 9,
                looked: true,
            }
        );
        let (stepped, text) = looping.stopped_with("stepped down; the lease is released");
        assert_eq!(stepped, 1, "{text}");
    }

    #[test]
    fn a_release_that_fails_at_the_word_is_tried_again_at_the_next_step_with_nobody_named() {
        let lease = MemoryLease::new();
        let releases = Arc::new(Mutex::new(Vec::new()));
        let store = FailsReleases {
            lease: lease.clone(),
            failing: 2,
            stalled: Duration::ZERO,
            releases: Arc::clone(&releases),
        };
        let elector = Elector::new(Box::new(store), "a@127.0.0.1:1", brisk())
            .expect("the settings are in order");
        let fence = elector.fence();
        let (standing, stood) = watch::channel(ElectionObservation::default());
        let looping = looping(elector, standing);
        until("the first lead", || {
            fence.leader_term(Instant::now()) == Some(1)
        });
        // The fence opens between two steps: the next one writes the line.
        until("the line of leading", || looping.said(": leading"));

        // The release fails at the word and at the step after it, and lands
        // at the third try. The fence is closed from the word, nobody is
        // named, and no step in between renews the lease.
        looping
            .stop
            .send(Word::StepDown(None))
            .expect("the loop runs");
        until("the release", || {
            lease
                .clone()
                .read()
                .expect("a read")
                .is_some_and(|record| record.is_released())
        });
        assert_eq!(fence.leader_term(Instant::now()), None);
        assert_eq!(releases.lock().len(), 3);
        assert!(stood.borrow().holder.is_empty());
        let (failed, text) = looping
            .stopped_with("stepped down; the release failed and is tried again at the next step");
        assert_eq!(failed, 1, "{text}");
        assert_eq!(text.matches(": leading").count(), 1, "{text}");
    }

    #[test]
    fn the_highest_term_a_word_may_say_leaves_the_last_term_to_take() {
        let elector = Elector::new(Box::new(MemoryLease::new()), "a@127.0.0.1:1", brisk())
            .expect("the settings are in order");
        let fence = elector.fence();
        let (standing, _stood) = watch::channel(ElectionObservation::default());
        let running = start(elector, Arc::new(MemoryStore::new()), standing).expect("a thread");
        until("the first lead", || {
            fence.leader_term(Instant::now()) == Some(1)
        });
        // The replica leaves under the term said, and the last term under
        // the ceiling is its next take.
        running
            .door()
            .saw(TERM_CEILING - 2)
            .expect("the last term a word may say");
        until("the retake", || {
            fence.leader_term(Instant::now()) == Some(TERM_CEILING - 1)
        });
        running.stop();
        ended(running);
    }

    #[test]
    fn a_term_said_at_the_door_is_the_handles_word() {
        let elector = Elector::alone("a@127.0.0.1:1", brisk()).expect("the settings are in order");
        let fence = elector.fence();
        let (standing, _stood) = watch::channel(ElectionObservation::default());
        let running = start(elector, Arc::new(MemoryStore::new()), standing).expect("a thread");
        until("the first lead", || {
            fence.leader_term(Instant::now()) == Some(1)
        });
        // The door says through the handle's words: a term said through it
        // is a word to leave that the thread takes.
        running
            .door()
            .saw(7)
            .expect("a term that leaves one to take");
        until("the retake above the term said", || {
            fence.leader_term(Instant::now()) == Some(8)
        });
        running.stop();
        ended(running);
    }

    #[test]
    fn the_door_refuses_a_term_that_leaves_none_to_take() {
        let (door, said) = Said::door();
        let refused = door
            .saw(TERM_CEILING - 1)
            .expect_err("no term is left to take");
        assert!(
            matches!(refused, ElectError::Term(next) if next == TERM_CEILING),
            "{refused}"
        );
        assert!(said.terms().is_empty(), "no word is said");
    }

    #[test]
    fn a_term_that_leaves_none_to_take_is_refused_at_the_door() {
        let elector = Elector::new(
            Box::new(MemoryLease::new()),
            "a@127.0.0.1:1",
            ElectionSettings::default(),
        )
        .expect("the settings are in order");
        let (standing, _stood) = watch::channel(ElectionObservation::default());
        let running = start(elector, Arc::new(MemoryStore::new()), standing).expect("a thread");
        let refused = running
            .door()
            .saw(TERM_CEILING - 1)
            .expect_err("no term is left to take");
        assert!(
            matches!(refused, ElectError::Term(next) if next == TERM_CEILING),
            "{refused}"
        );
        running.stop();
        ended(running);
    }
}
