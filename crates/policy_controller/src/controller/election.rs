//! The election wired into the run loop (design 5.3): where the lease lives,
//! the identity this replica holds it under, the thread that steps the
//! elector, and the standing the `Leader` call answers from on a replica
//! that does not lead.
//!
//! One path in every mode: a controller with no shared lease elects itself
//! over a record in its own process, so the [`Fence`](crate::elector::Fence)
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
//! lease at once; it then stops and never takes again.

use std::{
    fmt, io,
    net::{IpAddr, SocketAddr},
    path::{Path, PathBuf},
    sync::{mpsc, Arc},
    thread,
    time::Instant,
};

use tokio::sync::{oneshot, watch};
use tracing::{debug, info, warn};

use super::config::ConfigError;
#[cfg(feature = "kubernetes")]
use crate::elector::KubernetesLease;
use crate::{
    elector::{ElectError, ElectionSettings, Elector, FileLease, LeaseBackend, MemoryLease, Tick},
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
    fn is_shared(&self) -> bool {
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
pub struct Election {
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

impl Election {
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
pub(crate) async fn elector(election: &Election, identity: String) -> Result<Elector, LeaseError> {
    let backend: Box<dyn LeaseBackend> = match &election.lease {
        LeasePlace::Process => Box::new(MemoryLease::new()),
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
pub(crate) struct Standing {
    /// The lease's holder, `controller-id@host:port`; empty when the lease
    /// is free, when the last look failed, and when it read a term under
    /// the highest this replica saw: no write stored that holder under
    /// the term beside it.
    pub(crate) holder: String,
    /// The highest term this replica saw.
    pub(crate) term: u64,
    /// Whether a look at the lease has succeeded: readiness waits for the
    /// first, so a replica that cannot reach its lease never reads ready.
    pub(crate) looked: bool,
}

/// The running election: its stop signal and its end.
pub(crate) struct Running {
    stop: mpsc::Sender<()>,
    /// Taken once the thread's end was seen.
    end: Option<oneshot::Receiver<()>>,
}

impl Running {
    /// The planned stop: the thread closes the fence, releases the lease
    /// and ends.
    pub(crate) fn stop(&self) {
        // A thread that is gone has nothing to release.
        let _ = self.stop.send(());
    }

    /// Resolves when the thread is gone, after a stop or on its own; at
    /// once on every call after that.
    pub(crate) async fn ended(&mut self) {
        if let Some(end) = self.end.as_mut() {
            // Sent at the thread's end and dropped at its panic: gone
            // either way.
            let _ = end.await;
            self.end = None;
        }
    }
}

/// Starts the election thread over `elector`: the term goes to `store` at
/// every take, each step's sight of the lease to `standing`.
pub(crate) fn start(
    elector: Elector,
    store: Arc<MemoryStore>,
    standing: watch::Sender<Standing>,
) -> io::Result<Running> {
    let (stop, stopped) = mpsc::channel();
    let (end, ended) = oneshot::channel();
    thread::Builder::new()
        .name("election".to_owned())
        .spawn(move || {
            run(elector, &store, &standing, &stopped);
            // The run loop is gone with its receiver when the controller
            // returned first.
            let _ = end.send(());
        })?;
    Ok(Running {
        stop,
        end: Some(ended),
    })
}

fn run(
    mut elector: Elector,
    store: &MemoryStore,
    standing: &watch::Sender<Standing>,
    stopped: &mpsc::Receiver<()>,
) {
    let period = elector.settings().renew_period;
    let deadline = elector.settings().renew_deadline;
    let mut seen = Standing::default();
    let mut following = String::new();
    let mut gone = false;
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
        match elector.tick(began) {
            Ok(tick) => {
                failing = false;
                seen.looked = true;
                match &tick {
                    Tick::Acquired { .. } => {
                        stamped = Some(began);
                        late = false;
                    }
                    Tick::Renewed { .. } => {
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
                match elector.observed() {
                    // A record that is gone names nobody and has a line of
                    // its own.
                    _ if matches!(tick, Tick::Gone { .. }) => seen.holder.clear(),
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
                                "the lease's term fell under the highest this replica saw; nobody is named until it is back"
                            );
                        }
                        fallen = true;
                        seen.holder.clear();
                    }
                    Some(record) => {
                        fallen = false;
                        seen.holder.clone_from(&record.holder);
                        seen.term = record.term;
                    }
                    None => seen.holder.clear(),
                }
                report(&elector, &tick, store, &mut following, &mut gone);
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
        seen.term = seen.term.max(elector.fence().term());
        standing.send_replace(seen.clone());
        // A step that took the period or more is followed at once, never
        // sooner.
        match stopped.recv_timeout(period.saturating_sub(began.elapsed())) {
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    give_up(&mut elector);
    seen.holder.clear();
    seen.term = seen.term.max(elector.fence().term());
    standing.send_replace(seen);
}

/// One line a change of role or a gone streak; a take's term to the store.
fn report(
    elector: &Elector,
    tick: &Tick,
    store: &MemoryStore,
    following: &mut String,
    gone: &mut bool,
) {
    let was_gone = std::mem::replace(gone, matches!(tick, Tick::Gone { .. }));
    match tick {
        Tick::Acquired { term } => {
            following.clear();
            store.set_term(*term);
            info!(identity = elector.identity(), term, "leading");
        }
        Tick::Lost { term, holder } => {
            warn!(
                identity = elector.identity(),
                term,
                holder = holder.as_str(),
                "the lease moved; the fence is closed"
            );
        }
        Tick::Following { holder, term } => {
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
        Tick::Gone { term } => {
            // One line a streak; a holder followed after it is named anew.
            following.clear();
            if !was_gone {
                warn!(
                    identity = elector.identity(),
                    term,
                    "the lease's record is gone; nobody is named and it is taken after that record's duration"
                );
            }
        }
        Tick::Renewed { .. } | Tick::Raced => {
            debug!(identity = elector.identity(), ?tick, "election step");
        }
    }
}

/// The planned stop: the elector closes the fence, then rewrites the record
/// released, so the next candidate to look takes the lease at once. The
/// first release follows the loop's last step at once: a stop does not wait
/// for the period. A release that fails is tried again, a bounded number of
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
        dialable, elector, identity, run, split_identity, start, Election, LeaseError, LeasePlace,
        Running, Standing,
    };
    use crate::{
        controller::config::ConfigError,
        elector::{
            ElectError, ElectionSettings, Elector, LeaseBackend, LeaseRecord, MemoryLease, Written,
        },
        store::{LeasesStore, MemoryStore},
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
        let mut election = Election::default();
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
    async fn a_lease_file_needs_its_directory_and_writes_nothing_before_the_first_step() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let at = |path: PathBuf| Election {
            lease: LeasePlace::File(path),
            ..Election::default()
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
        let (standing, _stood) = watch::channel(Standing::default());
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

    /// Polls `met` every few milliseconds. No instant has two open fences:
    /// at every poll the first fence is read, then the second, then the
    /// first again, each at its own instant, and the first open under one
    /// term on both sides of an open second is two open fences at the
    /// second's instant. One instant for both reads would call a hand-over
    /// between them two leaders: the taker's stamp lies after that instant.
    fn wait_for(what: &str, fences: [&crate::elector::Fence; 2], met: impl Fn() -> bool) {
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

    /// Waits, bounded, for a stopped election's thread to be gone.
    fn ended(running: Running) {
        let mut end = running.end.expect("the end was not seen yet");
        let started = Instant::now();
        while end.try_recv().is_err() {
            assert!(
                started.elapsed() < Duration::from_secs(20),
                "the election thread did not end 20 s after its stop"
            );
            thread::sleep(Duration::from_millis(2));
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
        ) -> Result<Written, ElectError> {
            self.lease.write(expected, record)
        }

        fn place(&self) -> String {
            self.lease.place()
        }
    }

    #[test]
    fn a_look_that_fails_names_nobody_until_one_lands() {
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
        let (leader_standing, _led) = watch::channel(Standing::default());
        let (follower_standing, mut followed) = watch::channel(Standing::default());
        let leading =
            start(leader, Arc::new(MemoryStore::new()), leader_standing).expect("a thread");
        wait_for("the take", fences, || {
            leader_fence.leader_term(Instant::now()) == Some(1)
        });
        let following =
            start(follower, Arc::new(MemoryStore::new()), follower_standing).expect("a thread");
        wait_for("the follower's look", fences, || {
            followed.borrow().holder == "a@127.0.0.1:1"
        });

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
            Standing {
                holder: String::new(),
                term: 1,
                looked: true,
            }
        );

        // The store is back: the next look names the holder again.
        away.store(false, Ordering::SeqCst);
        wait_for("a look that lands", fences, || {
            followed.borrow().holder == "a@127.0.0.1:1"
        });
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
        assert_eq!(lease.write(None, &first).expect("a write"), Written::Done);
        let settings = ElectionSettings {
            duration: Duration::from_millis(600),
            renew_period: Duration::from_millis(20),
            renew_deadline: Duration::from_millis(400),
        };
        let follower = Elector::new(Box::new(lease.clone()), "b@127.0.0.1:2", settings)
            .expect("the settings are in order");
        let (standing, stood) = watch::channel(Standing::default());
        let running = start(follower, Arc::new(MemoryStore::new()), standing).expect("a thread");
        until("the first look", || stood.borrow().holder == first.holder);
        assert_eq!(stood.borrow().term, 5);

        assert_eq!(
            lease.write(Some(&first), &fallen).expect("a write"),
            Written::Done
        );
        until("the look at the fallen term", || {
            stood.borrow().holder.is_empty()
        });
        assert_eq!(stood.borrow().term, 5);

        // A holder again at the first look that reads a term at the highest.
        assert_eq!(
            lease.write(Some(&fallen), &back).expect("a write"),
            Written::Done
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
        ) -> Result<Written, ElectError> {
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
    fn looks(stood: &mut watch::Receiver<Standing>, count: u32) {
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
        assert_eq!(lease.write(None, &first).expect("a write"), Written::Done);
        let settings = ElectionSettings {
            duration: Duration::from_millis(600),
            renew_period: Duration::from_millis(20),
            renew_deadline: Duration::from_millis(400),
        };
        let follower = Elector::new(Box::new(lease.clone()), "b@127.0.0.1:2", settings)
            .expect("the settings are in order");
        let (standing, mut stood) = watch::channel(Standing::default());
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
            Written::Done
        );
        until("the look at the fallen term", || {
            stood.borrow().holder.is_empty()
        });
        looks(&mut stood, 5);
        // The term comes back and falls again: one line more.
        assert_eq!(
            lease.write(Some(&fallen), &back).expect("a write"),
            Written::Done
        );
        until("the look at the term come back", || {
            stood.borrow().holder == back.holder
        });
        assert_eq!(
            lease.write(Some(&back), &again).expect("a write"),
            Written::Done
        );
        until("the look at the second fall", || {
            stood.borrow().holder.is_empty()
        });
        looks(&mut stood, 5);
        stop.send(()).expect("the loop runs");
        // Bounded like every wait here: a loop that ignored its stop fails
        // this case by name.
        until("the loop's end after its stop", || looping.is_finished());
        looping.join().expect("the loop ends");

        let text = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
        let fell = text
            .lines()
            .filter(|line| {
                line.contains("the lease's term fell under the highest this replica saw")
            })
            .count();
        assert_eq!(fell, 2, "{text}");
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
        ) -> Result<Written, ElectError> {
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
        let (standing, stood) = watch::channel(Standing::default());
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
        assert_eq!((closed, unnamed), (0, 0), "of {samples} samples");
        // And no step met the store sooner than a period after the step
        // before it began: a hold only parts two writes further.
        assert!(slow.began.len() >= 4, "the take and three renewals");
        for pair in slow.began.windows(2) {
            let apart = pair[1].duration_since(pair[0]);
            assert!(apart >= period * 9 / 10, "two writes {apart:?} apart");
        }
    }

    #[test]
    fn the_loop_holds_the_fence_just_under_the_margin_and_follows_a_slow_step_at_once() {
        // A write of three quarters of the margin, 1.5 s under 2 s: half a
        // second of room for a held host. A wait that put any share of the
        // step's own time on top of the period would land the renewal after
        // the fence closed: half that time, here, a quarter of a second
        // after it.
        let slow =
            led_over_a_slow_store(Duration::from_millis(2000), Duration::from_millis(1500), 1);
        assert_eq!(
            (slow.closed, slow.unnamed),
            (0, 0),
            "of {} samples",
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
        let (standing, _stood) = watch::channel(Standing::default());
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
        stop: mpsc::Sender<()>,
        thread: thread::JoinHandle<()>,
        _beside: tracing::Dispatch,
    }

    fn looping(elector: Elector, standing: watch::Sender<Standing>) -> Looping {
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
        /// Stops the loop, waits its end out, bounded, and counts the lines
        /// that hold `words`; the whole text comes back for a failure's
        /// message.
        fn stopped_with(self, words: &str) -> (usize, String) {
            self.stop.send(()).expect("the loop runs");
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
        let (standing, stood) = watch::channel(Standing::default());
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
        let (standing, stood) = watch::channel(Standing::default());
        let following = looping(follower, standing);
        until("the first look", || stood.borrow().looked);
        let (moved, text) = following.stopped_with(MOVED);
        assert_eq!(moved, 0, "{text}");

        // A holder that stops over a quiet store releases, and says nothing
        // of a move.
        let quiet = Elector::new(Box::new(MemoryLease::new()), "d@127.0.0.1:4", settings)
            .expect("the settings are in order");
        let (standing, stood) = watch::channel(Standing::default());
        let released = looping(quiet, standing);
        until("the take", || stood.borrow().holder == "d@127.0.0.1:4");
        let (moved, text) = released.stopped_with(MOVED);
        assert_eq!(moved, 0, "{text}");
        assert!(text.contains("the lease is released"), "{text}");
    }

    #[test]
    fn a_late_renewal_logs_one_line_a_streak_and_one_more_when_renewals_turn_late_again() {
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
        let (standing, _stood) = watch::channel(Standing::default());
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
        assert_eq!(late, 2, "{text}");
    }

    /// A lease in this process that a case can lose: after `lose` it holds
    /// no record, as a deleted object or a deleted file. Every write that
    /// lands is counted.
    #[derive(Clone, Default)]
    struct Losable {
        lease: MemoryLease,
        landed: Arc<AtomicU32>,
    }

    impl Losable {
        fn lose(&self) {
            self.lease.lose();
        }

        /// A record no candidate wrote, stored over the one there.
        fn put(&self, record: &LeaseRecord) {
            let mut lease = self.lease.clone();
            let stored = lease.read().expect("a read");
            let written = lease.write(stored.as_ref(), record);
            assert_eq!(written.expect("a write"), Written::Done);
        }

        fn landed(&self) -> u32 {
            self.landed.load(Ordering::SeqCst)
        }
    }

    impl LeaseBackend for Losable {
        fn read(&mut self) -> Result<Option<LeaseRecord>, ElectError> {
            self.lease.read()
        }

        fn write(
            &mut self,
            expected: Option<&LeaseRecord>,
            record: &LeaseRecord,
        ) -> Result<Written, ElectError> {
            let written = self.lease.write(expected, record)?;
            if written == Written::Done {
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
        let (standing, mut stood) = watch::channel(Standing::default());
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
        let (gone, text) = looping.stopped_with("the lease's record is gone");
        assert_eq!(gone, 2, "{text}");
        assert!(!text.contains("the lease's term fell"), "{text}");
        // The holder followed again after the first streak is named anew.
        assert_eq!(text.matches(": following").count(), 2, "{text}");
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
        let (leader_standing, led) = watch::channel(Standing::default());
        let (follower_standing, followed) = watch::channel(Standing::default());
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
        let lost = Instant::now();
        store.lose();
        // Both fences through the loss, the first read on both sides of the
        // second as `wait_for` reads them, until a renew deadline after the
        // first take under the next term: a holder's fence closes by itself
        // inside that. `unnamed` is each replica seen with its fence closed
        // and nobody named: the old holder first, the standby second.
        let (mut samples, mut both, mut unnamed) = (0_u32, 0_u32, (false, false));
        // The first take: how long after the loss, whether the standby's,
        // and when it was seen.
        let mut taken: Option<(Duration, bool, Instant)> = None;
        while taken.is_none_or(|(_, _, seen)| seen.elapsed() < settings.renew_deadline) {
            let before = leader_fence.leader_term(Instant::now());
            let between = follower_fence.leader_term(Instant::now());
            let after = leader_fence.leader_term(Instant::now());
            samples += 1;
            both += u32::from(before.is_some() && between.is_some() && before == after);
            unnamed.0 |= before.is_none() && led.borrow().holder.is_empty();
            unnamed.1 |= between.is_none() && followed.borrow().holder.is_empty();
            if taken.is_none() && [before, between, after].contains(&Some(2)) {
                taken = Some((lost.elapsed(), between == Some(2), Instant::now()));
            }
            assert!(
                lost.elapsed() < Duration::from_secs(20),
                "neither replica took the lost record in 20 s"
            );
            thread::sleep(Duration::from_micros(200));
        }
        let (took, by_standby, _) = taken.expect("a replica leads under the next term");
        let (taker, named_nobody, other) = if by_standby {
            (standby, unnamed.1, &led)
        } else {
            (holder, unnamed.0, &followed)
        };
        // The page's row reads this line.
        writeln!(
            std::io::stdout(),
            "LOST_RECORD samples={samples} both_open={both} took_ms={} by={taker}",
            took.as_millis()
        )
        .expect("stdout");
        assert_eq!(
            both, 0,
            "two open fences in {both} of {samples} samples after the loss"
        );
        assert!(
            took >= settings.duration,
            "{taker} led {took:?} after the loss, sooner than the lost record's duration"
        );
        assert!(named_nobody, "{taker} named a holder through its wait");
        until("the other replica follows the taker", || {
            other.borrow().holder == taker
        });
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
        let (standing, _stood) = watch::channel(Standing::default());
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
        assert!(
            stop.releases[1] < period + period / 2,
            "the second release began {:?} after the stop",
            stop.releases[1]
        );
        assert!(
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
        assert!(
            stop.took < period,
            "the thread ended {:?} after its stop",
            stop.took
        );
        assert!(stop.free);
    }

    #[test]
    fn two_candidates_over_one_lease_lead_follow_and_hand_over_at_a_stop() {
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
        let (first_standing, first_stood) = watch::channel(Standing::default());
        let (second_standing, second_stood) = watch::channel(Standing::default());

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
        assert_eq!(
            *second_stood.borrow(),
            Standing {
                holder: "a@127.0.0.1:1".to_owned(),
                term: 1,
                looked: true,
            }
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
            Standing {
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
}
