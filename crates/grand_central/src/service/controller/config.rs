//! What an operator hands the controller at start (design 5: the addresses,
//! the credentials and the shutdown settings are read at start; the routing
//! policy is the spec's and never here): the listeners with their scopes and
//! TLS, the admin listener, the controller's identity, the election and the
//! shutdown bounds, and worker discovery settings when built with discovery.
//! The configuration is validated before anything binds.

#![expect(
    clippy::exhaustive_structs,
    reason = "a caller still builds or matches these types whole: each mark comes with its call"
)]

use std::{
    collections::BTreeMap,
    fmt,
    net::{IpAddr, SocketAddr},
    path::PathBuf,
    thread,
    time::Duration,
};

use smg_worker::health::overload::{
    OverloadThresholds, DEFAULT_TOKEN_USAGE_CEILING, DEFAULT_WAITING_REQUESTS,
};

#[cfg(feature = "discovery")]
pub use super::discovery::DiscoverySettings;
use super::election::{ElectionConfig, LeasePlace};
use crate::service::elector::ElectError;

/// Worker discovery's settings in a build without the `discovery` feature:
/// a type with no value. Such a build runs no provider, so its
/// configuration names none.
#[cfg(not(feature = "discovery"))]
#[derive(Clone, Debug, Eq, PartialEq)]
#[expect(clippy::exhaustive_enums, reason = "no case: discovery is not built")]
pub enum DiscoverySettings {}

/// A credentialed group of the controller's calls (design 8). Each scope is
/// mounted on exactly one listener, alone or beside others; a listener's
/// credential rule is its scopes' rule, so scopes with different rules go
/// on different listeners.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum Scope {
    /// `Session`, `Decide` and `WatchWorkers`: the gateways' calls. Mutual
    /// TLS, or a per-gateway bearer whose verified subject is the gateway id
    /// that keys its leases.
    Gateway,
    /// `PushLoad` and `PushKvEvents`: the relays' calls, a per-relay
    /// credential. Mounted beside the gateway scope by default: a relay is a
    /// gateway in front of a host-local worker.
    Ingestion,
    /// The unary operator calls and `WatchRegime`: the control-plane keys and
    /// roles, `user` reads and `admin` writes, every call audited.
    Control,
    /// `Replicate`: the standbys' call, a replica credential.
    Replication,
}

impl Scope {
    /// Every scope, in the order the design lists them.
    pub const ALL: [Scope; 4] = [
        Scope::Gateway,
        Scope::Ingestion,
        Scope::Control,
        Scope::Replication,
    ];

    /// The scope's name on the command line, in log lines and in refusals.
    pub const fn as_str(self) -> &'static str {
        match self {
            Scope::Gateway => "gateway",
            Scope::Ingestion => "ingestion",
            Scope::Control => "control",
            Scope::Replication => "replication",
        }
    }
}

impl fmt::Display for Scope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The TLS material of a listener, as paths read once at bind.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Tls {
    /// The server certificate chain, PEM.
    pub cert: PathBuf,
    /// The chain's private key, PEM.
    pub key: PathBuf,
    /// The authority the clients' certificates chain to. With it the
    /// listener is mutual TLS: every client presents a certificate the
    /// authority signed, and the verified subject is the scope's principal
    /// (design 8: the principal is transport metadata, never a message
    /// field). Without it the scope's credential rides the call metadata.
    pub client_ca: Option<PathBuf>,
}

/// One gRPC listener: an address, the scopes mounted on it and its TLS.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Listener {
    /// The listener's name in log lines and refusals; the binary names each
    /// after its flag (`gateway`, `control`, `replication`, `ingestion`).
    pub name: String,
    /// The address it binds. Port `0` takes an ephemeral port, reported by
    /// [`Controller::addresses`](super::Controller::addresses). A replica
    /// that follows a leader names its replication listener's port
    /// ([`ConfigError::ReplicationPortUnnamed`]).
    pub addr: SocketAddr,
    /// The scopes it serves; every scope is mounted once across the
    /// listeners.
    pub scopes: Vec<Scope>,
    /// Its TLS, or plaintext when `None` (a loopback, or a mesh that
    /// terminates TLS in front of the process).
    pub tls: Option<Tls>,
}

/// The controller's start-up configuration. It is compared by `PartialEq`
/// alone: its overload usage is a ratio.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// This controller's identity: the holder of the leases it takes
    /// (design 10, `controller-id@host:port`) and the name on its log lines.
    /// One token of letters, digits, `.`, `_` and `-`.
    pub controller_id: String,
    /// The gRPC listeners. The binary mounts the gateway and ingestion
    /// scopes on one and the control and replication scopes on one each.
    pub listeners: Vec<Listener>,
    /// The HTTP admin listener: the probes, the version and the metrics.
    pub admin: SocketAddr,
    /// How long the listeners keep accepting after readiness fails, so the
    /// probes and the gateways' `Leader` polls see the change before a
    /// connection is refused. At most half the grace period.
    pub shutdown_settle: Duration,
    /// The bound on a shutdown, the settle window included: when it runs
    /// out the listeners are closed with their streams. The controller's
    /// streams reconnect on a drain, so the default is far below the
    /// gateway's, which waits for generations to finish. A period the clock
    /// cannot add to the present is no bound. A start on a spec store that
    /// another controller holds waits this long for that one to stop
    /// ([`Controller::spec_store`](super::Controller::spec_store)).
    pub shutdown_grace: Duration,
    /// How often the load monitor, whose reports a decision reads, polls a
    /// worker whose stream is not serving it, and the freshness bound a
    /// pushed record must meet to stand in for the poll: the gateway's
    /// `--load-monitor-interval`. The monitor polls by the interval cut to
    /// whole seconds, and by one second for less.
    pub load_poll_interval: Duration,
    /// The overload thresholds the object of an engine's worker states for
    /// each signal its registration names none for (design, decision 10):
    /// the gateway's `--worker-overload-waiting-requests` and
    /// `--worker-overload-token-usage`. A registration's own threshold
    /// stands over each, and an external worker's object takes none of
    /// them. A worker at or above a threshold is vetoed: a count is from 1
    /// to `i64::MAX`, the most a load report states, and a usage is above 0
    /// and at most 1. A signal that states none has no default: a worker
    /// that registers none for it is not judged by it.
    pub worker_overload: OverloadThresholds,
    /// How many threads decide (design 5.5): a model's decisions all run on
    /// one of them, behind one bounded queue. From 1 to
    /// [`Config::MAX_DECIDE_SHARDS`].
    pub decide_shards: usize,
    /// The election (design 5.3): where the lease lives, its timing and
    /// this replica's dialable address. The default is one replica that
    /// leads itself over a lease in its own process.
    pub election: ElectionConfig,
    /// The worker discovery provider to run while this replica leads. A
    /// build without the `discovery` feature runs none: the type has no
    /// value there, and the field is always `None`.
    pub discovery: Option<DiscoverySettings>,
}

/// Whether two listener addresses certainly contend when bound. Mapped IPv4
/// addresses are compared as IPv4. An IPv6 wildcard's coverage of IPv4 is a
/// host socket setting, so addresses in different families do not overlap here.
fn addresses_overlap(first: SocketAddr, second: SocketAddr) -> bool {
    if first.port() == 0 || first.port() != second.port() {
        return false;
    }
    let (a, b) = (first.ip().to_canonical(), second.ip().to_canonical());
    first == second
        || (a.is_ipv6() && b.is_ipv6() && (a.is_unspecified() || b.is_unspecified()))
        || (a.is_ipv4() && b.is_ipv4() && (a == b || a.is_unspecified() || b.is_unspecified()))
}

impl Config {
    /// The default settle window, the gateway's.
    pub const DEFAULT_SHUTDOWN_SETTLE: Duration = Duration::from_secs(5);
    /// The default grace period.
    pub const DEFAULT_SHUTDOWN_GRACE: Duration = Duration::from_secs(30);
    /// The default load poll interval, the gateway's.
    pub const DEFAULT_LOAD_POLL_INTERVAL: Duration = Duration::from_secs(10);
    /// The default overload thresholds, those of the gateway's two flags:
    /// the worker crate's two numbers, 8 queued requests and a usage of
    /// 0.8. They are this crate's own two defaults, by which the store
    /// judges under a binding whose block is on
    /// ([`OVERLOAD_WAITING_REQUESTS`](crate::document::spec::OVERLOAD_WAITING_REQUESTS),
    /// [`OVERLOAD_TOKEN_USAGE`](crate::document::spec::OVERLOAD_TOKEN_USAGE)):
    /// no document names a threshold.
    ///
    /// The usage is the worker crate's 64-bit number, as an engine reports
    /// it. This crate's constant is 32 bits wide: widened, it reads just
    /// above the 0.8 an engine reports, and a worker at the threshold would
    /// not be over it.
    pub const DEFAULT_WORKER_OVERLOAD: OverloadThresholds = OverloadThresholds {
        waiting_requests: Some(DEFAULT_WAITING_REQUESTS),
        token_usage: Some(DEFAULT_TOKEN_USAGE_CEILING),
    };
    /// The most deciding threads a controller starts.
    pub const MAX_DECIDE_SHARDS: usize = 256;

    /// The deciding threads a controller starts when its operator names no
    /// count: one a CPU available to the process, eight at most, the
    /// design's count for its busiest fleet.
    pub fn default_decide_shards() -> usize {
        thread::available_parallelism().map_or(1, |cpus| cpus.get().min(8))
    }

    /// Holds default overload thresholds to their bounds, each signal that
    /// states one: what [`Self::validate`] asks of
    /// [`Self::worker_overload`], and the command line of the value each of
    /// its two flags is typed with.
    pub(super) fn overload_in_bounds(overload: OverloadThresholds) -> Result<(), ConfigError> {
        // A worker at or above a threshold is vetoed: a count or a usage of
        // zero would veto a worker with nothing queued and nothing in use.
        // So would a count over the most a report states, which the
        // comparison reads as a negative one. A usage is a fraction, at most
        // one, and one that is no number compares with none.
        let out_of_count = |count: &usize| *count == 0 || i64::try_from(*count).is_err();
        if let Some(count) = overload.waiting_requests.filter(out_of_count) {
            return Err(ConfigError::OverloadWaitingRequests(count));
        }
        let out_of_range = |usage: f64| !usage.is_finite() || usage <= 0.0 || usage > 1.0;
        if overload.token_usage.is_some_and(out_of_range) {
            return Err(ConfigError::OverloadTokenUsageOutOfRange);
        }
        Ok(())
    }

    /// Refuses the configuration by the first field that is wrong.
    ///
    /// Two listeners on port `0` do not share an address: each takes its
    /// own ephemeral port at bind. Within one address family after mapped IPv4
    /// addresses are canonicalized, a wildcard listener shares its port with
    /// every listener it covers. The refusal names the later listener's address.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.controller_id.is_empty() {
            return Err(ConfigError::EmptyControllerId);
        }
        if !self
            .controller_id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
        {
            return Err(ConfigError::MalformedControllerId(
                self.controller_id.clone(),
            ));
        }
        if self.listeners.is_empty() {
            return Err(ConfigError::NoListeners);
        }
        // A listener is told apart by its name: in a refusal, in a log
        // line and by a setting that names it.
        let mut names: Vec<&str> = Vec::with_capacity(self.listeners.len());
        for listener in &self.listeners {
            if names.contains(&listener.name.as_str()) {
                return Err(ConfigError::ListenerNameShared(listener.name.clone()));
            }
            names.push(&listener.name);
        }
        let mut mounted: BTreeMap<Scope, &str> = BTreeMap::new();
        for listener in &self.listeners {
            if listener.scopes.is_empty() {
                return Err(ConfigError::ListenerWithoutScope(listener.name.clone()));
            }
            for &scope in &listener.scopes {
                if let Some(first) = mounted.insert(scope, &listener.name) {
                    return Err(ConfigError::ScopeMountedTwice {
                        scope,
                        first: first.to_owned(),
                        second: listener.name.clone(),
                    });
                }
            }
        }
        if let Some(scope) = Scope::ALL.into_iter().find(|s| !mounted.contains_key(s)) {
            return Err(ConfigError::ScopeUnmounted(scope));
        }
        let mut seen: Vec<(SocketAddr, &str)> = Vec::with_capacity(self.listeners.len() + 1);
        let named = self
            .listeners
            .iter()
            .map(|l| (l.addr, l.name.as_str()))
            .chain(std::iter::once((self.admin, "admin")));
        for (addr, name) in named {
            if let Some((_, first)) = seen.iter().find(|(a, _)| addresses_overlap(*a, addr)) {
                return Err(ConfigError::AddressShared {
                    addr,
                    first: (*first).to_owned(),
                    second: name.to_owned(),
                });
            }
            seen.push((addr, name));
        }
        if self.shutdown_grace.is_zero() {
            return Err(ConfigError::ZeroShutdownGrace);
        }
        if self.shutdown_settle > self.shutdown_grace / 2 {
            return Err(ConfigError::SettleOverHalfTheGrace {
                settle: self.shutdown_settle,
                grace: self.shutdown_grace,
            });
        }
        if self.load_poll_interval.is_zero() {
            return Err(ConfigError::ZeroLoadPollInterval);
        }
        Self::overload_in_bounds(self.worker_overload)?;
        if !(1..=Self::MAX_DECIDE_SHARDS).contains(&self.decide_shards) {
            return Err(ConfigError::DecideShards(self.decide_shards));
        }
        // The holder identity names the gateway listener: every scope is
        // mounted by now, so the listener is there.
        if let Some(gateway) = self
            .listeners
            .iter()
            .find(|listener| listener.scopes.contains(&Scope::Gateway))
        {
            self.election.validate(gateway.addr)?;
        }
        Ok(())
    }
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

/// A controller setting refused by name.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ConfigError {
    /// `controller_id` is empty.
    #[error("the controller id is empty")]
    EmptyControllerId,
    /// `controller_id` holds a character outside letters, digits, `.`, `_`
    /// and `-`.
    #[error("the controller id {0:?} has a character outside letters, digits, '.', '_' and '-'")]
    MalformedControllerId(String),
    /// No gRPC listener at all.
    #[error("no listener is configured")]
    NoListeners,
    /// A listener with an empty scope list.
    #[error("listener {0} mounts no scope")]
    ListenerWithoutScope(String),
    /// Two listeners with one name: a listener is told apart by its name,
    /// in a refusal, in a log line and by a setting that names it.
    #[error("two listeners are named {0}: a listener's name is its own")]
    ListenerNameShared(String),
    /// One scope on two listeners.
    #[error("scope {scope} is mounted on listeners {first} and {second}")]
    ScopeMountedTwice {
        /// The scope.
        scope: Scope,
        /// The listener that mounted it first.
        first: String,
        /// The listener that mounted it again.
        second: String,
    },
    /// A scope on no listener.
    #[error("scope {0} is mounted on no listener")]
    ScopeUnmounted(Scope),
    /// Two listeners, the admin listener included, on overlapping addresses;
    /// `addr` is the later listener's address.
    #[error("listeners {first} and {second} share the address {addr}")]
    AddressShared {
        /// The address.
        addr: SocketAddr,
        /// The listener that named it first.
        first: String,
        /// The listener that named it again.
        second: String,
    },
    /// A listener named to take identities as told that the configuration
    /// does not hold.
    #[error("no listener is named {0:?}: identities are taken as told on a listener of the configuration")]
    UnknownListener(String),
    /// A listener named to take identities as told that verifies its
    /// callers against a client CA: the verified subject is the principal,
    /// and a listener has one credential rule.
    #[error("listener {0} verifies its callers against a client CA: it takes no identity as told")]
    AsToldBesideClientCa(String),
    /// The replication listener of a replica that follows a leader, on
    /// port `0`. Replicas are configured alike, and a standby dials its
    /// leader at the port of its own replication listener: a number the
    /// host drew for one replica is no other's. Refused at the serve's
    /// start.
    #[error("listener {0} serves the replication scope on port 0, and this replica follows a leader: a standby dials its leader at its own replication port, so name this listener's port")]
    ReplicationPortUnnamed(String),
    /// A grace period of zero: a drain needs a bound above zero.
    #[error("the shutdown grace period is zero")]
    ZeroShutdownGrace,
    /// The load poll interval is zero.
    #[error("the load poll interval is zero")]
    ZeroLoadPollInterval,
    /// A default overload threshold of queued requests outside its range:
    /// zero, or over the most a load report states. A worker at or above a
    /// threshold is vetoed, and a count over a report's is read as a
    /// negative one: either would veto a worker with nothing queued.
    #[error("a default overload threshold of {0} queued requests: the count is from 1 to {max}, the most a load report states", max = i64::MAX)]
    OverloadWaitingRequests(usize),
    /// The default overload threshold of usage is no ratio above 0 and at
    /// most 1.
    #[error("the default overload threshold of usage is not above 0 and at most 1: a worker at or above it is vetoed")]
    OverloadTokenUsageOutOfRange,
    /// A count of deciding threads outside its range.
    #[error("{0} deciding threads: the count is from 1 to {max}", max = Config::MAX_DECIDE_SHARDS)]
    DecideShards(usize),
    /// A settle window the grace period cannot hold twice.
    #[error("the shutdown settle window of {settle:?} is over half the grace period of {grace:?}")]
    SettleOverHalfTheGrace {
        /// The window.
        settle: Duration,
        /// The grace period.
        grace: Duration,
    },
    /// The election's timing is out of order: a renew period above zero, a
    /// renew deadline longer than it, a duration longer than the deadline.
    #[error("the election's settings: {0}")]
    ElectionSettings(&'static str),
    /// A shared lease, a gateway listener on the wildcard address and no
    /// advertised address: the holder identity would name an address no
    /// client can dial.
    #[error("a shared lease needs an advertised address: the gateway listener binds every address, and the holder identity is what a client dials")]
    AdvertiseAddressRequired,
    /// An advertised address that is neither a host nor a `host:port`.
    #[error("the advertised address {0:?} is not a host or a host:port a client could dial")]
    MalformedAdvertiseAddress(String),
    /// An advertised address that is the wildcard, in any spelling: a
    /// listener's address, which a client dials as its own host.
    #[error(
        "the advertised address {0:?} is the wildcard: a client that dials it reaches its own host"
    )]
    WildcardAdvertiseAddress(String),
    /// An advertised host that ends in a number and is no address in its
    /// standard form.
    #[error("the advertised address {0:?} ends in a number: a resolver reads it as an address, and an address is written in its standard form")]
    NumericAdvertiseHost(String),
    /// A lease file with no path, or a `Lease` object with no name.
    #[error("the lease's place has an empty name")]
    EmptyLeaseName,
    /// A `Lease` object's namespace given empty: `None` takes the client
    /// configuration's own.
    #[error("the lease's namespace is empty: none given takes the client configuration's own")]
    EmptyLeaseNamespace,
    /// A `Lease` object stores its duration in whole seconds.
    #[error("the lease duration of {0:?} has a fraction of a second: a Lease object stores whole seconds")]
    LeaseDurationNotWholeSeconds(Duration),
    /// A certificate without its key, or a key without its certificate.
    #[error(
        "--tls-{given} without --tls-{missing}: the server identity is a certificate with its key"
    )]
    TlsHalfConfigured {
        /// The half that was given.
        given: &'static str,
        /// The half that was not.
        missing: &'static str,
    },
    /// A client authority for a scope while no server certificate is set.
    #[error("a client CA for the {0} scope without a server certificate: mutual TLS needs TLS")]
    ClientCaWithoutTls(Scope),
    /// Two scopes on one listener, one with a client authority and one
    /// without: a listener has one credential rule.
    #[error("listener {listener} mounts {with} with a client CA and {without} without one: give the two their own listeners")]
    ScopesDisagreeOnClientCa {
        /// The listener.
        listener: String,
        /// The scope with a client authority.
        with: Scope,
        /// The scope without one.
        without: Scope,
    },
    /// Two scopes on one listener naming different client authorities: a
    /// listener verifies against one, and taking the first would admit the
    /// second scope's clients on neither.
    #[error("listener {listener} mounts {first} and {second} with different client CAs: give the two their own listeners")]
    ScopesNameDifferentClientCas {
        /// The listener.
        listener: String,
        /// The scope whose authority was read first.
        first: Scope,
        /// The scope naming another.
        second: Scope,
    },
    /// A discovery flag was refused.
    #[error("{flag}: {reason}")]
    DiscoveryFlag {
        /// The flag that was refused.
        flag: &'static str,
        /// Why the flag was refused.
        reason: String,
    },
}

#[cfg(test)]
pub(crate) mod tests {
    use std::{
        net::{Ipv4Addr, Ipv6Addr, SocketAddr},
        path::PathBuf,
        time::Duration,
    };

    use super::{dialable, Config, ConfigError, Listener, OverloadThresholds, Scope, Tls};
    use crate::service::{
        controller::election::{ElectionConfig, LeasePlace},
        elector::ElectionSettings,
    };

    fn loopback(port: u16) -> SocketAddr {
        SocketAddr::from((Ipv4Addr::LOCALHOST, port))
    }

    fn listener(name: &str, port: u16, scopes: &[Scope]) -> Listener {
        Listener {
            name: name.to_owned(),
            addr: loopback(port),
            scopes: scopes.to_vec(),
            tls: None,
        }
    }

    /// The binary's default shape on ephemeral ports.
    pub(crate) fn valid() -> Config {
        Config {
            controller_id: "controller-0".to_owned(),
            listeners: vec![
                listener("gateway", 0, &[Scope::Gateway, Scope::Ingestion]),
                listener("control", 0, &[Scope::Control]),
                listener("replication", 0, &[Scope::Replication]),
            ],
            admin: loopback(0),
            shutdown_settle: Duration::from_millis(100),
            shutdown_grace: Duration::from_secs(2),
            load_poll_interval: Config::DEFAULT_LOAD_POLL_INTERVAL,
            worker_overload: Config::DEFAULT_WORKER_OVERLOAD,
            decide_shards: 2,
            election: ElectionConfig::default(),
            discovery: None,
        }
    }

    /// A setting names a listener by its name, so no two share one.
    #[test]
    fn two_listeners_of_one_name_are_refused() {
        let mut config = valid();
        if let Some(second) = config.listeners.get_mut(1) {
            second.name = "gateway".to_owned();
        }
        assert_eq!(
            config.validate(),
            Err(ConfigError::ListenerNameShared("gateway".to_owned()))
        );
    }

    /// A default overload threshold is held to its bounds, by name. A
    /// worker at or above a threshold is vetoed: a count of zero would veto
    /// a worker with nothing queued, and so would a count over the most a
    /// load report states, which is read as a negative one. A usage of zero
    /// would veto a worker with nothing in use. A usage is a fraction, at
    /// most one, and one that is no number compares with none.
    /// The ends are taken: a count of 1 and the most a report states, and a
    /// usage of 1 and the least above 0. A signal that states none has no
    /// default and is no refusal, one signal or both.
    #[test]
    fn a_default_overload_threshold_is_one_a_worker_can_stay_under() {
        let stating = |waiting_requests, token_usage| OverloadThresholds {
            waiting_requests,
            token_usage,
        };
        // The most queued requests a load report states, and the least
        // usage above zero.
        let most = usize::try_from(i64::MAX).unwrap_or(usize::MAX);
        let least = f64::from_bits(1);
        let mut config = valid();
        assert_eq!(config.worker_overload, stating(Some(8), Some(0.8)));
        for taken in [
            stating(Some(1), Some(1.0)),
            stating(Some(most), Some(least)),
            stating(None, Some(0.5)),
            stating(Some(3), None),
            stating(None, None),
        ] {
            config.worker_overload = taken;
            assert_eq!(config.validate(), Ok(()), "{taken:?}");
        }
        for count in [Some(0), most.checked_add(1)].into_iter().flatten() {
            for usage in [None, Some(0.5)] {
                config.worker_overload = stating(Some(count), usage);
                assert_eq!(
                    config.validate(),
                    Err(ConfigError::OverloadWaitingRequests(count)),
                    "{count} beside {usage:?}"
                );
            }
        }
        for usage in [0.0, -0.5, 1.0 + f64::EPSILON, 1.5, f64::INFINITY, f64::NAN] {
            for waiting in [None, Some(1)] {
                config.worker_overload = stating(waiting, Some(usage));
                assert_eq!(
                    config.validate(),
                    Err(ConfigError::OverloadTokenUsageOutOfRange),
                    "{usage} beside {waiting:?}"
                );
            }
        }
        assert_eq!(
            ConfigError::OverloadWaitingRequests(0).to_string(),
            "a default overload threshold of 0 queued requests: the count is from 1 to 9223372036854775807, the most a load report states"
        );
        assert_eq!(
            ConfigError::OverloadTokenUsageOutOfRange.to_string(),
            "the default overload threshold of usage is not above 0 and at most 1: a worker at or above it is vetoed"
        );
    }

    #[test]
    fn the_deciding_threads_are_counted_from_one_to_their_ceiling() {
        let mut config = valid();
        for count in [1, Config::MAX_DECIDE_SHARDS] {
            config.decide_shards = count;
            assert_eq!(config.validate(), Ok(()), "{count}");
        }
        for count in [0, Config::MAX_DECIDE_SHARDS + 1] {
            config.decide_shards = count;
            assert_eq!(
                config.validate(),
                Err(ConfigError::DecideShards(count)),
                "{count}"
            );
        }
        assert_eq!(
            ConfigError::DecideShards(0).to_string(),
            "0 deciding threads: the count is from 1 to 256"
        );
        assert!((1..=8).contains(&Config::default_decide_shards()));
    }

    #[test]
    fn the_default_shape_is_accepted_with_or_without_tls() {
        let mut config = valid();
        assert_eq!(config.validate(), Ok(()));
        config.listeners[0].tls = Some(Tls {
            cert: PathBuf::from("server.pem"),
            key: PathBuf::from("server.key"),
            client_ca: Some(PathBuf::from("gateways-ca.pem")),
        });
        assert_eq!(config.validate(), Ok(()));
    }

    #[test]
    fn the_controller_id_is_one_plain_token() {
        let mut config = valid();
        config.controller_id = String::new();
        assert_eq!(config.validate(), Err(ConfigError::EmptyControllerId));
        for bad in ["a b", "a@b", "a/b", "a:b"] {
            config.controller_id = bad.to_owned();
            assert_eq!(
                config.validate(),
                Err(ConfigError::MalformedControllerId(bad.to_owned())),
                "{bad}"
            );
        }
        for good in ["controller-0", "pod.ns_1", "A-Z.0-9"] {
            config.controller_id = good.to_owned();
            assert_eq!(config.validate(), Ok(()), "{good}");
        }
    }

    #[test]
    fn every_scope_is_mounted_exactly_once() {
        let mut config = valid();
        config.listeners.clear();
        assert_eq!(config.validate(), Err(ConfigError::NoListeners));

        let mut config = valid();
        config.listeners[1].scopes.clear();
        assert_eq!(
            config.validate(),
            Err(ConfigError::ListenerWithoutScope("control".to_owned()))
        );

        let mut config = valid();
        config.listeners[2].scopes.push(Scope::Control);
        assert_eq!(
            config.validate(),
            Err(ConfigError::ScopeMountedTwice {
                scope: Scope::Control,
                first: "control".to_owned(),
                second: "replication".to_owned(),
            })
        );

        let mut config = valid();
        config.listeners[0].scopes = vec![Scope::Gateway];
        assert_eq!(
            config.validate(),
            Err(ConfigError::ScopeUnmounted(Scope::Ingestion))
        );

        let mut config = valid();
        config
            .listeners
            .push(listener("relays", 0, &[Scope::Ingestion]));
        config.listeners[0].scopes = vec![Scope::Gateway];
        assert_eq!(config.validate(), Ok(()), "the relays split onto a fourth");
    }

    #[test]
    fn two_listeners_never_share_a_real_address() {
        let mut config = valid();
        config.listeners[0].addr = loopback(30100);
        config.listeners[2].addr = loopback(30100);
        assert_eq!(
            config.validate(),
            Err(ConfigError::AddressShared {
                addr: loopback(30100),
                first: "gateway".to_owned(),
                second: "replication".to_owned(),
            })
        );

        let mut config = valid();
        config.listeners[1].addr = loopback(30103);
        config.admin = loopback(30103);
        assert_eq!(
            config.validate(),
            Err(ConfigError::AddressShared {
                addr: loopback(30103),
                first: "control".to_owned(),
                second: "admin".to_owned(),
            })
        );

        let config = valid();
        assert!(
            config.listeners.iter().all(|l| l.addr.port() == 0),
            "ephemeral ports are not shared addresses"
        );
        assert_eq!(config.validate(), Ok(()));
    }

    #[test]
    fn a_wildcard_and_a_specific_listener_never_share_a_real_port() {
        let wildcard = SocketAddr::from((Ipv4Addr::UNSPECIFIED, 30100));
        let specific = loopback(30100);

        let mut config = valid();
        config.listeners[0].addr = wildcard;
        config.listeners[1].addr = specific;
        assert_eq!(
            config.validate(),
            Err(ConfigError::AddressShared {
                addr: specific,
                first: "gateway".to_owned(),
                second: "control".to_owned(),
            })
        );

        let mut config = valid();
        config.listeners[0].addr = specific;
        config.listeners[1].addr = wildcard;
        assert_eq!(
            config.validate(),
            Err(ConfigError::AddressShared {
                addr: wildcard,
                first: "gateway".to_owned(),
                second: "control".to_owned(),
            })
        );

        let mut config = valid();
        config.listeners[1].addr = wildcard;
        config.admin = specific;
        assert_eq!(
            config.validate(),
            Err(ConfigError::AddressShared {
                addr: specific,
                first: "control".to_owned(),
                second: "admin".to_owned(),
            })
        );

        let mut config = valid();
        config.listeners[1].addr = specific;
        config.admin = wildcard;
        assert_eq!(
            config.validate(),
            Err(ConfigError::AddressShared {
                addr: wildcard,
                first: "control".to_owned(),
                second: "admin".to_owned(),
            })
        );

        let v6_wildcard = SocketAddr::from((Ipv6Addr::UNSPECIFIED, 30100));
        let v6_specific = SocketAddr::from((Ipv6Addr::LOCALHOST, 30100));
        let mut config = valid();
        config.listeners[0].addr = v6_wildcard;
        config.listeners[1].addr = v6_specific;
        assert_eq!(
            config.validate(),
            Err(ConfigError::AddressShared {
                addr: v6_specific,
                first: "gateway".to_owned(),
                second: "control".to_owned(),
            })
        );
    }

    #[test]
    fn wildcard_listeners_overlap_every_address_they_cover() {
        let specific = loopback(30100);

        let mapped_wildcard = SocketAddr::from((Ipv4Addr::UNSPECIFIED.to_ipv6_mapped(), 30100));
        let mut config = valid();
        config.listeners[0].addr = mapped_wildcard;
        config.listeners[1].addr = specific;
        assert_eq!(
            config.validate(),
            Err(ConfigError::AddressShared {
                addr: specific,
                first: "gateway".to_owned(),
                second: "control".to_owned(),
            })
        );

        let mapped_specific = SocketAddr::from((Ipv4Addr::LOCALHOST.to_ipv6_mapped(), 30100));
        let mut config = valid();
        config.listeners[0].addr = mapped_wildcard;
        config.listeners[1].addr = mapped_specific;
        assert_eq!(
            config.validate(),
            Err(ConfigError::AddressShared {
                addr: mapped_specific,
                first: "gateway".to_owned(),
                second: "control".to_owned(),
            })
        );
    }

    #[test]
    fn listener_addresses_that_do_not_overlap_are_accepted() {
        let mut config = valid();
        config.listeners[0].addr = SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0));
        config.listeners[1].addr = SocketAddr::from((Ipv4Addr::new(10, 0, 0, 1), 30100));
        config.listeners[2].addr = loopback(30100);
        config.admin = SocketAddr::from((Ipv6Addr::LOCALHOST, 30100));
        assert_eq!(
            config.validate(),
            Ok(()),
            "port 0, two specific addresses and two families share no address"
        );

        let mut config = valid();
        config.listeners[1].addr = SocketAddr::from((Ipv6Addr::UNSPECIFIED, 30100));
        config.listeners[2].addr = loopback(30100);
        assert_eq!(
            config.validate(),
            Ok(()),
            "an IPv6 wildcard is not known to cover the IPv4 family"
        );
    }

    #[test]
    fn the_shutdown_bounds_hold_each_other() {
        let mut config = valid();
        config.shutdown_grace = Duration::ZERO;
        assert_eq!(config.validate(), Err(ConfigError::ZeroShutdownGrace));

        let mut config = valid();
        config.shutdown_grace = Duration::from_secs(8);
        config.shutdown_settle = Duration::from_secs(5);
        assert_eq!(
            config.validate(),
            Err(ConfigError::SettleOverHalfTheGrace {
                settle: Duration::from_secs(5),
                grace: Duration::from_secs(8),
            })
        );
        config.shutdown_settle = Duration::from_secs(4);
        assert_eq!(config.validate(), Ok(()), "half the grace is the bound");
    }

    #[test]
    fn the_elections_settings_keep_their_order() {
        let mut config = valid();
        assert_eq!(config.election.settings, ElectionSettings::default());
        config.election.settings.renew_period = Duration::ZERO;
        assert_eq!(
            config.validate(),
            Err(ConfigError::ElectionSettings("the renew period is zero"))
        );
        let mut config = valid();
        config.election.settings.renew_deadline = Duration::from_secs(1);
        assert_eq!(
            config.validate(),
            Err(ConfigError::ElectionSettings(
                "the renew deadline is not longer than the renew period"
            ))
        );
        let mut config = valid();
        config.election.settings.duration = Duration::from_secs(2);
        assert_eq!(
            config.validate(),
            Err(ConfigError::ElectionSettings(
                "the duration is not longer than the renew deadline"
            ))
        );
    }

    #[test]
    fn a_shared_lease_needs_an_address_a_client_can_dial() {
        // A lone controller needs none, on any listener.
        let mut config = valid();
        config.listeners[0].addr = SocketAddr::from((Ipv4Addr::UNSPECIFIED, 30100));
        assert_eq!(config.validate(), Ok(()));

        // A shared lease behind the wildcard address is refused by name.
        config.election.lease = LeasePlace::File(PathBuf::from("/run/smg/lease"));
        assert_eq!(
            config.validate(),
            Err(ConfigError::AdvertiseAddressRequired)
        );
        // The IPv4 wildcard mapped into IPv6 is a wildcard listener too.
        config.listeners[0].addr =
            SocketAddr::from((Ipv4Addr::UNSPECIFIED.to_ipv6_mapped(), 30100));
        assert_eq!(
            config.validate(),
            Err(ConfigError::AdvertiseAddressRequired)
        );
        // ... until an address is advertised, or the listener binds one.
        config.election.advertise = Some("10.0.0.7".to_owned());
        assert_eq!(config.validate(), Ok(()));
        config.election.advertise = None;
        config.listeners[0].addr = loopback(30100);
        assert_eq!(config.validate(), Ok(()));

        for bad in [
            "",
            "a b",
            "controller@host",
            "host:0",
            "host:port",
            ":30100",
            "a/b",
        ] {
            config.election.advertise = Some(bad.to_owned());
            assert_eq!(
                config.validate(),
                Err(ConfigError::MalformedAdvertiseAddress(bad.to_owned())),
                "{bad:?}"
            );
        }
        // The wildcard is a listener's address, never a client's, and a
        // host that ends in a number is an address to a resolver.
        for wildcard in ["0.0.0.0", "[::]:30100"] {
            config.election.advertise = Some(wildcard.to_owned());
            assert_eq!(
                config.validate(),
                Err(ConfigError::WildcardAdvertiseAddress(wildcard.to_owned())),
                "{wildcard:?}"
            );
        }
        config.election.advertise = Some("0:30100".to_owned());
        assert_eq!(
            config.validate(),
            Err(ConfigError::NumericAdvertiseHost("0:30100".to_owned()))
        );
        for good in [
            "10.0.0.7",
            "10.0.0.7:30100",
            "fd00::7",
            "[fd00::7]",
            "[fd00::7]:30100",
            "controller-0.policy.svc",
            "controller-0.policy.svc:30100",
        ] {
            config.election.advertise = Some(good.to_owned());
            assert_eq!(config.validate(), Ok(()), "{good}");
        }

        config.election.advertise = None;
        config.election.lease = LeasePlace::File(PathBuf::new());
        assert_eq!(config.validate(), Err(ConfigError::EmptyLeaseName));
    }

    #[cfg(feature = "kubernetes")]
    #[test]
    fn a_lease_object_takes_a_name_and_whole_seconds() {
        let object = |namespace: Option<&str>, name: &str| LeasePlace::Kubernetes {
            namespace: namespace.map(str::to_owned),
            name: name.to_owned(),
        };
        let mut config = valid();
        config.election.lease = object(None, "smg-policy");
        assert_eq!(config.validate(), Ok(()));
        config.election.lease = object(Some("policy"), "smg-policy");
        assert_eq!(config.validate(), Ok(()));

        config.election.lease = object(None, "");
        assert_eq!(config.validate(), Err(ConfigError::EmptyLeaseName));
        config.election.lease = object(Some(""), "smg-policy");
        assert_eq!(config.validate(), Err(ConfigError::EmptyLeaseNamespace));

        // The object stores whole seconds: a fraction is refused here, not
        // at the first write.
        config.election.lease = object(None, "smg-policy");
        config.election.settings = ElectionSettings {
            duration: Duration::from_millis(2500),
            renew_period: Duration::from_millis(500),
            renew_deadline: Duration::from_millis(1500),
        };
        assert_eq!(
            config.validate(),
            Err(ConfigError::LeaseDurationNotWholeSeconds(
                Duration::from_millis(2500)
            ))
        );
        // A file takes the same settings.
        config.election.lease = LeasePlace::File(PathBuf::from("/run/smg/lease"));
        assert_eq!(config.validate(), Ok(()));
    }

    #[test]
    fn a_scope_names_itself_everywhere() {
        for scope in Scope::ALL {
            assert_eq!(scope.to_string(), scope.as_str());
        }
        assert_eq!(
            ConfigError::ScopeUnmounted(Scope::Replication).to_string(),
            "scope replication is mounted on no listener"
        );
    }

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
}
