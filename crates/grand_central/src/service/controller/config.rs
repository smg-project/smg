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
    /// The unary operator calls and `WatchRegime`: the control-plane keys and
    /// roles, `user` reads and `admin` writes, every call audited.
    Control,
    /// `Replicate`: the standbys' call, a replica credential.
    Replication,
}

impl Scope {
    /// Every scope, in the order the design lists them.
    pub const ALL: [Scope; 3] = [Scope::Gateway, Scope::Control, Scope::Replication];

    /// The scope's name on the command line, in log lines and in refusals.
    pub const fn as_str(self) -> &'static str {
        match self {
            Scope::Gateway => "gateway",
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
    /// after its flag (`gateway`, `control`, `replication`).
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
    /// The gRPC listeners. The binary mounts each scope on a listener of
    /// its own.
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

/// The number of [`Config::MAX_DECIDE_SHARDS`], named by the refusal too:
/// the refusal reads it here, not through `Config`, so `ConfigError` does
/// not know the record whose check returns it.
const DECIDE_SHARDS_CEILING: usize = 256;

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
    /// The most deciding threads a controller starts: 256.
    pub const MAX_DECIDE_SHARDS: usize = DECIDE_SHARDS_CEILING;

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
    #[error("{0} deciding threads: the count is from 1 to {max}", max = DECIDE_SHARDS_CEILING)]
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
#[path = "../../../tests/unit/service/controller/config/tests.rs"]
pub(crate) mod tests;
