//! What an operator hands the controller at start (design 5: the addresses,
//! the credentials and the shutdown settings are read at start; the routing
//! policy is the spec's and never here): the listeners with their scopes and
//! TLS, the admin listener, the controller's identity and the shutdown
//! bounds, validated as a whole before anything binds so a refusal names
//! the field it refuses.

use std::{collections::BTreeMap, fmt, net::SocketAddr, path::PathBuf, time::Duration};

/// A credentialed group of the controller's calls (design 8). Each scope is
/// mounted on exactly one listener, alone or beside others; a listener's
/// credential rule is its scopes' rule, so scopes with different rules go
/// on different listeners.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
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
    /// [`Controller::addresses`](super::Controller::addresses).
    pub addr: SocketAddr,
    /// The scopes it serves; every scope is mounted once across the
    /// listeners.
    pub scopes: Vec<Scope>,
    /// Its TLS, or plaintext when `None` (a loopback, or a mesh that
    /// terminates TLS in front of the process).
    pub tls: Option<Tls>,
}

/// The controller's start-up configuration.
#[derive(Clone, Debug, Eq, PartialEq)]
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
    /// gateway's, which waits for generations to finish.
    pub shutdown_grace: Duration,
}

impl Config {
    /// The default settle window, the gateway's.
    pub const DEFAULT_SHUTDOWN_SETTLE: Duration = Duration::from_secs(5);
    /// The default grace period.
    pub const DEFAULT_SHUTDOWN_GRACE: Duration = Duration::from_secs(30);

    /// Refuses the configuration by the first field that is wrong.
    ///
    /// Two listeners on port `0` do not share an address: each takes its
    /// own ephemeral port at bind.
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
            if addr.port() != 0 {
                if let Some((_, first)) = seen.iter().find(|(a, _)| *a == addr) {
                    return Err(ConfigError::AddressShared {
                        addr,
                        first: (*first).to_owned(),
                        second: name.to_owned(),
                    });
                }
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
        Ok(())
    }
}

/// A configuration refused, by name. The last four come from the binary's
/// command line, which maps one server identity and one client authority
/// per scope onto the listeners.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
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
    /// Two listeners, the admin listener included, on one address.
    #[error("listeners {first} and {second} share the address {addr}")]
    AddressShared {
        /// The address.
        addr: SocketAddr,
        /// The listener that named it first.
        first: String,
        /// The listener that named it again.
        second: String,
    },
    /// A grace period of zero: a drain needs a bound above zero.
    #[error("the shutdown grace period is zero")]
    ZeroShutdownGrace,
    /// A settle window the grace period cannot hold twice.
    #[error("the shutdown settle window of {settle:?} is over half the grace period of {grace:?}")]
    SettleOverHalfTheGrace {
        /// The window.
        settle: Duration,
        /// The grace period.
        grace: Duration,
    },
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
}

#[cfg(test)]
mod tests {
    use std::{
        net::{Ipv4Addr, SocketAddr},
        path::PathBuf,
        time::Duration,
    };

    use super::{Config, ConfigError, Listener, Scope, Tls};

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
    fn valid() -> Config {
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
        }
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
    fn a_scope_names_itself_everywhere() {
        for scope in Scope::ALL {
            assert_eq!(scope.to_string(), scope.as_str());
        }
        assert_eq!(
            ConfigError::ScopeUnmounted(Scope::Replication).to_string(),
            "scope replication is mounted on no listener"
        );
    }
}
