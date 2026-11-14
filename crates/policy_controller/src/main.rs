//! The standalone policy controller (design 2: the `remote` mode of the
//! policy engine). The command line is the operator's surface: the
//! controller's identity, its listeners, their TLS, the shutdown bounds and
//! the logging; the routing policy itself is the spec's and arrives over the
//! control scope.

use std::{
    ffi::OsStr, future::pending, io::IsTerminal, net::SocketAddr, path::PathBuf, process::ExitCode,
    time::Duration,
};

use clap::Parser;
use smg_policy_controller::controller::{
    Config, ConfigError, Controller, Listener, Scope, Tls, VERSION,
};
use tokio::{
    runtime::{Builder, Runtime},
    signal::unix::{signal, SignalKind},
};
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

/// The controller's ports sit in one block, apart from the gateway's data
/// (30000) and metrics (29000) ports.
const DEFAULT_GATEWAY_LISTEN: &str = "0.0.0.0:30100";
const DEFAULT_CONTROL_LISTEN: &str = "0.0.0.0:30101";
const DEFAULT_REPLICATION_LISTEN: &str = "0.0.0.0:30102";
const DEFAULT_ADMIN_LISTEN: &str = "0.0.0.0:30103";

#[derive(Debug, Parser)]
#[command(
    name = VERSION.name,
    version = VERSION.version,
    about = "The SMG policy controller: the owner of the routing data, serving decisions to the gateways"
)]
struct Cli {
    // ==================== Identity ====================
    /// This controller's identity: the holder of the leases it takes and the
    /// name on its log lines (letters, digits, '.', '_' and '-'); under
    /// Kubernetes the pod name, read from HOSTNAME
    #[arg(long, env = "HOSTNAME", help_heading = "Identity")]
    controller_id: String,

    // ==================== Listeners ====================
    /// The gateways' listener: Session, Decide and WatchWorkers, and the
    /// relays' PushLoad and PushKvEvents unless --ingestion-listen is set
    #[arg(long, default_value = DEFAULT_GATEWAY_LISTEN, help_heading = "Listeners")]
    gateway_listen: SocketAddr,

    /// A listener of the relays' own for PushLoad and PushKvEvents
    #[arg(long, help_heading = "Listeners")]
    ingestion_listen: Option<SocketAddr>,

    /// The operators' listener: the unary calls and WatchRegime
    #[arg(long, default_value = DEFAULT_CONTROL_LISTEN, help_heading = "Listeners")]
    control_listen: SocketAddr,

    /// The standbys' listener: Replicate
    #[arg(long, default_value = DEFAULT_REPLICATION_LISTEN, help_heading = "Listeners")]
    replication_listen: SocketAddr,

    /// The HTTP admin listener: /liveness, /readiness, /health, /version
    /// and /metrics
    #[arg(long, default_value = DEFAULT_ADMIN_LISTEN, help_heading = "Listeners")]
    admin_listen: SocketAddr,

    // ==================== TLS ====================
    /// The server certificate chain (PEM) of every gRPC listener; comes with
    /// --tls-key
    #[arg(long, help_heading = "TLS")]
    tls_cert: Option<PathBuf>,

    /// The private key (PEM) of --tls-cert
    #[arg(long, help_heading = "TLS")]
    tls_key: Option<PathBuf>,

    /// The authority the gateways' client certificates chain to: mutual TLS
    /// on the gateway scope's listener
    #[arg(long, help_heading = "TLS")]
    gateway_client_ca: Option<PathBuf>,

    /// The authority the relays' client certificates chain to: mutual TLS on
    /// the ingestion scope's listener (its own with --ingestion-listen, else
    /// the gateways' listener, whose authority must then be the same file)
    #[arg(long, help_heading = "TLS")]
    ingestion_client_ca: Option<PathBuf>,

    /// The authority the operators' client certificates chain to: mutual
    /// TLS on the control scope's listener
    #[arg(long, help_heading = "TLS")]
    control_client_ca: Option<PathBuf>,

    /// The authority the standbys' client certificates chain to: mutual TLS
    /// on the replication scope's listener
    #[arg(long, help_heading = "TLS")]
    replication_client_ca: Option<PathBuf>,

    // ==================== Load Monitoring ====================
    /// Seconds between two load polls of an engine whose stream is not
    /// serving it, and the freshness bound a pushed record must meet to
    /// stand in for the poll; the gateway's flag of the same name
    #[arg(long, default_value_t = Config::DEFAULT_LOAD_POLL_INTERVAL.as_secs(), help_heading = "Load Monitoring")]
    load_monitor_interval: u64,

    // ==================== Shutdown ====================
    /// Seconds the listeners keep accepting after readiness fails, so the
    /// probes see the change first; at most half the grace period
    #[arg(long, default_value_t = Config::DEFAULT_SHUTDOWN_SETTLE.as_secs(), help_heading = "Shutdown")]
    shutdown_settle_secs: u64,

    /// Seconds a shutdown may take in all, the settle window included; the
    /// listeners close with their streams when it runs out
    #[arg(long, default_value_t = Config::DEFAULT_SHUTDOWN_GRACE.as_secs(), help_heading = "Shutdown")]
    shutdown_grace_period_secs: u64,

    // ==================== Runtime ====================
    /// Explicit async runtime worker-thread count, from 1 to 4096 (a
    /// refusal names the ceiling); a count above 4 x the CPUs available to
    /// the process starts with a warning. Leave unset for tokio's default
    /// (`available_parallelism()`), which honors the cgroup CPU quota
    #[arg(long, value_parser = clap::value_parser!(u16).range(1..=4096), help_heading = "Runtime")]
    runtime_worker_threads: Option<u16>,

    // ==================== Logging ====================
    /// Set the logging level of this binary's own crates; RUST_LOG, when
    /// set, replaces it whole
    #[arg(long, default_value = "info", value_parser = ["debug", "info", "warn", "error"], help_heading = "Logging")]
    log_level: String,

    /// Output logs as JSON
    #[arg(long, default_value_t = false, help_heading = "Logging")]
    log_json: bool,
}

impl Cli {
    /// The listeners the flags describe: one server identity for every gRPC
    /// listener, one client authority per scope, one credential rule per
    /// listener.
    fn config(&self) -> Result<Config, ConfigError> {
        let identity = match (&self.tls_cert, &self.tls_key) {
            (Some(cert), Some(key)) => Some((cert, key)),
            (None, None) => None,
            (Some(_), None) => {
                return Err(ConfigError::TlsHalfConfigured {
                    given: "cert",
                    missing: "key",
                })
            }
            (None, Some(_)) => {
                return Err(ConfigError::TlsHalfConfigured {
                    given: "key",
                    missing: "cert",
                })
            }
        };
        let client_ca = |scope: Scope| -> Option<&PathBuf> {
            match scope {
                Scope::Gateway => self.gateway_client_ca.as_ref(),
                Scope::Ingestion => self.ingestion_client_ca.as_ref(),
                Scope::Control => self.control_client_ca.as_ref(),
                Scope::Replication => self.replication_client_ca.as_ref(),
            }
        };
        if identity.is_none() {
            if let Some(scope) = Scope::ALL.into_iter().find(|s| client_ca(*s).is_some()) {
                return Err(ConfigError::ClientCaWithoutTls(scope));
            }
        }

        let mut mounts: Vec<(&str, SocketAddr, Vec<Scope>)> = Vec::with_capacity(4);
        match self.ingestion_listen {
            Some(addr) => {
                mounts.push(("gateway", self.gateway_listen, vec![Scope::Gateway]));
                mounts.push(("ingestion", addr, vec![Scope::Ingestion]));
            }
            None => mounts.push((
                "gateway",
                self.gateway_listen,
                vec![Scope::Gateway, Scope::Ingestion],
            )),
        }
        mounts.push(("control", self.control_listen, vec![Scope::Control]));
        mounts.push((
            "replication",
            self.replication_listen,
            vec![Scope::Replication],
        ));

        let mut listeners = Vec::with_capacity(mounts.len());
        for (name, addr, scopes) in mounts {
            let tls = match identity {
                None => None,
                Some((cert, key)) => {
                    // One credential rule per listener: every scope on it
                    // names the same authority, or none of them does.
                    if let Some(first) = scopes.iter().copied().find(|s| client_ca(*s).is_some()) {
                        let authority = client_ca(first);
                        if let Some(second) =
                            scopes.iter().copied().find(|s| client_ca(*s) != authority)
                        {
                            return Err(if client_ca(second).is_none() {
                                ConfigError::ScopesDisagreeOnClientCa {
                                    listener: name.to_owned(),
                                    with: first,
                                    without: second,
                                }
                            } else {
                                ConfigError::ScopesNameDifferentClientCas {
                                    listener: name.to_owned(),
                                    first,
                                    second,
                                }
                            });
                        }
                    }
                    Some(Tls {
                        cert: cert.clone(),
                        key: key.clone(),
                        client_ca: scopes.first().and_then(|s| client_ca(*s)).cloned(),
                    })
                }
            };
            listeners.push(Listener {
                name: name.to_owned(),
                addr,
                scopes,
                tls,
            });
        }
        Ok(Config {
            controller_id: self.controller_id.clone(),
            listeners,
            admin: self.admin_listen,
            shutdown_settle: Duration::from_secs(self.shutdown_settle_secs),
            shutdown_grace: Duration::from_secs(self.shutdown_grace_period_secs),
            load_poll_interval: Duration::from_secs(self.load_monitor_interval),
        })
    }
}

/// `RUST_LOG` wins when set, as in the gateway and the servicer; otherwise
/// the flag sets this binary's own crates and the dependencies stay at
/// `warn`. The text log is colored by the gateway's rule
/// ([`color_for_stdout`]).
fn init_logging(level: &str, json: bool) {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(format!("warn,smg_policy_controller={level}")));
    let subscriber = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_ansi(color_for_stdout());
    if json {
        subscriber.json().init();
    } else {
        subscriber.init();
    }
}

/// Whether the text log is colored, by the gateway's rule and its three
/// inputs: ANSI escapes go to a terminal, not into a pipe or a file (a
/// redirected start, a container runtime collecting stdout), where they are
/// bytes on every line that nothing renders and a consumer has to strip.
/// `NO_COLOR` set to a non-empty value turns color off on a terminal too;
/// `FORCE_COLOR` or `CLICOLOR_FORCE` set to a non-empty value other than `0`
/// turns it on into a pipe too (the no-color.org and clicolors conventions,
/// where `0` means "do not force"; an explicit ask wins).
fn color_for_stdout() -> bool {
    color_for(
        std::io::stdout().is_terminal(),
        set(std::env::var_os("NO_COLOR").as_deref()),
        truthy(std::env::var_os("FORCE_COLOR").as_deref())
            || truthy(std::env::var_os("CLICOLOR_FORCE").as_deref()),
    )
}

/// The decision behind [`color_for_stdout`]: forced wins over refused, and
/// refused over the terminal.
fn color_for(terminal: bool, refused: bool, forced: bool) -> bool {
    forced || (terminal && !refused)
}

/// Whether an environment variable's value counts as set: present and
/// non-empty (an empty value counts as unset, as the conventions say).
fn set(value: Option<&OsStr>) -> bool {
    value.is_some_and(|value| !value.is_empty())
}

/// Whether an environment variable's value forces: present, non-empty and
/// not `0`, the clicolors reading of a force variable.
fn truthy(value: Option<&OsStr>) -> bool {
    value.is_some_and(|value| !value.is_empty() && value != "0")
}

/// Resolves on SIGTERM (the orchestrator's stop) or SIGINT (a terminal's).
/// A handler that cannot be installed is logged and its signal waits
/// forever, so the other one still stops the process; a failure of both
/// leaves the controller serving until it is killed.
async fn signals() {
    let mut term = match signal(SignalKind::terminate()) {
        Ok(term) => Some(term),
        Err(error) => {
            warn!(%error, "no SIGTERM handler; SIGINT alone stops the controller");
            None
        }
    };
    let term = async {
        match term.as_mut() {
            Some(term) => {
                term.recv().await;
            }
            None => pending::<()>().await,
        }
    };
    let interrupt = async {
        if let Err(error) = tokio::signal::ctrl_c().await {
            warn!(%error, "no SIGINT handler; SIGTERM alone stops the controller");
            pending::<()>().await;
        }
    };
    tokio::select! {
        () = term => info!("SIGTERM received"),
        () = interrupt => info!("SIGINT received"),
    }
}

/// The async runtime: tokio's default (one worker per CPU available to the
/// process), or the explicit count with the gateway's oversubscription
/// warning above four workers per CPU.
fn runtime(worker_threads: Option<u16>) -> std::io::Result<Runtime> {
    match worker_threads {
        Some(threads) => {
            let threads = usize::from(threads);
            if let Ok(cpus) = std::thread::available_parallelism() {
                if threads > 4 * cpus.get() {
                    warn!(
                        threads,
                        cpus = cpus.get(),
                        "runtime worker threads above four per CPU: a worker beyond the CPUs only waits for one"
                    );
                }
            }
            Builder::new_multi_thread()
                .worker_threads(threads)
                .enable_all()
                .build()
        }
        None => Runtime::new(),
    }
}

/// The start-up sequence: parse, log, configure, then the runtime, built
/// only now so a refusal costs no threads; bind and serve on it.
fn main() -> ExitCode {
    let cli = Cli::parse();
    init_logging(&cli.log_level, cli.log_json);
    let config = match cli.config() {
        Ok(config) => config,
        Err(error) => {
            error!(%error, "configuration refused");
            return ExitCode::FAILURE;
        }
    };
    let runtime = match runtime(cli.runtime_worker_threads) {
        Ok(runtime) => runtime,
        Err(error) => {
            error!(%error, "runtime refused");
            return ExitCode::FAILURE;
        }
    };
    runtime.block_on(async {
        let controller = match Controller::bind(config).await {
            Ok(controller) => controller,
            Err(error) => {
                error!(%error, "start refused");
                return ExitCode::FAILURE;
            }
        };
        match controller.serve(signals()).await {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                error!(%error, "stopped on an error");
                ExitCode::FAILURE
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use std::{
        ffi::OsStr,
        path::{Path, PathBuf},
        time::Duration,
    };

    use clap::Parser;
    use smg_policy_controller::controller::{Config, ConfigError, Scope};

    use super::{color_for, set, truthy, Cli};

    fn parse(args: &[&str]) -> Cli {
        let mut argv = vec!["smg-policy-controller", "--controller-id", "controller-0"];
        argv.extend_from_slice(args);
        Cli::try_parse_from(argv).expect("the flags parse")
    }

    fn config(args: &[&str]) -> Result<Config, ConfigError> {
        parse(args).config()
    }

    const IDENTITY: [&str; 4] = ["--tls-cert", "server.pem", "--tls-key", "server.key"];

    fn with_identity<'a>(args: &[&'a str]) -> Vec<&'a str> {
        let mut all = IDENTITY.to_vec();
        all.extend_from_slice(args);
        all
    }

    fn authorities(config: &Config) -> Vec<Option<PathBuf>> {
        config
            .listeners
            .iter()
            .map(|l| l.tls.as_ref().and_then(|t| t.client_ca.clone()))
            .collect()
    }

    #[test]
    fn the_defaults_mount_the_relays_beside_the_gateways() {
        let config = config(&[]).expect("the defaults");
        assert_eq!(config.controller_id, "controller-0");
        let mounts: Vec<(&str, &[Scope])> = config
            .listeners
            .iter()
            .map(|l| (l.name.as_str(), l.scopes.as_slice()))
            .collect();
        assert_eq!(
            mounts,
            vec![
                ("gateway", &[Scope::Gateway, Scope::Ingestion][..]),
                ("control", &[Scope::Control][..]),
                ("replication", &[Scope::Replication][..]),
            ]
        );
        assert_eq!(config.listeners[0].addr.port(), 30100);
        assert_eq!(config.listeners[1].addr.port(), 30101);
        assert_eq!(config.listeners[2].addr.port(), 30102);
        assert_eq!(config.admin.port(), 30103);
        assert!(config.listeners.iter().all(|l| l.tls.is_none()));
        assert_eq!(config.shutdown_settle, Duration::from_secs(5));
        assert_eq!(config.shutdown_grace, Duration::from_secs(30));
        assert_eq!(config.validate(), Ok(()));
    }

    #[test]
    fn an_ingestion_listener_splits_the_relays_off() {
        let config = config(&["--ingestion-listen", "127.0.0.1:30104"]).expect("four listeners");
        let names: Vec<&str> = config.listeners.iter().map(|l| l.name.as_str()).collect();
        assert_eq!(names, ["gateway", "ingestion", "control", "replication"]);
        assert_eq!(config.listeners[0].scopes, [Scope::Gateway]);
        assert_eq!(config.listeners[1].scopes, [Scope::Ingestion]);
        assert_eq!(config.listeners[1].addr.port(), 30104);
        assert_eq!(config.validate(), Ok(()));
    }

    #[test]
    fn one_identity_serves_every_listener_and_the_authorities_are_per_scope() {
        let config = config(&with_identity(&[
            "--gateway-client-ca",
            "gateways.pem",
            "--replication-client-ca",
            "replicas.pem",
            "--ingestion-listen",
            "127.0.0.1:30104",
        ]))
        .expect("tls on every listener");
        assert_eq!(
            authorities(&config),
            [
                Some(PathBuf::from("gateways.pem")),
                None,
                None,
                Some(PathBuf::from("replicas.pem")),
            ],
            "the relays and the operators stay server-TLS"
        );
        assert!(config.listeners.iter().all(|l| l
            .tls
            .as_ref()
            .is_some_and(|t| t.cert == Path::new("server.pem"))));
        assert_eq!(config.validate(), Ok(()));
    }

    #[test]
    fn scopes_sharing_a_listener_share_its_authority_or_get_their_own_listener() {
        let shared = config(&with_identity(&[
            "--gateway-client-ca",
            "gateways.pem",
            "--ingestion-client-ca",
            "relays.pem",
        ]));
        assert_eq!(
            shared,
            Err(ConfigError::ScopesNameDifferentClientCas {
                listener: "gateway".to_owned(),
                first: Scope::Gateway,
                second: Scope::Ingestion,
            }),
            "two authorities on one listener are refused, never reduced to the first"
        );

        let split = config(&with_identity(&[
            "--gateway-client-ca",
            "gateways.pem",
            "--ingestion-client-ca",
            "relays.pem",
            "--ingestion-listen",
            "127.0.0.1:30104",
        ]))
        .expect("each scope on its own listener");
        assert_eq!(
            authorities(&split),
            [
                Some(PathBuf::from("gateways.pem")),
                Some(PathBuf::from("relays.pem")),
                None,
                None,
            ]
        );
        assert_eq!(split.validate(), Ok(()));

        let same = config(&with_identity(&[
            "--gateway-client-ca",
            "fleet.pem",
            "--ingestion-client-ca",
            "fleet.pem",
        ]))
        .expect("one authority for both scopes is one rule");
        assert_eq!(
            authorities(&same),
            [Some(PathBuf::from("fleet.pem")), None, None]
        );
    }

    #[test]
    fn the_tls_refusals_are_named() {
        assert_eq!(
            config(&["--tls-cert", "server.pem"]),
            Err(ConfigError::TlsHalfConfigured {
                given: "cert",
                missing: "key"
            })
        );
        assert_eq!(
            config(&["--tls-key", "server.key"]),
            Err(ConfigError::TlsHalfConfigured {
                given: "key",
                missing: "cert"
            })
        );
        assert_eq!(
            config(&["--control-client-ca", "operators.pem"]),
            Err(ConfigError::ClientCaWithoutTls(Scope::Control))
        );
        assert_eq!(
            config(&with_identity(&["--gateway-client-ca", "gateways.pem"])),
            Err(ConfigError::ScopesDisagreeOnClientCa {
                listener: "gateway".to_owned(),
                with: Scope::Gateway,
                without: Scope::Ingestion,
            }),
            "the relays share the gateways' listener, so they share its rule or get their own"
        );
        assert_eq!(
            config(&with_identity(&["--ingestion-client-ca", "relays.pem"])),
            Err(ConfigError::ScopesDisagreeOnClientCa {
                listener: "gateway".to_owned(),
                with: Scope::Ingestion,
                without: Scope::Gateway,
            })
        );
        let split = config(&with_identity(&[
            "--gateway-client-ca",
            "gateways.pem",
            "--ingestion-listen",
            "127.0.0.1:30104",
        ]))
        .expect("the relays on their own listener, server-TLS");
        assert_eq!(
            authorities(&split),
            [Some(PathBuf::from("gateways.pem")), None, None, None]
        );
    }

    #[test]
    fn the_shutdown_flags_reach_the_configuration() {
        let bounds = config(&[
            "--shutdown-settle-secs",
            "2",
            "--shutdown-grace-period-secs",
            "10",
        ])
        .expect("bounds");
        assert_eq!(bounds.shutdown_settle, Duration::from_secs(2));
        assert_eq!(bounds.shutdown_grace, Duration::from_secs(10));
        let zero = config(&["--shutdown-grace-period-secs", "0"]).expect("parsed");
        assert_eq!(zero.validate(), Err(ConfigError::ZeroShutdownGrace));
    }

    #[test]
    fn the_load_poll_interval_is_the_gateways_default_and_refuses_zero() {
        let default = config(&[]).expect("parsed");
        assert_eq!(default.load_poll_interval, Duration::from_secs(10));
        let two = config(&["--load-monitor-interval", "2"]).expect("parsed");
        assert_eq!(two.load_poll_interval, Duration::from_secs(2));
        let zero = config(&["--load-monitor-interval", "0"]).expect("parsed");
        assert_eq!(zero.validate(), Err(ConfigError::ZeroLoadPollInterval));
    }

    #[test]
    fn the_command_line_refuses_what_it_cannot_read() {
        assert!(Cli::try_parse_from([
            "smg-policy-controller",
            "--controller-id",
            "c",
            "--gateway-listen",
            "nowhere"
        ])
        .is_err());
        assert!(Cli::try_parse_from([
            "smg-policy-controller",
            "--controller-id",
            "c",
            "--log-level",
            "loud"
        ])
        .is_err());
    }

    #[test]
    fn the_runtime_thread_count_is_bounded() {
        assert_eq!(
            parse(&["--runtime-worker-threads", "4"]).runtime_worker_threads,
            Some(4)
        );
        assert_eq!(parse(&[]).runtime_worker_threads, None);
        for bad in ["0", "4097", "many"] {
            assert!(
                Cli::try_parse_from([
                    "smg-policy-controller",
                    "--controller-id",
                    "c",
                    "--runtime-worker-threads",
                    bad
                ])
                .is_err(),
                "{bad}"
            );
        }
    }

    #[test]
    fn a_terminal_is_colored_unless_refused_and_a_pipe_only_when_forced() {
        // (terminal, refused, forced)
        assert!(color_for(true, false, false));
        assert!(!color_for(true, true, false));
        assert!(color_for(true, true, true));
        assert!(!color_for(false, false, false));
        assert!(!color_for(false, true, false));
        assert!(color_for(false, false, true));
        assert!(color_for(false, true, true));
    }

    /// A force variable set to `0` does not force (clicolors); `NO_COLOR` set
    /// to anything non-empty, `0` included, refuses (no-color.org); an empty
    /// value is unset for both. The readers are pure: the process environment
    /// is not touched.
    #[test]
    fn a_force_variable_set_to_zero_does_not_force_and_an_empty_one_is_unset() {
        let zero = Some(OsStr::new("0"));
        let empty = Some(OsStr::new(""));
        let one = Some(OsStr::new("1"));
        assert!(set(zero));
        assert!(!truthy(zero));
        assert!(!set(empty));
        assert!(!truthy(empty));
        assert!(set(one));
        assert!(truthy(one));
        assert!(!set(None));
        assert!(!truthy(None));
    }
}
