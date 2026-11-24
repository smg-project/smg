//! The standalone policy controller's command line and start-up (design 2:
//! the `remote` mode of the policy engine), as one function: the stock
//! binary calls [`run`] with the stock catalog, a host's own binary with
//! the stages it added. The command line is the operator's surface: the
//! controller's identity, its listeners, their TLS, the election, the
//! shutdown bounds, logging and worker discovery. The routing policy
//! arrives over the control scope.

use std::{
    ffi::{OsStr, OsString},
    future::pending,
    io::IsTerminal,
    net::SocketAddr,
    path::PathBuf,
    process::ExitCode,
    time::Duration,
};

use clap::Parser;
use tokio::runtime::{Builder, Runtime};
#[cfg(unix)]
use tokio::signal::unix::{signal, SignalKind};
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

#[cfg(feature = "discovery")]
use super::discovery::DiscoveryFlags;
use super::{
    Config, ConfigError, Controller, ElectionConfig, Error, LeasePlace, Listener, Scope, Tls,
    VERSION,
};
use crate::{service::elector::ElectionSettings, stages::registry::StageRegistry};

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
    about = "The SMG policy controller: the owner of the routing data, serving decisions to the gateways",
    group = clap::ArgGroup::new("spec_source").required(true).multiple(false).args(["spec_file", "spec_store"])
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

    /// The standbys' listener: Replicate, which restores a standby too. Under
    /// a shared lease a replica that does not lead follows its leader: it
    /// dials the leader's advertised host at this port, so every replica
    /// listens on the same one, and the listener takes identities as told
    /// in the clear. A replica that follows is refused port 0
    #[arg(long, default_value = DEFAULT_REPLICATION_LISTEN, help_heading = "Listeners")]
    replication_listen: SocketAddr,

    /// The HTTP admin listener: /liveness, /readiness, /health, /version
    /// and /metrics
    #[arg(long, default_value = DEFAULT_ADMIN_LISTEN, help_heading = "Listeners")]
    admin_listen: SocketAddr,

    /// The listeners, by name (gateway, control, replication, and ingestion
    /// where --ingestion-listen makes it a listener), that take a caller's
    /// identity as told: the caller states its name in the x-smg-principal
    /// metadata entry and nothing is verified, so whoever reaches such a
    /// listener acts as any principal of its scopes. For a loopback, or a
    /// mesh that verifies in front of the process; a listener with a client
    /// CA takes none. A listener that is not named refuses the calls that
    /// need a principal
    #[arg(
        long,
        value_name = "LISTENER",
        value_delimiter = ',',
        help_heading = "Listeners"
    )]
    identity_as_told: Vec<String>,

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

    // ==================== Election ====================
    /// The lease as a file on this host: the replicas of one host elect
    /// their leader over it; its directory must exist. Each renewal is a
    /// durable write: on a volume whose sync can take over a second the
    /// leader answers Busy for that time, ends its gateways' sessions
    /// where a renewal takes two, and can lose its lease where one takes
    /// three; keep the file on a quiet local volume or run a slower
    /// timing (15000, 10000 and 2000 for --lease-duration-ms,
    /// --lease-renew-deadline-ms and --lease-renew-period-ms). With no
    /// lease flag the controller runs alone and leads itself
    #[arg(long, help_heading = "Election")]
    lease_file: Option<PathBuf>,

    /// The lease as a Kubernetes Lease object of this name: the replicas of
    /// one namespace elect their leader over it
    #[cfg(feature = "kubernetes")]
    #[arg(long, conflicts_with = "lease_file", help_heading = "Election")]
    lease_name: Option<String>,

    /// The namespace of --lease-name; unset, the pod's own
    #[cfg(feature = "kubernetes")]
    #[arg(
        long,
        requires = "lease_name",
        conflicts_with = "lease_file",
        help_heading = "Election"
    )]
    lease_namespace: Option<String>,

    /// This replica's dialable address, in the lease's holder identity and
    /// in the Leader call's answer: a host, which takes the gateway
    /// listener's port, or host:port; under Kubernetes the pod's address.
    /// A lease flag needs it when the gateway listener binds every address
    #[arg(long, help_heading = "Election")]
    advertise_address: Option<String>,

    /// Milliseconds a lease stands unrenewed before a follower takes it
    #[arg(long, default_value_t = millis(ElectionSettings::default().duration), help_heading = "Election")]
    lease_duration_ms: u64,

    /// Milliseconds between two renewals of the leader, and between two
    /// looks of a follower
    #[arg(long, default_value_t = millis(ElectionSettings::default().renew_period), help_heading = "Election")]
    lease_renew_period_ms: u64,

    /// Milliseconds a leader serves on after its last renewal; past them
    /// it fences itself. Below --lease-duration-ms, above
    /// --lease-renew-period-ms
    #[arg(long, default_value_t = millis(ElectionSettings::default().renew_deadline), help_heading = "Election")]
    lease_renew_deadline_ms: u64,

    // ==================== Load Monitoring ====================
    /// Seconds between two load polls of an engine whose stream is not
    /// serving it, and the freshness bound a pushed record must meet to
    /// stand in for the poll; the gateway's flag of the same name
    #[arg(long, default_value_t = Config::DEFAULT_LOAD_POLL_INTERVAL.as_secs(), help_heading = "Load Monitoring")]
    load_monitor_interval: u64,

    /// The policy spec as a file, in YAML (JSON under a `.json` name): read
    /// and applied at the start, as generation 1 or under the number its
    /// `generation` line writes. Under `--lease-file` or `--lease-name` it
    /// is read and checked at the start, and applied when this replica
    /// leads: while it follows a leader it runs the leader's document, and
    /// once it leads it applies its own file's, unless that document ran
    /// on it since the file last changed. A file that cannot be read, or a
    /// document that is invalid, refuses the start. The file is watched
    /// from then on. While the replica leads, a changed one is applied as
    /// the next generation, or under the number it writes, and one that
    /// does not run is rejected whole and the running generation stays.
    /// While the replica follows, nothing of the file is applied and
    /// nothing is said of it. A start names one source of the spec, this
    /// file or `--spec-store`: with neither it is refused.
    ///
    /// YAML reads as JSON does: text that looks like a number or a boolean
    /// is quoted where a field takes text, and a key with no value is null,
    /// which no list or map is: write `[]` or `{}`, or leave the key out.
    #[arg(long, value_name = "PATH", help_heading = "Policy Spec")]
    spec_file: Option<PathBuf>,

    /// The policy spec as a directory the controller keeps: a document
    /// comes in through the apply call, is persisted in the directory, one
    /// file a generation, and is then applied as the next generation. The
    /// newest generation on file is applied at the start, and one that does
    /// not run refuses the start. A directory with none on file is a source
    /// too: the controller starts, reads ready and declines every decision
    /// until the first apply. The last 8 generations stay on file, and a
    /// rollback runs one of them again as the next generation. One replica
    /// serves it, its lease in the process: beside a lease another
    /// replica can take the start is refused, since a standby follows a
    /// file's generations alone in this build. One process holds the
    /// directory, under a lock on a file it makes there: a start that
    /// cannot make that file is refused, and one that finds it held waits
    /// for its holder to stop, refused where it is still held at the end
    /// of the grace period. The directory stands on a file system that
    /// links and locks: a generation's file is linked under its name.
    /// Where the locks do not reach a second host, a process there that
    /// started before another's last apply runs the generation it started
    /// with, and refuses each apply by name until it starts again
    #[arg(long, value_name = "DIR", help_heading = "Policy Spec")]
    spec_store: Option<PathBuf>,

    // ==================== Shutdown ====================
    /// Seconds the listeners keep accepting after readiness fails, so the
    /// probes see the change first; at most half the grace period
    #[arg(long, default_value_t = Config::DEFAULT_SHUTDOWN_SETTLE.as_secs(), help_heading = "Shutdown")]
    shutdown_settle_secs: u64,

    /// Seconds a shutdown may take in all, the settle window included; the
    /// listeners close with their streams when it runs out. A start on a
    /// spec store that another controller holds waits as long for that one
    /// to stop
    #[arg(long, default_value_t = Config::DEFAULT_SHUTDOWN_GRACE.as_secs(), help_heading = "Shutdown")]
    shutdown_grace_period_secs: u64,

    // ==================== Runtime ====================
    /// Explicit async runtime worker-thread count, from 1 to 4096 (a
    /// refusal names the ceiling); a count above 4 x the CPUs available to
    /// the process starts with a warning. Leave unset for tokio's default
    /// (`available_parallelism()`), which honors the cgroup CPU quota
    #[arg(long, value_parser = clap::value_parser!(u16).range(1..=4096), help_heading = "Runtime")]
    runtime_worker_threads: Option<u16>,

    /// Deciding threads, from 1 to 256, each behind a queue of 128
    /// requests. A model's decisions all run on one of them, so a fleet
    /// that serves one model decides on one thread. Default: one a CPU
    /// available to the process, eight at most
    #[arg(long, default_value_t = default_decide_shards(), value_parser = clap::value_parser!(u16).range(1..=256), help_heading = "Runtime")]
    decide_shards: u16,

    // ==================== Logging ====================
    /// Set the logging level of this binary's own crates; RUST_LOG, when
    /// set, replaces it whole
    #[arg(long, default_value = "info", value_parser = ["debug", "info", "warn", "error"], help_heading = "Logging")]
    log_level: String,

    /// Output logs as JSON
    #[arg(long, default_value_t = false, help_heading = "Logging")]
    log_json: bool,

    #[cfg(feature = "discovery")]
    #[command(flatten)]
    discovery: DiscoveryFlags,
}

/// A duration in whole milliseconds, as the election's flags take it.
fn millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn default_decide_shards() -> u16 {
    u16::try_from(Config::default_decide_shards()).unwrap_or(1)
}

impl Cli {
    /// Where the lease flags put the lease: a named object, a file, or
    /// this process when neither is given.
    fn lease(&self) -> LeasePlace {
        #[cfg(feature = "kubernetes")]
        if let Some(name) = &self.lease_name {
            return LeasePlace::Kubernetes {
                namespace: self.lease_namespace.clone(),
                name: name.clone(),
            };
        }
        match &self.lease_file {
            Some(path) => LeasePlace::File(path.clone()),
            None => LeasePlace::Process,
        }
    }

    /// The listeners the flags describe: one server identity for every gRPC
    /// listener, one client authority per scope, one credential rule per
    /// listener.
    fn config(&self) -> Result<Config, ConfigError> {
        self.config_with(|name| std::env::var_os(name))
    }

    /// [`Cli::config`] over `env`, the read of one environment variable.
    /// The worker key of a selected discovery provider alone is read so.
    #[cfg_attr(
        not(feature = "discovery"),
        expect(
            unused_variables,
            reason = "only the discovery flags read the environment"
        )
    )]
    fn config_with(&self, env: impl Fn(&str) -> Option<OsString>) -> Result<Config, ConfigError> {
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
            decide_shards: usize::from(self.decide_shards),
            election: ElectionConfig {
                lease: self.lease(),
                settings: ElectionSettings {
                    duration: Duration::from_millis(self.lease_duration_ms),
                    renew_period: Duration::from_millis(self.lease_renew_period_ms),
                    renew_deadline: Duration::from_millis(self.lease_renew_deadline_ms),
                },
                advertise: self.advertise_address.clone(),
            },
            #[cfg(feature = "discovery")]
            discovery: self.discovery.configured(env)?,
            #[cfg(not(feature = "discovery"))]
            discovery: None,
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
/// leaves the controller serving until it is killed. A host that is not
/// Unix has no SIGTERM: the terminal's interrupt alone stops it there.
async fn signals() {
    #[cfg(unix)]
    let mut term = match signal(SignalKind::terminate()) {
        Ok(term) => Some(term),
        Err(error) => {
            warn!(%error, "no SIGTERM handler; SIGINT alone stops the controller");
            None
        }
    };
    #[cfg(unix)]
    let term = async {
        match term.as_mut() {
            Some(term) => {
                term.recv().await;
            }
            None => pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let term = pending::<()>();
    let interrupt = async {
        if let Err(error) = tokio::signal::ctrl_c().await {
            #[cfg(unix)]
            warn!(%error, "no SIGINT handler; SIGTERM alone stops the controller");
            #[cfg(not(unix))]
            warn!(%error, "no interrupt handler; the controller serves until it is killed");
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

/// How long the process's end waits on a blocking call that has not
/// returned. A look at a spec file answers in well under a millisecond on a
/// file system that answers at all.
const END_WAIT: Duration = Duration::from_secs(1);

/// Runs `serve` on `runtime`, then ends the runtime. A dropped runtime
/// waits without end on a blocking call that has not returned, as a look at
/// a file system that stopped answering: the end waits [`END_WAIT`] on one
/// and leaves it, so the process ends at its own stop and not at its
/// supervisor's kill.
fn served<F: std::future::Future>(runtime: Runtime, serve: F) -> F::Output {
    let answer = runtime.block_on(serve);
    runtime.shutdown_timeout(END_WAIT);
    answer
}

/// The spec's source as the flags state it: one of the two.
#[derive(Clone, Debug, Eq, PartialEq)]
enum SpecSource {
    /// `--spec-file`.
    File(PathBuf),
    /// `--spec-store`.
    Store(PathBuf),
}

impl Cli {
    /// The spec's source: the one flag the parser took. The parser refuses
    /// a command line that names neither, so `None` is a state it rules
    /// out, and a start that meets it all the same refuses in its own words.
    fn spec_source(&self) -> Option<SpecSource> {
        let file = self.spec_file.clone().map(SpecSource::File);
        file.or_else(|| self.spec_store.clone().map(SpecSource::Store))
    }
}

/// Binds the controller over `catalog`, names the listeners that take
/// identities as told, then takes the spec's source: a refusal of any
/// arrives before anything is served.
async fn bind(
    config: Config,
    catalog: StageRegistry,
    as_told: &[String],
    spec: SpecSource,
) -> Result<Controller, Error> {
    let mut controller = Controller::bind_with(config, catalog).await?;
    for listener in as_told {
        controller.identity_as_told(listener)?;
    }
    match spec {
        SpecSource::File(path) => controller.spec_file(path).await?,
        SpecSource::Store(dir) => controller.spec_store(dir).await?,
    }
    Ok(controller)
}

/// Runs a controller to its end and returns the process's exit code. The
/// start-up sequence: parse, log, configure, then the runtime, built only
/// now so a refusal costs no threads; bind and serve on it. A policy
/// document names the stages `catalog` holds.
///
/// The command line is the process's own and this crate's: a flag it does
/// not know is refused, so a host's settings come from its environment or a
/// file. At `--help`, `--version` and a refused flag the parser ends the
/// process itself and the call does not return.
///
/// # Panics
///
/// When a global log subscriber is already set: `run` installs its own, so
/// the `main` that calls it installs none. When called inside an async
/// runtime: `run` builds its own and blocks on it, so that `main` is a
/// plain `fn main`.
pub fn run(catalog: StageRegistry) -> ExitCode {
    let cli = Cli::parse();
    init_logging(&cli.log_level, cli.log_json);
    run_with(cli, catalog)
}

/// [`run`] from its parsed command line on, the log set: configure, take
/// the spec's source, then the runtime, bind and serve on it. Apart from
/// the parse, which reads the process's own arguments, and from the log,
/// which a process sets once.
fn run_with(cli: Cli, catalog: StageRegistry) -> ExitCode {
    let config = match cli.config() {
        Ok(config) => config,
        Err(error) => {
            error!(%error, "configuration refused");
            return ExitCode::FAILURE;
        }
    };
    let Some(spec) = cli.spec_source() else {
        error!(
            error = "no spec source is named: a start takes --spec-file or --spec-store",
            "start refused"
        );
        return ExitCode::FAILURE;
    };
    let runtime = match runtime(cli.runtime_worker_threads) {
        Ok(runtime) => runtime,
        Err(error) => {
            error!(%error, "runtime refused");
            return ExitCode::FAILURE;
        }
    };
    served(runtime, async {
        let bound = bind(config, catalog, &cli.identity_as_told, spec);
        let controller = match bound.await {
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
        process::ExitCode,
        sync::{mpsc, Arc, Mutex},
        thread,
        time::Duration,
    };

    use clap::Parser;
    use tokio::{runtime::Handle, sync::oneshot};

    use super::{bind, color_for, set, truthy, Cli, SpecSource};
    use crate::{
        document::spec::SpecError,
        engine::{Candidate, Constraints, Filter, RequestFacts},
        service::{
            controller::{
                membership::tests::Logged, spec_store::SpecStore, Config, ConfigError, Controller,
                ElectionConfig, Error, LeasePlace, Scope, SpecRejected, StoreRefused,
            },
            elector::ElectionSettings,
        },
        stages::registry::StageRegistry,
    };

    /// A spec source for a command line that names none: a start names one,
    /// and the parser reads this one's flag and opens nothing.
    const SOURCE: [&str; 2] = ["--spec-store", "/var/lib/smg/spec"];

    fn parse(args: &[&str]) -> Cli {
        let mut argv = vec!["smg-policy-controller", "--controller-id", "controller-0"];
        if !args
            .iter()
            .any(|arg| ["--spec-file", "--spec-store"].contains(arg))
        {
            argv.extend(SOURCE);
        }
        argv.extend_from_slice(args);
        Cli::try_parse_from(argv).expect("the flags parse")
    }

    /// Whether the parser refuses `flags` beside an id and a spec source:
    /// the refusal is then the flags' own.
    fn refuses(flags: &[&str]) -> bool {
        let mut argv = vec!["smg-policy-controller", "--controller-id", "c"];
        argv.extend(SOURCE);
        argv.extend_from_slice(flags);
        Cli::try_parse_from(argv).is_err()
    }

    /// The configuration of `args` under an empty environment: a case
    /// reads nothing of the process's own.
    fn config(args: &[&str]) -> Result<Config, ConfigError> {
        parse(args).config_with(|_| None)
    }

    #[cfg(feature = "discovery")]
    #[test]
    fn the_discovery_flags_reach_the_configuration() {
        use smg_service_discovery::RuntimeDiscoveryConfig;

        let args = [
            "--discovery-provider",
            "file",
            "--discovery-file",
            "/run/workers.json",
            "--discovery-check-interval-secs",
            "13",
            "--api-key",
            "worker-key",
        ];
        let configured = config(&args).expect("discovery configuration");
        let settings = configured.discovery.as_ref().expect("selected provider");
        let RuntimeDiscoveryConfig::File(file) = settings.runtime() else {
            panic!("expected the file provider");
        };
        assert_eq!(file.path, Path::new("/run/workers.json"));
        assert_eq!(file.check_interval, Duration::from_secs(13));
        assert!(settings.credential().is_some());
        assert_eq!(configured, config(&args).expect("the same settings"));
        assert_ne!(configured, config(&args[..6]).expect("without a key"));
        let mut other_file = args;
        other_file[3] = "/run/other.json";
        assert_ne!(configured, config(&other_file).expect("another manifest"));
        let disaggregated = [&args[..], &["--prefill-selector", "role=prefill"]].concat();
        assert_ne!(
            configured,
            config(&disaggregated).expect("another set of roles")
        );
        assert!(config(&[]).expect("the defaults").discovery.is_none());
        assert!(matches!(
            config(&["--discovery-provider", "file"]),
            Err(ConfigError::DiscoveryFlag {
                flag: "--discovery-file",
                ..
            })
        ));
    }

    #[cfg(feature = "discovery")]
    #[test]
    fn the_worker_key_is_read_only_for_a_selected_provider() {
        use std::ffi::OsString;

        let file = [
            "--discovery-provider",
            "file",
            "--discovery-file",
            "/run/workers.json",
        ];
        let key = |name: &str| (name == "SMG_API_KEY").then(|| OsString::from("worker-key"));
        let keyed = parse(&file).config_with(key).expect("a key to send");
        assert!(keyed
            .discovery
            .as_ref()
            .is_some_and(|settings| settings.credential().is_some()));
        assert_ne!(keyed, config(&file).expect("no key to send"));

        // A key file that is not there fails every read of the key.
        let missing =
            |name: &str| (name == "SMG_API_KEY_FILE").then(|| OsString::from("/missing/key"));
        assert!(matches!(
            parse(&file).config_with(missing),
            Err(ConfigError::DiscoveryFlag {
                flag: "--api-key-file",
                ..
            })
        ));
        let unread = parse(&[]).config_with(missing).expect("no key is read");
        assert!(unread.discovery.is_none());

        for flag in [
            ["--api-key", "worker-key"],
            ["--api-key-file", "/missing/key"],
        ] {
            assert!(
                matches!(
                    config(&flag),
                    Err(ConfigError::DiscoveryFlag { flag: refused, .. }) if refused == flag[0]
                ),
                "{flag:?}"
            );
        }
    }

    #[cfg(feature = "discovery")]
    #[test]
    fn a_slurm_job_name_under_two_roles_is_refused_at_the_start() {
        let shared = [
            "--discovery-provider",
            "slurm",
            "--slurm-prefill-job-name",
            "shared",
            "--slurm-decode-job-name",
            "shared",
            "--slurm-port",
            "8000",
        ];
        let Err(ConfigError::DiscoveryFlag { flag, reason }) = config(&shared) else {
            panic!("one job name under two roles was taken");
        };
        assert_eq!(flag, "--discovery-provider");
        assert!(
            reason.contains("'shared' maps to both prefill and decode workers"),
            "{reason}"
        );
        let apart = [&shared[..5], &["other", "--slurm-port", "8000"]].concat();
        assert!(config(&apart).expect("two names").discovery.is_some());
    }

    #[cfg(not(feature = "discovery"))]
    #[test]
    fn a_build_without_discovery_refuses_its_flags() {
        for flags in [
            &["--service-discovery"][..],
            &["--discovery-provider", "file"],
            &["--discovery-file", "workers.json"],
            &["--api-key", "worker-key"],
            &["--api-key-file", "worker-key.txt"],
        ] {
            assert!(refuses(flags), "{flags:?}");
        }
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
    fn with_no_lease_flag_one_replica_leads_itself_on_the_designs_settings() {
        let alone = config(&[]).expect("parsed");
        assert_eq!(alone.election, ElectionConfig::default());
        assert_eq!(alone.election.lease, LeasePlace::Process);
        assert_eq!(alone.election.settings, ElectionSettings::default());
        assert_eq!(
            alone.validate(),
            Ok(()),
            "a wildcard listener is no refusal"
        );
    }

    #[test]
    fn a_lease_file_elects_under_an_address_a_client_can_dial() {
        // The default gateway listener binds every address: a shared lease
        // behind it is refused until an address is advertised.
        let file = config(&["--lease-file", "/run/smg/lease"]).expect("parsed");
        assert_eq!(
            file.election.lease,
            LeasePlace::File(PathBuf::from("/run/smg/lease"))
        );
        assert_eq!(file.validate(), Err(ConfigError::AdvertiseAddressRequired));

        let advertised = config(&[
            "--lease-file",
            "/run/smg/lease",
            "--advertise-address",
            "10.0.0.7",
        ])
        .expect("parsed");
        assert_eq!(advertised.election.advertise.as_deref(), Some("10.0.0.7"));
        assert_eq!(advertised.validate(), Ok(()));

        // A gateway listener that binds one address is the address.
        let bound = config(&[
            "--lease-file",
            "/run/smg/lease",
            "--gateway-listen",
            "127.0.0.1:30100",
        ])
        .expect("parsed");
        assert_eq!(bound.validate(), Ok(()));

        let malformed = config(&["--advertise-address", "controller@host"]).expect("parsed");
        assert_eq!(
            malformed.validate(),
            Err(ConfigError::MalformedAdvertiseAddress(
                "controller@host".to_owned()
            ))
        );
    }

    #[test]
    fn the_lease_timings_reach_the_configuration_and_keep_their_order() {
        let slow = config(&[
            "--lease-duration-ms",
            "15000",
            "--lease-renew-period-ms",
            "2000",
            "--lease-renew-deadline-ms",
            "10000",
        ])
        .expect("parsed");
        assert_eq!(
            slow.election.settings,
            ElectionSettings {
                duration: Duration::from_secs(15),
                renew_period: Duration::from_secs(2),
                renew_deadline: Duration::from_secs(10),
            }
        );
        assert_eq!(slow.validate(), Ok(()));

        let inverted = config(&["--lease-renew-deadline-ms", "3000"]).expect("parsed");
        assert_eq!(
            inverted.validate(),
            Err(ConfigError::ElectionSettings(
                "the duration is not longer than the renew deadline"
            ))
        );
    }

    #[cfg(feature = "kubernetes")]
    #[test]
    fn a_lease_object_is_named_and_takes_the_pods_namespace_unless_one_is_given() {
        let args = [
            "--advertise-address",
            "10.0.0.7",
            "--lease-name",
            "smg-policy",
        ];
        let object = config(&args).expect("parsed");
        assert_eq!(
            object.election.lease,
            LeasePlace::Kubernetes {
                namespace: None,
                name: "smg-policy".to_owned(),
            }
        );
        assert_eq!(object.validate(), Ok(()));

        let mut named = args.to_vec();
        named.extend(["--lease-namespace", "policy"]);
        assert_eq!(
            config(&named).expect("parsed").election.lease,
            LeasePlace::Kubernetes {
                namespace: Some("policy".to_owned()),
                name: "smg-policy".to_owned(),
            }
        );

        // The object stores whole seconds.
        let mut fraction = args.to_vec();
        fraction.extend(["--lease-duration-ms", "2500"]);
        assert_eq!(
            config(&fraction).expect("parsed").validate(),
            Err(ConfigError::LeaseDurationNotWholeSeconds(
                Duration::from_millis(2500)
            ))
        );

        // One lease, one place; a namespace names an object's.
        for refused in [
            &[
                "--lease-name",
                "smg-policy",
                "--lease-file",
                "/run/smg/lease",
            ][..],
            &["--lease-namespace", "policy"],
            &[
                "--lease-namespace",
                "policy",
                "--lease-file",
                "/run/smg/lease",
            ],
        ] {
            assert!(refuses(refused), "{refused:?}");
        }
    }

    #[test]
    fn no_listener_takes_identities_as_told_unless_it_is_named() {
        assert!(parse(&[]).identity_as_told.is_empty());
        assert_eq!(
            parse(&["--identity-as-told", "gateway,control"]).identity_as_told,
            ["gateway", "control"]
        );
        assert_eq!(
            parse(&[
                "--identity-as-told",
                "gateway",
                "--identity-as-told",
                "replication"
            ])
            .identity_as_told,
            ["gateway", "replication"]
        );
    }

    /// Every listener on loopback, at a port the host picks.
    const EPHEMERAL: [&str; 8] = [
        "--gateway-listen",
        "127.0.0.1:0",
        "--control-listen",
        "127.0.0.1:0",
        "--replication-listen",
        "127.0.0.1:0",
        "--admin-listen",
        "127.0.0.1:0",
    ];

    /// Keeps every worker: a stage of a host's own.
    #[derive(Debug)]
    struct Any;

    impl Filter for Any {
        fn name(&self) -> &'static str {
            "any"
        }

        fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
            true
        }
    }

    /// The stages the run is handed are the ones the bound controller holds.
    #[tokio::test]
    async fn the_bound_controller_holds_the_stages_the_run_is_handed() {
        let config = parse(&EPHEMERAL).config().expect("the flags are in order");
        let mut stages = StageRegistry::builtin();
        stages.filter("any", || Box::new(Any)).expect("a free name");
        let dir = tempfile::tempdir().expect("a temp dir");
        let source = SpecSource::Store(dir.path().to_owned());
        let controller = bind(config, stages, &[], source).await.expect("binds");
        assert!(controller
            .stage_registry()
            .filters()
            .any(|name| name == "any"));
        // The stock build holds no such stage.
        assert!(StageRegistry::builtin().filters().all(|name| name != "any"));
    }

    /// A spec file's document names the stages the run is handed: a stage
    /// of a host's own starts the controller on it, and the stock build
    /// refuses the same file by the stage's name.
    #[tokio::test]
    async fn a_spec_file_names_the_stages_the_run_is_handed() {
        let dir = tempfile::tempdir().expect("a directory");
        let path = dir.path().join("policy.yaml");
        let document =
            "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    filter: [any]\n    pick: random\n";
        std::fs::write(&path, document).expect("the file");
        let start = |stages: StageRegistry| {
            let config = parse(&EPHEMERAL).config().expect("the flags are in order");
            bind(config, stages, &[], SpecSource::File(path.clone()))
        };

        let mut stages = StageRegistry::builtin();
        stages.filter("any", || Box::new(Any)).expect("a free name");
        let controller = start(stages).await.expect("a host's stage starts");
        assert_eq!(controller.store().snapshot().spec().generation(), 1);
        drop(controller);

        match start(StageRegistry::builtin()).await {
            Err(Error::SpecFile {
                source: SpecRejected::Invalid(SpecError::UnknownFilter { profile, name }),
                ..
            }) => assert_eq!((profile.as_str(), name.as_str()), ("default", "any")),
            Err(other) => panic!("the stock build refused it as {other}"),
            Ok(_) => panic!("the stock build holds no such stage"),
        }
    }

    /// Scores every worker alike, and says whether the lifecycle's calls
    /// keep its state: a stage of a host's own.
    #[derive(Debug)]
    struct Level(bool);

    impl crate::stages::Lifecycle for Level {
        fn needs_lifecycle(&self) -> bool {
            self.0
        }
    }

    impl crate::engine::Scorer for Level {
        fn name(&self) -> &'static str {
            "level"
        }

        fn score(
            &self,
            _: &RequestFacts<'_>,
            _: &[Candidate],
            _: &[u32],
            out: &mut [f32],
        ) -> crate::engine::TieBand {
            out.fill(1.0);
            crate::engine::TieBand::NORMALIZED
        }
    }

    /// The controller makes the membership call and a lease's calls but
    /// the end of a lease whose gateway was let go: a spec file that names
    /// a stage in need of them refuses the start by the stage's field, and
    /// the same stage with no such need starts.
    #[tokio::test]
    async fn a_spec_file_that_names_a_stage_in_need_of_the_lifecycle_refuses_the_start() {
        let dir = tempfile::tempdir().expect("a directory");
        let path = dir.path().join("policy.yaml");
        let document =
            "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    score: [level]\n    pick: best\n";
        std::fs::write(&path, document).expect("the file");
        let start = |needs: bool| {
            let config = parse(&EPHEMERAL).config().expect("the flags are in order");
            let mut stages = StageRegistry::builtin();
            stages
                .scorer("level", move || Box::new(Level(needs)))
                .expect("a free name");
            bind(config, stages, &[], SpecSource::File(path.clone()))
        };

        let controller = start(false).await.expect("no need of the calls starts");
        assert_eq!(controller.store().snapshot().spec().generation(), 1);
        drop(controller);

        match start(true).await {
            Err(Error::SpecFile {
                source: SpecRejected::Invalid(SpecError::NotInBuild { path: field, what }),
                ..
            }) => assert_eq!(
                (field.as_str(), what),
                (
                    "profiles.default.score[0].level",
                    "the lifecycle this stage needs"
                )
            ),
            Err(other) => panic!("the start was refused as {other}"),
            Ok(_) => panic!("no served line makes the calls the stage needs"),
        }
    }

    /// The controller's file of leases keeps one session verdict for the
    /// two sides of a pair. So its own start refuses a spec file that binds
    /// a sessions rule under the pair plan, by the rule's place, and starts
    /// on the same file with no rule. The refusal is read through the
    /// catalog the controller itself builds: the case fails where that
    /// catalog comes to say every lifecycle call is made while the file of
    /// leases still keeps one verdict.
    #[tokio::test]
    async fn a_spec_file_that_binds_a_sessions_rule_under_a_pair_refuses_the_start() {
        let dir = tempfile::tempdir().expect("a directory");
        let path = dir.path().join("policy.yaml");
        let pair = "apiVersion: smg.policy/v1\nview:\n  headers: [x-session-id]\nprofiles:\n  q:\n    pick: random\nbindings:\n  - plan: pd-pair\n    profiles:\n      prefill: q\n      decode: q\n";
        let rule = "    sessions:\n      sticky_header: x-session-id\n      inflight_cap: 65535\n";
        let start = |document: String| {
            std::fs::write(&path, document).expect("the file");
            let config = parse(&EPHEMERAL).config().expect("the flags are in order");
            let stages = StageRegistry::builtin();
            bind(config, stages, &[], SpecSource::File(path.clone()))
        };

        let unruled = start(pair.to_string()).await;
        let controller = unruled.expect("a pair under no rule starts");
        assert_eq!(controller.store().snapshot().spec().generation(), 1);
        drop(controller);

        match start(format!("{pair}{rule}")).await {
            Err(Error::SpecFile {
                source: SpecRejected::Invalid(SpecError::NotInBuild { path: field, what }),
                ..
            }) => assert_eq!(
                (field.as_str(), what),
                (
                    "bindings[0].sessions",
                    "the session verdict a pair's decode side keeps of its own"
                )
            ),
            Err(other) => panic!("the start was refused as {other}"),
            Ok(_) => panic!("the file of leases keeps one verdict for a pair's two sides"),
        }
    }

    /// A served decision writes the record of sent prefixes at its lease's
    /// issue: a spec file starts under each source of prefix affinity, and
    /// one that names none reads the prefixes sent, as `merged` does.
    #[tokio::test]
    async fn a_spec_file_starts_under_each_source_of_prefix_affinity() {
        let dir = tempfile::tempdir().expect("a directory");
        let path = dir.path().join("policy.yaml");
        let start = |affinity: &str| {
            let document = format!(
                "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    score:\n      - prefix-affinity: {affinity}\n    pick: best\n"
            );
            std::fs::write(&path, document).expect("the file");
            let config = parse(&EPHEMERAL).config().expect("the flags are in order");
            bind(
                config,
                StageRegistry::builtin(),
                &[],
                SpecSource::File(path.clone()),
            )
        };
        for (affinity, sent) in [
            ("{source: index}", false),
            ("{source: predictor}", true),
            ("{source: merged}", true),
            ("{}", true),
        ] {
            let controller = start(affinity).await.expect(affinity);
            let needs = controller.store().snapshot().spec().needs();
            assert_eq!(needs.sent_prefixes, sent, "{affinity}");
        }
    }

    /// The flag's names reach the bound controller: a name the flags mount
    /// no listener under refuses the start, by that name.
    #[tokio::test]
    async fn a_listener_named_as_told_is_one_the_flags_mount() {
        // The start binds: its source is a store that is there.
        let dir = tempfile::tempdir().expect("a directory");
        let store = dir.path().to_str().expect("a path in UTF-8");
        let start = |flags: &[&str]| {
            let mut args = EPHEMERAL.to_vec();
            args.extend_from_slice(&["--spec-store", store]);
            args.extend_from_slice(flags);
            let cli = parse(&args);
            let config = cli.config().expect("the flags are in order");
            async move {
                bind(
                    config,
                    StageRegistry::builtin(),
                    &cli.identity_as_told,
                    cli.spec_source().expect("the flags name a store"),
                )
                .await
                .map(drop)
            }
        };
        // The relays' calls ride the gateways' listener unless they get
        // their own: there is none of that name to take them as told.
        let refused = start(&["--identity-as-told", "ingestion"]).await;
        assert!(
            matches!(
                &refused,
                Err(Error::Config(ConfigError::UnknownListener(name))) if name == "ingestion"
            ),
            "{refused:?}"
        );
        start(&[
            "--identity-as-told",
            "gateway,ingestion",
            "--ingestion-listen",
            "127.0.0.1:0",
        ])
        .await
        .expect("both are listeners the flags mount");
    }

    #[test]
    fn the_command_line_refuses_what_it_cannot_read() {
        assert!(refuses(&["--gateway-listen", "nowhere"]));
        assert!(refuses(&["--log-level", "loud"]));
    }

    /// A blocking call that has not returned does not hold the run's end:
    /// the serve's answer is back inside five seconds, where a dropped
    /// runtime waits on the call without end.
    #[test]
    fn the_runs_end_waits_on_no_blocked_call() {
        let runtime = super::runtime(Some(1)).expect("a runtime");
        let (release, blocked) = mpsc::channel::<()>();
        let (ended, end) = mpsc::channel();
        thread::spawn(move || {
            let answer = super::served(runtime, async move {
                let (started, start) = oneshot::channel::<()>();
                tokio::task::spawn_blocking(move || {
                    let _ = started.send(());
                    // Returns when `release` goes, at the case's end.
                    let _ = blocked.recv();
                });
                start.await.expect("the blocking call starts");
                7
            });
            let _ = ended.send(answer);
        });
        let waited = end.recv_timeout(Duration::from_secs(5));
        drop(release);
        assert_eq!(waited, Ok(7), "the end returns with a call still blocked");
    }

    /// The run's own call ends it so: a start refused after a stage of the
    /// host's left a blocking call gives its code inside five seconds.
    #[test]
    fn a_refused_start_ends_the_run_with_a_call_still_blocked() {
        let dir = tempfile::tempdir().expect("a directory");
        let path = dir.path().join("policy.yaml");
        // The first filter is built, then the second is refused by name.
        let document =
            "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    filter: [any, nowhere]\n    pick: random\n";
        std::fs::write(&path, document).expect("the file");
        let (release, blocked) = mpsc::channel::<()>();
        let blocked = Arc::new(Mutex::new(blocked));
        let (left, leaves) = mpsc::channel::<()>();
        let mut stages = StageRegistry::builtin();
        stages
            .filter("any", move || {
                // On the run's runtime alone: a registration builds the
                // stage once too, for its name.
                if let Ok(runtime) = Handle::try_current() {
                    let (begun, begin) = mpsc::channel::<()>();
                    let blocked = Arc::clone(&blocked);
                    runtime.spawn_blocking(move || {
                        let _ = begun.send(());
                        // Returns when `release` goes, at the case's end.
                        if let Ok(blocked) = blocked.lock() {
                            let _ = blocked.recv();
                        }
                    });
                    // The call is on its thread before the stage is handed
                    // out: an end leaves a call that has not begun unrun.
                    if begin.recv().is_ok() {
                        let _ = left.send(());
                    }
                }
                Box::new(Any)
            })
            .expect("a free name");
        let mut args: Vec<&str> = EPHEMERAL.to_vec();
        args.extend(["--spec-file", path.to_str().expect("a path in UTF-8")]);
        let cli = parse(&args);

        let (ended, end) = mpsc::channel();
        thread::spawn(move || {
            let _ = ended.send(super::run_with(cli, stages));
        });
        let waited = end.recv_timeout(Duration::from_secs(5));
        drop(release);
        leaves.try_recv().expect("the stage left a blocking call");
        assert_eq!(
            waited,
            Ok(ExitCode::FAILURE),
            "the run returns with a call still blocked"
        );
    }

    #[test]
    fn the_deciding_threads_take_the_operators_count_inside_their_range() {
        let default = config(&[]).expect("parsed");
        assert_eq!(default.decide_shards, Config::default_decide_shards());
        assert_eq!(default.validate(), Ok(()));
        let three = config(&["--decide-shards", "3"]).expect("parsed");
        assert_eq!(three.decide_shards, 3);
        for bad in ["0", "257", "many"] {
            assert!(refuses(&["--decide-shards", bad]), "{bad}");
        }
    }

    #[test]
    fn the_runtime_thread_count_is_bounded() {
        assert_eq!(
            parse(&["--runtime-worker-threads", "4"]).runtime_worker_threads,
            Some(4)
        );
        assert_eq!(parse(&[]).runtime_worker_threads, None);
        for bad in ["0", "4097", "many"] {
            assert!(refuses(&["--runtime-worker-threads", bad]), "{bad}");
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

    /// The flag's long help says how YAML reads: an operator meets the rule
    /// before a start refuses a number where a field takes text.
    #[test]
    fn the_spec_files_help_says_how_yaml_reads() {
        let help = <Cli as clap::CommandFactory>::command()
            .render_long_help()
            .to_string();
        // The help wraps where the terminal does.
        let words = help.split_whitespace().collect::<Vec<_>>().join(" ");
        assert!(words.contains("quoted where a field takes text"), "{help}");
        assert!(words.contains("a key with no value is null"), "{help}");
    }

    #[test]
    fn the_spec_file_is_a_flag_with_no_default() {
        assert_eq!(parse(&with_identity(&[])).spec_file, None);
        assert_eq!(
            parse(&with_identity(&[
                "--spec-file",
                "/etc/smg/policy/policy.yaml"
            ]))
            .spec_file,
            Some(PathBuf::from("/etc/smg/policy/policy.yaml"))
        );
    }

    /// A start names one source of its spec. The parser refuses a command
    /// line that names none, and one that names both, by both flags' names.
    #[test]
    fn a_start_names_one_spec_source_and_the_parser_refuses_none_and_both() {
        use clap::error::ErrorKind;

        let dir = PathBuf::from("/var/lib/smg/spec");
        let stored = parse(&with_identity(&["--spec-store", "/var/lib/smg/spec"]));
        assert_eq!(stored.spec_source(), Some(SpecSource::Store(dir)));
        let file = PathBuf::from("/etc/smg/policy/policy.yaml");
        let filed = parse(&with_identity(&[
            "--spec-file",
            "/etc/smg/policy/policy.yaml",
        ]));
        assert_eq!(filed.spec_source(), Some(SpecSource::File(file)));

        let mut argv = vec!["smg-policy-controller", "--controller-id", "controller-0"];
        argv.extend(with_identity(&[]));
        let neither = Cli::try_parse_from(argv.clone()).expect_err("a source is named");
        argv.extend(["--spec-file", "/a.yaml", "--spec-store", "/b"]);
        let both = Cli::try_parse_from(argv).expect_err("one source");
        for (refused, kind) in [
            (neither, ErrorKind::MissingRequiredArgument),
            (both, ErrorKind::ArgumentConflict),
        ] {
            assert_eq!(refused.kind(), kind);
            let told = refused.to_string();
            assert!(
                told.contains("--spec-file") && told.contains("--spec-store"),
                "{told}"
            );
        }
    }

    /// A command line that names neither source is the state the parser
    /// rules out. Past the parser it has no source, and the start refuses
    /// it before a runtime is built, in the refusal's own words: the case
    /// reads the line the start wrote, on its own thread.
    #[test]
    fn a_command_line_past_the_parser_with_no_spec_source_is_refused() {
        let mut cli = parse(&[]);
        cli.spec_store = None;
        assert_eq!(cli.spec_source(), None);
        // The case's own subscriber keeps this thread's lines, a second one
        // registered beside it, as in the door's audit case.
        let lines = Logged::default();
        let writer = lines.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_writer(move || writer.clone())
            .finish();
        let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
        let code = tracing::subscriber::with_default(subscriber, || {
            super::run_with(cli, StageRegistry::builtin())
        });
        assert_eq!(code, ExitCode::FAILURE);
        let said = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
        assert!(
            said.contains("start refused")
                && said
                    .contains("no spec source is named: a start takes --spec-file or --spec-store"),
            "{said}"
        );
    }

    /// A caller that drops the spec store call while its start waits on a
    /// held directory leaves the task to bind the store once the holder
    /// lets go: its series are bound with it, and a second call is refused
    /// as a second source.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_dropped_spec_store_call_that_binds_later_leaves_its_series() {
        let dir = tempfile::tempdir().expect("a directory");
        let holder = SpecStore::open(dir.path().to_owned()).expect("the holder");
        let config = parse(&EPHEMERAL).config().expect("the flags are in order");
        let mut controller = Controller::bind_with(config, StageRegistry::builtin())
            .await
            .expect("the controller binds");
        let call = controller.spec_store(dir.path().to_owned());
        let dropped = tokio::time::timeout(Duration::from_millis(200), call).await;
        assert!(dropped.is_err(), "the call waits on the held directory");
        drop(holder);
        let bound = async {
            while !controller.metrics.spec_bound() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        };
        tokio::time::timeout(Duration::from_secs(10), bound)
            .await
            .expect("the task binds the store with its series");
        assert!(controller.shared.generations.bound());
        match controller.spec_store(dir.path().to_owned()).await {
            Err(Error::SpecStore {
                source: StoreRefused::SecondSource,
                ..
            }) => {}
            other => panic!("a second source: {other:?}"),
        }
    }

    /// A spec store starts the controller on the newest generation on file,
    /// under that file's number; a path that leads to no directory refuses
    /// the start by name; and a controller that took a store takes no file
    /// and no second store, refused at once on the directory it holds.
    #[tokio::test]
    async fn a_spec_store_starts_on_its_newest_generation() {
        let dir = tempfile::tempdir().expect("a directory");
        let document =
            "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    filter: [healthy]\n    pick: random\n";
        let store = dir.path().join("store");
        std::fs::create_dir(&store).expect("the directory");
        for name in ["00000000000000000003.yaml", "00000000000000000004.yaml"] {
            std::fs::write(store.join(name), document).expect("the file");
        }
        let start = |path: PathBuf| {
            let config = parse(&EPHEMERAL).config().expect("the flags are in order");
            bind(
                config,
                StageRegistry::builtin(),
                &[],
                SpecSource::Store(path),
            )
        };

        let mut controller = start(store.clone()).await.expect("the store starts");
        assert_eq!(controller.store().snapshot().spec().generation(), 4);
        let file = store.join("00000000000000000004.yaml");
        match controller.spec_file(file.clone()).await {
            Err(Error::SpecFile {
                source: SpecRejected::SecondSource,
                ..
            }) => {}
            other => panic!("a second source: {other:?}"),
        }
        match controller.spec_store(store.clone()).await {
            Err(Error::SpecStore {
                source: StoreRefused::SecondSource,
                ..
            }) => {}
            other => panic!("a second source: {other:?}"),
        }
        drop(controller);

        match start(file).await {
            Err(
                refused @ Error::SpecStore {
                    source: StoreRefused::NotADirectory,
                    ..
                },
            ) => assert!(
                refused
                    .to_string()
                    .ends_with(": the path leads to no directory"),
                "{refused}"
            ),
            Err(other) => panic!("a file was refused as {other}"),
            Ok(_) => panic!("a file is no directory"),
        }
    }

    /// Beside a lease another replica can take, a spec store refuses the
    /// start by name: a standby follows a file's generations alone.
    #[tokio::test]
    async fn a_spec_store_refuses_the_start_beside_a_lease_another_replica_can_take() {
        let dir = tempfile::tempdir().expect("a directory");
        let store = dir.path().join("store");
        std::fs::create_dir(&store).expect("the directory");
        let lease = dir.path().join("lease");
        let flags = ["--lease-file", lease.to_str().expect("a path in UTF-8")];
        let mut args = EPHEMERAL.to_vec();
        args.extend_from_slice(&flags);
        let config = parse(&args).config().expect("the flags are in order");
        assert!(matches!(config.election.lease, LeasePlace::File(_)));

        let source = SpecSource::Store(store.clone());
        match bind(config, StageRegistry::builtin(), &[], source).await {
            Err(refused @ Error::SpecStore {
                source: StoreRefused::BesideASharedLease,
                ..
            }) => assert_eq!(
                refused.to_string(),
                format!(
                    "the spec store {}: another replica can take this controller's lease, and a standby follows a file's generations alone in this build: keep the lease in the process, or hand the spec as a file",
                    store.display()
                )
            ),
            Err(other) => panic!("refused as {other}"),
            Ok(_) => panic!("a shared lease takes no spec store"),
        }
    }

    /// Beside a lease another replica can take, a spec file is passed at
    /// the start and not applied: the replica runs no number of its own
    /// until it leads, so it follows its leader's generation whatever the
    /// number. The file is its source all the same, and writes while the
    /// replica leads. One that does not run refuses the start. Under a
    /// lease in the process the file is applied at the start, and writes
    /// always.
    #[tokio::test]
    async fn a_spec_file_is_applied_at_the_start_under_a_lease_in_the_process_alone() {
        let dir = tempfile::tempdir().expect("a directory");
        let document =
            "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    filter: [healthy]\n    pick: random\n";
        let file = dir.path().join("policy.yaml");
        std::fs::write(&file, document).expect("the file");
        let lease = dir.path().join("lease");
        let shared = ["--lease-file", lease.to_str().expect("a path in UTF-8")];
        let start = |flags: &[&str]| {
            let mut args = EPHEMERAL.to_vec();
            args.extend_from_slice(flags);
            let config = parse(&args).config().expect("the flags are in order");
            bind(
                config,
                StageRegistry::builtin(),
                &[],
                SpecSource::File(file.clone()),
            )
        };

        let alone = start(&[]).await.expect("the file starts");
        assert_eq!(alone.store().snapshot().spec().generation(), 1);
        assert!(alone.shared.generations.source_writes());
        drop(alone);

        let paired = start(&shared).await.expect("the file starts");
        assert_eq!(
            paired.store().snapshot().spec().generation(),
            0,
            "passed, not applied"
        );
        assert!(paired.shared.generations.bound(), "a source all the same");
        assert!(
            !paired.shared.generations.source_writes(),
            "its source writes while it leads, and it has not led"
        );
        drop(paired);

        std::fs::write(&file, document.replace("healthy", "no-such-stage")).expect("the file");
        match start(&shared).await {
            Err(Error::SpecFile { .. }) => {}
            Err(other) => panic!("refused as {other}"),
            Ok(_) => panic!("a document that does not run refuses the start"),
        }
    }
}
