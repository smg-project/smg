//! The standalone Grand Central's command line and start-up (design 2:
//! the `remote` mode of the engine), as one function: the stock
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
use smg_worker::health::overload::{
    OverloadThresholds, DEFAULT_TOKEN_USAGE_CEILING, DEFAULT_WAITING_REQUESTS,
};
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
    about = "SMG Grand Central: the owner of the routing data, serving decisions to the gateways",
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
    /// The gateways' listener: Session, Decide and WatchWorkers
    #[arg(long, default_value = DEFAULT_GATEWAY_LISTEN, help_heading = "Listeners")]
    gateway_listen: SocketAddr,

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

    /// The listeners, by name (gateway, control and replication, each
    /// after its own flag), that take a caller's
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

    /// Queued requests a worker's load report states, at or above which an
    /// engine's worker that registers no count of its own is vetoed as
    /// overloaded; from 1 to 2^63 - 1, the most a load report states; the
    /// gateway's flag of the same name. An external worker takes no default
    #[arg(long, default_value_t = DEFAULT_WAITING_REQUESTS, value_parser = overload_waiting_requests, allow_negative_numbers = true, help_heading = "Load Monitoring")]
    worker_overload_waiting_requests: usize,

    /// KV usage a worker's load report states, at or above which an
    /// engine's worker that registers no usage of its own is vetoed as
    /// overloaded; above 0 and at most 1; the gateway's flag of the same
    /// name. An external worker takes no default
    #[arg(long, default_value_t = DEFAULT_TOKEN_USAGE_CEILING, value_parser = overload_token_usage, allow_negative_numbers = true, help_heading = "Load Monitoring")]
    worker_overload_token_usage: f64,

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

/// Why the value a flag is typed with is refused: the parser says it beside
/// the flag's name and the value.
type Refusal = Box<dyn std::error::Error + Send + Sync>;

/// The count `--worker-overload-waiting-requests` is typed with, held to
/// the configuration's bounds where the command line is read. The flag
/// takes a negative number as its value, so that one is refused here too.
fn overload_waiting_requests(typed: &str) -> Result<usize, Refusal> {
    let count: usize = typed.parse()?;
    Config::overload_in_bounds(OverloadThresholds {
        waiting_requests: Some(count),
        token_usage: None,
    })?;
    Ok(count)
}

/// The usage `--worker-overload-token-usage` is typed with, held as the
/// count is.
fn overload_token_usage(typed: &str) -> Result<f64, Refusal> {
    let usage: f64 = typed.parse()?;
    Config::overload_in_bounds(OverloadThresholds {
        waiting_requests: None,
        token_usage: Some(usage),
    })?;
    Ok(usage)
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
                Scope::Control => self.control_client_ca.as_ref(),
                Scope::Replication => self.replication_client_ca.as_ref(),
            }
        };
        if identity.is_none() {
            if let Some(scope) = Scope::ALL.into_iter().find(|s| client_ca(*s).is_some()) {
                return Err(ConfigError::ClientCaWithoutTls(scope));
            }
        }

        // Each listener mounts one scope, so its credential rule is that
        // scope's authority.
        let mounts = [
            ("gateway", self.gateway_listen, Scope::Gateway),
            ("control", self.control_listen, Scope::Control),
            ("replication", self.replication_listen, Scope::Replication),
        ];
        let listeners = mounts
            .into_iter()
            .map(|(name, addr, scope)| Listener {
                name: name.to_owned(),
                addr,
                scopes: vec![scope],
                tls: identity.map(|(cert, key)| Tls {
                    cert: cert.clone(),
                    key: key.clone(),
                    client_ca: client_ca(scope).cloned(),
                }),
            })
            .collect();
        Ok(Config {
            controller_id: self.controller_id.clone(),
            listeners,
            admin: self.admin_listen,
            shutdown_settle: Duration::from_secs(self.shutdown_settle_secs),
            shutdown_grace: Duration::from_secs(self.shutdown_grace_period_secs),
            load_poll_interval: Duration::from_secs(self.load_monitor_interval),
            worker_overload: OverloadThresholds {
                waiting_requests: Some(self.worker_overload_waiting_requests),
                token_usage: Some(self.worker_overload_token_usage),
            },
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
        .unwrap_or_else(|_| EnvFilter::new(format!("warn,smg_grand_central={level}")));
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
#[path = "../../../tests/unit/service/controller/run/tests.rs"]
mod tests;
