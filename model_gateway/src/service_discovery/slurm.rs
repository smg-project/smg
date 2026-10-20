//! Slurm worker discovery: a worker on every node of each running job whose
//! name maps to a worker type.
//!
//! Each poll runs three commands, without a shell, each within
//! [`COMMAND_TIMEOUT`]:
//!
//! 1. `squeue --json --local --states=RUNNING --name=<every mapped name>`,
//!    scoped to one user (`--me` unless one is configured) and to the
//!    configured account and partition;
//! 2. `scontrol show hostnames <nodes>` for each job, so Slurm's own hostlist
//!    grammar expands an allocation such as `gpu[01-04]`;
//! 3. `scontrol --json show nodes <name,...>` once, for every node named.
//!
//! A node's worker listens on the configured port at the node's `NodeAddr`,
//! which Slurm defaults to `NodeHostname` and that to the node's name. Every
//! selected job's nodes run one server each on that port, so worker jobs
//! should run `--exclusive` (or on nodes of their own): Slurm otherwise shares
//! a node between jobs.
//!
//! A command that fails or times out, or an answer that does not parse,
//! rejects the whole poll, and the last good one stays. A job that cannot be
//! placed is left out alone: one whose hostlist does not expand, with a node
//! `scontrol` does not know or without a usable address, or running with no
//! start time. An address two placements claim stays with the job that
//! started first, whose server holds the port since a later one cannot bind
//! it, and is left out for the others. What a poll leaves out is counted on
//! `smg_discovery_skipped_total` and logged when it first appears.
//!
//! `squeue`'s answer is filtered again here, by cluster, state, name, user,
//! account and partition: under `--json`, Slurm 23.11 ignores every one of
//! those filters but `--user`, so a pending job, or another user's job with a
//! mapped name, would otherwise become a worker.
//!
//! Discovery stays within the local cluster, the one `squeue` reports running
//! in. A federation can show its other clusters' jobs by default
//! (`FederationParameters=fed_display`, or `SQUEUE_FEDERATION`), but
//! `scontrol` resolves node names in the local cluster only: a remote job's
//! `gpu01` would take the local `gpu01`'s address. `--local` asks for local
//! jobs, and the cluster each job names is checked as well. 23.11 is the oldest release this reads;
//! `squeue` reports its release, and older ones are refused (21.08, for one,
//! has no `scontrol --json`).
//!
//! A worker's id is its job's cluster, id, start time and restart count, plus
//! the node and the worker type. A requeued job keeps its id but starts again,
//! so its workers are new instances at the same addresses.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    net::Ipv6Addr,
    path::{Path, PathBuf},
    process::Stdio,
    sync::Arc,
    time::Duration,
};

use async_trait::async_trait;
use openai_protocol::worker::WorkerType;
use serde::Deserialize;
use tokio::{process::Command, task, time};
use tracing::{info, warn};

use super::{
    provider::{DiscoveredWorker, DiscoveryKind},
    reconciler::DesiredState,
    snapshot::{LastGood, SnapshotError},
};
use crate::{
    app_context::AppContext,
    config::{ConfigError, ConfigResult, RoutingMode, SlurmDiscoveryConfig},
    observability::metrics::{metrics_labels, Metrics},
    worker::{endpoint::Endpoint, EndpointKey},
};

/// The oldest Slurm release this provider reads, as (major, minor).
const MIN_RELEASE: (u32, u32) = (23, 11);

/// How long one Slurm command may run before the poll is given up.
const COMMAND_TIMEOUT: Duration = Duration::from_secs(30);

/// How much of a command's stderr an error repeats.
const STDERR_EXCERPT: usize = 200;

/// Slurm discovery as the running gateway needs it.
#[derive(Debug, Clone)]
pub struct SlurmProviderConfig {
    /// The worker type of each job name, in name order so that `--name` is
    /// the same on every poll.
    roles: BTreeMap<String, WorkerType>,
    port: u16,
    bootstrap_port: Option<u16>,
    user: Option<String>,
    account: Option<String>,
    partition: Option<String>,
    /// Interval between polls.
    pub check_interval: Duration,
}

impl SlurmProviderConfig {
    /// The router's validation has checked the mapping against the routing
    /// mode; a mode with no worker pools to fill is still refused here.
    pub fn new(config: &SlurmDiscoveryConfig, mode: &RoutingMode) -> ConfigResult<Self> {
        if !matches!(
            mode,
            RoutingMode::Regular { .. }
                | RoutingMode::PrefillDecode { .. }
                | RoutingMode::EncodePrefillDecode { .. }
        ) {
            return Err(ConfigError::ValidationFailed {
                reason: "Slurm discovery needs regular, prefill-decode or encode-prefill-decode \
                         routing"
                    .to_string(),
            });
        }
        let roles = [
            (&config.job_names, WorkerType::Regular),
            (&config.encode_job_names, WorkerType::Encode),
            (&config.prefill_job_names, WorkerType::Prefill),
            (&config.decode_job_names, WorkerType::Decode),
        ]
        .into_iter()
        .flat_map(|(names, worker_type)| names.iter().map(move |name| (name.clone(), worker_type)))
        .collect();
        Ok(Self {
            roles,
            port: config.port,
            bootstrap_port: config.bootstrap_port,
            user: config.user.clone(),
            account: config.account.clone(),
            partition: config.partition.clone(),
            check_interval: Duration::from_secs(config.check_interval_secs),
        })
    }

    /// The `squeue` arguments that ask for this configuration's jobs.
    fn squeue_args(&self) -> Vec<String> {
        let names: Vec<&str> = self.roles.keys().map(String::as_str).collect();
        let mut args = vec![
            "--json".to_string(),
            "--local".to_string(),
            "--states=RUNNING".to_string(),
            format!("--name={}", names.join(",")),
            match &self.user {
                Some(user) => format!("--user={user}"),
                None => "--me".to_string(),
            },
        ];
        args.extend(
            self.account
                .iter()
                .map(|account| format!("--account={account}")),
        );
        args.extend(
            self.partition
                .iter()
                .map(|partition| format!("--partition={partition}")),
        );
        args
    }

    /// Whether `job` is one `squeue`'s filters ask for: in `cluster`, running,
    /// submitted by `user` (a name, or a numeric id), and in the configured
    /// account and partition. Its name is checked against the mapping
    /// separately.
    ///
    /// Slurm stores a job's account lowercased (`-A ML-Team` runs as
    /// `ml-team`) and `squeue --account` matches it in any case, so the
    /// account is compared the same way. Partition names are case-sensitive.
    fn selects(&self, job: &Job, cluster: &str, user: &str) -> bool {
        job.cluster == cluster
            && job.states.iter().any(|state| state == "RUNNING")
            && (job.user_name == user || user.parse::<u64>() == Ok(job.user_id))
            && self
                .account
                .as_ref()
                .is_none_or(|account| account.eq_ignore_ascii_case(&job.account))
            && self
                .partition
                .as_ref()
                .is_none_or(|partition| *partition == job.partition)
    }
}

/// One job, as much of it as discovery reads.
#[derive(Debug, Clone, PartialEq)]
struct Job {
    cluster: String,
    /// Unique per array task: each task of an array job has its own.
    id: u64,
    name: String,
    /// The base state, then any state flags.
    states: Vec<String>,
    /// The allocation as a compressed hostlist, such as `gpu[01-04]`.
    nodes: String,
    /// Unset for a pending job with no estimate; a running one has it.
    start_time: Option<i64>,
    restart_count: u64,
    user_name: String,
    user_id: u64,
    account: String,
    partition: String,
}

/// `squeue`'s answer.
#[derive(Debug, Clone, PartialEq)]
struct Jobs {
    /// The cluster `squeue` ran in, the only one whose nodes `scontrol`
    /// resolves.
    cluster: Option<String>,
    /// The user `squeue` ran as, which is who the default scope selects.
    requested_by: Option<String>,
    jobs: Vec<Job>,
}

/// One node, as much of it as discovery reads.
#[derive(Debug, Clone, PartialEq)]
struct Node {
    name: String,
    /// `NodeAddr`.
    address: String,
    /// `NodeHostname`.
    hostname: String,
}

/// The three questions a poll asks Slurm. The commands answer them today;
/// `slurmrestd` could, behind the same interface.
#[async_trait]
trait SlurmClient: Send + Sync {
    /// The jobs `squeue` lists for this configuration, filtered or not.
    async fn running_jobs(&self, config: &SlurmProviderConfig) -> Result<Jobs, SlurmError>;
    /// The node names a hostlist stands for.
    async fn expand_hostlist(&self, hostlist: &str) -> Result<Vec<String>, SlurmError>;
    /// The records of the named nodes; a name Slurm does not know is left out.
    async fn nodes(&self, names: &[String]) -> Result<Vec<Node>, SlurmError>;
}

/// Why a poll produced no snapshot.
#[derive(Debug, thiserror::Error)]
enum SlurmError {
    /// A command could not start, failed or ran out of time.
    #[error("Slurm poll failed: `{command}` {reason}")]
    Command { command: String, reason: String },
    /// The commands ran, but what they answered cannot be used.
    #[error("Slurm poll rejected: {reason}")]
    Invalid { reason: String },
}

impl SnapshotError for SlurmError {
    fn metric_reason(&self) -> &'static str {
        match self {
            Self::Command { .. } => metrics_labels::SNAPSHOT_ERROR_UNREADABLE,
            Self::Invalid { .. } => metrics_labels::SNAPSHOT_ERROR_INVALID,
        }
    }
}

fn invalid(reason: impl Into<String>) -> SlurmError {
    SlurmError::Invalid {
        reason: reason.into(),
    }
}

/// What a poll found: the workers to publish, and what it left out.
#[derive(Debug, Default)]
struct PollOutcome {
    workers: Vec<DiscoveredWorker>,
    skipped: Vec<Skip>,
}

/// A job, or an address, a poll left out of an answer it otherwise applied.
#[derive(Debug, Clone, PartialEq)]
struct Skip {
    /// The `reason` of `smg_discovery_skipped_total`.
    reason: &'static str,
    message: String,
}

impl Skip {
    fn job(job: &Job, why: impl std::fmt::Display) -> Self {
        Self {
            reason: metrics_labels::DISCOVERY_SKIP_INVALID,
            message: format!("job {} ({}): {why}", job.id, job.name),
        }
    }
}

/// One poll: a worker on every node of every selected job, or why there is
/// no answer.
///
/// An answer that cannot be read rejects the poll: a command that fails or
/// times out, or one whose output does not parse. A job that cannot be placed
/// is left out alone, and the rest of the poll stands: one whose hostlist does
/// not expand, with a node `scontrol` does not know or that has no usable
/// address, or running with no start time. An address two placements claim
/// stays with the job that started first (the lower id within a second),
/// whose server holds the port, and is left out for the later ones.
async fn poll(
    client: &dyn SlurmClient,
    config: &SlurmProviderConfig,
) -> Result<PollOutcome, SlurmError> {
    let answer = client.running_jobs(config).await?;
    let user = match (&config.user, &answer.requested_by) {
        (Some(user), _) | (None, Some(user)) => user.as_str(),
        (None, None) => {
            return Err(invalid(
                "squeue did not name the user it ran as, so the jobs to discover are \
                 unknown; configure the Slurm user",
            ))
        }
    };
    let cluster = answer.cluster.as_deref().ok_or_else(|| {
        invalid(
            "squeue did not name the cluster it ran in, so which jobs' nodes scontrol \
             resolves is unknown",
        )
    })?;

    let mut skipped = Vec::new();
    let mut placements = Vec::new();
    for job in answer.jobs {
        let Some(&worker_type) = config.roles.get(&job.name) else {
            continue;
        };
        if !config.selects(&job, cluster, user) {
            continue;
        }
        // Part of the instance's identity, so only a placed job needs one.
        let Some(start_time) = job.start_time else {
            skipped.push(Skip::job(&job, "running with no start time"));
            continue;
        };
        let nodes = match client.expand_hostlist(&job.nodes).await {
            Ok(nodes) => nodes,
            Err(SlurmError::Invalid { reason }) => {
                skipped.push(Skip::job(&job, reason));
                continue;
            }
            Err(error) => return Err(error),
        };
        placements.push((job, start_time, worker_type, nodes));
    }
    // Asking `scontrol` about no node at all would list every node.
    if placements.is_empty() {
        return Ok(PollOutcome {
            workers: Vec::new(),
            skipped,
        });
    }

    let names: BTreeSet<&String> = placements
        .iter()
        .flat_map(|(_, _, _, nodes)| nodes.iter())
        .collect();
    let names: Vec<String> = names.into_iter().cloned().collect();
    let nodes: HashMap<String, Node> = client
        .nodes(&names)
        .await?
        .into_iter()
        .map(|node| (node.name.clone(), node))
        .collect();

    let mut placed = Vec::new();
    'jobs: for (job, start_time, worker_type, node_names) in &placements {
        let mut workers = Vec::new();
        for node_name in node_names {
            let Some(node) = nodes.get(node_name) else {
                skipped.push(Skip::job(
                    job,
                    format!("node {node_name} is missing from scontrol's answer"),
                ));
                continue 'jobs;
            };
            let endpoint = match node_endpoint(node, config.port) {
                Ok(endpoint) => endpoint,
                Err(why) => {
                    skipped.push(Skip::job(job, why));
                    continue 'jobs;
                }
            };
            workers.push(DiscoveredWorker {
                discovery_id: format!(
                    "{}/{}/{}/{}/{}/{worker_type}",
                    job.cluster, job.id, start_time, job.restart_count, node.name
                ),
                endpoint,
                worker_type: *worker_type,
                bootstrap_port: config
                    .bootstrap_port
                    .filter(|_| matches!(worker_type, WorkerType::Encode | WorkerType::Prefill)),
                model_id_override: None,
                kv_connector: None,
                kv_role: None,
                kv_engine_id: None,
                compat_labels: BTreeMap::new(),
            });
        }
        let started = (*start_time, job.id);
        placed.extend(workers.into_iter().map(|worker| (started, worker)));
    }

    // Two placements at one address: Slurm shares a node between jobs unless
    // they run `--exclusive`, but only one server can listen on the port. The
    // job that started first (the lower id within a second) holds it, as a
    // later server cannot bind a port in use, so it keeps the address.
    let mut holders: HashMap<EndpointKey, ((i64, u64), String)> = HashMap::new();
    for (started, worker) in &placed {
        let holder = holders
            .entry(worker.endpoint.key())
            .or_insert_with(|| (*started, worker.discovery_id.clone()));
        if *started < holder.0 {
            *holder = (*started, worker.discovery_id.clone());
        }
    }
    let mut workers = Vec::new();
    for (_, worker) in placed {
        let (_, holder) = &holders[&worker.endpoint.key()];
        if *holder == worker.discovery_id {
            workers.push(worker);
            continue;
        }
        skipped.push(Skip {
            reason: metrics_labels::DISCOVERY_SKIP_CONFLICT,
            message: format!(
                "{} places a worker at {}, where {holder} started first and holds the \
                 configured port; run worker jobs with --exclusive",
                worker.discovery_id,
                worker.endpoint.redacted()
            ),
        });
    }
    Ok(PollOutcome { workers, skipped })
}

/// The worker endpoint on `node`: its `NodeAddr`, else its `NodeHostname`,
/// on `port`. Never its name alone, which need not resolve.
fn node_endpoint(node: &Node, port: u16) -> Result<Endpoint, String> {
    let host = [&node.address, &node.hostname]
        .into_iter()
        .find(|host| !host.is_empty())
        .ok_or_else(|| format!("node {} has no address", node.name))?;
    let authority = if host.parse::<Ipv6Addr>().is_ok() {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    };
    Endpoint::parse(&authority).map_err(|error| {
        format!(
            "node {} has an address no worker can be reached at ({host}): {error}",
            node.name
        )
    })
}

// ---------------------------------------------------------------------------
// The commands, and the JSON they print.
// ---------------------------------------------------------------------------

/// [`SlurmClient`] over the `squeue` and `scontrol` commands.
struct SlurmCli {
    squeue: PathBuf,
    scontrol: PathBuf,
    timeout: Duration,
}

impl Default for SlurmCli {
    /// The commands on `PATH`.
    fn default() -> Self {
        Self {
            squeue: PathBuf::from("squeue"),
            scontrol: PathBuf::from("scontrol"),
            timeout: COMMAND_TIMEOUT,
        }
    }
}

/// What a command that exited successfully printed.
struct CommandOutput {
    /// The command line, for errors.
    command: String,
    stdout: Vec<u8>,
    /// The first line of stderr, shortened.
    stderr: Option<String>,
}

impl CommandOutput {
    /// An answer of this command that cannot be used.
    fn invalid(&self, reason: impl std::fmt::Display) -> SlurmError {
        let stderr = self
            .stderr
            .as_ref()
            .map(|stderr| format!(" (stderr: {stderr})"))
            .unwrap_or_default();
        invalid(format!("`{}` {reason}{stderr}", self.command))
    }
}

impl SlurmCli {
    /// Run a command to completion, without a shell, and return what it
    /// printed. Running out of time kills it.
    async fn run(&self, program: &Path, args: Vec<String>) -> Result<CommandOutput, SlurmError> {
        let command = std::iter::once(program.display().to_string())
            .chain(args.iter().cloned())
            .collect::<Vec<_>>()
            .join(" ");
        let failed = |reason: String| SlurmError::Command {
            command: command.clone(),
            reason,
        };
        let child = Command::new(program)
            .args(&args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|error| failed(format!("could not start: {error}")))?;
        // Dropping the child on the timeout kills it.
        let output = match time::timeout(self.timeout, child.wait_with_output()).await {
            Ok(Ok(output)) => output,
            Ok(Err(error)) => return Err(failed(format!("failed: {error}"))),
            Err(_) => {
                return Err(failed(format!(
                    "timed out after {}s",
                    self.timeout.as_secs_f64()
                )))
            }
        };
        let stderr = excerpt(&output.stderr);
        if !output.status.success() {
            let stderr = stderr
                .map(|stderr| format!(": {stderr}"))
                .unwrap_or_default();
            return Err(failed(format!("exited with {}{stderr}", output.status)));
        }
        Ok(CommandOutput {
            command,
            stdout: output.stdout,
            stderr,
        })
    }
}

/// The first line a command wrote to stderr, shortened.
fn excerpt(stderr: &[u8]) -> Option<String> {
    let text = String::from_utf8_lossy(stderr);
    let line = text.lines().map(str::trim).find(|line| !line.is_empty())?;
    Some(line.chars().take(STDERR_EXCERPT).collect())
}

#[async_trait]
impl SlurmClient for SlurmCli {
    async fn running_jobs(&self, config: &SlurmProviderConfig) -> Result<Jobs, SlurmError> {
        let output = self.run(&self.squeue, config.squeue_args()).await?;
        parse_jobs(&output.stdout).map_err(|reason| output.invalid(reason))
    }

    async fn expand_hostlist(&self, hostlist: &str) -> Result<Vec<String>, SlurmError> {
        let args = vec![
            "show".to_string(),
            "hostnames".to_string(),
            hostlist.to_string(),
        ];
        let output = self.run(&self.scontrol, args).await?;
        parse_hostnames(&output.stdout).map_err(|reason| output.invalid(reason))
    }

    async fn nodes(&self, names: &[String]) -> Result<Vec<Node>, SlurmError> {
        let args = vec![
            "--json".to_string(),
            "show".to_string(),
            "nodes".to_string(),
            names.join(","),
        ];
        let output = self.run(&self.scontrol, args).await?;
        parse_nodes(&output.stdout).map_err(|reason| output.invalid(reason))
    }
}

/// A number Slurm prints either plainly or, from 23.02, as
/// `{"set": .., "infinite": .., "number": ..}`.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Number {
    Plain(i64),
    Wrapped { set: bool, number: i64 },
}

impl Number {
    fn value(&self) -> Option<i64> {
        match *self {
            Self::Plain(number) => Some(number),
            Self::Wrapped { set, number } => set.then_some(number),
        }
    }
}

/// The release alone, read before the rest so an old one is reported as
/// such rather than as whatever field its shape lacks.
#[derive(Deserialize)]
struct Release {
    meta: ReleaseMeta,
}

#[derive(Deserialize)]
struct ReleaseMeta {
    /// `Slurm` before 23.02.
    #[serde(alias = "Slurm")]
    slurm: Option<ReleaseName>,
}

#[derive(Deserialize)]
struct ReleaseName {
    release: String,
}

#[derive(Deserialize)]
struct SqueueAnswer {
    meta: SqueueMeta,
    #[serde(default)]
    errors: Vec<serde_json::Value>,
    jobs: Vec<SqueueJob>,
}

#[derive(Deserialize)]
struct SqueueMeta {
    client: Option<SqueueClient>,
    slurm: Option<SqueueSlurm>,
}

#[derive(Deserialize)]
struct SqueueSlurm {
    cluster: Option<String>,
}

#[derive(Deserialize)]
struct SqueueClient {
    user: String,
}

#[derive(Deserialize)]
struct SqueueJob {
    cluster: String,
    job_id: u64,
    name: String,
    job_state: Vec<String>,
    nodes: String,
    start_time: Number,
    restart_cnt: u64,
    user_name: String,
    user_id: u64,
    #[serde(default)]
    account: String,
    #[serde(default)]
    partition: String,
}

#[derive(Deserialize)]
struct NodesAnswer {
    #[serde(default)]
    errors: Vec<serde_json::Value>,
    nodes: Vec<NodeRecord>,
}

#[derive(Deserialize)]
struct NodeRecord {
    name: String,
    #[serde(default)]
    address: String,
    #[serde(default)]
    hostname: String,
}

/// Refuse a release older than [`MIN_RELEASE`].
fn check_release(release: &str) -> Result<(), String> {
    let mut parts = release.split('.').map(str::parse::<u32>);
    let (Some(Ok(major)), Some(Ok(minor))) = (parts.next(), parts.next()) else {
        return Err(format!(
            "reported an unrecognized Slurm release {release:?}"
        ));
    };
    if (major, minor) < MIN_RELEASE {
        return Err(format!(
            "is from Slurm {release}; Slurm discovery reads {}.{} and later",
            MIN_RELEASE.0, MIN_RELEASE.1
        ));
    }
    Ok(())
}

/// The errors a JSON answer lists, as one line.
fn describe_errors(errors: &[serde_json::Value]) -> String {
    errors
        .iter()
        .map(
            |error| match error.get("description").and_then(|d| d.as_str()) {
                Some(description) if !description.is_empty() => description.to_string(),
                _ => error.to_string(),
            },
        )
        .collect::<Vec<_>>()
        .join("; ")
}

fn parse_jobs(stdout: &[u8]) -> Result<Jobs, String> {
    let Release { meta } =
        serde_json::from_slice(stdout).map_err(|error| format!("printed no JSON: {error}"))?;
    let release = meta.slurm.ok_or("reported no Slurm release")?.release;
    check_release(&release)?;

    let answer: SqueueAnswer = serde_json::from_slice(stdout)
        .map_err(|error| format!("printed JSON this cannot read: {error}"))?;
    if !answer.errors.is_empty() {
        return Err(format!(
            "reported errors: {}",
            describe_errors(&answer.errors)
        ));
    }
    let jobs = answer
        .jobs
        .into_iter()
        .map(|job| {
            Ok(Job {
                cluster: job.cluster,
                id: job.job_id,
                name: job.name,
                states: job.job_state,
                nodes: job.nodes,
                start_time: job.start_time.value(),
                restart_count: job.restart_cnt,
                user_name: job.user_name,
                user_id: job.user_id,
                account: job.account,
                partition: job.partition,
            })
        })
        .collect::<Result<_, String>>()?;
    Ok(Jobs {
        cluster: answer
            .meta
            .slurm
            .and_then(|slurm| slurm.cluster)
            .filter(|cluster| !cluster.is_empty()),
        requested_by: answer
            .meta
            .client
            .map(|client| client.user)
            .filter(|user| !user.is_empty()),
        jobs,
    })
}

/// One node name per line. An answer that names no node, or that is not a
/// node name at all, is refused: `scontrol` exits 0 even on a malformed
/// hostlist.
fn parse_hostnames(stdout: &[u8]) -> Result<Vec<String>, String> {
    let text = std::str::from_utf8(stdout).map_err(|_| "printed text that is not UTF-8")?;
    let names: Vec<String> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect();
    if names.is_empty() {
        return Err("expanded to no node".to_string());
    }
    if let Some(bad) = names
        .iter()
        .find(|name| name.contains(|c: char| c.is_whitespace() || "[],".contains(c)))
    {
        return Err(format!("expanded to {bad:?}, which is not a node name"));
    }
    Ok(names)
}

fn parse_nodes(stdout: &[u8]) -> Result<Vec<Node>, String> {
    let answer: NodesAnswer = serde_json::from_slice(stdout)
        .map_err(|error| format!("printed JSON this cannot read: {error}"))?;
    if !answer.errors.is_empty() {
        return Err(format!(
            "reported errors: {}",
            describe_errors(&answer.errors)
        ));
    }
    Ok(answer
        .nodes
        .into_iter()
        .map(|node| Node {
            name: node.name,
            address: node.address,
            hostname: node.hostname,
        })
        .collect())
}

// ---------------------------------------------------------------------------
// The provider's loop.
// ---------------------------------------------------------------------------

/// Poll Slurm now and on every tick, reconciling the last good answer each
/// time.
pub(super) fn start_slurm_discovery(
    config: SlurmProviderConfig,
    app_context: Arc<AppContext>,
) -> task::JoinHandle<()> {
    info!(
        "Starting Slurm service discovery | jobs: {} | port: {} | poll every {}s",
        config
            .roles
            .iter()
            .map(|(name, worker_type)| format!("{name} ({worker_type})"))
            .collect::<Vec<_>>()
            .join(", "),
        config.port,
        config.check_interval.as_secs_f64()
    );

    #[expect(
        clippy::disallowed_methods,
        reason = "worker discovery runs for the lifetime of the server; shutdown aborts the handle"
    )]
    let handle =
        task::spawn(SlurmDiscovery::new(config, app_context, Box::new(SlurmCli::default())).run());

    handle
}

/// The provider's loop and its state.
struct SlurmDiscovery {
    config: SlurmProviderConfig,
    app_context: Arc<AppContext>,
    client: Box<dyn SlurmClient>,
    last_good: LastGood,
    /// What the last poll left out, so a skip that lasts is logged once.
    skipped: HashSet<String>,
}

impl SlurmDiscovery {
    fn new(
        config: SlurmProviderConfig,
        app_context: Arc<AppContext>,
        client: Box<dyn SlurmClient>,
    ) -> Self {
        Self {
            config,
            app_context,
            client,
            last_good: LastGood::new(DiscoveryKind::Slurm, "Slurm".to_string(), "job list"),
            skipped: HashSet::new(),
        }
    }

    async fn run(mut self) {
        let mut interval = time::interval(self.config.check_interval);
        interval.set_missed_tick_behavior(time::MissedTickBehavior::Delay);
        loop {
            interval.tick().await;
            let started_at = time::Instant::now();
            self.repoll().await;
            self.last_good
                .reconcile(&self.app_context, started_at)
                .await;
        }
    }

    /// Poll into the last good snapshot. A poll that fails is reported and
    /// leaves the snapshot in place; what one leaves out is counted on every
    /// poll and logged when it first appears.
    async fn repoll(&mut self) {
        let read = poll(self.client.as_ref(), &self.config)
            .await
            .map(|outcome| {
                let mut skipped = HashSet::new();
                for skip in outcome.skipped {
                    Metrics::record_discovery_skip(
                        DiscoveryKind::Slurm.metric_label(),
                        skip.reason,
                    );
                    if !self.skipped.contains(&skip.message) {
                        warn!("Slurm discovery left out {}", skip.message);
                    }
                    skipped.insert(skip.message);
                }
                self.skipped = skipped;
                DesiredState::from_workers(outcome.workers)
            });
        self.last_good.update(read);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    const SQUEUE_23_11: &str = include_str!("testdata/slurm/squeue-23.11.json");
    const SQUEUE_24_11: &str = include_str!("testdata/slurm/squeue-24.11.json");
    const SQUEUE_21_08: &str = include_str!("testdata/slurm/squeue-21.08.json");
    const NODES_23_11: &str = include_str!("testdata/slurm/scontrol-nodes-23.11.json");
    const NODES_24_11: &str = include_str!("testdata/slurm/scontrol-nodes-24.11.json");

    fn regular() -> RoutingMode {
        RoutingMode::Regular {
            worker_urls: vec![],
        }
    }

    fn epd() -> RoutingMode {
        RoutingMode::EncodePrefillDecode {
            encode_urls: vec![],
            prefill_urls: vec![],
            decode_urls: vec![],
            encode_policy: None,
            prefill_policy: None,
            decode_policy: None,
        }
    }

    fn discovery_config(job_names: &[&str]) -> SlurmDiscoveryConfig {
        SlurmDiscoveryConfig {
            job_names: job_names.iter().map(|name| name.to_string()).collect(),
            encode_job_names: vec![],
            prefill_job_names: vec![],
            decode_job_names: vec![],
            port: 8000,
            bootstrap_port: None,
            user: None,
            account: None,
            partition: None,
            check_interval_secs: 30,
        }
    }

    fn config(job_names: &[&str]) -> SlurmProviderConfig {
        SlurmProviderConfig::new(&discovery_config(job_names), &regular()).unwrap()
    }

    fn job(jobs: &Jobs, id: u64) -> &Job {
        jobs.jobs.iter().find(|job| job.id == id).unwrap()
    }

    // ---- What the recordings say. ----

    /// Both supported releases read into the same jobs: the requeued one with
    /// its restart count, the two array tasks under their own ids, the
    /// pending one, alice's, and the user `squeue` ran as.
    #[test]
    fn the_recorded_job_lists_read_alike() {
        for (recording, alice, pending, second_task) in
            [(SQUEUE_23_11, 8, 6, 7), (SQUEUE_24_11, 6, 7, 8)]
        {
            let jobs = parse_jobs(recording.as_bytes()).unwrap();
            assert_eq!(jobs.cluster.as_deref(), Some("smgtest"));
            assert_eq!(jobs.requested_by.as_deref(), Some("root"));
            assert_eq!(jobs.jobs.len(), 8);

            let requeued = job(&jobs, 4);
            assert_eq!(requeued.name, "smg-decode");
            assert_eq!(requeued.restart_count, 1);
            assert_eq!(requeued.states, ["RUNNING"]);
            assert_eq!(requeued.nodes, "slurmnode");
            assert_eq!(requeued.cluster, "smgtest");
            assert!(requeued.start_time.is_some_and(|start| start > 0));

            for id in [3, second_task] {
                assert_eq!(job(&jobs, id).name, "smg-prefill");
                assert_eq!(job(&jobs, id).restart_count, 0);
            }
            assert_eq!(job(&jobs, alice).user_name, "alice");
            assert_eq!(job(&jobs, pending).states, ["PENDING"]);
            assert_eq!(job(&jobs, 2).account, "acct-b");
            assert_eq!(job(&jobs, 2).partition, "gpu");
            assert_eq!(job(&jobs, 1).account, "");
        }
    }

    /// 21.08 prints a different shape and has no `scontrol --json`: it is
    /// refused by its release, not by whatever field it lacks.
    #[test]
    fn an_older_release_is_refused_by_name() {
        let reason = parse_jobs(SQUEUE_21_08.as_bytes()).unwrap_err();
        assert!(reason.contains("Slurm 21.08.5"), "{reason}");
        assert!(reason.contains("23.11 and later"), "{reason}");

        assert!(check_release("23.11.4").is_ok());
        assert!(check_release("24.11.5").is_ok());
        assert!(check_release("25.05.0").is_ok());
        assert!(check_release("23.02.7").is_err());
        assert!(check_release("garbage").is_err());
    }

    #[test]
    fn an_unset_start_time_reads_as_none() {
        let mut answer: serde_json::Value = serde_json::from_str(SQUEUE_24_11).unwrap();
        answer["jobs"][0]["start_time"] =
            serde_json::json!({"set": false, "infinite": false, "number": 0});
        let jobs = parse_jobs(answer.to_string().as_bytes()).unwrap();
        assert_eq!(jobs.jobs[0].start_time, None);
    }

    #[test]
    fn errors_in_an_answer_reject_it() {
        let mut answer: serde_json::Value = serde_json::from_str(SQUEUE_24_11).unwrap();
        answer["errors"] = serde_json::json!([{"description": "controller unreachable"}]);
        let reason = parse_jobs(answer.to_string().as_bytes()).unwrap_err();
        assert!(reason.contains("controller unreachable"), "{reason}");

        assert!(parse_jobs(b"squeue: error: Invalid user").is_err());
    }

    /// `NodeAddr` is reported whether set or not: unset, Slurm reports
    /// `NodeHostname`, and unset too, the node's name.
    #[test]
    fn the_recorded_nodes_report_an_address_for_each() {
        for recording in [NODES_23_11, NODES_24_11] {
            let nodes = parse_nodes(recording.as_bytes()).unwrap();
            let addresses: Vec<(&str, &str, &str)> = nodes
                .iter()
                .map(|n| (n.name.as_str(), n.address.as_str(), n.hostname.as_str()))
                .collect();
            assert_eq!(
                addresses,
                [
                    ("slurmnode", "127.0.0.1", "slurmnode"),
                    ("gpu01", "10.0.0.11", "gpu01.cluster"),
                    ("gpu02", "gpu02.cluster", "gpu02.cluster"),
                    ("gpu03", "gpu03", "gpu03"),
                ]
            );
        }
    }

    /// `scontrol show hostnames` exits 0 on a malformed hostlist: 23.11 prints
    /// nothing, and 21.08 printed `gpu]`. Both are refused.
    #[test]
    fn hostname_expansion_refuses_what_is_not_a_node_list() {
        assert_eq!(
            parse_hostnames(b"gpu01\ngpu02\nnode-10\n").unwrap(),
            ["gpu01", "gpu02", "node-10"]
        );
        assert!(parse_hostnames(b"").is_err());
        assert!(parse_hostnames(b"gpu]\n").is_err());
        assert!(parse_hostnames(b"a b\n").is_err());
    }

    // ---- A poll, against a scripted Slurm. ----

    /// Answers from fixed data, and records the node lookups it was asked.
    struct FakeSlurm {
        jobs: Jobs,
        hostlists: HashMap<String, Vec<String>>,
        nodes: Vec<Node>,
        node_lookups: Mutex<Vec<Vec<String>>>,
        /// `scontrol show hostnames` failing as a command, not answering badly.
        expansion_fails: bool,
    }

    impl FakeSlurm {
        /// The 23.11 recording, which ignored every filter but `--user`, with
        /// each job given its own node `n<id>` at `10.0.1.<id>`.
        fn recorded() -> Self {
            let mut jobs = parse_jobs(SQUEUE_23_11.as_bytes()).unwrap();
            let mut hostlists = HashMap::new();
            let mut nodes = Vec::new();
            for job in &mut jobs.jobs {
                job.nodes = format!("n{}", job.id);
                hostlists.insert(job.nodes.clone(), vec![job.nodes.clone()]);
                nodes.push(Node {
                    name: job.nodes.clone(),
                    address: format!("10.0.1.{}", job.id),
                    hostname: format!("{}.cluster", job.nodes),
                });
            }
            Self {
                jobs,
                hostlists,
                nodes,
                node_lookups: Mutex::new(Vec::new()),
                expansion_fails: false,
            }
        }

        fn job_mut(&mut self, id: u64) -> &mut Job {
            self.jobs.jobs.iter_mut().find(|job| job.id == id).unwrap()
        }
    }

    #[async_trait]
    impl SlurmClient for FakeSlurm {
        async fn running_jobs(&self, _: &SlurmProviderConfig) -> Result<Jobs, SlurmError> {
            Ok(self.jobs.clone())
        }

        async fn expand_hostlist(&self, hostlist: &str) -> Result<Vec<String>, SlurmError> {
            if self.expansion_fails {
                return Err(SlurmError::Command {
                    command: format!("scontrol show hostnames {hostlist}"),
                    reason: "timed out after 30s".to_string(),
                });
            }
            self.hostlists
                .get(hostlist)
                .cloned()
                .ok_or_else(|| invalid(format!("unknown hostlist {hostlist}")))
        }

        async fn nodes(&self, names: &[String]) -> Result<Vec<Node>, SlurmError> {
            self.node_lookups.lock().unwrap().push(names.to_vec());
            Ok(self
                .nodes
                .iter()
                .filter(|node| names.contains(&node.name))
                .cloned()
                .collect())
        }
    }

    fn urls_of(outcome: &PollOutcome) -> Vec<String> {
        let mut urls: Vec<String> = outcome
            .workers
            .iter()
            .map(|worker| worker.endpoint.render())
            .collect();
        urls.sort();
        urls
    }

    async fn urls(slurm: &FakeSlurm, config: &SlurmProviderConfig) -> Vec<String> {
        let mut urls: Vec<String> = poll(slurm, config)
            .await
            .unwrap()
            .workers
            .iter()
            .map(|worker| worker.endpoint.render())
            .collect();
        urls.sort();
        urls
    }

    /// The recorded list holds a pending job, an unrelated name and another
    /// user's job beside the two running `smg-worker` jobs of `root`; only
    /// those two become workers, because the answer is filtered here.
    #[tokio::test]
    async fn the_answer_is_filtered_by_state_name_and_user() {
        let slurm = FakeSlurm::recorded();
        assert_eq!(
            urls(&slurm, &config(&["smg-worker"])).await,
            ["10.0.1.1:8000", "10.0.1.2:8000"]
        );

        let alice = SlurmProviderConfig {
            user: Some("alice".to_string()),
            ..config(&["smg-worker"])
        };
        assert_eq!(urls(&slurm, &alice).await, ["10.0.1.8:8000"]);
        // A numeric user is a uid; root's is 0.
        let uid = SlurmProviderConfig {
            user: Some("0".to_string()),
            ..config(&["smg-worker"])
        };
        assert_eq!(urls(&slurm, &uid).await, ["10.0.1.1:8000", "10.0.1.2:8000"]);
    }

    /// Slurm stores a job's account lowercased (checked against Slurm 24.11:
    /// `-A ACCT-B` runs as `acct-b`), so a configured account matches in any
    /// case.
    #[tokio::test]
    async fn the_account_matches_in_any_case() {
        let slurm = FakeSlurm::recorded();
        let upper = SlurmProviderConfig {
            account: Some("ACCT-B".to_string()),
            ..config(&["smg-worker"])
        };
        assert_eq!(urls(&slurm, &upper).await, ["10.0.1.2:8000"]);
    }

    /// 23.11 lists every job of the user, filters or not: one it would have
    /// filtered out may have no start time, and must not reject the poll. A
    /// job that is placed needs one, as its identity carries it.
    #[tokio::test]
    async fn only_a_placed_job_needs_a_start_time() {
        let mut slurm = FakeSlurm::recorded();
        // The pending job, and one with a name no mapping lists.
        slurm.job_mut(6).start_time = None;
        slurm.job_mut(5).start_time = None;
        assert_eq!(
            urls(&slurm, &config(&["smg-worker"])).await,
            ["10.0.1.1:8000", "10.0.1.2:8000"]
        );

        // A selected one is left out alone.
        slurm.job_mut(1).start_time = None;
        let outcome = poll(&slurm, &config(&["smg-worker"])).await.unwrap();
        assert_eq!(urls_of(&outcome), ["10.0.1.2:8000"]);
        assert_eq!(
            outcome.skipped,
            [Skip {
                reason: metrics_labels::DISCOVERY_SKIP_INVALID,
                message: "job 1 (smg-worker): running with no start time".to_string(),
            }]
        );
    }

    #[tokio::test]
    async fn the_answer_is_filtered_by_account_and_partition() {
        let slurm = FakeSlurm::recorded();
        for scoped in [
            SlurmProviderConfig {
                account: Some("acct-b".to_string()),
                ..config(&["smg-worker"])
            },
            SlurmProviderConfig {
                partition: Some("gpu".to_string()),
                ..config(&["smg-worker"])
            },
        ] {
            assert_eq!(urls(&slurm, &scoped).await, ["10.0.1.2:8000"]);
        }
    }

    /// Each mapped name gives its jobs' workers that type, and only encode and
    /// prefill workers carry the bootstrap port.
    #[tokio::test]
    async fn job_names_map_to_worker_types() {
        let slurm = FakeSlurm::recorded();
        let config = SlurmProviderConfig::new(
            &SlurmDiscoveryConfig {
                job_names: vec![],
                encode_job_names: vec!["smg-worker".to_string()],
                prefill_job_names: vec!["smg-prefill".to_string()],
                decode_job_names: vec!["smg-decode".to_string()],
                bootstrap_port: Some(8998),
                ..discovery_config(&[])
            },
            &epd(),
        )
        .unwrap();
        let mut workers: Vec<(String, WorkerType, Option<u16>)> = poll(&slurm, &config)
            .await
            .unwrap()
            .workers
            .into_iter()
            .map(|w| (w.endpoint.render(), w.worker_type, w.bootstrap_port))
            .collect();
        workers.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(
            workers,
            [
                ("10.0.1.1:8000".to_string(), WorkerType::Encode, Some(8998)),
                ("10.0.1.2:8000".to_string(), WorkerType::Encode, Some(8998)),
                ("10.0.1.3:8000".to_string(), WorkerType::Prefill, Some(8998)),
                ("10.0.1.4:8000".to_string(), WorkerType::Decode, None),
                ("10.0.1.7:8000".to_string(), WorkerType::Prefill, Some(8998)),
            ]
        );
    }

    /// A multi-node allocation places a worker on each node: at `NodeAddr`
    /// (bracketed when IPv6), else `NodeHostname`. One lookup names every node
    /// once.
    #[tokio::test]
    async fn every_node_of_an_allocation_gets_a_worker() {
        let mut slurm = FakeSlurm::recorded();
        slurm.job_mut(1).nodes = "gpu[01-04]".to_string();
        slurm.hostlists.insert(
            "gpu[01-04]".to_string(),
            ["gpu01", "gpu02", "gpu03", "gpu04"]
                .map(String::from)
                .to_vec(),
        );
        // The recorded gpu01 (NodeAddr set) and gpu02 (unset, so Slurm reports
        // NodeHostname), then an IPv6 NodeAddr and a record with only a
        // hostname.
        let recorded = parse_nodes(NODES_24_11.as_bytes()).unwrap();
        slurm.nodes.extend(
            recorded
                .into_iter()
                .filter(|node| node.name == "gpu01" || node.name == "gpu02"),
        );
        slurm.nodes.push(Node {
            name: "gpu03".to_string(),
            address: "fd00::13".to_string(),
            hostname: "gpu03".to_string(),
        });
        slurm.nodes.push(Node {
            name: "gpu04".to_string(),
            address: String::new(),
            hostname: "gpu04.cluster".to_string(),
        });

        let mut found: Vec<(String, String)> = poll(&slurm, &config(&["smg-worker"]))
            .await
            .unwrap()
            .workers
            .into_iter()
            .map(|worker| (worker.endpoint.render(), worker.discovery_id))
            .collect();
        found.sort();
        let start = |id| job(&slurm.jobs, id).start_time.unwrap();
        let on_job_1 = |node| format!("smgtest/1/{}/0/{node}/regular", start(1));
        let mut expected = vec![
            ("10.0.0.11:8000".to_string(), on_job_1("gpu01")),
            ("gpu02.cluster:8000".to_string(), on_job_1("gpu02")),
            ("[fd00::13]:8000".to_string(), on_job_1("gpu03")),
            ("gpu04.cluster:8000".to_string(), on_job_1("gpu04")),
            (
                "10.0.1.2:8000".to_string(),
                format!("smgtest/2/{}/0/n2/regular", start(2)),
            ),
        ];
        expected.sort();
        assert_eq!(found, expected);
        assert_eq!(
            *slurm.node_lookups.lock().unwrap(),
            [vec!["gpu01", "gpu02", "gpu03", "gpu04", "n2"]]
        );
    }

    /// A requeued job keeps its id but starts again: its worker is a new
    /// instance at the same address.
    #[tokio::test]
    async fn a_requeue_is_a_new_instance_at_the_same_address() {
        let mut slurm = FakeSlurm::recorded();
        let config = config(&["smg-decode"]);
        let before = poll(&slurm, &config).await.unwrap().workers;
        let job = slurm.job_mut(4);
        job.restart_count += 1;
        job.start_time = job.start_time.map(|start| start + 60);
        let after = poll(&slurm, &config).await.unwrap().workers;

        assert_eq!(before.len(), 1);
        assert_eq!(after.len(), 1);
        assert_eq!(before[0].endpoint, after[0].endpoint);
        assert_ne!(before[0].discovery_id, after[0].discovery_id);
        assert!(
            after[0].discovery_id.starts_with("smgtest/4/"),
            "{}",
            after[0].discovery_id
        );
    }

    /// A job with a node `scontrol` does not know is left out whole; the rest
    /// of the poll stands.
    #[tokio::test]
    async fn a_job_on_a_node_scontrol_does_not_know_is_left_out() {
        let mut slurm = FakeSlurm::recorded();
        slurm.nodes.retain(|node| node.name != "n2");
        let outcome = poll(&slurm, &config(&["smg-worker"])).await.unwrap();
        assert_eq!(urls_of(&outcome), ["10.0.1.1:8000"]);
        assert_eq!(
            outcome.skipped,
            [Skip {
                reason: metrics_labels::DISCOVERY_SKIP_INVALID,
                message: "job 2 (smg-worker): node n2 is missing from scontrol's answer"
                    .to_string(),
            }]
        );
    }

    /// Slurm shares a node between jobs unless they run `--exclusive`, but
    /// only one server can listen on the port: the one of the job that
    /// started first, as a later server cannot bind a port in use. That job
    /// keeps the address, a later one is left out there, and every other
    /// placement stands, the later job's other nodes included.
    #[tokio::test]
    async fn two_jobs_on_one_node_leave_it_to_the_first_started() {
        let mut slurm = FakeSlurm::recorded();
        // Job 2 shares job 1's node, and has a node of its own too.
        slurm.job_mut(2).nodes = "n1,n2".to_string();
        slurm
            .hostlists
            .insert("n1,n2".to_string(), vec!["n1".into(), "n2".into()]);
        let mut config = config(&["smg-worker", "smg-decode"]);
        config
            .roles
            .insert("smg-decode".to_string(), WorkerType::Regular);
        let on_n1 = |outcome: &PollOutcome| {
            let worker = outcome
                .workers
                .iter()
                .find(|worker| worker.endpoint.render() == "10.0.1.1:8000");
            worker.unwrap().discovery_id.clone()
        };
        let conflict = |outcome: &PollOutcome| {
            assert_eq!(outcome.skipped.len(), 1, "{:?}", outcome.skipped);
            let skip = outcome.skipped[0].clone();
            assert_eq!(skip.reason, metrics_labels::DISCOVERY_SKIP_CONFLICT);
            skip.message
        };

        // Started in the same second as job 1, job 2 is taken as the later.
        let outcome = poll(&slurm, &config).await.unwrap();
        let urls = ["10.0.1.1:8000", "10.0.1.2:8000", "10.0.1.4:8000"];
        assert_eq!(urls_of(&outcome), urls);
        assert!(on_n1(&outcome).starts_with("smgtest/1/"));
        let message = conflict(&outcome);
        assert!(
            message.starts_with("smgtest/2/")
                && message.contains("places a worker at 10.0.1.1:8000, where smgtest/1/"),
            "{message}"
        );

        // Started a second before job 1, job 2 keeps the node.
        let first = slurm.job_mut(1).start_time.unwrap();
        slurm.job_mut(2).start_time = Some(first - 1);
        let outcome = poll(&slurm, &config).await.unwrap();
        assert_eq!(urls_of(&outcome), urls);
        assert!(on_n1(&outcome).starts_with("smgtest/2/"));
        let message = conflict(&outcome);
        assert!(
            message.starts_with("smgtest/1/") && message.contains("where smgtest/2/"),
            "{message}"
        );
    }

    /// With no job to place, `scontrol` is not asked about nodes at all: asked
    /// about none, it lists every node.
    #[tokio::test]
    async fn no_selected_job_asks_about_no_node() {
        let slurm = FakeSlurm::recorded();
        assert!(poll(&slurm, &config(&["no-such-job"]))
            .await
            .unwrap()
            .workers
            .is_empty());
        assert!(slurm.node_lookups.lock().unwrap().is_empty());
    }

    /// A federation can list its other clusters' jobs (an older release may
    /// ignore `--local` under `--json`, as 23.11 ignores most filters there).
    /// `scontrol` resolves node names in the local cluster only, so a remote
    /// job on a node named like a local one must not take that node's
    /// address, nor a remote-only node reject the poll: both are left out.
    #[tokio::test]
    async fn jobs_of_another_cluster_are_left_out() {
        let mut slurm = FakeSlurm::recorded();
        // Job 2 runs in a remote cluster, on a node named like job 1's.
        let remote = slurm.job_mut(2);
        remote.cluster = "remote".to_string();
        remote.nodes = "n1".to_string();
        // Alice's job 8, as root's, runs remotely on a node the local cluster
        // does not have.
        let only_remote = slurm.job_mut(8);
        only_remote.cluster = "remote".to_string();
        only_remote.user_name = "root".to_string();
        only_remote.nodes = "remote-only".to_string();

        assert_eq!(
            urls(&slurm, &config(&["smg-worker"])).await,
            ["10.0.1.1:8000"]
        );
        assert_eq!(*slurm.node_lookups.lock().unwrap(), [vec!["n1"]]);
    }

    #[tokio::test]
    async fn the_local_cluster_must_be_named() {
        let mut slurm = FakeSlurm::recorded();
        slurm.jobs.cluster = None;
        let error = poll(&slurm, &config(&["smg-worker"])).await.unwrap_err();
        assert!(error.to_string().contains("cluster it ran in"), "{error}");
    }

    #[tokio::test]
    async fn the_default_scope_needs_squeue_to_name_its_user() {
        let mut slurm = FakeSlurm::recorded();
        slurm.jobs.requested_by = None;
        let error = poll(&slurm, &config(&["smg-worker"])).await.unwrap_err();
        assert!(
            error.to_string().contains("configure the Slurm user"),
            "{error}"
        );

        let named = SlurmProviderConfig {
            user: Some("root".to_string()),
            ..config(&["smg-worker"])
        };
        assert_eq!(urls(&slurm, &named).await.len(), 2);
    }

    /// A hostlist `scontrol` cannot expand leaves its job out; `scontrol`
    /// failing as a command leaves the answer unread, and rejects the poll.
    #[tokio::test]
    async fn an_expansion_that_fails_leaves_its_job_out_and_a_failed_command_the_poll() {
        let mut slurm = FakeSlurm::recorded();
        slurm.hostlists.remove("n1");
        let outcome = poll(&slurm, &config(&["smg-worker"])).await.unwrap();
        assert_eq!(urls_of(&outcome), ["10.0.1.2:8000"]);
        assert_eq!(outcome.skipped.len(), 1);
        assert!(outcome.skipped[0]
            .message
            .starts_with("job 1 (smg-worker): "));

        slurm.expansion_fails = true;
        let error = poll(&slurm, &config(&["smg-worker"])).await.unwrap_err();
        assert!(matches!(error, SlurmError::Command { .. }), "{error}");
    }

    #[test]
    fn squeue_is_asked_for_the_mapped_names_in_one_scope() {
        assert_eq!(
            config(&["smg-worker", "a-model"]).squeue_args(),
            [
                "--json",
                "--local",
                "--states=RUNNING",
                "--name=a-model,smg-worker",
                "--me"
            ]
        );
        let scoped = SlurmProviderConfig {
            user: Some("svc".to_string()),
            account: Some("acct".to_string()),
            partition: Some("gpu".to_string()),
            ..config(&["w"])
        };
        assert_eq!(
            scoped.squeue_args(),
            [
                "--json",
                "--local",
                "--states=RUNNING",
                "--name=w",
                "--user=svc",
                "--account=acct",
                "--partition=gpu"
            ]
        );
    }

    #[test]
    fn modes_without_worker_pools_cannot_use_slurm_discovery() {
        let openai = RoutingMode::OpenAI {
            worker_urls: vec![],
        };
        assert!(SlurmProviderConfig::new(&discovery_config(&["w"]), &openai).is_err());
    }

    // ---- The commands, against scripts standing in for them. ----

    #[cfg(unix)]
    fn script(dir: &Path, name: &str, body: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;

        let path = dir.join(name);
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    #[cfg(unix)]
    fn cli(squeue: PathBuf, scontrol: PathBuf) -> SlurmCli {
        SlurmCli {
            squeue,
            scontrol,
            timeout: Duration::from_secs(5),
        }
    }

    /// The real command path end to end: the arguments `squeue` and `scontrol`
    /// are given, and their recorded answers read back.
    #[cfg(unix)]
    #[tokio::test]
    async fn the_commands_are_run_with_their_arguments_and_read() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("squeue.json"), SQUEUE_24_11).unwrap();
        std::fs::write(dir.path().join("nodes.json"), NODES_24_11).unwrap();
        let log = dir.path().join("calls");
        let squeue = script(
            dir.path(),
            "squeue",
            &format!(
                r#"echo "squeue $*" >> {log}; cat {}"#,
                dir.path().join("squeue.json").display(),
                log = log.display()
            ),
        );
        let scontrol = script(
            dir.path(),
            "scontrol",
            &format!(
                r#"echo "scontrol $*" >> {log}
if [ "$1" = show ]; then echo slurmnode; else cat {}; fi"#,
                dir.path().join("nodes.json").display(),
                log = log.display()
            ),
        );
        let slurm = cli(squeue, scontrol);

        let workers = poll(&slurm, &config(&["smg-decode"]))
            .await
            .unwrap()
            .workers;
        assert_eq!(workers.len(), 1);
        assert_eq!(workers[0].endpoint.render(), "127.0.0.1:8000");
        assert_eq!(
            std::fs::read_to_string(&log).unwrap(),
            "squeue --json --local --states=RUNNING --name=smg-decode --me\n\
             scontrol show hostnames slurmnode\n\
             scontrol --json show nodes slurmnode\n"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_command_that_fails_says_how() {
        let dir = tempfile::tempdir().unwrap();
        let failing = script(
            dir.path(),
            "squeue",
            "echo 'squeue: error: Unable to contact slurm controller' >&2; exit 1",
        );
        let slow = script(dir.path(), "slow", "sleep 10");
        let missing = dir.path().join("absent");

        let error = cli(failing, missing.clone())
            .running_jobs(&config(&["w"]))
            .await
            .unwrap_err();
        assert!(matches!(error, SlurmError::Command { .. }), "{error}");
        assert!(
            error
                .to_string()
                .contains("Unable to contact slurm controller"),
            "{error}"
        );
        assert_eq!(
            error.metric_reason(),
            metrics_labels::SNAPSHOT_ERROR_UNREADABLE
        );

        let slurm = SlurmCli {
            timeout: Duration::from_millis(200),
            ..cli(slow, missing.clone())
        };
        let started = time::Instant::now();
        let error = slurm.running_jobs(&config(&["w"])).await.unwrap_err();
        assert!(error.to_string().contains("timed out"), "{error}");
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "the command was not killed"
        );

        let error = cli(missing.clone(), missing)
            .expand_hostlist("n1")
            .await
            .unwrap_err();
        assert!(error.to_string().contains("could not start"), "{error}");
    }

    /// An answer that exits 0 but cannot be used is invalid, and carries what
    /// the command wrote to stderr.
    #[cfg(unix)]
    #[tokio::test]
    async fn an_unusable_answer_is_invalid() {
        let dir = tempfile::tempdir().unwrap();
        let scontrol = script(
            dir.path(),
            "scontrol",
            "echo 'Invalid hostlist: gpu[01-' >&2",
        );
        let error = cli(dir.path().join("absent"), scontrol)
            .expand_hostlist("gpu[01-")
            .await
            .unwrap_err();
        assert!(matches!(error, SlurmError::Invalid { .. }), "{error}");
        assert!(error.to_string().contains("Invalid hostlist"), "{error}");
        assert_eq!(
            error.metric_reason(),
            metrics_labels::SNAPSHOT_ERROR_INVALID
        );
    }
}
