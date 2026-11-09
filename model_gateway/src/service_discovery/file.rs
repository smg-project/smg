//! File worker discovery: the workers listed in a JSON manifest on disk.
//!
//! The manifest is reread in full whenever it changes, and on every tick in
//! case a change went unreported. A read that succeeds replaces the last good
//! snapshot; one that fails is reported and leaves that snapshot in place, so a
//! half-written or broken file never removes a worker. Every read then
//! reconciles the last good snapshot through the shared reconciler, which
//! retries whatever an earlier pass left undone.
//!
//! ```json
//! {
//!   "version": 1,
//!   "workers": [
//!     { "id": "decode-a", "url": "10.0.0.23:8000", "worker_type": "decode" },
//!     { "id": "prefill-a", "url": "10.0.0.22:8000", "worker_type": "prefill",
//!       "bootstrap_port": 8998, "kv_connector": "MooncakeConnector" }
//!   ]
//! }
//! ```
//!
//! The manifest is strict: an unknown field, worker type or version rejects
//! the whole file, so a misspelled field fails visibly instead of being
//! dropped. `{"version": 1, "workers": []}` is a valid, intentionally empty
//! fleet. Writers should write a temporary file in the same directory and
//! rename it over the manifest.
//!
//! Editing a record in place, such as correcting its bootstrap port, updates
//! its worker in place with the new settings. A new `id` at a reused address
//! signals a new instance there, whose worker is removed and re-registered.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    path::PathBuf,
    sync::Arc,
    time::Duration,
};

use openai_protocol::worker::{WorkerInfo, WorkerType};
use serde::{Deserialize, Serialize};
use tokio::{sync::mpsc, task, time};
use tracing::{info, warn};

use super::{
    provider::{DiscoveredWorker, DiscoveryKind, Roles},
    reconciler::DesiredState,
    snapshot::{LastGood, SnapshotError},
};
use crate::{
    app_context::AppContext,
    config::{ConfigError, ConfigResult, RoutingMode},
    fs_watch::{settle, DirectoryWatch},
    observability::metrics::metrics_labels,
    worker::{endpoint::Endpoint, EndpointKey},
};

/// The manifest version this gateway reads.
const MANIFEST_VERSION: u64 = 1;

/// File discovery as the running gateway needs it.
#[derive(Debug, Clone)]
pub struct FileProviderConfig {
    /// The manifest. An absent file is a retryable error, not an empty fleet.
    pub path: PathBuf,
    /// Interval between full rereads.
    pub check_interval: Duration,
    /// The worker types the routing mode accepts.
    roles: Roles,
}

impl FileProviderConfig {
    /// The routing mode decides which worker types a manifest may list; a mode
    /// with no worker pools to fill is rejected.
    pub fn new(path: PathBuf, check_interval: Duration, mode: &RoutingMode) -> ConfigResult<Self> {
        let roles = Roles::of(mode).ok_or_else(|| ConfigError::ValidationFailed {
            reason: "File discovery needs regular, prefill-decode or encode-prefill-decode routing"
                .to_string(),
        })?;
        Ok(Self {
            path,
            check_interval,
            roles,
        })
    }
}

/// The version alone, read before the full parse so an unsupported version is
/// reported as one rather than as whatever unknown field it introduced.
#[derive(Deserialize)]
struct ManifestVersion {
    version: u64,
}

/// The manifest as the file holds it, and as `GET /workers?format=manifest`
/// writes it (see [`export_manifest`]).
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Manifest {
    /// Checked by [`ManifestVersion`]; named here only so it is a known field.
    #[serde(rename = "version")]
    _version: u64,
    workers: Vec<ManifestWorker>,
    /// What an export could not carry (a worker's key, an address's
    /// credentials), one line per worker. Ignored on read.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    warnings: Vec<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestWorker {
    /// Defaults to the endpoint. Giving one lets a producer signal a new
    /// instance at an address it reuses.
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    worker_type: Option<WorkerType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bootstrap_port: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    kv_connector: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    kv_role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    kv_engine_id: Option<String>,
}

/// The registered fleet in the manifest's shape, for
/// `GET /workers?format=manifest`: what a second gateway reads with
/// `--discovery-provider file`. A data-parallel group is one record at the
/// address it was registered with. No `id` is written: the reader derives
/// one from the address, and an id that changed with every restart of the
/// exporting gateway would re-register the fleet on every reader. A
/// worker's key is no manifest field and an address's credentials never
/// leave: both are dropped, and each such address is named in `warnings`.
pub(crate) fn export_manifest(workers: &[WorkerInfo]) -> Manifest {
    // One record per data-parallel group: the lowest rank speaks for it.
    let mut lowest_rank: HashMap<&str, usize> = HashMap::new();
    for info in workers {
        if let (Some(base), Some(rank)) = (info.spec.dp_base_url.as_deref(), info.spec.dp_rank) {
            let lowest = lowest_rank.entry(base).or_insert(rank);
            *lowest = (*lowest).min(rank);
        }
    }

    let mut warnings = Vec::new();
    let mut records = Vec::with_capacity(workers.len());
    for info in workers {
        let spec = &info.spec;
        if let (Some(base), Some(rank)) = (spec.dp_base_url.as_deref(), spec.dp_rank) {
            if lowest_rank.get(base) != Some(&rank) {
                continue;
            }
        }
        let address = spec.dp_base_url.as_deref().unwrap_or(&spec.url);
        // No registered address fails to parse; one that did is not written
        // raw, as it may carry a credential.
        let Ok(endpoint) = Endpoint::parse(address) else {
            warnings.push(format!(
                "{}: not exported, the address does not parse",
                Endpoint::redacted_str(address)
            ));
            continue;
        };
        let url = endpoint.without_credentials().render();
        if endpoint.has_credentials() {
            warnings.push(format!("{url}: url credentials not exported"));
        }
        if spec.api_key.is_some() {
            warnings.push(format!("{url}: api_key not exported"));
        }
        records.push(ManifestWorker {
            id: None,
            url,
            worker_type: Some(spec.worker_type),
            bootstrap_port: spec
                .bootstrap_port
                .filter(|_| takes_bootstrap_port(spec.worker_type)),
            model_id: info.model_id.clone(),
            kv_connector: spec.kv_connector.clone(),
            kv_role: spec.kv_role.clone(),
            kv_engine_id: spec.kv_engine_id.clone(),
        });
    }
    Manifest {
        _version: MANIFEST_VERSION,
        workers: records,
        warnings,
    }
}

impl ManifestWorker {
    fn into_worker(self, roles: Roles) -> Result<DiscoveredWorker, String> {
        // `parse` rejects an `@rank` suffix: ranks are the workflow's to
        // assign. Its errors name the host or port, never the userinfo.
        let endpoint = Endpoint::parse(&self.url).map_err(|e| format!("invalid url: {e}"))?;

        let worker_type = match self.worker_type {
            Some(worker_type) if roles.allows(worker_type) => worker_type,
            Some(worker_type) => {
                return Err(format!(
                    "worker_type '{worker_type}' is not used in {} routing",
                    roles.name()
                ))
            }
            None => roles
                .default_type()
                .ok_or_else(|| format!("worker_type is required in {} routing", roles.name()))?,
        };

        // Port 0 is no port, as static configuration and Kubernetes
        // annotations already refuse it.
        if self.bootstrap_port == Some(0) {
            return Err("bootstrap_port must be between 1 and 65535".to_string());
        }
        if self.bootstrap_port.is_some() && !takes_bootstrap_port(worker_type) {
            return Err(format!(
                "bootstrap_port applies only to encode and prefill workers, not {worker_type}"
            ));
        }

        let discovery_id = match self.id {
            Some(id) if id.is_empty() => return Err("id must not be empty".to_string()),
            Some(id) => id,
            // Not `endpoint.key()`: the key keeps any password in the URL,
            // and the id is written to a label anyone can read.
            None => endpoint.redacted(),
        };

        Ok(DiscoveredWorker {
            discovery_id,
            endpoint,
            worker_type,
            bootstrap_port: self.bootstrap_port,
            model_id_override: self.model_id,
            kv_connector: self.kv_connector,
            kv_role: self.kv_role,
            kv_engine_id: self.kv_engine_id,
            compat_labels: BTreeMap::new(),
        })
    }
}

/// Whether a worker of this type opens a bootstrap port: the reader refuses
/// one on any other type, and the writer leaves it out.
fn takes_bootstrap_port(worker_type: WorkerType) -> bool {
    matches!(worker_type, WorkerType::Encode | WorkerType::Prefill)
}

/// Parse a whole manifest, or reject it whole: no partial list is applied.
fn parse_manifest(bytes: &[u8], roles: Roles) -> Result<Vec<DiscoveredWorker>, String> {
    let ManifestVersion { version } = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    if version != MANIFEST_VERSION {
        return Err(format!(
            "unsupported version {version}; this gateway reads version {MANIFEST_VERSION}"
        ));
    }
    let manifest: Manifest = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;

    let mut workers = Vec::with_capacity(manifest.workers.len());
    let mut ids = HashSet::new();
    let mut endpoints: HashMap<EndpointKey, String> = HashMap::new();
    for (index, record) in manifest.workers.into_iter().enumerate() {
        let worker = record
            .into_worker(roles)
            .map_err(|reason| format!("workers[{index}]: {reason}"))?;
        if !ids.insert(worker.discovery_id.clone()) {
            // The default id is the address with its userinfo masked, so two
            // id-less entries at one address that differ only in their
            // credentials (different keys) collide here on an id nobody
            // wrote: say so. The same entry twice is a plain duplicate.
            let defaulted = worker.discovery_id == worker.endpoint.redacted()
                && !endpoints.contains_key(&worker.endpoint.key());
            return Err(format!(
                "workers[{index}]: duplicate id '{}'{}",
                worker.discovery_id,
                if defaulted {
                    " (the default id, the address with its userinfo masked; set an explicit id)"
                } else {
                    ""
                }
            ));
        }
        if let Some(other) = endpoints.insert(worker.endpoint.key(), worker.discovery_id.clone()) {
            return Err(format!(
                "workers[{index}]: '{}' and '{other}' name the same endpoint {}",
                worker.discovery_id,
                worker.endpoint.redacted()
            ));
        }
        workers.push(worker);
    }
    Ok(workers)
}

/// Why a reread produced no snapshot.
#[derive(Debug, thiserror::Error)]
enum ManifestError {
    /// Absent (a writer may create it later) or unreadable.
    #[error("cannot read discovery manifest {path}: {source}")]
    Unreadable {
        path: String,
        #[source]
        source: std::io::Error,
    },
    /// Read, but not a valid manifest.
    #[error("invalid discovery manifest {path}: {reason}")]
    Invalid { path: String, reason: String },
}

impl SnapshotError for ManifestError {
    fn metric_reason(&self) -> &'static str {
        match self {
            Self::Unreadable { .. } => metrics_labels::SNAPSHOT_ERROR_UNREADABLE,
            Self::Invalid { .. } => metrics_labels::SNAPSHOT_ERROR_INVALID,
        }
    }
}

async fn read_manifest(config: &FileProviderConfig) -> Result<DesiredState, ManifestError> {
    let path = || config.path.display().to_string();
    let bytes =
        tokio::fs::read(&config.path)
            .await
            .map_err(|source| ManifestError::Unreadable {
                path: path(),
                source,
            })?;
    let workers =
        parse_manifest(&bytes, config.roles).map_err(|reason| ManifestError::Invalid {
            path: path(),
            reason,
        })?;
    Ok(DesiredState::from_workers(workers))
}

/// How often an unwatched manifest's directory is tried again. A try on a
/// directory that is still missing is one `stat`, and a directory that a
/// deploy tool recreates is back within about this long.
const REWATCH: Duration = Duration::from_secs(1);

/// Read the manifest now, whenever it changes, and on every tick, reconciling
/// the last good snapshot each time.
///
/// A change is read as soon as it settles. The tick stays, for two reasons: a
/// change notification can be lost, or never come at all (a manifest behind a
/// symlink into another directory, a network filesystem); and every reconcile
/// retries what an earlier one left undone, such as an engine that was not yet
/// listening when its record first appeared.
pub(super) fn start_file_discovery(
    config: FileProviderConfig,
    app_context: Arc<AppContext>,
) -> task::JoinHandle<()> {
    info!(
        "Starting file service discovery | manifest: {} | reread on change and every {}s",
        config.path.display(),
        config.check_interval.as_secs_f64()
    );

    #[expect(
        clippy::disallowed_methods,
        reason = "worker discovery runs for the lifetime of the server; shutdown aborts the handle"
    )]
    let handle = task::spawn(FileDiscovery::new(config, app_context).run());

    handle
}

/// The provider's loop and its state.
///
/// `watch` is `None` while the manifest's directory cannot be watched (it does
/// not exist yet, or was removed): the loop then rereads on the tick alone and
/// tries the watch again every [`REWATCH`]. `warned` records that the failure
/// was logged, so the retries stay quiet and the watch starting is logged in
/// turn. `last_good` is the snapshot every pass reconciles; a read that fails
/// leaves it in place.
struct FileDiscovery {
    config: FileProviderConfig,
    app_context: Arc<AppContext>,
    /// The watch sends on `wake` whenever the manifest may have changed; the
    /// loop takes the wakes from `woken`.
    wake: mpsc::Sender<()>,
    woken: mpsc::Receiver<()>,
    /// Held for as long as the loop runs: dropping it stops the watch.
    watch: Option<DirectoryWatch>,
    warned: bool,
    last_good: LastGood,
}

/// The loop's timers, held across passes: the tick, and the retry of a watch
/// that could not be started. A timer rebuilt on every pass is set back by
/// every pass another cause starts: the retry, once a sleep rebuilt that way,
/// never fired under a check interval no longer than [`REWATCH`], because the
/// tick came first on every pass.
struct Timers {
    tick: time::Interval,
    rewatch: time::Interval,
}

impl Timers {
    fn new(check_interval: Duration) -> Self {
        let mut tick = time::interval(check_interval);
        tick.set_missed_tick_behavior(time::MissedTickBehavior::Delay);
        let mut rewatch = time::interval(REWATCH);
        rewatch.set_missed_tick_behavior(time::MissedTickBehavior::Delay);
        Self { tick, rewatch }
    }
}

impl FileDiscovery {
    fn new(config: FileProviderConfig, app_context: Arc<AppContext>) -> Self {
        let (wake, woken) = mpsc::channel(1);
        let last_good = LastGood::new(
            DiscoveryKind::File,
            format!("Discovery manifest {}", config.path.display()),
            "manifest",
        );
        Self {
            config,
            app_context,
            wake,
            woken,
            watch: None,
            warned: false,
            last_good,
        }
    }

    async fn run(mut self) {
        let mut timers = Timers::new(self.config.check_interval);
        self.start_watch(&mut timers);
        loop {
            self.pass(&mut timers).await;
        }
    }

    /// One pass: wait for its cause, then reread and reconcile.
    async fn pass(&mut self, timers: &mut Timers) {
        if !self.next_pass(timers).await {
            return;
        }
        self.rewatch_if_lost(timers);
        let started_at = time::Instant::now();
        self.reread().await;
        self.last_good
            .reconcile(&self.app_context, started_at)
            .await;
    }

    /// Wait for what causes the next pass: the tick, a burst of change events
    /// once it has settled, or the watch starting after the directory came
    /// back, since nothing reported changes until then. `false` when the
    /// directory is still missing, which is no pass at all.
    async fn next_pass(&mut self, timers: &mut Timers) -> bool {
        tokio::select! {
            _ = timers.tick.tick() => true,
            Some(()) = self.woken.recv() => {
                settle(&mut self.woken).await;
                true
            }
            // Unwatched, because the directory does not exist (yet, or any
            // more): retry on the watch's own cadence rather than on the tick.
            _ = timers.rewatch.tick(), if self.watch.is_none() => {
                self.start_watch(timers);
                self.watch.is_some()
            }
        }
    }

    /// Start the watch again once the directory it was on is gone.
    fn rewatch_if_lost(&mut self, timers: &mut Timers) {
        if self.watch.as_ref().is_some_and(DirectoryWatch::is_lost) {
            warn!(
                "The directory of discovery manifest {} was removed or moved; watching it again once it exists",
                self.config.path.display()
            );
            // Logged here, so a failed retry stays quiet.
            self.warned = true;
            self.start_watch(timers);
        }
    }

    /// Start watching, or say why the provider is polling only and have the
    /// watch tried again in [`REWATCH`]. Polling alone is still correct: the
    /// tick rereads whatever the watch would have reported.
    fn start_watch(&mut self, timers: &mut Timers) {
        self.watch = match DirectoryWatch::of_file(&self.config.path, self.wake.clone()) {
            Ok(watch) => {
                if self.warned {
                    info!(
                        "Watching discovery manifest {} for changes",
                        self.config.path.display()
                    );
                    self.warned = false;
                }
                Some(watch)
            }
            Err(error) => {
                if !self.warned {
                    warn!(
                        "Cannot watch discovery manifest {} ({error}); rereading every {}s, and retrying the watch every {}s",
                        self.config.path.display(),
                        self.config.check_interval.as_secs_f64(),
                        REWATCH.as_secs_f64()
                    );
                    self.warned = true;
                }
                // A full `REWATCH` from now, wherever the retry's schedule
                // stood while the watch was up.
                timers.rewatch.reset();
                None
            }
        };
    }

    /// Read the manifest into the last good snapshot. A read that fails is
    /// reported and leaves the snapshot in place, so a half-written or broken
    /// file never removes a worker.
    async fn reread(&mut self) {
        self.last_good.update(read_manifest(&self.config).await);
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::service_discovery::testing::create_test_app_context;

    fn parse(json: &str, roles: Roles) -> Result<Vec<DiscoveredWorker>, String> {
        parse_manifest(json.as_bytes(), roles)
    }

    fn parse_err(json: &str, roles: Roles) -> String {
        match parse(json, roles) {
            Ok(workers) => panic!("expected a rejection, parsed {workers:?}"),
            Err(reason) => reason,
        }
    }

    fn manifest(workers: &str) -> String {
        format!(r#"{{"version": 1, "workers": [{workers}]}}"#)
    }

    #[test]
    fn every_field_reaches_the_record() {
        let workers = parse(
            &manifest(
                r#"{"id": "prefill-a", "url": "10.0.0.22:8000", "worker_type": "prefill",
                    "bootstrap_port": 8998, "model_id": "llama", "kv_connector": "MooncakeConnector",
                    "kv_role": "kv_producer", "kv_engine_id": "engine-a"}"#,
            ),
            Roles::PrefillDecode,
        )
        .unwrap();
        assert_eq!(workers.len(), 1);
        let worker = &workers[0];
        assert_eq!(worker.discovery_id, "prefill-a");
        assert_eq!(worker.endpoint.render(), "10.0.0.22:8000");
        assert_eq!(worker.worker_type, WorkerType::Prefill);
        assert_eq!(worker.bootstrap_port, Some(8998));
        assert_eq!(worker.model_id_override.as_deref(), Some("llama"));
        assert_eq!(worker.kv_connector.as_deref(), Some("MooncakeConnector"));
        assert_eq!(worker.kv_role.as_deref(), Some("kv_producer"));
        assert_eq!(worker.kv_engine_id.as_deref(), Some("engine-a"));
        assert!(worker.compat_labels.is_empty());
    }

    /// Every EPD role in one manifest, over IPv4, IPv6, gRPC and IPC.
    #[test]
    fn all_worker_types_and_endpoint_forms_parse() {
        let workers = parse(
            &manifest(
                r#"{"url": "10.0.0.21:8000", "worker_type": "encode", "bootstrap_port": 8998},
                   {"url": "[2001:db8::1]:8000", "worker_type": "prefill"},
                   {"url": "grpc://decode.internal:9000", "worker_type": "decode"},
                   {"url": "ipc:///tmp/engine.sock", "worker_type": "decode"}"#,
            ),
            Roles::EncodePrefillDecode,
        )
        .unwrap();
        let types: Vec<WorkerType> = workers.iter().map(|w| w.worker_type).collect();
        assert_eq!(
            types,
            [
                WorkerType::Encode,
                WorkerType::Prefill,
                WorkerType::Decode,
                WorkerType::Decode
            ]
        );
    }

    #[test]
    fn an_empty_fleet_is_valid() {
        assert!(parse(&manifest(""), Roles::Regular).unwrap().is_empty());
    }

    /// Without an id the endpoint names the instance, rendered with any
    /// userinfo masked: the id is written to a label. Two entries at one
    /// address that differ only in their credentials need explicit ids.
    #[test]
    fn a_missing_id_defaults_to_the_redacted_endpoint() {
        let workers = parse(
            &manifest(r#"{"url": "http://admin:hunter2@10.0.0.11:8000"}"#),
            Roles::Regular,
        )
        .unwrap();
        assert_eq!(workers[0].discovery_id, "http://***@10.0.0.11:8000");
    }

    #[test]
    fn regular_routing_defaults_the_worker_type() {
        let workers = parse(&manifest(r#"{"url": "10.0.0.11:8000"}"#), Roles::Regular).unwrap();
        assert_eq!(workers[0].worker_type, WorkerType::Regular);
    }

    /// Each malformed manifest is rejected whole, with a reason that names
    /// the problem.
    #[test]
    fn malformed_manifests_are_rejected_with_the_reason() {
        let cases: [(&str, String, Roles, &str); 18] = [
            ("bad JSON", "{".to_string(), Roles::Regular, "EOF"),
            (
                "missing version",
                r#"{"workers": []}"#.to_string(),
                Roles::Regular,
                "missing field `version`",
            ),
            (
                "unsupported version",
                r#"{"version": 2, "workers": []}"#.to_string(),
                Roles::Regular,
                "unsupported version 2",
            ),
            (
                "unknown top-level field",
                r#"{"version": 1, "workers": [], "worker": []}"#.to_string(),
                Roles::Regular,
                "unknown field `worker`",
            ),
            (
                "misspelled worker field",
                manifest(r#"{"url": "10.0.0.11:8000", "worker_typ": "regular"}"#),
                Roles::Regular,
                "unknown field `worker_typ`",
            ),
            (
                "unknown worker type",
                manifest(r#"{"url": "10.0.0.11:8000", "worker_type": "router"}"#),
                Roles::Regular,
                "unknown variant `router`",
            ),
            (
                "type the routing mode does not use",
                manifest(r#"{"url": "10.0.0.11:8000", "worker_type": "prefill"}"#),
                Roles::Regular,
                "'prefill' is not used in regular routing",
            ),
            (
                "encode in prefill-decode routing",
                manifest(r#"{"url": "10.0.0.11:8000", "worker_type": "encode"}"#),
                Roles::PrefillDecode,
                "'encode' is not used in prefill-decode routing",
            ),
            (
                "no type in disaggregated routing",
                manifest(r#"{"url": "10.0.0.11:8000"}"#),
                Roles::PrefillDecode,
                "worker_type is required in prefill-decode routing",
            ),
            (
                "bootstrap port 0",
                manifest(
                    r#"{"url": "10.0.0.11:8000", "worker_type": "prefill", "bootstrap_port": 0}"#,
                ),
                Roles::PrefillDecode,
                "bootstrap_port must be between 1 and 65535",
            ),
            (
                "bootstrap port on a decode worker",
                manifest(
                    r#"{"url": "10.0.0.11:8000", "worker_type": "decode", "bootstrap_port": 8998}"#,
                ),
                Roles::PrefillDecode,
                "bootstrap_port applies only to encode and prefill workers, not decode",
            ),
            (
                "rank suffix",
                manifest(r#"{"url": "10.0.0.11:8000@1"}"#),
                Roles::Regular,
                "DP rank suffix",
            ),
            (
                "unparsable url",
                manifest(r#"{"url": "10.0.0.11:http"}"#),
                Roles::Regular,
                "workers[0]: invalid url",
            ),
            (
                "empty id",
                manifest(r#"{"id": "", "url": "10.0.0.11:8000"}"#),
                Roles::Regular,
                "id must not be empty",
            ),
            (
                "duplicate id",
                manifest(
                    r#"{"id": "a", "url": "10.0.0.11:8000"}, {"id": "a", "url": "10.0.0.12:8000"}"#,
                ),
                Roles::Regular,
                "workers[1]: duplicate id 'a'",
            ),
            (
                "the same id-less entry twice",
                manifest(r#"{"url": "10.0.0.11:8000"}, {"url": "10.0.0.11:8000"}"#),
                Roles::Regular,
                "workers[1]: duplicate id '10.0.0.11:8000'",
            ),
            (
                "two id-less entries at one address that differ only in their credentials",
                manifest(
                    r#"{"url": "http://alice:x@10.0.0.11:8000"}, {"url": "http://bob:y@10.0.0.11:8000"}"#,
                ),
                Roles::Regular,
                "workers[1]: duplicate id 'http://***@10.0.0.11:8000' (the default id, the address \
                 with its userinfo masked; set an explicit id)",
            ),
            (
                "two spellings of one endpoint",
                manifest(
                    r#"{"id": "a", "url": "10.0.0.11:8000"}, {"id": "b", "url": "http://10.0.0.11:8000"}"#,
                ),
                Roles::Regular,
                "'b' and 'a' name the same endpoint",
            ),
        ];
        for (case, json, roles, expected) in cases {
            let reason = parse_err(&json, roles);
            assert!(
                reason.contains(expected),
                "{case}: expected '{expected}' in '{reason}'"
            );
        }
        // Exact: the plain duplicate carries no default-id hint (the table's
        // `contains` would accept the hint's longer message as well).
        assert_eq!(
            parse_err(
                &manifest(r#"{"url": "10.0.0.11:8000"}, {"url": "10.0.0.11:8000"}"#),
                Roles::Regular,
            ),
            "workers[1]: duplicate id '10.0.0.11:8000'",
        );
    }

    #[test]
    fn modes_without_worker_pools_cannot_use_file_discovery() {
        let mode = RoutingMode::OpenAI {
            worker_urls: vec![],
        };
        assert!(
            FileProviderConfig::new(PathBuf::from("w.json"), Duration::from_secs(1), &mode)
                .is_err()
        );
    }

    #[tokio::test]
    async fn an_absent_manifest_is_unreadable_not_empty() {
        let dir = tempfile::tempdir().unwrap();
        let config = FileProviderConfig::new(
            dir.path().join("workers.json"),
            Duration::from_secs(1),
            &RoutingMode::Regular {
                worker_urls: vec![],
            },
        )
        .unwrap();
        let err = read_manifest(&config).await.unwrap_err();
        assert!(matches!(err, ManifestError::Unreadable { .. }), "{err}");
        assert_eq!(
            err.metric_reason(),
            metrics_labels::SNAPSHOT_ERROR_UNREADABLE
        );
    }

    fn discovery(path: &Path) -> FileDiscovery {
        discovery_checking_every(path, Duration::from_secs(60))
    }

    fn discovery_checking_every(path: &Path, check_interval: Duration) -> FileDiscovery {
        let config = FileProviderConfig::new(
            path.to_path_buf(),
            check_interval,
            &RoutingMode::Regular {
                worker_urls: vec![],
            },
        )
        .unwrap();
        FileDiscovery::new(config, create_test_app_context())
    }

    /// The endpoints of the last good snapshot, `None` before the first good
    /// read.
    fn listed(discovery: &FileDiscovery) -> Option<Vec<String>> {
        discovery.last_good.desired().map(|desired| {
            desired
                .workers
                .values()
                .map(|worker| worker.endpoint.render())
                .collect()
        })
    }

    /// A read that fails leaves the last good snapshot in place, so a
    /// half-written, broken or missing manifest never removes a worker; a
    /// valid empty fleet does replace it.
    #[tokio::test]
    async fn a_failed_read_keeps_the_last_good_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("workers.json");
        let mut discovery = discovery(&path);

        discovery.reread().await;
        assert_eq!(listed(&discovery), None, "nothing read yet");

        std::fs::write(&path, manifest(r#"{"url": "10.0.0.11:8000"}"#)).unwrap();
        discovery.reread().await;
        assert_eq!(listed(&discovery), Some(vec!["10.0.0.11:8000".to_string()]));

        std::fs::write(&path, "{").unwrap();
        discovery.reread().await;
        assert_eq!(
            listed(&discovery),
            Some(vec!["10.0.0.11:8000".to_string()]),
            "a half-written manifest replaced the snapshot"
        );

        std::fs::remove_file(&path).unwrap();
        discovery.reread().await;
        assert_eq!(
            listed(&discovery),
            Some(vec!["10.0.0.11:8000".to_string()]),
            "a missing manifest replaced the snapshot"
        );

        std::fs::write(&path, manifest("")).unwrap();
        discovery.reread().await;
        assert_eq!(listed(&discovery), Some(vec![]), "an empty fleet is valid");
    }

    /// Until the manifest's directory exists the provider polls only: the
    /// watch is retried after the one warning, and starts once the directory
    /// is there.
    #[tokio::test]
    async fn an_unwatchable_directory_leaves_the_provider_polling() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("smg");
        let mut discovery = discovery(&dir.join("workers.json"));
        let mut timers = Timers::new(discovery.config.check_interval);

        discovery.start_watch(&mut timers);
        assert!(discovery.watch.is_none());
        assert!(discovery.warned, "the failure is logged");
        discovery.start_watch(&mut timers);
        assert!(discovery.watch.is_none());
        assert!(discovery.warned, "a failed retry stays quiet");

        std::fs::create_dir(&dir).unwrap();
        discovery.start_watch(&mut timers);
        assert!(discovery.watch.is_some(), "the directory exists now");
        assert!(!discovery.warned, "the watch starting is logged in turn");
    }

    /// A watch lost with its directory is tried again on its own cadence,
    /// under a check interval no longer than that cadence too: the loop, as
    /// it runs, sees the directory removed (the watch reports its own loss)
    /// and recreated, and has the watch back within a few passes. A retry
    /// rebuilt on every pass lost to the tick on every pass and never fired.
    #[tokio::test]
    async fn a_lost_watch_is_retried_under_a_check_interval_as_short_as_the_retry() {
        for check_interval in [Duration::from_millis(250), REWATCH] {
            let root = tempfile::tempdir().unwrap();
            let dir = root.path().join("smg");
            std::fs::create_dir(&dir).unwrap();
            let path = dir.join("workers.json");
            std::fs::write(&path, manifest("")).unwrap();
            let mut discovery = discovery_checking_every(&path, check_interval);
            let mut timers = Timers::new(check_interval);
            discovery.start_watch(&mut timers);
            assert!(discovery.watch.is_some());

            std::fs::remove_file(&path).unwrap();
            std::fs::remove_dir(&dir).unwrap();
            let deadline = time::Instant::now() + Duration::from_secs(10);
            while discovery.watch.is_some() {
                assert!(time::Instant::now() < deadline, "the loss was never seen");
                discovery.pass(&mut timers).await;
            }
            assert!(discovery.warned, "polling only, and said so");

            std::fs::create_dir(&dir).unwrap();
            std::fs::write(&path, manifest("")).unwrap();
            let deadline = time::Instant::now() + Duration::from_secs(5);
            while discovery.watch.is_none() && time::Instant::now() < deadline {
                discovery.pass(&mut timers).await;
            }
            assert!(
                discovery.watch.is_some(),
                "the watch was not tried again under a check interval of {check_interval:?}"
            );
            assert!(!discovery.warned, "the watch starting is logged in turn");
        }
    }

    /// The export of a fleet, fed back through the parser, yields the same
    /// endpoints, types and settings; a key and an address's credentials
    /// stay behind and are named in the warnings.
    #[test]
    fn an_exported_fleet_parses_back_without_its_credentials() {
        use openai_protocol::model_card::ModelCard;

        use crate::worker::{basic::worker_to_info, BasicWorkerBuilder, Worker};

        fn info(worker: BasicWorkerBuilder) -> WorkerInfo {
            let worker: Arc<dyn Worker> = Arc::new(worker.build());
            worker_to_info(&worker)
        }
        let fleet = [
            info(
                BasicWorkerBuilder::new("http://10.0.0.22:8000")
                    .worker_type(WorkerType::Prefill)
                    .bootstrap_port(Some(8998))
                    .kv_connector("MooncakeConnector")
                    .kv_role("kv_producer")
                    .kv_engine_id("engine-a")
                    .model(ModelCard::new("llama")),
            ),
            info(
                BasicWorkerBuilder::new("grpc://decode.internal:9000")
                    .worker_type(WorkerType::Decode)
                    .model(ModelCard::new("llama")),
            ),
            info(
                BasicWorkerBuilder::new("http://admin:hunter2@10.0.0.23:8001")
                    .worker_type(WorkerType::Decode)
                    .api_key("s3cr3t"),
            ),
        ];

        let exported = export_manifest(&fleet);
        let text = serde_json::to_string(&exported).unwrap();

        for secret in ["hunter2", "s3cr3t", r#""api_key""#, "admin", r#""id""#] {
            assert!(
                !text.contains(secret),
                "{secret} reached the export: {text}"
            );
        }
        assert_eq!(
            exported.warnings,
            [
                "http://10.0.0.23:8001: url credentials not exported",
                "http://10.0.0.23:8001: api_key not exported"
            ]
        );

        let parsed = parse(&text, Roles::PrefillDecode).unwrap();
        let summary: Vec<_> = parsed
            .iter()
            .map(|w| {
                (
                    w.endpoint.render(),
                    w.worker_type,
                    w.bootstrap_port,
                    w.model_id_override.as_deref(),
                    w.kv_connector.as_deref(),
                )
            })
            .collect();
        assert_eq!(
            summary,
            [
                (
                    "http://10.0.0.22:8000".to_string(),
                    WorkerType::Prefill,
                    Some(8998),
                    Some("llama"),
                    Some("MooncakeConnector"),
                ),
                (
                    "grpc://decode.internal:9000".to_string(),
                    WorkerType::Decode,
                    None,
                    Some("llama"),
                    None,
                ),
                (
                    "http://10.0.0.23:8001".to_string(),
                    WorkerType::Decode,
                    None,
                    None,
                    None,
                ),
            ]
        );
        assert_eq!(parsed[0].kv_role.as_deref(), Some("kv_producer"));
        assert_eq!(parsed[0].kv_engine_id.as_deref(), Some("engine-a"));
    }

    /// A data-parallel group was registered at one address; its export is
    /// that one record, not one per rank.
    #[test]
    fn a_data_parallel_group_exports_as_one_record_at_its_base_address() {
        use crate::worker::{basic::worker_to_info, BasicWorkerBuilder, Worker};

        let fleet: Vec<WorkerInfo> = (0..2)
            .map(|rank| {
                let worker: Arc<dyn Worker> = Arc::new(
                    BasicWorkerBuilder::new("http://10.0.0.30:8000")
                        .worker_type(WorkerType::Regular)
                        .dp_config(rank, 2)
                        .build(),
                );
                worker_to_info(&worker)
            })
            .collect();

        let exported = export_manifest(&fleet);
        let text = serde_json::to_string(&exported).unwrap();

        assert_eq!(exported.workers.len(), 1, "{text}");
        assert_eq!(exported.workers[0].url, "http://10.0.0.30:8000");
        assert!(exported.workers[0].id.is_none(), "{text}");
        assert!(exported.warnings.is_empty(), "{text}");
        assert_eq!(parse(&text, Roles::Regular).unwrap().len(), 1);
    }

    /// An export with warnings is still a manifest: the field is known and
    /// ignored on read, so the fleet registers minus what was not carried.
    #[test]
    fn warnings_in_a_manifest_are_read_past() {
        let workers = parse(
            r#"{"version": 1, "workers": [{"url": "10.0.0.5:8000"}], "warnings": ["worker a: api_key not exported"]}"#,
            Roles::Regular,
        )
        .unwrap();
        assert_eq!(workers.len(), 1);
    }
}
