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
    ffi::OsStr,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use notify::{event::ModifyKind, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use openai_protocol::worker::WorkerType;
use serde::Deserialize;
use tokio::{sync::mpsc, task, time};
use tracing::{info, warn};

use super::{
    provider::{DiscoveredWorker, DiscoveryKind},
    reconciler::{self, DesiredState},
};
use crate::{
    app_context::AppContext,
    config::{ConfigError, ConfigResult, RoutingMode},
    observability::metrics::{metrics_labels, Metrics},
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

/// The worker types a routing mode accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Roles {
    Regular,
    PrefillDecode,
    EncodePrefillDecode,
}

impl Roles {
    fn of(mode: &RoutingMode) -> Option<Self> {
        match mode {
            RoutingMode::Regular { .. } => Some(Self::Regular),
            RoutingMode::PrefillDecode { .. } => Some(Self::PrefillDecode),
            RoutingMode::EncodePrefillDecode { .. } => Some(Self::EncodePrefillDecode),
            RoutingMode::OpenAI { .. }
            | RoutingMode::Anthropic { .. }
            | RoutingMode::Gemini { .. } => None,
        }
    }

    fn allows(self, worker_type: WorkerType) -> bool {
        matches!(
            (self, worker_type),
            (Self::Regular, WorkerType::Regular)
                | (
                    Self::PrefillDecode,
                    WorkerType::Prefill | WorkerType::Decode
                )
                | (
                    Self::EncodePrefillDecode,
                    WorkerType::Encode | WorkerType::Prefill | WorkerType::Decode
                )
        )
    }

    /// The type of a record that names none. Regular routing has only one; a
    /// disaggregated record must say which pool it joins.
    fn default_type(self) -> Option<WorkerType> {
        (self == Self::Regular).then_some(WorkerType::Regular)
    }

    fn name(self) -> &'static str {
        match self {
            Self::Regular => "regular",
            Self::PrefillDecode => "prefill-decode",
            Self::EncodePrefillDecode => "encode-prefill-decode",
        }
    }
}

/// The version alone, read before the full parse so an unsupported version is
/// reported as one rather than as whatever unknown field it introduced.
#[derive(Deserialize)]
struct ManifestVersion {
    version: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    /// Checked by [`ManifestVersion`]; named here only so it is a known field.
    #[serde(rename = "version")]
    _version: u64,
    workers: Vec<ManifestWorker>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestWorker {
    /// Defaults to the endpoint. Giving one lets a producer signal a new
    /// instance at an address it reuses.
    id: Option<String>,
    url: String,
    worker_type: Option<WorkerType>,
    bootstrap_port: Option<u16>,
    model_id: Option<String>,
    kv_connector: Option<String>,
    kv_role: Option<String>,
    kv_engine_id: Option<String>,
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

        if self.bootstrap_port.is_some()
            && !matches!(worker_type, WorkerType::Encode | WorkerType::Prefill)
        {
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
            return Err(format!(
                "workers[{index}]: duplicate id '{}'",
                worker.discovery_id
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

impl ManifestError {
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

/// How long a burst of filesystem events is left to settle before the one
/// reread it causes. A writer's rename, or an editor's several writes, arrive
/// as a handful of events within a few milliseconds.
const SETTLE: Duration = Duration::from_millis(200);

/// How often an unwatched manifest's directory is tried again. A try on a
/// directory that is still missing is one `stat`, and a directory that a
/// deploy tool recreates is back within about this long.
const REWATCH: Duration = Duration::from_secs(1);

/// A watch on the manifest's directory. Dropping it stops the watch.
struct ManifestWatch {
    _watcher: RecommendedWatcher,
    /// Set once the watched directory itself is removed or moved away. The
    /// watch then reports nothing more, not even for a directory recreated at
    /// the same path, so it has to be started again.
    lost: Arc<AtomicBool>,
}

impl ManifestWatch {
    fn is_lost(&self) -> bool {
        self.lost.load(Ordering::Relaxed)
    }
}

/// Watch the manifest's directory and send on `wake` whenever the manifest may
/// have changed, or the watch stopped working.
///
/// The directory rather than the file: a writer that renames a new file over
/// the manifest replaces its inode, and a watch on the old inode never
/// reports it. A send that finds the channel full is dropped, because a wake
/// is already pending and the reread it causes reads the latest file.
fn watch_manifest(path: &Path, wake: mpsc::Sender<()>) -> notify::Result<ManifestWatch> {
    let dir = match path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir,
        _ => Path::new("."),
    };
    // Watched by its absolute path, which is how inotify names the directory
    // in an event about the directory itself. FSEvents names it resolved.
    let dir = std::path::absolute(dir).map_err(notify::Error::io)?;
    // Checked before building the watcher, which on Linux starts an inotify
    // instance and a thread even when the watch then fails.
    match std::fs::metadata(&dir) {
        Ok(metadata) if metadata.is_dir() => {}
        Ok(_) => return Err(notify::Error::generic("not a directory").add_path(dir)),
        Err(error) => return Err(notify::Error::io(error).add_path(dir)),
    }
    let dir_names: Vec<PathBuf> = [Some(dir.clone()), dir.canonicalize().ok()]
        .into_iter()
        .flatten()
        .collect();
    let name = path.file_name().map(OsStr::to_os_string);
    let lost = Arc::new(AtomicBool::new(false));
    let lost_by_event = Arc::clone(&lost);
    let mut watcher = notify::recommended_watcher(move |event: notify::Result<Event>| {
        let wakes = match event {
            Ok(event) if removes_directory(&event, &dir_names) => {
                lost_by_event.store(true, Ordering::Relaxed);
                true
            }
            Ok(event) => concerns_manifest(&event, name.as_deref()),
            // A failed watch may have dropped a change, so it rereads too.
            Err(_) => true,
        };
        if wakes {
            let _ = wake.try_send(());
        }
    })?;
    watcher.watch(&dir, RecursiveMode::NonRecursive)?;
    Ok(ManifestWatch {
        _watcher: watcher,
        lost,
    })
}

/// Whether an event says the watched directory itself was removed or moved
/// away, given the names an event may call it by.
fn removes_directory(event: &Event, dir_names: &[PathBuf]) -> bool {
    matches!(
        event.kind,
        EventKind::Remove(_) | EventKind::Modify(ModifyKind::Name(_))
    ) && event.paths.iter().any(|path| dir_names.contains(path))
}

/// Whether an event in the manifest's directory may have changed the manifest.
fn concerns_manifest(event: &Event, name: Option<&OsStr>) -> bool {
    if event.need_rescan() {
        return true;
    }
    // Reads change nothing, and the provider's own reread is one of them.
    if matches!(event.kind, EventKind::Access(_)) {
        return false;
    }
    event.paths.iter().any(|path| {
        path.file_name().is_some_and(|file| {
            // Kubernetes updates a mounted ConfigMap or Secret by swapping a
            // `..data` symlink beside the file, never touching the file's name.
            Some(file) == name || file.as_encoded_bytes().starts_with(b"..data")
        })
    })
}

/// Let a burst of events settle, then take every wake it sent as one.
async fn settle(wake: &mut mpsc::Receiver<()>) {
    time::sleep(SETTLE).await;
    while wake.try_recv().is_ok() {}
}

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
    watch: Option<ManifestWatch>,
    warned: bool,
    last_good: Option<DesiredState>,
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
        Self {
            config,
            app_context,
            wake,
            woken,
            watch: None,
            warned: false,
            last_good: None,
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
        if let Some(desired) = &self.last_good {
            reconciler::reconcile(desired, DiscoveryKind::File, &self.app_context, started_at)
                .await;
        }
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
        if self.watch.as_ref().is_some_and(ManifestWatch::is_lost) {
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
        self.watch = match watch_manifest(&self.config.path, self.wake.clone()) {
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
        match read_manifest(&self.config).await {
            Ok(desired) => {
                if self.last_good.as_ref() != Some(&desired) {
                    info!(
                        "Discovery manifest {} lists {} worker(s)",
                        self.config.path.display(),
                        desired.workers.len()
                    );
                }
                self.last_good = Some(desired);
            }
            Err(error) => {
                Metrics::record_discovery_snapshot_error(
                    DiscoveryKind::File.metric_label(),
                    error.metric_reason(),
                );
                if self.last_good.is_some() {
                    warn!("{error}; keeping the last good manifest");
                } else {
                    warn!("{error}; no workers discovered yet");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
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
    /// password masked: the id is written to a label.
    #[test]
    fn a_missing_id_defaults_to_the_redacted_endpoint() {
        let workers = parse(
            &manifest(r#"{"url": "http://admin:hunter2@10.0.0.11:8000"}"#),
            Roles::Regular,
        )
        .unwrap();
        assert_eq!(workers[0].discovery_id, "http://admin:***@10.0.0.11:8000");
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
        let cases: [(&str, String, Roles, &str); 15] = [
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

    fn event(kind: EventKind, paths: &[&str]) -> Event {
        paths
            .iter()
            .fold(Event::new(kind), |event, path| event.add_path(path.into()))
    }

    /// Which events in the manifest's directory cause a reread.
    #[test]
    fn only_a_change_to_the_manifest_wakes_the_provider() {
        use notify::event::{
            AccessKind, CreateKind, DataChange, Flag, ModifyKind, RemoveKind, RenameMode,
        };

        let name = Some(OsStr::new("workers.json"));
        let wakes = [
            // A writer renaming its temporary file over the manifest.
            event(
                EventKind::Modify(ModifyKind::Name(RenameMode::Both)),
                &["/run/smg/workers.json.tmp", "/run/smg/workers.json"],
            ),
            event(
                EventKind::Modify(ModifyKind::Name(RenameMode::To)),
                &["/run/smg/workers.json"],
            ),
            // Written in place, created late, or deleted.
            event(
                EventKind::Modify(ModifyKind::Data(DataChange::Content)),
                &["/run/smg/workers.json"],
            ),
            event(
                EventKind::Create(CreateKind::File),
                &["/run/smg/workers.json"],
            ),
            event(
                EventKind::Remove(RemoveKind::File),
                &["/run/smg/workers.json"],
            ),
            // Kubernetes swapping a ConfigMap's `..data` symlink.
            event(
                EventKind::Modify(ModifyKind::Name(RenameMode::To)),
                &["/etc/smg/..data"],
            ),
            event(EventKind::Create(CreateKind::Any), &["/etc/smg/..data_tmp"]),
            // The watch dropped events; anything may have changed.
            event(EventKind::Other, &[]).set_flag(Flag::Rescan),
        ];
        for event in &wakes {
            assert!(concerns_manifest(event, name), "{event:?} should wake");
        }

        let ignored = [
            // The provider's own read.
            event(
                EventKind::Access(AccessKind::Read),
                &["/run/smg/workers.json"],
            ),
            event(
                EventKind::Access(AccessKind::Any),
                &["/run/smg/workers.json"],
            ),
            // Another file in the same directory, including the writer's
            // temporary file before its rename.
            event(
                EventKind::Modify(ModifyKind::Data(DataChange::Any)),
                &["/run/smg/workers.json.tmp"],
            ),
            event(
                EventKind::Create(CreateKind::File),
                &["/run/smg/other.json"],
            ),
        ];
        for event in &ignored {
            assert!(!concerns_manifest(event, name), "{event:?} should not wake");
        }
    }

    async fn woken(wake: &mut mpsc::Receiver<()>, within: Duration) -> bool {
        time::timeout(within, wake.recv()).await.is_ok()
    }

    /// The real watcher: a rename over the manifest wakes the provider, a burst
    /// of writes settles into one wake, and another file in the directory does
    /// not wake it.
    #[tokio::test]
    async fn the_watcher_reports_the_manifest_and_only_the_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("workers.json");
        let (tx, mut rx) = mpsc::channel(1);
        let _watcher = watch_manifest(&path, tx).unwrap();

        let staged = dir.path().join("workers.json.staged");
        std::fs::write(&staged, manifest("")).unwrap();
        std::fs::rename(&staged, &path).unwrap();
        assert!(
            woken(&mut rx, Duration::from_secs(10)).await,
            "rename not seen"
        );
        settle(&mut rx).await;

        for _ in 0..20 {
            std::fs::write(&path, manifest("")).unwrap();
        }
        assert!(
            woken(&mut rx, Duration::from_secs(10)).await,
            "writes not seen"
        );
        settle(&mut rx).await;
        assert!(rx.try_recv().is_err(), "the burst woke the provider twice");

        // Let any late event from the burst arrive and be drained first.
        time::sleep(Duration::from_secs(1)).await;
        while rx.try_recv().is_ok() {}
        std::fs::write(dir.path().join("other.json"), "{}").unwrap();
        assert!(
            !woken(&mut rx, Duration::from_secs(1)).await,
            "another file woke the provider"
        );
    }

    /// Only an event about the watched directory itself, removed or moved
    /// away, marks the watch lost; events about files inside it do not.
    #[test]
    fn only_the_directory_going_away_loses_the_watch() {
        use notify::event::{CreateKind, RemoveKind, RenameMode};

        let dir = [PathBuf::from("/run/smg")];
        for event in [
            // inotify's DELETE_SELF and MOVE_SELF.
            event(EventKind::Remove(RemoveKind::Folder), &["/run/smg"]),
            event(
                EventKind::Modify(ModifyKind::Name(RenameMode::From)),
                &["/run/smg"],
            ),
        ] {
            assert!(removes_directory(&event, &dir), "{event:?}");
        }
        for event in [
            event(
                EventKind::Remove(RemoveKind::File),
                &["/run/smg/workers.json"],
            ),
            event(
                EventKind::Modify(ModifyKind::Name(RenameMode::Both)),
                &["/run/smg/workers.json.tmp", "/run/smg/workers.json"],
            ),
            event(EventKind::Create(CreateKind::Folder), &["/run/smg"]),
        ] {
            assert!(!removes_directory(&event, &dir), "{event:?}");
        }
    }

    /// Removing the watched directory marks the watch lost and wakes the
    /// provider, which then watches the path again.
    #[tokio::test]
    async fn removing_the_directory_loses_the_watch() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("smg");
        std::fs::create_dir(&dir).unwrap();
        let (tx, mut rx) = mpsc::channel(1);
        let watch = watch_manifest(&dir.join("workers.json"), tx).unwrap();
        assert!(!watch.is_lost());

        std::fs::remove_dir(&dir).unwrap();
        assert!(
            woken(&mut rx, Duration::from_secs(10)).await,
            "removal not seen"
        );
        let deadline = time::Instant::now() + Duration::from_secs(10);
        while !watch.is_lost() {
            assert!(time::Instant::now() < deadline, "the watch never lost");
            time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// The directory must exist to be watched; until it does, the provider
    /// polls, and retries the watch every second. The `stat` that refuses it
    /// names the cause and the path.
    #[test]
    fn a_missing_directory_cannot_be_watched() {
        let dir = tempfile::tempdir().unwrap();
        let (tx, _rx) = mpsc::channel(1);
        let Err(error) = watch_manifest(&dir.path().join("absent/workers.json"), tx.clone()) else {
            panic!("watched a missing directory");
        };
        assert!(
            matches!(&error.kind, notify::ErrorKind::Io(io) if io.kind() == std::io::ErrorKind::NotFound),
            "{error}"
        );
        assert!(
            error.paths.iter().any(|path| path.ends_with("absent")),
            "{error}"
        );

        let file = dir.path().join("a-file");
        std::fs::write(&file, "").unwrap();
        assert!(watch_manifest(&file.join("workers.json"), tx).is_err());
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
        discovery.last_good.as_ref().map(|desired| {
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
}
