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
use smg_worker::identity::{
    endpoint::{Endpoint, EndpointKey},
    url_door::listed_worker_url_refusal,
};
use tokio::{sync::mpsc, task, time};
use tracing::{info, warn};

use super::{
    config::{DiscoveryConfigError, DiscoveryConfigResult},
    host::DiscoveryHost,
    provider::{DiscoveredWorker, DiscoveryKind, Roles},
    reconciler::DesiredState,
    series,
    snapshot::{LastGood, SnapshotError},
};
use crate::fs_watch::{settle, DirectoryWatch};

/// The manifest version this crate reads.
const MANIFEST_VERSION: u64 = 1;

/// File discovery as the running host needs it.
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
    /// The roles decide which worker types a manifest may list. A routing mode
    /// with no worker pools to fill has none, and is rejected.
    pub fn new(
        path: PathBuf,
        check_interval: Duration,
        roles: Option<Roles>,
    ) -> DiscoveryConfigResult<Self> {
        let roles = roles.ok_or_else(|| DiscoveryConfigError::ValidationFailed {
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

/// The manifest as the file holds it, and as a host writes its registry out
/// (see [`export_manifest`]).
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
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

/// The registered fleet in the manifest's shape, for a host to export: what
/// a second host's file provider reads. A data-parallel group is one record
/// at the address it was registered with. No `id` is written: the reader
/// derives one from the address, and an id that changed with every restart
/// of the exporting host would re-register the fleet on every reader. A
/// worker's key is no manifest field and an address's credentials never
/// leave: both are dropped, and each such address is named in `warnings`.
pub fn export_manifest(workers: &[WorkerInfo]) -> Manifest {
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
        // The worker URL door first, as the worker API and a start ask it:
        // an address its two readers read apart is not stored, and neither
        // is a `@rank` suffix, which is the workflow's to assign. The door's
        // reason names no part of the address behind its scheme. The parse
        // below names the host or the port as it reads them, and in an
        // address the two read apart that is a credential to a client.
        let written = self.url.trim();
        if let Some(reason) = listed_worker_url_refusal(written) {
            return Err(format!("invalid url: {reason}"));
        }
        let endpoint = Endpoint::parse(&self.url).map_err(|e| format!("invalid url: {e}"))?;
        // The door again, of the address as it is kept: the registration is
        // handed the endpoint as it renders, which is not always the text as
        // written. The rendering drops a path's trailing slashes, and the
        // door reads an address less its slashes under `http` and `https`
        // alone: `grpc://token@8080/` is a userinfo and a host as written,
        // and a host and its rank as kept.
        let kept = endpoint.render();
        let kept_refusal = (kept != written)
            .then(|| listed_worker_url_refusal(&kept))
            .flatten();
        if let Some(reason) = kept_refusal {
            return Err(format!("invalid url: {reason}"));
        }

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
            // The manifest has no pairing protocol field: only the one an
            // engine reports about itself applies.
            pairing_protocol: None,
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
            Self::Unreadable { .. } => series::SNAPSHOT_ERROR_UNREADABLE,
            Self::Invalid { .. } => series::SNAPSHOT_ERROR_INVALID,
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
    host: Arc<dyn DiscoveryHost>,
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
    let handle = task::spawn(FileDiscovery::new(config, host).run());

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
    host: Arc<dyn DiscoveryHost>,
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
    fn new(config: FileProviderConfig, host: Arc<dyn DiscoveryHost>) -> Self {
        let (wake, woken) = mpsc::channel(1);
        let last_good = LastGood::new(
            DiscoveryKind::File,
            format!("Discovery manifest {}", config.path.display()),
            "manifest",
        );
        Self {
            config,
            host,
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
            .reconcile(self.host.as_ref(), started_at)
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
#[path = "../tests/unit/file/tests.rs"]
mod tests;
