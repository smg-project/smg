//! File worker discovery: the workers listed in a JSON manifest on disk.
//!
//! The manifest is reread in full on every tick. A read that succeeds replaces
//! the last good snapshot; one that fails is reported and leaves that snapshot
//! in place, so a half-written or broken file never removes a worker. Every
//! tick then reconciles the last good snapshot through the shared reconciler,
//! which retries whatever an earlier pass left undone.
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

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    path::PathBuf,
    sync::Arc,
    time::Duration,
};

use openai_protocol::worker::WorkerType;
use serde::Deserialize;
use tokio::{task, time};
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

/// Read the manifest now and on every tick after, reconciling the last good
/// snapshot each time.
pub(super) fn start_file_discovery(
    config: FileProviderConfig,
    app_context: Arc<AppContext>,
) -> task::JoinHandle<()> {
    info!(
        "Starting file service discovery | manifest: {} | reread every {}s",
        config.path.display(),
        config.check_interval.as_secs_f64()
    );

    #[expect(
        clippy::disallowed_methods,
        reason = "worker discovery runs for the lifetime of the server; shutdown aborts the handle"
    )]
    let handle = task::spawn(async move {
        let kind = DiscoveryKind::File;
        let mut last_good: Option<DesiredState> = None;
        let mut interval = time::interval(config.check_interval);
        interval.set_missed_tick_behavior(time::MissedTickBehavior::Delay);
        loop {
            interval.tick().await;
            let started_at = time::Instant::now();
            match read_manifest(&config).await {
                Ok(desired) => {
                    if last_good.as_ref() != Some(&desired) {
                        info!(
                            "Discovery manifest {} lists {} worker(s)",
                            config.path.display(),
                            desired.workers.len()
                        );
                    }
                    last_good = Some(desired);
                }
                Err(error) => {
                    Metrics::record_discovery_snapshot_error(
                        kind.metric_label(),
                        error.metric_reason(),
                    );
                    if last_good.is_some() {
                        warn!("{error}; keeping the last good manifest");
                    } else {
                        warn!("{error}; no workers discovered yet");
                    }
                }
            }
            if let Some(desired) = &last_good {
                reconciler::reconcile(desired, kind, &app_context, started_at).await;
            }
        }
    });

    handle
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
