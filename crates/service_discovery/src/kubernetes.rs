//! Kubernetes worker discovery: the Pods a label selector names, as workers.
//!
//! A reflector keeps a store of the Pods in view consistent with the API
//! server (initial list, watch, re-list on desync, reconnect with backoff);
//! store changes rebuild the published workers, and a periodic tick reconciles
//! the last build again. What a Pod publishes:
//!
//! - It is in view when its labels match the worker selector, or in
//!   disaggregated routing any of the role selectors; its role is the first
//!   role selector it matches, in encode, prefill, decode order.
//! - It is published only while Ready and Running and not being deleted: a
//!   terminating Pod leaves the desired set at the start of its grace period.
//! - It publishes one worker per data port (the worker-ports annotation, else
//!   the configured port), identified as `{pod uid}:{port}`, so a Pod
//!   recreated at the same address is a new instance and a container restart
//!   inside it is not.
//! - Its pairing-protocol label, where it carries one, is the explicit
//!   prefill/decode pairing protocol of each of its workers: an operator
//!   that rolls both roles stamps one value on the Pods of one rollout, so a
//!   prefill of one rollout does not pair with a decode of another.
//!
//! Ownership, the registry diff and the job submissions belong to the shared
//! reconciler; nothing here reads the registry.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    net::{IpAddr, SocketAddr},
    sync::Arc,
    time::Duration,
};

use futures::{Stream, StreamExt};
use k8s_openapi::api::core::v1::{Pod, PodCondition};
use kube::{
    api::Api,
    runtime::{
        reflector::{self, Store},
        watcher::{self, watcher, Config},
        WatchStreamExt,
    },
    Client,
};
use openai_protocol::worker::WorkerType;
use rustls::crypto::ring;
use smg_worker::{
    identity::{
        basic::{MOONCAKE_CONNECTOR, NIXL_CONNECTOR},
        endpoint::Endpoint,
    },
    pd::pairing::{connector_profile, KvTransferProfile},
};
use tokio::{sync::Notify, task, time};
use tracing::{debug, error, info, warn};

use super::{
    host::DiscoveryHost,
    provider::{DiscoveredWorker, DiscoveryKind},
    reconciler::{self, DesiredState},
};

#[cfg(test)]
std::thread_local! {
    /// The `compute_desired_state` calls on this thread, isolated because the
    /// test harness runs each unit test on its own thread.
    static PROJECTION_PASSES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Labels Kubernetes discovery has always stamped on the workers it registers,
/// kept for existing selectors and dashboards. Ownership no longer depends on
/// them: the shared reconciler decides it from the generic provider label.
pub const POD_NAME_LABEL: &str = "smg.ai/pod-name";
pub const POD_UID_LABEL: &str = "smg.ai/pod-uid";

/// Source for per-worker model_id override during Kubernetes service discovery.
#[derive(Debug, Clone)]
pub enum ModelIdSource {
    /// Use the pod's namespace as the model_id.
    Namespace,
    /// Use a specific pod label value as the model_id.
    Label(String),
    /// Use a specific pod annotation value as the model_id.
    Annotation(String),
}

impl ModelIdSource {
    /// Parse a CLI string like `"namespace"`, `"label:key"`, or `"annotation:key"`.
    pub fn parse(s: &str) -> Result<Self, String> {
        if s.eq_ignore_ascii_case("namespace") {
            Ok(Self::Namespace)
        } else if let Some(key) = s.strip_prefix("label:") {
            if key.is_empty() {
                Err("label: requires a key name".to_string())
            } else {
                Ok(Self::Label(key.to_string()))
            }
        } else if let Some(key) = s.strip_prefix("annotation:") {
            if key.is_empty() {
                Err("annotation: requires a key name".to_string())
            } else {
                Ok(Self::Annotation(key.to_string()))
            }
        } else {
            Err(format!(
                "Invalid model-id-from value '{s}'. Expected: namespace, label:<key>, or annotation:<key>"
            ))
        }
    }

    /// Extract the model_id value from a Kubernetes Pod object.
    pub fn extract(&self, pod: &Pod) -> Option<String> {
        match self {
            Self::Namespace => pod.metadata.namespace.clone(),
            Self::Label(key) => pod
                .metadata
                .labels
                .as_ref()
                .and_then(|labels| labels.get(key).cloned()),
            Self::Annotation(key) => pod
                .metadata
                .annotations
                .as_ref()
                .and_then(|annotations| annotations.get(key).cloned()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ServiceDiscoveryConfig {
    pub selector: HashMap<String, String>,
    pub check_interval: Duration,
    pub port: u16,
    pub namespace: Option<String>,
    // Disaggregated mode specific configuration
    pub disaggregated_mode: bool,
    pub encode_selector: HashMap<String, String>,
    pub prefill_selector: HashMap<String, String>,
    pub decode_selector: HashMap<String, String>,
    // Bootstrap port annotation specific to mooncake implementation
    pub bootstrap_port_annotation: String,
    /// Annotation listing the pod's worker data ports (comma-separated).
    /// Absent = single worker at `port`.
    pub worker_ports_annotation: String,
    /// Editing the KV annotations on a live Pod updates its worker in place.
    pub kv_connector_annotation: String,
    pub kv_engine_id_annotation: String,
    /// Pod label holding the prefill/decode pairing protocol. Trimmed; an
    /// absent or empty label states none. Relabelling a live Pod updates its
    /// workers in place.
    pub pairing_protocol_label: String,
    /// Per-worker model_id override source from pod metadata.
    pub model_id_source: Option<ModelIdSource>,
}

impl ServiceDiscoveryConfig {
    /// Build a label selector string for K8s list calls.
    ///
    /// In regular mode, uses the worker selector directly.
    /// In disaggregated mode, uses labels common to role selectors so a single
    /// list call covers all selected pod types. If there are no common
    /// labels, returns an empty string (no server-side filtering).
    fn list_label_selector(&self) -> String {
        if self.disaggregated_mode {
            let selectors = self.disaggregated_selectors();
            let Some(first) = selectors.first() else {
                return String::new();
            };
            selector_string(first.iter().filter(|(k, v)| {
                selectors
                    .iter()
                    .skip(1)
                    .all(|selector| selector.get(*k) == Some(*v))
            }))
        } else {
            selector_string(&self.selector)
        }
    }

    /// The role selectors in the order a Pod is matched against them.
    fn role_selectors(&self) -> [(&HashMap<String, String>, PodType); 3] {
        [
            (&self.encode_selector, PodType::Encode),
            (&self.prefill_selector, PodType::Prefill),
            (&self.decode_selector, PodType::Decode),
        ]
    }

    /// The role selectors that are set.
    fn disaggregated_selectors(&self) -> Vec<&HashMap<String, String>> {
        self.role_selectors()
            .into_iter()
            .map(|(selector, _)| selector)
            .filter(|selector| !selector.is_empty())
            .collect()
    }

    /// The role a Pod's labels give it: the first role selector it matches, in
    /// encode, prefill, decode order, and `Regular` when none does. Outside
    /// disaggregated routing every Pod is `Regular`; the role selectors are
    /// not read.
    fn pod_type_of(&self, pod: &Pod) -> PodType {
        if !self.disaggregated_mode {
            return PodType::Regular;
        }
        self.role_selectors()
            .into_iter()
            .find(|(selector, _)| matches_selector(pod, selector))
            .map_or(PodType::Regular, |(_, role)| role)
    }
}

/// A selector as the API server reads it: `key=value` pairs joined by commas.
/// Shared with the mesh's router-peer watcher.
pub fn selector_string<'a>(pairs: impl IntoIterator<Item = (&'a String, &'a String)>) -> String {
    pairs
        .into_iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(",")
}

/// Build a kube watcher Config that pushes the given label selector down to
/// the API server, logging the `kind` watcher's start at INFO. An empty
/// selector falls back to `Config::default()` (no server-side label
/// filtering) so the watcher still functions when no selector is set.
/// Shared with the mesh's router-peer watcher.
pub fn build_watcher_config(kind: &str, label_selector: &str) -> Config {
    info!("Starting K8s {kind} watcher | selector: '{label_selector}'");
    if label_selector.is_empty() {
        Config::default()
    } else {
        Config::default().labels(label_selector)
    }
}

impl Default for ServiceDiscoveryConfig {
    fn default() -> Self {
        ServiceDiscoveryConfig {
            selector: HashMap::new(),
            check_interval: Duration::from_secs(60),
            port: 8000,
            namespace: None,
            disaggregated_mode: false,
            encode_selector: HashMap::new(),
            prefill_selector: HashMap::new(),
            decode_selector: HashMap::new(),
            bootstrap_port_annotation: "sglang.ai/bootstrap-port".to_string(),
            worker_ports_annotation: "smg.ai/worker-ports".to_string(),
            kv_connector_annotation: "smg.ai/kv-connector".to_string(),
            kv_engine_id_annotation: "smg.ai/kv-engine-id".to_string(),
            pairing_protocol_label: "smg.ai/pairing-protocol".to_string(),
            model_id_source: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PodType {
    Encode,
    Prefill,
    Decode,
    Regular,
}

impl PodType {
    /// The worker pool a Pod of this role joins.
    fn worker_type(&self) -> WorkerType {
        match self {
            Self::Encode => WorkerType::Encode,
            Self::Prefill => WorkerType::Prefill,
            Self::Decode => WorkerType::Decode,
            Self::Regular => WorkerType::Regular,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PodInfo {
    pub name: String,
    pub uid: String,
    pub ip: IpAddr,
    pub status: String,
    pub is_ready: bool,
    pub pod_type: Option<PodType>,
    /// Worker data ports: the worker-ports annotation when present, else the
    /// single configured discovery port.
    pub ports: Vec<u16>,
    /// Per-port bootstrap ports (Encode/Prefill only), aligned with `ports`.
    pub bootstrap_ports: Vec<Option<u16>>,
    pub kv_connector: Option<String>,
    pub kv_engine_ids: Vec<Option<String>>,
    /// The Pod's explicit pairing protocol: the value of the label
    /// [`ServiceDiscoveryConfig::pairing_protocol_label`] names, trimmed.
    /// `None` when the label is absent or empty.
    pub pairing_protocol: Option<String>,
    pub model_id_override: Option<String>,
}

/// Whether every label of `selector` is on `pod`; an empty selector matches
/// no Pod. Shared with the mesh's router-peer watcher.
pub fn matches_selector(pod: &Pod, selector: &HashMap<String, String>) -> bool {
    if selector.is_empty() {
        return false;
    }

    pod.metadata
        .labels
        .as_ref()
        .is_some_and(|labels| selector.iter().all(|(k, v)| labels.get(k) == Some(v)))
}

/// Whether a Pod's conditions carry `Ready = True`, Kubernetes' traffic-admission
/// signal. Shared with the mesh's router-peer watcher.
pub fn pod_is_ready(conditions: Option<&[PodCondition]>) -> bool {
    conditions.is_some_and(|conditions| {
        conditions
            .iter()
            .any(|condition| condition.type_ == "Ready" && condition.status == "True")
    })
}

impl PodInfo {
    pub fn should_include(pod: &Pod, config: &ServiceDiscoveryConfig) -> bool {
        if config.disaggregated_mode {
            let selectors = config.disaggregated_selectors();
            if selectors.is_empty() {
                warn!("Disaggregated mode enabled but all role selectors are empty");
                return false;
            }
            selectors
                .iter()
                .any(|selector| matches_selector(pod, selector))
        } else {
            if config.selector.is_empty() {
                warn!("Regular mode enabled but selector is empty");
                return false;
            }
            matches_selector(pod, &config.selector)
        }
    }

    pub fn from_pod(pod: &Pod, config: Option<&ServiceDiscoveryConfig>) -> Option<Self> {
        let name = pod.metadata.name.clone()?;
        let Some(uid) = pod.metadata.uid.clone() else {
            warn!(
                "Pod {} has no UID, skipping -- cannot track identity for reconciliation",
                name
            );
            return None;
        };
        let status = pod.status.clone()?;
        let ip = pod_ip(&name, &status.pod_ip?)?;
        let is_ready = pod_is_ready(status.conditions.as_deref());
        let pod_status = status.phase.unwrap_or_else(|| "Unknown".to_string());

        let pod_type = config.map(|config| config.pod_type_of(pod));
        let ports = config
            .map(|config| resolve_worker_ports(&name, pod, config))
            .unwrap_or_default();
        let bootstrap_ports = if matches!(&pod_type, Some(PodType::Encode | PodType::Prefill)) {
            config
                .map(|config| resolve_bootstrap_ports(&name, pod, config, ports.len()))
                .unwrap_or_default()
        } else {
            vec![None; ports.len()]
        };
        let kv_connector = config.and_then(|config| kv_connector_of(&name, pod, config));
        let kv_engine_ids = config
            .map(|config| resolve_kv_engine_ids(&name, pod, config, ports.len()))
            .unwrap_or_else(|| vec![None; ports.len()]);
        let pairing_protocol =
            config.and_then(|config| label_value(pod, &config.pairing_protocol_label));
        let model_id_override = config
            .and_then(|config| config.model_id_source.as_ref())
            .and_then(|source| source.extract(pod));

        Some(PodInfo {
            name,
            uid,
            ip,
            status: pod_status,
            is_ready,
            pod_type,
            ports,
            bootstrap_ports,
            kv_connector,
            kv_engine_ids,
            pairing_protocol,
            model_id_override,
        })
    }

    pub fn is_healthy(&self) -> bool {
        self.is_ready && self.status == "Running"
    }
}

/// The Pod's address, parsed rather than kept as a string so the worker URL
/// renders an IPv6 address in bracketed form; `format!("{ip}:{port}")` on the
/// string would produce the unusable `fd00::1:8080`.
pub fn pod_ip(name: &str, raw: &str) -> Option<IpAddr> {
    raw.parse()
        .inspect_err(|e| warn!("Pod {} has an unparsable Pod IP '{}': {e}", name, raw))
        .ok()
}

/// The KV connector a Pod announces. One the disaggregated pairing has no
/// profile for is passed through as named, with a warning.
fn kv_connector_of(name: &str, pod: &Pod, config: &ServiceDiscoveryConfig) -> Option<String> {
    let connector = annotation_value(pod, &config.kv_connector_annotation)?;
    if connector != MOONCAKE_CONNECTOR
        && connector != NIXL_CONNECTOR
        && connector_profile(connector) != KvTransferProfile::Store
    {
        warn!(
            "Pod {}: {} annotation '{}' is not a connector with explicit PD handling; \
             using passthrough behavior",
            name, config.kv_connector_annotation, connector
        );
    }
    Some(connector.to_string())
}

/// Parse a comma-separated port list. Preserves order, rejects 0 and junk.
fn parse_port_list(raw: &str) -> Option<Vec<u16>> {
    let mut ports = Vec::new();
    for part in raw.split(',') {
        let port: u16 = part.trim().parse().ok()?;
        if port == 0 {
            return None;
        }
        ports.push(port);
    }
    Some(ports)
}

/// Data ports for the pod's workers: the worker-ports annotation when present
/// (deduped, order-preserving), else the single configured discovery port.
fn resolve_worker_ports(pod_name: &str, pod: &Pod, config: &ServiceDiscoveryConfig) -> Vec<u16> {
    let Some(raw) = pod
        .metadata
        .annotations
        .as_ref()
        .and_then(|annotations| annotations.get(&config.worker_ports_annotation))
    else {
        return vec![config.port];
    };
    match parse_port_list(raw) {
        Some(mut ports) => {
            let mut seen = HashSet::new();
            ports.retain(|port| seen.insert(*port));
            ports
        }
        None => {
            warn!(
                "Pod {}: invalid {} annotation '{}', falling back to port {}",
                pod_name, config.worker_ports_annotation, raw, config.port
            );
            vec![config.port]
        }
    }
}

fn annotation_value<'a>(pod: &'a Pod, key: &str) -> Option<&'a str> {
    pod.metadata
        .annotations
        .as_ref()?
        .get(key)
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
}

/// The trimmed value of the Pod's label `key`; `None` when the label is
/// absent or empty.
fn label_value(pod: &Pod, key: &str) -> Option<String> {
    pod.metadata
        .labels
        .as_ref()?
        .get(key)
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

/// Resolve one unique KV engine ID per worker port; unlike bootstrap ports,
/// one ID cannot be broadcast because each engine core owns a distinct ID.
fn resolve_kv_engine_ids(
    pod_name: &str,
    pod: &Pod,
    config: &ServiceDiscoveryConfig,
    num_ports: usize,
) -> Vec<Option<String>> {
    let Some(raw) = annotation_value(pod, &config.kv_engine_id_annotation) else {
        return vec![None; num_ports];
    };
    let ids: Vec<&str> = raw.split(',').map(str::trim).collect();
    let mut seen = HashSet::new();
    if ids.len() != num_ports
        || ids.iter().any(|id| id.is_empty())
        || !ids.iter().all(|id| seen.insert(*id))
    {
        warn!(
            "Pod {}: {} annotation '{}' must contain one distinct non-empty ID for each of {} \
             worker port(s), ignoring",
            pod_name, config.kv_engine_id_annotation, raw, num_ports
        );
        return vec![None; num_ports];
    }
    ids.into_iter().map(|id| Some(id.to_string())).collect()
}

/// Bootstrap ports aligned with the pod's worker ports: a single value applies
/// to every worker; a list must match the worker port count.
fn resolve_bootstrap_ports(
    pod_name: &str,
    pod: &Pod,
    config: &ServiceDiscoveryConfig,
    num_ports: usize,
) -> Vec<Option<u16>> {
    let Some(raw) = pod
        .metadata
        .annotations
        .as_ref()
        .and_then(|annotations| annotations.get(&config.bootstrap_port_annotation))
    else {
        return vec![None; num_ports];
    };
    match parse_port_list(raw) {
        Some(ports) if ports.len() == 1 => vec![Some(ports[0]); num_ports],
        Some(ports) if ports.len() == num_ports => ports.into_iter().map(Some).collect(),
        _ => {
            warn!(
                "Pod {}: {} annotation '{}' does not align with {} worker port(s), ignoring",
                pod_name, config.bootstrap_port_annotation, raw, num_ports
            );
            vec![None; num_ports]
        }
    }
}

pub(super) async fn start_kubernetes_discovery(
    config: ServiceDiscoveryConfig,
    host: Arc<dyn DiscoveryHost>,
) -> Result<task::JoinHandle<()>, kube::Error> {
    let _ = ring::default_provider().install_default();

    let client = Client::try_default().await?;

    Ok(run_service_discovery(client, config, host))
}

/// Run discovery against an injected client. Compiled only for this crate's
/// own integration tests, which point it at a scripted API server.
#[cfg(feature = "test-util")]
pub fn start_service_discovery_with_client(
    client: Client,
    config: ServiceDiscoveryConfig,
    host: Arc<dyn DiscoveryHost>,
) -> task::JoinHandle<()> {
    let _ = ring::default_provider().install_default();
    run_service_discovery(client, config, host)
}

fn run_service_discovery(
    client: Client,
    config: ServiceDiscoveryConfig,
    host: Arc<dyn DiscoveryHost>,
) -> task::JoinHandle<()> {
    if config.disaggregated_mode {
        info!(
            "Starting K8s service discovery | disaggregated mode | encode: '{}' | prefill: '{}' | decode: '{}'",
            selector_string(&config.encode_selector),
            selector_string(&config.prefill_selector),
            selector_string(&config.decode_selector)
        );
    } else {
        info!(
            "Starting K8s service discovery | selector: '{}'",
            selector_string(&config.selector)
        );
    }

    #[expect(
        clippy::disallowed_methods,
        reason = "worker discovery runs for the lifetime of the server; shutdown aborts the handle"
    )]
    let handle = task::spawn(async move {
        let pods: Api<Pod> = if let Some(namespace) = &config.namespace {
            Api::namespaced(client, namespace)
        } else {
            Api::all(client)
        };

        debug!("K8s service discovery initialized");

        // Level-triggered reconcile: the reflector keeps `store` consistent
        // with the API server (initial LIST → watch, re-LIST on desync,
        // reconnect with internal backoff). Visible store changes rebuild the
        // desired workers; periodic ticks reconcile that last build again.
        let (store, writer) = reflector::store::<Pod>();
        let notify = Notify::new();

        let watcher_config = build_watcher_config("worker", &config.list_label_selector());
        // managed_fields dominate pod object size and nothing here reads them;
        // pruning keeps the long-lived Store lean.
        let stream = watcher(pods, watcher_config)
            .modify(|pod| pod.metadata.managed_fields = None)
            .default_backoff()
            .reflect(writer)
            .boxed();

        tokio::join!(
            drive_watch(stream, &notify),
            reconcile_on_change(&store, &notify, &config, host.as_ref())
        );
    });

    handle
}

/// Wake the reconcile loop when the store changes. The initial list wakes the
/// store's readiness waiter; later complete re-lists wake this loop too. A
/// watcher error is logged, and the stream retries with its own backoff.
async fn drive_watch(
    mut stream: impl Stream<Item = watcher::Result<watcher::Event<Pod>>> + Unpin,
    notify: &Notify,
) {
    let mut initial_sync = true;
    while let Some(item) = stream.next().await {
        match item {
            Ok(watcher::Event::Apply(_) | watcher::Event::Delete(_)) => notify.notify_one(),
            Ok(watcher::Event::InitDone) => {
                if initial_sync {
                    initial_sync = false;
                } else {
                    notify.notify_one();
                }
            }
            Ok(watcher::Event::Init | watcher::Event::InitApply(_)) => {}
            Err(e) => {
                error!("K8s worker watcher error (auto-retrying with backoff): {e}");
            }
        }
    }
    error!("K8s worker watcher stream ended; discovery no longer receives updates");
}

/// Rebuild after a store change and reconcile the last build on every tick,
/// once the store holds the initial list.
async fn reconcile_on_change(
    store: &Store<Pod>,
    notify: &Notify,
    config: &ServiceDiscoveryConfig,
    host: &dyn DiscoveryHost,
) {
    if store.wait_until_ready().await.is_err() {
        error!("K8s worker watcher dropped before initial sync; reconciliation disabled");
        return;
    }
    info!(
        "K8s worker store synced, reconciling on change and every {}s",
        config.check_interval.as_secs()
    );
    let mut interval = time::interval(config.check_interval);
    interval.set_missed_tick_behavior(time::MissedTickBehavior::Delay);
    interval.tick().await;

    let started_at = time::Instant::now();
    let mut desired = compute_desired_state(&store.state(), config);
    reconciler::reconcile(&desired, DiscoveryKind::Kubernetes, host, started_at).await;

    loop {
        let changed = tokio::select! {
            () = notify.notified() => {
                // Coalesce event bursts (relists, rollouts) into one pass.
                time::sleep(Duration::from_secs(1)).await;
                true
            }
            _ = interval.tick() => false
        };
        let started_at = time::Instant::now();
        if changed {
            desired = compute_desired_state(&store.state(), config);
        }
        reconciler::reconcile(&desired, DiscoveryKind::Kubernetes, host, started_at).await;
    }
}

fn compute_desired_state(pods: &[Arc<Pod>], config: &ServiceDiscoveryConfig) -> DesiredState {
    #[cfg(test)]
    PROJECTION_PASSES.set(PROJECTION_PASSES.get() + 1);

    let mut workers = Vec::new();
    for pod in pods {
        if !PodInfo::should_include(pod, config) {
            continue;
        }
        // Terminating pods leave the desired set immediately so their
        // workers drain at the start of the grace period, not the end.
        if pod.metadata.deletion_timestamp.is_some() {
            continue;
        }
        let Some(info) = PodInfo::from_pod(pod, Some(config)) else {
            continue;
        };
        // Ready is Kubernetes' traffic-admission signal — controllers drive it
        // through readiness gates, and direct Pod-IP routing honors it as a
        // Service/EndpointSlice consumer would. Running is required too, so a
        // Pod is either published or it is not: the shared reconciler has no
        // "present but not yet addable" state to hold one in.
        if !info.is_healthy() {
            continue;
        }
        for (index, port) in info.ports.iter().enumerate() {
            let url = SocketAddr::new(info.ip, *port).to_string();
            // A socket address always parses; the fallible form keeps the
            // provider honest if the address ever stops coming from one.
            let Ok(endpoint) = Endpoint::parse(&url) else {
                // A socket address carries no userinfo, and the masked form of
                // one the parser refused would only read as its marker.
                warn!(
                    "Pod {} has an unusable worker address '{}', skipping",
                    info.name, url
                );
                continue;
            };
            workers.push(DiscoveredWorker {
                discovery_id: format!("{}:{port}", info.uid),
                endpoint,
                worker_type: info
                    .pod_type
                    .as_ref()
                    .map_or(WorkerType::Regular, PodType::worker_type),
                bootstrap_port: info.bootstrap_ports.get(index).copied().flatten(),
                model_id_override: info.model_id_override.clone(),
                kv_connector: info.kv_connector.clone(),
                // Kubernetes has no kv_role annotation; `None` keeps the
                // registration workflow's existing label fallback.
                kv_role: None,
                kv_engine_id: info.kv_engine_ids.get(index).cloned().flatten(),
                pairing_protocol: info.pairing_protocol.clone(),
                compat_labels: BTreeMap::from([
                    (POD_NAME_LABEL.to_string(), info.name.clone()),
                    (POD_UID_LABEL.to_string(), info.uid.clone()),
                ]),
            });
        }
    }
    DesiredState::from_workers(workers)
}

#[cfg(test)]
mod tests;
