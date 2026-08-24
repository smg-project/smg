use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Duration,
};

use futures::StreamExt;
use k8s_openapi::api::core::v1::Pod;
use kube::{
    api::Api,
    runtime::{
        reflector::{self, Store},
        watcher::{watcher, Config, Event},
        WatchStreamExt,
    },
    Client,
};
use openai_protocol::worker::{WorkerSpec, WorkerType};
use rustls::crypto::ring;
use smg_mesh::{
    gossip::{NodeState, NodeStatus},
    ClusterState,
};
use tokio::{sync::Notify, task, time};
use tracing::{debug, error, info, warn};

use crate::{
    app_context::AppContext,
    observability::metrics::{metrics_labels, Metrics},
    worker::WorkerOrigin,
    workflow::{Job, WorkerRegistrationMode},
};

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
    pub enabled: bool,
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
    // Router node discovery for mesh
    pub router_selector: HashMap<String, String>,
    pub router_mesh_port_annotation: String,
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
            first
                .iter()
                .filter(|(k, v)| {
                    selectors
                        .iter()
                        .skip(1)
                        .all(|selector| selector.get(*k) == Some(*v))
                })
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(",")
        } else {
            self.selector
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(",")
        }
    }

    fn disaggregated_selectors(&self) -> Vec<&HashMap<String, String>> {
        [
            &self.encode_selector,
            &self.prefill_selector,
            &self.decode_selector,
        ]
        .into_iter()
        .filter(|selector| !selector.is_empty())
        .collect()
    }

    /// Build a label selector string for router pod K8s list/watch calls.
    /// Returns an empty string when the router selector is unset, in which
    /// case the watcher should fall back to listing without server-side
    /// label filtering.
    fn router_label_selector(&self) -> String {
        self.router_selector
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// Build a kube watcher Config that pushes the given label selector down to
/// the API server, logging watcher startup at INFO. An empty selector falls
/// back to `Config::default()` (no server-side label filtering) so the
/// watcher still functions when no selector is set.
fn build_watcher_config(watcher_kind: &str, label_selector: &str) -> Config {
    info!(
        "Starting K8s {} watcher | selector: '{}'",
        watcher_kind, label_selector
    );
    if label_selector.is_empty() {
        Config::default()
    } else {
        Config::default().labels(label_selector)
    }
}

impl Default for ServiceDiscoveryConfig {
    fn default() -> Self {
        ServiceDiscoveryConfig {
            enabled: false,
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
            router_selector: HashMap::new(),
            router_mesh_port_annotation: "sglang.ai/mesh-port".to_string(),
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

/// Labels stamped on every discovery-created worker; workers carrying
/// [`POD_UID_LABEL`] are owned (added/removed) by the K8s reconciler.
pub const POD_NAME_LABEL: &str = "smg.ai/pod-name";
pub const POD_UID_LABEL: &str = "smg.ai/pod-uid";

#[derive(Debug, Clone)]
pub struct PodInfo {
    pub name: String,
    pub uid: String,
    pub ip: String,
    pub status: String,
    pub is_ready: bool,
    pub pod_type: Option<PodType>,
    /// Worker data ports: the worker-ports annotation when present, else the
    /// single configured discovery port.
    pub ports: Vec<u16>,
    /// Per-port bootstrap ports (Encode/Prefill only), aligned with `ports`.
    pub bootstrap_ports: Vec<Option<u16>>,
    pub is_router: bool,
    pub mesh_port: Option<u16>,
    pub model_id_override: Option<String>,
}

impl PodInfo {
    fn matches_selector(pod: &Pod, selector: &HashMap<String, String>) -> bool {
        if selector.is_empty() {
            return false;
        }

        pod.metadata
            .labels
            .as_ref()
            .is_some_and(|labels| selector.iter().all(|(k, v)| labels.get(k) == Some(v)))
    }

    pub fn should_include(pod: &Pod, config: &ServiceDiscoveryConfig) -> bool {
        if config.disaggregated_mode {
            let selectors = config.disaggregated_selectors();
            if selectors.is_empty() {
                warn!("Disaggregated mode enabled but all role selectors are empty");
                return false;
            }
            selectors
                .iter()
                .any(|selector| Self::matches_selector(pod, selector))
        } else {
            if config.selector.is_empty() {
                warn!("Regular mode enabled but selector is empty");
                return false;
            }
            Self::matches_selector(pod, &config.selector)
        }
    }

    pub fn from_pod(pod: &Pod, config: Option<&ServiceDiscoveryConfig>) -> Option<Self> {
        let name = pod.metadata.name.clone()?;
        let uid = match pod.metadata.uid.clone() {
            Some(uid) => uid,
            None => {
                warn!(
                    "Pod {} has no UID, skipping -- cannot track identity for reconciliation",
                    name
                );
                return None;
            }
        };
        let status = pod.status.clone()?;
        let pod_ip = status.pod_ip?;

        let is_ready = if let Some(conditions) = &status.conditions {
            conditions
                .iter()
                .any(|condition| condition.type_ == "Ready" && condition.status == "True")
        } else {
            false
        };

        let pod_status = status.phase.unwrap_or_else(|| "Unknown".to_string());

        let pod_type = if let Some(config) = config {
            if config.disaggregated_mode {
                if Self::matches_selector(pod, &config.encode_selector) {
                    Some(PodType::Encode)
                } else if Self::matches_selector(pod, &config.prefill_selector) {
                    Some(PodType::Prefill)
                } else if Self::matches_selector(pod, &config.decode_selector) {
                    Some(PodType::Decode)
                } else {
                    Some(PodType::Regular)
                }
            } else {
                Some(PodType::Regular)
            }
        } else {
            None
        };

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

        // Check if this is a router pod
        let is_router = if let Some(config) = config {
            !config.router_selector.is_empty()
                && Self::matches_selector(pod, &config.router_selector)
        } else {
            false
        };

        // Extract mesh port from annotation if this is a router pod
        let mesh_port = if is_router {
            if let Some(config) = config {
                pod.metadata
                    .annotations
                    .as_ref()
                    .and_then(|annotations| annotations.get(&config.router_mesh_port_annotation))
                    .and_then(|port_str| port_str.parse::<u16>().ok())
            } else {
                None
            }
        } else {
            None
        };

        // Extract model_id override from pod metadata if source is configured
        let model_id_override = config
            .and_then(|c| c.model_id_source.as_ref())
            .and_then(|source| source.extract(pod));

        Some(PodInfo {
            name,
            uid,
            ip: pod_ip,
            status: pod_status,
            is_ready,
            pod_type,
            ports,
            bootstrap_ports,
            is_router,
            mesh_port,
            model_id_override,
        })
    }

    pub fn is_healthy(&self) -> bool {
        self.is_ready && self.status == "Running"
    }
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

pub async fn start_service_discovery(
    config: ServiceDiscoveryConfig,
    app_context: Arc<AppContext>,
    mesh_cluster_state: Option<ClusterState>,
    mesh_port: Option<u16>,
) -> Result<task::JoinHandle<()>, kube::Error> {
    if !config.enabled {
        return Err(kube::Error::Api(
            kube::core::Status::failure("Service discovery is disabled", "ConfigurationError")
                .with_code(400)
                .boxed(),
        ));
    }

    let _ = ring::default_provider().install_default();

    let client = Client::try_default().await?;

    Ok(run_service_discovery(
        client,
        config,
        app_context,
        mesh_cluster_state,
        mesh_port,
    ))
}

/// Run discovery against an injected client. Compiled only for this crate's
/// own integration tests, which point it at a scripted API server.
#[cfg(feature = "test-util")]
pub fn start_service_discovery_with_client(
    client: Client,
    config: ServiceDiscoveryConfig,
    app_context: Arc<AppContext>,
    mesh_cluster_state: Option<ClusterState>,
    mesh_port: Option<u16>,
) -> task::JoinHandle<()> {
    let _ = ring::default_provider().install_default();
    run_service_discovery(client, config, app_context, mesh_cluster_state, mesh_port)
}

fn run_service_discovery(
    client: Client,
    config: ServiceDiscoveryConfig,
    app_context: Arc<AppContext>,
    mesh_cluster_state: Option<ClusterState>,
    mesh_port: Option<u16>,
) -> task::JoinHandle<()> {
    // Log the appropriate selectors based on mode
    if config.disaggregated_mode {
        let encode_selector = config
            .encode_selector
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(",");

        let prefill_selector = config
            .prefill_selector
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(",");

        let decode_selector = config
            .decode_selector
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(",");

        info!(
            "Starting K8s service discovery | disaggregated mode | encode: '{}' | prefill: '{}' | decode: '{}'",
            encode_selector, prefill_selector, decode_selector
        );
    } else {
        let label_selector = config
            .selector
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(",");

        info!(
            "Starting K8s service discovery | selector: '{}'",
            label_selector
        );
    }

    // Log router discovery if enabled
    if !config.router_selector.is_empty() {
        let router_selector = config
            .router_selector
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(",");
        info!(
            "Router node discovery enabled | selector: '{}' | mesh port annotation: '{}'",
            router_selector, config.router_mesh_port_annotation
        );
    }

    #[expect(
        clippy::disallowed_methods,
        reason = "service discovery runs for the lifetime of the server; shutdown is handled by dropping the handle"
    )]
    let handle = task::spawn(async move {
        let pods: Api<Pod> = if let Some(namespace) = &config.namespace {
            Api::namespaced(client, namespace)
        } else {
            Api::all(client)
        };

        debug!("K8s service discovery initialized");

        let config_arc = Arc::new(config);

        // Spawn router discovery task if enabled and mesh is available
        // Router discovery requires mesh to be enabled to update cluster state
        // If mesh is not enabled, router discovery is skipped and service discovery works independently
        if !config_arc.router_selector.is_empty() {
            if let (Some(cluster_state), Some(mesh_port)) = (mesh_cluster_state.clone(), mesh_port)
            {
                let router_config = config_arc.clone();
                let router_pods = pods.clone();
                #[expect(
                    clippy::disallowed_methods,
                    reason = "router discovery runs for the lifetime of the server alongside worker discovery"
                )]
                let router_handle = tokio::spawn(async move {
                    start_router_discovery(router_config, router_pods, cluster_state, mesh_port)
                        .await;
                });
                #[expect(
                    clippy::disallowed_methods,
                    reason = "monitor task runs for the lifetime of the server"
                )]
                tokio::spawn(async move {
                    if let Err(e) = router_handle.await {
                        error!(
                            "Router discovery task panicked and is no longer running: {}",
                            e
                        );
                    }
                });
                info!("Router discovery enabled (requires mesh to be enabled)");
            } else {
                warn!(
                    "Router selector configured but mesh is not enabled (mesh cluster state or mesh port not provided). \
                    Router discovery requires mesh to be enabled. Skipping router discovery."
                );
            }
        }

        // Level-triggered reconcile: the reflector keeps `store` consistent
        // with the API server (initial LIST → watch, re-LIST on desync,
        // reconnect with internal backoff). Every watch event and a periodic
        // tick trigger one pass diffing desired workers against the registry.
        let (store, writer) = reflector::store::<Pod>();
        let notify = Arc::new(Notify::new());

        let watcher_config = build_watcher_config("worker", &config_arc.list_label_selector());
        // managed_fields dominate pod object size and nothing here reads them;
        // pruning keeps the long-lived Store lean.
        let mut stream = watcher(pods, watcher_config)
            .modify(|pod| pod.metadata.managed_fields = None)
            .default_backoff()
            .reflect(writer)
            .boxed();

        let driver = {
            let notify = Arc::clone(&notify);
            async move {
                while let Some(item) = stream.next().await {
                    match item {
                        Ok(_) => notify.notify_one(),
                        Err(e) => {
                            error!("K8s worker watcher error (auto-retrying with backoff): {e}");
                        }
                    }
                }
                error!("K8s worker watcher stream ended; discovery no longer receives updates");
            }
        };

        let reconciler = async {
            if store.wait_until_ready().await.is_err() {
                error!("K8s worker watcher dropped before initial sync; reconciliation disabled");
                return;
            }
            info!(
                "K8s worker store synced, reconciling on change and every {}s",
                config_arc.check_interval.as_secs()
            );
            let mut interval = time::interval(config_arc.check_interval);
            interval.set_missed_tick_behavior(time::MissedTickBehavior::Delay);
            loop {
                tokio::select! {
                    () = notify.notified() => {
                        // Coalesce event bursts (relists, rollouts) into one pass.
                        time::sleep(Duration::from_secs(1)).await;
                    }
                    _ = interval.tick() => {}
                }
                reconcile_workers(&store, &config_arc, &app_context).await;
            }
        };

        tokio::join!(driver, reconciler);
    });

    handle
}

/// One worker the reconciler wants registered: a single engine server
/// (pod IP + data port) plus the metadata needed to build its spec.
#[derive(Debug, Clone)]
struct DesiredWorker {
    /// Bare host:port so DetectConnectionModeStep dual-probes HTTP and gRPC.
    url: String,
    worker_type: WorkerType,
    bootstrap_port: Option<u16>,
    pod_name: String,
    pod_uid: String,
    model_id_override: Option<String>,
}

/// Desired view of the cluster derived from the store snapshot.
#[derive(Debug, Default)]
struct DesiredState {
    /// Owning pod uid per worker URL for Ready, non-terminating Pods.
    /// Registered workers whose URL is absent — or owned by a different Pod
    /// uid — enter the existing drain/remove workflow.
    uid_by_url: HashMap<String, String>,
    /// Workers on Running, Ready Pods — registration candidates.
    addable: Vec<DesiredWorker>,
}

fn worker_type_for(pod_type: Option<&PodType>, disaggregated_mode: bool) -> WorkerType {
    if disaggregated_mode {
        match pod_type {
            Some(PodType::Encode) => WorkerType::Encode,
            Some(PodType::Prefill) => WorkerType::Prefill,
            Some(PodType::Decode) => WorkerType::Decode,
            _ => WorkerType::Regular,
        }
    } else {
        WorkerType::Regular
    }
}

fn compute_desired_state(pods: &[Arc<Pod>], config: &ServiceDiscoveryConfig) -> DesiredState {
    let mut state = DesiredState::default();
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
        // Pod readiness is Kubernetes' standard traffic-admission signal.
        // Controllers can drive it through readiness gates; direct Pod-IP
        // routing must honor the aggregate Ready condition just like a
        // Service/EndpointSlice consumer would.
        if !info.is_ready {
            continue;
        }
        for (index, port) in info.ports.iter().enumerate() {
            let url = format!("{}:{}", info.ip, port);
            if state.uid_by_url.contains_key(&url) {
                continue;
            }
            state.uid_by_url.insert(url.clone(), info.uid.clone());
            if info.is_healthy() {
                state.addable.push(DesiredWorker {
                    url,
                    worker_type: worker_type_for(info.pod_type.as_ref(), config.disaggregated_mode),
                    bootstrap_port: info.bootstrap_ports.get(index).copied().flatten(),
                    pod_name: info.name.clone(),
                    pod_uid: info.uid.clone(),
                    model_id_override: info.model_id_override.clone(),
                });
            }
        }
    }
    state
}

/// A registry worker owned by K8s discovery (stamped with [`POD_UID_LABEL`]).
#[derive(Debug, Clone)]
struct OwnedWorker {
    /// Scheme- and DP-rank-stripped `host:port`.
    url: String,
    pod_uid: String,
    /// Revision guard for removal: a concurrently replaced worker is skipped
    /// and re-evaluated on the next pass instead of removed blindly.
    revision: u64,
}

/// `http://10.0.0.1:8080@2` → `10.0.0.1:8080`.
fn canonical_host_port(url: &str) -> &str {
    let stripped = ["http://", "https://", "grpc://", "grpcs://", "ipc://"]
        .iter()
        .find_map(|scheme| url.strip_prefix(scheme))
        .unwrap_or(url);
    stripped.split('@').next().unwrap_or(stripped)
}

/// Snapshot the registry workers this reconciler owns: locally registered
/// (never mesh-imported) and stamped with the pod-uid label. Manually added
/// workers lack the label and are never touched.
fn k8s_owned_workers(app_context: &AppContext) -> Vec<OwnedWorker> {
    app_context
        .worker_registry
        .get_all_with_ids()
        .into_iter()
        .filter_map(|(id, worker)| {
            if app_context.worker_registry.origin_of(&id) != Some(WorkerOrigin::Local) {
                return None;
            }
            let pod_uid = worker.metadata().spec.labels.get(POD_UID_LABEL)?.clone();
            Some(OwnedWorker {
                url: canonical_host_port(worker.url()).to_string(),
                pod_uid,
                revision: worker.revision(),
            })
        })
        .collect()
}

#[derive(Debug, Default)]
struct ReconcileActions {
    /// Workers to register: new URLs, plus same-URL pods whose uid changed
    /// (restart with a stable IP).
    add: Vec<DesiredWorker>,
    /// Workers to remove: URL gone from the desired set, or owned by a pod
    /// uid that no longer holds the URL (covers a stale-scheme sibling the
    /// same-URL Upsert cannot replace).
    remove: Vec<OwnedWorker>,
}

fn compute_actions(desired: &DesiredState, registered: &[OwnedWorker]) -> ReconcileActions {
    let mut actions = ReconcileActions::default();

    let mut registered_uid: HashMap<&str, &str> = HashMap::new();
    for worker in registered {
        registered_uid.insert(worker.url.as_str(), worker.pod_uid.as_str());
        match desired.uid_by_url.get(worker.url.as_str()) {
            Some(uid) if *uid == worker.pod_uid => {}
            _ => actions.remove.push(worker.clone()),
        }
    }
    // DP-rank expansions share one canonical URL; remove it once.
    actions.remove.sort_unstable_by(|a, b| a.url.cmp(&b.url));
    actions.remove.dedup_by(|a, b| a.url == b.url);

    for worker in &desired.addable {
        match registered_uid.get(worker.url.as_str()) {
            Some(uid) if *uid == worker.pod_uid => {}
            _ => actions.add.push(worker.clone()),
        }
    }
    actions
}

fn build_worker_spec(desired: &DesiredWorker, app_context: &AppContext) -> WorkerSpec {
    let mut spec = WorkerSpec::new(desired.url.clone());
    spec.worker_type = desired.worker_type;
    spec.bootstrap_port = desired.bootstrap_port;
    spec.labels
        .insert(POD_NAME_LABEL.to_string(), desired.pod_name.clone());
    spec.labels
        .insert(POD_UID_LABEL.to_string(), desired.pod_uid.clone());
    // served_model_name is priority #2 in create_worker's model_id chain.
    if let Some(ref model_id) = desired.model_id_override {
        spec.labels
            .insert("served_model_name".to_string(), model_id.clone());
    }
    spec.api_key.clone_from(&app_context.router_config.api_key);
    spec.max_connection_attempts = app_context
        .router_config
        .health_check
        .success_threshold
        .max(1)
        * 20;
    spec
}

/// One reconcile pass: diff desired workers (store snapshot) against the
/// K8s-owned registry entries and submit Add/Remove jobs for the gap.
/// Failed or missed work is retried on the next pass by construction.
async fn reconcile_workers(
    store: &Store<Pod>,
    config: &ServiceDiscoveryConfig,
    app_context: &Arc<AppContext>,
) {
    let start = time::Instant::now();
    let snapshot = store.state();
    let desired = compute_desired_state(&snapshot, config);
    let registered = k8s_owned_workers(app_context);
    let actions = compute_actions(&desired, &registered);

    let desired_count = desired.uid_by_url.len();
    if actions.add.is_empty() && actions.remove.is_empty() {
        Metrics::set_discovery_workers_discovered(
            metrics_labels::DISCOVERY_KUBERNETES,
            desired_count,
        );
        return;
    }

    let Some(job_queue) = app_context.worker_job_queue.get() else {
        warn!(
            "JobQueue not initialized; deferring {} addition(s), {} removal(s)",
            actions.add.len(),
            actions.remove.len()
        );
        return;
    };

    // One in-flight job per URL: a pending/processing job owns that worker's
    // transition (a duplicate removal would find it already Draining, skip
    // the settle sleep, and collapse the drain window). Completed/failed
    // statuses do not block, so failures retry on the next pass.
    let in_flight = |url: &str| {
        job_queue
            .get_status(url)
            .is_some_and(|status| status.status == "pending" || status.status == "processing")
    };
    let removals: Vec<&OwnedWorker> = actions
        .remove
        .iter()
        .filter(|worker| !in_flight(&worker.url))
        .collect();
    let additions: Vec<&DesiredWorker> = actions
        .add
        .iter()
        .filter(|worker| !in_flight(&worker.url))
        .collect();

    if removals.is_empty() && additions.is_empty() {
        Metrics::set_discovery_workers_discovered(
            metrics_labels::DISCOVERY_KUBERNETES,
            desired_count,
        );
        return;
    }

    info!(
        "Reconciling workers: {} to add, {} to remove ({} desired)",
        additions.len(),
        removals.len(),
        desired_count
    );

    for worker in removals {
        info!(
            "Removing worker {}: pod unready, gone, terminating, or replaced",
            worker.url
        );
        let job = Job::RemoveWorker {
            url: worker.url.clone(),
            expected_revision: Some(worker.revision),
        };
        match job_queue.submit(job).await {
            Ok(()) => Metrics::record_discovery_deregistration(
                metrics_labels::DISCOVERY_KUBERNETES,
                metrics_labels::DEREGISTRATION_RECONCILED,
            ),
            Err(e) => error!("Failed to submit worker removal for {}: {}", worker.url, e),
        }
    }

    for worker in additions {
        info!(
            "Registering worker {} ({:?}) for pod {}",
            worker.url, worker.worker_type, worker.pod_name
        );
        let job = Job::AddWorker {
            config: Box::new(build_worker_spec(worker, app_context)),
            registration_mode: WorkerRegistrationMode::Upsert,
        };
        match job_queue.submit(job).await {
            Ok(()) => Metrics::record_discovery_registration(
                metrics_labels::DISCOVERY_KUBERNETES,
                metrics_labels::REGISTRATION_SUCCESS,
            ),
            Err(e) => {
                error!("Failed to submit worker addition for {}: {}", worker.url, e);
                Metrics::record_discovery_registration(
                    metrics_labels::DISCOVERY_KUBERNETES,
                    metrics_labels::REGISTRATION_FAILED,
                );
            }
        }
    }

    Metrics::set_discovery_workers_discovered(metrics_labels::DISCOVERY_KUBERNETES, desired_count);
    Metrics::record_discovery_sync_duration(metrics_labels::DISCOVERY_KUBERNETES, start.elapsed());
}

/// Apply one router-watcher event to the mesh cluster state. A hard delete
/// or a set deletion timestamp marks the node Down; the mesh SWIM/CRDT layer
/// owns convergence, so no store snapshot is needed here.
fn apply_router_event(
    event: &Event<Pod>,
    config: &ServiceDiscoveryConfig,
    cluster_state: &ClusterState,
    default_mesh_port: u16,
) {
    let (pod, is_delete) = match event {
        Event::Apply(pod) | Event::InitApply(pod) => (pod, false),
        Event::Delete(pod) => (pod, true),
        Event::Init | Event::InitDone => return,
    };

    if !PodInfo::matches_selector(pod, &config.router_selector) {
        return;
    }
    let Some(pod_info) = PodInfo::from_pod(pod, Some(config)) else {
        return;
    };
    if !pod_info.is_router {
        return;
    }

    if is_delete || pod.metadata.deletion_timestamp.is_some() {
        let mut state = cluster_state.write();
        if let Some(node) = state.get_mut(&pod_info.name) {
            node.status = NodeStatus::Down as i32;
            node.version += 1;
            info!("Router node {} marked as Down (pod deleted)", pod_info.name);
        } else {
            debug!(
                "Router node {} not found in cluster state (already removed)",
                pod_info.name
            );
        }
    } else if pod_info.is_healthy() {
        let mesh_port = pod_info.mesh_port.unwrap_or(default_mesh_port);
        let node_address = format!("{}:{}", pod_info.ip, mesh_port);
        let mut state = cluster_state.write();
        let existing_version = state.get(&pod_info.name).map(|n| n.version).unwrap_or(0);

        let node_state = NodeState {
            name: pod_info.name.clone(),
            address: node_address,
            status: NodeStatus::Alive as i32,
            version: existing_version + 1,
            metadata: HashMap::new(),
        };

        state.insert(pod_info.name.clone(), node_state.clone());
        info!(
            "Router node {} added/updated in mesh cluster (address: {})",
            pod_info.name, node_state.address
        );
    } else {
        let mut state = cluster_state.write();
        if let Some(node) = state.get_mut(&pod_info.name) {
            if node.status != NodeStatus::Down as i32 {
                node.status = NodeStatus::Suspected as i32;
                node.version += 1;
                debug!(
                    "Router node {} marked as Suspected (pod not healthy)",
                    pod_info.name
                );
            }
        }
    }
}

/// Start router node discovery for mesh cluster
async fn start_router_discovery(
    config: Arc<ServiceDiscoveryConfig>,
    pods: Api<Pod>,
    cluster_state: ClusterState,
    default_mesh_port: u16,
) {
    let watcher_config = build_watcher_config("router", &config.router_label_selector());
    let mut stream = watcher(pods, watcher_config).default_backoff().boxed();

    while let Some(item) = stream.next().await {
        match item {
            Ok(event) => apply_router_event(&event, &config, &cluster_state, default_mesh_port),
            Err(e) => error!("Router watcher error (auto-retrying with backoff): {e}"),
        }
    }

    error!("K8s router watcher stream ended; router discovery no longer receives updates");
}

#[cfg(test)]
mod tests {
    use k8s_openapi::{
        api::core::v1::{Pod, PodCondition, PodSpec, PodStatus},
        apimachinery::pkg::apis::meta::v1::{ObjectMeta, Time},
    };
    use tracing_test::traced_test;

    use super::*;
    use crate::routers::{common::openai_bridge, grpc::multimodal::MultimodalConfigRegistry};

    fn create_k8s_pod(
        name: Option<&str>,
        ip: Option<&str>,
        phase: Option<&str>,
        ready_status: Option<&str>,
        deletion_timestamp: Option<Time>,
    ) -> Pod {
        let mut pod = Pod {
            metadata: ObjectMeta {
                name: name.map(String::from),
                uid: name.map(|n| format!("uid-{n}")),
                deletion_timestamp,
                ..Default::default()
            },
            spec: Some(PodSpec::default()),
            status: None,
        };

        if ip.is_some() || phase.is_some() || ready_status.is_some() {
            let mut pod_status = PodStatus {
                pod_ip: ip.map(String::from),
                phase: phase.map(String::from),
                conditions: None,
                ..Default::default()
            };

            if let Some(status_str) = ready_status {
                let condition = PodCondition {
                    type_: "Ready".to_string(),
                    status: status_str.to_string(),
                    last_probe_time: None,
                    last_transition_time: None,
                    message: None,
                    reason: None,
                    observed_generation: None,
                };
                pod_status.conditions = Some(vec![condition]);
            }
            pod.status = Some(pod_status);
        }
        pod
    }

    fn create_pd_k8s_pod(name: &str, ip: &str, pod_type: &str, bootstrap_port: Option<u16>) -> Pod {
        let mut labels = std::collections::BTreeMap::new();
        labels.insert("app".to_string(), "sglang".to_string());
        labels.insert("component".to_string(), pod_type.to_string());

        let mut annotations = std::collections::BTreeMap::new();
        if let Some(port) = bootstrap_port {
            annotations.insert("sglang.ai/bootstrap-port".to_string(), port.to_string());
        }

        Pod {
            metadata: ObjectMeta {
                name: Some(name.to_string()),
                uid: Some(format!("uid-{name}")),
                labels: Some(labels),
                annotations: Some(annotations),
                ..Default::default()
            },
            spec: Some(PodSpec::default()),
            status: Some(PodStatus {
                pod_ip: Some(ip.to_string()),
                phase: Some("Running".to_string()),
                conditions: Some(vec![PodCondition {
                    type_: "Ready".to_string(),
                    status: "True".to_string(),
                    last_probe_time: None,
                    last_transition_time: None,
                    message: None,
                    reason: None,
                    observed_generation: None,
                }]),
                ..Default::default()
            }),
        }
    }

    fn create_test_app_context() -> Arc<AppContext> {
        use crate::{
            config::RouterConfig, middleware::TokenBucket,
            observability::inflight_tracker::InFlightRequestTracker,
            routers::common::realtime::RealtimeRegistry, worker::WorkerService,
        };

        let router_config = RouterConfig::builder()
            .worker_startup_timeout_secs(1)
            .build_unchecked();

        let worker_registry = Arc::new(crate::worker::WorkerRegistry::new());
        let worker_job_queue = Arc::new(std::sync::OnceLock::new());

        // Note: Using uninitialized queue for tests to avoid spawning background workers
        // Jobs submitted during tests will queue but not be processed
        Arc::new(AppContext {
            client: reqwest::Client::new(),
            router_config: router_config.clone(),
            rate_limiter: Some(Arc::new(TokenBucket::new(1000, 1000))),
            rate_limit_manager: None,
            worker_registry: worker_registry.clone(),
            policy_registry: Arc::new(crate::policies::PolicyRegistry::with_override(
                router_config.policy.clone(),
                router_config.routing_key_override.clone(),
            )),
            reasoning_parser_factory: None,
            tool_parser_factory: None,
            router_manager: None,
            response_storage: Arc::new(smg_data_connector::MemoryResponseStorage::new()),
            conversation_storage: Arc::new(smg_data_connector::MemoryConversationStorage::new()),
            conversation_item_storage: Arc::new(
                smg_data_connector::MemoryConversationItemStorage::new(),
            ),
            worker_monitor: None,
            configured_reasoning_parser: None,
            configured_tool_parser: None,
            worker_job_queue: worker_job_queue.clone(),
            workflow_engines: Arc::new(std::sync::OnceLock::new()),
            mcp_orchestrator: Arc::new(std::sync::OnceLock::new()),
            mcp_format_registry: openai_bridge::FormatRegistry::new(),
            tokenizer_registry: Arc::new(llm_tokenizer::registry::TokenizerRegistry::new()),
            multimodal_config_registry: Arc::new(MultimodalConfigRegistry::new()),
            wasm_manager: None,
            worker_client_cache: Arc::new(crate::worker::WorkerHttpClientCache::new(
                &router_config,
            )),
            worker_service: Arc::new(WorkerService::new(
                worker_registry,
                worker_job_queue,
                router_config,
            )),
            inflight_tracker: InFlightRequestTracker::new(),
            kv_event_monitor: None,
            realtime_registry: Arc::new(RealtimeRegistry::new()),
            webrtc_bind_addr: None,
            webrtc_stun_server: None,
        })
    }

    fn create_pd_config() -> ServiceDiscoveryConfig {
        let mut prefill_selector = HashMap::new();
        prefill_selector.insert("app".to_string(), "sglang".to_string());
        prefill_selector.insert("component".to_string(), "prefill".to_string());

        let mut decode_selector = HashMap::new();
        decode_selector.insert("app".to_string(), "sglang".to_string());
        decode_selector.insert("component".to_string(), "decode".to_string());

        ServiceDiscoveryConfig {
            enabled: true,
            selector: HashMap::new(),
            check_interval: Duration::from_secs(60),
            port: 8080,
            namespace: None,
            disaggregated_mode: true,
            encode_selector: HashMap::new(),
            prefill_selector,
            decode_selector,
            bootstrap_port_annotation: "sglang.ai/bootstrap-port".to_string(),
            worker_ports_annotation: "smg.ai/worker-ports".to_string(),
            router_selector: HashMap::new(),
            router_mesh_port_annotation: "sglang.ai/mesh-port".to_string(),
            model_id_source: None,
        }
    }

    fn create_epd_config() -> ServiceDiscoveryConfig {
        let mut config = create_pd_config();
        config
            .encode_selector
            .insert("app".to_string(), "sglang".to_string());
        config
            .encode_selector
            .insert("component".to_string(), "encode".to_string());
        config
    }

    #[test]
    fn test_pod_info_should_include() {
        let config = create_pd_config();

        let prefill_pod = create_pd_k8s_pod("prefill-pod", "10.0.0.1", "prefill", Some(8081));
        assert!(PodInfo::should_include(&prefill_pod, &config));

        let decode_pod = create_pd_k8s_pod("decode-pod", "10.0.0.2", "decode", None);
        assert!(PodInfo::should_include(&decode_pod, &config));

        let unmatched_pod = create_pd_k8s_pod("other-pod", "10.0.0.3", "other", None);
        assert!(!PodInfo::should_include(&unmatched_pod, &config));

        let mut regular_config = ServiceDiscoveryConfig::default();
        regular_config
            .selector
            .insert("app".to_string(), "sglang".to_string());
        regular_config.disaggregated_mode = false;

        let regular_pod = create_pd_k8s_pod("worker-pod", "10.0.0.4", "worker", None);
        assert!(PodInfo::should_include(&regular_pod, &regular_config));
    }

    #[test]
    fn test_pod_info_should_include_epd_encode_pod() {
        let config = create_epd_config();

        let encode_pod = create_pd_k8s_pod("encode-pod", "10.0.0.5", "encode", Some(8091));
        assert!(PodInfo::should_include(&encode_pod, &config));

        let prefill_pod = create_pd_k8s_pod("prefill-pod", "10.0.0.1", "prefill", Some(8081));
        assert!(PodInfo::should_include(&prefill_pod, &config));

        let decode_pod = create_pd_k8s_pod("decode-pod", "10.0.0.2", "decode", None);
        assert!(PodInfo::should_include(&decode_pod, &config));
    }

    #[test]
    fn test_service_discovery_config_default() {
        let config = ServiceDiscoveryConfig::default();
        assert!(!config.enabled);
        assert!(config.selector.is_empty());
        assert_eq!(config.check_interval, Duration::from_secs(60));
        assert_eq!(config.port, 8000);
        assert!(config.namespace.is_none());
        assert!(!config.disaggregated_mode);
        assert!(config.encode_selector.is_empty());
        assert!(config.prefill_selector.is_empty());
        assert!(config.decode_selector.is_empty());
        assert_eq!(config.bootstrap_port_annotation, "sglang.ai/bootstrap-port");
        assert_eq!(config.worker_ports_annotation, "smg.ai/worker-ports");
    }

    #[test]
    fn test_pod_type_enum() {
        let encode = PodType::Encode;
        let prefill = PodType::Prefill;
        let decode = PodType::Decode;
        let regular = PodType::Regular;

        assert_eq!(format!("{encode:?}"), "Encode");
        assert_eq!(format!("{prefill:?}"), "Prefill");
        assert_eq!(format!("{decode:?}"), "Decode");
        assert_eq!(format!("{regular:?}"), "Regular");
    }

    #[test]
    fn test_pod_info_from_pod_valid() {
        let k8s_pod = create_k8s_pod(
            Some("test-pod"),
            Some("10.0.0.1"),
            Some("Running"),
            Some("True"),
            None,
        );
        let pod_info = PodInfo::from_pod(&k8s_pod, None).unwrap();
        assert_eq!(pod_info.name, "test-pod");
        assert_eq!(pod_info.ip, "10.0.0.1");
        assert_eq!(pod_info.status, "Running");
        assert!(pod_info.is_ready);
        assert!(pod_info.pod_type.is_none());
        assert!(pod_info.ports.is_empty());
        assert!(pod_info.bootstrap_ports.is_empty());
    }

    #[test]
    fn test_pod_info_from_pod_with_pd_config_prefill() {
        let k8s_pod = create_pd_k8s_pod("prefill-pod", "10.0.0.1", "prefill", Some(8081));
        let config = create_pd_config();

        let pod_info = PodInfo::from_pod(&k8s_pod, Some(&config)).unwrap();
        assert_eq!(pod_info.name, "prefill-pod");
        assert_eq!(pod_info.ip, "10.0.0.1");
        assert_eq!(pod_info.status, "Running");
        assert!(pod_info.is_ready);
        assert_eq!(pod_info.pod_type, Some(PodType::Prefill));
        assert_eq!(pod_info.ports, vec![8080]);
        assert_eq!(pod_info.bootstrap_ports, vec![Some(8081)]);
    }

    #[test]
    fn test_pod_info_from_pod_with_epd_config_encode() {
        let k8s_pod = create_pd_k8s_pod("encode-pod", "10.0.0.5", "encode", Some(8091));
        let config = create_epd_config();

        let pod_info = PodInfo::from_pod(&k8s_pod, Some(&config)).unwrap();
        assert_eq!(pod_info.name, "encode-pod");
        assert_eq!(pod_info.ip, "10.0.0.5");
        assert_eq!(pod_info.status, "Running");
        assert!(pod_info.is_ready);
        assert_eq!(pod_info.pod_type, Some(PodType::Encode));
        assert_eq!(pod_info.ports, vec![8080]);
        assert_eq!(pod_info.bootstrap_ports, vec![Some(8091)]);
    }

    #[test]
    fn test_pod_info_from_pod_with_pd_config_decode() {
        let k8s_pod = create_pd_k8s_pod("decode-pod", "10.0.0.2", "decode", None);
        let config = create_pd_config();

        let pod_info = PodInfo::from_pod(&k8s_pod, Some(&config)).unwrap();
        assert_eq!(pod_info.name, "decode-pod");
        assert_eq!(pod_info.ip, "10.0.0.2");
        assert_eq!(pod_info.status, "Running");
        assert!(pod_info.is_ready);
        assert_eq!(pod_info.pod_type, Some(PodType::Decode));
        assert_eq!(pod_info.bootstrap_ports, vec![None]);
    }

    #[test]
    fn test_pod_info_from_pod_with_pd_config_regular_mode() {
        let k8s_pod = create_pd_k8s_pod("regular-pod", "10.0.0.3", "worker", None);
        let mut config = create_pd_config();
        config.disaggregated_mode = false;

        let pod_info = PodInfo::from_pod(&k8s_pod, Some(&config)).unwrap();
        assert_eq!(pod_info.name, "regular-pod");
        assert_eq!(pod_info.ip, "10.0.0.3");
        assert_eq!(pod_info.status, "Running");
        assert!(pod_info.is_ready);
        assert_eq!(pod_info.pod_type, Some(PodType::Regular));
        assert_eq!(pod_info.bootstrap_ports, vec![None]);
    }

    #[test]
    fn test_pod_info_from_pod_with_pd_config_unmatched_labels() {
        let k8s_pod = create_pd_k8s_pod("unknown-pod", "10.0.0.4", "unknown", None);
        let config = create_pd_config();

        let pod_info = PodInfo::from_pod(&k8s_pod, Some(&config)).unwrap();
        assert_eq!(pod_info.name, "unknown-pod");
        assert_eq!(pod_info.ip, "10.0.0.4");
        assert_eq!(pod_info.status, "Running");
        assert!(pod_info.is_ready);
        assert_eq!(pod_info.pod_type, Some(PodType::Regular));
        assert_eq!(pod_info.bootstrap_ports, vec![None]);
    }

    #[test]
    fn test_pod_info_from_pod_with_pd_config_invalid_bootstrap_port() {
        let mut pod = create_pd_k8s_pod("prefill-pod", "10.0.0.1", "prefill", None);
        pod.metadata.annotations.as_mut().unwrap().insert(
            "sglang.ai/bootstrap-port".to_string(),
            "invalid".to_string(),
        );
        let config = create_pd_config();

        let pod_info = PodInfo::from_pod(&pod, Some(&config)).unwrap();
        assert_eq!(pod_info.pod_type, Some(PodType::Prefill));
        assert_eq!(pod_info.bootstrap_ports, vec![None]);
    }

    #[test]
    fn test_pod_info_from_pod_not_ready() {
        let k8s_pod = create_k8s_pod(
            Some("test-pod"),
            Some("10.0.0.1"),
            Some("Running"),
            Some("False"),
            None,
        );
        let pod_info = PodInfo::from_pod(&k8s_pod, None).unwrap();
        assert!(!pod_info.is_ready);
    }

    #[test]
    fn test_pod_info_from_pod_no_conditions() {
        let k8s_pod = create_k8s_pod(
            Some("test-pod"),
            Some("10.0.0.1"),
            Some("Running"),
            None,
            None,
        );
        let pod_info = PodInfo::from_pod(&k8s_pod, None).unwrap();
        assert!(!pod_info.is_ready);
    }

    #[test]
    fn test_pod_info_from_pod_missing_name() {
        let k8s_pod = create_k8s_pod(None, Some("10.0.0.1"), Some("Running"), Some("True"), None);
        assert!(PodInfo::from_pod(&k8s_pod, None).is_none());
    }

    #[test]
    fn test_pod_info_from_pod_missing_ip() {
        let k8s_pod = create_k8s_pod(Some("test-pod"), None, Some("Running"), Some("True"), None);
        assert!(PodInfo::from_pod(&k8s_pod, None).is_none());
    }

    #[test]
    fn test_pod_info_from_pod_missing_status_phase() {
        let k8s_pod = create_k8s_pod(Some("test-pod"), Some("10.0.0.1"), None, Some("True"), None);
        let pod_info = PodInfo::from_pod(&k8s_pod, None).unwrap();
        assert_eq!(pod_info.status, "Unknown");
    }

    #[test]
    fn test_pod_info_from_pod_no_status_object() {
        let mut k8s_pod = create_k8s_pod(Some("test-pod"), None, None, None, None);
        k8s_pod.status = None;
        assert!(PodInfo::from_pod(&k8s_pod, None).is_none());
    }

    #[test]
    fn test_pod_info_is_healthy() {
        let healthy_pod = PodInfo {
            name: "p1".into(),
            uid: "uid-p1".into(),
            ip: "1.1.1.1".into(),
            status: "Running".into(),
            is_ready: true,
            pod_type: None,
            ports: vec![],
            bootstrap_ports: vec![],
            is_router: false,
            mesh_port: None,
            model_id_override: None,
        };
        assert!(healthy_pod.is_healthy());

        let not_ready_pod = PodInfo {
            name: "p2".into(),
            uid: "uid-p2".into(),
            ip: "1.1.1.2".into(),
            status: "Running".into(),
            is_ready: false,
            pod_type: None,
            ports: vec![],
            bootstrap_ports: vec![],
            is_router: false,
            mesh_port: None,
            model_id_override: None,
        };
        assert!(!not_ready_pod.is_healthy());

        let not_running_pod = PodInfo {
            name: "p3".into(),
            uid: "uid-p3".into(),
            ip: "1.1.1.3".into(),
            status: "Pending".into(),
            is_ready: true,
            pod_type: None,
            ports: vec![],
            bootstrap_ports: vec![],
            is_router: false,
            mesh_port: None,
            model_id_override: None,
        };
        assert!(!not_running_pod.is_healthy());
    }

    // ========== Port annotation parsing ==========

    #[test]
    fn test_parse_port_list() {
        assert_eq!(parse_port_list("8080"), Some(vec![8080]));
        assert_eq!(
            parse_port_list(" 8080, 8081 ,8082"),
            Some(vec![8080, 8081, 8082])
        );
        assert_eq!(parse_port_list("8080,abc"), None);
        assert_eq!(parse_port_list("0"), None);
        assert_eq!(parse_port_list(""), None);
        assert_eq!(parse_port_list("70000"), None);
    }

    fn pod_with_annotations(name: &str, annotations: &[(&str, &str)]) -> Pod {
        let mut pod = make_labeled_pod(name, "10.0.0.1", &[("app", "sglang")]);
        pod.metadata.annotations = Some(
            annotations
                .iter()
                .map(|&(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        );
        pod
    }

    #[test]
    fn test_from_pod_ports_default_to_config_port() {
        let config = make_regular_config();
        let pod = make_labeled_pod("w", "10.0.0.1", &[("app", "sglang")]);
        let info = PodInfo::from_pod(&pod, Some(&config)).unwrap();
        assert_eq!(info.ports, vec![config.port]);
        assert_eq!(info.bootstrap_ports, vec![None]);
    }

    #[test]
    fn test_from_pod_ports_from_annotation() {
        let config = make_regular_config();
        let pod = pod_with_annotations("w", &[("smg.ai/worker-ports", "8080,8081,8082,8083")]);
        let info = PodInfo::from_pod(&pod, Some(&config)).unwrap();
        assert_eq!(info.ports, vec![8080, 8081, 8082, 8083]);
        assert_eq!(info.bootstrap_ports, vec![None; 4]);
    }

    #[test]
    fn test_from_pod_ports_annotation_dedupes_preserving_order() {
        let config = make_regular_config();
        let pod = pod_with_annotations("w", &[("smg.ai/worker-ports", "8081,8080,8081")]);
        let info = PodInfo::from_pod(&pod, Some(&config)).unwrap();
        assert_eq!(info.ports, vec![8081, 8080]);
    }

    #[test]
    fn test_from_pod_invalid_ports_annotation_falls_back_to_config_port() {
        let config = make_regular_config();
        let pod = pod_with_annotations("w", &[("smg.ai/worker-ports", "8080,nope")]);
        let info = PodInfo::from_pod(&pod, Some(&config)).unwrap();
        assert_eq!(info.ports, vec![config.port]);
    }

    #[test]
    fn test_from_pod_bootstrap_broadcasts_to_all_ports() {
        let config = create_pd_config();
        let mut pod = create_pd_k8s_pod("prefill-0", "10.0.0.1", "prefill", Some(9080));
        pod.metadata
            .annotations
            .as_mut()
            .unwrap()
            .insert("smg.ai/worker-ports".to_string(), "8080,8081".to_string());
        let info = PodInfo::from_pod(&pod, Some(&config)).unwrap();
        assert_eq!(info.ports, vec![8080, 8081]);
        assert_eq!(info.bootstrap_ports, vec![Some(9080), Some(9080)]);
    }

    #[test]
    fn test_from_pod_bootstrap_list_zips_with_ports() {
        let config = create_pd_config();
        let mut pod = create_pd_k8s_pod("prefill-0", "10.0.0.1", "prefill", None);
        let annotations = pod.metadata.annotations.as_mut().unwrap();
        annotations.insert("smg.ai/worker-ports".to_string(), "8080,8081".to_string());
        annotations.insert(
            "sglang.ai/bootstrap-port".to_string(),
            "9080,9081".to_string(),
        );
        let info = PodInfo::from_pod(&pod, Some(&config)).unwrap();
        assert_eq!(info.bootstrap_ports, vec![Some(9080), Some(9081)]);
    }

    #[test]
    fn test_from_pod_bootstrap_count_mismatch_ignored() {
        let config = create_pd_config();
        let mut pod = create_pd_k8s_pod("prefill-0", "10.0.0.1", "prefill", None);
        let annotations = pod.metadata.annotations.as_mut().unwrap();
        annotations.insert("smg.ai/worker-ports".to_string(), "8080,8081".to_string());
        annotations.insert(
            "sglang.ai/bootstrap-port".to_string(),
            "9080,9081,9082".to_string(),
        );
        let info = PodInfo::from_pod(&pod, Some(&config)).unwrap();
        assert_eq!(info.bootstrap_ports, vec![None, None]);
    }

    // ========== Desired state ==========

    fn store_snapshot(pods: Vec<Pod>) -> Vec<Arc<Pod>> {
        pods.into_iter().map(Arc::new).collect()
    }

    #[test]
    fn test_compute_desired_state_multi_port_pod() {
        let config = make_regular_config();
        let pod = pod_with_annotations("w", &[("smg.ai/worker-ports", "8080,8081")]);
        let desired = compute_desired_state(&store_snapshot(vec![pod]), &config);

        assert_eq!(desired.uid_by_url.len(), 2);
        assert_eq!(
            desired.uid_by_url.get("10.0.0.1:8080"),
            Some(&"uid-w".to_string())
        );
        assert_eq!(
            desired.uid_by_url.get("10.0.0.1:8081"),
            Some(&"uid-w".to_string())
        );
        assert_eq!(desired.addable.len(), 2);
        assert!(desired.addable.iter().all(|w| w.pod_uid == "uid-w"));
        assert!(desired.addable.iter().all(|w| w.pod_name == "w"));
    }

    #[test]
    fn test_compute_desired_state_terminating_pod_excluded() {
        let config = make_regular_config();
        let mut pod = make_labeled_pod("w", "10.0.0.1", &[("app", "sglang")]);
        pod.metadata.deletion_timestamp = Some(Time(k8s_openapi::jiff::Timestamp::now()));
        let desired = compute_desired_state(&store_snapshot(vec![pod]), &config);
        assert!(desired.uid_by_url.is_empty());
        assert!(desired.addable.is_empty());
    }

    #[test]
    fn test_compute_desired_state_unready_pod_excluded() {
        let config = make_regular_config();
        let mut pod = pod_with_annotations("w", &[("smg.ai/worker-ports", "8080,8081")]);
        if let Some(status) = pod.status.as_mut() {
            status.conditions = Some(vec![PodCondition {
                type_: "Ready".to_string(),
                status: "False".to_string(),
                last_probe_time: None,
                last_transition_time: None,
                message: None,
                reason: None,
                observed_generation: None,
            }]);
        }
        let desired = compute_desired_state(&store_snapshot(vec![pod]), &config);
        assert!(desired.uid_by_url.is_empty());
        assert!(desired.addable.is_empty());
    }

    #[test]
    fn test_compute_desired_state_unknown_readiness_excluded() {
        let config = make_regular_config();
        let mut pod = make_labeled_pod("w", "10.0.0.1", &[("app", "sglang")]);
        if let Some(status) = pod.status.as_mut() {
            status.conditions = Some(vec![PodCondition {
                type_: "Ready".to_string(),
                status: "Unknown".to_string(),
                last_probe_time: None,
                last_transition_time: None,
                message: None,
                reason: None,
                observed_generation: None,
            }]);
        }
        let desired = compute_desired_state(&store_snapshot(vec![pod]), &config);
        assert!(desired.uid_by_url.is_empty());
        assert!(desired.addable.is_empty());
    }

    #[test]
    fn test_compute_desired_state_ignores_non_matching_pods() {
        let config = make_regular_config();
        let pod = make_labeled_pod("w", "10.0.0.1", &[("app", "other")]);
        let desired = compute_desired_state(&store_snapshot(vec![pod]), &config);
        assert!(desired.uid_by_url.is_empty());
        assert!(desired.addable.is_empty());
    }

    #[test]
    fn test_compute_desired_state_pd_prefill_bootstrap_alignment() {
        let config = create_pd_config();
        let mut pod = create_pd_k8s_pod("prefill-0", "10.0.0.1", "prefill", None);
        let annotations = pod.metadata.annotations.as_mut().unwrap();
        annotations.insert("smg.ai/worker-ports".to_string(), "8080,8081".to_string());
        annotations.insert(
            "sglang.ai/bootstrap-port".to_string(),
            "9080,9081".to_string(),
        );

        let desired = compute_desired_state(&store_snapshot(vec![pod]), &config);
        assert_eq!(desired.addable.len(), 2);
        let by_url: HashMap<&str, &DesiredWorker> = desired
            .addable
            .iter()
            .map(|w| (w.url.as_str(), w))
            .collect();
        let first = by_url["10.0.0.1:8080"];
        assert_eq!(first.worker_type, WorkerType::Prefill);
        assert_eq!(first.bootstrap_port, Some(9080));
        let second = by_url["10.0.0.1:8081"];
        assert_eq!(second.bootstrap_port, Some(9081));
    }

    #[test]
    fn test_compute_desired_state_carries_model_id_override() {
        let mut config = make_regular_config();
        config.model_id_source = Some(ModelIdSource::Namespace);
        let mut pod = make_labeled_pod("w", "10.0.0.1", &[("app", "sglang")]);
        pod.metadata.namespace = Some("team-a".to_string());
        let desired = compute_desired_state(&store_snapshot(vec![pod]), &config);
        assert_eq!(
            desired.addable[0].model_id_override,
            Some("team-a".to_string())
        );
    }

    #[test]
    fn test_desired_state_from_reflector_store() {
        let (store, mut writer) = reflector::store::<Pod>();
        let pod = pod_with_annotations("w", &[("smg.ai/worker-ports", "8080,8081")]);
        writer.apply_watcher_event(&Event::Init);
        writer.apply_watcher_event(&Event::InitApply(pod));
        writer.apply_watcher_event(&Event::InitDone);

        let config = make_regular_config();
        let desired = compute_desired_state(&store.state(), &config);
        assert_eq!(desired.uid_by_url.len(), 2);
    }

    // ========== Registry diff ==========

    #[test]
    fn test_canonical_host_port() {
        assert_eq!(canonical_host_port("10.0.0.1:8080"), "10.0.0.1:8080");
        assert_eq!(canonical_host_port("http://10.0.0.1:8080"), "10.0.0.1:8080");
        assert_eq!(
            canonical_host_port("grpc://10.0.0.1:8080@2"),
            "10.0.0.1:8080"
        );
        assert_eq!(canonical_host_port("10.0.0.1:8080@0"), "10.0.0.1:8080");
    }

    fn desired_worker(url: &str, uid: &str) -> DesiredWorker {
        DesiredWorker {
            url: url.to_string(),
            worker_type: WorkerType::Regular,
            bootstrap_port: None,
            pod_name: "w".to_string(),
            pod_uid: uid.to_string(),
            model_id_override: None,
        }
    }

    fn desired_state_of(workers: &[DesiredWorker]) -> DesiredState {
        let mut state = DesiredState::default();
        for worker in workers {
            state
                .uid_by_url
                .insert(worker.url.clone(), worker.pod_uid.clone());
            state.addable.push(worker.clone());
        }
        state
    }

    fn owned(url: &str, uid: &str) -> OwnedWorker {
        OwnedWorker {
            url: url.to_string(),
            pod_uid: uid.to_string(),
            revision: 1,
        }
    }

    #[test]
    fn test_compute_actions_adds_missing_workers() {
        let desired = desired_state_of(&[
            desired_worker("10.0.0.1:8080", "u1"),
            desired_worker("10.0.0.1:8081", "u1"),
        ]);
        let actions = compute_actions(&desired, &[]);
        assert_eq!(actions.add.len(), 2);
        assert!(actions.remove.is_empty());
    }

    #[test]
    fn test_compute_actions_noop_when_converged() {
        let desired = desired_state_of(&[desired_worker("10.0.0.1:8080", "u1")]);
        let registered = [owned("10.0.0.1:8080", "u1")];
        let actions = compute_actions(&desired, &registered);
        assert!(actions.add.is_empty());
        assert!(actions.remove.is_empty());
    }

    #[test]
    fn test_compute_actions_removes_workers_for_gone_pods() {
        let desired = desired_state_of(&[desired_worker("10.0.0.1:8080", "u1")]);
        let registered = [owned("10.0.0.1:8080", "u1"), owned("10.0.0.2:8080", "u2")];
        let actions = compute_actions(&desired, &registered);
        assert!(actions.add.is_empty());
        assert_eq!(actions.remove.len(), 1);
        assert_eq!(actions.remove[0].url, "10.0.0.2:8080");
    }

    #[test]
    fn test_compute_actions_uid_change_removes_and_reregisters_same_url() {
        // Same-IP pod restart (hostNetwork / stable IP): URL unchanged but
        // uid differs → the stale worker is removed (covers a scheme-flipped
        // sibling the Upsert cannot replace) and the new one registered.
        let desired = desired_state_of(&[desired_worker("10.0.0.1:8080", "uid-new")]);
        let registered = [owned("10.0.0.1:8080", "uid-old")];
        let actions = compute_actions(&desired, &registered);
        assert_eq!(actions.add.len(), 1);
        assert_eq!(actions.add[0].pod_uid, "uid-new");
        assert_eq!(actions.remove.len(), 1);
        assert_eq!(actions.remove[0].pod_uid, "uid-old");
    }

    #[test]
    fn test_compute_actions_unready_pod_removes_registered_workers() {
        let config = make_regular_config();
        let mut pod = pod_with_annotations("w", &[("smg.ai/worker-ports", "8080,8081")]);
        if let Some(status) = pod.status.as_mut() {
            status.conditions = Some(vec![PodCondition {
                type_: "Ready".to_string(),
                status: "False".to_string(),
                last_probe_time: None,
                last_transition_time: None,
                message: None,
                reason: None,
                observed_generation: None,
            }]);
        }
        let desired = compute_desired_state(&store_snapshot(vec![pod]), &config);
        let registered = [
            owned("10.0.0.1:8080", "uid-w"),
            owned("10.0.0.1:8081", "uid-w"),
        ];
        let actions = compute_actions(&desired, &registered);
        assert!(actions.add.is_empty());
        assert_eq!(actions.remove.len(), 2);
        assert_eq!(actions.remove[0].pod_uid, "uid-w");
        assert_eq!(actions.remove[1].pod_uid, "uid-w");
    }

    #[test]
    fn test_compute_actions_dp_ranks_removed_once() {
        let registered = [owned("10.0.0.1:8080", "u1"), owned("10.0.0.1:8080", "u1")];
        let actions = compute_actions(&DesiredState::default(), &registered);
        assert_eq!(actions.remove.len(), 1);
        assert_eq!(actions.remove[0].url, "10.0.0.1:8080");
        assert_eq!(actions.remove[0].revision, 1);
        assert!(actions.add.is_empty());
    }

    #[test]
    fn test_k8s_owned_workers_scoped_by_label_and_origin() {
        use openai_protocol::model_card::ModelCard;

        use crate::worker::BasicWorkerBuilder;

        let app_context = create_test_app_context();

        let mut labels = HashMap::new();
        labels.insert(POD_UID_LABEL.to_string(), "uid-1".to_string());
        let discovered = Arc::new(
            BasicWorkerBuilder::new("http://10.0.0.1:8080")
                .model(ModelCard::new("m"))
                .labels(labels)
                .build(),
        );
        let manual = Arc::new(
            BasicWorkerBuilder::new("http://10.0.0.2:8080")
                .model(ModelCard::new("m"))
                .build(),
        );
        app_context.worker_registry.register(discovered).unwrap();
        app_context.worker_registry.register(manual).unwrap();

        let owned = k8s_owned_workers(&app_context);
        assert_eq!(owned.len(), 1);
        assert_eq!(owned[0].url, "10.0.0.1:8080");
        assert_eq!(owned[0].pod_uid, "uid-1");
    }

    #[test]
    fn test_k8s_owned_workers_excludes_mesh_imported_workers() {
        // A peer's discovered worker arrives via mesh sync carrying the
        // pod-uid label; its pod is absent from this node's store, so without
        // the Local-origin filter the reconciler would remove it every pass.
        let app_context = create_test_app_context();

        let mut spec = WorkerSpec::new("http://10.0.0.3:8080");
        spec.labels
            .insert(POD_UID_LABEL.to_string(), "uid-mesh".to_string());
        let state = smg_mesh::WorkerState {
            worker_id: "peer-w1".to_string(),
            model_id: "m".to_string(),
            url: "http://10.0.0.3:8080".to_string(),
            health: true,
            load: 0.0,
            version: 0,
            spec: serde_json::to_vec(&spec).unwrap(),
        };
        app_context.worker_registry.on_remote_worker_state(&state);
        assert!(app_context
            .worker_registry
            .get_by_url("http://10.0.0.3:8080")
            .is_some());

        assert!(k8s_owned_workers(&app_context).is_empty());
    }

    #[test]
    fn test_build_worker_spec_stamps_ownership_labels() {
        let app_context = create_test_app_context();
        let desired = DesiredWorker {
            url: "10.0.0.1:8081".to_string(),
            worker_type: WorkerType::Prefill,
            bootstrap_port: Some(9080),
            pod_name: "prefill-0".to_string(),
            pod_uid: "uid-1".to_string(),
            model_id_override: Some("llama".to_string()),
        };
        let spec = build_worker_spec(&desired, &app_context);
        assert_eq!(spec.url, "10.0.0.1:8081");
        assert_eq!(spec.worker_type, WorkerType::Prefill);
        assert_eq!(spec.bootstrap_port, Some(9080));
        assert_eq!(
            spec.labels.get(POD_NAME_LABEL),
            Some(&"prefill-0".to_string())
        );
        assert_eq!(spec.labels.get(POD_UID_LABEL), Some(&"uid-1".to_string()));
        assert_eq!(
            spec.labels.get("served_model_name"),
            Some(&"llama".to_string())
        );
    }

    #[tokio::test]
    async fn test_reconcile_workers_without_job_queue_is_safe() {
        let app_context = create_test_app_context();
        let (store, mut writer) = reflector::store::<Pod>();
        writer.apply_watcher_event(&Event::Init);
        writer.apply_watcher_event(&Event::InitApply(make_labeled_pod(
            "w",
            "10.0.0.1",
            &[("app", "sglang")],
        )));
        writer.apply_watcher_event(&Event::InitDone);
        let config = make_regular_config();
        reconcile_workers(&store, &config, &app_context).await;
    }

    // ========== Router events ==========

    fn router_config_and_pod() -> (ServiceDiscoveryConfig, Pod) {
        let mut router_selector = HashMap::new();
        router_selector.insert("role".to_string(), "router".to_string());
        let config = ServiceDiscoveryConfig {
            enabled: true,
            router_selector,
            ..Default::default()
        };
        let mut pod = make_labeled_pod("r1", "10.1.0.1", &[("role", "router")]);
        pod.metadata.annotations = Some(
            [("sglang.ai/mesh-port".to_string(), "7100".to_string())]
                .into_iter()
                .collect(),
        );
        (config, pod)
    }

    #[test]
    fn test_apply_router_event_apply_then_delete() {
        let (config, pod) = router_config_and_pod();
        let cluster_state: ClusterState = Arc::default();

        apply_router_event(&Event::Apply(pod.clone()), &config, &cluster_state, 7000);
        {
            let state = cluster_state.read();
            let node = state.get("r1").unwrap();
            assert_eq!(node.status, NodeStatus::Alive as i32);
            assert_eq!(node.address, "10.1.0.1:7100");
        }

        apply_router_event(&Event::Delete(pod), &config, &cluster_state, 7000);
        assert_eq!(
            cluster_state.read().get("r1").unwrap().status,
            NodeStatus::Down as i32
        );
    }

    #[test]
    fn test_apply_router_event_terminating_marks_down() {
        let (config, mut pod) = router_config_and_pod();
        let cluster_state: ClusterState = Arc::default();

        apply_router_event(&Event::Apply(pod.clone()), &config, &cluster_state, 7000);
        pod.metadata.deletion_timestamp = Some(Time(k8s_openapi::jiff::Timestamp::now()));
        apply_router_event(&Event::Apply(pod), &config, &cluster_state, 7000);
        assert_eq!(
            cluster_state.read().get("r1").unwrap().status,
            NodeStatus::Down as i32
        );
    }

    #[test]
    fn test_apply_router_event_unready_marks_suspected() {
        let (config, mut pod) = router_config_and_pod();
        let cluster_state: ClusterState = Arc::default();

        apply_router_event(&Event::Apply(pod.clone()), &config, &cluster_state, 7000);
        if let Some(status) = pod.status.as_mut() {
            status.conditions = Some(vec![PodCondition {
                type_: "Ready".to_string(),
                status: "False".to_string(),
                last_probe_time: None,
                last_transition_time: None,
                message: None,
                reason: None,
                observed_generation: None,
            }]);
        }
        apply_router_event(&Event::Apply(pod), &config, &cluster_state, 7000);
        assert_eq!(
            cluster_state.read().get("r1").unwrap().status,
            NodeStatus::Suspected as i32
        );
    }

    // ========== ModelIdSource tests ==========

    #[test]
    fn test_model_id_source_parse_namespace() {
        let source = ModelIdSource::parse("namespace").unwrap();
        assert!(matches!(source, ModelIdSource::Namespace));
    }

    #[test]
    fn test_model_id_source_parse_namespace_case_insensitive() {
        let source = ModelIdSource::parse("Namespace").unwrap();
        assert!(matches!(source, ModelIdSource::Namespace));
    }

    #[test]
    fn test_model_id_source_parse_label() {
        let source = ModelIdSource::parse("label:model-name").unwrap();
        match source {
            ModelIdSource::Label(key) => assert_eq!(key, "model-name"),
            _ => panic!("Expected Label variant"),
        }
    }

    #[test]
    fn test_model_id_source_parse_annotation() {
        let source = ModelIdSource::parse("annotation:serving.example.com/model-id").unwrap();
        match source {
            ModelIdSource::Annotation(key) => {
                assert_eq!(key, "serving.example.com/model-id");
            }
            _ => panic!("Expected Annotation variant"),
        }
    }

    #[test]
    fn test_model_id_source_parse_label_empty_key() {
        assert!(ModelIdSource::parse("label:").is_err());
    }

    #[test]
    fn test_model_id_source_parse_annotation_empty_key() {
        assert!(ModelIdSource::parse("annotation:").is_err());
    }

    #[test]
    fn test_model_id_source_parse_invalid() {
        assert!(ModelIdSource::parse("hostname").is_err());
        assert!(ModelIdSource::parse("").is_err());
    }

    #[test]
    fn test_model_id_source_extract_namespace() {
        let source = ModelIdSource::Namespace;
        let pod = Pod {
            metadata: ObjectMeta {
                name: Some("pod1".to_string()),
                namespace: Some("team-a-serving".to_string()),
                ..Default::default()
            },
            spec: Some(PodSpec::default()),
            status: None,
        };
        assert_eq!(source.extract(&pod), Some("team-a-serving".to_string()));
    }

    #[test]
    fn test_model_id_source_extract_namespace_missing() {
        let source = ModelIdSource::Namespace;
        let pod = Pod {
            metadata: ObjectMeta {
                name: Some("pod1".to_string()),
                namespace: None,
                ..Default::default()
            },
            spec: Some(PodSpec::default()),
            status: None,
        };
        assert_eq!(source.extract(&pod), None);
    }

    #[test]
    fn test_model_id_source_extract_label() {
        let source = ModelIdSource::Label("model-name".to_string());
        let mut labels = std::collections::BTreeMap::new();
        labels.insert("model-name".to_string(), "llama-70b".to_string());
        let pod = Pod {
            metadata: ObjectMeta {
                name: Some("pod1".to_string()),
                labels: Some(labels),
                ..Default::default()
            },
            spec: Some(PodSpec::default()),
            status: None,
        };
        assert_eq!(source.extract(&pod), Some("llama-70b".to_string()));
    }

    #[test]
    fn test_model_id_source_extract_label_missing() {
        let source = ModelIdSource::Label("model-name".to_string());
        let pod = Pod {
            metadata: ObjectMeta {
                name: Some("pod1".to_string()),
                labels: None,
                ..Default::default()
            },
            spec: Some(PodSpec::default()),
            status: None,
        };
        assert_eq!(source.extract(&pod), None);
    }

    #[test]
    fn test_model_id_source_extract_annotation() {
        let source = ModelIdSource::Annotation("serving.example.com/model-id".to_string());
        let mut annotations = std::collections::BTreeMap::new();
        annotations.insert(
            "serving.example.com/model-id".to_string(),
            "my-model".to_string(),
        );
        let pod = Pod {
            metadata: ObjectMeta {
                name: Some("pod1".to_string()),
                annotations: Some(annotations),
                ..Default::default()
            },
            spec: Some(PodSpec::default()),
            status: None,
        };
        assert_eq!(source.extract(&pod), Some("my-model".to_string()));
    }

    #[test]
    fn test_pod_info_from_pod_with_model_id_override() {
        let mut pod = create_k8s_pod(
            Some("test-pod"),
            Some("10.0.0.1"),
            Some("Running"),
            Some("True"),
            None,
        );
        pod.metadata.namespace = Some("team-a".to_string());

        let config = ServiceDiscoveryConfig {
            model_id_source: Some(ModelIdSource::Namespace),
            ..Default::default()
        };

        let info = PodInfo::from_pod(&pod, Some(&config)).unwrap();
        assert_eq!(info.model_id_override, Some("team-a".to_string()));
    }

    #[test]
    fn test_pod_info_from_pod_without_model_id_source() {
        let pod = create_k8s_pod(
            Some("test-pod"),
            Some("10.0.0.1"),
            Some("Running"),
            Some("True"),
            None,
        );

        let config = ServiceDiscoveryConfig::default();
        let info = PodInfo::from_pod(&pod, Some(&config)).unwrap();
        assert_eq!(info.model_id_override, None);
    }

    fn make_regular_config() -> ServiceDiscoveryConfig {
        let mut selector = HashMap::new();
        selector.insert("app".to_string(), "sglang".to_string());
        ServiceDiscoveryConfig {
            enabled: true,
            selector,
            disaggregated_mode: false,
            ..Default::default()
        }
    }

    fn make_labeled_pod(name: &str, ip: &str, labels: &[(&str, &str)]) -> Pod {
        let mut label_map = std::collections::BTreeMap::new();
        for &(k, v) in labels {
            label_map.insert(k.to_string(), v.to_string());
        }
        Pod {
            metadata: ObjectMeta {
                name: Some(name.to_string()),
                uid: Some(format!("uid-{name}")),
                labels: Some(label_map),
                ..Default::default()
            },
            spec: Some(PodSpec::default()),
            status: Some(PodStatus {
                pod_ip: Some(ip.to_string()),
                phase: Some("Running".to_string()),
                conditions: Some(vec![PodCondition {
                    type_: "Ready".to_string(),
                    status: "True".to_string(),
                    last_probe_time: None,
                    last_transition_time: None,
                    message: None,
                    reason: None,
                    observed_generation: None,
                }]),
                ..Default::default()
            }),
        }
    }

    #[test]
    fn test_pod_info_from_pod_missing_uid() {
        let mut k8s_pod = create_k8s_pod(
            Some("test-pod"),
            Some("10.0.0.1"),
            Some("Running"),
            Some("True"),
            None,
        );
        k8s_pod.metadata.uid = None;
        assert!(PodInfo::from_pod(&k8s_pod, None).is_none());
    }

    #[test]
    fn test_list_label_selector_regular_mode() {
        let mut selector = HashMap::new();
        selector.insert("app".to_string(), "sglang".to_string());
        let config = ServiceDiscoveryConfig {
            selector,
            disaggregated_mode: false,
            ..Default::default()
        };
        assert_eq!(config.list_label_selector(), "app=sglang");
    }

    #[test]
    fn test_list_label_selector_pd_mode_common_labels() {
        let mut prefill = HashMap::new();
        prefill.insert("app".to_string(), "sglang".to_string());
        prefill.insert("component".to_string(), "prefill".to_string());
        let mut decode = HashMap::new();
        decode.insert("app".to_string(), "sglang".to_string());
        decode.insert("component".to_string(), "decode".to_string());
        let config = ServiceDiscoveryConfig {
            disaggregated_mode: true,
            prefill_selector: prefill,
            decode_selector: decode,
            ..Default::default()
        };
        // Only the common label "app=sglang" should be in the selector.
        assert_eq!(config.list_label_selector(), "app=sglang");
    }

    #[test]
    fn test_list_label_selector_epd_mode_common_labels() {
        let config = create_epd_config();

        assert_eq!(config.list_label_selector(), "app=sglang");
    }

    #[test]
    fn test_list_label_selector_pd_mode_no_common_labels() {
        let mut prefill = HashMap::new();
        prefill.insert("role".to_string(), "prefill".to_string());
        let mut decode = HashMap::new();
        decode.insert("role".to_string(), "decode".to_string());
        let config = ServiceDiscoveryConfig {
            disaggregated_mode: true,
            prefill_selector: prefill,
            decode_selector: decode,
            ..Default::default()
        };
        // No common labels → empty selector (falls back to listing all pods).
        assert!(config.list_label_selector().is_empty());
    }

    #[test]
    fn test_deregistration_reconciled_metric_label() {
        // Verify the metric label constant exists and has expected value
        assert_eq!(metrics_labels::DEREGISTRATION_RECONCILED, "reconciled");
    }

    #[test]
    fn test_build_watcher_config_with_selector_pushes_label_selector() {
        let cfg = build_watcher_config("worker", "app=sglang");
        assert_eq!(cfg.label_selector.as_deref(), Some("app=sglang"));
    }

    #[test]
    fn test_build_watcher_config_empty_selector_falls_back_to_default() {
        let cfg = build_watcher_config("worker", "");
        assert!(cfg.label_selector.is_none());
    }

    #[test]
    fn test_build_watcher_config_for_regular_mode_pushes_worker_selector() {
        let mut selector = HashMap::new();
        selector.insert("app".to_string(), "sglang".to_string());
        let config = ServiceDiscoveryConfig {
            selector,
            disaggregated_mode: false,
            ..Default::default()
        };
        let watcher_config = build_watcher_config("worker", &config.list_label_selector());
        assert_eq!(watcher_config.label_selector.as_deref(), Some("app=sglang"));
    }

    #[test]
    fn test_build_watcher_config_for_pd_mode_pushes_intersection() {
        let mut prefill = HashMap::new();
        prefill.insert("app".to_string(), "sglang".to_string());
        prefill.insert("component".to_string(), "prefill".to_string());
        let mut decode = HashMap::new();
        decode.insert("app".to_string(), "sglang".to_string());
        decode.insert("component".to_string(), "decode".to_string());
        let config = ServiceDiscoveryConfig {
            disaggregated_mode: true,
            prefill_selector: prefill,
            decode_selector: decode,
            ..Default::default()
        };
        let watcher_config = build_watcher_config("worker", &config.list_label_selector());
        assert_eq!(watcher_config.label_selector.as_deref(), Some("app=sglang"));
    }

    #[test]
    fn test_build_watcher_config_for_epd_mode_pushes_intersection() {
        let config = create_epd_config();

        let watcher_config = build_watcher_config("worker", &config.list_label_selector());
        assert_eq!(watcher_config.label_selector.as_deref(), Some("app=sglang"));
    }

    #[test]
    fn test_build_watcher_config_for_pd_mode_no_common_labels_omits_filter() {
        let mut prefill = HashMap::new();
        prefill.insert("role".to_string(), "prefill".to_string());
        let mut decode = HashMap::new();
        decode.insert("role".to_string(), "decode".to_string());
        let config = ServiceDiscoveryConfig {
            disaggregated_mode: true,
            prefill_selector: prefill,
            decode_selector: decode,
            ..Default::default()
        };
        let watcher_config = build_watcher_config("worker", &config.list_label_selector());
        assert!(watcher_config.label_selector.is_none());
    }

    #[test]
    fn test_router_label_selector_serializes_router_selector() {
        let mut router = HashMap::new();
        router.insert("app".to_string(), "smg".to_string());
        let config = ServiceDiscoveryConfig {
            router_selector: router,
            ..Default::default()
        };
        assert_eq!(config.router_label_selector(), "app=smg");
    }

    #[test]
    fn test_router_label_selector_empty_when_unset() {
        let config = ServiceDiscoveryConfig::default();
        assert!(config.router_label_selector().is_empty());
    }

    #[test]
    fn test_build_watcher_config_for_router_pushes_router_selector() {
        let mut router = HashMap::new();
        router.insert("app".to_string(), "smg".to_string());
        let config = ServiceDiscoveryConfig {
            router_selector: router,
            ..Default::default()
        };
        let watcher_config = build_watcher_config("router", &config.router_label_selector());
        assert_eq!(watcher_config.label_selector.as_deref(), Some("app=smg"));
    }

    #[traced_test]
    #[test]
    fn test_build_watcher_config_logs_selector_at_info_level() {
        let _ = build_watcher_config("worker", "app=sglang");
        assert!(logs_contain("Starting K8s worker watcher"));
        assert!(logs_contain("app=sglang"));
    }

    #[traced_test]
    #[test]
    fn test_build_watcher_config_logs_router_kind_with_empty_selector() {
        let _ = build_watcher_config("router", "");
        assert!(logs_contain("Starting K8s router watcher"));
        assert!(logs_contain("selector: ''"));
    }
}
