use std::{
    collections::{HashMap, HashSet},
    net::{IpAddr, SocketAddr},
    sync::Arc,
    time::Duration,
};

use futures::StreamExt;
use k8s_openapi::api::core::v1::Pod;
use kube::{
    api::Api,
    runtime::{
        reflector::{self, Store},
        watcher::{watcher, Config},
        WatchStreamExt,
    },
    Client,
};
use openai_protocol::worker::WorkerType;
use rustls::crypto::ring;
use tokio::{sync::Notify, task, time};
use tracing::{debug, error, info, warn};

use super::reconciler::{self, DesiredState, DesiredWorker};
use crate::{
    app_context::AppContext,
    worker::{MOONCAKE_CONNECTOR, NIXL_CONNECTOR},
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
    /// KV metadata is captured at worker registration; replace a Pod after changing it.
    pub kv_connector_annotation: String,
    pub kv_engine_id_annotation: String,
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
}

/// Build a kube watcher Config that pushes the given label selector down to
/// the API server, logging watcher startup at INFO. An empty selector falls
/// back to `Config::default()` (no server-side label filtering) so the
/// watcher still functions when no selector is set.
fn build_watcher_config(label_selector: &str) -> Config {
    info!(
        "Starting K8s worker watcher | selector: '{}'",
        label_selector
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
            kv_connector_annotation: "smg.ai/kv-connector".to_string(),
            kv_engine_id_annotation: "smg.ai/kv-engine-id".to_string(),
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
        let raw_ip = status.pod_ip?;
        // Parsed rather than kept as a string so the worker URL renders an
        // IPv6 address in bracketed form; `format!("{ip}:{port}")` would
        // produce the unusable `fd00::1:8080`.
        let pod_ip: IpAddr = match raw_ip.parse() {
            Ok(ip) => ip,
            Err(e) => {
                warn!("Pod {} has an unparsable Pod IP '{}': {e}", name, raw_ip);
                return None;
            }
        };

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
        let kv_connector = config.and_then(|config| {
            let connector = annotation_value(pod, &config.kv_connector_annotation)?;
            if connector != MOONCAKE_CONNECTOR && connector != NIXL_CONNECTOR {
                warn!(
                    "Pod {}: {} annotation '{}' is not a connector with explicit PD handling; \
                     using passthrough behavior",
                    name, config.kv_connector_annotation, connector
                );
            }
            Some(connector.to_string())
        });
        let kv_engine_ids = config
            .map(|config| resolve_kv_engine_ids(&name, pod, config, ports.len()))
            .unwrap_or_else(|| vec![None; ports.len()]);

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
            kv_connector,
            kv_engine_ids,
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

fn annotation_value<'a>(pod: &'a Pod, key: &str) -> Option<&'a str> {
    pod.metadata
        .annotations
        .as_ref()?
        .get(key)
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
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

pub async fn start_service_discovery(
    config: ServiceDiscoveryConfig,
    app_context: Arc<AppContext>,
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

    Ok(run_service_discovery(client, config, app_context))
}

/// Run discovery against an injected client. Compiled only for this crate's
/// own integration tests, which point it at a scripted API server.
#[cfg(feature = "test-util")]
pub fn start_service_discovery_with_client(
    client: Client,
    config: ServiceDiscoveryConfig,
    app_context: Arc<AppContext>,
) -> task::JoinHandle<()> {
    let _ = ring::default_provider().install_default();
    run_service_discovery(client, config, app_context)
}

fn run_service_discovery(
    client: Client,
    config: ServiceDiscoveryConfig,
    app_context: Arc<AppContext>,
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

        let config_arc = Arc::new(config);

        // Level-triggered reconcile: the reflector keeps `store` consistent
        // with the API server (initial LIST → watch, re-LIST on desync,
        // reconnect with internal backoff). Every watch event and a periodic
        // tick trigger one pass diffing desired workers against the registry.
        let (store, writer) = reflector::store::<Pod>();
        let notify = Arc::new(Notify::new());

        let watcher_config = build_watcher_config(&config_arc.list_label_selector());
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
                reconcile_once(&store, &config_arc, &app_context).await;
            }
        };

        tokio::join!(driver, reconciler);
    });

    handle
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
            let url = SocketAddr::new(info.ip, *port).to_string();
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
                    kv_connector: info.kv_connector.clone(),
                    kv_engine_id: info.kv_engine_ids.get(index).cloned().flatten(),
                });
            }
        }
    }
    state
}

/// Convert the current reflector store snapshot into desired workers and hand
/// it to the shared reconciler. Kubernetes-specific conversion stops here.
async fn reconcile_once(
    store: &Store<Pod>,
    config: &ServiceDiscoveryConfig,
    app_context: &Arc<AppContext>,
) {
    let started_at = time::Instant::now();
    let desired = compute_desired_state(&store.state(), config);
    reconciler::reconcile(&desired, app_context, started_at).await;
}

#[cfg(test)]
mod tests {
    use k8s_openapi::{
        api::core::v1::{Pod, PodCondition, PodSpec, PodStatus},
        apimachinery::pkg::apis::meta::v1::{ObjectMeta, Time},
    };
    use kube::runtime::watcher::Event;
    use tracing_test::traced_test;

    use super::*;

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
            kv_connector_annotation: "smg.ai/kv-connector".to_string(),
            kv_engine_id_annotation: "smg.ai/kv-engine-id".to_string(),
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
        assert_eq!(config.kv_connector_annotation, "smg.ai/kv-connector");
        assert_eq!(config.kv_engine_id_annotation, "smg.ai/kv-engine-id");
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
        assert_eq!(pod_info.ip.to_string(), "10.0.0.1");
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
        assert_eq!(pod_info.ip.to_string(), "10.0.0.1");
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
        assert_eq!(pod_info.ip.to_string(), "10.0.0.5");
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
        assert_eq!(pod_info.ip.to_string(), "10.0.0.2");
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
        assert_eq!(pod_info.ip.to_string(), "10.0.0.3");
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
        assert_eq!(pod_info.ip.to_string(), "10.0.0.4");
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
            ip: "1.1.1.1".parse().unwrap(),
            status: "Running".into(),
            is_ready: true,
            pod_type: None,
            ports: vec![],
            bootstrap_ports: vec![],
            kv_connector: None,
            kv_engine_ids: vec![],
            model_id_override: None,
        };
        assert!(healthy_pod.is_healthy());

        let not_ready_pod = PodInfo {
            name: "p2".into(),
            uid: "uid-p2".into(),
            ip: "1.1.1.2".parse().unwrap(),
            status: "Running".into(),
            is_ready: false,
            pod_type: None,
            ports: vec![],
            bootstrap_ports: vec![],
            kv_connector: None,
            kv_engine_ids: vec![],
            model_id_override: None,
        };
        assert!(!not_ready_pod.is_healthy());

        let not_running_pod = PodInfo {
            name: "p3".into(),
            uid: "uid-p3".into(),
            ip: "1.1.1.3".parse().unwrap(),
            status: "Pending".into(),
            is_ready: true,
            pod_type: None,
            ports: vec![],
            bootstrap_ports: vec![],
            kv_connector: None,
            kv_engine_ids: vec![],
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
    fn test_from_pod_kv_metadata_aligns_with_worker_ports() {
        let config = make_regular_config();
        let pod = pod_with_annotations(
            "w",
            &[
                ("smg.ai/worker-ports", "8080,8081"),
                ("smg.ai/kv-connector", "MooncakeConnector"),
                ("smg.ai/kv-engine-id", "engine-0,engine-1"),
            ],
        );
        let info = PodInfo::from_pod(&pod, Some(&config)).unwrap();
        assert_eq!(info.kv_connector.as_deref(), Some("MooncakeConnector"));
        assert_eq!(
            info.kv_engine_ids,
            vec![Some("engine-0".to_string()), Some("engine-1".to_string())]
        );

        for invalid_ids in ["shared", "shared,shared"] {
            let invalid = pod_with_annotations(
                "w",
                &[
                    ("smg.ai/worker-ports", "8080,8081"),
                    ("smg.ai/kv-connector", "NixlConnector"),
                    ("smg.ai/kv-engine-id", invalid_ids),
                ],
            );
            let info = PodInfo::from_pod(&invalid, Some(&config)).unwrap();
            assert_eq!(info.kv_connector.as_deref(), Some("NixlConnector"));
            assert_eq!(info.kv_engine_ids, vec![None, None]);
        }
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

    fn owned(url: &str, uid: &str) -> reconciler::OwnedWorker {
        use crate::worker::registry::WorkerId;

        reconciler::OwnedWorker {
            id: WorkerId::from_string(url.to_string()),
            url: url.to_string(),
            pod_uid: uid.to_string(),
            revision: 1,
        }
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
        let actions = reconciler::compute_actions(&desired, &registered);
        assert!(actions.add.is_empty());
        assert_eq!(actions.remove.len(), 2);
        assert_eq!(actions.remove[0].pod_uid, "uid-w");
        assert_eq!(actions.remove[1].pod_uid, "uid-w");
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
    fn test_compute_desired_state_brackets_ipv6_worker_urls() {
        let config = make_regular_config();
        let mut pod = make_labeled_pod("w", "10.0.0.1", &[("app", "sglang")]);
        if let Some(status) = pod.status.as_mut() {
            status.pod_ip = Some("fd00::1".to_string());
        }
        let desired = compute_desired_state(&store_snapshot(vec![pod]), &config);
        assert_eq!(
            desired
                .addable
                .iter()
                .map(|w| w.url.as_str())
                .collect::<Vec<_>>(),
            vec!["[fd00::1]:8000"],
            "an IPv6 Pod IP must render as a bracketed authority"
        );
    }

    #[traced_test]
    #[test]
    fn test_from_pod_rejects_an_unparsable_pod_ip() {
        let config = make_regular_config();
        let mut pod = make_labeled_pod("w", "10.0.0.1", &[("app", "sglang")]);
        if let Some(status) = pod.status.as_mut() {
            status.pod_ip = Some("not-an-ip".to_string());
        }
        assert!(PodInfo::from_pod(&pod, Some(&config)).is_none());
        // The warning is the only signal an operator gets for why a Ready,
        // selector-matching Pod never became a worker.
        assert!(logs_contain("has an unparsable Pod IP"));
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
    fn test_build_watcher_config_with_selector_pushes_label_selector() {
        let cfg = build_watcher_config("app=sglang");
        assert_eq!(cfg.label_selector.as_deref(), Some("app=sglang"));
    }

    #[test]
    fn test_build_watcher_config_empty_selector_falls_back_to_default() {
        let cfg = build_watcher_config("");
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
        let watcher_config = build_watcher_config(&config.list_label_selector());
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
        let watcher_config = build_watcher_config(&config.list_label_selector());
        assert_eq!(watcher_config.label_selector.as_deref(), Some("app=sglang"));
    }

    #[test]
    fn test_build_watcher_config_for_epd_mode_pushes_intersection() {
        let config = create_epd_config();

        let watcher_config = build_watcher_config(&config.list_label_selector());
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
        let watcher_config = build_watcher_config(&config.list_label_selector());
        assert!(watcher_config.label_selector.is_none());
    }

    #[traced_test]
    #[test]
    fn test_build_watcher_config_logs_selector_at_info_level() {
        let _ = build_watcher_config("app=sglang");
        assert!(logs_contain("Starting K8s worker watcher"));
        assert!(logs_contain("app=sglang"));
    }
}
