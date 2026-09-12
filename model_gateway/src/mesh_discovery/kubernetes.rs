//! Kubernetes watcher for SMG router Pods.
//!
//! Router peers are applied straight to the mesh `ClusterState`: the mesh
//! SWIM/CRDT layer owns convergence, so this is edge-triggered off the watch
//! stream and keeps no store of its own.

use std::collections::HashMap;

use futures::StreamExt;
use k8s_openapi::api::core::v1::Pod;
use kube::{
    api::Api,
    runtime::{
        watcher::{watcher, Config, Event},
        WatchStreamExt,
    },
    Client,
};
use rustls::crypto::ring;
use smg_mesh::{
    gossip::{NodeState, NodeStatus},
    ClusterState,
};
use tokio::task;
use tracing::{debug, error, info, warn};

/// Configuration for Kubernetes router-peer discovery.
#[derive(Debug, Clone)]
pub struct MeshDiscoveryConfig {
    /// `None` = all namespaces.
    pub namespace: Option<String>,
    /// Label selector identifying router Pods. Empty means disabled.
    pub router_selector: HashMap<String, String>,
    /// Pod annotation carrying the router's mesh port.
    pub router_mesh_port_annotation: String,
}

impl Default for MeshDiscoveryConfig {
    fn default() -> Self {
        Self {
            namespace: None,
            router_selector: HashMap::new(),
            router_mesh_port_annotation: "sglang.ai/mesh-port".to_string(),
        }
    }
}

impl MeshDiscoveryConfig {
    /// Router discovery only runs with a selector; without one there is no way
    /// to tell a router Pod from any other Pod in the namespace.
    pub fn is_enabled(&self) -> bool {
        !self.router_selector.is_empty()
    }

    /// Build a label selector string for router Pod list/watch calls. Empty
    /// when unset, in which case the watcher lists without server-side
    /// label filtering.
    fn label_selector(&self) -> String {
        self.router_selector
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// The router Pod fields the mesh cares about.
struct RouterPodInfo {
    name: String,
    ip: String,
    phase: String,
    is_ready: bool,
    mesh_port: Option<u16>,
}

impl RouterPodInfo {
    fn from_pod(pod: &Pod, config: &MeshDiscoveryConfig) -> Option<Self> {
        let name = pod.metadata.name.clone()?;
        if pod.metadata.uid.is_none() {
            warn!("Router pod {} has no UID, skipping", name);
            return None;
        }
        let status = pod.status.clone()?;
        let ip = status.pod_ip?;

        let is_ready = status.conditions.as_ref().is_some_and(|conditions| {
            conditions
                .iter()
                .any(|condition| condition.type_ == "Ready" && condition.status == "True")
        });

        let mesh_port = pod
            .metadata
            .annotations
            .as_ref()
            .and_then(|annotations| annotations.get(&config.router_mesh_port_annotation))
            .and_then(|port| port.parse::<u16>().ok());

        Some(RouterPodInfo {
            name,
            ip,
            phase: status.phase.unwrap_or_else(|| "Unknown".to_string()),
            is_ready,
            mesh_port,
        })
    }

    fn is_healthy(&self) -> bool {
        self.is_ready && self.phase == "Running"
    }
}

fn matches_selector(pod: &Pod, selector: &HashMap<String, String>) -> bool {
    if selector.is_empty() {
        return false;
    }
    pod.metadata
        .labels
        .as_ref()
        .is_some_and(|labels| selector.iter().all(|(k, v)| labels.get(k) == Some(v)))
}

/// Build a kube watcher Config that pushes the given label selector down to
/// the API server. An empty selector falls back to `Config::default()`.
fn build_watcher_config(label_selector: &str) -> Config {
    info!(
        "Starting K8s router watcher | selector: '{}'",
        label_selector
    );
    if label_selector.is_empty() {
        Config::default()
    } else {
        Config::default().labels(label_selector)
    }
}

/// Apply one router-watcher event to the mesh cluster state. A hard delete
/// or a set deletion timestamp marks the node Down; the mesh SWIM/CRDT layer
/// owns convergence, so no store snapshot is needed here.
fn apply_router_event(
    event: &Event<Pod>,
    config: &MeshDiscoveryConfig,
    cluster_state: &ClusterState,
    default_mesh_port: u16,
) {
    let (pod, is_delete) = match event {
        Event::Apply(pod) | Event::InitApply(pod) => (pod, false),
        Event::Delete(pod) => (pod, true),
        Event::Init | Event::InitDone => return,
    };

    if !matches_selector(pod, &config.router_selector) {
        return;
    }
    let Some(pod_info) = RouterPodInfo::from_pod(pod, config) else {
        return;
    };

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

/// Start Kubernetes router-peer discovery against the default in-cluster or
/// kubeconfig client.
///
/// Errors when the config is disabled. Without a selector the watcher would
/// LIST/WATCH every Pod in scope — cluster-wide when no namespace is set — and
/// then discard all of them, so refusing is safer than starting a watch that
/// can never produce a router.
pub async fn start_mesh_discovery(
    config: MeshDiscoveryConfig,
    cluster_state: ClusterState,
    default_mesh_port: u16,
) -> Result<task::JoinHandle<()>, kube::Error> {
    if !config.is_enabled() {
        return Err(kube::Error::Api(
            kube::core::Status::failure(
                "Mesh router discovery is disabled: no router selector configured",
                "ConfigurationError",
            )
            .with_code(400)
            .boxed(),
        ));
    }

    let _ = ring::default_provider().install_default();
    let client = Client::try_default().await?;
    Ok(run_mesh_discovery(
        client,
        config,
        cluster_state,
        default_mesh_port,
    ))
}

/// Run router discovery against an injected client. Compiled only for this
/// crate's own integration tests, which point it at a scripted API server.
#[cfg(feature = "test-util")]
pub fn start_mesh_discovery_with_client(
    client: Client,
    config: MeshDiscoveryConfig,
    cluster_state: ClusterState,
    default_mesh_port: u16,
) -> task::JoinHandle<()> {
    assert!(
        config.is_enabled(),
        "mesh router discovery requires a router selector"
    );
    let _ = ring::default_provider().install_default();
    run_mesh_discovery(client, config, cluster_state, default_mesh_port)
}

fn run_mesh_discovery(
    client: Client,
    config: MeshDiscoveryConfig,
    cluster_state: ClusterState,
    default_mesh_port: u16,
) -> task::JoinHandle<()> {
    info!(
        "Router node discovery enabled | selector: '{}' | mesh port annotation: '{}'",
        config.label_selector(),
        config.router_mesh_port_annotation
    );

    #[expect(
        clippy::disallowed_methods,
        reason = "router discovery runs for the lifetime of the server; shutdown aborts the handle"
    )]
    task::spawn(async move {
        let pods: Api<Pod> = if let Some(namespace) = &config.namespace {
            Api::namespaced(client, namespace)
        } else {
            Api::all(client)
        };

        let watcher_config = build_watcher_config(&config.label_selector());
        let mut stream = watcher(pods, watcher_config).default_backoff().boxed();

        while let Some(item) = stream.next().await {
            match item {
                Ok(event) => apply_router_event(&event, &config, &cluster_state, default_mesh_port),
                Err(e) => error!("Router watcher error (auto-retrying with backoff): {e}"),
            }
        }

        error!("K8s router watcher stream ended; router discovery no longer receives updates");
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use k8s_openapi::{
        api::core::v1::{PodCondition, PodSpec, PodStatus},
        apimachinery::pkg::apis::meta::v1::{ObjectMeta, Time},
    };
    use tracing_test::traced_test;

    use super::*;

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

    fn router_config_and_pod() -> (MeshDiscoveryConfig, Pod) {
        let mut router_selector = HashMap::new();
        router_selector.insert("role".to_string(), "router".to_string());
        let config = MeshDiscoveryConfig {
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

    #[test]
    fn test_apply_router_event_ignores_non_router_pods() {
        let (config, _) = router_config_and_pod();
        let cluster_state: ClusterState = Arc::default();
        let worker = make_labeled_pod("w1", "10.1.0.9", &[("app", "sglang")]);

        apply_router_event(&Event::Apply(worker), &config, &cluster_state, 7000);
        assert!(cluster_state.read().is_empty());
    }

    #[tokio::test]
    async fn test_start_mesh_discovery_refuses_an_empty_selector() {
        // An empty selector must not open an unfiltered, cluster-wide Pod
        // watch: refuse before a client is even built.
        let err = start_mesh_discovery(MeshDiscoveryConfig::default(), Arc::default(), 7000)
            .await
            .expect_err("disabled mesh discovery must not start a watch");
        assert!(
            err.to_string().contains("disabled"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_is_enabled_requires_a_selector() {
        assert!(!MeshDiscoveryConfig::default().is_enabled());
        let (config, _) = router_config_and_pod();
        assert!(config.is_enabled());
    }

    #[test]
    fn test_label_selector_serializes_router_selector() {
        let mut router = HashMap::new();
        router.insert("app".to_string(), "smg".to_string());
        let config = MeshDiscoveryConfig {
            router_selector: router,
            ..Default::default()
        };
        assert_eq!(config.label_selector(), "app=smg");
    }

    #[test]
    fn test_label_selector_empty_when_unset() {
        assert!(MeshDiscoveryConfig::default().label_selector().is_empty());
    }

    #[test]
    fn test_build_watcher_config_pushes_router_selector() {
        let mut router = HashMap::new();
        router.insert("app".to_string(), "smg".to_string());
        let config = MeshDiscoveryConfig {
            router_selector: router,
            ..Default::default()
        };
        let watcher_config = build_watcher_config(&config.label_selector());
        assert_eq!(watcher_config.label_selector.as_deref(), Some("app=smg"));
    }

    #[traced_test]
    #[test]
    fn test_build_watcher_config_logs_router_kind_with_empty_selector() {
        let _ = build_watcher_config("");
        assert!(logs_contain("Starting K8s router watcher"));
        assert!(logs_contain("selector: ''"));
    }
}
