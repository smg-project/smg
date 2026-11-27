//! Kubernetes watcher for SMG router Pods.
//!
//! Router peers are applied straight to the mesh `ClusterState`: the mesh
//! SWIM/CRDT layer owns convergence, so this is edge-triggered off the watch
//! stream and keeps no store of its own.

use std::{
    collections::HashMap,
    net::{IpAddr, SocketAddr},
};

use futures::StreamExt;
use k8s_openapi::api::core::v1::Pod;
use kube::{
    api::Api,
    runtime::{
        watcher::{watcher, Event},
        WatchStreamExt,
    },
};
use rustls::crypto::ring;
use smg_mesh::{
    gossip::{NodeState, NodeStatus},
    member_transition_line, status_of, ClusterState,
};
use smg_service_discovery::{
    build_watcher_config, matches_selector, pod_ip, pod_is_ready, selector_string, watch_client,
};
use tokio::task;
use tracing::{debug, error, info, warn};

use crate::config::DiscoveryConfig;

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
    /// Router discovery as configured today: inside Kubernetes worker
    /// discovery, so it runs only alongside it. `None` without a router
    /// selector, and under any other worker provider.
    pub fn from_discovery(discovery: &DiscoveryConfig) -> Option<Self> {
        let DiscoveryConfig::Kubernetes(kubernetes) = discovery else {
            return None;
        };
        Some(Self {
            namespace: kubernetes.namespace.clone(),
            router_selector: kubernetes.router_selector.clone(),
            router_mesh_port_annotation: kubernetes.router_mesh_port_annotation.clone(),
        })
        .filter(Self::is_enabled)
    }

    /// Router discovery only runs with a selector; without one there is no way
    /// to tell a router Pod from any other Pod in the namespace.
    pub fn is_enabled(&self) -> bool {
        !self.router_selector.is_empty()
    }

    /// Build a label selector string for router Pod list/watch calls. Empty
    /// when unset, in which case the watcher lists without server-side
    /// label filtering.
    fn label_selector(&self) -> String {
        selector_string(&self.router_selector)
    }
}

/// The router Pod fields the mesh cares about.
struct RouterPodInfo {
    name: String,
    ip: IpAddr,
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
        let ip = pod_ip(&name, &status.pod_ip?)?;
        let is_ready = pod_is_ready(status.conditions.as_deref());

        let mesh_port = pod
            .metadata
            .annotations
            .as_ref()
            .and_then(|annotations| annotations.get(&config.router_mesh_port_annotation))
            .and_then(|raw| match raw.trim().parse::<u16>() {
                Ok(port) if port != 0 => Some(port),
                _ => {
                    warn!(
                        "Router pod {}: invalid {} annotation '{}', falling back to the \
                         gateway's own mesh port",
                        name, config.router_mesh_port_annotation, raw
                    );
                    None
                }
            });

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

/// Apply one router-watcher event to the mesh cluster state. A hard delete
/// or a set deletion timestamp marks the node Down; the mesh SWIM/CRDT layer
/// owns convergence, so no store snapshot is needed here. A status change
/// writes the membership's transition line, as the gossip loop and the
/// probe handler do; an event that changes no status writes nothing at info.
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
            let held = status_of(node);
            node.status = NodeStatus::Down as i32;
            node.version += 1;
            if held != NodeStatus::Down {
                info!(
                    "{}",
                    member_transition_line(node, format_args!("a deleted pod; was {held:?}"))
                );
            }
            debug!("Router node {} marked as Down (pod deleted)", pod_info.name);
        } else {
            debug!(
                "Router node {} not found in cluster state (already removed)",
                pod_info.name
            );
        }
    } else if pod_info.is_healthy() {
        let mesh_port = pod_info.mesh_port.unwrap_or(default_mesh_port);
        let node_address = SocketAddr::new(pod_info.ip, mesh_port).to_string();
        let mut state = cluster_state.write();
        let existing = state.get(&pod_info.name);
        let held = existing.map(|node| (status_of(node), node.address.clone()));
        let existing_version = existing.map_or(0, |node| node.version);

        let node_state = NodeState {
            name: pod_info.name.clone(),
            address: node_address,
            status: NodeStatus::Alive as i32,
            version: existing_version + 1,
            metadata: HashMap::new(),
        };

        state.insert(pod_info.name.clone(), node_state.clone());
        // A ready pod seen again at its address changes nothing. One at another
        // address while the table held it `Alive` is a change the peers dial
        // (a pod replaced while the watch was away: the relist brings the new
        // pod and no delete of the old), so it is written, naming the address
        // held before.
        let unchanged = held.as_ref().is_some_and(|(status, address)| {
            *status == NodeStatus::Alive && *address == node_state.address
        });
        if !unchanged {
            info!(
                "{}",
                member_transition_line(
                    &node_state,
                    format_args!(
                        "a ready pod at {}{}",
                        node_state.address,
                        was(held.as_ref(), &node_state.address)
                    )
                )
            );
        }
        debug!(
            "Router node {} added/updated in mesh cluster (address: {})",
            pod_info.name, node_state.address
        );
    } else {
        let mut state = cluster_state.write();
        if let Some(node) = state.get_mut(&pod_info.name) {
            if node.status != NodeStatus::Down as i32 {
                let held = status_of(node);
                node.status = NodeStatus::Suspected as i32;
                node.version += 1;
                if held != NodeStatus::Suspected {
                    info!(
                        "{}",
                        member_transition_line(node, format_args!("a pod not ready; was {held:?}"))
                    );
                }
                debug!(
                    "Router node {} marked as Suspected (pod not healthy)",
                    pod_info.name
                );
            }
        }
    }
}

/// The `was` clause of a transition line: the status held before, when the
/// member was in the table, and the address held before, when it moved.
fn was(held: Option<&(NodeStatus, String)>, address: &str) -> String {
    match held {
        None => String::new(),
        Some((status, held_address)) if held_address == address => format!("; was {status:?}"),
        Some((status, held_address)) => format!("; was {status:?} at {held_address}"),
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

    let inferred = kube::Config::infer()
        .await
        .map_err(kube::Error::InferConfig)?;
    start_mesh_discovery_on(inferred, config, cluster_state, default_mesh_port)
}

/// Run router discovery on the client configuration `inferred`, for a config
/// that names a router selector: the one start of the watch, so nothing runs
/// it on a client with the client library's defaults. The module exports it
/// only under `test-util`, for this crate's own integration tests: they hand
/// it the configuration of a scripted API server, and the watch they start
/// runs on the client a served gateway's watch runs on.
pub fn start_mesh_discovery_on(
    inferred: kube::Config,
    config: MeshDiscoveryConfig,
    cluster_state: ClusterState,
    default_mesh_port: u16,
) -> Result<task::JoinHandle<()>, kube::Error> {
    debug_assert!(
        config.is_enabled(),
        "mesh router discovery requires a router selector"
    );
    let _ = ring::default_provider().install_default();
    let client = watch_client(inferred)?;

    info!(
        "Router node discovery enabled | selector: '{}' | mesh port annotation: '{}'",
        config.label_selector(),
        config.router_mesh_port_annotation
    );

    #[expect(
        clippy::disallowed_methods,
        reason = "router discovery runs for the lifetime of the server; shutdown aborts the handle"
    )]
    let handle = task::spawn(async move {
        let pods: Api<Pod> = if let Some(namespace) = &config.namespace {
            Api::namespaced(client, namespace)
        } else {
            Api::all(client)
        };

        let watcher_config = build_watcher_config("router", &config.label_selector());
        let mut stream = watcher(pods, watcher_config).default_backoff().boxed();

        while let Some(item) = stream.next().await {
            match item {
                Ok(event) => apply_router_event(&event, &config, &cluster_state, default_mesh_port),
                Err(e) => error!("Router watcher error (auto-retrying with backoff): {e}"),
            }
        }

        error!("K8s router watcher stream ended; router discovery no longer receives updates");
    });

    Ok(handle)
}

#[cfg(test)]
#[path = "../../../tests/unit/mesh/discovery/kubernetes/tests.rs"]
mod tests;
