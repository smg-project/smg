use std::sync::Arc;

use k8s_openapi::{api::core::v1::PodCondition, apimachinery::pkg::apis::meta::v1::Time};
use smg_service_discovery::testing::make_labeled_pod;
use tracing_test::traced_test;

use super::*;

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

/// A membership change the discovery watcher makes writes the same
/// transition line as the gossip loop and the probe handler, once per
/// change and naming the status held before: a ready pod that joins, a
/// ready pod at another address (the address held before is named), a
/// pod no longer ready, a pod ready again at its address, a deleted pod,
/// a replacement pod at another address after a delete. A ready pod seen
/// again at its address changes nothing and writes no second line.
#[traced_test]
#[test]
fn a_pod_event_that_changes_the_membership_writes_the_transition_line() {
    let (config, mut pod) = router_config_and_pod();
    let cluster_state: ClusterState = Arc::default();

    apply_router_event(&Event::Apply(pod.clone()), &config, &cluster_state, 7000);
    assert!(logs_contain(
        "Member r1 is Alive at version 1 (a ready pod at 10.1.0.1:7100)"
    ));
    apply_router_event(&Event::Apply(pod.clone()), &config, &cluster_state, 7000);
    logs_assert(|lines: &[&str]| {
        match lines
            .iter()
            .filter(|line| line.contains("Member r1 is Alive"))
            .count()
        {
            1 => Ok(()),
            n => Err(format!("{n} lines for one join")),
        }
    });

    if let Some(status) = pod.status.as_mut() {
        status.pod_ip = Some("10.1.0.2".to_string());
    }
    apply_router_event(&Event::Apply(pod.clone()), &config, &cluster_state, 7000);
    assert!(logs_contain(
        "Member r1 is Alive at version 3 (a ready pod at 10.1.0.2:7100; was Alive at 10.1.0.1:7100)"
    ));

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
    apply_router_event(&Event::Apply(pod.clone()), &config, &cluster_state, 7000);
    assert!(logs_contain(
        "Member r1 is Suspected at version 4 (a pod not ready; was Alive)"
    ));

    if let Some(status) = pod.status.as_mut() {
        status.conditions = Some(vec![PodCondition {
            type_: "Ready".to_string(),
            status: "True".to_string(),
            last_probe_time: None,
            last_transition_time: None,
            message: None,
            reason: None,
            observed_generation: None,
        }]);
    }
    apply_router_event(&Event::Apply(pod.clone()), &config, &cluster_state, 7000);
    assert!(logs_contain(
        "Member r1 is Alive at version 5 (a ready pod at 10.1.0.2:7100; was Suspected)"
    ));

    apply_router_event(&Event::Delete(pod.clone()), &config, &cluster_state, 7000);
    assert!(logs_contain(
        "Member r1 is Down at version 6 (a deleted pod; was Alive)"
    ));

    if let Some(status) = pod.status.as_mut() {
        status.pod_ip = Some("10.1.0.3".to_string());
    }
    apply_router_event(&Event::Apply(pod), &config, &cluster_state, 7000);
    assert!(logs_contain(
        "Member r1 is Alive at version 7 (a ready pod at 10.1.0.3:7100; was Down at 10.1.0.2:7100)"
    ));
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
fn test_apply_router_event_brackets_ipv6_addresses() {
    let (config, mut pod) = router_config_and_pod();
    if let Some(status) = pod.status.as_mut() {
        status.pod_ip = Some("fd00::1".to_string());
    }
    let cluster_state: ClusterState = Arc::default();

    apply_router_event(&Event::Apply(pod), &config, &cluster_state, 7000);
    assert_eq!(
        cluster_state.read().get("r1").unwrap().address,
        "[fd00::1]:7100"
    );
}

#[test]
fn test_from_pod_rejects_an_unparsable_pod_ip() {
    let (config, mut pod) = router_config_and_pod();
    if let Some(status) = pod.status.as_mut() {
        status.pod_ip = Some("not-an-ip".to_string());
    }
    assert!(RouterPodInfo::from_pod(&pod, &config).is_none());
}

#[traced_test]
#[test]
fn test_invalid_mesh_port_annotation_warns_and_falls_back() {
    let (config, mut pod) = router_config_and_pod();
    pod.metadata.annotations = Some(
        [("sglang.ai/mesh-port".to_string(), "not-a-port".to_string())]
            .into_iter()
            .collect(),
    );
    let cluster_state: ClusterState = Arc::default();

    apply_router_event(&Event::Apply(pod), &config, &cluster_state, 7000);
    assert_eq!(
        cluster_state.read().get("r1").unwrap().address,
        "10.1.0.1:7000",
        "an unusable annotation must fall back to the configured mesh port"
    );
    assert!(logs_contain("invalid sglang.ai/mesh-port annotation"));
}

#[traced_test]
#[test]
fn test_zero_mesh_port_annotation_is_rejected() {
    let (config, mut pod) = router_config_and_pod();
    pod.metadata.annotations = Some(
        [("sglang.ai/mesh-port".to_string(), "0".to_string())]
            .into_iter()
            .collect(),
    );
    let cluster_state: ClusterState = Arc::default();

    apply_router_event(&Event::Apply(pod), &config, &cluster_state, 7000);
    assert_eq!(
        cluster_state.read().get("r1").unwrap().address,
        "10.1.0.1:7000"
    );
    assert!(logs_contain("invalid sglang.ai/mesh-port annotation"));
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
    let watcher_config = build_watcher_config("router", &config.label_selector());
    assert_eq!(watcher_config.label_selector.as_deref(), Some("app=smg"));
}

#[traced_test]
#[test]
fn test_build_watcher_config_logs_router_kind_with_empty_selector() {
    let _ = build_watcher_config("router", "");
    assert!(logs_contain("Starting K8s router watcher"));
    assert!(logs_contain("selector: ''"));
}
