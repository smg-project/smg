use std::sync::atomic::{AtomicUsize, Ordering};

use futures::{channel::mpsc, FutureExt, SinkExt};
use k8s_openapi::{
    api::core::v1::{Pod, PodCondition, PodSpec, PodStatus},
    apimachinery::pkg::apis::meta::v1::{ObjectMeta, Time},
};
use kube::runtime::watcher::{self, Event};
use openai_protocol::worker::WorkerSpec;
use smg_worker::{
    identity::{
        basic::Worker,
        endpoint::{endpoint_key, Endpoint, EndpointKey},
    },
    jobs::WorkerJobs,
    registry::{WorkerId, WorkerOrigin},
};
use tracing_test::traced_test;

use super::*;
use crate::testing::make_labeled_pod;

/// The pod's Ready condition at `status` ("True", "False" or "Unknown"),
/// every other field unset.
fn ready_condition(status: &str) -> PodCondition {
    PodCondition {
        type_: "Ready".to_string(),
        status: status.to_string(),
        last_probe_time: None,
        last_transition_time: None,
        message: None,
        reason: None,
        observed_generation: None,
    }
}

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
            let condition = ready_condition(status_str);
            pod_status.conditions = Some(vec![condition]);
        }
        pod.status = Some(pod_status);
    }
    pod
}

/// A running, ready pod of the PD deployment: the `app` and `component`
/// labels the selectors read, and the bootstrap-port annotation when the
/// role has one (an empty, not absent, annotation map otherwise).
fn create_pd_k8s_pod(name: &str, ip: &str, pod_type: &str, bootstrap_port: Option<u16>) -> Pod {
    let mut pod = make_labeled_pod(name, ip, &[("app", "sglang"), ("component", pod_type)]);
    pod.metadata.annotations = Some(
        bootstrap_port
            .into_iter()
            .map(|port| ("sglang.ai/bootstrap-port".to_string(), port.to_string()))
            .collect(),
    );
    pod
}

fn create_pd_config() -> ServiceDiscoveryConfig {
    let mut prefill_selector = HashMap::new();
    prefill_selector.insert("app".to_string(), "sglang".to_string());
    prefill_selector.insert("component".to_string(), "prefill".to_string());

    let mut decode_selector = HashMap::new();
    decode_selector.insert("app".to_string(), "sglang".to_string());
    decode_selector.insert("component".to_string(), "decode".to_string());

    ServiceDiscoveryConfig {
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
        pairing_protocol_label: "smg.ai/pairing-protocol".to_string(),
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
    assert_eq!(config.pairing_protocol_label, "smg.ai/pairing-protocol");
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

/// The role is the first role selector a Pod matches, in encode, prefill,
/// decode order; a Pod matching none is Regular, and outside disaggregated
/// routing every Pod is Regular whatever its labels say.
#[test]
fn the_first_matching_role_selector_names_the_pod_type() {
    let mut config = create_pd_config();
    config.encode_selector = HashMap::from([("component".to_string(), "prefill".to_string())]);
    let prefill_pod = create_pd_k8s_pod("prefill-pod", "10.0.0.1", "prefill", None);
    let other_pod = make_labeled_pod("other-pod", "10.0.0.2", &[("app", "other")]);

    assert_eq!(
        config.pod_type_of(&prefill_pod),
        PodType::Encode,
        "a Pod matching the encode and the prefill selector is encode: the first match"
    );
    config.encode_selector.clear();
    assert_eq!(config.pod_type_of(&prefill_pod), PodType::Prefill);
    assert_eq!(config.pod_type_of(&other_pod), PodType::Regular);

    config.disaggregated_mode = false;
    assert_eq!(config.pod_type_of(&prefill_pod), PodType::Regular);
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
        pairing_protocol: None,
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
        pairing_protocol: None,
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
        pairing_protocol: None,
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

// ========== Pairing protocol label ==========

/// `pod` with one more label.
fn with_label(mut pod: Pod, key: &str, value: &str) -> Pod {
    pod.metadata
        .labels
        .get_or_insert_with(BTreeMap::new)
        .insert(key.to_string(), value.to_string());
    pod
}

/// A Pod's pairing-protocol label is read as its explicit pairing protocol:
/// trimmed, and none where the label is absent, empty or blank.
#[test]
fn a_pods_pairing_protocol_label_is_read_trimmed_and_an_empty_one_states_none() {
    let config = create_pd_config();
    let read = |value: Option<&str>| {
        let pod = create_pd_k8s_pod("decode-0", "10.0.0.1", "decode", None);
        let pod = match value {
            Some(value) => with_label(pod, "smg.ai/pairing-protocol", value),
            None => pod,
        };
        let info = PodInfo::from_pod(&pod, Some(&config)).unwrap();
        info.pairing_protocol
    };
    assert_eq!(read(Some("rollout-7")).as_deref(), Some("rollout-7"));
    assert_eq!(read(Some("  rollout-7 ")).as_deref(), Some("rollout-7"));
    assert_eq!(read(Some("")), None);
    assert_eq!(read(Some("   ")), None);
    assert_eq!(read(None), None);
}

/// The label's name is the configuration's: a Pod that carries both the
/// configured label and the default one is read by the configured one.
#[test]
fn the_pairing_protocol_is_read_from_the_label_the_configuration_names() {
    let mut config = create_pd_config();
    config.pairing_protocol_label = "example.com/rollout".to_string();
    let pod = create_pd_k8s_pod("decode-0", "10.0.0.1", "decode", None);
    let pod = with_label(pod, "example.com/rollout", "blue");
    let pod = with_label(pod, "smg.ai/pairing-protocol", "green");
    let info = PodInfo::from_pod(&pod, Some(&config)).unwrap();
    assert_eq!(info.pairing_protocol.as_deref(), Some("blue"));
}

/// Every worker a Pod publishes carries the Pod's pairing protocol. A Pod
/// that is relabelled publishes the same instances under another
/// fingerprint, which the reconciler answers with an update in place.
#[test]
fn each_worker_of_a_pod_carries_its_pairing_protocol_and_a_relabel_moves_its_fingerprint() {
    let config = create_pd_config();
    let published = |protocol: &str| {
        let pod = create_pd_k8s_pod("decode-0", "10.0.0.1", "decode", None);
        let mut pod = with_label(pod, "smg.ai/pairing-protocol", protocol);
        let annotations = pod.metadata.annotations.as_mut().unwrap();
        annotations.insert("smg.ai/worker-ports".to_string(), "8080,8081".to_string());
        compute_desired_state(&store_snapshot(vec![pod]), &config)
    };
    let (first, relabelled) = (published("v1"), published("v2"));
    assert_eq!(first.workers.len(), 2);
    for (endpoint, worker) in &first.workers {
        assert_eq!(worker.pairing_protocol.as_deref(), Some("v1"));
        let again = &relabelled.workers[endpoint];
        assert_eq!(again.discovery_id, worker.discovery_id);
        assert_eq!(again.pairing_protocol.as_deref(), Some("v2"));
        assert_ne!(again.fingerprint(), worker.fingerprint());
    }
}

// ========== Desired state ==========

fn store_snapshot(pods: Vec<Pod>) -> Vec<Arc<Pod>> {
    pods.into_iter().map(Arc::new).collect()
}

#[derive(Default)]
struct ReconcileCountingHost {
    passes: AtomicUsize,
}

impl DiscoveryHost for ReconcileCountingHost {
    fn workers(&self) -> Vec<(WorkerId, Arc<dyn Worker>)> {
        self.passes.fetch_add(1, Ordering::Relaxed);
        Vec::new()
    }

    fn origin_of(&self, _worker_id: &WorkerId) -> Option<WorkerOrigin> {
        None
    }

    fn stamp(&self, _spec: &mut WorkerSpec) {}

    fn jobs(&self) -> Option<Arc<dyn WorkerJobs>> {
        None
    }
}

async fn assert_watch_event_wake(
    events: &mut mpsc::Sender<watcher::Result<Event<Pod>>>,
    notify: &Notify,
    event: watcher::Result<Event<Pod>>,
    expected: bool,
) {
    assert!(events.send(event).await.is_ok());
    task::yield_now().await;
    assert_eq!(notify.notified().now_or_never().is_some(), expected);
}

#[tokio::test]
async fn drive_watch_notifies_only_after_visible_store_changes() {
    let (mut events, stream) = mpsc::channel(1);
    let notify = Arc::new(Notify::new());
    let loop_notify = Arc::clone(&notify);
    #[expect(
        clippy::disallowed_methods,
        reason = "the finite stream task is joined at the end of the test"
    )]
    let handle = task::spawn(async move {
        drive_watch(stream, loop_notify.as_ref()).await;
    });

    let pod = Pod::default;
    assert_watch_event_wake(&mut events, &notify, Ok(Event::Init), false).await;
    assert_watch_event_wake(&mut events, &notify, Ok(Event::InitApply(pod())), false).await;
    assert_watch_event_wake(&mut events, &notify, Ok(Event::InitDone), false).await;
    assert_watch_event_wake(&mut events, &notify, Ok(Event::Apply(pod())), true).await;
    assert_watch_event_wake(&mut events, &notify, Ok(Event::Delete(pod())), true).await;
    assert_watch_event_wake(&mut events, &notify, Ok(Event::Init), false).await;
    assert_watch_event_wake(&mut events, &notify, Ok(Event::InitApply(pod())), false).await;
    assert_watch_event_wake(&mut events, &notify, Ok(Event::InitDone), true).await;
    assert_watch_event_wake(
        &mut events,
        &notify,
        Err(watcher::Error::NoResourceVersion),
        false,
    )
    .await;

    drop(events);
    assert!(handle.await.is_ok());
}

#[tokio::test(start_paused = true)]
async fn unchanged_ticks_reuse_the_projected_pod_snapshot() {
    PROJECTION_PASSES.set(0);
    let (store, mut writer) = reflector::store::<Pod>();
    writer.apply_watcher_event(&Event::Init);
    writer.apply_watcher_event(&Event::InitApply(make_labeled_pod(
        "worker-0",
        "10.0.0.1",
        &[("app", "sglang")],
    )));
    writer.apply_watcher_event(&Event::InitDone);

    let notify = Arc::new(Notify::new());
    let host = Arc::new(ReconcileCountingHost::default());
    let mut config = make_regular_config();
    config.check_interval = Duration::from_secs(60);
    let loop_notify = Arc::clone(&notify);
    let loop_host = Arc::clone(&host);
    #[expect(
        clippy::disallowed_methods,
        reason = "the loop under test is aborted at the end of the test"
    )]
    let handle = task::spawn(async move {
        reconcile_on_change(&store, loop_notify.as_ref(), &config, loop_host.as_ref()).await;
    });

    task::yield_now().await;
    for _ in 0..2 {
        time::advance(Duration::from_secs(60)).await;
        task::yield_now().await;
    }
    assert_eq!(host.passes.load(Ordering::Relaxed), 3);
    assert_eq!(PROJECTION_PASSES.get(), 1);

    let changed = make_labeled_pod("worker-0", "10.0.0.2", &[("app", "sglang")]);
    writer.apply_watcher_event(&Event::Apply(changed));
    notify.notify_one();
    task::yield_now().await;
    time::advance(Duration::from_secs(1)).await;
    task::yield_now().await;

    assert_eq!(host.passes.load(Ordering::Relaxed), 4);
    assert_eq!(PROJECTION_PASSES.get(), 2);
    handle.abort();
}

fn key(url: &str) -> EndpointKey {
    endpoint_key(url).expect(url)
}

/// A registered worker whose Pod the tests then stop publishing, so its
/// fingerprint never comes into play.
fn owned(url: &str, discovery_id: &str) -> reconciler::OwnedWorker {
    reconciler::OwnedWorker {
        worker_id: WorkerId::from_string(url.to_string()),
        discovery_id: Some(discovery_id.to_string()),
        spec_hash: None,
        endpoint: Endpoint::parse_with_rank(url).expect(url).0,
        revision: 1,
    }
}

#[test]
fn test_compute_actions_unready_pod_removes_registered_workers() {
    let config = make_regular_config();
    let mut pod = pod_with_annotations("w", &[("smg.ai/worker-ports", "8080,8081")]);
    if let Some(status) = pod.status.as_mut() {
        status.conditions = Some(vec![ready_condition("False")]);
    }
    let desired = compute_desired_state(&store_snapshot(vec![pod]), &config);
    let registered = [
        owned("10.0.0.1:8080", "uid-w:8080"),
        owned("10.0.0.1:8081", "uid-w:8081"),
    ];
    let actions = reconciler::compute_actions(&desired, &registered);
    assert!(actions.add.is_empty());
    assert_eq!(actions.remove.len(), 2);
    assert_eq!(
        actions.remove[0].discovery_id.as_deref(),
        Some("uid-w:8080")
    );
    assert_eq!(
        actions.remove[1].discovery_id.as_deref(),
        Some("uid-w:8081")
    );
}

#[test]
fn test_compute_desired_state_multi_port_pod() {
    let config = make_regular_config();
    let pod = pod_with_annotations("w", &[("smg.ai/worker-ports", "8080,8081")]);
    let desired = compute_desired_state(&store_snapshot(vec![pod]), &config);

    // One worker per port, each its own instance, so the id carries the
    // port: one Pod UID alone would not be unique within the snapshot.
    assert_eq!(desired.workers.len(), 2);
    for port in [8080, 8081] {
        let worker = &desired.workers[&key(&format!("10.0.0.1:{port}"))];
        assert_eq!(worker.discovery_id, format!("uid-w:{port}"));
        assert_eq!(
            worker.compat_labels.get(POD_UID_LABEL).map(String::as_str),
            Some("uid-w")
        );
        assert_eq!(
            worker.compat_labels.get(POD_NAME_LABEL).map(String::as_str),
            Some("w")
        );
    }
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
            .workers
            .values()
            .map(|w| w.endpoint.render())
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
    assert!(desired.workers.is_empty());
}

#[test]
fn test_compute_desired_state_unready_pod_excluded() {
    let config = make_regular_config();
    let mut pod = pod_with_annotations("w", &[("smg.ai/worker-ports", "8080,8081")]);
    if let Some(status) = pod.status.as_mut() {
        status.conditions = Some(vec![ready_condition("False")]);
    }
    let desired = compute_desired_state(&store_snapshot(vec![pod]), &config);
    assert!(desired.workers.is_empty());
}

#[test]
fn test_compute_desired_state_unknown_readiness_excluded() {
    let config = make_regular_config();
    let mut pod = make_labeled_pod("w", "10.0.0.1", &[("app", "sglang")]);
    if let Some(status) = pod.status.as_mut() {
        status.conditions = Some(vec![ready_condition("Unknown")]);
    }
    let desired = compute_desired_state(&store_snapshot(vec![pod]), &config);
    assert!(desired.workers.is_empty());
}

/// A Ready Pod that is not Running used to sit in a hold state: protected
/// from removal, yet never added. The shared reconciler has no such state,
/// so the Pod is simply not published and a worker registered for it is
/// removed.
#[test]
fn test_ready_but_not_running_pod_is_not_published() {
    let config = make_regular_config();
    let mut pod = make_labeled_pod("w", "10.0.0.1", &[("app", "sglang")]);
    if let Some(status) = pod.status.as_mut() {
        status.phase = Some("Pending".to_string());
    }
    let desired = compute_desired_state(&store_snapshot(vec![pod]), &config);
    assert!(desired.workers.is_empty());

    let registered = [owned("10.0.0.1:8000", "uid-w:8000")];
    let actions = reconciler::compute_actions(&desired, &registered);
    assert_eq!(actions.remove.len(), 1);
    assert!(actions.add.is_empty());
}

#[test]
fn test_compute_desired_state_ignores_non_matching_pods() {
    let config = make_regular_config();
    let pod = make_labeled_pod("w", "10.0.0.1", &[("app", "other")]);
    let desired = compute_desired_state(&store_snapshot(vec![pod]), &config);
    assert!(desired.workers.is_empty());
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
    assert_eq!(desired.workers.len(), 2);
    let by_url: HashMap<String, &DiscoveredWorker> = desired
        .workers
        .values()
        .map(|w| (w.endpoint.render(), w))
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
        desired.workers.values().next().unwrap().model_id_override,
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
    assert_eq!(desired.workers.len(), 2);
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
    let mut labels = BTreeMap::new();
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
    let mut annotations = BTreeMap::new();
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
        selector,
        disaggregated_mode: false,
        ..Default::default()
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
    let config = create_pd_config();
    // Only the common label should be in the selector.
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
    let config = create_pd_config();
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

#[traced_test]
#[test]
fn test_build_watcher_config_logs_selector_at_info_level() {
    let _ = build_watcher_config("worker", "app=sglang");
    assert!(logs_contain("Starting K8s worker watcher"));
    assert!(logs_contain("app=sglang"));
}
