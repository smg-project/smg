//! Shared test fixtures for the discovery modules.

use std::{collections::BTreeMap, sync::Arc};

use k8s_openapi::{
    api::core::v1::{Pod, PodCondition, PodSpec, PodStatus},
    apimachinery::pkg::apis::meta::v1::ObjectMeta,
};

use crate::app_context::AppContext;

/// An `AppContext` with an uninitialized worker job queue: submitted jobs are
/// accepted and queued but never processed, so tests observe what discovery
/// *submits* without spawning workflow background tasks. [`live_app_context`]
/// is the one whose jobs run.
pub(crate) fn create_test_app_context() -> Arc<AppContext> {
    use crate::{app_context::testing::app_context_for_tests, config::RouterConfig};

    let router_config = RouterConfig::builder()
        .worker_startup_timeout_secs(1)
        .build_unchecked();

    // Note: Using uninitialized queue for tests to avoid spawning background workers
    // Jobs submitted during tests will queue but not be processed
    app_context_for_tests(
        router_config,
        Arc::new(crate::worker::WorkerRegistry::new()),
        None,
    )
}

/// An `AppContext` whose jobs run: a live job queue and workflow engines, so
/// tests observe what the registry *ends up with* after discovery's passes.
/// Regular routing with round-robin, a five-second startup window for an
/// engine slow to answer its probes, and no drain wait on removal.
/// [`create_test_app_context`] is the one whose jobs never run.
pub(crate) fn live_app_context() -> Arc<AppContext> {
    use std::sync::OnceLock;

    use llm_tokenizer::registry::TokenizerRegistry;
    use smg_data_connector::{
        MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
    };

    use crate::{
        config::{HealthCheckConfig, PolicyConfig, RouterConfig, RoutingMode},
        policies::PolicyRegistry,
        worker::WorkerRegistry,
        workflow::{JobQueue, JobQueueConfig, WorkflowEngines},
    };

    let config = RouterConfig::builder()
        .mode(RoutingMode::Regular {
            worker_urls: vec![],
        })
        .policy(PolicyConfig::RoundRobin)
        .worker_startup_timeout_secs(5)
        .worker_startup_check_interval_secs(1)
        .health_check_config(HealthCheckConfig {
            drain_settle_secs: 0,
            ..HealthCheckConfig::default()
        })
        .build_unchecked();
    let context = Arc::new(
        AppContext::builder()
            .router_config(config.clone())
            .client(crate::server::http_client())
            .tokenizer_registry(Arc::new(TokenizerRegistry::new()))
            .worker_registry(Arc::new(WorkerRegistry::new()))
            .policy_registry(Arc::new(PolicyRegistry::new(config.policy.clone())))
            .response_storage(Arc::new(MemoryResponseStorage::new()))
            .conversation_storage(Arc::new(MemoryConversationStorage::new()))
            .conversation_item_storage(Arc::new(MemoryConversationItemStorage::new()))
            .worker_job_queue(Arc::new(OnceLock::new()))
            .workflow_engines(Arc::new(OnceLock::new()))
            .mcp_orchestrator(Arc::new(OnceLock::new()))
            .build()
            .unwrap(),
    );
    context
        .worker_job_queue
        .set(JobQueue::new(
            JobQueueConfig::default(),
            Arc::downgrade(&context),
        ))
        .ok();
    context
        .workflow_engines
        .set(WorkflowEngines::new(&context.router_config))
        .ok();
    context
}

/// A running, ready pod named `name` at `ip`, carrying `labels`: the shape
/// the Kubernetes watchers' tests feed to their label selectors.
pub(crate) fn make_labeled_pod(name: &str, ip: &str, labels: &[(&str, &str)]) -> Pod {
    let mut label_map = BTreeMap::new();
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
