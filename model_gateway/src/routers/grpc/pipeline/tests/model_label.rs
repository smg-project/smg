//! The model label of the pipeline's request and error series: the served model's id, or `other`
//! for a model no worker serves.

use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};

use super::*;
use crate::{
    config::types::PolicyConfig,
    observability::metrics::test_support::render_with_recorder,
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerType},
};

/// smg-lab #288: the pipeline's request and error records carry the
/// served model's id, or `other` for a model no worker serves, so a
/// client cannot mint series with unknown names.
#[test]
fn the_pipeline_counts_an_unserved_model_under_other() {
    let worker_registry = Arc::new(WorkerRegistry::new());
    let worker = BasicWorkerBuilder::new("grpc://worker:30000")
        .worker_type(WorkerType::Regular)
        .connection_mode(ConnectionMode::Grpc)
        .runtime_type(RuntimeType::Sglang)
        .model(ModelCard::new("m1"))
        .health_config(HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        })
        .build();
    worker_registry.register(Arc::new(worker)).unwrap();
    let deps = PipelineDeps::pair(
        worker_registry,
        Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
        None,
        None,
        Duration::from_secs(60),
    );
    let pipeline =
        RequestPipeline::build(Endpoint::Chat, Mode::Regular, &deps).expect("chat pipeline");
    let rendered = render_with_recorder(|| {
        pipeline.record_request(metrics_labels::ENDPOINT_CHAT, "ghost-1", false);
        pipeline.record_request(metrics_labels::ENDPOINT_CHAT, "m1", false);
        pipeline.record_error(
            Some(metrics_labels::ENDPOINT_CHAT),
            "ghost-1",
            &error::model_not_found("ghost-1"),
        );
    });
    assert!(!rendered.contains("ghost-"), "{rendered}");
    let has = |family: &str, label: &str| {
        rendered
            .lines()
            .any(|line| line.starts_with(&format!("{family}{{")) && line.contains(label))
    };
    assert!(
        has("smg_router_requests_total", r#"model="other""#),
        "{rendered}"
    );
    assert!(
        has("smg_router_requests_total", r#"model="m1""#),
        "{rendered}"
    );
    assert!(
        has("smg_router_request_errors_total", r#"model="other""#),
        "{rendered}"
    );
}
