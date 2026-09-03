//! Pre-pipeline model validation shared by every gRPC entry point.

use std::sync::Arc;

use axum::response::Response;

use crate::{routers::error, worker::WorkerRegistry};

/// Validate that workers are available for the requested model.
///
/// Runs on the client-supplied name, before the pipeline canonicalizes it, so
/// it has to accept aliases as well as canonical model IDs. `contains_model`
/// covers both; listing `get_models()` and testing membership would reject
/// every alias here.
pub(crate) fn validate_worker_availability(
    worker_registry: &Arc<WorkerRegistry>,
    model: &str,
) -> Option<Response> {
    if !worker_registry.contains_model(model) {
        return Some(error::model_not_found(model));
    }

    None
}

#[cfg(test)]
mod tests {
    use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};

    use super::*;
    use crate::worker::{BasicWorkerBuilder, UNKNOWN_MODEL_ID};

    fn registry_with_aliased_worker() -> Arc<WorkerRegistry> {
        let registry = Arc::new(WorkerRegistry::new());
        let worker = BasicWorkerBuilder::new("http://worker:8080")
            .model(ModelCard::new("canonical-model").with_alias("model-alias"))
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build();
        registry.register_or_replace(Arc::new(worker));
        registry
    }

    #[test]
    fn worker_availability_accepts_alias_and_preserves_unknown_rejection() {
        let registry = registry_with_aliased_worker();

        assert!(validate_worker_availability(&registry, "canonical-model").is_none());
        assert!(validate_worker_availability(&registry, "model-alias").is_none());

        let response = validate_worker_availability(&registry, UNKNOWN_MODEL_ID)
            .expect("unknown model should remain rejected for Responses");
        assert_eq!(response.status(), http::StatusCode::NOT_FOUND);
    }

    #[test]
    fn worker_availability_rejects_alias_once_its_worker_is_gone() {
        let registry = registry_with_aliased_worker();
        let worker_id = registry.get_id_by_url("http://worker:8080").unwrap();
        assert!(registry.remove(&worker_id).is_some());

        let response = validate_worker_availability(&registry, "model-alias")
            .expect("alias must stop resolving with no workers behind it");
        assert_eq!(response.status(), http::StatusCode::NOT_FOUND);
    }
}
