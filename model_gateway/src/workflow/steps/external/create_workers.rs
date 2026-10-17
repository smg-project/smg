//! External worker creation step.

use std::{collections::HashMap, sync::Arc, time::Duration};

use async_trait::async_trait;
use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
use tracing::{debug, info};
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::{
    worker::{
        circuit_breaker::CircuitBreakerConfig,
        resilience::{resolve_resilience, ResolvedResilience},
        worker::{RuntimeType, WorkerType},
        BasicWorker, BasicWorkerBuilder, ConnectionMode, Worker,
    },
    workflow::data::{WorkerKind, WorkerList, WorkerWorkflowData},
};

/// Normalize URL for external APIs (ensure https://).
fn normalize_external_url(url: &str) -> String {
    if url.starts_with("http://") || url.starts_with("https://") {
        url.to_string()
    } else {
        format!("https://{url}")
    }
}

/// What the external worker is built from, once the step has resolved it from
/// the request and the router's settings.
struct ExternalWorkerParts<'a> {
    /// The endpoint, normalized to `https://`.
    url: &'a str,
    /// The discovered models; none for a wildcard worker, which accepts any
    /// model and forwards the user's auth.
    model_cards: &'a [ModelCard],
    api_key: Option<&'a str>,
    labels: &'a HashMap<String, String>,
    priority: u32,
    cost: f32,
    http_client: Arc<reqwest::Client>,
    http2: bool,
    resilience: ResolvedResilience,
    circuit_breaker: CircuitBreakerConfig,
    health: HealthCheckConfig,
    health_endpoint: &'a str,
}

/// The one worker for an external endpoint: with its models when there are
/// any, else a wildcard worker. The builder sets the initial status: Pending
/// if health-checked, Ready if not.
fn build_external_worker(parts: ExternalWorkerParts<'_>) -> BasicWorker {
    // No cards is the wildcard worker: the builder reads an empty list so.
    let mut builder = BasicWorkerBuilder::new(parts.url)
        .models(parts.model_cards.to_vec())
        .worker_type(WorkerType::Regular)
        .connection_mode(ConnectionMode::Http)
        .runtime_type(RuntimeType::External)
        .circuit_breaker_config(parts.circuit_breaker)
        .http_client(parts.http_client)
        .http2(parts.http2)
        .resilience(parts.resilience)
        .health_config(parts.health)
        .health_endpoint(parts.health_endpoint)
        .priority(parts.priority)
        .cost(parts.cost);

    if let Some(api_key) = parts.api_key {
        builder = builder.api_key(api_key);
    }

    if !parts.labels.is_empty() {
        builder = builder.labels(parts.labels.clone());
    }

    builder.build()
}

/// Step 2: Create a worker object for the external endpoint.
pub struct CreateExternalWorkersStep;

#[async_trait]
impl StepExecutor<WorkerWorkflowData> for CreateExternalWorkersStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        if context.data.worker_kind != Some(WorkerKind::External) {
            return Ok(StepResult::Skip);
        }

        let config = &context.data.config;
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        let model_cards = &context.data.model_cards;

        // Build configs from router settings — resolve per-worker resilience
        let base_retry = app_context.router_config.effective_retry_config();
        let base_cb = {
            let cfg = app_context.router_config.effective_circuit_breaker_config();
            CircuitBreakerConfig {
                failure_threshold: cfg.failure_threshold,
                success_threshold: cfg.success_threshold,
                timeout_duration: Duration::from_secs(cfg.timeout_duration_secs),
                window_duration: Duration::from_secs(cfg.window_duration_secs),
            }
        };

        let (resolved_resilience, circuit_breaker_config) = resolve_resilience(
            &base_retry,
            &base_cb,
            !app_context.router_config.disable_retries,
            !app_context.router_config.disable_circuit_breaker,
            &config.resilience,
        );

        // TLS negotiates the HTTP version via ALPN; only an explicit
        // `http_pool.http2` pins it.
        let http2 = config.http_pool.http2.unwrap_or(false);
        let http_client = app_context
            .worker_client_cache
            .get(&config.http_pool, http2)
            .map_err(|e| WorkflowError::StepFailed {
                step_id: StepId::new("create_external_workers"),
                message: e,
            })?;

        let (health_config, health_endpoint) = {
            let base = app_context.router_config.health_check.to_protocol_config();
            let mut merged = config.health.apply_to(&base);
            // External workers (OpenAI, Anthropic, etc.) should not be health-checked
            // by default — they are third-party APIs that don't expose a /health endpoint.
            // Only apply the default if the user hasn't explicitly overridden it.
            if config.health.disable_health_check.is_none() {
                merged.disable_health_check = true;
            }
            (
                merged,
                app_context.router_config.health_check.endpoint.clone(),
            )
        };

        // Build labels from config
        let labels: HashMap<String, String> = config.labels.clone();

        // Normalize URL (ensure https:// for external APIs)
        let normalized_url = normalize_external_url(&config.url);

        if model_cards.is_empty() {
            debug!("Creating wildcard worker (no models) for {}", config.url);
        } else {
            debug!(
                "Creating one external worker with {} discovered models for {}",
                model_cards.len(),
                config.url
            );
        }

        let worker = Arc::new(build_external_worker(ExternalWorkerParts {
            url: &normalized_url,
            model_cards,
            api_key: config.api_key.as_deref(),
            labels: &labels,
            priority: config.priority,
            cost: config.cost,
            http_client,
            http2,
            resilience: resolved_resilience,
            circuit_breaker: circuit_breaker_config,
            health: health_config,
            health_endpoint: &health_endpoint,
        })) as Arc<dyn Worker>;

        if model_cards.is_empty() {
            info!(
                "Created wildcard worker at {} (accepts any model, user auth forwarded)",
                normalized_url
            );
        } else {
            info!(
                "Created external worker at {} with {} discovered models",
                normalized_url,
                model_cards.len()
            );
        }

        let workers = vec![worker];

        // Store results in workflow data
        context.data.workers = Some(WorkerList::from_workers(&workers));
        context.data.actual_workers = Some(workers);
        context.data.final_labels = labels;
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parts<'a>(
        cards: &'a [ModelCard],
        labels: &'a HashMap<String, String>,
        client: &Arc<reqwest::Client>,
    ) -> ExternalWorkerParts<'a> {
        ExternalWorkerParts {
            url: "https://api.example.com",
            model_cards: cards,
            api_key: Some("key"),
            labels,
            priority: 3,
            cost: 1.5,
            http_client: Arc::clone(client),
            http2: true,
            resilience: ResolvedResilience::default(),
            circuit_breaker: CircuitBreakerConfig::default(),
            health: HealthCheckConfig::default(),
            health_endpoint: "/health",
        }
    }

    /// The wildcard worker and the worker with discovered models are the same
    /// worker but for the models: one build, one set of settings.
    #[test]
    fn the_worker_is_built_the_same_way_with_and_without_model_cards() {
        let client = Arc::new(crate::http_client::client());
        let labels = HashMap::from([("tier".to_string(), "gold".to_string())]);
        let cards = [ModelCard::new("m1"), ModelCard::new("m2")];
        let wildcard = build_external_worker(parts(&[], &labels, &client));
        let listed = build_external_worker(parts(&cards, &labels, &client));

        assert!(wildcard.metadata().is_wildcard());
        assert_eq!(wildcard.metadata().model_ids().count(), 0);
        assert!(!listed.metadata().is_wildcard());
        assert_eq!(
            listed.metadata().model_ids().collect::<Vec<_>>(),
            ["m1", "m2"]
        );

        for worker in [&wildcard, &listed] {
            let spec = &worker.metadata().spec;
            assert_eq!(worker.url(), "https://api.example.com");
            assert_eq!(spec.worker_type, WorkerType::Regular);
            assert_eq!(spec.connection_mode, ConnectionMode::Http);
            assert_eq!(spec.runtime_type, RuntimeType::External);
            assert_eq!(worker.api_key().map(String::as_str), Some("key"));
            assert_eq!(worker.priority(), 3);
            assert_eq!(worker.cost(), 1.5);
            assert_eq!(spec.labels, labels);
            assert_eq!(worker.metadata().health_endpoint, "/health");
        }
    }
}
