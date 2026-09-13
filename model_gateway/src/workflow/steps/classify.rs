//! Worker type classification step.
//!
//! Classifies a worker as Local or External based on the `runtime_type` field
//! and URL-based heuristics. Only `RuntimeType::External` or known cloud
//! provider URLs (OpenAI, Anthropic, xAI, Gemini) yield an external worker.
//! When `Unspecified` (the default) and the URL is not a known provider,
//! the step probes the endpoint and defaults to Local.

use std::time::Duration;

use async_trait::async_trait;
use openai_protocol::worker::ProviderType;
use reqwest::Client;
use tracing::debug;
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use super::util::{http_base_url, try_grpc_reachable, try_http_reachable};
use crate::{
    routers::provider_support,
    worker::worker::RuntimeType,
    workflow::data::{WorkerKind, WorkerWorkflowData},
};

/// Quick-probe timeout for classification. Deliberately short — the full
/// connection timeout is applied later by `DetectConnectionModeStep`.
const CLASSIFY_PROBE_TIMEOUT_SECS: u64 = 2;

/// Known local backend `owned_by` values returned by `/v1/models`.
const LOCAL_OWNED_BY: &[&str] = &["sglang", "vllm", "trtllm", "nvidia"];

/// Fetch `/v1/models` and check the `owned_by` field of the first model.
/// Returns `Some("sglang")`, `Some("vllm")`, etc. if recognized as a local
/// backend, or `None` if the response is missing, not parsable, or the
/// `owned_by` value does not match a known local backend.
async fn probe_models_owned_by(
    url: &str,
    timeout_secs: u64,
    client: &Client,
    api_key: Option<&str>,
) -> Option<String> {
    let base = http_base_url(url);
    let models_url = format!("{base}/v1/models");
    let mut req = client
        .get(&models_url)
        .timeout(Duration::from_secs(timeout_secs));
    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }
    let resp = req.send().await.ok()?;
    if resp.status().is_server_error() {
        return None;
    }
    let body: serde_json::Value = resp.json().await.ok()?;
    let owned_by = body
        .get("data")?
        .as_array()?
        .first()?
        .get("owned_by")?
        .as_str()?
        .to_lowercase();
    if LOCAL_OWNED_BY.iter().any(|&local| owned_by == local) {
        Some(owned_by)
    } else {
        None
    }
}

/// Step 0: Classify the worker as Local or External.
///
/// Detection logic:
/// 1. Any explicit runtime → classify immediately (External or Local)
/// 2. URL matches known cloud provider (OpenAI, Anthropic, xAI, Gemini) → External
/// 3. `/health` responds → Local (only local backends expose `/health`)
/// 4. gRPC health responds → Local (external APIs never use gRPC)
/// 5. `/v1/models` responds with `owned_by` matching a local backend → Local
/// 6. Nothing conclusive → default Local (backend may still be starting)
///
/// Note: external providers on private IPs (e.g., a proxy to OpenAI) must set
/// `runtime_type: external` explicitly — URL-based detection cannot identify them.
pub struct ClassifyWorkerTypeStep;

#[async_trait]
impl StepExecutor<WorkerWorkflowData> for ClassifyWorkerTypeStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let config = &context.data.config;

        // A provider target needs the provider's router, which exists only in
        // a build that compiled it in; a worker nothing could route to must
        // not enter the registry.
        if let Some(missing) = provider_support::missing_router(config) {
            return Err(WorkflowError::StepFailed {
                step_id: StepId::new("classify_worker_type"),
                message: format!(
                    "worker {} targets the {} provider, but this build carries no {} router; \
                     rebuild with the `{}` Cargo feature to admit it",
                    config.url, missing.label, missing.label, missing.feature
                ),
            });
        }

        // 1. Any explicit runtime → classify immediately, no probing needed
        if config.runtime_type.is_specified() {
            let kind = if config.runtime_type == RuntimeType::External {
                WorkerKind::External
            } else {
                WorkerKind::Local
            };
            debug!(
                "Worker {} explicitly configured as {} → {:?}",
                config.url, config.runtime_type, kind
            );
            context.data.worker_kind = Some(kind);
            return Ok(StepResult::Success);
        }

        // 3. URL matches known cloud provider → External (no probing needed)
        if let Some(provider) = ProviderType::from_url(&config.url) {
            debug!(
                "Worker {} URL matches known provider ({}) → External",
                config.url, provider
            );
            context.data.worker_kind = Some(WorkerKind::External);
            return Ok(StepResult::Success);
        }

        // Unspecified + unknown URL — probe the endpoint
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        let timeout = CLASSIFY_PROBE_TIMEOUT_SECS;
        let client = &app_context.client;

        // 4. /health → Local (only local backends expose this)
        if try_http_reachable(&config.url, timeout, client)
            .await
            .is_ok()
        {
            debug!("Worker {} responded to /health → Local", config.url);
            context.data.worker_kind = Some(WorkerKind::Local);
            return Ok(StepResult::Success);
        }

        // 5. gRPC health → Local (external APIs never use gRPC)
        if try_grpc_reachable(&config.url, timeout).await.is_ok() {
            debug!("Worker {} responded to gRPC health → Local", config.url);
            context.data.worker_kind = Some(WorkerKind::Local);
            return Ok(StepResult::Success);
        }

        // 6. /v1/models with recognized local owned_by → Local
        if let Some(owned_by) =
            probe_models_owned_by(&config.url, timeout, client, config.api_key.as_deref()).await
        {
            debug!(
                "Worker {} /v1/models owned_by={} → Local",
                config.url, owned_by
            );
            context.data.worker_kind = Some(WorkerKind::Local);
            return Ok(StepResult::Success);
        }

        // 7. Nothing conclusive — assume Local (backend may still be starting;
        // detect_connection_mode will retry with the full startup timeout).
        debug!(
            "Worker {} not reachable on any probe → defaulting to Local",
            config.url
        );
        context.data.worker_kind = Some(WorkerKind::Local);
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use axum::{routing::get, Json, Router};
    use reqwest::Client;
    use serde_json::json;
    use tokio::net::TcpListener;

    use super::probe_models_owned_by;

    #[tokio::test]
    async fn probe_models_owned_by_accepts_nvidia_as_local() {
        async fn models() -> Json<serde_json::Value> {
            Json(json!({
                "data": [{
                    "id": "test-model",
                    "object": "model",
                    "owned_by": "nvidia"
                }],
                "object": "list"
            }))
        }

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        #[expect(
            clippy::disallowed_methods,
            reason = "test-only mock /v1/models server; handle is aborted at test end"
        )]
        let server = tokio::spawn(async move {
            axum::serve(listener, Router::new().route("/v1/models", get(models)))
                .await
                .unwrap();
        });

        let owned_by = probe_models_owned_by(&format!("http://{addr}"), 5, &Client::new(), None)
            .await
            .unwrap();
        server.abort();

        assert_eq!(owned_by, "nvidia");
    }
}
