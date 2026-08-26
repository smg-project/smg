//! Shared worker selection for all routers.
//!
//! Single public API: [`WorkerSelector::select_worker`].

use std::{sync::Arc, time::Duration};

use axum::{
    http::{HeaderMap, HeaderValue},
    response::Response,
};
use futures_util::future::join_all;
use openai_protocol::models::ListModelsResponse;

use crate::{
    routers::{
        common::{
            header_utils::{apply_provider_headers, extract_auth_header},
            overload,
        },
        error,
    },
    worker::{ConnectionMode, ProviderType, RuntimeType, Worker, WorkerRegistry, WorkerType},
};

/// Holds references to shared infrastructure needed for worker selection.
///
/// Created once per router (or per-request where lifetimes differ) and
/// reused across calls.
pub struct WorkerSelector<'a> {
    registry: &'a WorkerRegistry,
    client: &'a reqwest::Client,
}

/// Input for [`WorkerSelector::select_worker`].
///
/// Combines the model to resolve with optional registry filters and
/// the caller's HTTP headers (used for auth passthrough during
/// upstream model refresh).
#[derive(Debug, Default)]
pub struct SelectWorkerRequest<'a> {
    /// Model ID to select a worker for (required).
    pub model_id: &'a str,

    /// Caller's HTTP headers — used to extract the auth token for
    /// upstream `/v1/models` refresh on cache miss.
    pub headers: Option<&'a HeaderMap>,

    /// Provider-based security filtering for multi-provider setups.
    /// When set, prevents credentials from leaking to workers of a
    /// different provider (e.g. Anthropic key to OpenAI worker).
    pub provider: Option<ProviderType>,

    /// Filter by worker type (Regular, Prefill, Decode). `None` = any.
    pub worker_type: Option<WorkerType>,

    /// Filter by connection mode (Http, Grpc). `None` = any.
    pub connection_mode: Option<ConnectionMode>,

    /// Filter by runtime type (External, Sglang, Vllm, Trtllm). `None` = any.
    pub runtime_type: Option<RuntimeType>,

    /// When `true`, restrict candidates to workers advertising realtime
    /// capability (the `realtime` label). Used by the realtime routes so
    /// they never proxy to a worker that can't serve realtime.
    pub require_realtime_capable: bool,
}

impl<'a> WorkerSelector<'a> {
    pub fn new(registry: &'a WorkerRegistry, client: &'a reqwest::Client) -> Self {
        Self { registry, client }
    }

    fn matches_worker_filters(worker: &Arc<dyn Worker>, req: &SelectWorkerRequest<'_>) -> bool {
        req.worker_type
            .is_none_or(|worker_type| *worker.worker_type() == worker_type)
            && req
                .connection_mode
                .is_none_or(|mode| *worker.connection_mode() == mode)
            && req
                .runtime_type
                .is_none_or(|runtime| worker.metadata().spec.runtime_type == runtime)
    }

    /// Select the best worker for a model with refresh-on-miss.
    ///
    /// 1. Filter available workers by the request criteria.
    /// 2. Pick the least-loaded worker that supports the model.
    /// 3. On miss, refresh external model lists by calling `/v1/models`
    ///    on all external workers (vendor-aware parsing), then retry.
    /// 4. Return an error distinguishing "model not found" from "all
    ///    workers circuit-broken".
    pub async fn select_worker(
        &self,
        req: &SelectWorkerRequest<'_>,
    ) -> Result<Arc<dyn Worker>, Response> {
        if let Some(worker) = self.find_best_worker(req) {
            return Ok(worker);
        }

        // Shed before the refresh, not after. Refresh-on-miss is the expensive
        // branch — a second registry walk plus a `/v1/models` fan-out under a
        // 5 s timeout — and a fleet whose every worker is vetoed will not be
        // un-vetoed by re-reading model lists. Without this, saturation turns
        // each of these requests into three registry walks and up to 5 s of
        // network wait to reach a 503 that carries neither the shed error code
        // nor the shed counter.
        if let Some(shed) = self.shed_if_all_overloaded(req) {
            return Err(shed);
        }

        tracing::debug!(
            model = req.model_id,
            "No worker found, refreshing external worker models"
        );

        let auth = extract_auth_header(req.headers, None);
        self.refresh_external_models(auth.as_ref(), req.provider.as_ref())
            .await;

        self.find_best_worker(req).ok_or_else(|| {
            if self.any_worker_supports_model(req) {
                error::service_unavailable(
                    "service_unavailable",
                    format!(
                        "All workers for model '{}' are temporarily unavailable",
                        req.model_id
                    ),
                )
            } else {
                error::model_not_found(req.model_id)
            }
        })
    }

    /// The pool selection walks. `require_available` adds the `is_available()`
    /// veto; the shed path takes the same pool without it, so the shed verdict
    /// always describes exactly what selection saw.
    fn candidate_pool(
        &self,
        req: &SelectWorkerRequest<'_>,
        require_available: bool,
    ) -> Vec<Arc<dyn Worker>> {
        let workers: Vec<_> = self
            .registry
            .get_routing_workers()
            .iter()
            .filter(|worker| Self::matches_worker_filters(worker, req))
            .filter(|worker| !require_available || worker.is_available())
            .cloned()
            .collect();
        let candidates = match &req.provider {
            Some(provider) => filter_by_provider(workers, provider),
            None => workers,
        };
        candidates
            .into_iter()
            .filter(|w| w.supports_model(req.model_id))
            .filter(|w| !req.require_realtime_capable || w.is_realtime_capable())
            .collect()
    }

    fn find_best_worker(&self, req: &SelectWorkerRequest<'_>) -> Option<Arc<dyn Worker>> {
        self.candidate_pool(req, true)
            .into_iter()
            .min_by_key(|w| w.load())
    }

    /// Shed when every worker this request could have selected is vetoed.
    /// Runs only on the miss path.
    fn shed_if_all_overloaded(&self, req: &SelectWorkerRequest<'_>) -> Option<Response> {
        let candidates = self.candidate_pool(req, false);
        overload::shed_if_all_overloaded(&candidates, req.model_id)
    }

    /// Check if any healthy worker supports the model (regardless of circuit breaker).
    /// Used to distinguish "model not found" from "all workers circuit-broken".
    fn any_worker_supports_model(&self, req: &SelectWorkerRequest<'_>) -> bool {
        let workers: Vec<_> = self
            .registry
            .get_routing_workers()
            .iter()
            .filter(|worker| Self::matches_worker_filters(worker, req))
            .filter(|worker| worker.is_healthy())
            .cloned()
            .collect();
        let candidates = match &req.provider {
            Some(p) => filter_by_provider(workers, p),
            None => workers,
        };
        candidates.iter().any(|w| {
            w.supports_model(req.model_id)
                && (!req.require_realtime_capable || w.is_realtime_capable())
        })
    }

    /// Refresh model lists for healthy external workers in parallel.
    ///
    /// When `provider` is set, only workers matching that provider are refreshed
    /// to prevent credential leakage across providers. Each worker falls back to
    /// its own configured API key when the caller provides no auth.
    async fn refresh_external_models(
        &self,
        auth_header: Option<&HeaderValue>,
        provider: Option<&ProviderType>,
    ) {
        let mut external_workers: Vec<_> = self
            .registry
            .get_routing_workers()
            .iter()
            .filter(|worker| {
                worker.metadata().spec.runtime_type == RuntimeType::External && worker.is_healthy()
            })
            .cloned()
            .collect();

        // Only refresh workers matching the request's provider to avoid sending
        // e.g. an OpenAI key to Anthropic workers during model discovery.
        if let Some(p) = provider {
            external_workers.retain(|w| matches!(w.default_provider(), Some(wp) if wp == p));
        }

        if external_workers.is_empty() {
            return;
        }

        tracing::debug!(
            "Refreshing models for {} external workers",
            external_workers.len()
        );

        let futures: Vec<_> = external_workers
            .iter()
            .map(|w| refresh_worker_models(self.client, w, auth_header))
            .collect();

        // Timeout prevents a slow/unresponsive worker from blocking all
        // requests that trigger refresh-on-miss.
        const REFRESH_TIMEOUT: Duration = Duration::from_secs(5);
        let _ = tokio::time::timeout(REFRESH_TIMEOUT, join_all(futures)).await;
    }
}

/// In multi-provider setups, filter to only workers matching the target provider.
/// In single-provider (or no-provider) setups, returns all workers unchanged.
fn filter_by_provider(
    workers: Vec<Arc<dyn Worker>>,
    target: &ProviderType,
) -> Vec<Arc<dyn Worker>> {
    let mut first_provider: Option<Option<ProviderType>> = None;
    let has_multiple_providers = workers.iter().any(|w| {
        let provider = w.default_provider().cloned();
        match first_provider {
            None => {
                first_provider = Some(provider);
                false
            }
            Some(ref first) => *first != provider,
        }
    });

    if has_multiple_providers {
        workers
            .into_iter()
            .filter(|w| matches!(w.default_provider(), Some(p) if p == target))
            .collect()
    } else {
        workers
    }
}

/// Refresh a single worker's model list by calling its `/v1/models` endpoint.
///
/// Auth headers are adapted per-vendor via [`apply_provider_headers`] (e.g.
/// Anthropic uses `x-api-key`, OpenAI uses `Authorization: Bearer`). The
/// response is parsed via [`ListModelsResponse::parse_upstream`].
async fn refresh_worker_models(
    client: &reqwest::Client,
    worker: &Arc<dyn Worker>,
    auth_header: Option<&HeaderValue>,
) -> bool {
    let url = format!("{}/v1/models", worker.url());
    let mut backend_req = client.get(&url);

    // Use caller's auth if provided, otherwise fall back to worker's configured API key.
    // This matches how auth is handled in request routing (e.g. openai/router.rs).
    let worker_auth = auth_header.cloned().or_else(|| {
        worker
            .api_key()
            .and_then(|k| HeaderValue::from_str(&format!("Bearer {k}")).ok())
    });
    if let Some(ref auth) = worker_auth {
        backend_req = apply_provider_headers(backend_req, &url, Some(auth));
    }

    match backend_req.send().await {
        Ok(response) if response.status().is_success() => {
            match response.json::<serde_json::Value>().await {
                Ok(json) => {
                    let provider = ProviderType::from_url(&url);
                    let model_cards = ListModelsResponse::parse_upstream(&json, provider);

                    if !model_cards.is_empty() {
                        tracing::info!(
                            "Model refresh: found {} models from {}",
                            model_cards.len(),
                            url
                        );
                        worker.set_models(model_cards);
                        return true;
                    }
                    false
                }
                Err(e) => {
                    tracing::warn!("Failed to parse models response: {}", e);
                    false
                }
            }
        }
        Ok(response) => {
            tracing::debug!(
                "Model refresh returned non-success status {} from {}",
                response.status(),
                url
            );
            false
        }
        Err(e) => {
            tracing::warn!("Failed to fetch models from backend: {}", e);
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::worker::HealthCheckConfig;

    use super::*;
    use crate::worker::BasicWorkerBuilder;

    fn no_health_check() -> HealthCheckConfig {
        HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        }
    }

    // Wildcard model support (no `.models()`), so `supports_model` is always true;
    // this isolates the realtime-capability gate.
    fn worker(url: &str, realtime: bool) -> Arc<dyn Worker> {
        let mut b = BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check());
        if realtime {
            b = b.label("realtime", "true");
        }
        Arc::new(b.build())
    }

    #[tokio::test]
    async fn requires_realtime_selects_only_labeled() {
        let registry = WorkerRegistry::new();
        registry.register_or_replace(worker("http://127.0.0.1:18080", false));
        registry.register_or_replace(worker("http://127.0.0.1:18081", true));
        let client = reqwest::Client::new();

        let picked = WorkerSelector::new(&registry, &client)
            .select_worker(&SelectWorkerRequest {
                model_id: "m",
                require_realtime_capable: true,
                ..Default::default()
            })
            .await
            .expect("a realtime-capable worker should be selected");
        assert_eq!(picked.url(), "http://127.0.0.1:18081");
    }

    #[tokio::test]
    async fn requires_realtime_errors_when_none_capable() {
        let registry = WorkerRegistry::new();
        registry.register_or_replace(worker("http://127.0.0.1:18080", false));
        let client = reqwest::Client::new();

        let res = WorkerSelector::new(&registry, &client)
            .select_worker(&SelectWorkerRequest {
                model_id: "m",
                require_realtime_capable: true,
                ..Default::default()
            })
            .await;
        assert!(
            res.is_err(),
            "no realtime-capable worker => selection fails"
        );
    }

    #[tokio::test]
    async fn without_realtime_flag_any_worker_eligible() {
        let registry = WorkerRegistry::new();
        registry.register_or_replace(worker("http://127.0.0.1:18080", false));
        let client = reqwest::Client::new();

        let res = WorkerSelector::new(&registry, &client)
            .select_worker(&SelectWorkerRequest {
                model_id: "m",
                ..Default::default()
            })
            .await;
        assert!(res.is_ok(), "gate off => a plain worker is eligible");
    }

    #[test]
    fn default_request_does_not_require_realtime() {
        assert!(!SelectWorkerRequest::default().require_realtime_capable);
    }
}
