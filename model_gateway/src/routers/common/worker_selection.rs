//! Shared worker selection for all routers.
//!
//! Single public API: [`WorkerSelector::select_worker`].

use std::{sync::Arc, time::Duration};

use axum::response::Response;
use futures_util::future::join_all;
use smg_external_router::ExternalRouterSpec;

use crate::{
    routers::{common::overload, error, provider_support},
    worker::{overload::STAGE_SELECTION, ProviderType, RuntimeType, Worker, WorkerRegistry},
    workflow::steps::external::{fetch_models, resolve_discovery_api_key},
};

/// Holds references to shared infrastructure needed for worker selection.
///
/// Created once per router (or per-request where lifetimes differ) and
/// reused across calls.
pub struct WorkerSelector<'a> {
    registry: &'a WorkerRegistry,
}

pub use smg_external_router::worker::SelectWorkerRequest;

impl<'a> WorkerSelector<'a> {
    pub fn new(registry: &'a WorkerRegistry) -> Self {
        Self { registry }
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
    /// 3. On miss, re-discover the external workers' models with the key the
    ///    operator configured for each (registration's discovery, run again,
    ///    at most once per [`REFRESH_MIN_INTERVAL_MS`] per worker), then
    ///    retry.
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
        if let Some(verdict) = self.all_overloaded_verdict(req) {
            return verdict;
        }

        tracing::debug!(
            model = req.model_id,
            "No worker found, refreshing external worker models"
        );

        self.refresh_external_models(req.router.as_ref()).await;

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
        let candidates = match &req.router {
            Some(router) => filter_by_router(workers, router),
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

    /// The verdict on a pool whose every worker is vetoed, judged on the miss
    /// path only: a shed when the pool is all-overloaded under
    /// `--worker-overload-shed`, else the least-loaded routable worker (an
    /// overloaded one never under shedding; a liveness veto steers either
    /// way). `None` when the pool is not all-vetoed (or nothing in it is
    /// routable), so the usual miss handling continues.
    fn all_overloaded_verdict(
        &self,
        req: &SelectWorkerRequest<'_>,
    ) -> Option<Result<Arc<dyn Worker>, Response>> {
        let candidates = self.candidate_pool(req, false);
        let shedding = self.registry.overload_shed_enabled();
        if let Some(shed) = overload::shed_if_all_overloaded(&candidates, req.model_id, shedding) {
            return Some(Err(shed));
        }
        overload::fallback_if_all_vetoed(&candidates, req.model_id, STAGE_SELECTION, shedding)
            .map(Ok)
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
        let candidates = match &req.router {
            Some(router) => filter_by_router(workers, router),
            None => workers,
        };
        candidates.iter().any(|w| {
            w.supports_model(req.model_id)
                && (!req.require_realtime_capable || w.is_realtime_capable())
        })
    }

    /// Re-discover the models of the healthy external workers in parallel,
    /// each with the key the operator configured for it and at most once per
    /// [`REFRESH_MIN_INTERVAL_MS`] per worker.
    ///
    /// When `router` is set, only workers it takes are refreshed: the others
    /// cannot serve this request, so re-reading their catalogues would be a
    /// wasted call. Nothing of the caller's request takes part.
    async fn refresh_external_models(&self, router: Option<&ExternalRouterSpec>) {
        let mut external_workers: Vec<_> = self
            .registry
            .get_routing_workers()
            .iter()
            .filter(|worker| {
                worker.metadata().spec.runtime_type == RuntimeType::External && worker.is_healthy()
            })
            .cloned()
            .collect();

        // Only refresh the workers the request's router takes: the others
        // cannot serve this request, so re-reading their catalogues would be
        // a wasted call (each worker is asked with its own configured key).
        if let Some(router) = router {
            external_workers.retain(|w| router.takes(w.default_provider()));
        }

        if external_workers.is_empty() {
            return;
        }

        tracing::debug!(
            "Refreshing models for {} external workers",
            external_workers.len()
        );

        let futures: Vec<_> = external_workers.iter().map(refresh_worker_models).collect();

        // Timeout prevents a slow/unresponsive worker from blocking all
        // requests that trigger refresh-on-miss.
        const REFRESH_TIMEOUT: Duration = Duration::from_secs(5);
        let _ = tokio::time::timeout(REFRESH_TIMEOUT, join_all(futures)).await;
    }
}

/// In multi-provider setups, keep only the workers `router` takes. In
/// single-provider (or no-provider) setups, return all workers unchanged.
fn filter_by_router(
    workers: Vec<Arc<dyn Worker>>,
    router: &ExternalRouterSpec,
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
            .filter(|w| router.takes(w.default_provider()))
            .collect()
    } else {
        workers
    }
}

/// The least time between two re-discoveries of one worker's models. A miss
/// is one request's doing and a provider's catalogue changes rarely; without
/// a floor, a run of requests for unknown model ids (a loop of made-up names,
/// a client retrying a typo) made the gateway send its own `/v1/models`
/// request on the operator's key once per request per worker, concurrently,
/// and spend the operator's provider rate limit on it. Within the interval a
/// miss is answered from the cards as they stand; a model the provider added
/// since is admitted by the first miss after it. The claim counts the
/// attempt, so a refresh that fails (a refused key, a transient error, the
/// refresh timeout) is not repeated per miss either: the next attempt is the
/// first miss after the interval.
pub(crate) const REFRESH_MIN_INTERVAL_MS: u64 = 30_000;

/// Re-discover one external worker's models and store them: the registration
/// discovery run again, with the key the operator configured for the worker
/// (the provider's admin key from the environment, else the worker's
/// `api_key`), the cards grouped as registration groups them and narrowed to
/// the providers this build routes. A worker without a configured key was
/// registered as a wildcard, which admits every model and had nothing to
/// discover, and is left alone. So is a worker refreshed within
/// [`REFRESH_MIN_INTERVAL_MS`]; of concurrent misses one asks the provider
/// and the others are answered from the cards as they stand, without waiting
/// for its answer (see [`Worker::claim_models_refresh`]): concurrent first
/// requests for a model the provider just added see the winner served and
/// the rest refused until its refresh lands.
///
/// No credential of the caller's request takes part: the cards are shared by
/// every caller, so only the operator's view of the provider may be written
/// to them.
async fn refresh_worker_models(worker: &Arc<dyn Worker>) {
    let spec = &worker.metadata().spec;
    let provider = ProviderType::from_url(&spec.url);
    let Some(api_key) =
        resolve_discovery_api_key(provider.as_ref(), &spec.url, spec.api_key.as_deref())
    else {
        return;
    };
    if !worker.claim_models_refresh(REFRESH_MIN_INTERVAL_MS) {
        tracing::debug!(
            "Model refresh from {} skipped: one was claimed within the last {} ms",
            spec.url,
            REFRESH_MIN_INTERVAL_MS
        );
        return;
    }

    let cards = match fetch_models(&spec.url, Some(&api_key), provider.as_ref()).await {
        Ok(cards) => cards,
        Err(e) => {
            tracing::warn!("Model refresh from {} failed: {}", spec.url, e);
            return;
        }
    };
    let (cards, unroutable) = provider_support::partition_routable(spec, cards);
    if !unroutable.is_empty() {
        tracing::debug!(
            "Model refresh: {} of the models {} lists need a router this build does not carry",
            unroutable.len(),
            spec.url
        );
    }
    if cards.is_empty() {
        return;
    }
    tracing::info!("Model refresh: {} models from {}", cards.len(), spec.url);
    worker.set_models(cards);
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use openai_protocol::worker::{HealthCheckConfig, WorkerSpec};
    use smg_external_router::known;

    use super::*;
    use crate::worker::{BasicWorkerBuilder, WorkerType};

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

    /// A worker of `provider` serving exactly `model`, as a spec would declare it.
    fn provider_worker(url: &str, provider: &str, model: &str) -> Arc<dyn Worker> {
        let spec: WorkerSpec = serde_json::from_value(serde_json::json!({
            "url": url,
            "runtime_type": "external",
            "provider": provider,
            "models": [{"id": model}],
        }))
        .expect("worker spec");
        Arc::new(
            BasicWorkerBuilder::from_spec(spec)
                .health_config(no_health_check())
                .build(),
        )
    }

    #[tokio::test]
    async fn a_router_takes_every_provider_it_serves() {
        // Mixed providers, so the selector filters by router: the
        // OpenAI-compatible router must still reach the xAI worker it is
        // dispatched for, and the Anthropic router must not.
        let registry = WorkerRegistry::new();
        registry.register_or_replace(provider_worker(
            "http://127.0.0.1:18090",
            "openai",
            "gpt-4o",
        ));
        registry.register_or_replace(provider_worker("http://127.0.0.1:18091", "xai", "grok-3"));

        let picked = WorkerSelector::new(&registry)
            .select_worker(&SelectWorkerRequest {
                model_id: "grok-3",
                router: Some(known::OPENAI),
                ..Default::default()
            })
            .await
            .expect("the OpenAI-compatible router serves xAI workers");
        assert_eq!(picked.url(), "http://127.0.0.1:18091");

        let refused = WorkerSelector::new(&registry)
            .select_worker(&SelectWorkerRequest {
                model_id: "gpt-4o",
                router: Some(known::ANTHROPIC),
                ..Default::default()
            })
            .await;
        assert!(
            refused.is_err(),
            "an OpenAI worker is not an Anthropic candidate"
        );
    }

    #[tokio::test]
    async fn requires_realtime_selects_only_labeled() {
        let registry = WorkerRegistry::new();
        registry.register_or_replace(worker("http://127.0.0.1:18080", false));
        registry.register_or_replace(worker("http://127.0.0.1:18081", true));

        let picked = WorkerSelector::new(&registry)
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

        let res = WorkerSelector::new(&registry)
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

        let res = WorkerSelector::new(&registry)
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

    /// Every candidate over the thresholds: by default the least-loaded one
    /// serves; under `--worker-overload-shed` the request is refused.
    #[tokio::test]
    async fn all_overloaded_steers_to_the_least_loaded_unless_shedding() {
        let registry = WorkerRegistry::new();
        let busy = worker("http://127.0.0.1:18180", false);
        let quiet = worker("http://127.0.0.1:18181", false);
        for _ in 0..4 {
            busy.increment_load();
        }
        quiet.increment_load();
        registry.register_or_replace(Arc::clone(&busy));
        registry.register_or_replace(Arc::clone(&quiet));
        registry.set_worker_overloaded(&busy, true);
        registry.set_worker_overloaded(&quiet, true);

        let picked = WorkerSelector::new(&registry)
            .select_worker(&SelectWorkerRequest {
                model_id: "m",
                ..Default::default()
            })
            .await
            .expect("steered to the least-loaded worker");
        assert_eq!(picked.url(), "http://127.0.0.1:18181");

        registry.set_overload_shed(true);
        let refused = WorkerSelector::new(&registry)
            .select_worker(&SelectWorkerRequest {
                model_id: "m",
                ..Default::default()
            })
            .await
            .expect_err("shedding refuses the all-overloaded pool");
        assert_eq!(
            error::extract_error_code_from_response(&refused),
            "worker_overload_protection_shed"
        );
    }

    type Seen = Arc<Mutex<Vec<Option<String>>>>;

    /// What a provider stub answers on `/v1/models` under an `Authorization`
    /// value: a catalogue of model ids, or a refusal.
    type Catalogue = fn(Option<&str>) -> Option<Vec<&'static str>>;

    /// A provider stub whose `/v1/models` records the `Authorization` it is
    /// shown and answers `catalogue`. Returns its URL, the record and the
    /// server task to abort.
    async fn stub_provider(catalogue: Catalogue) -> (String, Seen, tokio::task::JoinHandle<()>) {
        use axum::{
            extract::State,
            http::{HeaderMap, StatusCode},
            response::IntoResponse,
            routing::get,
            Json, Router,
        };
        use tokio::net::TcpListener;

        async fn models(
            State((seen, catalogue)): State<(Seen, Catalogue)>,
            headers: HeaderMap,
        ) -> Response {
            let auth = headers
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .map(str::to_owned);
            seen.lock().expect("record").push(auth.clone());
            match catalogue(auth.as_deref()) {
                Some(ids) => {
                    let data: Vec<_> = ids
                        .iter()
                        .map(|id| serde_json::json!({"id": id, "object": "model"}))
                        .collect();
                    Json(serde_json::json!({"object": "list", "data": data})).into_response()
                }
                None => StatusCode::UNAUTHORIZED.into_response(),
            }
        }

        let seen: Seen = Arc::default();
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let url = format!("http://{}", listener.local_addr().expect("addr"));
        let app = Router::new()
            .route("/v1/models", get(models))
            .with_state((Arc::clone(&seen), catalogue));
        #[expect(
            clippy::disallowed_methods,
            reason = "test-only provider stub; the handle is aborted at test end"
        )]
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.expect("serve");
        });
        (url, seen, server)
    }

    /// An external worker as registration builds it: with the key the
    /// operator configured, if any, and the cards discovered with it (none
    /// for a wildcard worker, which admits every model).
    fn external_worker(url: &str, api_key: Option<&str>, models: &[&str]) -> Arc<dyn Worker> {
        let models: Vec<_> = models
            .iter()
            .map(|id| serde_json::json!({"id": id}))
            .collect();
        let spec: WorkerSpec = serde_json::from_value(serde_json::json!({
            "url": url,
            "runtime_type": "external",
            "api_key": api_key,
            "models": models,
        }))
        .expect("worker spec");
        Arc::new(
            BasicWorkerBuilder::from_spec(spec)
                .health_config(no_health_check())
                .build(),
        )
    }

    /// The refresh on a miss is the registration's discovery run again: the
    /// key the operator configured for the worker goes out and no other, and
    /// the cards come back as registration shapes them, a dated variant an
    /// alias of its base model. A model the provider added since
    /// registration is admitted after it.
    #[tokio::test]
    async fn the_refresh_on_a_miss_rediscovers_with_the_operators_key() {
        let (url, seen, server) = stub_provider(|auth| {
            (auth == Some("Bearer operator-key")).then(|| {
                vec![
                    "operator-model",
                    "released-later",
                    "released-later-2026-01-01",
                ]
            })
        })
        .await;
        let registry = WorkerRegistry::new();
        let keyed = external_worker(&url, Some("operator-key"), &["operator-model"]);
        registry.register_or_replace(Arc::clone(&keyed));

        let picked = WorkerSelector::new(&registry)
            .select_worker(&SelectWorkerRequest {
                model_id: "released-later-2026-01-01",
                ..Default::default()
            })
            .await
            .expect("a model the operator's key sees is admitted after the refresh");
        server.abort();
        assert_eq!(picked.url(), url);
        assert_eq!(
            *seen.lock().expect("record"),
            vec![Some("Bearer operator-key".to_owned())],
            "the configured key, once"
        );

        let mut cards = keyed.models();
        cards.sort_by(|a, b| a.id.cmp(&b.id));
        let ids: Vec<&str> = cards.iter().map(|card| card.id.as_str()).collect();
        assert_eq!(ids, ["operator-model", "released-later"]);
        assert_eq!(cards[1].aliases, ["released-later-2026-01-01"]);
    }

    /// A burst of misses asks the provider once: of concurrent misses for
    /// distinct unknown ids one refreshes, a miss after them within the
    /// interval makes no request, and the cards hold what that one refresh
    /// found. Before, every miss sent its own `/v1/models` request.
    #[tokio::test]
    async fn a_burst_of_misses_asks_the_provider_once_per_interval() {
        let (url, seen, server) =
            stub_provider(|_| Some(vec!["operator-model", "released-later"])).await;
        let registry = WorkerRegistry::new();
        let keyed = external_worker(&url, Some("operator-key"), &["operator-model"]);
        registry.register_or_replace(Arc::clone(&keyed));

        let ids: Vec<String> = (0..16).map(|i| format!("unknown-{i}")).collect();
        let misses = ids.iter().map(|id| async {
            WorkerSelector::new(&registry)
                .select_worker(&SelectWorkerRequest {
                    model_id: id,
                    ..Default::default()
                })
                .await
                .is_err()
        });
        assert!(
            join_all(misses).await.into_iter().all(|refused| refused),
            "unknown ids stay unknown"
        );
        assert_eq!(
            seen.lock().expect("record").len(),
            1,
            "one refresh for the burst"
        );

        let refused = WorkerSelector::new(&registry)
            .select_worker(&SelectWorkerRequest {
                model_id: "unknown-later",
                ..Default::default()
            })
            .await;
        assert!(refused.is_err());
        server.abort();
        assert_eq!(
            seen.lock().expect("record").len(),
            1,
            "a miss within the interval asks nothing"
        );
        assert!(
            keyed.supports_model("released-later"),
            "the one refresh's cards stand"
        );
    }

    /// A worker registered without a key is a wildcard: registration
    /// discovered nothing for it and admits every model, and a miss discovers
    /// nothing either. Before, the miss fetched its catalogue without a
    /// credential and the wildcard collapsed into that answer for every
    /// caller.
    #[tokio::test]
    async fn a_worker_without_a_configured_key_is_not_refreshed_on_a_miss() {
        let (url, seen, server) = stub_provider(|_| Some(vec!["one-model"])).await;
        let registry = WorkerRegistry::new();
        let wildcard = external_worker(&url, None, &[]);
        registry.register_or_replace(Arc::clone(&wildcard));

        // A realtime request no worker can take is a miss, and the miss path
        // refreshes every external worker.
        let refused = WorkerSelector::new(&registry)
            .select_worker(&SelectWorkerRequest {
                model_id: "any-model",
                require_realtime_capable: true,
                ..Default::default()
            })
            .await;
        assert!(refused.is_err(), "no realtime-capable worker");
        server.abort();
        assert!(
            seen.lock().expect("record").is_empty(),
            "no configured key, no request"
        );

        assert!(wildcard.models().is_empty());
        assert!(wildcard.supports_model("any-model"), "the wildcard stands");
        let picked = WorkerSelector::new(&registry)
            .select_worker(&SelectWorkerRequest {
                model_id: "any-model",
                ..Default::default()
            })
            .await
            .expect("the wildcard worker serves any model");
        assert_eq!(picked.url(), url);
    }
}
