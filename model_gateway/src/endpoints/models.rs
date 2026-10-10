//! `GET /v1/models` and `GET /v1/models/{id}`: the gateway's model inventory.
//!
//! Self-hosted models are read from the worker registry. A caller presenting
//! its own provider credential (bring your own key) is answered from the
//! external upstreams instead, but never when that credential is one of the
//! gateway's own keys: a tenant-scoped key must not be forwarded to a third
//! party as if it were the caller's.
//!
//! The list and the single model come in the shape of the API family the
//! caller speaks: a request carrying the `anthropic-version` header gets the
//! Anthropic objects (`type`, `display_name`, `created_at`, page markers),
//! every other request the OpenAI ones (`object`, `created`, `owned_by`).

use std::collections::HashSet;

use axum::{
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use futures::future::select_all;
use openai_protocol::{messages, model_card::ModelCard, models::ListModelsResponse};
use serde_json::Value;
use tracing::{debug, warn};

use crate::{
    app_context::AppContext,
    middleware::AuthConfig,
    routers::{common::header_utils::apply_provider_headers, error},
    worker::{ProviderType, RuntimeType, WorkerRegistry},
};

/// The API family a models request speaks, read from its headers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelShape {
    OpenAi,
    Anthropic,
}

impl ModelShape {
    /// Anthropic clients send `anthropic-version` on every request; no
    /// OpenAI-compatible client does.
    fn from_headers(headers: &HeaderMap) -> Self {
        if headers.contains_key("anthropic-version") {
            Self::Anthropic
        } else {
            Self::OpenAi
        }
    }

    fn list(self, cards: Vec<ModelCard>) -> Response {
        match self {
            Self::OpenAi => Json(ListModelsResponse::from_model_cards(cards)).into_response(),
            Self::Anthropic => {
                Json(messages::ListModelsResponse::from_model_cards(cards)).into_response()
            }
        }
    }

    fn single(self, card: ModelCard) -> Response {
        match self {
            Self::OpenAi => Json(card.into_model_object()).into_response(),
            Self::Anthropic => Json(messages::ModelInfo::from_model_card(&card)).into_response(),
        }
    }
}

/// Answer `GET /v1/models` for the caller identified by `headers`.
pub async fn list_models(context: &AppContext, headers: &HeaderMap) -> Response {
    list_models_with(
        &context.worker_registry,
        &context.client,
        &context.gateway_auth,
        headers,
    )
    .await
}

/// [`list_models`] over explicit collaborators.
pub(crate) async fn list_models_with(
    registry: &WorkerRegistry,
    client: &reqwest::Client,
    gateway_auth: &AuthConfig,
    headers: &HeaderMap,
) -> Response {
    let bearer_token = bearer_token(headers);
    let shape = ModelShape::from_headers(headers);

    // Short-circuit: a token that is one of the gateway's own credentials
    // is answered from the registry. It must never reach the BYOK fan-out
    // below, which would forward it externally.
    if let Some(ref token) = bearer_token {
        if gateway_auth.contains_token(token) {
            return registry_models_response(registry, shape);
        }
    }

    // A foreign token is tried against the external upstreams first (bring
    // your own key). On total failure fall through to the registry.
    if let Some(ref token) = bearer_token {
        let upstream_cards = fetch_upstream_models(registry, client, token).await;
        if !upstream_cards.is_empty() {
            return shape.list(upstream_cards);
        }
    }

    registry_models_response(registry, shape)
}

/// Answer `GET /v1/models/{id}` from the self-hosted inventory: the card of
/// the model `model_id` names (its id, or an alias the registry resolves),
/// in the caller's API shape, or the same 404 an unknown model gets on every
/// other route.
pub fn retrieve_model(context: &AppContext, headers: &HeaderMap, model_id: &str) -> Response {
    retrieve_model_with(&context.worker_registry, headers, model_id)
}

/// [`retrieve_model`] over an explicit registry. The lookup is by the
/// resolved canonical id alone: a canonical id that is also another card's
/// alias names its own card, as it does on every other route.
pub(crate) fn retrieve_model_with(
    registry: &WorkerRegistry,
    headers: &HeaderMap,
    model_id: &str,
) -> Response {
    let canonical = registry.resolve_model_alias(model_id);
    let wanted = canonical.as_deref().unwrap_or(model_id);
    let card = self_hosted_cards(registry)
        .into_iter()
        .find(|card| card.id == wanted);
    match card {
        Some(card) => ModelShape::from_headers(headers).single(card),
        None => error::model_not_found(model_id),
    }
}

/// The caller's credential: a `Bearer` token (case-insensitive scheme per
/// RFC 7235) or an Anthropic-style `x-api-key` header.
fn bearer_token(headers: &HeaderMap) -> Option<String> {
    headers
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .and_then(|h| {
            let lower = h.to_ascii_lowercase();
            lower.starts_with("bearer ").then(|| h[7..].to_string())
        })
        .or_else(|| {
            headers
                .get("x-api-key")
                .and_then(|h| h.to_str().ok())
                .map(String::from)
        })
}

/// The self-hosted inventory: every model card of every non-external worker.
fn self_hosted_cards(registry: &WorkerRegistry) -> Vec<ModelCard> {
    registry
        .get_all()
        .iter()
        .filter(|w| !matches!(w.metadata().spec.runtime_type, RuntimeType::External))
        .flat_map(|w| w.models())
        .collect()
}

/// The self-hosted inventory as a list in the caller's shape.
fn registry_models_response(registry: &WorkerRegistry, shape: ModelShape) -> Response {
    let cards = self_hosted_cards(registry);
    if cards.is_empty() {
        (StatusCode::SERVICE_UNAVAILABLE, "No models available").into_response()
    } else {
        shape.list(cards)
    }
}

/// Fan out to all healthy external upstreams concurrently with the caller's
/// bearer token and return the first successful model inventory. Returns an
/// empty vec on total failure.
async fn fetch_upstream_models(
    registry: &WorkerRegistry,
    client: &reqwest::Client,
    bearer_token: &str,
) -> Vec<ModelCard> {
    let unique_urls: Vec<_> = registry
        .get_workers_filtered(None, None, None, Some(RuntimeType::External), true)
        .iter()
        .map(|w| w.url().to_string())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();

    if unique_urls.is_empty() {
        return Vec::new();
    }

    debug!(
        "Trying {} upstream(s) for model discovery",
        unique_urls.len()
    );

    let auth = match HeaderValue::from_str(&format!("Bearer {bearer_token}")) {
        Ok(v) => Some(v),
        Err(e) => {
            warn!("Bearer token contains invalid header characters: {e}");
            return Vec::new();
        }
    };

    // Fan out concurrently; return the first non-empty result.
    let mut pending: Vec<_> = unique_urls
        .into_iter()
        .map(|url| Box::pin(fetch_models_from(client.clone(), url, auth.clone())))
        .collect();

    while !pending.is_empty() {
        let (cards, _index, remaining) = select_all(pending).await;
        if !cards.is_empty() {
            return cards;
        }
        pending = remaining;
    }

    Vec::new()
}

/// Fetch models from a single upstream endpoint.
async fn fetch_models_from(
    client: reqwest::Client,
    base_url: String,
    auth: Option<HeaderValue>,
) -> Vec<ModelCard> {
    let url = format!("{base_url}/v1/models");
    let req = apply_provider_headers(client.get(&url), &url, auth.as_ref());

    let resp = match req.send().await {
        Ok(r) => r,
        Err(e) => {
            debug!("Failed to reach upstream {url}: {e}");
            return Vec::new();
        }
    };

    if !resp.status().is_success() {
        debug!(
            "Upstream {url} returned {} for model discovery",
            resp.status()
        );
        return Vec::new();
    }

    match resp.json::<Value>().await {
        Ok(json) => ListModelsResponse::parse_upstream(&json, ProviderType::from_url(&url)),
        Err(e) => {
            warn!("Failed to parse upstream models from {url}: {e}");
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    use axum::http::Request;
    use openai_protocol::worker::HealthCheckConfig;

    use super::*;
    use crate::{
        config::types::TenantApiKeyEntry,
        worker::{BasicWorkerBuilder, Worker, WorkerType},
    };

    /// A tenant-scoped credential must never reach the upstream fan-out:
    /// that would forward the gateway's own secret to an external provider
    /// as if it were the caller's BYOK token. Verified against a real mock
    /// upstream that counts hits, since a stubbed HTTP client can't
    /// distinguish "short-circuited" from "fell through and failed" by
    /// response alone: both end up returning registry models on failure.
    #[tokio::test]
    #[expect(clippy::disallowed_methods, reason = "test infrastructure")]
    async fn a_gateway_credential_never_fans_out_upstream() {
        let hit_count = Arc::new(AtomicUsize::new(0));
        let hit_count_clone = hit_count.clone();
        let mock_app = axum::Router::new().route(
            "/v1/models",
            axum::routing::get(move || {
                let hit_count = hit_count_clone.clone();
                async move {
                    hit_count.fetch_add(1, Ordering::SeqCst);
                    Json(serde_json::json!({"object": "list", "data": []}))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, mock_app).await;
        });

        let registry = WorkerRegistry::new();
        let external_worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(format!("http://{addr}"))
                .worker_type(WorkerType::Regular)
                .runtime_type(RuntimeType::External)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        registry.register(external_worker);
        // A non-external worker so the short-circuit path has something to
        // return besides 503.
        let internal_worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://internal.invalid:8000")
                .worker_type(WorkerType::Regular)
                .models(vec![ModelCard::new("internal-model")])
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        registry.register(internal_worker);

        let client = reqwest::Client::new();
        let gateway_auth = AuthConfig::with_tenant_keys(
            Some("shared-secret".to_string()),
            &[TenantApiKeyEntry {
                tenant_id: "team-red".to_string(),
                key: "team-red-secret".to_string(),
            }],
        );

        let req = Request::builder()
            .header(header::AUTHORIZATION, "Bearer team-red-secret")
            .body(())
            .unwrap();
        let response = list_models_with(&registry, &client, &gateway_auth, req.headers()).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            hit_count.load(Ordering::SeqCst),
            0,
            "a tenant-scoped key must never trigger upstream BYOK fan-out"
        );

        // Control: a genuinely unrecognized token still fans out, which
        // confirms the short-circuit is keyed on the gateway's own
        // credentials rather than disabled entirely.
        let req = Request::builder()
            .header(header::AUTHORIZATION, "Bearer not-a-gateway-key")
            .body(())
            .unwrap();
        let _ = list_models_with(&registry, &client, &gateway_auth, req.headers()).await;
        assert_eq!(
            hit_count.load(Ordering::SeqCst),
            1,
            "an unrecognized token should still fan out to upstream providers"
        );
    }

    fn registry_with(cards: Vec<ModelCard>) -> WorkerRegistry {
        let registry = WorkerRegistry::new();
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://internal.invalid:8000")
                .worker_type(WorkerType::Regular)
                .models(cards)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        registry.register(worker);
        registry
    }

    async fn json_body(response: Response) -> Value {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    /// The same inventory answers both API families, each in its own shape:
    /// the OpenAI list by default, the Anthropic list (page markers, `type`,
    /// `display_name`, `created_at`) when the request carries the header
    /// every Anthropic client sends.
    #[tokio::test]
    async fn the_list_takes_the_shape_of_the_callers_api_family() {
        let mut card = ModelCard::new("m1");
        card.display_name = Some("Model One".to_owned());
        let registry = registry_with(vec![card]);
        let client = reqwest::Client::new();
        let gateway_auth = AuthConfig::with_tenant_keys(None, &[]);

        let openai =
            json_body(list_models_with(&registry, &client, &gateway_auth, &HeaderMap::new()).await)
                .await;
        assert_eq!(openai["object"], "list");
        assert_eq!(openai["data"][0]["object"], "model");
        assert_eq!(openai["data"][0]["id"], "m1");
        assert!(openai.get("has_more").is_none());

        let mut headers = HeaderMap::new();
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        let anthropic =
            json_body(list_models_with(&registry, &client, &gateway_auth, &headers).await).await;
        assert_eq!(anthropic["has_more"], false);
        assert_eq!(anthropic["first_id"], "m1");
        assert_eq!(anthropic["last_id"], "m1");
        assert_eq!(
            anthropic["data"][0],
            serde_json::json!({
                "type": "model",
                "id": "m1",
                "display_name": "Model One",
                "created_at": "1970-01-01T00:00:00Z"
            })
        );
    }

    /// A single model is found by its id or by an alias and answered in the
    /// caller's shape; an unknown id is the 404 every other route gives.
    #[tokio::test]
    async fn a_single_model_is_retrieved_by_id_or_alias() {
        let mut card = ModelCard::new("m1");
        card.aliases = vec!["alias-1".to_owned()];
        let registry = registry_with(vec![card]);

        let by_id = retrieve_model_with(&registry, &HeaderMap::new(), "m1");
        assert_eq!(by_id.status(), StatusCode::OK);
        let body = json_body(by_id).await;
        assert_eq!(body["object"], "model");
        assert_eq!(body["id"], "m1");

        let mut headers = HeaderMap::new();
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        let by_alias = retrieve_model_with(&registry, &headers, "alias-1");
        assert_eq!(by_alias.status(), StatusCode::OK);
        let body = json_body(by_alias).await;
        assert_eq!(body["type"], "model");
        assert_eq!(body["id"], "m1");

        let unknown = retrieve_model_with(&registry, &headers, "nope");
        assert_eq!(unknown.status(), StatusCode::NOT_FOUND);
    }

    /// A canonical id that is also another card's alias names its own card:
    /// the lookup goes by the resolved canonical id, never by a raw alias.
    #[tokio::test]
    async fn a_canonical_id_that_is_another_cards_alias_answers_its_own_card() {
        let mut aliased = ModelCard::new("org/first");
        aliased.aliases = vec!["org/second".to_owned()];
        let registry = registry_with(vec![aliased, ModelCard::new("org/second")]);

        let response = retrieve_model_with(&registry, &HeaderMap::new(), "org/second");
        assert_eq!(response.status(), StatusCode::OK);
        let body = json_body(response).await;
        assert_eq!(body["id"], "org/second");
    }
}
