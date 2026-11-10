//! `GET /v1/models` and `GET /v1/models/{id}`: the gateway's model inventory.
//!
//! Self-hosted models are read from the worker registry; every alias of a
//! self-hosted model is listed as an entry of its own (the same model under
//! the alias name), so every name a client may send is one it can discover.
//! A caller presenting its own provider credential (bring your own key) is
//! answered from the external upstreams of the API family it speaks instead:
//! a credential presented with the Messages API's headers reaches the
//! Anthropic upstreams alone, one presented the OpenAI way the
//! OpenAI-compatible ones (the providers each family's router takes, a
//! worker's provider judged as admission judges it: declared, else read from
//! its address), and never another family's provider. And never when that
//! credential is one of the gateway's own keys: a tenant-scoped key must not
//! be forwarded to a third party as if it were the caller's. A Messages
//! client's upstream is asked for the page the client's query names, and an
//! Anthropic-shaped page it serves is relayed as it is (the upstream's own
//! `lifecycle`, token limits and page markers); an OpenAI-shaped answer is
//! the whole inventory, paged here.
//!
//! The list and the single model come in the shape of the API family the
//! caller speaks: a request carrying the `anthropic-version` header gets the
//! Anthropic objects (`type`, `display_name`, `created_at`, `lifecycle`, the
//! token limits, page markers) and the page its query asks for (`limit`,
//! `after_id`, `before_id`, `lifecycle`); every other request the OpenAI
//! ones (`object`, `created`, `owned_by`), the whole inventory. Both list
//! the self-hosted inventory in id order (the cards, then the aliases).

use std::collections::HashSet;

use axum::{
    http::{header, HeaderMap, HeaderValue},
    response::{IntoResponse, Response},
    Json,
};
use futures::future::select_all;
use openai_protocol::{messages, model_card::ModelCard, models::ListModelsResponse};
use serde_json::Value;
use smg_external_router::{known, ExternalRouterSpec};
use tracing::{debug, warn};

use crate::{
    app_context::AppContext,
    middleware::{into_messages_envelope, AuthConfig},
    routers::{common::header_utils::apply_provider_headers, error, provider_support},
    worker::{endpoint::Endpoint, ProviderType, RuntimeType, WorkerRegistry},
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

    /// The external router of the caller's API family, whose providers are
    /// the only upstreams a credential presented in this family may reach:
    /// the scope the routers define for their workers, which worker selection
    /// applies to a serving request's credential when a miss refreshes the
    /// external workers' models.
    fn router(self) -> ExternalRouterSpec {
        match self {
            Self::OpenAi => known::OPENAI,
            Self::Anthropic => known::ANTHROPIC,
        }
    }

    /// The inventory as a list: the page `page` asks for in the Anthropic
    /// shape (a cursor that names no entry is a 400 in the Messages
    /// envelope), the whole of it in the OpenAI shape, which has no paging.
    async fn list(
        self,
        cards: Vec<ModelCard>,
        page: &messages::ListModelsQuery,
        request_id: String,
    ) -> Response {
        match self {
            Self::OpenAi => Json(ListModelsResponse::from_model_cards(cards)).into_response(),
            Self::Anthropic => match messages::ListModelsResponse::from_model_cards(cards, page) {
                Ok(list) => Json(list).into_response(),
                Err(why) => {
                    self.error(
                        error::bad_request("invalid_query_parameter", why),
                        request_id,
                    )
                    .await
                }
            },
        }
    }

    fn single(self, card: ModelCard) -> Response {
        match self {
            Self::OpenAi => Json(card.into_model_object()).into_response(),
            Self::Anthropic => Json(messages::ModelInfo::from_model_card(&card)).into_response(),
        }
    }

    /// The 404 of an unknown id: the same OpenAI-shaped envelope an unknown
    /// model gets on every other route, re-shaped into the Messages envelope
    /// (`not_found_error`, top-level `type`, `request_id`) for a Messages
    /// client, as `/v1/messages` answers it.
    async fn not_found(self, model_id: &str, request_id: String) -> Response {
        self.error(error::model_not_found(model_id), request_id)
            .await
    }

    /// The 503 of an inventory with no model in it (no worker registered, or
    /// every worker gone): the gateway's error envelope with a typed code, as
    /// every other refusal answers, in the caller's shape.
    async fn no_models(self, request_id: String) -> Response {
        self.error(
            error::service_unavailable("no_models_available", "No models available"),
            request_id,
        )
        .await
    }

    /// An OpenAI-shaped error response in the caller's shape: as it is for an
    /// OpenAI client, re-shaped into the Messages envelope for a Messages
    /// client.
    async fn error(self, response: Response, request_id: String) -> Response {
        match self {
            Self::OpenAi => response,
            Self::Anthropic => into_messages_envelope(response, request_id).await,
        }
    }
}

/// Answer `GET /v1/models` for the caller identified by `headers`, `query`
/// being the request URI's query (`request_id` names the request in the
/// Messages envelope of a refusal).
pub async fn list_models(
    context: &AppContext,
    headers: &HeaderMap,
    query: Option<&str>,
    request_id: String,
) -> Response {
    list_models_with(
        &context.worker_registry,
        &context.client,
        &context.gateway_auth,
        headers,
        query,
        request_id,
    )
    .await
}

/// [`list_models`] over explicit collaborators.
pub(crate) async fn list_models_with(
    registry: &WorkerRegistry,
    client: &reqwest::Client,
    gateway_auth: &AuthConfig,
    headers: &HeaderMap,
    query: Option<&str>,
    request_id: String,
) -> Response {
    let bearer_token = bearer_token(headers);
    let shape = ModelShape::from_headers(headers);
    // The page the Messages-shaped list is asked for, refused in the Messages
    // envelope when the query names what the contract does not take; the
    // OpenAI-shaped list has no paging and reads nothing.
    let page = match shape {
        ModelShape::Anthropic => match messages::ListModelsQuery::parse(query) {
            Ok(page) => page,
            Err(why) => {
                return shape
                    .error(
                        error::bad_request("invalid_query_parameter", why),
                        request_id,
                    )
                    .await
            }
        },
        ModelShape::OpenAi => messages::ListModelsQuery::default(),
    };

    // Short-circuit: a token that is one of the gateway's own credentials
    // is answered from the registry. It must never reach the BYOK fan-out
    // below, which would forward it externally.
    if let Some(ref token) = bearer_token {
        if gateway_auth.contains_token(token) {
            return registry_models_response(registry, shape, &page, request_id).await;
        }
    }

    // A foreign token is tried against the external upstreams of the caller's
    // API family first (bring your own key): a Messages client's upstream is
    // asked for the client's page, as the contract names it, and a page it
    // serves in the Messages shape is the answer as it served it. On total
    // failure fall through to the registry.
    if let Some(ref token) = bearer_token {
        let asked = match shape {
            ModelShape::Anthropic => Some(&page),
            ModelShape::OpenAi => None,
        };
        match fetch_upstream_models(registry, client, token, shape.router(), asked).await {
            Some(UpstreamModels::Page(list)) => return Json(list).into_response(),
            Some(UpstreamModels::Cards(cards)) => {
                return shape.list(cards, &page, request_id).await
            }
            None => {}
        }
    }

    registry_models_response(registry, shape, &page, request_id).await
}

/// Answer `GET /v1/models/{id}` from the self-hosted inventory: the card of
/// the model `model_id` names (its id, or an alias the registry resolves),
/// in the caller's API shape, or the 404 an unknown model gets on every other
/// route, in the caller's error envelope (`request_id` names the request in
/// the Messages envelope).
pub async fn retrieve_model(
    context: &AppContext,
    headers: &HeaderMap,
    model_id: &str,
    request_id: String,
) -> Response {
    retrieve_model_with(&context.worker_registry, headers, model_id, request_id).await
}

/// [`retrieve_model`] over an explicit registry. The lookup is by the
/// resolved canonical id alone: a canonical id that is also another card's
/// alias names its own card, as it does on every other route.
pub(crate) async fn retrieve_model_with(
    registry: &WorkerRegistry,
    headers: &HeaderMap,
    model_id: &str,
    request_id: String,
) -> Response {
    let canonical = registry.resolve_model_alias(model_id);
    let wanted = canonical.as_deref().unwrap_or(model_id);
    let card = self_hosted_cards(registry)
        .into_iter()
        .find(|card| card.id == wanted);
    let shape = ModelShape::from_headers(headers);
    match card {
        Some(card) => shape.single(card),
        None => shape.not_found(model_id, request_id).await,
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

/// The self-hosted inventory as the list presents it: every card, then one
/// entry per alias (the same card under the alias name), so every name a
/// client may send is one it can discover; the cards in id order and the
/// aliases in id order, because the registry's map iterates in no fixed
/// order (another per process, another after a worker came or went) and a
/// page cursor names a position that must mean the same on every replica
/// the client's next request reaches. An alias is listed under the model
/// the registry resolves it to, and nowhere else: a name that is a
/// registered model's id, a contested alias's losing declaration or the
/// reserved wildcard name is never shown as an alias routing does not honour.
fn listed_cards(registry: &WorkerRegistry) -> Vec<ModelCard> {
    let mut cards = self_hosted_cards(registry);
    cards.sort_by(|a, b| a.id.cmp(&b.id));
    let mut aliased: Vec<ModelCard> = cards
        .iter()
        .flat_map(|card| {
            card.aliases
                .iter()
                .filter(|alias| {
                    registry.resolve_model_alias(alias).as_deref() == Some(card.id.as_str())
                })
                .map(|alias| ModelCard {
                    id: alias.clone(),
                    aliases: Vec::new(),
                    ..card.clone()
                })
        })
        .collect();
    aliased.sort_by(|a, b| a.id.cmp(&b.id));
    cards.extend(aliased);
    cards
}

/// What an upstream answered `GET /v1/models` with: the page an
/// Anthropic-shaped upstream served (relayed as it is: its entries' own
/// `lifecycle` and token limits, the markers of the page the caller's query
/// asked it for), or the whole inventory of an OpenAI-shaped one as cards,
/// which the caller's shape lists and pages here.
enum UpstreamModels {
    Page(messages::ListModelsResponse),
    Cards(Vec<ModelCard>),
}

/// The self-hosted inventory as a list in the caller's shape (the page
/// `page` asks for in the Anthropic shape), or the 503 of an empty one in
/// the caller's error envelope.
async fn registry_models_response(
    registry: &WorkerRegistry,
    shape: ModelShape,
    page: &messages::ListModelsQuery,
    request_id: String,
) -> Response {
    let cards = listed_cards(registry);
    if cards.is_empty() {
        shape.no_models(request_id).await
    } else {
        shape.list(cards, page, request_id).await
    }
}

/// Fan out to the healthy external upstreams of the caller's API family (the
/// providers `router` takes) concurrently with the caller's bearer token and
/// return the first successful model inventory. `page` is a Messages
/// client's query, asked of the upstream; an OpenAI client has none. An
/// upstream of another family is not asked: a credential presented in one
/// family's API never reaches another family's provider. Returns `None` on
/// total failure.
async fn fetch_upstream_models(
    registry: &WorkerRegistry,
    client: &reqwest::Client,
    bearer_token: &str,
    router: ExternalRouterSpec,
    page: Option<&messages::ListModelsQuery>,
) -> Option<UpstreamModels> {
    let unique_urls: Vec<_> = registry
        .get_workers_filtered(None, None, None, Some(RuntimeType::External), true)
        .iter()
        .filter(|w| router.takes(provider_support::provider_of(&w.metadata().spec).as_ref()))
        .map(|w| w.url().to_string())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();

    if unique_urls.is_empty() {
        return None;
    }

    debug!(
        "Trying {} upstream(s) for model discovery",
        unique_urls.len()
    );

    let auth = match HeaderValue::from_str(&format!("Bearer {bearer_token}")) {
        Ok(v) => Some(v),
        Err(e) => {
            warn!("Bearer token contains invalid header characters: {e}");
            return None;
        }
    };

    // Fan out concurrently; return the first answer that lists something.
    let mut pending: Vec<_> = unique_urls
        .into_iter()
        .map(|url| {
            Box::pin(fetch_models_from(
                client.clone(),
                url,
                auth.clone(),
                page.cloned(),
            ))
        })
        .collect();

    while !pending.is_empty() {
        let (models, _index, remaining) = select_all(pending).await;
        if models.is_some() {
            return models;
        }
        pending = remaining;
    }

    None
}

/// Fetch models from a single upstream endpoint, asked for the page `page`
/// (a Messages client's query; `None` for an OpenAI client).
async fn fetch_models_from(
    client: reqwest::Client,
    base_url: String,
    auth: Option<HeaderValue>,
    page: Option<messages::ListModelsQuery>,
) -> Option<UpstreamModels> {
    let url = format!("{base_url}/v1/models");
    let asked = match page
        .as_ref()
        .map(messages::ListModelsQuery::to_query_string)
    {
        Some(query) if !query.is_empty() => format!("{url}?{query}"),
        _ => url.clone(),
    };
    // The address as the lines below write it: a userinfo in it masked.
    let shown = Endpoint::redacted_str(&url);
    let req = apply_provider_headers(client.get(&asked), &url, auth.as_ref());

    let resp = match req.send().await {
        Ok(r) => r,
        Err(e) => {
            debug!("Failed to reach upstream {shown}: {e}");
            return None;
        }
    };

    if !resp.status().is_success() {
        debug!(
            "Upstream {shown} returned {} for model discovery",
            resp.status()
        );
        return None;
    }

    match resp.json::<Value>().await {
        Ok(json) => upstream_models(&json, ProviderType::from_url(&url), page.as_ref()),
        Err(e) => {
            warn!("Failed to parse upstream models from {shown}: {e}");
            None
        }
    }
}

/// The models an upstream's `GET /v1/models` body lists. For a Messages
/// client (`page` is its query, which the upstream was asked for) a body in
/// the Messages shape (it carries `has_more`) is the page as the upstream
/// served it: an empty page answers a query that narrows the list (a cursor
/// at the end, a lifecycle nothing has), an empty inventory nothing; a body
/// in that shape the contract's entries cannot be read from is an upstream
/// failure, never the whole inventory to page a second time. Any other body
/// lists the cards of its `data[].id`, an OpenAI-shaped list being the whole
/// inventory; `None` when it lists nothing.
fn upstream_models(
    json: &Value,
    provider: Option<ProviderType>,
    page: Option<&messages::ListModelsQuery>,
) -> Option<UpstreamModels> {
    if let Some(query) = page {
        if json.get("has_more").is_some_and(Value::is_boolean) {
            return match serde_json::from_value::<messages::ListModelsResponse>(json.clone()) {
                Ok(list) if !list.data.is_empty() || query.narrows() => {
                    Some(UpstreamModels::Page(list))
                }
                Ok(_) => None,
                Err(e) => {
                    debug!("Upstream page is not in the Messages shape: {e}");
                    None
                }
            };
        }
    }
    let mut cards = ListModelsResponse::parse_upstream(json, provider);
    if page.is_some() {
        // A cursor names the same position whichever upstream answered.
        cards.sort_by(|a, b| a.id.cmp(&b.id));
    }
    (!cards.is_empty()).then_some(UpstreamModels::Cards(cards))
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    };

    use axum::http::{Request, StatusCode, Uri};
    use openai_protocol::worker::WorkerSpec;

    use super::*;
    use crate::{
        config::types::TenantApiKeyEntry,
        middleware::test_support::json_of,
        worker::{test_support::no_health_check, BasicWorkerBuilder, Worker, WorkerType},
    };

    /// An inventory with no model in it answers the gateway's JSON error
    /// envelope with a typed code, as every other refusal does, in the
    /// caller's shape: the OpenAI envelope, or the Messages envelope with its
    /// `request_id` for a Messages client.
    #[tokio::test]
    async fn an_empty_inventory_answers_the_error_envelope_in_the_callers_shape() {
        let registry = WorkerRegistry::new();

        let response = registry_models_response(
            &registry,
            ModelShape::OpenAi,
            &Default::default(),
            "req_1".to_string(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            response
                .headers()
                .get(header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("application/json")
        );
        let body = json_of(response).await;
        assert_eq!(body["error"]["code"], "no_models_available", "{body}");
        assert_eq!(body["error"]["message"], "No models available", "{body}");
        assert!(body["error"]["type"].is_string(), "{body}");

        let response = registry_models_response(
            &registry,
            ModelShape::Anthropic,
            &Default::default(),
            "req_2".to_string(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = json_of(response).await;
        assert_eq!(body["type"], "error", "{body}");
        assert_eq!(body["error"]["message"], "No models available", "{body}");
        assert_eq!(body["request_id"], "req_2", "{body}");
    }

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
                .health_config(no_health_check())
                .build(),
        );
        registry.register(external_worker);
        // A non-external worker so the short-circuit path has something to
        // return besides 503.
        let internal_worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://internal.invalid:8000")
                .worker_type(WorkerType::Regular)
                .models(vec![ModelCard::new("internal-model")])
                .health_config(no_health_check())
                .build(),
        );
        registry.register(internal_worker);

        let client = crate::server::http_client();
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
        let response = list_models_with(
            &registry,
            &client,
            &gateway_auth,
            req.headers(),
            None,
            "req_test".to_string(),
        )
        .await;
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
        let _ = list_models_with(
            &registry,
            &client,
            &gateway_auth,
            req.headers(),
            None,
            "req_test".to_string(),
        )
        .await;
        assert_eq!(
            hit_count.load(Ordering::SeqCst),
            1,
            "an unrecognized token should still fan out to upstream providers"
        );
    }

    /// A stub provider at a loopback address: it records the credential
    /// headers of every request it gets and answers a one-model catalogue
    /// named after `label`.
    #[expect(clippy::disallowed_methods, reason = "test infrastructure")]
    async fn recording_provider(label: &'static str) -> (String, Arc<Mutex<Vec<String>>>) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen_by_app = seen.clone();
        let app = axum::Router::new().route(
            "/v1/models",
            axum::routing::get(move |headers: HeaderMap| {
                let seen = seen_by_app.clone();
                async move {
                    let credentials: Vec<String> = ["authorization", "x-api-key"]
                        .iter()
                        .filter_map(|name| {
                            headers
                                .get(*name)
                                .and_then(|value| value.to_str().ok())
                                .map(|value| format!("{name}={value}"))
                        })
                        .collect();
                    seen.lock().unwrap().push(credentials.join(" "));
                    Json(serde_json::json!({
                        "object": "list",
                        "data": [{"id": format!("{label}-model"), "object": "model"}]
                    }))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        (format!("http://{addr}"), seen)
    }

    /// An external worker built from the spec a registration would carry.
    fn external_worker(spec: Value) -> Arc<dyn Worker> {
        let spec: WorkerSpec = serde_json::from_value(spec).unwrap();
        Arc::new(
            BasicWorkerBuilder::from_spec(spec)
                .health_config(no_health_check())
                .build(),
        )
    }

    /// A caller's own credential reaches the providers of the API family the
    /// caller speaks and no other: an OpenAI-style bearer goes to the
    /// OpenAI-compatible upstreams, a Messages-style key to the Anthropic
    /// ones, so a key for one provider is never presented to another family's
    /// provider by the listing, and each caller is listed its own family's
    /// catalogue.
    #[tokio::test]
    async fn a_callers_credential_reaches_the_providers_of_its_api_family_alone() {
        let (unnamed_url, seen_by_unnamed) = recording_provider("open").await;
        let (anthropic_url, seen_by_anthropic) = recording_provider("claude").await;
        let registry = WorkerRegistry::new();
        // An address naming no provider: the OpenAI-compatible router's
        // fallback, as an OpenAI-compatible endpoint registers.
        registry.register(external_worker(serde_json::json!({
            "url": unnamed_url,
            "runtime_type": "external",
        })));
        registry.register(external_worker(serde_json::json!({
            "url": anthropic_url,
            "runtime_type": "external",
            "provider": "anthropic",
        })));
        let client = crate::server::http_client();
        let gateway_auth = AuthConfig::with_tenant_keys(None, &[]);

        let mut openai_style = HeaderMap::new();
        openai_style.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer key-for-one-provider"),
        );
        let listed = json_of(
            list_models_with(
                &registry,
                &client,
                &gateway_auth,
                &openai_style,
                None,
                rid(),
            )
            .await,
        )
        .await;
        assert_eq!(listed_ids(&listed), ["open-model"], "{listed}");
        assert_eq!(
            *seen_by_unnamed.lock().unwrap(),
            ["authorization=Bearer key-for-one-provider"],
            "the OpenAI-compatible upstream is asked with the caller's credential"
        );
        assert!(
            seen_by_anthropic.lock().unwrap().is_empty(),
            "an OpenAI-style credential reached the Anthropic upstream: {:?}",
            seen_by_anthropic.lock().unwrap()
        );

        let mut messages_style = HeaderMap::new();
        messages_style.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        messages_style.insert(
            "x-api-key",
            HeaderValue::from_static("key-for-one-provider"),
        );
        let listed = json_of(
            list_models_with(
                &registry,
                &client,
                &gateway_auth,
                &messages_style,
                None,
                rid(),
            )
            .await,
        )
        .await;
        assert_eq!(listed_ids(&listed), ["claude-model"], "{listed}");
        assert_eq!(
            *seen_by_anthropic.lock().unwrap(),
            ["authorization=Bearer key-for-one-provider"],
            "the Anthropic upstream is asked with the caller's credential"
        );
        assert_eq!(
            seen_by_unnamed.lock().unwrap().len(),
            1,
            "a Messages-style credential reached the OpenAI-compatible upstream: {:?}",
            seen_by_unnamed.lock().unwrap()
        );
    }

    /// The list is in id order, not the registry's: the registry's map
    /// iterates in no fixed order, so a cursor would name another position
    /// on another replica or after a worker came or went. Two registries
    /// holding one inventory in different insertion orders serve the same
    /// pages, the first page is the first id, and the OpenAI-shaped list
    /// reads the same order.
    #[tokio::test]
    async fn the_list_is_in_id_order_whatever_the_registrys() {
        let inventory = || {
            let mut aliased = ModelCard::new("m-b");
            aliased.aliases = vec!["alias-z".to_owned(), "alias-a".to_owned()];
            vec![ModelCard::new("m-c"), aliased, ModelCard::new("m-a")]
        };
        let one = WorkerRegistry::new();
        for (index, card) in inventory().into_iter().enumerate() {
            register_worker(
                &one,
                &format!("http://one-{index}.invalid:8000"),
                vec![card],
            );
        }
        let other = WorkerRegistry::new();
        for (index, card) in inventory().into_iter().rev().enumerate() {
            register_worker(
                &other,
                &format!("http://other-{index}.invalid:8000"),
                vec![card],
            );
        }
        async fn list(
            registry: &WorkerRegistry,
            headers: &HeaderMap,
            query: Option<&str>,
        ) -> Response {
            let client = crate::server::http_client();
            let gateway_auth = AuthConfig::with_tenant_keys(None, &[]);
            list_models_with(registry, &client, &gateway_auth, headers, query, rid()).await
        }
        let mut headers = HeaderMap::new();
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));

        for query in [
            None,
            Some("limit=1"),
            Some("limit=2&after_id=m-b"),
            Some("before_id=alias-z"),
        ] {
            let listed = json_of(list(&one, &headers, query).await).await;
            let same = json_of(list(&other, &headers, query).await).await;
            assert_eq!(listed, same, "{query:?}");
        }
        let all = json_of(list(&one, &headers, None).await).await;
        assert_eq!(
            listed_ids(&all),
            ["m-a", "m-b", "m-c", "alias-a", "alias-z"],
            "{all}"
        );
        let first = json_of(list(&other, &headers, Some("limit=1")).await).await;
        assert_eq!(listed_ids(&first), ["m-a"], "{first}");
        assert_eq!(first["has_more"], true, "{first}");
        let openai = json_of(list(&other, &HeaderMap::new(), None).await).await;
        assert_eq!(
            listed_ids(&openai),
            ["m-a", "m-b", "m-c", "alias-a", "alias-z"],
            "{openai}"
        );
    }

    /// A stub Anthropic upstream at a loopback address: it records the query
    /// of every request and answers one page in the Messages shape, an
    /// entry the upstream marks `deprecated` with `has_more` for the rest.
    #[expect(clippy::disallowed_methods, reason = "test infrastructure")]
    async fn paging_provider() -> (String, Arc<Mutex<Vec<String>>>) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen_by_app = seen.clone();
        let app = axum::Router::new().route(
            "/v1/models",
            axum::routing::get(move |uri: Uri| {
                let seen = seen_by_app.clone();
                async move {
                    seen.lock()
                        .unwrap()
                        .push(uri.query().unwrap_or("").to_owned());
                    Json(serde_json::json!({
                        "data": [{
                            "type": "model", "id": "claude-old", "display_name": "Claude Old",
                            "created_at": "2024-01-01T00:00:00Z", "lifecycle": "deprecated",
                            "deprecated_at": "2025-06-01T00:00:00Z", "retires_at": "2026-06-01T00:00:00Z",
                            "max_tokens": 8192, "max_input_tokens": 200000
                        }],
                        "has_more": true, "first_id": "claude-old", "last_id": "claude-old"
                    }))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        (format!("http://{addr}"), seen)
    }

    /// A Messages client's own credential is answered with the page its
    /// upstream serves for the client's query, as the upstream served it:
    /// the upstream is asked for the query's parameters (and nothing the
    /// contract does not name), and its entry's `lifecycle`, token limits
    /// and dates and its page markers reach the client unchanged, where the
    /// gateway's own inventory would call every model `active` and page the
    /// upstream's first page as if it were the whole.
    #[tokio::test]
    async fn a_messages_clients_upstream_page_is_relayed_as_served() {
        let (upstream, seen) = paging_provider().await;
        let registry = WorkerRegistry::new();
        registry.register(external_worker(serde_json::json!({
            "url": upstream,
            "runtime_type": "external",
            "provider": "anthropic",
        })));
        let client = crate::server::http_client();
        let gateway_auth = AuthConfig::with_tenant_keys(None, &[]);
        let mut headers = HeaderMap::new();
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        headers.insert("x-api-key", HeaderValue::from_static("the-callers-own-key"));

        let response = list_models_with(
            &registry,
            &client,
            &gateway_auth,
            &headers,
            Some("limit=1&after_id=claude-new&foo=bar"),
            rid(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let page = json_of(response).await;
        assert_eq!(
            *seen.lock().unwrap(),
            ["limit=1&after_id=claude-new"],
            "the upstream is asked for the client's page, the contract's parameters alone"
        );
        assert_eq!(listed_ids(&page), ["claude-old"], "{page}");
        assert_eq!(page["data"][0]["lifecycle"], "deprecated", "{page}");
        assert_eq!(
            page["data"][0]["deprecated_at"], "2025-06-01T00:00:00Z",
            "{page}"
        );
        assert_eq!(page["data"][0]["max_tokens"], 8192, "{page}");
        assert_eq!(page["data"][0]["max_input_tokens"], 200000, "{page}");
        assert_eq!(page["has_more"], true, "{page}");
        assert_eq!(page["last_id"], "claude-old", "{page}");

        let response =
            list_models_with(&registry, &client, &gateway_auth, &headers, None, rid()).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            seen.lock().unwrap().last().map(String::as_str),
            Some(""),
            "a query naming nothing asks the upstream for its defaults"
        );
    }

    /// What an upstream's body is taken for: a Messages-shaped page is the
    /// page (an empty one only when the query narrows the list), a
    /// Messages-shaped body whose entries the contract cannot read is an
    /// upstream failure (never the whole inventory paged a second time),
    /// and for an OpenAI client any body is its ids.
    #[test]
    fn an_upstream_body_is_a_page_the_ids_or_a_failure() {
        let query = |q: &str| messages::ListModelsQuery::parse(Some(q)).expect("a query");
        let page = serde_json::json!({
            "data": [{"type": "model", "id": "m", "display_name": "M", "created_at": "2024-01-01T00:00:00Z", "lifecycle": "active"}],
            "has_more": true, "first_id": "m", "last_id": "m"
        });
        let slim = serde_json::json!({"data": [{"id": "m"}], "has_more": false, "first_id": "m", "last_id": "m"});
        let empty =
            serde_json::json!({"data": [], "has_more": false, "first_id": null, "last_id": null});
        let whole = serde_json::json!({"object": "list", "data": [{"id": "m", "object": "model"}]});

        assert!(
            matches!(upstream_models(&page, None, Some(&query(""))), Some(UpstreamModels::Page(list)) if list.has_more)
        );
        assert!(
            matches!(upstream_models(&page, None, None), Some(UpstreamModels::Cards(cards)) if cards.len() == 1)
        );
        assert!(upstream_models(&slim, None, Some(&query("after_id=x"))).is_none());
        assert!(
            matches!(upstream_models(&slim, None, None), Some(UpstreamModels::Cards(cards)) if cards[0].id == "m")
        );
        assert!(upstream_models(&empty, None, Some(&query(""))).is_none());
        assert!(upstream_models(&empty, None, Some(&query("limit=5"))).is_none());
        assert!(
            matches!(upstream_models(&empty, None, Some(&query("after_id=x"))), Some(UpstreamModels::Page(list)) if list.data.is_empty())
        );
        assert!(matches!(
            upstream_models(&empty, None, Some(&query("lifecycle=retired"))),
            Some(UpstreamModels::Page(_))
        ));
        assert!(matches!(
            upstream_models(&whole, None, Some(&query("limit=1"))),
            Some(UpstreamModels::Cards(_))
        ));
        assert!(upstream_models(
            &serde_json::json!({"data": []}),
            None,
            Some(&query("after_id=x"))
        )
        .is_none());
    }

    /// An OpenAI-shaped inventory the gateway pages is sorted by id first, so
    /// a cursor names the same position whichever upstream answered; listed
    /// whole for an OpenAI client, it keeps the upstream's order.
    #[test]
    fn an_inventory_the_gateway_pages_is_sorted_by_id() {
        let query = |q: &str| messages::ListModelsQuery::parse(Some(q)).expect("a query");
        let whole = serde_json::json!({
            "object": "list",
            "data": [{"id": "m-b", "object": "model"}, {"id": "m-a", "object": "model"}]
        });
        let ids = |models: Option<UpstreamModels>| match models {
            Some(UpstreamModels::Cards(cards)) => {
                cards.into_iter().map(|card| card.id).collect::<Vec<_>>()
            }
            _ => panic!("the whole inventory lists as cards"),
        };
        assert_eq!(
            ids(upstream_models(&whole, None, Some(&query("limit=1")))),
            ["m-a", "m-b"]
        );
        assert_eq!(
            ids(upstream_models(&whole, None, Some(&query("")))),
            ["m-a", "m-b"]
        );
        assert_eq!(ids(upstream_models(&whole, None, None)), ["m-b", "m-a"]);
    }

    fn registry_with(cards: Vec<ModelCard>) -> WorkerRegistry {
        let registry = WorkerRegistry::new();
        register_worker(&registry, "http://internal.invalid:8000", cards);
        registry
    }

    /// Register a self-hosted worker at `url` serving `cards`.
    fn register_worker(registry: &WorkerRegistry, url: &str, cards: Vec<ModelCard>) {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .models(cards)
                .health_config(no_health_check())
                .build(),
        );
        registry.register(worker);
    }

    /// A request id for the tests that do not read it back.
    fn rid() -> String {
        "req_test".to_owned()
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
        let client = crate::server::http_client();
        let gateway_auth = AuthConfig::with_tenant_keys(None, &[]);

        let openai = json_of(
            list_models_with(
                &registry,
                &client,
                &gateway_auth,
                &HeaderMap::new(),
                None,
                "req_test".to_string(),
            )
            .await,
        )
        .await;
        assert_eq!(openai["object"], "list");
        assert_eq!(openai["data"][0]["object"], "model");
        assert_eq!(openai["data"][0]["id"], "m1");
        assert!(openai.get("has_more").is_none());

        let mut headers = HeaderMap::new();
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        let anthropic = json_of(
            list_models_with(
                &registry,
                &client,
                &gateway_auth,
                &headers,
                None,
                "req_test".to_string(),
            )
            .await,
        )
        .await;
        assert_eq!(anthropic["has_more"], false);
        assert_eq!(anthropic["first_id"], "m1");
        assert_eq!(anthropic["last_id"], "m1");
        assert_eq!(
            anthropic["data"][0],
            serde_json::json!({
                "type": "model",
                "id": "m1",
                "display_name": "Model One",
                "created_at": "1970-01-01T00:00:00Z",
                "lifecycle": "active",
                "capabilities": null,
                "line": null,
                "max_tokens": null,
                "max_input_tokens": null,
                "deprecated_at": null,
                "retires_at": null
            })
        );
    }

    /// The Messages-shaped list serves the page its query asks for over the
    /// listed names (the models, then their aliases): `limit` entries from
    /// the start, after the `after_id` entry or just before the `before_id`
    /// one, `has_more` for the rest; a value the contract does not take is a
    /// 400 in the Messages envelope naming the parameter. The entry carries
    /// `lifecycle` and the card's context length as both token limits. The
    /// OpenAI-shaped list has no paging and reads no query.
    #[tokio::test]
    async fn the_messages_shaped_list_pages_by_limit_and_cursor() {
        async fn page(
            registry: &WorkerRegistry,
            headers: &HeaderMap,
            query: Option<&str>,
        ) -> (StatusCode, Value) {
            let client = crate::server::http_client();
            let gateway_auth = AuthConfig::with_tenant_keys(None, &[]);
            let response =
                list_models_with(registry, &client, &gateway_auth, headers, query, rid()).await;
            (response.status(), json_of(response).await)
        }
        let mut aliased = ModelCard::new("m1");
        aliased.aliases = vec!["alias-1a".to_owned(), "alias-1b".to_owned()];
        aliased.context_length = Some(4096);
        let registry = registry_with(vec![aliased, ModelCard::new("m2")]);
        let mut headers = HeaderMap::new();
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));

        let (status, first) = page(&registry, &headers, Some("limit=1")).await;
        assert_eq!(status, StatusCode::OK, "{first}");
        assert_eq!(listed_ids(&first), ["m1"], "{first}");
        assert_eq!(first["has_more"], true, "{first}");
        assert_eq!(first["last_id"], "m1", "{first}");
        assert_eq!(first["data"][0]["lifecycle"], "active", "{first}");
        assert_eq!(first["data"][0]["max_tokens"], 4096, "{first}");
        assert_eq!(first["data"][0]["max_input_tokens"], 4096, "{first}");
        assert!(first["data"][0]["capabilities"].is_null(), "{first}");
        assert!(first["data"][0]["retires_at"].is_null(), "{first}");

        let (_, second) = page(&registry, &headers, Some("after_id=m1&limit=2")).await;
        assert_eq!(listed_ids(&second), ["m2", "alias-1a"], "{second}");
        assert_eq!(second["has_more"], true, "{second}");
        let (_, last) = page(&registry, &headers, Some("after_id=alias-1a")).await;
        assert_eq!(listed_ids(&last), ["alias-1b"], "{last}");
        assert_eq!(last["has_more"], false, "{last}");
        let (_, back) = page(&registry, &headers, Some("before_id=alias-1b&limit=2")).await;
        assert_eq!(listed_ids(&back), ["m2", "alias-1a"], "{back}");
        assert_eq!(back["has_more"], true, "{back}");
        assert_eq!(back["first_id"], "m2", "{back}");
        let (_, start) = page(&registry, &headers, Some("before_id=alias-1a&limit=5")).await;
        assert_eq!(listed_ids(&start), ["m1", "m2"], "{start}");
        assert_eq!(start["has_more"], false, "{start}");
        let (_, none) = page(&registry, &headers, Some("lifecycle=retired")).await;
        assert!(listed_ids(&none).is_empty(), "{none}");
        assert_eq!(none["has_more"], false, "{none}");
        assert!(none["first_id"].is_null(), "{none}");
        let (_, all) = page(&registry, &headers, None).await;
        assert_eq!(
            listed_ids(&all),
            ["m1", "m2", "alias-1a", "alias-1b"],
            "{all}"
        );
        assert_eq!(all["has_more"], false, "{all}");

        for (query, names) in [
            ("limit=0", "limit"),
            ("limit=1001", "limit"),
            ("limit=many", "limit"),
            ("after_id=nope", "after_id"),
            ("after_id=m1&before_id=m2", "after_id and before_id"),
            ("lifecycle=bogus", "lifecycle"),
        ] {
            let (status, body) = page(&registry, &headers, Some(query)).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{query}: {body}");
            assert_eq!(body["type"], "error", "{query}: {body}");
            assert_eq!(body["error"]["type"], "invalid_request_error", "{body}");
            assert!(
                body["error"]["message"]
                    .as_str()
                    .is_some_and(|message| message.starts_with(names)),
                "{query}: {body}"
            );
        }

        let (status, openai) = page(&registry, &HeaderMap::new(), Some("limit=0")).await;
        assert_eq!(status, StatusCode::OK, "{openai}");
        assert_eq!(
            listed_ids(&openai),
            ["m1", "m2", "alias-1a", "alias-1b"],
            "{openai}"
        );
    }

    /// A single model is found by its id or by an alias and answered in the
    /// caller's shape; an unknown id is the 404 every other route gives.
    #[tokio::test]
    async fn a_single_model_is_retrieved_by_id_or_alias() {
        let mut card = ModelCard::new("m1");
        card.aliases = vec!["alias-1".to_owned()];
        let registry = registry_with(vec![card]);

        let by_id = retrieve_model_with(&registry, &HeaderMap::new(), "m1", rid()).await;
        assert_eq!(by_id.status(), StatusCode::OK);
        let body = json_of(by_id).await;
        assert_eq!(body["object"], "model");
        assert_eq!(body["id"], "m1");

        let mut headers = HeaderMap::new();
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        let by_alias = retrieve_model_with(&registry, &headers, "alias-1", rid()).await;
        assert_eq!(by_alias.status(), StatusCode::OK);
        let body = json_of(by_alias).await;
        assert_eq!(body["type"], "model");
        assert_eq!(body["id"], "m1");

        let unknown = retrieve_model_with(&registry, &headers, "nope", rid()).await;
        assert_eq!(unknown.status(), StatusCode::NOT_FOUND);
    }

    /// An unknown id is refused in the envelope of the API the caller speaks:
    /// the OpenAI-shaped one by default, the Messages envelope (top-level
    /// `type`, `error.type` `not_found_error`, `request_id`) under the header
    /// every Anthropic client sends, as `/v1/messages` refuses the same model.
    #[tokio::test]
    async fn an_unknown_model_is_refused_in_the_callers_envelope() {
        let registry = registry_with(vec![ModelCard::new("m1")]);

        let openai = retrieve_model_with(&registry, &HeaderMap::new(), "nope", rid()).await;
        assert_eq!(openai.status(), StatusCode::NOT_FOUND);
        let body = json_of(openai).await;
        assert_eq!(body["error"]["code"], "model_not_found", "{body}");
        assert_eq!(body["error"]["type"], "invalid_request_error", "{body}");
        assert!(body.get("type").is_none(), "{body}");

        let mut headers = HeaderMap::new();
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        let messages =
            retrieve_model_with(&registry, &headers, "nope", "req_models_1".to_owned()).await;
        assert_eq!(messages.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            messages.headers()["content-type"],
            "application/json",
            "{:?}",
            messages.headers()
        );
        let body = json_of(messages).await;
        assert_eq!(body["type"], "error", "{body}");
        assert_eq!(body["error"]["type"], "not_found_error", "{body}");
        assert_eq!(
            body["error"]["message"], "No worker available for model 'nope'",
            "{body}"
        );
        assert_eq!(body["request_id"], "req_models_1", "{body}");
        assert!(body["error"].get("code").is_none(), "{body}");
        assert!(body["error"].get("param").is_none(), "{body}");
    }

    /// Every alias of a self-hosted model is listed as an entry of its own,
    /// after the models' own ids, with its model's `owned_by`, `created` and
    /// display name, in either shape of the list: a name a client may send
    /// is a name it can discover. A model without aliases lists as before.
    #[tokio::test]
    async fn every_alias_is_listed_as_an_entry_of_its_own() {
        let mut aliased = ModelCard::new("m1");
        aliased.display_name = Some("Model One".to_owned());
        aliased.aliases = vec!["alias-1a".to_owned(), "alias-1b".to_owned()];
        let registry = registry_with(vec![aliased, ModelCard::new("m2")]);

        let openai = json_of(
            registry_models_response(&registry, ModelShape::OpenAi, &Default::default(), rid())
                .await,
        )
        .await;
        assert_eq!(
            listed_ids(&openai),
            ["m1", "m2", "alias-1a", "alias-1b"],
            "{openai}"
        );
        assert_eq!(
            openai["data"][2],
            serde_json::json!({
                "id": "alias-1a",
                "object": "model",
                "created": 0,
                "owned_by": "self_hosted"
            }),
            "{openai}"
        );

        let anthropic = json_of(
            registry_models_response(&registry, ModelShape::Anthropic, &Default::default(), rid())
                .await,
        )
        .await;
        assert_eq!(
            listed_ids(&anthropic),
            ["m1", "m2", "alias-1a", "alias-1b"],
            "{anthropic}"
        );
        assert_eq!(anthropic["data"][1]["display_name"], "m2", "{anthropic}");
        assert_eq!(
            anthropic["data"][3]["display_name"], "Model One",
            "{anthropic}"
        );
        assert_eq!(anthropic["first_id"], "m1", "{anthropic}");
        assert_eq!(anthropic["last_id"], "alias-1b", "{anthropic}");
    }

    /// An alias two workers' cards declare is listed once, under the card
    /// the registry resolves it to (the first registered keeps it), whatever
    /// order the workers come back in: the entry describes the model routing
    /// sends the name to.
    #[tokio::test]
    async fn a_contested_alias_is_listed_under_the_card_the_registry_resolves_it_to() {
        let registry = WorkerRegistry::new();
        let mut first = ModelCard::new("org/first");
        first.display_name = Some("First".to_owned());
        first.aliases = vec!["shared".to_owned()];
        register_worker(&registry, "http://first.invalid:8000", vec![first]);
        let mut second = ModelCard::new("org/second");
        second.display_name = Some("Second".to_owned());
        second.aliases = vec!["shared".to_owned()];
        register_worker(&registry, "http://second.invalid:8000", vec![second]);
        assert_eq!(
            registry.resolve_model_alias("shared").as_deref(),
            Some("org/first")
        );

        let body = json_of(
            registry_models_response(&registry, ModelShape::Anthropic, &Default::default(), rid())
                .await,
        )
        .await;
        let shared: Vec<&Value> = body["data"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|entry| entry["id"] == "shared")
            .collect();
        assert_eq!(shared.len(), 1, "{body}");
        assert_eq!(shared[0]["display_name"], "First", "{body}");
    }

    /// A name is listed once: a model's own id is never re-listed as another
    /// card's alias, and an alias two cards declare makes one entry.
    #[tokio::test]
    async fn a_name_is_listed_once_and_a_models_own_id_wins() {
        let mut first = ModelCard::new("org/first");
        first.aliases = vec!["org/second".to_owned(), "shared".to_owned()];
        let mut third = ModelCard::new("org/third");
        third.aliases = vec!["shared".to_owned()];
        let registry = registry_with(vec![first, ModelCard::new("org/second"), third]);

        let body = json_of(
            registry_models_response(&registry, ModelShape::OpenAi, &Default::default(), rid())
                .await,
        )
        .await;
        assert_eq!(
            listed_ids(&body),
            ["org/first", "org/second", "org/third", "shared"],
            "{body}"
        );
    }

    /// The ids of a list body, in order.
    fn listed_ids(body: &Value) -> Vec<&str> {
        body["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| entry["id"].as_str().unwrap())
            .collect()
    }

    /// A canonical id that is also another card's alias names its own card:
    /// the lookup goes by the resolved canonical id, never by a raw alias.
    #[tokio::test]
    async fn a_canonical_id_that_is_another_cards_alias_answers_its_own_card() {
        let mut aliased = ModelCard::new("org/first");
        aliased.aliases = vec!["org/second".to_owned()];
        let registry = registry_with(vec![aliased, ModelCard::new("org/second")]);

        let response = retrieve_model_with(&registry, &HeaderMap::new(), "org/second", rid()).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = json_of(response).await;
        assert_eq!(body["id"], "org/second");
    }
}
