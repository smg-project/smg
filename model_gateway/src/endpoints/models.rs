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
use openai_protocol::{
    messages, model_card::ModelCard, models::ListModelsResponse, worker::ProviderType,
};
use serde::Deserialize;
use serde_json::Value;
use smg_external_router::{known, ExternalRouterSpec};
use smg_worker::identity::{basic::RuntimeType, endpoint::Endpoint};
use tracing::{debug, warn};

use crate::{
    app_context::AppContext,
    middleware::{into_messages_envelope, AuthConfig},
    routers::{
        common::{header_utils::apply_provider_headers, snapshot::WorkerRegistry},
        error, provider_support,
    },
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
    let card = self_hosted_card(registry, wanted);
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
            // The scheme compared on the header's first seven bytes; no
            // lowercased copy of the whole header.
            let (scheme, token) = h.split_at_checked(7)?;
            scheme
                .eq_ignore_ascii_case("bearer ")
                .then(|| token.to_string())
        })
        .or_else(|| {
            headers
                .get("x-api-key")
                .and_then(|h| h.to_str().ok())
                .map(String::from)
        })
}

/// The first self-hosted card with id `wanted`, read in place through the
/// worker's card reader: only the match is copied, no worker's card list.
fn self_hosted_card(registry: &WorkerRegistry, wanted: &str) -> Option<ModelCard> {
    let mut card = None;
    for worker in &registry.get_all() {
        if matches!(worker.metadata().spec.runtime_type, RuntimeType::External) {
            continue;
        }
        let found = worker.any_model(&mut |candidate| {
            let hit = candidate.id == wanted;
            if hit {
                card = Some(candidate.clone());
            }
            hit
        });
        if found {
            break;
        }
    }
    card
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
            debug!("Failed to reach upstream {shown}: {}", e.without_url());
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
            warn!(
                "Failed to parse upstream models from {shown}: {}",
                e.without_url()
            );
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
            return match messages::ListModelsResponse::deserialize(json) {
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
#[path = "../../tests/unit/endpoints/models/tests.rs"]
mod tests;
