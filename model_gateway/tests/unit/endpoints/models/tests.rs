use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

use axum::http::{Request, StatusCode, Uri};
use openai_protocol::worker::WorkerSpec;
use smg_worker::{
    identity::basic::{Worker, WorkerType},
    test_support::no_health_check,
};

use super::*;
use crate::{
    config::types::TenantApiKeyEntry, middleware::test_support::json_of,
    routers::grpc::backend_client::BasicWorkerBuilder, server::http_client,
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

    let client = http_client();
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
    let client = http_client();
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
    async fn list(registry: &WorkerRegistry, headers: &HeaderMap, query: Option<&str>) -> Response {
        let client = http_client();
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
    let client = http_client();
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

    let response = list_models_with(&registry, &client, &gateway_auth, &headers, None, rid()).await;
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
    let client = http_client();
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
        let client = http_client();
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
        registry_models_response(&registry, ModelShape::OpenAi, &Default::default(), rid()).await,
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
        registry_models_response(&registry, ModelShape::OpenAi, &Default::default(), rid()).await,
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

/// The in-place walk skips external workers and reads past the first
/// self-hosted one: an external worker's card is not in the self-hosted
/// inventory (404), each self-hosted worker's card is (200), whichever of
/// the two the walk visits first.
#[tokio::test]
async fn the_card_walk_skips_external_workers_and_reads_every_self_hosted_one() {
    let registry = WorkerRegistry::new();
    let external_worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://external.invalid:8000")
            .worker_type(WorkerType::Regular)
            .runtime_type(RuntimeType::External)
            .models(vec![ModelCard::new("m1")])
            .health_config(no_health_check())
            .build(),
    );
    registry.register(external_worker);
    register_worker(
        &registry,
        "http://internal-a.invalid:8000",
        vec![ModelCard::new("m0")],
    );
    register_worker(
        &registry,
        "http://internal-b.invalid:8000",
        vec![ModelCard::new("m2")],
    );

    let response = retrieve_model_with(&registry, &HeaderMap::new(), "m1", rid()).await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    // `get_all` order is the DashMap's, not registration order: looking
    // up both cards means one of them always sits on the second
    // self-hosted worker the walk visits.
    for id in ["m0", "m2"] {
        let response = retrieve_model_with(&registry, &HeaderMap::new(), id, rid()).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = json_of(response).await;
        assert_eq!(body["id"], id);
    }
}

/// The bearer scheme is matched without regard to case on the header's
/// first seven bytes; a header too short for a scheme and a space is no
/// bearer token and the lookup falls through to `x-api-key`.
#[test]
fn the_bearer_scheme_is_case_insensitive_and_a_short_header_falls_through() {
    let mut headers = HeaderMap::new();
    headers.insert(
        header::AUTHORIZATION,
        HeaderValue::from_static("BEARER abc"),
    );
    assert_eq!(bearer_token(&headers).as_deref(), Some("abc"));
    headers.insert(
        header::AUTHORIZATION,
        HeaderValue::from_static("bearer xyz"),
    );
    assert_eq!(bearer_token(&headers).as_deref(), Some("xyz"));

    headers.insert(header::AUTHORIZATION, HeaderValue::from_static("Bearer"));
    assert_eq!(bearer_token(&headers), None);
    headers.insert("x-api-key", HeaderValue::from_static("key-1"));
    assert_eq!(bearer_token(&headers).as_deref(), Some("key-1"));
}

/// An upstream at an address with a query whose model list is no JSON:
/// the line about it does not repeat the query. The HTTP client's text of
/// a failed body read ends with the request URL too.
#[tracing_test::traced_test]
#[tokio::test]
async fn an_unparsed_upstream_list_goes_without_the_clients_url_in_its_line() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    #[expect(
        clippy::disallowed_methods,
        reason = "test-only server answering no JSON; handle is aborted at test end"
    )]
    let server = tokio::spawn(async move {
        axum::serve(
            listener,
            axum::Router::new().fallback(|| async { "not json" }),
        )
        .await
        .unwrap();
    });

    let models = fetch_models_from(
        http_client(),
        format!("http://127.0.0.1:{port}?token=s3cret"),
        None,
        None,
    )
    .await;
    server.abort();

    assert!(models.is_none());
    assert!(logs_contain("Failed to parse upstream models"));
    assert!(!logs_contain("s3cret"));
}

/// The opt-in lists immutable registration cards, never lazy caller catalogues,
/// and makes no provider request under either shared or tenant gateway keys.
#[tokio::test]
async fn external_inventory_opt_in_uses_registration_cards_without_provider_requests() {
    let (url, seen) = recording_provider("live-provider").await;
    let registry = WorkerRegistry::new();
    register_worker(
        &registry,
        "http://local.invalid",
        vec![ModelCard::new("local")],
    );
    let external = external_worker(serde_json::json!({
        "url": url, "runtime_type": "external", "api_key": "operator-secret",
        "models": [
            {"id": "registered-external", "aliases": ["external-alias"],
             "metadata": {"private": "operator-private-metadata"}},
            {"id": "local", "provider": "openai"}
        ]
    }));
    registry.register(Arc::clone(&external));
    let wildcard = external_worker(serde_json::json!({
        "url": "http://wildcard.invalid", "runtime_type": "external"
    }));
    registry.register(Arc::clone(&wildcard));
    let auth = AuthConfig::with_tenant_keys(
        Some("shared".into()),
        &[TenantApiKeyEntry {
            tenant_id: "red".into(),
            key: "tenant".into(),
        }],
    );
    let client = http_client();
    for private in ["tenant-a-private", "tenant-b-private"] {
        registry.refresh_models(&external, vec![ModelCard::new(private)]);
        registry.refresh_models(&wildcard, vec![ModelCard::new("wildcard-private")]);
        for key in ["shared", "tenant"] {
            for anthropic in [false, true] {
                let mut headers = HeaderMap::new();
                headers.insert(
                    header::AUTHORIZATION,
                    format!("Bearer {key}").parse().unwrap(),
                );
                if anthropic {
                    headers.insert("anthropic-version", "2023-06-01".parse().unwrap());
                }
                for query in [
                    None,
                    Some("include_external=false"),
                    Some("include_external=true"),
                ] {
                    let response =
                        list_models_with(&registry, &client, &auth, &headers, query, rid()).await;
                    assert_eq!(response.status(), StatusCode::OK);
                    let body = json_of(response).await;
                    let ids: Vec<_> = body["data"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|card| card["id"].as_str().unwrap())
                        .collect();
                    let expected = if query == Some("include_external=true") {
                        vec!["local", "registered-external"]
                    } else {
                        vec!["local"]
                    };
                    assert_eq!(ids, expected, "{query:?}, anthropic={anthropic}, {body}");
                    assert!(!body.to_string().contains(private));
                    assert!(!body.to_string().contains("wildcard-private"));
                    assert!(!body.to_string().contains("operator-secret"));
                    assert!(!body.to_string().contains("operator-private-metadata"));
                    if !anthropic {
                        assert_eq!(body["data"][0]["owned_by"], "self_hosted");
                    }
                }
                if anthropic {
                    let response = list_models_with(
                        &registry,
                        &client,
                        &auth,
                        &headers,
                        Some("include_external=true&limit=1&after_id=local"),
                        rid(),
                    )
                    .await;
                    let body = json_of(response).await;
                    assert_eq!(body["data"].as_array().unwrap().len(), 1);
                    assert_eq!(body["data"][0]["id"], "registered-external");
                }
            }
        }
    }
    assert!(
        seen.lock().unwrap().is_empty(),
        "inventory must never call a provider"
    );
}

/// Opting in cannot turn a BYOK credential, or an unauthenticated request,
/// into access to the operator catalogue, nor trigger a provider request.
#[tokio::test]
async fn external_inventory_opt_in_requires_a_gateway_key_and_valid_boolean() {
    let (url, seen) = recording_provider("provider").await;
    let registry = WorkerRegistry::new();
    registry.register(external_worker(serde_json::json!({
        "url": url, "runtime_type": "external", "models": [{"id":"operator-model"}]
    })));
    let auth = AuthConfig::with_tenant_keys(Some("gateway".into()), &[]);
    let client = http_client();
    for credential in [None, Some("foreign")] {
        for header_name in ["authorization", "x-api-key"] {
            let mut headers = HeaderMap::new();
            if let Some(key) = credential {
                let value = if header_name == "authorization" {
                    format!("Bearer {key}")
                } else {
                    key.into()
                };
                headers.insert(header_name, value.parse().unwrap());
            }
            let response = list_models_with(
                &registry,
                &client,
                &auth,
                &headers,
                Some("include_external=true"),
                rid(),
            )
            .await;
            assert_eq!(response.status(), StatusCode::FORBIDDEN);
            assert_eq!(
                json_of(response).await["error"]["code"],
                "external_inventory_requires_gateway_key"
            );
        }
    }
    let mut headers = HeaderMap::new();
    headers.insert("x-api-key", "gateway".parse().unwrap());
    for query in [
        "include_external=1",
        "include_external=",
        "include_external=true&include_external=false",
    ] {
        let response =
            list_models_with(&registry, &client, &auth, &headers, Some(query), rid()).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{query}");
        assert_eq!(
            json_of(response).await["error"]["code"],
            "invalid_query_parameter"
        );
    }
    let response = list_models_with(
        &registry,
        &client,
        &auth,
        &headers,
        Some("include_external=true"),
        rid(),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(json_of(response).await["data"][0]["id"], "operator-model");
    assert!(seen.lock().unwrap().is_empty());
}

/// Only the opt-in distinguishes a registered wildcard's empty catalogue
/// from a gateway with no workers; the default empty-inventory 503 stays.
#[tokio::test]
async fn external_inventory_opt_in_keeps_empty_catalogue_and_empty_registry_distinct() {
    let registry = WorkerRegistry::new();
    let auth = AuthConfig::with_tenant_keys(Some("gateway".into()), &[]);
    let client = http_client();
    let mut headers = HeaderMap::new();
    headers.insert(header::AUTHORIZATION, "Bearer gateway".parse().unwrap());
    for anthropic in [false, true] {
        if anthropic {
            headers.insert("anthropic-version", "2023-06-01".parse().unwrap());
        }
        let response = list_models_with(
            &registry,
            &client,
            &auth,
            &headers,
            Some("include_external=true"),
            rid(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }
    registry.register(external_worker(serde_json::json!({
        "url":"http://wildcard.invalid", "runtime_type":"external"
    })));
    for query in [
        None,
        Some("include_external=false"),
        Some("include_external=true"),
    ] {
        let response = list_models_with(&registry, &client, &auth, &headers, query, rid()).await;
        if query == Some("include_external=true") {
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(json_of(response).await["data"], serde_json::json!([]));
        } else {
            assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        }
    }
}
