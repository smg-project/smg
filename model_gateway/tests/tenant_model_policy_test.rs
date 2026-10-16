//! The per-tenant model allow-list through the real app: a tenant held to a
//! list is served the models on it and told the others do not exist (the
//! 404 of an unknown model, in the caller's envelope), `/v1/models` shows it
//! the allowed subset, every other caller gets the default policy, and
//! without the policy nothing changes. Over an in-process scripted engine
//! serving two models.

#[path = "common/mod.rs"]
mod common;

#[path = "common/scripted_worker.rs"]
mod scripted_worker;

use std::sync::Arc;

use axum::{
    body::{to_bytes, Body},
    http::{header::CONTENT_TYPE, Method, Request, StatusCode},
    Router,
};
use llm_tokenizer::{traits::Tokenizer, MockTokenizer, TokenizerRegistry};
use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
use serde_json::{json, Value};
use smg::{
    config::{RouterConfig, RoutingMode, TenantApiKeyEntry},
    routers::RouterFactory,
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerType},
};
use tokio::net::TcpListener;
use tower::ServiceExt;

const MODEL_A: &str = "org/model-a";
const MODEL_B: &str = "org/model-b";
const TENANT_HEADER: &str = "x-smg-tenant-id";

/// `team-red` may use model A only; `team-blue` has no entry and gets the
/// default (every model); the trusted-header tenant `team-green` (reachable
/// on the public routes, which need no key) may use model B only.
const POLICY: &str = r#"default_policy:
  models: ["*"]
tenants:
  - tenant_key: auth:team-red
    models: ["org/model-a"]
  - tenant_key: header:team-green
    models: ["org/model-b*"]
"#;

/// The caller of a request: a tenant API key, a trusted tenant header, or
/// nobody.
#[derive(Clone, Copy)]
enum Caller {
    Key(&'static str),
    Header(&'static str),
    Nobody,
}

/// The app over one scripted engine serving both models, with two tenant
/// keys and the trusted header; `policy` switches the allow-list on. The
/// engine's task is aborted when the handle drops; the policy file lives as
/// long as the directory.
#[expect(
    clippy::unwrap_used,
    clippy::disallowed_methods,
    reason = "test helper: failures should panic; the in-process engine is aborted with its handle"
)]
async fn app(policy: Option<&str>) -> (Router, tokio::task::AbortHandle, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let engine = tokio::spawn(
        scripted_worker::ScriptedWorker {
            output_tokens: 2,
            finish_reasons: vec!["stop"],
            generation: Default::default(),
            requests: Default::default(),
        }
        .serve(listener),
    );
    let worker = Arc::new(
        BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::TokenSpeed)
            .models(vec![ModelCard::new(MODEL_A), ModelCard::new(MODEL_B)])
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build(),
    );
    let mut config = RouterConfig::builder()
        .mode(RoutingMode::Regular {
            worker_urls: vec![],
        })
        .grpc_connection()
        .random_policy()
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(1024 * 1024)
        .build_unchecked();
    config.health_check.disable_health_check = true;
    config.tenant_resolution.trust_tenant_header = true;
    config.tenant_api_keys = vec![
        TenantApiKeyEntry {
            tenant_id: "team-red".to_owned(),
            key: "red-secret".to_owned(),
        },
        TenantApiKeyEntry {
            tenant_id: "team-blue".to_owned(),
            key: "blue-secret".to_owned(),
        },
    ];
    if let Some(policy) = policy {
        let path = dir.path().join("model-policy.yaml");
        std::fs::write(&path, policy).unwrap();
        config.tenant_model_policy_config = Some(path.to_str().unwrap().to_owned());
    }
    let tokenizers = Arc::new(TokenizerRegistry::new());
    for (id, model) in [("tokenizer-a", MODEL_A), ("tokenizer-b", MODEL_B)] {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
        tokenizers
            .load(id, model, "test", || async move { Ok(tokenizer) })
            .await
            .unwrap();
    }
    let context = common::create_test_context_with_tokenizer_registry(config, tokenizers).await;
    context.worker_registry.register(worker).unwrap();
    let router = RouterFactory::create_router(&context).await.unwrap();
    let app = common::test_app::create_test_app_with_context(Arc::from(router), context);
    (app, engine.abort_handle(), dir)
}

/// One call as `caller`; `anthropic` adds the header every Messages client
/// sends, for the Messages envelope.
#[expect(clippy::unwrap_used, reason = "test helper: failures should panic")]
async fn call(
    app: &Router,
    method: Method,
    path: &str,
    caller: Caller,
    body: Option<Value>,
    anthropic: bool,
) -> (StatusCode, Value, Option<String>) {
    let mut request = Request::builder().method(method).uri(path);
    request = match caller {
        Caller::Key(key) => request.header("authorization", format!("Bearer {key}")),
        Caller::Header(tenant) => request.header(TENANT_HEADER, tenant),
        Caller::Nobody => request,
    };
    if anthropic {
        request = request.header("anthropic-version", "2023-06-01");
    }
    let request = match body {
        Some(body) => request
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap(),
        None => request.body(Body::empty()).unwrap(),
    };
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let error_code = response
        .headers()
        .get("x-smg-error-code")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let json = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, json, error_code)
}

fn chat(model: &str) -> Value {
    json!({"model": model, "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 4})
}

/// The ids of a `/v1/models` list, sorted.
#[expect(clippy::unwrap_used, reason = "test helper: failures should panic")]
fn listed(list: &Value) -> Vec<String> {
    let mut ids: Vec<String> = list["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|card| card["id"].as_str().unwrap().to_owned())
        .collect();
    ids.sort();
    ids
}

/// A tenant on a list is served the models on it and refused the others as
/// not found, on every serving route that names a model and in the caller's
/// envelope; the refusal is the one an unknown model gets. A tenant without
/// an entry gets the default policy.
#[tokio::test]
async fn a_tenant_is_held_to_its_allow_list_on_the_serving_routes() {
    let (app, _engine, _dir) = app(Some(POLICY)).await;
    let red = Caller::Key("red-secret");
    let blue = Caller::Key("blue-secret");

    let (status, body, _) = call(
        &app,
        Method::POST,
        "/v1/chat/completions",
        red,
        Some(chat(MODEL_A)),
        false,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let (status, refused, code) = call(
        &app,
        Method::POST,
        "/v1/chat/completions",
        red,
        Some(chat(MODEL_B)),
        false,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{refused}");
    assert_eq!(code.as_deref(), Some("model_not_found"));
    assert_eq!(refused["error"]["code"], "model_not_found", "{refused}");
    // Indistinguishable from a model nobody serves: the same envelope, the
    // same code, the same message with the model's name.
    let (status, unknown, unknown_code) = call(
        &app,
        Method::POST,
        "/v1/chat/completions",
        red,
        Some(chat("org/model-zzz")),
        false,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{unknown}");
    assert_eq!(code, unknown_code);
    assert_eq!(
        refused.to_string(),
        unknown.to_string().replace("org/model-zzz", MODEL_B),
        "{refused} vs {unknown}"
    );

    // The other routes that name a model refuse the same way.
    for (path, body) in [
        (
            "/v1/completions",
            json!({"model": MODEL_B, "prompt": "Hello", "max_tokens": 4}),
        ),
        (
            "/v1/responses",
            json!({"model": MODEL_B, "input": "Hello", "max_output_tokens": 4}),
        ),
        (
            "/v1/embeddings",
            json!({"model": MODEL_B, "input": "Hello"}),
        ),
    ] {
        let (status, refused, code) = call(&app, Method::POST, path, red, Some(body), false).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{path}: {refused}");
        assert_eq!(code.as_deref(), Some("model_not_found"), "{path}");
    }

    // A Messages client is refused in the Messages envelope.
    let (status, refused, _) = call(
        &app,
        Method::POST,
        "/v1/messages",
        red,
        Some(json!({"model": MODEL_B, "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 4})),
        true,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{refused}");
    assert_eq!(refused["type"], "error", "{refused}");
    assert_eq!(refused["error"]["type"], "not_found_error", "{refused}");

    // A tenant without an entry gets the default policy: every model.
    for model in [MODEL_A, MODEL_B] {
        let (status, body, _) = call(
            &app,
            Method::POST,
            "/v1/chat/completions",
            blue,
            Some(chat(model)),
            false,
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{model}: {body}");
    }
}

/// `/v1/models` lists the caller's allowed subset: the tenant of the key it
/// presents, the trusted-header tenant without one, and the default policy
/// for a caller without either; `/v1/models/{id}` answers a model outside
/// the list as unknown.
#[tokio::test]
async fn the_model_inventory_is_filtered_to_the_callers_allow_list() {
    let (app, _engine, _dir) = app(Some(POLICY)).await;
    let cases: [(Caller, &[&str]); 4] = [
        (Caller::Key("red-secret"), &[MODEL_A]),
        (Caller::Key("blue-secret"), &[MODEL_A, MODEL_B]),
        (Caller::Header("team-green"), &[MODEL_B]),
        (Caller::Nobody, &[MODEL_A, MODEL_B]),
    ];
    for (caller, expected) in cases {
        let (status, list, _) = call(&app, Method::GET, "/v1/models", caller, None, false).await;
        assert_eq!(status, StatusCode::OK, "{list}");
        assert_eq!(listed(&list), expected, "{list}");
        for model in [MODEL_A, MODEL_B] {
            let (status, body, code) = call(
                &app,
                Method::GET,
                &format!("/v1/models/{model}"),
                caller,
                None,
                false,
            )
            .await;
            if expected.contains(&model) {
                assert_eq!(status, StatusCode::OK, "{model}: {body}");
                assert_eq!(body["id"], model, "{body}");
            } else {
                assert_eq!(status, StatusCode::NOT_FOUND, "{model}: {body}");
                assert_eq!(code.as_deref(), Some("model_not_found"), "{model}");
            }
        }
    }
    // A model nobody serves is unknown to everyone, list or no list.
    let (status, _, code) = call(
        &app,
        Method::GET,
        "/v1/models/org/model-zzz",
        Caller::Key("blue-secret"),
        None,
        false,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(code.as_deref(), Some("model_not_found"));
}

/// Without the policy key nothing changes: every tenant is served every
/// model and sees the whole inventory.
#[tokio::test]
async fn without_a_policy_every_tenant_may_name_every_model() {
    let (app, _engine, _dir) = app(None).await;
    let red = Caller::Key("red-secret");
    for model in [MODEL_A, MODEL_B] {
        let (status, body, _) = call(
            &app,
            Method::POST,
            "/v1/chat/completions",
            red,
            Some(chat(model)),
            false,
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{model}: {body}");
    }
    for caller in [red, Caller::Header("team-green"), Caller::Nobody] {
        let (status, list, _) = call(&app, Method::GET, "/v1/models", caller, None, false).await;
        assert_eq!(status, StatusCode::OK, "{list}");
        assert_eq!(listed(&list), [MODEL_A, MODEL_B], "{list}");
    }
}
