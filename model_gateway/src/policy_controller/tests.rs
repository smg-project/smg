//! Cases of the store's start and of the router a server answers with.

use std::{error::Error, sync::Arc};

use axum::{
    body::{to_bytes, Body},
    extract::Request,
    http::StatusCode,
    response::Response,
};
use openai_protocol::{
    model_card::ModelCard,
    transcription::{AudioFile, TranscriptionRequest},
    worker::WorkerStatus,
    UNKNOWN_MODEL_ID,
};
use serde::de::DeserializeOwned;
use serde_json::{json, Value};
use smg_worker::identity::basic::{ConnectionMode, RuntimeType};

use super::{refused, start, undecided::ROUTER_TYPE, StartError};
use crate::{
    app_context::AppContext,
    config::{ManualAssignmentMode, PolicyConfig, RouterConfig},
    middleware::{RouteRequestMeta, TenantKey},
    routers::{
        common::snapshot::WorkerRegistry, grpc::backend_client::BasicWorkerBuilder,
        http::router::Router, RouterTrait,
    },
    server::{serving_router, ServerConfig},
};

type Case = Result<(), Box<dyn Error>>;

/// The refusal's code as a client reads it. A literal: the product's
/// constant may not move it in silence.
const BY_NAME: &str = "not_decided_by_policy_controller";

#[tokio::test]
async fn refused_names_its_shape_under_the_one_code() -> Case {
    let answer = refused("a test request");
    assert_eq!(answer.status(), StatusCode::NOT_IMPLEMENTED);
    let bytes = to_bytes(answer.into_body(), 1024).await?;
    let error: Value = serde_json::from_slice(&bytes)?;
    assert_eq!(error["error"]["code"], BY_NAME);
    assert_eq!(
        error["error"]["message"],
        "The policy controller is picked for this gateway and does not decide a test request yet"
    );
    Ok(())
}

/// `config` with the policy controller picked.
fn picked(mut config: RouterConfig) -> RouterConfig {
    config.policy_controller = true;
    config
}

#[test]
fn the_classic_policies_start_no_store() -> Case {
    assert!(start(&RouterConfig::default())?.is_none());
    Ok(())
}

#[test]
fn the_store_starts_with_the_document_the_routing_configuration_means() -> Case {
    let config = picked(RouterConfig::default());
    let store = start(&config)?.ok_or("the controller was picked and no store started")?;
    let world = store.world();
    assert_eq!(world.spec().generation(), 1);
    assert_eq!(
        world.spec().digest(),
        Some(config.policy_document()?.digest()?)
    );
    Ok(())
}

#[test]
fn a_setting_no_document_says_stops_the_start_by_its_name() -> Case {
    let mut config = picked(RouterConfig::default());
    config.routing_key_override.enabled = true;
    let refusal = start(&config)
        .err()
        .ok_or("the start took a setting no document says")?;
    assert!(matches!(refusal, StartError::Document(_)), "{refusal:?}");
    assert_eq!(
        refusal.to_string(),
        "routing_key_override.enabled = true: the controller holds no recorded form of the sticky sessions' rule yet"
    );
    Ok(())
}

#[test]
fn a_document_this_build_does_not_run_stops_the_start_by_its_name() -> Case {
    let mut policy = PolicyConfig::cache_aware(0.3, 64, 1.5, 120, 1 << 20);
    if let PolicyConfig::CacheAware { overlap_decay, .. } = &mut policy {
        *overlap_decay = 0.5;
    }
    let mut config = picked(RouterConfig::default());
    config.policy = policy;
    let refusal = start(&config)
        .err()
        .ok_or("the start took a document this build does not run")?;
    assert!(matches!(refusal, StartError::Spec(_)), "{refusal:?}");
    assert_eq!(
        refusal.to_string(),
        "profiles.cache_aware.score[1].prefix-affinity: the prefix-affinity decay is not in this build"
    );
    Ok(())
}

#[test]
fn a_view_that_reads_the_gateways_own_header_stops_the_start_by_its_name() -> Case {
    // The tenant header is the gateway's own: the view of a document this
    // gateway hosts may not read it. The manual policy's view reads the
    // routing key headers.
    let mut config = picked(RouterConfig::default());
    config.policy = PolicyConfig::Manual {
        eviction_interval_secs: 60,
        max_idle_secs: 14_400,
        assignment_mode: ManualAssignmentMode::Random,
    };
    let tenant_header = config.tenant_resolution.tenant_header_name.clone();
    config.routing_key_override.headers = vec![tenant_header.clone()];
    let refusal = start(&config)
        .err()
        .ok_or("the start took a view that reads the tenant header")?;
    assert!(matches!(refusal, StartError::Spec(_)), "{refusal:?}");
    assert_eq!(
        refusal.to_string(),
        format!("view.headers[0]: header {tenant_header:?} is on the deny-list")
    );
    Ok(())
}

/// A request body from its JSON.
fn body<T: DeserializeOwned>(json: Value) -> T {
    serde_json::from_value(json).expect("a request body")
}

/// The status of an answer and the code of its error object.
async fn refusal(answer: Response) -> (StatusCode, String) {
    let status = answer.status();
    let bytes = to_bytes(answer.into_body(), usize::MAX)
        .await
        .expect("the answer's body");
    let error: Value = serde_json::from_slice(&bytes).expect("a JSON error object");
    let code = error["error"]["code"].as_str().unwrap_or_default();
    (status, code.to_owned())
}

/// The asks of `answers` not refused by the controller's name, with what
/// each read.
async fn not_by_name(answers: Vec<(&'static str, Response)>) -> Vec<String> {
    let mut other = Vec::new();
    for (ask, answer) in answers {
        let (status, code) = refusal(answer).await;
        if (status, code.as_str()) != (StatusCode::NOT_IMPLEMENTED, BY_NAME) {
            other.push(format!("{ask}: {status} {code}"));
        }
    }
    other
}

/// Registers a ready regular worker of `model` at `url`, on `mode`, of
/// `runtime` where one is named.
fn register(
    registry: &WorkerRegistry,
    url: &str,
    model: &str,
    mode: ConnectionMode,
    runtime: Option<RuntimeType>,
) {
    let mut worker = BasicWorkerBuilder::new(url)
        .model(ModelCard::new(model))
        .connection_mode(mode)
        .status(WorkerStatus::Ready);
    if let Some(runtime) = runtime {
        worker = worker.runtime_type(runtime);
    }
    registry
        .register(Arc::new(worker.build()))
        .expect("the worker registers");
}

/// A chat request for `model`.
fn chat(model: &str) -> Value {
    json!({"model": model, "messages": [{"role": "user", "content": "hi"}]})
}

/// Every route of the router trait, asked of `router` for `model` in the
/// trait's order. The messages route is asked twice: its second request
/// asks for no token, which the gateway routes by a rule of its own.
async fn every_route(router: &dyn RouterTrait, model: &str) -> Vec<(&'static str, Response)> {
    let meta = RouteRequestMeta::new(TenantKey::from("test-tenant"));
    let empty = || Request::new(Body::empty());
    let text = json!({"model": model, "input": "hi"});
    let message = |max_tokens: u32| {
        let mut request = chat(model);
        request["max_tokens"] = json!(max_tokens);
        request
    };
    let audio = AudioFile {
        bytes: bytes::Bytes::new(),
        file_name: "a.wav".to_owned(),
        content_type: None,
    };
    let decisions = json!({
        "model": model, "input": "evidence",
        "questions": [{"type": "predicate", "instructions": "Is it evidence?"}]
    });
    let systemone = json!({
        "model": model, "state": {"ticket": "evidence"},
        "questions": {"evidence": {
            "type": "noul", "criteria": {"true": "Has evidence", "false": "No evidence"}
        }}
    });
    let rerank = json!({"query": "q", "documents": ["d"], "model": model});
    vec![
        ("health_generate", router.health_generate(empty()).await),
        ("get_server_info", router.get_server_info(empty()).await),
        ("get_model_info", router.get_model_info(empty()).await),
        (
            "route_generate",
            router
                .route_generate(None, &meta, body(json!({"text": "hi"})), model)
                .await,
        ),
        (
            "route_chat",
            router
                .route_chat(None, &meta, Arc::new(body(chat(model))), model)
                .await,
        ),
        (
            "route_completion",
            router
                .route_completion(
                    None,
                    &meta,
                    body(json!({"model": model, "prompt": "hi"})),
                    model,
                )
                .await,
        ),
        (
            "route_responses",
            router
                .route_responses(None, &meta, body(text.clone()), model)
                .await,
        ),
        (
            "cancel_response",
            router.cancel_response(None, &meta, "resp_1").await,
        ),
        (
            "route_embeddings",
            router
                .route_embeddings(None, &meta, body(text.clone()), model)
                .await,
        ),
        (
            "route_classify",
            router.route_classify(None, &meta, body(text), model).await,
        ),
        (
            "route_decisions",
            router
                .route_decisions(None, &meta, body(decisions), model)
                .await,
        ),
        (
            "route_systemone",
            router
                .route_systemone(None, &meta, body(systemone), model)
                .await,
        ),
        (
            "route_audio_transcriptions",
            router
                .route_audio_transcriptions(
                    None,
                    &meta,
                    &TranscriptionRequest::default(),
                    audio,
                    model,
                )
                .await,
        ),
        (
            "route_rerank",
            router.route_rerank(None, &meta, body(rerank), model).await,
        ),
        (
            "route_messages",
            router
                .route_messages(None, &meta, Arc::new(body(message(1))), model)
                .await,
        ),
        (
            "route_messages, no token asked",
            router
                .route_messages(None, &meta, Arc::new(body(message(0))), model)
                .await,
        ),
        (
            "route_messages_count_tokens",
            router
                .route_messages_count_tokens(None, &meta, body(chat(model)), model)
                .await,
        ),
        (
            "route_interactions",
            router
                .route_interactions(None, &meta, body(json!({"input": "hi"})), Some(model))
                .await,
        ),
        (
            "route_realtime_session",
            router
                .route_realtime_session(None, &body(json!({"type": "realtime"})))
                .await,
        ),
        (
            "route_realtime_client_secret",
            router
                .route_realtime_client_secret(None, &body(json!({"session": {"type": "realtime"}})))
                .await,
        ),
        (
            "route_realtime_transcription_session",
            router
                .route_realtime_transcription_session(None, &body(json!({"type": "transcription"})))
                .await,
        ),
        (
            "route_realtime_ws",
            router.route_realtime_ws(empty(), model).await,
        ),
        (
            "route_realtime_webrtc",
            router.route_realtime_webrtc(empty(), model).await,
        ),
    ]
}

/// The adapter request shapes: the model names the adapter, or the
/// request's own field does.
async fn adapter_asks(router: &dyn RouterTrait) -> Vec<(&'static str, Response)> {
    let meta = RouteRequestMeta::new(TenantKey::from("test-tenant"));
    let named = chat("base:adapter");
    let mut by_field = chat("base");
    by_field["lora_path"] = json!("adapter");
    let completion = json!({"model": "base:adapter", "prompt": "hi"});
    let generate = json!({"text": "hi", "lora_path": "adapter"});
    vec![
        (
            "chat, the model names the adapter",
            router
                .route_chat(None, &meta, Arc::new(body(named)), "base:adapter")
                .await,
        ),
        (
            "chat, the request's field names the adapter",
            router
                .route_chat(None, &meta, Arc::new(body(by_field)), "base")
                .await,
        ),
        (
            "completion, the model names the adapter",
            router
                .route_completion(None, &meta, body(completion), "base:adapter")
                .await,
        ),
        (
            "generate, the request's field names the adapter",
            router
                .route_generate(None, &meta, body(generate), "base")
                .await,
        ),
    ]
}

/// A configuration with no health probe behind its cases.
fn quiet() -> RouterConfig {
    let mut config = RouterConfig::default();
    config.health_check.disable_health_check = true;
    config
}

/// Every route and every adapter shape, asked of the router a server answers
/// with under `config`: each is refused by the controller's name. The
/// registry holds the workers a gateway object would choose a router by, and
/// no gateway object is built to choose.
async fn refused_whole(config: RouterConfig) -> Case {
    let context = Arc::new(AppContext::from_config(config.clone(), 30, None, None).await?);
    let registry = &context.worker_registry;
    register(registry, "http://w1:8080", "m", ConnectionMode::Http, None);
    let runtime = Some(RuntimeType::Sglang);
    register(
        registry,
        "grpc://w2:8080",
        "base",
        ConnectionMode::Grpc,
        runtime,
    );
    let (router, gateway) = serving_router(&ServerConfig::minimal(config), &context).await?;
    assert!(gateway.is_none(), "a gateway object is built");
    assert_eq!(router.router_type(), ROUTER_TYPE);
    let mut other = not_by_name(every_route(router.as_ref(), "m").await).await;
    other.extend(not_by_name(adapter_asks(router.as_ref()).await).await);
    assert!(other.is_empty(), "not refused by name: {other:?}");
    Ok(())
}

#[tokio::test]
async fn every_request_is_refused_by_name_where_one_router_would_be_built() -> Case {
    Box::pin(refused_whole(picked(quiet()))).await
}

#[tokio::test]
async fn every_request_is_refused_by_name_where_many_routers_would_be_built() -> Case {
    let mut config = picked(quiet());
    config.enable_igw = true;
    Box::pin(refused_whole(config)).await
}

#[tokio::test]
async fn the_classic_policies_build_the_gateway_object() -> Case {
    let config = quiet();
    let context = Arc::new(AppContext::from_config(config.clone(), 30, None, None).await?);
    assert!(context.policy_store.is_none());
    let (router, gateway) = serving_router(&ServerConfig::minimal(config), &context).await?;
    assert!(gateway.is_some(), "no gateway object is built");
    assert_ne!(router.router_type(), ROUTER_TYPE);
    Ok(())
}

#[tokio::test]
async fn a_starts_context_holds_the_store_where_the_controller_was_picked() -> Case {
    let hosted = AppContext::from_config(picked(quiet()), 30, None, None).await?;
    let generation = hosted
        .policy_store
        .as_ref()
        .map(|store| store.world().spec().generation());
    assert_eq!(generation, Some(1));
    let mut unsaid = picked(quiet());
    unsaid.routing_key_override.enabled = true;
    let refusal = AppContext::from_config(unsaid, 30, None, None).await.err();
    assert_eq!(
        refusal.as_deref(),
        Some("routing_key_override.enabled = true: the controller holds no recorded form of the sticky sessions' rule yet")
    );
    Ok(())
}

#[tokio::test]
async fn a_starts_context_holds_the_host_where_it_holds_the_store() -> Case {
    let hosted = AppContext::from_config(picked(quiet()), 30, None, None).await?;
    assert!(hosted.policy_store.is_some());
    assert!(hosted.policy_host.is_some());
    let router = Router::new(&Arc::new(hosted)).await?;
    let meta = RouteRequestMeta::new(TenantKey::from("test-tenant"));
    let routed = router
        .route_generate(None, &meta, body(json!({"text": "hi"})), UNKNOWN_MODEL_ID)
        .await;
    assert_eq!(
        refusal(routed).await,
        (StatusCode::NOT_IMPLEMENTED, BY_NAME.to_owned())
    );

    let classic = AppContext::from_config(quiet(), 30, None, None).await?;
    assert!(classic.policy_store.is_none());
    assert!(classic.policy_host.is_none());
    Ok(())
}
