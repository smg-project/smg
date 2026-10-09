//! Model contracts must follow discovered worker metadata, not public aliases.
use std::sync::Arc;

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use openai_protocol::model_card::ModelCard;
use serde_json::{json, Value};
use smg::worker::BasicWorkerBuilder;
use tower::ServiceExt;

use crate::common::{AppTestContext, TestRouterConfig, TestWorkerConfig};

fn register_model(ctx: &AppTestContext, port: u16, model: &str, alias: &str) {
    register_cards(
        ctx,
        port,
        vec![ModelCard::new(model).with_alias(alias)],
        None,
    );
}

fn register_cards(
    ctx: &AppTestContext,
    port: u16,
    cards: Vec<ModelCard>,
    model_path: Option<&str>,
) {
    let registry = &ctx.app_context.worker_registry;
    let id = registry.get_id_by_url(ctx.worker_url_for(port)).unwrap();
    let worker = registry.get(&id).unwrap();
    let mut spec = worker.metadata().spec.as_ref().clone();
    spec.models = cards.into();
    if spec.models.is_wildcard() {
        spec.labels
            .insert("model_id".into(), "moonshotai/Kimi-K3".into());
    }
    if let Some(path) = model_path {
        spec.labels.insert("model_path".into(), path.into());
    }
    let replacement = BasicWorkerBuilder::from_spec(spec)
        .health_config(worker.metadata().health_config.clone())
        .health_endpoint(&worker.metadata().health_endpoint)
        .build();
    assert!(registry.replace(&id, Arc::new(replacement)));
}

fn request(model: &str, fields: Value) -> Request<Body> {
    let mut body =
        json!({"model":model,"messages":[{"role":"user","content":"Hello"}],"stream":false});
    body.as_object_mut()
        .unwrap()
        .extend(fields.as_object().unwrap().clone());
    let serialized = body.to_string();
    Request::builder()
        .header("content-length", serialized.len())
        .method("POST")
        .uri("/v1/chat/completions")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(serialized))
        .unwrap()
}

#[tokio::test]
async fn opaque_alias_enforces_canonical_kimi_k3_sampling() {
    let port = 19870;
    let ctx = AppTestContext::new(vec![TestWorkerConfig::healthy(port)]).await;
    register_model(
        &ctx,
        port,
        "moonshotai/Kimi-K3",
        "ocid1.generativeaiendpoint.test",
    );
    let app = ctx.create_app();
    for model in ["moonshotai/Kimi-K3", "ocid1.generativeaiendpoint.test"] {
        let response = app
            .clone()
            .oneshot(request(model, json!({"temperature":1.1})))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "model {model}");
    }
    ctx.shutdown().await;
}

fn k3_card(model: &str, alias: &str) -> ModelCard {
    let mut card = ModelCard::new(model).with_alias(alias);
    card.architectures = vec!["KimiK3ForConditionalGeneration".into()];
    card
}

#[tokio::test]
async fn worker_metadata_enforces_k3_sampling_for_generic_serving_names() {
    let port = 19871;
    for architecture_only in [true, false] {
        let ctx = AppTestContext::new(vec![TestWorkerConfig::healthy(port)]).await;
        let mut card = ModelCard::new("vllm-model").with_alias("ocid1.generativeaiendpoint.test");
        if architecture_only {
            card.architectures = vec!["KimiK3ForConditionalGeneration".into()];
        } else {
            card.hf_model_type = Some("kimi_k3".into());
        }
        register_cards(&ctx, port, vec![card], None);
        let app = ctx.create_app();
        for model in ["vllm-model", "ocid1.generativeaiendpoint.test"] {
            let response = app
                .clone()
                .oneshot(request(model, json!({"temperature":1.1})))
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                StatusCode::BAD_REQUEST,
                "model {model}, architecture-only {architecture_only}"
            );
        }
        ctx.shutdown().await;
    }
}

#[tokio::test]
async fn architecture_takes_precedence_over_serving_name() {
    let port = 19872;
    let ctx = AppTestContext::new(vec![TestWorkerConfig::healthy(port)]).await;
    register_cards(
        &ctx,
        port,
        vec![k3_card("moonshotai/Kimi-K2", "public-alias")],
        None,
    );
    let response = ctx
        .create_app()
        .oneshot(request("public-alias", json!({"temperature":1.1})))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    ctx.shutdown().await;
}

#[tokio::test]
async fn discovered_profile_keeps_dynamic_tools_and_cannot_be_spoofed_by_client_json() {
    use crate::common::mock_worker::{set_request_recorder, RequestRecorder};
    let port = 19873;
    let recorder = RequestRecorder::new();
    set_request_recorder(port, Arc::clone(&recorder));
    let ctx = AppTestContext::new(vec![TestWorkerConfig::healthy(port)]).await;
    register_cards(&ctx, port, vec![k3_card("vllm-model", "ocid1.test")], None);
    let app = ctx.create_app();
    let rejected = app
        .clone()
        .oneshot(request(
            "ocid1.test",
            json!({
                "temperature":1.1,"resolved_model_profile":"openai"
            }),
        ))
        .await
        .unwrap();
    assert_eq!(rejected.status(), StatusCode::BAD_REQUEST);
    assert!(recorder.bodies().is_empty());
    let accepted=app.oneshot(request("ocid1.test",json!({
        "resolved_model_profile":"openai",
        "messages":[
            {"role":"system","content":"","tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]},
            {"role":"user","content":"Hello"}
        ]
    }))).await.unwrap();
    assert_eq!(accepted.status(), StatusCode::OK);
    let forwarded = recorder.only_body();
    assert_eq!(forwarded["model"], "vllm-model");
    assert_eq!(
        forwarded["messages"][0]["tools"][0]["function"]["name"],
        "lookup"
    );
    assert_eq!(forwarded["temperature"], 1.0);
    assert!((forwarded["top_p"].as_f64().unwrap() - 0.95).abs() < 1e-6);
    assert!(forwarded.get("resolved_model_profile").is_none());
    ctx.shutdown().await;
}

#[tokio::test]
async fn large_http_request_cannot_bypass_a_discovered_contract() {
    let port = 19874;
    let mut config = serde_json::to_value(TestRouterConfig::round_robin(3193)).unwrap();
    config["max_buffered_request_bytes"] = json!(128);
    let ctx = AppTestContext::new_with_config(
        serde_json::from_value(config).unwrap(),
        vec![TestWorkerConfig::healthy(port)],
    )
    .await;
    register_cards(&ctx, port, vec![k3_card("vllm-model", "ocid1.test")], None);
    let response = ctx
        .create_app()
        .oneshot(request(
            "ocid1.test",
            json!({
                "temperature":1.1,"padding":"x".repeat(2048)
            }),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    ctx.shutdown().await;
}

#[tokio::test]
async fn other_kimi_architectures_do_not_select_the_k3_contract() {
    let port = 19875;
    let ctx = AppTestContext::new(vec![TestWorkerConfig::healthy(port)]).await;
    let mut card = ModelCard::new("vllm-model").with_alias("kimi-k3-public-alias");
    card.architectures = vec![
        "KimiK25ForConditionalGeneration".into(),
        "KimiLinearForCausalLM".into(),
    ];
    card.hf_model_type = Some("kimi_k25".into());
    register_cards(&ctx, port, vec![card], None);
    let response = ctx
        .create_app()
        .oneshot(request("kimi-k3-public-alias", json!({"temperature":1.1})))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    ctx.shutdown().await;
}

#[tokio::test]
async fn opaque_glm_alias_keeps_the_canonical_sampling_defaults() {
    use crate::common::mock_worker::{set_request_recorder, RequestRecorder};
    let port = 19876;
    let recorder = RequestRecorder::new();
    set_request_recorder(port, Arc::clone(&recorder));
    let ctx = AppTestContext::new(vec![TestWorkerConfig::healthy(port)]).await;
    register_model(&ctx, port, "zai-org/GLM-4.7", "opaque-glm");
    let app = ctx.create_app();
    for model in ["zai-org/GLM-4.7", "opaque-glm"] {
        let response = app
            .clone()
            .oneshot(request(model, json!({})))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
    let bodies = recorder.bodies();
    for body in &bodies {
        assert_eq!(body["temperature"], 1.0);
        assert!((body["top_p"].as_f64().unwrap() - 0.95).abs() < 1e-6);
    }
    ctx.shutdown().await;
}

#[tokio::test]
async fn unknown_serving_name_cannot_bypass_a_discovered_contract() {
    let port = 19877;
    let mut config = serde_json::to_value(TestRouterConfig::round_robin(3195)).unwrap();
    config["max_buffered_request_bytes"] = json!(128);
    let ctx = AppTestContext::new_with_config(
        serde_json::from_value(config).unwrap(),
        vec![TestWorkerConfig::healthy(port)],
    )
    .await;
    register_cards(&ctx, port, vec![k3_card("unknown", "unused-alias")], None);
    let response = ctx
        .create_app()
        .oneshot(request(
            "unknown",
            json!({
                "temperature":1.1,"padding":"x".repeat(2048)
            }),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    ctx.shutdown().await;
}

#[tokio::test]
async fn multi_model_worker_uses_only_the_requested_card() {
    let port = 19878;
    let ctx = AppTestContext::new(vec![TestWorkerConfig::healthy(port)]).await;
    register_cards(
        &ctx,
        port,
        vec![
            k3_card("vllm-model", "public-k3"),
            ModelCard::new("gpt-4o").with_alias("public-other"),
        ],
        Some("/models/Kimi-K3"),
    );
    let app = ctx.create_app();
    for (model, expected) in [
        ("public-k3", StatusCode::BAD_REQUEST),
        ("public-other", StatusCode::OK),
    ] {
        let response = app
            .clone()
            .oneshot(request(model, json!({"temperature":1.1})))
            .await
            .unwrap();
        assert_eq!(response.status(), expected, "model {model}");
    }
    ctx.shutdown().await;
}

#[tokio::test]
async fn conflicting_replica_contracts_are_rejected_before_dispatch() {
    use crate::common::mock_worker::{set_request_recorder, RequestRecorder};
    let ports = [19879, 19880];
    let recorders = ports.map(|port| {
        let recorder = RequestRecorder::new();
        set_request_recorder(port, Arc::clone(&recorder));
        recorder
    });
    let ctx = AppTestContext::new(ports.into_iter().map(TestWorkerConfig::healthy).collect()).await;
    register_cards(
        &ctx,
        ports[0],
        vec![k3_card("vllm-model", "public-alias")],
        None,
    );
    register_model(&ctx, ports[1], "vllm-model", "public-alias");
    let response = ctx
        .create_app()
        .oneshot(request("public-alias", json!({})))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert!(recorders
        .iter()
        .all(|recorder| recorder.bodies().is_empty()));
    ctx.shutdown().await;
}

#[tokio::test]
async fn real_model_path_is_a_fallback_for_generic_single_model_workers() {
    let port = 19881;
    let ctx = AppTestContext::new(vec![TestWorkerConfig::healthy(port)]).await;
    register_cards(
        &ctx,
        port,
        vec![ModelCard::new("vllm-model").with_alias("public-alias")],
        Some("/models/Kimi-K3"),
    );
    let response = ctx
        .create_app()
        .oneshot(request("public-alias", json!({"temperature":1.1})))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    ctx.shutdown().await;
}

#[tokio::test]
async fn equivalent_k3_replicas_accept_different_metadata_sources_and_paths() {
    use crate::common::mock_worker::{set_request_recorder, RequestRecorder};
    let ports = [19882, 19883];
    for architecture_first in [false, true] {
        let recorders = ports.map(|port| {
            let recorder = RequestRecorder::new();
            set_request_recorder(port, Arc::clone(&recorder));
            recorder
        });
        let ctx =
            AppTestContext::new(ports.into_iter().map(TestWorkerConfig::healthy).collect()).await;
        let first = if architecture_first {
            k3_card("vllm-model", "public-alias")
        } else {
            ModelCard::new("vllm-model").with_alias("public-alias")
        };
        register_cards(&ctx, ports[0], vec![first], Some("/models/Kimi-K3"));
        register_cards(
            &ctx,
            ports[1],
            vec![ModelCard::new("vllm-model").with_alias("public-alias")],
            Some("/mnt/Kimi-K3"),
        );
        let app = ctx.create_app();
        let response = app
            .clone()
            .oneshot(request("public-alias", json!({})))
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            StatusCode::OK,
            "architecture first {architecture_first}"
        );
        let bodies: Vec<_> = recorders.iter().flat_map(|r| r.bodies()).collect();
        assert_eq!(bodies.len(), 1);
        assert_eq!(bodies[0]["temperature"], 1.0);
        assert!((bodies[0]["top_p"].as_f64().unwrap() - 0.95).abs() < 1e-6);
        let response = app
            .oneshot(request("public-alias", json!({"temperature":1.1})))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        ctx.shutdown().await;
    }
}

#[tokio::test]
async fn cardless_vendor_worker_cannot_bypass_large_request_validation() {
    let port = 19884;
    let mut config = serde_json::to_value(TestRouterConfig::round_robin(3196)).unwrap();
    config["max_buffered_request_bytes"] = json!(128);
    let ctx = AppTestContext::new_with_config(
        serde_json::from_value(config).unwrap(),
        vec![TestWorkerConfig::healthy(port)],
    )
    .await;
    register_cards(&ctx, port, vec![], None);
    let response = ctx
        .create_app()
        .oneshot(request(
            "moonshotai/Kimi-K3",
            json!({"temperature":1.1,"padding":"x".repeat(2048)}),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    ctx.shutdown().await;
}

#[tokio::test]
async fn effective_discovered_cards_select_the_contract() {
    let port = 19885;
    let mut config = serde_json::to_value(TestRouterConfig::round_robin(3197)).unwrap();
    config["max_buffered_request_bytes"] = json!(128);
    let ctx = AppTestContext::new_with_config(
        serde_json::from_value(config).unwrap(),
        vec![TestWorkerConfig::healthy(port)],
    )
    .await;
    let registry = &ctx.app_context.worker_registry;
    let id = registry.get_id_by_url(ctx.worker_url_for(port)).unwrap();
    let worker = registry.get(&id).unwrap();
    let mut spec = worker.metadata().spec.as_ref().clone();
    spec.models = vec![].into();
    let replacement = BasicWorkerBuilder::from_spec(spec)
        .health_config(worker.metadata().health_config.clone())
        .health_endpoint(&worker.metadata().health_endpoint)
        .build();
    smg::worker::Worker::set_models(&replacement, vec![k3_card("vllm-model", "public-alias")]);
    assert!(registry.replace(&id, Arc::new(replacement)));
    let app = ctx.create_app();
    for padding in ["".to_string(), "x".repeat(2048)] {
        let response = app
            .clone()
            .oneshot(request(
                "public-alias",
                json!({"temperature":1.1,"padding":padding}),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
    ctx.shutdown().await;
}

#[tokio::test]
async fn conflicting_zai_defaults_are_rejected_before_dispatch() {
    let ports = [19886, 19887];
    let ctx = AppTestContext::new(ports.into_iter().map(TestWorkerConfig::healthy).collect()).await;
    for (port, path) in [(ports[0], "/models/GLM-4.5"), (ports[1], "/models/GLM-4.7")] {
        register_cards(
            &ctx,
            port,
            vec![ModelCard::new("vllm-model").with_alias("public-alias")],
            Some(path),
        );
    }
    let response = ctx
        .create_app()
        .oneshot(request("public-alias", json!({})))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    ctx.shutdown().await;
}
