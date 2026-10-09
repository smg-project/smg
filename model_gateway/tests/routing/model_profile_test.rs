//! Model contracts must follow trusted deployment configuration, not public aliases.
use std::sync::Arc;

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use openai_protocol::model_card::ModelCard;
use serde_json::{json, Value};
use smg::{config::RouterConfig, worker::BasicWorkerBuilder};
use tower::ServiceExt;

use crate::common::{AppTestContext, TestRouterConfig, TestWorkerConfig};

fn register_model(ctx: &AppTestContext, port: u16, model: &str, alias: &str) {
    let registry = &ctx.app_context.worker_registry;
    let id = registry.get_id_by_url(ctx.worker_url_for(port)).unwrap();
    let worker = registry.get(&id).unwrap();
    let mut spec = worker.metadata().spec.as_ref().clone();
    spec.models = vec![ModelCard::new(model).with_alias(alias)].into();
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

#[tokio::test]
async fn configured_kimi_k3_enforces_sampling_for_generic_serving_name() {
    let port = 19871;
    let mut config = serde_json::to_value(TestRouterConfig::round_robin(3190)).unwrap();
    config["model_profiles"] = json!({"vllm-model":"kimi_k3"});
    let config: RouterConfig = serde_json::from_value(config).unwrap();
    let ctx = AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(port)]).await;
    register_model(&ctx, port, "vllm-model", "ocid1.generativeaiendpoint.test");
    let app = ctx.create_app();
    for model in ["vllm-model", "ocid1.generativeaiendpoint.test"] {
        let response = app
            .clone()
            .oneshot(request(model, json!({"temperature":1.1})))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "model {model}");
    }
    ctx.shutdown().await;
}

#[tokio::test]
async fn explicit_variants_apply_their_own_contract_to_opaque_models() {
    let port = 19872;
    for (profile, fields, expected) in [
        ("kimi_k3", json!({"top_p":0.8}), StatusCode::BAD_REQUEST),
        ("kimi", json!({"temperature":1.1}), StatusCode::OK),
        ("openai", json!({"temperature":1.1}), StatusCode::OK),
        (
            "minimax",
            json!({"messages":[{"role":"root","content":"Hello"}]}),
            StatusCode::OK,
        ),
        (
            "zai_glm_5_3",
            json!({"thinking":{"type":"disabled"}}),
            StatusCode::BAD_REQUEST,
        ),
        (
            "zai",
            json!({"thinking":{"type":"disabled"}}),
            StatusCode::OK,
        ),
        (
            "deepseek_v4",
            json!({"thinking":{"type":"enabled","effort":"75"}}),
            StatusCode::BAD_REQUEST,
        ),
        (
            "deepseek_v4_1",
            json!({"thinking":{"type":"enabled","effort":"75"}}),
            StatusCode::OK,
        ),
    ] {
        let mut config = serde_json::to_value(TestRouterConfig::round_robin(3191)).unwrap();
        config["model_profiles"] = json!({"vllm-model":profile});
        let ctx = AppTestContext::new_with_config(
            serde_json::from_value(config).unwrap(),
            vec![TestWorkerConfig::healthy(port)],
        )
        .await;
        register_model(&ctx, port, "vllm-model", "public-alias");
        let response = ctx
            .create_app()
            .oneshot(request("public-alias", fields))
            .await
            .unwrap();
        assert_eq!(response.status(), expected, "profile {profile}");
        ctx.shutdown().await;
    }
}

#[tokio::test]
async fn configured_profile_keeps_dynamic_tools_and_cannot_be_spoofed_by_client_json() {
    use crate::common::mock_worker::{set_request_recorder, RequestRecorder};
    let port = 19873;
    let recorder = RequestRecorder::new();
    set_request_recorder(port, Arc::clone(&recorder));
    let mut config = serde_json::to_value(TestRouterConfig::round_robin(3192)).unwrap();
    config["model_profiles"] = json!({"vllm-model":"kimi_k3"});
    let ctx = AppTestContext::new_with_config(
        serde_json::from_value(config).unwrap(),
        vec![TestWorkerConfig::healthy(port)],
    )
    .await;
    register_model(&ctx, port, "vllm-model", "ocid1.test");
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
        "resolved_model_id":"gpt-4o",
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
    assert!(forwarded.get("resolved_model_id").is_none());
    ctx.shutdown().await;
}

#[tokio::test]
async fn large_http_request_cannot_bypass_an_explicit_contract() {
    let port = 19874;
    let mut config = serde_json::to_value(TestRouterConfig::round_robin(3193)).unwrap();
    config["model_profiles"] = json!({"vllm-model":"kimi_k3"});
    config["max_buffered_request_bytes"] = json!(128);
    let ctx = AppTestContext::new_with_config(
        serde_json::from_value(config).unwrap(),
        vec![TestWorkerConfig::healthy(port)],
    )
    .await;
    register_model(&ctx, port, "vllm-model", "ocid1.test");
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
async fn explicit_openai_profile_overrides_kimi_name_inference() {
    let port = 19875;
    let mut config = serde_json::to_value(TestRouterConfig::round_robin(3194)).unwrap();
    config["model_profiles"] = json!({"moonshotai/Kimi-K3":"openai"});
    let ctx = AppTestContext::new_with_config(
        serde_json::from_value(config).unwrap(),
        vec![TestWorkerConfig::healthy(port)],
    )
    .await;
    register_model(&ctx, port, "moonshotai/Kimi-K3", "kimi-looking-alias");
    let response = ctx
        .create_app()
        .oneshot(request("kimi-looking-alias", json!({"temperature":1.1})))
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
async fn wildcard_worker_cannot_bypass_a_requested_model_contract() {
    let port = 19877;
    let mut config = serde_json::to_value(TestRouterConfig::round_robin(3195)).unwrap();
    config["model_profiles"] = json!({"vllm-model":"kimi_k3"});
    config["max_buffered_request_bytes"] = json!(128);
    let ctx = AppTestContext::new_with_config(
        serde_json::from_value(config).unwrap(),
        vec![TestWorkerConfig::healthy(port)],
    )
    .await;
    register_model(&ctx, port, "unknown", "unused-alias");
    let response = ctx
        .create_app()
        .oneshot(request(
            "vllm-model",
            json!({
                "temperature":1.1,"padding":"x".repeat(2048)
            }),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    ctx.shutdown().await;
}
