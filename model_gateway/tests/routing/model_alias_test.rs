//! Model alias integration tests for the regular (non-PD) HTTP router.
//!
//! A worker is registered under its canonical model ID and declares aliases
//! alongside it. A client may address the model by either name, but the worker
//! itself only knows the canonical one, so the gateway resolves the alias for
//! its own routing decisions and forwards the canonical name downstream.
//!
//! The PD equivalent lives in `pd_routing_test::test_pd_model_alias`.

use std::sync::Arc;

use axum::{
    body::{to_bytes, Body},
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use openai_protocol::model_card::ModelCard;
use serde_json::json;
use smg::worker::BasicWorkerBuilder;
use tower::ServiceExt;

use crate::common::{
    mock_worker::{set_request_recorder, RequestRecorder},
    AppTestContext, TestWorkerConfig,
};

#[cfg(test)]
mod model_alias_tests {
    use super::*;

    const WORKER_PORT: u16 = 19860;
    const CANONICAL_MODEL: &str = "GLM-5.2";
    const MODEL_ALIAS: &str = "GLM-5.2-Coding";

    /// Re-register the started worker under a canonical model ID plus an
    /// alias. Mock workers advertise a generic model card, so the alias has to
    /// be installed after startup.
    fn declare_alias(ctx: &AppTestContext, url: &str) {
        let registry = &ctx.app_context.worker_registry;
        let worker_id = registry.get_id_by_url(url).unwrap();
        let worker = registry.get(&worker_id).unwrap();
        let mut spec = worker.metadata().spec.as_ref().clone();
        spec.models = vec![ModelCard::new(CANONICAL_MODEL).with_alias(MODEL_ALIAS)].into();
        spec.labels
            .insert("realtime".to_string(), "true".to_string());
        let replacement = BasicWorkerBuilder::from_spec(spec)
            .health_config(worker.metadata().health_config.clone())
            .health_endpoint(&worker.metadata().health_endpoint)
            .build();
        assert!(registry.replace(&worker_id, Arc::new(replacement)));
    }

    fn chat_request(model: &str) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(
                json!({
                    "model": model,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": false
                })
                .to_string(),
            ))
            .unwrap()
    }

    /// The worker must receive the canonical model ID, not the alias the
    /// client sent. Forwarding the alias would hand the backend a model it
    /// cannot serve.
    #[tokio::test]
    async fn test_regular_routing_forwards_canonical_model_for_alias() {
        let recorder = RequestRecorder::new();
        set_request_recorder(WORKER_PORT, Arc::clone(&recorder));

        let ctx = AppTestContext::new(vec![TestWorkerConfig::healthy(WORKER_PORT)]).await;
        let app = ctx.create_app();
        declare_alias(&ctx, ctx.worker_url_for(WORKER_PORT));

        let response = app
            .clone()
            .oneshot(chat_request(MODEL_ALIAS))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let forwarded = recorder.only_body();
        assert_eq!(
            forwarded["model"], CANONICAL_MODEL,
            "worker must receive the canonical model ID, got {}",
            forwarded["model"]
        );

        ctx.shutdown().await;
    }

    /// The canonical ID keeps working and is forwarded unchanged.
    #[tokio::test]
    async fn test_regular_routing_leaves_canonical_model_untouched() {
        let recorder = RequestRecorder::new();
        set_request_recorder(WORKER_PORT + 1, Arc::clone(&recorder));

        let ctx = AppTestContext::new(vec![TestWorkerConfig::healthy(WORKER_PORT + 1)]).await;
        let app = ctx.create_app();
        declare_alias(&ctx, ctx.worker_url_for(WORKER_PORT + 1));

        let response = app
            .clone()
            .oneshot(chat_request(CANONICAL_MODEL))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(recorder.only_body()["model"], CANONICAL_MODEL);

        ctx.shutdown().await;
    }

    /// Rerank is the one HTTP route whose response the gateway builds itself
    /// rather than passing the worker's through, so it needs its own check
    /// that the reported model is the one that actually ran.
    ///
    /// Uses `/rerank`, not `/v1/rerank`: the latter takes only `query` and
    /// `documents` and hardcodes the model to the `unknown` wildcard, so no
    /// alias can reach the router through it.
    #[tokio::test]
    async fn test_rerank_forwards_and_reports_the_canonical_model() {
        let recorder = RequestRecorder::new();
        set_request_recorder(WORKER_PORT + 3, Arc::clone(&recorder));

        let ctx = AppTestContext::new(vec![TestWorkerConfig::healthy(WORKER_PORT + 3)]).await;
        let app = ctx.create_app();
        declare_alias(&ctx, ctx.worker_url_for(WORKER_PORT + 3));

        let request = Request::builder()
            .method("POST")
            .uri("/rerank")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(
                json!({
                    "model": MODEL_ALIAS,
                    "query": "what is rust",
                    "documents": ["a systems language", "a kind of oxide"]
                })
                .to_string(),
            ))
            .unwrap();
        let response = app.clone().oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        assert_eq!(
            recorder.only_body()["model"],
            CANONICAL_MODEL,
            "worker must receive the canonical model ID"
        );

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            json["model"], CANONICAL_MODEL,
            "rerank must report the model that ran, got {}",
            json["model"]
        );

        ctx.shutdown().await;
    }

    /// `/v1/audio/transcriptions` builds a multipart form instead of a JSON
    /// body and never passes through `route_typed_request`, so it canonicalizes
    /// on its own path.
    #[tokio::test]
    async fn test_transcription_form_carries_the_canonical_model() {
        let recorder = RequestRecorder::new();
        set_request_recorder(WORKER_PORT + 4, Arc::clone(&recorder));

        let ctx = AppTestContext::new(vec![TestWorkerConfig::healthy(WORKER_PORT + 4)]).await;
        let app = ctx.create_app();
        declare_alias(&ctx, ctx.worker_url_for(WORKER_PORT + 4));

        let boundary = "alias-test-boundary";
        let form = format!(
            "--{boundary}\r\n\
             Content-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\
             Content-Type: audio/wav\r\n\r\n\
             RIFFmock\r\n\
             --{boundary}\r\n\
             Content-Disposition: form-data; name=\"model\"\r\n\r\n\
             {MODEL_ALIAS}\r\n\
             --{boundary}--\r\n"
        );
        let request = Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header(
                CONTENT_TYPE,
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(Body::from(form))
            .unwrap();

        let response = app.clone().oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        assert_eq!(
            recorder.only_body()["model"],
            CANONICAL_MODEL,
            "the multipart form must carry the canonical model ID"
        );

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_realtime_rest_routes_forward_the_canonical_model() {
        let recorder = RequestRecorder::new();
        set_request_recorder(WORKER_PORT + 5, Arc::clone(&recorder));

        let ctx = AppTestContext::new(vec![TestWorkerConfig::healthy(WORKER_PORT + 5)]).await;
        let app = ctx.create_app();
        declare_alias(&ctx, ctx.worker_url_for(WORKER_PORT + 5));

        for (endpoint, payload) in [
            (
                "/v1/realtime/sessions",
                json!({"type": "realtime", "model": MODEL_ALIAS}),
            ),
            (
                "/v1/realtime/client_secrets",
                json!({
                    "session": {"type": "realtime", "model": MODEL_ALIAS}
                }),
            ),
            (
                "/v1/realtime/transcription_sessions",
                json!({"type": "transcription", "model": MODEL_ALIAS}),
            ),
        ] {
            let request = Request::builder()
                .method("POST")
                .uri(endpoint)
                .header(CONTENT_TYPE, "application/json")
                .header("Authorization", "Bearer test-key")
                .body(Body::from(payload.to_string()))
                .unwrap();
            let response = app.clone().oneshot(request).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK, "endpoint {endpoint}");
        }

        let forwarded = recorder.bodies();
        assert_eq!(forwarded.len(), 3);
        assert_eq!(forwarded[0]["model"], CANONICAL_MODEL);
        assert_eq!(forwarded[1]["session"]["model"], CANONICAL_MODEL);
        assert_eq!(forwarded[2]["model"], CANONICAL_MODEL);

        ctx.shutdown().await;
    }

    /// A name that is neither a canonical model ID nor an alias is still
    /// rejected, and nothing reaches the worker.
    #[tokio::test]
    async fn test_regular_routing_rejects_an_unknown_model() {
        let recorder = RequestRecorder::new();
        set_request_recorder(WORKER_PORT + 2, Arc::clone(&recorder));

        let ctx = AppTestContext::new(vec![TestWorkerConfig::healthy(WORKER_PORT + 2)]).await;
        let app = ctx.create_app();
        declare_alias(&ctx, ctx.worker_url_for(WORKER_PORT + 2));

        let response = app
            .clone()
            .oneshot(chat_request("GLM-5.2-Nonexistent"))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert!(!body.is_empty());
        assert!(recorder.bodies().is_empty());

        ctx.shutdown().await;
    }
}
