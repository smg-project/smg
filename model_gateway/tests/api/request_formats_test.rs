use serde_json::json;

use crate::common::{
    mock_worker::{HealthStatus, MockWorkerConfig, WorkerType},
    WorkerTestContext,
};

#[cfg(test)]
mod request_format_tests {
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use axum::{
        body::Body,
        http::{
            header::{CONTENT_ENCODING, CONTENT_TYPE},
            HeaderMap, Request, StatusCode,
        },
        response::{IntoResponse, Response},
    };
    use openai_protocol::chat::ChatCompletionRequest;
    use smg::{middleware::TenantRequestMeta, routers::RouterTrait};
    use tower::ServiceExt;

    use super::*;
    use crate::common::test_app::{create_test_app_context, create_test_app_with_context};

    #[derive(Debug, Default)]
    struct CapturingChatRouter {
        request: Mutex<Option<Option<String>>>,
    }

    #[async_trait]
    impl RouterTrait for CapturingChatRouter {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        async fn route_chat(
            &self,
            _headers: Option<&HeaderMap>,
            _tenant_meta: &TenantRequestMeta,
            body: ChatCompletionRequest,
            _model_id: &str,
        ) -> Response {
            *self.request.lock().unwrap() = Some(body.reasoning_effort.clone());
            StatusCode::OK.into_response()
        }

        fn router_type(&self) -> &'static str {
            "capture"
        }
    }

    #[tokio::test]
    async fn test_chat_completions_handler_does_not_materialize_default_effort() {
        let capture = Arc::new(CapturingChatRouter::default());
        let app_context = create_test_app_context().await;
        let app = create_test_app_with_context(capture.clone(), app_context);
        let payload = json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Hello!"}]
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(payload.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let request = capture.request.lock().unwrap();
        let reasoning_effort = request
            .as_ref()
            .expect("route_chat should receive the request");
        // The router forwards effort verbatim and injects no default; the chat
        // template applies its own default when it is absent.
        assert!(reasoning_effort.is_none());
    }

    /// `{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"max_tokens":1}`
    /// under `gzip -n -9` (90 bytes): a valid chat request under a content
    /// coding the gateway does not decode.
    const GZIP_CHAT_BODY: &[u8] = &[
        0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x03, 0xab, 0x56, 0xca, 0xcd, 0x4f, 0x49, 0xcd, 0x51, 0xb2, 0x02, 0xd2, 0xc9, 0xd9, 0xba, 0x10, 0x8e, 0x8e, 0x52, 0x6e, 0x6a, 0x71, 0x71, 0x62, 0x7a, 0x6a, 0xb1, 0x92, 0x55, 0x74, 0xb5, 0x52, 0x51, 0x7e, 0x4e, 0x2a, 0x50, 0x41, 0x69, 0x71, 0x6a, 0x11, 0x50, 0x2a, 0x39, 0x3f, 0xaf, 0x24, 0x35, 0xaf, 0x04, 0x28, 0x90, 0x91, 0xa9, 0x54, 0x1b, 0x0b, 0x54, 0x9b, 0x58, 0x11, 0x5f, 0x92, 0x9f, 0x9d, 0x9a, 0x07, 0x54, 0x6d, 0x58, 0x0b, 0x00, 0x1b, 0x7f, 0x8c, 0xf1, 0x51, 0x00, 0x00, 0x00
    ];

    /// A request body under a content coding the gateway does not decode is
    /// refused by the coding's name (`415`, the envelope, the remedy) before
    /// any parser reads it; the same request under `identity` reaches the
    /// router. Red over the edge as it stood: the gzip bytes reached the JSON
    /// extractor and were refused as malformed JSON (`400 invalid_json`).
    #[tokio::test]
    async fn a_request_body_under_a_content_coding_is_refused_by_its_coding_before_the_parser() {
        let capture = Arc::new(CapturingChatRouter::default());
        let app_context = create_test_app_context().await;
        let app = create_test_app_with_context(capture.clone(), app_context);

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header(CONTENT_TYPE, "application/json")
                    .header(CONTENT_ENCODING, "gzip")
                    .body(Body::from(GZIP_CHAT_BODY))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "a gzip body is refused by its coding, not parsed as JSON"
        );
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let envelope: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(envelope["error"]["type"], "invalid_request_error", "{envelope}");
        assert_eq!(envelope["error"]["code"], "json_parse_error", "{envelope}");
        let message = envelope["error"]["message"].as_str().unwrap_or("");
        assert!(
            message.contains("'gzip'") && message.contains("identity"),
            "the message names the coding and the remedy: {message}"
        );
        assert!(
            capture.request.lock().unwrap().is_none(),
            "the router never saw the encoded request"
        );

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header(CONTENT_TYPE, "application/json")
                    .header(CONTENT_ENCODING, "identity")
                    .body(Body::from(
                        r#"{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"max_tokens":1}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK, "identity is no coding");
        assert!(capture.request.lock().unwrap().is_some());
    }

    #[tokio::test]
    async fn test_generate_request_formats() {
        let ctx = WorkerTestContext::new(vec![MockWorkerConfig {
            port: 19001,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let payload = json!({
            "text": "Hello, world!",
            "stream": false
        });

        let result = ctx.make_request("/generate", payload).await;
        assert!(result.is_ok());

        let payload = json!({
            "text": "Tell me a story",
            "sampling_params": {
                "temperature": 0.7,
                "max_new_tokens": 100,
                "top_p": 0.9
            },
            "stream": false
        });

        let result = ctx.make_request("/generate", payload).await;
        assert!(result.is_ok());

        let payload = json!({
            "input_ids": [1, 2, 3, 4, 5],
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 50
            },
            "stream": false
        });

        let result = ctx.make_request("/generate", payload).await;
        assert!(result.is_ok());

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_v1_chat_completions_formats() {
        let ctx = WorkerTestContext::new(vec![MockWorkerConfig {
            port: 19002,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let payload = json!({
            "model": "mock-model",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ],
            "stream": false
        });

        let result = ctx.make_request("/v1/chat/completions", payload).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert!(response.get("choices").is_some());
        assert!(response.get("id").is_some());
        assert_eq!(
            response.get("object").and_then(|v| v.as_str()),
            Some("chat.completion")
        );

        let payload = json!({
            "model": "mock-model",
            "messages": [
                {"role": "user", "content": "Tell me a joke"}
            ],
            "temperature": 0.8,
            "max_tokens": 150,
            "top_p": 0.95,
            "stream": false
        });

        let result = ctx.make_request("/v1/chat/completions", payload).await;
        assert!(result.is_ok());

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_v1_completions_formats() {
        let ctx = WorkerTestContext::new(vec![MockWorkerConfig {
            port: 19003,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let payload = json!({
            "model": "mock-model",
            "prompt": "Once upon a time",
            "max_tokens": 50,
            "stream": false
        });

        let result = ctx.make_request("/v1/completions", payload).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert!(response.get("choices").is_some());
        assert_eq!(
            response.get("object").and_then(|v| v.as_str()),
            Some("text_completion")
        );

        let payload = json!({
            "model": "mock-model",
            "prompt": ["First prompt", "Second prompt"],
            "temperature": 0.5,
            "stream": false
        });

        let result = ctx.make_request("/v1/completions", payload).await;
        assert!(result.is_ok());

        let payload = json!({
            "model": "mock-model",
            "prompt": "The capital of France is",
            "max_tokens": 10,
            "logprobs": 5,
            "stream": false
        });

        let result = ctx.make_request("/v1/completions", payload).await;
        assert!(result.is_ok());

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_batch_requests() {
        let ctx = WorkerTestContext::new(vec![MockWorkerConfig {
            port: 19004,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let payload = json!({
            "text": ["First text", "Second text", "Third text"],
            "sampling_params": {
                "temperature": 0.7,
                "max_new_tokens": 50
            },
            "stream": false
        });

        let result = ctx.make_request("/generate", payload).await;
        assert!(result.is_ok());

        let payload = json!({
            "input_ids": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "stream": false
        });

        let result = ctx.make_request("/generate", payload).await;
        assert!(result.is_ok());

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_special_parameters() {
        let ctx = WorkerTestContext::new(vec![MockWorkerConfig {
            port: 19005,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let payload = json!({
            "text": "Test",
            "return_logprob": true,
            "stream": false
        });

        let result = ctx.make_request("/generate", payload).await;
        assert!(result.is_ok());

        let payload = json!({
            "text": "Generate JSON",
            "sampling_params": {
                "temperature": 0.0,
                "json_schema": "$$ANY$$"
            },
            "stream": false
        });

        let result = ctx.make_request("/generate", payload).await;
        assert!(result.is_ok());

        let payload = json!({
            "text": "Continue forever",
            "sampling_params": {
                "temperature": 0.7,
                "max_new_tokens": 100,
                "ignore_eos": true
            },
            "stream": false
        });

        let result = ctx.make_request("/generate", payload).await;
        assert!(result.is_ok());

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_error_handling() {
        let ctx = WorkerTestContext::new(vec![MockWorkerConfig {
            port: 19006,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let payload = json!({});

        let result = ctx.make_request("/generate", payload).await;
        // Mock worker accepts empty body
        assert!(result.is_ok());

        ctx.shutdown().await;
    }
}
