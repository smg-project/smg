use std::sync::Arc;

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use reqwest::Client;
use serde_json::json;
use smg::{
    app_context::AppContext,
    config::{RouterConfig, RoutingMode},
    routers::{RouterFactory, RouterTrait},
};
use tower::ServiceExt;

use crate::common::{
    mock_worker::{MockWorker, MockWorkerConfig},
    test_app::create_test_app_with_context,
};

/// Test context that manages mock workers and app
struct ParserTestContext {
    workers: Vec<MockWorker>,
    router: Arc<dyn RouterTrait>,
    _client: Client,
    _config: RouterConfig,
    app_context: Arc<AppContext>,
}

impl ParserTestContext {
    async fn new(worker_configs: Vec<MockWorkerConfig>) -> Self {
        // Create router config with parser support enabled
        let mut config = RouterConfig::builder()
            .regular_mode(vec![])
            .random_policy()
            .host("127.0.0.1")
            .port(3003)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(1)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();
        config.health_check.disable_health_check = true;

        Self::new_with_config(config, worker_configs).await
    }

    #[expect(clippy::unwrap_used, reason = "test helper with known-valid setup")]
    async fn new_with_config(
        mut config: RouterConfig,
        worker_configs: Vec<MockWorkerConfig>,
    ) -> Self {
        let mut workers = Vec::new();
        let mut worker_urls = Vec::new();

        // Start mock workers if any
        for worker_config in worker_configs {
            let mut worker = MockWorker::new(worker_config);
            let url = worker.start().await.unwrap();
            worker_urls.push(url);
            workers.push(worker);
        }

        if !workers.is_empty() {
            tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        }

        // Update config with worker URLs if not already set
        match &mut config.mode {
            RoutingMode::Regular {
                worker_urls: ref mut urls,
            }
            | RoutingMode::OpenAI {
                worker_urls: ref mut urls,
            } if urls.is_empty() => {
                urls.clone_from(&worker_urls);
            }
            _ => {} // PrefillDecode mode has its own setup
        }

        let client = smg::server::http_client_builder()
            .timeout(std::time::Duration::from_secs(config.request_timeout_secs))
            .build()
            .unwrap();

        // Create app context with parser factories initialized
        let app_context = crate::common::create_test_context_with_parsers(config.clone()).await;

        // Create router
        let router = RouterFactory::create_router(&app_context).await.unwrap();
        let router = Arc::from(router);

        Self {
            workers,
            router,
            _client: client,
            _config: config,
            app_context,
        }
    }

    fn create_app(&self) -> axum::Router {
        create_test_app_with_context(Arc::clone(&self.router), Arc::clone(&self.app_context))
    }

    async fn shutdown(mut self) {
        for worker in &mut self.workers {
            worker.stop().await;
        }
    }
}

#[cfg(test)]
mod parse_function_call_tests {
    use super::*;

    #[tokio::test]
    async fn test_parse_function_call_success() {
        let ctx = ParserTestContext::new(vec![]).await;
        let app = ctx.create_app();

        let payload = json!({
            "text": r#"I need to call the weather function <tool_call>{"function_name": "get_weather", "parameters": {"location": "Beijing"}}</tool_call>"#,
            "tool_call_parser": "json",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather information",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The location"
                                }
                            },
                            "required": ["location"]
                        }
                    }
                }
            ]
        });

        let req = Request::builder()
            .method("POST")
            .uri("/parse/function_call")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();

        // Parser endpoint should return 200 for valid requests
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Expected OK (200), got {}",
            resp.status()
        );

        // Verify response contains tool_calls
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body_json["success"], true);
        assert!(body_json["tool_calls"].is_array());

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_parse_function_call_invalid_parser() {
        let ctx = ParserTestContext::new(vec![]).await;
        let app = ctx.create_app();

        let payload = json!({
            "text": "some text",
            "tool_call_parser": "nonexistent_parser",
            "tools": []
        });

        let req = Request::builder()
            .method("POST")
            .uri("/parse/function_call")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();

        // Should return 400 (parser not found)
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "Expected BAD_REQUEST (400), got {}",
            resp.status()
        );

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(body_json["success"], false);

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_parse_function_call_missing_fields() {
        let ctx = ParserTestContext::new(vec![]).await;
        let app = ctx.create_app();

        // Missing 'text' field
        let payload = json!({
            "tool_call_parser": "json",
            "tools": []
        });

        let req = Request::builder()
            .method("POST")
            .uri("/parse/function_call")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        // Missing 'tool_call_parser' field
        let payload = json!({
            "text": "some text",
            "tools": []
        });

        let req = Request::builder()
            .method("POST")
            .uri("/parse/function_call")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_parse_function_call_empty_text() {
        let ctx = ParserTestContext::new(vec![]).await;
        let app = ctx.create_app();

        let payload = json!({
            "text": "",
            "tool_call_parser": "json",
            "tools": []
        });

        let req = Request::builder()
            .method("POST")
            .uri("/parse/function_call")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();

        // Parser should handle empty text gracefully - return 200
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Expected OK (200), got {}",
            resp.status()
        );

        ctx.shutdown().await;
    }
}

#[cfg(test)]
mod separate_reasoning_tests {
    use super::*;

    #[tokio::test]
    async fn test_separate_reasoning_success() {
        let ctx = ParserTestContext::new(vec![]).await;
        let app = ctx.create_app();

        let payload = json!({
            "text": "<think>Let me think about this problem. The user is asking for help.</think>Sure, I can help you with that.",
            "reasoning_parser": "step3"
        });

        let req = Request::builder()
            .method("POST")
            .uri("/parse/reasoning")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();

        // Should return 200 with parser factory initialized
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Expected OK (200), got {}",
            resp.status()
        );

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // Check response structure
        assert_eq!(body_json["success"], true);
        assert!(body_json.get("normal_text").is_some());
        assert!(body_json.get("reasoning_text").is_some());

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_separate_reasoning_invalid_parser() {
        let ctx = ParserTestContext::new(vec![]).await;
        let app = ctx.create_app();

        let payload = json!({
            "text": "some text",
            "reasoning_parser": "invalid_parser_type"
        });

        let req = Request::builder()
            .method("POST")
            .uri("/parse/reasoning")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();

        // Should return 400 (parser not found)
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "Expected BAD_REQUEST (400), got {}",
            resp.status()
        );

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(body_json["success"], false);

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_separate_reasoning_missing_fields() {
        let ctx = ParserTestContext::new(vec![]).await;
        let app = ctx.create_app();

        // Missing 'text' field
        let payload = json!({
            "reasoning_parser": "step3"
        });

        let req = Request::builder()
            .method("POST")
            .uri("/parse/reasoning")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        // Missing 'reasoning_parser' field
        let payload = json!({
            "text": "some text"
        });

        let req = Request::builder()
            .method("POST")
            .uri("/parse/reasoning")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_separate_reasoning_empty_text() {
        let ctx = ParserTestContext::new(vec![]).await;
        let app = ctx.create_app();

        let payload = json!({
            "text": "",
            "reasoning_parser": "step3"
        });

        let req = Request::builder()
            .method("POST")
            .uri("/parse/reasoning")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();

        // Parser should handle empty text gracefully
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Expected OK (200), got {}",
            resp.status()
        );

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_separate_reasoning_without_reasoning_tags() {
        let ctx = ParserTestContext::new(vec![]).await;
        let app = ctx.create_app();

        let payload = json!({
            "text": "Just a normal text without any reasoning tags",
            "reasoning_parser": "step3"
        });

        let req = Request::builder()
            .method("POST")
            .uri("/parse/reasoning")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();

        // Should return 200, parser should handle gracefully
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Expected OK (200), got {}",
            resp.status()
        );

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(body_json["success"], true);
        // When there are no reasoning tags, parser returns empty normal_text and empty reasoning_text
        // since the detect_and_parse_reasoning method only extracts if it finds reasoning markers
        assert!(body_json.get("normal_text").is_some());
        assert!(body_json.get("reasoning_text").is_some());

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_separate_reasoning_multiple_reasoning_blocks() {
        let ctx = ParserTestContext::new(vec![]).await;
        let app = ctx.create_app();

        // Some parsers may handle multiple reasoning blocks
        let payload = json!({
            "text": "<think>First thought</think>Text 1<think>Second thought</think>Text 2",
            "reasoning_parser": "step3"
        });

        let req = Request::builder()
            .method("POST")
            .uri("/parse/reasoning")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();

        // Should handle multiple blocks gracefully
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Expected OK (200), got {}",
            resp.status()
        );

        ctx.shutdown().await;
    }
}

#[cfg(test)]
mod api_routing_tests {
    use super::*;

    #[tokio::test]
    async fn test_admin_routes_accessible() {
        let ctx = ParserTestContext::new(vec![]).await;
        let app = ctx.create_app();

        // Test that both endpoints exist and are accessible (even if parser factory not initialized)
        let payload = json!({
            "text": "test",
            "tool_call_parser": "json",
            "tools": []
        });

        let req = Request::builder()
            .method("POST")
            .uri("/parse/function_call")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();

        // Should not be 404
        assert_ne!(resp.status(), StatusCode::NOT_FOUND);

        let payload = json!({
            "text": "test",
            "reasoning_parser": "step3"
        });

        let req = Request::builder()
            .method("POST")
            .uri("/parse/reasoning")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();

        // Should not be 404
        assert_ne!(resp.status(), StatusCode::NOT_FOUND);

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_endpoints_only_accept_post() {
        let ctx = ParserTestContext::new(vec![]).await;
        let app = ctx.create_app();

        // Test GET request to parse/function_call
        let req = Request::builder()
            .method("GET")
            .uri("/parse/function_call")
            .body(Body::empty())
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();

        // Should not accept GET (should be 405 or 404)
        assert!(
            resp.status() == StatusCode::METHOD_NOT_ALLOWED
                || resp.status() == StatusCode::NOT_FOUND,
            "Expected METHOD_NOT_ALLOWED (405) or NOT_FOUND (404), got {}",
            resp.status()
        );

        // Test GET request to parse/reasoning
        let req = Request::builder()
            .method("GET")
            .uri("/parse/reasoning")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();

        // Should not accept GET
        assert!(
            resp.status() == StatusCode::METHOD_NOT_ALLOWED
                || resp.status() == StatusCode::NOT_FOUND,
            "Expected METHOD_NOT_ALLOWED (405) or NOT_FOUND (404), got {}",
            resp.status()
        );

        ctx.shutdown().await;
    }
}

/// The plain-body routes of the admin plane (the two parse routes, the two
/// profile routes) refuse a body the type does not take in the control
/// plane's envelope (`{"error", "code"}`), as the worker, WASM module and
/// tokenizer routes do, not with the framework's text/plain answer.
#[cfg(test)]
mod admin_body_refusals {
    use super::*;

    /// The status and the body of `body` posted to `path` under
    /// `content_type` (no header when `None`).
    async fn post(
        app: &axum::Router,
        path: &str,
        content_type: Option<&str>,
        body: &str,
    ) -> (StatusCode, String) {
        let mut req = Request::builder().method("POST").uri(path);
        if let Some(content_type) = content_type {
            req = req.header(CONTENT_TYPE, content_type);
        }
        let resp = app
            .clone()
            .oneshot(req.body(Body::from(body.to_string())).unwrap())
            .await
            .unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        (status, String::from_utf8_lossy(&bytes).into_owned())
    }

    /// `text` is the control plane's envelope: an `error` string and the
    /// code `BAD_REQUEST`; the panic names the route and the body otherwise.
    fn assert_refusal(case: &str, status: StatusCode, text: &str) {
        assert_eq!(status, StatusCode::BAD_REQUEST, "{case}: {text}");
        let body: serde_json::Value = serde_json::from_str(text)
            .unwrap_or_else(|e| panic!("{case}: not the control plane's envelope ({e}): {text}"));
        assert_eq!(body["code"], "BAD_REQUEST", "{case}: {text}");
        assert!(body["error"].is_string(), "{case}: {text}");
    }

    /// An array, null, a string and broken JSON under `application/json`
    /// are a 400 in the envelope on the parse routes and on `/start_profile`;
    /// `/stop_profile` refuses the same bodies but an array, which serde
    /// reads as a struct of defaults from an empty positional form. An object
    /// under no content type is refused on the parse routes, whose body is
    /// required, and read as no body on the profile routes, whose optional
    /// form reads a request without a content-type header as no body, as
    /// the framework's does. A profile route with no body at all answers as
    /// it does with an empty object: the defaults.
    #[tokio::test]
    async fn the_plain_body_routes_refuse_a_body_the_type_does_not_take_in_the_envelope() {
        let ctx = ParserTestContext::new(vec![]).await;
        let app = ctx.create_app();
        let bodies = [
            (Some("application/json"), "[]"),
            (Some("application/json"), "null"),
            (Some("application/json"), r#""text""#),
            (Some("application/json"), "{not json"),
            (None, "{}"),
        ];
        for path in [
            "/parse/function_call",
            "/parse/reasoning",
            "/start_profile",
            "/stop_profile",
        ] {
            for (content_type, body) in bodies {
                if path == "/stop_profile" && body == "[]" {
                    continue;
                }
                if path.ends_with("_profile") && content_type.is_none() {
                    continue;
                }
                let (status, text) = post(&app, path, content_type, body).await;
                assert_refusal(
                    &format!("{path} {content_type:?} {body}"),
                    status,
                    text.as_str(),
                );
            }
        }
        for path in ["/start_profile", "/stop_profile"] {
            let (without_body, _) = post(&app, path, None, "").await;
            let (empty_object, _) = post(&app, path, Some("application/json"), "{}").await;
            assert_eq!(
                without_body, empty_object,
                "{path}: no body and an empty object take the same defaults"
            );
            assert_ne!(
                without_body,
                StatusCode::BAD_REQUEST,
                "{path}: no body is not a refusal"
            );
            let (no_content_type, _) = post(&app, path, None, "{}").await;
            assert_eq!(
                without_body, no_content_type,
                "{path}: an object under no content type is read as no body"
            );
        }
        ctx.shutdown().await;
    }
}
