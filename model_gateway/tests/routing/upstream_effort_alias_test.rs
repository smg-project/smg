//! What the HTTP routers forward for a reasoning effort the gateway's own
//! renderer and the model vendor's API read differently.
//!
//! A DeepSeek V4.1 `medium` is a level of the checkpoint's chat template to
//! the gateway's renderer and the vendor API's alias of `high`. An upstream
//! the HTTP routers proxy to renders the prompt itself, so the top-level
//! fields reach it with the vendor's reading, while a chat-template kwarg,
//! the client's word to the upstream's template, goes out as sent.

use std::sync::Arc;

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use openai_protocol::{model_card::ModelCard, profile::ProviderProfile};
use serde_json::{json, Value};
use smg::{config::RouterConfig, worker::BasicWorkerBuilder};
use tower::ServiceExt;

use crate::common::{
    mock_worker::{set_request_recorder, RequestRecorder},
    AppTestContext, TestWorkerConfig,
};

#[cfg(test)]
mod upstream_effort_alias_tests {
    use super::*;

    const V41_MODEL: &str = "deepseek-v4.1-flash";
    const V4_MODEL: &str = "deepseek-v4-flash";

    /// Mock workers advertise a generic model card; the started worker is
    /// re-registered under the DeepSeek model ids so a request for them
    /// selects the vendor profile and routes to it.
    fn serve_deepseek_models(ctx: &AppTestContext, url: &str) {
        serve_models(
            ctx,
            url,
            vec![ModelCard::new(V41_MODEL), ModelCard::new(V4_MODEL)],
        );
    }

    /// Re-register the started worker at `url` under `models`.
    fn serve_models(ctx: &AppTestContext, url: &str, models: Vec<ModelCard>) {
        let registry = &ctx.app_context.worker_registry;
        let worker_id = registry.get_id_by_url(url).unwrap();
        let worker = registry.get(&worker_id).unwrap();
        let mut spec = worker.metadata().spec.as_ref().clone();
        spec.models = models.into();
        let replacement = BasicWorkerBuilder::from_spec(spec)
            .health_config(worker.metadata().health_config.clone())
            .health_endpoint(&worker.metadata().health_endpoint)
            .build();
        assert!(registry.replace(&worker_id, Arc::new(replacement)));
    }

    fn chat_request(body: &Value) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    /// Send one chat request and return the body the worker recorded for it.
    async fn forwarded(app: &axum::Router, recorder: &RequestRecorder, body: Value) -> Value {
        let seen = recorder.bodies().len();
        let response = app.clone().oneshot(chat_request(&body)).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK, "{body}");
        let bodies = recorder.bodies();
        assert_eq!(bodies.len(), seen + 1, "one request reaches the worker");
        bodies.into_iter().last().unwrap()
    }

    #[tokio::test]
    async fn regular_router_forwards_a_v41_top_level_medium_as_the_vendors_high() {
        let recorder = RequestRecorder::new();
        set_request_recorder(19870, Arc::clone(&recorder));
        let ctx = AppTestContext::new(vec![TestWorkerConfig::healthy(19870)]).await;
        let app = ctx.create_app();
        serve_deepseek_models(&ctx, ctx.worker_url_for(19870));

        // Both top-level spellings reach the upstream as `high`.
        let body = forwarded(
            &app,
            &recorder,
            json!({
                "model": V41_MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
                "reasoning_effort": "medium",
                "thinking": {"effort": "medium"},
                "stream": false
            }),
        )
        .await;
        assert_eq!(body["reasoning_effort"], "high", "{body}");
        assert_eq!(body["thinking"]["effort"], "high", "{body}");

        // A template kwarg is the client's word to the upstream's template
        // and goes out as sent; so does `xhigh`.
        let body = forwarded(
            &app,
            &recorder,
            json!({
                "model": V41_MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
                "reasoning_effort": "xhigh",
                "chat_template_kwargs": {"reasoning_effort": "medium"},
                "stream": false
            }),
        )
        .await;
        assert_eq!(body["reasoning_effort"], "xhigh", "{body}");
        assert_eq!(
            body["chat_template_kwargs"]["reasoning_effort"], "medium",
            "{body}"
        );

        // V4: the profile's alias at validation, as before.
        let body = forwarded(
            &app,
            &recorder,
            json!({
                "model": V4_MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
                "reasoning_effort": "medium",
                "stream": false
            }),
        )
        .await;
        assert_eq!(body["reasoning_effort"], "high", "{body}");

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn pd_router_forwards_a_v41_top_level_medium_as_the_vendors_high() {
        let prefill_recorder = RequestRecorder::new();
        let decode_recorder = RequestRecorder::new();
        set_request_recorder(19871, Arc::clone(&prefill_recorder));
        set_request_recorder(19872, Arc::clone(&decode_recorder));

        let mut config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![("http://127.0.0.1:19871".to_string(), None)],
                vec!["http://127.0.0.1:19872".to_string()],
            )
            .round_robin_policy()
            .host("127.0.0.1")
            .port(3806)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();
        config.health_check.disable_health_check = true;

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::prefill(19871),
                TestWorkerConfig::decode(19872),
            ],
        )
        .await;
        let app = ctx.create_app();
        for port in [19871, 19872] {
            serve_deepseek_models(&ctx, ctx.worker_url_for(port));
        }

        let body = json!({
            "model": V41_MODEL,
            "messages": [{"role": "user", "content": "Hello"}],
            "reasoning_effort": "medium",
            "chat_template_kwargs": {"reasoning_effort": "medium"},
            "stream": false
        });
        let response = app.clone().oneshot(chat_request(&body)).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        for (side, recorder) in [("prefill", &prefill_recorder), ("decode", &decode_recorder)] {
            let body = recorder.only_body();
            assert_eq!(body["reasoning_effort"], "high", "{side}: {body}");
            assert_eq!(
                body["chat_template_kwargs"]["reasoning_effort"], "medium",
                "{side}: {body}"
            );
        }

        ctx.shutdown().await;
    }

    /// A registered alias of a V4.1 model: the profile reads the request as
    /// the model the alias resolves to, so what the router forwards under
    /// the alias is what it forwards under the model's own name.
    #[tokio::test]
    async fn regular_router_forwards_a_v41_alias_like_the_model_itself() {
        const V41_ALIAS: &str = "company-chat-large";
        let recorder = RequestRecorder::new();
        set_request_recorder(19873, Arc::clone(&recorder));
        let ctx = AppTestContext::new(vec![TestWorkerConfig::healthy(19873)]).await;
        let app = ctx.create_app();
        ProviderProfile::register_model_aliases([(V41_ALIAS, V41_MODEL)]);
        serve_models(
            &ctx,
            ctx.worker_url_for(19873),
            vec![ModelCard::new(V41_MODEL).with_alias(V41_ALIAS)],
        );

        for request in [
            // A template kwarg `medium` is a level of V4.1's own table: it
            // switches thinking on ahead of the typed toggle.
            json!({
                "messages": [{"role": "user", "content": "Hello"}],
                "reasoning_effort": "medium",
                "thinking": {"type": "disabled"},
                "chat_template_kwargs": {"reasoning_effort": "medium"},
                "stream": false
            }),
            // An integer budget is V4.1's extension.
            json!({
                "messages": [{"role": "user", "content": "Hello"}],
                "reasoning_effort": 62,
                "stream": false
            }),
        ] {
            let mut under_alias = request.clone();
            under_alias["model"] = json!(V41_ALIAS);
            let mut under_name = request;
            under_name["model"] = json!(V41_MODEL);
            let alias_body = forwarded(&app, &recorder, under_alias).await;
            let name_body = forwarded(&app, &recorder, under_name).await;
            for field in [
                "model",
                "reasoning_effort",
                "thinking",
                "chat_template_kwargs",
            ] {
                assert_eq!(
                    alias_body[field], name_body[field],
                    "{field}: {alias_body} vs {name_body}"
                );
            }
        }

        ctx.shutdown().await;
    }
}
