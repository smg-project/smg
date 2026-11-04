//! The gRPC router's `/v1/responses` route counts in the router request
//! families once per client request, as the HTTP router counts the same
//! route: one request and one duration sample for a turn whose tool loop
//! runs the pipeline twice, streamed or not, and one error sample under its
//! class and reason for a request the route refuses on its own.
//!
//! The recorder is this binary's, installed once for all its tests (see
//! `recorder()` in the group's `mod.rs`); the series read here carry this
//! test's own model label.

#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "test failures should panic; the in-process worker and MCP server live for the test"
)]

use std::{sync::Arc, time::Duration};

use axum::{body::to_bytes, http::StatusCode};
use llm_tokenizer::{traits::Tokenizer, TokenizerRegistry};
use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
use serde_json::{json, Value};
use smg::{
    config::{RouterConfig, RoutingMode},
    middleware::TenantRequestMeta,
    routers::{RouterFactory, RouterTrait},
    tenant::TenantKey,
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerType},
};
use smg_test_harness::{mock_mcp_server::MockMCPServer, scripted_tokenizer, scripted_worker};
use tokio::{net::TcpListener, time::timeout};

use crate::common;

const MODEL: &str = "responses-route-series-model";
const UNSERVED_MODEL: &str = "responses-route-series-nobody-serves";
const ALIASED_MODEL: &str = "responses-route-series-aliased-model";
const ALIAS: &str = "responses-route-series-alias";
const REFUSING_MODEL: &str = "responses-route-series-refusing-model";
const TOOL_CALL: &str =
    "<tool_call>\n{\"name\":\"brave_web_search\",\"arguments\":{\"query\":\"test\"}}\n</tool_call>\n";
const ROUTE_LABELS: [&str; 4] = [
    "router_type=\"grpc\"",
    "backend_type=\"regular\"",
    "connection_mode=\"grpc\"",
    "endpoint=\"responses\"",
];

/// The value of the one series of `family` whose labels contain every label
/// of `ROUTE_LABELS` and of `labels`, if the series exists.
fn series(rendered: &str, family: &str, labels: &[&str]) -> Option<f64> {
    rendered
        .lines()
        .find(|line| {
            line.starts_with(&format!("{family}{{"))
                && ROUTE_LABELS.iter().all(|label| line.contains(label))
                && labels.iter().all(|label| line.contains(label))
        })
        .map(|line| line.rsplit(' ').next().unwrap().parse().unwrap())
}

/// A gRPC router in front of a scripted worker serving `card`, whose every
/// answer is one MCP tool call under the generations' `finish_reasons`
/// (`"unavailable"` in a generation's place refuses it), under a tokenizer
/// that decodes the answer as such.
async fn router_with_tool_calling_worker(
    card: ModelCard,
    finish_reasons: Vec<&'static str>,
) -> (Box<dyn RouterTrait>, tokio::task::JoinHandle<()>) {
    let model = card.id.clone();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = tokio::spawn(
        scripted_worker::ScriptedWorker {
            output_tokens: 2,
            finish_reasons,
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
            .model(card)
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
    config.tool_call_parser = Some("qwen".into());
    let tokenizers = Arc::new(TokenizerRegistry::new());
    let tokenizer: Arc<dyn Tokenizer> =
        Arc::new(scripted_tokenizer::ScriptedTokenizer::new(TOOL_CALL));
    tokenizers
        .load(
            "tokenizer-id",
            &model,
            "test",
            || async move { Ok(tokenizer) },
        )
        .await
        .unwrap();
    let context = common::create_test_context_with_tokenizer_registry(config, tokenizers).await;
    context.worker_registry.register(worker).unwrap();
    let router = RouterFactory::create_router(&context).await.unwrap();
    (router, server)
}

/// One Responses turn of `model` through `router`, `stream`ed or not, read to
/// its end: the status and the final response object (the JSON answer, or
/// the terminal event's response).
async fn responses_turn(
    router: &dyn RouterTrait,
    model: &str,
    stream: bool,
    tools: Value,
) -> (StatusCode, Value) {
    let tenant = TenantRequestMeta::new(TenantKey::new("test-tenant"));
    let request = serde_json::from_value(json!({
        "model": model, "input": "Hello", "stream": stream,
        "max_output_tokens": 16, "tools": tools, "max_tool_calls": 1,
    }))
    .unwrap();
    timeout(Duration::from_secs(30), async {
        let response = router.route_responses(None, &tenant, request, model).await;
        let status = response.status();
        let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let body = std::str::from_utf8(&bytes).unwrap();
        let object = if stream && status.is_success() {
            body.lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .map(|data| serde_json::from_str::<Value>(data).unwrap())
                .next_back()
                .map(|terminal| terminal["response"].clone())
                .unwrap()
        } else {
            serde_json::from_str(body).unwrap()
        };
        (status, object)
    })
    .await
    .expect("the Responses turn ends")
}

#[tokio::test]
async fn a_responses_turn_counts_once_at_the_route_whatever_its_tool_loop_ran() {
    let handle = crate::metrics::recorder();
    let mut mcp = MockMCPServer::start().await.unwrap();
    let tools = json!([{
        "type": "mcp", "server_label": "test-tools", "server_url": mcp.url(),
        "require_approval": "never"
    }]);
    let (router, server) =
        router_with_tool_calling_worker(ModelCard::new(MODEL), vec!["stop"]).await;
    let model = format!("model=\"{MODEL}\"");

    for (stream, turns) in [(false, 1.0), (true, 2.0)] {
        let (status, object) = responses_turn(router.as_ref(), MODEL, stream, tools.clone()).await;
        assert_eq!(status, StatusCode::OK, "{object}");
        assert_eq!(object["status"], "completed", "{object}");
        // Two model calls per turn: the executed call, then the call the cap
        // of one leaves unexecuted, as the Responses contract tests pin it.
        assert_eq!(object["usage"]["input_tokens"], 2, "{object}");

        let rendered = handle.render();
        let streaming = format!("streaming=\"{stream}\"");
        assert_eq!(
            series(
                &rendered,
                "smg_router_requests_total",
                &[&model, &streaming]
            ),
            Some(1.0),
            "one request per client request, not per iteration (stream {stream}): {rendered}"
        );
        assert_eq!(
            series(
                &rendered,
                "smg_router_request_duration_seconds_count",
                &[&model]
            ),
            Some(turns),
            "one duration sample per answered turn (stream {stream})"
        );
        assert_eq!(
            series(&rendered, "smg_router_request_errors_total", &[&model]),
            None,
            "an answered turn is no error (stream {stream})"
        );
        assert!(
            !rendered.lines().any(|line| {
                line.starts_with("smg_router_requests_total{")
                    && line.contains(&model)
                    && line.contains("endpoint=\"chat\"")
            }),
            "the iterations count under no endpoint of their own (stream {stream}): {rendered}"
        );
    }

    // A request the route refuses on its own, before any pipeline run.
    let (status, object) = responses_turn(router.as_ref(), UNSERVED_MODEL, false, tools).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{object}");
    let rendered = handle.render();
    assert_eq!(
        series(
            &rendered,
            "smg_router_requests_total",
            &["model=\"other\"", "streaming=\"false\""],
        ),
        Some(1.0),
        "the refused request counts, under the bounded model label"
    );
    assert_eq!(
        series(
            &rendered,
            "smg_router_request_errors_total",
            &[
                "model=\"other\"",
                "error_type=\"no_workers\"",
                "reason=\"model_not_found\""
            ],
        ),
        Some(1.0),
        "the refusal counts once under its class and reason: {rendered}"
    );

    mcp.stop().await;
    server.abort();
}

/// A request under a served alias counts under the served id, the label the
/// run's own samples carry, so the request and whatever its run records sit
/// on one series.
#[tokio::test]
async fn a_request_under_an_alias_counts_under_the_served_id() {
    let handle = crate::metrics::recorder();
    let (router, server) = router_with_tool_calling_worker(
        ModelCard::new(ALIASED_MODEL).with_alias(ALIAS),
        vec!["stop"],
    )
    .await;

    let (status, object) = responses_turn(router.as_ref(), ALIAS, false, json!([])).await;
    assert_eq!(status, StatusCode::OK, "{object}");
    let rendered = handle.render();
    assert_eq!(
        series(
            &rendered,
            "smg_router_requests_total",
            &[&format!("model=\"{ALIASED_MODEL}\""), "streaming=\"false\""]
        ),
        Some(1.0),
        "the request counts under the served id: {rendered}"
    );
    assert!(
        !rendered.contains(&format!("model=\"{ALIAS}\"")),
        "nothing counts under the alias: {rendered}"
    );

    server.abort();
}

/// A tool loop whose later iteration the worker refuses fails inside the
/// answered stream, where the route's own answer cannot count it: the turn
/// counts once as a request and once as an error under the route, and not
/// under the iterations' chat endpoint.
#[tokio::test]
async fn a_later_iteration_refused_inside_the_stream_counts_one_error_under_the_route() {
    let handle = crate::metrics::recorder();
    let mut mcp = MockMCPServer::start().await.unwrap();
    let tools = json!([{
        "type": "mcp", "server_label": "test-tools", "server_url": mcp.url(),
        "require_approval": "never"
    }]);
    let (router, server) = router_with_tool_calling_worker(
        ModelCard::new(REFUSING_MODEL),
        vec!["stop", "unavailable"],
    )
    .await;
    let model = format!("model=\"{REFUSING_MODEL}\"");

    // The first generation's tool call runs; the second generation is refused.
    let (status, _) = responses_turn(router.as_ref(), REFUSING_MODEL, true, tools).await;
    assert_eq!(status, StatusCode::OK, "the stream's head is answered");
    let rendered = handle.render();
    assert_eq!(
        series(
            &rendered,
            "smg_router_requests_total",
            &[&model, "streaming=\"true\""]
        ),
        Some(1.0),
        "one request for the turn: {rendered}"
    );
    assert_eq!(
        series(&rendered, "smg_router_request_errors_total", &[&model]),
        Some(1.0),
        "the refused iteration counts once under the route: {rendered}"
    );
    assert!(
        !rendered.lines().any(|line| {
            line.starts_with("smg_router_request_errors_total{")
                && line.contains(&model)
                && line.contains("endpoint=\"chat\"")
        }),
        "the iteration's failure counts under no endpoint of its own: {rendered}"
    );

    mcp.stop().await;
    server.abort();
}
