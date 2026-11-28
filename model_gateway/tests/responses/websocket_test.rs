//! The experimental transport must reuse production Responses routing.
use std::{net::SocketAddr, sync::Arc, time::Duration};

use axum::{body::Body, http::Request};
use futures::{SinkExt, StreamExt};
use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
use serde_json::{json, Value};
use smg::{
    config::{RouterConfig, RoutingMode},
    routers::{grpc::backend_client::BasicWorkerBuilder, RouterFactory},
};
use smg_test_harness::{
    grpc_harness::grpc_test_context_with_tokenizer, scripted_tokenizer::ScriptedTokenizer,
    scripted_worker::ScriptedWorker, test_app::create_test_app_with_context,
};
use smg_worker::identity::basic::{ConnectionMode, RuntimeType, WorkerType};
use tokio::{net::TcpListener, task::AbortHandle, time::timeout};
use tokio_tungstenite::{
    connect_async,
    tungstenite::{client::IntoClientRequest, Error as WsError, Message},
};
use tower::ServiceExt;

struct Servers(Vec<AbortHandle>);
impl Drop for Servers {
    fn drop(&mut self) {
        for server in &self.0 {
            server.abort();
        }
    }
}

#[expect(
    clippy::disallowed_methods,
    reason = "synthetic CPU fixture: failed startup or protocol assertions must fail the test"
)]
#[tokio::test]
async fn native_responses_websocket_uses_auth_validation_and_grpc_tool_pipeline() {
    const MODEL: &str = "responses-websocket-fixture";
    let worker_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let worker_address = worker_listener.local_addr().unwrap();
    let worker = ScriptedWorker {
        output_tokens: 2,
        finish_reasons: vec!["tool_calls"],
        generation: Default::default(),
        requests: Default::default(),
    };
    let requests = worker.requests.clone();
    let worker_server = tokio::spawn(worker.serve(worker_listener));
    let mut servers = Servers(vec![worker_server.abort_handle()]);
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
    config.api_key = Some("fixture-key".into());
    config.enable_responses_websocket = true;
    config.max_concurrent_requests = 1;
    config.cors_allowed_origins = vec!["https://allowed.example".into()];
    let tokenizer = Arc::new(ScriptedTokenizer::new(
        r#"<tool_call>{"name":"echo","arguments":{"message":"fixture"}}</tool_call>"#,
    ));
    let context = grpc_test_context_with_tokenizer(config, MODEL, tokenizer).await;
    context
        .worker_registry
        .register(Arc::new(
            BasicWorkerBuilder::new(format!("grpc://{worker_address}"))
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Grpc)
                .runtime_type(RuntimeType::TokenSpeed)
                .model(ModelCard::new(MODEL))
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        ))
        .unwrap();
    let router = RouterFactory::create_router(&context).await.unwrap();
    let tracker = context.inflight_tracker.clone();
    let app = create_test_app_with_context(router.into(), context);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await
        .unwrap();
    });
    servers.0.push(server.abort_handle());
    let url = format!("ws://{address}/v1/responses");
    let denied = connect_async(&url).await.unwrap_err();
    assert!(matches!(denied, WsError::Http(response) if response.status() == 401));
    let mut denied_origin = url.as_str().into_client_request().unwrap();
    denied_origin
        .headers_mut()
        .insert("authorization", "Bearer fixture-key".parse().unwrap());
    denied_origin
        .headers_mut()
        .insert("origin", "https://unlisted.example".parse().unwrap());
    let denied_origin = connect_async(denied_origin).await.unwrap_err();
    assert!(matches!(denied_origin, WsError::Http(response) if response.status() == 403));
    let mut upgrade = url.as_str().into_client_request().unwrap();
    upgrade
        .headers_mut()
        .insert("authorization", "Bearer fixture-key".parse().unwrap());
    upgrade
        .headers_mut()
        .insert("origin", "https://allowed.example".parse().unwrap());
    let (mut socket, _) = connect_async(upgrade).await.unwrap();
    socket
        .send(Message::Text(
            json!({
                "type": "response.create", "model": "missing", "generate": false, "input": "fixture",
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
    let refusal = timeout(Duration::from_secs(3), socket.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap()
        .into_text()
        .unwrap();
    let refusal: Value = serde_json::from_str(&refusal).unwrap();
    assert_eq!(refusal["status"], 404);
    assert_eq!(refusal["error"]["code"], "model_not_found");
    assert!(requests.lock().unwrap().is_empty());
    socket
        .send(Message::Text(
            json!({"type":"response.create","model":MODEL,"store":null,"input":"fixture input"})
                .to_string()
                .into(),
        ))
        .await
        .unwrap();
    let rejected = timeout(Duration::from_secs(3), socket.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap()
        .into_text()
        .unwrap();
    let rejected: Value = serde_json::from_str(&rejected).unwrap();
    assert_eq!(rejected["error"]["code"], "invalid_type");
    assert_eq!(rejected["error"]["param"], "store");
    assert!(requests.lock().unwrap().is_empty());
    let tools = json!([{"type":"function","name":"echo","parameters":{"type":"object","properties":{"message":{"type":"string"}}}}]);
    let mut parent = Value::Null;
    let mut call_id = Value::Null;
    for turn in 0..3 {
        let input = if turn == 2 {
            json!([{"type":"function_call_output","call_id":call_id,"output":"fixture result"}])
        } else {
            json!("fixture input")
        };
        let mut request = json!({"type":"response.create","model":MODEL,"store":false,"generate":turn != 0,"input":input,"tools":tools,"max_output_tokens":16});
        if !parent.is_null() {
            request["previous_response_id"] = parent.clone();
        }
        socket
            .send(Message::Text(request.to_string().into()))
            .await
            .unwrap();
        let terminal = timeout(Duration::from_secs(20), async {
            loop {
                let event = socket.next().await.unwrap().unwrap().into_text().unwrap();
                let event: Value = serde_json::from_str(&event).unwrap();
                assert_ne!(event["type"], "error", "{event}");
                if matches!(
                    event["type"].as_str(),
                    Some("response.completed" | "response.incomplete" | "response.failed")
                ) {
                    break event;
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(terminal["type"], "response.completed", "{terminal}");
        parent = terminal["response"]["id"].clone();
        if turn == 0 {
            assert_eq!(terminal["response"]["store"], false);
            assert_eq!(terminal["response"]["usage"]["total_tokens"], 0);
            assert_eq!(
                terminal["response"]["usage"]["input_tokens_details"]["cached_tokens"],
                0
            );
            let tool = &terminal["response"]["tools"][0];
            assert_eq!(tool["type"], "function");
            assert_eq!(tool["name"], "echo");
            assert_eq!(tool["parameters"], tools[0]["parameters"]);
            assert!(terminal["response"]["output"]
                .as_array()
                .unwrap()
                .is_empty());
            assert!(
                requests.lock().unwrap().is_empty(),
                "prewarming must not dispatch inference"
            );
        } else {
            let output = &terminal["response"]["output"][0];
            assert_eq!(output["type"], "function_call");
            assert_eq!(output["name"], "echo");
            assert_eq!(
                serde_json::from_str::<Value>(output["arguments"].as_str().unwrap()).unwrap(),
                json!({"message":"fixture"})
            );
            assert_ne!(output["call_id"], call_id);
            call_id = output["call_id"].clone();
        }
    }
    assert_eq!(requests.lock().unwrap().len(), 2);
    assert!(tracker.wait_for_drain(Duration::from_secs(3)).await);
    tracker.begin_drain();
    let close = timeout(Duration::from_secs(3), socket.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(matches!(close, Message::Close(Some(frame)) if u16::from(frame.code) == 1012));
}

#[tokio::test]
async fn responses_get_is_disabled_by_default() {
    let config = RouterConfig::default();
    let context = smg_test_harness::test_app::create_test_app_context_with_config(&config).await;
    let router = RouterFactory::create_router(&context).await.unwrap();
    let app = create_test_app_with_context(router.into(), context);
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/responses")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 405);
}

#[cfg(feature = "provider-openai")]
#[expect(
    clippy::disallowed_methods,
    reason = "synthetic loopback servers are owned by the fixture"
)]
#[tokio::test]
async fn prewarm_discovers_openai_models_before_the_first_generation() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use axum::{http::HeaderMap, routing::get, Json, Router};
    use smg_test_harness::test_app::create_test_app_context_with_config;

    const MODEL: &str = "cold-openai-fixture";
    let discovery_calls = Arc::new(AtomicUsize::new(0));
    let observed = discovery_calls.clone();
    // The fixture publishes model inventory but has no generation endpoint.
    let worker = Router::new().route(
        "/v1/models",
        get(move |headers: HeaderMap| async move {
            assert!(!headers.contains_key("authorization"));
            observed.fetch_add(1, Ordering::SeqCst);
            Json(json!({"object": "list", "data": [{"id": MODEL, "object": "model"}]}))
        }),
    );
    let worker_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let worker_url = format!("http://{}", worker_listener.local_addr().unwrap());
    let worker_server =
        tokio::spawn(async move { axum::serve(worker_listener, worker).await.unwrap() });
    let mut servers = Servers(vec![worker_server.abort_handle()]);
    let mut config = RouterConfig::builder()
        .openai_mode(vec![])
        .build_unchecked();
    config.api_key = Some("fixture-key".into());
    config.enable_responses_websocket = true;
    config.health_check.disable_health_check = true;
    let context = create_test_app_context_with_config(&config).await;
    context
        .worker_registry
        .register(Arc::new(
            BasicWorkerBuilder::new(worker_url)
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Http)
                .runtime_type(RuntimeType::External)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        ))
        .unwrap();
    let registry = context.worker_registry.clone();
    assert!(!registry.contains_model(MODEL));
    let router = RouterFactory::create_router(&context).await.unwrap();
    let app = create_test_app_with_context(router.into(), context);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("ws://{}/v1/responses", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    servers.0.push(server.abort_handle());

    let mut request = url.into_client_request().unwrap();
    request
        .headers_mut()
        .insert("authorization", "Bearer fixture-key".parse().unwrap());
    let (mut socket, _) = connect_async(request).await.unwrap();
    for expected_model in [MODEL, MODEL, "unknown-fixture"] {
        socket.send(Message::Text(json!({
            "type": "response.create", "generate": false, "model": expected_model, "input": "fixture"
        }).to_string().into())).await.unwrap();
        let result = timeout(Duration::from_secs(6), async {
            loop {
                let event = socket.next().await.unwrap().unwrap().into_text().unwrap();
                let value: Value = serde_json::from_str(&event).unwrap();
                if value["type"] != "response.created" {
                    break value;
                }
            }
        })
        .await
        .unwrap();
        if expected_model == MODEL {
            assert_eq!(result["type"], "response.completed", "{result}");
            assert_eq!(result["response"]["usage"]["total_tokens"], 0);
            assert_eq!(discovery_calls.load(Ordering::SeqCst), 1);
        } else {
            assert_eq!(result["status"], 404, "{result}");
            assert_eq!(result["error"]["code"], "model_not_found");
        }
    }
}
