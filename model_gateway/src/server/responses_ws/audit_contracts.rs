use futures::{SinkExt, StreamExt};
use tokio::sync::Notify;
use tokio_tungstenite::{
    connect_async, tungstenite::Message as ClientMessage, MaybeTlsStream, WebSocketStream,
};

use super::*;

struct Fixture {
    socket: WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>,
    app: axum::Router,
    calls: Arc<AtomicUsize>,
    server: tokio::task::JoinHandle<()>,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        self.server.abort();
    }
}
struct NativeFixture {
    fixture: Fixture,
    auth: crate::middleware::AuthConfig,
    tracker: Arc<InFlightRequestTracker>,
    drops: Arc<AtomicUsize>,
    post_body_ends: Arc<AtomicUsize>,
    resume: Arc<Notify>,
    url: String,
    bucket: Arc<crate::middleware::TokenBucket>,
}

fn admission_state(bucket: Arc<crate::middleware::TokenBucket>) -> Arc<crate::server::AppState> {
    use std::sync::OnceLock;

    use llm_tokenizer::TokenizerRegistry;
    use smg_data_connector::{
        MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
    };

    use crate::{
        app_context::AppContext,
        config::RouterConfig,
        policies::PolicyRegistry,
        routers::gateway::Gateway,
        server::{health::ProbeState, http_client, AppState},
        worker::WorkerRegistry,
    };
    let config = RouterConfig::default();
    let context = Arc::new(
        AppContext::builder()
            .client(http_client())
            .rate_limiter(Some(bucket))
            .tokenizer_registry(Arc::new(TokenizerRegistry::new()))
            .reasoning_parser_factory(None)
            .tool_parser_factory(None)
            .worker_registry(Arc::new(WorkerRegistry::new()))
            .policy_registry(Arc::new(PolicyRegistry::new(config.policy.clone())))
            .router_config(config)
            .response_storage(Arc::new(MemoryResponseStorage::new()))
            .conversation_storage(Arc::new(MemoryConversationStorage::new()))
            .conversation_item_storage(Arc::new(MemoryConversationItemStorage::new()))
            .worker_monitor(None)
            .worker_job_queue(Arc::new(OnceLock::new()))
            .workflow_engines(Arc::new(OnceLock::new()))
            .mcp_orchestrator(Arc::new(OnceLock::new()))
            .build()
            .unwrap(),
    );
    Arc::new(AppState {
        secret_sources: Default::default(),
        secrets: None,
        router: Arc::new(Gateway::new(context.worker_registry.clone())),
        probe_state: ProbeState::new(context.inflight_tracker.clone()),
        context,
        admission_queue: None,
        gateway: None,
        mesh_handler: None,
        mesh_adapters: None,
    })
}
impl Fixture {
    #[expect(
        clippy::disallowed_methods,
        reason = "Fixture owns and aborts the loopback server in Drop"
    )]
    async fn new(deny: bool) -> Self {
        let calls = Arc::new(AtomicUsize::new(0));
        let counted = calls.clone();
        let protected = axum::Router::new().route("/v1/responses", axum::routing::post(move |axum::Json(body): axum::Json<Value>| {
            let counted = counted.clone();
            async move {
                let number = counted.fetch_add(1, Ordering::SeqCst);
                if deny {
                    return (StatusCode::FORBIDDEN, axum::Json(json!({"error":{"code":"fixture_denied"}}))).into_response();
                }
                let fail = body["input"].as_array().is_some_and(|items| items.last().is_some_and(|item| item["content"] == "fail"));
                let response = json!({"id":format!("resp_{number}"),"status":if fail {"failed"} else {"completed"},"output":[]});
                let event = json!({"type":if fail {"response.failed"} else {"response.completed"},"response":response});
                ([("content-type","text/event-stream")], format!("data: {event}\n\n")).into_response()
            }
        }));
        let dispatch = Dispatch {
            service: protected
                .clone()
                .layer(axum::extract::DefaultBodyLimit::max(512)),
            tracker: InFlightRequestTracker::new(),
            max_payload_bytes: 512,
            origins: Arc::default(),
            connections: Arc::default(),
        };
        let websocket = axum::Router::new()
            .route("/v1/responses", axum::routing::get(upgrade))
            .layer(Extension(dispatch))
            .layer(Extension(DataPlaneCaller::new("fixture".into())))
            .layer(axum::middleware::from_fn(capture_credential));
        let app = protected
            .merge(websocket)
            .layer(axum::extract::DefaultBodyLimit::max(512));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let served = app.clone();
        let server = tokio::spawn(async move {
            axum::serve(listener, served).await.unwrap();
        });
        let (socket, _) = connect_async(format!("ws://{address}/v1/responses"))
            .await
            .unwrap();
        Self {
            socket,
            app,
            calls,
            server,
        }
    }
    async fn turn(&mut self, body: Value) -> Value {
        self.socket
            .send(ClientMessage::Text(body.to_string().into()))
            .await
            .unwrap();
        loop {
            let message = tokio::time::timeout(Duration::from_secs(3), self.socket.next())
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            if let ClientMessage::Text(text) = message {
                let value: Value = serde_json::from_str(&text).unwrap();
                if matches!(
                    value["type"].as_str(),
                    Some(
                        "error" | "response.completed" | "response.incomplete" | "response.failed"
                    )
                ) {
                    return value;
                }
            }
        }
    }

    #[expect(
        clippy::disallowed_methods,
        reason = "Fixture owns and aborts the loopback server in Drop"
    )]
    async fn native() -> NativeFixture {
        use openai_protocol::validated::ValidatedJson;
        use tokio_tungstenite::tungstenite::client::IntoClientRequest;

        use crate::{
            config::{TenantApiKeyEntry, TenantResolutionConfig},
            middleware::{auth_middleware, AuthConfig, TenantRequestMeta, TenantResolutionState},
            observability::inflight_tracker::InFlightRequestTracker,
        };
        let keys = [
            TenantApiKeyEntry {
                tenant_id: "team-a".into(),
                key: "fixture-key-a".into(),
            },
            TenantApiKeyEntry {
                tenant_id: "team-b".into(),
                key: "fixture-key-b".into(),
            },
        ];
        let auth = AuthConfig::with_tenant_keys(None, &keys);
        let auth_layer = axum::middleware::from_fn_with_state(auth.clone(), auth_middleware);
        let tenant = TenantResolutionState::from_config(&TenantResolutionConfig {
            trust_tenant_header: true,
            tenant_header_name: "x-tenant-id".into(),
        })
        .unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let post_body_ends = Arc::new(AtomicUsize::new(0));
        let resume = Arc::new(Notify::new());
        let bucket = Arc::new(crate::middleware::TokenBucket::new(1, 0));
        let admission = admission_state(bucket.clone());
        let counted = calls.clone();
        let dropped = drops.clone();
        let body_ends = post_body_ends.clone();
        let release = resume.clone();
        let tracker = InFlightRequestTracker::new();
        struct DropProbe(Arc<AtomicUsize>);
        impl Drop for DropProbe {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let protected = axum::Router::new().route("/v1/responses", axum::routing::post(move |Extension(meta): Extension<TenantRequestMeta>, turn: Option<Extension<SocketTurn>>, ValidatedJson(body): ValidatedJson<ResponsesRequest>| {
            let counted = counted.clone();
            let dropped = dropped.clone();
            let body_ends = body_ends.clone();
            let release = release.clone();
            async move {
                let number = counted.fetch_add(1, Ordering::SeqCst);
                if turn.is_some_and(|Extension(turn)| turn.prewarm) { return prewarm_response(body); }
                if serde_json::to_string(&body.input).unwrap().contains("stall-fixture") {
                    let probe = DropProbe(dropped);
                    let event = json!({"type":"response.output_text.delta","delta":"x".repeat(1024 * 1024)});
                    let frame = axum::body::Bytes::from(format!("data: {event}\n\n"));
                    let stream = futures::stream::unfold((probe, frame), |(probe, frame)| async move {
                        Some((Ok::<_, io::Error>(frame.clone()), (probe, frame)))
                    });
                    return Response::builder().header("content-type", "text/event-stream").body(Body::from_stream(stream)).unwrap();
                }
                if serde_json::to_string(&body.input).unwrap().contains("wait-fixture") {
                    let probe = DropProbe(dropped);
                    let pending = futures::stream::once(async move {
                        let _probe = probe;
                        release.notified().await;
                        let event = json!({
                            "type": "response.completed",
                            "response": {"id": "resp_wait", "status": "completed", "output": []},
                        });
                        Ok::<_, io::Error>(axum::body::Bytes::from(format!("data: {event}\n\n")))
                    });
                    let first = futures::stream::once(async { Ok::<_, io::Error>(axum::body::Bytes::from_static(b"data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_wait\"}}\n\n")) });
                    let end = futures::stream::poll_fn(move |_| {
                        body_ends.fetch_add(1, Ordering::SeqCst);
                        std::task::Poll::Ready(None)
                    });
                    return Response::builder().header("content-type", "text/event-stream").body(Body::from_stream(first.chain(pending).chain(end))).unwrap();
                }
                let tenant = meta.tenant_key().as_str().to_owned();
                let event = json!({"type":"response.completed","response":{"id":format!("resp_native_{number}"),"status":"completed","output":[{"type":"message","id":"msg_fixture","role":"assistant","content":[{"type":"output_text","text":tenant}]}]}});
                ([("content-type","text/event-stream")], format!("data: {event}\n\n")).into_response()
            }
        }))
        .layer(crate::middleware::ConcurrencyLimitLayer::new(admission.clone()))
        .layer(axum::middleware::from_fn(tenant_binding_guard))
        .layer(tenant.clone().layer())
        .layer(auth_layer.clone())
        .layer(crate::middleware::HttpMetricsLayer::new(tracker.clone()))
        .layer(axum::extract::DefaultBodyLimit::max(4096));
        let dispatch = Dispatch {
            service: protected.clone(),
            tracker: tracker.clone(),
            max_payload_bytes: 4096,
            origins: Arc::default(),
            connections: Arc::default(),
        };
        let websocket = axum::Router::new()
            .route("/v1/responses", axum::routing::get(upgrade))
            .layer(Extension(dispatch))
            .layer(crate::middleware::ConcurrencyLimitLayer::new(admission))
            .layer(crate::middleware::HttpMetricsLayer::new(
                InFlightRequestTracker::new(),
            ))
            .layer(tenant.layer())
            .layer(auth_layer)
            .layer(axum::middleware::from_fn(capture_credential));
        let app = protected.merge(websocket);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("ws://{}/v1/responses", listener.local_addr().unwrap());
        let served = app.clone();
        let server = tokio::spawn(async move {
            axum::serve(
                listener,
                served.into_make_service_with_connect_info::<SocketAddr>(),
            )
            .await
            .unwrap();
        });
        let mut request = url.clone().into_client_request().unwrap();
        request
            .headers_mut()
            .insert("authorization", "Bearer fixture-key-a".parse().unwrap());
        request
            .headers_mut()
            .insert("x-tenant-id", "team-b".parse().unwrap());
        let (socket, _) = connect_async(request).await.unwrap();
        NativeFixture {
            fixture: Self {
                socket,
                app,
                calls,
                server,
            },
            auth,
            tracker,
            drops,
            post_body_ends,
            resume,
            url,
            bucket,
        }
    }
}

#[tokio::test]
async fn native_auth_tenant_scope_rotation_prewarm_and_drain_are_preserved() {
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;

    use crate::{config::TenantApiKeyEntry, middleware::AuthConfig};
    let NativeFixture {
        mut fixture,
        auth,
        tracker,
        url,
        bucket,
        ..
    } = Fixture::native().await;
    for credential in [None, Some("Bearer wrong")] {
        let mut request = Request::builder().uri("/v1/responses");
        if let Some(credential) = credential {
            request = request.header("authorization", credential);
        }
        assert_eq!(
            fixture
                .app
                .clone()
                .oneshot(request.body(Body::empty()).unwrap())
                .await
                .unwrap()
                .status(),
            StatusCode::UNAUTHORIZED
        );
    }
    let first = fixture
        .turn(
            json!({"type":"response.create","model":"fixture-model","store":false,"input":"seed"}),
        )
        .await;
    assert_eq!(
        first["response"]["output"][0]["content"][0]["text"],
        "auth:team-a"
    );
    let parent = first["response"]["id"].clone();
    bucket.try_acquire(1.0).unwrap();
    let shed = fixture.turn(json!({"type":"response.create","model":"fixture-model","generate":false,"input":"warm"})).await;
    assert_eq!(
        shed["status"], 429,
        "prewarm must acquire its own native admission slot"
    );
    assert_eq!(fixture.calls.load(Ordering::SeqCst), 1);
    bucket.return_tokens(1.0);
    let warm = fixture.turn(json!({"type":"response.create","model":"fixture-model","store":false,"generate":false,"previous_response_id":parent,"input":"warm"})).await;
    assert_eq!(warm["type"], "response.completed");
    assert_eq!(
        fixture.calls.load(Ordering::SeqCst),
        2,
        "prewarming enters protected dispatch, native auth, tenant resolution and validation"
    );

    let mut request = url.clone().into_client_request().unwrap();
    request
        .headers_mut()
        .insert("authorization", "Bearer fixture-key-b".parse().unwrap());
    let (mut other, _) = connect_async(request).await.unwrap();
    other.send(ClientMessage::Text(json!({"type":"response.create","model":"fixture-model","previous_response_id":parent,"input":"cross-tenant"}).to_string().into())).await.unwrap();
    let event = other.next().await.unwrap().unwrap().into_text().unwrap();
    let event: Value = serde_json::from_str(&event).unwrap();
    assert_eq!(event["error"]["code"], "previous_response_not_found");
    assert_eq!(fixture.calls.load(Ordering::SeqCst), 2);

    auth.replace(AuthConfig::key_map(
        None,
        &[TenantApiKeyEntry {
            tenant_id: "team-b".into(),
            key: "fixture-key-b".into(),
        }],
    ));
    let revoked = fixture.turn(json!({"type":"response.create","model":"fixture-model","generate":false,"input":"warm"})).await;
    assert_eq!(
        revoked["status"], 401,
        "open sockets reauthenticate even on prewarming"
    );
    assert_eq!(fixture.calls.load(Ordering::SeqCst), 2);
    assert!(tracker.wait_for_drain(Duration::from_secs(1)).await);
    fixture.socket = other;
    tracker.begin_drain();
    let closed = tokio::time::timeout(Duration::from_secs(1), fixture.socket.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(matches!(closed, ClientMessage::Close(Some(frame)) if u16::from(frame.code) == 1012));
    assert!(tracker.wait_for_shutdown(Duration::from_secs(1)).await);
}

#[tokio::test]
async fn removing_the_last_key_refuses_new_upgrades_and_existing_turns() {
    use tokio_tungstenite::tungstenite::Error as WsError;
    let NativeFixture {
        mut fixture,
        auth,
        tracker,
        url,
        ..
    } = Fixture::native().await;
    auth.replace(Default::default());
    let denied = connect_async(url).await.unwrap_err();
    assert!(matches!(denied, WsError::Http(response) if response.status() == 401));
    let refusal = fixture
        .turn(json!({
            "type": "response.create", "model": "fixture-model", "generate": false, "input": "warm",
        }))
        .await;
    assert_eq!(refusal["error"]["code"], "websocket_tenant_changed");
    assert_eq!(fixture.calls.load(Ordering::SeqCst), 0);
    assert!(tracker.wait_for_shutdown(Duration::from_secs(1)).await);
}

#[tokio::test]
async fn a_reassigned_key_cannot_replay_another_tenants_retained_history() {
    use crate::{config::TenantApiKeyEntry, middleware::AuthConfig};
    let NativeFixture {
        mut fixture,
        auth,
        bucket,
        tracker,
        ..
    } = Fixture::native().await;
    let first = fixture
        .turn(
            json!({"type":"response.create","model":"fixture-model","input":"only-team-a-context"}),
        )
        .await;
    assert_eq!(first["type"], "response.completed");
    auth.replace(AuthConfig::key_map(
        None,
        &[TenantApiKeyEntry {
            tenant_id: "team-b".into(),
            key: "fixture-key-a".into(),
        }],
    ));
    let denied = fixture.turn(json!({"type":"response.create","model":"fixture-model","previous_response_id":first["response"]["id"],"input":"next"})).await;
    assert_eq!(denied["status"], 403);
    assert_eq!(denied["error"]["code"], "websocket_tenant_changed");
    assert_eq!(
        fixture.calls.load(Ordering::SeqCst),
        1,
        "old context never reaches the reassigned tenant's backend"
    );
    assert_eq!(
        bucket.available_tokens(),
        1.0,
        "identity changes are refused before admission"
    );
    assert!(tracker.wait_for_drain(Duration::from_secs(1)).await);

    // A fresh HTTP call still uses the new tenant; only the retained
    // history is bound to the identity under which it was accepted.
    let request = Request::builder()
        .method("POST")
        .uri("/v1/responses")
        .header("authorization", "Bearer fixture-key-a")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"model":"fixture-model","input":"fresh","stream":true,"store":false})
                .to_string(),
        ))
        .unwrap();
    assert_eq!(
        fixture.app.clone().oneshot(request).await.unwrap().status(),
        StatusCode::OK
    );
}

#[test]
fn durable_storage_and_conversation_modes_are_explicitly_unsupported() {
    for store in [Value::Null, json!("false"), json!(0), json!([]), json!({})] {
        let mut cache = SessionCache::default();
        let error = prepare(
            json!({"type":"response.create","model":"fixture-model","store":store,"input":"seed"}),
            &mut cache,
        )
        .err()
        .unwrap();
        assert_eq!(error["error"]["code"], "invalid_type");
        assert_eq!(error["error"]["param"], "store");
    }
    for generate in [true, false] {
        let mut cache = SessionCache::default();
        let error = prepare(json!({"type":"response.create","model":"fixture-model","generate":generate,"store":true,"input":"seed"}), &mut cache).err().unwrap();
        assert_eq!(error["error"]["code"], "unsupported_parameter");
        assert_eq!(error["error"]["param"], "store");
        let error = prepare(json!({"type":"response.create","model":"fixture-model","conversation":"conv_fixture","input":"seed"}), &mut cache).err().unwrap();
        assert_eq!(error["error"]["param"], "conversation");
    }
}

#[tokio::test]
async fn the_lifetime_limit_cancels_an_active_response_and_clears_its_guard() {
    let NativeFixture {
        mut fixture,
        tracker,
        drops,
        bucket,
        ..
    } = Fixture::native().await;
    fixture
        .socket
        .send(ClientMessage::Text(
            json!({"type":"response.create","model":"fixture-model","input":"wait-fixture"})
                .to_string()
                .into(),
        ))
        .await
        .unwrap();
    let first = fixture
        .socket
        .next()
        .await
        .unwrap()
        .unwrap()
        .into_text()
        .unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&first).unwrap()["type"],
        "response.created"
    );
    tokio::time::pause();
    tokio::time::advance(CONNECTION_LIFETIME + Duration::from_secs(1)).await;
    let error = loop {
        let message = fixture.socket.next().await.unwrap().unwrap();
        if let ClientMessage::Text(error) = message {
            break error;
        }
        assert!(message.is_ping());
    };
    assert_eq!(
        serde_json::from_str::<Value>(&error).unwrap()["error"]["code"],
        "websocket_connection_limit_reached"
    );
    assert!(tracker.wait_for_drain(Duration::from_secs(1)).await);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert_eq!(bucket.available_tokens(), 1.0);
}

#[tokio::test]
async fn disconnect_drops_the_native_body_and_releases_the_drain_guard() {
    let NativeFixture {
        mut fixture,
        tracker,
        drops,
        bucket,
        ..
    } = Fixture::native().await;
    fixture
        .socket
        .send(ClientMessage::Text(
            json!({"type":"response.create","model":"fixture-model","input":"wait-fixture"})
                .to_string()
                .into(),
        ))
        .await
        .unwrap();
    let first = fixture
        .socket
        .next()
        .await
        .unwrap()
        .unwrap()
        .into_text()
        .unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&first).unwrap()["type"],
        "response.created"
    );
    assert_eq!(tracker.len(), 1);
    assert_eq!(bucket.available_tokens(), 0.0);
    fixture.socket.close(None).await.unwrap();
    assert!(tracker.wait_for_drain(Duration::from_secs(2)).await);
    assert_eq!(
        drops.load(Ordering::SeqCst),
        1,
        "the upstream stream is cancelled on disconnect"
    );
    assert_eq!(bucket.available_tokens(), 1.0);
}

#[tokio::test]
async fn a_client_that_stops_reading_releases_the_body_and_guard() {
    let NativeFixture {
        mut fixture,
        tracker,
        drops,
        bucket,
        ..
    } = Fixture::native().await;
    fixture
        .socket
        .send(ClientMessage::Text(
            json!({"type":"response.create","model":"fixture-model","input":"stall-fixture"})
                .to_string()
                .into(),
        ))
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(2), async {
        while tracker.is_empty() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    // The client deliberately never polls its reader. A finite socket
    // window fills and the per-write deadline must cancel the body.
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(tracker.len(), 1);
    assert_eq!(bucket.available_tokens(), 0.0);
    assert!(
        tracker
            .wait_for_drain(WRITE_TIMEOUT + Duration::from_secs(3))
            .await
    );
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert_eq!(bucket.available_tokens(), 1.0);
}

#[tokio::test]
async fn prewarm_obeys_protected_request_admission() {
    let mut fixture = Fixture::new(true).await;
    let terminal = fixture.turn(json!({"type":"response.create","model":"fixture-model","generate":false,"input":"warm"})).await;
    assert_eq!(
        terminal["type"], "error",
        "prewarming must not bypass a protected service refusal"
    );
    assert_eq!(fixture.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn failed_replacement_preserves_the_previous_response() {
    let mut fixture = Fixture::new(false).await;
    let first = fixture
        .turn(json!({"type":"response.create","model":"fixture-model","input":"seed"}))
        .await;
    let parent = first["response"]["id"].clone();
    let failed = fixture.turn(json!({"type":"response.create","model":"fixture-model","previous_response_id":null,"input":"fail"})).await;
    assert_eq!(failed["type"], "response.failed");
    let retry = fixture.turn(json!({"type":"response.create","model":"fixture-model","previous_response_id":parent,"input":"retry"})).await;
    assert_eq!(
        retry["type"], "response.completed",
        "a failed replacement must not delete a valid parent"
    );
}

#[tokio::test]
async fn websocket_obeys_the_configured_http_payload_cap() {
    let mut fixture = Fixture::new(false).await;
    let body = json!({"model":"fixture-model","input":"x".repeat(1024)});
    let http = Request::builder()
        .method("POST")
        .uri("/v1/responses")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    assert_eq!(
        fixture.app.clone().oneshot(http).await.unwrap().status(),
        StatusCode::PAYLOAD_TOO_LARGE
    );
    let first = fixture
        .turn(json!({"type":"response.create","model":"fixture-model","input":"y".repeat(250)}))
        .await;
    assert_eq!(first["type"], "response.completed");
    let terminal = fixture.turn(json!({"type":"response.create","model":"fixture-model","previous_response_id":first["response"]["id"],"input":"z".repeat(250)})).await;
    assert_eq!(
        terminal["type"], "error",
        "expanded continuations must preserve the operator's payload limit"
    );
    assert_eq!(fixture.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn drain_allows_an_active_response_to_finish_and_then_closes_the_socket() {
    let NativeFixture {
        mut fixture,
        tracker,
        resume,
        bucket,
        post_body_ends,
        ..
    } = Fixture::native().await;
    fixture
        .socket
        .send(ClientMessage::Text(
            json!({"type": "response.create", "model": "fixture-model", "input": "wait-fixture"})
                .to_string()
                .into(),
        ))
        .await
        .unwrap();
    let first = tokio::time::timeout(Duration::from_secs(2), fixture.socket.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap()
        .into_text()
        .unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&first).unwrap()["type"],
        "response.created"
    );
    assert_eq!(tracker.len(), 1);
    assert_eq!(bucket.available_tokens(), 0.0);
    tokio::time::pause();
    tokio::time::advance(HEARTBEAT_INTERVAL).await;
    let ping = tokio::time::timeout(Duration::from_secs(2), fixture.socket.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(ping.is_ping());
    tokio::time::resume();
    tracker.begin_drain();
    assert!(!tracker.wait_for_shutdown(Duration::ZERO).await);
    resume.notify_one();
    let terminal = tokio::time::timeout(Duration::from_secs(2), fixture.socket.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap()
        .into_text()
        .unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&terminal).unwrap()["type"],
        "response.completed"
    );
    let close = tokio::time::timeout(Duration::from_secs(2), fixture.socket.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(matches!(close, ClientMessage::Close(Some(frame)) if u16::from(frame.code) == 1012));
    assert!(tracker.wait_for_shutdown(Duration::from_secs(1)).await);
    assert_eq!(bucket.available_tokens(), 1.0);
    assert_eq!(post_body_ends.load(Ordering::SeqCst), 1);
}
