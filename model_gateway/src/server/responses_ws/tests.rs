use super::*;
fn user(text: &str) -> Value {
    json!({"role":"user","content":text})
}
#[test]
fn continuation_preserves_order_and_only_appends_new_items() {
    let mut cache = SessionCache::default();
    cache.remember("resp_1", &[user("first")], &[json!({"type":"message","id":"m1","role":"assistant","content":[{"type":"output_text","text":"answer"}]})]);
    let request = prepare(json!({"type":"response.create","model":"fixture-model","previous_response_id":"resp_1","input":[user("next")],"store":false}), &mut cache, MAX_PAYLOAD_BYTES).unwrap();
    assert_eq!(request.input.len(), 3);
    assert_eq!(request.input[0], user("first"));
    assert_eq!(request.input[2], user("next"));
    assert_eq!(request.body["store"], false);
}
#[test]
fn ids_are_connection_scoped_and_missing_parent_never_generates() {
    let mut a = SessionCache::default();
    a.remember("resp_1", &[user("private")], &[]);
    let mut b = SessionCache::default();
    let error = prepare(json!({"type":"response.create","model":"fixture-model","previous_response_id":"resp_1","input":"next"}), &mut b, MAX_PAYLOAD_BYTES).err().unwrap();
    assert_eq!(error["error"]["code"], "previous_response_not_found");
}
#[test]
fn lru_entries_are_bounded_and_clear_releases_owned_state() {
    let mut cache = SessionCache::default();
    for i in 0..8 {
        cache.remember(&format!("resp_{i}"), &[user("small")], &[]);
    }
    assert_eq!(cache.entries.len(), MAX_CACHE_ENTRIES);
    assert!(cache.expand("resp_0", vec![], MAX_PAYLOAD_BYTES).is_err());
    cache.clear();
    assert!(cache.entries.is_empty());
    assert_eq!(cache.bytes, 0);
}
#[test]
fn oversized_history_is_not_retained() {
    let mut cache = SessionCache::default();
    cache.remember("large", &[user(&"x".repeat(MAX_CACHE_BYTES / 3 + 1))], &[]);
    assert_eq!(cache.bytes, 0);
    assert!(cache.entries.is_empty());
}
#[test]
fn history_charges_the_id_and_rejects_replay_over_the_payload_limit() {
    let mut cache = SessionCache::default();
    cache.remember(&"x".repeat(MAX_CACHE_BYTES / 3), &[], &[]);
    assert!(cache.entries.is_empty());

    cache.remember("parent", &[user("longer than the replay cap")], &[]);
    let entry = &cache.entries[0];
    assert_eq!(cache.bytes, (entry.id.len() + entry.items.len()) * 3);
    let failure = cache.expand("parent", Vec::new(), 10).unwrap_err();
    assert_eq!(failure["status"], 413);
    assert_eq!(failure["error"]["code"], "request_body_too_large");
    assert!(cache
        .expand("parent", Vec::new(), MAX_PAYLOAD_BYTES)
        .is_ok());
}
#[test]
fn a_history_charge_releases_the_gauge_it_reserved() {
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let charge = metrics::with_local_recorder(&recorder, || Charge::reserve(123).unwrap());
    assert!(handle
        .render()
        .contains("smg_responses_ws_cache_bytes 123\n"));
    drop(charge);
    assert!(handle.render().contains("smg_responses_ws_cache_bytes 0\n"));
}
#[test]
fn prewarm_and_control_validation() {
    let mut cache = SessionCache::default();
    assert!(
        !prepare(
            json!({"type":"response.create","model":"fixture-model","generate":false,"input":[]}),
            &mut cache,
            MAX_PAYLOAD_BYTES,
        )
        .unwrap()
        .generate
    );
    assert!(prepare(json!({"type":"other"}), &mut cache, MAX_PAYLOAD_BYTES).is_err());
    assert!(prepare(
        json!({"type":"response.create","generate":"false"}),
        &mut cache,
        MAX_PAYLOAD_BYTES,
    )
    .is_err());
    assert!(prepare(
        json!({"type":"response.create","stream_id":"x"}),
        &mut cache,
        MAX_PAYLOAD_BYTES,
    )
    .is_err());
}
#[test]
fn sse_handles_split_utf8_crlf_comments_and_multiline_data() {
    let raw = ": ping\r\n\r\nevent: delta\r\ndata: {\"type\":\"response.output_text.delta\",\r\ndata: \"delta\":\"café\"}\r\n\r\n";
    let mut decoder = Events::default();
    let mut events = Vec::new();
    for byte in raw.as_bytes() {
        events.extend(decoder.push(&[*byte]).unwrap());
    }
    assert_eq!(events.len(), 1);
    assert_eq!(events[0]["delta"], "café");
}
#[expect(
    clippy::disallowed_methods,
    reason = "owned loopback server is aborted after the fixture and by test runtime shutdown"
)]
#[tokio::test]
async fn heartbeat_keeps_the_transport_alive_without_extending_session_idle_time() {
    use futures::{SinkExt, StreamExt};
    use tokio::sync::oneshot;
    use tokio_tungstenite::{connect_async, tungstenite::Message as ClientMessage};
    let (finished, done) = oneshot::channel();
    let finished = Arc::new(std::sync::Mutex::new(Some(finished)));
    let app = axum::Router::new().route(
        "/",
        axum::routing::get(move |upgrade: WebSocketUpgrade| {
            let finished = finished.clone();
            async move {
                upgrade.on_upgrade(move |socket| async move {
                    serve(
                        socket,
                        Dispatch {
                            service: axum::Router::new().route(
                                "/v1/responses",
                                axum::routing::post(
                                    |axum::Json(request): axum::Json<ResponsesRequest>| async move {
                                        prewarm_response(request)
                                    },
                                ),
                            ),
                            tracker: InFlightRequestTracker::new(),
                            max_payload_bytes: MAX_MESSAGE_BYTES,
                            origins: Arc::default(),
                            connections: Arc::default(),
                        },
                        HeaderMap::new(),
                        None,
                        "fixture".into(),
                        HEARTBEAT_INTERVAL + Duration::from_secs(5),
                    )
                    .await;
                    if let Some(finished) = finished.lock().unwrap().take() {
                        let _ = finished.send(());
                    }
                })
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let (mut socket, _) = connect_async(format!("ws://{address}/")).await.unwrap();
    socket.send(ClientMessage::Text(json!({"type":"response.create","model":"fixture-model","generate":false,"input":"private root"}).to_string().into())).await.unwrap();
    for _ in 0..2 {
        assert!(socket.next().await.unwrap().unwrap().is_text());
    }
    tokio::time::pause();
    tokio::time::advance(HEARTBEAT_INTERVAL).await;
    let ping = socket.next().await.unwrap().unwrap();
    let ClientMessage::Ping(payload) = ping else {
        panic!("expected a server heartbeat, got {ping:?}");
    };
    socket.send(ClientMessage::Pong(payload)).await.unwrap();
    tokio::time::advance(Duration::from_secs(6)).await;
    let close = tokio::time::timeout(Duration::from_secs(3), socket.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(close.is_close());
    tokio::time::timeout(Duration::from_secs(3), done)
        .await
        .unwrap()
        .unwrap();
    server.abort();
}

#[test]
fn background_mode_is_rejected_before_loading_history() {
    let mut cache = SessionCache::default();
    let failure = prepare(json!({"type":"response.create","background":true,"previous_response_id":"missing","input":[]}), &mut cache, MAX_PAYLOAD_BYTES).err().unwrap();
    assert_eq!(failure["error"]["code"], "unsupported_parameter");
    assert_eq!(failure["error"]["param"], "background");
}

#[tokio::test]
async fn response_body_reads_preserve_json_and_refuse_invalid_or_oversized_bodies() {
    assert_eq!(
        read_response_json(Body::from("{\"ok\":true}"), 64)
            .await
            .unwrap(),
        json!({"ok":true})
    );
    assert_eq!(
        read_response_json(Body::from("{\"ok\":true}"), 2).await,
        Err(ResponseBodyError::Read)
    );
    assert_eq!(
        read_response_json(Body::from("{"), 64).await,
        Err(ResponseBodyError::InvalidJson)
    );
}

#[tokio::test(start_paused = true)]
async fn a_stalled_response_body_is_dropped_at_the_read_deadline() {
    use std::{
        pin::Pin,
        sync::atomic::AtomicBool,
        task::{Context, Poll},
    };

    use axum::body::Bytes;
    use futures::Stream;

    struct PendingBody(Arc<AtomicBool>);
    impl Stream for PendingBody {
        type Item = Result<Bytes, io::Error>;
        fn poll_next(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            Poll::Pending
        }
    }
    impl Drop for PendingBody {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }
    let dropped = Arc::new(AtomicBool::new(false));
    let result = tokio::time::timeout(
        RESPONSE_READ_TIMEOUT + Duration::from_secs(1),
        read_response_json(Body::from_stream(PendingBody(dropped.clone())), 64),
    )
    .await;
    assert_eq!(result.unwrap(), Err(ResponseBodyError::Timeout));
    assert!(dropped.load(Ordering::SeqCst));
}

#[test]
fn connection_permits_bound_tenants_and_the_listener_and_return_on_drop() {
    let connections = Arc::new(Connections::default());
    let mut permits = Vec::new();
    for _ in 0..MAX_CONNECTIONS_PER_TENANT {
        permits.push(connections.try_acquire("tenant-a".into()).unwrap());
    }
    assert!(connections.try_acquire("tenant-a".into()).is_none());
    for n in permits.len()..MAX_CONNECTIONS {
        permits.push(
            connections
                .try_acquire(TenantKey::from(format!("tenant-{n}")))
                .unwrap(),
        );
    }
    assert!(connections.try_acquire("tenant-b".into()).is_none());
    permits.clear();
    assert_eq!(connections.0.lock().total, 0);
    assert!(connections.0.lock().tenants.is_empty());
    assert!(connections.try_acquire("tenant-a".into()).is_some());
}
