use super::*;
fn user(text: &str) -> Value {
    json!({"role":"user","content":text})
}
#[test]
fn continuation_preserves_order_and_only_appends_new_items() {
    let mut cache = SessionCache::default();
    cache.remember("resp_1".into(), vec![user("first")], vec![json!({"type":"message","id":"m1","role":"assistant","content":[{"type":"output_text","text":"answer"}]})]);
    let request = prepare(json!({"type":"response.create","model":"fixture-model","previous_response_id":"resp_1","input":[user("next")],"store":false}), &mut cache).unwrap();
    assert_eq!(request.input.len(), 3);
    assert_eq!(request.input[0], user("first"));
    assert_eq!(request.input[2], user("next"));
    assert_eq!(request.body["store"], false);
}
#[test]
fn ids_are_connection_scoped_and_missing_parent_never_generates() {
    let mut a = SessionCache::default();
    a.remember("resp_1".into(), vec![user("private")], vec![]);
    let mut b = SessionCache::default();
    let error = prepare(json!({"type":"response.create","model":"fixture-model","previous_response_id":"resp_1","input":"next"}), &mut b).err().unwrap();
    assert_eq!(error["error"]["code"], "previous_response_not_found");
}
#[test]
fn lru_entries_are_bounded_and_clear_releases_owned_state() {
    let mut cache = SessionCache::default();
    for i in 0..8 {
        cache.remember(format!("resp_{i}"), vec![user("small")], vec![]);
    }
    assert_eq!(cache.entries.len(), MAX_CACHE_ENTRIES);
    assert!(cache.expand("resp_0", vec![]).is_err());
    cache.clear();
    assert!(cache.entries.is_empty());
    assert_eq!(cache.bytes, 0);
}
#[test]
fn oversized_history_is_not_retained() {
    let mut cache = SessionCache::default();
    cache.remember(
        "large".into(),
        vec![user(&"x".repeat(MAX_CACHE_BYTES / 3 + 1))],
        vec![],
    );
    assert_eq!(cache.bytes, 0);
    assert!(cache.entries.is_empty());
}
#[test]
fn prewarm_and_control_validation() {
    let mut cache = SessionCache::default();
    assert!(
        !prepare(
            json!({"type":"response.create","model":"fixture-model","generate":false,"input":[]}),
            &mut cache
        )
        .unwrap()
        .generate
    );
    assert!(prepare(json!({"type":"other"}), &mut cache).is_err());
    assert!(prepare(
        json!({"type":"response.create","generate":"false"}),
        &mut cache
    )
    .is_err());
    assert!(prepare(
        json!({"type":"response.create","stream_id":"x"}),
        &mut cache
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
async fn idle_socket_expires_after_prewarm_and_releases_session() {
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
                            track_response: Arc::new(|| Ok(Box::new(()))),
                            max_payload_bytes: MAX_MESSAGE_BYTES,
                        },
                        HeaderMap::new(),
                        None,
                        None,
                        Duration::from_millis(500),
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
    let failure = prepare(json!({"type":"response.create","background":true,"previous_response_id":"missing","input":[]}), &mut cache).err().unwrap();
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
        type Item = Result<Bytes, std::io::Error>;
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
