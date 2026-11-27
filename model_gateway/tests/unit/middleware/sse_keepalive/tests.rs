use axum::{
    http::{header::CONTENT_TYPE, Method, StatusCode},
    routing::post,
    Router,
};
use http_body_util::BodyExt;
use openai_protocol::messages::MessageStreamEvent;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tower::ServiceExt;

use super::*;
use crate::{
    middleware::{edge_body::at_the_edge, HttpMetricsLayer},
    observability::inflight_tracker::InFlightRequestTracker,
};

const SECS: u64 = 15;
type Chunk = Result<Bytes, std::io::Error>;

/// What a read of the body gave within its time budget.
#[derive(Debug, PartialEq)]
enum Read {
    Frame(Bytes),
    End,
    Silence,
}

/// The next data frame within `secs` of (virtual) time.
async fn next_within(body: &mut Body, secs: u64) -> Read {
    match tokio::time::timeout(Duration::from_secs(secs), body.frame()).await {
        Ok(Some(Ok(frame))) => frame.into_data().map_or(Read::End, Read::Frame),
        Ok(_) => Read::End,
        Err(_) => Read::Silence,
    }
}

fn sse(rx: mpsc::Receiver<Chunk>) -> Response {
    Response::builder()
        .header(CONTENT_TYPE, "text/event-stream; charset=utf-8")
        .body(Body::from_stream(ReceiverStream::new(rx)))
        .unwrap()
}

/// An SSE body under the keep-alive, at the edge, fed through the
/// returned sender.
fn wrapped(secs: u64, messages: bool) -> (mpsc::Sender<Chunk>, Body) {
    let (tx, rx) = mpsc::channel(16);
    let response = SseKeepAlive::every_secs(secs).apply(sse(rx), messages);
    (tx, at_the_edge(response).into_body())
}

async fn send(tx: &mpsc::Sender<Chunk>, bytes: &'static [u8]) {
    tx.send(Ok(Bytes::from_static(bytes))).await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn a_quiet_stream_gets_a_comment_every_interval_from_the_headers() {
    let (_tx, mut body) = wrapped(SECS, false);
    assert_eq!(next_within(&mut body, SECS - 1).await, Read::Silence);
    assert_eq!(
        next_within(&mut body, 2).await,
        Read::Frame(Bytes::from_static(COMMENT))
    );
    // The next one an interval after the keep-alive, not sooner.
    assert_eq!(next_within(&mut body, SECS - 1).await, Read::Silence);
    assert_eq!(
        next_within(&mut body, 2).await,
        Read::Frame(Bytes::from_static(COMMENT))
    );
}

#[tokio::test(start_paused = true)]
async fn a_flowing_stream_carries_no_keep_alive() {
    let (tx, mut body) = wrapped(SECS, false);
    for _ in 0..6 {
        tokio::time::sleep(Duration::from_secs(10)).await;
        send(&tx, b"data: {\"delta\":\"x\"}\n\n").await;
        assert_eq!(
            next_within(&mut body, 0).await,
            Read::Frame(Bytes::from_static(b"data: {\"delta\":\"x\"}\n\n"))
        );
    }
    // The interval counts from the last frame.
    assert_eq!(next_within(&mut body, SECS - 1).await, Read::Silence);
    assert_eq!(
        next_within(&mut body, 2).await,
        Read::Frame(Bytes::from_static(COMMENT))
    );
}

#[tokio::test(start_paused = true)]
async fn a_keep_alive_never_splits_a_frame() {
    let (tx, mut body) = wrapped(SECS, false);
    send(&tx, b"data: {\"delta\":").await;
    assert_eq!(
        next_within(&mut body, 0).await,
        Read::Frame(Bytes::from_static(b"data: {\"delta\":"))
    );
    // Two intervals inside the frame: nothing is written.
    assert_eq!(next_within(&mut body, 3 * SECS - 1).await, Read::Silence);
    send(&tx, b"\"x\"}\n\n").await;
    assert_eq!(
        next_within(&mut body, 0).await,
        Read::Frame(Bytes::from_static(b"\"x\"}\n\n"))
    );
    // Back on a boundary: the keep-alive an interval after the frame.
    assert_eq!(next_within(&mut body, SECS - 1).await, Read::Silence);
    assert_eq!(
        next_within(&mut body, 2).await,
        Read::Frame(Bytes::from_static(COMMENT))
    );
}

#[tokio::test(start_paused = true)]
async fn an_interval_of_zero_sends_nothing() {
    let (_tx, mut body) = wrapped(0, false);
    assert_eq!(next_within(&mut body, 3600).await, Read::Silence);
}

#[tokio::test(start_paused = true)]
async fn a_messages_stream_gets_the_api_s_ping_event() {
    let (_tx, mut body) = wrapped(SECS, true);
    assert_eq!(next_within(&mut body, SECS - 1).await, Read::Silence);
    assert_eq!(
        next_within(&mut body, 2).await,
        Read::Frame(Bytes::from_static(MESSAGES_PING))
    );
    // The frame is the protocol's own ping event.
    let ping = serde_json::to_string(&MessageStreamEvent::Ping).unwrap();
    assert_eq!(
        format!("event: ping\ndata: {ping}\n\n").as_bytes(),
        MESSAGES_PING
    );
}

#[tokio::test(start_paused = true)]
async fn the_end_of_the_stream_passes_at_once() {
    let (tx, mut body) = wrapped(SECS, false);
    send(&tx, b"data: [DONE]\n\n").await;
    drop(tx);
    assert_eq!(
        next_within(&mut body, 0).await,
        Read::Frame(Bytes::from_static(b"data: [DONE]\n\n"))
    );
    assert_eq!(next_within(&mut body, 0).await, Read::End);
}

fn app(response: impl Fn() -> Response + Clone + Send + Sync + 'static) -> Router {
    let chat = response.clone();
    Router::new()
        .route("/v1/chat/completions", post(move || async move { chat() }))
        .route(MESSAGES_ROUTE, post(move || async move { response() }))
        .route_layer(SseKeepAlive::every_secs(SECS).layer())
        .layer(HttpMetricsLayer::new(InFlightRequestTracker::new()))
}

async fn call(app: Router, path: &str) -> Response {
    app.oneshot(
        Request::builder()
            .method(Method::POST)
            .uri(path)
            .body(Body::empty())
            .unwrap(),
    )
    .await
    .unwrap()
}

/// An SSE response whose upstream never speaks.
fn silent_sse() -> Response {
    Response::builder()
        .header(CONTENT_TYPE, "text/event-stream")
        .body(Body::from_stream(futures::stream::pending::<Chunk>()))
        .unwrap()
}

#[tokio::test(start_paused = true)]
async fn the_middleware_writes_the_comment_on_an_openai_route_and_the_ping_on_messages() {
    let mut chat = call(app(silent_sse), "/v1/chat/completions")
        .await
        .into_body();
    assert_eq!(
        next_within(&mut chat, SECS + 1).await,
        Read::Frame(Bytes::from_static(COMMENT))
    );
    let mut messages = call(app(silent_sse), MESSAGES_ROUTE).await.into_body();
    assert_eq!(
        next_within(&mut messages, SECS + 1).await,
        Read::Frame(Bytes::from_static(MESSAGES_PING))
    );
}

#[tokio::test(start_paused = true)]
async fn a_response_that_is_not_an_event_stream_passes_untouched() {
    let json = || {
        Response::builder()
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from_stream(futures::stream::pending::<Chunk>()))
            .unwrap()
    };
    let response = call(app(json), "/v1/chat/completions").await;
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body();
    assert_eq!(next_within(&mut body, 10 * SECS).await, Read::Silence);
}

/// A streamed body under an upstream's `Content-Length` (a non-SSE
/// upstream body relabelled as SSE keeps the header; the relay drops it
/// from a streamed SSE body) is left alone: a keep-alive would push it
/// past the length.
#[tokio::test(start_paused = true)]
async fn an_sse_answer_under_a_content_length_header_passes_untouched() {
    let declared = || {
        Response::builder()
            .header(CONTENT_TYPE, "text/event-stream")
            .header(CONTENT_LENGTH, "14")
            .body(Body::from_stream(futures::stream::pending::<Chunk>()))
            .unwrap()
    };
    let response = call(app(declared), "/v1/chat/completions").await;
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body();
    assert_eq!(next_within(&mut body, 10 * SECS).await, Read::Silence);
}

#[tokio::test(start_paused = true)]
async fn a_complete_sse_answer_passes_untouched() {
    let done = || {
        Response::builder()
            .header(CONTENT_TYPE, "text/event-stream")
            .body(Body::from("data: [DONE]\n\n"))
            .unwrap()
    };
    let response = call(app(done), "/v1/chat/completions").await;
    assert_eq!(
        response.body().size_hint().exact(),
        Some("data: [DONE]\n\n".len() as u64),
        "a sized body keeps its length"
    );
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(&bytes[..], b"data: [DONE]\n\n");
}
