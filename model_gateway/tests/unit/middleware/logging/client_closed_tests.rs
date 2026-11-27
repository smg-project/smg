use std::{
    convert::Infallible,
    sync::{Arc, Mutex},
    task::Poll,
};

use axum::{
    body::{Body, Bytes},
    http::Request,
    response::Response,
    routing::get,
    Router,
};
use http::header::CONTENT_LENGTH;
use http_body_util::BodyExt;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tower::ServiceExt;
use tracing::Level;

use super::*;
use crate::{
    middleware::HttpMetricsLayer,
    observability::{
        inflight_tracker::InFlightRequestTracker,
        logging::test_support::{test_subscriber, ResponseLines},
    },
};

fn line(level: Level, message: &str) -> (Level, String) {
    (level, message.to_owned())
}

fn request(uri: &str) -> Request<Body> {
    Request::builder()
        .uri(uri)
        .body(Body::empty())
        .expect("request")
}

/// The response `build`'s router streams at `uri`, its first chunk `first`
/// already sent; the sender keeps the stream open until the test drops it.
async fn streamed(
    build: fn(Body) -> Router,
    uri: &str,
    first: &'static [u8],
) -> (Response, mpsc::Sender<Result<Bytes, Infallible>>) {
    let (tx, rx) = mpsc::channel(4);
    let body = Body::from_stream(ReceiverStream::new(rx));
    let response = build(body).oneshot(request(uri)).await.expect("response");
    tx.send(Ok(Bytes::from_static(first))).await.expect("send");
    (response, tx)
}

/// `/stream` hands out `body` once; `/slow` never answers.
fn app(body: Body) -> Router {
    let slot = Arc::new(Mutex::new(Some(body)));
    Router::new()
        .route(
            "/stream",
            get(move || {
                let body = slot
                    .lock()
                    .expect("slot")
                    .take()
                    .expect("handler called once");
                async move { Response::new(body) }
            }),
        )
        .route(
            "/slow",
            get(|| async {
                std::future::pending::<()>().await;
                "never"
            }),
        )
        .layer(create_logging_layer(AccessLogFields::NONE))
        .layer(HttpMetricsLayer::new(InFlightRequestTracker::new()))
}

/// `/relay` answers with `body` under a `Content-Length` of nine bytes,
/// as the HTTP relay answers an upstream that declared one.
fn relay_app(body: Body) -> Router {
    let slot = Arc::new(Mutex::new(Some(body)));
    Router::new()
        .route(
            "/relay",
            get(move || {
                let body = slot
                    .lock()
                    .expect("slot")
                    .take()
                    .expect("handler called once");
                async move {
                    Response::builder()
                        .header(CONTENT_LENGTH, 9)
                        .body(body)
                        .expect("response")
                }
            }),
        )
        .layer(create_logging_layer(AccessLogFields::NONE))
        .layer(HttpMetricsLayer::new(InFlightRequestTracker::new()))
}

#[tokio::test]
async fn a_body_under_a_content_length_dropped_after_its_declared_bytes_logs_no_client_close() {
    // On HTTP/1.1 the server stops asking for frames once the declared
    // bytes are out and drops the body: sent in full, not a client that
    // left.
    let sink = ResponseLines::default();
    let _subscriber = test_subscriber(sink.clone());
    let (response, tx) = streamed(relay_app, "/relay", b"data: x\n\n").await;
    let mut body = response.into_body();
    let frame = body.frame().await.expect("one frame").expect("frame ok");
    assert_eq!(frame.data_ref().map(Bytes::len), Some(9));
    drop(body);
    drop(tx);
    assert_eq!(
        sink.lines(),
        vec![line(Level::INFO, "finished processing request")]
    );
}

#[tokio::test]
async fn a_body_under_a_content_length_dropped_before_its_declared_bytes_logs_a_client_close() {
    // The other side of the declared-length rule: the client left with
    // the body short of its declared bytes.
    let sink = ResponseLines::default();
    let _subscriber = test_subscriber(sink.clone());
    let (response, tx) = streamed(relay_app, "/relay", b"data:").await;
    let mut body = response.into_body();
    let frame = body.frame().await.expect("one frame").expect("frame ok");
    assert_eq!(frame.data_ref().map(Bytes::len), Some(5));
    drop(body);
    drop(tx);
    assert_eq!(
        sink.lines(),
        vec![
            line(Level::INFO, "finished processing request"),
            line(
                Level::INFO,
                "client closed the connection before the response stream ended"
            ),
        ]
    );
}

#[tokio::test]
async fn a_request_the_client_dropped_before_the_response_logs_one_info_line() {
    let sink = ResponseLines::default();
    let _subscriber = test_subscriber(sink.clone());
    let mut pending = Box::pin(app(Body::empty()).oneshot(request("/slow")));
    assert!(matches!(futures::poll!(pending.as_mut()), Poll::Pending));
    drop(pending);
    assert_eq!(
        sink.lines(),
        vec![line(
            Level::INFO,
            "client closed the connection before the response"
        )]
    );
}

#[tokio::test]
async fn a_stream_the_client_dropped_logs_one_info_line_after_the_access_log() {
    let sink = ResponseLines::default();
    let _subscriber = test_subscriber(sink.clone());
    let (response, tx) = streamed(app, "/stream", b"data: x\n\n").await;
    let mut body = response.into_body();
    let frame = body.frame().await.expect("one frame").expect("frame ok");
    assert!(frame.is_data());
    drop(body);
    drop(tx);
    assert_eq!(
        sink.lines(),
        vec![
            line(Level::INFO, "finished processing request"),
            line(
                Level::INFO,
                "client closed the connection before the response stream ended"
            ),
        ]
    );
}

#[tokio::test]
async fn a_stream_that_ends_logs_no_client_close() {
    let sink = ResponseLines::default();
    let _subscriber = test_subscriber(sink.clone());
    let (response, tx) = streamed(app, "/stream", b"data: x\n\n").await;
    drop(tx);
    let _ = response.into_body().collect().await.expect("body");
    assert_eq!(
        sink.lines(),
        vec![line(Level::INFO, "finished processing request")]
    );
}
