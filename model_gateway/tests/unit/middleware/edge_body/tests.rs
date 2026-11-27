use std::{
    convert::Infallible,
    sync::{Arc, Mutex},
    time::Duration,
};

use axum::{
    http::{header::CONTENT_TYPE, Request, StatusCode},
    routing::get,
    Router,
};
use http_body_util::BodyExt;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tower::ServiceExt;
use tracing::Level;

use super::*;
use crate::{
    middleware::{create_logging_layer, AccessLogFields, HttpMetricsLayer, SseKeepAlive},
    observability::logging::test_support::{test_subscriber, ResponseLines},
};

/// A permit that records its release and refuses the first frame when
/// told to.
struct TestPermit {
    released: Arc<Mutex<bool>>,
    lets_first_frame_pass: bool,
}

impl Permit for TestPermit {
    fn first_data_frame(&mut self) -> bool {
        self.lets_first_frame_pass
    }
}

impl Drop for TestPermit {
    fn drop(&mut self) {
        *self.released.lock().expect("released") = true;
    }
}

/// The server's edge over `/stream`, which hands out `body` once as an
/// SSE response with a permit: the admission and keep-alive layers inside,
/// the request span and the metrics layer outside, in the server's order.
fn app(body: Body, permit: TestPermit, tracker: Arc<InFlightRequestTracker>) -> Router {
    let slot = Arc::new(Mutex::new(Some((body, permit))));
    Router::new()
        .route(
            "/v1/chat/completions",
            get(move || {
                let (body, permit) = slot
                    .lock()
                    .expect("slot")
                    .take()
                    .expect("handler called once");
                async move {
                    let mut response = Response::builder()
                        .header(CONTENT_TYPE, "text/event-stream")
                        .body(body)
                        .expect("response");
                    BodyDuties::hand(&mut response, |duties| {
                        duties.permit = Some(Box::new(permit));
                    });
                    response
                }
            }),
        )
        .route_layer(SseKeepAlive::every_secs(15).layer())
        .layer(create_logging_layer(AccessLogFields::NONE))
        .layer(HttpMetricsLayer::new(tracker))
}

fn request() -> Request<Body> {
    Request::builder()
        .uri("/v1/chat/completions")
        .body(Body::empty())
        .expect("request")
}

fn permit(released: &Arc<Mutex<bool>>, lets_first_frame_pass: bool) -> TestPermit {
    TestPermit {
        released: released.clone(),
        lets_first_frame_pass,
    }
}

fn line(level: Level, message: &str) -> (Level, String) {
    (level, message.to_owned())
}

#[tokio::test(start_paused = true)]
async fn every_duty_ends_with_the_body() {
    let sink = ResponseLines::default();
    let _subscriber = test_subscriber(sink.clone());
    let tracker = InFlightRequestTracker::new();
    let released = Arc::new(Mutex::new(false));
    let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(4);
    let response = app(
        Body::from_stream(ReceiverStream::new(rx)),
        permit(&released, true),
        tracker.clone(),
    )
    .oneshot(request())
    .await
    .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body();

    // The handler has answered, the body is open: in flight, the permit
    // held, the access log at its response line.
    assert_eq!(tracker.len(), 1);
    assert!(!*released.lock().expect("released"));
    tx.send(Ok(Bytes::from_static(b"data: x\n\n")))
        .await
        .expect("send");
    let frame = body.frame().await.expect("a frame").expect("frame ok");
    assert_eq!(
        frame.data_ref().map(|data| &data[..]),
        Some(&b"data: x\n\n"[..])
    );
    assert_eq!(tracker.len(), 1);
    assert!(!*released.lock().expect("released"));

    // Silence: the keep-alive comment comes through the one wrapper.
    let next = tokio::time::timeout(Duration::from_secs(16), body.frame())
        .await
        .expect("a keep-alive within the interval")
        .expect("a frame")
        .expect("frame ok");
    assert_eq!(
        next.data_ref().map(|data| &data[..]),
        Some(&b": ping\n\n"[..])
    );

    // The end of the stream: everything goes, the response is complete.
    drop(tx);
    assert!(body.frame().await.is_none());
    assert_eq!(tracker.len(), 0);
    assert!(*released.lock().expect("released"));
    assert_eq!(
        sink.lines(),
        vec![line(Level::INFO, "finished processing request")]
    );
}

#[tokio::test]
async fn a_body_the_client_dropped_mid_stream_releases_every_duty_and_logs_the_close_once() {
    let sink = ResponseLines::default();
    let _subscriber = test_subscriber(sink.clone());
    let tracker = InFlightRequestTracker::new();
    let released = Arc::new(Mutex::new(false));
    let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(4);
    let response = app(
        Body::from_stream(ReceiverStream::new(rx)),
        permit(&released, true),
        tracker.clone(),
    )
    .oneshot(request())
    .await
    .expect("response");
    tx.send(Ok(Bytes::from_static(b"data: x\n\n")))
        .await
        .expect("send");
    let mut body = response.into_body();
    body.frame().await.expect("a frame").expect("frame ok");
    drop(body);
    assert_eq!(tracker.len(), 0);
    assert!(*released.lock().expect("released"));
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
async fn a_permit_that_refuses_the_first_frame_ends_the_body_without_it() {
    let tracker = InFlightRequestTracker::new();
    let released = Arc::new(Mutex::new(false));
    let response = app(
        Body::from("data: never\n\n"),
        permit(&released, false),
        tracker.clone(),
    )
    .oneshot(request())
    .await
    .expect("response");
    let mut body = response.into_body();
    assert!(!body.is_end_stream());
    assert!(
        body.frame().await.is_none(),
        "the refused frame is not sent"
    );
    assert!(body.is_end_stream(), "the body stays ended");
    assert!(body.frame().await.is_none());
    assert_eq!(tracker.len(), 0);
    assert!(*released.lock().expect("released"));
}

#[tokio::test]
async fn an_already_complete_body_is_finished_at_the_edge() {
    let tracker = InFlightRequestTracker::new();
    let released = Arc::new(Mutex::new(false));
    let response = app(Body::empty(), permit(&released, true), tracker.clone())
        .oneshot(request())
        .await
        .expect("response");
    assert_eq!(tracker.len(), 0);
    assert!(*released.lock().expect("released"));
    drop(response);
}

/// A stack with no metrics layer outside takes no duties: the span ends
/// with the response head, without a client-close line for a stream
/// nobody cut.
#[tokio::test]
async fn duties_nobody_took_keep_the_access_log_quiet() {
    let sink = ResponseLines::default();
    let _subscriber = test_subscriber(sink.clone());
    let (_tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(1);
    let slot = Arc::new(Mutex::new(Some(Body::from_stream(ReceiverStream::new(rx)))));
    let app = Router::new()
        .route(
            "/stream",
            get(move || {
                let body = slot.lock().expect("slot").take().expect("once");
                async move { Response::new(body) }
            }),
        )
        .layer(create_logging_layer(AccessLogFields::NONE));
    let response = app
        .oneshot(
            Request::builder()
                .uri("/stream")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    let (parts, body) = response.into_parts();
    drop(parts);
    assert_eq!(
        sink.lines(),
        vec![line(Level::INFO, "finished processing request")]
    );
    drop(body);
}

/// The producer's word and the duties share the response's one cell,
/// whichever reached the response first: the duties come out at the
/// edge with the word's cell, and the word the body tells is the word
/// the edge reads.
#[test]
fn the_producers_word_and_the_duties_share_one_cell() {
    for duty_first in [true, false] {
        let mut response = Response::new(Body::empty());
        let head_at = Instant::now();
        let word = BodyFailure::default();
        if duty_first {
            BodyDuties::hand(&mut response, |duties| duties.head_at = Some(head_at));
            word.carry(&mut response);
        } else {
            word.carry(&mut response);
            BodyDuties::hand(&mut response, |duties| duties.head_at = Some(head_at));
        }
        let mut duties = BodyDuties::take(&mut response);
        assert_eq!(duties.head_at, Some(head_at), "duty first: {duty_first}");
        let edge = duties.failure.take().expect("the cell comes along");
        assert!(!edge.told(), "duty first: {duty_first}");
        word.tell();
        assert!(edge.told(), "duty first: {duty_first}");
    }
}

/// A response whose producer gave no word hands the edge its duties
/// alone: nothing to read at the body's end.
#[test]
fn a_response_without_a_producers_word_hands_the_edge_no_cell() {
    let mut response = Response::new(Body::empty());
    let head_at = Instant::now();
    BodyDuties::hand(&mut response, |duties| duties.head_at = Some(head_at));
    let duties = BodyDuties::take(&mut response);
    assert_eq!(duties.head_at, Some(head_at));
    assert!(duties.failure.is_none());
}
