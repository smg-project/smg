use std::convert::Infallible;

use axum::{body::to_bytes, Router};
use futures::{stream, StreamExt};
use tokio::time::Instant;
use tower::ServiceExt;

use super::{super::request_start::RequestStarted, *};
use crate::middleware::{
    test_support::{echo_request, echo_route, json_of, ECHO_PATH},
    RequestBodyLimitLayer,
};

fn app(timeouts: RequestBodyTimeouts) -> Router {
    Router::new()
        .route(ECHO_PATH, echo_route())
        .layer(timeouts.layer())
}

fn timeouts(stall_secs: u64, total_secs: u64) -> RequestBodyTimeouts {
    RequestBodyTimeouts {
        stall: (stall_secs > 0).then(|| Duration::from_secs(stall_secs)),
        total: Duration::from_secs(total_secs),
    }
}

/// One byte per `interval`, `frames` times, then the end of the body.
fn trickle(frames: usize, interval: Duration) -> Body {
    Body::from_stream(stream::unfold(0usize, move |sent| async move {
        if sent == frames {
            return None;
        }
        tokio::time::sleep(interval).await;
        Some((Ok::<_, Infallible>(Bytes::from_static(b"x")), sent + 1))
    }))
}

async fn error_code(response: Response) -> String {
    let json = json_of(response).await;
    json["error"]["code"].as_str().unwrap().to_string()
}

#[tokio::test(start_paused = true)]
async fn a_complete_body_passes_through_untouched() {
    let response = app(timeouts(5, 60))
        .oneshot(echo_request(Body::from("payload")))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(&body[..], b"payload");
}

/// The upload wait of a body runs from the first poll that found it
/// pending to where its reader drops it, before the handler has the
/// bytes: under the entry stamp, a body that comes in two parts with a
/// pause before each leaves its handler a stamp whose upload wait is the
/// two pauses, on the clock the test holds, so a wait counted from the
/// entry reads that much less. A body that came whole takes nothing off.
#[tokio::test(start_paused = true)]
async fn the_upload_wait_of_a_body_that_came_in_parts_is_what_its_read_waited() {
    const PAUSE: Duration = Duration::from_secs(2);
    const WAIT: Duration = Duration::from_secs(100);
    let app = || {
        Router::new()
            .route(
                ECHO_PATH,
                axum::routing::post(|body: Bytes| async move {
                    let stamp = RequestStarted::carried().expect("the request's stamp");
                    format!(
                        "{} {}",
                        body.len(),
                        stamp.less_upload_wait(WAIT).as_millis()
                    )
                }),
            )
            .layer(timeouts(5, 60).layer())
    };
    for (body, expected, what) in [
        (
            trickle(2, PAUSE),
            &b"2 96000"[..],
            "two bytes, the two pauses off the wait",
        ),
        (
            Body::from("ab"),
            &b"2 100000"[..],
            "a body that came whole takes nothing off",
        ),
    ] {
        let response = RequestStarted::now()
            .scope(app().oneshot(echo_request(body)))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let read = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(&read[..], expected, "{what}");
    }
}

/// A reader outside the layer measures its body the same way: a body in
/// two parts read through the measure leaves the two pauses in the
/// request's stamp, and one that came whole leaves nothing.
#[tokio::test(start_paused = true)]
async fn a_body_read_through_the_measure_outside_the_layer_adds_its_wait() {
    const PAUSE: Duration = Duration::from_secs(2);
    const WAIT: Duration = Duration::from_secs(100);
    for (body, expected) in [
        (trickle(2, PAUSE), Duration::from_secs(96)),
        (Body::from("ab"), WAIT),
    ] {
        let left = RequestStarted::now()
            .scope(async {
                let read = to_bytes(upload_measured(body), usize::MAX).await.unwrap();
                assert_eq!(read.len(), 2);
                RequestStarted::carried()
                    .expect("the request's stamp")
                    .less_upload_wait(WAIT)
            })
            .await;
        assert_eq!(left, expected);
    }
}

/// A body that stops after its first bytes ends with 408 once a single
/// client wait reaches the stall timeout, long before the request
/// timeout.
#[tokio::test(start_paused = true)]
async fn a_stalled_body_is_408_request_body_stalled() {
    let body = Body::from_stream(
        stream::iter([Ok::<_, Infallible>(Bytes::from_static(b"{\"partial\":"))])
            .chain(stream::pending()),
    );
    let started = Instant::now();
    let response = app(timeouts(5, 120))
        .oneshot(echo_request(body))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);
    assert_eq!(error_code(response).await, REQUEST_BODY_STALLED);
    let waited = started.elapsed();
    assert!(
        waited >= Duration::from_secs(5) && waited < Duration::from_secs(6),
        "ended after {waited:?}, expected the 5 s stall timeout"
    );
}

/// A trickle whose waits each stay under the stall timeout is still cut
/// at the request timeout.
#[tokio::test(start_paused = true)]
async fn a_trickling_body_is_cut_at_the_request_timeout() {
    let started = Instant::now();
    let response = app(timeouts(5, 20))
        .oneshot(echo_request(trickle(usize::MAX, Duration::from_secs(1))))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);
    assert_eq!(error_code(response).await, REQUEST_BODY_TIMEOUT);
    let waited = started.elapsed();
    assert!(
        waited >= Duration::from_secs(20) && waited < Duration::from_secs(22),
        "ended after {waited:?}, expected the 20 s request timeout"
    );
}

/// The deadline is armed where this layer is reached, inside the
/// admission layers: time the request spent before it (a wait for a
/// permit; here an entry stamp 15 s in the past) is not the client's
/// budget, which stays the whole request timeout.
#[tokio::test(start_paused = true)]
async fn time_before_this_layer_leaves_the_request_timeout_whole() {
    let started = Instant::now();
    let earlier = started
        .into_std()
        .checked_sub(Duration::from_secs(15))
        .expect("the clock has been running for fifteen seconds");
    let request = echo_request(trickle(usize::MAX, Duration::from_secs(1)));
    let response = RequestStarted::at(earlier)
        .scope(app(timeouts(5, 20)).oneshot(request))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);
    assert_eq!(error_code(response).await, REQUEST_BODY_TIMEOUT);
    let waited = started.elapsed();
    assert!(
        waited >= Duration::from_secs(20) && waited < Duration::from_secs(22),
        "ended after {waited:?}, expected the whole 20 s request timeout from this layer"
    );
}

/// The request timeout counts from the wrap, where this layer is
/// reached, not from the first read of the body: a handler that reads
/// late is cut at the timeout the wrap started.
#[tokio::test(start_paused = true)]
async fn the_request_timeout_counts_from_the_wrap_not_from_the_first_read() {
    let wrapped_at = Instant::now();
    let (request, outcome) = timed(
        echo_request(trickle(usize::MAX, Duration::from_secs(1))),
        timeouts(0, 5),
    );
    tokio::time::sleep(Duration::from_secs(3)).await;
    let read = to_bytes(request.into_body(), usize::MAX).await;
    assert!(read.is_err(), "the body ends with the timeout's error");
    assert_eq!(outcome.load(Ordering::Acquire), TIMED_OUT);
    let waited = wrapped_at.elapsed();
    assert!(
        waited >= Duration::from_secs(5) && waited < Duration::from_secs(6),
        "cut {waited:?} after the wrap: the 5 s timeout counted from the wrap, not from the first read at 3 s"
    );
}

/// A `request_timeout_secs` no instant can hold (the "no timeout" idiom;
/// validation rejects only zero) is a deadline that never comes, as the
/// replaced `sleep` saturated to, not an overflow at the wrap.
#[tokio::test(start_paused = true)]
async fn a_total_no_instant_can_hold_is_a_deadline_that_never_comes() {
    let (request, outcome) = timed(
        echo_request(trickle(3, Duration::from_secs(86_400 * 365))),
        timeouts(0, u64::MAX),
    );
    let read = to_bytes(request.into_body(), usize::MAX).await;
    assert_eq!(
        read.expect("three frames a year apart arrive whole"),
        Bytes::from_static(b"xxx")
    );
    assert_ne!(outcome.load(Ordering::Acquire), TIMED_OUT);
}

/// Progress inside the stall window resets the watchdog: a slow but
/// moving upload that completes before the request timeout is served.
#[tokio::test(start_paused = true)]
async fn a_slow_but_moving_body_is_served() {
    let response = app(timeouts(5, 60))
        .oneshot(echo_request(trickle(4, Duration::from_secs(3))))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(&body[..], b"xxxx");
}

/// With the stall watchdog off (0), only the request timeout bounds the
/// read.
#[tokio::test(start_paused = true)]
async fn stall_watchdog_off_leaves_the_request_timeout() {
    let body = Body::from_stream(stream::pending::<Result<Bytes, Infallible>>());
    let started = Instant::now();
    let response = app(timeouts(0, 10))
        .oneshot(echo_request(body))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);
    assert_eq!(error_code(response).await, REQUEST_BODY_TIMEOUT);
    assert!(started.elapsed() >= Duration::from_secs(10));
}

/// A read the payload limit cut short is 413 request_body_too_large,
/// whatever the handler made of the failed read: the limit layer sits
/// outside the watchdog, so its error reaches the handler boxed once
/// more, past where the extractors look for it.
#[tokio::test(start_paused = true)]
async fn a_body_over_the_payload_limit_is_413_request_body_too_large() {
    let app = app(timeouts(5, 60)).layer(RequestBodyLimitLayer::new(8));
    let response = app
        .oneshot(echo_request(Body::from("more than eight bytes")))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(error_code(response).await, "request_body_too_large");
}

/// Any other failed read keeps the handler's own answer.
#[tokio::test(start_paused = true)]
async fn another_failed_read_keeps_the_handlers_answer() {
    let body = Body::from_stream(stream::iter([Err::<Bytes, _>(std::io::Error::other(
        "connection reset",
    ))]));
    let response = app(timeouts(5, 60))
        .oneshot(echo_request(body))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}
