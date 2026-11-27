use std::{
    convert::Infallible,
    io,
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::Duration,
};

use axum::{
    body::{Body, Bytes},
    extract::connect_info::ConnectInfo,
    http::{header, HeaderValue, Request, StatusCode},
    middleware::from_fn_with_state,
    response::{IntoResponse, Response},
    routing::post,
    Router,
};
use http_body_util::BodyExt;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tower::ServiceExt;
use tracing_subscriber::{fmt::MakeWriter, Layer, Registry};

use super::*;
use crate::{
    config::{PolicyConfig, RouterConfig, RoutingMode},
    middleware::{
        auth_middleware, route_request_meta_middleware, AuthConfig, HttpMetricsLayer, SseKeepAlive,
        TenantResolutionState,
    },
    observability::{
        inflight_tracker::InFlightRequestTracker,
        log_format::{JsonLine, SpanFields, TextFields, TextLine},
        logging::test_support::test_subscriber,
    },
};

const KEY: &str = "sk-test-5e1f0c9a7b3d2e4f";
const PAYLOAD: &[u8] = br#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#;
const REPLY: &str = r#"{"id":"chatcmpl-1","choices":[]}"#;
const WORKER: &str = "http://127.0.0.1:8000";
const PEER: &str = "198.51.100.7:51514";
const FORWARDED: &str = "203.0.113.9, 198.51.100.2";
const AGENT: &str = "test-client/1.0";
const FRAME: &[u8] = b"data: x\n\n";

/// The lines a subscriber wrote.
#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<u8>>>);

impl Capture {
    fn text(&self) -> String {
        String::from_utf8(self.0.lock().expect("lines").clone()).expect("utf-8 lines")
    }
}

impl io::Write for Capture {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.lock().expect("lines").extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl<'a> MakeWriter<'a> for Capture {
    type Writer = Capture;

    fn make_writer(&'a self) -> Capture {
        self.clone()
    }
}

/// The gateway's JSON line format, into `capture`.
fn json_subscriber(capture: &Capture) -> impl Layer<Registry> + Send + Sync {
    tracing_subscriber::fmt::layer()
        .with_ansi(false)
        .with_writer(capture.clone())
        .fmt_fields(SpanFields)
        .event_format(JsonLine::new())
}

/// The gateway's text format, into `capture`.
fn text_subscriber(capture: &Capture) -> impl Layer<Registry> + Send + Sync {
    tracing_subscriber::fmt::layer()
        .with_ansi(false)
        .with_writer(capture.clone())
        .fmt_fields(TextFields)
        .event_format(TextLine::new())
}

fn routed(mut response: Response) -> Response {
    response
        .headers_mut()
        .insert("x-smg-routed-worker-id", HeaderValue::from_static(WORKER));
    response
}

/// `/v1/chat/completions` answers [`REPLY`]; `/v1/stream` answers
/// `stream` as an SSE body; both read the request body and name the
/// worker. The serving path's auth and tenant layers sit inside the
/// access log, as they do in the gateway.
fn app(fields: AccessLogFields, stream: Option<Body>) -> Router {
    let config = RouterConfig::new(
        RoutingMode::Regular {
            worker_urls: vec![WORKER.to_string()],
        },
        PolicyConfig::Random,
    );
    let tenants = TenantResolutionState::new(&config).expect("a tenant header name");
    let auth = AuthConfig::new(Some(KEY.to_string()));
    let slot = Arc::new(Mutex::new(stream));
    Router::new()
        .route(
            "/v1/chat/completions",
            post(|body: Bytes| async move {
                assert_eq!(body.len(), PAYLOAD.len());
                routed((StatusCode::OK, REPLY).into_response())
            }),
        )
        .route(
            "/v1/stream",
            post({
                let slot = Arc::clone(&slot);
                move |body: Bytes| {
                    let stream = slot
                        .lock()
                        .expect("slot")
                        .take()
                        .expect("the stream handler is called once");
                    async move {
                        assert_eq!(body.len(), PAYLOAD.len());
                        routed(
                            ([(header::CONTENT_TYPE, "text/event-stream")], stream).into_response(),
                        )
                    }
                }
            }),
        )
        .route(
            "/v1/sized",
            post(move |body: Bytes| {
                let stream = slot
                    .lock()
                    .expect("slot")
                    .take()
                    .expect("the sized handler is called once");
                async move {
                    assert_eq!(body.len(), PAYLOAD.len());
                    routed(
                        Response::builder()
                            .header(header::CONTENT_LENGTH, 2 * FRAME.len())
                            .body(stream)
                            .expect("response"),
                    )
                }
            }),
        )
        .route_layer(SseKeepAlive::every_secs(15).layer())
        .route_layer(from_fn_with_state(tenants, route_request_meta_middleware))
        .route_layer(from_fn_with_state(auth, auth_middleware))
        .layer(create_logging_layer(fields))
        .layer(HttpMetricsLayer::new(InFlightRequestTracker::new()))
}

fn request(path: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header(header::AUTHORIZATION, format!("Bearer {KEY}"))
        .header(USER_AGENT, AGENT)
        .header("x-forwarded-for", FORWARDED)
        .extension(ConnectInfo(PEER.parse::<SocketAddr>().expect("a peer")))
        .body(Body::from(PAYLOAD))
        .expect("request")
}

/// The access log's JSON lines (the `smg::request` and `smg::response`
/// targets), as (message, span object).
fn access_lines(capture: &Capture) -> Vec<(String, serde_json::Map<String, Value>)> {
    capture
        .text()
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("a JSON line"))
        .filter(|line| {
            line["target"]
                .as_str()
                .is_some_and(|target| target.starts_with("smg::"))
        })
        .map(|line| {
            (
                line["message"].as_str().expect("a message").to_owned(),
                line["span"].as_object().expect("the span").clone(),
            )
        })
        .collect()
}

/// The span's keys, sorted (the line writes `name` last).
fn keys(span: &serde_json::Map<String, Value>) -> Vec<&str> {
    let mut keys: Vec<&str> = span.keys().map(String::as_str).collect();
    keys.sort_unstable();
    keys
}

/// The finish line's span fields as they have always been (no request
/// id layer in these apps).
const DEFAULT_FINISH_KEYS: [&str; 7] = [
    "latency",
    "method",
    "module",
    "name",
    "path",
    "status_code",
    "version",
];

/// Without a selection, the record is byte for byte what it was: the
/// same two lines, the same span keys.
#[tokio::test]
async fn the_default_record_keeps_its_fields_under_the_json_format() {
    let capture = Capture::default();
    let _subscriber = test_subscriber(json_subscriber(&capture));
    let response = app(AccessLogFields::NONE, None)
        .oneshot(request("/v1/chat/completions"))
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let body = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    assert_eq!(body.len(), REPLY.len());

    let lines = access_lines(&capture);
    assert_eq!(lines.len(), 2, "{lines:?}");
    assert_eq!(lines[0].0, "started processing request");
    assert_eq!(lines[1].0, "finished processing request");
    assert_eq!(keys(&lines[1].1), DEFAULT_FINISH_KEYS);
}

/// The finish line's latency runs from the edge's entry stamp when the
/// request runs inside its scope (the metrics layer's, ahead of this
/// layer in the gateway's stack), not from a read of this layer's own.
#[tokio::test]
async fn the_latency_runs_from_the_edge_s_entry_stamp() {
    let capture = Capture::default();
    let _subscriber = test_subscriber(json_subscriber(&capture));
    let app = Router::new()
        .route("/v1/chat/completions", post(|| async { "ok" }))
        .layer(create_logging_layer(AccessLogFields::NONE));
    let earlier = clock_now()
        .checked_sub(Duration::from_secs(5))
        .expect("the clock has been running for five seconds");
    let response = RequestStarted::at(earlier)
        .scope(app.oneshot(request("/v1/chat/completions")))
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let _ = response.into_body().collect().await.expect("body");

    let lines = access_lines(&capture);
    assert_eq!(lines.len(), 2, "{lines:?}");
    let latency = lines[1].1["latency"]
        .as_u64()
        .expect("the latency, in microseconds");
    assert!(
        latency >= 5_000_000,
        "latency {latency} us, expected at least the 5 s since the entry stamp"
    );
}

/// Every selected field is on the finish line with its value; the
/// request-side ones are on the first line too; the key itself is
/// nowhere.
#[tokio::test]
async fn every_selected_field_is_on_the_finish_line_under_the_json_format() {
    let capture = Capture::default();
    let _subscriber = test_subscriber(json_subscriber(&capture));
    let response = app(AccessLogFields::all(), None)
        .oneshot(request("/v1/chat/completions"))
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let _ = response.into_body().collect().await.expect("body");

    let lines = access_lines(&capture);
    assert_eq!(lines.len(), 2, "{lines:?}");
    let (message, span) = &lines[1];
    assert_eq!(message, "finished processing request");
    let hex: String = Sha256::digest(KEY.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    assert_eq!(span["client_addr"], PEER);
    assert_eq!(span["forwarded_for"], FORWARDED);
    assert_eq!(span["user_agent"], AGENT);
    assert_eq!(span["tenant"], format!("auth:{hex}"));
    assert_eq!(span["key_id"], &hex[..16]);
    assert_eq!(span["request_bytes"], PAYLOAD.len());
    assert_eq!(span["response_bytes"], REPLY.len());
    assert_eq!(span["worker"], WORKER);
    let mut expected: Vec<&str> = DEFAULT_FINISH_KEYS.to_vec();
    expected.extend(AccessLogField::ALL.iter().map(|field| field.as_str()));
    expected.sort_unstable();
    assert_eq!(keys(span), expected);

    let (message, span) = &lines[0];
    assert_eq!(message, "started processing request");
    assert_eq!(span["client_addr"], PEER);
    assert_eq!(span["forwarded_for"], FORWARDED);
    assert_eq!(span["user_agent"], AGENT);
    assert!(
        !capture.text().contains(KEY),
        "the key reached the log: {}",
        capture.text()
    );
}

/// A streamed body's bytes are known at its end: the finish line carries
/// no count, a line at the end does; without the field, no third line.
#[tokio::test]
async fn a_streamed_response_gets_its_bytes_on_an_end_line() {
    for (fields, expected_lines) in [(AccessLogFields::all(), 3), (AccessLogFields::NONE, 2)] {
        let capture = Capture::default();
        let _subscriber = test_subscriber(json_subscriber(&capture));
        let frames =
            futures::stream::iter((0..3).map(|_| Ok::<_, Infallible>(Bytes::from_static(FRAME))));
        let response = app(fields, Some(Body::from_stream(frames)))
            .oneshot(request("/v1/stream"))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);
        let body = response
            .into_body()
            .collect()
            .await
            .expect("body")
            .to_bytes();
        assert_eq!(body.len(), 3 * FRAME.len());

        let lines = access_lines(&capture);
        assert_eq!(lines.len(), expected_lines, "{fields:?}: {lines:?}");
        assert_eq!(lines[1].0, "finished processing request");
        assert!(
            !lines[1].1.contains_key("response_bytes"),
            "a stream's bytes are not known at its head: {:?}",
            lines[1]
        );
        if expected_lines == 3 {
            let (message, span) = &lines[2];
            assert_eq!(message, "finished streaming the response");
            assert_eq!(span["response_bytes"], 3 * FRAME.len());
            assert_eq!(span["worker"], WORKER);
            assert_eq!(span["status_code"], 200);
        }
    }
}

/// A keep-alive frame the edge wrote into a quiet stream reached the
/// client like any frame: the end line's bytes count it.
#[tokio::test(start_paused = true)]
async fn a_keep_alive_frame_counts_in_the_streamed_bytes() {
    let capture = Capture::default();
    let _subscriber = test_subscriber(json_subscriber(&capture));
    let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(4);
    let response = app(
        AccessLogFields::all(),
        Some(Body::from_stream(ReceiverStream::new(rx))),
    )
    .oneshot(request("/v1/stream"))
    .await
    .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body();
    tx.send(Ok(Bytes::from_static(FRAME))).await.expect("send");
    let first = body.frame().await.expect("a frame").expect("frame ok");
    assert_eq!(first.data_ref().map(|data| data.len()), Some(FRAME.len()));
    // Silence: the keep-alive comment comes through the edge.
    let ping = tokio::time::timeout(Duration::from_secs(16), body.frame())
        .await
        .expect("a keep-alive within the interval")
        .expect("a frame")
        .expect("frame ok");
    let ping_len = ping
        .data_ref()
        .map(|data| data.len())
        .expect("a data frame");
    assert_eq!(ping_len, b": ping\n\n".len());
    drop(tx);
    assert!(body.frame().await.is_none());

    let lines = access_lines(&capture);
    assert_eq!(lines.len(), 3, "{lines:?}");
    let (message, span) = &lines[2];
    assert_eq!(message, "finished streaming the response");
    assert_eq!(span["response_bytes"], FRAME.len() + ping_len);
}

/// A client that leaves mid-stream gets the bytes that went out on its
/// line.
#[tokio::test]
async fn a_client_that_leaves_mid_stream_gets_the_bytes_out_so_far() {
    let capture = Capture::default();
    let _subscriber = test_subscriber(json_subscriber(&capture));
    let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(4);
    let response = app(
        AccessLogFields::all(),
        Some(Body::from_stream(ReceiverStream::new(rx))),
    )
    .oneshot(request("/v1/stream"))
    .await
    .expect("response");
    tx.send(Ok(Bytes::from_static(FRAME))).await.expect("send");
    let mut body = response.into_body();
    let frame = body.frame().await.expect("one frame").expect("frame ok");
    assert_eq!(frame.data_ref().map(Bytes::len), Some(FRAME.len()));
    drop(body);
    drop(tx);

    let lines = access_lines(&capture);
    let (message, span) = lines.last().expect("a line");
    assert_eq!(
        message,
        "client closed the connection before the response stream ended"
    );
    assert_eq!(span["response_bytes_sent"], FRAME.len());
    assert!(
        !span.contains_key("response_bytes"),
        "a stream cut short has no length: {span:?}"
    );
    assert_eq!(span["request_bytes"], PAYLOAD.len());
}

/// A body under a declared length that the client leaves short of it:
/// the finish line said the length once, and the client-closed line says
/// the bytes that went out under a key of its own, so the text format
/// (which appends a re-recorded field rather than replacing it) carries
/// each value once.
#[tokio::test]
async fn a_declared_body_cut_short_keeps_the_length_once_and_adds_the_bytes_sent() {
    let capture = Capture::default();
    let _subscriber = test_subscriber(text_subscriber(&capture));
    let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(4);
    let response = app(
        AccessLogFields::all(),
        Some(Body::from_stream(ReceiverStream::new(rx))),
    )
    .oneshot(request("/v1/sized"))
    .await
    .expect("response");
    tx.send(Ok(Bytes::from_static(FRAME))).await.expect("send");
    let mut body = response.into_body();
    let frame = body.frame().await.expect("one frame").expect("frame ok");
    assert_eq!(frame.data_ref().map(Bytes::len), Some(FRAME.len()));
    drop(body);
    drop(tx);

    let text = capture.text();
    let finish = text
        .lines()
        .find(|line| line.contains("finished processing request"))
        .expect("the finish line");
    let declared = format!("response_bytes={}", 2 * FRAME.len());
    assert_eq!(finish.matches(&declared).count(), 1, "{finish}");
    assert!(!finish.contains("response_bytes_sent="), "{finish}");
    let closed = text
        .lines()
        .find(|line| line.contains("client closed the connection before the response stream ended"))
        .expect("the client-closed line");
    assert_eq!(closed.matches("response_bytes=").count(), 1, "{closed}");
    assert!(closed.contains(&declared), "{closed}");
    assert_eq!(
        closed.matches("response_bytes_sent=").count(),
        1,
        "{closed}"
    );
    assert!(
        closed.contains(&format!("response_bytes_sent={}", FRAME.len())),
        "{closed}"
    );
}

/// A stream that fails after the head went out: the failure line carries
/// the bytes out so far as `response_bytes_sent`, once, and a declared
/// length stays on its own key, once, under the text format.
#[tokio::test]
async fn a_stream_that_fails_keeps_the_length_once_and_adds_the_bytes_sent() {
    let capture = Capture::default();
    let _subscriber = test_subscriber(text_subscriber(&capture));
    let (tx, rx) = mpsc::channel::<Result<Bytes, io::Error>>(4);
    let response = app(
        AccessLogFields::all(),
        Some(Body::from_stream(ReceiverStream::new(rx))),
    )
    .oneshot(request("/v1/sized"))
    .await
    .expect("response");
    tx.send(Ok(Bytes::from_static(FRAME))).await.expect("send");
    tx.send(Err(io::Error::other("upstream cut")))
        .await
        .expect("send the failure");
    drop(tx);
    let mut body = response.into_body();
    let frame = body.frame().await.expect("one frame").expect("frame ok");
    assert_eq!(frame.data_ref().map(Bytes::len), Some(FRAME.len()));
    let failure = body.frame().await.expect("the failure");
    assert!(failure.is_err(), "the body ends in the failure");
    drop(body);

    let text = capture.text();
    let failed = text
        .lines()
        .find(|line| line.contains("response stream failed after the head was sent"))
        .expect("the failure line");
    assert_eq!(
        failed.matches("response_bytes_sent=").count(),
        1,
        "{failed}"
    );
    assert!(
        failed.contains(&format!("response_bytes_sent={}", FRAME.len())),
        "{failed}"
    );
    assert!(failed.matches("response_bytes=").count() <= 1, "{failed}");
    assert!(
        failed.contains(&format!("response_bytes={}", 2 * FRAME.len())),
        "{failed}"
    );
}

/// The text format carries the selected fields on the finish line, and
/// none of them without a selection.
#[tokio::test]
async fn the_text_format_carries_the_selected_fields_only() {
    for (fields, selected) in [
        (AccessLogFields::NONE, false),
        (AccessLogFields::all(), true),
    ] {
        let capture = Capture::default();
        let _subscriber = test_subscriber(text_subscriber(&capture));
        let response = app(fields, None)
            .oneshot(request("/v1/chat/completions"))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);
        let _ = response.into_body().collect().await.expect("body");

        let text = capture.text();
        let finish = text
            .lines()
            .find(|line| line.contains("finished processing request"))
            .expect("the finish line");
        assert!(finish.contains("status_code=200"), "{finish}");
        for needle in [
            format!("client_addr=\"{PEER}\""),
            format!("forwarded_for=\"{FORWARDED}\""),
            format!("user_agent=\"{AGENT}\""),
            "tenant=\"auth:".to_owned(),
            "key_id=\"".to_owned(),
            format!("request_bytes={}", PAYLOAD.len()),
            format!("response_bytes={}", REPLY.len()),
            format!("worker=\"{WORKER}\""),
        ] {
            assert_eq!(
                finish.contains(&needle),
                selected,
                "{needle} selected={selected}: {finish}"
            );
        }
    }
}

/// The record handle rides the request only while a field an inner
/// layer learns (the tenant, the key) is selected.
#[tokio::test]
async fn the_record_handle_rides_the_request_only_for_inner_fields() {
    for (fields, present) in [
        (
            AccessLogFields::from_fields(&[AccessLogField::ClientAddr, AccessLogField::Worker]),
            false,
        ),
        (
            AccessLogFields::from_fields(&[AccessLogField::Tenant]),
            true,
        ),
        (AccessLogFields::from_fields(&[AccessLogField::KeyId]), true),
    ] {
        let app = Router::new()
            .route(
                "/",
                post(move |request: Request<Body>| async move {
                    if request.extensions().get::<AccessLogRecord>().is_some() == present {
                        StatusCode::OK
                    } else {
                        StatusCode::INTERNAL_SERVER_ERROR
                    }
                }),
            )
            .layer(create_logging_layer(fields))
            .layer(HttpMetricsLayer::new(InFlightRequestTracker::new()));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(
            response.status(),
            StatusCode::OK,
            "handle present = {present} for {fields:?}"
        );
    }
}

/// A header value is carried as text, lossily decoded and cut at the
/// bound on a character boundary.
#[test]
fn header_fields_are_bounded_at_a_character_boundary() {
    let mut headers = HeaderMap::new();
    headers.insert(USER_AGENT, HeaderValue::from_static(AGENT));
    assert_eq!(header_field(&headers, &USER_AGENT).as_deref(), Some(AGENT));

    // 255 bytes, then a two-byte character straddling the bound.
    let long = format!("{}é", "a".repeat(HEADER_FIELD_MAX_BYTES - 1));
    headers.insert(
        USER_AGENT,
        HeaderValue::from_bytes(long.as_bytes()).expect("header"),
    );
    let field = header_field(&headers, &USER_AGENT).expect("a value");
    assert_eq!(field.len(), HEADER_FIELD_MAX_BYTES - 1);
    assert!(field.bytes().all(|byte| byte == b'a'));

    headers.insert(
        USER_AGENT,
        HeaderValue::from_bytes(b"\xff\xfeagent").expect("header"),
    );
    assert_eq!(
        header_field(&headers, &USER_AGENT).as_deref(),
        Some("\u{fffd}\u{fffd}agent")
    );
    assert_eq!(header_field(&headers, &X_FORWARDED_FOR), None);
    assert_eq!(header_lines_field(&headers, &X_FORWARDED_FOR), None);
}

/// Every `X-Forwarded-For` field line is in the record, in order, joined
/// as the combined value: a proxy that adds a line of its own is not
/// hidden behind the line the client sent.
#[test]
fn every_forwarded_for_line_is_in_the_record_in_order() {
    let mut headers = HeaderMap::new();
    headers.append(
        X_FORWARDED_FOR.clone(),
        HeaderValue::from_static("203.0.113.9"),
    );
    headers.append(
        X_FORWARDED_FOR.clone(),
        HeaderValue::from_static("198.51.100.2, 192.0.2.1"),
    );
    assert_eq!(
        header_lines_field(&headers, &X_FORWARDED_FOR).as_deref(),
        Some("203.0.113.9, 198.51.100.2, 192.0.2.1")
    );
    assert_eq!(
        header_field(&headers, &X_FORWARDED_FOR).as_deref(),
        Some("203.0.113.9"),
        "the single-line reader sees the first line alone"
    );

    // The joined text is bounded like a single value, at a character
    // boundary.
    let mut headers = HeaderMap::new();
    headers.append(
        X_FORWARDED_FOR.clone(),
        HeaderValue::from_bytes("a".repeat(HEADER_FIELD_MAX_BYTES - 3).as_bytes()).expect("header"),
    );
    headers.append(
        X_FORWARDED_FOR.clone(),
        HeaderValue::from_bytes("é".as_bytes()).expect("header"),
    );
    let field = header_lines_field(&headers, &X_FORWARDED_FOR).expect("a value");
    assert_eq!(field.len(), HEADER_FIELD_MAX_BYTES - 1);
    assert!(field.ends_with(", "), "{field}");
}

#[test]
fn the_field_set_holds_what_was_selected() {
    assert_eq!(AccessLogFields::from_fields(&[]), AccessLogFields::NONE);
    let set = AccessLogFields::from_fields(&[AccessLogField::Worker, AccessLogField::Tenant]);
    for field in AccessLogField::ALL {
        assert_eq!(
            set.contains(field),
            matches!(field, AccessLogField::Worker | AccessLogField::Tenant),
            "{field}"
        );
        assert!(AccessLogFields::all().contains(field), "{field}");
        assert!(!AccessLogFields::NONE.contains(field), "{field}");
    }
}
