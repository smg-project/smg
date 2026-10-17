//! The access log's cost per request: the `http_request` span and its lines
//! under the JSON line format, written to a sink, for a unary response and
//! for a streamed one, with the record's optional fields off (the default)
//! and all of them on. The reference case runs the same handlers without the
//! access-log layer, so the difference is the log's own cost.
#![expect(clippy::unwrap_used, clippy::expect_used, reason = "a benchmark")]

use std::{hint::black_box, io, net::SocketAddr};

use axum::{
    body::{Body, Bytes},
    extract::connect_info::ConnectInfo,
    http::{header, HeaderValue, Request, StatusCode},
    middleware::from_fn_with_state,
    response::{IntoResponse, Response},
    routing::post,
    Router,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use http_body_util::BodyExt;
use smg::{
    config::{PolicyConfig, RouterConfig, RoutingMode},
    middleware::{
        auth_middleware, create_logging_layer, route_request_meta_middleware, AccessLogFields,
        AuthConfig, TenantResolutionState,
    },
    observability::log_format::{JsonLine, SpanFields},
};
use tower::ServiceExt;
use tracing_subscriber::{layer::SubscriberExt, Registry};

const KEY: &str = "sk-bench-0123456789abcdef";
const PAYLOAD: &[u8] = br#"{"model":"bench-model","messages":[{"role":"system","content":"You are a benchmark."},{"role":"user","content":"Write one line about access logs."}],"max_tokens":16,"stream":false}"#;
const REPLY: &str = r#"{"id":"chatcmpl-bench","object":"chat.completion","created":1700000000,"model":"bench-model","choices":[{"index":0,"message":{"role":"assistant","content":"Access logs record every request."},"finish_reason":"stop"}],"usage":{"prompt_tokens":24,"completion_tokens":7,"total_tokens":31}}"#;
const FRAME: &[u8] = b"data: {\"id\":\"chatcmpl-bench\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"}}]}\n\n";
const FRAMES: usize = 16;

/// A unary answer, routed: the worker header the routers set.
async fn chat(body: Bytes) -> Response {
    black_box(body.len());
    let mut response = (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        REPLY,
    )
        .into_response();
    response.headers_mut().insert(
        "x-smg-routed-worker-id",
        HeaderValue::from_static("http://127.0.0.1:8000"),
    );
    response
}

/// A streamed answer of [`FRAMES`] SSE frames.
async fn stream(body: Bytes) -> Response {
    black_box(body.len());
    let frames =
        futures::stream::iter((0..FRAMES).map(|_| Ok::<_, io::Error>(Bytes::from_static(FRAME))));
    let mut response = (
        [(header::CONTENT_TYPE, "text/event-stream")],
        Body::from_stream(frames),
    )
        .into_response();
    response.headers_mut().insert(
        "x-smg-routed-worker-id",
        HeaderValue::from_static("http://127.0.0.1:8000"),
    );
    response
}

/// The serving path's inner layers (auth, tenant resolution) around the two
/// handlers, with the access-log layer outside them when `fields` is set.
fn app(fields: Option<AccessLogFields>) -> Router {
    let config = RouterConfig::new(
        RoutingMode::Regular {
            worker_urls: vec!["http://127.0.0.1:8000".to_string()],
        },
        PolicyConfig::Random,
    );
    let tenants = TenantResolutionState::new(&config).expect("a tenant header name");
    let auth = AuthConfig::new(Some(KEY.to_string()));
    let routes = Router::new()
        .route("/v1/chat/completions", post(chat))
        .route("/v1/stream", post(stream))
        .route_layer(from_fn_with_state(tenants, route_request_meta_middleware))
        .route_layer(from_fn_with_state(auth, auth_middleware));
    match fields {
        Some(fields) => routes.layer(create_logging_layer(fields)),
        None => routes,
    }
}

fn request(path: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header(header::AUTHORIZATION, format!("Bearer {KEY}"))
        .header(header::USER_AGENT, "bench-client/1.0 (+criterion)")
        .header("x-forwarded-for", "203.0.113.9, 198.51.100.2")
        .header(header::CONTENT_TYPE, "application/json")
        .extension(ConnectInfo(
            "198.51.100.7:51514".parse::<SocketAddr>().unwrap(),
        ))
        .body(Body::from(PAYLOAD))
        .unwrap()
}

fn bench_access_log(c: &mut Criterion) {
    // The gateway's JSON lines, into a sink: the formatting is measured, the
    // write is not.
    let layer = tracing_subscriber::fmt::layer()
        .with_ansi(false)
        .with_file(true)
        .with_line_number(true)
        .with_writer(io::sink)
        .fmt_fields(SpanFields)
        .event_format(JsonLine::new());
    let _subscriber = tracing::subscriber::set_default(Registry::default().with(layer));
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a runtime");

    let mut group = c.benchmark_group("access_log");
    group.throughput(Throughput::Elements(1));
    for (path, kind) in [("/v1/chat/completions", "unary"), ("/v1/stream", "stream")] {
        for (fields, case) in [
            (None, "no_layer"),
            (Some(AccessLogFields::NONE), "default_fields"),
            (Some(AccessLogFields::all()), "all_fields"),
        ] {
            let app = app(fields);
            group.bench_with_input(BenchmarkId::new(kind, case), &path, |b, path| {
                b.iter(|| {
                    runtime.block_on(async {
                        let response = app.clone().oneshot(request(path)).await.unwrap();
                        assert_eq!(response.status(), StatusCode::OK);
                        let body = response.into_body().collect().await.unwrap().to_bytes();
                        black_box(body.len())
                    })
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_access_log);
criterion_main!(benches);
