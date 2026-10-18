//! The bytes the HTTP routers relay from a streaming worker, byte for byte:
//! the regular router (with and without the provider re-chunking) and the
//! disaggregated prefill/decode router (its decode side's stream), over the
//! same scripted upstream bodies, compared with the dumps recorded under
//! `tests/fixtures/relay_dumps`. `RELAY_DUMP_RECORD=1` rewrites the dumps.
//!
//! The script splits frames mid-line and mid-frame, carries an empty chunk
//! and a CRLF frame, and sends a frame after the `[DONE]` sentinel in a chunk
//! of its own: the regular relay forwards everything, the decode relay ends
//! at the sentinel, and both are recorded as they are.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::allow_attributes
)]

#[path = "common/mod.rs"]
mod common;

use std::{fs, path::PathBuf};

use axum::{
    body::Body,
    http::{header::CONTENT_TYPE, Method, Request},
};
use common::{
    mock_worker::{set_stream_script, HealthStatus, MockWorkerConfig, StreamScript, WorkerType},
    AppTestContext,
};
use http_body_util::BodyExt;
use openai_protocol::profile::ProviderProfile;
use serde_json::{json, Value};
use smg::config::RouterConfig;
use tower::ServiceExt;

/// An alias under the provider whose profile turns the regular router's
/// re-chunking on; registered for the served model, the profile applies to
/// requests for the served model's own name.
const RECHUNKING_ALIAS: &str = "MiniMax-M2";

fn script() -> StreamScript {
    StreamScript {
        chunks: vec![
            b"data: {\"id\":\"one\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"},\"finish_reason\":null}]}\n".to_vec(),
            b"\ndata: {\"id\":\"two\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"con".to_vec(),
            b"tent\":\"lo, \"},\"finish_reason\":null}]}\n\n".to_vec(),
            Vec::new(),
            b"data: {\"id\":\"three\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"world\"},\"finish_reason\":\"stop\"}]}\r\n\r\n".to_vec(),
            b"data: [DONE]\n\n".to_vec(),
            b"data: {\"id\":\"after\",\"object\":\"chat.completion.chunk\",\"choices\":[]}\n\n".to_vec(),
        ],
    }
}

fn worker(port: u16, worker_type: WorkerType) -> MockWorkerConfig {
    set_stream_script(port, script());
    MockWorkerConfig {
        port,
        worker_type,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    }
}

fn base() -> smg::config::RouterConfigBuilder {
    RouterConfig::builder()
        .random_policy()
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(256 * 1024 * 1024)
        .request_timeout_secs(600)
        .worker_startup_timeout_secs(1)
        .worker_startup_check_interval_secs(1)
        .max_concurrent_requests(64)
        .queue_timeout_secs(60)
}

async fn regular_app(port: u16) -> AppTestContext {
    let mut config = base().regular_mode(vec![]).build_unchecked();
    config.health_check.disable_health_check = true;
    AppTestContext::new_with_config(config, vec![worker(port, WorkerType::Regular)]).await
}

async fn pd_app(prefill_port: u16, decode_port: u16) -> AppTestContext {
    let mut config = base()
        .prefill_decode_mode(
            vec![(
                format!("http://127.0.0.1:{prefill_port}"),
                Some(prefill_port),
            )],
            vec![format!("http://127.0.0.1:{decode_port}")],
        )
        .build_unchecked();
    config.health_check.disable_health_check = true;
    AppTestContext::new_with_config(
        config,
        vec![
            worker(prefill_port, WorkerType::Prefill),
            worker(decode_port, WorkerType::Decode),
        ],
    )
    .await
}

/// The status, content type and body the client gets for `body` on `path`.
async fn dump(ctx: &AppTestContext, path: &str, body: Value) -> Vec<u8> {
    let app = ctx.create_app();
    let request = Request::builder()
        .method(Method::POST)
        .uri(path)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let content_type = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let mut out = format!("HTTP {status} {content_type}\n\n").into_bytes();
    out.extend_from_slice(&bytes);
    out
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/relay_dumps")
        .join(format!("{name}.sse"))
}

/// Compare `got` with the recorded dump, or record it when asked.
fn check(name: &str, got: &[u8]) {
    let path = fixture(name);
    if std::env::var_os("RELAY_DUMP_RECORD").is_some() {
        fs::write(&path, got).unwrap();
        return;
    }
    let want = fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert!(
        want == got,
        "{name}: the relayed bytes differ from the recorded dump\n--- recorded\n{}\n--- got\n{}",
        String::from_utf8_lossy(&want),
        String::from_utf8_lossy(got)
    );
}

fn chat(model: &str) -> Value {
    json!({"model": model, "messages": [{"role": "user", "content": "hi"}], "stream": true})
}

#[tokio::test]
async fn the_regular_router_relays_the_scripted_streams_byte_for_byte() {
    let ctx = regular_app(18901).await;
    check(
        "regular_chat",
        &dump(&ctx, "/v1/chat/completions", chat("mock-model")).await,
    );
    // The same stream once the served model carries the re-chunking
    // provider's profile (the registration is process-wide; the plain case
    // above ran first, and the decode relay never re-chunks).
    ProviderProfile::register_model_aliases([(RECHUNKING_ALIAS, "mock-model")]);
    check(
        "regular_chat_rechunked",
        &dump(&ctx, "/v1/chat/completions", chat("mock-model")).await,
    );
    check(
        "regular_completions",
        &dump(
            &ctx,
            "/v1/completions",
            json!({"model": "mock-model", "prompt": "hi", "stream": true}),
        )
        .await,
    );
    check(
        "regular_generate",
        &dump(&ctx, "/generate", json!({"text": "hi", "stream": true})).await,
    );
    ctx.shutdown().await;
}

#[tokio::test]
async fn the_disaggregated_router_relays_the_decode_streams_byte_for_byte() {
    let ctx = pd_app(18902, 18903).await;
    check(
        "pd_chat",
        &dump(&ctx, "/v1/chat/completions", chat("mock-model")).await,
    );
    check(
        "pd_generate",
        &dump(&ctx, "/generate", json!({"text": "hi", "stream": true})).await,
    );
    ctx.shutdown().await;
}
