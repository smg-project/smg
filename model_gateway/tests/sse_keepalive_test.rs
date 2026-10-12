//! A keep-alive on every open SSE stream while the engine is silent.
//!
//! The gateway sends a streamed response's headers at once and its first
//! delta when the engine produces it; a long prefill leaves the connection
//! carrying nothing in between, which an idle-timeout proxy cuts. With the
//! keep-alive interval at one second and an engine that speaks after three
//! and a half, each of the four streaming endpoints carries keep-alive frames
//! through the silence and none once deltas flow: an SSE comment on the
//! OpenAI-shaped endpoints, the `ping` event on the Messages API. With the
//! interval at zero nothing is sent, as before.

#[path = "common/mod.rs"]
mod common;

#[path = "common/held_worker.rs"]
mod held_worker;

#[path = "common/sse_keepalive_support.rs"]
mod support;

use std::time::Duration;

use serde_json::{json, Value};
use support::{app, stream, stream_to_end, Arrival, MODEL};

const COMMENT: &str = ": ping\n\n";
const PING: &str = "event: ping\ndata: {\"type\":\"ping\"}\n\n";
/// The engine's silence before its first token.
const SILENCE: Duration = Duration::from_millis(3500);
const BUDGET: Duration = Duration::from_secs(20);

/// A streaming endpoint: its path, request, keep-alive frame and the frame
/// that ends the stream.
struct Endpoint {
    path: &'static str,
    body: Value,
    keep_alive: &'static str,
    end: &'static str,
}

fn endpoints() -> Vec<Endpoint> {
    vec![
        Endpoint {
            path: "/v1/chat/completions",
            body: json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": true,
                "max_tokens": 3,
            }),
            keep_alive: COMMENT,
            end: "data: [DONE]",
        },
        Endpoint {
            path: "/v1/completions",
            body: json!({"model": MODEL, "prompt": "Hello", "stream": true, "max_tokens": 3}),
            keep_alive: COMMENT,
            end: "data: [DONE]",
        },
        Endpoint {
            path: "/v1/responses",
            body: json!({
                "model": MODEL,
                "input": "Hello",
                "stream": true,
                "store": false,
                // The API's minimum; the held worker stops after three tokens anyway.
                "max_output_tokens": 16,
            }),
            keep_alive: COMMENT,
            end: "response.completed",
        },
        Endpoint {
            path: "/v1/messages",
            body: json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": true,
                "max_tokens": 3,
            }),
            keep_alive: PING,
            end: "message_stop",
        },
    ]
}

/// The arrivals split at the engine's first token: the first frame that is
/// not a keep-alive and arrived once the engine's silence was over (the
/// early events of a stream come well before, the tokens only after).
fn split_at_first_token<'a>(
    arrivals: &'a [Arrival],
    keep_alive: &str,
    silence: Duration,
) -> (&'a [Arrival], &'a [Arrival]) {
    let first = arrivals
        .iter()
        .position(|arrival| arrival.at >= silence && arrival.text() != keep_alive);
    assert!(
        first.is_some(),
        "no frame after the {silence:?} silence: {arrivals:#?}"
    );
    arrivals.split_at(first.unwrap_or(arrivals.len()))
}

fn keep_alives<'a>(arrivals: &'a [Arrival], keep_alive: &str) -> Vec<&'a Arrival> {
    arrivals
        .iter()
        .filter(|arrival| arrival.text() == keep_alive)
        .collect()
}

#[tokio::test]
async fn every_streaming_endpoint_carries_a_keep_alive_through_the_engine_s_silence() {
    let (app, server) = app(SILENCE, Some(1)).await;
    for endpoint in endpoints() {
        let (arrivals, eof) =
            stream_to_end(&app, endpoint.path, endpoint.body.clone(), BUDGET).await;
        let (silence, flowing) = split_at_first_token(&arrivals, endpoint.keep_alive, SILENCE);
        let during = keep_alives(silence, endpoint.keep_alive);
        assert!(
            (2..=4).contains(&during.len()),
            "{}: {} keep-alive frames during a {SILENCE:?} silence at a 1 s interval: {arrivals:#?}",
            endpoint.path,
            during.len()
        );
        // Each one an interval after the previous frame, whatever it was.
        for pair in silence.windows(2) {
            if pair[1].text() == endpoint.keep_alive {
                let gap = pair[1].at.saturating_sub(pair[0].at);
                assert!(
                    gap >= Duration::from_millis(900),
                    "{}: a keep-alive {gap:?} after the previous frame: {arrivals:#?}",
                    endpoint.path
                );
            }
        }
        assert!(
            keep_alives(flowing, endpoint.keep_alive).is_empty(),
            "{}: a keep-alive while deltas flow: {flowing:#?}",
            endpoint.path
        );
        let whole: String = arrivals.iter().map(Arrival::text).collect();
        assert!(
            eof && whole.contains(endpoint.end),
            "{}: the stream ends as before and the body closes (eof={eof}): {whole:?}",
            endpoint.path
        );
        // Every data line is still whole JSON (or the terminator): no
        // keep-alive landed inside a frame.
        for data in whole.lines().filter_map(|line| line.strip_prefix("data: ")) {
            assert!(
                data == "[DONE]" || serde_json::from_str::<Value>(data).is_ok(),
                "{}: a data line that is not JSON: {data:?}",
                endpoint.path
            );
        }
    }
    server.abort();
}

#[tokio::test]
async fn an_interval_of_zero_sends_nothing_through_the_silence() {
    let (app, server) = app(SILENCE, Some(0)).await;
    let endpoints = endpoints();
    let chat = &endpoints[0];
    let arrivals = stream(&app, chat.path, chat.body.clone(), BUDGET).await;
    let (silence, _) = split_at_first_token(&arrivals, chat.keep_alive, SILENCE);
    assert!(
        silence.is_empty(),
        "nothing before the first token with the keep-alive off: {arrivals:#?}"
    );
    server.abort();
}
