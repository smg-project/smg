//! By default an open SSE stream carries a keep-alive after fifteen seconds
//! of engine silence, inside the idle window of common proxies: a chat
//! stream whose engine has not spoken gets an SSE comment before the
//! sixteenth second, where before nothing was sent until the first delta.

#[path = "common/mod.rs"]
mod common;

#[path = "common/held_worker.rs"]
mod held_worker;

#[path = "common/sse_keepalive_support.rs"]
mod support;

use std::time::Duration;

use serde_json::json;
use support::{app, stream, MODEL};

#[tokio::test]
async fn the_default_configuration_keeps_a_silent_chat_stream_alive_within_the_proxy_idle_window() {
    // The engine never speaks within the budget: whatever arrives is the
    // gateway's own keep-alive.
    let (app, server) = app(Duration::from_secs(120), None).await;
    let body = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": true,
        "max_tokens": 3,
    });
    let budget = Duration::from_millis(16_500);
    let arrivals = stream(&app, "/v1/chat/completions", body, budget).await;
    server.abort();
    assert!(
        !arrivals.is_empty(),
        "nothing on the stream within {budget:?} of silence"
    );
    let first = &arrivals[0];
    assert_eq!(first.text(), ": ping\n\n", "{arrivals:#?}");
    assert!(
        first.at >= Duration::from_secs(14) && first.at < Duration::from_secs(16),
        "the first keep-alive at {:?}, not at the 15 s default",
        first.at
    );
    assert!(
        arrivals
            .iter()
            .all(|arrival| arrival.text() == ": ping\n\n"),
        "only keep-alive frames while the engine is silent: {arrivals:#?}"
    );
}
