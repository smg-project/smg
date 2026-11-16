//! By default an open SSE stream carries a keep-alive after fifteen seconds
//! of engine silence, inside the idle window of common proxies: a chat
//! stream whose engine has not spoken gets an SSE comment at the fifteenth
//! second, where before nothing was sent until the first delta.
//!
//! The clock pauses once the response headers are in. The request and its
//! hops to the engine run on the wall clock, where a paused clock's
//! auto-advance could walk past a socket to a timeout; the silence after
//! the headers has only the gateway's timers, which the runtime advances to
//! with no wall time spent. The keep-alive is due fifteen seconds after the
//! headers on the timer wheel's millisecond, from a timer armed at most the
//! request's own duration before the pause.

use std::time::Duration;

use serde_json::json;
use support::{app, stream_paused_from_the_headers, MODEL};

use crate::support;

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
    let (arrivals, slack) =
        stream_paused_from_the_headers(&app, "/v1/chat/completions", body, budget).await;
    server.abort();
    assert!(
        !arrivals.is_empty(),
        "nothing on the stream within {budget:?} of silence"
    );
    let first = &arrivals[0];
    assert_eq!(first.text(), ": ping\n\n", "{arrivals:#?}");
    let default = Duration::from_secs(15);
    assert!(
        first.at <= default + Duration::from_millis(1),
        "the first keep-alive is late against the 15 s default, past the timer wheel's millisecond: {arrivals:#?}"
    );
    assert!(
        first.at + slack >= default,
        "the first keep-alive is early against the 15 s default, armed at most {slack:?} before the clock paused: {arrivals:#?}"
    );
    assert!(
        arrivals
            .iter()
            .all(|arrival| arrival.text() == ": ping\n\n"),
        "only keep-alive frames while the engine is silent: {arrivals:#?}"
    );
}
