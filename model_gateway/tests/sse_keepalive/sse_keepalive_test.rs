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
//!
//! The clock pauses once each response's headers are in (the request and
//! its hops to the engine run on the wall clock, as the default test's doc
//! says) and resumes a margin before the engine speaks, as its frames cross
//! a real socket: the keep-alive timer is tokio's, so the runtime advances
//! to it with no wall time spent and the tests read its frames as exact
//! instants instead of judging on windows.

use std::time::Duration;

use serde_json::{json, Value};
use support::{app, stream_to_end_paused_through, Arrival, MODEL};

use crate::support;

const COMMENT: &str = ": ping\n\n";
const PING: &str = "event: ping\ndata: {\"type\":\"ping\"}\n\n";
/// The engine's silence before its first token.
const SILENCE: Duration = Duration::from_millis(3500);
/// The keep-alive interval the tests configure.
const INTERVAL: Duration = Duration::from_secs(1);
const BUDGET: Duration = Duration::from_secs(20);
/// The timer wheel's resolution: a timer fires on its deadline's millisecond.
const WHEEL: Duration = Duration::from_millis(1);

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

/// The arrivals split at the engine's first frame: the first frame after the
/// first keep-alive that is not a keep-alive. A stream's early events come
/// with the response headers, before any keep-alive is due; the engine's
/// frames come after its silence, once keep-alive frames went out. The split
/// reads the frame order and not the arrival clock: that clock starts at the
/// response headers, while the engine's silence started at the request it
/// was sent before them, so on a loaded host the engine's frames land a few
/// milliseconds before the silence's length on the arrival clock.
fn split_at_first_engine_frame<'a>(
    arrivals: &'a [Arrival],
    keep_alive: &str,
) -> (&'a [Arrival], &'a [Arrival]) {
    let first_keep_alive = arrivals
        .iter()
        .position(|arrival| arrival.text() == keep_alive);
    assert!(
        first_keep_alive.is_some(),
        "no keep-alive before the engine's first frame: {arrivals:#?}"
    );
    let searched_from = first_keep_alive.unwrap_or(arrivals.len());
    let first_frame = arrivals[searched_from..]
        .iter()
        .position(|arrival| arrival.text() != keep_alive)
        .map(|offset| searched_from + offset);
    assert!(
        first_frame.is_some(),
        "no frame of the engine's after the keep-alive frames: {arrivals:#?}"
    );
    arrivals.split_at(first_frame.unwrap_or(arrivals.len()))
}

fn keep_alives<'a>(arrivals: &'a [Arrival], keep_alive: &str) -> Vec<&'a Arrival> {
    arrivals
        .iter()
        .filter(|arrival| arrival.text() == keep_alive)
        .collect()
}

#[tokio::test]
async fn every_streaming_endpoint_carries_a_keep_alive_through_the_engine_s_silence() {
    let (app, server) = app(SILENCE, Some(INTERVAL.as_secs())).await;
    for endpoint in endpoints() {
        let read = stream_to_end_paused_through(
            &app,
            endpoint.path,
            endpoint.body.clone(),
            BUDGET,
            SILENCE,
        )
        .await;
        let arrivals = &read.arrivals;
        let (silence, flowing) = split_at_first_engine_frame(arrivals, endpoint.keep_alive);
        let during = keep_alives(silence, endpoint.keep_alive);
        // One keep-alive per whole interval of the silence: every one due
        // before the clock resumed is in, read exact on the paused clock,
        // and the silence holds no more than its length allows.
        let at_least = read.resumed_at.as_millis() / INTERVAL.as_millis();
        let at_most = SILENCE.as_millis() / INTERVAL.as_millis();
        assert!(
            (at_least..=at_most).contains(&(during.len() as u128)),
            "{}: {} keep-alive frames during a {SILENCE:?} silence at a {INTERVAL:?} interval (the \
             clock resumed at {:?}, slack {:?}): {arrivals:#?}",
            endpoint.path,
            during.len(),
            read.resumed_at,
            read.slack
        );
        // Each one read on the paused clock an interval after the previous
        // frame, whatever it was, on the timer wheel's millisecond.
        for pair in silence.windows(2) {
            if pair[1].text() == endpoint.keep_alive && pair[1].at < read.resumed_at {
                let gap = pair[1].at.saturating_sub(pair[0].at);
                assert!(
                    gap >= INTERVAL && gap <= INTERVAL + WHEEL,
                    "{}: a keep-alive {gap:?} after the previous frame, not {INTERVAL:?}: {arrivals:#?}",
                    endpoint.path
                );
            }
        }
        // The first one interval after the headers, where the timer was
        // armed at most `slack` before the clock paused.
        if read.resumed_at >= INTERVAL {
            let first = during
                .first()
                .unwrap_or_else(|| {
                    panic!(
                        "{}: no keep-alive arrived in the silence, the first was due one interval \
                         after the headers: {arrivals:#?}",
                        endpoint.path
                    )
                })
                .at;
            assert!(
                first <= INTERVAL + WHEEL && first + read.slack >= INTERVAL,
                "{}: the first keep-alive at {first:?}, not one interval after the headers (slack \
                 {:?}): {arrivals:#?}",
                endpoint.path,
                read.slack
            );
        }
        assert!(
            keep_alives(flowing, endpoint.keep_alive).is_empty(),
            "{}: a keep-alive while deltas flow: {flowing:#?}",
            endpoint.path
        );
        let whole: String = arrivals.iter().map(Arrival::text).collect();
        assert!(
            read.eof && whole.contains(endpoint.end),
            "{}: the stream ends as before and the body closes (eof={}): {whole:?}",
            endpoint.path,
            read.eof
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
    let read =
        stream_to_end_paused_through(&app, chat.path, chat.body.clone(), BUDGET, SILENCE).await;
    let arrivals = &read.arrivals;
    assert!(
        keep_alives(arrivals, chat.keep_alive).is_empty(),
        "a keep-alive with the interval at zero: {arrivals:#?}"
    );
    // The engine's frames are the first on the stream, at the end of its
    // silence (started at the request, up to `slack` before the arrival
    // clock), through which a 1 s interval would have sent three keep-alive
    // frames.
    let first = arrivals
        .first()
        .unwrap_or_else(|| {
            panic!(
                "no frame arrived within the budget (eof {}): the engine's frames were due at the \
                 end of its {SILENCE:?} silence",
                read.eof
            )
        })
        .at;
    assert!(
        first + read.slack + WHEEL >= SILENCE,
        "the engine's first frame at {first:?}, before the end of its {SILENCE:?} silence (slack \
         {:?}): {arrivals:#?}",
        read.slack
    );
    assert!(
        read.eof,
        "the stream ends as before and the body closes: {arrivals:#?}"
    );
    server.abort();
}
