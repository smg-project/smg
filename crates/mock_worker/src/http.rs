//! Mock HTTP worker endpoints — the vLLM/SGLang-compatible surface the SMG
//! gateway probes and routes to.
//!
//! In canned mode every response is fixed (no model). In realistic mode each
//! worker is backed by the engine simulator ([`crate::engine`]); since the HTTP
//! path carries text rather than token ids, the prompt is approximated into
//! synthetic token ids (one per whitespace word) so shared text prefixes still
//! produce cache hits and prompt length still drives prefill latency. Token-id
//! KV events (event-driven `cache_aware`) remain a gRPC-path feature.

use std::{
    convert::Infallible,
    io,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use axum::{
    body::Bytes,
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    serve::ListenerExt,
    Json, Router,
};
use futures::{stream, Stream};
use serde_json::{json, Value};
use tokio::{net::TcpListener, sync::mpsc};

use crate::{
    config::Config,
    engine::{self, Engine, NewRequest},
};

/// Per-listener HTTP state: shared config plus an optional engine simulator.
struct AppState {
    cfg: Arc<Config>,
    engine: Option<Engine>,
}

/// Build the router serving the mock HTTP worker contract.
fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/generate", post(chat_completions))
        .route("/v1/loads", get(loads))
        .with_state(state)
}

/// Serve the mock HTTP worker contract on `port` until the process exits.
pub async fn serve(cfg: Arc<Config>, host: String, port: u16) {
    let listener = match TcpListener::bind((host.as_str(), port)).await {
        Ok(listener) => listener,
        Err(e) => {
            tracing::error!("http worker bind {host}:{port} failed: {e}");
            return;
        }
    };
    // One simulated engine per listener (i.e. per virtual worker), registered
    // in the process fleet under `http:<port>` for the oracle and admin API.
    let engine = cfg
        .realistic
        .then(|| Engine::spawn_named(cfg.engine.clone(), format!("http:{port}"), true));
    let state = Arc::new(AppState { cfg, engine });
    // TCP_NODELAY: without it Nagle holds each small SSE frame until the
    // gateway's delayed ACK (~40ms) arrives, which stalls every streamed
    // response on a pooled keep-alive connection and distorts ITL/CPU numbers.
    let listener = listener.tap_io(move |tcp| {
        if let Err(e) = tcp.set_nodelay(true) {
            tracing::warn!("http worker {port}: set_nodelay failed: {e}");
        }
    });
    if let Err(e) = axum::serve(listener, router(state)).await {
        tracing::error!("http worker {port} stopped: {e}");
    }
}

async fn health() -> &'static str {
    "OK"
}

async fn models(State(state): State<Arc<AppState>>) -> Response {
    Json(json!({
        "object": "list",
        "data": [{
            "id": state.cfg.model_id,
            "object": "model",
            "created": 0,
            "owned_by": "sglang",
            "root": state.cfg.model_id,
            "max_model_len": state.cfg.context_length,
        }],
    }))
    .into_response()
}

async fn loads(State(state): State<Arc<AppState>>) -> Response {
    let load = state
        .engine
        .as_ref()
        .map(|e| e.load().as_reported_by(state.cfg.loads_like));
    let value = match load {
        Some(s) => json!({
            "dp_rank": 0,
            "num_running_reqs": s.num_running_reqs,
            "num_waiting_reqs": s.num_waiting_reqs,
            "num_waiting_uncached_tokens": s.num_waiting_uncached_tokens,
            "num_total_reqs": s.num_running_reqs + s.num_waiting_reqs,
            "num_used_tokens": s.num_used_tokens,
            "max_total_num_tokens": s.max_total_num_tokens,
            "token_usage": s.token_usage,
            "gen_throughput": s.gen_throughput,
            "cache_hit_rate": s.cache_hit_rate,
            "utilization": s.token_usage,
            "max_running_requests": s.max_running_requests,
        }),
        None => json!({
            "dp_rank": 0,
            "num_running_reqs": 0,
            "num_waiting_reqs": 0,
            "num_waiting_uncached_tokens": 0,
            "num_total_reqs": 0,
            "num_used_tokens": 0,
            "max_total_num_tokens": 1_000_000,
            "token_usage": 0.0,
            "gen_throughput": 0.0,
            "cache_hit_rate": 0.0,
            "utilization": 0.0,
            "max_running_requests": 0,
        }),
    };
    Json(json!({ "timestamp": "", "dp_rank_count": 1, "loads": [value] })).into_response()
}

/// OpenAI response shape this worker replies in. Chat emits
/// `choices[].message` / `choices[].delta.content`; completions emits
/// `choices[].text`. The load generator parses the two differently, so
/// `/v1/completions` must not be answered with chat-shaped frames.
#[derive(Clone, Copy)]
enum Endpoint {
    Chat,
    Completions,
}

async fn chat_completions(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    handle(Endpoint::Chat, state, body).await
}

async fn completions(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    handle(Endpoint::Completions, state, body).await
}

async fn handle(endpoint: Endpoint, state: Arc<AppState>, body: Bytes) -> Response {
    let parsed: Option<Value> = serde_json::from_slice(&body).ok();
    let stream_requested = parsed
        .as_ref()
        .and_then(|v| v.get("stream").and_then(Value::as_bool))
        .unwrap_or(false);

    // Realistic mode: drive the engine simulator.
    if let Some(engine) = &state.engine {
        // The fault hook's answer for this request, when one is armed: a
        // stall, a status before admission, or a cut after some output.
        let injected = engine.inject();
        if let Some(fault) = injected {
            if !fault.stall.is_zero() {
                tokio::time::sleep(fault.stall).await;
            }
            if fault.status != 0 && fault.after_tokens.is_none() {
                engine.record_failure();
                return injected_error(fault.status);
            }
        }
        let cut = injected.and_then(|fault| {
            fault
                .after_tokens
                .map(|after| (after, fault.status, engine.cut_counter()))
        });
        let parsed = parsed.unwrap_or(Value::Null);
        let prompt_ids = synth_token_ids(&extract_prompt_text(&parsed));
        let prompt_tokens = prompt_ids.len() as u32;
        let max_new = extract_max_tokens(&parsed).unwrap_or(state.cfg.output_tokens);
        let request_id = next_request_id();
        let (tx, rx) = mpsc::unbounded_channel();
        engine.submit(NewRequest {
            request_id,
            prompt_token_ids: prompt_ids,
            max_new,
            events: tx,
        });
        let model = state.cfg.model_id.clone();
        return if stream_requested {
            realistic_sse(rx, model, endpoint, cut).into_response()
        } else {
            realistic_completion(rx, model, prompt_tokens, endpoint, cut).await
        };
    }

    // Canned mode: a single up-front delay, then a fixed chat-shaped response,
    // whichever endpoint was called.
    if !state.cfg.gen_delay.is_zero() {
        tokio::time::sleep(state.cfg.gen_delay).await;
    }
    if stream_requested {
        stream_chat(&state.cfg).into_response()
    } else {
        Json(completion(&state.cfg)).into_response()
    }
}

// ── Canned responses ───────────────────────────────────────────────────────

fn completion(cfg: &Config) -> Value {
    json!({
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": 0,
        "model": cfg.model_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "mock"},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": cfg.output_tokens,
            "total_tokens": u64::from(cfg.output_tokens) + 1,
        },
    })
}

fn stream_chat(cfg: &Config) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let mut events: Vec<Result<Event, Infallible>> = Vec::new();
    for _ in 0..cfg.output_tokens {
        let frame = json!({
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": null}],
        });
        events.push(Ok(Event::default().data(frame.to_string())));
    }
    let final_frame = json!({
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    });
    events.push(Ok(Event::default().data(final_frame.to_string())));
    events.push(Ok(Event::default().data("[DONE]")));
    Sse::new(stream::iter(events))
}

// ── Realistic responses ─────────────────────────────────────────────────────

/// The fault hook's answer: the status (an error status; anything else
/// becomes 500) with an OpenAI-shaped error body.
fn injected_error(status: u16) -> Response {
    let code = StatusCode::from_u16(status)
        .ok()
        .filter(|code| code.is_client_error() || code.is_server_error())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    (
        code,
        Json(json!({"error": {"message": "injected", "type": "fault"}})),
    )
        .into_response()
}

/// Non-streaming: drain the engine's events and assemble one completion JSON.
/// With `cut`, the answer is that status instead once the engine has produced
/// more than the given number of tokens (the fault hook's `after_tokens`).
async fn realistic_completion(
    mut rx: mpsc::UnboundedReceiver<engine::GenEvent>,
    model: String,
    prompt_tokens: u32,
    endpoint: Endpoint,
    cut: Option<(u32, u16, Arc<AtomicU64>)>,
) -> Response {
    let mut completion_tokens = 0u32;
    let mut cached_tokens = 0u32;
    while let Some(ev) = rx.recv().await {
        match ev {
            engine::GenEvent::Token { .. } => {
                completion_tokens += 1;
                if let Some((after, status, cut_total)) = &cut {
                    if completion_tokens > *after {
                        cut_total.fetch_add(1, Ordering::Relaxed);
                        return injected_error(*status);
                    }
                }
            }
            engine::GenEvent::Done {
                completion_tokens: c,
                cached_tokens: cached,
                ..
            } => {
                completion_tokens = c;
                cached_tokens = cached;
            }
        }
    }
    let usage = json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cached_tokens": cached_tokens,
        "prompt_tokens_details": { "cached_tokens": cached_tokens },
    });
    let body = match endpoint {
        Endpoint::Chat => json!({
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "mock"},
                "finish_reason": "stop",
            }],
            "usage": usage,
        }),
        Endpoint::Completions => json!({
            "id": "cmpl-mock",
            "object": "text_completion",
            "created": 0,
            "model": model,
            "choices": [{
                "index": 0,
                "text": "mock",
                "finish_reason": "stop",
            }],
            "usage": usage,
        }),
    };
    Json(body).into_response()
}

/// Streaming: map the engine's events to SSE chunks, ending with a finish frame
/// and `[DONE]`. Frame shape follows `endpoint` (chat delta vs completion text).
/// With `cut`, the body ends in an error instead (no finish frame, no `[DONE]`:
/// the connection is cut) once the engine has produced more than the given
/// number of tokens (the fault hook's `after_tokens`).
fn realistic_sse(
    rx: mpsc::UnboundedReceiver<engine::GenEvent>,
    model: String,
    endpoint: Endpoint,
    cut: Option<(u32, u16, Arc<AtomicU64>)>,
) -> Sse<impl Stream<Item = Result<Event, io::Error>>> {
    Sse::new(realistic_frames(rx, model, endpoint, cut))
}

/// The frames of [`realistic_sse`].
fn realistic_frames(
    rx: mpsc::UnboundedReceiver<engine::GenEvent>,
    model: String,
    endpoint: Endpoint,
    cut: Option<(u32, u16, Arc<AtomicU64>)>,
) -> impl Stream<Item = Result<Event, io::Error>> {
    enum St {
        Active {
            rx: mpsc::UnboundedReceiver<engine::GenEvent>,
            model: String,
            endpoint: Endpoint,
            served: u32,
            cut: Option<(u32, u16, Arc<AtomicU64>)>,
        },
        Closing,
        Ended,
    }

    stream::unfold(
        St::Active {
            rx,
            model,
            endpoint,
            served: 0,
            cut,
        },
        |st| async move {
            match st {
                St::Active {
                    mut rx,
                    model,
                    endpoint,
                    served,
                    cut,
                } => match rx.recv().await {
                    Some(engine::GenEvent::Token { .. }) => {
                        if let Some((after, status, cut_total)) = &cut {
                            if served >= *after {
                                cut_total.fetch_add(1, Ordering::Relaxed);
                                let cut = io::Error::other(format!("injected {status}"));
                                return Some((Err(cut), St::Ended));
                            }
                        }
                        let frame = token_chunk(endpoint, &model);
                        Some((
                            Ok(Event::default().data(frame.to_string())),
                            St::Active {
                                rx,
                                model,
                                endpoint,
                                served: served + 1,
                                cut,
                            },
                        ))
                    }
                    Some(engine::GenEvent::Done { .. }) => {
                        let frame = final_chunk(endpoint, &model);
                        Some((Ok(Event::default().data(frame.to_string())), St::Closing))
                    }
                    None => None,
                },
                St::Closing => Some((Ok(Event::default().data("[DONE]")), St::Ended)),
                St::Ended => None,
            }
        },
    )
}

/// One streamed token frame in the shape the requesting endpoint expects.
fn token_chunk(endpoint: Endpoint, model: &str) -> Value {
    match endpoint {
        Endpoint::Chat => json!({
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": null}],
        }),
        Endpoint::Completions => json!({
            "id": "cmpl-mock",
            "object": "text_completion",
            "model": model,
            "choices": [{"index": 0, "text": "x", "finish_reason": null}],
        }),
    }
}

/// The terminal frame (`finish_reason = stop`) for the requesting endpoint.
fn final_chunk(endpoint: Endpoint, model: &str) -> Value {
    match endpoint {
        Endpoint::Chat => json!({
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }),
        Endpoint::Completions => json!({
            "id": "cmpl-mock",
            "object": "text_completion",
            "model": model,
            "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
        }),
    }
}

// ── HTTP prompt helpers ──────────────────────────────────────────────────────

/// Extract prompt text from a chat/completions/generate body.
fn extract_prompt_text(v: &Value) -> String {
    if let Some(messages) = v.get("messages").and_then(Value::as_array) {
        return messages
            .iter()
            .filter_map(|m| m.get("content").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(" ");
    }
    for key in ["prompt", "text", "inputs"] {
        if let Some(s) = v.get(key).and_then(Value::as_str) {
            return s.to_string();
        }
    }
    String::new()
}

/// Approximate a prompt's token ids from its text: one id per whitespace word,
/// each a stable hash of the word. Identical leading words yield identical
/// leading ids, so shared prefixes still produce cache hits.
fn synth_token_ids(text: &str) -> Vec<u32> {
    text.split_whitespace().map(hash_word).collect()
}

fn hash_word(w: &str) -> u32 {
    let mut h: u32 = 2_166_136_261;
    for b in w.bytes() {
        h ^= u32::from(b);
        h = h.wrapping_mul(16_777_619);
    }
    h % 30_000
}

fn extract_max_tokens(v: &Value) -> Option<u32> {
    for key in ["max_tokens", "max_new_tokens"] {
        if let Some(n) = v.get(key).and_then(Value::as_u64) {
            return Some(n as u32);
        }
    }
    None
}

fn next_request_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    format!("mock-http-{}", COUNTER.fetch_add(1, Ordering::Relaxed))
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use super::*;

    fn token() -> engine::GenEvent {
        engine::GenEvent::Token {
            token_id: 7,
            prompt_tokens: 3,
            cached_tokens: 0,
        }
    }

    fn done(tokens: u32) -> engine::GenEvent {
        engine::GenEvent::Done {
            finish_reason: "stop",
            prompt_tokens: 3,
            completion_tokens: tokens,
            cached_tokens: 0,
        }
    }

    #[tokio::test]
    async fn a_cut_stream_ends_in_an_error_after_the_given_frames() {
        let (tx, rx) = mpsc::unbounded_channel();
        for _ in 0..3 {
            tx.send(token()).unwrap();
        }
        tx.send(done(3)).unwrap();
        drop(tx);
        let cuts = Arc::new(AtomicU64::new(0));
        let frames: Vec<Result<Event, io::Error>> = realistic_frames(
            rx,
            "m".to_string(),
            Endpoint::Chat,
            Some((1, 502, Arc::clone(&cuts))),
        )
        .collect()
        .await;
        assert_eq!(frames.len(), 2, "one token frame, then the cut");
        assert_eq!(cuts.load(Ordering::Relaxed), 1, "the cut is counted once");
        assert!(frames[0].is_ok());
        assert!(
            matches!(&frames[1], Err(e) if e.to_string() == "injected 502"),
            "{:?}",
            frames[1]
        );
    }

    #[tokio::test]
    async fn an_uncut_stream_still_finishes_with_done() {
        let (tx, rx) = mpsc::unbounded_channel();
        tx.send(token()).unwrap();
        tx.send(done(1)).unwrap();
        drop(tx);
        let cuts = Arc::new(AtomicU64::new(0));
        let frames: Vec<Result<Event, io::Error>> = realistic_frames(
            rx,
            "m".to_string(),
            Endpoint::Chat,
            Some((5, 502, Arc::clone(&cuts))),
        )
        .collect()
        .await;
        assert_eq!(frames.len(), 3, "token, finish, [DONE]");
        assert!(frames.iter().all(Result::is_ok));
        assert_eq!(cuts.load(Ordering::Relaxed), 0, "nothing was cut");
    }

    #[tokio::test]
    async fn a_cut_completion_answers_the_status_and_counts_the_cut() {
        let (tx, rx) = mpsc::unbounded_channel();
        for _ in 0..3 {
            tx.send(token()).unwrap();
        }
        tx.send(done(3)).unwrap();
        drop(tx);
        let cuts = Arc::new(AtomicU64::new(0));
        let response = realistic_completion(
            rx,
            "m".to_string(),
            3,
            Endpoint::Chat,
            Some((1, 502, Arc::clone(&cuts))),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(cuts.load(Ordering::Relaxed), 1, "the cut is counted once");
    }

    #[tokio::test]
    async fn an_injected_error_carries_the_status_and_an_error_body() {
        let response = injected_error(503);
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["error"]["type"], "fault");
        assert_eq!(body["error"]["message"], "injected");
        assert_eq!(
            injected_error(999).status(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
        assert_eq!(
            injected_error(200).status(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
        assert_eq!(injected_error(429).status(), StatusCode::TOO_MANY_REQUESTS);
    }
}
