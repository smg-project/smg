//! Mock HTTP worker endpoints — the vLLM/SGLang-compatible surface the SMG
//! gateway probes and routes to.
//!
//! In canned mode every response is fixed (no model). In realistic mode each
//! worker is backed by the engine simulator ([`crate::engine`]); since the HTTP
//! path carries text rather than token ids, the prompt is approximated into
//! synthetic token ids (one per whitespace word) so shared text prefixes still
//! produce cache hits and prompt length still drives prefill latency. Token-id
//! KV events (event-driven `cache_aware`) remain a gRPC-path feature.
//!
//! `/generate` is SGLang-native in realistic mode: it reads `input_ids` when
//! the body carries them (text is the fallback), and answers in the native
//! shape (`output_ids` + `meta_info`), so a load generator that speaks the
//! production `/generate` contract scores cached tokens and rebuilds
//! multi-turn context from the worker's own output. In canned mode it stays
//! the historical chat-shaped alias the existing rigs depend on.

use std::{
    convert::Infallible,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use axum::{
    body::Bytes,
    extract::State,
    response::{
        sse::{Event, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
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
pub struct AppState {
    cfg: Arc<Config>,
    engine: Option<Engine>,
    /// The listener's port, echoed in native `meta_info.worker_port` so a
    /// client can attribute a response to a worker without trusting routing.
    port: u16,
}

/// Build the router serving the mock HTTP worker contract.
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/generate", post(generate))
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
    // One simulated engine per listener (i.e. per virtual worker).
    let engine = cfg.realistic.then(|| Engine::spawn(cfg.engine.clone()));
    let state = Arc::new(AppState { cfg, engine, port });
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
            "max_model_len": 32768,
        }],
    }))
    .into_response()
}

async fn loads(State(state): State<Arc<AppState>>) -> Response {
    let load = state.engine.as_ref().map(|e| e.load());
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

/// `/generate`: SGLang-native in realistic mode, chat-shaped alias otherwise.
async fn generate(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    if state.engine.is_none() {
        return handle(Endpoint::Chat, state, body).await;
    }
    let parsed: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    drop(body);
    let stream_requested = parsed
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let prompt_ids = extract_input_ids(&parsed)
        .unwrap_or_else(|| synth_token_ids(&extract_prompt_text(&parsed)));
    let max_new = extract_native_max_new(&parsed).unwrap_or(state.cfg.output_tokens);
    drop(parsed);
    let request_id = next_request_id();
    let (tx, rx) = mpsc::unbounded_channel();
    state
        .engine
        .as_ref()
        .expect("checked above")
        .submit(NewRequest {
            request_id: request_id.clone(),
            prompt_token_ids: prompt_ids,
            max_new,
            events: tx,
        });
    if stream_requested {
        native_sse(rx, request_id, state.port).into_response()
    } else {
        Json(native_completion(rx, request_id, state.port).await).into_response()
    }
}

/// Everything a native response reports, accumulated from the engine's events.
struct NativeProgress {
    request_id: String,
    worker_port: u16,
    prompt_tokens: u32,
    cached_tokens: u32,
    output_ids: Vec<u32>,
    finished: bool,
}

impl NativeProgress {
    fn new(request_id: String, worker_port: u16) -> Self {
        Self {
            request_id,
            worker_port,
            prompt_tokens: 0,
            cached_tokens: 0,
            output_ids: Vec::new(),
            finished: false,
        }
    }

    fn absorb(&mut self, ev: engine::GenEvent) {
        match ev {
            engine::GenEvent::Token {
                token_id,
                prompt_tokens,
                cached_tokens,
            } => {
                self.prompt_tokens = prompt_tokens;
                self.cached_tokens = cached_tokens;
                self.output_ids.push(token_id);
            }
            engine::GenEvent::Done {
                prompt_tokens,
                cached_tokens,
                ..
            } => {
                self.prompt_tokens = prompt_tokens;
                self.cached_tokens = cached_tokens;
                self.finished = true;
            }
        }
    }

    /// The SGLang-native frame: `output_ids` and completion accounting are
    /// reported once, on the terminal frame; earlier frames carry the prompt
    /// accounting only (first frame = time to first token).
    fn frame(&self) -> Value {
        let completion = if self.finished {
            self.output_ids.len()
        } else {
            0
        };
        json!({
            "text": if self.finished { "mock" } else { "" },
            "output_ids": if self.finished { Value::from(self.output_ids.clone()) } else { Value::from(Vec::<u32>::new()) },
            "meta_info": {
                "id": self.request_id,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": completion,
                "cached_tokens": self.cached_tokens,
                "finish_reason": if self.finished {
                    json!({"type": "length", "length": completion})
                } else {
                    Value::Null
                },
                "worker_port": self.worker_port,
            },
        })
    }
}

/// Non-streaming native `/generate`: one JSON object after the last token.
async fn native_completion(
    mut rx: mpsc::UnboundedReceiver<engine::GenEvent>,
    request_id: String,
    worker_port: u16,
) -> Value {
    let mut progress = NativeProgress::new(request_id, worker_port);
    while let Some(ev) = rx.recv().await {
        progress.absorb(ev);
    }
    progress.frame()
}

/// Streaming native `/generate`: a first frame at the first token (so the
/// client's TTFT is the engine's), the terminal frame with `output_ids` at
/// completion, then `[DONE]`. Intermediate tokens are accumulated, not
/// streamed one per frame: the production `/generate` clients this mock
/// serves read the first and last frame, and a frame per token through the
/// gateway would make the mock fleet's SSE volume the bottleneck.
fn native_sse(
    rx: mpsc::UnboundedReceiver<engine::GenEvent>,
    request_id: String,
    worker_port: u16,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    enum St {
        Active {
            rx: mpsc::UnboundedReceiver<engine::GenEvent>,
            progress: NativeProgress,
            first_sent: bool,
        },
        Closing,
        Ended,
    }

    let body = stream::unfold(
        St::Active {
            rx,
            progress: NativeProgress::new(request_id, worker_port),
            first_sent: false,
        },
        |st| async move {
            match st {
                St::Active {
                    mut rx,
                    mut progress,
                    mut first_sent,
                } => loop {
                    match rx.recv().await {
                        Some(ev) => {
                            progress.absorb(ev);
                            if progress.finished {
                                let frame = progress.frame();
                                return Some((
                                    Ok(Event::default().data(frame.to_string())),
                                    St::Closing,
                                ));
                            }
                            if !first_sent {
                                first_sent = true;
                                let frame = progress.frame();
                                return Some((
                                    Ok(Event::default().data(frame.to_string())),
                                    St::Active {
                                        rx,
                                        progress,
                                        first_sent,
                                    },
                                ));
                            }
                        }
                        // Engine dropped the request without a terminal
                        // event: end the stream so the client sees an
                        // incomplete response, not a hang.
                        None => return None,
                    }
                },
                St::Closing => Some((Ok(Event::default().data("[DONE]")), St::Ended)),
                St::Ended => None,
            }
        },
    );
    Sse::new(body)
}

async fn handle(endpoint: Endpoint, state: Arc<AppState>, body: Bytes) -> Response {
    let parsed: Option<Value> = serde_json::from_slice(&body).ok();
    let stream_requested = parsed
        .as_ref()
        .and_then(|v| v.get("stream").and_then(Value::as_bool))
        .unwrap_or(false);

    // Realistic mode: drive the engine simulator.
    if let Some(engine) = &state.engine {
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
            realistic_sse(rx, model, endpoint).into_response()
        } else {
            realistic_completion(rx, model, prompt_tokens, endpoint)
                .await
                .into_response()
        };
    }

    // Canned mode: a single up-front delay, then a fixed response. Always
    // chat-shaped (unchanged) so the existing scale rig is unaffected.
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

/// Non-streaming: drain the engine's events and assemble one completion JSON.
async fn realistic_completion(
    mut rx: mpsc::UnboundedReceiver<engine::GenEvent>,
    model: String,
    prompt_tokens: u32,
    endpoint: Endpoint,
) -> Json<Value> {
    let mut completion_tokens = 0u32;
    let mut cached_tokens = 0u32;
    while let Some(ev) = rx.recv().await {
        match ev {
            engine::GenEvent::Token { .. } => completion_tokens += 1,
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
    });
    Json(match endpoint {
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
    })
}

/// Streaming: map the engine's events to SSE chunks, ending with a finish frame
/// and `[DONE]`. Frame shape follows `endpoint` (chat delta vs completion text).
fn realistic_sse(
    rx: mpsc::UnboundedReceiver<engine::GenEvent>,
    model: String,
    endpoint: Endpoint,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    enum St {
        Active {
            rx: mpsc::UnboundedReceiver<engine::GenEvent>,
            model: String,
            endpoint: Endpoint,
        },
        Closing,
        Ended,
    }

    let body = stream::unfold(
        St::Active {
            rx,
            model,
            endpoint,
        },
        |st| async move {
            match st {
                St::Active {
                    mut rx,
                    model,
                    endpoint,
                } => match rx.recv().await {
                    Some(engine::GenEvent::Token { .. }) => {
                        let frame = token_chunk(endpoint, &model);
                        Some((
                            Ok(Event::default().data(frame.to_string())),
                            St::Active {
                                rx,
                                model,
                                endpoint,
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
    );
    Sse::new(body)
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

/// Native `input_ids`: a flat token list, or a batch of one (`[[...]]`).
fn extract_input_ids(v: &Value) -> Option<Vec<u32>> {
    let ids = v.get("input_ids")?.as_array()?;
    let seq = match ids.first() {
        Some(Value::Array(inner)) => inner,
        _ => ids,
    };
    Some(
        seq.iter()
            .filter_map(Value::as_u64)
            .map(|id| id as u32)
            .collect(),
    )
}

/// Native limit: `sampling_params.max_new_tokens` first, then the top-level
/// OpenAI-style keys.
fn extract_native_max_new(v: &Value) -> Option<u32> {
    v.get("sampling_params")
        .and_then(|sp| sp.get("max_new_tokens"))
        .and_then(Value::as_u64)
        .map(|n| n as u32)
        .or_else(|| extract_max_tokens(v))
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
    use super::*;

    #[test]
    fn native_input_ids_accepts_flat_and_batched_shapes() {
        let flat = json!({"input_ids": [1, 2, 3]});
        let batched = json!({"input_ids": [[4, 5]]});
        let text_only = json!({"text": "a b"});
        assert_eq!(extract_input_ids(&flat), Some(vec![1, 2, 3]));
        assert_eq!(extract_input_ids(&batched), Some(vec![4, 5]));
        assert_eq!(extract_input_ids(&text_only), None);
    }

    #[test]
    fn native_max_new_prefers_sampling_params() {
        let native = json!({"sampling_params": {"max_new_tokens": 7}, "max_tokens": 99});
        let openai = json!({"max_tokens": 99});
        assert_eq!(extract_native_max_new(&native), Some(7));
        assert_eq!(extract_native_max_new(&openai), Some(99));
        assert_eq!(extract_native_max_new(&json!({})), None);
    }

    #[test]
    fn native_frames_report_output_only_when_finished() {
        let mut p = NativeProgress::new("r1".into(), 9007);
        p.absorb(engine::GenEvent::Token {
            token_id: 11,
            prompt_tokens: 100,
            cached_tokens: 64,
        });
        let first = p.frame();
        assert_eq!(first["output_ids"].as_array().unwrap().len(), 0);
        assert_eq!(first["meta_info"]["completion_tokens"], 0);
        assert_eq!(first["meta_info"]["cached_tokens"], 64);
        assert!(first["meta_info"]["finish_reason"].is_null());
        p.absorb(engine::GenEvent::Token {
            token_id: 12,
            prompt_tokens: 100,
            cached_tokens: 64,
        });
        p.absorb(engine::GenEvent::Done {
            finish_reason: "length",
            prompt_tokens: 100,
            completion_tokens: 2,
            cached_tokens: 64,
        });
        let last = p.frame();
        assert_eq!(last["output_ids"], json!([11, 12]));
        assert_eq!(last["meta_info"]["completion_tokens"], 2);
        assert_eq!(last["meta_info"]["prompt_tokens"], 100);
        assert_eq!(last["meta_info"]["worker_port"], 9007);
        assert_eq!(last["meta_info"]["finish_reason"]["type"], "length");
    }
}
