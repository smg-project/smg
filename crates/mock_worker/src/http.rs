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
    future::{pending, Future},
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
    admin_ops::{AdminOps, ProfileOptions},
    config::Config,
    drain::{InFlight, WorkerInFlight},
    engine::{self, Engine, NewRequest},
    script::{self, Answer},
};

/// Per-listener HTTP state: shared config plus an optional engine simulator.
struct AppState {
    cfg: Arc<Config>,
    /// The worker's name in the process (`http:<port>`): what the admin API
    /// and the script store address it by.
    name: String,
    engine: Option<Engine>,
    /// The requests this worker is serving: what its load report counts as
    /// running when no engine simulation runs.
    in_flight: WorkerInFlight,
    /// The flush and profile operations the gateway fans out, and their counts.
    ops: Arc<AdminOps>,
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
        .route("/flush_cache", post(flush_cache))
        .route("/start_profile", post(start_profile))
        .route("/stop_profile", post(stop_profile))
        .with_state(state)
}

/// Serve the mock HTTP worker contract on `port` until the process exits.
pub async fn serve(cfg: Arc<Config>, host: String, port: u16) {
    serve_until(cfg, host, port, pending()).await;
}

/// Serve on `port` until `shutdown` resolves, then stop accepting connections
/// and let the requests in flight finish (the drain of a termination signal).
pub async fn serve_until(
    cfg: Arc<Config>,
    host: String,
    port: u16,
    shutdown: impl Future<Output = ()> + Send + 'static,
) {
    let listener = match TcpListener::bind((host.as_str(), port)).await {
        Ok(listener) => listener,
        Err(e) => {
            tracing::error!("http worker bind {host}:{port} failed: {e}");
            return;
        }
    };
    serve_with_listener_until(cfg, listener, shutdown).await;
}

/// Serve on an already-bound listener until the process exits. Callers that
/// need a free port bind port 0 and read it back instead of picking one and
/// binding it later.
pub async fn serve_with_listener(cfg: Arc<Config>, listener: TcpListener) {
    serve_with_listener_until(cfg, listener, pending()).await;
}

/// [`serve_with_listener`] with the shutdown of [`serve_until`].
pub async fn serve_with_listener_until(
    cfg: Arc<Config>,
    listener: TcpListener,
    shutdown: impl Future<Output = ()> + Send + 'static,
) {
    let port = listener.local_addr().map(|a| a.port()).unwrap_or(0);
    if let Err(e) = script::ensure_loaded(&cfg) {
        tracing::error!("http worker {port} cannot load script: {e}");
        return;
    }
    // One simulated engine per listener (i.e. per virtual worker), registered
    // in the process fleet under `http:<port>` for the oracle and admin API.
    let name = format!("http:{port}");
    script::register(&name);
    let ops = AdminOps::register(&name);
    let engine = cfg
        .realistic
        .then(|| Engine::spawn_named(cfg.engine.clone(), name.clone(), true));
    let state = Arc::new(AppState {
        cfg,
        name,
        engine,
        in_flight: WorkerInFlight::default(),
        ops,
    });
    // TCP_NODELAY: without it Nagle holds each small SSE frame until the
    // gateway's delayed ACK (~40ms) arrives, which stalls every streamed
    // response on a pooled keep-alive connection and distorts ITL/CPU numbers.
    let listener = listener.tap_io(move |tcp| {
        if let Err(e) = tcp.set_nodelay(true) {
            tracing::warn!("http worker {port}: set_nodelay failed: {e}");
        }
    });
    if let Err(e) = axum::serve(listener, router(state))
        .with_graceful_shutdown(shutdown)
        .await
    {
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
        None => {
            // Without an engine simulation the work in flight is the canned
            // and scripted answers whose responses are still open on this
            // worker (not the process's other workers): the running count a
            // load poll asks about, and nothing waits.
            let running = state.in_flight.get();
            json!({
                "dp_rank": 0,
                "num_running_reqs": running,
                "num_waiting_reqs": 0,
                "num_waiting_uncached_tokens": 0,
                "num_total_reqs": running,
                "num_used_tokens": 0,
                "max_total_num_tokens": 1_000_000,
                "token_usage": 0.0,
                "gen_throughput": 0.0,
                "cache_hit_rate": 0.0,
                "utilization": 0.0,
                "max_running_requests": 0,
            })
        }
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

/// `POST /flush_cache`: the engine's cache cleared at once (a canned worker
/// has none), answered 200 as the engines' servers answer it.
async fn flush_cache(State(state): State<Arc<AppState>>) -> Response {
    let message = state.ops.flush(state.engine.iter());
    Json(json!({ "status": "ok", "message": message })).into_response()
}

/// `POST /start_profile` with the profile options as its JSON body (an
/// empty body is the defaults); a run already in progress is a 409.
async fn start_profile(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    let options = if body.is_empty() {
        ProfileOptions::default()
    } else {
        match serde_json::from_slice::<ProfileOptions>(&body) {
            Ok(options) => options,
            Err(e) => {
                return (StatusCode::BAD_REQUEST, format!("profile options: {e}")).into_response()
            }
        }
    };
    profile_answer(state.ops.start_profile(options))
}

/// `POST /stop_profile`: the run's trace file written; no run is a 409.
async fn stop_profile(State(state): State<Arc<AppState>>) -> Response {
    profile_answer(state.ops.stop_profile())
}

fn profile_answer(result: Result<String, String>) -> Response {
    match result {
        Ok(message) => Json(json!({ "status": "ok", "message": message })).into_response(),
        Err(message) => (StatusCode::CONFLICT, message).into_response(),
    }
}

async fn handle(endpoint: Endpoint, state: Arc<AppState>, body: Bytes) -> Response {
    // In flight from here until the response is complete or its stream is
    // done, so a shutdown knows what it waits for and this worker's load
    // report counts it.
    let in_flight = InFlight::enter_on(&state.in_flight);
    let parsed: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    let stream_requested = parsed
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    // The prompt as this wire knows it: its text, and synthetic ids, one per
    // whitespace word, the realistic engine's prompt and the script's key
    // alike. The canned path does without them until something is scripted.
    let prompt_text = if script::active() || state.engine.is_some() {
        extract_prompt_text(&parsed)
    } else {
        String::new()
    };
    let prompt_ids = synth_token_ids(&prompt_text);

    // A scripted answer, ahead of either engine mode: the worker's queue, the
    // prompt's text or its key decides ([`crate::script`]).
    // One scripted answer per choice the request samples (`n`).
    match script::take_per_choice(
        &state.name,
        &prompt_ids,
        Some(&prompt_text),
        extract_n(&parsed),
    ) {
        Ok(Some(answers)) => {
            // The request's limit ends each answer as it would an engine's.
            let limit = extract_max_tokens(&parsed);
            let answers = answers
                .into_iter()
                .map(|answer| answer.cut_to(limit))
                .collect();
            return scripted_response(
                endpoint,
                &state.cfg,
                &parsed,
                answers,
                prompt_ids.len(),
                stream_requested,
                in_flight,
            )
            .await;
        }
        Ok(None) => {}
        // A precondition the script does not meet (the gRPC wire's
        // `FAILED_PRECONDITION`): a status the gateway does not retry, so the
        // reason reaches the test instead of a retry's canned choice.
        Err(why) => return (StatusCode::PRECONDITION_FAILED, why).into_response(),
    }

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
        // An HTTP worker has no engine-failure frame on its wire: an armed
        // `finish=error` cuts with a 500 here.
        let cut = injected.and_then(|fault| {
            let status = if fault.finish_error {
                500
            } else {
                fault.status
            };
            fault
                .after_tokens
                .map(|after| (after, status, engine.cut_counter()))
        });
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
            realistic_sse(rx, model, endpoint, cut, in_flight).into_response()
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
        stream_chat(&state.cfg, in_flight).into_response()
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

fn stream_chat(
    cfg: &Config,
    in_flight: InFlight,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
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
    Sse::new(in_flight.until_done(stream::iter(events)))
}

// ── Scripted responses ──────────────────────────────────────────────────────

/// Whether the request asked for logprobs: chat's `logprobs: true`, a
/// completion's `logprobs: <n>` (`0` included: the sampled token's logprob
/// with no alternatives, as the engines read it).
fn wants_logprobs(endpoint: Endpoint, request: &Value) -> bool {
    match (endpoint, request.get("logprobs")) {
        (Endpoint::Chat, Some(value)) => value.as_bool().unwrap_or(false),
        (Endpoint::Completions, Some(value)) => value.as_u64().is_some(),
        (_, None) => false,
    }
}

/// How many top alternatives the request asked for per token: chat's
/// `top_logprobs`, a completion's `logprobs: <n>` (the sampled token counts
/// as one of them).
fn wants_top(endpoint: Endpoint, request: &Value) -> usize {
    let key = match endpoint {
        Endpoint::Chat => "top_logprobs",
        Endpoint::Completions => "logprobs",
    };
    request
        .get(key)
        .and_then(Value::as_u64)
        .map_or(0, |n| usize::try_from(n).unwrap_or(usize::MAX))
}

/// The usage of a request's scripted answers: the prompt once, every
/// choice's completion, as an engine reports a request that samples `n` times.
fn scripted_usage(answers: &[Answer], prompt_len: usize) -> Value {
    let first = answers.first();
    let prompt_tokens = first.map_or(0, |answer| answer.prompt_tokens(prompt_len));
    let completion_tokens: u32 = answers.iter().map(Answer::completion_tokens).sum();
    json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": { "cached_tokens": first.map_or(0, |answer| answer.cached_tokens) },
    })
}

/// The logprobs of some text pieces in the endpoint's shape: chat's
/// `content` entries, a completion's parallel `tokens` and `token_logprobs`,
/// with the answer's `top` alternatives per piece where the request asked for
/// them (chat's entry list, a completion's piece-to-logprob map).
fn scripted_logprobs(
    endpoint: Endpoint,
    answer: &Answer,
    pieces: &[String],
    logprobs: &[f32],
    top: usize,
) -> Value {
    let mut alternatives = answer.top_pieces(pieces, logprobs, top).into_iter();
    match endpoint {
        Endpoint::Chat => json!({
            "content": pieces.iter().zip(logprobs).map(|(piece, logprob)| json!({
                "token": piece,
                "logprob": logprob,
                "bytes": Value::Null,
                "top_logprobs": alternatives.next().unwrap_or_default().into_iter().map(|(token, logprob)| json!({
                    "token": token,
                    "logprob": logprob,
                    "bytes": Value::Null,
                })).collect::<Vec<Value>>(),
            })).collect::<Vec<Value>>(),
        }),
        Endpoint::Completions => json!({
            "tokens": pieces,
            "token_logprobs": logprobs,
            "top_logprobs": pieces.iter().map(|_| {
                alternatives
                    .next()
                    .unwrap_or_default()
                    .into_iter()
                    .map(|(token, logprob)| (token, json!(logprob)))
                    .collect::<serde_json::Map<String, Value>>()
            }).collect::<Vec<serde_json::Map<String, Value>>>(),
            "text_offset": [],
        }),
    }
}

/// The scripted answer in the endpoint's shape: one JSON body after the
/// answer's pauses, or an SSE stream of one frame per cut of the text, the
/// finish frame, the usage chunk when the request asked for it in its
/// stream (a chunk of its own with no choices, as the engines send it) and
/// `[DONE]`, with the pause before every frame but the first. The finish
/// carries `matched_stop` when the answer names one.
async fn scripted_response(
    endpoint: Endpoint,
    cfg: &Config,
    request: &Value,
    answers: Vec<Answer>,
    prompt_len: usize,
    stream: bool,
    in_flight: InFlight,
) -> Response {
    let want_logprobs = wants_logprobs(endpoint, request);
    let top = if want_logprobs {
        wants_top(endpoint, request)
    } else {
        0
    };
    let usage = scripted_usage(&answers, prompt_len);
    let model = cfg.model_id.clone();
    if stream {
        let include_usage = request
            .pointer("/stream_options/include_usage")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let usage = include_usage.then_some(usage);
        return Sse::new(in_flight.until_done(scripted_frames(
            endpoint,
            model,
            answers,
            want_logprobs,
            top,
            usage,
        )))
        .into_response();
    }
    // The pauses the choices' frames would take one after the other, as one wait.
    let pauses: std::time::Duration = answers
        .iter()
        .map(|answer| {
            answer.delay
                * u32::try_from(answer.text_frames().len().saturating_sub(1)).unwrap_or(u32::MAX)
        })
        .sum();
    if !pauses.is_zero() {
        tokio::time::sleep(pauses).await;
    }
    let choices: Vec<Value> = answers
        .iter()
        .enumerate()
        .map(|(index, answer)| {
            let text = answer.text();
            let mut choice = match endpoint {
                Endpoint::Chat => json!({
                    "index": index,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": answer.finish_reason,
                }),
                Endpoint::Completions => json!({
                    "index": index,
                    "text": text,
                    "finish_reason": answer.finish_reason,
                }),
            };
            if let Some(stop) = &answer.matched_stop {
                choice["matched_stop"] = stop.to_json();
            }
            if let (true, Some(pieces), Some(logprobs)) =
                (want_logprobs, &answer.output_text, &answer.logprobs)
            {
                if pieces.len() == logprobs.len() {
                    choice["logprobs"] = scripted_logprobs(endpoint, answer, pieces, logprobs, top);
                }
            }
            choice
        })
        .collect();
    let body = match endpoint {
        Endpoint::Chat => json!({
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": choices,
            "usage": usage,
        }),
        Endpoint::Completions => json!({
            "id": "cmpl-mock",
            "object": "text_completion",
            "created": 0,
            "model": model,
            "choices": choices,
            "usage": usage,
        }),
    };
    Json(body).into_response()
}

/// The frames of a scripted stream, choice by choice in order: each choice's
/// text frames and its finish frame, then the usage chunk when asked for,
/// then `[DONE]`.
fn scripted_frames(
    endpoint: Endpoint,
    model: String,
    answers: Vec<Answer>,
    want_logprobs: bool,
    top: usize,
    usage: Option<Value>,
) -> impl Stream<Item = Result<Event, Infallible>> {
    struct Choice {
        index: usize,
        answer: Answer,
        frames: std::collections::VecDeque<script::Frame>,
        started: bool,
    }
    struct Scripted {
        endpoint: Endpoint,
        model: String,
        choices: std::collections::VecDeque<Choice>,
        want_logprobs: bool,
        top: usize,
        usage: Option<Value>,
        ended: bool,
    }
    let choices = answers
        .into_iter()
        .enumerate()
        .map(|(index, answer)| Choice {
            index,
            frames: answer.text_frames().into(),
            answer,
            started: false,
        })
        .collect();
    let init = Scripted {
        endpoint,
        model,
        choices,
        want_logprobs,
        top,
        usage,
        ended: false,
    };
    stream::unfold(init, |mut st| async move {
        if st.ended {
            return None;
        }
        if st
            .choices
            .front()
            .is_some_and(|choice| choice.frames.is_empty())
        {
            // Every frame of the choice went out: its finish frame, then the
            // next choice from its first frame.
            let choice = st.choices.pop_front()?;
            let mut finish = match st.endpoint {
                Endpoint::Chat => json!({
                    "index": choice.index, "delta": {}, "finish_reason": choice.answer.finish_reason
                }),
                Endpoint::Completions => json!({
                    "index": choice.index, "text": "", "finish_reason": choice.answer.finish_reason
                }),
            };
            if let Some(stop) = &choice.answer.matched_stop {
                finish["matched_stop"] = stop.to_json();
            }
            let frame = scripted_chunk(st.endpoint, &st.model, finish);
            return Some((Ok(Event::default().data(frame.to_string())), st));
        }
        let Some(choice) = st.choices.front_mut() else {
            // Every choice finished. The usage rides in a chunk of its own
            // after the finish frames, with no choices, as the engines send
            // it; then the end.
            if let Some(usage) = st.usage.take() {
                let mut chunk = scripted_chunk(st.endpoint, &st.model, Value::Null);
                chunk["choices"] = json!([]);
                chunk["usage"] = usage;
                return Some((Ok(Event::default().data(chunk.to_string())), st));
            }
            st.ended = true;
            return Some((Ok(Event::default().data("[DONE]")), st));
        };
        let frame = choice.frames.pop_front()?;
        if choice.started && !choice.answer.delay.is_zero() {
            tokio::time::sleep(choice.answer.delay).await;
        }
        let text = frame.text();
        let mut delta = match st.endpoint {
            // The first delta of a choice names the role, as an engine's does.
            Endpoint::Chat if !choice.started => json!({
                "index": choice.index,
                "delta": {"role": "assistant", "content": text},
                "finish_reason": Value::Null,
            }),
            Endpoint::Chat => json!({
                "index": choice.index, "delta": {"content": text}, "finish_reason": Value::Null
            }),
            Endpoint::Completions => json!({
                "index": choice.index, "text": text, "finish_reason": Value::Null
            }),
        };
        if st.want_logprobs
            && !frame.logprobs.is_empty()
            && frame.logprobs.len() == frame.pieces.len()
        {
            delta["logprobs"] = scripted_logprobs(
                st.endpoint,
                &choice.answer,
                &frame.pieces,
                &frame.logprobs,
                st.top,
            );
        }
        choice.started = true;
        let frame = scripted_chunk(st.endpoint, &st.model, delta);
        Some((Ok(Event::default().data(frame.to_string())), st))
    })
}

/// One streamed frame of the endpoint's shape around `choice`.
fn scripted_chunk(endpoint: Endpoint, model: &str, choice: Value) -> Value {
    match endpoint {
        Endpoint::Chat => json!({
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [choice],
        }),
        Endpoint::Completions => json!({
            "id": "cmpl-mock",
            "object": "text_completion",
            "created": 0,
            "model": model,
            "choices": [choice],
        }),
    }
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
    in_flight: InFlight,
) -> Sse<impl Stream<Item = Result<Event, io::Error>>> {
    Sse::new(in_flight.until_done(realistic_frames(rx, model, endpoint, cut)))
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

/// Extract prompt text from a chat/completions/generate body: the messages'
/// content (a string, or the `text` of its parts), else the prompt (a
/// string, or an array of them), joined by spaces.
fn extract_prompt_text(v: &Value) -> String {
    if let Some(messages) = v.get("messages").and_then(Value::as_array) {
        return messages
            .iter()
            .filter_map(|m| m.get("content"))
            .flat_map(texts_of)
            .collect::<Vec<_>>()
            .join(" ");
    }
    for key in ["prompt", "text", "inputs"] {
        if let Some(value) = v.get(key) {
            let texts = texts_of(value);
            if !texts.is_empty() {
                return texts.join(" ");
            }
        }
    }
    String::new()
}

/// The texts of a content value: the string itself, or the `text` fields of
/// an array's parts (`{"type": "text", "text": ...}`) and its plain strings.
fn texts_of(value: &Value) -> Vec<&str> {
    match value {
        Value::String(s) => vec![s.as_str()],
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| {
                part.as_str()
                    .or_else(|| part.get("text").and_then(Value::as_str))
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// Approximate a prompt's token ids from its text: one id per whitespace word,
/// each a stable hash of the word. Identical leading words yield identical
/// leading ids, so shared prefixes still produce cache hits. These are also
/// the ids an HTTP worker's script key is computed over
/// ([`crate::script::script_key`]).
pub fn synth_token_ids(text: &str) -> Vec<u32> {
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

/// The choices a request samples: its `n`, one when absent or zero.
fn extract_n(v: &Value) -> usize {
    v.get("n")
        .and_then(Value::as_u64)
        .map_or(1, |n| usize::try_from(n).unwrap_or(usize::MAX))
        .max(1)
}

fn extract_max_tokens(v: &Value) -> Option<u32> {
    // The chat API's current name first, then the one it deprecates and the
    // engines' native one.
    for key in ["max_completion_tokens", "max_tokens", "max_new_tokens"] {
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

    #[test]
    fn the_prompt_text_is_read_from_string_content_text_parts_and_prompt_arrays() {
        use serde_json::json;
        assert_eq!(
            extract_prompt_text(&json!({"messages": [{"role": "user", "content": "say apricot"}]})),
            "say apricot"
        );
        assert_eq!(
            extract_prompt_text(&json!({"messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": [
                    {"type": "text", "text": "say"},
                    {"type": "image_url", "image_url": {"url": "data:,"}},
                    {"type": "text", "text": "apricot"},
                ]},
            ]})),
            "be brief say apricot"
        );
        assert_eq!(
            extract_prompt_text(&json!({"prompt": "say apricot"})),
            "say apricot"
        );
        assert_eq!(
            extract_prompt_text(&json!({"prompt": ["say", "apricot"]})),
            "say apricot"
        );
        assert_eq!(extract_prompt_text(&json!({"prompt": [1, 2, 3]})), "");
    }

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

    #[test]
    fn the_request_limit_is_read_under_each_of_its_names() {
        for key in ["max_completion_tokens", "max_tokens", "max_new_tokens"] {
            assert_eq!(extract_max_tokens(&json!({ key: 7 })), Some(7), "{key}");
        }
        assert_eq!(extract_max_tokens(&json!({ "stream": true })), None);
        assert_eq!(
            extract_max_tokens(&json!({ "max_completion_tokens": 3, "max_tokens": 9 })),
            Some(3),
            "the current name wins over the deprecated one"
        );
    }

    /// The admin fan-out's three routes on a canned worker, through their
    /// handlers: the flush is a 200, a profiling run ends with one trace file
    /// in the requested directory, a second start and a stop with no run are
    /// 409s, a body that is not the options is a 400.
    #[tokio::test]
    async fn the_admin_fan_out_routes_answer_on_a_canned_worker() {
        let state = Arc::new(AppState {
            cfg: Arc::new(Config::default()),
            name: "http:test".to_string(),
            engine: None,
            in_flight: WorkerInFlight::default(),
            ops: AdminOps::register("http:test"),
        });
        let start =
            |body: &str| start_profile(State(Arc::clone(&state)), Bytes::from(body.to_string()));
        let stop = || stop_profile(State(Arc::clone(&state)));

        assert_eq!(
            flush_cache(State(Arc::clone(&state))).await.status(),
            StatusCode::OK
        );

        let dir = tempfile::tempdir().expect("tempdir");
        let traces = dir.path().join("traces");
        let options =
            json!({ "output_dir": traces.display().to_string(), "num_steps": 2 }).to_string();
        assert_eq!(start(&options).await.status(), StatusCode::OK);
        assert_eq!(
            start(&options).await.status(),
            StatusCode::CONFLICT,
            "a run is in progress"
        );
        assert_eq!(stop().await.status(), StatusCode::OK);
        assert_eq!(
            std::fs::read_dir(&traces)
                .expect("the output directory")
                .count(),
            1
        );
        assert_eq!(
            stop().await.status(),
            StatusCode::CONFLICT,
            "no run in progress"
        );
        assert_eq!(start("{not json").await.status(), StatusCode::BAD_REQUEST);
        let ops = AdminOps::find("http:test").expect("the worker's operations");
        assert_eq!(ops.status()["flushes"], 1);
        assert_eq!(ops.status()["profiles_stopped"], 1);
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
