//! Mock HTTP worker endpoints — the vLLM/SGLang-compatible surface the SMG
//! gateway probes and routes to.
//!
//! In canned mode every response is fixed (no model), in the shape of the
//! route it answers. In realistic mode each worker is backed by the engine
//! simulator ([`crate::engine`]); since the HTTP path carries text rather
//! than token ids, the prompt is approximated into synthetic token ids (one
//! per whitespace word) so shared text prefixes still produce cache hits and
//! prompt length still drives prefill latency. Token-id KV events
//! (event-driven `cache_aware`) remain a gRPC-path feature.

use std::{
    convert::Infallible,
    future::{pending, Future},
    io,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
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
use futures::{stream, Stream, StreamExt};
use serde_json::{json, Value};
use tokio::{
    net::TcpListener,
    sync::{mpsc, watch},
};

use crate::{
    admin_ops::{AdminOps, ProfileOptions},
    config::{Config, Role},
    drain::{InFlight, WorkerInFlight},
    engine::{self, Engine, NewRequest},
    script::{self, Answer, MatchedStop},
};

/// The bootstrap port a scheduler takes without being told one, and the
/// one the gateway probes for a prefill worker registered without a port:
/// what a missing or null `bootstrap_port` in a request means.
const DEFAULT_BOOTSTRAP_PORT: u16 = 8998;

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
    /// The bootstrap port a prefill worker bound and advertises: where its
    /// decode partner's engine would rendezvous, the port the gateway tells
    /// both sides of (`None` on every other worker).
    bootstrap_port: Option<u16>,
}

/// Build the router serving the mock HTTP worker contract.
fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/generate", post(generate))
        .route("/v1/messages", post(messages))
        .route("/v1/loads", get(loads))
        .route("/server_info", get(server_info))
        .route("/get_server_info", get(server_info))
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
    // A prefill worker's bootstrap port, numbered by the worker's offset from
    // the base port as a gRPC worker's is: bound before the worker serves, as
    // an engine's bootstrap server is up when its health answers; a port that
    // cannot be bound stops the worker as its own port would.
    let bootstrap_port = port
        .checked_sub(cfg.http_base_port)
        .and_then(|index| cfg.bootstrap_port_for(index));
    let bootstrap = match crate::bootstrap::bind(&cfg.host, bootstrap_port).await {
        Ok(listener) => listener,
        Err(e) => {
            tracing::error!("http worker {port} {e}");
            return;
        }
    };
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
        bootstrap_port,
    });
    // TCP_NODELAY: without it Nagle holds each small SSE frame until the
    // gateway's delayed ACK (~40ms) arrives, which stalls every streamed
    // response on a pooled keep-alive connection and distorts ITL/CPU numbers.
    let listener = listener.tap_io(move |tcp| {
        if let Err(e) = tcp.set_nodelay(true) {
            tracing::warn!("http worker {port}: set_nodelay failed: {e}");
        }
    });
    // The bootstrap listener ends with the worker: the shutdown that stops
    // the server stops its accept loop.
    let (stop, stopped) = watch::channel(false);
    let shutdown = async move {
        shutdown.await;
        let _ = stop.send(true);
    };
    let bootstrap = async {
        if let Some(listener) = bootstrap {
            crate::bootstrap::accept_and_close(listener, stopped).await;
        }
    };
    let server = async {
        if let Err(e) = axum::serve(listener, router(state))
            .with_graceful_shutdown(shutdown)
            .await
        {
            tracing::error!("http worker {port} stopped: {e}");
        }
    };
    tokio::join!(server, bootstrap);
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

/// The completion routes as the request named them. The canned answer takes
/// the route's own shape; the scripted, realistic and prompt-side answers
/// take the shape of [`Route::endpoint`].
#[derive(Clone, Copy)]
enum Route {
    Chat,
    Completions,
    Generate,
}

impl Route {
    /// The shape of the scripted, realistic and prompt-side answers on the
    /// route: `/generate` takes the chat completion's until something reads
    /// the generate shape there (the gateway's generate tests drive the gRPC
    /// wire).
    fn endpoint(self) -> Endpoint {
        match self {
            Route::Chat | Route::Generate => Endpoint::Chat,
            Route::Completions => Endpoint::Completions,
        }
    }
}

async fn chat_completions(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    handle(Route::Chat, state, body).await
}

async fn completions(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    handle(Route::Completions, state, body).await
}

/// `POST /generate`, the engine-native route.
async fn generate(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    handle(Route::Generate, state, body).await
}

/// `POST /v1/messages`, the Messages API: the same answers as chat, in the
/// Messages shape ([`handle_messages`]).
async fn messages(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    handle_messages(state, body).await
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

/// `GET /server_info` (and its older spelling `/get_server_info`): what the
/// gateway's metadata read of an HTTP engine asks for. A disaggregated worker
/// reports its role and, on the prefill side, the port its partners
/// rendezvous on, as the engines' server info carries them; the mock moves
/// no KV cache and names the one transport the gateway pairs its workers on.
async fn server_info(State(state): State<Arc<AppState>>) -> Response {
    let mut info = json!({ "model_path": state.cfg.model_id });
    if let Some(role) = state.cfg.role {
        info["disaggregation_mode"] = json!(role.as_str());
        info["disaggregation_transfer_backend"] = json!(crate::grpc::KV_TRANSPORT);
        if let Some(port) = state.bootstrap_port {
            info["disaggregation_bootstrap_port"] = json!(port);
        }
    }
    Json(info).into_response()
}

/// What a disaggregated worker reads of the rendezvous the gateway injected
/// into the request: a `bootstrap_room` (one, or one per prompt of a batch)
/// on either side, as the engines' disaggregated servers require of every
/// request; on a prefill side that bound its port, a `bootstrap_port` naming
/// that port, the one its decode partner would come to for the handoff this
/// mock stands in for.
fn bootstrap_of(request: &Value, own_port: Option<u16>) -> Result<(), String> {
    match request.get("bootstrap_room") {
        Some(room) if !room.is_null() => {}
        _ => return Err("a disaggregated request carries no bootstrap_room".to_string()),
    }
    let Some(own) = own_port else {
        return Ok(());
    };
    // A missing or null port is the engines' default: a scheduler fills it
    // with its own bootstrap port (8998 unless told otherwise), and the
    // gateway, which sends null for a prefill worker registered without a
    // port, probes that default for it. A worker on another port refuses
    // it, as the pairing would send the decode side elsewhere.
    let names_own = |port: &Value| match port {
        Value::Null => own == DEFAULT_BOOTSTRAP_PORT,
        port => port.as_u64() == Some(u64::from(own)),
    };
    let named: &[Value] = match request.get("bootstrap_port") {
        Some(Value::Array(ports)) => ports,
        Some(port) => std::slice::from_ref(port),
        None => &[Value::Null],
    };
    if named.is_empty() || !named.iter().all(names_own) {
        return Err(format!(
            "bootstrap_port {} is not this worker's {own} (a missing port means {DEFAULT_BOOTSTRAP_PORT})",
            request.get("bootstrap_port").unwrap_or(&Value::Null)
        ));
    }
    Ok(())
}

/// A request this side cannot serve, refused as the engines' servers refuse
/// one: a 400 with an OpenAI-shaped error body naming the reason.
fn bad_request(why: String) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({"error": {"message": why, "type": "invalid_request_error"}})),
    )
        .into_response()
}

/// The prompt side's admission: the realistic engine's fault and admission
/// hooks as the regular path counts them ([`Engine::admit_without_tokens`]),
/// then the canned delay. The finish reason of the completion to answer
/// (`error` for an armed `finish=error`), or the status to answer instead.
async fn prompt_side_admission(state: &AppState) -> Result<&'static str, Response> {
    let mut finish_reason = "stop";
    if let Some(engine) = &state.engine {
        if engine
            .admit_without_tokens()
            .await
            .map_err(injected_error)?
        {
            finish_reason = "error";
        }
    }
    if !state.cfg.gen_delay.is_zero() {
        tokio::time::sleep(state.cfg.gen_delay).await;
    }
    Ok(finish_reason)
}

/// The prompt side's completion: no output, the prompt's tokens and the
/// finish reason, in the endpoint's shape; one body, or a stream of its
/// finish frame and `[DONE]` when the request asked for one (what the
/// gateway drains on the prefill side).
fn prompt_side_response(
    endpoint: Endpoint,
    model: &str,
    prompt_len: usize,
    finish_reason: &str,
    stream: bool,
    in_flight: InFlight,
) -> Response {
    if stream {
        let finish = match endpoint {
            Endpoint::Chat => json!({"index": 0, "delta": {}, "finish_reason": finish_reason}),
            Endpoint::Completions => {
                json!({"index": 0, "text": "", "finish_reason": finish_reason})
            }
        };
        let frames: [Result<Event, Infallible>; 2] = [
            Ok(Event::default().data(scripted_chunk(endpoint, model, finish).to_string())),
            Ok(Event::default().data("[DONE]")),
        ];
        return Sse::new(in_flight.until_done(stream::iter(frames))).into_response();
    }
    let usage = json!({
        "prompt_tokens": prompt_len,
        "completion_tokens": 0,
        "total_tokens": prompt_len,
    });
    let body = match endpoint {
        Endpoint::Chat => json!({
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": finish_reason,
            }],
            "usage": usage,
        }),
        Endpoint::Completions => json!({
            "id": "cmpl-mock",
            "object": "text_completion",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "text": "", "finish_reason": finish_reason}],
            "usage": usage,
        }),
    };
    Json(body).into_response()
}

/// A request as every completion route and the Messages route read it: the
/// body, whether it asked for a stream, and the prompt's text and synthetic
/// ids.
struct Incoming {
    parsed: Value,
    stream_requested: bool,
    prompt_text: String,
    prompt_ids: Vec<u32>,
}

fn incoming(state: &AppState, body: &[u8]) -> Result<Incoming, Response> {
    let parsed: Value = serde_json::from_slice(body).unwrap_or(Value::Null);
    // A disaggregated worker reads the rendezvous the gateway injected before
    // anything is spent on the request: a request without it went to the
    // wrong pool, or came from a gateway that is not pairing. The check is
    // pure, so an armed fault waits for a request this side serves.
    if state.cfg.role.is_some() {
        if let Err(why) = bootstrap_of(&parsed, state.bootstrap_port) {
            return Err(bad_request(why));
        }
    }
    let stream_requested = parsed
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    // The prompt as this wire knows it: its text, and synthetic ids, one per
    // whitespace word, the realistic engine's prompt and the script's key
    // alike (and the prompt side's count). The canned path does without them
    // until something is scripted.
    let prompt_text =
        if script::active() || state.engine.is_some() || state.cfg.role == Some(Role::Prefill) {
            extract_prompt_text(&parsed)
        } else {
            String::new()
        };
    let prompt_ids = synth_token_ids(&prompt_text);
    Ok(Incoming {
        parsed,
        stream_requested,
        prompt_text,
        prompt_ids,
    })
}

/// The engine simulator's side of a request: the answer channel, the cut an
/// armed fault takes after some output, and the prompt's token count.
struct Submitted {
    rx: mpsc::UnboundedReceiver<engine::GenEvent>,
    cut: Option<(u32, u16, Arc<AtomicU64>)>,
    prompt_tokens: u32,
}

/// Submit a request to the engine simulator: the fault hook's stall served
/// first, its status before admission answered instead (`Err`), its cut
/// after some output carried along for the answer to apply.
async fn submit_realistic(
    engine: &Engine,
    cfg: &Config,
    parsed: &Value,
    prompt_ids: Vec<u32>,
) -> Result<Submitted, Response> {
    // The fault hook's answer for this request, when one is armed: a
    // stall, a status before admission, or a cut after some output.
    let injected = engine.inject();
    if let Some(fault) = injected {
        if !fault.stall.is_zero() {
            tokio::time::sleep(fault.stall).await;
        }
        if fault.status != 0 && fault.after_tokens.is_none() {
            engine.record_failure();
            return Err(injected_error(fault.status));
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
    let prompt_tokens = u32::try_from(prompt_ids.len()).unwrap_or(u32::MAX);
    let max_new = extract_max_tokens(parsed).unwrap_or(cfg.output_tokens);
    let request_id = next_request_id();
    let (tx, rx) = mpsc::unbounded_channel();
    engine.submit(NewRequest {
        request_id,
        prompt_token_ids: prompt_ids,
        max_new,
        choice: 0,
        events: tx,
    });
    Ok(Submitted {
        rx,
        cut,
        prompt_tokens,
    })
}

async fn handle(route: Route, state: Arc<AppState>, body: Bytes) -> Response {
    let endpoint = route.endpoint();
    // In flight from here until the response is complete or its stream is
    // done, so a shutdown knows what it waits for and this worker's load
    // report counts it.
    let in_flight = InFlight::enter_on(&state.in_flight);
    let Incoming {
        parsed,
        stream_requested,
        prompt_text,
        prompt_ids,
    } = match incoming(&state, &body) {
        Ok(incoming) => incoming,
        Err(refused) => return refused,
    };

    // A prefill worker's answer: the prompt side's completion alone, ahead
    // of the script and of either engine mode. The decode side of the pair
    // has the prompt and generates the output, so nothing is scripted here
    // and nothing counts as a miss; the realistic engine's fault and
    // admission hooks apply as to any request, and the canned delay paces
    // it as it paces a canned answer.
    if state.cfg.role == Some(Role::Prefill) {
        let finish_reason = match prompt_side_admission(&state).await {
            Ok(finish_reason) => finish_reason,
            Err(refused) => return refused,
        };
        return prompt_side_response(
            endpoint,
            &state.cfg.model_id,
            prompt_ids.len(),
            finish_reason,
            stream_requested,
            in_flight,
        );
    }

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
        let Submitted {
            rx,
            cut,
            prompt_tokens,
        } = match submit_realistic(engine, &state.cfg, &parsed, prompt_ids).await {
            Ok(submitted) => submitted,
            Err(refused) => return refused,
        };
        let model = state.cfg.model_id.clone();
        return if stream_requested {
            realistic_sse(rx, model, endpoint, cut, in_flight).into_response()
        } else {
            realistic_completion(rx, model, prompt_tokens, endpoint, cut).await
        };
    }

    // Canned mode: a single up-front delay, then a fixed answer in the shape
    // of the route that was called.
    if !state.cfg.gen_delay.is_zero() {
        tokio::time::sleep(state.cfg.gen_delay).await;
    }
    if stream_requested {
        canned_stream(route, &state.cfg, in_flight).into_response()
    } else {
        Json(canned_answer(route, &state.cfg)).into_response()
    }
}

// ── Messages responses ─────────────────────────────────────────────────────

/// The Messages route's answer: the decisions of [`handle`] (the prompt side,
/// a scripted answer, the engine simulator, the canned answer) in the
/// Messages shape. One JSON message, or the event stream `message_start`,
/// `content_block_start`, one `content_block_delta` per text piece,
/// `content_block_stop`, `message_delta` with the stop reason and the output
/// tokens, then `message_stop`; no `[DONE]`. The stop reason comes from the
/// answer's finish reason ([`stop_reason`]); logprobs and `n` have no
/// Messages form, so one answer is taken and none are read.
async fn handle_messages(state: Arc<AppState>, body: Bytes) -> Response {
    let in_flight = InFlight::enter_on(&state.in_flight);
    let Incoming {
        parsed,
        stream_requested,
        prompt_text,
        prompt_ids,
    } = match incoming(&state, &body) {
        Ok(incoming) => incoming,
        Err(refused) => return refused,
    };
    let model = state.cfg.model_id.clone();
    let prompt_tokens = u32::try_from(prompt_ids.len()).unwrap_or(u32::MAX);

    if state.cfg.role == Some(Role::Prefill) {
        let finish_reason = match prompt_side_admission(&state).await {
            Ok(finish_reason) => finish_reason,
            Err(refused) => return refused,
        };
        let answer = MessageAnswer {
            text: String::new(),
            pieces: Vec::new(),
            finish_reason: finish_reason.to_string(),
            stop_sequence: None,
            input_tokens: prompt_tokens,
            output_tokens: 0,
            delay: Duration::ZERO,
        };
        return answer.response(&model, stream_requested, in_flight).await;
    }

    match script::take_per_choice(&state.name, &prompt_ids, Some(&prompt_text), 1) {
        Ok(Some(answers)) => {
            if let Some(answer) = answers.into_iter().next() {
                let answer = answer.cut_to(extract_max_tokens(&parsed));
                let answer = MessageAnswer::scripted(&answer, prompt_ids.len());
                return answer.response(&model, stream_requested, in_flight).await;
            }
        }
        Ok(None) => {}
        Err(why) => return (StatusCode::PRECONDITION_FAILED, why).into_response(),
    }

    // Realistic mode: drive the engine simulator.
    if let Some(engine) = &state.engine {
        let Submitted {
            rx,
            cut,
            prompt_tokens,
        } = match submit_realistic(engine, &state.cfg, &parsed, prompt_ids).await {
            Ok(submitted) => submitted,
            Err(refused) => return refused,
        };
        return if stream_requested {
            let frames = messages_realistic_frames(rx, model, prompt_tokens, cut);
            Sse::new(in_flight.until_done(frames)).into_response()
        } else {
            messages_realistic_body(rx, &model, prompt_tokens, cut).await
        };
    }

    if !state.cfg.gen_delay.is_zero() {
        tokio::time::sleep(state.cfg.gen_delay).await;
    }
    let answer = MessageAnswer {
        text: "mock".to_string(),
        pieces: vec![
            "x".to_string();
            usize::try_from(state.cfg.output_tokens).unwrap_or(usize::MAX)
        ],
        finish_reason: "stop".to_string(),
        stop_sequence: None,
        input_tokens: 1,
        output_tokens: state.cfg.output_tokens,
        delay: Duration::ZERO,
    };
    answer.response(&model, stream_requested, in_flight).await
}

/// One Messages answer, however it was decided: the text pieces its stream
/// carries as one delta each and the text its JSON body carries, the finish
/// reason and matched stop that become its stop reason, the token counts
/// its usage reports, and the pause between two deltas.
struct MessageAnswer {
    text: String,
    pieces: Vec<String>,
    finish_reason: String,
    stop_sequence: Option<Value>,
    input_tokens: u32,
    output_tokens: u32,
    delay: Duration,
}

impl MessageAnswer {
    /// A scripted answer in the Messages shape: its text frames as the deltas.
    fn scripted(answer: &Answer, prompt_len: usize) -> Self {
        MessageAnswer {
            text: answer.text(),
            pieces: answer
                .text_frames()
                .iter()
                .map(script::Frame::text)
                .collect(),
            finish_reason: answer.finish_reason.clone(),
            // A text stop is the sequence the Messages shape names; a token id
            // is no sequence, so that answer keeps its finish reason's stop reason.
            stop_sequence: match answer.matched_stop.as_ref() {
                Some(MatchedStop::Text(text)) => Some(json!(text)),
                Some(MatchedStop::Token(_)) | None => None,
            },
            input_tokens: answer.prompt_tokens(prompt_len),
            output_tokens: answer.completion_tokens(),
            delay: answer.delay,
        }
    }

    fn stop_reason(&self) -> &'static str {
        stop_reason(&self.finish_reason, self.stop_sequence.is_some())
    }

    /// The answer on the wire: the event stream, or one JSON body after the
    /// pauses its deltas would have taken.
    async fn response(self, model: &str, stream: bool, in_flight: InFlight) -> Response {
        // An HTTP worker has no engine-failure frame on its wire and the Messages
        // shape has no `error` stop reason: an `error` finish, armed or scripted,
        // answers a 500 here, as the completion routes' cut does.
        if self.finish_reason == "error" {
            return injected_error(500);
        }
        if stream {
            return Sse::new(in_flight.until_done(self.events(model))).into_response();
        }
        let pauses =
            self.delay * u32::try_from(self.pieces.len().saturating_sub(1)).unwrap_or(u32::MAX);
        if !pauses.is_zero() {
            tokio::time::sleep(pauses).await;
        }
        Json(self.json(model)).into_response()
    }

    fn json(&self, model: &str) -> Value {
        json!({
            "id": "msg-mock",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": self.text}],
            "stop_reason": self.stop_reason(),
            "stop_sequence": self.stop_sequence,
            "usage": {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens},
        })
    }

    /// The Messages event stream of the answer, one `content_block_delta`
    /// per piece with the pause before every piece but the first.
    fn events(self, model: &str) -> impl Stream<Item = Result<Event, Infallible>> {
        let prelude = [
            message_start_event(model, self.input_tokens),
            content_block_start_event(),
        ];
        let tail = [
            content_block_stop_event(),
            message_delta_event(self.stop_reason(), &self.stop_sequence, self.output_tokens),
            message_stop_event(),
        ];
        let delay = self.delay;
        let deltas = stream::iter(self.pieces.into_iter().enumerate()).then(
            move |(index, piece)| async move {
                if index > 0 && !delay.is_zero() {
                    tokio::time::sleep(delay).await;
                }
                text_delta_event(&piece)
            },
        );
        stream::iter(prelude)
            .chain(deltas)
            .chain(stream::iter(tail))
            .map(Ok::<Event, Infallible>)
    }
}

/// The Messages stop reason of a finish reason: `length` is `max_tokens`, a
/// tool call is `tool_use`, a matched stop is `stop_sequence`, and the end of
/// an answer is `end_turn`, whatever else the engine wire called it.
fn stop_reason(finish_reason: &str, stopped_on_sequence: bool) -> &'static str {
    if stopped_on_sequence {
        return "stop_sequence";
    }
    match finish_reason {
        "length" => "max_tokens",
        "tool_calls" => "tool_use",
        _ => "end_turn",
    }
}

fn message_start_event(model: &str, input_tokens: u32) -> Event {
    let message = json!({
        "type": "message_start",
        "message": {
            "id": "msg-mock",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": Value::Null,
            "stop_sequence": Value::Null,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        },
    });
    Event::default()
        .event("message_start")
        .data(message.to_string())
}

fn content_block_start_event() -> Event {
    let start = json!({
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    });
    Event::default()
        .event("content_block_start")
        .data(start.to_string())
}

fn text_delta_event(text: &str) -> Event {
    let delta = json!({
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": text},
    });
    Event::default()
        .event("content_block_delta")
        .data(delta.to_string())
}

fn content_block_stop_event() -> Event {
    let stop = json!({"type": "content_block_stop", "index": 0});
    Event::default()
        .event("content_block_stop")
        .data(stop.to_string())
}

fn message_delta_event(
    stop_reason: &str,
    stop_sequence: &Option<Value>,
    output_tokens: u32,
) -> Event {
    let delta = json!({
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence},
        "usage": {"output_tokens": output_tokens},
    });
    Event::default()
        .event("message_delta")
        .data(delta.to_string())
}

fn message_stop_event() -> Event {
    Event::default()
        .event("message_stop")
        .data(json!({"type": "message_stop"}).to_string())
}

/// The engine simulator's answer as one Messages body: the output counted
/// as it arrives (a cut armed after some output answers its status instead).
async fn messages_realistic_body(
    mut rx: mpsc::UnboundedReceiver<engine::GenEvent>,
    model: &str,
    prompt_tokens: u32,
    cut: Option<(u32, u16, Arc<AtomicU64>)>,
) -> Response {
    let mut completion_tokens = 0u32;
    let mut finish_reason = "stop";
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
                finish_reason: reason,
                ..
            } => {
                completion_tokens = c;
                finish_reason = reason;
            }
        }
    }
    let answer = MessageAnswer {
        text: "mock".to_string(),
        pieces: Vec::new(),
        finish_reason: finish_reason.to_string(),
        stop_sequence: None,
        input_tokens: prompt_tokens,
        output_tokens: completion_tokens,
        delay: Duration::ZERO,
    };
    Json(answer.json(model)).into_response()
}

/// The engine simulator's answer as the Messages event stream: one delta per
/// generated token, the stop events after the engine's `Done`, and a cut
/// armed after some output ending the stream with an error, as the chat
/// stream does.
fn messages_realistic_frames(
    rx: mpsc::UnboundedReceiver<engine::GenEvent>,
    model: String,
    prompt_tokens: u32,
    cut: Option<(u32, u16, Arc<AtomicU64>)>,
) -> impl Stream<Item = Result<Event, io::Error>> {
    enum St {
        Start {
            rx: mpsc::UnboundedReceiver<engine::GenEvent>,
            cut: Option<(u32, u16, Arc<AtomicU64>)>,
        },
        BlockStart {
            rx: mpsc::UnboundedReceiver<engine::GenEvent>,
            cut: Option<(u32, u16, Arc<AtomicU64>)>,
        },
        Active {
            rx: mpsc::UnboundedReceiver<engine::GenEvent>,
            cut: Option<(u32, u16, Arc<AtomicU64>)>,
            served: u32,
        },
        MessageDelta {
            served: u32,
            finish_reason: &'static str,
        },
        MessageStop,
        Ended,
    }
    stream::unfold(
        (St::Start { rx, cut }, model),
        move |(st, model)| async move {
            let (event, next) = match st {
                St::Start { rx, cut } => (
                    message_start_event(&model, prompt_tokens),
                    St::BlockStart { rx, cut },
                ),
                St::BlockStart { rx, cut } => (
                    content_block_start_event(),
                    St::Active { rx, cut, served: 0 },
                ),
                St::Active {
                    mut rx,
                    cut,
                    served,
                } => match rx.recv().await {
                    Some(engine::GenEvent::Token { .. }) => {
                        if let Some((after, status, cut_total)) = &cut {
                            if served >= *after {
                                cut_total.fetch_add(1, Ordering::Relaxed);
                                let cut = io::Error::other(format!("injected {status}"));
                                return Some((Err(cut), (St::Ended, model)));
                            }
                        }
                        (
                            text_delta_event("x"),
                            St::Active {
                                rx,
                                cut,
                                served: served + 1,
                            },
                        )
                    }
                    Some(engine::GenEvent::Done { finish_reason, .. }) => (
                        content_block_stop_event(),
                        St::MessageDelta {
                            served,
                            finish_reason,
                        },
                    ),
                    None => return None,
                },
                St::MessageDelta {
                    served,
                    finish_reason,
                } => (
                    message_delta_event(stop_reason(finish_reason, false), &None, served),
                    St::MessageStop,
                ),
                St::MessageStop => (message_stop_event(), St::Ended),
                St::Ended => return None,
            };
            Some((Ok(event), (next, model)))
        },
    )
}

// ── Canned responses ───────────────────────────────────────────────────────

/// The canned answer in the route's own shape: a chat completion, a text
/// completion, or the generate body (`text` and `meta_info`, as the engines
/// answer it).
fn canned_answer(route: Route, cfg: &Config) -> Value {
    let usage = json!({
        "prompt_tokens": 1,
        "completion_tokens": cfg.output_tokens,
        "total_tokens": u64::from(cfg.output_tokens) + 1,
    });
    match route {
        Route::Chat => json!({
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": 0,
            "model": cfg.model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "mock"},
                "finish_reason": "stop",
            }],
            "usage": usage,
        }),
        Route::Completions => json!({
            "id": "cmpl-mock",
            "object": "text_completion",
            "created": 0,
            "model": cfg.model_id,
            "choices": [{
                "index": 0,
                "text": "mock",
                "logprobs": null,
                "finish_reason": "stop",
            }],
            "usage": usage,
        }),
        Route::Generate => json!({
            "text": "mock",
            "meta_info": {
                "prompt_tokens": 1,
                "completion_tokens": cfg.output_tokens,
                "completion_tokens_wo_jump_forward": cfg.output_tokens,
                "input_token_logprobs": null,
                "output_token_logprobs": null,
                "first_token_latency": cfg.gen_delay.as_secs_f64(),
                "time_to_first_token": cfg.gen_delay.as_secs_f64(),
                "time_per_output_token": 0.0,
                "finish_reason": {"type": "stop", "reason": "stop"},
            },
        }),
    }
}

/// The canned stream in the route's own shape: one frame per output token
/// and the finish frame on the chat and text completion routes; one frame
/// with the whole answer on the generate route, as the engines' last frame
/// carries it. `[DONE]` ends each.
fn canned_stream(
    route: Route,
    cfg: &Config,
    in_flight: InFlight,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let mut events: Vec<Result<Event, Infallible>> = Vec::new();
    match route {
        Route::Generate => {
            events.push(Ok(
                Event::default().data(canned_answer(route, cfg).to_string())
            ));
        }
        Route::Chat | Route::Completions => {
            let endpoint = route.endpoint();
            for _ in 0..cfg.output_tokens {
                let frame = token_chunk(endpoint, &cfg.model_id);
                events.push(Ok(Event::default().data(frame.to_string())));
            }
            let frame = final_chunk(endpoint, &cfg.model_id);
            events.push(Ok(Event::default().data(frame.to_string())));
        }
    }
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

/// Extract prompt text from a chat/completions/generate/messages body: the
/// top-level `system` of a Messages request (a string, or text blocks) and
/// the messages' content (a string, or the `text` of its parts), else the
/// prompt (a string, or an array of them), joined by spaces.
fn extract_prompt_text(v: &Value) -> String {
    if let Some(messages) = v.get("messages").and_then(Value::as_array) {
        let system = v.get("system").map(texts_of).unwrap_or_default();
        return system
            .into_iter()
            .chain(
                messages
                    .iter()
                    .filter_map(|m| m.get("content"))
                    .flat_map(texts_of),
            )
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
        // A Messages request carries its system prompt as a top-level field.
        assert_eq!(
            extract_prompt_text(
                &json!({"system": "be brief", "messages": [{"role": "user", "content": "say apricot"}]})
            ),
            "be brief say apricot"
        );
        assert_eq!(
            extract_prompt_text(&json!({
                "system": [{"type": "text", "text": "be"}, {"type": "text", "text": "brief"}],
                "messages": [{"role": "user", "content": "say apricot"}],
            })),
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
            bootstrap_port: None,
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

    /// A canned worker of `role` for the handler-level tests: no engine, a
    /// name of its own in the script store, the bootstrap port it would have
    /// bound.
    fn state_of(role: Option<Role>, bootstrap_port: Option<u16>, name: &str) -> Arc<AppState> {
        script::register(name);
        Arc::new(AppState {
            cfg: Arc::new(Config {
                role,
                output_tokens: 2,
                ..Config::default()
            }),
            name: name.to_string(),
            engine: None,
            in_flight: WorkerInFlight::default(),
            ops: AdminOps::register(name),
            bootstrap_port,
        })
    }

    /// A chat request as the gateway's disaggregated dispatch sends it to
    /// both sides: the prompt, and the rendezvous it injected (the room and
    /// the prefill side's port).
    fn disaggregated_chat(room: Option<u64>, port: Option<Value>, stream: bool) -> Bytes {
        let mut body = json!({
            "messages": [{"role": "user", "content": "say apricot now"}],
            "stream": stream,
        });
        if let Some(room) = room {
            body["bootstrap_host"] = json!("127.0.0.1");
            body["bootstrap_room"] = json!(room);
        }
        if let Some(port) = port {
            body["bootstrap_port"] = port;
        }
        Bytes::from(body.to_string())
    }

    async fn text_of(response: Response) -> String {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("the body");
        String::from_utf8(bytes.to_vec()).expect("utf-8")
    }

    async fn json_of(response: Response) -> Value {
        serde_json::from_str(&text_of(response).await).expect("json")
    }

    /// The Messages stop reason follows the finish reason: a realistic answer
    /// that hit its token limit says `max_tokens` in the body and in the
    /// `message_delta`, a scripted text stop is the `stop_sequence`, and a
    /// scripted token-id stop is no sequence at all.
    #[tokio::test]
    async fn the_messages_stop_reason_follows_the_finish_reason() {
        let limit = || {
            let (tx, rx) = mpsc::unbounded_channel();
            tx.send(token()).unwrap();
            tx.send(engine::GenEvent::Done {
                finish_reason: "length",
                prompt_tokens: 3,
                completion_tokens: 1,
                cached_tokens: 0,
            })
            .unwrap();
            rx
        };
        let body = json_of(messages_realistic_body(limit(), "m", 3, None).await).await;
        assert_eq!(body["stop_reason"], "max_tokens", "{body}");
        let frames = text_of(
            Sse::new(messages_realistic_frames(limit(), "m".to_string(), 3, None)).into_response(),
        )
        .await;
        let delta: Value = frames
            .split("event: message_delta\n")
            .nth(1)
            .and_then(|rest| rest.lines().next())
            .and_then(|line| line.strip_prefix("data: "))
            .map(|data| serde_json::from_str(data).expect("delta json"))
            .expect("a message_delta frame");
        assert_eq!(delta["delta"]["stop_reason"], "max_tokens", "{frames}");
        // Scripted: a text stop is the stop sequence, a token id is none.
        let state = state_of(None, None, "http:messages-stops");
        let request = || {
            Bytes::from(
                json!({
                    "model": "m",
                    "max_tokens": 8,
                    "messages": [{"role": "user", "content": "say apricot"}],
                })
                .to_string(),
            )
        };
        let answer = |stop: Value| {
            script::Answer::parse(&json!({
                "output_text": "apricot",
                "finish_reason": "stop",
                "matched_stop": stop,
            }))
            .expect("an answer")
        };
        script::enqueue(
            "http:messages-stops",
            vec![answer(json!("</s>")), answer(json!(2))],
        )
        .expect("a known worker");
        let text_stop = json_of(handle_messages(Arc::clone(&state), request()).await).await;
        assert_eq!(text_stop["stop_reason"], "stop_sequence", "{text_stop}");
        assert_eq!(text_stop["stop_sequence"], "</s>");
        let token_stop = json_of(handle_messages(Arc::clone(&state), request()).await).await;
        assert_eq!(token_stop["stop_reason"], "end_turn", "{token_stop}");
        assert!(token_stop["stop_sequence"].is_null(), "{token_stop}");
        // An error finish, armed or scripted, answers a 500: the shape has no
        // error stop reason and the wire has no failure frame.
        script::enqueue(
            "http:messages-stops",
            vec![
                script::Answer::parse(&json!({"output_text": "x", "finish_reason": "error"}))
                    .expect("an answer"),
            ],
        )
        .expect("a known worker");
        let failed = handle_messages(Arc::clone(&state), request()).await;
        assert_eq!(failed.status(), 500);
    }

    /// The prompt side's completion in both shapes and as a stream, with
    /// the scripted answer left where the decode side takes it.
    #[tokio::test]
    async fn a_prefill_worker_answers_the_prompt_side_alone_and_leaves_the_script_to_decode() {
        let state = state_of(Some(Role::Prefill), Some(8998), "http:prefill-test");
        let scripted =
            script::Answer::parse(&json!({"output_text": "apricot"})).expect("an answer");
        script::enqueue("http:prefill-test", vec![scripted]).expect("a known worker");

        for stream in [false, true] {
            let response = handle(
                Route::Chat,
                Arc::clone(&state),
                disaggregated_chat(Some(7), Some(json!(8998)), stream),
            )
            .await;
            assert_eq!(response.status(), StatusCode::OK);
            let text = text_of(response).await;
            if stream {
                assert!(
                    text.contains(r#""finish_reason":"stop""#) && text.contains("[DONE]"),
                    "the finish frame and the end: {text}"
                );
                assert!(
                    !text.contains(r#""content":"x""#),
                    "no output on the prompt side: {text}"
                );
            } else {
                let body: Value = serde_json::from_str(&text).expect("json");
                assert_eq!(body["choices"][0]["message"]["content"], "");
                assert_eq!(body["choices"][0]["finish_reason"], "stop");
                assert_eq!(
                    (
                        body["usage"]["prompt_tokens"].as_u64(),
                        body["usage"]["completion_tokens"].as_u64()
                    ),
                    (Some(3), Some(0)),
                    "the prompt's words, no output: {body}"
                );
            }
        }
        let completion = json_of(
            handle(
                Route::Completions,
                Arc::clone(&state),
                Bytes::from(
                    json!({"prompt": "say apricot", "bootstrap_room": 1, "bootstrap_port": 8998})
                        .to_string(),
                ),
            )
            .await,
        )
        .await;
        assert_eq!(completion["object"], "text_completion");
        assert_eq!(completion["choices"][0]["text"], "");
        let status = script::status("http:prefill-test").expect("a known worker");
        assert_eq!(
            (
                status["queued"].as_u64(),
                status["hits"].as_u64(),
                status["misses"].as_u64()
            ),
            (Some(1), Some(0), Some(0)),
            "the scripted answer waits for the decode side, and no miss is counted: {status}"
        );
    }

    /// The Messages route's answers in the Messages shape: one message, or the
    /// event stream between `message_start` and `message_stop` with no `[DONE]`,
    /// for the canned, scripted and prompt-side answers.
    #[tokio::test]
    async fn the_messages_route_answers_in_the_messages_shape() {
        let state = state_of(None, None, "http:messages-test");
        let request = |stream: bool| {
            Bytes::from(
                json!({
                    "model": "m",
                    "max_tokens": 8,
                    "messages": [{"role": "user", "content": "say apricot"}],
                    "stream": stream,
                })
                .to_string(),
            )
        };
        let events_of = |frames: &str| -> Vec<String> {
            frames
                .lines()
                .filter_map(|line| line.strip_prefix("event: "))
                .map(str::to_string)
                .collect()
        };
        // Canned: one message with the mock text, one delta per output token.
        let canned = json_of(handle_messages(Arc::clone(&state), request(false)).await).await;
        assert_eq!(canned["type"], "message", "{canned}");
        assert_eq!(canned["content"][0]["text"], "mock");
        assert_eq!(canned["stop_reason"], "end_turn");
        assert_eq!(canned["usage"]["output_tokens"], 2);
        assert!(canned.get("choices").is_none(), "{canned}");
        let frames = text_of(handle_messages(Arc::clone(&state), request(true)).await).await;
        assert_eq!(
            events_of(&frames),
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ],
            "{frames}"
        );
        assert!(!frames.contains("[DONE]"), "{frames}");
        // Scripted: the answer's text in the body and as the deltas, its finish
        // reason as the stop reason, its usage.
        let parse = || {
            script::Answer::parse(&json!({"output_text": "apricot", "finish_reason": "length"}))
                .expect("an answer")
        };
        let deltas = parse().answer.text_frames().len();
        script::enqueue("http:messages-test", vec![parse(), parse()]).expect("a known worker");
        let body = json_of(handle_messages(Arc::clone(&state), request(false)).await).await;
        assert_eq!(body["content"][0]["text"], "apricot", "{body}");
        assert_eq!(body["stop_reason"], "max_tokens");
        assert_eq!(
            body["usage"]["input_tokens"], 2,
            "the prompt's two words: {body}"
        );
        assert!(
            body["usage"]["output_tokens"]
                .as_u64()
                .is_some_and(|n| n >= 1),
            "{body}"
        );
        let frames = text_of(handle_messages(Arc::clone(&state), request(true)).await).await;
        let events = events_of(&frames);
        assert_eq!(
            events
                .iter()
                .filter(|e| *e == "content_block_delta")
                .count(),
            deltas,
            "{frames}"
        );
        assert!(
            events.first().is_some_and(|e| e == "message_start")
                && events.last().is_some_and(|e| e == "message_stop")
                && frames.contains(r#""stop_reason":"max_tokens""#)
                && frames.contains("apricot"),
            "{frames}"
        );
        // The prompt side answers the message with no content and no delta.
        let prefill = state_of(
            Some(Role::Prefill),
            Some(8998),
            "http:messages-prefill-test",
        );
        let on_the_prompt_side = |stream: bool| {
            Bytes::from(
                json!({
                    "max_tokens": 8,
                    "messages": [{"role": "user", "content": "say apricot"}],
                    "stream": stream,
                    "bootstrap_room": 7,
                    "bootstrap_port": 8998,
                })
                .to_string(),
            )
        };
        let body =
            json_of(handle_messages(Arc::clone(&prefill), on_the_prompt_side(false)).await).await;
        assert_eq!(body["content"][0]["text"], "", "{body}");
        assert_eq!(
            (
                body["usage"]["input_tokens"].as_u64(),
                body["usage"]["output_tokens"].as_u64()
            ),
            (Some(2), Some(0))
        );
        let frames = text_of(handle_messages(prefill, on_the_prompt_side(true)).await).await;
        assert!(
            !events_of(&frames)
                .iter()
                .any(|e| e == "content_block_delta")
                && frames.contains("message_stop"),
            "{frames}"
        );
    }

    /// The canned answers in each route's own shape: a chat completion, a
    /// text completion (`choices[].text`, no logprobs, a `cmpl-` id) and the
    /// generate body (`text`, `meta_info`); as a stream, the route's frames
    /// and `[DONE]`.
    #[tokio::test]
    async fn the_canned_answers_take_the_shape_of_their_route() {
        let state = state_of(None, None, "http:canned-shape-test");
        let request = |stream: bool| {
            Bytes::from(json!({"prompt": "say apricot", "stream": stream}).to_string())
        };

        let chat = json_of(handle(Route::Chat, Arc::clone(&state), request(false)).await).await;
        assert_eq!(chat["object"], "chat.completion", "{chat}");
        assert_eq!(chat["choices"][0]["message"]["content"], "mock");

        let completion =
            json_of(handle(Route::Completions, Arc::clone(&state), request(false)).await).await;
        assert_eq!(completion["object"], "text_completion", "{completion}");
        assert_eq!(completion["id"], "cmpl-mock");
        assert_eq!(completion["choices"][0]["text"], "mock");
        assert_eq!(completion["choices"][0]["logprobs"], Value::Null);
        assert_eq!(completion["choices"][0]["finish_reason"], "stop");
        assert!(
            completion["choices"][0].get("message").is_none(),
            "{completion}"
        );
        assert_eq!(completion["usage"]["completion_tokens"], 2);

        let generate =
            json_of(handle(Route::Generate, Arc::clone(&state), request(false)).await).await;
        assert_eq!(generate["text"], "mock", "{generate}");
        assert!(generate.get("choices").is_none(), "{generate}");
        assert_eq!(generate["meta_info"]["prompt_tokens"], 1);
        assert_eq!(generate["meta_info"]["completion_tokens"], 2);
        assert_eq!(generate["meta_info"]["finish_reason"]["type"], "stop");

        let frames =
            text_of(handle(Route::Completions, Arc::clone(&state), request(true)).await).await;
        assert_eq!(
            frames.matches(r#""text":"x""#).count(),
            2,
            "one frame per output token: {frames}"
        );
        assert!(
            frames.contains(r#""object":"text_completion""#)
                && !frames.contains("delta")
                && frames.ends_with("data: [DONE]\n\n"),
            "{frames}"
        );
        let frames = text_of(handle(Route::Generate, state, request(true)).await).await;
        assert!(
            frames.contains(r#""text":"mock""#)
                && frames.contains(r#""meta_info""#)
                && frames.ends_with("data: [DONE]\n\n"),
            "{frames}"
        );
    }

    /// Either side of a disaggregated pair refuses a request that carries no
    /// rendezvous; a regular worker reads none.
    #[tokio::test]
    async fn a_disaggregated_worker_refuses_a_request_without_its_bootstrap_room() {
        for role in [Role::Prefill, Role::Decode] {
            let state = state_of(
                Some(role),
                None,
                &format!("http:{}-room-test", role.as_str()),
            );
            let refused = handle(
                Route::Chat,
                Arc::clone(&state),
                disaggregated_chat(None, None, false),
            )
            .await;
            assert_eq!(refused.status(), StatusCode::BAD_REQUEST, "{role:?}");
            let body = json_of(refused).await;
            assert!(
                body["error"]["message"]
                    .as_str()
                    .is_some_and(|message| message.contains("bootstrap_room")),
                "{body}"
            );
            let served = handle(
                Route::Chat,
                state,
                disaggregated_chat(Some(3), Some(json!(8998)), false),
            )
            .await;
            assert_eq!(served.status(), StatusCode::OK, "{role:?}");
        }
        let regular = state_of(None, None, "http:regular-room-test");
        assert_eq!(
            handle(Route::Chat, regular, disaggregated_chat(None, None, false))
                .await
                .status(),
            StatusCode::OK,
            "a regular worker reads no rendezvous"
        );
    }

    /// A prefill worker that bound its port takes requests naming that port
    /// alone (one, or one per prompt of a batch). A missing or null port is
    /// the engines' default, 8998: the worker on that port takes it, one on
    /// another refuses it. One that bound none is told nothing to check.
    #[tokio::test]
    async fn a_prefill_worker_takes_the_bootstrap_port_it_bound_alone() {
        let bound = state_of(Some(Role::Prefill), Some(8998), "http:port-test");
        for (port, status) in [
            (json!(8998), StatusCode::OK),
            (json!([8998, 8998]), StatusCode::OK),
            (json!(null), StatusCode::OK),
            (json!([null, 8998]), StatusCode::OK),
            (json!(8999), StatusCode::BAD_REQUEST),
            (json!([8998, 8999]), StatusCode::BAD_REQUEST),
            (json!("8998"), StatusCode::BAD_REQUEST),
            (json!([]), StatusCode::BAD_REQUEST),
        ] {
            let response = handle(
                Route::Chat,
                Arc::clone(&bound),
                disaggregated_chat(Some(1), Some(port.clone()), false),
            )
            .await;
            assert_eq!(response.status(), status, "bootstrap_port {port}");
        }
        let missing = handle(
            Route::Chat,
            Arc::clone(&bound),
            disaggregated_chat(Some(1), None, false),
        )
        .await;
        assert_eq!(
            missing.status(),
            StatusCode::OK,
            "no port named: the default"
        );
        let elsewhere = state_of(Some(Role::Prefill), Some(9001), "http:elsewhere-test");
        for port in [Some(json!(null)), None] {
            let response = handle(
                Route::Chat,
                Arc::clone(&elsewhere),
                disaggregated_chat(Some(1), port.clone(), false),
            )
            .await;
            assert_eq!(
                response.status(),
                StatusCode::BAD_REQUEST,
                "bootstrap_port {port:?} on a worker off the default port"
            );
        }
        let unbound = state_of(Some(Role::Prefill), None, "http:unbound-test");
        assert_eq!(
            handle(
                Route::Chat,
                unbound,
                disaggregated_chat(Some(1), Some(json!(8999)), false)
            )
            .await
            .status(),
            StatusCode::OK
        );
    }

    /// The server info names the role and, on the prefill side, the
    /// bootstrap port; a regular worker reports neither.
    #[tokio::test]
    async fn the_server_info_reports_the_role_and_the_bootstrap_port() {
        let regular =
            json_of(server_info(State(state_of(None, None, "http:info-regular"))).await).await;
        assert_eq!(regular["model_path"], Config::default().model_id);
        assert!(regular.get("disaggregation_mode").is_none(), "{regular}");
        let prefill = json_of(
            server_info(State(state_of(
                Some(Role::Prefill),
                Some(8998),
                "http:info-prefill",
            )))
            .await,
        )
        .await;
        assert_eq!(prefill["disaggregation_mode"], "prefill");
        assert_eq!(prefill["disaggregation_bootstrap_port"], 8998);
        assert_eq!(
            prefill["disaggregation_transfer_backend"],
            crate::grpc::KV_TRANSPORT
        );
        let decode = json_of(
            server_info(State(state_of(
                Some(Role::Decode),
                None,
                "http:info-decode",
            )))
            .await,
        )
        .await;
        assert_eq!(decode["disaggregation_mode"], "decode");
        assert!(
            decode.get("disaggregation_bootstrap_port").is_none(),
            "{decode}"
        );
    }
}
