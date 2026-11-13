//! Process-wide admin API for the simulated fleet: the ground truth a routing
//! benchmark needs that real engines do not expose, and the fault hooks a
//! recovery test switches on.
//!
//! - `GET /admin/fleet`: every worker of the process: a realistic engine with
//!   its cache size and load and, in a canned process, each listener with its
//!   name and mode.
//! - `GET /admin/requests?since=<seq>&limit=<n>`: admitted-request records
//!   (`request_id`, serving worker, prompt/cached/oracle tokens, queue wait),
//!   oldest first, `seq` strictly greater than `since`.
//! - `GET /admin/cache/{worker}`: the worker's cached block keys.
//! - `POST /admin/reset/{worker}` and `POST /admin/reset`: clear one or every
//!   cache and publish `AllBlocksCleared` (an engine restart, to the index).
//! - `POST /admin/fault/{worker}/drop?batches=N`, `.../delay?ms=D`,
//!   `.../restart-publisher`, `.../pause`, `.../resume`,
//!   `.../fail?status=S[&count=N|&secs=T][&after_tokens=K][&stall_ms=M]` and
//!   `GET /admin/fault/{worker}`: the fault hooks (see the README).
//! - `GET /admin/ops/{worker}`: what the gateway's admin fan-out reached on
//!   the worker: its flushes, its profiling runs and the last trace written
//!   (`all`: every worker).
//! - `POST /admin/truth/{worker}` with `{"token_ids": [...]}`: what the
//!   worker would serve from cache for that prompt right now;
//!   `GET /admin/truth`: per worker, what it served over every admitted
//!   request.
//! - `POST /admin/script/{worker}` with one scripted answer or an array of
//!   them: what the worker's next requests get, in order (`all`: whichever
//!   worker gets the request), or, with a `key` or `input_ids`, what every
//!   worker answers to that prompt; `GET` reads the queue length and the hit
//!   and miss counts, `DELETE` drops the queue ([`crate::script`]); a name no
//!   worker of the process has is a 404, as on the fault routes.
//! - `GET /admin/zmq/{worker}`: a ZMQ rank's `batch_messages` (the output
//!   messages that carried request outputs) and `outputs` (the outputs in
//!   them) since it connected, and its side of the wave protocol when it is
//!   a rank of a lockstep group: `parks` (the waves it drained and parked
//!   on), `wakes` (the wave starts it accepted) and `wave` (its current
//!   wave); cumulative, a reset does not clear them; a rank name, not `all`.
//!
//! Worker names are `grpc:<port>` / `http:<port>` / `zmq:<index>`; `all`
//! addresses every worker on the routes that take it.

use std::{
    collections::{BTreeMap, HashMap},
    future::{pending, Future},
    sync::Arc,
    time::{Duration, Instant},
};

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use tokio::net::TcpListener;

use crate::{
    admin_ops::AdminOps,
    config::Config,
    engine::{self, Engine, FaultScope, RequestFault},
    script,
};

struct AdminState {
    cfg: Arc<Config>,
}

fn router(state: Arc<AdminState>) -> Router {
    Router::new()
        .route("/admin/health", get(health))
        .route("/admin/fleet", get(fleet))
        .route("/admin/requests", get(requests))
        .route("/admin/cache/{worker}", get(cache))
        .route("/admin/reset", post(reset_all))
        .route("/admin/reset/{worker}", post(reset_one))
        .route("/admin/fault/{worker}", get(fault_status))
        .route("/admin/fault/{worker}/drop", post(fault_drop))
        .route("/admin/fault/{worker}/delay", post(fault_delay))
        .route("/admin/fault/{worker}/admit-delay", post(fault_admit_delay))
        .route("/admin/fault/{worker}/admit-rate", post(fault_admit_rate))
        .route("/admin/fault/{worker}/cold-shape", post(fault_cold_shape))
        .route(
            "/admin/fault/{worker}/restart-publisher",
            post(fault_restart_publisher),
        )
        .route("/admin/fault/{worker}/pause", post(fault_pause))
        .route("/admin/fault/{worker}/resume", post(fault_resume))
        .route("/admin/fault/{worker}/fail", post(fault_fail))
        .route("/admin/ops/{worker}", get(ops_status))
        .route("/admin/truth", get(truth_served))
        .route("/admin/truth/{worker}", post(truth_prompt))
        .route(
            "/admin/script/{worker}",
            get(script_status).post(script_enqueue).delete(script_clear),
        )
        // What a gRPC worker reports about itself, by port: whether its
        // health check answers serving, and the info the gateway turns into
        // labels. For beds that change an engine's reported settings between
        // two health transitions.
        .route("/admin/serving/{port}", get(serving_get).post(serving_set))
        .route("/admin/zmq/{worker}", get(zmq_rank))
        .route("/admin/info/{port}", get(info_get).post(info_patch))
        .with_state(state)
}

async fn serving_get(Path(port): Path<u16>) -> Json<Value> {
    Json(json!({ "port": port, "serving": crate::overrides::for_port(port).serving() }))
}

/// Body `{"serving": true|false}`: whether the worker's gRPC health check
/// answers healthy from now on.
async fn serving_set(Path(port): Path<u16>, Json(body): Json<Value>) -> Response {
    match body.get("serving").and_then(Value::as_bool) {
        Some(serving) => {
            crate::overrides::for_port(port).set_serving(serving);
            Json(json!({ "port": port, "serving": serving })).into_response()
        }
        None => (StatusCode::BAD_REQUEST, "body: {\"serving\": true|false}").into_response(),
    }
}

async fn info_get(Path(port): Path<u16>) -> Json<Value> {
    Json(Value::Object(
        crate::overrides::for_port(port)
            .info()
            .into_iter()
            .collect(),
    ))
}

/// Body: a JSON object merged into the info the worker reports (its gRPC
/// `server_args`, and `supports_vision` of its model info); a `null` removes
/// a key. Answers the info as it is now.
async fn info_patch(Path(port): Path<u16>, Json(body): Json<Value>) -> Response {
    match body.as_object() {
        Some(patch) => {
            let overrides = crate::overrides::for_port(port);
            overrides.patch_info(patch);
            Json(Value::Object(overrides.info().into_iter().collect())).into_response()
        }
        None => (StatusCode::BAD_REQUEST, "body: a JSON object").into_response(),
    }
}

/// Body: one scripted answer or an array of them ([`script::Answer::parse`]
/// names the fields). Each goes to the end of the worker's queue (`all`: the
/// queue every worker falls back to), under its prompt's key when it has
/// one, or under the text a prompt must hold (`prompt_contains`). Answers
/// the worker's script state.
async fn script_enqueue(Path(worker): Path<String>, Json(body): Json<Value>) -> Response {
    let items: Vec<&Value> = match &body {
        Value::Array(items) => items.iter().collect(),
        one => vec![one],
    };
    let mut answers = Vec::with_capacity(items.len());
    for (n, item) in items.into_iter().enumerate() {
        match script::Answer::parse(item) {
            Ok(scripted) => answers.push(scripted),
            Err(why) => {
                return (StatusCode::BAD_REQUEST, format!("answer {n}: {why}")).into_response()
            }
        }
    }
    script_state(script::enqueue(&worker, answers))
}

/// The worker's queued answers, the ones filed under a text, its hits and
/// misses, and the keyed answers of the process.
async fn script_status(Path(worker): Path<String>) -> Response {
    script_state(script::status(&worker))
}

/// Drop the worker's queued answers and the ones filed under a text (`all`:
/// every worker's); keyed answers stay.
async fn script_clear(Path(worker): Path<String>) -> Response {
    script_state(script::clear(&worker))
}

/// A script route's answer: the state, or 404 for a name no worker of the
/// process has (the fault routes' answer to the same mistake).
fn script_state(state: Option<Value>) -> Response {
    match state {
        Some(state) => Json(state).into_response(),
        None => (StatusCode::NOT_FOUND, "unknown worker").into_response(),
    }
}

/// Serve the admin API on `port` until the process exits.
pub async fn serve(cfg: Arc<Config>, host: String, port: u16) {
    serve_until(cfg, host, port, pending()).await;
}

/// Serve the admin API on `port` until `shutdown` resolves (the drain of a
/// termination signal): the requests in flight are answered, no new one is
/// accepted. A port another listener holds stops the server before it
/// serves: the bind failure is logged and the future returns.
pub async fn serve_until(
    cfg: Arc<Config>,
    host: String,
    port: u16,
    shutdown: impl Future<Output = ()> + Send + 'static,
) {
    let listener = match TcpListener::bind((host.as_str(), port)).await {
        Ok(listener) => listener,
        Err(e) => {
            tracing::error!("admin bind {host}:{port} failed: {e}");
            return;
        }
    };
    serve_with_listener_until(cfg, listener, shutdown).await;
}

/// Serve the admin API on an already-bound listener until the process exits.
/// A test that needs a free port binds port 0 and reads it back instead of
/// probing one and binding it later: nothing takes the port in between.
pub async fn serve_with_listener(cfg: Arc<Config>, listener: TcpListener) {
    serve_with_listener_until(cfg, listener, pending()).await;
}

/// [`serve_with_listener`] with the shutdown of [`serve_until`].
pub async fn serve_with_listener_until(
    cfg: Arc<Config>,
    listener: TcpListener,
    shutdown: impl Future<Output = ()> + Send + 'static,
) {
    let state = Arc::new(AdminState { cfg });
    if let Err(e) = axum::serve(listener, router(state))
        .with_graceful_shutdown(shutdown)
        .await
    {
        tracing::error!("admin server stopped: {e}");
    }
}

async fn health() -> &'static str {
    "ok"
}

fn find(name: &str) -> Option<Engine> {
    engine::fleet_engines()
        .into_iter()
        .find(|e| e.name() == name)
}

/// The engines a `{worker}` path segment addresses: one by name, or `all`.
fn select(worker: &str) -> Result<Vec<Engine>, Response> {
    if worker == "all" {
        return Ok(engine::fleet_engines());
    }
    find(worker)
        .map(|e| vec![e])
        .ok_or_else(|| (StatusCode::NOT_FOUND, "unknown worker").into_response())
}

fn status_json(e: &Engine) -> Value {
    let s = e.fault_status();
    let fail = s.fail;
    let (fail_pending, fail_secs_left) = match fail.map(|f| f.scope) {
        Some(FaultScope::Requests(n)) => (Some(n), None),
        Some(FaultScope::Until(at)) => (
            None,
            Some(at.saturating_duration_since(Instant::now()).as_secs_f64()),
        ),
        Some(FaultScope::Open) | None => (None, None),
    };
    json!({
        "worker": e.name(),
        "drop_pending": s.drop_pending,
        "dropped_total": s.dropped_total,
        "delay_ms": s.delay_ms,
        "admit_delay_ms": s.admit_delay_ms,
        "admit_per_sec": s.admit_per_sec,
        "cold_shape_ms": s.cold_shape_ms,
        "paused": s.paused,
        "generation": s.generation,
        "restarts": s.restarts,
        "fail_status": fail.map_or(0, |f| f.status),
        "fail_pending": fail_pending,
        "fail_secs_left": fail_secs_left,
        "fail_after_tokens": fail.and_then(|f| f.after_tokens),
        "fail_finish_error": fail.is_some_and(|f| f.finish_error),
        "fail_stall_ms": fail.map_or(0, |f| f.stall.as_millis() as u64),
        "failed_total": s.failed_total,
        "cut_total": s.cut_total,
        "stalled_total": s.stalled_total,
    })
}

fn param<T: std::str::FromStr>(q: &HashMap<String, String>, name: &str) -> Result<T, Response> {
    q.get(name).and_then(|v| v.parse().ok()).ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            format!("missing or invalid query parameter {name}"),
        )
            .into_response()
    })
}

/// A query parameter that may be absent; present, it must parse.
fn optional<T: std::str::FromStr>(
    q: &HashMap<String, String>,
    name: &str,
) -> Result<Option<T>, String> {
    match q.get(name) {
        None => Ok(None),
        Some(v) => v
            .parse()
            .map(Some)
            .map_err(|_| format!("invalid query parameter {name}: {v}")),
    }
}

/// The request fault a `fail` query arms, or `None` to clear it:
/// `status=S` (0, or 400 to 599) answered instead of serving, for the next
/// `count=N` requests or every request for `secs=T` (neither: until cleared),
/// after a `stall_ms=M` hold; `after_tokens=K` serves K tokens first and cuts
/// the stream with S (on the gRPC wire, `status=422` is the engine's report
/// that it could not continue the request under its constraint). `finish=error`
/// ends the stream as an engine that failed the request does, a `Complete`
/// with finish reason `error` after K tokens (0 when unset), with no status
/// needed. `status=0` alone clears; with `stall_ms` it stalls and then serves.
fn fail_query(q: &HashMap<String, String>) -> Result<Option<RequestFault>, String> {
    let status: u16 = optional(q, "status")?.unwrap_or(0);
    let count: Option<u32> = optional(q, "count")?;
    let secs: Option<f64> = optional(q, "secs")?;
    let mut after_tokens: Option<u32> = optional(q, "after_tokens")?;
    let stall_ms: u64 = optional(q, "stall_ms")?.unwrap_or(0);
    let finish_error = match q.get("finish").map(String::as_str) {
        None => false,
        Some("error") => true,
        Some(other) => return Err(format!("finish must be \"error\", got {other}")),
    };
    if status != 0 && !(400..=599).contains(&status) {
        return Err(format!(
            "status must be 0 or an error status (400-599), got {status}"
        ));
    }
    if finish_error {
        after_tokens = Some(after_tokens.unwrap_or(0));
    } else if status == 0 && after_tokens.is_some() {
        return Err("after_tokens needs a status to cut the stream with".to_string());
    }
    if status == 0 && stall_ms == 0 && !finish_error {
        return Ok(None);
    }
    let scope = match (count, secs) {
        (Some(_), Some(_)) => return Err("count and secs exclude each other".to_string()),
        (Some(0), None) => return Err("count must be at least 1".to_string()),
        (Some(n), None) => FaultScope::Requests(n),
        (None, Some(t)) if t.is_finite() && t > 0.0 => Duration::try_from_secs_f64(t)
            .ok()
            .and_then(|d| Instant::now().checked_add(d))
            .map(FaultScope::Until)
            .ok_or_else(|| format!("secs out of range, got {t}"))?,
        (None, Some(t)) => return Err(format!("secs must be positive, got {t}")),
        (None, None) => FaultScope::Open,
    };
    Ok(Some(RequestFault {
        status,
        stall: Duration::from_millis(stall_ms),
        after_tokens,
        finish_error,
        scope,
    }))
}

async fn fleet(State(state): State<Arc<AdminState>>) -> Json<Value> {
    Json(json!({ "workers": fleet_rows(&state.cfg) }))
}

/// One row per worker of the process, in name order: a realistic engine with
/// its load (a data-parallel rank is a row of its own, named `<worker>:r<k>`)
/// and, in a canned process, each listener with its name and mode alone, since
/// no engine stands behind it. A listener registers its name when it starts,
/// before a realistic engine spawns (a ZMQ rank's after its handshake), so a
/// realistic process lists its engines alone and a canned fleet counts here as
/// it serves; an engine no listener claims is listed after them.
fn fleet_rows(cfg: &Config) -> Vec<Value> {
    let engines = engine::fleet_engines();
    let mut listed = vec![false; engines.len()];
    let mut rows = Vec::with_capacity(engines.len());
    for worker in script::registered() {
        let mut found = false;
        for (index, e) in engines.iter().enumerate() {
            let rank_of_worker = e
                .name()
                .strip_prefix(worker.as_str())
                .is_some_and(|rest| rest.starts_with(":r"));
            if e.name() == worker || rank_of_worker {
                rows.push(engine_row(e, cfg));
                listed[index] = true;
                found = true;
            }
        }
        if !found && !cfg.realistic {
            rows.push(json!({ "worker": worker, "engine": "canned" }));
        }
    }
    for (index, e) in engines.iter().enumerate() {
        if !listed[index] {
            rows.push(engine_row(e, cfg));
        }
    }
    rows
}

fn engine_row(e: &Engine, cfg: &Config) -> Value {
    let load = e.load();
    json!({
        "worker": e.name(),
        "engine": "realistic",
        "cache_blocks": e.cache_len(),
        "block_size": cfg.engine.block_size,
        "num_running_reqs": load.num_running_reqs,
        "num_waiting_reqs": load.num_waiting_reqs,
        "num_waiting_uncached_tokens": load.num_waiting_uncached_tokens,
        "token_usage": load.token_usage,
        "cache_hit_rate": load.cache_hit_rate,
        "num_cached_blocks": load.num_cached_blocks,
        "num_preemptions": load.num_preemptions,
        "num_kv_batches": load.num_kv_batches,
    })
}

async fn requests(Query(q): Query<HashMap<String, String>>) -> Json<Value> {
    let since = q.get("since").and_then(|v| v.parse().ok()).unwrap_or(0u64);
    let limit = q
        .get("limit")
        .and_then(|v| v.parse().ok())
        .unwrap_or(100_000usize);
    let records = engine::records_since(since, limit);
    let next = records.last().map(|r| r.seq).unwrap_or(since);
    let rows: Vec<Value> = records
        .iter()
        .map(|r| {
            json!({
                "seq": r.seq,
                "request_id": r.request_id,
                "worker": r.worker,
                "prompt_tokens": r.prompt_tokens,
                "cached_tokens": r.cached_tokens,
                "oracle_tokens": r.oracle_tokens,
                "queued_ms": r.queued_ms,
                "running_at_admit": r.running_at_admit,
                "waiting_at_admit": r.waiting_at_admit,
                "admitted_unix_ms": r.admitted_unix_ms,
            })
        })
        .collect();
    Json(json!({
        "records": rows,
        "next": next,
        "injected_failures": injected_failures(),
    }))
}

/// Per worker, the requests its fault hook answered with a status before
/// admission (they left no record) and the streams it cut after output.
fn injected_failures() -> Value {
    let workers: serde_json::Map<String, Value> = engine::fleet_engines()
        .iter()
        .map(|e| {
            let s = e.fault_status();
            (
                e.name().to_string(),
                json!({ "failed": s.failed_total, "cut": s.cut_total }),
            )
        })
        .collect();
    Value::Object(workers)
}

/// What a ZMQ rank has pushed since it connected: the messages that carried
/// request outputs and the outputs in them, cumulative (a reset does not
/// clear them), so a test tells an engine's step of many outputs from one
/// message per token; and the waves a rank of a lockstep group parked on,
/// the wave starts it accepted and its current wave, so a test tells a park
/// from a wake.
async fn zmq_rank(Path(worker): Path<String>) -> Response {
    match crate::zmq::rank_output_counts(&worker) {
        Some(counts) => {
            let (batch_messages, outputs) = counts.read();
            let (parks, wakes, wave) = counts.waves();
            Json(json!({
                "worker": worker,
                "batch_messages": batch_messages,
                "outputs": outputs,
                "parks": parks,
                "wakes": wakes,
                "wave": wave,
            }))
            .into_response()
        }
        None => (StatusCode::NOT_FOUND, "unknown ZMQ rank").into_response(),
    }
}

async fn cache(Path(worker): Path<String>) -> Response {
    match find(&worker) {
        Some(e) => {
            let mut keys = e.cache_keys();
            keys.sort_unstable();
            Json(json!({ "worker": worker, "blocks": keys })).into_response()
        }
        None => (StatusCode::NOT_FOUND, "unknown worker").into_response(),
    }
}

async fn reset_one(Path(worker): Path<String>) -> Response {
    match select(&worker) {
        Ok(engines) => {
            let names: Vec<String> = engines
                .iter()
                .map(|e| {
                    e.reset();
                    e.name().to_string()
                })
                .collect();
            Json(json!({ "reset": names })).into_response()
        }
        Err(response) => response,
    }
}

/// The flush and profile operations the gateway fanned out to the worker.
async fn ops_status(Path(worker): Path<String>) -> Response {
    if worker == "all" {
        let workers: Vec<Value> = AdminOps::all().iter().map(|ops| ops.status()).collect();
        return Json(json!({ "workers": workers })).into_response();
    }
    match AdminOps::find(&worker) {
        Some(ops) => Json(ops.status()).into_response(),
        None => (StatusCode::NOT_FOUND, "unknown worker").into_response(),
    }
}

async fn reset_all() -> Json<Value> {
    let names: Vec<String> = engine::fleet_engines()
        .iter()
        .map(|e| {
            e.reset();
            e.name().to_string()
        })
        .collect();
    Json(json!({ "reset": names }))
}

async fn fault_status(Path(worker): Path<String>) -> Response {
    match select(&worker) {
        Ok(engines) => {
            let workers: Vec<Value> = engines.iter().map(status_json).collect();
            Json(json!({ "workers": workers })).into_response()
        }
        Err(response) => response,
    }
}

async fn fault_drop(
    Path(worker): Path<String>,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    let batches: u32 = match param(&q, "batches") {
        Ok(v) => v,
        Err(response) => return response,
    };
    apply(&worker, |e| e.fault_drop(batches))
}

async fn fault_delay(
    Path(worker): Path<String>,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    let ms: u64 = match param(&q, "ms") {
        Ok(v) => v,
        Err(response) => return response,
    };
    apply(&worker, |e| e.fault_delay_ms(ms))
}

async fn fault_admit_delay(
    Path(worker): Path<String>,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    let ms: u64 = match param(&q, "ms") {
        Ok(v) => v,
        Err(response) => return response,
    };
    apply(&worker, |e| e.fault_admit_delay_ms(ms))
}

async fn fault_cold_shape(
    Path(worker): Path<String>,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    let ms: u64 = match param(&q, "ms") {
        Ok(v) => v,
        Err(response) => return response,
    };
    apply(&worker, |e| e.fault_cold_shape_ms(ms))
}

async fn fault_admit_rate(
    Path(worker): Path<String>,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    let per_sec: u64 = match param(&q, "per_sec") {
        Ok(v) => v,
        Err(response) => return response,
    };
    apply(&worker, |e| e.fault_admit_per_sec(per_sec))
}

async fn fault_restart_publisher(Path(worker): Path<String>) -> Response {
    match select(&worker) {
        Ok(engines) => {
            for e in &engines {
                e.restart_publisher().await;
            }
            let workers: Vec<Value> = engines.iter().map(status_json).collect();
            Json(json!({ "workers": workers })).into_response()
        }
        Err(response) => response,
    }
}

async fn fault_pause(Path(worker): Path<String>) -> Response {
    apply(&worker, Engine::pause)
}

async fn fault_fail(
    Path(worker): Path<String>,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    let fault = match fail_query(&q) {
        Ok(fault) => fault,
        Err(why) => return (StatusCode::BAD_REQUEST, why).into_response(),
    };
    apply(&worker, |e| e.fault_fail(fault))
}

async fn fault_resume(Path(worker): Path<String>) -> Response {
    apply(&worker, Engine::resume)
}

/// Run `hook` on the addressed engines and answer with their fault state.
fn apply(worker: &str, hook: impl Fn(&Engine)) -> Response {
    match select(worker) {
        Ok(engines) => {
            for e in &engines {
                hook(e);
            }
            let workers: Vec<Value> = engines.iter().map(status_json).collect();
            Json(json!({ "workers": workers })).into_response()
        }
        Err(response) => response,
    }
}

async fn truth_prompt(Path(worker): Path<String>, Json(body): Json<Value>) -> Response {
    let Some(token_ids) = body.get("token_ids").and_then(Value::as_array).map(|ids| {
        ids.iter()
            .filter_map(Value::as_u64)
            .map(|t| u32::try_from(t).unwrap_or(u32::MAX))
            .collect::<Vec<u32>>()
    }) else {
        return (StatusCode::BAD_REQUEST, "body needs token_ids: [u32]").into_response();
    };
    match select(&worker) {
        Ok(engines) => {
            let workers: Vec<Value> = engines
                .iter()
                .map(|e| {
                    let truth = e.cached_tokens_for(&token_ids);
                    json!({
                        "worker": e.name(),
                        "cached_tokens": truth.cached_tokens,
                        "cached_blocks": truth.cached_blocks,
                        "block_size": truth.block_size,
                        "prompt_tokens": token_ids.len(),
                    })
                })
                .collect();
            Json(json!({ "workers": workers })).into_response()
        }
        Err(response) => response,
    }
}

/// Per worker, what it actually served over every admitted request, and
/// what its fault hook refused or cut.
async fn truth_served() -> Json<Value> {
    let mut totals: BTreeMap<String, (u64, u64, u64, u64)> = BTreeMap::new();
    for r in engine::records_since(0, usize::MAX) {
        let t = totals.entry(r.worker).or_insert((0, 0, 0, 0));
        t.0 += 1;
        t.1 += u64::from(r.prompt_tokens);
        t.2 += u64::from(r.cached_tokens);
        t.3 += u64::from(r.oracle_tokens);
    }
    let injected: BTreeMap<String, (u64, u64)> = engine::fleet_engines()
        .iter()
        .map(|e| {
            let s = e.fault_status();
            (e.name().to_string(), (s.failed_total, s.cut_total))
        })
        .collect();
    for worker in injected.keys() {
        totals.entry(worker.clone()).or_insert((0, 0, 0, 0));
    }
    let workers: Vec<Value> = totals
        .iter()
        .map(|(worker, (requests, prompt, cached, oracle))| {
            let (failed, cut) = injected.get(worker).copied().unwrap_or((0, 0));
            json!({
                "worker": worker,
                "requests": requests,
                "prompt_tokens": prompt,
                "cached_tokens": cached,
                "oracle_tokens": oracle,
                "injected_failures": failed,
                "cut_streams": cut,
            })
        })
        .collect();
    Json(json!({ "workers": workers }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    }

    /// A listener whose bootstrap port is taken stops before it serves, and
    /// the fleet never lists it: a worker joins the registries and spawns its
    /// engine once that port is bound.
    #[tokio::test]
    async fn a_listener_that_cannot_bind_its_bootstrap_port_never_joins_the_fleet() {
        let taken = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("a port to occupy");
        let taken_port = taken.local_addr().expect("address").port();
        // The worker's port is drawn from the OS and kept while its name is
        // new to the registries, which are process-wide and never forget a
        // name: a port an earlier test's worker served would read as
        // registered here. The listener is held through the check, so no
        // other test takes the port meanwhile.
        let (listener, port) = loop {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("the worker's port");
            let port = listener.local_addr().expect("address").port();
            let name = format!("grpc:{port}");
            let known = script::registered().contains(&name)
                || engine::fleet_engines().iter().any(|e| e.name() == name);
            if !known {
                break (listener, port);
            }
        };
        let cfg = std::sync::Arc::new(Config {
            host: "127.0.0.1".to_string(),
            grpc_base_port: port,
            grpc_count: 1,
            realistic: true,
            role: Some(crate::config::Role::Prefill),
            bootstrap_base_port: Some(taken_port),
            ..Config::default()
        });
        // The worker stops at the bind: serving returns instead of running on.
        tokio::time::timeout(
            std::time::Duration::from_secs(10),
            crate::grpc::serve_with_listener(std::sync::Arc::clone(&cfg), listener),
        )
        .await
        .expect("the worker stops at the bootstrap bind");
        let name = format!("grpc:{port}");
        assert!(
            !script::registered().contains(&name),
            "a worker that never served is registered"
        );
        assert!(
            engine::fleet_engines().iter().all(|e| e.name() != name),
            "a worker that never served spawned its engine"
        );
        assert!(
            fleet_rows(&cfg).iter().all(|row| row["worker"] != name),
            "a worker that never served is listed: {:?}",
            fleet_rows(&cfg)
        );
        drop(taken);
    }

    /// The admin API serves a listener its caller bound: the port the caller
    /// read back answers, and the shutdown ends the server.
    #[tokio::test]
    async fn the_admin_api_serves_a_listener_the_caller_bound_until_its_shutdown() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind the admin listener");
        let port = listener.local_addr().expect("address").port();
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        #[expect(
            clippy::disallowed_methods,
            reason = "the server task is joined at the end of the test"
        )]
        let server = tokio::spawn(serve_with_listener_until(
            std::sync::Arc::new(Config::default()),
            listener,
            async {
                let _ = stopped.await;
            },
        ));
        crate::tls::install_crypto_provider();
        #[expect(
            clippy::disallowed_methods,
            reason = "a test's client; the process TLS crypto provider is installed first"
        )]
        let http = reqwest::Client::new();
        let health = http
            .get(format!("http://127.0.0.1:{port}/admin/health"))
            .send()
            .await
            .expect("the admin API answers on the port read back");
        assert_eq!(health.text().await.expect("the body"), "ok");
        stop.send(()).expect("the server waits on its shutdown");
        tokio::time::timeout(std::time::Duration::from_secs(10), server)
            .await
            .expect("the server stops at its shutdown")
            .expect("the server task");
    }

    /// A port another listener holds stops the server before it serves: the
    /// bind failure is logged and the future returns, which is why a test
    /// binds the admin listener itself and reads the port back.
    #[tokio::test]
    async fn a_taken_admin_port_stops_the_server_before_it_serves() {
        let taken = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("a port to occupy");
        let port = taken.local_addr().expect("address").port();
        tokio::time::timeout(
            std::time::Duration::from_secs(10),
            serve_until(
                std::sync::Arc::new(Config::default()),
                "127.0.0.1".to_string(),
                port,
                pending(),
            ),
        )
        .await
        .expect("the server stops at the bind");
        drop(taken);
    }

    #[tokio::test]
    async fn the_fleet_lists_canned_listeners_beside_the_engines() {
        // The registries are process-wide; the names below are this test's own.
        script::register("http:fleet-canned");
        script::register("grpc:fleet-real");
        script::register("grpc:fleet-dp");
        let engines = [
            Engine::spawn_named(
                engine::EngineParams::default(),
                "grpc:fleet-real".into(),
                true,
            ),
            Engine::spawn_named(
                engine::EngineParams::default(),
                "grpc:fleet-dp:r0".into(),
                true,
            ),
            Engine::spawn_named(
                engine::EngineParams::default(),
                "grpc:fleet-dp:r1".into(),
                true,
            ),
            // No listener registered this name: the engine is listed after the claimed ones.
            Engine::spawn_named(
                engine::EngineParams::default(),
                "grpc:fleet-orphan".into(),
                true,
            ),
        ];
        let cfg = Config::default();
        let rows = fleet_rows(&cfg);
        let row = |name: &str| {
            rows.iter()
                .find(|row| row["worker"] == name)
                .cloned()
                .expect("a row for the name")
        };
        let canned = row("http:fleet-canned");
        assert_eq!(canned["engine"], "canned");
        assert!(
            canned.get("num_running_reqs").is_none(),
            "a canned row carries no load"
        );
        let real = row("grpc:fleet-real");
        assert_eq!(real["engine"], "realistic");
        assert_eq!(real["num_running_reqs"], 0);
        assert_eq!(real["block_size"], json!(cfg.engine.block_size));
        // A data-parallel worker lists its ranks, each once, and no canned row of its own.
        assert_eq!(row("grpc:fleet-dp:r0")["engine"], "realistic");
        assert_eq!(row("grpc:fleet-dp:r1")["engine"], "realistic");
        let named = |name: &str| rows.iter().filter(|row| row["worker"] == name).count();
        assert_eq!((named("grpc:fleet-dp"), named("grpc:fleet-real")), (0, 1));
        assert_eq!(named("grpc:fleet-orphan"), 1);
        assert_eq!(row("grpc:fleet-orphan")["engine"], "realistic");
        // A realistic process lists its engines alone: a listener registers before
        // its engine spawns, so a name without an engine is not a canned worker there.
        let realistic = fleet_rows(&Config {
            realistic: true,
            ..Config::default()
        });
        assert!(realistic.iter().all(|row| row["engine"] == "realistic"));
        let listed = |name: &str| realistic.iter().filter(|row| row["worker"] == name).count();
        assert_eq!(
            (listed("http:fleet-canned"), listed("grpc:fleet-orphan")),
            (0, 1)
        );
        drop(engines);
    }

    #[test]
    fn fail_query_arms_counts_and_clears() {
        assert_eq!(fail_query(&query(&[("status", "0")])), Ok(None));
        assert_eq!(fail_query(&query(&[])), Ok(None));
        let next_two = fail_query(&query(&[("status", "503"), ("count", "2")]))
            .unwrap()
            .expect("armed");
        assert_eq!(
            (next_two.status, next_two.scope, next_two.after_tokens),
            (503, FaultScope::Requests(2), None)
        );
        assert_eq!(next_two.stall, Duration::ZERO);
        let open = fail_query(&query(&[("status", "500")]))
            .unwrap()
            .expect("armed");
        assert_eq!(open.scope, FaultScope::Open);
        let cut = fail_query(&query(&[
            ("status", "502"),
            ("after_tokens", "4"),
            ("stall_ms", "250"),
        ]))
        .unwrap()
        .expect("armed");
        assert_eq!(cut.after_tokens, Some(4));
        assert_eq!(cut.stall, Duration::from_millis(250));
        let stall_only = fail_query(&query(&[("stall_ms", "3000"), ("count", "1")]))
            .unwrap()
            .expect("armed");
        assert_eq!(
            (stall_only.status, stall_only.scope),
            (0, FaultScope::Requests(1))
        );
        let timed = fail_query(&query(&[("status", "429"), ("secs", "0.5")]))
            .unwrap()
            .expect("armed");
        assert!(matches!(timed.scope, FaultScope::Until(at) if at > Instant::now()));
    }

    #[test]
    fn fail_query_rejects_what_it_cannot_mean() {
        for bad in [
            query(&[("status", "200")]),
            query(&[("status", "503"), ("count", "1"), ("secs", "1")]),
            query(&[("status", "503"), ("count", "0")]),
            query(&[("status", "503"), ("secs", "0")]),
            query(&[("status", "503"), ("secs", "1e30")]),
            query(&[("after_tokens", "2")]),
            query(&[("status", "many")]),
        ] {
            assert!(fail_query(&bad).is_err(), "{bad:?}");
        }
    }
}
