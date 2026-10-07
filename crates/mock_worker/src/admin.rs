//! Process-wide admin API for the simulated fleet: the ground truth a routing
//! benchmark needs that real engines do not expose, and the fault hooks a
//! recovery test switches on.
//!
//! - `GET /admin/fleet`: every registered engine with its cache size and load.
//! - `GET /admin/requests?since=<seq>&limit=<n>`: admitted-request records
//!   (`request_id`, serving worker, prompt/cached/oracle tokens, queue wait),
//!   oldest first, `seq` strictly greater than `since`.
//! - `GET /admin/cache/{worker}`: the worker's cached block keys.
//! - `POST /admin/reset/{worker}` and `POST /admin/reset`: clear one or every
//!   cache and publish `AllBlocksCleared` (an engine restart, to the index).
//! - `POST /admin/fault/{worker}/drop?batches=N`, `.../delay?ms=D`,
//!   `.../restart-publisher`, `.../pause`, `.../resume` and
//!   `GET /admin/fault/{worker}`: the fault hooks (see the README).
//! - `POST /admin/truth/{worker}` with `{"token_ids": [...]}`: what the
//!   worker would serve from cache for that prompt right now;
//!   `GET /admin/truth`: per worker, what it served over every admitted
//!   request.
//!
//! Worker names are `grpc:<port>` / `http:<port>` / `zmq:<index>`; `all`
//! addresses every worker.

use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
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
    config::Config,
    engine::{self, Engine},
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
        .route(
            "/admin/fault/{worker}/restart-publisher",
            post(fault_restart_publisher),
        )
        .route("/admin/fault/{worker}/pause", post(fault_pause))
        .route("/admin/fault/{worker}/resume", post(fault_resume))
        .route("/admin/truth", get(truth_served))
        .route("/admin/truth/{worker}", post(truth_prompt))
        .with_state(state)
}

/// Serve the admin API on `port` until the process exits.
pub async fn serve(cfg: Arc<Config>, host: String, port: u16) {
    let listener = match TcpListener::bind((host.as_str(), port)).await {
        Ok(listener) => listener,
        Err(e) => {
            tracing::error!("admin bind {host}:{port} failed: {e}");
            return;
        }
    };
    let state = Arc::new(AdminState { cfg });
    if let Err(e) = axum::serve(listener, router(state)).await {
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
    json!({
        "worker": e.name(),
        "drop_pending": s.drop_pending,
        "dropped_total": s.dropped_total,
        "delay_ms": s.delay_ms,
        "paused": s.paused,
        "generation": s.generation,
        "restarts": s.restarts,
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

async fn fleet(State(state): State<Arc<AdminState>>) -> Json<Value> {
    let workers: Vec<Value> = engine::fleet_engines()
        .iter()
        .map(|e| {
            let load = e.load();
            json!({
                "worker": e.name(),
                "cache_blocks": e.cache_keys().len(),
                "block_size": state.cfg.engine.block_size,
                "num_running_reqs": load.num_running_reqs,
                "num_waiting_reqs": load.num_waiting_reqs,
                "num_waiting_uncached_tokens": load.num_waiting_uncached_tokens,
                "token_usage": load.token_usage,
                "cache_hit_rate": load.cache_hit_rate,
                "num_cached_blocks": load.num_cached_blocks,
                "num_preemptions": load.num_preemptions,
                "num_kv_batches": load.num_kv_batches,
            })
        })
        .collect();
    Json(json!({ "workers": workers }))
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
    Json(json!({ "records": rows, "next": next }))
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

/// Per worker, what it actually served over every admitted request.
async fn truth_served() -> Json<Value> {
    let mut totals: BTreeMap<String, (u64, u64, u64, u64)> = BTreeMap::new();
    for r in engine::records_since(0, usize::MAX) {
        let t = totals.entry(r.worker).or_insert((0, 0, 0, 0));
        t.0 += 1;
        t.1 += u64::from(r.prompt_tokens);
        t.2 += u64::from(r.cached_tokens);
        t.3 += u64::from(r.oracle_tokens);
    }
    let workers: Vec<Value> = totals
        .iter()
        .map(|(worker, (requests, prompt, cached, oracle))| {
            json!({
                "worker": worker,
                "requests": requests,
                "prompt_tokens": prompt,
                "cached_tokens": cached,
                "oracle_tokens": oracle,
            })
        })
        .collect();
    Json(json!({ "workers": workers }))
}
