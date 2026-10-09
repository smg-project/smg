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
//!   `.../restart-publisher`, `.../pause`, `.../resume`,
//!   `.../fail?status=S[&count=N|&secs=T][&after_tokens=K][&stall_ms=M]` and
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
    config::Config,
    engine::{self, Engine, FaultScope, RequestFault},
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
        .route(
            "/admin/fault/{worker}/restart-publisher",
            post(fault_restart_publisher),
        )
        .route("/admin/fault/{worker}/pause", post(fault_pause))
        .route("/admin/fault/{worker}/resume", post(fault_resume))
        .route("/admin/fault/{worker}/fail", post(fault_fail))
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
        "paused": s.paused,
        "generation": s.generation,
        "restarts": s.restarts,
        "fail_status": fail.map_or(0, |f| f.status),
        "fail_pending": fail_pending,
        "fail_secs_left": fail_secs_left,
        "fail_after_tokens": fail.and_then(|f| f.after_tokens),
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
/// the stream with S. `status=0` alone clears; with `stall_ms` it stalls and
/// then serves.
fn fail_query(q: &HashMap<String, String>) -> Result<Option<RequestFault>, String> {
    let status: u16 = optional(q, "status")?.unwrap_or(0);
    let count: Option<u32> = optional(q, "count")?;
    let secs: Option<f64> = optional(q, "secs")?;
    let after_tokens: Option<u32> = optional(q, "after_tokens")?;
    let stall_ms: u64 = optional(q, "stall_ms")?.unwrap_or(0);
    if status != 0 && !(400..=599).contains(&status) {
        return Err(format!(
            "status must be 0 or an error status (400-599), got {status}"
        ));
    }
    if status == 0 && after_tokens.is_some() {
        return Err("after_tokens needs a status to cut the stream with".to_string());
    }
    if status == 0 && stall_ms == 0 {
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
        scope,
    }))
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
