//! Fan a proxied engine call out to every worker matching a selector.

use std::{sync::Arc, time::Instant};

use axum::{
    body::Bytes,
    extract::{Path, Query, RawQuery, State},
    http::{HeaderMap, Method, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use futures::{stream, StreamExt};
use openai_protocol::rl::{RlCallOutcome, RlFailedCall, RlFanoutResponse};
use serde::Deserialize;
use tracing::info;

use crate::{
    discovery::{collapse, merged_labels},
    error::RlError,
    metrics::record_fanout,
    proxy::{call_worker, ProxyRequest},
    selector::Selector,
    state::RlState,
    view::{RlWorkerInfo, RlWorkerView},
};

#[derive(Deserialize)]
pub(crate) struct FanoutQuery {
    selector: Option<String>,
}

/// DP-collapsed workers matching `selector`, sorted by base URL.
pub fn resolve_targets(view: &dyn RlWorkerView, selector: &Selector) -> Vec<RlWorkerInfo> {
    collapse(view.list())
        .into_iter()
        .map(|(w, _)| w)
        .filter(|w| selector.matches(&merged_labels(w)))
        .collect()
}

/// HTTP status of a fan-out envelope. Partial success is never reported
/// as success.
pub fn fanout_status(report: &RlFanoutResponse) -> StatusCode {
    if report.failed.is_empty() {
        StatusCode::OK
    } else {
        StatusCode::MULTI_STATUS
    }
}

/// Run `req` against every target with bounded concurrency on the current
/// task (no spawn: a client disconnect cancels outstanding engine calls).
pub async fn run_fanout(
    state: &RlState,
    targets: Vec<RlWorkerInfo>,
    req: &ProxyRequest,
) -> RlFanoutResponse {
    let total = targets.len();
    let outcomes: Vec<(RlWorkerInfo, Result<RlCallOutcome, RlError>)> = stream::iter(targets)
        .map(|w| async move {
            let r = call_worker(state, &w, req).await;
            (w, r)
        })
        .buffer_unordered(state.config.fanout_concurrency.max(1))
        .collect()
        .await;

    let mut report = RlFanoutResponse {
        total,
        ..RlFanoutResponse::default()
    };
    for (w, r) in outcomes {
        match r {
            Ok(o) if o.is_success() => {
                report.succeeded += 1;
                report.results.insert(w.id.clone(), o);
            }
            Ok(o) => {
                report.failed.push(RlFailedCall {
                    worker_id: w.id.clone(),
                    url: w.url.clone(),
                    error: "upstream_error".to_string(),
                    message: format!("HTTP {}", o.status),
                    status: Some(o.status),
                    connection_mode: None,
                });
                report.results.insert(w.id.clone(), o);
            }
            Err(e) => {
                let connection_mode = match &e {
                    RlError::NoControlEndpoint {
                        connection_mode, ..
                    } => Some(connection_mode.clone()),
                    _ => None,
                };
                report.failed.push(RlFailedCall {
                    worker_id: w.id.clone(),
                    url: w.url.clone(),
                    error: e.code().to_string(),
                    message: e.to_string(),
                    status: None,
                    connection_mode,
                });
            }
        }
    }
    report.failed.sort_by(|a, b| a.worker_id.cmp(&b.worker_id));
    report
}

pub(crate) async fn fanout_handler(
    State(state): State<Arc<RlState>>,
    Path(raw_path): Path<String>,
    method: Method,
    Query(FanoutQuery { selector }): Query<FanoutQuery>,
    RawQuery(raw_query): RawQuery,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let started = Instant::now();
    let selector = match selector.as_deref().map(str::trim) {
        Some(s) if !s.is_empty() => match Selector::parse(s) {
            Ok(sel) => sel,
            Err(e) => return e.into_response(),
        },
        _ => return RlError::SelectorRequired.into_response(),
    };
    let req =
        match ProxyRequest::from_parts(method, &raw_path, raw_query.as_deref(), &headers, body) {
            Ok(r) => r,
            Err(e) => return e.into_response(),
        };
    let targets = resolve_targets(state.view.as_ref(), &selector);
    if targets.is_empty() {
        record_fanout("no_match", started.elapsed());
        return RlError::NoWorkersMatch(selector.source().to_string()).into_response();
    }
    let report = run_fanout(&state, targets, &req).await;
    let elapsed = started.elapsed();
    record_fanout(
        if report.failed.is_empty() {
            "ok"
        } else {
            "partial"
        },
        elapsed,
    );
    info!(
        target: "smg_rl",
        path = %req.path, selector = %selector.source(), total = report.total,
        succeeded = report.succeeded, failed = report.failed.len(),
        latency_ms = elapsed.as_millis() as u64, "rl.fanout"
    );
    (fanout_status(&report), Json(report)).into_response()
}

#[cfg(test)]
#[path = "../tests/unit/fanout/tests.rs"]
mod tests;
