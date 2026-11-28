//! `GET /workers` and `GET /workers/{id}`: a projection of the registry view
//! plus the static capability table, with DP-aware ranks collapsed.

use std::{collections::HashMap, sync::Arc};

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use openai_protocol::rl::{RlWorkerEntry, RlWorkersResponse, RL_PROTOCOL_VERSION};
use serde::Serialize;

use crate::{capability::capabilities_for, error::RlError, state::RlState, view::RlWorkerInfo};

/// Serialize an enum through serde to get its canonical wire string.
pub(crate) fn enum_str<T: Serialize>(v: &T) -> String {
    serde_json::to_value(v)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_else(|| "unknown".to_string())
}

fn engine_name(info: &RlWorkerInfo) -> String {
    let s = enum_str(&info.runtime);
    if s == "unspecified" {
        "unknown".to_string()
    } else {
        s
    }
}

fn int_label(labels: &HashMap<String, String>, key: &str) -> Option<u64> {
    labels.get(key).and_then(|v| v.trim().parse().ok())
}

fn dp_rank(url: &str) -> usize {
    url.rsplit_once('@')
        .and_then(|(_, r)| r.parse().ok())
        .unwrap_or(0)
}

/// Collapse DP-aware ranks that share a `base_url` into one row (lowest rank
/// wins) and sort rows by `base_url`. The count is the number of ranks.
///
/// Keyed on `(is_dp_aware, base_url_or_url)` rather than a plain string: a
/// non-DP worker's `url` and a DP group's `base_url` can be textually
/// identical (e.g. a non-DP worker at `http://b:1` alongside DP ranks
/// `http://b:1@0`, `http://b:1@1`), and those are distinct registry
/// entries that must not be merged.
pub fn collapse(workers: Vec<RlWorkerInfo>) -> Vec<(RlWorkerInfo, usize)> {
    let mut groups: HashMap<(bool, String), (RlWorkerInfo, usize)> = HashMap::new();
    for w in workers {
        let key = if w.is_dp_aware {
            (true, w.base_url.clone())
        } else {
            (false, w.url.clone())
        };
        match groups.get_mut(&key) {
            None => {
                groups.insert(key, (w, 1));
            }
            Some((kept, n)) => {
                *n += 1;
                if dp_rank(&w.url) < dp_rank(&kept.url) {
                    *kept = w;
                }
            }
        }
    }
    let mut rows: Vec<(RlWorkerInfo, usize)> = groups.into_values().collect();
    rows.sort_by(|a, b| a.0.base_url.cmp(&b.0.base_url).then(a.0.url.cmp(&b.0.url)));
    rows
}

/// Labels plus synthetic keys, for selector matching. Synthetic keys shadow.
pub fn merged_labels(info: &RlWorkerInfo) -> HashMap<String, String> {
    let mut m = info.labels.clone();
    m.insert("id".to_string(), info.id.clone());
    m.insert("url".to_string(), info.url.clone());
    m.insert("base_url".to_string(), info.base_url.clone());
    m.insert("engine".to_string(), engine_name(info));
    m.insert("model".to_string(), info.model_id.clone());
    m.insert("worker_type".to_string(), enum_str(&info.worker_type));
    m.insert(
        "connection_mode".to_string(),
        enum_str(&info.connection_mode),
    );
    m.insert("health".to_string(), enum_str(&info.status));
    if let Some(v) = info.labels.get("weight_version") {
        m.insert("weight_version".to_string(), v.clone());
    }
    if let Some(v) = info.labels.get("role") {
        m.insert("role".to_string(), v.clone());
    }
    m
}

/// One row of `GET /workers` for `info`, with `dp_ranks` collapsed into it.
pub fn entry(info: &RlWorkerInfo, dp_ranks: usize) -> RlWorkerEntry {
    RlWorkerEntry {
        id: info.id.clone(),
        url: info.url.clone(),
        base_url: info.base_url.clone(),
        engine: engine_name(info),
        engine_version: info.labels.get("version").cloned(),
        model_id: info.model_id.clone(),
        worker_type: enum_str(&info.worker_type),
        connection_mode: enum_str(&info.connection_mode),
        control_url: info.control_url.clone(),
        engine_control_calls: info.control_calls.is_some(),
        tp_size: int_label(&info.labels, "tp_size"),
        dp_size: int_label(&info.labels, "dp_size"),
        pp_size: int_label(&info.labels, "pp_size"),
        dp_ranks,
        role: info.labels.get("role").cloned(),
        health: enum_str(&info.status),
        weight_version: info.labels.get("weight_version").cloned(),
        labels: info.labels.clone(),
        capabilities: capabilities_for(info.runtime, &info.labels),
    }
}

pub(crate) async fn list_workers(State(state): State<Arc<RlState>>) -> Response {
    let workers: Vec<RlWorkerEntry> = collapse(state.view.list())
        .iter()
        .map(|(w, n)| entry(w, *n))
        .collect();
    let total = workers.len();
    (
        StatusCode::OK,
        Json(RlWorkersResponse {
            protocol_version: RL_PROTOCOL_VERSION,
            workers,
            total,
        }),
    )
        .into_response()
}

pub(crate) async fn get_worker(
    State(state): State<Arc<RlState>>,
    Path(id): Path<String>,
) -> Response {
    match state.view.get(&id) {
        Some(w) => {
            let ranks = if w.is_dp_aware {
                state
                    .view
                    .list()
                    .iter()
                    .filter(|other| other.is_dp_aware && other.base_url == w.base_url)
                    .count()
                    .max(1)
            } else {
                1
            };
            (StatusCode::OK, Json(entry(&w, ranks))).into_response()
        }
        None => RlError::WorkerNotFound(id).into_response(),
    }
}

#[cfg(test)]
#[path = "../tests/unit/discovery/tests.rs"]
mod tests;
