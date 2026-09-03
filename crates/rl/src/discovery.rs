//! `GET /workers` and `GET /workers/{id}`: a projection of the registry view
//! plus the static capability table, with DP-aware ranks collapsed.

use std::{collections::HashMap, sync::Arc};

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;
use serde_json::json;

use crate::{
    capability::{capabilities_for, Capabilities},
    error::RlError,
    state::RlState,
    view::RlWorkerInfo,
};

/// One row of `GET /workers`.
#[derive(Debug, Serialize)]
pub struct WorkerEntry {
    pub id: String,
    pub url: String,
    pub base_url: String,
    pub engine: String,
    pub engine_version: Option<String>,
    pub model_id: String,
    pub worker_type: String,
    pub connection_mode: String,
    pub tp_size: Option<u64>,
    pub dp_size: Option<u64>,
    pub pp_size: Option<u64>,
    pub dp_ranks: usize,
    pub role: Option<String>,
    pub health: String,
    pub weight_version: Option<String>,
    pub labels: HashMap<String, String>,
    pub capabilities: Capabilities,
}

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
pub fn collapse(workers: Vec<RlWorkerInfo>) -> Vec<(RlWorkerInfo, usize)> {
    let mut groups: HashMap<String, (RlWorkerInfo, usize)> = HashMap::new();
    for w in workers {
        let key = if w.is_dp_aware {
            w.base_url.clone()
        } else {
            w.url.clone()
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

pub fn entry(info: &RlWorkerInfo, dp_ranks: usize) -> WorkerEntry {
    WorkerEntry {
        id: info.id.clone(),
        url: info.url.clone(),
        base_url: info.base_url.clone(),
        engine: engine_name(info),
        engine_version: info.labels.get("version").cloned(),
        model_id: info.model_id.clone(),
        worker_type: enum_str(&info.worker_type),
        connection_mode: enum_str(&info.connection_mode),
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
    let rows: Vec<WorkerEntry> = collapse(state.view.list())
        .iter()
        .map(|(w, n)| entry(w, *n))
        .collect();
    let total = rows.len();
    (
        StatusCode::OK,
        Json(json!({ "workers": rows, "total": total })),
    )
        .into_response()
}

pub(crate) async fn get_worker(
    State(state): State<Arc<RlState>>,
    Path(id): Path<String>,
) -> Response {
    match state.view.get(&id) {
        Some(w) => {
            let ranks = collapse(state.view.list())
                .into_iter()
                .find(|(row, _)| row.base_url == w.base_url)
                .map_or(1, |(_, n)| n);
            (StatusCode::OK, Json(entry(&w, ranks))).into_response()
        }
        None => RlError::WorkerNotFound(id).into_response(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::{body::Body, http::Request};
    use http_body_util::BodyExt;
    use openai_protocol::worker::RuntimeType;
    use tower::ServiceExt;

    use super::*;
    use crate::{
        config::RlConfig,
        state::RlState,
        testing::{worker, FakeView},
    };

    fn state(workers: Vec<RlWorkerInfo>) -> Arc<RlState> {
        Arc::new(RlState::new(Arc::new(FakeView(workers)), RlConfig::default(), false).unwrap())
    }

    #[test]
    fn collapse_groups_dp_ranks_and_sorts() {
        let ws = vec![
            worker("b1", "http://b:1@1", RuntimeType::Sglang),
            worker("a0", "http://a:1", RuntimeType::Sglang),
            worker("b0", "http://b:1@0", RuntimeType::Sglang),
            worker("b2", "http://b:1@2", RuntimeType::Sglang),
        ];
        let rows = collapse(ws);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].0.id, "a0");
        assert_eq!(rows[0].1, 1);
        assert_eq!(rows[1].0.id, "b0", "lowest rank wins");
        assert_eq!(rows[1].1, 3);
    }

    #[test]
    fn merged_labels_add_synthetic_keys_that_shadow() {
        let mut w = worker("w1", "http://a:1", RuntimeType::Vllm);
        w.labels.insert("engine".to_string(), "spoofed".to_string());
        w.labels.insert("role".to_string(), "reward".to_string());
        let m = merged_labels(&w);
        assert_eq!(m["engine"], "vllm");
        assert_eq!(m["id"], "w1");
        assert_eq!(m["url"], "http://a:1");
        assert_eq!(m["model"], "mock-model");
        assert_eq!(m["health"], "ready");
        assert_eq!(m["weight_version"], "default");
        assert_eq!(m["role"], "reward");
        assert_eq!(m["tp_size"], "1");
    }

    #[tokio::test]
    async fn list_and_get_workers() {
        let mut w = worker("w1", "http://a:1", RuntimeType::Sglang);
        w.labels.insert("tp_size".to_string(), "x".to_string());
        w.labels.insert("version".to_string(), "0.5.15".to_string());
        let app = crate::router::<()>(state(vec![w]));

        let resp = app
            .clone()
            .oneshot(Request::get("/workers").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body: serde_json::Value =
            serde_json::from_slice(&resp.into_body().collect().await.unwrap().to_bytes()).unwrap();
        assert_eq!(body["total"], 1);
        let e = &body["workers"][0];
        assert_eq!(e["id"], "w1");
        assert_eq!(e["engine"], "sglang");
        assert_eq!(e["engine_version"], "0.5.15");
        assert_eq!(
            e["tp_size"],
            serde_json::Value::Null,
            "garbage label -> null"
        );
        assert_eq!(e["dp_size"], 1);
        assert_eq!(e["dp_ranks"], 1);
        assert_eq!(e["health"], "ready");
        assert_eq!(e["weight_version"], "default");
        assert_eq!(e["capabilities"]["source"], "static");
        assert_eq!(e["capabilities"]["pause_modes"][0], "abort");

        let resp = app
            .clone()
            .oneshot(Request::get("/workers/w1").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let resp = app
            .oneshot(Request::get("/workers/nope").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }
}
