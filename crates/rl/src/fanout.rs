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
                    RlError::UnsupportedConnectionMode { mode, .. } => Some(mode.clone()),
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
mod tests {
    use std::sync::Arc;

    use axum::{body::Body, http::Request};
    use http_body_util::BodyExt;
    use openai_protocol::worker::{ConnectionMode, RuntimeType};
    use serde_json::{json, Value};
    use tower::ServiceExt;

    use super::*;
    use crate::{
        config::RlConfig,
        testing::{worker, FakeEngine, FakeView},
    };

    fn state(workers: Vec<RlWorkerInfo>, concurrency: usize) -> Arc<RlState> {
        let cfg = RlConfig {
            enabled: true,
            control_timeout_secs: 5,
            fanout_concurrency: concurrency,
        };
        Arc::new(RlState::new(Arc::new(FakeView(workers)), cfg))
    }

    async fn json_body(resp: Response) -> Value {
        serde_json::from_slice(&resp.into_body().collect().await.unwrap().to_bytes()).unwrap()
    }

    #[tokio::test]
    async fn all_success_is_200_and_each_engine_called_once() {
        let e1 = FakeEngine::start(StatusCode::OK, json!({"ok": 1}), 0).await;
        let e2 = FakeEngine::start(StatusCode::OK, json!({"ok": 2}), 0).await;
        let app = crate::router::<()>(state(
            vec![
                worker("w1", &e1.url, RuntimeType::Sglang),
                worker("w2", &e2.url, RuntimeType::Sglang),
                worker("v1", &format!("{}/v", e2.url), RuntimeType::Vllm),
            ],
            8,
        ));
        let req = Request::post("/engine/pause_generation?selector=engine%3Dsglang")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"mode":"abort"}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json_body(resp).await;
        assert_eq!(body["total"], 2);
        assert_eq!(body["succeeded"], 2);
        assert_eq!(body["failed"].as_array().unwrap().len(), 0);
        assert_eq!(body["results"]["w1"]["body"]["ok"], 1);
        assert_eq!(body["results"]["w2"]["status"], 200);
        assert_eq!(e1.seen().len(), 1);
        assert_eq!(
            e2.seen().len(),
            1,
            "vllm worker sharing e2 was not selected"
        );
        assert_eq!(&e1.seen()[0].body[..], br#"{"mode":"abort"}"#);
        assert_eq!(
            e1.seen()[0].query,
            None,
            "selector stripped from engine query"
        );
    }

    #[tokio::test]
    async fn one_failure_is_207_and_names_the_worker() {
        let good = FakeEngine::start(StatusCode::OK, json!({}), 0).await;
        let bad = FakeEngine::start(StatusCode::INTERNAL_SERVER_ERROR, json!({"e": 1}), 0).await;
        let mut grpc = worker("g1", &format!("{}/grpc", good.url), RuntimeType::Sglang);
        grpc.connection_mode = ConnectionMode::Grpc;
        let app = crate::router::<()>(state(
            vec![
                worker("w1", &good.url, RuntimeType::Sglang),
                worker("w2", &bad.url, RuntimeType::Sglang),
                worker("w3", "http://127.0.0.1:9", RuntimeType::Sglang),
                grpc,
            ],
            8,
        ));
        let req = Request::post("/engine/flush_cache?selector=engine%3Dsglang")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::MULTI_STATUS);
        let body = json_body(resp).await;
        assert_eq!(body["total"], 4);
        assert_eq!(body["succeeded"], 1);
        let failed: Vec<&str> = body["failed"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| f["worker_id"].as_str().unwrap())
            .collect();
        assert_eq!(failed, ["g1", "w2", "w3"]);
        let by_id = |id: &str| {
            body["failed"]
                .as_array()
                .unwrap()
                .iter()
                .find(|f| f["worker_id"] == id)
                .cloned()
                .unwrap()
        };
        assert_eq!(by_id("w2")["error"], "upstream_error");
        assert_eq!(by_id("w2")["status"], 500);
        assert_eq!(by_id("w3")["error"], "upstream_unreachable");
        assert_eq!(by_id("g1")["error"], "unsupported_connection_mode");
        assert_eq!(
            body["results"]["w2"]["status"], 500,
            "failed also in results"
        );
        assert!(
            body["results"].get("w3").is_none(),
            "no outcome for transport failure"
        );
    }

    #[tokio::test]
    async fn selector_errors_are_400() {
        let e = FakeEngine::start(StatusCode::OK, json!({}), 0).await;
        let app = crate::router::<()>(state(vec![worker("w1", &e.url, RuntimeType::Sglang)], 8));

        let r = app
            .clone()
            .oneshot(Request::post("/engine/pause").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::BAD_REQUEST);
        assert_eq!(json_body(r).await["error"], "selector_required");

        let r = app
            .clone()
            .oneshot(
                Request::post("/engine/pause?selector=")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(json_body(r).await["error"], "selector_required");

        let r = app
            .clone()
            .oneshot(
                Request::post("/engine/pause?selector=engine%3D%3D")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::BAD_REQUEST);
        assert_eq!(json_body(r).await["error"], "invalid_selector");

        let r = app
            .oneshot(
                Request::post("/engine/pause?selector=engine%3Dtrtllm")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::BAD_REQUEST);
        let b = json_body(r).await;
        assert_eq!(b["error"], "no_workers_match");
        assert_eq!(b["selector"], "engine=trtllm");
        assert!(e.seen().is_empty());
    }

    #[tokio::test]
    async fn dp_ranks_collapse_to_one_call_and_url_selector_targets_one() {
        let e = FakeEngine::start(StatusCode::OK, json!({}), 0).await;
        let app = crate::router::<()>(state(
            vec![
                worker("r0", &format!("{}@0", e.url), RuntimeType::Sglang),
                worker("r1", &format!("{}@1", e.url), RuntimeType::Sglang),
                worker("r2", &format!("{}@2", e.url), RuntimeType::Sglang),
            ],
            8,
        ));
        let sel = format!("url={}@0", e.url);
        let uri = format!("/engine/pause?selector={}", urlenc(&sel));
        let r = app
            .oneshot(Request::post(uri).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::OK);
        assert_eq!(json_body(r).await["total"], 1);
        assert_eq!(e.seen().len(), 1);
    }

    #[tokio::test]
    async fn concurrency_is_capped() {
        let e = FakeEngine::start(StatusCode::OK, json!({}), 150).await;
        let workers: Vec<RlWorkerInfo> = (0..6)
            .map(|i| {
                worker(
                    &format!("w{i}"),
                    &format!("{}/{i}", e.url),
                    RuntimeType::Sglang,
                )
            })
            .collect();
        // base_url differs per worker (path suffix) so they are 6 targets on one engine.
        let app = crate::router::<()>(state(workers, 2));
        let r = app
            .oneshot(
                Request::post("/engine/pause?selector=engine%3Dsglang")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::OK);
        assert_eq!(e.seen().len(), 6);
        assert!(e.peak_concurrency() <= 2, "peak {}", e.peak_concurrency());
    }

    fn urlenc(s: &str) -> String {
        let mut out = String::new();
        for b in s.bytes() {
            match b {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                    out.push(b as char);
                }
                _ => out.push_str(&format!("%{b:02X}")),
            }
        }
        out
    }
}
