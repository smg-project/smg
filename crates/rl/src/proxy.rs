//! Verbatim proxy of one engine-native route to one worker.

use std::{sync::Arc, time::Instant};

use axum::{
    body::Bytes,
    extract::{Path, RawQuery, State},
    http::{
        header::{AUTHORIZATION, CONTENT_TYPE},
        HeaderMap, HeaderName, HeaderValue, Method, StatusCode,
    },
    response::{IntoResponse, Response},
    Json,
};
use openai_protocol::worker::ConnectionMode;
use serde_json::{json, Value};
use tracing::info;

use crate::{
    discovery::enum_str,
    error::RlError,
    metrics::{op_label, record_control_call},
    path::{passthrough_query, validate_engine_path},
    state::RlState,
    view::RlWorkerInfo,
};

/// Largest response body kept in the envelope (bytes).
pub(crate) const BODY_CAP: usize = 1 << 20;

/// Request headers copied from the caller to the engine. `authorization` is
/// deliberately absent: it authenticates the caller to SMG, not to the engine.
const FORWARDED_HEADERS: &[&str] = &["x-request-id", "traceparent", "tracestate"];

/// Everything needed to replay a caller's request against one engine.
#[derive(Debug, Clone)]
pub struct ProxyRequest {
    pub method: Method,
    pub path: String,
    pub query: Option<String>,
    pub content_type: Option<HeaderValue>,
    pub forward: Vec<(HeaderName, HeaderValue)>,
    pub body: Bytes,
}

impl ProxyRequest {
    pub fn from_parts(
        method: Method,
        raw_path: &str,
        raw_query: Option<&str>,
        headers: &HeaderMap,
        body: Bytes,
    ) -> Result<Self, RlError> {
        let path = validate_engine_path(raw_path)?;
        let content_type = headers
            .get(CONTENT_TYPE)
            .cloned()
            .or_else(|| (!body.is_empty()).then(|| HeaderValue::from_static("application/json")));
        let forward = FORWARDED_HEADERS
            .iter()
            .filter_map(|name| {
                let hn = HeaderName::from_static(name);
                headers.get(&hn).map(|v| (hn, v.clone()))
            })
            .collect();
        Ok(Self {
            method,
            path,
            query: passthrough_query(raw_query),
            content_type,
            forward,
            body,
        })
    }

    fn url_for(&self, worker: &RlWorkerInfo) -> String {
        let base = worker.base_url.trim_end_matches('/');
        match &self.query {
            Some(q) => format!("{base}/{}?{q}", self.path),
            None => format!("{base}/{}", self.path),
        }
    }
}

/// One completed engine call (any HTTP status).
#[derive(Debug, Clone)]
pub struct CallOutcome {
    pub worker_id: String,
    pub url: String,
    pub status: u16,
    pub latency_ms: u64,
    pub body: Value,
    pub body_truncated: bool,
}

impl CallOutcome {
    pub fn to_json(&self) -> Value {
        let mut v = json!({
            "worker_id": self.worker_id,
            "url": self.url,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "body": self.body,
        });
        if self.body_truncated {
            v["body_truncated"] = json!(true);
        }
        v
    }

    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }
}

/// JSON when the content type says JSON and the bytes parse; otherwise text,
/// capped at `BODY_CAP`.
pub(crate) fn parse_body(content_type: Option<&HeaderValue>, bytes: &[u8]) -> (Value, bool) {
    let is_json = content_type
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.contains("json"));
    if is_json {
        if let Ok(v) = serde_json::from_slice::<Value>(bytes) {
            return (v, false);
        }
    }
    let truncated = bytes.len() > BODY_CAP;
    let slice = if truncated { &bytes[..BODY_CAP] } else { bytes };
    (
        Value::String(String::from_utf8_lossy(slice).into_owned()),
        truncated,
    )
}

/// Send `req` to `worker`. `Err` only for transport failures; an upstream
/// 4xx/5xx is a successful proxy with that status in the outcome.
pub async fn call_worker(
    state: &RlState,
    worker: &RlWorkerInfo,
    req: &ProxyRequest,
) -> Result<CallOutcome, RlError> {
    if worker.connection_mode != ConnectionMode::Http {
        return Err(RlError::UnsupportedConnectionMode {
            worker_id: worker.id.clone(),
            url: worker.url.clone(),
            mode: enum_str(&worker.connection_mode),
        });
    }
    let url = req.url_for(worker);
    let mut builder = state.client.request(req.method.clone(), &url);
    if let Some(ct) = &req.content_type {
        builder = builder.header(CONTENT_TYPE, ct.clone());
    }
    for (name, value) in &req.forward {
        builder = builder.header(name.clone(), value.clone());
    }
    if let Some(key) = &worker.api_key {
        builder = builder.header(AUTHORIZATION, format!("Bearer {key}"));
    }
    if !req.body.is_empty() {
        builder = builder.body(req.body.clone());
    }

    let op = op_label(&req.path);
    let started = Instant::now();
    let response = match builder.send().await {
        Ok(r) => r,
        Err(e) => {
            let elapsed = started.elapsed();
            let err = if e.is_timeout() {
                record_control_call(op, "timeout", elapsed);
                RlError::UpstreamTimeout {
                    worker_id: worker.id.clone(),
                    url: worker.url.clone(),
                    timeout_secs: state.config.control_timeout_secs,
                }
            } else {
                record_control_call(op, "unreachable", elapsed);
                RlError::UpstreamUnreachable {
                    worker_id: worker.id.clone(),
                    url: worker.url.clone(),
                    message: e.to_string(),
                }
            };
            return Err(err);
        }
    };
    let status = response.status().as_u16();
    let content_type = response.headers().get(CONTENT_TYPE).cloned();
    let bytes = response.bytes().await.map_err(|e| {
        record_control_call(op, "unreachable", started.elapsed());
        RlError::UpstreamUnreachable {
            worker_id: worker.id.clone(),
            url: worker.url.clone(),
            message: format!("reading response body: {e}"),
        }
    })?;
    let elapsed = started.elapsed();
    let (body, body_truncated) = parse_body(content_type.as_ref(), &bytes);
    let outcome = CallOutcome {
        worker_id: worker.id.clone(),
        url: worker.url.clone(),
        status,
        latency_ms: elapsed.as_millis() as u64,
        body,
        body_truncated,
    };
    record_control_call(
        op,
        if outcome.is_success() {
            "ok"
        } else {
            "upstream_error"
        },
        elapsed,
    );
    info!(
        target: "smg_rl",
        worker_id = %worker.id, url = %worker.url, method = %req.method,
        path = %req.path, status, latency_ms = outcome.latency_ms, "rl.proxy"
    );
    Ok(outcome)
}

pub(crate) async fn proxy_handler(
    State(state): State<Arc<RlState>>,
    Path((id, raw_path)): Path<(String, String)>,
    method: Method,
    RawQuery(raw_query): RawQuery,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let req =
        match ProxyRequest::from_parts(method, &raw_path, raw_query.as_deref(), &headers, body) {
            Ok(r) => r,
            Err(e) => return e.into_response(),
        };
    let Some(worker) = state.view.get(&id) else {
        return RlError::WorkerNotFound(id).into_response();
    };
    match call_worker(&state, &worker, &req).await {
        Ok(outcome) => {
            let status = StatusCode::from_u16(outcome.status).unwrap_or(StatusCode::BAD_GATEWAY);
            (status, Json(outcome.to_json())).into_response()
        }
        Err(e) => e.into_response(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::{body::Body, http::Request};
    use http_body_util::BodyExt;
    use openai_protocol::worker::{ConnectionMode, RuntimeType};
    use serde_json::json;
    use tower::ServiceExt;

    use super::*;
    use crate::{
        config::RlConfig,
        testing::{worker, FakeEngine, FakeView},
    };

    fn state(workers: Vec<RlWorkerInfo>, timeout_secs: u64) -> Arc<RlState> {
        let cfg = RlConfig {
            enabled: true,
            control_timeout_secs: timeout_secs,
            fanout_concurrency: 4,
        };
        Arc::new(RlState::new(Arc::new(FakeView(workers)), cfg, false).unwrap())
    }

    async fn json_body(resp: Response) -> Value {
        serde_json::from_slice(&resp.into_body().collect().await.unwrap().to_bytes()).unwrap()
    }

    #[tokio::test]
    async fn forwards_method_path_query_body_and_worker_key_only() {
        let engine = FakeEngine::start(StatusCode::OK, json!({"success": true}), 0).await;
        let mut w = worker("w1", &engine.url, RuntimeType::Vllm);
        w.api_key = Some("engine-secret".to_string());
        let app = crate::router::<()>(state(vec![w], 5));

        let req = Request::post("/workers/w1/engine/pause?mode=keep&selector=engine%3Dvllm")
            .header("content-type", "application/json")
            .header("authorization", "Bearer caller-token")
            .header("x-request-id", "req-1")
            .header("x-smg-routing-key", "leak")
            .body(Body::from(r#"{"raw": "bytes"}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json_body(resp).await;
        assert_eq!(body["worker_id"], "w1");
        assert_eq!(body["status"], 200);
        assert_eq!(body["body"]["success"], true);
        assert!(body["latency_ms"].is_u64());

        let seen = engine.seen();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].method, Method::POST);
        assert_eq!(seen[0].path, "/pause");
        assert_eq!(seen[0].query.as_deref(), Some("mode=keep"));
        assert_eq!(&seen[0].body[..], br#"{"raw": "bytes"}"#);
        assert_eq!(seen[0].headers["authorization"], "Bearer engine-secret");
        assert_eq!(seen[0].headers["x-request-id"], "req-1");
        assert!(seen[0].headers.get("x-smg-routing-key").is_none());
    }

    #[tokio::test]
    async fn defaults_content_type_when_body_has_no_header() {
        let engine = FakeEngine::start(StatusCode::OK, json!({}), 0).await;
        let app = crate::router::<()>(state(
            vec![worker("w1", &engine.url, RuntimeType::Sglang)],
            5,
        ));

        let req = Request::post("/workers/w1/engine/update_weight_version")
            .header(
                "traceparent",
                "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
            )
            .body(Body::from(r#"{"new_version":"3"}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let seen = engine.seen();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].headers["content-type"], "application/json");
        assert_eq!(
            seen[0].headers["traceparent"],
            "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        );
        assert_eq!(&seen[0].body[..], br#"{"new_version":"3"}"#);
    }

    #[tokio::test]
    async fn get_is_forwarded_and_text_bodies_are_wrapped() {
        let engine = FakeEngine::start(StatusCode::OK, json!("plain"), 0).await;
        let app = crate::router::<()>(state(
            vec![worker("w1", &engine.url, RuntimeType::Sglang)],
            5,
        ));
        let resp = app
            .oneshot(
                Request::get("/workers/w1/engine/server_info")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(json_body(resp).await["body"], "plain");
        assert_eq!(engine.seen()[0].method, Method::GET);
    }

    #[tokio::test]
    async fn upstream_status_is_mirrored_not_translated() {
        let engine = FakeEngine::start(
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({"error": "boom"}),
            0,
        )
        .await;
        let app = crate::router::<()>(state(
            vec![worker("w1", &engine.url, RuntimeType::Sglang)],
            5,
        ));
        let resp = app
            .oneshot(
                Request::post("/workers/w1/engine/flush_cache")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = json_body(resp).await;
        assert_eq!(body["status"], 500);
        assert_eq!(body["body"]["error"], "boom");
    }

    #[tokio::test]
    async fn errors_404_422_400_405() {
        let engine = FakeEngine::start(StatusCode::OK, json!({}), 0).await;
        let mut grpc = worker("g1", &engine.url, RuntimeType::Sglang);
        grpc.connection_mode = ConnectionMode::Grpc;
        let app = crate::router::<()>(state(
            vec![worker("w1", &engine.url, RuntimeType::Sglang), grpc],
            5,
        ));

        let r = app
            .clone()
            .oneshot(
                Request::post("/workers/zz/engine/pause")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::NOT_FOUND);
        assert_eq!(json_body(r).await["error"], "worker_not_found");

        let r = app
            .clone()
            .oneshot(
                Request::post("/workers/g1/engine/pause")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(json_body(r).await["error"], "unsupported_connection_mode");

        let r = app
            .clone()
            .oneshot(
                Request::post("/workers/w1/engine/a/b/c/d/e")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::BAD_REQUEST);
        assert_eq!(json_body(r).await["error"], "invalid_engine_path");

        let r = app
            .oneshot(
                Request::delete("/workers/w1/engine/pause")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::METHOD_NOT_ALLOWED);
        assert!(engine.seen().is_empty());
    }

    #[tokio::test]
    async fn unreachable_and_timeout_map_to_502_and_504() {
        let dead = worker("d1", "http://127.0.0.1:9", RuntimeType::Sglang);
        let slow_engine = FakeEngine::start(StatusCode::OK, json!({}), 1500).await;
        let slow = worker("s1", &slow_engine.url, RuntimeType::Sglang);
        let app = crate::router::<()>(state(vec![dead, slow], 1));

        let r = app
            .clone()
            .oneshot(
                Request::post("/workers/d1/engine/pause")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(json_body(r).await["error"], "upstream_unreachable");

        let r = app
            .oneshot(
                Request::post("/workers/s1/engine/pause")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::GATEWAY_TIMEOUT);
        assert_eq!(json_body(r).await["error"], "upstream_timeout");
    }

    #[test]
    fn large_text_bodies_are_truncated() {
        let big = vec![b'x'; BODY_CAP + 10];
        let (v, truncated) = parse_body(None, &big);
        assert!(truncated);
        assert_eq!(v.as_str().unwrap().len(), BODY_CAP);
    }
}
