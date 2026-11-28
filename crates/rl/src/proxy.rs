//! Verbatim proxy of one engine-native route to one worker.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

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
use openai_protocol::rl::RlCallOutcome;
use serde_json::Value;
use tracing::info;

use crate::{
    calls::{route_answer, CallFailure, RlControlCalls},
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

    fn url_for(&self, base: &str) -> String {
        let base = base.trim_end_matches('/');
        match &self.query {
            Some(q) => format!("{base}/{}?{q}", self.path),
            None => format!("{base}/{}", self.path),
        }
    }
}

/// JSON when the content type says JSON and the bytes parse; otherwise text,
/// capped at `BODY_CAP`. When `truncated_by_read` is true, the bytes were
/// already cut off while reading the response stream, so JSON parsing is
/// skipped (the bytes are not valid JSON by construction) and the result is
/// always reported truncated.
pub(crate) fn parse_body(
    content_type: Option<&HeaderValue>,
    bytes: &[u8],
    truncated_by_read: bool,
) -> (Value, bool) {
    if truncated_by_read {
        return (
            Value::String(String::from_utf8_lossy(bytes).into_owned()),
            true,
        );
    }
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

/// The worker has nowhere to send control calls, or nothing to send them with.
fn no_control_endpoint(worker: &RlWorkerInfo, message: &str) -> RlError {
    RlError::NoControlEndpoint {
        worker_id: worker.id.clone(),
        url: worker.url.clone(),
        connection_mode: enum_str(&worker.connection_mode),
        message: message.to_string(),
    }
}

/// Send `req` to `worker`. `Err` only for transport failures; an upstream
/// 4xx/5xx is a successful proxy with that status in the outcome.
pub async fn call_worker(
    state: &RlState,
    worker: &RlWorkerInfo,
    req: &ProxyRequest,
) -> Result<RlCallOutcome, RlError> {
    // Control is independent of the data transport: what matters is that the
    // gateway knows an HTTP control endpoint and has a client to reach it, or,
    // without one, that the worker's engine connection carries the calls.
    let Some(base) = worker.control_url.as_deref() else {
        if let Some(calls) = &worker.control_calls {
            return call_over_engine_connection(state, worker, calls.as_ref(), req).await;
        }
        return Err(no_control_endpoint(
            worker,
            "no `rl.control_url` label: set it at registration or through the worker update route (a TokenSpeed gRPC engine advertises it when launched with a routable --rl-control-host; a TokenSpeed engine on the direct ZMQ lane carries the calls itself)",
        ));
    };
    let client = worker.control_client.as_ref().map_err(|e| {
        no_control_endpoint(
            worker,
            &format!("no HTTP client for the control endpoint: {e}"),
        )
    })?;
    let url = req.url_for(base);
    // The worker's client carries the gateway's request timeout; a refit
    // can outlive it, so the control deadline is set per request, as the
    // gateway's own flush/profile admin calls do.
    let mut builder = client
        .request(req.method.clone(), &url)
        .timeout(Duration::from_secs(state.config.control_timeout_secs));
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
    let mut response = match builder.send().await {
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
    // Bounded streaming read: stop once BODY_CAP bytes are collected instead
    // of buffering an unbounded upstream body.
    let mut collected: Vec<u8> = Vec::new();
    let mut truncated_by_read = false;
    loop {
        let chunk = match response.chunk().await {
            Ok(Some(c)) => c,
            Ok(None) => break,
            Err(e) => {
                record_control_call(op, "unreachable", started.elapsed());
                return Err(RlError::UpstreamUnreachable {
                    worker_id: worker.id.clone(),
                    url: worker.url.clone(),
                    message: format!("reading response body: {e}"),
                });
            }
        };
        let remaining = BODY_CAP.saturating_sub(collected.len());
        if remaining == 0 {
            truncated_by_read = true;
            break;
        }
        if chunk.len() > remaining {
            collected.extend_from_slice(&chunk[..remaining]);
            truncated_by_read = true;
            break;
        }
        collected.extend_from_slice(&chunk);
    }
    let elapsed = started.elapsed();
    let (body, body_truncated) = parse_body(content_type.as_ref(), &collected, truncated_by_read);
    let outcome = RlCallOutcome {
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
        worker_id = %worker.id, url = %worker.url, control_url = %base,
        method = %req.method, path = %req.path, status,
        latency_ms = outcome.latency_ms, "rl.proxy"
    );
    Ok(outcome)
}

/// Send `req` to `worker` as a control call on its engine connection: the
/// route is the method, its JSON body (a map, or nothing) the argument, and
/// the ranks' replies come back in the route's own shape (see
/// [`route_answer`]). The connection carries the engine's POST operations
/// only: any other method, or a route below one segment, has no call.
async fn call_over_engine_connection(
    state: &RlState,
    worker: &RlWorkerInfo,
    calls: &dyn RlControlCalls,
    req: &ProxyRequest,
) -> Result<RlCallOutcome, RlError> {
    if req.method != Method::POST || req.path.contains('/') {
        return Err(no_control_endpoint(
            worker,
            &format!(
                "its engine connection carries the engine's POST control operations only, not \
                 {} /{}; label `rl.control_url` to reach the engine's own control app",
                req.method, req.path
            ),
        ));
    }
    // An empty object is the route's defaults, as no body is (callers send
    // `{}` to the routes whose framework requires a JSON body), so it is a
    // call without an argument.
    let body = if req.body.is_empty() {
        None
    } else {
        match serde_json::from_slice::<Value>(&req.body) {
            Ok(Value::Object(fields)) if fields.is_empty() => None,
            Ok(body @ Value::Object(_)) => Some(body),
            Ok(other) => {
                return Err(RlError::InvalidBody(format!(
                    "a control call takes a JSON object, got {other}"
                )))
            }
            Err(e) => return Err(RlError::InvalidBody(e.to_string())),
        }
    };
    let op = op_label(&req.path);
    let timeout = Duration::from_secs(state.config.control_timeout_secs);
    let started = Instant::now();
    let replies = calls.call(&req.path, body, timeout).await;
    let elapsed = started.elapsed();
    let replies = match replies {
        Ok(replies) => replies,
        Err(CallFailure::Timeout) => {
            record_control_call(op, "timeout", elapsed);
            return Err(RlError::UpstreamTimeout {
                worker_id: worker.id.clone(),
                url: worker.url.clone(),
                timeout_secs: state.config.control_timeout_secs,
            });
        }
        Err(CallFailure::Failed(message)) => {
            record_control_call(op, "unreachable", elapsed);
            return Err(RlError::UpstreamUnreachable {
                worker_id: worker.id.clone(),
                url: worker.url.clone(),
                message,
            });
        }
        // The connection cannot carry the call: the same answer as a worker
        // without a control endpoint, with the engine's reason and the way out.
        Err(CallFailure::Unsupported(message)) => {
            return Err(no_control_endpoint(
                worker,
                &format!("{message}; label `rl.control_url` to reach the engine's own control app"),
            ));
        }
    };
    let (status, body) = route_answer(&replies);
    let outcome = RlCallOutcome {
        worker_id: worker.id.clone(),
        url: worker.url.clone(),
        status,
        latency_ms: elapsed.as_millis() as u64,
        body,
        body_truncated: false,
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
        worker_id = %worker.id, url = %worker.url, transport = "engine_connection",
        path = %req.path, status, ranks = replies.len(),
        latency_ms = outcome.latency_ms, "rl.proxy"
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
            (status, Json(outcome)).into_response()
        }
        Err(e) => e.into_response(),
    }
}

#[cfg(test)]
#[path = "../tests/unit/proxy/tests.rs"]
mod tests;
