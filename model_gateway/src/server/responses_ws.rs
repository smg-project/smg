//! Connection-scoped Responses continuations over WebSocket.
//!
//! Reuses the native validated Responses pipeline. `store=false` histories
//! live only on this authenticated socket, with bounded LRU state. Dropping
//! the socket drops its history and any in-flight response body; this never
//! flushes the inference engine's shared radix/KV cache.
use std::{
    collections::VecDeque,
    net::SocketAddr,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use axum::{
    body::{to_bytes, Body},
    extract::{
        ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade},
        Extension,
    },
    http::{header, HeaderMap, HeaderValue, Method, Request, StatusCode, Uri},
    response::{IntoResponse, Response},
};
use futures::StreamExt;
use openai_protocol::responses::ResponsesRequest;
use serde_json::{json, Value};
use smg_external_router::sse::SseDecoder;
use tower::ServiceExt;

/// In-process access to the existing protected POST route: authentication,
/// tenant accounting, admission, validation and cancellation remain native.
pub(super) type TrackResponse = Arc<dyn Fn() -> Result<Box<dyn Send>, Value> + Send + Sync>;
#[derive(Clone)]
pub(super) struct Dispatch {
    pub(super) service: axum::Router,
    pub(super) track_response: TrackResponse,
    pub(super) max_payload_bytes: usize,
}

/// An internal request extension; a network caller cannot supply this marker.
#[derive(Clone)]
pub(super) struct Prewarm;

#[derive(Clone)]
struct ConnectionTenant(String);
#[derive(Clone)]
struct TenantChanged;

/// Bound histories cannot change tenants when a serving key is reassigned.
/// This runs after authentication/tenant resolution, before admission.
#[cfg(feature = "experimental-responses-websocket")]
pub(super) async fn tenant_binding_guard(
    request: Request<Body>,
    next: axum::middleware::Next,
) -> Response {
    if let Some(expected) = request.extensions().get::<ConnectionTenant>() {
        let matches = request
            .extensions()
            .get::<crate::middleware::TenantRequestMeta>()
            .is_some_and(|current| current.tenant_key().as_str() == expected.0);
        if !matches {
            let mut response = (
                StatusCode::FORBIDDEN,
                axum::Json(json!({
                    "error": {
                        "type": "invalid_request_error",
                        "code": "websocket_tenant_changed",
                        "message": "The connection's authenticated tenant changed; reconnect without retained history.",
                    }
                })),
            ).into_response();
            response.extensions_mut().insert(TenantChanged);
            return response;
        }
    }
    next.run(request).await
}

/// Called by the normal POST handler after auth, tenant resolution, admission,
/// body limits and typed validation. No generation or durable storage is done.
pub(super) fn prewarm_response(request: ResponsesRequest) -> Response {
    axum::Json(json!({
        "id": format!("resp_{}", uuid::Uuid::now_v7().simple()),
        "object": "response",
        "created_at": chrono::Utc::now().timestamp(),
        "status": "completed",
        "model": request.model,
        "output": [],
        "store": false,
        "error": null,
        "incomplete_details": null,
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }))
    .into_response()
}

#[cfg(test)]
const MAX_MESSAGE_BYTES: usize = 64 * 1024 * 1024;
const MAX_EVENT_BYTES: usize = 16 * 1024 * 1024;
const MAX_CACHE_BYTES: usize = 32 * 1024 * 1024;
const MAX_CACHE_ENTRIES: usize = 4;
const GLOBAL_CACHE_BYTES: usize = 256 * 1024 * 1024;
const IDLE_TIMEOUT: Duration = Duration::from_secs(900);
const WRITE_TIMEOUT: Duration = Duration::from_secs(15);
const RESPONSE_READ_TIMEOUT: Duration = Duration::from_secs(15);
const MAX_ERROR_BYTES: usize = 1024 * 1024;
const CONNECTION_LIFETIME: Duration = Duration::from_secs(3600);
static CACHE_BYTES: AtomicUsize = AtomicUsize::new(0);

struct Charge(usize);
impl Charge {
    fn reserve(bytes: usize) -> Option<Self> {
        CACHE_BYTES
            .try_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(bytes)
                    .filter(|total| *total <= GLOBAL_CACHE_BYTES)
            })
            .ok()?;
        metrics::gauge!("smg_responses_ws_cache_bytes")
            .set(CACHE_BYTES.load(Ordering::Acquire) as f64);
        Some(Self(bytes))
    }
}
impl Drop for Charge {
    fn drop(&mut self) {
        CACHE_BYTES.fetch_sub(self.0, Ordering::AcqRel);
        metrics::gauge!("smg_responses_ws_cache_bytes")
            .set(CACHE_BYTES.load(Ordering::Acquire) as f64);
    }
}
struct Entry {
    id: String,
    input: Vec<Value>,
    output: Vec<Value>,
    charge: Charge,
}
#[derive(Default)]
struct SessionCache {
    entries: VecDeque<Entry>,
    bytes: usize,
}
impl SessionCache {
    fn clear(&mut self) {
        self.entries.clear();
        self.bytes = 0;
    }
    fn expand(&mut self, parent: &str, input: Vec<Value>) -> Result<Vec<Value>, Value> {
        let index = self.entries.iter().position(|entry| entry.id == parent);
        let Some(entry) = index.and_then(|index| self.entries.remove(index)) else {
            metrics::counter!("smg_responses_ws_cache_misses_total").increment(1);
            return Err(error("previous_response_not_found", "Previous response is not cached on this connection. Resend the complete input without previous_response_id.", Some("previous_response_id")));
        };
        let mut expanded = entry.input.clone();
        expanded.extend(entry.output.iter().cloned());
        expanded.extend(input);
        self.entries.push_back(entry);
        metrics::counter!("smg_responses_ws_cache_hits_total").increment(1);
        Ok(expanded)
    }
    fn remember(&mut self, id: String, input: Vec<Value>, output: Vec<Value>) {
        // Retention is charged in logical bytes (3x serialized JSON), not RSS.
        // Frame and HTTP body limits bound individual incoming requests.
        let bytes = match serde_json::to_vec(&(&input, &output))
            .ok()
            .and_then(|v| v.len().checked_mul(3))
        {
            Some(bytes) if bytes <= MAX_CACHE_BYTES => bytes,
            _ => {
                self.clear();
                return;
            }
        };
        while !self.entries.is_empty()
            && (self.entries.len() >= MAX_CACHE_ENTRIES || self.bytes + bytes > MAX_CACHE_BYTES)
        {
            self.evict();
        }
        let charge = loop {
            if let Some(charge) = Charge::reserve(bytes) {
                break charge;
            }
            if self.entries.is_empty() {
                return;
            }
            self.evict();
        };
        self.bytes += bytes;
        self.entries.push_back(Entry {
            id,
            input,
            output,
            charge,
        });
    }
    fn evict(&mut self) {
        if let Some(entry) = self.entries.pop_front() {
            self.bytes -= entry.charge.0;
            metrics::counter!("smg_responses_ws_cache_evictions_total").increment(1);
        }
    }
}
fn error(code: &str, message: &str, param: Option<&str>) -> Value {
    json!({
        "type": "error",
        "status": 400,
        "error": {"type": "invalid_request_error", "code": code, "message": message, "param": param},
    })
}
fn connection_expired_error() -> Value {
    error(
        "websocket_connection_limit_reached",
        "Connection lifetime reached; reconnect and replay full context",
        None,
    )
}

fn input_items(value: Option<Value>) -> Result<Vec<Value>, Value> {
    match value {
        None => Ok(Vec::new()),
        Some(Value::Array(items)) => Ok(items),
        Some(Value::String(text)) => Ok(vec![json!({"role":"user","content":text})]),
        _ => Err(error(
            "invalid_type",
            "input must be a string or an array",
            Some("input"),
        )),
    }
}
struct Prepared {
    body: Value,
    input: Vec<Value>,
    generate: bool,
    full_replacement: bool,
}
fn prepare(value: Value, cache: &mut SessionCache) -> Result<Prepared, Value> {
    let Value::Object(mut body) = value else {
        return Err(error(
            "invalid_type",
            "Expected a response.create object",
            None,
        ));
    };
    if body.remove("type").as_ref().and_then(Value::as_str) != Some("response.create") {
        return Err(error(
            "invalid_value",
            "Expected type response.create",
            Some("type"),
        ));
    }
    // A connection serializes its responses. Never silently accept multiplexed
    // stream routing that this handler cannot honor.
    if body.contains_key("stream_id") {
        return Err(error(
            "unsupported_parameter",
            "stream_id multiplexing is not supported on this connection",
            Some("stream_id"),
        ));
    }
    if body.get("background").and_then(Value::as_bool) == Some(true) {
        return Err(error(
            "unsupported_parameter",
            "background is not supported over WebSocket",
            Some("background"),
        ));
    }
    match body.get("store") {
        None | Some(Value::Bool(false)) => {}
        Some(Value::Bool(true)) => {
            return Err(error(
                "unsupported_parameter",
                "This experimental transport supports only store:false",
                Some("store"),
            ));
        }
        _ => {
            return Err(error(
                "invalid_type",
                "store must be false or omitted for this transport",
                Some("store"),
            ));
        }
    }
    if body
        .get("conversation")
        .is_some_and(|value| !value.is_null())
    {
        return Err(error(
            "unsupported_parameter",
            "conversation storage is not supported by connection-local continuations",
            Some("conversation"),
        ));
    }
    let generate = match body.remove("generate") {
        None | Some(Value::Bool(true)) => true,
        Some(Value::Bool(false)) => false,
        _ => {
            return Err(error(
                "invalid_type",
                "generate must be boolean",
                Some("generate"),
            ))
        }
    };
    let parent = body.remove("previous_response_id");
    let full_replacement = matches!(parent.as_ref(), None | Some(Value::Null));
    let input = input_items(body.remove("input"))?;
    let input = match parent {
        Some(Value::String(parent)) => cache.expand(&parent, input)?,
        None | Some(Value::Null) => input,
        _ => {
            return Err(error(
                "invalid_type",
                "previous_response_id must be a string",
                Some("previous_response_id"),
            ))
        }
    };
    body.insert("input".into(), Value::Array(input.clone()));
    body.insert("stream".into(), Value::Bool(true));
    // This pilot supports only socket-scoped history; unstored responses
    // are never promoted to persistent history.
    body.insert("store".into(), Value::Bool(false));
    Ok(Prepared {
        body: Value::Object(body),
        input,
        generate,
        full_replacement,
    })
}
async fn send(socket: &mut WebSocket, value: &Value) -> bool {
    write_message(socket, Message::Text(value.to_string().into())).await
}
async fn write_message(socket: &mut WebSocket, message: Message) -> bool {
    matches!(
        tokio::time::timeout(WRITE_TIMEOUT, socket.send(message)).await,
        Ok(Ok(()))
    )
}

#[derive(Debug, PartialEq, Eq)]
enum ResponseBodyError {
    Timeout,
    Read,
    InvalidJson,
}
impl ResponseBodyError {
    fn message(&self) -> &'static str {
        match self {
            Self::Timeout => "Responses body read timed out",
            Self::Read => "Responses body could not be read within its byte limit",
            Self::InvalidJson => "Responses body is not valid JSON",
        }
    }
}

async fn read_response_json(body: Body, limit: usize) -> Result<Value, ResponseBodyError> {
    let bytes = tokio::time::timeout(RESPONSE_READ_TIMEOUT, to_bytes(body, limit))
        .await
        .map_err(|_| ResponseBodyError::Timeout)?
        .map_err(|_| ResponseBodyError::Read)?;
    serde_json::from_slice(&bytes).map_err(|_| ResponseBodyError::InvalidJson)
}

async fn upstream_error(socket: &mut WebSocket, response: Response) -> bool {
    let status = response.status().as_u16();
    let envelope = read_response_json(response.into_body(), MAX_ERROR_BYTES)
        .await
        .unwrap_or_else(|failure| {
            json!({
                "error": {
                    "type": "server_error",
                    "code": "upstream_error",
                    "message": failure.message(),
                }
            })
        });
    send(
        socket,
        &json!({"type":"error","status":status,"error":envelope.get("error").unwrap_or(&envelope)}),
    )
    .await
}
/// Decode native SSE with the shared codec; forward JSON events without SSE framing.
struct Events(SseDecoder);
impl Default for Events {
    fn default() -> Self {
        Self(SseDecoder::with_max_size(MAX_EVENT_BYTES))
    }
}
impl Events {
    fn push(&mut self, chunk: &[u8]) -> Result<Vec<Value>, ()> {
        self.0.push(chunk).map_err(|_| ())?;
        let mut events = Vec::new();
        while let Some(frame) = self.0.next_frame() {
            let frame = frame.map_err(|_| ())?;
            if !frame.is_done() && !frame.data.is_empty() {
                events.push(frame.decode_data().map_err(|_| ())?);
            }
        }
        self.0.compact();
        Ok(events)
    }
}

// Capture before serving auth consumes the bearer header. Only an internal
// request extension crosses that boundary; clients cannot manufacture it.
// Each generated turn still reauthenticates through the protected POST route.
#[derive(Clone)]
pub(super) struct UpgradeCredential(Option<HeaderValue>, Option<SocketAddr>);
pub(super) async fn capture_credential(
    mut request: Request<Body>,
    next: axum::middleware::Next,
) -> Response {
    if request.method() == Method::GET && request.uri().path() == "/v1/responses" {
        let credential = request.headers().get(header::AUTHORIZATION).cloned();
        let peer = request
            .extensions()
            .get::<axum::extract::ConnectInfo<SocketAddr>>()
            .map(|peer| peer.0);
        request
            .extensions_mut()
            .insert(UpgradeCredential(credential, peer));
    }
    next.run(request).await
}

pub(super) async fn upgrade(
    mut headers: HeaderMap,
    Extension(dispatch): Extension<Dispatch>,
    Extension(credential): Extension<UpgradeCredential>,
    #[cfg(feature = "experimental-responses-websocket")] Extension(tenant): Extension<
        crate::middleware::TenantRequestMeta,
    >,
    upgrade: WebSocketUpgrade,
) -> Response {
    if let Some(credential) = credential.0 {
        headers.insert(header::AUTHORIZATION, credential);
    }
    let peer = credential.1;
    #[cfg(feature = "experimental-responses-websocket")]
    let bound_tenant = Some(tenant.tenant_key().as_str().to_owned());
    #[cfg(not(feature = "experimental-responses-websocket"))]
    let bound_tenant = None;
    upgrade
        .max_message_size(dispatch.max_payload_bytes)
        .max_frame_size(dispatch.max_payload_bytes)
        .on_upgrade(move |socket| {
            serve(socket, dispatch, headers, peer, bound_tenant, IDLE_TIMEOUT)
        })
        .into_response()
}
async fn busy_message(
    socket: &mut WebSocket,
    message: Option<Result<Message, axum::Error>>,
) -> bool {
    match message {
        Some(Ok(Message::Ping(bytes))) => write_message(socket, Message::Pong(bytes)).await,
        Some(Ok(Message::Pong(_))) => true,
        Some(Ok(Message::Close(_))) | None | Some(Err(_)) => false,
        Some(Ok(_)) => send(
            socket,
            &error(
                "response_in_progress",
                "Wait for the current response to finish before sending another response.create",
                None,
            ),
        )
        .await,
    }
}
async fn serve(
    mut socket: WebSocket,
    dispatch: Dispatch,
    headers: HeaderMap,
    peer: Option<SocketAddr>,
    bound_tenant: Option<String>,
    idle_timeout: Duration,
) {
    let mut cache = SessionCache::default();
    let expires = tokio::time::Instant::now() + CONNECTION_LIFETIME;
    loop {
        let deadline = expires.min(tokio::time::Instant::now() + idle_timeout);
        let message = match tokio::time::timeout_at(deadline, socket.recv()).await {
            Ok(Some(Ok(Message::Text(text)))) => text,
            Ok(Some(Ok(Message::Ping(bytes)))) => {
                if !write_message(&mut socket, Message::Pong(bytes)).await {
                    return;
                }
                continue;
            }
            Ok(Some(Ok(Message::Pong(_)))) => continue,
            Err(_) => {
                if tokio::time::Instant::now() >= expires {
                    let _ = send(&mut socket, &connection_expired_error()).await;
                }
                let _ = write_message(
                    &mut socket,
                    Message::Close(Some(CloseFrame {
                        code: 1000,
                        reason: "Idle connection expired".into(),
                    })),
                )
                .await;
                return;
            }
            _ => return,
        };
        let value = match serde_json::from_str(&message) {
            Ok(value) => value,
            Err(_) => {
                if !send(
                    &mut socket,
                    &error("invalid_json", "Invalid JSON response.create", None),
                )
                .await
                {
                    return;
                }
                continue;
            }
        };
        let prepared = match prepare(value, &mut cache) {
            Ok(prepared) => prepared,
            Err(error) => {
                if !send(&mut socket, &error).await {
                    return;
                }
                continue;
            }
        };
        // Count each logical response, including its streaming body, in the
        // server's existing drain tracker. A closed socket drops this guard.
        let _active = match (dispatch.track_response)() {
            Ok(active) => active,
            Err(error) => {
                let _ = send(&mut socket, &error).await;
                return;
            }
        };
        let mut body = Request::new(Body::from(prepared.body.to_string()));
        *body.method_mut() = Method::POST;
        *body.uri_mut() = Uri::from_static("/v1/responses");
        *body.headers_mut() = headers.clone();
        if let Some(peer) = peer {
            body.extensions_mut()
                .insert(axum::extract::ConnectInfo(peer));
        }
        if let Some(tenant) = &bound_tenant {
            body.extensions_mut()
                .insert(ConnectionTenant(tenant.clone()));
        }
        for name in [
            "host",
            "connection",
            "upgrade",
            "sec-websocket-key",
            "sec-websocket-version",
            "sec-websocket-protocol",
            "sec-websocket-extensions",
            "content-length",
            "transfer-encoding",
            "x-request-id",
        ] {
            body.headers_mut().remove(name);
        }
        body.headers_mut()
            .insert("content-type", HeaderValue::from_static("application/json"));
        axum::extract::DefaultBodyLimit::max(dispatch.max_payload_bytes).apply(&mut body);
        if !prepared.generate {
            body.extensions_mut().insert(Prewarm);
        }
        metrics::counter!("smg_responses_ws_requests_total").increment(1);
        // Invoke the real protected POST service in process. No extra network
        // hop, alternative tokenizer, parser, routing or generation path.
        let future = dispatch.service.clone().oneshot(body);
        tokio::pin!(future);
        let response = loop {
            tokio::select! {
                () = tokio::time::sleep_until(expires) => {
                    let _ = send(&mut socket, &connection_expired_error()).await;
                    return;
                },
                response = &mut future => break match response {
                    Ok(response) => response,
                    Err(never) => match never {},
                },
                message = socket.recv() => {
                    if !busy_message(&mut socket, message).await { return; }
                },
            }
        };
        if !response.status().is_success() {
            let authentication_lost = response.status() == StatusCode::UNAUTHORIZED
                || response.extensions().get::<TenantChanged>().is_some();
            if !upstream_error(&mut socket, response).await {
                return;
            }
            if authentication_lost {
                let _ = write_message(
                    &mut socket,
                    Message::Close(Some(CloseFrame {
                        code: 1008,
                        reason: "Connection authorization changed".into(),
                    })),
                )
                .await;
                return;
            }
            continue;
        }
        if !prepared.generate {
            let response =
                match read_response_json(response.into_body(), dispatch.max_payload_bytes).await {
                    Ok(value) => value,
                    Err(failure) => {
                        let _ = send(
                            &mut socket,
                            &error("upstream_stream_error", failure.message(), None),
                        )
                        .await;
                        return;
                    }
                };
            let Some(id) = response["id"]
                .as_str()
                .filter(|_| response["status"] == "completed")
            else {
                let _ = send(
                    &mut socket,
                    &error("upstream_stream_error", "Invalid prewarm response", None),
                )
                .await;
                return;
            };
            let id = id.to_owned();
            let mut created = response.clone();
            created["status"] = json!("in_progress");
            if !send(
                &mut socket,
                &json!({"type":"response.created","sequence_number":0,"response":created}),
            )
            .await
            {
                return;
            }
            if prepared.full_replacement {
                cache.clear();
            }
            cache.remember(id, prepared.input, Vec::new());
            if !send(
                &mut socket,
                &json!({"type":"response.completed","sequence_number":1,"response":response}),
            )
            .await
            {
                return;
            }
            continue;
        }
        let mut stream = response.into_body().into_data_stream();
        let mut decoder = Events::default();
        let mut complete = false;
        loop {
            tokio::select! {
                () = tokio::time::sleep_until(expires) => {
                    let _ = send(&mut socket, &connection_expired_error()).await;
                    return;
                },
                message = socket.recv() => {
                    if !busy_message(&mut socket, message).await { return; }
                },
                chunk = stream.next() => {
                    let chunk = match chunk {
                        Some(Ok(chunk)) => chunk,
                        _ => break,
                    };
                    let events = match decoder.push(&chunk) {
                        Ok(events) => events,
                        Err(()) => {
                            let _ = send(&mut socket, &error(
                                "upstream_stream_error", "Invalid or oversized upstream event", None,
                            )).await;
                            return;
                        }
                    };
                    for event in events {
                        match event.get("type").and_then(Value::as_str) {
                            Some("response.completed" | "response.incomplete") => {
                                if prepared.full_replacement && event["type"] == "response.completed" {
                                    cache.clear();
                                }
                                let response = &event["response"];
                                if let (Some(id), Some(output)) = (response["id"].as_str(), response["output"].as_array()) {
                                    cache.remember(id.to_owned(), prepared.input.clone(), output.clone());
                                }
                                complete = true;
                            }
                            Some("response.failed") => complete = true,
                            _ => {}
                        }
                        if !send(&mut socket, &event).await { return; }
                    }
                    if complete { break; }
                }
            }
        }
        if !complete
            && !send(
                &mut socket,
                &error(
                    "upstream_stream_error",
                    "Upstream ended before a terminal response event",
                    None,
                ),
            )
            .await
        {
            return;
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod audit_contracts;
