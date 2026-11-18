//! Connection-scoped Responses continuations over WebSocket.
//!
//! Reuses the native validated Responses pipeline. `store=false` histories
//! live only on this authenticated socket, with bounded LRU state. Dropping
//! the socket drops its history and any in-flight response body; this never
//! flushes the inference engine's shared radix/KV cache.
use std::{
    collections::{HashMap, VecDeque},
    io::{self, Write},
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
use openai_protocol::responses::{
    ResponseStatus, ResponseUsage, ResponsesRequest, ResponsesResponse, ResponsesUsage,
};
use parking_lot::Mutex;
use serde::{Serialize, Serializer as _};
use serde_json::{json, Map, Value};
use smg_external_router::sse::SseDecoder;
use tower::ServiceExt;

use crate::{
    observability::inflight_tracker::InFlightRequestTracker,
    routers::error as route_error,
    tenant::{DataPlaneCaller, TenantKey},
};

/// The protected POST service and limits shared by this listener's sessions.
#[derive(Clone)]
pub(super) struct Dispatch {
    pub(super) service: axum::Router,
    pub(super) tracker: Arc<InFlightRequestTracker>,
    pub(super) max_payload_bytes: usize,
    pub(super) origins: Arc<[String]>,
    pub(super) connections: Arc<Connections>,
}

/// Marks a socket turn after it is expanded to a normal POST request.
#[derive(Clone)]
pub(super) struct SocketTurn {
    tenant: TenantKey,
    pub(super) prewarm: bool,
}

#[derive(Default)]
pub(super) struct Connections(Mutex<ConnectionCounts>);

#[derive(Default)]
struct ConnectionCounts {
    total: usize,
    tenants: HashMap<TenantKey, usize>,
}

impl Connections {
    fn try_acquire(self: &Arc<Self>, tenant: TenantKey) -> Option<ConnectionPermit> {
        let mut counts = self.0.lock();
        if counts.total >= MAX_CONNECTIONS {
            return None;
        }
        let count = counts.tenants.entry(tenant.clone()).or_default();
        if *count >= MAX_CONNECTIONS_PER_TENANT {
            return None;
        }
        *count += 1;
        counts.total += 1;
        Some(ConnectionPermit {
            connections: self.clone(),
            tenant,
        })
    }
}

struct ConnectionPermit {
    connections: Arc<Connections>,
    tenant: TenantKey,
}

impl Drop for ConnectionPermit {
    fn drop(&mut self) {
        let mut counts = self.connections.0.lock();
        if let Some(count) = counts.tenants.get_mut(&self.tenant) {
            *count -= 1;
            if *count == 0 {
                counts.tenants.remove(&self.tenant);
            }
        }
        counts.total -= 1;
    }
}

#[derive(Clone)]
struct TenantChanged;

/// Bound histories cannot change tenants when a serving key is reassigned.
/// This runs after authentication/tenant resolution, before admission.
pub(super) async fn tenant_binding_guard(
    request: Request<Body>,
    next: axum::middleware::Next,
) -> Response {
    if let Some(expected) = request.extensions().get::<SocketTurn>() {
        let matches = request
            .extensions()
            .get::<DataPlaneCaller>()
            .is_some_and(|current| current.tenant_key() == &expected.tenant);
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
    let response = ResponsesResponse::builder(
        format!("resp_{}", uuid::Uuid::now_v7().simple()),
        &request.model,
    )
    .copy_from_request(&request)
    .status(ResponseStatus::Completed)
    .completed_at(chrono::Utc::now().timestamp())
    .store(false)
    .usage(ResponsesUsage::Modern(ResponseUsage {
        input_tokens: 0,
        output_tokens: 0,
        total_tokens: 0,
        input_tokens_details: Default::default(),
        output_tokens_details: Default::default(),
    }))
    .build();
    axum::Json(response).into_response()
}

#[cfg(test)]
const MAX_MESSAGE_BYTES: usize = 64 * 1024 * 1024;
pub(super) const MAX_PAYLOAD_BYTES: usize = 4 * 1024 * 1024;
const MAX_CONNECTIONS: usize = 32;
const MAX_CONNECTIONS_PER_TENANT: usize = 4;
const MAX_EVENT_BYTES: usize = 8 * 1024 * 1024;
const MAX_CACHE_BYTES: usize = 16 * 1024 * 1024;
const MAX_CACHE_ENTRIES: usize = 4;
const GLOBAL_CACHE_BYTES: usize = 128 * 1024 * 1024;
const IDLE_TIMEOUT: Duration = Duration::from_secs(900);
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(15);
const WRITE_TIMEOUT: Duration = Duration::from_secs(15);
const RESPONSE_READ_TIMEOUT: Duration = Duration::from_secs(15);
const POST_TAIL_TIMEOUT: Duration = Duration::from_secs(1);
const MAX_POST_TAIL_FRAMES: usize = 8;
const MAX_ERROR_BYTES: usize = 1024 * 1024;
const CONNECTION_LIFETIME: Duration = Duration::from_secs(3600);
static CACHE_BYTES: AtomicUsize = AtomicUsize::new(0);

struct Charge {
    bytes: usize,
    gauge: metrics::Gauge,
}
impl Charge {
    fn reserve(bytes: usize) -> Option<Self> {
        CACHE_BYTES
            .try_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(bytes)
                    .filter(|total| *total <= GLOBAL_CACHE_BYTES)
            })
            .ok()?;
        // Add each charge to the gauge before handing it out. Publishing
        // absolute snapshots allows concurrent drops to restore older values.
        let gauge = metrics::gauge!("smg_responses_ws_cache_bytes");
        gauge.increment(bytes as f64);
        Some(Self { bytes, gauge })
    }
}
impl Drop for Charge {
    fn drop(&mut self) {
        CACHE_BYTES.fetch_sub(self.bytes, Ordering::AcqRel);
        self.gauge.decrement(self.bytes as f64);
    }
}
struct Entry {
    id: Box<str>,
    items: Box<[u8]>,
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
    fn expand(
        &mut self,
        parent: &str,
        input: Vec<Value>,
        max_payload_bytes: usize,
    ) -> Result<Vec<Value>, Value> {
        let Some(index) = self
            .entries
            .iter()
            .position(|entry| entry.id.as_ref() == parent)
        else {
            metrics::counter!("smg_responses_ws_cache_misses_total").increment(1);
            return Err(error("previous_response_not_found", "Previous response is not cached on this connection. Resend the complete input without previous_response_id.", Some("previous_response_id")));
        };
        let entry = &self.entries[index];
        if entry.items.len() > max_payload_bytes {
            return Err(payload_limit_error());
        }
        let mut expanded: Vec<Value> = serde_json::from_slice(&entry.items).map_err(|_| {
            error(
                "previous_response_not_found",
                "Previous response could not be replayed. Resend the complete input without previous_response_id.",
                Some("previous_response_id"),
            )
        })?;
        expanded.extend(input);
        self.entries.make_contiguous()[index..].rotate_left(1);
        metrics::counter!("smg_responses_ws_cache_hits_total").increment(1);
        Ok(expanded)
    }
    fn remember(&mut self, id: &str, input: &[Value], output: &[Value]) {
        // Cache the serialized transcript so small JSON items cannot expand
        // into an unaccounted tree of allocations for the lifetime of a session.
        let Some(limit) = (MAX_CACHE_BYTES / 3).checked_sub(id.len()) else {
            return;
        };
        let mut encoded = JsonLimit::new(Vec::new(), limit);
        if serde_json::Serializer::new(&mut encoded)
            .collect_seq(input.iter().chain(output))
            .is_err()
        {
            return;
        }
        let bytes = (encoded.written + id.len()) * 3;
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
            id: id.into(),
            items: encoded.writer.into_boxed_slice(),
            charge,
        });
    }
    fn evict(&mut self) {
        if let Some(entry) = self.entries.pop_front() {
            self.bytes -= entry.charge.bytes;
            metrics::counter!("smg_responses_ws_cache_evictions_total").increment(1);
        }
    }
}
/// Stops encoding at the byte budget. The POST retains the bytes and the
/// history charge counts them without allocating another copy of the JSON.
struct JsonLimit<W> {
    writer: W,
    written: usize,
    limit: usize,
}

impl<W> JsonLimit<W> {
    fn new(writer: W, limit: usize) -> Self {
        Self {
            writer,
            written: 0,
            limit,
        }
    }
}

impl<W: Write> Write for JsonLimit<W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.limit - self.written {
            return Err(io::Error::other(
                "Responses WebSocket JSON exceeds its byte limit",
            ));
        }
        let written = self.writer.write(bytes)?;
        self.written += written;
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
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

fn payload_limit_error() -> Value {
    let mut refusal = error(
        "request_body_too_large",
        "Expanded Responses request exceeds the WebSocket payload limit",
        None,
    );
    refusal["status"] = 413.into();
    refusal
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
#[derive(Serialize)]
struct Prepared {
    #[serde(flatten)]
    body: Map<String, Value>,
    input: Vec<Value>,
    #[serde(skip)]
    generate: bool,
    #[serde(skip)]
    full_replacement: bool,
}
fn prepare(
    value: Value,
    cache: &mut SessionCache,
    max_payload_bytes: usize,
) -> Result<Prepared, Value> {
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
        Some(Value::String(parent)) => cache.expand(&parent, input, max_payload_bytes)?,
        None | Some(Value::Null) => input,
        _ => {
            return Err(error(
                "invalid_type",
                "previous_response_id must be a string",
                Some("previous_response_id"),
            ))
        }
    };
    body.insert("stream".into(), Value::Bool(true));
    // This pilot supports only socket-scoped history; unstored responses
    // are never promoted to persistent history.
    body.insert("store".into(), Value::Bool(false));
    Ok(Prepared {
        body,
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

async fn heartbeat(socket: &mut WebSocket) -> bool {
    write_message(socket, Message::Ping(Default::default())).await
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
    caller: Option<Extension<DataPlaneCaller>>,
    upgrade: WebSocketUpgrade,
) -> Response {
    let Some(Extension(caller)) = caller else {
        let mut response = route_error::create_error(
            StatusCode::UNAUTHORIZED,
            "websocket_auth_required",
            "Responses WebSockets require a gateway API key",
        );
        response
            .headers_mut()
            .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
        return response;
    };
    // Browsers may only upgrade from a single, explicitly allowed Origin.
    let mut origins = headers.get_all(header::ORIGIN).iter();
    if let Some(origin) = origins.next() {
        let allowed = origin.to_str().is_ok_and(|origin| {
            origin != "null"
                && dispatch
                    .origins
                    .iter()
                    .any(|allowed| allowed != "*" && allowed == origin)
        });
        if !allowed || origins.next().is_some() {
            return route_error::create_error(
                StatusCode::FORBIDDEN,
                "websocket_origin_forbidden",
                "WebSocket Origin is not allowed",
            );
        }
    }
    let tenant = caller.tenant_key().clone();
    let Some(permit) = dispatch.connections.try_acquire(tenant.clone()) else {
        let mut response = route_error::create_error(
            StatusCode::TOO_MANY_REQUESTS,
            "websocket_connections_exhausted",
            "Responses WebSocket connection limit reached",
        );
        response
            .headers_mut()
            .insert(header::RETRY_AFTER, HeaderValue::from_static("1"));
        return response;
    };
    let upgrade_guard = dispatch.tracker.track_upgrade();
    if dispatch.tracker.is_draining() {
        return route_error::service_unavailable("gateway_draining", "Gateway is draining");
    }
    if let Some(credential) = credential.0 {
        headers.insert(header::AUTHORIZATION, credential);
    }
    let peer = credential.1;
    upgrade
        .max_message_size(dispatch.max_payload_bytes)
        .max_frame_size(dispatch.max_payload_bytes)
        .on_upgrade(move |socket| async move {
            let _guards = (upgrade_guard, permit);
            serve(socket, dispatch, headers, peer, tenant, IDLE_TIMEOUT).await;
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
    tenant: TenantKey,
    idle_timeout: Duration,
) {
    let mut cache = SessionCache::default();
    let opened_at = tokio::time::Instant::now();
    let expires = opened_at + CONNECTION_LIFETIME;
    let mut ping = tokio::time::interval_at(opened_at + HEARTBEAT_INTERVAL, HEARTBEAT_INTERVAL);
    ping.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        let deadline = expires.min(tokio::time::Instant::now() + idle_timeout);
        // Control frames keep the transport alive without extending the time
        // the session can wait for its next turn.
        let message = loop {
            tokio::select! {
                biased;
                () = dispatch.tracker.draining() => {
                    let _ = write_message(&mut socket, Message::Close(Some(CloseFrame {
                        code: 1012,
                        reason: "Gateway is draining".into(),
                    }))).await;
                    return;
                }
                () = tokio::time::sleep_until(deadline) => {
                    if tokio::time::Instant::now() >= expires {
                        let _ = send(&mut socket, &connection_expired_error()).await;
                    }
                    let _ = write_message(&mut socket, Message::Close(Some(CloseFrame {
                        code: 1000,
                        reason: "Idle connection expired".into(),
                    }))).await;
                    return;
                }
                _ = ping.tick() => {
                    if !heartbeat(&mut socket).await { return; }
                }
                incoming = socket.recv() => match incoming {
                    Some(Ok(Message::Text(text))) => break text,
                    Some(Ok(Message::Ping(bytes))) => {
                        if !write_message(&mut socket, Message::Pong(bytes)).await { return; }
                    }
                    Some(Ok(Message::Pong(_))) => {}
                    _ => return,
                }
            }
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
        let prepared = match prepare(value, &mut cache, dispatch.max_payload_bytes) {
            Ok(prepared) => prepared,
            Err(error) => {
                if !send(&mut socket, &error).await {
                    return;
                }
                continue;
            }
        };
        // The protected service's edge tracks the response body. The upgrade
        // guard keeps shutdown waiting while a turn enters that service.
        if dispatch.tracker.is_draining() {
            let _ = write_message(
                &mut socket,
                Message::Close(Some(CloseFrame {
                    code: 1012,
                    reason: "Gateway is draining".into(),
                })),
            )
            .await;
            return;
        }
        let mut payload = JsonLimit::new(Vec::new(), dispatch.max_payload_bytes);
        if serde_json::to_writer(&mut payload, &prepared).is_err() {
            if !send(&mut socket, &payload_limit_error()).await {
                return;
            }
            continue;
        }
        drop(prepared.body);
        drop(message);
        let mut body = Request::new(Body::from(payload.writer));
        *body.method_mut() = Method::POST;
        *body.uri_mut() = Uri::from_static("/v1/responses");
        *body.headers_mut() = headers.clone();
        if let Some(peer) = peer {
            body.extensions_mut()
                .insert(axum::extract::ConnectInfo(peer));
        }
        body.extensions_mut().insert(SocketTurn {
            tenant: tenant.clone(),
            prewarm: !prepared.generate,
        });
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
                _ = ping.tick() => {
                    if !heartbeat(&mut socket).await { return; }
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
            let mut created = response.clone();
            created["status"] = json!("in_progress");
            created["completed_at"] = Value::Null;
            created["usage"] = Value::Null;
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
            cache.remember(id, &prepared.input, &[]);
            drop(prepared.input);
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
                _ = ping.tick() => {
                    if !heartbeat(&mut socket).await { return; }
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
                                    cache.remember(id, &prepared.input, output);
                                }
                                complete = true;
                            }
                            Some("response.failed" | "response.cancelled" | "error") => complete = true,
                            _ => {}
                        }
                        if !send(&mut socket, &event).await { return; }
                        if complete { break; }
                    }
                    if complete { break; }
                }
            }
        }
        drop(prepared.input);
        if complete {
            // Let the HTTP edge observe EOF and release its permit. The
            // terminal event has already reached the client.
            let deadline = expires.min(tokio::time::Instant::now() + POST_TAIL_TIMEOUT);
            let _ = tokio::time::timeout_at(deadline, async {
                for _ in 0..MAX_POST_TAIL_FRAMES {
                    if !matches!(stream.next().await, Some(Ok(_))) {
                        break;
                    }
                }
            })
            .await;
        } else if !send(
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
