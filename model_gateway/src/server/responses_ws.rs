//! Connection-scoped Responses continuations over WebSocket.
//!
//! Reuses the native validated Responses pipeline. `store=false` histories
//! live only on this authenticated socket, with bounded LRU state. Dropping
//! the socket drops its history and any in-flight response body; this never
//! flushes the inference engine's shared radix/KV cache.
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
pub(super) struct Dispatch(
    pub(super) axum::Router,
    pub(super) TrackResponse,
    pub(super) usize,
);

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
            let mut response = (StatusCode::FORBIDDEN, axum::Json(json!({"error":{"type":"invalid_request_error","code":"websocket_tenant_changed","message":"The connection's authenticated tenant changed; reconnect without retained history."}}))).into_response();
            response.extensions_mut().insert(TenantChanged);
            return response;
        }
    }
    next.run(request).await
}

/// Called by the normal POST handler after auth, tenant resolution, admission,
/// body limits and typed validation. No generation or durable storage is done.
pub(super) fn prewarm_response(request: ResponsesRequest) -> Response {
    axum::Json(json!({"id":format!("resp_{}", uuid::Uuid::now_v7().simple()),"object":"response","created_at":chrono::Utc::now().timestamp(),"status":"completed","model":request.model,"output":[],"store":false,"error":null,"incomplete_details":null,"usage":{"input_tokens":0,"output_tokens":0,"total_tokens":0}})).into_response()
}
use std::{
    collections::VecDeque,
    net::SocketAddr,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

#[cfg(test)]
const MAX_MESSAGE_BYTES: usize = 64 * 1024 * 1024;
const MAX_EVENT_BYTES: usize = 16 * 1024 * 1024;
const MAX_CACHE_BYTES: usize = 32 * 1024 * 1024;
const MAX_CACHE_ENTRIES: usize = 4;
const GLOBAL_CACHE_BYTES: usize = 256 * 1024 * 1024;
const IDLE_TIMEOUT: Duration = Duration::from_secs(900);
const WRITE_TIMEOUT: Duration = Duration::from_secs(15);
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
        let Some(index) = self.entries.iter().position(|entry| entry.id == parent) else {
            metrics::counter!("smg_responses_ws_cache_misses_total").increment(1);
            return Err(error("previous_response_not_found", "Previous response is not cached on this connection. Resend the complete input without previous_response_id.", Some("previous_response_id")));
        };
        let Some(entry) = self.entries.remove(index) else {
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
    json!({"type":"error","status":400,"error":{"type":"invalid_request_error","code":code,"message":message,"param":param}})
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
        Some(Value::String(parent)) => {
            if body
                .get("conversation")
                .is_some_and(|value| !value.is_null())
            {
                return Err(error(
                    "invalid_value",
                    "conversation and previous_response_id are mutually exclusive",
                    Some("conversation"),
                ));
            }
            cache.expand(&parent, input)?
        }
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
async fn upstream_error(socket: &mut WebSocket, response: Response) -> bool {
    let status = response.status().as_u16();
    let raw = to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap_or_default();
    let envelope: Value = serde_json::from_slice(&raw).unwrap_or_else(|_| json!({"error":{"type":"server_error","code":"upstream_error","message":"Responses pipeline refused the request"}}));
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
        .max_message_size(dispatch.2)
        .max_frame_size(dispatch.2)
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
                    let _ = send(
                        &mut socket,
                        &error(
                            "websocket_connection_limit_reached",
                            "Connection lifetime reached; reconnect and replay full context",
                            None,
                        ),
                    )
                    .await;
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
        let _active = match (dispatch.1)() {
            Ok(active) => active,
            Err(error) => {
                let _ = send(&mut socket, &error).await;
                return;
            }
        };
        if prepared.body.get("background").and_then(Value::as_bool) == Some(true) {
            if !send(
                &mut socket,
                &error(
                    "unsupported_parameter",
                    "background is not supported over WebSocket",
                    Some("background"),
                ),
            )
            .await
            {
                return;
            }
            continue;
        }
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
        axum::extract::DefaultBodyLimit::max(dispatch.2).apply(&mut body);
        if !prepared.generate {
            body.extensions_mut().insert(Prewarm);
        }
        metrics::counter!("smg_responses_ws_requests_total").increment(1);
        // Invoke the real protected POST service in process. No extra network
        // hop, alternative tokenizer, parser, routing or generation path.
        let future = dispatch.0.clone().oneshot(body);
        tokio::pin!(future);
        let response = loop {
            tokio::select! {
                () = tokio::time::sleep_until(expires) => { let _ = send(&mut socket, &error("websocket_connection_limit_reached", "Connection lifetime reached; reconnect and replay full context", None)).await; return; },
                response = &mut future => break match response { Ok(response) => response, Err(never) => match never {} },
                message = socket.recv() => if !busy_message(&mut socket, message).await { return; },
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
            let response: Value = match to_bytes(response.into_body(), dispatch.2).await {
                Ok(bytes) => match serde_json::from_slice(&bytes) {
                    Ok(value) => value,
                    Err(_) => {
                        let _ = send(
                            &mut socket,
                            &error("upstream_stream_error", "Invalid prewarm response", None),
                        )
                        .await;
                        return;
                    }
                },
                Err(_) => {
                    let _ = send(
                        &mut socket,
                        &error("upstream_stream_error", "Oversized prewarm response", None),
                    )
                    .await;
                    return;
                }
            };
            let Some(id) = response["id"].as_str() else {
                return;
            };
            let id = id.to_owned();
            if response["status"] != "completed" {
                return;
            }
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
                () = tokio::time::sleep_until(expires) => { let _ = send(&mut socket, &error("websocket_connection_limit_reached", "Connection lifetime reached; reconnect and replay full context", None)).await; return; },
                message = socket.recv() => if !busy_message(&mut socket, message).await { return; },
                chunk = stream.next() => {
                    let chunk = match chunk { Some(Ok(chunk)) => chunk, _ => break };
                    let events = match decoder.push(&chunk) {
                        Ok(events) => events,
                        Err(()) => { let _ = send(&mut socket, &error("upstream_stream_error", "Invalid or oversized upstream event", None)).await; return; }
                    };
                    for event in events {
                        let terminal = matches!(event.get("type").and_then(Value::as_str), Some("response.completed" | "response.incomplete"));
                        if terminal {
                            if prepared.full_replacement && event["type"] == "response.completed" { cache.clear(); }
                            let response = &event["response"];
                            if let (Some(id), Some(output)) = (response["id"].as_str(), response["output"].as_array()) {
                                cache.remember(id.to_owned(), prepared.input.clone(), output.clone());
                            }
                            complete = true;
                        }
                        if event.get("type").and_then(Value::as_str) == Some("response.failed") { complete = true; }
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
mod tests {
    use super::*;
    fn user(text: &str) -> Value {
        json!({"role":"user","content":text})
    }
    #[test]
    fn continuation_preserves_order_and_only_appends_new_items() {
        let mut cache = SessionCache::default();
        cache.remember("resp_1".into(), vec![user("first")], vec![json!({"type":"message","id":"m1","role":"assistant","content":[{"type":"output_text","text":"answer"}]})]);
        let request = prepare(json!({"type":"response.create","model":"fixture-model","previous_response_id":"resp_1","input":[user("next")],"store":false}), &mut cache).unwrap();
        assert_eq!(request.input.len(), 3);
        assert_eq!(request.input[0], user("first"));
        assert_eq!(request.input[2], user("next"));
        assert_eq!(request.body["store"], false);
    }
    #[test]
    fn ids_are_connection_scoped_and_missing_parent_never_generates() {
        let mut a = SessionCache::default();
        a.remember("resp_1".into(), vec![user("private")], vec![]);
        let mut b = SessionCache::default();
        let error = prepare(json!({"type":"response.create","model":"fixture-model","previous_response_id":"resp_1","input":"next"}), &mut b).err().unwrap();
        assert_eq!(error["error"]["code"], "previous_response_not_found");
    }
    #[test]
    fn lru_entries_are_bounded_and_clear_releases_owned_state() {
        let mut cache = SessionCache::default();
        for i in 0..8 {
            cache.remember(format!("resp_{i}"), vec![user("small")], vec![]);
        }
        assert_eq!(cache.entries.len(), MAX_CACHE_ENTRIES);
        assert!(cache.expand("resp_0", vec![]).is_err());
        cache.clear();
        assert!(cache.entries.is_empty());
        assert_eq!(cache.bytes, 0);
    }
    #[test]
    fn oversized_history_is_not_retained() {
        let mut cache = SessionCache::default();
        cache.remember(
            "large".into(),
            vec![user(&"x".repeat(MAX_CACHE_BYTES / 3 + 1))],
            vec![],
        );
        assert_eq!(cache.bytes, 0);
        assert!(cache.entries.is_empty());
    }
    #[test]
    fn prewarm_and_control_validation() {
        let mut cache = SessionCache::default();
        assert!(
            !prepare(
                json!({"type":"response.create","model":"fixture-model","generate":false,"input":[]}),
                &mut cache
            )
            .unwrap()
            .generate
        );
        assert!(prepare(json!({"type":"other"}), &mut cache).is_err());
        assert!(prepare(
            json!({"type":"response.create","generate":"false"}),
            &mut cache
        )
        .is_err());
        assert!(prepare(
            json!({"type":"response.create","stream_id":"x"}),
            &mut cache
        )
        .is_err());
    }
    #[test]
    fn sse_handles_split_utf8_crlf_comments_and_multiline_data() {
        let raw = ": ping\r\n\r\nevent: delta\r\ndata: {\"type\":\"response.output_text.delta\",\r\ndata: \"delta\":\"café\"}\r\n\r\n";
        let mut decoder = Events::default();
        let mut events = Vec::new();
        for byte in raw.as_bytes() {
            events.extend(decoder.push(&[*byte]).unwrap());
        }
        assert_eq!(events.len(), 1);
        assert_eq!(events[0]["delta"], "café");
    }
    #[expect(
        clippy::disallowed_methods,
        reason = "owned loopback server is aborted after the fixture and by test runtime shutdown"
    )]
    #[tokio::test]
    async fn idle_socket_expires_after_prewarm_and_releases_session() {
        use futures::{SinkExt, StreamExt};
        use tokio::sync::oneshot;
        use tokio_tungstenite::{connect_async, tungstenite::Message as ClientMessage};
        let (finished, done) = oneshot::channel();
        let finished = Arc::new(std::sync::Mutex::new(Some(finished)));
        let app = axum::Router::new().route(
            "/",
            axum::routing::get(move |upgrade: WebSocketUpgrade| {
                let finished = finished.clone();
                async move {
                    upgrade.on_upgrade(move |socket| async move {
                        serve(
                            socket,
                            Dispatch(axum::Router::new().route("/v1/responses", axum::routing::post(|axum::Json(request): axum::Json<ResponsesRequest>| async move { prewarm_response(request) })), Arc::new(|| Ok(Box::new(()))), MAX_MESSAGE_BYTES),
                            HeaderMap::new(),
                            None,
                            None,
                            Duration::from_millis(500),
                        )
                        .await;
                        if let Some(finished) = finished.lock().unwrap().take() {
                            let _ = finished.send(());
                        }
                    })
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let (mut socket, _) = connect_async(format!("ws://{address}/")).await.unwrap();
        socket.send(ClientMessage::Text(json!({"type":"response.create","model":"fixture-model","generate":false,"input":"private root"}).to_string().into())).await.unwrap();
        for _ in 0..2 {
            assert!(socket.next().await.unwrap().unwrap().is_text());
        }
        let close = tokio::time::timeout(Duration::from_secs(3), socket.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(close.is_close());
        tokio::time::timeout(Duration::from_secs(3), done)
            .await
            .unwrap()
            .unwrap();
        server.abort();
    }
}

#[cfg(test)]
mod audit_contracts {
    use futures::{SinkExt, StreamExt};
    use tokio_tungstenite::{
        connect_async, tungstenite::Message as ClientMessage, MaybeTlsStream, WebSocketStream,
    };

    use super::*;
    #[cfg(feature = "experimental-responses-websocket")]
    use crate::observability::inflight_tracker::InFlightRequestTracker;

    struct Fixture {
        socket: WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>,
        app: axum::Router,
        calls: Arc<AtomicUsize>,
        server: tokio::task::JoinHandle<()>,
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            self.server.abort();
        }
    }
    #[cfg(feature = "experimental-responses-websocket")]
    struct NativeFixture {
        fixture: Fixture,
        auth: crate::middleware::AuthConfig,
        tracker: Arc<InFlightRequestTracker>,
        drops: Arc<AtomicUsize>,
        url: String,
        bucket: Arc<crate::middleware::TokenBucket>,
    }

    #[cfg(feature = "experimental-responses-websocket")]
    fn admission_state(
        bucket: Arc<crate::middleware::TokenBucket>,
    ) -> Arc<crate::server::AppState> {
        use std::sync::OnceLock;

        use llm_tokenizer::TokenizerRegistry;
        use smg_data_connector::{
            MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
        };

        use crate::{
            app_context::AppContext,
            config::RouterConfig,
            health::ProbeState,
            policies::PolicyRegistry,
            routers::gateway::Gateway,
            server::{http_client, AppState},
            worker::WorkerRegistry,
        };
        let config = RouterConfig::default();
        let context = Arc::new(
            AppContext::builder()
                .client(http_client())
                .rate_limiter(Some(bucket))
                .tokenizer_registry(Arc::new(TokenizerRegistry::new()))
                .reasoning_parser_factory(None)
                .tool_parser_factory(None)
                .worker_registry(Arc::new(WorkerRegistry::new()))
                .policy_registry(Arc::new(PolicyRegistry::new(config.policy.clone())))
                .router_config(config)
                .response_storage(Arc::new(MemoryResponseStorage::new()))
                .conversation_storage(Arc::new(MemoryConversationStorage::new()))
                .conversation_item_storage(Arc::new(MemoryConversationItemStorage::new()))
                .worker_monitor(None)
                .worker_job_queue(Arc::new(OnceLock::new()))
                .workflow_engines(Arc::new(OnceLock::new()))
                .mcp_orchestrator(Arc::new(OnceLock::new()))
                .build()
                .unwrap(),
        );
        Arc::new(AppState {
            secret_sources: Default::default(),
            secrets: None,
            router: Arc::new(Gateway::new(context.worker_registry.clone())),
            probe_state: ProbeState::new(context.inflight_tracker.clone()),
            context,
            admission_queue: None,
            gateway: None,
            mesh_handler: None,
            mesh_adapters: None,
        })
    }
    impl Fixture {
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture owns and aborts the loopback server in Drop"
        )]
        async fn new(deny: bool) -> Self {
            let calls = Arc::new(AtomicUsize::new(0));
            let counted = calls.clone();
            let protected = axum::Router::new().route("/v1/responses", axum::routing::post(move |axum::Json(body): axum::Json<Value>| {
                let counted = counted.clone();
                async move {
                    let number = counted.fetch_add(1, Ordering::SeqCst);
                    if deny {
                        return (StatusCode::FORBIDDEN, axum::Json(json!({"error":{"code":"fixture_denied"}}))).into_response();
                    }
                    let fail = body["input"].as_array().is_some_and(|items| items.last().is_some_and(|item| item["content"] == "fail"));
                    let response = json!({"id":format!("resp_{number}"),"status":if fail {"failed"} else {"completed"},"output":[]});
                    let event = json!({"type":if fail {"response.failed"} else {"response.completed"},"response":response});
                    ([("content-type","text/event-stream")], format!("data: {event}\n\n")).into_response()
                }
            }));
            let dispatch = Dispatch(
                protected
                    .clone()
                    .layer(axum::extract::DefaultBodyLimit::max(512)),
                Arc::new(|| Ok(Box::new(()))),
                512,
            );
            let websocket = axum::Router::new()
                .route("/v1/responses", axum::routing::get(upgrade))
                .layer(Extension(dispatch))
                .layer(axum::middleware::from_fn(capture_credential));
            // The gateway's configured payload cap is outside protected_routes.
            #[cfg(feature = "experimental-responses-websocket")]
            let websocket = websocket.layer(
                crate::middleware::TenantResolutionState::new(
                    &crate::config::RouterConfig::default(),
                )
                .unwrap()
                .layer(),
            );
            let app = protected
                .merge(websocket)
                .layer(axum::extract::DefaultBodyLimit::max(512));
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let served = app.clone();
            let server = tokio::spawn(async move {
                axum::serve(listener, served).await.unwrap();
            });
            let (socket, _) = connect_async(format!("ws://{address}/v1/responses"))
                .await
                .unwrap();
            Self {
                socket,
                app,
                calls,
                server,
            }
        }
        async fn turn(&mut self, body: Value) -> Value {
            self.socket
                .send(ClientMessage::Text(body.to_string().into()))
                .await
                .unwrap();
            loop {
                let message = tokio::time::timeout(Duration::from_secs(3), self.socket.next())
                    .await
                    .unwrap()
                    .unwrap()
                    .unwrap();
                if let ClientMessage::Text(text) = message {
                    let value: Value = serde_json::from_str(&text).unwrap();
                    if matches!(
                        value["type"].as_str(),
                        Some(
                            "error"
                                | "response.completed"
                                | "response.incomplete"
                                | "response.failed"
                        )
                    ) {
                        return value;
                    }
                }
            }
        }

        #[cfg(feature = "experimental-responses-websocket")]
        async fn native() -> NativeFixture {
            Self::native_with_auth(true).await
        }

        #[cfg(feature = "experimental-responses-websocket")]
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture owns and aborts the loopback server in Drop"
        )]
        async fn native_with_auth(authenticated: bool) -> NativeFixture {
            use openai_protocol::validated::ValidatedJson;
            use tokio_tungstenite::tungstenite::client::IntoClientRequest;

            use crate::{
                config::{TenantApiKeyEntry, TenantResolutionConfig},
                middleware::{
                    auth_middleware, AuthConfig, TenantRequestMeta, TenantResolutionState,
                },
                observability::inflight_tracker::InFlightRequestTracker,
            };
            let keys = [
                TenantApiKeyEntry {
                    tenant_id: "team-a".into(),
                    key: "fixture-key-a".into(),
                },
                TenantApiKeyEntry {
                    tenant_id: "team-b".into(),
                    key: "fixture-key-b".into(),
                },
            ];
            let auth = AuthConfig::with_tenant_keys(None, if authenticated { &keys } else { &[] });
            let auth_layer = axum::middleware::from_fn_with_state(auth.clone(), auth_middleware);
            let tenant = TenantResolutionState::from_config(&TenantResolutionConfig {
                trust_tenant_header: authenticated,
                tenant_header_name: "x-tenant-id".into(),
            })
            .unwrap();
            let calls = Arc::new(AtomicUsize::new(0));
            let drops = Arc::new(AtomicUsize::new(0));
            let bucket = Arc::new(crate::middleware::TokenBucket::new(1, 0));
            let admission = admission_state(bucket.clone());
            let counted = calls.clone();
            let dropped = drops.clone();
            struct DropProbe(Arc<AtomicUsize>);
            impl Drop for DropProbe {
                fn drop(&mut self) {
                    self.0.fetch_add(1, Ordering::SeqCst);
                }
            }
            let protected = axum::Router::new().route("/v1/responses", axum::routing::post(move |Extension(meta): Extension<TenantRequestMeta>, prewarm: Option<Extension<Prewarm>>, ValidatedJson(body): ValidatedJson<ResponsesRequest>| {
                let counted = counted.clone();
                let dropped = dropped.clone();
                async move {
                    let number = counted.fetch_add(1, Ordering::SeqCst);
                    if prewarm.is_some() { return prewarm_response(body); }
                    if serde_json::to_string(&body.input).unwrap().contains("stall-fixture") {
                        let probe = DropProbe(dropped);
                        let event = json!({"type":"response.output_text.delta","delta":"x".repeat(1024 * 1024)});
                        let frame = axum::body::Bytes::from(format!("data: {event}\n\n"));
                        let stream = futures::stream::unfold((probe, frame), |(probe, frame)| async move {
                            Some((Ok::<_, std::io::Error>(frame.clone()), (probe, frame)))
                        });
                        return Response::builder().header("content-type", "text/event-stream").body(Body::from_stream(stream)).unwrap();
                    }
                    if serde_json::to_string(&body.input).unwrap().contains("wait-fixture") {
                        let probe = DropProbe(dropped);
                        let pending = futures::stream::unfold(probe, |probe| async move {
                            futures::future::pending::<Option<(Result<axum::body::Bytes, std::io::Error>, DropProbe)>>().await.map(|(chunk, _)| (chunk, probe))
                        });
                        let first = futures::stream::once(async { Ok::<_, std::io::Error>(axum::body::Bytes::from_static(b"data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_wait\"}}\n\n")) });
                        return Response::builder().header("content-type", "text/event-stream").body(Body::from_stream(first.chain(pending))).unwrap();
                    }
                    let tenant = meta.tenant_key().as_str().to_owned();
                    let event = json!({"type":"response.completed","response":{"id":format!("resp_native_{number}"),"status":"completed","output":[{"type":"message","id":"msg_fixture","role":"assistant","content":[{"type":"output_text","text":tenant}]}]}});
                    ([("content-type","text/event-stream")], format!("data: {event}\n\n")).into_response()
                }
            })).layer(crate::middleware::ConcurrencyLimitLayer::new(admission.clone()))
                .layer(crate::middleware::HttpMetricsLayer::new(InFlightRequestTracker::new()))
                .layer(axum::middleware::from_fn(tenant_binding_guard))
            .layer(tenant.clone().layer()).layer(auth_layer.clone()).layer(axum::extract::DefaultBodyLimit::max(4096));
            let tracker = InFlightRequestTracker::new();
            let track = tracker.clone();
            let dispatch = Dispatch(
                protected.clone(),
                Arc::new(move || {
                    let guard = track.track();
                    if track.is_draining() {
                        return Err(
                            json!({"type":"error","status":503,"error":{"code":"gateway_draining"}}),
                        );
                    }
                    Ok(Box::new(guard))
                }),
                4096,
            );
            let websocket = axum::Router::new()
                .route("/v1/responses", axum::routing::get(upgrade))
                .layer(Extension(dispatch))
                .layer(crate::middleware::ConcurrencyLimitLayer::new(admission))
                .layer(crate::middleware::HttpMetricsLayer::new(
                    InFlightRequestTracker::new(),
                ))
                .layer(tenant.layer())
                .layer(auth_layer)
                .layer(axum::middleware::from_fn(capture_credential));
            let app = protected.merge(websocket);
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("ws://{}/v1/responses", listener.local_addr().unwrap());
            let served = app.clone();
            let server = tokio::spawn(async move {
                axum::serve(
                    listener,
                    served.into_make_service_with_connect_info::<SocketAddr>(),
                )
                .await
                .unwrap();
            });
            let mut request = url.clone().into_client_request().unwrap();
            request
                .headers_mut()
                .insert("authorization", "Bearer fixture-key-a".parse().unwrap());
            request
                .headers_mut()
                .insert("x-tenant-id", "team-b".parse().unwrap());
            let (socket, _) = connect_async(request).await.unwrap();
            NativeFixture {
                fixture: Self {
                    socket,
                    app,
                    calls,
                    server,
                },
                auth,
                tracker,
                drops,
                url,
                bucket,
            }
        }
    }

    #[cfg(feature = "experimental-responses-websocket")]
    #[tokio::test]
    async fn native_auth_tenant_scope_rotation_prewarm_and_drain_are_preserved() {
        use tokio_tungstenite::tungstenite::client::IntoClientRequest;

        use crate::{config::TenantApiKeyEntry, middleware::AuthConfig};
        let NativeFixture {
            mut fixture,
            auth,
            tracker,
            url,
            bucket,
            ..
        } = Fixture::native().await;
        for credential in [None, Some("Bearer wrong")] {
            let mut request = Request::builder().uri("/v1/responses");
            if let Some(credential) = credential {
                request = request.header("authorization", credential);
            }
            assert_eq!(
                fixture
                    .app
                    .clone()
                    .oneshot(request.body(Body::empty()).unwrap())
                    .await
                    .unwrap()
                    .status(),
                StatusCode::UNAUTHORIZED
            );
        }
        let first = fixture.turn(json!({"type":"response.create","model":"fixture-model","store":false,"input":"seed"})).await;
        assert_eq!(
            first["response"]["output"][0]["content"][0]["text"],
            "auth:team-a"
        );
        let parent = first["response"]["id"].clone();
        bucket.try_acquire(1.0).unwrap();
        let shed = fixture.turn(json!({"type":"response.create","model":"fixture-model","generate":false,"input":"warm"})).await;
        assert_eq!(
            shed["status"], 429,
            "prewarm must acquire its own native admission slot"
        );
        assert_eq!(fixture.calls.load(Ordering::SeqCst), 1);
        bucket.return_tokens(1.0);
        let warm = fixture.turn(json!({"type":"response.create","model":"fixture-model","store":false,"generate":false,"previous_response_id":parent,"input":"warm"})).await;
        assert_eq!(warm["type"], "response.completed");
        assert_eq!(
            fixture.calls.load(Ordering::SeqCst),
            2,
            "prewarming enters protected dispatch, native auth, tenant resolution and validation"
        );

        let mut request = url.clone().into_client_request().unwrap();
        request
            .headers_mut()
            .insert("authorization", "Bearer fixture-key-b".parse().unwrap());
        let (mut other, _) = connect_async(request).await.unwrap();
        other.send(ClientMessage::Text(json!({"type":"response.create","model":"fixture-model","previous_response_id":parent,"input":"cross-tenant"}).to_string().into())).await.unwrap();
        let event = other.next().await.unwrap().unwrap().into_text().unwrap();
        let event: Value = serde_json::from_str(&event).unwrap();
        assert_eq!(event["error"]["code"], "previous_response_not_found");
        assert_eq!(fixture.calls.load(Ordering::SeqCst), 2);

        auth.replace(AuthConfig::key_map(
            None,
            &[TenantApiKeyEntry {
                tenant_id: "team-b".into(),
                key: "fixture-key-b".into(),
            }],
        ));
        let revoked = fixture.turn(json!({"type":"response.create","model":"fixture-model","generate":false,"input":"warm"})).await;
        assert_eq!(
            revoked["status"], 401,
            "open sockets reauthenticate even on prewarming"
        );
        assert_eq!(fixture.calls.load(Ordering::SeqCst), 2);
        assert!(tracker.wait_for_drain(Duration::from_secs(1)).await);
        fixture.socket = other;
        tracker.begin_drain();
        let draining = fixture
            .turn(json!({"type":"response.create","model":"fixture-model","input":"next"}))
            .await;
        assert_eq!(draining["status"], 503);
        assert!(tracker.wait_for_drain(Duration::from_secs(1)).await);
    }

    #[cfg(feature = "experimental-responses-websocket")]
    #[tokio::test]
    async fn an_unauthenticated_caller_keeps_its_peer_based_admission_identity() {
        let NativeFixture { mut fixture, .. } = Fixture::native_with_auth(false).await;
        let response = fixture
            .turn(json!({"type":"response.create","model":"fixture-model","input":"seed"}))
            .await;
        assert_eq!(
            response["response"]["output"][0]["content"][0]["text"],
            "ip:127.0.0.1"
        );
    }

    #[cfg(feature = "experimental-responses-websocket")]
    #[tokio::test]
    async fn a_reassigned_key_cannot_replay_another_tenants_retained_history() {
        use crate::{config::TenantApiKeyEntry, middleware::AuthConfig};
        let NativeFixture {
            mut fixture,
            auth,
            bucket,
            tracker,
            ..
        } = Fixture::native().await;
        let first = fixture.turn(json!({"type":"response.create","model":"fixture-model","input":"only-team-a-context"})).await;
        assert_eq!(first["type"], "response.completed");
        auth.replace(AuthConfig::key_map(
            None,
            &[TenantApiKeyEntry {
                tenant_id: "team-b".into(),
                key: "fixture-key-a".into(),
            }],
        ));
        let denied = fixture.turn(json!({"type":"response.create","model":"fixture-model","previous_response_id":first["response"]["id"],"input":"next"})).await;
        assert_eq!(denied["status"], 403);
        assert_eq!(denied["error"]["code"], "websocket_tenant_changed");
        assert_eq!(
            fixture.calls.load(Ordering::SeqCst),
            1,
            "old context never reaches the reassigned tenant's backend"
        );
        assert_eq!(
            bucket.available_tokens(),
            1.0,
            "identity changes are refused before admission"
        );
        assert!(tracker.wait_for_drain(Duration::from_secs(1)).await);

        // A fresh HTTP call still uses the new tenant; only the retained
        // history is bound to the identity under which it was accepted.
        let request = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("authorization", "Bearer fixture-key-a")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({"model":"fixture-model","input":"fresh","stream":true,"store":false})
                    .to_string(),
            ))
            .unwrap();
        assert_eq!(
            fixture.app.clone().oneshot(request).await.unwrap().status(),
            StatusCode::OK
        );
    }

    #[test]
    fn durable_storage_and_conversation_modes_are_explicitly_unsupported() {
        for store in [Value::Null, json!("false"), json!(0), json!([]), json!({})] {
            let mut cache = SessionCache::default();
            let error = prepare(json!({"type":"response.create","model":"fixture-model","store":store,"input":"seed"}), &mut cache).err().unwrap();
            assert_eq!(error["error"]["code"], "invalid_type");
            assert_eq!(error["error"]["param"], "store");
        }
        for generate in [true, false] {
            let mut cache = SessionCache::default();
            let error = prepare(json!({"type":"response.create","model":"fixture-model","generate":generate,"store":true,"input":"seed"}), &mut cache).err().unwrap();
            assert_eq!(error["error"]["code"], "unsupported_parameter");
            assert_eq!(error["error"]["param"], "store");
            let error = prepare(json!({"type":"response.create","model":"fixture-model","conversation":"conv_fixture","input":"seed"}), &mut cache).err().unwrap();
            assert_eq!(error["error"]["param"], "conversation");
        }
    }

    #[cfg(feature = "experimental-responses-websocket")]
    #[tokio::test]
    async fn the_lifetime_limit_cancels_an_active_response_and_clears_its_guard() {
        let NativeFixture {
            mut fixture,
            tracker,
            drops,
            bucket,
            ..
        } = Fixture::native().await;
        fixture
            .socket
            .send(ClientMessage::Text(
                json!({"type":"response.create","model":"fixture-model","input":"wait-fixture"})
                    .to_string()
                    .into(),
            ))
            .await
            .unwrap();
        let first = fixture
            .socket
            .next()
            .await
            .unwrap()
            .unwrap()
            .into_text()
            .unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&first).unwrap()["type"],
            "response.created"
        );
        tokio::time::pause();
        tokio::time::advance(CONNECTION_LIFETIME + Duration::from_secs(1)).await;
        let error = fixture
            .socket
            .next()
            .await
            .unwrap()
            .unwrap()
            .into_text()
            .unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&error).unwrap()["error"]["code"],
            "websocket_connection_limit_reached"
        );
        assert!(tracker.wait_for_drain(Duration::from_secs(1)).await);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(bucket.available_tokens(), 1.0);
    }

    #[cfg(feature = "experimental-responses-websocket")]
    #[tokio::test]
    async fn disconnect_drops_the_native_body_and_releases_the_drain_guard() {
        let NativeFixture {
            mut fixture,
            tracker,
            drops,
            bucket,
            ..
        } = Fixture::native().await;
        fixture
            .socket
            .send(ClientMessage::Text(
                json!({"type":"response.create","model":"fixture-model","input":"wait-fixture"})
                    .to_string()
                    .into(),
            ))
            .await
            .unwrap();
        let first = fixture
            .socket
            .next()
            .await
            .unwrap()
            .unwrap()
            .into_text()
            .unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&first).unwrap()["type"],
            "response.created"
        );
        assert_eq!(tracker.len(), 1);
        assert_eq!(bucket.available_tokens(), 0.0);
        fixture.socket.close(None).await.unwrap();
        assert!(tracker.wait_for_drain(Duration::from_secs(2)).await);
        assert_eq!(
            drops.load(Ordering::SeqCst),
            1,
            "the upstream stream is cancelled on disconnect"
        );
        assert_eq!(bucket.available_tokens(), 1.0);
    }

    #[cfg(feature = "experimental-responses-websocket")]
    #[tokio::test]
    async fn a_client_that_stops_reading_releases_the_body_and_guard() {
        let NativeFixture {
            mut fixture,
            tracker,
            drops,
            bucket,
            ..
        } = Fixture::native().await;
        fixture
            .socket
            .send(ClientMessage::Text(
                json!({"type":"response.create","model":"fixture-model","input":"stall-fixture"})
                    .to_string()
                    .into(),
            ))
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            while tracker.is_empty() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        // The client deliberately never polls its reader. A finite socket
        // window fills and the per-write deadline must cancel the body.
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(tracker.len(), 1);
        assert_eq!(bucket.available_tokens(), 0.0);
        assert!(
            tracker
                .wait_for_drain(WRITE_TIMEOUT + Duration::from_secs(3))
                .await
        );
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(bucket.available_tokens(), 1.0);
    }

    #[tokio::test]
    async fn prewarm_obeys_protected_request_admission() {
        let mut fixture = Fixture::new(true).await;
        let terminal = fixture.turn(json!({"type":"response.create","model":"fixture-model","generate":false,"input":"warm"})).await;
        assert_eq!(
            terminal["type"], "error",
            "prewarming must not bypass a protected service refusal"
        );
        assert_eq!(fixture.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn failed_replacement_preserves_the_previous_response() {
        let mut fixture = Fixture::new(false).await;
        let first = fixture
            .turn(json!({"type":"response.create","model":"fixture-model","input":"seed"}))
            .await;
        let parent = first["response"]["id"].clone();
        let failed = fixture.turn(json!({"type":"response.create","model":"fixture-model","previous_response_id":null,"input":"fail"})).await;
        assert_eq!(failed["type"], "response.failed");
        let retry = fixture.turn(json!({"type":"response.create","model":"fixture-model","previous_response_id":parent,"input":"retry"})).await;
        assert_eq!(
            retry["type"], "response.completed",
            "a failed replacement must not delete a valid parent"
        );
    }

    #[tokio::test]
    async fn websocket_obeys_the_configured_http_payload_cap() {
        let mut fixture = Fixture::new(false).await;
        let body = json!({"model":"fixture-model","input":"x".repeat(1024)});
        let http = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        assert_eq!(
            fixture.app.clone().oneshot(http).await.unwrap().status(),
            StatusCode::PAYLOAD_TOO_LARGE
        );
        let first = fixture
            .turn(json!({"type":"response.create","model":"fixture-model","input":"y".repeat(250)}))
            .await;
        assert_eq!(first["type"], "response.completed");
        let terminal = fixture.turn(json!({"type":"response.create","model":"fixture-model","previous_response_id":first["response"]["id"],"input":"z".repeat(250)})).await;
        assert_eq!(
            terminal["type"], "error",
            "expanded continuations must preserve the operator's payload limit"
        );
        assert_eq!(fixture.calls.load(Ordering::SeqCst), 1);
    }
}
