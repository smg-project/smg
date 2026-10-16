//! The exact-match response cache (`--response-cache memory`).
//!
//! An identical deterministic request is answered from memory instead of
//! being generated again. A request is cacheable when it is not streamed,
//! asks for one choice, uses no tools and no engine session, and samples at
//! temperature 0 (or with a seed). Its key is the tenant it was resolved to
//! and the canonical form of its body: the typed request serialized with
//! every object's keys sorted and the fields that identify the caller rather
//! than the answer (`user`, `metadata`, ...) left out, so two callers of one
//! tenant who send the same request in a different key order share an entry
//! and two tenants never do. Without tenant identity the tenant key is the
//! client address, so entries are shared by one client address only.
//!
//! Only a complete `200` JSON answer is stored, recorded while it streams to
//! the client (so a miss costs no extra latency) and dropped unrecorded when
//! the body fails, is cut short or outgrows the per-entry cap. Entries live
//! for the configured TTL (a request may shorten its own with
//! `x-smg-cache: ttl=N`, or opt out with `x-smg-cache: no-store`) inside an
//! entry and byte budget the least recently used entries leave first.
//!
//! Every answer the cache saw carries `x-smg-cache-status`: `hit` (served
//! from memory, with `age`), `miss` (generated) or `bypass` (not cacheable).
//! Off by default: nothing on the serving path changes while it is off.

use std::{
    future::Future,
    num::NonZeroUsize,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::{Duration, Instant},
};

use axum::{
    body::Body,
    http::{
        header::{AGE, CONTENT_TYPE},
        HeaderMap, HeaderName, HeaderValue, StatusCode,
    },
    response::Response,
};
use bytes::Bytes;
use lru::LruCache;
use openai_protocol::{
    chat::ChatCompletionRequest, completion::CompletionRequest, messages::CreateMessageRequest,
};
use parking_lot::Mutex;
use serde::Serialize;
use serde_json::Value;
use tracing::{debug, info};

use crate::{
    config::{ResponseCacheBackend, ResponseCacheConfig},
    middleware::TenantRequestMeta,
    observability::metrics::Metrics,
};

/// The request header a caller steers the cache with: `no-store` (neither
/// looked up nor stored) or `ttl=<seconds>` (a shorter life for this
/// answer; `ttl=0` is `no-store`). Unknown directives are ignored.
pub static HEADER_CACHE_CONTROL: HeaderName = HeaderName::from_static("x-smg-cache");
/// The response header every answer the cache saw carries.
pub static HEADER_CACHE_STATUS: HeaderName = HeaderName::from_static("x-smg-cache-status");

/// `x-smg-cache-status` and the `result` label of
/// `smg_response_cache_requests_total`.
pub const STATUS_HIT: &str = "hit";
pub const STATUS_MISS: &str = "miss";
pub const STATUS_BYPASS: &str = "bypass";

/// The key of an entry: a 256-bit hash of the scope, the tenant and the
/// canonical request.
pub type Key = [u8; 32];

/// Fixed per-entry cost charged against the byte budget besides the body:
/// the key, the timestamps, the content type and the LRU links.
const ENTRY_OVERHEAD_BYTES: usize = 128;

/// The fields a request's canonical form leaves out: they identify the
/// caller or the delivery, not the answer. Streaming and the choice count
/// are already pinned by eligibility, so `stream: false` and `n: 1` key as
/// their absence does.
const STRIPPED_FIELDS: [&str; 9] = [
    "stream",
    "stream_options",
    "n",
    "best_of",
    "user",
    "metadata",
    "safety_identifier",
    "prompt_cache_key",
    "rid",
];

/// The endpoint a request came through. Part of the key, so a chat
/// completion and a completion with the same JSON never share an answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    Chat,
    Completion,
    Messages,
}

impl Scope {
    fn tag(self) -> &'static [u8] {
        match self {
            Scope::Chat => b"chat",
            Scope::Completion => b"completion",
            Scope::Messages => b"messages",
        }
    }
}

/// A request type the cache can answer.
pub trait Cacheable: Serialize {
    const SCOPE: Scope;

    /// Why the request cannot be cached, or `None` when it can: streamed,
    /// more than one choice, tool use, an engine session, or sampling.
    fn bypass_reason(&self) -> Option<&'static str>;
}

impl Cacheable for ChatCompletionRequest {
    const SCOPE: Scope = Scope::Chat;

    #[expect(
        deprecated,
        reason = "the legacy function fields and seed still arrive and still decide"
    )]
    fn bypass_reason(&self) -> Option<&'static str> {
        if self.stream {
            return Some("stream");
        }
        if self.n.is_some_and(|n| n > 1) {
            return Some("choices");
        }
        if self.tools.as_ref().is_some_and(|tools| !tools.is_empty())
            || self.tool_choice.is_some()
            || self.functions.is_some()
            || self.function_call.is_some()
        {
            return Some("tools");
        }
        if self.session_params.is_some() {
            return Some("session");
        }
        if is_deterministic(
            self.temperature,
            self.seed.is_some() || self.sampling_seed.is_some(),
        ) {
            None
        } else {
            Some("sampled")
        }
    }
}

impl Cacheable for CompletionRequest {
    const SCOPE: Scope = Scope::Completion;

    fn bypass_reason(&self) -> Option<&'static str> {
        if self.stream {
            return Some("stream");
        }
        if self.n.is_some_and(|n| n > 1) || self.best_of.is_some_and(|best_of| best_of > 1) {
            return Some("choices");
        }
        if self.session_params.is_some() {
            return Some("session");
        }
        if is_deterministic(self.temperature, self.seed.is_some()) {
            None
        } else {
            Some("sampled")
        }
    }
}

impl Cacheable for CreateMessageRequest {
    const SCOPE: Scope = Scope::Messages;

    fn bypass_reason(&self) -> Option<&'static str> {
        if self.stream == Some(true) {
            return Some("stream");
        }
        if self.tools.as_ref().is_some_and(|tools| !tools.is_empty())
            || self.tool_choice.is_some()
            || self.mcp_servers.is_some()
            || self.container.is_some()
        {
            return Some("tools");
        }
        if self.temperature == Some(0.0) {
            None
        } else {
            Some("sampled")
        }
    }
}

/// Deterministic sampling: temperature 0, or a seed the engine replays.
fn is_deterministic<T: Into<f64> + Copy>(temperature: Option<T>, seeded: bool) -> bool {
    temperature.is_some_and(|t| t.into() == 0.0) || seeded
}

/// What the cache decided about a request before the router runs.
#[derive(Debug)]
pub enum Decision {
    /// Not cacheable: the router's answer is tagged `bypass` and not stored.
    Bypass,
    /// Cacheable: look `key` up, and store the router's answer under it for
    /// `ttl` on a miss.
    Lookup { key: Key, ttl: Duration },
}

/// What `x-smg-cache` asked for.
#[derive(Debug, PartialEq, Eq)]
enum Directive {
    Default,
    NoStore,
    Ttl(Duration),
}

impl Directive {
    fn parse(headers: Option<&HeaderMap>) -> Self {
        let Some(value) = headers
            .and_then(|headers| headers.get(&HEADER_CACHE_CONTROL))
            .and_then(|value| value.to_str().ok())
        else {
            return Directive::Default;
        };
        let mut directive = Directive::Default;
        for part in value.split(',') {
            let part = part.trim();
            if part.eq_ignore_ascii_case("no-store") {
                return Directive::NoStore;
            }
            if let Some(secs) = part
                .get(..4)
                .filter(|prefix| prefix.eq_ignore_ascii_case("ttl="))
                .and_then(|_| part[4..].trim().parse::<u64>().ok())
            {
                directive = match secs {
                    0 => return Directive::NoStore,
                    secs => Directive::Ttl(Duration::from_secs(secs)),
                };
            }
        }
        directive
    }
}

/// The canonical bytes of a request: its JSON with the caller-identifying
/// fields removed and every object's keys sorted, compact. `None` when the
/// request does not serialize (it always does; the cache then stands aside).
pub fn canonical_bytes<R: Serialize>(request: &R) -> Option<Vec<u8>> {
    let mut value = serde_json::to_value(request).ok()?;
    if let Value::Object(map) = &mut value {
        for field in STRIPPED_FIELDS {
            map.remove(field);
        }
    }
    sort_keys(&mut value);
    serde_json::to_vec(&value).ok()
}

fn sort_keys(value: &mut Value) {
    match value {
        Value::Object(map) => {
            map.sort_keys();
            for value in map.values_mut() {
                sort_keys(value);
            }
        }
        Value::Array(items) => items.iter_mut().for_each(sort_keys),
        _ => {}
    }
}

/// The key of `canonical` for `tenant` under `scope`.
pub fn key_for(scope: Scope, tenant: &str, canonical: &[u8]) -> Key {
    let mut hasher = blake3::Hasher::new();
    hasher.update(scope.tag());
    hasher.update(&[0]);
    hasher.update(tenant.as_bytes());
    hasher.update(&[0]);
    hasher.update(canonical);
    *hasher.finalize().as_bytes()
}

/// A stored answer.
#[derive(Clone, Debug)]
struct Entry {
    body: Bytes,
    content_type: HeaderValue,
    stored_at: Instant,
    expires_at: Instant,
}

impl Entry {
    fn cost(&self) -> usize {
        self.body.len() + self.content_type.len() + ENTRY_OVERHEAD_BYTES
    }
}

struct Store {
    entries: LruCache<Key, Entry>,
    bytes: usize,
}

/// The in-memory exact-match response cache: one per gateway, shared by
/// every tenant, bounded by entries and bytes, with a TTL per entry.
pub struct ResponseCache {
    store: Mutex<Store>,
    ttl: Duration,
    max_bytes: usize,
    max_entry_bytes: usize,
}

impl std::fmt::Debug for ResponseCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let store = self.store.lock();
        f.debug_struct("ResponseCache")
            .field("entries", &store.entries.len())
            .field("bytes", &store.bytes)
            .field("ttl", &self.ttl)
            .field("max_bytes", &self.max_bytes)
            .field("max_entry_bytes", &self.max_entry_bytes)
            .finish()
    }
}

impl ResponseCache {
    /// The cache the configuration asks for: `None` while it is off.
    pub fn from_config(config: &ResponseCacheConfig) -> Option<Arc<Self>> {
        match config.backend {
            ResponseCacheBackend::Off => None,
            ResponseCacheBackend::Memory => {
                info!(
                    ttl_secs = config.ttl_secs,
                    max_entries = config.max_entries,
                    max_bytes = config.max_bytes,
                    max_entry_bytes = config.max_entry_bytes,
                    "Response cache on: exact-match, in memory"
                );
                Some(Arc::new(Self::new(
                    Duration::from_secs(config.ttl_secs),
                    config.max_entries,
                    config.max_bytes,
                    config.max_entry_bytes,
                )))
            }
        }
    }

    /// A cache of at most `max_entries` answers and `max_bytes` in all, each
    /// at most `max_entry_bytes`, kept for `ttl`.
    pub fn new(
        ttl: Duration,
        max_entries: usize,
        max_bytes: usize,
        max_entry_bytes: usize,
    ) -> Self {
        Metrics::init_response_cache_series();
        Self {
            store: Mutex::new(Store {
                entries: LruCache::new(NonZeroUsize::new(max_entries).unwrap_or(NonZeroUsize::MIN)),
                bytes: 0,
            }),
            ttl,
            max_bytes,
            max_entry_bytes: max_entry_bytes.min(max_bytes),
        }
    }

    /// What to do with `request` from `tenant`, before the router runs.
    pub fn decide<R: Cacheable>(
        &self,
        headers: Option<&HeaderMap>,
        tenant: &TenantRequestMeta,
        request: &R,
    ) -> Decision {
        let directive = Directive::parse(headers);
        if directive == Directive::NoStore {
            return bypass("no-store");
        }
        if let Some(reason) = request.bypass_reason() {
            return bypass(reason);
        }
        let Some(canonical) = canonical_bytes(request) else {
            return bypass("serialization");
        };
        let key = key_for(R::SCOPE, tenant.tenant_key().as_str(), &canonical);
        let ttl = match directive {
            Directive::Ttl(ttl) => ttl.min(self.ttl),
            Directive::Default | Directive::NoStore => self.ttl,
        };
        Decision::Lookup { key, ttl }
    }

    /// The stored answer for `key` as the client's response, or `None`.
    pub fn hit(&self, key: &Key) -> Option<Response> {
        let now = Instant::now();
        let entry = self.lookup_at(key, now)?;
        Metrics::record_response_cache_request(STATUS_HIT);
        let age = now.saturating_duration_since(entry.stored_at).as_secs();
        let mut response = Response::new(Body::from(entry.body));
        let headers = response.headers_mut();
        headers.insert(CONTENT_TYPE, entry.content_type);
        headers.insert(AGE, HeaderValue::from(age));
        headers.insert(
            HEADER_CACHE_STATUS.clone(),
            HeaderValue::from_static(STATUS_HIT),
        );
        Some(response)
    }

    /// The router's answer on a miss: tagged `miss`, and when it is a `200`
    /// JSON answer, recorded as it streams to the client and stored under
    /// `key` for `ttl` once it is complete.
    pub fn record(self: &Arc<Self>, key: Key, ttl: Duration, response: Response) -> Response {
        Metrics::record_response_cache_request(STATUS_MISS);
        let mut response = tag(response, STATUS_MISS);
        if response.status() != StatusCode::OK {
            return response;
        }
        let Some(content_type) = response
            .headers()
            .get(CONTENT_TYPE)
            .filter(|value| is_json(value))
            .cloned()
        else {
            return response;
        };
        let body = std::mem::take(response.body_mut());
        let hint = http_body::Body::size_hint(&body);
        let cap = self.max_entry_bytes;
        if hint.lower() > cap as u64 || http_body::Body::is_end_stream(&body) {
            *response.body_mut() = body;
            return response;
        }
        let expected = hint.upper().unwrap_or_else(|| hint.lower()).min(cap as u64) as usize;
        *response.body_mut() = Body::new(RecordingBody {
            inner: body,
            recorded: Vec::with_capacity(expected),
            overflowed: false,
            pending: Some(Pending {
                cache: Arc::clone(self),
                key,
                ttl,
                content_type,
            }),
        });
        response
    }

    /// Serve `request` through the cache: `run` generates the answer on a
    /// miss or a bypass.
    pub async fn serve<R, Fut>(
        self: &Arc<Self>,
        headers: Option<&HeaderMap>,
        tenant: &TenantRequestMeta,
        request: R,
        run: impl FnOnce(R) -> Fut,
    ) -> Response
    where
        R: Cacheable,
        Fut: Future<Output = Response>,
    {
        match self.decide(headers, tenant, &request) {
            Decision::Bypass => tag(run(request).await, STATUS_BYPASS),
            Decision::Lookup { key, ttl } => {
                if let Some(hit) = self.hit(&key) {
                    return hit;
                }
                self.record(key, ttl, run(request).await)
            }
        }
    }

    /// The live entry for `key` at `now`, made the most recently used; an
    /// expired one is dropped and a miss.
    fn lookup_at(&self, key: &Key, now: Instant) -> Option<Entry> {
        let mut store = self.store.lock();
        let expired = match store.entries.get(key) {
            None => return None,
            Some(entry) if now < entry.expires_at => return Some(entry.clone()),
            Some(_) => store.entries.pop(key)?,
        };
        store.bytes -= expired.cost();
        Metrics::record_response_cache_change(1, store.bytes, store.entries.len());
        None
    }

    /// Store `body` under `key` for `ttl` from `now`; the least recently
    /// used entries leave until the budgets hold. A body over the per-entry
    /// cap is not stored.
    fn insert_at(
        &self,
        key: Key,
        body: Bytes,
        content_type: HeaderValue,
        ttl: Duration,
        now: Instant,
    ) {
        let entry = Entry {
            body,
            content_type,
            stored_at: now,
            expires_at: now + ttl,
        };
        let cost = entry.cost();
        if cost > self.max_entry_bytes {
            return;
        }
        let mut store = self.store.lock();
        let mut evicted = 0;
        // `push` hands back the entry this one replaced under the same key
        // or the one the entry budget pushed out; either leaves the budget.
        if let Some((_, old)) = store.entries.push(key, entry) {
            store.bytes -= old.cost();
            evicted += 1;
        }
        store.bytes += cost;
        while store.bytes > self.max_bytes {
            let Some((_, old)) = store.entries.pop_lru() else {
                break;
            };
            store.bytes -= old.cost();
            evicted += 1;
        }
        Metrics::record_response_cache_change(evicted, store.bytes, store.entries.len());
    }

    /// Answers held and bytes charged, for tests and diagnostics.
    pub fn occupancy(&self) -> (usize, usize) {
        let store = self.store.lock();
        (store.entries.len(), store.bytes)
    }
}

/// The decision for a request the cache stands aside from, counted.
fn bypass(reason: &'static str) -> Decision {
    debug!(reason, "Response cache bypassed");
    Metrics::record_response_cache_request(STATUS_BYPASS);
    Decision::Bypass
}

/// Tag `response` with its `x-smg-cache-status`.
pub fn tag(mut response: Response, status: &'static str) -> Response {
    response.headers_mut().insert(
        HEADER_CACHE_STATUS.clone(),
        HeaderValue::from_static(status),
    );
    response
}

fn is_json(content_type: &HeaderValue) -> bool {
    content_type.to_str().is_ok_and(|value| {
        value
            .trim_start()
            .get(..16)
            .is_some_and(|mime| mime.eq_ignore_ascii_case("application/json"))
    })
}

/// Where a recorded body goes once it is complete.
struct Pending {
    cache: Arc<ResponseCache>,
    key: Key,
    ttl: Duration,
    content_type: HeaderValue,
}

/// A response body copied into the cache as it streams to the client. The
/// copy is stored at the end of a complete body, never for one that failed,
/// was dropped early (the client went away) or outgrew the per-entry cap.
struct RecordingBody {
    inner: Body,
    recorded: Vec<u8>,
    overflowed: bool,
    pending: Option<Pending>,
}

impl RecordingBody {
    fn record(&mut self, data: &Bytes) {
        if self.overflowed || self.pending.is_none() {
            return;
        }
        let cap = self
            .pending
            .as_ref()
            .map_or(0, |pending| pending.cache.max_entry_bytes);
        if self.recorded.len() + data.len() > cap.saturating_sub(ENTRY_OVERHEAD_BYTES) {
            self.overflowed = true;
            self.recorded = Vec::new();
            self.pending = None;
            return;
        }
        self.recorded.extend_from_slice(data);
    }

    fn complete(&mut self) {
        let Some(pending) = self.pending.take() else {
            return;
        };
        if self.overflowed {
            return;
        }
        let body = Bytes::from(std::mem::take(&mut self.recorded));
        pending.cache.insert_at(
            pending.key,
            body,
            pending.content_type,
            pending.ttl,
            Instant::now(),
        );
    }
}

impl http_body::Body for RecordingBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();
        let poll = Pin::new(&mut this.inner).poll_frame(cx);
        match &poll {
            Poll::Ready(None) => this.complete(),
            Poll::Ready(Some(Err(_))) => this.pending = None,
            Poll::Ready(Some(Ok(frame))) => {
                if let Some(data) = frame.data_ref() {
                    this.record(data);
                }
                // A consumer may stop at the last frame and never ask for
                // the trailing None (an HTTP/1.1 server stops at the
                // declared length).
                if this.inner.is_end_stream() {
                    this.complete();
                }
            }
            Poll::Pending => {}
        }
        poll
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        future::poll_fn,
        io,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use axum::{body::to_bytes, http::header::CONTENT_TYPE, response::IntoResponse, Json};
    use futures_util::stream;
    use serde_json::json;

    use super::*;
    use crate::middleware::{RouteRequestMeta, TenantKey};

    fn chat(value: Value) -> ChatCompletionRequest {
        serde_json::from_value(value).unwrap()
    }

    fn completion(value: Value) -> CompletionRequest {
        serde_json::from_value(value).unwrap()
    }

    fn messages(value: Value) -> CreateMessageRequest {
        serde_json::from_value(value).unwrap()
    }

    fn deterministic_chat() -> Value {
        json!({"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": 0})
    }

    fn tenant(name: &str) -> TenantRequestMeta {
        RouteRequestMeta::new(TenantKey::from(name))
    }

    fn cache() -> Arc<ResponseCache> {
        Arc::new(ResponseCache::new(
            Duration::from_secs(60),
            16,
            1 << 20,
            1 << 16,
        ))
    }

    fn key_of<R: Cacheable>(request: &R, tenant_name: &str) -> Key {
        match cache().decide(None, &tenant(tenant_name), request) {
            Decision::Lookup { key, .. } => key,
            Decision::Bypass => panic!("expected a cacheable request"),
        }
    }

    fn cache_header(value: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            HEADER_CACHE_CONTROL.clone(),
            HeaderValue::from_str(value).unwrap(),
        );
        headers
    }

    fn json_response(value: Value) -> Response {
        Json(value).into_response()
    }

    async fn drain(response: Response) -> (StatusCode, HeaderMap, Bytes) {
        let status = response.status();
        let headers = response.headers().clone();
        let body = to_bytes(response.into_body(), 1 << 20).await.unwrap();
        (status, headers, body)
    }

    #[test]
    fn a_deterministic_unstreamed_chat_request_is_cacheable_by_temperature_or_seed() {
        assert_eq!(chat(deterministic_chat()).bypass_reason(), None);
        let mut seeded = deterministic_chat();
        seeded["temperature"] = json!(0.9);
        seeded["seed"] = json!(7);
        assert_eq!(chat(seeded).bypass_reason(), None);
        let mut engine_seeded = deterministic_chat();
        engine_seeded.as_object_mut().unwrap().remove("temperature");
        engine_seeded["sampling_seed"] = json!(7);
        assert_eq!(chat(engine_seeded).bypass_reason(), None);
    }

    #[test]
    fn sampled_streamed_multi_choice_tool_and_session_chat_requests_bypass() {
        let mut sampled = deterministic_chat();
        sampled["temperature"] = json!(0.7);
        assert_eq!(chat(sampled).bypass_reason(), Some("sampled"));
        let mut unset = deterministic_chat();
        unset.as_object_mut().unwrap().remove("temperature");
        assert_eq!(chat(unset).bypass_reason(), Some("sampled"));
        let mut streamed = deterministic_chat();
        streamed["stream"] = json!(true);
        assert_eq!(chat(streamed).bypass_reason(), Some("stream"));
        let mut choices = deterministic_chat();
        choices["n"] = json!(2);
        assert_eq!(chat(choices).bypass_reason(), Some("choices"));
        let mut tools = deterministic_chat();
        tools["tools"] = json!([{"type": "function", "function": {"name": "f"}}]);
        assert_eq!(chat(tools).bypass_reason(), Some("tools"));
        let mut functions = deterministic_chat();
        functions["functions"] = json!([{"name": "f"}]);
        assert_eq!(chat(functions).bypass_reason(), Some("tools"));
        let mut session = deterministic_chat();
        session["session_params"] = json!({"id": "s"});
        assert_eq!(chat(session).bypass_reason(), Some("session"));
        // One choice and an empty tool list are the plain request.
        let mut plain = deterministic_chat();
        plain["n"] = json!(1);
        plain["tools"] = json!([]);
        assert_eq!(chat(plain).bypass_reason(), None);
    }

    #[test]
    fn completion_and_message_eligibility_follow_their_own_fields() {
        let base = json!({"model": "m", "prompt": "hi", "temperature": 0});
        assert_eq!(completion(base.clone()).bypass_reason(), None);
        let mut best_of = base.clone();
        best_of["best_of"] = json!(3);
        assert_eq!(completion(best_of).bypass_reason(), Some("choices"));
        let mut seeded = base.clone();
        seeded["temperature"] = json!(1.0);
        seeded["seed"] = json!(3);
        assert_eq!(completion(seeded).bypass_reason(), None);
        let mut streamed = base;
        streamed["stream"] = json!(true);
        assert_eq!(completion(streamed).bypass_reason(), Some("stream"));

        let base = json!({"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}], "temperature": 0});
        assert_eq!(messages(base.clone()).bypass_reason(), None);
        let mut sampled = base.clone();
        sampled.as_object_mut().unwrap().remove("temperature");
        assert_eq!(messages(sampled).bypass_reason(), Some("sampled"));
        let mut tools = base.clone();
        tools["tools"] = json!([{"name": "t", "input_schema": {"type": "object"}}]);
        assert_eq!(messages(tools).bypass_reason(), Some("tools"));
        let mut mcp = base.clone();
        mcp["mcp_servers"] = json!([{"type": "url", "url": "http://127.0.0.1:1/mcp", "name": "s"}]);
        assert_eq!(messages(mcp).bypass_reason(), Some("tools"));
        let mut streamed = base;
        streamed["stream"] = json!(true);
        assert_eq!(messages(streamed).bypass_reason(), Some("stream"));
    }

    #[test]
    fn the_key_ignores_key_order_map_order_and_caller_identity() {
        let a = chat(json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0,
            "logit_bias": {"1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0},
            "user": "alice",
            "metadata": {"trace": "x"},
            "stream": false,
            "n": 1,
            "x_custom": {"b": 1, "a": 2}
        }));
        let b = chat(json!({
            "x_custom": {"a": 2, "b": 1},
            "logit_bias": {"4": 4.0, "3": 3.0, "2": 2.0, "1": 1.0},
            "temperature": 0.0,
            "messages": [{"content": "hi", "role": "user"}],
            "model": "m",
            "user": "bob",
            "prompt_cache_key": "p"
        }));
        assert_eq!(key_of(&a, "t"), key_of(&b, "t"));
        assert_eq!(canonical_bytes(&a).unwrap(), canonical_bytes(&b).unwrap());
    }

    #[test]
    fn the_key_separates_tenants_scopes_and_answers_that_differ() {
        let request = chat(deterministic_chat());
        assert_ne!(key_of(&request, "header:a"), key_of(&request, "header:b"));
        let mut other = deterministic_chat();
        other["messages"][0]["content"] = json!("hello");
        assert_ne!(key_of(&request, "t"), key_of(&chat(other), "t"));
        let mut max_tokens = deterministic_chat();
        max_tokens["max_tokens"] = json!(4);
        assert_ne!(key_of(&request, "t"), key_of(&chat(max_tokens), "t"));
        let canonical = canonical_bytes(&request).unwrap();
        assert_ne!(
            key_for(Scope::Chat, "t", &canonical),
            key_for(Scope::Completion, "t", &canonical)
        );
    }

    #[test]
    fn the_cache_header_opts_out_or_shortens_the_ttl() {
        assert_eq!(Directive::parse(None), Directive::Default);
        assert_eq!(
            Directive::parse(Some(&cache_header("no-store"))),
            Directive::NoStore
        );
        assert_eq!(
            Directive::parse(Some(&cache_header("No-Store"))),
            Directive::NoStore
        );
        assert_eq!(
            Directive::parse(Some(&cache_header("ttl=0"))),
            Directive::NoStore
        );
        assert_eq!(
            Directive::parse(Some(&cache_header("ttl=5"))),
            Directive::Ttl(Duration::from_secs(5))
        );
        assert_eq!(
            Directive::parse(Some(&cache_header("other, TTL= 9"))),
            Directive::Ttl(Duration::from_secs(9))
        );
        assert_eq!(
            Directive::parse(Some(&cache_header("ttl=x"))),
            Directive::Default
        );
        assert_eq!(
            Directive::parse(Some(&cache_header("whatever"))),
            Directive::Default
        );

        let cache = cache();
        let request = chat(deterministic_chat());
        assert!(matches!(
            cache.decide(Some(&cache_header("no-store")), &tenant("t"), &request),
            Decision::Bypass
        ));
        match cache.decide(Some(&cache_header("ttl=5")), &tenant("t"), &request) {
            Decision::Lookup { ttl, .. } => assert_eq!(ttl, Duration::from_secs(5)),
            Decision::Bypass => panic!("cacheable"),
        }
        // A requested TTL never outlives the configured one.
        match cache.decide(Some(&cache_header("ttl=5000")), &tenant("t"), &request) {
            Decision::Lookup { ttl, .. } => assert_eq!(ttl, Duration::from_secs(60)),
            Decision::Bypass => panic!("cacheable"),
        }
    }

    #[test]
    fn entries_expire_on_touch_and_the_oldest_leave_at_the_budgets() {
        let cache = ResponseCache::new(Duration::from_secs(10), 2, 10_000, 10_000);
        let now = Instant::now();
        let ct = HeaderValue::from_static("application/json");
        let body = |n: u8| Bytes::from(vec![n; 100]);
        cache.insert_at([1; 32], body(1), ct.clone(), Duration::from_secs(10), now);
        assert!(cache
            .lookup_at(&[1; 32], now + Duration::from_secs(9))
            .is_some());
        assert!(cache
            .lookup_at(&[1; 32], now + Duration::from_secs(10))
            .is_none());
        assert_eq!(cache.occupancy(), (0, 0));

        // Three entries into an entry budget of two: the least recently used goes.
        cache.insert_at([1; 32], body(1), ct.clone(), Duration::from_secs(10), now);
        cache.insert_at([2; 32], body(2), ct.clone(), Duration::from_secs(10), now);
        assert!(cache.lookup_at(&[1; 32], now).is_some());
        cache.insert_at([3; 32], body(3), ct.clone(), Duration::from_secs(10), now);
        assert!(cache.lookup_at(&[2; 32], now).is_none());
        assert!(cache.lookup_at(&[1; 32], now).is_some());
        assert!(cache.lookup_at(&[3; 32], now).is_some());
        let per_entry = 100 + ct.len() + ENTRY_OVERHEAD_BYTES;
        assert_eq!(cache.occupancy(), (2, 2 * per_entry));

        // A byte budget below two entries keeps one; an entry over the cap is not stored.
        let small = ResponseCache::new(Duration::from_secs(10), 16, 2 * per_entry - 1, per_entry);
        small.insert_at([1; 32], body(1), ct.clone(), Duration::from_secs(10), now);
        small.insert_at([2; 32], body(2), ct.clone(), Duration::from_secs(10), now);
        assert_eq!(small.occupancy(), (1, per_entry));
        assert!(small.lookup_at(&[2; 32], now).is_some());
        small.insert_at(
            [3; 32],
            Bytes::from(vec![3; 101]),
            ct,
            Duration::from_secs(10),
            now,
        );
        assert_eq!(small.occupancy(), (1, per_entry));
    }

    #[tokio::test]
    async fn a_complete_json_answer_is_stored_and_replayed_with_the_hit_headers() {
        let cache = cache();
        let key = [7; 32];
        let answer = json!({"id": "chatcmpl-1", "choices": []});
        let response = cache.record(key, Duration::from_secs(30), json_response(answer.clone()));
        let (status, headers, body) = drain(response).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(headers.get(&HEADER_CACHE_STATUS).unwrap(), STATUS_MISS);
        assert_eq!(serde_json::from_slice::<Value>(&body).unwrap(), answer);
        assert_eq!(cache.occupancy().0, 1);

        let (status, headers, replay) = drain(cache.hit(&key).unwrap()).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(headers.get(&HEADER_CACHE_STATUS).unwrap(), STATUS_HIT);
        assert_eq!(headers.get(CONTENT_TYPE).unwrap(), "application/json");
        assert!(headers.contains_key(AGE));
        assert_eq!(replay, body);
    }

    #[tokio::test]
    async fn errors_non_json_cut_and_oversized_answers_are_never_stored() {
        let cache = cache();
        let error = (StatusCode::BAD_GATEWAY, Json(json!({"error": "x"}))).into_response();
        let (_, headers, _) = drain(cache.record([1; 32], Duration::from_secs(30), error)).await;
        assert_eq!(headers.get(&HEADER_CACHE_STATUS).unwrap(), STATUS_MISS);
        assert!(cache.hit(&[1; 32]).is_none());

        let text = (StatusCode::OK, "plain").into_response();
        drain(cache.record([2; 32], Duration::from_secs(30), text)).await;
        assert!(cache.hit(&[2; 32]).is_none());

        let failing = Response::builder()
            .status(StatusCode::OK)
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from_stream(stream::iter([
                Ok::<_, io::Error>(Bytes::from_static(b"{\"partial\":")),
                Err(io::Error::other("cut")),
            ])))
            .unwrap();
        let response = cache.record([3; 32], Duration::from_secs(30), failing);
        assert!(to_bytes(response.into_body(), 1 << 20).await.is_err());
        assert!(cache.hit(&[3; 32]).is_none());

        let dropped = Response::builder()
            .status(StatusCode::OK)
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from_stream(stream::iter([
                Ok::<_, io::Error>(Bytes::from_static(b"{\"partial\":")),
                Ok(Bytes::from_static(b"true}")),
            ])))
            .unwrap();
        let response = cache.record([4; 32], Duration::from_secs(30), dropped);
        let mut body = response.into_body();
        let first = poll_fn(|cx| http_body::Body::poll_frame(Pin::new(&mut body), cx)).await;
        assert!(first.is_some());
        drop(body);
        assert!(cache.hit(&[4; 32]).is_none());

        let big = json!({"text": "x".repeat(1 << 16)});
        drain(cache.record([5; 32], Duration::from_secs(30), json_response(big))).await;
        assert!(cache.hit(&[5; 32]).is_none());
        assert_eq!(cache.occupancy(), (0, 0));
    }

    #[tokio::test]
    async fn serve_answers_the_second_identical_request_from_memory_and_bypasses_the_rest() {
        let cache = cache();
        let calls = Arc::new(AtomicUsize::new(0));
        let run = |calls: Arc<AtomicUsize>| {
            move |_request: ChatCompletionRequest| {
                let n = calls.fetch_add(1, Ordering::SeqCst) + 1;
                async move { json_response(json!({"id": format!("chatcmpl-{n}")})) }
            }
        };
        let request = chat(deterministic_chat());
        let first = cache
            .serve(None, &tenant("t"), request.clone(), run(calls.clone()))
            .await;
        let (_, headers, first_body) = drain(first).await;
        assert_eq!(headers.get(&HEADER_CACHE_STATUS).unwrap(), STATUS_MISS);
        let second = cache
            .serve(None, &tenant("t"), request.clone(), run(calls.clone()))
            .await;
        let (_, headers, second_body) = drain(second).await;
        assert_eq!(headers.get(&HEADER_CACHE_STATUS).unwrap(), STATUS_HIT);
        assert_eq!(first_body, second_body);
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        // Another tenant generates its own answer.
        let other = cache
            .serve(None, &tenant("u"), request.clone(), run(calls.clone()))
            .await;
        let (_, headers, _) = drain(other).await;
        assert_eq!(headers.get(&HEADER_CACHE_STATUS).unwrap(), STATUS_MISS);
        assert_eq!(calls.load(Ordering::SeqCst), 2);

        // A sampled request is generated every time and tagged bypass.
        let mut sampled = deterministic_chat();
        sampled["temperature"] = json!(0.5);
        let sampled = chat(sampled);
        for _ in 0..2 {
            let response = cache
                .serve(None, &tenant("t"), sampled.clone(), run(calls.clone()))
                .await;
            let (_, headers, _) = drain(response).await;
            assert_eq!(headers.get(&HEADER_CACHE_STATUS).unwrap(), STATUS_BYPASS);
        }
        assert_eq!(calls.load(Ordering::SeqCst), 4);

        // no-store is neither served from nor stored.
        let no_store = cache_header("no-store");
        let response = cache
            .serve(
                Some(&no_store),
                &tenant("t"),
                request.clone(),
                run(calls.clone()),
            )
            .await;
        let (_, headers, _) = drain(response).await;
        assert_eq!(headers.get(&HEADER_CACHE_STATUS).unwrap(), STATUS_BYPASS);
        assert_eq!(calls.load(Ordering::SeqCst), 5);
    }
}
