//! Cross-router shared utilities.
//!
//! This module collects helpers that every router (HTTP, gRPC,
//! OpenAI, Anthropic, Gemini, etc.) needs but no individual router
//! owns. Putting them here keeps `routers/mod.rs` focused on the
//! `RouterTrait` definition and the per-protocol submodules.
//!
//! Submodules:
//! - [`body_policy`] — per-family [`body_policy::BodyPolicy`] capability,
//!   the per-request buffer-vs-stream decision matrix and its counter
//!   vocabulary
//! - [`header_utils`] — request header parsing helpers
//!   (`extract_routing_key`, `extract_target_worker`, etc.)
//! - [`mcp_utils`] — Model Context Protocol tool-call orchestration
//! - [`persistence_utils`] — response/conversation persistence
//!   helpers shared across the chat / responses / messages routes
//! - [`realtime`] — Realtime API transport (WS/WebRTC/REST relay +
//!   session registry) shared by the OpenAI and HTTP routers
//! - [`overload`] — shed responses for the absolute worker-overload
//!   guard, shared by the HTTP and gRPC selection paths
//! - [`pd_admission`] — the disaggregated dispatch's admission gate: never
//!   post more bootstrap rooms to a pair than its decode engine can admit
//! - `pd_failure` — the one vocabulary a prefill or decode side's failure is
//!   named by (the side, the phase, the failure's own status), the HTTP and
//!   gRPC prefill-decode paths' answers built from it
//! - `placement` — policy-driven worker placement over the routing
//!   pools (single worker and prefill/decode pairs), the one sequence the
//!   HTTP and gRPC families used to each carry a copy of
//! - [`worker_selection`] — per-request worker-selection helpers used
//!   by every routing path (regular, PD, fallback, external provider)
//! - `test_support` (test builds only) — the fast retry policy the HTTP
//!   and gRPC routers' tests retry under
//! - `attempt_ledger` — one request's bookkeeping across its retry
//!   attempts: the circuit breaker charged once per request per worker,
//!   and a worker's definitive answer never replayed on it
//! - `rate_limit_cell` — one logical request's tenant rate-limit
//!   reservation: reserved once before any dispatch, then settled, handed
//!   to a streaming body, or closed unsettled
//! - [`request_lease`] — dispatch-phase owner of a request's parsed
//!   body, routing derivatives and serialized upstream bytes, with a
//!   retry-aware release point
//! - [`retry`] — generic async retry executor + backoff calculator,
//!   used by every router for transport-level retries. Has zero
//!   coupling to the `Worker` trait — it lived in `worker/` for
//!   historical reasons before this extraction.
//! - [`sglang_fields`] — the engine-specific request fields the HTTP proxy
//!   strips at their defaults and the provider transformers drop outright
//! - [`sse`] — shared SSE codec (encoder + decoder) for streaming
//!   responses to clients and parsing upstream SSE byte streams

pub(crate) mod attempt_ledger;
pub mod body_policy;
pub(crate) mod decisions;
pub use smg_external_router::header_utils;
pub(crate) mod kv_transfer;
pub use smg_external_router::{mcp_utils, openai_bridge};
pub mod overload;
pub mod pd_admission;
pub(crate) mod pd_failure;
pub use smg_external_router::persistence_utils;
pub(crate) mod placement;
pub(crate) mod rate_limit_cell;
pub use smg_external_router::realtime;
pub mod request_lease;
pub(crate) use smg_external_router::sglang_fields;
pub use smg_external_router::{retry, sse};
pub mod sse_rechunk;
pub mod worker_selection;

/// Threshold above which upstream request bodies are sent as one-shot
/// streams. A streamed body's `try_clone()` is `None` in every layer that
/// would otherwise hold a refcount until response headers (the stale-conn
/// resend guard, reqwest's internal retry and redirect clones), so the
/// allocation frees at upload completion instead of living for the whole
/// generation on buffered upstreams. The explicit Content-Length keeps h1
/// framing non-chunked; h2 is unaffected.
pub(crate) const STREAM_UPSTREAM_BODY_OVER: usize = 1 << 20;

/// Buffer capacity for reserializing a parsed request of `raw_len` incoming
/// bytes: the round trip stays close to raw size, and the 1/16 + 512B slack
/// absorbs injected fields (bootstrap, kv_transfer_params, dp ranks).
fn serialized_capacity(raw_len: usize) -> usize {
    raw_len + raw_len / 16 + 512
}

/// `serde_json::to_vec` into a buffer pre-sized from the raw request length,
/// avoiding doubling growth (and its final huge memcpy) for large bodies.
pub(crate) fn serialize_json_sized<T: serde::Serialize>(
    value: &T,
    raw_len: Option<usize>,
) -> serde_json::Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(raw_len.map_or(128, serialized_capacity));
    serde_json::to_writer(&mut buf, value)?;
    Ok(buf)
}

/// The capacity a thread's JSON answer buffer keeps between answers: a chat
/// completion, a Messages answer or a JSON-schema answer of a few hundred
/// bytes fits it many times over; the buffer a larger answer grew goes with
/// that answer.
const JSON_ANSWER_RETAINED: usize = 64 * 1024;

thread_local! {
    /// The buffer a thread's typed JSON answers are serialized into.
    static JSON_ANSWER_BUFFER: std::cell::RefCell<Vec<u8>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// A typed answer as a JSON response: serialized once into the thread's
/// buffer (serde's own sink, one `extend_from_slice` per write) and copied
/// once into a body of exactly its size, with the `application/json`
/// content type. That is one allocation per answer: a buffer sized per
/// answer and handed to the body by refcount cost two, the buffer and the
/// shared header `Bytes::from` boxes for a buffer with spare capacity, and
/// grew again for a body past its size. `axum::Json` writes the same bytes
/// through `BytesMut::with_capacity(128).writer()`: the `bytes` writer
/// adapter slices every write to the buffer's remaining room before it
/// appends, which prices the one-byte JSON tokens (2.5 % of a chat
/// request's user instructions on the mock bed), and the 128-byte start
/// reallocates and copies a 700-byte body three times. A value that does
/// not serialize answers as axum does: a 500 with the error's text.
pub fn json_response<T: serde::Serialize>(value: &T) -> axum::response::Response {
    JSON_ANSWER_BUFFER.with(|cell| match cell.try_borrow_mut() {
        Ok(mut buffer) => {
            buffer.clear();
            let result = serde_json::to_writer(&mut *buffer, value);
            let response = json_answer(&buffer, result);
            if buffer.capacity() > JSON_ANSWER_RETAINED {
                *buffer = Vec::new();
            }
            response
        }
        // A serializer that answers while this thread's buffer is borrowed
        // (none does today): a buffer of this answer's own.
        Err(_) => {
            let mut buffer = Vec::new();
            let result = serde_json::to_writer(&mut buffer, value);
            json_answer(&buffer, result)
        }
    })
}

/// The response of a serialized answer, compiled once for every `T`.
fn json_answer(body: &[u8], result: serde_json::Result<()>) -> axum::response::Response {
    use axum::{
        http::{header, HeaderValue, StatusCode},
        response::IntoResponse,
    };
    match result {
        Ok(()) => (
            [(
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/json"),
            )],
            bytes::Bytes::copy_from_slice(body),
        )
            .into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            [(
                header::CONTENT_TYPE,
                HeaderValue::from_static("text/plain; charset=utf-8"),
            )],
            error.to_string(),
        )
            .into_response(),
    }
}

/// A typed request as a `Value`, for the forwarding paths that edit it:
/// `openai_protocol::common::to_value_exact` with the buffer pre-sized from
/// the raw request length. Every typed `f32` goes out as the client wrote
/// it (`serde_json::to_value` would widen `0.95` to `0.949999988079071`).
pub(crate) fn request_to_value<T: serde::Serialize>(
    value: &T,
    raw_len: Option<usize>,
) -> serde_json::Result<serde_json::Value> {
    serde_json::from_slice(&serialize_json_sized(value, raw_len)?)
}

/// `Bytes::from(Vec)` keeps the Vec's capacity, so unshrunk doubling growth
/// (up to 2x len) would be retained for the buffer's whole lifetime; trade
/// one bounded memcpy to free it. The threshold is double the intended
/// pre-size slack so well-hinted buffers are never re-copied.
pub(crate) fn trim_serialization_slack(buf: &mut Vec<u8>) {
    if buf.capacity() > buf.len() + buf.len() / 8 + 1024 {
        buf.shrink_to_fit();
    }
}

pub(crate) fn attach_sized_body(
    builder: reqwest::RequestBuilder,
    body: bytes::Bytes,
) -> reqwest::RequestBuilder {
    let len = body.len();
    let builder = builder.header(reqwest::header::CONTENT_LENGTH, len);
    if len >= STREAM_UPSTREAM_BODY_OVER {
        builder.body(reqwest::Body::wrap_stream(futures::stream::once(
            async move { Ok::<_, std::convert::Infallible>(body) },
        )))
    } else {
        builder.body(body)
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    //! What the HTTP and gRPC routers' tests share.

    use crate::config::types::RetryConfig;

    /// `max_retries` attempts with one to two milliseconds between them and
    /// no jitter: a retrying test's whole schedule fits in its own bound.
    pub(crate) fn fast_retry_config(max_retries: u32) -> RetryConfig {
        RetryConfig {
            max_retries,
            initial_backoff_ms: 1,
            max_backoff_ms: 2,
            backoff_multiplier: 1.0,
            jitter_factor: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sized_serialization_never_doubles_for_large_bodies() {
        let raw = serde_json::to_vec(&serde_json::json!({
            "text": "x".repeat(8 << 20),
            "stream": false,
        }))
        .unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&raw).unwrap();

        let body = serialize_json_sized(&parsed, Some(raw.len())).unwrap();

        assert_eq!(body.len(), raw.len());
        assert_eq!(body.capacity(), serialized_capacity(raw.len()));
        assert!(body.capacity() <= body.len() + body.len() / 8 + 1024);
    }

    #[test]
    fn a_json_answer_is_the_serializer_bytes_under_the_json_content_type() {
        use axum::http::{header, StatusCode};
        let value = serde_json::json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "x".repeat(600)}}],
        });
        let response = json_response(&value);
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE).unwrap(),
            "application/json"
        );
        let body =
            futures::executor::block_on(axum::body::to_bytes(response.into_body(), usize::MAX))
                .unwrap();
        assert_eq!(body, serde_json::to_vec(&value).unwrap());
    }

    #[test]
    fn a_json_answer_that_does_not_serialize_is_a_plain_text_500() {
        use axum::http::{header, StatusCode};
        // A map keyed by a tuple has no JSON key: the serializer refuses it.
        let value: std::collections::BTreeMap<(u8, u8), u8> = [((1, 2), 3)].into_iter().collect();
        let response = json_response(&value);
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE).unwrap(),
            "text/plain; charset=utf-8"
        );
    }

    #[test]
    fn trim_frees_doubling_slack_but_keeps_presized_buffers() {
        // Mimic a 5MiB body landing in an 8MiB doubling-growth buffer.
        let mut grown = Vec::with_capacity(8 << 20);
        grown.resize(5 << 20, b'x');
        trim_serialization_slack(&mut grown);
        assert!(grown.capacity() <= grown.len() + grown.len() / 8 + 1024);
        assert_eq!(grown.len(), 5 << 20);

        let mut sized = Vec::with_capacity(serialized_capacity(1 << 20));
        sized.resize(1 << 20, b'y');
        let capacity = sized.capacity();
        trim_serialization_slack(&mut sized);
        assert_eq!(sized.capacity(), capacity);
    }
}
