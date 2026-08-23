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
//! - [`worker_selection`] — per-request worker-selection helpers used
//!   by every routing path (regular, PD, fallback, external provider)
//! - [`request_lease`] — dispatch-phase owner of a request's parsed
//!   body, routing derivatives and serialized upstream bytes, with a
//!   retry-aware release point
//! - [`retry`] — generic async retry executor + backoff calculator,
//!   used by every router for transport-level retries. Has zero
//!   coupling to the `Worker` trait — it lived in `worker/` for
//!   historical reasons before this extraction.
//! - [`sse`] — shared SSE codec (encoder + decoder) for streaming
//!   responses to clients and parsing upstream SSE byte streams

pub mod body_policy;
pub mod header_utils;
pub(crate) mod kv_transfer;
pub mod mcp_utils;
pub mod openai_bridge;
pub mod overload;
pub mod persistence_utils;
pub mod realtime;
pub mod request_lease;
pub mod retry;
pub mod sse;
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
/// absorbs injected fields (bootstrap, kv_transfer_params, dp ranks) and
/// widened floats.
pub(crate) fn serialized_capacity(raw_len: usize) -> usize {
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
