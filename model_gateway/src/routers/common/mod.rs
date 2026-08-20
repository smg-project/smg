//! Cross-router shared utilities.
//!
//! This module collects helpers that every router (HTTP, gRPC,
//! OpenAI, Anthropic, Gemini, etc.) needs but no individual router
//! owns. Putting them here keeps `routers/mod.rs` focused on the
//! `RouterTrait` definition and the per-protocol submodules.
//!
//! Submodules:
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

pub mod header_utils;
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
