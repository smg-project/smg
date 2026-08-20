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
//! - [`worker_selection`] — per-request worker-selection helpers used
//!   by every routing path (regular, PD, fallback, external provider)
//! - [`retry`] — generic async retry executor + backoff calculator,
//!   used by every router for transport-level retries. Has zero
//!   coupling to the `Worker` trait — it lived in `worker/` for
//!   historical reasons before this extraction.
//! - [`sse`] — shared SSE codec (encoder + decoder) for streaming
//!   responses to clients and parsing upstream SSE byte streams

pub mod header_utils;
pub mod mcp_utils;
pub mod openai_bridge;
pub mod persistence_utils;
pub mod realtime;
pub mod retry;
pub(crate) mod routing_tokens;
pub mod sse;
pub mod worker_selection;
