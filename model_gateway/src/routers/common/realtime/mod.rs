//! Realtime API transport, shared across routers (OpenAI, HTTP).
//!
//! Supports three transport mechanisms:
//! - **WebSocket** (server-to-server): Bidirectional WS proxy with transparent MCP interception
//! - **WebRTC** (browser-to-server): Dual peer-connection relay; SMG terminates both sides
//! - **REST**: Ephemeral token generation (`client_secrets`, `sessions`, `transcription_sessions`)

pub mod proxy;
pub mod registry;
pub mod rest;
pub mod webrtc;
pub mod webrtc_bridge;
pub mod ws;

pub use registry::RealtimeRegistry;

use crate::observability::metrics::metrics_labels;

/// Metrics labels identifying which router drives a realtime relay.
///
/// The shared transport helpers are provider-agnostic; each router passes
/// its own labels so metrics attribute realtime traffic correctly — the
/// HTTP router's local workers are `regular`, not `external`.
#[derive(Clone, Copy)]
pub(crate) struct RealtimeLabels {
    /// Router label (e.g. `openai`, `http`).
    pub router: &'static str,
    /// Backend label (e.g. `external`, `regular`).
    pub backend: &'static str,
}

impl RealtimeLabels {
    /// Labels for the OpenAI router relaying to an external provider.
    #[cfg(feature = "provider-openai")]
    pub const OPENAI: Self = Self {
        router: metrics_labels::ROUTER_OPENAI,
        backend: metrics_labels::BACKEND_EXTERNAL,
    };

    /// Labels for the HTTP router relaying to a local self-hosted worker.
    pub const HTTP: Self = Self {
        router: metrics_labels::ROUTER_HTTP,
        backend: metrics_labels::BACKEND_REGULAR,
    };
}
