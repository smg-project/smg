//! Context types for the Gemini Interactions router.
//!
//! Two-level context design:
//! - `SharedComponents`: created once per router, `Arc`-cloned for each request.
//! - `RequestContext`: created fresh per request, owned and mutated by steps.

use std::{sync::Arc, time::Duration};

use axum::http::HeaderMap;
use openai_protocol::interactions::InteractionsRequest;
use serde_json::Value;
use smg_mcp::McpOrchestrator;

use super::state::RequestState;
use crate::{
    openai_bridge::FormatRegistry,
    tenant::TenantRequestMeta,
    worker::{ExternalWorker, WorkerSource},
};

// ============================================================================
// SharedComponents (per-router)
// ============================================================================

/// Immutable state shared across all requests.
///
/// Created once during `GeminiRouter::new()` and cheaply `Arc`-cloned
/// into every `RequestContext`.
///
/// TODO: Create InteractionsStorage and add in context
pub(crate) struct SharedComponents {
    /// HTTP client for upstream requests.
    pub client: reqwest::Client,

    /// Worker registry for model → worker resolution.
    pub workers: Arc<dyn WorkerSource>,

    /// MCP orchestrator for creating tool sessions (used in Phase 2).
    #[expect(dead_code, reason = "MCP tool integration is Phase 2")]
    pub mcp_orchestrator: Arc<McpOrchestrator>,

    #[expect(dead_code, reason = "MCP tool integration is Phase 2")]
    pub mcp_format_registry: FormatRegistry,

    /// Per-request timeout from router config.
    pub request_timeout: Duration,
}

// ============================================================================
// RequestContext (per-request)
// ============================================================================

/// Per-request mutable state passed through the state machine.
///
/// Steps read and write fields on this struct. The `state` field
/// determines which step the driver executes next.
///
/// Streaming-specific state (SSE channel, event counters) is **not** stored
/// here — it lives inside the background task spawned by the terminal
/// streaming step, following the same pattern as `process_streaming_response`
/// in the gRPC router.
pub(crate) struct RequestContext {
    /// Immutable request data from the client.
    pub input: RequestInput,

    /// Reference to the per-router shared components.
    pub components: Arc<SharedComponents>,

    /// Current position in the state machine.
    pub state: RequestState,

    /// Mutable processing state populated incrementally by steps.
    pub processing: ProcessingState,
}

/// Immutable request data captured at the start of processing.
pub(crate) struct RequestInput {
    /// Original client request (`Arc` for cheap cloning into spawned tasks).
    pub original_request: Arc<InteractionsRequest>,

    /// HTTP headers forwarded from the client.
    pub headers: Option<HeaderMap>,

    /// Optional model ID override (e.g. from URL path or query parameter).
    /// When set, takes precedence over `original_request.model`.
    pub model_id: Option<String>,
    #[expect(
        dead_code,
        reason = "tenant-aware Gemini consumers land after the shared routing contract"
    )]
    pub tenant_request_meta: TenantRequestMeta,
}

/// Mutable processing state populated incrementally by steps.
#[derive(Default)]
pub(crate) struct ProcessingState {
    /// Selected upstream worker (set by `WorkerSelection`).
    pub worker: Option<Arc<dyn ExternalWorker>>,

    /// Upstream URL: `{worker.url()}/v1beta/interactions` (set by `WorkerSelection`).
    pub upstream_url: Option<String>,

    /// JSON payload to POST upstream (set by `RequestBuilding`, mutated on tool-loop resume).
    pub payload: Option<Value>,

    /// Latest upstream response JSON (set by execution steps).
    pub upstream_response: Option<Value>,
}

impl RequestContext {
    /// Create a new `RequestContext` in the `SelectWorker` state.
    pub fn new(
        original_request: Arc<InteractionsRequest>,
        headers: Option<HeaderMap>,
        model_id: Option<String>,
        tenant_request_meta: TenantRequestMeta,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self {
            input: RequestInput {
                original_request,
                headers,
                model_id,
                tenant_request_meta,
            },
            components,
            state: RequestState::SelectWorker,
            processing: ProcessingState::default(),
        }
    }
}
