//! Gemini Interactions API router implementation
//!
//! This module implements the Gemini Interactions API using a state machine pattern.
//! The state machine drives request processing through explicit states and steps,
//! supporting both streaming and non-streaming flows with MCP tool interception.
//!
//! # Architecture
//!
//! - **State machine**: A single `execute()` driver dispatches steps based on `RequestState`.
//! - **Two-level context**: `SharedComponents` (per-router) and `RequestContext` (per-request).
//! - **Tool loop**: MCP tool calls cause state transitions back to the execution state,
//!   forming an explicit loop in the state machine rather than a nested `loop {}`.

mod context;
mod driver;
mod router;
mod state;
mod steps;

use std::sync::Arc;

use openai_protocol::worker::ProviderType;
pub use router::GeminiRouter;

use crate::{ids, BuildFuture, ExternalContext, ExternalRouter, ExternalRouterSpec};

fn serves(provider: &ProviderType) -> bool {
    matches!(provider, ProviderType::Gemini)
}

fn build(ctx: ExternalContext) -> BuildFuture {
    Box::pin(async move {
        GeminiRouter::new(&ctx).map(|router| Arc::new(router) as Arc<dyn ExternalRouter>)
    })
}

/// How the gateway mounts this router.
pub fn spec() -> ExternalRouterSpec {
    ExternalRouterSpec {
        router_id: ids::GEMINI,
        backend: "gemini",
        label: "Gemini",
        feature: "provider-gemini",
        serves,
        fallback: false,
        build,
    }
}
