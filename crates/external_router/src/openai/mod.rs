//! OpenAI-compatible router implementation
//!
//! This module provides OpenAI-compatible API routing with support for:
//! - Streaming and non-streaming responses
//! - MCP (Model Context Protocol) tool calling
//! - Response storage and conversation management
//! - Multi-turn tool execution loops
//! - SSE (Server-Sent Events) streaming

mod chat;
mod context;
mod health;
pub(crate) mod mcp;
mod provider;
pub mod responses;
mod router;

use std::sync::Arc;

use openai_protocol::worker::ProviderType;
pub use router::OpenAIRouter;

use crate::{ids, BuildFuture, ExternalContext, ExternalRouter, ExternalRouterSpec};

fn serves(provider: &ProviderType) -> bool {
    matches!(
        provider,
        ProviderType::OpenAI | ProviderType::XAI | ProviderType::Custom(_)
    )
}

fn build(ctx: ExternalContext) -> BuildFuture {
    Box::pin(async move {
        OpenAIRouter::new(&ctx)
            .await
            .map(|router| Arc::new(router) as Arc<dyn ExternalRouter>)
    })
}

/// How the gateway mounts this router.
pub fn spec() -> ExternalRouterSpec {
    ExternalRouterSpec {
        router_id: ids::OPENAI,
        backend: "openai",
        label: "OpenAI",
        feature: "provider-openai",
        serves,
        fallback: true,
        build,
    }
}
