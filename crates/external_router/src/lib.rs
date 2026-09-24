//! Routers for third-party providers, and the contract they are built on.
//!
//! The contract is [`ExternalRouter`] (what a router implements) and
//! [`ExternalContext`] (what the gateway hands it). Around them sits the
//! protocol-level glue every router family shares: the error envelope, SSE
//! parsing, the MCP-to-OpenAI bridge, request-scoped MCP helpers, provider
//! header handling, retries, persistence, the realtime proxy and the router
//! metrics. The gateway re-exports the glue at its old paths, so its own
//! routers do not know the difference. The built-in routers sit behind one
//! Cargo feature each; a self-hosted build takes none of them.

#[cfg(feature = "anthropic")]
pub mod anthropic;
pub mod context;
pub mod error;
#[cfg(feature = "gemini")]
pub mod gemini;
pub mod header_utils;
pub mod mcp_utils;
pub mod metrics;
#[cfg(feature = "openai")]
pub mod openai;
pub mod openai_bridge;
pub mod persistence_utils;
pub mod realtime;
pub mod retry;
pub mod retry_config;
pub mod router;
pub mod sglang_fields;
pub mod sse;
pub mod tenant;
pub mod worker;

pub use context::ExternalContext;
pub use retry_config::RetryConfig;
pub use router::{
    builtin_routers, home_among, ids, known, spec_for_backend, spec_for_provider, BuildFuture,
    ExternalRouter, ExternalRouterSpec,
};
