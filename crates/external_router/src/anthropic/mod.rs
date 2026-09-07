pub(crate) mod context;
pub(crate) mod mcp;
pub(crate) mod non_streaming;
mod router;
pub(crate) mod sse;
pub(crate) mod streaming;
pub(crate) mod utils;
pub(crate) mod worker;

use std::sync::Arc;

use openai_protocol::worker::ProviderType;
pub use router::AnthropicRouter;

use crate::{ids, BuildFuture, ExternalContext, ExternalRouter, ExternalRouterSpec};

fn serves(provider: &ProviderType) -> bool {
    matches!(provider, ProviderType::Anthropic)
}

fn build(ctx: ExternalContext) -> BuildFuture {
    Box::pin(async move {
        AnthropicRouter::new(&ctx).map(|router| Arc::new(router) as Arc<dyn ExternalRouter>)
    })
}

/// How the gateway mounts this router.
pub fn spec() -> ExternalRouterSpec {
    ExternalRouterSpec {
        router_id: ids::ANTHROPIC,
        backend: "anthropic",
        label: "Anthropic",
        feature: "provider-anthropic",
        serves,
        fallback: false,
        build,
    }
}
