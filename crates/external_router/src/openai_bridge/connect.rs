//! Atomic `orchestrator.connect_*` + `registry.populate_*` helpers.
//!
//! Use these in place of calling [`McpOrchestrator::connect_static_server`]
//! / [`McpOrchestrator::connect_dynamic_server`] directly so the
//! gateway-side [`FormatRegistry`] can never drift out of sync with the
//! orchestrator's tool inventory.
//!
//! Forgetting `populate_from_server_config` after a connect leaves
//! hosted-tool dispatch silently downgraded to `mcp_call`. The
//! `lookup_tool_format` debug log added in 283f27f1 fingerprints the bug
//! at runtime, but a wrapper makes the mistake structurally impossible.

use smg_mcp::{McpOrchestrator, McpResult, McpServerConfig};

use super::FormatRegistry;

pub async fn connect_static_server(
    orchestrator: &McpOrchestrator,
    registry: &FormatRegistry,
    config: &McpServerConfig,
) -> McpResult<()> {
    orchestrator.connect_static_server(config).await?;
    registry.populate_from_server_config(config);
    Ok(())
}

/// Returns the assigned server key.
pub async fn connect_dynamic_server(
    orchestrator: &McpOrchestrator,
    registry: &FormatRegistry,
    config: McpServerConfig,
) -> McpResult<String> {
    let server_key = orchestrator.connect_dynamic_server(config.clone()).await?;
    registry.populate_from_server_config(&config);
    Ok(server_key)
}
