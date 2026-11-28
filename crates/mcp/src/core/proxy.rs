//! HTTP proxy configuration for MCP connections.
//!
//! Resolves proxy settings and applies them to HTTP client builders.

use super::config::{McpProxyConfig, McpServerConfig};
use crate::error::{McpError, McpResult};

/// Resolve proxy configuration for a server
/// Priority: server.proxy > global.proxy > None
///
/// # Arguments
/// * `server_config` - Server-specific configuration
/// * `global_proxy` - Global proxy configuration from McpConfig
///
/// # Returns
/// The resolved proxy configuration, or None when none is configured (the
/// client then falls back to reqwest's environment proxies, if any)
pub(crate) fn resolve_proxy_config<'a>(
    server_config: &'a McpServerConfig,
    global_proxy: Option<&'a McpProxyConfig>,
) -> Option<&'a McpProxyConfig> {
    // Priority 1: the server's own proxy mapping. A missing field and an
    // explicit `null` are one state (`None`): both fall back to the global
    // proxy.
    if server_config.proxy.is_some() {
        server_config.proxy.as_ref()
    } else {
        // Priority 2: Fall back to global proxy
        global_proxy
    }
}

/// Finish a proxy the `scheme` constructor made: a construction error becomes
/// `McpError::Config` naming `scheme`, then `proxy_cfg`'s exclusions and credentials apply.
fn configured(
    made: reqwest::Result<reqwest::Proxy>,
    scheme: &str,
    proxy_cfg: &McpProxyConfig,
) -> McpResult<reqwest::Proxy> {
    let mut proxy = made.map_err(|e| McpError::Config(format!("Invalid {scheme} proxy: {e}")))?;
    if let Some(ref no_proxy) = proxy_cfg.no_proxy {
        proxy = proxy.no_proxy(reqwest::NoProxy::from_string(no_proxy));
    }
    if let (Some(ref username), Some(ref password)) = (&proxy_cfg.username, &proxy_cfg.password) {
        proxy = proxy.basic_auth(username, password);
    }
    Ok(proxy)
}

/// Apply proxy configuration to a ClientBuilder
///
/// This is a reusable helper that applies proxy settings without building the client,
/// allowing additional configuration (like auth headers) to be added afterward.
///
/// # Arguments
/// * `builder` - The reqwest::ClientBuilder to configure
/// * `proxy_config` - The proxy configuration to apply
///
/// # Returns
/// The configured builder or error
pub(super) fn apply_proxy_to_builder(
    mut builder: reqwest::ClientBuilder,
    proxy_cfg: &McpProxyConfig,
) -> McpResult<reqwest::ClientBuilder> {
    if let Some(ref http_proxy) = proxy_cfg.http {
        builder = builder.proxy(configured(
            reqwest::Proxy::http(http_proxy),
            "HTTP",
            proxy_cfg,
        )?);
    }

    if let Some(ref https_proxy) = proxy_cfg.https {
        builder = builder.proxy(configured(
            reqwest::Proxy::https(https_proxy),
            "HTTPS",
            proxy_cfg,
        )?);
    }

    Ok(builder)
}

#[cfg(test)]
#[path = "../../tests/unit/core/proxy/tests.rs"]
mod tests;
