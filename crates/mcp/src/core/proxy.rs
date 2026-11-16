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
mod tests {
    use super::*;
    use crate::core::{config::test_support::sse_server, orchestrator::http_client_builder};

    #[test]
    fn test_resolve_proxy_no_config() {
        let server = sse_server("test", "http://localhost:3000/sse");

        let result = resolve_proxy_config(&server, None);
        assert!(
            result.is_none(),
            "Should return None when no proxy configured"
        );
    }

    #[test]
    fn test_resolve_proxy_global_only() {
        let server = sse_server("test", "http://localhost:3000/sse");

        let global = McpProxyConfig {
            http: Some("http://global-proxy:8080".to_string()),
            https: None,
            no_proxy: None,
            username: None,
            password: None,
        };

        let result = resolve_proxy_config(&server, Some(&global));
        assert!(result.is_some(), "Should use global proxy");
        assert_eq!(
            result.unwrap().http.as_ref().unwrap(),
            "http://global-proxy:8080"
        );
    }

    #[test]
    fn test_resolve_proxy_server_override() {
        let server_proxy = McpProxyConfig {
            http: Some("http://server-proxy:9090".to_string()),
            https: None,
            no_proxy: None,
            username: None,
            password: None,
        };

        let server = McpServerConfig {
            proxy: Some(server_proxy),
            ..sse_server("test", "http://localhost:3000/sse")
        };

        let global = McpProxyConfig {
            http: Some("http://global-proxy:8080".to_string()),
            https: None,
            no_proxy: None,
            username: None,
            password: None,
        };

        let result = resolve_proxy_config(&server, Some(&global));
        assert!(result.is_some(), "Should use server-specific proxy");
        assert_eq!(
            result.unwrap().http.as_ref().unwrap(),
            "http://server-proxy:9090",
            "Server proxy should override global"
        );
    }

    #[test]
    fn test_apply_proxy_to_builder_with_http() {
        let proxy = McpProxyConfig {
            http: Some("http://proxy.example.com:8080".to_string()),
            https: None,
            no_proxy: Some("localhost,127.0.0.1".to_string()),
            username: None,
            password: None,
        };

        let builder = http_client_builder();
        let result = apply_proxy_to_builder(builder, &proxy);
        assert!(result.is_ok(), "Should apply proxy to builder");
        assert!(result.unwrap().build().is_ok(), "Should build client");
    }

    #[test]
    fn test_apply_proxy_to_builder_with_auth() {
        let proxy = McpProxyConfig {
            http: Some("http://proxy.example.com:8080".to_string()),
            https: None,
            no_proxy: None,
            username: Some("user".to_string()),
            password: Some("pass".to_string()),
        };

        let builder = http_client_builder();
        let result = apply_proxy_to_builder(builder, &proxy);
        assert!(
            result.is_ok(),
            "Should apply proxy with authentication to builder"
        );
        assert!(result.unwrap().build().is_ok(), "Should build client");
    }

    #[test]
    fn test_apply_proxy_to_builder_invalid_url() {
        let proxy = McpProxyConfig {
            http: Some("://invalid".to_string()), // Invalid URL format
            https: None,
            no_proxy: None,
            username: None,
            password: None,
        };

        let builder = http_client_builder();
        let result = apply_proxy_to_builder(builder, &proxy);
        assert!(result.is_err(), "Should fail with invalid proxy URL");
    }

    #[test]
    fn test_apply_proxy_to_builder_with_http_and_https() {
        let proxy = McpProxyConfig {
            http: Some("http://proxy.example.com:8080".to_string()),
            https: Some("http://proxy.example.com:8443".to_string()),
            no_proxy: Some("localhost,127.0.0.1".to_string()),
            username: Some("user".to_string()),
            password: Some("pass".to_string()),
        };

        let builder = http_client_builder();
        let result = apply_proxy_to_builder(builder, &proxy);
        assert!(result.is_ok(), "Should apply both proxies to builder");
        assert!(result.unwrap().build().is_ok(), "Should build client");
    }

    #[test]
    fn test_apply_proxy_to_builder_invalid_url_names_the_scheme() {
        for (http, https, named) in [
            (Some("://invalid"), None, "Invalid HTTP proxy"),
            (None, Some("://invalid"), "Invalid HTTPS proxy"),
        ] {
            let proxy = McpProxyConfig {
                http: http.map(str::to_string),
                https: https.map(str::to_string),
                no_proxy: None,
                username: None,
                password: None,
            };
            let Err(err) = apply_proxy_to_builder(http_client_builder(), &proxy) else {
                panic!("an invalid proxy URL must be refused");
            };
            assert!(
                matches!(&err, McpError::Config(m) if m.starts_with(named)),
                "{named}: {err:?}"
            );
        }
    }
}
