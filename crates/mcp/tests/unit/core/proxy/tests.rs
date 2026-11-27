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
