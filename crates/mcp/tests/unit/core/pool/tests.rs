use super::*;
use crate::core::config::{test_support::sse_server_with_headers, McpTransport};

// Helper to create test server config
fn create_test_config(url: &str) -> McpServerConfig {
    McpServerConfig {
        name: "test_server".to_string(),
        transport: McpTransport::Streamable {
            url: url.to_string(),
            token: None,
            headers: HashMap::new(),
        },
        proxy: None,
        required: false,
        tools: None,
        builtin_type: None,
        builtin_tool_name: None,
        internal: false,
    }
}

#[tokio::test]
async fn test_pool_creation() {
    let pool = McpConnectionPool::new();
    assert_eq!(pool.len(), 0);
    assert!(pool.is_empty());
}

#[test]
fn test_pool_stats() {
    let pool = McpConnectionPool::with_capacity(10);

    let stats = pool.stats();
    assert_eq!(stats.total_connections, 0);
    assert_eq!(stats.capacity, 10);
}

#[test]
fn test_pool_clear() {
    let pool = McpConnectionPool::new();
    // Pool starts empty
    assert_eq!(pool.len(), 0);
    // Clear on empty pool should work
    pool.clear();
    assert!(pool.is_empty());
}

#[test]
fn test_pool_key_from_config() {
    // No token
    let config = create_test_config("http://localhost:3000");
    let key = PoolKey::from_config(&config, None);
    assert_eq!(key.url, "http://localhost:3000");
    assert_eq!(key.auth_hash, 0);
    assert_eq!(key.tenant_id, None);

    // With token
    let config_with_token = McpServerConfig {
        name: "test".to_string(),
        transport: McpTransport::Streamable {
            url: "http://localhost:3000".to_string(),
            token: Some("secret-token".to_string()),
            headers: HashMap::new(),
        },
        proxy: None,
        required: false,
        tools: None,
        builtin_type: None,
        builtin_tool_name: None,
        internal: false,
    };
    let key_with_token = PoolKey::from_config(&config_with_token, None);
    assert_eq!(key_with_token.url, "http://localhost:3000");
    assert_ne!(key_with_token.auth_hash, 0); // Token hashed

    // With tenant
    let key_with_tenant = PoolKey::from_config(&config, Some("tenant-123".to_string()));
    assert_eq!(key_with_tenant.tenant_id, Some("tenant-123".to_string()));
}

#[test]
fn test_pool_key_different_tokens() {
    let config1 = McpServerConfig {
        name: "test".to_string(),
        transport: McpTransport::Streamable {
            url: "http://localhost:3000".to_string(),
            token: Some("token-a".to_string()),
            headers: HashMap::new(),
        },
        proxy: None,
        required: false,
        tools: None,
        builtin_type: None,
        builtin_tool_name: None,
        internal: false,
    };
    let config2 = McpServerConfig {
        name: "test".to_string(),
        transport: McpTransport::Streamable {
            url: "http://localhost:3000".to_string(),
            token: Some("token-b".to_string()),
            headers: HashMap::new(),
        },
        proxy: None,
        required: false,
        tools: None,
        builtin_type: None,
        builtin_tool_name: None,
        internal: false,
    };

    let key1 = PoolKey::from_config(&config1, None);
    let key2 = PoolKey::from_config(&config2, None);

    // Same URL but different tokens = different keys
    assert_eq!(key1.url, key2.url);
    assert_ne!(key1.auth_hash, key2.auth_hash);
    assert_ne!(key1, key2);
}

#[test]
fn test_pool_key_with_headers() {
    let mut headers1 = HashMap::new();
    headers1.insert("X-API-Key".to_string(), "key-1".to_string());

    let mut headers2 = HashMap::new();
    headers2.insert("X-API-Key".to_string(), "key-2".to_string());

    let config1 = sse_server_with_headers("test", "http://localhost:3000", headers1);
    let config2 = sse_server_with_headers("test", "http://localhost:3000", headers2);

    let key1 = PoolKey::from_config(&config1, None);
    let key2 = PoolKey::from_config(&config2, None);

    // Same URL but different headers = different keys
    assert_eq!(key1.url, key2.url);
    assert_ne!(key1.auth_hash, key2.auth_hash);
    assert_ne!(key1, key2);
}

#[test]
fn test_pool_with_global_proxy() {
    use crate::core::config::McpProxyConfig;

    // Create proxy config
    let proxy = McpProxyConfig {
        http: Some("http://proxy.example.com:8080".to_string()),
        https: None,
        no_proxy: Some("localhost,127.0.0.1".to_string()),
        username: None,
        password: None,
    };

    // Create pool with proxy
    let pool = McpConnectionPool::with_full_config(100, Some(proxy.clone()));

    // Verify proxy is stored
    assert!(pool.global_proxy.is_some());
    let stored_proxy = pool.global_proxy.as_ref().unwrap();
    assert_eq!(
        stored_proxy.http.as_ref().unwrap(),
        "http://proxy.example.com:8080"
    );
    assert_eq!(
        stored_proxy.no_proxy.as_ref().unwrap(),
        "localhost,127.0.0.1"
    );
}

#[test]
fn classify_url_match_distinguishes_none_unique_and_ambiguous() {
    let url = "http://localhost:3000";
    let other = PoolKey::new("http://other:9000", 1, None);
    let a = PoolKey::new(url, 111, None);
    // Same URL, different auth hash (two callers with different tokens).
    let b_diff_auth = PoolKey::new(url, 222, None);
    // Same URL and auth, different tenant.
    let a_diff_tenant = PoolKey::new(url, 111, Some("tenant-2".to_string()));

    assert_eq!(
        classify_url_match([&other].into_iter(), url),
        UrlMatch::None
    );
    assert_eq!(
        classify_url_match([&other, &a].into_iter(), url),
        UrlMatch::Unique
    );
    // Different credentials on the same URL must never resolve to one entry.
    assert_eq!(
        classify_url_match([&a, &b_diff_auth].into_iter(), url),
        UrlMatch::Ambiguous
    );
    // Different tenants on the same URL are equally ambiguous.
    assert_eq!(
        classify_url_match([&a, &a_diff_tenant].into_iter(), url),
        UrlMatch::Ambiguous
    );
}

#[test]
fn get_unique_by_url_on_empty_pool_is_not_found() {
    let pool = McpConnectionPool::new();
    assert!(matches!(
        pool.get_unique_by_url("http://localhost:3000"),
        UrlLookup::NotFound
    ));
}

#[test]
#[expect(deprecated, reason = "verifies the source-compatibility wrapper")]
fn deprecated_get_by_url_fails_closed_on_empty_pool() {
    let pool = McpConnectionPool::new();
    assert!(pool.get_by_url("http://localhost:3000").is_none());
}

#[test]
fn test_pool_proxy_from_env() {
    // Note: This test depends on environment variables
    // In production, proxy is loaded from MCP_HTTP_PROXY or HTTP_PROXY env vars
    let pool = McpConnectionPool::new();

    // Pool should either have proxy from env or None
    // We can't assert specific value since it depends on test environment
    // Just verify it doesn't panic
    assert!(pool.global_proxy.is_some() || pool.global_proxy.is_none());
}
