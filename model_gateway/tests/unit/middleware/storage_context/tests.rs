use axum::http::{HeaderMap, HeaderValue};

use super::*;

#[test]
fn build_storage_request_context_maps_configured_headers() {
    let config = RouterConfig {
        storage_context_headers: std::collections::HashMap::from([
            ("x-tenant-id".to_string(), "tenant_id".to_string()),
            ("x-user-id".to_string(), "user_id".to_string()),
        ]),
        ..Default::default()
    };

    let mut headers = HeaderMap::new();
    headers.insert("x-tenant-id", HeaderValue::from_static("tenant-abc"));
    headers.insert("x-user-id", HeaderValue::from_static("user-123"));

    let ctx = build_storage_request_context(&config, &headers).unwrap();

    assert_eq!(ctx.get("tenant_id"), Some("tenant-abc"));
    assert_eq!(ctx.get("user_id"), Some("user-123"));
}

#[test]
fn build_storage_request_context_ignores_empty_entries_and_missing_headers() {
    let config = RouterConfig {
        storage_context_headers: std::collections::HashMap::from([
            (" ".to_string(), "tenant_id".to_string()),
            ("x-empty-key".to_string(), " ".to_string()),
            ("x-present".to_string(), "present_key".to_string()),
        ]),
        ..Default::default()
    };

    let mut headers = HeaderMap::new();
    headers.insert("x-present", HeaderValue::from_static("  keep-me  "));

    let ctx = build_storage_request_context(&config, &headers).unwrap();

    assert_eq!(ctx.get("present_key"), Some("keep-me"));
    assert_eq!(ctx.data().len(), 1);
}
