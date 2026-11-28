use super::*;

#[test]
fn request_context_new_is_empty() {
    let ctx = RequestContext::new();
    assert!(ctx.data().is_empty());
    assert!(ctx.get("anything").is_none());
}

#[test]
fn request_context_set_and_get() {
    let mut ctx = RequestContext::new();
    ctx.set("tenant_id", "abc");
    assert_eq!(ctx.get("tenant_id"), Some("abc"));
    assert!(ctx.get("missing").is_none());
}

#[test]
fn request_context_with_data() {
    let mut data = HashMap::new();
    data.insert("key".to_string(), "value".to_string());
    let ctx = RequestContext::with_data(data);
    assert_eq!(ctx.get("key"), Some("value"));
}

#[tokio::test]
async fn current_request_context_returns_none_outside_scope() {
    assert!(current_request_context().is_none());
}

#[tokio::test]
async fn with_request_context_makes_context_available() {
    let mut ctx = RequestContext::new();
    ctx.set("store_id", "123");

    let result = with_request_context(ctx, async {
        let inner = current_request_context().expect("should be set");
        inner.get("store_id").unwrap().to_string()
    })
    .await;

    assert_eq!(result, "123");
}

#[tokio::test]
async fn context_not_available_after_scope_exits() {
    let ctx = RequestContext::new();
    with_request_context(ctx, async {}).await;
    assert!(current_request_context().is_none());
}

// ── ExtraColumns task-local ──────────────────────────────────────────

#[tokio::test]
async fn extra_columns_returns_none_outside_scope() {
    assert!(current_extra_columns().is_none());
}

#[tokio::test]
async fn extra_columns_available_inside_scope() {
    let mut extra = ExtraColumns::new();
    extra.insert(
        "tenant_id".to_string(),
        serde_json::Value::String("t-123".to_string()),
    );

    let result = with_extra_columns(extra, async {
        let ec = current_extra_columns().expect("should be set");
        ec.get("tenant_id").unwrap().as_str().unwrap().to_string()
    })
    .await;

    assert_eq!(result, "t-123");
}

#[tokio::test]
async fn extra_columns_not_leaked_after_scope() {
    let extra = ExtraColumns::new();
    with_extra_columns(extra, async {}).await;
    assert!(current_extra_columns().is_none());
}
