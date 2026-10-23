//! Per-request context and hook data bridge for storage hooks.
//!
//! Uses tokio task-local storage so that per-request data (e.g. tenant ID,
//! conversation store ID from HTTP headers) can reach hooks without threading
//! it through every storage method signature.
//!
//! Also provides a task-local [`ExtraColumns`] bridge so that hooked storage
//! wrappers can pass hook-provided extra column values to backends without
//! changing any storage trait signatures.

use std::collections::HashMap;

use crate::hooks::ExtraColumns;

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

/// Per-request context passed to storage hooks.
///
/// Populated by the gateway layer (from HTTP headers, middleware output, etc.)
/// before each storage operation. Hooks access it via [`current_request_context`].
#[derive(Debug, Clone, Default)]
pub struct RequestContext {
    data: HashMap<String, String>,
}

impl RequestContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a context pre-populated with the given key-value pairs.
    pub fn with_data(data: HashMap<String, String>) -> Self {
        Self { data }
    }

    /// Get a value by key.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.data.get(key).map(String::as_str)
    }

    /// Set a key-value pair.
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.data.insert(key.into(), value.into());
    }

    /// Borrow the underlying data map.
    pub fn data(&self) -> &HashMap<String, String> {
        &self.data
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Task-local storage
// ────────────────────────────────────────────────────────────────────────────

tokio::task_local! {
    static REQUEST_CONTEXT: RequestContext;
    static HOOK_EXTRA_COLUMNS: ExtraColumns;
}

/// Run an async block with the given [`RequestContext`] available via task-local.
///
/// Called by the gateway HTTP handler before invoking storage operations.
/// The context is available inside `f` via [`current_request_context`].
pub async fn with_request_context<F, T>(ctx: RequestContext, f: F) -> T
where
    F: std::future::Future<Output = T>,
{
    REQUEST_CONTEXT.scope(ctx, f).await
}

/// Read the current request context, if one is set for this task.
///
/// Returns `None` when called outside a [`with_request_context`] scope.
pub fn current_request_context() -> Option<RequestContext> {
    REQUEST_CONTEXT.try_with(|ctx| ctx.clone()).ok()
}

/// Run an async block with the given [`ExtraColumns`] available via task-local.
///
/// Called by `HookedStorage` wrappers to make hook-provided
/// extra column values visible to the inner backend during write operations.
pub async fn with_extra_columns<F, T>(extra: ExtraColumns, f: F) -> T
where
    F: std::future::Future<Output = T>,
{
    HOOK_EXTRA_COLUMNS.scope(extra, f).await
}

/// Read the current hook extra columns, if set for this task.
///
/// Backends call this during INSERT to pick up values provided by hooks.
/// Returns `None` when called outside a [`with_extra_columns`] scope.
pub fn current_extra_columns() -> Option<ExtraColumns> {
    HOOK_EXTRA_COLUMNS.try_with(|ec| ec.clone()).ok()
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
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
}
