use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::json;

use super::*;
use crate::{
    context::{current_extra_columns, with_request_context, RequestContext},
    core::{SortOrder, StorageFailure},
    hooks::HookError,
    memory::{MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage},
};

// ── Capturing conversation storage ───────────────────────────────────

/// A memory conversation store that records the task-local `ExtraColumns`
/// in force while a conversation is created: what a hooked wrapper handed
/// its backend.
struct CapturingConversationStorage {
    inner: MemoryConversationStorage,
    captured: parking_lot::Mutex<Option<ExtraColumns>>,
}

impl CapturingConversationStorage {
    fn new() -> Self {
        Self {
            inner: MemoryConversationStorage::new(),
            captured: parking_lot::Mutex::new(None),
        }
    }
}

#[async_trait]
impl ConversationStorage for CapturingConversationStorage {
    async fn create_conversation(
        &self,
        input: NewConversation,
    ) -> ConversationResult<Conversation> {
        *self.captured.lock() = current_extra_columns();
        self.inner.create_conversation(input).await
    }

    async fn get_conversation(
        &self,
        id: &ConversationId,
    ) -> ConversationResult<Option<Conversation>> {
        self.inner.get_conversation(id).await
    }

    async fn update_conversation(
        &self,
        id: &ConversationId,
        metadata: Option<ConversationMetadata>,
    ) -> ConversationResult<Option<Conversation>> {
        self.inner.update_conversation(id, metadata).await
    }

    async fn delete_conversation(&self, id: &ConversationId) -> ConversationResult<bool> {
        self.inner.delete_conversation(id).await
    }
}

// ── Mock hook ────────────────────────────────────────────────────────

/// A test hook that counts invocations and can be configured to reject.
struct MockHook {
    before_count: AtomicUsize,
    after_count: AtomicUsize,
    after_results: parking_lot::Mutex<Vec<serde_json::Value>>,
    reject_before: parking_lot::Mutex<Option<String>>,
}

impl MockHook {
    fn new() -> Self {
        Self {
            before_count: AtomicUsize::new(0),
            after_count: AtomicUsize::new(0),
            after_results: parking_lot::Mutex::new(Vec::new()),
            reject_before: parking_lot::Mutex::new(None),
        }
    }

    /// The `result` of every after-hook call, in order.
    fn after_results(&self) -> Vec<serde_json::Value> {
        self.after_results.lock().clone()
    }

    fn set_reject(&self, msg: &str) {
        *self.reject_before.lock() = Some(msg.to_string());
    }

    fn before_calls(&self) -> usize {
        self.before_count.load(Ordering::SeqCst)
    }

    fn after_calls(&self) -> usize {
        self.after_count.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl StorageHook for MockHook {
    async fn before(
        &self,
        _operation: StorageOperation,
        _context: Option<&RequestContext>,
        _payload: &serde_json::Value,
    ) -> Result<BeforeHookResult, HookError> {
        self.before_count.fetch_add(1, Ordering::SeqCst);
        if let Some(msg) = self.reject_before.lock().as_ref() {
            return Ok(BeforeHookResult::Reject(msg.clone()));
        }
        Ok(BeforeHookResult::Continue(ExtraColumns::new()))
    }

    async fn after(
        &self,
        _operation: StorageOperation,
        _context: Option<&RequestContext>,
        _payload: &serde_json::Value,
        result: &serde_json::Value,
        extra: &ExtraColumns,
    ) -> Result<ExtraColumns, HookError> {
        self.after_count.fetch_add(1, Ordering::SeqCst);
        self.after_results.lock().push(result.clone());
        Ok(extra.clone())
    }
}

// ── Conversation tests ───────────────────────────────────────────────

#[tokio::test]
async fn hooked_conversation_create_calls_before_and_after() {
    let inner = Arc::new(MemoryConversationStorage::new());
    let hook = Arc::new(MockHook::new());
    let hooked = HookedConversationStorage::new(inner, hook.clone());

    let conv = hooked
        .create_conversation(NewConversation::default())
        .await
        .unwrap();

    assert_eq!(hook.before_calls(), 1);
    assert_eq!(hook.after_calls(), 1);
    assert!(!conv.id.0.is_empty());
}

#[tokio::test]
async fn hooked_conversation_reject_prevents_create() {
    let inner = Arc::new(MemoryConversationStorage::new());
    let hook = Arc::new(MockHook::new());
    hook.set_reject("not allowed");
    let hooked = HookedConversationStorage::new(inner.clone(), hook.clone());

    let err = hooked
        .create_conversation(NewConversation::default())
        .await
        .unwrap_err();

    assert!(err.to_string().contains("not allowed"));
    // A refusal on purpose: the store is fine, so it is not the 503 of a
    // store that is not available.
    assert!(
        matches!(err, ConversationStorageError::Rejected(_)),
        "{err:?}"
    );
    assert!(!err.is_unavailable(), "{err}");
    assert_eq!(hook.before_calls(), 1);
    assert_eq!(hook.after_calls(), 0); // after not called on rejection
}

#[tokio::test]
async fn hooked_conversation_get_calls_hooks() {
    let inner = Arc::new(MemoryConversationStorage::new());
    let conv = inner
        .create_conversation(NewConversation::default())
        .await
        .unwrap();

    let hook = Arc::new(MockHook::new());
    let hooked = HookedConversationStorage::new(inner, hook.clone());

    let found = hooked.get_conversation(&conv.id).await.unwrap();
    assert!(found.is_some());
    assert_eq!(hook.before_calls(), 1);
    assert_eq!(hook.after_calls(), 1);
}

#[tokio::test]
async fn hooked_conversation_delete_calls_hooks() {
    let inner = Arc::new(MemoryConversationStorage::new());
    let conv = inner
        .create_conversation(NewConversation::default())
        .await
        .unwrap();

    let hook = Arc::new(MockHook::new());
    let hooked = HookedConversationStorage::new(inner, hook.clone());

    let deleted = hooked.delete_conversation(&conv.id).await.unwrap();
    assert!(deleted);
    assert_eq!(hook.before_calls(), 1);
    assert_eq!(hook.after_calls(), 1);
}

// ── Response tests ───────────────────────────────────────────────────

#[tokio::test]
async fn hooked_response_store_calls_hooks() {
    let inner = Arc::new(MemoryResponseStorage::new());
    let hook = Arc::new(MockHook::new());
    let hooked = HookedResponseStorage::new(inner, hook.clone());

    let mut resp = StoredResponse::new(None);
    resp.input = json!("hello");
    let id = hooked.store_response(resp).await.unwrap();

    assert_eq!(hook.before_calls(), 1);
    assert_eq!(hook.after_calls(), 1);
    assert!(!id.0.is_empty());
}

#[tokio::test]
async fn hooked_response_reject_prevents_store() {
    let inner = Arc::new(MemoryResponseStorage::new());
    let hook = Arc::new(MockHook::new());
    hook.set_reject("denied");
    let hooked = HookedResponseStorage::new(inner, hook.clone());

    let err = hooked
        .store_response(StoredResponse::new(None))
        .await
        .unwrap_err();

    assert!(err.to_string().contains("denied"));
    assert_eq!(hook.after_calls(), 0);
}

#[tokio::test]
async fn hooked_response_get_calls_hooks() {
    let inner = Arc::new(MemoryResponseStorage::new());
    let id = inner
        .store_response(StoredResponse::new(None))
        .await
        .unwrap();

    let hook = Arc::new(MockHook::new());
    let hooked = HookedResponseStorage::new(inner, hook.clone());

    let found = hooked.get_response(&id).await.unwrap();
    assert!(found.is_some());
    assert_eq!(hook.before_calls(), 1);
    assert_eq!(hook.after_calls(), 1);
}

#[tokio::test]
async fn hooked_response_chain_for_a_caller_is_one_chain_operation_on_the_cut_chain() {
    // What each hook call was handed: the payload before, the result after.
    struct RecordingHook(parking_lot::Mutex<Vec<(StorageOperation, serde_json::Value)>>);

    #[async_trait]
    impl StorageHook for RecordingHook {
        async fn before(
            &self,
            operation: StorageOperation,
            _context: Option<&RequestContext>,
            payload: &serde_json::Value,
        ) -> Result<BeforeHookResult, HookError> {
            self.0.lock().push((operation, payload.clone()));
            Ok(BeforeHookResult::Continue(ExtraColumns::new()))
        }

        async fn after(
            &self,
            operation: StorageOperation,
            _context: Option<&RequestContext>,
            _payload: &serde_json::Value,
            result: &serde_json::Value,
            extra: &ExtraColumns,
        ) -> Result<ExtraColumns, HookError> {
            self.0.lock().push((operation, result.clone()));
            Ok(extra.clone())
        }
    }

    let inner = Arc::new(MemoryResponseStorage::new());
    let mut parent = StoredResponse::new(None);
    parent.tenant = Some("tenant-b".to_string());
    let parent_id = inner.store_response(parent).await.unwrap();
    let mut head = StoredResponse::new(Some(parent_id));
    head.tenant = Some("tenant-a".to_string());
    let head_id = inner.store_response(head).await.unwrap();

    let hook = Arc::new(RecordingHook(parking_lot::Mutex::new(Vec::new())));
    let hooked = HookedResponseStorage::new(inner, hook.clone());

    let own = hooked
        .get_response_chain_for(&head_id, Some("tenant-a"))
        .await
        .unwrap();
    let refused = hooked
        .get_response_chain_for(&head_id, Some("tenant-b"))
        .await
        .unwrap();

    // the head alone for a caller that may read it and not its parent,
    // nothing for a caller that may not read the head
    assert_eq!(
        own.responses.iter().map(|r| &r.id).collect::<Vec<_>>(),
        [&head_id]
    );
    assert!(refused.responses.is_empty());
    let payload = json!({ "response_id": head_id, "max_depth": null });
    assert_eq!(
        *hook.0.lock(),
        [
            (StorageOperation::GetResponseChain, payload.clone()),
            (StorageOperation::GetResponseChain, json!(own)),
            (StorageOperation::GetResponseChain, payload),
            (StorageOperation::GetResponseChain, json!(refused)),
        ]
    );
}

fn response_with_output_item(item_id: &str) -> StoredResponse {
    let mut stored = StoredResponse::new(None);
    stored.raw_response = json!({
        "id": "resp_1",
        "output": [{"type": "message", "id": item_id, "role": "assistant",
                    "status": "completed", "content": []}]
    });
    stored
}

#[tokio::test]
async fn hooked_response_output_item_lookup_runs_the_response_read_hooks() {
    let inner = Arc::new(MemoryResponseStorage::new());
    let id = inner
        .store_response(response_with_output_item("msg_1"))
        .await
        .unwrap();

    let hook = Arc::new(MockHook::new());
    let hooked = HookedResponseStorage::new(inner, hook.clone());

    let found = hooked.find_response_by_output_item("msg_1").await.unwrap();
    assert_eq!(found.map(|r| r.id), Some(id));
    assert_eq!(hook.before_calls(), 1);
    assert_eq!(hook.after_calls(), 1);

    // an unknown item reads no response: the hooks do not run
    let missing = hooked
        .find_response_by_output_item("msg_unknown")
        .await
        .unwrap();
    assert!(missing.is_none());
    assert_eq!(hook.before_calls(), 1);
    assert_eq!(hook.after_calls(), 1);
}

#[tokio::test]
async fn hooked_response_reject_prevents_output_item_lookup() {
    let inner = Arc::new(MemoryResponseStorage::new());
    inner
        .store_response(response_with_output_item("msg_1"))
        .await
        .unwrap();

    let hook = Arc::new(MockHook::new());
    hook.set_reject("denied");
    let hooked = HookedResponseStorage::new(inner, hook.clone());

    // A read the hook rejects is a miss, not an error: the caller cannot
    // tell a response it may not read from one that does not exist.
    let denied = hooked.find_response_by_output_item("msg_1").await.unwrap();
    assert!(denied.is_none());
    assert_eq!(hook.before_calls(), 1);
    assert_eq!(hook.after_calls(), 0);
}

// ── Item tests ───────────────────────────────────────────────────────

#[tokio::test]
async fn hooked_item_create_calls_hooks() {
    let inner = Arc::new(MemoryConversationItemStorage::new());
    let hook = Arc::new(MockHook::new());
    let hooked = HookedConversationItemStorage::new(inner, hook.clone());

    let item = hooked
        .create_item(NewConversationItem {
            id: None,
            response_id: None,
            item_type: "message".to_string(),
            role: Some("user".to_string()),
            content: json!([]),
            status: Some("completed".to_string()),
            tenant: None,
        })
        .await
        .unwrap();

    assert_eq!(hook.before_calls(), 1);
    assert_eq!(hook.after_calls(), 1);
    assert!(!item.id.0.is_empty());
}

// ── Request context integration ──────────────────────────────────────

#[tokio::test]
async fn hooked_storage_receives_request_context() {
    /// Hook that captures the request context value.
    struct ContextCapturingHook {
        captured: parking_lot::Mutex<Option<String>>,
    }

    #[async_trait]
    impl StorageHook for ContextCapturingHook {
        async fn before(
            &self,
            _op: StorageOperation,
            context: Option<&RequestContext>,
            _payload: &serde_json::Value,
        ) -> Result<BeforeHookResult, HookError> {
            if let Some(ctx) = context {
                if let Some(val) = ctx.get("tenant_id") {
                    *self.captured.lock() = Some(val.to_string());
                }
            }
            Ok(BeforeHookResult::default())
        }

        async fn after(
            &self,
            _op: StorageOperation,
            _ctx: Option<&RequestContext>,
            _payload: &serde_json::Value,
            _result: &serde_json::Value,
            extra: &ExtraColumns,
        ) -> Result<ExtraColumns, HookError> {
            Ok(extra.clone())
        }
    }

    let inner = Arc::new(MemoryConversationStorage::new());
    let hook = Arc::new(ContextCapturingHook {
        captured: parking_lot::Mutex::new(None),
    });
    let hooked = HookedConversationStorage::new(inner, hook.clone());

    let mut ctx = RequestContext::new();
    ctx.set("tenant_id", "tenant_abc");

    with_request_context(ctx, async {
        hooked
            .create_conversation(NewConversation::default())
            .await
            .unwrap();
    })
    .await;

    assert_eq!(hook.captured.lock().as_deref(), Some("tenant_abc"),);
}

// ── ExtraColumns flow through task-local ───────────────────────────

#[tokio::test]
async fn extra_columns_available_to_backend_during_write() {
    /// Hook that returns extra columns from `before`.
    struct ExtraColumnsHook;

    #[async_trait]
    impl StorageHook for ExtraColumnsHook {
        async fn before(
            &self,
            _op: StorageOperation,
            _ctx: Option<&RequestContext>,
            _payload: &serde_json::Value,
        ) -> Result<BeforeHookResult, HookError> {
            let mut extra = ExtraColumns::new();
            extra.insert(
                "tenant_id".to_string(),
                serde_json::Value::String("hook-tenant".to_string()),
            );
            Ok(BeforeHookResult::Continue(extra))
        }

        async fn after(
            &self,
            _op: StorageOperation,
            _ctx: Option<&RequestContext>,
            _payload: &serde_json::Value,
            _result: &serde_json::Value,
            extra: &ExtraColumns,
        ) -> Result<ExtraColumns, HookError> {
            Ok(extra.clone())
        }
    }

    let capturing = Arc::new(CapturingConversationStorage::new());
    let hooked = HookedConversationStorage::new(
        capturing.clone() as Arc<dyn ConversationStorage>,
        Arc::new(ExtraColumnsHook),
    );

    hooked
        .create_conversation(NewConversation::default())
        .await
        .unwrap();

    let captured = capturing.captured.lock().clone().expect("should be set");
    assert_eq!(
        captured.get("tenant_id").and_then(|v| v.as_str()),
        Some("hook-tenant"),
    );
}

// ── Hooks + extra columns: end-to-end ──

/// Configures `extra_columns` on the schema with a hook that provides a
/// TENANT_ID value, and verifies:
/// 1. Hook `before()` is called with `StoreResponse`
/// 2. ExtraColumns from the hook are available via task-local during write
/// 3. Hook `after()` is called with the result
/// 4. The response round-trips correctly (store → get)
#[tokio::test]
async fn hook_with_extra_columns_replaces_forked_backend() {
    /// Hook that returns a TENANT_ID extra column value.
    struct TenantHook;

    #[async_trait]
    impl StorageHook for TenantHook {
        async fn before(
            &self,
            _op: StorageOperation,
            _ctx: Option<&RequestContext>,
            _payload: &serde_json::Value,
        ) -> Result<BeforeHookResult, HookError> {
            let mut extra = ExtraColumns::new();
            extra.insert(
                "TENANT_ID".to_string(),
                serde_json::Value::String("acme-corp".to_string()),
            );
            Ok(BeforeHookResult::Continue(extra))
        }

        async fn after(
            &self,
            _op: StorageOperation,
            _ctx: Option<&RequestContext>,
            _payload: &serde_json::Value,
            _result: &serde_json::Value,
            extra: &ExtraColumns,
        ) -> Result<ExtraColumns, HookError> {
            Ok(extra.clone())
        }
    }

    /// Storage wrapper that captures task-local ExtraColumns during writes.
    struct CapturingResponseStorage {
        inner: MemoryResponseStorage,
        captured: parking_lot::Mutex<Option<ExtraColumns>>,
    }

    #[async_trait]
    impl ResponseStorage for CapturingResponseStorage {
        async fn store_response(
            &self,
            resp: StoredResponse,
        ) -> Result<ResponseId, ResponseStorageError> {
            *self.captured.lock() = current_extra_columns();
            self.inner.store_response(resp).await
        }

        async fn get_response(
            &self,
            id: &ResponseId,
        ) -> Result<Option<StoredResponse>, ResponseStorageError> {
            self.inner.get_response(id).await
        }

        async fn delete_response(&self, id: &ResponseId) -> Result<(), ResponseStorageError> {
            self.inner.delete_response(id).await
        }

        async fn list_identifier_responses(
            &self,
            identifier: &str,
            tenant: Option<&str>,
            limit: Option<usize>,
        ) -> Result<Vec<StoredResponse>, ResponseStorageError> {
            self.inner
                .list_identifier_responses(identifier, tenant, limit)
                .await
        }

        async fn delete_identifier_responses(
            &self,
            identifier: &str,
            tenant: Option<&str>,
        ) -> Result<usize, ResponseStorageError> {
            self.inner
                .delete_identifier_responses(identifier, tenant)
                .await
        }
    }

    let capturing = Arc::new(CapturingResponseStorage {
        inner: MemoryResponseStorage::new(),
        captured: parking_lot::Mutex::new(None),
    });
    let hooked = HookedResponseStorage::new(
        capturing.clone() as Arc<dyn ResponseStorage>,
        Arc::new(TenantHook),
    );

    // Store a response — hook provides TENANT_ID extra column
    let mut resp = StoredResponse::new(None);
    resp.input = json!("hello");
    let id = hooked.store_response(resp).await.unwrap();

    // Verify extra columns were available to the backend during write
    let captured = capturing
        .captured
        .lock()
        .clone()
        .expect("ExtraColumns should be set during store_response");
    assert_eq!(
        captured.get("TENANT_ID").and_then(|v| v.as_str()),
        Some("acme-corp"),
    );

    // Response round-trips correctly
    let stored = hooked.get_response(&id).await.unwrap();
    assert!(stored.is_some());
    assert_eq!(stored.unwrap().input, json!("hello"));
}

/// Proves that RequestContext flows through hooks and can be used to
/// drive extra column values — the full pipeline from request to storage.
#[tokio::test]
async fn request_context_drives_hook_extra_columns() {
    /// Hook that reads tenant_id from RequestContext and produces ExtraColumns.
    struct ContextDrivenHook;

    #[async_trait]
    impl StorageHook for ContextDrivenHook {
        async fn before(
            &self,
            _op: StorageOperation,
            context: Option<&RequestContext>,
            _payload: &serde_json::Value,
        ) -> Result<BeforeHookResult, HookError> {
            let mut extra = ExtraColumns::new();
            if let Some(ctx) = context {
                if let Some(tid) = ctx.get("tenant_id") {
                    extra.insert(
                        "TENANT_ID".to_string(),
                        serde_json::Value::String(tid.to_string()),
                    );
                }
            }
            Ok(BeforeHookResult::Continue(extra))
        }

        async fn after(
            &self,
            _op: StorageOperation,
            _ctx: Option<&RequestContext>,
            _payload: &serde_json::Value,
            _result: &serde_json::Value,
            extra: &ExtraColumns,
        ) -> Result<ExtraColumns, HookError> {
            Ok(extra.clone())
        }
    }

    let capturing = Arc::new(CapturingConversationStorage::new());
    let hooked = HookedConversationStorage::new(
        capturing.clone() as Arc<dyn ConversationStorage>,
        Arc::new(ContextDrivenHook),
    );

    // Set up RequestContext with tenant_id (simulating middleware)
    let mut ctx = RequestContext::new();
    ctx.set("tenant_id", "acme-corp");

    with_request_context(ctx, async {
        hooked
            .create_conversation(NewConversation::default())
            .await
            .unwrap();
    })
    .await;

    // Hook read tenant_id from context → produced ExtraColumns → backend received them
    let captured = capturing
        .captured
        .lock()
        .clone()
        .expect("ExtraColumns should be set during create_conversation");
    assert_eq!(
        captured.get("TENANT_ID").and_then(|v| v.as_str()),
        Some("acme-corp"),
    );
}

// ── Hook error is non-fatal ──────────────────────────────────────────

#[tokio::test]
async fn before_hook_error_does_not_block_operation() {
    struct FailingHook;

    #[async_trait]
    impl StorageHook for FailingHook {
        async fn before(
            &self,
            _op: StorageOperation,
            _ctx: Option<&RequestContext>,
            _payload: &serde_json::Value,
        ) -> Result<BeforeHookResult, HookError> {
            Err(HookError::Internal("oops".to_string()))
        }

        async fn after(
            &self,
            _op: StorageOperation,
            _ctx: Option<&RequestContext>,
            _payload: &serde_json::Value,
            _result: &serde_json::Value,
            extra: &ExtraColumns,
        ) -> Result<ExtraColumns, HookError> {
            Ok(extra.clone())
        }
    }

    let inner = Arc::new(MemoryConversationStorage::new());
    let hooked = HookedConversationStorage::new(inner, Arc::new(FailingHook));

    // Should succeed despite the hook error
    let conv = hooked
        .create_conversation(NewConversation::default())
        .await
        .unwrap();
    assert!(!conv.id.0.is_empty());
}

// ── Integration tests: full decorator-chain round-trips ──────────

#[tokio::test]
async fn hooked_response_store_and_get_round_trips() {
    let inner = Arc::new(MemoryResponseStorage::new());
    let hook = Arc::new(MockHook::new());
    let hooked = HookedResponseStorage::new(inner, hook.clone());

    let mut resp = StoredResponse::new(None);
    resp.input = json!("round-trip-input");
    resp.raw_response = json!({"output": ["round-trip-output"]});
    resp.safety_identifier = Some("user-rt".to_string());

    let id = hooked.store_response(resp).await.unwrap();
    assert_eq!(hook.before_calls(), 1);
    assert_eq!(hook.after_calls(), 1);

    let fetched = hooked.get_response(&id).await.unwrap();
    assert!(fetched.is_some(), "stored response should be retrievable");
    let fetched = fetched.unwrap();
    assert_eq!(fetched.input, json!("round-trip-input"));
    assert_eq!(fetched.raw_response["output"], json!(["round-trip-output"]));
    assert_eq!(fetched.safety_identifier.as_deref(), Some("user-rt"));

    // get_response also triggers before/after hooks
    assert_eq!(hook.before_calls(), 2);
    assert_eq!(hook.after_calls(), 2);
}

#[tokio::test]
async fn hooked_response_list_identifier_responses_with_hook() {
    let inner = Arc::new(MemoryResponseStorage::new());
    let hook = Arc::new(MockHook::new());
    let hooked = HookedResponseStorage::new(inner, hook.clone());

    // Store two responses with the same safety_identifier
    let mut r1 = StoredResponse::new(None);
    r1.input = json!("first");
    r1.safety_identifier = Some("user-1".to_string());
    hooked.store_response(r1).await.unwrap();

    let mut r2 = StoredResponse::new(None);
    r2.input = json!("second");
    r2.safety_identifier = Some("user-1".to_string());
    hooked.store_response(r2).await.unwrap();

    // 2 stores = 2 before + 2 after
    assert_eq!(hook.before_calls(), 2);
    assert_eq!(hook.after_calls(), 2);

    let listed = hooked
        .list_identifier_responses("user-1", None, None)
        .await
        .unwrap();
    assert_eq!(listed.len(), 2, "should list both responses for user-1");

    // list_identifier_responses triggers its own before/after hook
    assert_eq!(hook.before_calls(), 3);
    assert_eq!(hook.after_calls(), 3);
}

#[tokio::test]
async fn hooked_conditional_store_runs_the_store_hooks_and_keeps_a_closed_record() {
    let inner = Arc::new(MemoryResponseStorage::new());
    let hook = Arc::new(MockHook::new());
    let hooked = HookedResponseStorage::new(inner, hook.clone());

    let record = |status: &str| {
        let mut stored = StoredResponse::new(None);
        stored.id = ResponseId::from("resp_1");
        stored.raw_response = json!({"background": true, "status": status});
        stored
    };
    hooked.store_response(record("queued")).await.unwrap();
    assert!(matches!(
        hooked
            .store_response_if_open(record("cancelled"))
            .await
            .unwrap(),
        ConditionalStore::Stored
    ));
    assert!(matches!(
        hooked
            .store_response_if_open(record("completed"))
            .await
            .unwrap(),
        ConditionalStore::Closed(current) if current.raw_response["status"] == "cancelled"
    ));
    assert_eq!(hook.before_calls(), 3, "each store runs the before-hook");
    assert_eq!(
        hook.after_calls(),
        2,
        "the after-hook runs for the two writes, not for the refused one"
    );
    assert_eq!(
        hook.after_results(),
        vec![json!("resp_1"), json!("resp_1")],
        "the after-hook's result is the response's id, as for a plain store"
    );
    let stored = hooked
        .get_response(&ResponseId::from("resp_1"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(stored.raw_response["status"], "cancelled");
}

#[tokio::test]
async fn hooked_transition_runs_the_store_hooks_and_moves_the_object_alone() {
    let inner = Arc::new(MemoryResponseStorage::new());
    let hook = Arc::new(MockHook::new());
    let hooked = HookedResponseStorage::new(inner, hook.clone());

    let mut stored = StoredResponse::new(None);
    stored.id = ResponseId::from("resp_1");
    stored.safety_identifier = Some("user-1".to_string());
    stored.raw_response = json!({"background": true, "status": "queued"});
    hooked.store_response(stored).await.unwrap();
    assert!(matches!(
        hooked
            .update_response_object_if_open(
                &ResponseId::from("resp_1"),
                json!({"background": true, "status": "failed"})
            )
            .await
            .unwrap(),
        ConditionalStore::Stored
    ));
    assert!(matches!(
        hooked
            .update_response_object_if_open(
                &ResponseId::from("resp_1"),
                json!({"background": true, "status": "completed"})
            )
            .await
            .unwrap(),
        ConditionalStore::Closed(current) if current.raw_response["status"] == "failed"
    ));
    assert_eq!(hook.before_calls(), 3, "each attempt runs the before-hook");
    assert_eq!(
        hook.after_results(),
        vec![json!("resp_1"), json!("resp_1")],
        "the after-hook runs for the two writes, with the id"
    );
    let stored = hooked
        .get_response(&ResponseId::from("resp_1"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(stored.raw_response["status"], "failed");
    assert_eq!(
        stored.safety_identifier.as_deref(),
        Some("user-1"),
        "the rest of the record stays"
    );
}

/// A hook's refusal of a conditional store is the refusal it is on a
/// plain store: `Rejected` with the hook's message, nothing written, no
/// after-hook; not the `StorageError` of a failed backend.
#[tokio::test]
async fn hooked_conditional_store_refused_by_the_hook_is_rejected() {
    let inner = Arc::new(MemoryResponseStorage::new());
    let hook = Arc::new(MockHook::new());
    let hooked = HookedResponseStorage::new(inner.clone(), hook.clone());
    hook.set_reject("tenant quota");

    let mut stored = StoredResponse::new(None);
    stored.id = ResponseId::from("resp_1");
    stored.raw_response = json!({"background": true, "status": "queued"});
    let err = hooked
        .store_response_if_open(stored)
        .await
        .expect_err("the hook refuses the store");
    assert!(
        matches!(err, ResponseStorageError::Rejected(ref msg) if msg == "tenant quota"),
        "{err}"
    );
    assert_eq!(hook.before_calls(), 1);
    assert_eq!(hook.after_calls(), 0, "nothing was written");
    assert!(inner
        .get_response(&ResponseId::from("resp_1"))
        .await
        .unwrap()
        .is_none());
}

/// The same for a status transition: the refusal is `Rejected`, and the
/// object does not move.
#[tokio::test]
async fn hooked_transition_refused_by_the_hook_is_rejected() {
    let inner = Arc::new(MemoryResponseStorage::new());
    let hook = Arc::new(MockHook::new());
    let hooked = HookedResponseStorage::new(inner.clone(), hook.clone());
    let mut stored = StoredResponse::new(None);
    stored.id = ResponseId::from("resp_1");
    stored.raw_response = json!({"background": true, "status": "queued"});
    hooked.store_response(stored).await.unwrap();
    hook.set_reject("tenant quota");

    let err = hooked
        .update_response_object_if_open(
            &ResponseId::from("resp_1"),
            json!({"background": true, "status": "failed"}),
        )
        .await
        .expect_err("the hook refuses the transition");
    assert!(
        matches!(err, ResponseStorageError::Rejected(ref msg) if msg == "tenant quota"),
        "{err}"
    );
    assert_eq!(hook.after_calls(), 1, "the one write before the refusal");
    let current = inner
        .get_response(&ResponseId::from("resp_1"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        current.raw_response["status"], "queued",
        "the object did not move"
    );
}

#[tokio::test]
async fn hooked_response_delete_identifier_responses_with_hook() {
    let inner = Arc::new(MemoryResponseStorage::new());
    let hook = Arc::new(MockHook::new());
    let hooked = HookedResponseStorage::new(inner, hook.clone());

    // Store two responses with the same safety_identifier
    let mut r1 = StoredResponse::new(None);
    r1.safety_identifier = Some("user-1".to_string());
    hooked.store_response(r1).await.unwrap();

    let mut r2 = StoredResponse::new(None);
    r2.safety_identifier = Some("user-1".to_string());
    hooked.store_response(r2).await.unwrap();

    assert_eq!(hook.before_calls(), 2);
    assert_eq!(hook.after_calls(), 2);

    // Delete all responses for user-1
    let deleted = hooked
        .delete_identifier_responses("user-1", None)
        .await
        .unwrap();
    assert_eq!(deleted, 2, "should delete both responses");
    assert_eq!(hook.before_calls(), 3);
    assert_eq!(hook.after_calls(), 3);

    // Verify list is now empty
    let listed = hooked
        .list_identifier_responses("user-1", None, None)
        .await
        .unwrap();
    assert!(listed.is_empty(), "no responses should remain after delete");
    assert_eq!(hook.before_calls(), 4);
    assert_eq!(hook.after_calls(), 4);
}

#[tokio::test]
async fn hooked_conversation_update_calls_hooks() {
    let inner = Arc::new(MemoryConversationStorage::new());
    let hook = Arc::new(MockHook::new());
    let hooked = HookedConversationStorage::new(inner, hook.clone());

    // Create a conversation
    let conv = hooked
        .create_conversation(NewConversation::default())
        .await
        .unwrap();
    assert_eq!(hook.before_calls(), 1);
    assert_eq!(hook.after_calls(), 1);

    // Update its metadata
    let mut metadata = ConversationMetadata::new();
    metadata.insert("key".to_string(), json!("value"));

    let updated = hooked
        .update_conversation(&conv.id, Some(metadata.clone()))
        .await
        .unwrap();
    assert_eq!(hook.before_calls(), 2);
    assert_eq!(hook.after_calls(), 2);

    // Verify the metadata was persisted correctly
    let updated = updated.expect("update should return the conversation");
    assert_eq!(updated.metadata, Some(metadata));

    // Read it back via get to confirm persistence
    let fetched = hooked
        .get_conversation(&conv.id)
        .await
        .unwrap()
        .expect("conversation should exist");
    assert_eq!(
        fetched.metadata.as_ref().and_then(|m| m.get("key")),
        Some(&json!("value")),
    );
    assert_eq!(hook.before_calls(), 3);
    assert_eq!(hook.after_calls(), 3);
}

#[tokio::test]
async fn hooked_item_link_and_list_calls_hooks() {
    let conv_store = Arc::new(MemoryConversationStorage::new());
    let conv = conv_store
        .create_conversation(NewConversation::default())
        .await
        .unwrap();

    let inner = Arc::new(MemoryConversationItemStorage::new());
    let hook = Arc::new(MockHook::new());
    let hooked = HookedConversationItemStorage::new(inner, hook.clone());

    // Create an item
    let item = hooked
        .create_item(NewConversationItem {
            id: None,
            response_id: Some("resp-1".to_string()),
            item_type: "message".to_string(),
            role: Some("user".to_string()),
            content: json!("hello"),
            status: None,
            tenant: None,
        })
        .await
        .unwrap();
    assert_eq!(hook.before_calls(), 1);
    assert_eq!(hook.after_calls(), 1);

    // Link item to conversation
    hooked
        .link_item(&conv.id, &item.id, Utc::now())
        .await
        .unwrap();
    assert_eq!(hook.before_calls(), 2);
    assert_eq!(hook.after_calls(), 2);

    // List items for the conversation
    let items = hooked
        .list_items(
            &conv.id,
            ListParams {
                limit: 10,
                order: SortOrder::Asc,
                after: None,
            },
        )
        .await
        .unwrap();
    assert_eq!(hook.before_calls(), 3);
    assert_eq!(hook.after_calls(), 3);

    // Verify the listed item matches what was created
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].id, item.id);
    assert_eq!(items[0].item_type, "message");
    assert_eq!(items[0].role.as_deref(), Some("user"));
    assert_eq!(items[0].content, json!("hello"));
    assert_eq!(items[0].response_id.as_deref(), Some("resp-1"));
}

#[tokio::test]
async fn multiple_operations_accumulate_hook_calls() {
    let conv_inner = Arc::new(MemoryConversationStorage::new());
    let resp_inner = Arc::new(MemoryResponseStorage::new());
    let item_inner = Arc::new(MemoryConversationItemStorage::new());
    let hook = Arc::new(MockHook::new());

    let hooked_conv = HookedConversationStorage::new(conv_inner, hook.clone());
    let hooked_resp = HookedResponseStorage::new(resp_inner, hook.clone());
    let hooked_item = HookedConversationItemStorage::new(item_inner, hook.clone());

    // 1. create_conversation
    let conv = hooked_conv
        .create_conversation(NewConversation::default())
        .await
        .unwrap();
    assert_eq!(hook.before_calls(), 1);
    assert_eq!(hook.after_calls(), 1);

    // 2. store_response
    let mut resp = StoredResponse::new(None);
    resp.input = json!("multi-op");
    hooked_resp.store_response(resp).await.unwrap();
    assert_eq!(hook.before_calls(), 2);
    assert_eq!(hook.after_calls(), 2);

    // 3. create_item
    let item = hooked_item
        .create_item(NewConversationItem {
            id: None,
            response_id: None,
            item_type: "message".to_string(),
            role: Some("user".to_string()),
            content: json!("test"),
            status: None,
            tenant: None,
        })
        .await
        .unwrap();
    assert_eq!(hook.before_calls(), 3);
    assert_eq!(hook.after_calls(), 3);

    // 4. link_item
    hooked_item
        .link_item(&conv.id, &item.id, Utc::now())
        .await
        .unwrap();
    assert_eq!(hook.before_calls(), 4);
    assert_eq!(hook.after_calls(), 4);
}
