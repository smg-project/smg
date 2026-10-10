//! Hooked storage wrappers.
//!
//! Each wrapper implements the public storage trait by delegating to an inner
//! backend while running before/after hooks around every operation.  The public
//! trait signatures are unchanged — callers don't need to know about hooks.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use tracing::warn;

use crate::{
    context::{current_request_context, with_extra_columns},
    core::{
        Conversation, ConversationId, ConversationItem, ConversationItemId, ConversationItemResult,
        ConversationItemStorage, ConversationItemStorageError, ConversationMetadata,
        ConversationResult, ConversationStorage, ConversationStorageError, ListParams,
        NewConversation, NewConversationItem, ResponseChain, ResponseId, ResponseResult,
        ResponseStorage, ResponseStorageError, StoredResponse,
    },
    hooks::{BeforeHookResult, ExtraColumns, StorageHook, StorageOperation},
};

// ────────────────────────────────────────────────────────────────────────────
// Helper
// ────────────────────────────────────────────────────────────────────────────

/// Run the before-hook, returning the extra columns on success or mapping
/// `Reject` / errors to the appropriate storage error via `map_err`.
async fn run_before<E>(
    hook: &dyn StorageHook,
    op: StorageOperation,
    payload: &serde_json::Value,
    map_err: fn(String) -> E,
) -> Result<ExtraColumns, E> {
    let ctx = current_request_context();
    match hook.before(op, ctx.as_ref(), payload).await {
        Ok(BeforeHookResult::Continue(extra)) => Ok(extra),
        Ok(BeforeHookResult::Reject(msg)) => Err(map_err(msg)),
        Err(e) => {
            warn!("before hook error for {op:?}, continuing: {e}");
            Ok(ExtraColumns::new())
        }
    }
}

/// Run the after-hook, logging any errors (non-fatal).
async fn run_after(
    hook: &dyn StorageHook,
    op: StorageOperation,
    payload: &serde_json::Value,
    result: &serde_json::Value,
    extra: &ExtraColumns,
) {
    let ctx = current_request_context();
    if let Err(e) = hook.after(op, ctx.as_ref(), payload, result, extra).await {
        warn!("after hook error for {op:?}: {e}");
    }
}

// ────────────────────────────────────────────────────────────────────────────
// HookedConversationStorage
// ────────────────────────────────────────────────────────────────────────────

pub struct HookedConversationStorage {
    inner: Arc<dyn ConversationStorage>,
    hook: Arc<dyn StorageHook>,
}

impl HookedConversationStorage {
    pub fn new(inner: Arc<dyn ConversationStorage>, hook: Arc<dyn StorageHook>) -> Self {
        Self { inner, hook }
    }
}

#[async_trait]
impl ConversationStorage for HookedConversationStorage {
    async fn create_conversation(
        &self,
        input: NewConversation,
    ) -> ConversationResult<Conversation> {
        let payload = serde_json::to_value(&input).unwrap_or_default();
        let extra = run_before(
            &*self.hook,
            StorageOperation::CreateConversation,
            &payload,
            ConversationStorageError::StorageError,
        )
        .await?;

        let result =
            with_extra_columns(extra.clone(), self.inner.create_conversation(input)).await?;

        let result_json = serde_json::to_value(&result).unwrap_or_default();
        run_after(
            &*self.hook,
            StorageOperation::CreateConversation,
            &payload,
            &result_json,
            &extra,
        )
        .await;

        Ok(result)
    }

    async fn get_conversation(
        &self,
        id: &ConversationId,
    ) -> ConversationResult<Option<Conversation>> {
        let payload = serde_json::to_value(id).unwrap_or_default();
        let extra = run_before(
            &*self.hook,
            StorageOperation::GetConversation,
            &payload,
            ConversationStorageError::StorageError,
        )
        .await?;

        let result = self.inner.get_conversation(id).await?;

        let result_json = serde_json::to_value(&result).unwrap_or_default();
        run_after(
            &*self.hook,
            StorageOperation::GetConversation,
            &payload,
            &result_json,
            &extra,
        )
        .await;

        Ok(result)
    }

    async fn update_conversation(
        &self,
        id: &ConversationId,
        metadata: Option<ConversationMetadata>,
    ) -> ConversationResult<Option<Conversation>> {
        let payload = serde_json::json!({ "id": id, "metadata": metadata });
        let extra = run_before(
            &*self.hook,
            StorageOperation::UpdateConversation,
            &payload,
            ConversationStorageError::StorageError,
        )
        .await?;

        let result =
            with_extra_columns(extra.clone(), self.inner.update_conversation(id, metadata)).await?;

        let result_json = serde_json::to_value(&result).unwrap_or_default();
        run_after(
            &*self.hook,
            StorageOperation::UpdateConversation,
            &payload,
            &result_json,
            &extra,
        )
        .await;

        Ok(result)
    }

    async fn delete_conversation(&self, id: &ConversationId) -> ConversationResult<bool> {
        let payload = serde_json::to_value(id).unwrap_or_default();
        let extra = run_before(
            &*self.hook,
            StorageOperation::DeleteConversation,
            &payload,
            ConversationStorageError::StorageError,
        )
        .await?;

        let result = self.inner.delete_conversation(id).await?;

        let result_json = serde_json::to_value(result).unwrap_or_default();
        run_after(
            &*self.hook,
            StorageOperation::DeleteConversation,
            &payload,
            &result_json,
            &extra,
        )
        .await;

        Ok(result)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// HookedConversationItemStorage
// ────────────────────────────────────────────────────────────────────────────

pub struct HookedConversationItemStorage {
    inner: Arc<dyn ConversationItemStorage>,
    hook: Arc<dyn StorageHook>,
}

impl HookedConversationItemStorage {
    pub fn new(inner: Arc<dyn ConversationItemStorage>, hook: Arc<dyn StorageHook>) -> Self {
        Self { inner, hook }
    }
}

#[async_trait]
impl ConversationItemStorage for HookedConversationItemStorage {
    async fn create_item(
        &self,
        item: NewConversationItem,
    ) -> ConversationItemResult<ConversationItem> {
        let payload = serde_json::to_value(&item).unwrap_or_default();
        let extra = run_before(
            &*self.hook,
            StorageOperation::CreateItem,
            &payload,
            ConversationItemStorageError::StorageError,
        )
        .await?;

        let result = with_extra_columns(extra.clone(), self.inner.create_item(item)).await?;

        let result_json = serde_json::to_value(&result).unwrap_or_default();
        run_after(
            &*self.hook,
            StorageOperation::CreateItem,
            &payload,
            &result_json,
            &extra,
        )
        .await;

        Ok(result)
    }

    async fn link_item(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
        added_at: DateTime<Utc>,
    ) -> ConversationItemResult<()> {
        let payload = serde_json::json!({ "conversation_id": conversation_id, "item_id": item_id, "added_at": added_at });
        let extra = run_before(
            &*self.hook,
            StorageOperation::LinkItem,
            &payload,
            ConversationItemStorageError::StorageError,
        )
        .await?;

        let result = with_extra_columns(
            extra.clone(),
            self.inner.link_item(conversation_id, item_id, added_at),
        )
        .await?;

        run_after(
            &*self.hook,
            StorageOperation::LinkItem,
            &payload,
            &serde_json::Value::Null,
            &extra,
        )
        .await;

        Ok(result)
    }

    async fn link_items(
        &self,
        conversation_id: &ConversationId,
        items: &[(ConversationItemId, DateTime<Utc>)],
    ) -> ConversationItemResult<()> {
        let payload =
            serde_json::json!({ "conversation_id": conversation_id, "items_count": items.len() });
        let extra = run_before(
            &*self.hook,
            StorageOperation::LinkItems,
            &payload,
            ConversationItemStorageError::StorageError,
        )
        .await?;

        let result =
            with_extra_columns(extra.clone(), self.inner.link_items(conversation_id, items))
                .await?;

        run_after(
            &*self.hook,
            StorageOperation::LinkItems,
            &payload,
            &serde_json::Value::Null,
            &extra,
        )
        .await;

        Ok(result)
    }

    async fn list_items(
        &self,
        conversation_id: &ConversationId,
        params: ListParams,
    ) -> ConversationItemResult<Vec<ConversationItem>> {
        let payload = serde_json::json!({ "conversation_id": conversation_id, "params": params });
        let extra = run_before(
            &*self.hook,
            StorageOperation::ListItems,
            &payload,
            ConversationItemStorageError::StorageError,
        )
        .await?;

        let result = self.inner.list_items(conversation_id, params).await?;

        let result_json = serde_json::to_value(&result).unwrap_or_default();
        run_after(
            &*self.hook,
            StorageOperation::ListItems,
            &payload,
            &result_json,
            &extra,
        )
        .await;

        Ok(result)
    }

    async fn get_item(
        &self,
        item_id: &ConversationItemId,
    ) -> ConversationItemResult<Option<ConversationItem>> {
        let payload = serde_json::to_value(item_id).unwrap_or_default();
        let extra = run_before(
            &*self.hook,
            StorageOperation::GetItem,
            &payload,
            ConversationItemStorageError::StorageError,
        )
        .await?;

        let result = self.inner.get_item(item_id).await?;

        let result_json = serde_json::to_value(&result).unwrap_or_default();
        run_after(
            &*self.hook,
            StorageOperation::GetItem,
            &payload,
            &result_json,
            &extra,
        )
        .await;

        Ok(result)
    }

    async fn is_item_linked(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
    ) -> ConversationItemResult<bool> {
        let payload = serde_json::json!({ "conversation_id": conversation_id, "item_id": item_id });
        let extra = run_before(
            &*self.hook,
            StorageOperation::IsItemLinked,
            &payload,
            ConversationItemStorageError::StorageError,
        )
        .await?;

        let result = self.inner.is_item_linked(conversation_id, item_id).await?;

        let result_json = serde_json::to_value(result).unwrap_or_default();
        run_after(
            &*self.hook,
            StorageOperation::IsItemLinked,
            &payload,
            &result_json,
            &extra,
        )
        .await;

        Ok(result)
    }

    async fn delete_item(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
    ) -> ConversationItemResult<()> {
        let payload = serde_json::json!({ "conversation_id": conversation_id, "item_id": item_id });
        let extra = run_before(
            &*self.hook,
            StorageOperation::DeleteItem,
            &payload,
            ConversationItemStorageError::StorageError,
        )
        .await?;

        self.inner.delete_item(conversation_id, item_id).await?;

        run_after(
            &*self.hook,
            StorageOperation::DeleteItem,
            &payload,
            &serde_json::Value::Null,
            &extra,
        )
        .await;

        Ok(())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// HookedResponseStorage
// ────────────────────────────────────────────────────────────────────────────

pub struct HookedResponseStorage {
    inner: Arc<dyn ResponseStorage>,
    hook: Arc<dyn StorageHook>,
}

impl HookedResponseStorage {
    pub fn new(inner: Arc<dyn ResponseStorage>, hook: Arc<dyn StorageHook>) -> Self {
        Self { inner, hook }
    }
}

#[async_trait]
impl ResponseStorage for HookedResponseStorage {
    async fn store_response(&self, response: StoredResponse) -> ResponseResult<ResponseId> {
        let payload = serde_json::to_value(&response).unwrap_or_default();
        let extra = run_before(
            &*self.hook,
            StorageOperation::StoreResponse,
            &payload,
            ResponseStorageError::StorageError,
        )
        .await?;

        let result = with_extra_columns(extra.clone(), self.inner.store_response(response)).await?;

        let result_json = serde_json::to_value(&result).unwrap_or_default();
        run_after(
            &*self.hook,
            StorageOperation::StoreResponse,
            &payload,
            &result_json,
            &extra,
        )
        .await;

        Ok(result)
    }

    async fn get_response(
        &self,
        response_id: &ResponseId,
    ) -> ResponseResult<Option<StoredResponse>> {
        let payload = serde_json::to_value(response_id).unwrap_or_default();
        let extra = run_before(
            &*self.hook,
            StorageOperation::GetResponse,
            &payload,
            ResponseStorageError::StorageError,
        )
        .await?;

        let result = self.inner.get_response(response_id).await?;

        let result_json = serde_json::to_value(&result).unwrap_or_default();
        run_after(
            &*self.hook,
            StorageOperation::GetResponse,
            &payload,
            &result_json,
            &extra,
        )
        .await;

        Ok(result)
    }

    async fn delete_response(&self, response_id: &ResponseId) -> ResponseResult<()> {
        let payload = serde_json::to_value(response_id).unwrap_or_default();
        let extra = run_before(
            &*self.hook,
            StorageOperation::DeleteResponse,
            &payload,
            ResponseStorageError::StorageError,
        )
        .await?;

        self.inner.delete_response(response_id).await?;

        run_after(
            &*self.hook,
            StorageOperation::DeleteResponse,
            &payload,
            &serde_json::Value::Null,
            &extra,
        )
        .await;

        Ok(())
    }

    async fn get_response_chain(
        &self,
        response_id: &ResponseId,
        max_depth: Option<usize>,
    ) -> ResponseResult<ResponseChain> {
        let payload = serde_json::json!({ "response_id": response_id, "max_depth": max_depth });
        let extra = run_before(
            &*self.hook,
            StorageOperation::GetResponseChain,
            &payload,
            ResponseStorageError::StorageError,
        )
        .await?;

        let result = self
            .inner
            .get_response_chain(response_id, max_depth)
            .await?;

        let result_json = serde_json::to_value(&result).unwrap_or_default();
        run_after(
            &*self.hook,
            StorageOperation::GetResponseChain,
            &payload,
            &result_json,
            &extra,
        )
        .await;

        Ok(result)
    }

    /// The lookup by output item id is a read of the response it finds, so the
    /// `GetResponse` hooks run on the found response's id, the policy of
    /// [`get_response`](Self::get_response); no hook operation of its own, so a
    /// hook that never heard of the lookup still sees it as the response read
    /// it is. The id has to be found before a hook can judge it, so a read the
    /// hook rejects is answered as a miss: the caller learns no more about an
    /// item it may not read than about one that does not exist.
    async fn find_response_by_output_item(
        &self,
        item_id: &str,
    ) -> ResponseResult<Option<StoredResponse>> {
        let Some(found) = self.inner.find_response_by_output_item(item_id).await? else {
            return Ok(None);
        };
        let payload = serde_json::to_value(&found.id).unwrap_or_default();
        let Ok(extra) = run_before(
            &*self.hook,
            StorageOperation::GetResponse,
            &payload,
            ResponseStorageError::StorageError,
        )
        .await
        else {
            return Ok(None);
        };

        let result = Some(found);
        let result_json = serde_json::to_value(&result).unwrap_or_default();
        run_after(
            &*self.hook,
            StorageOperation::GetResponse,
            &payload,
            &result_json,
            &extra,
        )
        .await;

        Ok(result)
    }

    async fn list_identifier_responses(
        &self,
        identifier: &str,
        limit: Option<usize>,
    ) -> ResponseResult<Vec<StoredResponse>> {
        let payload = serde_json::json!({ "identifier": identifier, "limit": limit });
        let extra = run_before(
            &*self.hook,
            StorageOperation::ListIdentifierResponses,
            &payload,
            ResponseStorageError::StorageError,
        )
        .await?;

        let result = self
            .inner
            .list_identifier_responses(identifier, limit)
            .await?;

        let result_json = serde_json::to_value(&result).unwrap_or_default();
        run_after(
            &*self.hook,
            StorageOperation::ListIdentifierResponses,
            &payload,
            &result_json,
            &extra,
        )
        .await;

        Ok(result)
    }

    async fn delete_identifier_responses(&self, identifier: &str) -> ResponseResult<usize> {
        let payload = serde_json::json!({ "identifier": identifier });
        let extra = run_before(
            &*self.hook,
            StorageOperation::DeleteIdentifierResponses,
            &payload,
            ResponseStorageError::StorageError,
        )
        .await?;

        let result = self.inner.delete_identifier_responses(identifier).await?;

        let result_json = serde_json::to_value(result).unwrap_or_default();
        run_after(
            &*self.hook,
            StorageOperation::DeleteIdentifierResponses,
            &payload,
            &result_json,
            &extra,
        )
        .await;

        Ok(result)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use serde_json::json;

    use super::*;
    use crate::{
        context::{with_request_context, RequestContext},
        hooks::HookError,
        memory::{MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage},
    };

    // ── Mock hook ────────────────────────────────────────────────────────

    /// A test hook that counts invocations and can be configured to reject.
    struct MockHook {
        before_count: AtomicUsize,
        after_count: AtomicUsize,
        reject_before: parking_lot::Mutex<Option<String>>,
    }

    impl MockHook {
        fn new() -> Self {
            Self {
                before_count: AtomicUsize::new(0),
                after_count: AtomicUsize::new(0),
                reject_before: parking_lot::Mutex::new(None),
            }
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
            _result: &serde_json::Value,
            extra: &ExtraColumns,
        ) -> Result<ExtraColumns, HookError> {
            self.after_count.fetch_add(1, Ordering::SeqCst);
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
        use crate::context::current_extra_columns;

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

        /// Storage that captures the task-local ExtraColumns during writes.
        struct CapturingStorage {
            inner: MemoryConversationStorage,
            captured: parking_lot::Mutex<Option<ExtraColumns>>,
        }

        #[async_trait]
        impl ConversationStorage for CapturingStorage {
            async fn create_conversation(
                &self,
                input: NewConversation,
            ) -> ConversationResult<Conversation> {
                // Capture the extra columns that the hooked wrapper set
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

        let capturing = Arc::new(CapturingStorage {
            inner: MemoryConversationStorage::new(),
            captured: parking_lot::Mutex::new(None),
        });
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
        use crate::context::current_extra_columns;

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
                limit: Option<usize>,
            ) -> Result<Vec<StoredResponse>, ResponseStorageError> {
                self.inner
                    .list_identifier_responses(identifier, limit)
                    .await
            }

            async fn delete_identifier_responses(
                &self,
                identifier: &str,
            ) -> Result<usize, ResponseStorageError> {
                self.inner.delete_identifier_responses(identifier).await
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
        use crate::context::{current_extra_columns, with_request_context};

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

        /// Storage that captures task-local ExtraColumns during writes.
        struct CapturingConvStorage {
            inner: MemoryConversationStorage,
            captured: parking_lot::Mutex<Option<ExtraColumns>>,
        }

        #[async_trait]
        impl ConversationStorage for CapturingConvStorage {
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

        let capturing = Arc::new(CapturingConvStorage {
            inner: MemoryConversationStorage::new(),
            captured: parking_lot::Mutex::new(None),
        });
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
            .list_identifier_responses("user-1", None)
            .await
            .unwrap();
        assert_eq!(listed.len(), 2, "should list both responses for user-1");

        // list_identifier_responses triggers its own before/after hook
        assert_eq!(hook.before_calls(), 3);
        assert_eq!(hook.after_calls(), 3);
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
        let deleted = hooked.delete_identifier_responses("user-1").await.unwrap();
        assert_eq!(deleted, 2, "should delete both responses");
        assert_eq!(hook.before_calls(), 3);
        assert_eq!(hook.after_calls(), 3);

        // Verify list is now empty
        let listed = hooked
            .list_identifier_responses("user-1", None)
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
                    order: crate::core::SortOrder::Asc,
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
}
