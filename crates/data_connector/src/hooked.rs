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
        ConditionalStore, Conversation, ConversationId, ConversationItem, ConversationItemId,
        ConversationItemResult, ConversationItemStorage, ConversationItemStorageError,
        ConversationMetadata, ConversationResult, ConversationStorage, ConversationStorageError,
        ListParams, NewConversation, NewConversationItem, ResponseChain, ResponseId,
        ResponseResult, ResponseStorage, ResponseStorageError, StoredResponse,
    },
    hooks::{BeforeHookResult, ExtraColumns, StorageHook, StorageOperation},
};

// ────────────────────────────────────────────────────────────────────────────
// Helper
// ────────────────────────────────────────────────────────────────────────────

/// Run the before-hook, returning the extra columns on success or mapping
/// `Reject` to the store's `Rejected` error via `map_err` (a refusal on purpose,
/// never the store being unavailable); a hook that fails is logged and skipped.
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
    fn stored_record_count(&self) -> Option<usize> {
        self.inner.stored_record_count()
    }

    async fn create_conversation(
        &self,
        input: NewConversation,
    ) -> ConversationResult<Conversation> {
        let payload = serde_json::to_value(&input).unwrap_or_default();
        let extra = run_before(
            &*self.hook,
            StorageOperation::CreateConversation,
            &payload,
            ConversationStorageError::Rejected,
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
            ConversationStorageError::Rejected,
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
            ConversationStorageError::Rejected,
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
            ConversationStorageError::Rejected,
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
    fn stored_record_count(&self) -> Option<usize> {
        self.inner.stored_record_count()
    }

    async fn create_item(
        &self,
        item: NewConversationItem,
    ) -> ConversationItemResult<ConversationItem> {
        let payload = serde_json::to_value(&item).unwrap_or_default();
        let extra = run_before(
            &*self.hook,
            StorageOperation::CreateItem,
            &payload,
            ConversationItemStorageError::Rejected,
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
            ConversationItemStorageError::Rejected,
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
            ConversationItemStorageError::Rejected,
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
            ConversationItemStorageError::Rejected,
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
            ConversationItemStorageError::Rejected,
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
            ConversationItemStorageError::Rejected,
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
            ConversationItemStorageError::Rejected,
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
    fn stored_record_count(&self) -> Option<usize> {
        self.inner.stored_record_count()
    }

    async fn store_response(&self, response: StoredResponse) -> ResponseResult<ResponseId> {
        let payload = serde_json::to_value(&response).unwrap_or_default();
        let extra = run_before(
            &*self.hook,
            StorageOperation::StoreResponse,
            &payload,
            ResponseStorageError::Rejected,
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

    /// A store that may be refused by the record's status: the before-hook
    /// of [`Self::store_response`] runs for every attempt; the after-hook
    /// runs only when the record was written, with the response's id as its
    /// result, as a plain store's does (a hook that mirrors or audits writes
    /// sees one contract on both paths, and nothing for a refused write).
    async fn store_response_if_open(
        &self,
        response: StoredResponse,
    ) -> ResponseResult<ConditionalStore> {
        let payload = serde_json::to_value(&response).unwrap_or_default();
        let id = serde_json::to_value(&response.id).unwrap_or_default();
        let extra = run_before(
            &*self.hook,
            StorageOperation::StoreResponse,
            &payload,
            ResponseStorageError::Rejected,
        )
        .await?;

        let result =
            with_extra_columns(extra.clone(), self.inner.store_response_if_open(response)).await?;

        if matches!(result, ConditionalStore::Stored) {
            run_after(
                &*self.hook,
                StorageOperation::StoreResponse,
                &payload,
                &id,
                &extra,
            )
            .await;
        }

        Ok(result)
    }

    /// A status transition: the store hooks run as for a store, the
    /// after-hook only when the object was written, with the id as its
    /// result; the record's other columns are not rewritten, so a hook that
    /// supplies none (a transition under no request) resets nothing.
    async fn update_response_object_if_open(
        &self,
        id: &ResponseId,
        raw_response: serde_json::Value,
    ) -> ResponseResult<ConditionalStore> {
        let payload = serde_json::json!({ "id": id, "raw_response": raw_response });
        let id_json = serde_json::to_value(id).unwrap_or_default();
        let extra = run_before(
            &*self.hook,
            StorageOperation::StoreResponse,
            &payload,
            ResponseStorageError::Rejected,
        )
        .await?;

        let result = with_extra_columns(
            extra.clone(),
            self.inner.update_response_object_if_open(id, raw_response),
        )
        .await?;

        if matches!(result, ConditionalStore::Stored) {
            run_after(
                &*self.hook,
                StorageOperation::StoreResponse,
                &payload,
                &id_json,
                &extra,
            )
            .await;
        }

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
            ResponseStorageError::Rejected,
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
            ResponseStorageError::Rejected,
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
            ResponseStorageError::Rejected,
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

    async fn get_response_chain_for(
        &self,
        response_id: &ResponseId,
        tenant: Option<&str>,
    ) -> ResponseResult<ResponseChain> {
        // One chain operation for the hook, as for the walk it stands for.
        let payload = serde_json::json!({ "response_id": response_id, "max_depth": null });
        let extra = run_before(
            &*self.hook,
            StorageOperation::GetResponseChain,
            &payload,
            ResponseStorageError::Rejected,
        )
        .await?;

        let result = self
            .inner
            .get_response_chain_for(response_id, tenant)
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
            ResponseStorageError::Rejected,
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
        tenant: Option<&str>,
        limit: Option<usize>,
    ) -> ResponseResult<Vec<StoredResponse>> {
        let payload =
            serde_json::json!({ "identifier": identifier, "tenant": tenant, "limit": limit });
        let extra = run_before(
            &*self.hook,
            StorageOperation::ListIdentifierResponses,
            &payload,
            ResponseStorageError::Rejected,
        )
        .await?;

        let result = self
            .inner
            .list_identifier_responses(identifier, tenant, limit)
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

    async fn expire_response(
        &self,
        response_id: &ResponseId,
        after: std::time::Duration,
    ) -> ResponseResult<()> {
        self.inner.expire_response(response_id, after).await
    }

    /// Delegated without the hooks: the sweep is the store's own housekeeping
    /// over records whose clients asked for nothing to be kept, and it names no
    /// record before the store finds them, so no hook runs for its deletions.
    async fn delete_unstored_older_than(&self, cutoff: DateTime<Utc>) -> ResponseResult<usize> {
        self.inner.delete_unstored_older_than(cutoff).await
    }

    async fn delete_identifier_responses(
        &self,
        identifier: &str,
        tenant: Option<&str>,
    ) -> ResponseResult<usize> {
        let payload = serde_json::json!({ "identifier": identifier, "tenant": tenant });
        let extra = run_before(
            &*self.hook,
            StorageOperation::DeleteIdentifierResponses,
            &payload,
            ResponseStorageError::Rejected,
        )
        .await?;

        let result = self
            .inner
            .delete_identifier_responses(identifier, tenant)
            .await?;

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
#[path = "../tests/unit/hooked/tests.rs"]
mod tests;
