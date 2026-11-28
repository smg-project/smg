//! Bounded storage wrappers: every call to a remote history store ends within
//! the backend's `op_timeout_secs`.
//!
//! The connection pool and the client library bound their own steps (waiting
//! for a pooled connection, connecting, a response), but nothing bounded a
//! call as a whole on Postgres, on Redis the client library's own response
//! timeout was the only bound, and on Oracle the driver's blocking call had
//! none. A store that accepts bytes and does not answer (a frozen or
//! partitioned server, a stalled proxy, a half-open connection) held every
//! request that touched it for as long as the client waited.
//!
//! These wrappers put one bound around each call of the three storage traits.
//! A call that exceeds it is answered as the error type's `Timeout`, with one
//! WARN naming the backend, the operation and the bound, and its future is
//! dropped. On the async drivers (Postgres, Redis) that ends the call: the
//! pooled connection goes back to the pool and a reply that arrives later is
//! discarded by the client library. On a driver that runs each call on a
//! blocking thread (Oracle) the bound ends the wait, not the statement: it
//! runs on, is applied if it completes (the pooled connections autocommit),
//! and its connection returns to the pool when the driver returns (each of
//! its round trips ends within the driver's own call timeout of twice the
//! bound; a statement with several round trips takes longer in total); so
//! a write answered as a timeout may have been applied. Like the hooked
//! wrappers, they forward every
//! method of a trait: a method added to a storage trait is forwarded here too.
//! The exceptions are `get_response_chain` and `get_response_chain_for`, left
//! to the trait's default so that a chain walk is bounded per record: a long
//! `previous_response_id` chain over a slow store is not refused as a whole.

use std::{future::Future, sync::Arc, time::Duration};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde_json::Value;
use tracing::warn;

use crate::core::{
    ConditionalStore, Conversation, ConversationId, ConversationItem, ConversationItemId,
    ConversationItemResult, ConversationItemStorage, ConversationItemStorageError,
    ConversationMetadata, ConversationResult, ConversationStorage, ConversationStorageError,
    ListParams, NewConversation, NewConversationItem, ResponseId, ResponseResult, ResponseStorage,
    ResponseStorageError, StoredResponse,
};

/// The bound of one store call, with the backend's name for the error message
/// and the log line.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CallBound {
    backend: &'static str,
    timeout: Duration,
}

impl CallBound {
    pub(crate) fn new(backend: &'static str, timeout: Duration) -> Self {
        Self { backend, timeout }
    }

    /// Runs one store call under the bound.
    async fn run<T, E, F>(
        &self,
        operation: &'static str,
        call: F,
        timeout_error: fn(String) -> E,
    ) -> Result<T, E>
    where
        F: Future<Output = Result<T, E>>,
    {
        match tokio::time::timeout(self.timeout, call).await {
            Ok(result) => result,
            Err(_) => {
                let bound_secs = self.timeout.as_secs_f64();
                warn!(
                    backend = self.backend,
                    operation, bound_secs, "History store call exceeded its bound"
                );
                Err(timeout_error(format!(
                    "the {} {operation} call exceeded its bound of {bound_secs} s",
                    self.backend
                )))
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Responses
// ────────────────────────────────────────────────────────────────────────────

pub(crate) struct BoundedResponseStorage {
    inner: Arc<dyn ResponseStorage>,
    bound: CallBound,
}

impl BoundedResponseStorage {
    pub(crate) fn new(inner: Arc<dyn ResponseStorage>, bound: CallBound) -> Self {
        Self { inner, bound }
    }
}

#[async_trait]
impl ResponseStorage for BoundedResponseStorage {
    async fn store_response(&self, response: StoredResponse) -> ResponseResult<ResponseId> {
        self.bound
            .run(
                "store_response",
                self.inner.store_response(response),
                ResponseStorageError::Timeout,
            )
            .await
    }

    async fn get_response(
        &self,
        response_id: &ResponseId,
    ) -> ResponseResult<Option<StoredResponse>> {
        self.bound
            .run(
                "get_response",
                self.inner.get_response(response_id),
                ResponseStorageError::Timeout,
            )
            .await
    }

    async fn delete_response(&self, response_id: &ResponseId) -> ResponseResult<()> {
        self.bound
            .run(
                "delete_response",
                self.inner.delete_response(response_id),
                ResponseStorageError::Timeout,
            )
            .await
    }

    // `get_response_chain` and `get_response_chain_for` stay the trait's
    // default: each record of a walk is one bounded `get_response` (see the
    // module doc).

    async fn find_response_by_output_item(
        &self,
        item_id: &str,
    ) -> ResponseResult<Option<StoredResponse>> {
        self.bound
            .run(
                "find_response_by_output_item",
                self.inner.find_response_by_output_item(item_id),
                ResponseStorageError::Timeout,
            )
            .await
    }

    async fn list_identifier_responses(
        &self,
        identifier: &str,
        tenant: Option<&str>,
        limit: Option<usize>,
    ) -> ResponseResult<Vec<StoredResponse>> {
        self.bound
            .run(
                "list_identifier_responses",
                self.inner
                    .list_identifier_responses(identifier, tenant, limit),
                ResponseStorageError::Timeout,
            )
            .await
    }

    async fn delete_identifier_responses(
        &self,
        identifier: &str,
        tenant: Option<&str>,
    ) -> ResponseResult<usize> {
        self.bound
            .run(
                "delete_identifier_responses",
                self.inner.delete_identifier_responses(identifier, tenant),
                ResponseStorageError::Timeout,
            )
            .await
    }

    async fn expire_response(
        &self,
        response_id: &ResponseId,
        after: Duration,
    ) -> ResponseResult<()> {
        self.bound
            .run(
                "expire_response",
                self.inner.expire_response(response_id, after),
                ResponseStorageError::Timeout,
            )
            .await
    }

    async fn store_response_if_open(
        &self,
        response: StoredResponse,
    ) -> ResponseResult<ConditionalStore> {
        self.bound
            .run(
                "store_response_if_open",
                self.inner.store_response_if_open(response),
                ResponseStorageError::Timeout,
            )
            .await
    }

    async fn update_response_object_if_open(
        &self,
        response_id: &ResponseId,
        raw_response: Value,
    ) -> ResponseResult<ConditionalStore> {
        self.bound
            .run(
                "update_response_object_if_open",
                self.inner
                    .update_response_object_if_open(response_id, raw_response),
                ResponseStorageError::Timeout,
            )
            .await
    }

    async fn delete_unstored_older_than(&self, cutoff: DateTime<Utc>) -> ResponseResult<usize> {
        self.bound
            .run(
                "delete_unstored_older_than",
                self.inner.delete_unstored_older_than(cutoff),
                ResponseStorageError::Timeout,
            )
            .await
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Conversations
// ────────────────────────────────────────────────────────────────────────────

pub(crate) struct BoundedConversationStorage {
    inner: Arc<dyn ConversationStorage>,
    bound: CallBound,
}

impl BoundedConversationStorage {
    pub(crate) fn new(inner: Arc<dyn ConversationStorage>, bound: CallBound) -> Self {
        Self { inner, bound }
    }
}

#[async_trait]
impl ConversationStorage for BoundedConversationStorage {
    async fn create_conversation(
        &self,
        input: NewConversation,
    ) -> ConversationResult<Conversation> {
        self.bound
            .run(
                "create_conversation",
                self.inner.create_conversation(input),
                ConversationStorageError::Timeout,
            )
            .await
    }

    async fn get_conversation(
        &self,
        id: &ConversationId,
    ) -> ConversationResult<Option<Conversation>> {
        self.bound
            .run(
                "get_conversation",
                self.inner.get_conversation(id),
                ConversationStorageError::Timeout,
            )
            .await
    }

    async fn update_conversation(
        &self,
        id: &ConversationId,
        metadata: Option<ConversationMetadata>,
    ) -> ConversationResult<Option<Conversation>> {
        self.bound
            .run(
                "update_conversation",
                self.inner.update_conversation(id, metadata),
                ConversationStorageError::Timeout,
            )
            .await
    }

    async fn delete_conversation(&self, id: &ConversationId) -> ConversationResult<bool> {
        self.bound
            .run(
                "delete_conversation",
                self.inner.delete_conversation(id),
                ConversationStorageError::Timeout,
            )
            .await
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Conversation items
// ────────────────────────────────────────────────────────────────────────────

pub(crate) struct BoundedConversationItemStorage {
    inner: Arc<dyn ConversationItemStorage>,
    bound: CallBound,
}

impl BoundedConversationItemStorage {
    pub(crate) fn new(inner: Arc<dyn ConversationItemStorage>, bound: CallBound) -> Self {
        Self { inner, bound }
    }
}

#[async_trait]
impl ConversationItemStorage for BoundedConversationItemStorage {
    async fn create_item(
        &self,
        item: NewConversationItem,
    ) -> ConversationItemResult<ConversationItem> {
        self.bound
            .run(
                "create_item",
                self.inner.create_item(item),
                ConversationItemStorageError::Timeout,
            )
            .await
    }

    async fn link_item(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
        added_at: DateTime<Utc>,
    ) -> ConversationItemResult<()> {
        self.bound
            .run(
                "link_item",
                self.inner.link_item(conversation_id, item_id, added_at),
                ConversationItemStorageError::Timeout,
            )
            .await
    }

    /// Forwarded as one bounded call: a backend may link the batch in one
    /// round trip.
    async fn link_items(
        &self,
        conversation_id: &ConversationId,
        items: &[(ConversationItemId, DateTime<Utc>)],
    ) -> ConversationItemResult<()> {
        self.bound
            .run(
                "link_items",
                self.inner.link_items(conversation_id, items),
                ConversationItemStorageError::Timeout,
            )
            .await
    }

    async fn list_items(
        &self,
        conversation_id: &ConversationId,
        params: ListParams,
    ) -> ConversationItemResult<Vec<ConversationItem>> {
        self.bound
            .run(
                "list_items",
                self.inner.list_items(conversation_id, params),
                ConversationItemStorageError::Timeout,
            )
            .await
    }

    async fn get_item(
        &self,
        item_id: &ConversationItemId,
    ) -> ConversationItemResult<Option<ConversationItem>> {
        self.bound
            .run(
                "get_item",
                self.inner.get_item(item_id),
                ConversationItemStorageError::Timeout,
            )
            .await
    }

    async fn is_item_linked(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
    ) -> ConversationItemResult<bool> {
        self.bound
            .run(
                "is_item_linked",
                self.inner.is_item_linked(conversation_id, item_id),
                ConversationItemStorageError::Timeout,
            )
            .await
    }

    async fn delete_item(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
    ) -> ConversationItemResult<()> {
        self.bound
            .run(
                "delete_item",
                self.inner.delete_item(conversation_id, item_id),
                ConversationItemStorageError::Timeout,
            )
            .await
    }
}

#[cfg(test)]
#[path = "../tests/unit/bounded/tests.rs"]
mod tests;
