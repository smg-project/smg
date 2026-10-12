//! Bounded storage wrappers: every call to a remote history store ends within
//! the backend's `op_timeout_secs`.
//!
//! The connection pool and the client library bound their own steps (waiting
//! for a pooled connection, connecting, a response), but nothing bounded a
//! call as a whole on Postgres, and on Redis the client library's own
//! response timeout was the only bound. A store that accepts bytes and does
//! not answer (a frozen or partitioned server, a stalled proxy, a half-open
//! connection) held every request that touched it for as long as the client
//! waited.
//!
//! These wrappers put one bound around each call of the three storage traits.
//! A call that exceeds it is dropped (its pooled connection goes back to the
//! pool; a reply that arrives later is discarded by the client library) and
//! answered as the error type's `Timeout`, with one WARN naming the backend,
//! the operation and the bound. Like the hooked wrappers, they forward every
//! method of a trait: a method added to a storage trait is forwarded here too.
//! The one exception is `get_response_chain`, left to the trait's default so
//! that a chain walk is bounded per record: a long `previous_response_id`
//! chain over a slow store is not refused as a whole.

use std::{future::Future, sync::Arc, time::Duration};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use tracing::warn;

use crate::core::{
    Conversation, ConversationId, ConversationItem, ConversationItemId, ConversationItemResult,
    ConversationItemStorage, ConversationItemStorageError, ConversationMetadata,
    ConversationResult, ConversationStorage, ConversationStorageError, ListParams, NewConversation,
    NewConversationItem, ResponseId, ResponseResult, ResponseStorage, ResponseStorageError,
    StoredResponse,
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

    // `get_response_chain` stays the trait's default: each record of the walk
    // is one bounded `get_response` (see the module doc).

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
        limit: Option<usize>,
    ) -> ResponseResult<Vec<StoredResponse>> {
        self.bound
            .run(
                "list_identifier_responses",
                self.inner.list_identifier_responses(identifier, limit),
                ResponseStorageError::Timeout,
            )
            .await
    }

    async fn delete_identifier_responses(&self, identifier: &str) -> ResponseResult<usize> {
        self.bound
            .run(
                "delete_identifier_responses",
                self.inner.delete_identifier_responses(identifier),
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
mod tests {
    use std::{
        sync::atomic::{AtomicUsize, Ordering},
        time::Instant,
    };

    use serde_json::json;

    use super::*;
    use crate::{core::StorageFailure, memory::MemoryConversationItemStorage};

    /// A response store whose reads never answer and whose writes answer at
    /// once (the shape of a store that froze after the connection was made).
    struct Stalled;

    #[async_trait]
    impl ResponseStorage for Stalled {
        async fn store_response(&self, response: StoredResponse) -> ResponseResult<ResponseId> {
            Ok(response.id)
        }

        async fn get_response(
            &self,
            _response_id: &ResponseId,
        ) -> ResponseResult<Option<StoredResponse>> {
            std::future::pending().await
        }

        async fn delete_response(&self, _response_id: &ResponseId) -> ResponseResult<()> {
            Err(ResponseStorageError::StorageError("refused".to_string()))
        }

        async fn list_identifier_responses(
            &self,
            _identifier: &str,
            _limit: Option<usize>,
        ) -> ResponseResult<Vec<StoredResponse>> {
            Ok(Vec::new())
        }

        async fn delete_identifier_responses(&self, _identifier: &str) -> ResponseResult<usize> {
            Ok(0)
        }
    }

    #[tokio::test]
    async fn a_call_that_answers_in_time_passes_through_unchanged() {
        let bounded = BoundedResponseStorage::new(
            Arc::new(Stalled),
            CallBound::new("test", Duration::from_millis(200)),
        );
        let stored = StoredResponse::new(None);
        let id = stored.id.clone();
        assert_eq!(bounded.store_response(stored).await.unwrap(), id);
        let err = bounded.delete_response(&id).await.unwrap_err();
        assert!(matches!(err, ResponseStorageError::StorageError(ref m) if m == "refused"));
        assert!(!err.timed_out());
    }

    #[tokio::test]
    async fn a_call_the_store_never_answers_fails_at_the_bound_as_a_timeout() {
        let bounded = BoundedResponseStorage::new(
            Arc::new(Stalled),
            CallBound::new("test", Duration::from_millis(50)),
        );
        let started = Instant::now();
        let err = bounded
            .get_response(&ResponseId::from("resp_1"))
            .await
            .expect_err("a read the store never answers must fail");
        let elapsed = started.elapsed();
        assert!(
            elapsed >= Duration::from_millis(50) && elapsed < Duration::from_secs(5),
            "failed after {elapsed:?}, the bound is 50 ms"
        );
        assert!(err.timed_out(), "{err}");
        assert_eq!(
            err.to_string(),
            "Storage timeout: the test get_response call exceeded its bound of 0.05 s"
        );
    }

    /// A batch link is forwarded as one call, so a backend that links the
    /// batch in one round trip keeps doing so behind the bound.
    #[tokio::test]
    async fn a_batch_link_is_forwarded_as_one_call() {
        struct CountingLinks {
            inner: MemoryConversationItemStorage,
            batch_calls: AtomicUsize,
        }

        #[async_trait]
        impl ConversationItemStorage for CountingLinks {
            async fn create_item(
                &self,
                item: NewConversationItem,
            ) -> ConversationItemResult<ConversationItem> {
                self.inner.create_item(item).await
            }

            async fn link_item(
                &self,
                conversation_id: &ConversationId,
                item_id: &ConversationItemId,
                added_at: DateTime<Utc>,
            ) -> ConversationItemResult<()> {
                self.inner
                    .link_item(conversation_id, item_id, added_at)
                    .await
            }

            async fn link_items(
                &self,
                conversation_id: &ConversationId,
                items: &[(ConversationItemId, DateTime<Utc>)],
            ) -> ConversationItemResult<()> {
                self.batch_calls.fetch_add(1, Ordering::SeqCst);
                self.inner.link_items(conversation_id, items).await
            }

            async fn list_items(
                &self,
                conversation_id: &ConversationId,
                params: ListParams,
            ) -> ConversationItemResult<Vec<ConversationItem>> {
                self.inner.list_items(conversation_id, params).await
            }

            async fn get_item(
                &self,
                item_id: &ConversationItemId,
            ) -> ConversationItemResult<Option<ConversationItem>> {
                self.inner.get_item(item_id).await
            }

            async fn is_item_linked(
                &self,
                conversation_id: &ConversationId,
                item_id: &ConversationItemId,
            ) -> ConversationItemResult<bool> {
                self.inner.is_item_linked(conversation_id, item_id).await
            }

            async fn delete_item(
                &self,
                conversation_id: &ConversationId,
                item_id: &ConversationItemId,
            ) -> ConversationItemResult<()> {
                self.inner.delete_item(conversation_id, item_id).await
            }
        }

        let counting = Arc::new(CountingLinks {
            inner: MemoryConversationItemStorage::new(),
            batch_calls: AtomicUsize::new(0),
        });
        let bounded = BoundedConversationItemStorage::new(
            counting.clone(),
            CallBound::new("test", Duration::from_secs(1)),
        );
        let conversation_id = ConversationId::from("conv_1");
        let mut links = Vec::new();
        for n in 0..3 {
            let item = bounded
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
            links.push((item.id, Utc::now() + chrono::TimeDelta::seconds(n)));
        }
        bounded.link_items(&conversation_id, &links).await.unwrap();
        assert_eq!(counting.batch_calls.load(Ordering::SeqCst), 1);
        assert!(bounded
            .is_item_linked(&conversation_id, &links[2].0)
            .await
            .unwrap());
    }
}
