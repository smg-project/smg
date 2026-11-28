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
        _tenant: Option<&str>,
        _limit: Option<usize>,
    ) -> ResponseResult<Vec<StoredResponse>> {
        Ok(Vec::new())
    }

    async fn delete_identifier_responses(
        &self,
        _identifier: &str,
        _tenant: Option<&str>,
    ) -> ResponseResult<usize> {
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

/// The sweep of unstored records and the expiry of one reach the inner
/// store through the wrapper: left to the trait's defaults, a deployment
/// built through the factory swept nothing and expired nothing.
#[tokio::test]
async fn the_sweep_and_the_expiry_reach_the_inner_store() {
    struct Sweeps {
        expired_after: std::sync::Mutex<Option<Duration>>,
    }
    #[async_trait]
    impl ResponseStorage for Sweeps {
        async fn store_response(&self, response: StoredResponse) -> ResponseResult<ResponseId> {
            Ok(response.id)
        }
        async fn get_response(
            &self,
            _response_id: &ResponseId,
        ) -> ResponseResult<Option<StoredResponse>> {
            Ok(None)
        }
        async fn delete_response(&self, _response_id: &ResponseId) -> ResponseResult<()> {
            Ok(())
        }
        async fn list_identifier_responses(
            &self,
            _identifier: &str,
            _tenant: Option<&str>,
            _limit: Option<usize>,
        ) -> ResponseResult<Vec<StoredResponse>> {
            Ok(Vec::new())
        }
        async fn delete_identifier_responses(
            &self,
            _identifier: &str,
            _tenant: Option<&str>,
        ) -> ResponseResult<usize> {
            Ok(0)
        }
        async fn expire_response(
            &self,
            _response_id: &ResponseId,
            after: Duration,
        ) -> ResponseResult<()> {
            *self.expired_after.lock().unwrap() = Some(after);
            Ok(())
        }
        async fn delete_unstored_older_than(
            &self,
            _cutoff: DateTime<Utc>,
        ) -> ResponseResult<usize> {
            Ok(7)
        }
    }
    let inner = Arc::new(Sweeps {
        expired_after: std::sync::Mutex::new(None),
    });
    let bounded = BoundedResponseStorage::new(
        inner.clone(),
        CallBound::new("test", Duration::from_millis(200)),
    );
    assert_eq!(
        bounded
            .delete_unstored_older_than(Utc::now())
            .await
            .unwrap(),
        7,
        "the sweep's count is the inner store's"
    );
    bounded
        .expire_response(&ResponseId::from("resp_1"), Duration::from_secs(600))
        .await
        .unwrap();
    assert_eq!(
        *inner.expired_after.lock().unwrap(),
        Some(Duration::from_secs(600)),
        "the expiry reached the inner store with its deadline"
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
                tenant: None,
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
