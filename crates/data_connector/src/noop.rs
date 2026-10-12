//! NoOp storage implementations
//!
//! These implementations do nothing - useful for when persistence is disabled.

use async_trait::async_trait;
use chrono::{DateTime, Utc};

use super::core::*;

// ============================================================================
// PART 1: NoOpConversationStorage
// ============================================================================

/// No-op implementation that synthesizes conversation responses without persistence
#[derive(Default, Debug, Clone)]
pub(super) struct NoOpConversationStorage;

impl NoOpConversationStorage {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ConversationStorage for NoOpConversationStorage {
    async fn create_conversation(
        &self,
        input: NewConversation,
    ) -> ConversationResult<Conversation> {
        Ok(Conversation::new(input))
    }

    async fn get_conversation(
        &self,
        _id: &ConversationId,
    ) -> ConversationResult<Option<Conversation>> {
        Ok(None)
    }

    async fn update_conversation(
        &self,
        _id: &ConversationId,
        _metadata: Option<ConversationMetadata>,
    ) -> ConversationResult<Option<Conversation>> {
        Ok(None)
    }

    async fn delete_conversation(&self, _id: &ConversationId) -> ConversationResult<bool> {
        Ok(false)
    }
}

// ============================================================================
// PART 2: NoOpConversationItemStorage
// ============================================================================

/// No-op conversation item storage (does nothing)
#[derive(Clone, Copy, Default)]
pub(super) struct NoOpConversationItemStorage;

impl NoOpConversationItemStorage {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ConversationItemStorage for NoOpConversationItemStorage {
    async fn create_item(
        &self,
        item: NewConversationItem,
    ) -> ConversationItemResult<ConversationItem> {
        let id = item
            .id
            .clone()
            .unwrap_or_else(|| make_item_id(&item.item_type));
        Ok(ConversationItem {
            id,
            response_id: item.response_id,
            item_type: item.item_type,
            role: item.role,
            content: item.content,
            status: item.status,
            created_at: Utc::now(),
            tenant: item.tenant,
        })
    }

    async fn link_item(
        &self,
        _conversation_id: &ConversationId,
        _item_id: &ConversationItemId,
        _added_at: DateTime<Utc>,
    ) -> ConversationItemResult<()> {
        Ok(())
    }

    async fn list_items(
        &self,
        _conversation_id: &ConversationId,
        _params: ListParams,
    ) -> ConversationItemResult<Vec<ConversationItem>> {
        Ok(Vec::new())
    }

    async fn get_item(
        &self,
        _item_id: &ConversationItemId,
    ) -> ConversationItemResult<Option<ConversationItem>> {
        Ok(None)
    }

    async fn is_item_linked(
        &self,
        _conversation_id: &ConversationId,
        _item_id: &ConversationItemId,
    ) -> ConversationItemResult<bool> {
        Ok(false)
    }

    async fn delete_item(
        &self,
        _conversation_id: &ConversationId,
        _item_id: &ConversationItemId,
    ) -> ConversationItemResult<()> {
        Ok(())
    }
}

// ============================================================================
// PART 3: NoOpResponseStorage
// ============================================================================

/// No-op implementation of response storage (does nothing)
pub(super) struct NoOpResponseStorage;

impl NoOpResponseStorage {
    pub fn new() -> Self {
        Self
    }
}

impl Default for NoOpResponseStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ResponseStorage for NoOpResponseStorage {
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

    async fn get_response_chain(
        &self,
        _response_id: &ResponseId,
        _max_depth: Option<usize>,
    ) -> ResponseResult<ResponseChain> {
        Ok(ResponseChain::new())
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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    // ========================================================================
    // NoOpConversationStorage Tests
    // ========================================================================

    #[tokio::test]
    async fn test_create_conversation_generates_id() {
        let store = NoOpConversationStorage::new();
        let conv = store
            .create_conversation(NewConversation::default())
            .await
            .expect("create_conversation should succeed");
        assert!(
            conv.id.0.starts_with("conv_"),
            "generated ID should start with conv_ prefix"
        );
    }

    #[tokio::test]
    async fn test_create_conversation_with_provided_id() {
        let store = NoOpConversationStorage::new();
        let input = NewConversation {
            id: Some(ConversationId::from("conv_custom_123")),
            metadata: None,
            tenant: None,
        };
        let conv = store
            .create_conversation(input)
            .await
            .expect("create_conversation should succeed");
        assert_eq!(conv.id.0, "conv_custom_123");
    }

    #[tokio::test]
    async fn test_get_conversation_always_returns_none() {
        let store = NoOpConversationStorage::new();
        let result = store
            .get_conversation(&ConversationId::from("any_id"))
            .await
            .expect("get_conversation should succeed");
        assert!(
            result.is_none(),
            "NoOp get_conversation should always return None"
        );
    }

    #[tokio::test]
    async fn test_update_conversation_always_returns_none() {
        let store = NoOpConversationStorage::new();
        let result = store
            .update_conversation(&ConversationId::from("any_id"), None)
            .await
            .expect("update_conversation should succeed");
        assert!(
            result.is_none(),
            "NoOp update_conversation should always return None"
        );
    }

    #[tokio::test]
    async fn test_delete_conversation_always_returns_false() {
        let store = NoOpConversationStorage::new();
        let result = store
            .delete_conversation(&ConversationId::from("any_id"))
            .await
            .expect("delete_conversation should succeed");
        assert!(
            !result,
            "NoOp delete_conversation should always return false"
        );
    }

    // ========================================================================
    // NoOpConversationItemStorage Tests
    // ========================================================================

    fn make_test_item(item_type: &str, role: Option<&str>) -> NewConversationItem {
        NewConversationItem {
            id: None,
            response_id: None,
            item_type: item_type.to_string(),
            role: role.map(|r| r.to_string()),
            content: json!([]),
            status: Some("completed".to_string()),
            tenant: None,
        }
    }

    #[tokio::test]
    async fn test_create_item_generates_id() {
        let store = NoOpConversationItemStorage::new();
        let item = store
            .create_item(make_test_item("message", Some("user")))
            .await
            .expect("create_item should succeed");
        assert!(
            item.id.0.starts_with("msg_"),
            "message item ID should start with msg_ prefix"
        );
        assert_eq!(item.item_type, "message");
        assert_eq!(item.role.as_deref(), Some("user"));
    }

    #[tokio::test]
    async fn test_create_item_with_provided_id() {
        let store = NoOpConversationItemStorage::new();
        let input = NewConversationItem {
            id: Some(ConversationItemId::from("msg_custom_456")),
            response_id: None,
            item_type: "message".to_string(),
            role: Some("assistant".to_string()),
            content: json!(["hello"]),
            status: Some("completed".to_string()),
            tenant: None,
        };
        let item = store
            .create_item(input)
            .await
            .expect("create_item should succeed");
        assert_eq!(item.id.0, "msg_custom_456");
    }

    #[tokio::test]
    async fn test_link_item_succeeds() {
        let store = NoOpConversationItemStorage::new();
        store
            .link_item(
                &ConversationId::from("conv_1"),
                &ConversationItemId::from("msg_1"),
                Utc::now(),
            )
            .await
            .expect("link_item should succeed");
    }

    #[tokio::test]
    async fn test_list_items_returns_empty() {
        let store = NoOpConversationItemStorage::new();
        let items = store
            .list_items(
                &ConversationId::from("conv_1"),
                ListParams {
                    limit: 100,
                    order: SortOrder::Asc,
                    after: None,
                },
            )
            .await
            .expect("list_items should succeed");
        assert!(
            items.is_empty(),
            "NoOp list_items should always return empty Vec"
        );
    }

    #[tokio::test]
    async fn test_get_item_always_returns_none() {
        let store = NoOpConversationItemStorage::new();
        let result = store
            .get_item(&ConversationItemId::from("msg_any"))
            .await
            .expect("get_item should succeed");
        assert!(result.is_none(), "NoOp get_item should always return None");
    }

    #[tokio::test]
    async fn test_is_item_linked_always_returns_false() {
        let store = NoOpConversationItemStorage::new();
        let result = store
            .is_item_linked(
                &ConversationId::from("conv_1"),
                &ConversationItemId::from("msg_1"),
            )
            .await
            .expect("is_item_linked should succeed");
        assert!(!result, "NoOp is_item_linked should always return false");
    }

    #[tokio::test]
    async fn test_delete_item_succeeds() {
        let store = NoOpConversationItemStorage::new();
        store
            .delete_item(
                &ConversationId::from("conv_1"),
                &ConversationItemId::from("msg_1"),
            )
            .await
            .expect("delete_item should succeed");
    }

    // ========================================================================
    // NoOpResponseStorage Tests
    // ========================================================================

    #[tokio::test]
    async fn test_store_response_returns_id() {
        let store = NoOpResponseStorage::new();
        let response = StoredResponse::new(None);
        let expected_id = response.id.clone();
        let id = store
            .store_response(response)
            .await
            .expect("store_response should succeed");
        assert_eq!(
            id, expected_id,
            "store_response should return the response's own ID"
        );
    }

    #[tokio::test]
    async fn test_get_response_always_returns_none() {
        let store = NoOpResponseStorage::new();
        let result = store
            .get_response(&ResponseId::from("resp_any"))
            .await
            .expect("get_response should succeed");
        assert!(
            result.is_none(),
            "NoOp get_response should always return None"
        );
    }

    #[tokio::test]
    async fn test_delete_response_succeeds() {
        let store = NoOpResponseStorage::new();
        store
            .delete_response(&ResponseId::from("resp_any"))
            .await
            .expect("delete_response should succeed");
    }

    #[tokio::test]
    async fn test_get_response_chain_returns_empty() {
        let store = NoOpResponseStorage::new();
        let chain = store
            .get_response_chain(&ResponseId::from("resp_any"), None)
            .await
            .expect("get_response_chain should succeed");
        assert!(
            chain.responses.is_empty(),
            "NoOp get_response_chain should return empty chain"
        );
    }

    #[tokio::test]
    async fn test_list_identifier_responses_returns_empty() {
        let store = NoOpResponseStorage::new();
        let responses = store
            .list_identifier_responses("any_identifier", None, None)
            .await
            .expect("list_identifier_responses should succeed");
        assert!(
            responses.is_empty(),
            "NoOp list_identifier_responses should return empty Vec"
        );
    }

    #[tokio::test]
    async fn test_delete_identifier_responses_returns_zero() {
        let store = NoOpResponseStorage::new();
        let count = store
            .delete_identifier_responses("any_identifier", None)
            .await
            .expect("delete_identifier_responses should succeed");
        assert_eq!(count, 0, "NoOp delete_identifier_responses should return 0");
    }
}
