//! In-memory storage implementations
//!
//! Used for development and testing - no persistence.

use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;

use super::core::*;

// ============================================================================
// PART 1: MemoryConversationStorage
// ============================================================================

/// In-memory conversation storage used for development and tests
#[derive(Default, Clone)]
pub struct MemoryConversationStorage {
    inner: Arc<RwLock<HashMap<ConversationId, Conversation>>>,
}

impl MemoryConversationStorage {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl ConversationStorage for MemoryConversationStorage {
    async fn create_conversation(
        &self,
        input: NewConversation,
    ) -> ConversationResult<Conversation> {
        let conversation = Conversation::new(input);
        self.inner
            .write()
            .insert(conversation.id.clone(), conversation.clone());
        Ok(conversation)
    }

    async fn get_conversation(
        &self,
        id: &ConversationId,
    ) -> ConversationResult<Option<Conversation>> {
        Ok(self.inner.read().get(id).cloned())
    }

    async fn update_conversation(
        &self,
        id: &ConversationId,
        metadata: Option<ConversationMetadata>,
    ) -> ConversationResult<Option<Conversation>> {
        let mut store = self.inner.write();
        if let Some(entry) = store.get_mut(id) {
            entry.metadata = metadata;
            return Ok(Some(entry.clone()));
        }

        Ok(None)
    }

    async fn delete_conversation(&self, id: &ConversationId) -> ConversationResult<bool> {
        let removed = self.inner.write().remove(id).is_some();
        Ok(removed)
    }
}

// ============================================================================
// PART 2: MemoryConversationItemStorage
// ============================================================================

/// Internal store for conversation items, protected by a single lock to prevent
/// lock ordering inversions between the three maps.
#[derive(Default)]
struct ConversationItemInner {
    /// All items indexed by ID
    items: HashMap<ConversationItemId, ConversationItem>,
    /// Per-conversation sorted links: (timestamp, item_id_str) -> ConversationItemId
    links: HashMap<ConversationId, BTreeMap<(i64, String), ConversationItemId>>,
    /// Per-conversation reverse index: item_id_str -> (timestamp, item_id_str)
    rev_index: HashMap<ConversationId, HashMap<String, (i64, String)>>,
}

#[derive(Default, Clone)]
pub struct MemoryConversationItemStorage {
    inner: Arc<RwLock<ConversationItemInner>>,
}

impl MemoryConversationItemStorage {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl ConversationItemStorage for MemoryConversationItemStorage {
    async fn create_item(
        &self,
        new_item: NewConversationItem,
    ) -> ConversationItemResult<ConversationItem> {
        let id = new_item
            .id
            .clone()
            .unwrap_or_else(|| make_item_id(&new_item.item_type));
        let created_at = Utc::now();
        let item = ConversationItem {
            id: id.clone(),
            response_id: new_item.response_id,
            item_type: new_item.item_type,
            role: new_item.role,
            content: new_item.content,
            status: new_item.status,
            created_at,
        };
        self.inner.write().items.insert(id, item.clone());
        Ok(item)
    }

    async fn link_item(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
        added_at: DateTime<Utc>,
    ) -> ConversationItemResult<()> {
        let mut store = self.inner.write();
        store
            .links
            .entry(conversation_id.clone())
            .or_default()
            .insert((added_at.timestamp(), item_id.0.clone()), item_id.clone());
        store
            .rev_index
            .entry(conversation_id.clone())
            .or_default()
            .insert(item_id.0.clone(), (added_at.timestamp(), item_id.0.clone()));
        Ok(())
    }

    async fn link_items(
        &self,
        conversation_id: &ConversationId,
        items: &[(ConversationItemId, DateTime<Utc>)],
    ) -> ConversationItemResult<()> {
        let mut store = self.inner.write();
        let links = store.links.entry(conversation_id.clone()).or_default();
        for (item_id, added_at) in items {
            links.insert((added_at.timestamp(), item_id.0.clone()), item_id.clone());
        }
        let rev = store.rev_index.entry(conversation_id.clone()).or_default();
        for (item_id, added_at) in items {
            rev.insert(item_id.0.clone(), (added_at.timestamp(), item_id.0.clone()));
        }
        Ok(())
    }

    async fn list_items(
        &self,
        conversation_id: &ConversationId,
        params: ListParams,
    ) -> ConversationItemResult<Vec<ConversationItem>> {
        let store = self.inner.read();
        let map = match store.links.get(conversation_id) {
            Some(m) => m,
            None => return Ok(Vec::new()),
        };

        let after_key: Option<(i64, String)> = if let Some(after_id) = &params.after {
            store
                .rev_index
                .get(conversation_id)
                .and_then(|idx| idx.get(after_id).cloned())
        } else {
            None
        };

        let take = params.limit;
        let mut results: Vec<ConversationItem> = Vec::new();

        use std::ops::Bound::{Excluded, Unbounded};

        let mut push_item = |key: &ConversationItemId| -> bool {
            if let Some(it) = store.items.get(key) {
                results.push(it.clone());
                if results.len() == take {
                    return true;
                }
            }
            false
        };

        match (params.order, after_key) {
            (SortOrder::Desc, Some(k)) => {
                for ((_ts, _id), item_key) in map.range(..k).rev() {
                    if push_item(item_key) {
                        break;
                    }
                }
            }
            (SortOrder::Desc, None) => {
                for ((_ts, _id), item_key) in map.iter().rev() {
                    if push_item(item_key) {
                        break;
                    }
                }
            }
            (SortOrder::Asc, Some(k)) => {
                for ((_ts, _id), item_key) in map.range((Excluded(k), Unbounded)) {
                    if push_item(item_key) {
                        break;
                    }
                }
            }
            (SortOrder::Asc, None) => {
                for ((_ts, _id), item_key) in map {
                    if push_item(item_key) {
                        break;
                    }
                }
            }
        }

        Ok(results)
    }

    async fn get_item(
        &self,
        item_id: &ConversationItemId,
    ) -> ConversationItemResult<Option<ConversationItem>> {
        Ok(self.inner.read().items.get(item_id).cloned())
    }

    async fn is_item_linked(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
    ) -> ConversationItemResult<bool> {
        let store = self.inner.read();
        Ok(store
            .rev_index
            .get(conversation_id)
            .is_some_and(|idx| idx.contains_key(&item_id.0)))
    }

    async fn delete_item(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
    ) -> ConversationItemResult<()> {
        let mut store = self.inner.write();
        let key_to_remove = store
            .rev_index
            .get_mut(conversation_id)
            .and_then(|idx| idx.remove(&item_id.0));

        if let Some(key) = key_to_remove {
            if let Some(conv_links) = store.links.get_mut(conversation_id) {
                conv_links.remove(&key);
            }
        }

        Ok(())
    }
}

// ============================================================================
// PART 3: MemoryResponseStorage
// ============================================================================

/// Internal store structure holding both maps together
#[derive(Default)]
struct InnerStore {
    /// All stored responses indexed by ID
    responses: HashMap<ResponseId, StoredResponse>,
    /// Index of response IDs by safety identifier
    identifier_index: HashMap<String, Vec<ResponseId>>,
}

/// In-memory implementation of response storage
pub struct MemoryResponseStorage {
    /// Single lock wrapping both maps to prevent deadlocks and ensure atomic updates
    store: Arc<RwLock<InnerStore>>,
}

impl MemoryResponseStorage {
    pub fn new() -> Self {
        Self {
            store: Arc::new(RwLock::new(InnerStore::default())),
        }
    }

    /// Get statistics about the store
    #[cfg(test)]
    pub(super) fn stats(&self) -> MemoryStoreStats {
        let store = self.store.read();
        MemoryStoreStats {
            response_count: store.responses.len(),
            identifier_count: store.identifier_index.len(),
        }
    }

    /// Clear all data (useful for testing)
    pub fn clear(&self) {
        let mut store = self.store.write();
        store.responses.clear();
        store.identifier_index.clear();
    }
}

impl Default for MemoryResponseStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ResponseStorage for MemoryResponseStorage {
    async fn store_response(&self, mut response: StoredResponse) -> ResponseResult<ResponseId> {
        // Generate ID if not set
        if response.id.0.is_empty() {
            response.id = ResponseId::new();
        }

        let response_id = response.id.clone();

        // Single lock acquisition for atomic update
        let mut store = self.store.write();

        // Update safety identifier index if specified
        if let Some(ref safety_identifier) = response.safety_identifier {
            store
                .identifier_index
                .entry(safety_identifier.clone())
                .or_default()
                .push(response_id.clone());
        }

        store.responses.insert(response_id.clone(), response);
        tracing::debug!(
            memory_store_size = store.responses.len(),
            "Response stored in memory"
        );

        Ok(response_id)
    }

    async fn get_response(
        &self,
        response_id: &ResponseId,
    ) -> ResponseResult<Option<StoredResponse>> {
        let store = self.store.read();
        let result = store.responses.get(response_id).cloned();
        tracing::debug!(response_id = %response_id.0, found = result.is_some(), "Memory response lookup");
        Ok(result)
    }

    async fn delete_response(&self, response_id: &ResponseId) -> ResponseResult<()> {
        let mut store = self.store.write();

        // Remove the response and update user index if needed
        if let Some(response) = store.responses.remove(response_id) {
            if let Some(ref safety_identifier) = response.safety_identifier {
                if let Some(user_responses) = store.identifier_index.get_mut(safety_identifier) {
                    user_responses.retain(|id| id != response_id);
                }
            }
        }

        Ok(())
    }

    async fn get_response_chain(
        &self,
        response_id: &ResponseId,
        max_depth: Option<usize>,
    ) -> ResponseResult<ResponseChain> {
        let mut chain = ResponseChain::new();
        let max_depth = max_depth.unwrap_or(100); // Default max depth to prevent infinite loops

        // Single lock acquisition: walk the chain and collect responses atomically
        // to prevent concurrent writers from causing silent data loss between reads.
        let store = self.store.read();
        let mut current_id = Some(response_id.clone());
        let mut depth = 0;

        while let Some(id) = current_id {
            if depth >= max_depth {
                break;
            }

            if let Some(response) = store.responses.get(&id) {
                #[expect(
                    clippy::assigning_clones,
                    reason = "false positive: while-let moves out of current_id, making clone_from invalid"
                )]
                {
                    current_id = response.previous_response_id.clone();
                }
                chain.add_response(response.clone());
                depth += 1;
            } else {
                break;
            }
        }
        drop(store);

        // Reverse to get chronological order (oldest first)
        chain.responses.reverse();

        Ok(chain)
    }

    async fn find_response_by_output_item(
        &self,
        item_id: &str,
    ) -> ResponseResult<Option<StoredResponse>> {
        let store = self.store.read();
        let found = store
            .responses
            .values()
            .find(|response| {
                response
                    .raw_response
                    .get("output")
                    .and_then(serde_json::Value::as_array)
                    .is_some_and(|items| {
                        items.iter().any(|item| {
                            item.get("id").and_then(serde_json::Value::as_str) == Some(item_id)
                        })
                    })
            })
            .cloned();
        Ok(found)
    }

    async fn list_identifier_responses(
        &self,
        identifier: &str,
        limit: Option<usize>,
    ) -> ResponseResult<Vec<StoredResponse>> {
        let store = self.store.read();

        if let Some(user_response_ids) = store.identifier_index.get(identifier) {
            // Collect responses with their timestamps for sorting
            let mut responses_with_time: Vec<_> = user_response_ids
                .iter()
                .filter_map(|id| store.responses.get(id).map(|r| (r.created_at, r)))
                .collect();

            // Sort by creation time (newest first)
            responses_with_time.sort_by_key(|(created_at, _)| std::cmp::Reverse(*created_at));

            // Apply limit and collect the actual responses
            let limit = limit.unwrap_or(responses_with_time.len());
            let user_responses: Vec<StoredResponse> = responses_with_time
                .into_iter()
                .take(limit)
                .map(|(_, r)| r.clone())
                .collect();

            Ok(user_responses)
        } else {
            Ok(Vec::new())
        }
    }

    async fn delete_identifier_responses(&self, identifier: &str) -> ResponseResult<usize> {
        let mut store = self.store.write();

        if let Some(user_response_ids) = store.identifier_index.remove(identifier) {
            let count = user_response_ids.len();
            for id in user_response_ids {
                store.responses.remove(&id);
            }
            Ok(count)
        } else {
            Ok(0)
        }
    }
}

/// Statistics for the memory store
#[cfg(test)]
#[derive(Debug, Clone)]
pub(super) struct MemoryStoreStats {
    pub response_count: usize,
    pub identifier_count: usize,
}

#[cfg(test)]
mod tests {
    use chrono::{TimeZone, Utc};
    use serde_json::json;

    use super::*;

    // ========================================================================
    // ConversationItem Tests
    // ========================================================================

    fn make_item(
        item_type: &str,
        role: Option<&str>,
        content: serde_json::Value,
    ) -> NewConversationItem {
        NewConversationItem {
            id: None,
            response_id: None,
            item_type: item_type.to_string(),
            role: role.map(|r| r.to_string()),
            content,
            status: Some("completed".to_string()),
        }
    }

    #[tokio::test]
    async fn test_list_ordering_and_cursors() {
        let store = MemoryConversationItemStorage::new();
        let conv: ConversationId = "conv_test".into();

        // Create 3 items and link them at controlled timestamps
        let i1 = store
            .create_item(make_item("message", Some("user"), json!([])))
            .await
            .unwrap();
        let i2 = store
            .create_item(make_item("message", Some("assistant"), json!([])))
            .await
            .unwrap();
        let i3 = store
            .create_item(make_item("reasoning", None, json!([])))
            .await
            .unwrap();

        let t1 = Utc.timestamp_opt(1_700_000_001, 0).single().unwrap();
        let t2 = Utc.timestamp_opt(1_700_000_002, 0).single().unwrap();
        let t3 = Utc.timestamp_opt(1_700_000_003, 0).single().unwrap();

        store.link_item(&conv, &i1.id, t1).await.unwrap();
        store.link_item(&conv, &i2.id, t2).await.unwrap();
        store.link_item(&conv, &i3.id, t3).await.unwrap();

        // Desc order, no cursor
        let desc = store
            .list_items(
                &conv,
                ListParams {
                    limit: 2,
                    order: SortOrder::Desc,
                    after: None,
                },
            )
            .await
            .unwrap();
        assert!(desc.len() >= 2);
        assert_eq!(desc[0].id, i3.id);
        assert_eq!(desc[1].id, i2.id);

        // Desc with cursor = i2 -> expect i1 next
        let desc_after = store
            .list_items(
                &conv,
                ListParams {
                    limit: 2,
                    order: SortOrder::Desc,
                    after: Some(i2.id.0.clone()),
                },
            )
            .await
            .unwrap();
        assert!(!desc_after.is_empty());
        assert_eq!(desc_after[0].id, i1.id);

        // Asc order, no cursor
        let asc = store
            .list_items(
                &conv,
                ListParams {
                    limit: 2,
                    order: SortOrder::Asc,
                    after: None,
                },
            )
            .await
            .unwrap();
        assert!(asc.len() >= 2);
        assert_eq!(asc[0].id, i1.id);
        assert_eq!(asc[1].id, i2.id);

        // Asc with cursor = i2 -> expect i3 next
        let asc_after = store
            .list_items(
                &conv,
                ListParams {
                    limit: 2,
                    order: SortOrder::Asc,
                    after: Some(i2.id.0.clone()),
                },
            )
            .await
            .unwrap();
        assert!(!asc_after.is_empty());
        assert_eq!(asc_after[0].id, i3.id);
    }

    // ========================================================================
    // Response Tests
    // ========================================================================

    #[tokio::test]
    async fn test_store_with_custom_id() {
        let store = MemoryResponseStorage::new();
        let mut response = StoredResponse::new(None);
        response.id = ResponseId::from("resp_custom");
        response.input = json!("Input");
        response.raw_response = json!({"output": "Output"});
        store.store_response(response.clone()).await.unwrap();
        let retrieved = store
            .get_response(&ResponseId::from("resp_custom"))
            .await
            .unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().raw_response["output"], json!("Output"));
    }

    #[tokio::test]
    async fn find_response_by_output_item_matches_an_output_item_id() {
        let store = MemoryResponseStorage::new();

        let mut response = StoredResponse::new(None);
        response.input = json!([{"type": "message", "role": "user", "content": "hi"}]);
        response.raw_response = json!({
            "id": "resp_1",
            "output": [
                {"type": "reasoning", "id": "rs_1", "summary": []},
                {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                 "content": [{"type": "output_text", "text": "hello", "annotations": []}]}
            ]
        });
        let response_id = store.store_response(response).await.unwrap();

        let found = store.find_response_by_output_item("msg_1").await.unwrap();
        assert_eq!(found.map(|r| r.id), Some(response_id.clone()));
        let found = store.find_response_by_output_item("rs_1").await.unwrap();
        assert_eq!(found.map(|r| r.id), Some(response_id));
        assert!(store
            .find_response_by_output_item("msg_unknown")
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn test_memory_store_basic() {
        let store = MemoryResponseStorage::new();

        // Store a response
        let mut response = StoredResponse::new(None);
        response.input = json!("Hello");
        response.raw_response = json!({"output": "Hi there!"});
        let response_id = store.store_response(response).await.unwrap();

        // Retrieve it
        let retrieved = store.get_response(&response_id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().input, json!("Hello"));

        // Delete it
        store.delete_response(&response_id).await.unwrap();
        let deleted = store.get_response(&response_id).await.unwrap();
        assert!(deleted.is_none());
    }

    #[tokio::test]
    async fn test_response_chain() {
        let store = MemoryResponseStorage::new();

        // Create a chain of responses
        let mut response1 = StoredResponse::new(None);
        response1.input = json!("First");
        response1.raw_response = json!({"output": "First response"});
        let id1 = store.store_response(response1).await.unwrap();

        let mut response2 = StoredResponse::new(Some(id1.clone()));
        response2.input = json!("Second");
        response2.raw_response = json!({"output": "Second response"});
        let id2 = store.store_response(response2).await.unwrap();

        let mut response3 = StoredResponse::new(Some(id2.clone()));
        response3.input = json!("Third");
        response3.raw_response = json!({"output": "Third response"});
        let id3 = store.store_response(response3).await.unwrap();

        // Get the chain
        let chain = store.get_response_chain(&id3, None).await.unwrap();
        assert_eq!(chain.responses.len(), 3);
        assert_eq!(chain.responses[0].input, json!("First"));
        assert_eq!(chain.responses[1].input, json!("Second"));
        assert_eq!(chain.responses[2].input, json!("Third"));

        let limited_chain = store.get_response_chain(&id3, Some(2)).await.unwrap();
        assert_eq!(limited_chain.responses.len(), 2);
        assert_eq!(limited_chain.responses[0].input, json!("Second"));
        assert_eq!(limited_chain.responses[1].input, json!("Third"));
    }

    #[tokio::test]
    async fn test_user_responses() {
        let store = MemoryResponseStorage::new();

        // Store responses for different users
        let mut response1 = StoredResponse::new(None);
        response1.input = json!("User1 message");
        response1.safety_identifier = Some("user1".to_string());
        store.store_response(response1).await.unwrap();

        let mut response2 = StoredResponse::new(None);
        response2.input = json!("Another user1 message");
        response2.safety_identifier = Some("user1".to_string());
        store.store_response(response2).await.unwrap();

        let mut response3 = StoredResponse::new(None);
        response3.input = json!("User2 message");
        response3.safety_identifier = Some("user2".to_string());
        store.store_response(response3).await.unwrap();

        // List user1's responses
        let user1_responses = store
            .list_identifier_responses("user1", None)
            .await
            .unwrap();
        assert_eq!(user1_responses.len(), 2);

        // List user2's responses
        let user2_responses = store
            .list_identifier_responses("user2", None)
            .await
            .unwrap();
        assert_eq!(user2_responses.len(), 1);

        // Delete user1's responses
        let deleted_count = store.delete_identifier_responses("user1").await.unwrap();
        assert_eq!(deleted_count, 2);

        let user1_responses_after = store
            .list_identifier_responses("user1", None)
            .await
            .unwrap();
        assert_eq!(user1_responses_after.len(), 0);

        // User2's responses should still be there
        let user2_responses_after = store
            .list_identifier_responses("user2", None)
            .await
            .unwrap();
        assert_eq!(user2_responses_after.len(), 1);
    }

    #[tokio::test]
    async fn test_memory_store_stats() {
        let store = MemoryResponseStorage::new();

        let mut response1 = StoredResponse::new(None);
        response1.input = json!("Test1");
        response1.safety_identifier = Some("user1".to_string());
        store.store_response(response1).await.unwrap();

        let mut response2 = StoredResponse::new(None);
        response2.input = json!("Test2");
        response2.safety_identifier = Some("user2".to_string());
        store.store_response(response2).await.unwrap();

        let stats = store.stats();
        assert_eq!(stats.response_count, 2);
        assert_eq!(stats.identifier_count, 2);
    }

    #[tokio::test]
    async fn test_conversation_item_storage_clone_shares_state() {
        let store = MemoryConversationItemStorage::new();
        let clone = store.clone();

        // Write through original
        let item = store
            .create_item(make_item("message", Some("user"), json!([])))
            .await
            .unwrap();

        // Read through clone — should see the same item
        let found = clone.get_item(&item.id).await.unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().id, item.id);
    }

    // ========================================================================
    // MemoryConversationStorage Tests
    // ========================================================================

    #[tokio::test]
    async fn test_conversation_create_generates_id() {
        let store = MemoryConversationStorage::new();
        let conv = store
            .create_conversation(NewConversation::default())
            .await
            .expect("create_conversation should succeed");
        assert!(
            conv.id.0.starts_with("conv_"),
            "generated ID should have conv_ prefix"
        );
    }

    #[tokio::test]
    async fn test_conversation_create_with_custom_id() {
        let store = MemoryConversationStorage::new();
        let input = NewConversation {
            id: Some(ConversationId::from("conv_my_custom")),
            metadata: None,
        };
        let conv = store
            .create_conversation(input)
            .await
            .expect("create_conversation should succeed");
        assert_eq!(conv.id.0, "conv_my_custom");
    }

    #[tokio::test]
    async fn test_conversation_create_preserves_metadata() {
        let store = MemoryConversationStorage::new();
        let mut metadata = serde_json::Map::new();
        metadata.insert("key".to_string(), json!("value"));
        metadata.insert("count".to_string(), json!(42));

        let input = NewConversation {
            id: None,
            metadata: Some(metadata.clone()),
        };
        let conv = store
            .create_conversation(input)
            .await
            .expect("create_conversation should succeed");
        let stored_metadata = conv.metadata.expect("metadata should be present");
        assert_eq!(stored_metadata["key"], json!("value"));
        assert_eq!(stored_metadata["count"], json!(42));
    }

    #[tokio::test]
    async fn test_conversation_get_nonexistent_returns_none() {
        let store = MemoryConversationStorage::new();
        let result = store
            .get_conversation(&ConversationId::from("conv_does_not_exist"))
            .await
            .expect("get_conversation should succeed");
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_conversation_get_returns_stored() {
        let store = MemoryConversationStorage::new();
        let conv = store
            .create_conversation(NewConversation {
                id: Some(ConversationId::from("conv_stored")),
                metadata: None,
            })
            .await
            .expect("create_conversation should succeed");

        let retrieved = store
            .get_conversation(&conv.id)
            .await
            .expect("get_conversation should succeed")
            .expect("conversation should exist");
        assert_eq!(retrieved.id, conv.id);
        assert_eq!(retrieved.created_at, conv.created_at);
    }

    #[tokio::test]
    async fn test_conversation_update_metadata() {
        let store = MemoryConversationStorage::new();
        let conv = store
            .create_conversation(NewConversation {
                id: Some(ConversationId::from("conv_update")),
                metadata: None,
            })
            .await
            .expect("create_conversation should succeed");

        // Update with new metadata
        let mut new_metadata = serde_json::Map::new();
        new_metadata.insert("updated".to_string(), json!(true));

        let updated = store
            .update_conversation(&conv.id, Some(new_metadata))
            .await
            .expect("update_conversation should succeed")
            .expect("conversation should exist for update");
        let meta = updated
            .metadata
            .expect("metadata should be present after update");
        assert_eq!(meta["updated"], json!(true));

        // Verify the update persists on subsequent get
        let fetched = store
            .get_conversation(&conv.id)
            .await
            .expect("get_conversation should succeed")
            .expect("conversation should still exist");
        let fetched_meta = fetched.metadata.expect("metadata should persist");
        assert_eq!(fetched_meta["updated"], json!(true));
    }

    #[tokio::test]
    async fn test_conversation_update_nonexistent_returns_none() {
        let store = MemoryConversationStorage::new();
        let result = store
            .update_conversation(&ConversationId::from("conv_ghost"), None)
            .await
            .expect("update_conversation should succeed");
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_conversation_delete_removes() {
        let store = MemoryConversationStorage::new();
        let conv = store
            .create_conversation(NewConversation {
                id: Some(ConversationId::from("conv_to_delete")),
                metadata: None,
            })
            .await
            .expect("create_conversation should succeed");

        let deleted = store
            .delete_conversation(&conv.id)
            .await
            .expect("delete_conversation should succeed");
        assert!(
            deleted,
            "delete should return true for existing conversation"
        );

        let after = store
            .get_conversation(&conv.id)
            .await
            .expect("get_conversation should succeed");
        assert!(after.is_none(), "conversation should be gone after delete");
    }

    #[tokio::test]
    async fn test_conversation_delete_nonexistent_returns_false() {
        let store = MemoryConversationStorage::new();
        let deleted = store
            .delete_conversation(&ConversationId::from("conv_never_existed"))
            .await
            .expect("delete_conversation should succeed");
        assert!(
            !deleted,
            "delete should return false for non-existent conversation"
        );
    }

    #[tokio::test]
    async fn test_multiple_conversations_coexist() {
        let store = MemoryConversationStorage::new();

        let conv1 = store
            .create_conversation(NewConversation {
                id: Some(ConversationId::from("conv_alpha")),
                metadata: None,
            })
            .await
            .expect("create conv1 should succeed");

        let conv2 = store
            .create_conversation(NewConversation {
                id: Some(ConversationId::from("conv_beta")),
                metadata: None,
            })
            .await
            .expect("create conv2 should succeed");

        // Both should be retrievable
        let got1 = store
            .get_conversation(&conv1.id)
            .await
            .expect("get conv1 should succeed")
            .expect("conv1 should exist");
        let got2 = store
            .get_conversation(&conv2.id)
            .await
            .expect("get conv2 should succeed")
            .expect("conv2 should exist");

        assert_eq!(got1.id.0, "conv_alpha");
        assert_eq!(got2.id.0, "conv_beta");

        // Deleting one doesn't affect the other
        store
            .delete_conversation(&conv1.id)
            .await
            .expect("delete conv1 should succeed");

        assert!(store
            .get_conversation(&conv1.id)
            .await
            .expect("get should succeed")
            .is_none());
        assert!(store
            .get_conversation(&conv2.id)
            .await
            .expect("get should succeed")
            .is_some());
    }

    #[tokio::test]
    async fn test_delete_item_unlinks_but_preserves_item() {
        let store = MemoryConversationItemStorage::new();
        let conv: ConversationId = "conv_del".into();

        let item = store
            .create_item(make_item("message", Some("user"), json!([])))
            .await
            .unwrap();
        let t = Utc::now();
        store.link_item(&conv, &item.id, t).await.unwrap();

        // Item is linked
        assert!(store.is_item_linked(&conv, &item.id).await.unwrap());

        // Delete (unlink)
        store.delete_item(&conv, &item.id).await.unwrap();

        // No longer linked
        assert!(!store.is_item_linked(&conv, &item.id).await.unwrap());

        // But item data itself is still retrievable
        assert!(store.get_item(&item.id).await.unwrap().is_some());
    }
}
