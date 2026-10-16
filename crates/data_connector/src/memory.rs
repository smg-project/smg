//! In-memory storage implementations
//!
//! Used for development and testing - no persistence. The stores are bounded
//! ([`MemoryConfig`]): the oldest records go when a count bound is crossed or
//! a record is older than the retention, checked on every write.

use std::{
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    hash::Hash,
    sync::Arc,
};

use async_trait::async_trait;
use chrono::{DateTime, TimeDelta, Utc};
use parking_lot::RwLock;

use super::core::*;
use crate::config::MemoryConfig;

// ============================================================================
// Bounds
// ============================================================================

/// The bounds of one store: the count it keeps at most and the age past which
/// a record goes.
#[derive(Debug, Clone, Copy)]
struct Bounds {
    max: Option<usize>,
    retention: Option<TimeDelta>,
}

impl Bounds {
    fn new(max: Option<usize>, retention: Option<std::time::Duration>) -> Self {
        Self {
            max,
            retention: retention.and_then(|retention| TimeDelta::from_std(retention).ok()),
        }
    }

    /// Whether a record stamped `stamp` is older than the retention at `now`.
    fn expired(&self, stamp: DateTime<Utc>, now: DateTime<Utc>) -> bool {
        self.retention
            .is_some_and(|retention| now.signed_duration_since(stamp) >= retention)
    }

    /// Whether `count` records are more than the store keeps.
    fn over(&self, count: usize) -> bool {
        self.max.is_some_and(|max| count > max)
    }
}

/// Why an entry at the front of the order is up for eviction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Why {
    /// Older than the retention.
    Expired,
    /// The store holds more than its count bound.
    Over,
}

/// What became of the record an eviction asked a store to drop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fate {
    /// The record went.
    Removed,
    /// There was no record under the id.
    Absent,
    /// The store keeps the record for now; it is queued again behind the
    /// others with its original stamp (the retention still counts from it).
    Stays,
}

/// The ids of a store in insertion order with their stamps, oldest first.
/// Every entry carries the sequence number it was queued with, and `live`
/// keeps the sequence of the entry that stands for each id the store holds.
/// An entry a delete or a re-insert under the same id left behind stands for
/// nothing: it is skipped at the front, so it never drops the id's newer
/// record, and such entries are compacted away when they outnumber the live
/// ones. The store's real count is what the count bound is held against.
struct Order<K> {
    queue: VecDeque<(DateTime<Utc>, u64, K)>,
    live: HashMap<K, u64>,
    next: u64,
}

impl<K> Default for Order<K> {
    fn default() -> Self {
        Self {
            queue: VecDeque::new(),
            live: HashMap::new(),
            next: 0,
        }
    }
}

impl<K: Clone + Eq + Hash> Order<K> {
    /// Queues `id` at `stamp` as the entry that stands for it; a re-insert or
    /// a touch of an id moves it to the back.
    fn push(&mut self, stamp: DateTime<Utc>, id: K) {
        self.next += 1;
        self.live.insert(id.clone(), self.next);
        self.queue.push_back((stamp, self.next, id));
    }

    /// Moves a live `id` to the back, stamped `now`, and compacts (a read-heavy
    /// workload leaves an entry behind per touch); an id the store does not
    /// hold is left alone.
    fn touch(&mut self, now: DateTime<Utc>, id: &K) {
        if self.live.contains_key(id) {
            self.push(now, id.clone());
            self.compact();
        }
    }

    /// No entry stands for `id` any more (its record went); compacts, as a
    /// delete leaves the entry behind.
    fn forget(&mut self, id: &K) {
        self.live.remove(id);
        self.compact();
    }

    fn clear(&mut self) {
        self.queue.clear();
        self.live.clear();
    }

    /// Drops the entries that stand for nothing once they outnumber the live
    /// ones (a store-then-delete workload leaves one behind per write);
    /// amortised constant time per write.
    fn compact(&mut self) {
        if self.queue.len() > 2 * self.live.len().max(64) {
            let live = &self.live;
            self.queue.retain(|(_, seq, id)| live.get(id) == Some(seq));
        }
    }

    /// The stamp of the entry that stands for `id`.
    #[cfg(test)]
    fn stamp_of(&self, id: &K) -> Option<DateTime<Utc>> {
        let seq = *self.live.get(id)?;
        self.queue
            .iter()
            .find(|(_, entry, _)| *entry == seq)
            .map(|(stamp, _, _)| *stamp)
    }

    /// Evicts from the front while the entry there is expired at `now` or the
    /// store holds more than its count bound: `remove` drops the record under
    /// a live id, told why, and answers what became of it; `len` is the store's
    /// count before. One pass over the entries queued when called (a record
    /// that stays is queued again behind them, with its stamp: the retention
    /// counts from when it was stored, not from the pass that kept it).
    /// Answers how many records went.
    fn evict(
        &mut self,
        bounds: &Bounds,
        now: DateTime<Utc>,
        mut len: usize,
        mut remove: impl FnMut(&K, Why) -> Fate,
    ) -> usize {
        let mut evicted = 0;
        let mut budget = self.queue.len();
        while budget > 0 {
            let Some((stamp, _, _)) = self.queue.front() else {
                break;
            };
            let why = if bounds.expired(*stamp, now) {
                Why::Expired
            } else if bounds.over(len) {
                Why::Over
            } else {
                break;
            };
            budget -= 1;
            let Some((stamp, seq, id)) = self.queue.pop_front() else {
                break;
            };
            if self.live.get(&id) != Some(&seq) {
                continue;
            }
            match remove(&id, why) {
                Fate::Removed => {
                    self.live.remove(&id);
                    evicted += 1;
                    len -= 1;
                }
                Fate::Absent => {
                    self.live.remove(&id);
                }
                Fate::Stays => self.push(stamp, id),
            }
        }
        self.compact();
        evicted
    }
}

// ============================================================================
// PART 1: MemoryConversationStorage
// ============================================================================

#[derive(Default)]
struct ConversationInner {
    conversations: HashMap<ConversationId, Conversation>,
    order: Order<ConversationId>,
}

impl ConversationInner {
    fn evict(&mut self, bounds: &Bounds, now: DateTime<Utc>) -> usize {
        let Self {
            conversations,
            order,
        } = self;
        let len = conversations.len();
        let evicted = order.evict(bounds, now, len, |id, _| {
            if conversations.remove(id).is_some() {
                Fate::Removed
            } else {
                Fate::Absent
            }
        });
        if evicted > 0 {
            tracing::debug!(
                evicted,
                left = conversations.len(),
                "Memory history: conversations evicted"
            );
        }
        evicted
    }
}

/// In-memory conversation storage used for development and tests; the
/// conversations follow the retention of its [`MemoryConfig`], counted from
/// the conversation's last use (created, read or updated), so one a client
/// keeps using is kept (a conversation record is small; its items have a
/// bound of their own).
#[derive(Clone)]
pub struct MemoryConversationStorage {
    inner: Arc<RwLock<ConversationInner>>,
    bounds: Bounds,
}

impl Default for MemoryConversationStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryConversationStorage {
    /// A store with the default bounds ([`MemoryConfig::default`]).
    pub fn new() -> Self {
        Self::with_config(&MemoryConfig::default())
    }

    /// A store bounded as `config` says.
    pub fn with_config(config: &MemoryConfig) -> Self {
        Self {
            inner: Arc::new(RwLock::new(ConversationInner::default())),
            bounds: Bounds::new(None, config.retention()),
        }
    }

    /// Drops the conversations whose last use is older than the retention at
    /// `now`; answers how many went. Every write does this at the current time.
    pub fn evict_at(&self, now: DateTime<Utc>) -> usize {
        self.inner.write().evict(&self.bounds, now)
    }

    /// The stamp the retention of `id` counts from.
    #[cfg(test)]
    pub(super) fn stamp_of(&self, id: &ConversationId) -> Option<DateTime<Utc>> {
        self.inner.read().order.stamp_of(id)
    }
}

#[async_trait]
impl ConversationStorage for MemoryConversationStorage {
    async fn create_conversation(
        &self,
        input: NewConversation,
    ) -> ConversationResult<Conversation> {
        let conversation = Conversation::new(input);
        let mut store = self.inner.write();
        let now = Utc::now();
        if store
            .conversations
            .insert(conversation.id.clone(), conversation.clone())
            .is_none()
        {
            store.order.push(now, conversation.id.clone());
        }
        store.evict(&self.bounds, now);
        Ok(conversation)
    }

    async fn get_conversation(
        &self,
        id: &ConversationId,
    ) -> ConversationResult<Option<Conversation>> {
        // A read is a use: the retention counts from it.
        let mut store = self.inner.write();
        let conversation = store.conversations.get(id).cloned();
        if conversation.is_some() {
            store.order.touch(Utc::now(), id);
        }
        Ok(conversation)
    }

    async fn update_conversation(
        &self,
        id: &ConversationId,
        metadata: Option<ConversationMetadata>,
    ) -> ConversationResult<Option<Conversation>> {
        let mut store = self.inner.write();
        if let Some(entry) = store.conversations.get_mut(id) {
            entry.metadata = metadata;
            let updated = entry.clone();
            store.order.touch(Utc::now(), id);
            return Ok(Some(updated));
        }

        Ok(None)
    }

    async fn delete_conversation(&self, id: &ConversationId) -> ConversationResult<bool> {
        let mut store = self.inner.write();
        let removed = store.conversations.remove(id).is_some();
        if removed {
            store.order.forget(id);
        }
        Ok(removed)
    }

    fn stored_record_count(&self) -> Option<usize> {
        Some(self.inner.read().conversations.len())
    }
}

// ============================================================================
// PART 2: MemoryConversationItemStorage
// ============================================================================

/// The order of a conversation's links: the full `added_at`, then the item id (the pair the
/// SQL backends order by; they keep `added_at` at microsecond resolution).
type LinkKey = (DateTime<Utc>, String);

/// Internal store for conversation items, protected by a single lock to prevent
/// lock ordering inversions between the maps.
#[derive(Default)]
struct ConversationItemInner {
    /// All items indexed by ID
    items: HashMap<ConversationItemId, ConversationItem>,
    /// Per-conversation sorted links: (added_at, item_id_str) -> ConversationItemId
    links: HashMap<ConversationId, BTreeMap<LinkKey, ConversationItemId>>,
    /// Per-conversation reverse index: item_id_str -> (added_at, item_id_str)
    rev_index: HashMap<ConversationId, HashMap<String, LinkKey>>,
    /// The conversations each item is linked into: an evicted item leaves all of them.
    item_conversations: HashMap<ConversationItemId, HashSet<ConversationId>>,
    /// The items in insertion order, for the bound.
    order: Order<ConversationItemId>,
}

impl ConversationItemInner {
    /// Links `item_id` to the conversation at `added_at`. An item the conversation already
    /// has moves to the new stamp: its previous key, found through the reverse index, goes,
    /// so one item never has two keys (a list would show it twice, a delete remove one).
    fn link(
        &mut self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
        added_at: DateTime<Utc>,
    ) {
        let key: LinkKey = (added_at, item_id.0.clone());
        let links = self.links.entry(conversation_id.clone()).or_default();
        let rev = self.rev_index.entry(conversation_id.clone()).or_default();
        if let Some(previous) = rev.insert(item_id.0.clone(), key.clone()) {
            links.remove(&previous);
        }
        links.insert(key, item_id.clone());
        self.item_conversations
            .entry(item_id.clone())
            .or_default()
            .insert(conversation_id.clone());
    }

    /// Removes the link of `item_id` in `conversation_id`; a conversation left
    /// without links drops its maps, an item linked nowhere its entry.
    fn unlink(&mut self, conversation_id: &ConversationId, item_id: &ConversationItemId) {
        let key = self
            .rev_index
            .get_mut(conversation_id)
            .and_then(|idx| idx.remove(&item_id.0));
        if let Some(key) = key {
            if let Some(links) = self.links.get_mut(conversation_id) {
                links.remove(&key);
                if links.is_empty() {
                    self.links.remove(conversation_id);
                    self.rev_index.remove(conversation_id);
                }
            }
        }
        if let Some(conversations) = self.item_conversations.get_mut(item_id) {
            conversations.remove(conversation_id);
            if conversations.is_empty() {
                self.item_conversations.remove(item_id);
            }
        }
    }

    /// Drops the item and its links in every conversation; whether an item went.
    fn evict_item(&mut self, item_id: &ConversationItemId) -> bool {
        let existed = self.items.remove(item_id).is_some();
        if let Some(conversations) = self.item_conversations.remove(item_id) {
            for conversation_id in conversations {
                let key = self
                    .rev_index
                    .get_mut(&conversation_id)
                    .and_then(|idx| idx.remove(&item_id.0));
                if let Some(key) = key {
                    if let Some(links) = self.links.get_mut(&conversation_id) {
                        links.remove(&key);
                        if links.is_empty() {
                            self.links.remove(&conversation_id);
                            self.rev_index.remove(&conversation_id);
                        }
                    }
                }
            }
        }
        existed
    }

    fn evict(&mut self, bounds: &Bounds, now: DateTime<Utc>) -> usize {
        let len = self.items.len();
        let mut order = std::mem::take(&mut self.order);
        let evicted = order.evict(bounds, now, len, |id, _| {
            if self.evict_item(id) {
                Fate::Removed
            } else {
                Fate::Absent
            }
        });
        self.order = order;
        if evicted > 0 {
            tracing::debug!(
                evicted,
                left = self.items.len(),
                "Memory history: conversation items evicted"
            );
        }
        evicted
    }
}

/// In-memory conversation item storage; bounded by the item count and the
/// retention of its [`MemoryConfig`].
#[derive(Clone)]
pub struct MemoryConversationItemStorage {
    inner: Arc<RwLock<ConversationItemInner>>,
    bounds: Bounds,
}

impl Default for MemoryConversationItemStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryConversationItemStorage {
    /// A store with the default bounds ([`MemoryConfig::default`]).
    pub fn new() -> Self {
        Self::with_config(&MemoryConfig::default())
    }

    /// A store bounded as `config` says.
    pub fn with_config(config: &MemoryConfig) -> Self {
        Self {
            inner: Arc::new(RwLock::new(ConversationItemInner::default())),
            bounds: Bounds::new(Some(config.max_items), config.retention()),
        }
    }

    /// Drops the items over the bound or older than the retention at `now`,
    /// unlinking them from every conversation; answers how many went. Every
    /// write does this at the current time.
    pub fn evict_at(&self, now: DateTime<Utc>) -> usize {
        self.inner.write().evict(&self.bounds, now)
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
            tenant: new_item.tenant,
        };
        let mut store = self.inner.write();
        if store.items.insert(id.clone(), item.clone()).is_none() {
            store.order.push(created_at, id);
        }
        store.evict(&self.bounds, created_at);
        Ok(item)
    }

    async fn link_item(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
        added_at: DateTime<Utc>,
    ) -> ConversationItemResult<()> {
        self.inner.write().link(conversation_id, item_id, added_at);
        Ok(())
    }

    async fn link_items(
        &self,
        conversation_id: &ConversationId,
        items: &[(ConversationItemId, DateTime<Utc>)],
    ) -> ConversationItemResult<()> {
        let mut store = self.inner.write();
        for (item_id, added_at) in items {
            store.link(conversation_id, item_id, *added_at);
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

        let after_key: Option<LinkKey> = if let Some(after_id) = &params.after {
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
        self.inner.write().unlink(conversation_id, item_id);
        Ok(())
    }

    fn stored_record_count(&self) -> Option<usize> {
        Some(self.inner.read().items.len())
    }
}

// ============================================================================
// PART 3: MemoryResponseStorage
// ============================================================================

/// Internal store structure holding the maps together
#[derive(Default)]
struct InnerStore {
    /// All stored responses indexed by ID, shared with the chains handed out:
    /// a reader of a chain of ten takes ten references, not ten copies.
    responses: HashMap<ResponseId, Arc<StoredResponse>>,
    /// Index of response IDs by safety identifier
    identifier_index: HashMap<String, Vec<ResponseId>>,
    /// The responses in insertion order (a record replaced under its id keeps
    /// its place), for the bound.
    order: Order<ResponseId>,
}

impl InnerStore {
    /// Whether the record under `id` is a background response still running
    /// (`queued` or `in_progress`): its client is still to read it, so a count
    /// bound passes it over (the retention does not).
    fn running(&self, id: &ResponseId) -> bool {
        self.responses.get(id).is_some_and(|response| {
            matches!(
                response
                    .raw_response
                    .get("status")
                    .and_then(serde_json::Value::as_str),
                Some("queued" | "in_progress")
            )
        })
    }

    fn remove(&mut self, response_id: &ResponseId) -> bool {
        let Some(response) = self.responses.remove(response_id) else {
            return false;
        };
        self.order.forget(response_id);
        if let Some(ref safety_identifier) = response.safety_identifier {
            if let Some(user_responses) = self.identifier_index.get_mut(safety_identifier) {
                user_responses.retain(|id| id != response_id);
                if user_responses.is_empty() {
                    self.identifier_index.remove(safety_identifier);
                }
            }
        }
        true
    }

    fn evict(&mut self, bounds: &Bounds, now: DateTime<Utc>) -> usize {
        let len = self.responses.len();
        let mut order = std::mem::take(&mut self.order);
        let evicted = order.evict(bounds, now, len, |id, why| {
            if why == Why::Over && self.running(id) {
                Fate::Stays
            } else if self.remove(id) {
                Fate::Removed
            } else {
                Fate::Absent
            }
        });
        self.order = order;
        if evicted > 0 {
            tracing::debug!(
                evicted,
                left = self.responses.len(),
                "Memory history: responses evicted"
            );
        }
        evicted
    }
}

/// In-memory implementation of response storage; bounded by the response
/// count and the retention of its [`MemoryConfig`]. A background response
/// still running (`queued` or `in_progress`) is not dropped by the count
/// bound, so its client finds it when it polls; the retention drops it like
/// any record.
pub struct MemoryResponseStorage {
    /// Single lock wrapping the maps to prevent deadlocks and ensure atomic updates
    store: Arc<RwLock<InnerStore>>,
    bounds: Bounds,
}

impl MemoryResponseStorage {
    /// A store with the default bounds ([`MemoryConfig::default`]).
    pub fn new() -> Self {
        Self::with_config(&MemoryConfig::default())
    }

    /// A store bounded as `config` says.
    pub fn with_config(config: &MemoryConfig) -> Self {
        Self {
            store: Arc::new(RwLock::new(InnerStore::default())),
            bounds: Bounds::new(Some(config.max_responses), config.retention()),
        }
    }

    /// Drops the responses over the bound or older than the retention at
    /// `now`; answers how many went. Every write does this at the current time.
    pub fn evict_at(&self, now: DateTime<Utc>) -> usize {
        self.store.write().evict(&self.bounds, now)
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

    /// The entries the order holds, the live ones and the ones left behind.
    #[cfg(test)]
    pub(super) fn order_len(&self) -> usize {
        self.store.read().order.queue.len()
    }

    /// Clear all data (useful for testing)
    pub fn clear(&self) {
        let mut store = self.store.write();
        store.responses.clear();
        store.identifier_index.clear();
        store.order.clear();
    }
}

impl Default for MemoryResponseStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl InnerStore {
    /// Puts `response` under its id (the index keeps one entry per id); a
    /// new id joins the order at `now`, a replaced record keeps its place.
    fn insert(&mut self, response: StoredResponse, now: DateTime<Utc>) -> ResponseId {
        let response_id = response.id.clone();
        if let Some(ref safety_identifier) = response.safety_identifier {
            let ids = self
                .identifier_index
                .entry(safety_identifier.clone())
                .or_default();
            if !ids.contains(&response_id) {
                ids.push(response_id.clone());
            }
        }
        if self
            .responses
            .insert(response_id.clone(), Arc::new(response))
            .is_none()
        {
            self.order.push(now, response_id.clone());
        }
        tracing::debug!(
            memory_store_size = self.responses.len(),
            "Response stored in memory"
        );
        response_id
    }
}

#[async_trait]
impl ResponseStorage for MemoryResponseStorage {
    async fn store_response(&self, mut response: StoredResponse) -> ResponseResult<ResponseId> {
        // Generate ID if not set
        if response.id.0.is_empty() {
            response.id = ResponseId::new();
        }

        // Single lock acquisition for atomic update; a record replaced
        // under its id keeps its one index entry. Every write drops what is
        // over the bounds at this time.
        let now = Utc::now();
        let mut store = self.store.write();
        let response_id = store.insert(response, now);
        store.evict(&self.bounds, now);
        Ok(response_id)
    }

    async fn store_response_if_open(
        &self,
        response: StoredResponse,
    ) -> ResponseResult<ConditionalStore> {
        // The check and the write under the one lock: nothing closes the
        // record in between.
        let mut store = self.store.write();
        match store.responses.get(&response.id) {
            None => Ok(ConditionalStore::Missing),
            Some(current) if current.is_closed() => Ok(ConditionalStore::Closed(Box::new(
                StoredResponse::clone(current),
            ))),
            Some(_) => {
                let now = Utc::now();
                store.insert(response, now);
                store.evict(&self.bounds, now);
                Ok(ConditionalStore::Stored)
            }
        }
    }

    async fn update_response_object_if_open(
        &self,
        id: &ResponseId,
        raw_response: serde_json::Value,
    ) -> ResponseResult<ConditionalStore> {
        // The check and the write under the one lock; the rest of the record
        // (its identifier, model, input) stays as it was stored.
        let mut store = self.store.write();
        match store.responses.get_mut(id) {
            None => Ok(ConditionalStore::Missing),
            Some(current) if current.is_closed() => Ok(ConditionalStore::Closed(Box::new(
                StoredResponse::clone(current),
            ))),
            Some(current) => {
                // In place unless a chain handed out earlier still reads the
                // record, which then keeps the object it was given.
                Arc::make_mut(current).raw_response = raw_response;
                Ok(ConditionalStore::Stored)
            }
        }
    }

    async fn get_response(
        &self,
        response_id: &ResponseId,
    ) -> ResponseResult<Option<StoredResponse>> {
        let store = self.store.read();
        let result = store
            .responses
            .get(response_id)
            .map(|response| StoredResponse::clone(response));
        tracing::debug!(response_id = %response_id.0, found = result.is_some(), "Memory response lookup");
        Ok(result)
    }

    async fn delete_response(&self, response_id: &ResponseId) -> ResponseResult<()> {
        self.store.write().remove(response_id);
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
                chain.add_shared(Arc::clone(response));
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
            .map(|response| StoredResponse::clone(response));
        Ok(found)
    }

    async fn list_identifier_responses(
        &self,
        identifier: &str,
        tenant: Option<&str>,
        limit: Option<usize>,
    ) -> ResponseResult<Vec<StoredResponse>> {
        let store = self.store.read();

        if let Some(user_response_ids) = store.identifier_index.get(identifier) {
            // Collect the responses the caller may see with their timestamps for sorting
            let mut responses_with_time: Vec<_> = user_response_ids
                .iter()
                .filter_map(|id| store.responses.get(id))
                .filter(|r| r.is_visible_to(tenant))
                .map(|r| (r.created_at, r))
                .collect();

            // Sort by creation time (newest first)
            responses_with_time.sort_by_key(|(created_at, _)| std::cmp::Reverse(*created_at));

            // Apply limit and collect the actual responses
            let limit = limit.unwrap_or(responses_with_time.len());
            let user_responses: Vec<StoredResponse> = responses_with_time
                .into_iter()
                .take(limit)
                .map(|(_, r)| StoredResponse::clone(r))
                .collect();

            Ok(user_responses)
        } else {
            Ok(Vec::new())
        }
    }

    async fn delete_unstored_older_than(&self, cutoff: DateTime<Utc>) -> ResponseResult<usize> {
        let mut store = self.store.write();
        let expired: Vec<ResponseId> = store
            .responses
            .iter()
            .filter(|(_, response)| unstored_terminal_before(response, cutoff))
            .map(|(id, _)| id.clone())
            .collect();
        // Through the one remover: the order forgets the id and an emptied
        // identifier index entry goes with the record.
        for id in &expired {
            store.remove(id);
        }
        Ok(expired.len())
    }

    async fn delete_identifier_responses(
        &self,
        identifier: &str,
        tenant: Option<&str>,
    ) -> ResponseResult<usize> {
        let mut store = self.store.write();

        let Some(user_response_ids) = store.identifier_index.remove(identifier) else {
            return Ok(0);
        };
        // The records the caller may see go; the others keep their index entry.
        let (gone, kept): (Vec<ResponseId>, Vec<ResponseId>) =
            user_response_ids.into_iter().partition(|id| {
                store
                    .responses
                    .get(id)
                    .is_none_or(|r| r.is_visible_to(tenant))
            });
        let count = gone
            .iter()
            .filter(|id| store.responses.remove(id).is_some())
            .count();
        for id in &gone {
            store.order.forget(id);
        }
        if !kept.is_empty() {
            store.identifier_index.insert(identifier.to_string(), kept);
        }
        Ok(count)
    }

    fn stored_record_count(&self) -> Option<usize> {
        Some(self.store.read().responses.len())
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
            tenant: None,
        }
    }

    #[tokio::test]
    async fn links_a_microsecond_apart_within_one_second_list_in_added_at_order() {
        let store = MemoryConversationItemStorage::new();
        let conv: ConversationId = "conv_same_second".into();

        // Ids in the reverse of the link order: a store that keys its links by whole seconds
        // ties the three and lists them by id, backwards.
        let mut ids = Vec::new();
        for id in ["msg_z", "msg_m", "msg_a"] {
            let mut item = make_item("message", Some("user"), json!([]));
            item.id = Some(ConversationItemId::from(id));
            ids.push(store.create_item(item).await.unwrap().id);
        }
        let base = Utc.timestamp_opt(1_700_000_000, 0).single().unwrap();
        for (i, id) in ids.iter().enumerate() {
            let added_at = base + TimeDelta::microseconds(i as i64);
            store.link_item(&conv, id, added_at).await.unwrap();
        }

        let list = |order: SortOrder, after: Option<&str>| {
            let store = store.clone();
            let conv = conv.clone();
            let params = ListParams {
                limit: 10,
                order,
                after: after.map(String::from),
            };
            async move {
                store
                    .list_items(&conv, params)
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|item| item.id.0)
                    .collect::<Vec<_>>()
            }
        };

        assert_eq!(
            list(SortOrder::Asc, None).await,
            ["msg_z", "msg_m", "msg_a"]
        );
        assert_eq!(
            list(SortOrder::Desc, None).await,
            ["msg_a", "msg_m", "msg_z"]
        );
        assert_eq!(
            list(SortOrder::Asc, Some("msg_z")).await,
            ["msg_m", "msg_a"]
        );
        assert_eq!(
            list(SortOrder::Desc, Some("msg_a")).await,
            ["msg_m", "msg_z"]
        );
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
    async fn unstored_terminal_records_past_the_cutoff_are_deleted_and_the_rest_kept() {
        let store = MemoryResponseStorage::new();
        let cutoff = Utc::now();
        let past = cutoff.timestamp() - 60;
        let later = cutoff.timestamp() + 60;
        for (id, raw) in [
            (
                "resp_old_unstored",
                json!({"status": "completed", "store": false, "completed_at": past}),
            ),
            (
                "resp_fresh_unstored",
                json!({"status": "failed", "store": false, "completed_at": later}),
            ),
            (
                "resp_old_stored",
                json!({"status": "completed", "store": true, "completed_at": past}),
            ),
            (
                "resp_running_unstored",
                json!({"status": "in_progress", "store": false}),
            ),
        ] {
            let mut response = StoredResponse::new(None);
            response.id = ResponseId::from(id);
            response.safety_identifier = Some("user-1".to_string());
            response.raw_response = raw;
            store.store_response(response).await.unwrap();
        }

        assert_eq!(store.delete_unstored_older_than(cutoff).await.unwrap(), 1);

        assert!(store
            .get_response(&ResponseId::from("resp_old_unstored"))
            .await
            .unwrap()
            .is_none());
        for kept in [
            "resp_fresh_unstored",
            "resp_old_stored",
            "resp_running_unstored",
        ] {
            assert!(
                store
                    .get_response(&ResponseId::from(kept))
                    .await
                    .unwrap()
                    .is_some(),
                "{kept}"
            );
        }
        // The identifier index lost the one entry with the record.
        let listed = store
            .list_identifier_responses("user-1", None, None)
            .await
            .unwrap();
        assert_eq!(listed.len(), 3);
        // A second pass finds nothing left to do.
        assert_eq!(store.delete_unstored_older_than(cutoff).await.unwrap(), 0);
    }

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
    async fn storing_under_an_existing_id_replaces_the_record_and_keeps_one_index_entry() {
        let store = MemoryResponseStorage::new();
        let mut queued = StoredResponse::new(None);
        queued.id = ResponseId::from("resp_bg");
        queued.safety_identifier = Some("user-1".to_string());
        queued.raw_response = json!({"id": "resp_bg", "status": "queued", "output": []});
        store.store_response(queued.clone()).await.unwrap();

        let mut done = queued;
        done.raw_response =
            json!({"id": "resp_bg", "status": "completed", "output": [{"id": "msg_1"}]});
        store.store_response(done).await.unwrap();

        let stored = store
            .get_response(&ResponseId::from("resp_bg"))
            .await
            .unwrap()
            .expect("the record stays under its id");
        assert_eq!(stored.raw_response["status"], "completed");
        let listed = store
            .list_identifier_responses("user-1", None, None)
            .await
            .unwrap();
        assert_eq!(listed.len(), 1, "one record, one index entry");
        assert_eq!(store.stats().response_count, 1);
    }

    #[tokio::test]
    async fn a_conditional_store_replaces_an_open_record_and_refuses_a_closed_or_missing_one() {
        let store = MemoryResponseStorage::new();
        let record = |status: &str| {
            let mut response = StoredResponse::new(None);
            response.id = ResponseId::from("resp_1");
            response.safety_identifier = Some("user-1".to_string());
            response.raw_response = json!({"id": "resp_1", "background": true, "status": status});
            response
        };
        assert!(matches!(
            store
                .store_response_if_open(record("in_progress"))
                .await
                .unwrap(),
            ConditionalStore::Missing
        ));
        assert!(store
            .get_response(&ResponseId::from("resp_1"))
            .await
            .unwrap()
            .is_none());

        store.store_response(record("queued")).await.unwrap();
        assert!(matches!(
            store
                .store_response_if_open(record("in_progress"))
                .await
                .unwrap(),
            ConditionalStore::Stored
        ));
        let stored = store
            .get_response(&ResponseId::from("resp_1"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(stored.raw_response["status"], "in_progress");
        assert!(matches!(
            store
                .store_response_if_open(record("cancelled"))
                .await
                .unwrap(),
            ConditionalStore::Stored
        ));

        // Closed: the write is refused and the record handed back as it is.
        let ConditionalStore::Closed(current) = store
            .store_response_if_open(record("completed"))
            .await
            .unwrap()
        else {
            panic!("a cancelled record is closed");
        };
        assert_eq!(current.raw_response["status"], "cancelled");
        let stored = store
            .get_response(&ResponseId::from("resp_1"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(stored.raw_response["status"], "cancelled");
        assert_eq!(store.stats().response_count, 1);
        assert_eq!(
            store
                .list_identifier_responses("user-1", None, None)
                .await
                .unwrap()
                .len(),
            1,
            "one record, one index entry"
        );
    }

    #[tokio::test]
    async fn a_transition_moves_the_stored_object_and_keeps_the_rest_of_the_record() {
        let store = MemoryResponseStorage::new();
        let mut response = StoredResponse::new(None);
        response.id = ResponseId::from("resp_1");
        response.safety_identifier = Some("user-1".to_string());
        response.model = Some("m".to_string());
        response.raw_response = json!({"id": "resp_1", "background": true, "status": "queued"});
        assert!(matches!(
            store
                .update_response_object_if_open(
                    &ResponseId::from("resp_1"),
                    json!({"id": "resp_1", "status": "in_progress"})
                )
                .await
                .unwrap(),
            ConditionalStore::Missing
        ));
        store.store_response(response).await.unwrap();
        assert!(matches!(
            store
                .update_response_object_if_open(
                    &ResponseId::from("resp_1"),
                    json!({"id": "resp_1", "background": true, "status": "cancelled"})
                )
                .await
                .unwrap(),
            ConditionalStore::Stored
        ));
        let stored = store
            .get_response(&ResponseId::from("resp_1"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(stored.raw_response["status"], "cancelled");
        assert_eq!(stored.safety_identifier.as_deref(), Some("user-1"));
        assert_eq!(stored.model.as_deref(), Some("m"));
        let ConditionalStore::Closed(current) = store
            .update_response_object_if_open(
                &ResponseId::from("resp_1"),
                json!({"id": "resp_1", "status": "completed"}),
            )
            .await
            .unwrap()
        else {
            panic!("a cancelled record is closed");
        };
        assert_eq!(current.raw_response["status"], "cancelled");
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
    async fn a_chain_walk_shares_the_stored_records_and_a_status_write_leaves_them_as_read() {
        let store = MemoryResponseStorage::new();
        let mut first = StoredResponse::new(None);
        first.input = json!("First");
        first.raw_response = json!({"status": "completed", "output": []});
        let id1 = store.store_response(first).await.unwrap();
        let mut second = StoredResponse::new(Some(id1.clone()));
        second.raw_response = json!({"status": "in_progress", "output": []});
        let id2 = store.store_response(second).await.unwrap();

        let chain = store.get_response_chain(&id2, None).await.unwrap();
        assert_eq!(chain.responses.len(), 2);
        // The chain holds the store's own records: a second reference each, no copy.
        assert_eq!(Arc::strong_count(&chain.responses[0]), 2);
        assert_eq!(Arc::strong_count(&chain.responses[1]), 2);

        // A status write on a record the chain is reading: the store moves on, the chain keeps
        // the object it was given, as it did when it held a copy.
        let written = store
            .update_response_object_if_open(
                &id2,
                json!({"status": "completed", "output": [{"id": "msg_1"}]}),
            )
            .await
            .unwrap();
        assert!(matches!(written, ConditionalStore::Stored));
        assert_eq!(chain.responses[1].raw_response["status"], "in_progress");
        let stored = store.get_response(&id2).await.unwrap().unwrap();
        assert_eq!(stored.raw_response["status"], "completed");
        drop(chain);
        let chain = store.get_response_chain(&id2, None).await.unwrap();
        assert_eq!(chain.responses[1].raw_response["output"][0]["id"], "msg_1");
        assert_eq!(Arc::strong_count(&chain.responses[1]), 2);
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
            .list_identifier_responses("user1", None, None)
            .await
            .unwrap();
        assert_eq!(user1_responses.len(), 2);

        // List user2's responses
        let user2_responses = store
            .list_identifier_responses("user2", None, None)
            .await
            .unwrap();
        assert_eq!(user2_responses.len(), 1);

        // Delete user1's responses
        let deleted_count = store
            .delete_identifier_responses("user1", None)
            .await
            .unwrap();
        assert_eq!(deleted_count, 2);

        let user1_responses_after = store
            .list_identifier_responses("user1", None, None)
            .await
            .unwrap();
        assert_eq!(user1_responses_after.len(), 0);

        // User2's responses should still be there
        let user2_responses_after = store
            .list_identifier_responses("user2", None, None)
            .await
            .unwrap();
        assert_eq!(user2_responses_after.len(), 1);
    }

    /// A record keeps the tenant it was stored under, and the listing and the
    /// deletion by safety identifier see only the records of the caller's
    /// tenant (plus those stored under none, as before).
    #[tokio::test]
    async fn the_identifier_listing_and_deletion_are_scoped_to_the_callers_tenant() {
        let store = MemoryResponseStorage::new();
        let mut of_a = StoredResponse::new(None);
        of_a.safety_identifier = Some("user-1".to_string());
        of_a.tenant = Some("header:a".to_string());
        let id_a = store.store_response(of_a).await.unwrap();
        let mut of_b = StoredResponse::new(None);
        of_b.safety_identifier = Some("user-1".to_string());
        of_b.tenant = Some("header:b".to_string());
        store.store_response(of_b).await.unwrap();
        let mut of_nobody = StoredResponse::new(None);
        of_nobody.safety_identifier = Some("user-1".to_string());
        let id_nobody = store.store_response(of_nobody).await.unwrap();

        let stored = store.get_response(&id_a).await.unwrap().unwrap();
        assert_eq!(stored.tenant.as_deref(), Some("header:a"));

        let ids = |responses: Vec<StoredResponse>| {
            responses.into_iter().map(|r| r.id).collect::<Vec<_>>()
        };
        let listed = store
            .list_identifier_responses("user-1", Some("header:a"), None)
            .await
            .unwrap();
        assert_eq!(listed.len(), 2, "a's record and the untenanted one");
        assert!(ids(listed).iter().all(|id| *id == id_a || *id == id_nobody));
        let listed = store
            .list_identifier_responses("user-1", None, None)
            .await
            .unwrap();
        assert_eq!(
            ids(listed),
            vec![id_nobody.clone()],
            "no tenant: the untenanted record only"
        );

        // a's deletion takes a's record and the untenanted one; b's stays listed for b.
        let deleted = store
            .delete_identifier_responses("user-1", Some("header:a"))
            .await
            .unwrap();
        assert_eq!(deleted, 2);
        assert!(store.get_response(&id_a).await.unwrap().is_none());
        let left = store
            .list_identifier_responses("user-1", Some("header:b"), None)
            .await
            .unwrap();
        assert_eq!(left.len(), 1);
        assert_eq!(left[0].tenant.as_deref(), Some("header:b"));
        assert_eq!(
            store.stats().identifier_count,
            1,
            "the index entry stays for b"
        );
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

    /// The default store keeps at most 10,000 responses: the oldest go.
    #[tokio::test]
    async fn the_default_store_keeps_at_most_ten_thousand_responses() {
        let store = MemoryResponseStorage::new();
        let mut first = None;
        for i in 0..10_001 {
            let mut response = StoredResponse::new(None);
            response.input = json!(format!("request {i}"));
            let id = store.store_response(response).await.unwrap();
            first.get_or_insert(id);
        }
        assert_eq!(store.stats().response_count, 10_000);
        assert!(
            store.get_response(&first.unwrap()).await.unwrap().is_none(),
            "the oldest record went"
        );
    }

    /// The default store keeps at most 100,000 conversation items.
    #[tokio::test]
    async fn the_default_store_keeps_at_most_one_hundred_thousand_items() {
        let store = MemoryConversationItemStorage::new();
        let conv: ConversationId = "conv_bound".into();
        let mut first = None;
        for _ in 0..100_001 {
            let item = store
                .create_item(make_item("message", Some("user"), json!([])))
                .await
                .unwrap();
            store.link_item(&conv, &item.id, Utc::now()).await.unwrap();
            first.get_or_insert(item.id);
        }
        let first = first.unwrap();
        assert!(
            store.get_item(&first).await.unwrap().is_none(),
            "the oldest item went"
        );
        assert!(
            !store.is_item_linked(&conv, &first).await.unwrap(),
            "and left the conversation"
        );
        let listed = store
            .list_items(
                &conv,
                ListParams {
                    limit: 1_000_000,
                    order: SortOrder::Asc,
                    after: None,
                },
            )
            .await
            .unwrap();
        assert_eq!(listed.len(), 100_000);
    }

    /// A configured count bound evicts the oldest records on every write; a
    /// record deleted by a client is no longer counted.
    #[tokio::test]
    async fn the_response_bound_evicts_the_oldest_records_on_write() {
        let config = MemoryConfig {
            max_responses: 2,
            max_items: 1,
            retention_secs: None,
        };
        let store = MemoryResponseStorage::with_config(&config);
        let mut ids = Vec::new();
        for i in 0..3 {
            let mut response = StoredResponse::new(None);
            response.safety_identifier = Some("user-1".to_string());
            response.input = json!(i);
            ids.push(store.store_response(response).await.unwrap());
        }
        assert!(store.get_response(&ids[0]).await.unwrap().is_none());
        assert!(store.get_response(&ids[1]).await.unwrap().is_some());
        assert!(store.get_response(&ids[2]).await.unwrap().is_some());
        assert_eq!(store.stored_record_count(), Some(2));
        // The index follows: the evicted record is not listed.
        let listed = store
            .list_identifier_responses("user-1", None, None)
            .await
            .unwrap();
        assert_eq!(listed.len(), 2);
        // A record re-stored under its id keeps its place: it is not younger.
        let mut again = StoredResponse::new(None);
        again.id = ids[1].clone();
        again.input = json!("again");
        store.store_response(again).await.unwrap();
        assert_eq!(store.stored_record_count(), Some(2));
        let mut fourth = StoredResponse::new(None);
        fourth.input = json!(4);
        let fourth = store.store_response(fourth).await.unwrap();
        assert!(
            store.get_response(&ids[1]).await.unwrap().is_none(),
            "the re-stored record was still the oldest"
        );
        assert!(store.get_response(&fourth).await.unwrap().is_some());
        // A deleted record leaves room: its id in the order is skipped.
        store.delete_response(&ids[2]).await.unwrap();
        let mut fifth = StoredResponse::new(None);
        fifth.input = json!(5);
        let fifth = store.store_response(fifth).await.unwrap();
        assert!(store.get_response(&fourth).await.unwrap().is_some());
        assert!(store.get_response(&fifth).await.unwrap().is_some());
        assert_eq!(store.stored_record_count(), Some(2));
    }

    /// The retention drops the records older than it, at the time it is
    /// checked against; the newer ones stay.
    #[tokio::test]
    async fn the_retention_evicts_the_records_older_than_it() {
        let config = MemoryConfig {
            max_responses: 1_000,
            max_items: 1_000,
            retention_secs: Some(60),
        };
        let responses = MemoryResponseStorage::with_config(&config);
        let old = responses
            .store_response(StoredResponse::new(None))
            .await
            .unwrap();
        let items = MemoryConversationItemStorage::with_config(&config);
        let conv: ConversationId = "conv_retention".into();
        let old_item = items
            .create_item(make_item("message", Some("user"), json!([])))
            .await
            .unwrap();
        items
            .link_item(&conv, &old_item.id, Utc::now())
            .await
            .unwrap();
        let conversations = MemoryConversationStorage::with_config(&config);
        let old_conversation = conversations
            .create_conversation(NewConversation::default())
            .await
            .unwrap();

        // Fifty-nine seconds later nothing goes; sixty-one seconds later everything does.
        let later = Utc::now() + TimeDelta::seconds(59);
        assert_eq!(responses.evict_at(later), 0);
        assert_eq!(items.evict_at(later), 0);
        assert_eq!(conversations.evict_at(later), 0);
        let later = Utc::now() + TimeDelta::seconds(61);
        assert_eq!(responses.evict_at(later), 1);
        assert_eq!(items.evict_at(later), 1);
        assert_eq!(conversations.evict_at(later), 1);
        assert!(responses.get_response(&old).await.unwrap().is_none());
        assert!(items.get_item(&old_item.id).await.unwrap().is_none());
        assert!(!items.is_item_linked(&conv, &old_item.id).await.unwrap());
        assert!(conversations
            .get_conversation(&old_conversation.id)
            .await
            .unwrap()
            .is_none());
        assert_eq!(responses.stored_record_count(), Some(0));
        assert_eq!(items.stored_record_count(), Some(0));
        assert_eq!(conversations.stored_record_count(), Some(0));
    }

    /// An evicted item leaves every conversation that listed it; the counts
    /// the gauges read follow the stores.
    #[tokio::test]
    async fn an_evicted_item_leaves_every_conversation_and_the_counts_follow() {
        let config = MemoryConfig {
            max_responses: 10,
            max_items: 2,
            retention_secs: None,
        };
        let store = MemoryConversationItemStorage::with_config(&config);
        let a: ConversationId = "conv_a".into();
        let b: ConversationId = "conv_b".into();
        let shared = store
            .create_item(make_item("message", Some("user"), json!([])))
            .await
            .unwrap();
        store.link_item(&a, &shared.id, Utc::now()).await.unwrap();
        store.link_item(&b, &shared.id, Utc::now()).await.unwrap();
        assert_eq!(store.stored_record_count(), Some(1));
        for _ in 0..2 {
            let item = store
                .create_item(make_item("message", Some("user"), json!([])))
                .await
                .unwrap();
            store.link_item(&a, &item.id, Utc::now()).await.unwrap();
        }
        assert_eq!(store.stored_record_count(), Some(2));
        assert!(store.get_item(&shared.id).await.unwrap().is_none());
        assert!(!store.is_item_linked(&a, &shared.id).await.unwrap());
        assert!(!store.is_item_linked(&b, &shared.id).await.unwrap());
        let params = |order| ListParams {
            limit: 10,
            order,
            after: None,
        };
        assert_eq!(
            store
                .list_items(&a, params(SortOrder::Asc))
                .await
                .unwrap()
                .len(),
            2
        );
        assert!(store
            .list_items(&b, params(SortOrder::Asc))
            .await
            .unwrap()
            .is_empty());
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
            tenant: None,
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
            tenant: None,
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
                tenant: None,
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
                tenant: None,
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
                tenant: None,
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
                tenant: None,
            })
            .await
            .expect("create conv1 should succeed");

        let conv2 = store
            .create_conversation(NewConversation {
                id: Some(ConversationId::from("conv_beta")),
                metadata: None,
                tenant: None,
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

    #[tokio::test]
    async fn relinking_an_item_keeps_one_link_so_it_lists_once_and_delete_removes_it() {
        let store = MemoryConversationItemStorage::new();
        let conv: ConversationId = "conv_relink".into();
        let mut ids = Vec::new();
        for id in ["msg_a", "msg_b"] {
            let mut item = make_item("message", Some("user"), json!([]));
            item.id = Some(ConversationItemId::from(id));
            ids.push(store.create_item(item).await.unwrap().id);
        }
        let base = Utc.timestamp_opt(1_700_000_000, 0).single().unwrap();
        let at = |seconds: i64| base + TimeDelta::seconds(seconds);
        store.link_item(&conv, &ids[0], at(0)).await.unwrap();
        store.link_item(&conv, &ids[1], at(1)).await.unwrap();

        // The first item linked again, later, through both link paths: it moves to the new
        // stamp instead of being listed under the old one as well.
        store.link_item(&conv, &ids[0], at(2)).await.unwrap();
        store
            .link_items(&conv, &[(ids[0].clone(), at(3))])
            .await
            .unwrap();

        let list = |order: SortOrder| {
            let store = store.clone();
            let conv = conv.clone();
            let params = ListParams {
                limit: 10,
                order,
                after: None,
            };
            async move {
                store
                    .list_items(&conv, params)
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|item| item.id.0)
                    .collect::<Vec<_>>()
            }
        };
        assert_eq!(list(SortOrder::Asc).await, ["msg_b", "msg_a"]);
        assert_eq!(list(SortOrder::Desc).await, ["msg_a", "msg_b"]);

        // One link per item: the delete takes the item out of the conversation entirely.
        store.delete_item(&conv, &ids[0]).await.unwrap();
        assert!(!store.is_item_linked(&conv, &ids[0]).await.unwrap());
        assert_eq!(list(SortOrder::Asc).await, ["msg_b"]);
    }

    /// An id stored again after a delete is a new record: the entry its
    /// predecessor left in the order stands for nothing, so it drops no one.
    #[tokio::test]
    async fn a_reused_id_is_not_evicted_by_the_entry_its_predecessor_left() {
        let config = MemoryConfig {
            max_responses: 2,
            max_items: 1,
            retention_secs: None,
        };
        let store = MemoryResponseStorage::with_config(&config);
        let a: ResponseId = "resp_a".into();
        let b: ResponseId = "resp_b".into();
        let mut first = StoredResponse::new(None);
        first.id = a.clone();
        store.store_response(first).await.unwrap();
        let mut second = StoredResponse::new(None);
        second.id = b.clone();
        store.store_response(second).await.unwrap();
        store.delete_response(&a).await.unwrap();
        let mut again = StoredResponse::new(None);
        again.id = a.clone();
        again.input = json!("again");
        store.store_response(again).await.unwrap();
        // Over the bound: the oldest live record (b) goes, not the re-stored a
        // behind the stale entry of its predecessor.
        let third = store
            .store_response(StoredResponse::new(None))
            .await
            .unwrap();
        assert!(
            store.get_response(&b).await.unwrap().is_none(),
            "b was the oldest"
        );
        assert_eq!(
            store.get_response(&a).await.unwrap().map(|r| r.input),
            Some(json!("again"))
        );
        assert!(store.get_response(&third).await.unwrap().is_some());
        assert_eq!(store.stored_record_count(), Some(2));
    }

    /// A background response still running is passed over by the count bound
    /// (its client is still to read it); once terminal it goes like any other.
    #[tokio::test]
    async fn a_running_background_response_is_not_dropped_by_the_count_bound() {
        let config = MemoryConfig {
            max_responses: 2,
            max_items: 1,
            retention_secs: None,
        };
        let store = MemoryResponseStorage::with_config(&config);
        let running: ResponseId = "resp_bg".into();
        let mut queued = StoredResponse::new(None);
        queued.id = running.clone();
        queued.raw_response = json!({"id": "resp_bg", "status": "queued"});
        store.store_response(queued).await.unwrap();
        let mut done = Vec::new();
        for i in 0..3 {
            let mut response = StoredResponse::new(None);
            response.raw_response = json!({"status": "completed", "n": i});
            done.push(store.store_response(response).await.unwrap());
        }
        assert!(
            store.get_response(&running).await.unwrap().is_some(),
            "the queued record outlived three stores past the bound"
        );
        assert!(store.get_response(&done[0]).await.unwrap().is_none());
        assert!(store.get_response(&done[1]).await.unwrap().is_none());
        assert!(store.get_response(&done[2]).await.unwrap().is_some());
        assert_eq!(store.stored_record_count(), Some(2));
        // Terminal now: the next write past the bound drops it (it is the oldest).
        let mut completed = StoredResponse::new(None);
        completed.id = running.clone();
        completed.raw_response = json!({"id": "resp_bg", "status": "completed"});
        store.store_response(completed).await.unwrap();
        store
            .store_response(StoredResponse::new(None))
            .await
            .unwrap();
        assert!(store.get_response(&running).await.unwrap().is_none());
        assert_eq!(store.stored_record_count(), Some(2));
        // A store of running responses only: the pass ends, the bound yields.
        let store = MemoryResponseStorage::with_config(&config);
        for i in 0..4 {
            let mut response = StoredResponse::new(None);
            response.raw_response = json!({"status": "in_progress", "n": i});
            store.store_response(response).await.unwrap();
        }
        assert_eq!(store.stored_record_count(), Some(4));
    }

    /// Deletes leave entries behind in the order; they are compacted away, so
    /// a store-then-delete workload under no retention does not grow it.
    #[tokio::test]
    async fn the_order_is_compacted_when_deletes_leave_entries_behind() {
        let config = MemoryConfig {
            max_responses: 10_000,
            max_items: 1,
            retention_secs: None,
        };
        let store = MemoryResponseStorage::with_config(&config);
        for _ in 0..1_000 {
            let id = store
                .store_response(StoredResponse::new(None))
                .await
                .unwrap();
            store.delete_response(&id).await.unwrap();
        }
        assert_eq!(store.stored_record_count(), Some(0));
        assert!(
            store.order_len() <= 128,
            "{} entries left",
            store.order_len()
        );
        let conversations = MemoryConversationStorage::with_config(&config);
        for _ in 0..1_000 {
            let conversation = conversations
                .create_conversation(NewConversation::default())
                .await
                .unwrap();
            conversations
                .delete_conversation(&conversation.id)
                .await
                .unwrap();
        }
        assert_eq!(conversations.stored_record_count(), Some(0));
        assert!(conversations.inner.read().order.queue.len() <= 128);
    }

    /// A conversation's retention counts from its last use: a read or an
    /// update moves its stamp, so one in use outlives one created with it.
    #[tokio::test]
    async fn a_conversation_in_use_outlives_the_retention_counted_from_its_creation() {
        let config = MemoryConfig {
            max_responses: 10,
            max_items: 10,
            retention_secs: Some(3_600),
        };
        let conversations = MemoryConversationStorage::with_config(&config);
        let idle = conversations
            .create_conversation(NewConversation::default())
            .await
            .unwrap();
        let used = conversations
            .create_conversation(NewConversation::default())
            .await
            .unwrap();
        let created = conversations.stamp_of(&used.id).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert!(conversations
            .get_conversation(&used.id)
            .await
            .unwrap()
            .is_some());
        let read = conversations.stamp_of(&used.id).unwrap();
        assert!(read > created, "the read moved the stamp");
        std::thread::sleep(std::time::Duration::from_millis(5));
        conversations
            .update_conversation(&used.id, None)
            .await
            .unwrap();
        let updated = conversations.stamp_of(&used.id).unwrap();
        assert!(updated > read, "the update moved the stamp");
        // An hour after the used one's creation the idle one (created before
        // it) goes; the used one stays until an hour after its last use.
        assert_eq!(
            conversations.evict_at(created + TimeDelta::seconds(3_600)),
            1
        );
        assert!(conversations
            .get_conversation(&idle.id)
            .await
            .unwrap()
            .is_none());
        assert!(conversations
            .get_conversation(&used.id)
            .await
            .unwrap()
            .is_some());
        let last = conversations.stamp_of(&used.id).unwrap();
        assert_eq!(conversations.evict_at(last + TimeDelta::seconds(3_600)), 1);
        assert_eq!(conversations.stored_record_count(), Some(0));
    }

    /// A read or an update of a conversation moves its entry and leaves one
    /// behind; those are compacted away too, so a read-heavy workload under
    /// no retention does not grow the order.
    #[tokio::test]
    async fn the_order_is_compacted_when_touches_leave_entries_behind() {
        let config = MemoryConfig {
            max_responses: 10_000,
            max_items: 1,
            retention_secs: None,
        };
        let conversations = MemoryConversationStorage::with_config(&config);
        let conversation = conversations
            .create_conversation(NewConversation::default())
            .await
            .unwrap();
        for _ in 0..1_000 {
            assert!(conversations
                .get_conversation(&conversation.id)
                .await
                .unwrap()
                .is_some());
        }
        assert_eq!(conversations.stored_record_count(), Some(1));
        assert!(
            conversations.inner.read().order.queue.len() <= 128,
            "{} entries left",
            conversations.inner.read().order.queue.len()
        );
    }

    /// A running background response the count bound passes over keeps the
    /// stamp it was stored with, so the retention drops it when that passes,
    /// not a retention after the last pass that kept it.
    #[tokio::test]
    async fn a_running_response_passed_over_keeps_its_stamp_for_the_retention() {
        let config = MemoryConfig {
            max_responses: 1,
            max_items: 1,
            retention_secs: Some(3_600),
        };
        let store = MemoryResponseStorage::with_config(&config);
        let stored_at = Utc::now();
        for id in ["resp_bg_1", "resp_bg_2"] {
            let mut queued = StoredResponse::new(None);
            queued.id = id.into();
            queued.raw_response = json!({"id": id, "status": "queued"});
            store.store_response(queued).await.unwrap();
        }
        // Over the bound and both running: a pass half-way through the
        // retention keeps both.
        assert_eq!(store.evict_at(stored_at + TimeDelta::seconds(1_800)), 0);
        assert_eq!(store.stored_record_count(), Some(2));
        // The retention counted from the store drops them, running or not.
        assert_eq!(store.evict_at(stored_at + TimeDelta::seconds(3_601)), 2);
        assert_eq!(store.stored_record_count(), Some(0));
    }

    /// The unstored-response sweep removes through the one remover: the order
    /// holds no live entry for a swept record, the entries left behind are
    /// compacted away, and an emptied identifier index entry goes with it.
    #[tokio::test]
    async fn the_unstored_sweep_leaves_no_order_entry_behind() {
        let config = MemoryConfig {
            max_responses: 10_000,
            max_items: 1,
            retention_secs: None,
        };
        let store = MemoryResponseStorage::with_config(&config);
        let cutoff = Utc::now();
        let past = cutoff.timestamp() - 60;
        for i in 0..1_000 {
            let mut response = StoredResponse::new(None);
            response.safety_identifier = Some(format!("user-{}", i % 10));
            response.raw_response =
                json!({"status": "completed", "store": false, "completed_at": past});
            store.store_response(response).await.unwrap();
        }
        assert_eq!(
            store.delete_unstored_older_than(cutoff).await.unwrap(),
            1_000
        );
        assert_eq!(store.stored_record_count(), Some(0));
        assert_eq!(store.stats().identifier_count, 0);
        assert!(store.store.read().order.live.is_empty());
        assert!(
            store.order_len() <= 128,
            "{} entries left",
            store.order_len()
        );
    }
}
