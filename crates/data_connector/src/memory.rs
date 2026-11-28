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
        // A link lives and dies with its item (an eviction unlinks through the item), so none
        // is written for an item the bound has already dropped.
        if !self.items.contains_key(item_id) {
            return;
        }
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
        // An id already stored answers its item, untouched (the contract of
        // [`ConversationItemStorage::create_item`]); a replace rewrote it.
        if let Some(stored) = store.items.get(&id) {
            return Ok(stored.clone());
        }
        store.items.insert(id.clone(), item.clone());
        store.order.push(created_at, id);
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
        let Some(map) = store.links.get(conversation_id) else {
            return Ok(Vec::new());
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
#[derive(Clone)]
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

    /// The record written under its id, in place of what was there (a
    /// record replaced under its id keeps its one index entry); every write
    /// drops what is over the bounds at this time. The store trait's
    /// `store_response`.
    pub fn replace_response(&self, response: StoredResponse) -> ResponseId {
        let now = Utc::now();
        let mut store = self.store.write();
        let response_id = store.insert(response, now).id.clone();
        store.evict(&self.bounds, now);
        response_id
    }

    /// The record written under its id in place of `held` alone, the record
    /// [`Self::store_response_if_absent`] handed back: a place the bounds
    /// dropped meanwhile, and whatever took its id since, is left as it is.
    /// Whether the write happened. The streamed thread turn's final write.
    pub fn replace_response_if_held(
        &self,
        held: &Arc<StoredResponse>,
        response: StoredResponse,
    ) -> bool {
        let now = Utc::now();
        let mut store = self.store.write();
        if !store.holds(held) {
            return false;
        }
        store.insert(response, now);
        store.evict(&self.bounds, now);
        true
    }

    /// The record under `response_id` removed, when there is one.
    pub fn remove_response(&self, response_id: &ResponseId) {
        self.store.write().remove(response_id);
    }

    /// The record `held` removed while it is the one under its id: a place
    /// the bounds dropped meanwhile, and whatever took its id since, is left
    /// as it is. Whether the removal happened.
    pub fn remove_response_if_held(&self, held: &Arc<StoredResponse>) -> bool {
        let mut store = self.store.write();
        store.holds(held) && store.remove(&held.id)
    }

    /// Puts `response` under its id when nothing is stored under it, and
    /// hands back the record as stored, the identity the `_if_held` writes
    /// compare against; a taken id hands the record back. The eviction runs
    /// before the check, so absent means absent or expired, and after the
    /// write for the count bound; the check and the write are under the one
    /// lock, so two writers of one id store one record.
    pub fn store_response_if_absent(
        &self,
        response: StoredResponse,
    ) -> Result<Arc<StoredResponse>, Box<StoredResponse>> {
        self.store_response_if_absent_at(response, Utc::now())
    }

    /// [`Self::store_response_if_absent`] at `now`, the time the bounds are
    /// judged at.
    fn store_response_if_absent_at(
        &self,
        response: StoredResponse,
        now: DateTime<Utc>,
    ) -> Result<Arc<StoredResponse>, Box<StoredResponse>> {
        let mut store = self.store.write();
        store.evict(&self.bounds, now);
        if store.responses.contains_key(&response.id) {
            return Err(Box::new(response));
        }
        let stored = store.insert(response, now);
        store.evict(&self.bounds, now);
        Ok(stored)
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
    /// Whether `held` is the record under its id: the same allocation, not an
    /// equal record, so a place the bounds dropped and another record took
    /// since is not it.
    fn holds(&self, held: &Arc<StoredResponse>) -> bool {
        self.responses
            .get(&held.id)
            .is_some_and(|stored| Arc::ptr_eq(stored, held))
    }

    /// Puts `response` under its id (the index keeps one entry per id); a
    /// new id joins the order at `now`, a replaced record keeps its place.
    /// Answers the record as stored.
    fn insert(&mut self, response: StoredResponse, now: DateTime<Utc>) -> Arc<StoredResponse> {
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
        let stored = Arc::new(response);
        if self
            .responses
            .insert(response_id.clone(), stored.clone())
            .is_none()
        {
            self.order.push(now, response_id);
        }
        tracing::debug!(
            memory_store_size = self.responses.len(),
            "Response stored in memory"
        );
        stored
    }
}

#[async_trait]
impl ResponseStorage for MemoryResponseStorage {
    async fn store_response(&self, mut response: StoredResponse) -> ResponseResult<ResponseId> {
        // Generate ID if not set
        if response.id.0.is_empty() {
            response.id = ResponseId::new();
        }

        Ok(self.replace_response(response))
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
        self.remove_response(response_id);
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

    async fn get_response_chain_for(
        &self,
        response_id: &ResponseId,
        tenant: Option<&str>,
    ) -> ResponseResult<ResponseChain> {
        // One pass under one lock, as for the whole chain (one snapshot, one
        // depth limit); the cut is made on what that pass holds.
        let mut chain = self.get_response_chain(response_id, None).await?;
        if chain.responses.last().is_some_and(|head| is_unstored(head)) {
            chain.responses.clear();
        }
        chain.responses = visible_chain(chain.responses, tenant);
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
#[path = "../tests/unit/memory/tests.rs"]
mod tests;
