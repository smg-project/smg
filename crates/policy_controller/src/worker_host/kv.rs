//! The worker's KV port over the store's index. The adapter decodes one
//! batch at a time into the store's events, in the batch's order.
//!
//! # What the adapter keeps
//!
//! A worker names a cache group's kind on the group's stores and leaves it
//! off the group's other events. Which events are the index's is therefore
//! decided with what the row's stream showed before: whether a main
//! attention group was shown, which groups were shown as another kind, and
//! which groups are the main signal of a worker that has shown no main
//! group. The adapter keeps those three a row until the worker leaves.
//!
//! They are the admission rule of the gateway's index, word for word and
//! under its names, with the order it applies a batch in: what the batch's
//! stores show is read first, and the stores of groups other than main
//! attention are decoded last. This is a second body of that rule, and the
//! ingestion is to move below both hosts, not to be copied: the change that
//! moves the gateway's body deletes this one, and the gateway's cases of
//! the rule stand here under their names until then. No other rule with a
//! state is copied.
//!
//! # Limits
//!
//! - A block with copies on both indexed tiers leaves the index at its first
//!   qualifying removal.
//! - A store narrower than the worker's block size of record is indexed.
//! - The events of a cache group other than main attention feed no view of
//!   the group: they are passed over, unless the group is the main signal.
//! - No block size is learned from a store: a model's size is the one its
//!   workers name.
//! - A row above rank 0 of a data-parallel worker opens no feed, so its rank
//!   is not indexed: a stream is its engine's and resumes by rank 0.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use kv_index::salt::{content_hash_with_seed, namespace_seed};
use parking_lot::Mutex;
use smg_worker::{
    common_proto::{
        kv_cache_event::Data, KvBlock, KvBlocksStored, KvCacheLocality, KvCacheTier, KvEventBatch,
    },
    identity::basic::Worker,
    kv_events::{
        port::{BatchFacts, BatchOutcome, KvCursor, KvEventSink, ResyncReason},
        KvEventFeed, KvEventHost,
    },
    registry::Generation,
};

use super::Objects;
use crate::{
    store::{self, KvEvent, SequenceHash, StoredBlock},
    view::WorkerId,
};

/// The worker crate's KV event port over a store's KV event sink.
pub struct KvAdapter<S: ?Sized> {
    /// What each row's stream showed of its cache groups.
    rows: Mutex<HashMap<WorkerId, GroupsShown>>,
    store: Arc<S>,
}

impl<S: ?Sized> KvAdapter<S> {
    /// An adapter that writes `store`.
    pub fn new(store: Arc<S>) -> Self {
        Self {
            rows: Mutex::new(HashMap::new()),
            store,
        }
    }
}

impl<S> KvEventSink for KvAdapter<S>
where
    S: store::KvEventSink<Batch = [KvEvent]> + ?Sized + 'static,
{
    type Batch = KvEventBatch;
    type Id = WorkerId;

    fn admit(
        &self,
        id: WorkerId,
        rank: u16,
        facts: BatchFacts,
        batch: &KvEventBatch,
    ) -> BatchOutcome {
        // The batch is decoded against a copy of what the row showed. The
        // copy stands only where the store took the batch: one it skipped
        // shows nothing.
        let shown = self.rows.lock().get(&id).cloned().unwrap_or_default();
        let mut groups = shown.clone();
        let (restarted, mut events) = events(batch, &mut groups);
        // The store reads a clear among a batch's events as the header's
        // fact: the engine's own clear, which makes a batch at or under the
        // row's cursor a restarted publisher. The header counts an unowned
        // clear alone. So where the header counts none, a clear reaches the
        // store only above the cursor.
        let cleared = |event: &KvEvent| matches!(event, KvEvent::Cleared);
        if restarted || (!facts.clears && events.iter().any(cleared)) {
            // One stream writes a row's rank, so the cursor read here is the
            // one the batch meets.
            let late = facts.snapshot.is_none()
                && self
                    .store
                    .cursor(id, rank)
                    .is_some_and(|cursor| cursor.applied > 0 && facts.sequence <= cursor.applied);
            if !late {
                if restarted {
                    // The main signal's blocks go ahead of the batch's own
                    // events.
                    events.insert(0, KvEvent::Cleared);
                }
            } else if !facts.clears {
                // The store skips the batch or empties the rank itself, and
                // what stands ahead of a batch's last clear is gone by it:
                // the events behind that clear are the batch's.
                if let Some(last) = events.iter().rposition(cleared) {
                    events.drain(..=last);
                }
            }
        }
        let facts = store::BatchFacts {
            sequence: facts.sequence,
            clears: facts.clears,
            snapshot: facts.snapshot.map(|chunk| store::SnapshotChunk {
                unknown_before: chunk.unknown_before,
            }),
        };
        match self.store.admit(id, rank, facts, &events) {
            store::BatchOutcome::Applied => {
                if groups != shown {
                    self.rows.lock().insert(id, groups);
                }
                BatchOutcome::Applied
            }
            store::BatchOutcome::Skipped => BatchOutcome::Skipped,
            store::BatchOutcome::Gap { expected, received } => {
                BatchOutcome::Gap { expected, received }
            }
        }
    }

    fn reconnected(&self, id: WorkerId) {
        self.store.reconnected(id);
    }

    fn resync(&self, id: WorkerId, rank: Option<u16>, reason: ResyncReason) {
        let reason = match reason {
            ResyncReason::OutOfRange => store::ResyncReason::OutOfRange,
            ResyncReason::DataLoss => store::ResyncReason::DataLoss,
            ResyncReason::SnapshotAbandoned => store::ResyncReason::SnapshotAbandoned,
            ResyncReason::PublisherRestart => store::ResyncReason::PublisherRestart,
            ResyncReason::GapCleared => store::ResyncReason::GapCleared,
            ResyncReason::Snapshot => store::ResyncReason::Snapshot,
        };
        self.store.resync(id, rank, reason);
    }

    fn cursor(&self, id: WorkerId, rank: u16) -> Option<KvCursor> {
        self.store.cursor(id, rank).map(|cursor| KvCursor {
            applied: cursor.applied,
            replay_pending: cursor.replay_pending,
            degraded: cursor.degraded,
        })
    }

    fn removed(&self, id: WorkerId) {
        self.rows.lock().remove(&id);
        self.store.removed(id);
    }

    fn block_size(&self, model: &str) -> Option<u32> {
        self.store.block_size(model)
    }

    fn set_block_size(&self, model: &str, size: u32) {
        self.store.set_block_size(model, size);
    }
}

/// What a row's stream showed of its cache groups: the three facts the
/// gateway's index keeps a worker to say where an event goes, under the
/// gateway's names, and no more of its state.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct GroupsShown {
    /// The groups a store named with a kind other than main attention.
    non_main_groups: Vec<u32>,
    /// Whether a store named a main attention kind, or a plain store came.
    main_shown: bool,
    /// The groups whose stores with tokens are the worker's main signal: a
    /// worker that has shown no main group.
    main_signal_groups: Vec<u32>,
}

/// Where an event goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Admission {
    /// The prefix index's: a main attention store, a plain worker's event,
    /// or the main signal of a worker without a main group.
    Main,
    /// A group's of another kind: the index holds no view of it here.
    Group,
    /// A removal of a group that is the worker's main signal and is known
    /// as another kind too: the index lets go.
    MainAndGroup,
    /// Not applied.
    Skip,
}

impl GroupsShown {
    /// Where an event with these attributes goes. A store names its group's
    /// kind; later events for that group may omit it, so the groups shown
    /// decide for them, and an unlearned group without a kind is the main
    /// group (an engine with one group names nothing). A store of a group
    /// other than main attention that carries tokens, on a worker that has
    /// shown no main group, is the worker's main signal: the index's, and
    /// its group's kind-less removals follow it there.
    fn admit(
        &mut self,
        kind: Option<&str>,
        group_idx: Option<u32>,
        tokenful: bool,
        locality: Option<i32>,
        ownership: Option<&str>,
    ) -> Admission {
        if foreign_owner(ownership) || remote(locality) {
            return Admission::Skip;
        }
        match (kind, group_idx) {
            (Some(kind), group) if MAIN_ATTENTION_KINDS.contains(&kind) => {
                self.main_shown = true;
                if let Some(group) = group {
                    self.non_main_groups.retain(|&known| known != group);
                }
                Admission::Main
            }
            (Some(_), Some(group)) if tokenful && !self.main_shown => {
                self.main_signal(group);
                Admission::Main
            }
            (Some(_), Some(group)) => {
                self.show_group(group);
                Admission::Group
            }
            (Some(_), None) => Admission::Skip,
            (None, Some(group)) => match (self.is_main_signal(group), self.knows_group(group)) {
                (true, true) => Admission::MainAndGroup,
                (false, true) => Admission::Group,
                _ => Admission::Main,
            },
            (None, None) => Admission::Main,
        }
    }

    /// Whether the worker has shown `group` as other than main attention.
    fn knows_group(&self, group: u32) -> bool {
        self.non_main_groups.contains(&group)
    }

    /// A store named `group` with a kind other than main attention.
    fn show_group(&mut self, group: u32) {
        if !self.non_main_groups.contains(&group) {
            self.non_main_groups.push(group);
        }
    }

    /// Whether tokenful stores of `group` were the worker's main signal.
    fn is_main_signal(&self, group: u32) -> bool {
        self.main_signal_groups.contains(&group)
    }

    /// A tokenful store of `group` on a worker that has shown no main group.
    fn main_signal(&mut self, group: u32) {
        if !self.main_signal_groups.contains(&group) {
            self.main_signal_groups.push(group);
        }
    }

    /// Whether a store belongs to a cache group other than main attention
    /// ([`Self::admit`] without a write): the stores a batch decodes last.
    fn is_group_store(&self, stored: &KvBlocksStored) -> bool {
        match (stored.kv_cache_spec_kind.as_deref(), stored.group_idx) {
            (Some(kind), Some(_)) => {
                !MAIN_ATTENTION_KINDS.contains(&kind) && (self.main_shown || !tokenful(stored))
            }
            (None, Some(group)) => self.knows_group(group) && !self.is_main_signal(group),
            _ => false,
        }
    }

    /// A main group shows. The first on a worker whose main signal was a
    /// window group restarts it: `true`, and the index lets the signal's
    /// blocks go, the main group's stores rebuild it, and the signal's
    /// groups are groups of another kind from here.
    fn main_group_shown(&mut self) -> bool {
        if self.main_shown {
            return false;
        }
        self.main_shown = true;
        if self.main_signal_groups.is_empty() {
            return false;
        }
        for group in std::mem::take(&mut self.main_signal_groups) {
            self.show_group(group);
        }
        true
    }
}

/// Decode the events the store can index against what the row's stream
/// showed of its groups, as the gateway's index applies a batch: what the
/// batch's stores show is read ahead of its events, and the stores of groups
/// other than main attention come last. `true` with the events when the
/// batch restarts the worker: the index is to let the main signal's blocks
/// go ahead of them.
fn events(batch: &KvEventBatch, groups: &mut GroupsShown) -> (bool, Vec<KvEvent>) {
    let stores = || {
        batch
            .events
            .iter()
            .filter_map(|event| match event.data.as_ref()? {
                // A store the index skips shows no group here either.
                Data::Stored(stored) if !foreign(stored) => Some(stored),
                Data::Stored(_) | Data::Removed(_) | Data::Cleared(_) => None,
            })
    };
    // The main group first, so a window group's store ahead of it in the
    // batch reads as a group's, not as the worker's main signal.
    let mut restarted = false;
    for stored in stores() {
        let main = match (stored.kv_cache_spec_kind.as_deref(), stored.group_idx) {
            (Some(kind), _) => MAIN_ATTENTION_KINDS.contains(&kind),
            (None, None) => true,
            (None, Some(_)) => false,
        };
        if main && groups.main_group_shown() {
            restarted = true;
        }
    }
    for stored in stores() {
        if let (Some(kind), Some(group)) = (stored.kv_cache_spec_kind.as_deref(), stored.group_idx)
        {
            if MAIN_ATTENTION_KINDS.contains(&kind) {
                continue;
            }
            if groups.main_shown || !tokenful(stored) {
                groups.show_group(group);
            } else {
                groups.main_signal(group);
            }
        }
    }
    let mut events = Vec::with_capacity(batch.events.len() + 1);
    let mut deferred = Vec::new();
    for data in batch.events.iter().filter_map(|event| event.data.as_ref()) {
        if matches!(data, Data::Stored(stored) if groups.is_group_store(stored)) {
            deferred.push(data);
            continue;
        }
        if let Some(event) = decode(data, groups) {
            events.push(event);
        }
    }
    for data in deferred {
        if let Some(event) = decode(data, groups) {
            events.push(event);
        }
    }
    (restarted, events)
}

/// One wire event as the store's, `None` for one that is not the index's.
fn decode(data: &Data, groups: &mut GroupsShown) -> Option<KvEvent> {
    match data {
        Data::Stored(stored) => {
            let tokens = tokenful(stored);
            let admission = groups.admit(
                stored.kv_cache_spec_kind.as_deref(),
                stored.group_idx,
                tokens,
                stored.locality,
                stored.ownership.as_deref(),
            );
            // A kind-less store of a group that is the main signal is the
            // index's, as its tokenful stores are.
            let indexed = matches!(admission, Admission::Main | Admission::MainAndGroup);
            if !(indexed && tokens && stored_tier(stored)) {
                return None;
            }
            let seed = namespace_seed(stored.lora_name.as_deref(), stored.cache_salt.as_deref());
            Some(KvEvent::Stored {
                parent: stored.parent_block_hash.map(SequenceHash::from),
                blocks: stored
                    .blocks
                    .iter()
                    .map(|block| convert_kv_block(block, seed))
                    .collect(),
            })
        }
        Data::Removed(removed) => {
            let admission = groups.admit(
                None,
                removed.group_idx,
                false,
                removed.locality,
                removed.ownership.as_deref(),
            );
            let indexed = matches!(admission, Admission::Main | Admission::MainAndGroup);
            if !(indexed && indexed_tier(removed.tier, removed.cache_level)) {
                return None;
            }
            Some(KvEvent::Removed(
                removed
                    .block_hashes
                    .iter()
                    .copied()
                    .map(SequenceHash::from)
                    .collect(),
            ))
        }
        Data::Cleared(cleared) => {
            let admission = groups.admit(None, None, false, None, cleared.ownership.as_deref());
            (admission == Admission::Main).then_some(KvEvent::Cleared)
        }
    }
}

/// Whether a residency agent owns the blocks.
fn foreign_owner(ownership: Option<&str>) -> bool {
    ownership.is_some_and(|owner| owner.eq_ignore_ascii_case("kvcr"))
}

/// Whether the blocks live on a remote tier.
fn remote(locality: Option<i32>) -> bool {
    locality == Some(KvCacheLocality::Remote as i32)
}

/// Whether the admission skips a store before it reads the store's group:
/// the batch's first passes skip it too, so a foreign store shows neither a
/// main group nor a window group.
fn foreign(stored: &KvBlocksStored) -> bool {
    foreign_owner(stored.ownership.as_deref()) || remote(stored.locality)
}

/// Whether the named tier is indexed, falling back to the older cache level.
fn indexed_tier(tier: Option<i32>, cache_level: Option<i32>) -> bool {
    match tier.and_then(|tier| KvCacheTier::try_from(tier).ok()) {
        Some(KvCacheTier::Device | KvCacheTier::Host) => true,
        Some(KvCacheTier::Disk | KvCacheTier::External) => false,
        Some(KvCacheTier::Unspecified) | None => matches!(cache_level, None | Some(0 | 1)),
    }
}

/// Whether the store's tier is indexed, using its first block for the cache level.
fn stored_tier(stored: &KvBlocksStored) -> bool {
    let first_level = stored.blocks.first().and_then(|block| block.cache_level);
    indexed_tier(stored.tier, first_level)
}

/// The kinds whose blocks feed the main attention index.
const MAIN_ATTENTION_KINDS: [&str; 3] = ["full_attention", "mla_attention", "sink_full_attention"];

/// Whether the store has blocks and carries the tokens of every block.
fn tokenful(stored: &KvBlocksStored) -> bool {
    !stored.blocks.is_empty()
        && stored
            .blocks
            .iter()
            .all(|block| !block.token_ids.is_empty())
}

/// Keep the block's sequence hash and hash its tokens under the namespace seed.
fn convert_kv_block(block: &KvBlock, seed: u64) -> StoredBlock {
    StoredBlock {
        seq_hash: SequenceHash::from(block.block_hash),
        content_hash: content_hash_with_seed(&block.token_ids, seed),
    }
}

/// Why a subscription got no feed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoFeed {
    /// The holder keeps no row for the worker's object: it laid none, or it
    /// has laid another object in its place.
    NoRow,
    /// The worker's row is a rank above 0 of a data-parallel worker.
    UpperRank,
}

/// How many subscriptions a host gave no feed, by the reason.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Unfed {
    /// The holder kept no row for the worker's object.
    pub no_row: u64,
    /// The worker's row was a rank above 0.
    pub upper_rank: u64,
}

/// The KV collector's host over the store an events subscription feeds.
pub struct KvHost<G, S: ?Sized> {
    objects: Arc<Objects<G>>,
    sink: Arc<KvAdapter<S>>,
    no_row: AtomicU64,
    upper_rank: AtomicU64,
}

impl<G, S: ?Sized> KvHost<G, S> {
    /// A host whose subscriptions feed `store`, each under the row
    /// `objects` holds for its worker.
    pub fn new(objects: Arc<Objects<G>>, store: Arc<S>) -> Self {
        Self {
            objects,
            sink: Arc::new(KvAdapter::new(store)),
            no_row: AtomicU64::new(0),
            upper_rank: AtomicU64::new(0),
        }
    }

    /// How many subscriptions got no feed so far, by the reason.
    pub fn unfed(&self) -> Unfed {
        Unfed {
            no_row: self.no_row.load(Ordering::Relaxed),
            upper_rank: self.upper_rank.load(Ordering::Relaxed),
        }
    }
}

impl<G, S> KvEventHost for KvHost<G, S>
where
    G: Generation,
    S: store::KvEventSink<Batch = [KvEvent]> + ?Sized + 'static,
{
    type Sink = KvAdapter<S>;
    type Ticket = Result<WorkerId, NoFeed>;

    fn enroll(&self, worker: &Arc<dyn Worker>, model: &str) -> Self::Ticket {
        // A size the store's width cannot hold is not a size of record.
        if let Some(size) = worker
            .metadata()
            .spec
            .kv_block_size
            .and_then(|size| u32::try_from(size).ok())
            .filter(|&size| size > 0)
        {
            self.sink.set_block_size(model, size);
        }
        // A stream is its engine's and resumes by rank 0: a feed that is one
        // row cannot hold a rank above it to that.
        if worker.dp_rank().is_some_and(|rank| rank > 0) {
            return Err(NoFeed::UpperRank);
        }
        let id = self
            .objects
            .registry()
            .get_id_by_url(worker.url())
            .ok_or(NoFeed::NoRow)?;
        // The row this very object stands for: one the holder has laid
        // another object in the place of has none.
        self.objects
            .standing_of(&id, worker)
            .map(|standing| standing.row)
            .ok_or(NoFeed::NoRow)
    }

    /// The feed names the worker's row. A subscription with no feed is
    /// counted and said here, by its reason.
    fn open(
        &self,
        ticket: Self::Ticket,
        worker: &Arc<dyn Worker>,
        _model: &str,
    ) -> Option<KvEventFeed<Self::Sink>> {
        match ticket {
            Ok(row) => Some((Arc::clone(&self.sink), row)),
            Err(NoFeed::NoRow) => {
                self.no_row.fetch_add(1, Ordering::Relaxed);
                tracing::warn!(
                    worker_url = %worker.redacted_url(),
                    "no KV event feed: the holder keeps no row for the worker's object"
                );
                None
            }
            Err(NoFeed::UpperRank) => {
                self.upper_rank.fetch_add(1, Ordering::Relaxed);
                tracing::debug!(
                    worker_url = %worker.redacted_url(),
                    "no KV event feed for a rank above 0: its rank is not indexed"
                );
                None
            }
        }
    }

    fn model_left(&self, model: &str) {
        self.sink.store.forget_block_size(model);
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use smg_worker::{
        common_proto::{KvBlocksRemoved, KvCacheCleared, KvCacheEvent},
        identity::builder::BasicWorkerBuilder,
        kv_events::port::SnapshotChunk,
        registry::WorkerRegistry,
    };

    use super::*;
    use crate::{
        store::{
            Ask, GatewayId, KvEventSink as _, MemoryStore, Named, Naming, Signals, ViewBuffers,
            WorkerFacts, WorkersStore,
        },
        view::Role,
        worker_host::Members,
    };

    const MODEL: &str = "m";
    const URL: &str = "http://a:8000";
    const RANK: u16 = 1;
    const BLOCK_SIZE: u32 = 32;

    type Store = dyn store::KvEventSink<Batch = [KvEvent]>;

    struct Fixture {
        store: Arc<MemoryStore>,
        sink: Arc<KvAdapter<Store>>,
        row: WorkerId,
    }

    impl Fixture {
        /// A row at rank 1 of a worker of two: the adapter's cases name the
        /// rank a batch is forwarded under.
        fn new() -> Self {
            Self::at(RANK, RANK + 1, Some(BLOCK_SIZE))
        }

        /// A worker of one rank: the row a host gives a feed.
        fn single(kv_block_size: Option<u32>) -> Self {
            Self::at(0, 1, kv_block_size)
        }

        fn at(dp_rank: u16, dp_size: u16, kv_block_size: Option<u32>) -> Self {
            let store = Arc::new(MemoryStore::new());
            let row = store
                .upsert(WorkerFacts {
                    key: URL.into(),
                    engine: URL.into(),
                    incarnation: 1,
                    models: vec![MODEL.into()],
                    dp_rank,
                    dp_size,
                    kv_block_size,
                    signals: Signals {
                        events: true,
                        ..Signals::default()
                    },
                    ..WorkerFacts::default()
                })
                .id();
            let sink = Arc::new(KvAdapter::new(Arc::clone(&store) as Arc<Store>));
            Self { store, sink, row }
        }

        fn applied(&self, sequence: u64, batch: &KvEventBatch) {
            assert_eq!(
                self.sink.admit(self.row, RANK, facts(sequence), batch),
                BatchOutcome::Applied
            );
        }

        fn blocks(&self) -> usize {
            self.store.indexed_blocks(self.row, RANK)
        }

        fn host(&self) -> KvHost<Members, Store> {
            let objects = Arc::new(Objects::new(Arc::new(WorkerRegistry::new())));
            objects.follow(self.store.as_ref());
            KvHost::new(objects, Arc::clone(&self.store) as Arc<Store>)
        }
    }

    fn facts(sequence: u64) -> BatchFacts {
        BatchFacts {
            sequence,
            ..BatchFacts::default()
        }
    }

    fn batch(data: impl IntoIterator<Item = Data>) -> KvEventBatch {
        KvEventBatch {
            events: data
                .into_iter()
                .map(|data| KvCacheEvent {
                    data: Some(data),
                    ..KvCacheEvent::default()
                })
                .collect(),
            ..KvEventBatch::default()
        }
    }

    fn block(block_hash: i64, tokens: &[u32]) -> KvBlock {
        KvBlock {
            block_hash,
            token_ids: tokens.to_vec(),
            ..KvBlock::default()
        }
    }

    fn stored(hash: i64) -> KvBlocksStored {
        KvBlocksStored {
            blocks: vec![block(hash, &[1, 2])],
            ..KvBlocksStored::default()
        }
    }

    fn removed(hashes: &[i64]) -> KvBlocksRemoved {
        KvBlocksRemoved {
            block_hashes: hashes.to_vec(),
            ..KvBlocksRemoved::default()
        }
    }

    /// The tokens of the block `hash` in a case: its own, so two root blocks
    /// are two blocks of the index.
    fn tokens_of(hash: i64) -> [u32; 2] {
        [7, u32::try_from(hash).unwrap_or_default()]
    }

    /// A main attention store of one block under a group number, as an
    /// engine that names its groups writes it.
    fn main_store(hash: i64, group: u32) -> KvBlocksStored {
        KvBlocksStored {
            blocks: vec![block(hash, &tokens_of(hash))],
            group_idx: Some(group),
            kv_cache_spec_kind: Some("full_attention".into()),
            ..KvBlocksStored::default()
        }
    }

    /// A window group's store of the blocks: with their tokens, or their
    /// hashes alone.
    fn window_store(hashes: &[i64], group: u32, tokens: bool) -> KvBlocksStored {
        KvBlocksStored {
            blocks: hashes
                .iter()
                .map(|&hash| {
                    let own = tokens_of(hash);
                    block(hash, if tokens { &own } else { &[] })
                })
                .collect(),
            group_idx: Some(group),
            kv_cache_spec_kind: Some("sliding_window".into()),
            ..KvBlocksStored::default()
        }
    }

    /// A removal that names its group and no kind, as every removal does.
    fn group_removal(hashes: &[i64], group: u32) -> KvBlocksRemoved {
        KvBlocksRemoved {
            group_idx: Some(group),
            ..removed(hashes)
        }
    }

    fn assert_store_is_skipped(event: KvBlocksStored) {
        let f = Fixture::new();
        f.applied(1, &batch([Data::Stored(event)]));
        assert_eq!(f.blocks(), 0);
    }

    #[test]
    fn a_main_store_is_indexed_under_the_namespaces_seed() {
        let f = Fixture::new();
        let parent = block(-1, &[1, 2]);
        let first = block(2, &[3, 4]);
        let second = block(3, &[5, 6]);
        let seed = namespace_seed(Some("adapter"), Some("tenant"));
        let hashes =
            [&parent, &first, &second].map(|b| content_hash_with_seed(&b.token_ids, seed).0);
        let parent_hash = parent.block_hash;
        let mut event = KvBlocksStored {
            blocks: vec![parent],
            group_idx: Some(0),
            kv_cache_spec_kind: Some("full_attention".into()),
            lora_name: Some("adapter".into()),
            cache_salt: Some("tenant".into()),
            ..KvBlocksStored::default()
        };
        f.applied(1, &batch([Data::Stored(event.clone())]));
        event.parent_block_hash = Some(parent_hash);
        event.blocks = vec![first, second];
        f.applied(2, &batch([Data::Stored(event)]));
        assert_eq!(f.blocks(), 3);

        f.store.publish_now();
        let gateway = GatewayId(0);
        let ask = Ask {
            gateway,
            now: Instant::now(),
            pinned: None,
            affinity: false,
        };
        f.store.deciding(|world| {
            let mut view = ViewBuffers::default();
            let built = world
                .build(MODEL, Role::Single, &ask, &mut view)
                .expect("the pool");
            let naming = Naming {
                gateway,
                block_hashes: &hashes,
                seed: 0,
                cap: 0,
                wires: None,
                now_micros: built.now_micros,
            };
            let mut named = Named::default();
            world.name(MODEL, Role::Single, &naming, &view.candidates, &mut named);
            assert_eq!(named.holders.len(), 1);
            let holder = &named.holders[0];
            assert_eq!(view.candidates[holder.row as usize].id, f.row);
            assert_eq!(holder.overlap_blocks, 3);
        });
    }

    #[test]
    fn a_foreign_owners_store_is_passed_over() {
        assert_store_is_skipped(KvBlocksStored {
            ownership: Some("KvCr".into()),
            ..stored(1)
        });
    }

    #[test]
    fn a_remote_store_is_passed_over() {
        assert_store_is_skipped(KvBlocksStored {
            locality: Some(KvCacheLocality::Remote as i32),
            ..stored(1)
        });
    }

    #[test]
    fn a_store_on_an_unindexed_tier_is_passed_over() {
        for (tier, level) in [
            (Some(KvCacheTier::Disk as i32), Some(0)),
            (Some(KvCacheTier::External as i32), None),
            (None, Some(2)),
            (Some(KvCacheTier::Unspecified as i32), Some(3)),
            (Some(99), Some(2)),
        ] {
            let mut event = stored(1);
            event.tier = tier;
            event.blocks[0].cache_level = level;
            assert_store_is_skipped(event);
        }
    }

    #[test]
    fn a_store_uses_the_named_tier_then_the_first_blocks_cache_level() {
        for (tier, level) in [
            (Some(KvCacheTier::Device as i32), Some(2)),
            (Some(KvCacheTier::Host as i32), Some(2)),
            (None, Some(1)),
            (Some(KvCacheTier::Unspecified as i32), Some(1)),
            (Some(99), Some(1)),
        ] {
            let f = Fixture::new();
            let mut event = stored(1);
            event.tier = tier;
            event.blocks[0].cache_level = level;
            event.blocks.push(KvBlock {
                cache_level: Some(2),
                ..block(2, &[3, 4])
            });
            f.applied(1, &batch([Data::Stored(event)]));
            assert_eq!(f.blocks(), 2, "tier {tier:?}, cache level {level:?}");
        }
    }

    #[test]
    fn each_main_attention_kind_is_indexed_with_a_group() {
        for kind in ["full_attention", "mla_attention", "sink_full_attention"] {
            let f = Fixture::new();
            let event = KvBlocksStored {
                group_idx: Some(1),
                kv_cache_spec_kind: Some(kind.into()),
                ..stored(1)
            };
            f.applied(1, &batch([Data::Stored(event)]));
            assert_eq!(f.blocks(), 1, "{kind}");
        }
    }

    #[test]
    fn a_kindless_store_of_a_group_not_shown_as_another_kind_is_the_main_groups() {
        let f = Fixture::new();
        let event = KvBlocksStored {
            group_idx: Some(0),
            ..stored(1)
        };
        f.applied(1, &batch([Data::Stored(event)]));
        assert_eq!(f.blocks(), 1);
    }

    #[test]
    fn a_kindless_store_of_a_group_shown_as_another_kind_is_passed_over() {
        let f = Fixture::new();
        f.applied(
            1,
            &batch([
                Data::Stored(main_store(1, 0)),
                Data::Stored(window_store(&[1], 1, false)),
            ]),
        );
        let event = KvBlocksStored {
            group_idx: Some(1),
            ..stored(2)
        };
        f.applied(2, &batch([Data::Stored(event)]));
        assert_eq!(f.blocks(), 1);
    }

    #[test]
    fn a_window_group_with_tokens_is_the_main_signal_until_a_main_group_shows() {
        let f = Fixture::new();
        f.applied(1, &batch([Data::Stored(window_store(&[1, 2], 1, true))]));
        assert_eq!(f.blocks(), 2);
        // Its kind-less removals follow it to the index.
        f.applied(2, &batch([Data::Removed(group_removal(&[2], 1))]));
        assert_eq!(f.blocks(), 1);
        // A main group shows: the signal's blocks go, the main group's stand.
        f.applied(3, &batch([Data::Stored(main_store(7, 0))]));
        assert_eq!(f.blocks(), 1);
        // The window group is a group of another kind from here.
        f.applied(
            4,
            &batch([
                Data::Stored(window_store(&[8], 1, true)),
                Data::Removed(group_removal(&[7], 1)),
            ]),
        );
        assert_eq!(f.blocks(), 1);
    }

    // The gateway's cases of the admission rule, under their names: what
    // each says of the prefix index. They hold both bodies of the rule
    // until the gateway's moves below both hosts.

    /// A removal that names a group without its kind is the group's where
    /// the group was shown, even ahead of the group's first store in the
    /// batch, and even for a group no view could place: the main block of
    /// the same hash stays indexed. A main store without tokens indexes
    /// nothing.
    #[test]
    fn a_kind_less_group_event_never_reaches_the_main_index() {
        let f = Fixture::new();
        f.applied(
            1,
            &batch([
                Data::Stored(stored(1)),
                Data::Removed(group_removal(&[1], 1)),
                Data::Stored(window_store(&[1], 1, false)),
            ]),
        );
        assert_eq!(f.blocks(), 1, "the main block stays");

        let mut hidden = window_store(&[2], 17, false);
        hidden.kv_cache_spec_kind = Some("mamba".into());
        let second = KvBlocksStored {
            parent_block_hash: Some(1),
            ..stored(2)
        };
        f.applied(2, &batch([Data::Stored(hidden), Data::Stored(second)]));
        assert_eq!(f.blocks(), 2);
        f.applied(3, &batch([Data::Removed(group_removal(&[2], 17))]));
        assert_eq!(f.blocks(), 2, "the main block stays");

        let bare = KvBlocksStored {
            blocks: vec![block(3, &[])],
            ..KvBlocksStored::default()
        };
        f.applied(4, &batch([Data::Stored(bare)]));
        assert_eq!(f.blocks(), 2);
    }

    /// A worker whose attention is all sliding window has no main group:
    /// its window stores carry tokens and are the prefix index's, as a plain
    /// worker's, and a kind-less removal of the group leaves the index.
    #[test]
    fn a_window_only_workers_stores_and_removals_are_the_prefix_indexs() {
        let f = Fixture::new();
        let second = KvBlocksStored {
            parent_block_hash: Some(1),
            ..window_store(&[2], 0, true)
        };
        f.applied(
            1,
            &batch([
                Data::Stored(window_store(&[1], 0, true)),
                Data::Stored(second),
            ]),
        );
        assert_eq!(f.blocks(), 2);
        f.applied(2, &batch([Data::Removed(group_removal(&[2], 0))]));
        assert_eq!(f.blocks(), 1);
    }

    /// A store the index skips (a residency agent's, a remote tier's) shows
    /// no group in a batch: a foreign main store ahead of a window-only
    /// worker's store leaves that store in the prefix index, and a foreign
    /// window store shows no group.
    #[test]
    fn a_foreign_store_in_a_batch_shows_no_group() {
        let f = Fixture::new();
        let agent_main = KvBlocksStored {
            ownership: Some("kvcr".into()),
            ..main_store(7, 1)
        };
        let remote_window = KvBlocksStored {
            locality: Some(KvCacheLocality::Remote as i32),
            ..window_store(&[8], 2, true)
        };
        f.applied(
            1,
            &batch([
                Data::Stored(agent_main),
                Data::Stored(window_store(&[1], 0, true)),
                Data::Stored(remote_window),
            ]),
        );
        assert_eq!(f.blocks(), 1, "the window store is the prefix index's");

        // The group pass's own case: a worker whose main group has shown
        // gets a foreign window store, which would show that group. Skipped,
        // the group stays unshown, and its kind-less removal is the main
        // group's.
        let f = Fixture::new();
        let remote_window = KvBlocksStored {
            locality: Some(KvCacheLocality::Remote as i32),
            ..window_store(&[9], 2, true)
        };
        f.applied(
            1,
            &batch([Data::Stored(stored(1)), Data::Stored(remote_window)]),
        );
        assert_eq!(f.blocks(), 1);
        f.applied(2, &batch([Data::Removed(group_removal(&[1], 2))]));
        assert_eq!(f.blocks(), 0);
    }

    /// A relay snapshot replays the stores after a clear in one batch: the
    /// batch reads as the stream did, and a removal after it still leaves
    /// the index.
    #[test]
    fn a_window_only_workers_snapshot_replay_reads_as_its_stream() {
        let f = Fixture::new();
        f.applied(1, &batch([Data::Stored(window_store(&[1], 0, true))]));
        let second = KvBlocksStored {
            parent_block_hash: Some(1),
            ..window_store(&[2], 0, true)
        };
        f.applied(
            2,
            &batch([
                Data::Cleared(KvCacheCleared::default()),
                Data::Stored(window_store(&[1], 0, true)),
                Data::Stored(second),
            ]),
        );
        assert_eq!(f.blocks(), 2);
        f.applied(
            3,
            &batch([
                Data::Removed(group_removal(&[1], 0)),
                Data::Removed(group_removal(&[2], 0)),
            ]),
        );
        assert_eq!(f.blocks(), 0);
    }

    /// A worker whose first batches carried only a window group's stores
    /// (its main signal) and then shows a main group: the index lets the
    /// signal's blocks go and the main group's stores rebuild it, and the
    /// window group's removals are never the main group's blocks' from then
    /// on.
    #[test]
    fn a_main_group_shown_later_moves_the_window_group_to_the_view() {
        let f = Fixture::new();
        f.applied(1, &batch([Data::Stored(window_store(&[1], 0, true))]));
        assert_eq!(f.blocks(), 1);

        // The main lane's batch of the same block: the signal's copy goes,
        // the main group's comes.
        f.applied(2, &batch([Data::Stored(main_store(1, 1))]));
        assert_eq!(f.blocks(), 1);

        // The window group's later store and its removals leave the index
        // alone: the main group's blocks stand until the main group lets
        // them go.
        f.applied(3, &batch([Data::Stored(window_store(&[1], 0, false))]));
        f.applied(4, &batch([Data::Stored(main_store(4, 1))]));
        f.applied(
            5,
            &batch([
                Data::Removed(group_removal(&[1], 0)),
                Data::Removed(group_removal(&[4], 0)),
            ]),
        );
        assert_eq!(
            f.blocks(),
            2,
            "the window group's removals leave the index alone"
        );
        f.applied(6, &batch([Data::Removed(group_removal(&[1], 1))]));
        assert_eq!(f.blocks(), 1, "the main group let it go");
    }

    /// A hybrid worker whose window stores carry tokens: the main group's
    /// store in the same batch keeps them out of the prefix index.
    #[test]
    fn a_hybrid_workers_tokenful_window_store_stays_in_the_view() {
        let f = Fixture::new();
        f.applied(
            1,
            &batch([
                Data::Stored(window_store(&[1], 1, true)),
                Data::Stored(main_store(1, 0)),
            ]),
        );
        assert_eq!(f.blocks(), 1);
        // The window group's kind-less removal is the group's.
        f.applied(2, &batch([Data::Removed(group_removal(&[1], 1))]));
        assert_eq!(f.blocks(), 1);
    }

    /// A store that names a kind other than main attention and no group is
    /// not applied, and a plain store after it is indexed.
    #[test]
    fn a_nameless_group_store_is_counted_and_a_plain_worker_has_no_view() {
        let f = Fixture::new();
        let nameless = KvBlocksStored {
            kv_cache_spec_kind: Some("sliding_window".into()),
            ..stored(1)
        };
        f.applied(1, &batch([Data::Stored(nameless)]));
        assert_eq!(f.blocks(), 0);
        f.applied(2, &batch([Data::Stored(stored(1))]));
        assert_eq!(f.blocks(), 1);
    }

    #[test]
    fn a_batchs_stores_of_other_groups_are_decoded_last() {
        let f = Fixture::new();
        f.applied(
            1,
            &batch([
                Data::Stored(main_store(1, 0)),
                Data::Stored(window_store(&[1], 1, false)),
            ]),
        );
        // Group 1 is a group of another kind, so its kind-less store waits
        // for the batch's other events. The main store behind it takes the
        // group back, and the store is the main group's when it is decoded.
        let kindless = KvBlocksStored {
            group_idx: Some(1),
            ..stored(2)
        };
        f.applied(
            2,
            &batch([Data::Stored(kindless), Data::Stored(main_store(3, 1))]),
        );
        assert_eq!(f.blocks(), 3);
    }

    #[test]
    fn a_late_batch_that_shows_a_main_group_is_no_restarted_publisher() {
        let f = Fixture::new();
        f.applied(4, &batch([Data::Stored(window_store(&[1], 0, true))]));
        f.applied(5, &batch([Data::Stored(window_store(&[2], 0, true))]));
        // A main store at a sequence the row holds would restart the worker.
        // Its clear is not the engine's own: the store skips the batch.
        let late = batch([Data::Stored(main_store(7, 1))]);
        assert_eq!(
            f.sink.admit(f.row, RANK, facts(4), &late),
            BatchOutcome::Skipped
        );
        assert_eq!(f.blocks(), 2);
        // The window group is the worker's main signal still.
        f.applied(6, &batch([Data::Stored(window_store(&[3], 0, true))]));
        assert_eq!(f.blocks(), 3);
    }

    /// A clear owned by a name the header does not count, at a sequence the
    /// row holds: the batch is late, and the store skips it as any other.
    #[test]
    fn a_late_batch_with_an_owned_clear_is_skipped_as_any_late_batch() {
        let f = Fixture::new();
        f.applied(9, &batch([Data::Stored(stored(1))]));
        let second = KvBlocksStored {
            parent_block_hash: Some(1),
            ..stored(2)
        };
        f.applied(10, &batch([Data::Stored(second)]));
        let late = batch([
            Data::Cleared(KvCacheCleared {
                ownership: Some("engine".into()),
            }),
            Data::Stored(main_store(3, 0)),
        ]);
        assert_eq!(
            f.sink.admit(f.row, RANK, facts(9), &late),
            BatchOutcome::Skipped
        );
        assert_eq!(f.blocks(), 2);
    }

    /// The first batch of a new connection under the cursor is a restarted
    /// publisher by the store's own rule. Its owned clear still wipes what
    /// stood ahead of it in the batch.
    #[test]
    fn an_owned_clear_in_a_late_batch_the_store_takes_wipes_what_stood_ahead_of_it() {
        let f = Fixture::new();
        f.applied(9, &batch([Data::Stored(stored(1))]));
        f.sink.reconnected(f.row);
        let restarted = batch([
            Data::Stored(main_store(5, 0)),
            Data::Cleared(KvCacheCleared {
                ownership: Some("engine".into()),
            }),
            Data::Stored(main_store(6, 0)),
        ]);
        f.applied(3, &restarted);
        assert_eq!(f.blocks(), 1);
    }

    #[test]
    fn a_batch_the_store_does_not_take_shows_nothing() {
        let f = Fixture::new();
        f.applied(1, &batch([Data::Stored(main_store(1, 0))]));
        // A batch at a sequence the store holds: its store shows no group.
        let late = batch([Data::Stored(window_store(&[1], 0, false))]);
        assert_eq!(
            f.sink.admit(f.row, RANK, facts(1), &late),
            BatchOutcome::Skipped
        );
        f.applied(2, &batch([Data::Removed(group_removal(&[1], 0))]));
        assert_eq!(f.blocks(), 0);
    }

    #[test]
    fn a_worker_that_left_takes_what_its_stream_showed_with_it() {
        let f = Fixture::new();
        f.applied(
            1,
            &batch([
                Data::Stored(main_store(1, 0)),
                Data::Stored(window_store(&[1], 1, false)),
            ]),
        );
        f.sink.removed(f.row);
        // Nothing is shown of group 1 now: its kind-less store is the main
        // group's.
        let event = KvBlocksStored {
            group_idx: Some(1),
            ..stored(2)
        };
        f.applied(1, &batch([Data::Stored(event)]));
        assert_eq!(f.blocks(), 1);
    }

    #[test]
    fn a_store_with_a_block_without_tokens_is_passed_over() {
        let mut event = stored(1);
        event.blocks.push(block(2, &[]));
        assert_store_is_skipped(event);
    }

    #[test]
    fn a_store_without_blocks_is_passed_over() {
        let f = Fixture::new();
        let empty = batch([Data::Stored(KvBlocksStored::default())]);
        f.applied(1, &empty);
        assert!(events(&empty, &mut GroupsShown::default()).1.is_empty());
    }

    #[test]
    fn a_removal_takes_the_blocks_out() {
        let f = Fixture::new();
        let mut event = stored(-1);
        event.blocks.push(block(2, &[3, 4]));
        f.applied(1, &batch([Data::Stored(event)]));
        assert_eq!(f.blocks(), 2);
        let removal = KvBlocksRemoved {
            tier: Some(KvCacheTier::Host as i32),
            cache_level: Some(2),
            ..removed(&[-1, 2])
        };
        f.applied(2, &batch([Data::Removed(removal)]));
        assert_eq!(f.blocks(), 0);
    }

    #[test]
    fn a_removal_of_the_main_group_is_taken_under_its_group_number() {
        let f = Fixture::new();
        f.applied(1, &batch([Data::Stored(main_store(1, 0))]));
        f.applied(2, &batch([Data::Removed(group_removal(&[1], 0))]));
        assert_eq!(f.blocks(), 0);
    }

    #[test]
    fn a_foreign_remote_or_unindexed_removal_is_passed_over() {
        let f = Fixture::new();
        f.applied(1, &batch([Data::Stored(stored(1))]));
        let options = [
            (Some("KvCr"), None, None, None),
            (None, Some(KvCacheLocality::Remote as i32), None, None),
            (None, None, Some(KvCacheTier::Disk as i32), Some(0)),
            (None, None, None, Some(2)),
        ];
        for (sequence, (owner, locality, tier, cache_level)) in (2..).zip(options) {
            let removal = KvBlocksRemoved {
                ownership: owner.map(str::to_owned),
                locality,
                tier,
                cache_level,
                ..removed(&[1])
            };
            f.applied(sequence, &batch([Data::Removed(removal)]));
            assert_eq!(f.blocks(), 1, "removal at sequence {sequence}");
        }
    }

    #[test]
    fn a_clear_empties_the_rank() {
        let f = Fixture::new();
        f.applied(1, &batch([Data::Stored(stored(1))]));
        assert_eq!(f.blocks(), 1);
        f.applied(2, &batch([Data::Cleared(KvCacheCleared::default())]));
        assert_eq!(f.blocks(), 0);
    }

    #[test]
    fn a_foreign_owners_clear_is_passed_over() {
        let f = Fixture::new();
        f.applied(1, &batch([Data::Stored(stored(1))]));
        f.applied(
            2,
            &batch([Data::Cleared(KvCacheCleared {
                ownership: Some("KvCr".into()),
            })]),
        );
        assert_eq!(f.blocks(), 1);
    }

    #[test]
    fn an_event_with_no_data_gives_none() {
        let f = Fixture::new();
        let missing = KvEventBatch {
            events: vec![KvCacheEvent::default()],
            ..KvEventBatch::default()
        };
        f.applied(1, &missing);
        assert!(events(&missing, &mut GroupsShown::default()).1.is_empty());
    }

    #[test]
    fn the_order_of_a_batch_of_three_event_kinds_is_kept() {
        let f = Fixture::new();
        f.applied(1, &batch([Data::Stored(stored(1))]));
        let ordered = batch([
            Data::Removed(removed(&[1])),
            Data::Cleared(KvCacheCleared::default()),
            Data::Stored(stored(2)),
        ]);
        f.applied(2, &ordered);
        assert_eq!(f.blocks(), 1);
        assert!(matches!(
            events(&ordered, &mut GroupsShown::default()).1.as_slice(),
            [
                KvEvent::Removed(_),
                KvEvent::Cleared,
                KvEvent::Stored { .. }
            ]
        ));
    }

    #[test]
    fn the_facts_and_the_outcome_cross_as_they_are() {
        let f = Fixture::new();
        f.applied(21, &batch([Data::Stored(stored(1))]));
        let empty = batch([]);
        f.applied(22, &empty);
        assert_eq!(
            f.sink.admit(f.row, RANK, facts(26), &empty),
            BatchOutcome::Gap {
                expected: 23,
                received: 26
            }
        );
        assert_eq!(
            f.sink.admit(f.row, RANK, facts(22), &empty),
            BatchOutcome::Skipped
        );

        let clear = BatchFacts {
            clears: true,
            ..facts(20)
        };
        assert_eq!(
            f.sink.admit(f.row, RANK, clear, &empty),
            BatchOutcome::Applied
        );
        assert_eq!(f.blocks(), 0);
        for (sequence, unknown_before) in [(70, true), (71, false)] {
            let chunk = BatchFacts {
                snapshot: Some(SnapshotChunk { unknown_before }),
                ..facts(sequence)
            };
            assert_eq!(
                f.sink.admit(f.row, RANK, chunk, &empty),
                BatchOutcome::Applied
            );
            assert_eq!(
                f.store.cursor(f.row, RANK),
                Some(store::KvCursor {
                    applied: sequence,
                    degraded: unknown_before,
                    replay_pending: false
                })
            );
        }
    }

    #[test]
    fn each_reason_reaches_the_store_under_its_own_name() {
        use store::ResyncReason as To;

        let f = Fixture::new();
        let store = batch([Data::Stored(stored(1))]);
        for (reason, expected) in [
            (ResyncReason::OutOfRange, To::OutOfRange),
            (ResyncReason::DataLoss, To::DataLoss),
            (ResyncReason::SnapshotAbandoned, To::SnapshotAbandoned),
            (ResyncReason::PublisherRestart, To::PublisherRestart),
            (ResyncReason::GapCleared, To::GapCleared),
            (ResyncReason::Snapshot, To::Snapshot),
        ] {
            f.sink.resync(f.row, Some(RANK), reason);
            assert_eq!(f.store.resyncs(expected), 1, "{reason:?}");

            f.applied(7, &store);
            f.sink.resync(f.row, Some(0), reason);
            assert_eq!(f.blocks(), 1);
            f.sink.resync(f.row, Some(RANK), reason);
            assert_eq!(f.blocks(), 0);
        }

        f.applied(7, &store);
        f.sink.resync(f.row, None, ResyncReason::OutOfRange);
        assert_eq!(f.blocks(), 0);
    }

    #[test]
    fn a_cursor_reads_as_the_stores() {
        let f = Fixture::new();
        assert_eq!(f.sink.cursor(f.row, RANK), None);
        f.store.admit(
            f.row,
            RANK,
            store::BatchFacts {
                sequence: 41,
                clears: false,
                snapshot: Some(store::SnapshotChunk {
                    unknown_before: true,
                }),
            },
            &[],
        );
        f.store.admit(
            f.row,
            RANK,
            store::BatchFacts {
                sequence: 43,
                ..Default::default()
            },
            &[],
        );
        let expected = f.store.cursor(f.row, RANK).expect("the store's cursor");
        assert!(expected.replay_pending && expected.degraded);
        assert_eq!(
            f.sink.cursor(f.row, RANK),
            Some(KvCursor {
                applied: expected.applied,
                replay_pending: expected.replay_pending,
                degraded: expected.degraded,
            })
        );
        assert_eq!(f.sink.cursor(f.row, 0), None);
    }

    #[test]
    fn a_reconnection_is_reported_to_the_store() {
        let f = Fixture::new();
        f.applied(7, &batch([Data::Stored(stored(1))]));
        f.sink.reconnected(f.row);
        f.applied(6, &batch([]));
        assert_eq!(f.blocks(), 0);
    }

    #[test]
    fn a_worker_reported_gone_loses_its_blocks_and_cursor() {
        let f = Fixture::new();
        f.applied(1, &batch([Data::Stored(stored(1))]));
        assert_eq!(f.blocks(), 1);
        f.sink.removed(f.row);
        assert_eq!(f.blocks(), 0);
        assert_eq!(f.store.cursor(f.row, RANK), None);
    }

    #[test]
    fn the_block_size_is_read_and_set_at_the_store() {
        let f = Fixture::new();
        f.store.set_block_size(MODEL, 16);
        assert_eq!(f.sink.block_size(MODEL), Some(16));
        f.sink.set_block_size(MODEL, 32);
        assert_eq!(f.store.block_size(MODEL), Some(32));
    }

    #[test]
    fn a_worker_with_a_row_is_enrolled_under_the_rows_id_and_seeds_the_block_size() {
        let f = Fixture::single(Some(BLOCK_SIZE));
        let host = f.host();
        let worker = host
            .objects
            .registry()
            .get_by_url(URL)
            .expect("the worker's object");
        let ticket = host.enroll(&worker, MODEL);
        assert_eq!(ticket, Ok(f.row));
        assert_eq!(f.store.block_size(MODEL), Some(BLOCK_SIZE));
        let (sink, row) = host.open(ticket, &worker, MODEL).expect("the row's feed");
        assert_eq!(row, f.row);
        assert!(Arc::ptr_eq(&sink, &host.sink));
        assert_eq!(host.unfed(), Unfed::default());
    }

    #[test]
    fn a_row_above_rank_0_opens_no_feed_and_is_counted() {
        let f = Fixture::new();
        let host = f.host();
        let worker = host
            .objects
            .registry()
            .get_by_url(URL)
            .expect("the worker's object");
        let ticket = host.enroll(&worker, MODEL);
        assert_eq!(ticket, Err(NoFeed::UpperRank));
        // The model's size is seeded all the same.
        assert_eq!(f.store.block_size(MODEL), Some(BLOCK_SIZE));
        assert!(host.open(ticket, &worker, MODEL).is_none());
        let unfed = Unfed {
            no_row: 0,
            upper_rank: 1,
        };
        assert_eq!(host.unfed(), unfed);
    }

    #[test]
    fn an_object_the_holder_replaced_opens_nothing() {
        let f = Fixture::single(Some(BLOCK_SIZE));
        let host = f.host();
        let registry = host.objects.registry();
        let old = registry.get_by_url(URL).expect("the worker's object");
        // The row's facts move, so the holder lays another object in the
        // place of the first.
        f.store.upsert(WorkerFacts {
            key: URL.into(),
            engine: URL.into(),
            incarnation: 1,
            models: vec![MODEL.into()],
            dp_rank: 0,
            dp_size: 1,
            kv_block_size: Some(BLOCK_SIZE * 2),
            signals: Signals {
                events: true,
                ..Signals::default()
            },
            ..WorkerFacts::default()
        });
        host.objects.follow(f.store.as_ref());
        let new = registry.get_by_url(URL).expect("the row's new object");
        assert!(!Arc::ptr_eq(&old, &new));
        assert_eq!(host.enroll(&old, MODEL), Err(NoFeed::NoRow));
        assert_eq!(host.enroll(&new, MODEL), Ok(f.row));
    }

    #[test]
    fn a_worker_the_holder_did_not_lay_opens_nothing() {
        let f = Fixture::new();
        let host = f.host();
        let worker: Arc<dyn Worker> = Arc::new(<BasicWorkerBuilder>::new("http://b:8000").build());
        assert!(host
            .objects
            .registry()
            .register(Arc::clone(&worker))
            .is_some());
        let ticket = host.enroll(&worker, MODEL);
        assert_eq!(ticket, Err(NoFeed::NoRow));
        assert!(host.open(ticket, &worker, MODEL).is_none());
        let unfed = Unfed {
            no_row: 1,
            upper_rank: 0,
        };
        assert_eq!(host.unfed(), unfed);
    }

    #[test]
    fn a_worker_with_no_positive_block_size_leaves_it_unset() {
        for size in [None, Some(0)] {
            let f = Fixture::single(size);
            let host = f.host();
            let worker = host
                .objects
                .registry()
                .get_by_url(URL)
                .expect("the worker's object");
            assert_eq!(host.enroll(&worker, MODEL), Ok(f.row));
            assert_eq!(f.store.block_size(MODEL), None);
        }
    }

    #[test]
    fn a_model_that_left_forgets_its_block_size() {
        let f = Fixture::new();
        let host = f.host();
        f.store.set_block_size(MODEL, 32);
        host.model_left(MODEL);
        assert_eq!(f.store.block_size(MODEL), None);
        f.store.set_block_size(MODEL, 16);
        assert_eq!(f.store.block_size(MODEL), Some(16));
    }
}
