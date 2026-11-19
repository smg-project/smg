//! The worker's KV port over the store's index. The adapter decodes one
//! batch at a time, with at most one store event per wire event.
//!
//! # Limits
//!
//! - A block with copies on both indexed tiers leaves the index at its first
//!   qualifying removal.
//! - A store narrower than the worker's block size of record is indexed.
//! - A store that names a cache group without a kind is skipped, as is a
//!   removal of grouped main-attention blocks.

use std::sync::Arc;

use kv_index::salt::{content_hash_with_seed, namespace_seed};
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
    store: Arc<S>,
}

impl<S: ?Sized> KvAdapter<S> {
    /// An adapter that writes `store`.
    pub fn new(store: Arc<S>) -> Self {
        Self { store }
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
        let facts = store::BatchFacts {
            sequence: facts.sequence,
            clears: facts.clears,
            snapshot: facts.snapshot.map(|chunk| store::SnapshotChunk {
                unknown_before: chunk.unknown_before,
            }),
        };
        match self.store.admit(id, rank, facts, &events(batch)) {
            store::BatchOutcome::Applied => BatchOutcome::Applied,
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
        self.store.removed(id);
    }

    fn block_size(&self, model: &str) -> Option<u32> {
        self.store.block_size(model)
    }

    fn set_block_size(&self, model: &str, size: u32) {
        self.store.set_block_size(model, size);
    }
}

/// Decode the events the store can index, in batch order.
fn events(batch: &KvEventBatch) -> Vec<KvEvent> {
    batch
        .events
        .iter()
        .filter_map(|event| match event.data.as_ref()? {
            Data::Stored(stored)
                if !foreign_owner(stored.ownership.as_deref())
                    && !remote(stored.locality)
                    && stored_tier(stored)
                    && main_attention(stored)
                    && tokenful(stored) =>
            {
                let seed =
                    namespace_seed(stored.lora_name.as_deref(), stored.cache_salt.as_deref());
                Some(KvEvent::Stored {
                    parent: stored.parent_block_hash.map(SequenceHash::from),
                    blocks: stored
                        .blocks
                        .iter()
                        .map(|block| convert_kv_block(block, seed))
                        .collect(),
                })
            }
            Data::Removed(removed)
                if !foreign_owner(removed.ownership.as_deref())
                    && !remote(removed.locality)
                    && indexed_tier(removed.tier, removed.cache_level)
                    && removed.group_idx.is_none() =>
            {
                Some(KvEvent::Removed(
                    removed
                        .block_hashes
                        .iter()
                        .copied()
                        .map(SequenceHash::from)
                        .collect(),
                ))
            }
            Data::Cleared(cleared) if !foreign_owner(cleared.ownership.as_deref()) => {
                Some(KvEvent::Cleared)
            }
            Data::Stored(_) | Data::Removed(_) | Data::Cleared(_) => None,
        })
        .collect()
}

/// Whether a residency agent owns the blocks.
fn foreign_owner(ownership: Option<&str>) -> bool {
    ownership.is_some_and(|owner| owner.eq_ignore_ascii_case("kvcr"))
}

/// Whether the blocks live on a remote tier.
fn remote(locality: Option<i32>) -> bool {
    locality == Some(KvCacheLocality::Remote as i32)
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

/// Whether the store names a main attention kind or names neither kind nor group.
fn main_attention(stored: &KvBlocksStored) -> bool {
    match stored.kv_cache_spec_kind.as_deref() {
        Some(kind) => MAIN_ATTENTION_KINDS.contains(&kind),
        None => stored.group_idx.is_none(),
    }
}

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

/// The KV collector's host over the store an events subscription feeds.
pub struct KvHost<G, S: ?Sized> {
    objects: Arc<Objects<G>>,
    sink: Arc<KvAdapter<S>>,
}

impl<G, S: ?Sized> KvHost<G, S> {
    /// A host whose subscriptions feed `store`, each under the row
    /// `objects` holds for its worker.
    pub fn new(objects: Arc<Objects<G>>, store: Arc<S>) -> Self {
        Self {
            objects,
            sink: Arc::new(KvAdapter::new(store)),
        }
    }
}

impl<G, S> KvEventHost for KvHost<G, S>
where
    G: Generation,
    S: store::KvEventSink<Batch = [KvEvent]> + ?Sized + 'static,
{
    type Sink = KvAdapter<S>;
    type Ticket = Option<WorkerId>;

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
        let id = self.objects.registry().get_id_by_url(worker.url())?;
        self.objects.standing(&id).map(|standing| standing.row)
    }

    /// The feed names the worker's row, and `None` means the holder has no row for the worker.
    fn open(
        &self,
        ticket: Self::Ticket,
        _worker: &Arc<dyn Worker>,
        _model: &str,
    ) -> Option<KvEventFeed<Self::Sink>> {
        ticket.map(|row| (Arc::clone(&self.sink), row))
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
        fn new() -> Self {
            Self::with_block_size(Some(BLOCK_SIZE))
        }

        fn with_block_size(kv_block_size: Option<u32>) -> Self {
            let store = Arc::new(MemoryStore::new());
            let row = store
                .upsert(WorkerFacts {
                    key: URL.into(),
                    engine: URL.into(),
                    incarnation: 1,
                    models: vec![MODEL.into()],
                    dp_rank: RANK,
                    dp_size: RANK + 1,
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
    fn a_store_for_another_attention_kind_is_passed_over() {
        assert_store_is_skipped(KvBlocksStored {
            group_idx: Some(1),
            kv_cache_spec_kind: Some("sliding_window".into()),
            ..stored(1)
        });
    }

    #[test]
    fn a_store_that_names_a_group_but_no_kind_is_passed_over() {
        assert_store_is_skipped(KvBlocksStored {
            group_idx: Some(0),
            ..stored(1)
        });
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
        assert!(events(&empty).is_empty());
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
    fn a_removal_that_names_a_group_is_passed_over() {
        let f = Fixture::new();
        f.applied(1, &batch([Data::Stored(stored(1))]));
        let removal = KvBlocksRemoved {
            group_idx: Some(0),
            ..removed(&[1])
        };
        f.applied(2, &batch([Data::Removed(removal)]));
        assert_eq!(f.blocks(), 1);
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
        assert!(events(&missing).is_empty());
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
            events(&ordered).as_slice(),
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
        let f = Fixture::new();
        let host = f.host();
        let worker = host
            .objects
            .registry()
            .get_by_url(URL)
            .expect("the worker's object");
        let ticket = host.enroll(&worker, MODEL);
        assert_eq!(ticket, Some(f.row));
        assert_eq!(f.store.block_size(MODEL), Some(BLOCK_SIZE));
        let (sink, row) = host.open(ticket, &worker, MODEL).expect("the row's feed");
        assert_eq!(row, f.row);
        assert!(Arc::ptr_eq(&sink, &host.sink));
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
        assert_eq!(ticket, None);
        assert!(host.open(ticket, &worker, MODEL).is_none());
    }

    #[test]
    fn a_worker_with_no_positive_block_size_leaves_it_unset() {
        for size in [None, Some(0)] {
            let f = Fixture::with_block_size(size);
            let host = f.host();
            let worker = host
                .objects
                .registry()
                .get_by_url(URL)
                .expect("the worker's object");
            assert_eq!(host.enroll(&worker, MODEL), Some(f.row));
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
