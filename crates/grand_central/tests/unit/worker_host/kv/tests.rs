use std::sync::mpsc;

use smg_worker::{
    common_proto::{
        KvBlock, KvBlocksRemoved, KvBlocksStored, KvCacheCleared, KvCacheEvent, KvCacheLocality,
        KvCacheTier,
    },
    identity::builder::BasicWorkerBuilder,
    kv_events::port::SnapshotChunk,
    registry::WorkerRegistry,
};

use super::*;
use crate::{
    store::{MemoryStore, Registrant, Signals, WorkerFacts, WorkersStore},
    worker_host::{objects::tests::lay_standing, Members},
};

const MODEL: &str = "m";
const URL: &str = "http://a:8000";
const RANK: u16 = 1;
const BLOCK_SIZE: u32 = 32;

impl KvAdapter {
    /// An adapter that writes `tables`, every batch to its stream's row. It
    /// has no holder to state its rows: the cases that build one state
    /// theirs to the tables themselves.
    fn new(tables: Arc<KvTables>) -> Self {
        Self::over(tables, None)
    }
}

/// A store's rows, the tables beside them and an adapter with no
/// holder: the cases state a row to the tables themselves
/// ([`Fixture::state`]), as a holder does where it lays the row.
struct Fixture {
    store: Arc<MemoryStore>,
    tables: Arc<KvTables>,
    sink: Arc<KvAdapter>,
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
        let tables = Arc::new(KvTables::default());
        let sink = Arc::new(KvAdapter::new(Arc::clone(&tables)));
        let fixture = Self {
            store,
            tables,
            sink,
            row,
        };
        fixture.state(row);
        fixture
    }

    /// State the row `id` to the tables by the facts the store holds
    /// for it now, as a holder does where it lays the row.
    fn state(&self, id: WorkerId) {
        let row = self.store.get(id).expect("the row");
        self.tables.member(&row);
    }

    fn applied(&self, sequence: u64, batch: &KvEventBatch) {
        assert_eq!(
            self.sink
                .admit(self.row, RANK, facts_of(sequence, batch), batch),
            BatchOutcome::Applied
        );
    }

    fn blocks(&self) -> usize {
        self.tables.indexed_blocks(self.row, RANK)
    }

    /// A host over a holder that was handed the fixture's tables and
    /// has laid the store's rows.
    fn host(&self) -> KvHost<Members> {
        let objects = Arc::new(
            Objects::new(Arc::new(WorkerRegistry::new())).with_kv_tables(Arc::clone(&self.tables)),
        );
        lay_standing(&objects, &self.store);
        KvHost::new(objects)
    }
}

fn facts(sequence: u64) -> BatchFacts {
    BatchFacts {
        sequence,
        ..BatchFacts::default()
    }
}

/// Whether the adapter keeps `row` as its own stream's, begun from
/// nothing: no batch was applied under its state.
fn begun(sink: &KvAdapter, row: WorkerId) -> bool {
    let rows = sink.rows.lock();
    rows.written.get(&row).is_some_and(|written| {
        written.stream == row && !written.leaving && written.applied.lock().emptied.is_none()
    })
}

/// The header's facts of `batch` at `sequence`, by the collector's rule:
/// a clear no name owns is the header's clear, and a clear a name owns
/// is not.
fn facts_of(sequence: u64, batch: &KvEventBatch) -> BatchFacts {
    let clears = batch.events.iter().any(
        |event| matches!(&event.data, Some(Data::Cleared(cleared)) if cleared.ownership.is_none()),
    );
    BatchFacts {
        sequence,
        clears,
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

/// A main store that names a namespace, an adapter and a tenant's salt,
/// is indexed with its chain as any other, over two batches. The hashes
/// under the namespace's seed are the apply rules' own.
#[test]
fn a_main_store_of_a_namespace_is_indexed_with_its_chain() {
    let f = Fixture::new();
    let parent = block(-1, &[1, 2]);
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
    event.blocks = vec![block(2, &[3, 4]), block(3, &[5, 6])];
    f.applied(2, &batch([Data::Stored(event)]));
    assert_eq!(f.blocks(), 3);
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
    // Its clear is not the engine's own: the cursor skips the batch.
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
/// row holds: the batch is late, and the cursor skips it as any other.
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
/// publisher by the cursor's own rule. Its owned clear still wipes what
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
    // The block that stands is the one behind the clear.
    f.applied(4, &batch([Data::Removed(group_removal(&[6], 0))]));
    assert_eq!(f.blocks(), 0);
}

/// Engines number batches from 0, and the row's cursor reads 0 behind
/// batch 0. A batch 0 that arrives again is late there as at any other
/// sequence: its owned clear empties nothing.
#[test]
fn a_repeated_first_batch_with_an_owned_clear_is_skipped() {
    let f = Fixture::new();
    f.applied(0, &batch([Data::Stored(stored(1))]));
    let late = batch([
        Data::Cleared(KvCacheCleared {
            ownership: Some("engine".into()),
        }),
        Data::Stored(main_store(3, 0)),
    ]);
    assert_eq!(
        f.sink.admit(f.row, RANK, facts_of(0, &late), &late),
        BatchOutcome::Skipped
    );
    // The first batch's block stands: its removal empties the rank.
    assert_eq!(f.blocks(), 1);
    f.applied(1, &batch([Data::Removed(removed(&[1]))]));
    assert_eq!(f.blocks(), 0);
}

/// A main store in a batch 0 that arrives again would restart the
/// worker by the adapter's own clear. Behind batch 0 it is late as
/// well, and the cursor skips it.
#[test]
fn a_repeated_first_batch_that_shows_a_main_group_is_no_restarted_publisher() {
    let f = Fixture::new();
    f.applied(0, &batch([Data::Stored(window_store(&[1], 0, true))]));
    let late = batch([Data::Stored(main_store(7, 1))]);
    assert_eq!(
        f.sink.admit(f.row, RANK, facts_of(0, &late), &late),
        BatchOutcome::Skipped
    );
    assert_eq!(f.blocks(), 1);
    // The window group is the worker's main signal still.
    f.applied(1, &batch([Data::Stored(window_store(&[2], 0, true))]));
    assert_eq!(f.blocks(), 2);
}

/// A cursor reads 0 behind a resync too, and the resync emptied the
/// rank. A batch 0 there is the stream's first: the cursor takes it, and
/// what it holds behind its last clear stands.
#[test]
fn a_first_batch_behind_a_resync_stands_from_its_last_clear() {
    let f = Fixture::new();
    f.applied(4, &batch([Data::Stored(stored(1))]));
    f.sink.resync(f.row, None, ResyncReason::DataLoss);
    assert_eq!(f.blocks(), 0);
    let first = batch([
        Data::Stored(main_store(5, 0)),
        Data::Cleared(KvCacheCleared {
            ownership: Some("engine".into()),
        }),
        Data::Stored(main_store(6, 0)),
    ]);
    f.applied(0, &first);
    assert_eq!(f.blocks(), 1);
    f.applied(1, &batch([Data::Removed(group_removal(&[6], 0))]));
    assert_eq!(f.blocks(), 0);
}

/// A main group shown in a batch 0 behind a resync: the adapter's own
/// clear is read as late there, and the resync emptied the rank already.
#[test]
fn a_first_batch_behind_a_resync_that_shows_a_main_group_stands() {
    let f = Fixture::new();
    f.applied(4, &batch([Data::Stored(window_store(&[1], 0, true))]));
    f.sink.resync(f.row, None, ResyncReason::DataLoss);
    f.applied(0, &batch([Data::Stored(main_store(7, 1))]));
    assert_eq!(f.blocks(), 1);
    // The window group is a group of another kind now: its store is not
    // the index's.
    f.applied(1, &batch([Data::Stored(window_store(&[2], 0, true))]));
    assert_eq!(f.blocks(), 1);
    // The block that stands is the main group's.
    f.applied(2, &batch([Data::Removed(group_removal(&[7], 1))]));
    assert_eq!(f.blocks(), 0);
}

#[test]
fn a_batch_the_store_does_not_take_shows_nothing() {
    let f = Fixture::new();
    f.applied(1, &batch([Data::Stored(main_store(1, 0))]));
    // A batch at a sequence the cursor holds: its store shows no group.
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
    assert_eq!(f.blocks(), 0);
}

#[test]
fn a_removal_takes_the_blocks_out() {
    let f = Fixture::new();
    let mut event = stored(-1);
    event.blocks.push(block(2, &[3, 4]));
    f.applied(1, &batch([Data::Stored(event)]));
    assert_eq!(f.blocks(), 2);
    // The rules count a block's copies a tier. A removal on the tier
    // that holds no copy of the blocks takes none out.
    let from_host = KvBlocksRemoved {
        tier: Some(KvCacheTier::Host as i32),
        cache_level: Some(2),
        ..removed(&[-1, 2])
    };
    f.applied(2, &batch([Data::Removed(from_host)]));
    assert_eq!(f.blocks(), 2);
    f.applied(3, &batch([Data::Removed(removed(&[-1, 2]))]));
    assert_eq!(f.blocks(), 0);
}

/// A block stored on the device and on the host is one block of the
/// index with two copies. It leaves with its last copy.
#[test]
fn a_block_leaves_the_index_with_its_last_copy() {
    let f = Fixture::new();
    let on_host = KvBlocksStored {
        tier: Some(KvCacheTier::Host as i32),
        ..stored(1)
    };
    f.applied(1, &batch([Data::Stored(stored(1)), Data::Stored(on_host)]));
    assert_eq!(f.blocks(), 1);
    f.applied(2, &batch([Data::Removed(removed(&[1]))]));
    assert_eq!(f.blocks(), 1, "the host still holds a copy");
    let from_host = KvBlocksRemoved {
        tier: Some(KvCacheTier::Host as i32),
        ..removed(&[1])
    };
    f.applied(3, &batch([Data::Removed(from_host)]));
    assert_eq!(f.blocks(), 0);
}

/// Two copies on one tier need two removals.
#[test]
fn two_copies_on_a_tier_need_two_removals() {
    let f = Fixture::new();
    f.applied(
        1,
        &batch([Data::Stored(stored(1)), Data::Stored(stored(1))]),
    );
    assert_eq!(f.blocks(), 1);
    f.applied(2, &batch([Data::Removed(removed(&[1]))]));
    assert_eq!(f.blocks(), 1, "the other copy is still cached");
    f.applied(3, &batch([Data::Removed(removed(&[1]))]));
    assert_eq!(f.blocks(), 0);
}

/// The tables empty a rank in their own turn, here at a resync. The
/// row's next batch reads another value of its lane and forgets the
/// copies it counted: a block stored once behind the resync leaves at
/// its one removal.
#[test]
fn a_rank_the_store_emptied_forgets_the_copies_it_counted() {
    let f = Fixture::new();
    f.applied(
        1,
        &batch([Data::Stored(stored(1)), Data::Stored(stored(1))]),
    );
    f.tables.resync(f.row, Some(RANK), ResyncReason::DataLoss);
    assert_eq!(f.blocks(), 0);
    f.applied(5, &batch([Data::Stored(stored(1))]));
    assert_eq!(f.blocks(), 1);
    f.applied(6, &batch([Data::Removed(removed(&[1]))]));
    assert_eq!(f.blocks(), 0, "one copy stood behind the resync");
}

/// A store narrower than the row's block of record is left out: an
/// engine's prefix entry at its hash granularity, which no request
/// chunked at the block matches.
#[test]
fn a_store_narrower_than_the_block_of_record_is_left_out() {
    let f = Fixture::new();
    f.applied(1, &wide(1, 32));
    assert_eq!(f.blocks(), 1);
    f.applied(2, &wide(77, 4));
    assert_eq!(f.blocks(), 1);
    // A wider store raises the record, and what was indexed stays.
    f.applied(3, &wide(2, 64));
    assert_eq!(f.blocks(), 2);
    f.applied(4, &wide(3, 32));
    assert_eq!(f.blocks(), 2);
}

/// The tables empty a rank in their own turn, and the row's next batch
/// forgets its block of record: an engine that comes back with a
/// narrower block has its stores indexed as they come.
#[test]
fn a_rank_the_store_emptied_forgets_its_block_of_record() {
    let f = Fixture::new();
    f.applied(1, &wide(1, 32));
    f.tables.resync(f.row, Some(RANK), ResyncReason::DataLoss);
    f.applied(5, &wide(2, 16));
    assert_eq!(f.blocks(), 1, "the narrower block is the row's block now");
}

/// The row's view in the tables' index, read by a lent batch with no
/// event at `sequence`.
fn view(f: &Fixture, sequence: u64) -> Arc<WorkerGroups> {
    let mut view = None;
    let outcome = f
        .tables
        .admit_through(f.row, RANK, facts(sequence), &mut |lane| {
            view = Some(lane.groups());
        });
    assert_eq!(outcome, BatchOutcome::Applied);
    view.expect("the body ran")
}

/// A main store and a window group's store of the same block, as a
/// hybrid engine writes them.
fn hybrid() -> KvEventBatch {
    batch([
        Data::Stored(main_store(1, 0)),
        Data::Stored(window_store(&[1], 1, false)),
    ])
}

/// A store of a cache group other than main attention feeds the row's
/// view in the tables' index, where a decision can read it.
#[test]
fn a_group_store_feeds_the_rows_view_in_the_stores_index() {
    let f = Fixture::new();
    f.applied(1, &hybrid());
    assert_eq!(f.blocks(), 1);
    let view = view(&f, 2);
    assert!(view.knows(1), "the window group is the view's");
    assert_eq!(view.named_blocks(), 1, "the main block is named in it");
}

/// A row's lane is made again under the row's id, and the index holds
/// another view for the row. The row's next batch reads another value
/// of its lane and takes that view: what its group stores name is in
/// the tables' index, not in the view of the lane before.
#[test]
fn a_lane_made_again_gives_the_rows_state_the_indexs_new_view() {
    let f = Fixture::new();
    f.applied(1, &hybrid());
    let old = view(&f, 2);
    // The row stops asking for its events, and asks again: each fact
    // is stated to the tables as a holder states it.
    let facts = f.store.get(f.row).expect("the row").facts;
    let quiet = WorkerFacts {
        signals: Signals::default(),
        ..facts.clone()
    };
    assert_eq!(f.store.upsert(quiet).id(), f.row);
    f.state(f.row);
    assert_eq!(f.store.upsert(facts).id(), f.row);
    f.state(f.row);
    f.applied(1, &hybrid());
    let new = view(&f, 2);
    assert!(!Arc::ptr_eq(&old, &new));
    assert!(new.knows(1), "the new view is the one the row writes");
    assert_eq!(new.named_blocks(), 1);
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
    assert_eq!(f.blocks(), 0);
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
    // The store behind the clear stands: the three were applied in the
    // batch's order.
    f.applied(2, &ordered);
    assert_eq!(f.blocks(), 1);
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
            f.tables.cursor(f.row, RANK),
            Some(KvCursor {
                applied: sequence,
                degraded: unknown_before,
                replay_pending: false
            })
        );
    }
}

#[test]
fn each_reason_reaches_the_store_under_its_own_name() {
    let f = Fixture::new();
    let store = batch([Data::Stored(stored(1))]);
    for reason in [
        ResyncReason::OutOfRange,
        ResyncReason::DataLoss,
        ResyncReason::SnapshotAbandoned,
        ResyncReason::PublisherRestart,
        ResyncReason::GapCleared,
        ResyncReason::Snapshot,
    ] {
        f.sink.resync(f.row, Some(RANK), reason);
        assert_eq!(f.tables.resyncs(reason), 1, "{reason:?}");

        f.applied(7, &store);
        f.sink.resync(f.row, Some(0), reason);
        assert_eq!(f.blocks(), 1);
        f.sink.resync(f.row, Some(RANK), reason);
        // An abandoned snapshot keeps the block it showed until the
        // next reason empties the rank. Every other reason empties it.
        let kept = usize::from(reason == ResyncReason::SnapshotAbandoned);
        assert_eq!(f.blocks(), kept, "{reason:?}");
    }

    f.applied(7, &store);
    f.sink.resync(f.row, None, ResyncReason::OutOfRange);
    assert_eq!(f.blocks(), 0);
}

#[test]
fn a_cursor_reads_as_the_stores() {
    let f = Fixture::new();
    assert_eq!(f.sink.cursor(f.row, RANK), None);
    f.tables.admit_through(
        f.row,
        RANK,
        BatchFacts {
            sequence: 41,
            clears: false,
            snapshot: Some(SnapshotChunk {
                unknown_before: true,
            }),
        },
        &mut |_| {},
    );
    f.tables.admit_through(
        f.row,
        RANK,
        BatchFacts {
            sequence: 43,
            ..Default::default()
        },
        &mut |_| {},
    );
    let expected = f.tables.cursor(f.row, RANK).expect("the tables' cursor");
    assert!(expected.replay_pending && expected.degraded);
    assert_eq!(f.sink.cursor(f.row, RANK), Some(expected));
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
    assert_eq!(f.tables.cursor(f.row, RANK), None);
}

/// The feed of the fixture's one worker, opened by its host.
fn opened(f: &Fixture) -> (KvHost<Members>, Arc<KvAdapter>) {
    let host = f.host();
    let worker = host
        .objects
        .registry()
        .get_by_url(URL)
        .expect("the worker's object");
    let ticket = host.enroll(&worker, MODEL);
    let (sink, row) = host.open(ticket, &worker, MODEL).expect("the row's feed");
    assert_eq!(row, f.row);
    (host, sink)
}

/// One store of one block `block_size` tokens wide.
fn wide_store(hash: i64, block_size: i32) -> KvBlocksStored {
    KvBlocksStored {
        blocks: vec![KvBlock {
            block_size,
            ..block(hash, &tokens_of(hash))
        }],
        ..KvBlocksStored::default()
    }
}

/// One store of one block `block_size` tokens wide, alone in its batch.
fn wide(hash: i64, block_size: i32) -> KvEventBatch {
    batch([Data::Stored(wide_store(hash, block_size))])
}

/// A batch the cursor takes names the model's block size, whatever the
/// worker stated at its registration: none, a narrower size or a wider
/// one. A narrower store behind it moves nothing, and a batch the cursor
/// skips names none.
#[test]
fn a_batch_the_store_takes_names_the_models_block_size() {
    for stated in [None, Some(16), Some(64)] {
        let f = Fixture::single(stated);
        let (_host, sink) = opened(&f);
        assert_eq!(f.tables.block_size(MODEL), stated);
        let mut first = wide(1, 16);
        first.events.extend(wide(2, 32).events);
        assert_eq!(
            sink.admit(f.row, 0, facts(1), &first),
            BatchOutcome::Applied
        );
        assert_eq!(f.tables.block_size(MODEL), Some(32), "stated: {stated:?}");
        // A narrower store is a prefix entry.
        assert_eq!(
            sink.admit(f.row, 0, facts(2), &wide(3, 8)),
            BatchOutcome::Applied
        );
        assert_eq!(f.tables.block_size(MODEL), Some(32));
        // A duplicate of the batch the row holds is skipped.
        assert_eq!(
            sink.admit(f.row, 0, facts(2), &wide(4, 64)),
            BatchOutcome::Skipped
        );
        assert_eq!(f.tables.block_size(MODEL), Some(32));
    }
}

/// A store the index passes over names no size. One on a tier the index
/// does not track is wider than the engine's block here, and the size of
/// record stays the indexed store's.
#[test]
fn a_store_the_index_passes_over_names_no_block_size() {
    let f = Fixture::single(None);
    let (_host, sink) = opened(&f);
    let passed_over = KvBlocksStored {
        tier: Some(KvCacheTier::Disk as i32),
        ..wide_store(2, 64)
    };
    let both = batch([Data::Stored(wide_store(1, 16)), Data::Stored(passed_over)]);
    assert_eq!(sink.admit(f.row, 0, facts(1), &both), BatchOutcome::Applied);
    assert_eq!(f.tables.block_size(MODEL), Some(16));
}

/// A registration that names a wider size behind a stored block is put
/// right by the next stored block, a prefix entry too: the engine's word
/// only rises. A model that left learns its size anew.
#[test]
fn a_stored_block_keeps_the_size_of_record_the_engines_own() {
    let f = Fixture::single(None);
    let (host, sink) = opened(&f);
    let mut sequence: u32 = 0;
    let mut stored = |block_size: i32| {
        sequence += 1;
        let batch = wide(i64::from(sequence), block_size);
        assert_eq!(
            sink.admit(f.row, 0, facts(u64::from(sequence)), &batch),
            BatchOutcome::Applied
        );
        f.tables.block_size(MODEL)
    };
    assert_eq!(stored(16), Some(16));
    // Another worker of the model registers and states a wider size.
    f.tables.set_block_size(MODEL, 64);
    assert_eq!(f.tables.block_size(MODEL), Some(64));
    assert_eq!(stored(16), Some(16));
    // The model leaves, and comes back with a registration's size.
    host.model_left(MODEL);
    assert_eq!(f.tables.block_size(MODEL), None);
    f.tables.set_block_size(MODEL, 64);
    assert_eq!(stored(16), Some(16));
    // A wider block raises it.
    assert_eq!(stored(32), Some(32));
    // A prefix entry behind a wider registration puts the engine's block
    // back, not its own width.
    f.tables.set_block_size(MODEL, 64);
    assert_eq!(stored(8), Some(32));
}

#[test]
fn the_block_size_is_read_and_set_at_the_store() {
    let f = Fixture::new();
    f.tables.set_block_size(MODEL, 16);
    assert_eq!(f.sink.block_size(MODEL), Some(16));
    f.sink.set_block_size(MODEL, 32);
    assert_eq!(f.tables.block_size(MODEL), Some(32));
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
    assert_eq!(f.tables.block_size(MODEL), Some(BLOCK_SIZE));
    let (sink, row) = host.open(ticket, &worker, MODEL).expect("the row's feed");
    assert_eq!(row, f.row);
    assert!(Arc::ptr_eq(&sink, &host.sink));
    assert_eq!(
        host.skipped_subscriptions(),
        SkippedSubscriptions::default()
    );
}

#[test]
fn a_row_the_adapter_never_fed_begins_from_nothing() {
    let f = Fixture::single(Some(BLOCK_SIZE));
    let stores = batch([
        Data::Stored(main_store(1, 0)),
        Data::Stored(window_store(&[1], 2, false)),
    ]);
    assert_eq!(
        f.sink.admit(f.row, 0, facts(7), &stores),
        BatchOutcome::Applied
    );
    assert_eq!(
        f.tables.cursor(f.row, 0).map(|cursor| cursor.applied),
        Some(7)
    );
    assert_eq!(f.tables.indexed_blocks(f.row, 0), 1);

    let host = f.host();
    let worker = host
        .objects
        .registry()
        .get_by_url(URL)
        .expect("the worker's object");
    let (sink, row) = host
        .open(host.enroll(&worker, MODEL), &worker, MODEL)
        .expect("the row's feed");
    assert_eq!(sink.cursor(row, 0), None);
    assert_eq!(f.tables.indexed_blocks(row, 0), 0);
    assert!(begun(&sink, row));

    assert_eq!(sink.admit(row, 0, facts(9), &stores), BatchOutcome::Applied);
    let (sink, row) = host
        .open(host.enroll(&worker, MODEL), &worker, MODEL)
        .expect("the row's feed");
    assert_eq!(sink.cursor(row, 0).map(|cursor| cursor.applied), Some(9));
    assert_eq!(f.tables.indexed_blocks(row, 0), 1);
    assert_eq!(
        sink.admit(
            row,
            0,
            facts(10),
            &batch([Data::Removed(group_removal(&[1], 2))])
        ),
        BatchOutcome::Applied
    );
    assert_eq!(f.tables.indexed_blocks(row, 0), 1);
}

/// The tables keep a row's lane at the row's own rank. A row the holder
/// stands above rank 0 is its engine's rank 0 stream's. A ticket for
/// it opens no feed, and what the tables keep of the row stands. It is
/// counted as an upper rank's: the holder keeps the row.
#[test]
fn a_ticket_for_a_row_above_rank_0_opens_no_feed() {
    let f = Fixture::new();
    let stores = batch([
        Data::Stored(main_store(1, 0)),
        Data::Stored(window_store(&[1], 2, false)),
    ]);
    f.applied(7, &stores);
    assert_eq!(
        f.tables.cursor(f.row, RANK).map(|cursor| cursor.applied),
        Some(7)
    );
    assert_eq!(f.blocks(), 1);

    // A host over the same store: its holder stands the row at rank 1,
    // and the row's object stands for it.
    let host = f.host();
    let worker = host
        .objects
        .registry()
        .get_by_url(URL)
        .expect("the worker's object");
    assert!(host.open(Ok(f.row), &worker, MODEL).is_none());
    assert_eq!(
        f.tables.cursor(f.row, RANK).map(|cursor| cursor.applied),
        Some(7)
    );
    assert_eq!(f.blocks(), 1);
    let unfed = host.skipped_subscriptions();
    assert_eq!((unfed.no_row, unfed.upper_rank), (0, 1));
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
    assert_eq!(ticket, Err(SkipReason::UpperRank));
    // The model's size is seeded all the same.
    assert_eq!(f.tables.block_size(MODEL), Some(BLOCK_SIZE));
    assert!(host.open(ticket, &worker, MODEL).is_none());
    let unfed = SkippedSubscriptions {
        no_row: 0,
        upper_rank: 1,
    };
    assert_eq!(host.skipped_subscriptions(), unfed);
}

#[test]
fn an_object_the_holder_replaced_opens_nothing() {
    let f = Fixture::single(Some(BLOCK_SIZE));
    let host = f.host();
    let registry = host.objects.registry();
    let old = registry.get_by_url(URL).expect("the worker's object");
    // The row's facts move, so the holder lays another object in the
    // place of the first.
    let moved = WorkerFacts {
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
    };
    let written = host.objects.upsert(&f.store, moved, Registrant::Control);
    assert!(written.is_some(), "the store takes the write");
    let new = registry.get_by_url(URL).expect("the row's new object");
    assert!(!Arc::ptr_eq(&old, &new));
    assert_eq!(host.enroll(&old, MODEL), Err(SkipReason::NoRow));
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
    assert_eq!(ticket, Err(SkipReason::NoRow));
    assert!(host.open(ticket, &worker, MODEL).is_none());
    let unfed = SkippedSubscriptions {
        no_row: 1,
        upper_rank: 0,
    };
    assert_eq!(host.skipped_subscriptions(), unfed);
}

/// A host over a holder that was handed no tables indexes nothing. The
/// row's feed opens all the same, since a stream brings more than
/// blocks. No row holds a lane, so its batch is passed over and no
/// cursor stands for it.
#[test]
fn a_host_over_a_holder_handed_no_tables_indexes_nothing() {
    let f = Fixture::single(Some(BLOCK_SIZE));
    let objects = Arc::new(Objects::new(Arc::new(WorkerRegistry::new())));
    lay_standing(&objects, &f.store);
    assert!(objects.kv_tables().is_none());
    let host: KvHost<Members> = KvHost::new(objects);
    let worker = host
        .objects
        .registry()
        .get_by_url(URL)
        .expect("the worker's object");
    let (sink, row) = host
        .open(host.enroll(&worker, MODEL), &worker, MODEL)
        .expect("the row's feed");
    assert_eq!(row, f.row);
    let one = batch([Data::Stored(stored(1))]);
    assert_eq!(sink.admit(row, 0, facts(1), &one), BatchOutcome::Skipped);
    assert_eq!(sink.cursor(row, 0), None);
    assert_eq!(sink.tables.indexed_blocks(row, 0), 0);
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
        assert_eq!(f.tables.block_size(MODEL), None);
    }
}

#[test]
fn a_model_that_left_forgets_its_block_size() {
    let f = Fixture::new();
    let host = f.host();
    f.tables.set_block_size(MODEL, 32);
    host.model_left(MODEL);
    assert_eq!(f.tables.block_size(MODEL), None);
    f.tables.set_block_size(MODEL, 16);
    assert_eq!(f.tables.block_size(MODEL), Some(16));
}

/// A model that left takes its own word alone: another model keeps the
/// engine's word and the tables' size.
#[test]
fn a_model_that_left_takes_its_own_block_size_alone() {
    const OTHER_MODEL: &str = "n";
    let f = Fixture::single(None);
    let (host, sink) = opened(&f);
    let first = sink.admit(f.row, 0, facts(1), &wide(1, 32));
    assert_eq!(first, BatchOutcome::Applied);
    // What a feed of the other model learned.
    sink.sizes.lock().insert(Arc::from(OTHER_MODEL), 64);
    f.tables.set_block_size(OTHER_MODEL, 64);
    host.model_left(MODEL);
    assert_eq!(f.tables.block_size(MODEL), None);
    assert_eq!(f.tables.block_size(OTHER_MODEL), Some(64));
    let words = sink.sizes.lock();
    assert_eq!(words.get(MODEL), None);
    assert_eq!(words.get(OTHER_MODEL), Some(&64));
}

/// A model that left takes the engine's word with it: the first stored
/// block behind its return names its size, a narrower one too.
#[test]
fn a_model_that_left_names_its_block_size_anew() {
    let f = Fixture::single(None);
    let (host, sink) = opened(&f);
    let first = sink.admit(f.row, 0, facts(1), &wide(1, 32));
    assert_eq!(first, BatchOutcome::Applied);
    assert_eq!(f.tables.block_size(MODEL), Some(32));
    // The model's one worker leaves, and the model with it.
    sink.removed(f.row);
    host.model_left(MODEL);
    assert_eq!(f.tables.block_size(MODEL), None);
    // It returns on the same host with a narrower block.
    let worker = host
        .objects
        .registry()
        .get_by_url(URL)
        .expect("the worker's object");
    let ticket = host.enroll(&worker, MODEL);
    let (sink, _) = host.open(ticket, &worker, MODEL).expect("the row's feed");
    let again = sink.admit(f.row, 0, facts(1), &wide(2, 16));
    assert_eq!(again, BatchOutcome::Applied);
    assert_eq!(f.tables.block_size(MODEL), Some(16));
}

/// A stream that left takes its feed's record with it: a batch for its
/// row behind it names no size. What the batch itself is answered is no
/// part of this case.
#[test]
fn a_stream_that_left_names_no_block_size() {
    let f = Fixture::single(None);
    let (_host, sink) = opened(&f);
    sink.removed(f.row);
    let _ = sink.admit(f.row, 0, facts(1), &wide(1, 32));
    assert_eq!(f.tables.block_size(MODEL), None);
    assert!(sink.sizes.lock().is_empty());
}

/// A batch reads its feed's record where it begins and learns at its
/// end. One that began ahead of its stream's removal and learns behind it
/// names no size: the feed is its row's no longer, and the model's
/// departure cleared the size in between. A feed opened anew for the row
/// is another feed, so the old batch's learning is not its either, and
/// its own learning names the size.
#[test]
fn a_learning_behind_its_streams_removal_names_no_block_size() {
    let f = Fixture::single(None);
    let (host, sink) = opened(&f);
    let of_row = |sink: &KvAdapter| sink.feeds.lock().get(&f.row).cloned();
    let began = of_row(&sink).expect("the feed the batch found open");
    sink.removed(f.row);
    host.model_left(MODEL);
    sink.learn(f.row, &began, 32);
    assert_eq!(f.tables.block_size(MODEL), None);
    assert!(sink.sizes.lock().is_empty());
    // The model returns on the same host, under the same row.
    let worker = host
        .objects
        .registry()
        .get_by_url(URL)
        .expect("the worker's object");
    let ticket = host.enroll(&worker, MODEL);
    let (sink, _) = host.open(ticket, &worker, MODEL).expect("the row's feed");
    sink.learn(f.row, &began, 32);
    assert_eq!(f.tables.block_size(MODEL), None);
    let anew = of_row(&sink).expect("the feed opened anew");
    sink.learn(f.row, &anew, 16);
    assert_eq!(f.tables.block_size(MODEL), Some(16));
}

/// A model's departure takes the engine's word and the tables' size in
/// one hold of the lock a learning holds, so no row of the model learns
/// between the two. The table's own lock is the probe: the case holds
/// it, and the departure cannot get past its call of the tables.
#[test]
fn a_models_departure_holds_the_learnings_lock_through_both_removals() {
    let f = Fixture::single(None);
    let (host, sink) = opened(&f);
    let first = sink.admit(f.row, 0, facts(1), &wide(1, 32));
    assert_eq!(first, BatchOutcome::Applied);
    // The engine's words, looked at while the departure stands at the
    // tables. Free, with the model's word gone: the departure let the
    // lock go ahead of the tables. Free, with the word still there: the
    // departure has not begun. Held at every look over a quarter second,
    // which is long against a removal's own hold: the departure holds
    // them, at the tables or still ahead of them.
    let held_at_the_tables = || {
        let window = std::time::Instant::now() + std::time::Duration::from_millis(250);
        loop {
            if let Some(sizes) = sink.sizes.try_lock() {
                return (!sizes.contains_key(MODEL)).then_some(false);
            }
            if std::time::Instant::now() >= window {
                return Some(true);
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    };
    let held = std::thread::scope(|scope| {
        f.tables.locked(|| {
            // The collector's own call: the host's.
            scope.spawn(|| host.model_left(MODEL));
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            loop {
                if let Some(held) = held_at_the_tables() {
                    break Some(held);
                }
                if std::time::Instant::now() >= deadline {
                    break None;
                }
                std::thread::yield_now();
            }
        })
    });
    assert_eq!(held, Some(true), "let go ahead of the tables");
    assert_eq!(f.tables.block_size(MODEL), None);
    assert!(sink.sizes.lock().is_empty());
}

const UPPER: &str = "http://a:8000#1";

const OTHER: &str = "http://b:8000";
const THIRD: &str = "http://a:8000#2";
const FOURTH: &str = "http://b:8000#1";

/// The row of one rank of the engine at `engine`, which asks for events.
fn rank_facts(key: &str, engine: &str, dp_rank: u16) -> WorkerFacts {
    WorkerFacts {
        key: key.into(),
        engine: engine.into(),
        incarnation: 1,
        models: vec![MODEL.into()],
        dp_rank,
        dp_size: 3,
        signals: Signals {
            events: true,
            ..Signals::default()
        },
        ..WorkerFacts::default()
    }
}

/// The cache-event series of the fixture's worker, read from a recorder
/// of the case's own. The recorder's crates come with the controller.
#[cfg(feature = "controller")]
mod series {
    use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle, PrometheusRecorder};

    use super::*;

    struct Series {
        recorder: PrometheusRecorder,
        handle: PrometheusHandle,
    }

    impl Series {
        fn new() -> Self {
            let recorder = PrometheusBuilder::new().build_recorder();
            let handle = recorder.handle();
            Self { recorder, handle }
        }

        /// The value of `name` for the fixture's worker under `labels`,
        /// where the series stands.
        fn read(&self, name: &str, labels: &[&str]) -> Option<f64> {
            let worker = format!("worker=\"{URL}\"");
            let open = format!("{name}{{");
            self.handle.render().lines().find_map(|line| {
                let (series, value) = line.rsplit_once(' ')?;
                let named = series.starts_with(&open) && series.contains(&worker);
                let all = labels.iter().all(|label| series.contains(label));
                (named && all).then(|| value.parse().ok()).flatten()
            })
        }

        /// Whether any cache-event series names the fixture's worker.
        /// The worker's object has series of its own.
        fn names_the_worker(&self) -> bool {
            let worker = format!("worker=\"{URL}\"");
            let rendered = self.handle.render();
            let mut lines = rendered.lines();
            lines.any(|line| line.starts_with("smg_kv_") && line.contains(&worker))
        }
    }

    /// A batch the cursor takes counts its blocks by operation and
    /// itself as applied, publishes its pace and sets the worker's
    /// index blocks. A removal behind it counts its blocks and moves
    /// the gauge.
    #[test]
    fn an_applied_batch_counts_its_blocks_and_sets_the_workers_gauge() {
        let series = Series::new();
        metrics::with_local_recorder(&series.recorder, || {
            let f = Fixture::single(Some(BLOCK_SIZE));
            let (_host, sink) = opened(&f);
            let mut first = wide(1, 32);
            first.events.extend(wide(2, 32).events);
            assert_eq!(
                sink.admit(f.row, 0, facts(1), &first),
                BatchOutcome::Applied
            );
            let stored = series.read("smg_kv_event_blocks_total", &["op=\"stored\""]);
            assert_eq!(stored, Some(2.0));
            let applied = &["disposition=\"applied\""];
            assert_eq!(
                series.read("smg_kv_event_batches_total", applied),
                Some(1.0)
            );
            assert_eq!(series.read("smg_kv_index_blocks", &[]), Some(2.0));
            assert_eq!(
                series.read("smg_kv_event_apply_seconds_count", &[]),
                Some(1.0)
            );

            let gone = batch([Data::Removed(removed(&[2]))]);
            assert_eq!(sink.admit(f.row, 0, facts(2), &gone), BatchOutcome::Applied);
            let removed = series.read("smg_kv_event_blocks_total", &["op=\"removed\""]);
            assert_eq!(removed, Some(1.0));
            assert_eq!(
                series.read("smg_kv_event_batches_total", applied),
                Some(2.0)
            );
            assert_eq!(series.read("smg_kv_index_blocks", &[]), Some(1.0));
        });
    }

    /// A snapshot chunk counts its blocks and itself under the
    /// snapshot's own words, and publishes no pace.
    #[test]
    fn a_snapshot_chunk_counts_under_its_own_words_and_publishes_no_pace() {
        let series = Series::new();
        metrics::with_local_recorder(&series.recorder, || {
            let f = Fixture::single(Some(BLOCK_SIZE));
            let (_host, sink) = opened(&f);
            let chunk = BatchFacts {
                snapshot: Some(SnapshotChunk {
                    unknown_before: false,
                }),
                ..facts(7)
            };
            assert_eq!(
                sink.admit(f.row, 0, chunk, &wide(1, 32)),
                BatchOutcome::Applied
            );
            let stored = series.read("smg_kv_event_blocks_total", &["op=\"snapshot\""]);
            assert_eq!(stored, Some(1.0));
            let chunks = &["disposition=\"snapshot\""];
            assert_eq!(series.read("smg_kv_event_batches_total", chunks), Some(1.0));
            assert_eq!(series.read("smg_kv_index_blocks", &[]), Some(1.0));
            assert_eq!(series.read("smg_kv_event_apply_seconds_count", &[]), None);
        });
    }

    /// A batch behind its lane's cursor counts as stale, and a gap
    /// counts the replay the collector asks.
    #[test]
    fn a_stale_batch_and_a_gap_count_under_their_own_words() {
        let series = Series::new();
        metrics::with_local_recorder(&series.recorder, || {
            let f = Fixture::single(Some(BLOCK_SIZE));
            let (_host, sink) = opened(&f);
            assert_eq!(
                sink.admit(f.row, 0, facts(1), &wide(1, 32)),
                BatchOutcome::Applied
            );
            assert_eq!(
                sink.admit(f.row, 0, facts(1), &wide(1, 32)),
                BatchOutcome::Skipped
            );
            let stale = &["disposition=\"stale\""];
            assert_eq!(series.read("smg_kv_event_batches_total", stale), Some(1.0));
            assert_eq!(
                sink.admit(f.row, 0, facts(3), &wide(3, 32)),
                BatchOutcome::Gap {
                    expected: 2,
                    received: 3
                }
            );
            let asked = &["outcome=\"replay_requested\""];
            assert_eq!(series.read("smg_kv_event_gaps_total", asked), Some(1.0));
            // Neither moved the applied count or the stored blocks.
            let applied = &["disposition=\"applied\""];
            assert_eq!(
                series.read("smg_kv_event_batches_total", applied),
                Some(1.0)
            );
            let stored = series.read("smg_kv_event_blocks_total", &["op=\"stored\""]);
            assert_eq!(stored, Some(1.0));
        });
    }

    /// A reset counts under its reason, each reason by itself.
    #[test]
    fn a_reset_counts_under_its_reason() {
        let series = Series::new();
        metrics::with_local_recorder(&series.recorder, || {
            let f = Fixture::single(Some(BLOCK_SIZE));
            let (_host, sink) = opened(&f);
            sink.resync(f.row, None, ResyncReason::DataLoss);
            sink.resync(f.row, Some(0), ResyncReason::Snapshot);
            sink.resync(f.row, None, ResyncReason::DataLoss);
            let lost = &["reason=\"data_loss\""];
            assert_eq!(series.read("smg_kv_event_resyncs_total", lost), Some(2.0));
            let snapshot = &["reason=\"snapshot\""];
            assert_eq!(
                series.read("smg_kv_event_resyncs_total", snapshot),
                Some(1.0)
            );
        });
    }

    /// A store whose parent the row does not hold is placed as a chain
    /// of its own and counted for the worker: the rules' own count
    /// reaches a series.
    #[test]
    fn a_parentless_store_is_counted_for_the_worker() {
        let series = Series::new();
        metrics::with_local_recorder(&series.recorder, || {
            let f = Fixture::single(Some(BLOCK_SIZE));
            let (_host, sink) = opened(&f);
            let orphan = KvBlocksStored {
                parent_block_hash: Some(99),
                ..wide_store(1, 32)
            };
            let orphan = batch([Data::Stored(orphan)]);
            assert_eq!(
                sink.admit(f.row, 0, facts(1), &orphan),
                BatchOutcome::Applied
            );
            let stores = series.read("smg_kv_event_parentless_stores_total", &[]);
            assert_eq!(stores, Some(1.0));
            let blocks = series.read("smg_kv_event_parentless_blocks_total", &[]);
            assert_eq!(blocks, Some(1.0));
            assert_eq!(first_block_holders(&f, 1), [(f.row, 1)]);
        });
    }

    /// The rows that hold the block `hash` as the first block of a
    /// request, each with its count of the request's leading blocks.
    fn first_block_holders(f: &Fixture, hash: i64) -> Vec<(WorkerId, u32)> {
        let request = [kv_index::compute_content_hash(&tokens_of(hash)).0];
        f.tables.overlaps(MODEL, &request)
    }

    /// A worker that holds blocks stores under a parent it does not
    /// hold: the store is counted for the worker and starts a chain of
    /// its own, which a request reads from its first block.
    #[test]
    fn a_parentless_store_behind_held_blocks_is_counted_and_starts_a_chain() {
        let series = Series::new();
        metrics::with_local_recorder(&series.recorder, || {
            let f = Fixture::single(Some(BLOCK_SIZE));
            let (_host, sink) = opened(&f);
            let count = |name: &str| series.read(name, &[]);
            assert_eq!(
                sink.admit(f.row, 0, facts(1), &wide(1, 32)),
                BatchOutcome::Applied
            );
            assert_eq!(count("smg_kv_event_parentless_stores_total"), None);
            let orphan = KvBlocksStored {
                parent_block_hash: Some(99),
                ..wide_store(2, 32)
            };
            let orphan = batch([Data::Stored(orphan)]);
            assert_eq!(
                sink.admit(f.row, 0, facts(2), &orphan),
                BatchOutcome::Applied
            );
            assert_eq!(count("smg_kv_event_parentless_stores_total"), Some(1.0));
            assert_eq!(count("smg_kv_event_parentless_blocks_total"), Some(1.0));
            assert_eq!(f.tables.indexed_blocks(f.row, 0), 2);
            for hash in [1, 2] {
                assert_eq!(first_block_holders(&f, hash), [(f.row, 1)], "block {hash}");
            }
        });
    }

    /// One store from `parent` of the block 1 and the block `hash`
    /// behind it, each as wide as the block of record.
    fn forked(hash: i64, parent: Option<i64>) -> KvBlocksStored {
        let wide = |hash: i64| KvBlock {
            block_size: 32,
            ..block(hash, &tokens_of(hash))
        };
        KvBlocksStored {
            blocks: vec![wide(1), wide(hash)],
            parent_block_hash: parent,
            ..KvBlocksStored::default()
        }
    }

    /// Blocks the index declines at a size cap are counted for the
    /// worker: on a store from the root, and on the retry from the root
    /// of a store whose parent the worker does not hold.
    #[test]
    fn blocks_declined_at_a_size_cap_are_counted_for_the_worker() {
        // The first chain's second block lengthens the first block's
        // own run. Each later chain's second block is one more entry of
        // the table behind that run, which holds 196,608 at the most.
        const CHAINS: i64 = 196_609;
        let series = Series::new();
        metrics::with_local_recorder(&series.recorder, || {
            let f = Fixture::single(Some(BLOCK_SIZE));
            let (_host, sink) = opened(&f);
            let count = |name: &str| series.read(name, &[]);
            let stores = "smg_kv_event_cap_refused_stores_total";
            let blocks = "smg_kv_event_cap_refused_blocks_total";
            let fill = (0..CHAINS).map(|chain| Data::Stored(forked(1_000 + chain, None)));
            assert_eq!(
                sink.admit(f.row, 0, facts(1), &batch(fill)),
                BatchOutcome::Applied
            );
            assert_eq!(
                count(stores),
                None,
                "the table takes its entries up to the cap"
            );
            let held = usize::try_from(CHAINS).expect("a count of chains") + 1;
            assert_eq!(f.tables.indexed_blocks(f.row, 0), held);
            // The first block is held, and the second is declined.
            let over = batch([Data::Stored(forked(2_000_000, None))]);
            assert_eq!(sink.admit(f.row, 0, facts(2), &over), BatchOutcome::Applied);
            assert_eq!((count(stores), count(blocks)), (Some(1.0), Some(1.0)));
            // Under a parent the worker does not hold, the retry from
            // the root is declined the same way.
            let orphan = batch([Data::Stored(forked(2_000_001, Some(99)))]);
            assert_eq!(
                sink.admit(f.row, 0, facts(3), &orphan),
                BatchOutcome::Applied
            );
            assert_eq!((count(stores), count(blocks)), (Some(2.0), Some(2.0)));
            let parentless = count("smg_kv_event_parentless_stores_total");
            assert_eq!(parentless, Some(1.0));
            assert_eq!(f.tables.indexed_blocks(f.row, 0), held);
        });
    }

    /// A row no feed was opened for names no series: its batch is
    /// applied by the rules alone. A feed that left names none either.
    #[test]
    fn a_row_with_no_open_feed_names_no_series() {
        let series = Series::new();
        metrics::with_local_recorder(&series.recorder, || {
            let f = Fixture::single(Some(BLOCK_SIZE));
            assert_eq!(
                f.sink.admit(f.row, 0, facts(1), &wide(1, 32)),
                BatchOutcome::Applied
            );
            assert_eq!(f.tables.indexed_blocks(f.row, 0), 1);
            assert!(!series.names_the_worker());

            let (_host, sink) = opened(&f);
            sink.removed(f.row);
            sink.resync(f.row, None, ResyncReason::DataLoss);
            assert!(!series.names_the_worker());
        });
    }
}

/// An engine of three ranks: its rank 0 row, the host whose holder
/// writes the store, and the feed the host opened for that row.
struct Ranked {
    store: Arc<MemoryStore>,
    /// The tables the holder was handed: it makes each row's lane.
    tables: Arc<KvTables>,
    host: KvHost<Members>,
    sink: Arc<KvAdapter>,
    lower: WorkerId,
}

impl Ranked {
    /// Rank 0's row alone: rank 1's is laid by [`Self::upper`].
    fn new() -> Self {
        let store = Arc::new(MemoryStore::new());
        let tables = Arc::new(KvTables::default());
        let objects = Arc::new(
            Objects::new(Arc::new(WorkerRegistry::new())).with_kv_tables(Arc::clone(&tables)),
        );
        let lower = objects.upsert(&store, rank_facts(URL, URL, 0), Registrant::Control);
        let lower = lower.expect("the store takes the write").id();
        let host = KvHost::new(objects);
        let worker = host
            .objects
            .registry()
            .get_by_url(URL)
            .expect("the rank 0 row's object");
        let (sink, row) = host
            .open(host.enroll(&worker, MODEL), &worker, MODEL)
            .expect("the rank 0 row's feed");
        assert_eq!(row, lower);
        Self {
            store,
            tables,
            host,
            sink,
            lower,
        }
    }

    /// Write the row `facts` state through the host's holder: its id.
    fn write(&self, facts: WorkerFacts) -> WorkerId {
        let objects = &self.host.objects;
        let written = objects.upsert(&self.store, facts, Registrant::Control);
        written.expect("the store takes the write").id()
    }

    /// Rank 1's row, laid where it is not yet.
    fn upper(&self) -> WorkerId {
        self.write(rank_facts(UPPER, URL, 1))
    }

    /// The rank 0 row of another engine at `key`, laid, with its feed
    /// opened by the host.
    fn other(&self, key: &str) -> WorkerId {
        let row = self.write(rank_facts(key, key, 0));
        let worker = self
            .host
            .objects
            .registry()
            .get_by_url(key)
            .expect("the other engine's object");
        let (_, opened) = self
            .host
            .open(self.host.enroll(&worker, MODEL), &worker, MODEL)
            .expect("the other engine's feed");
        assert_eq!(opened, row);
        row
    }

    /// One batch of `rank` on the engine's stream, at `sequence`.
    fn admit(&self, rank: u16, sequence: u64, data: Data) -> BatchOutcome {
        self.sink
            .admit(self.lower, rank, facts(sequence), &batch([data]))
    }

    /// The blocks rank 0's row and rank 1's row hold.
    fn blocks(&self, upper: WorkerId) -> (usize, usize) {
        (
            self.tables.indexed_blocks(self.lower, 0),
            self.tables.indexed_blocks(upper, 1),
        )
    }
}

/// A batch of rank 1 on the engine's one stream is indexed under rank
/// 1's row, and a batch of rank 0 under the stream's own. A rank's store
/// names the model's block size by its stream's feed, which is opened
/// for the stream's own row.
#[test]
fn a_batch_of_rank_one_is_indexed_under_rank_ones_row() {
    let f = Ranked::new();
    let upper = f.upper();
    let store = |hash| Data::Stored(wide_store(hash, 16));
    assert_eq!(f.tables.block_size(MODEL), None);
    assert_eq!(f.admit(1, 1, store(1)), BatchOutcome::Applied);
    assert_eq!(f.tables.block_size(MODEL), Some(16));
    assert_eq!(f.blocks(upper), (0, 1));
    assert_eq!(f.admit(0, 1, store(2)), BatchOutcome::Applied);
    assert_eq!(f.admit(0, 2, store(3)), BatchOutcome::Applied);
    assert_eq!(f.blocks(upper), (2, 1));
}

/// A stream that left takes its own feed's record alone: another
/// engine's stream still names the model's block size by its own.
#[test]
fn a_stream_that_left_takes_its_own_feed_alone() {
    let f = Ranked::new();
    let other = f.other(OTHER);
    f.sink.removed(f.lower);
    let first = f.sink.admit(other, 0, facts(1), &wide(1, 16));
    assert_eq!(first, BatchOutcome::Applied);
    assert_eq!(f.tables.block_size(MODEL), Some(16));
}

/// A batch of a rank whose row the holder has not laid is passed over,
/// and the same batch is taken once the row is laid.
#[test]
fn a_batch_of_a_rank_with_no_row_is_passed_over() {
    let f = Ranked::new();
    let store = Data::Stored(wide_store(1, 16));
    assert_eq!(f.admit(1, 1, store.clone()), BatchOutcome::Skipped);
    assert_eq!(f.sink.cursor(f.lower, 1), None);
    let upper = f.upper();
    assert_eq!(f.admit(1, 1, store), BatchOutcome::Applied);
    assert_eq!(f.blocks(upper), (0, 1));
}

/// A rank's cursor and its reset are its own row's. A rank with no row
/// has no cursor and resets nothing. The stream's reset reaches every
/// rank it fed.
#[test]
fn a_ranks_cursor_and_resync_reach_its_own_row() {
    let f = Ranked::new();
    let upper = f.upper();
    let store = |hash| Data::Stored(wide_store(hash, 16));
    assert_eq!(f.admit(0, 3, store(1)), BatchOutcome::Applied);
    assert_eq!(f.admit(1, 7, store(2)), BatchOutcome::Applied);
    let applied = |rank| f.sink.cursor(f.lower, rank).map(|cursor| cursor.applied);
    assert_eq!(
        (applied(0), applied(1), applied(2)),
        (Some(3), Some(7), None)
    );

    f.sink.resync(f.lower, Some(1), ResyncReason::GapCleared);
    assert_eq!(f.blocks(upper), (1, 0));
    f.sink.resync(f.lower, Some(2), ResyncReason::GapCleared);
    assert_eq!(f.tables.resyncs(ResyncReason::GapCleared), 1);

    assert_eq!(f.admit(1, 8, store(3)), BatchOutcome::Applied);
    assert_eq!(f.blocks(upper), (1, 1));
    f.sink.resync(f.lower, None, ResyncReason::OutOfRange);
    assert_eq!(f.blocks(upper), (0, 0));
    assert_eq!(f.tables.resyncs(ResyncReason::OutOfRange), 2);
}

/// An engine that leaves clears every rank its stream fed, and the
/// adapter keeps nothing of them.
#[test]
fn a_removed_engine_clears_every_rank() {
    let f = Ranked::new();
    let upper = f.upper();
    let store = |hash| Data::Stored(wide_store(hash, 16));
    assert_eq!(f.admit(0, 1, store(1)), BatchOutcome::Applied);
    assert_eq!(f.admit(1, 1, store(2)), BatchOutcome::Applied);
    assert_eq!(f.blocks(upper), (1, 1));
    f.sink.removed(f.lower);
    assert_eq!(f.blocks(upper), (0, 0));
    assert_eq!(f.tables.cursor(upper, 1), None);
    let rows = f.sink.rows.lock();
    assert!(rows.written.is_empty() && rows.fed.is_empty());
}

/// A new connection of the engine's stream is every fed rank's: a
/// rank's first batch below its cursor then reads as a publisher that
/// started over.
#[test]
fn a_reconnect_reaches_every_rank() {
    let f = Ranked::new();
    let upper = f.upper();
    assert_eq!(
        f.admit(1, 7, Data::Stored(wide_store(1, 16))),
        BatchOutcome::Applied
    );
    f.sink.reconnected(f.lower);
    assert_eq!(
        f.sink.admit(f.lower, 1, facts(6), &batch([])),
        BatchOutcome::Applied
    );
    assert_eq!(f.blocks(upper), (0, 0));
}

/// What a rank's batches showed of its groups is its own row's. A group
/// shown as another kind on rank 1 keeps that group's removal from rank
/// 1's index, and means nothing for rank 0.
#[test]
fn the_admission_state_is_the_routed_rows() {
    let f = Ranked::new();
    let upper = f.upper();
    let shown = batch([
        Data::Stored(main_store(1, 0)),
        Data::Stored(window_store(&[1], 2, false)),
    ]);
    assert_eq!(
        f.sink.admit(f.lower, 1, facts(1), &shown),
        BatchOutcome::Applied
    );
    assert_eq!(
        f.admit(0, 1, Data::Stored(stored(1))),
        BatchOutcome::Applied
    );
    assert_eq!(f.blocks(upper), (1, 1));
    let removal = || Data::Removed(group_removal(&[1], 2));
    // On rank 1 the removal is the window group's: the main block stays.
    assert_eq!(f.admit(1, 2, removal()), BatchOutcome::Applied);
    // Rank 0 showed no group: the same removal is its main block's.
    assert_eq!(f.admit(0, 2, removal()), BatchOutcome::Applied);
    assert_eq!(f.blocks(upper), (0, 1));
}

/// A rank's row the stream feeds for the first time begins from
/// nothing, as a feed's own row does at its open: what the tables kept
/// of it was another writer's.
#[test]
fn a_ranks_row_the_stream_feeds_first_begins_from_nothing() {
    let f = Ranked::new();
    let upper = f.upper();
    // Another adapter over the same store wrote the row before.
    let other = KvAdapter::new(Arc::clone(&f.tables));
    let before = batch([Data::Stored(wide_store(1, 16))]);
    assert_eq!(
        other.admit(upper, 1, facts(7), &before),
        BatchOutcome::Applied
    );
    assert_eq!(f.blocks(upper), (0, 1));
    // A batch below that cursor is the stream's first for the row.
    assert_eq!(
        f.admit(1, 3, Data::Stored(wide_store(2, 16))),
        BatchOutcome::Applied
    );
    assert_eq!(f.blocks(upper), (0, 1));
    let applied = f.sink.cursor(f.lower, 1).map(|cursor| cursor.applied);
    assert_eq!(applied, Some(3));
}

/// A row whose facts move it under another engine is that engine's
/// stream's from then on. It begins from nothing there, the old
/// stream's removal leaves it alone, and the new stream's reset reaches
/// it.
#[test]
fn a_row_that_moves_to_another_engine_is_its_new_streams() {
    let f = Ranked::new();
    let upper = f.upper();
    let store = |hash| Data::Stored(wide_store(hash, 16));
    assert_eq!(f.admit(1, 5, store(1)), BatchOutcome::Applied);
    assert_eq!(f.blocks(upper), (0, 1));

    // The same registration names the other engine now: the row keeps
    // its id, and the first engine has no row of rank 1.
    let other = f.other(OTHER);
    let moved = f.write(rank_facts(UPPER, OTHER, 1));
    assert_eq!(moved, upper);
    assert_eq!(f.admit(1, 6, store(2)), BatchOutcome::Skipped);

    // The other engine's stream takes the row, below its old cursor.
    assert_eq!(
        f.sink.admit(other, 1, facts(2), &batch([store(3)])),
        BatchOutcome::Applied
    );
    assert_eq!(f.tables.indexed_blocks(upper, 1), 1);
    let applied = f.sink.cursor(other, 1).map(|cursor| cursor.applied);
    assert_eq!(applied, Some(2));

    f.sink.removed(f.lower);
    assert_eq!(f.tables.indexed_blocks(upper, 1), 1);
    f.sink.resync(other, None, ResyncReason::OutOfRange);
    assert_eq!(f.tables.indexed_blocks(upper, 1), 0);
}

/// A feed's own row whose facts make it a rank of another engine is
/// that engine's stream's. Its own stream, which is ending, writes no
/// row of any rank from then on: not its own, and no row of the other
/// engine by its new facts. That stream's removal leaves what the
/// other engine's stream wrote.
#[test]
fn a_feeds_row_that_becomes_another_engines_rank_leaves_its_own_stream() {
    let f = Ranked::new();
    let own = f.other(OTHER);
    let third = f.write(rank_facts(THIRD, URL, 2));
    let store = |hash| Data::Stored(wide_store(hash, 16));
    // One batch on the row's own stream.
    let late = |rank, sequence, hash| {
        let one = batch([store(hash)]);
        f.sink.admit(own, rank, facts(sequence), &one)
    };
    assert_eq!(late(0, 5, 1), BatchOutcome::Applied);
    assert_eq!(f.admit(2, 5, store(9)), BatchOutcome::Applied);
    let thirds = || f.tables.indexed_blocks(third, 2);
    assert_eq!(thirds(), 1);

    // The same registration states rank 1 of the first engine now. A
    // late batch of the row's own stream reaches no row of that engine,
    // before its stream takes the row and after.
    let moved = f.write(rank_facts(OTHER, URL, 1));
    assert_eq!(moved, own);
    assert_eq!(late(2, 4, 7), BatchOutcome::Skipped);
    assert_eq!(f.admit(1, 2, store(2)), BatchOutcome::Applied);
    assert_eq!(f.tables.indexed_blocks(own, 1), 1);
    assert_eq!(late(0, 6, 3), BatchOutcome::Skipped);
    assert_eq!(late(2, 4, 8), BatchOutcome::Skipped);
    assert_eq!(thirds(), 1);

    // Its facts state rank 0 of its first engine again and no feed was
    // opened for that: the row is still the other stream's, so its old
    // stream writes no rank of its first engine either.
    let fourth = f.write(rank_facts(FOURTH, OTHER, 1));
    let back = f.write(rank_facts(OTHER, OTHER, 0));
    assert_eq!(back, own);
    assert_eq!(late(1, 1, 4), BatchOutcome::Skipped);
    assert_eq!(f.tables.indexed_blocks(fourth, 1), 0);

    f.sink.removed(own);
    assert_eq!(thirds(), 1);
    f.sink.removed(f.lower);
    assert_eq!(thirds(), 0);
}

/// A ticket from before the row's object was laid again opens nothing
/// and is counted. The row is another stream's by then, and what that
/// stream wrote stands.
#[test]
fn a_ticket_from_before_the_row_was_laid_again_opens_nothing() {
    let f = Ranked::new();
    let row = f.write(rank_facts(OTHER, OTHER, 0));
    let worker = f
        .host
        .objects
        .registry()
        .get_by_url(OTHER)
        .expect("the other engine's object");
    let ticket = f.host.enroll(&worker, MODEL);
    assert_eq!(ticket, Ok(row));

    // Between the enrollment and the open the row becomes the first
    // engine's rank 1, and that engine's stream writes it.
    let moved = f.write(rank_facts(OTHER, URL, 1));
    assert_eq!(moved, row);
    let store = Data::Stored(wide_store(1, 16));
    assert_eq!(f.admit(1, 2, store), BatchOutcome::Applied);
    assert_eq!(f.tables.indexed_blocks(row, 1), 1);

    assert!(f.host.open(ticket, &worker, MODEL).is_none());
    assert_eq!(f.tables.indexed_blocks(row, 1), 1);
    assert_eq!(f.host.skipped_subscriptions().no_row, 1);
}

/// No stream takes a row its leaving stream still clears: a batch
/// routed to it is passed over, and the first one behind the removal
/// takes it. So a removal clears nothing a new stream wrote.
#[test]
fn a_row_its_leaving_stream_still_clears_is_passed_over() {
    let f = Ranked::new();
    let upper = f.upper();
    let store = |hash| Data::Stored(wide_store(hash, 16));
    assert_eq!(f.admit(1, 5, store(1)), BatchOutcome::Applied);
    let other = f.other(OTHER);
    let moved = f.write(rank_facts(UPPER, OTHER, 1));
    assert_eq!(moved, upper);
    // One batch of rank 1 on the other engine's stream.
    let late = |sequence, hash| {
        let one = batch([store(hash)]);
        f.sink.admit(other, 1, facts(sequence), &one)
    };

    // The first engine's stream is leaving: its removal marked the row
    // and has not cleared it yet.
    f.sink
        .rows
        .lock()
        .written
        .get_mut(&upper)
        .expect("the row's entry")
        .leaving = true;
    assert_eq!(late(2, 2), BatchOutcome::Skipped);
    assert_eq!(f.tables.indexed_blocks(upper, 1), 1);

    // The removal ends: the row is cleared and forgotten, and the
    // other engine's next batch takes it.
    f.sink.removed(f.lower);
    assert_eq!(f.tables.indexed_blocks(upper, 1), 0);
    assert_eq!(late(3, 3), BatchOutcome::Applied);
    assert_eq!(f.tables.indexed_blocks(upper, 1), 1);
    let rows = f.sink.rows.lock();
    let stream = rows.written.get(&upper).map(|written| written.stream);
    assert_eq!(stream, Some(other));
    assert_eq!(rows.fed.get(&other), Some(&vec![upper]));
    assert_eq!(rows.fed.get(&f.lower), None);
}

/// A route read before a row was laid again takes no open feed's own
/// row. The row stands at rank 0 of its own engine and its feed is
/// open: the old stream's late batch is passed over, and the row's own
/// stream goes on writing it.
#[test]
fn a_stale_route_takes_no_open_feeds_own_row() {
    let f = Ranked::new();
    let upper = f.upper();
    let store = |hash| Data::Stored(wide_store(hash, 16));
    assert_eq!(f.admit(1, 5, store(1)), BatchOutcome::Applied);
    // The row is laid again in place as rank 0 of its own engine, and
    // its feed opens.
    assert_eq!(f.other(UPPER), upper);

    // A batch of the old stream that read its route before the lay.
    assert!(f.sink.applied(f.lower, upper).is_none());
    let own = batch([store(2)]);
    assert_eq!(
        f.sink.admit(upper, 0, facts(1), &own),
        BatchOutcome::Applied
    );
    assert_eq!(f.tables.indexed_blocks(upper, 0), 1);
}

/// A feed's open asks whether its object still stands for the row in
/// the hold that writes the row's entry: the lock is taken while the
/// answer is read.
#[test]
fn a_feeds_open_asks_the_holder_in_the_hold_that_writes_its_row() {
    let f = Fixture::single(Some(BLOCK_SIZE));
    let sink = &f.sink;
    let held = sink.begin(f.row, || sink.rows.try_lock().is_none());
    assert_eq!(held, Ok(()));
    {
        let rows = sink.rows.lock();
        let begun = rows.written.get(&f.row).expect("the row's entry");
        assert_eq!((begun.stream, begun.leaving), (f.row, false));
    }
    // A row its object stands for no longer begins nothing.
    sink.removed(f.row);
    assert_eq!(sink.begin(f.row, || false), Err(SkipReason::NoRow));
    assert!(sink.rows.lock().written.is_empty());
}

/// What `seen` reads while `step` stands still inside its clear of
/// `row`. The tables' own lock is the probe: a clear of a row takes its
/// lane's lock, and here a batch's body holds that lock until `seen`
/// has answered, so the step cannot get past its clear before then.
fn at_its_clear<T>(
    tables: &KvTables,
    row: WorkerId,
    sequence: u64,
    step: impl FnOnce() + Send,
    seen: impl Fn() -> Option<T>,
) -> T {
    let (entered, at_body) = mpsc::channel();
    let (release, held) = mpsc::channel::<()>();
    std::thread::scope(|scope| {
        scope.spawn(move || {
            let outcome = tables.admit_through(row, 0, facts(sequence), &mut |_| {
                entered.send(()).expect("the case waits for the body");
                held.recv().expect("the case lets the body go");
            });
            assert_eq!(outcome, BatchOutcome::Applied);
        });
        at_body.recv().expect("the body holds the lane");
        let stepping = scope.spawn(step);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let read = loop {
            if let Some(read) = seen() {
                break Some(read);
            }
            if std::time::Instant::now() >= deadline {
                break None;
            }
            std::thread::yield_now();
        };
        // The body is let go before any failure is said, so a case that
        // fails ends: no thread of it waits on the lane.
        release.send(()).expect("the body waits");
        stepping.join().expect("the step ends");
        read.expect("the step never showed what the case waits for")
    })
}

/// A row's first clear runs in the hold that writes its entry, at a
/// stream's take and at a feed's open. No stream claims the row, and
/// writes it, between the two, so the clear takes nothing a later
/// stream wrote. A removal's clear runs outside the lock, behind its
/// mark: it takes every block the stream stored.
#[test]
fn a_claimed_rows_first_clear_runs_in_the_hold_that_writes_its_entry() {
    let store = Arc::new(MemoryStore::new());
    let tables = Arc::new(KvTables::default());
    let sink = Arc::new(KvAdapter::new(Arc::clone(&tables)));
    let row = store.upsert(rank_facts(URL, URL, 0)).id();
    let other = store.upsert(rank_facts(OTHER, OTHER, 0)).id();
    for id in [row, other] {
        tables.member(&store.get(id).expect("the row"));
    }
    // One look at the adapter's list while the step cannot get past its
    // clear. Held: the step holds it, at its clear or still ahead of
    // it, so the case looks again behind a wait that is long against a
    // claim's own hold. A clear inside the hold still holds the list
    // then. Free, with the row's entry naming `stream`: the step wrote
    // its entry and let the list go ahead of its clear.
    let held_at_its_clear = |stream: WorkerId| {
        let Some(rows) = sink.rows.try_lock() else {
            std::thread::sleep(std::time::Duration::from_millis(20));
            return Some(sink.rows.try_lock().is_none());
        };
        let written = rows.written.get(&row);
        let wrote = written.is_some_and(|written| written.stream == stream);
        wrote.then_some(false)
    };

    // Another stream's take of the row, then the open of its own feed.
    let take = || assert!(sink.take(other, row).is_some());
    let held = at_its_clear(&tables, row, 1, take, || held_at_its_clear(other));
    assert!(held, "a take's clear runs in the hold of its claim");
    let begin = || assert_eq!(sink.begin(row, || true), Ok(()));
    let held = at_its_clear(&tables, row, 1, begin, || held_at_its_clear(row));
    assert!(held, "a begin's clear runs in the hold of its claim");

    // The feed's removal: its mark stands in the list, which is free,
    // while its clear still waits for the lane.
    let marked = || {
        let rows = sink.rows.try_lock()?;
        let leaving = rows
            .written
            .get(&row)
            .is_some_and(|written| written.leaving);
        leaving.then_some(())
    };
    at_its_clear(&tables, row, 1, || sink.removed(row), marked);
}
