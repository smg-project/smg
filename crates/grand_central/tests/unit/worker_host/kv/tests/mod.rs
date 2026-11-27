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

mod application;
mod feeds_and_block_size;
mod ranks;
#[cfg(feature = "controller")]
mod series;

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
/// ([`Fixture::state`]), as a holder does where it lays the row. The
/// row's feed is open on the adapter ([`open_feed`]): a stream writes
/// only while its feed stands open.
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
        open_feed(&fixture.sink, row);
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

/// Open `row`'s feed on `sink`, as a host's opening does: the row
/// begins, and the feed's record is kept.
fn open_feed(sink: &KvAdapter, row: WorkerId) {
    assert_eq!(sink.begin(row, || true), Ok(()), "the row begins");
    sink.opened(row, MODEL, URL);
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
