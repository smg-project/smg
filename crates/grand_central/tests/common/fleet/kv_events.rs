//! The harness's listener on the KV events: one stream an engine, what each
//! stream delivered, and what a case reads of it beside the controller's KV
//! tables.
//! A child of the harness's module: it reads the handle's private fields and
//! the record of a probed engine as they are, so nothing is widened to move
//! it out.

use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use kv_index::{compute_content_hash, SequenceHash, StoredBlock};
use parking_lot::Mutex;
use smg_grand_central::engine::WorkerId;
use smg_grpc_client::{
    common_proto::{kv_cache_event::Data, KvEventBatch},
    VllmEngineClient,
};
use smg_worker::kv_events::port::{KvCursor, ResyncReason};
use tokio::sync::watch;

use super::{Harness, Probed, DIAL_TIMEOUT, PROBE_INTERVAL};

/// Every reason the KV tables count a resync under.
const RESYNCS: [ResyncReason; 6] = [
    ResyncReason::OutOfRange,
    ResyncReason::DataLoss,
    ResyncReason::SnapshotAbandoned,
    ResyncReason::PublisherRestart,
    ResyncReason::GapCleared,
    ResyncReason::Snapshot,
];

/// A reason's place in [`RESYNCS`]. The match names every reason, so a new
/// one fails to build here, beside the list that must name it.
const fn place(reason: ResyncReason) -> usize {
    match reason {
        ResyncReason::OutOfRange => 0,
        ResyncReason::DataLoss => 1,
        ResyncReason::SnapshotAbandoned => 2,
        ResyncReason::PublisherRestart => 3,
        ResyncReason::GapCleared => 4,
        ResyncReason::Snapshot => 5,
    }
}

// Every place of the list holds its reason, read at the build.
const _: () = {
    let mut at = 0;
    while at < RESYNCS.len() {
        assert!(
            place(RESYNCS[at]) == at,
            "a resync reason is out of its place"
        );
        at += 1;
    }
};

/// One rank's KV events: what the listener's stream delivered beside what
/// the controller's KV tables hold of it, read at one instant. The two come
/// by streams of their own, so they agree only at rest.
#[derive(Clone, Copy, Debug)]
pub struct Heard {
    /// The engine's data-parallel rank the row stands for.
    pub rank: u16,
    /// The last sequence the engine's stream delivered for the rank; `0`
    /// when none came.
    pub last: u64,
    /// The blocks the stream's events left the engine holding.
    pub held: usize,
    /// The blocks the controller's index holds for the row.
    pub indexed: usize,
    /// The rank's cursor in the KV tables, when the rank has been heard
    /// from.
    pub cursor: Option<KvCursor>,
}

/// What the listeners heard, a row at a time. The controller's KV tables
/// have a writer of their own, its feed, so a read sees the two in step
/// only at rest.
#[derive(Default)]
pub(super) struct Events {
    rows: Mutex<HashMap<WorkerId, Stream>>,
    /// What refuses each engine's stream, by the engine's index: a dial or
    /// a subscription that failed or outlasted its bound, a stream that
    /// broke. An engine whose stream stands open has none.
    refused: Mutex<HashMap<usize, String>>,
    /// The snapshot chunks that came: an engine's own stream carries none.
    snapshots: AtomicU64,
}

/// One row's stream as its listener heard it.
#[derive(Default)]
struct Stream {
    /// The last sequence the stream delivered.
    last: u64,
    /// The blocks the stream's events left the engine holding, by its
    /// hashes. An engine's own stream carries no snapshot chunk (a relay's
    /// does), so stores, removals and clears are the whole of it. A stream
    /// opened again past the engine's replay buffer would part it from the
    /// index: the replay starts this set over with what the buffer still
    /// holds, and the index keeps what its own stream brought. No engine of
    /// the fleet opens one.
    held: HashSet<SequenceHash>,
    /// The first chain the stream stored, in the chain's order: the blocks of
    /// the first store under no parent, then of each store whose parent is
    /// the chain's last block. A prompt's head as the engine holds it.
    head: Vec<StoredBlock>,
}

/// One event of an engine's batch as the listener keeps it.
enum KvEvent {
    /// Blocks stored in order after `parent`; `None` starts a chain.
    Stored {
        parent: Option<SequenceHash>,
        blocks: Vec<StoredBlock>,
    },
    /// Blocks evicted, by the engine's hashes.
    Removed(Vec<SequenceHash>),
    /// The engine emptied its cache.
    Cleared,
}

/// One engine batch as the events an index applies: every block under the
/// engine's own hash, its content hash made of its token ids. The harness's
/// engines publish one cache group in one tier, so this decode chooses
/// nothing: which blocks reach an index is a collector's decision.
fn decode(batch: &KvEventBatch) -> Vec<KvEvent> {
    batch
        .events
        .iter()
        .filter_map(|event| {
            Some(match event.data.as_ref()? {
                Data::Stored(stored) => KvEvent::Stored {
                    parent: stored.parent_block_hash.map(SequenceHash::from),
                    blocks: stored
                        .blocks
                        .iter()
                        .map(|block| StoredBlock {
                            seq_hash: SequenceHash::from(block.block_hash),
                            content_hash: compute_content_hash(&block.token_ids),
                        })
                        .collect(),
                },
                Data::Removed(removed) => KvEvent::Removed(
                    removed
                        .block_hashes
                        .iter()
                        .copied()
                        .map(SequenceHash::from)
                        .collect(),
                ),
                Data::Cleared(_) => KvEvent::Cleared,
            })
        })
        .collect()
}

/// One batch as its rank's row keeps it: the last sequence, and the blocks
/// its events leave the engine holding. The controller is handed nothing
/// here: its KV tables have its own feed. A batch that carries a load
/// record alone is passed by, and its record is not written here: loads
/// are the monitor's. A snapshot chunk is a relay's: its start is a
/// collector's word to every row of the engine, which this listener does
/// not say, so a chunk is counted and passed by.
fn keep(engine: &Probed, events: &Events, batch: &KvEventBatch) {
    if batch.load.as_ref().is_some_and(|load| load.load_only) {
        return;
    }
    if batch.snapshot.is_some() {
        events.snapshots.fetch_add(1, Ordering::SeqCst);
        return;
    }
    let rank = u16::try_from(batch.dp_rank.unwrap_or(0)).unwrap_or(u16::MAX);
    let Some(&id) = engine.ids.get(usize::from(rank)) else {
        return;
    };
    let decoded = decode(batch);
    let mut rows = events.rows.lock();
    let row = rows.entry(id).or_default();
    row.last = batch.sequence_number;
    for event in decoded {
        match event {
            KvEvent::Stored { parent, blocks } => {
                if parent == row.head.last().map(|block| block.seq_hash) {
                    row.head.extend(&blocks);
                }
                row.held.extend(blocks.iter().map(|block| block.seq_hash));
            }
            KvEvent::Removed(hashes) => {
                for hash in hashes {
                    row.held.remove(&hash);
                }
            }
            KvEvent::Cleared => row.held.clear(),
        }
    }
}

/// The case's listener on the KV events: one stream an engine, every rank's
/// batches on it, each kept as it arrives ([`keep`]). The stream is asked
/// from zero each time it opens, and the engine's rows start over: an
/// engine asked from zero replays what its buffer still holds. A stream
/// that ends is opened again after one probe interval. What refuses a dial
/// or a stream is kept until a stream opens, so a wait that runs out can
/// say why.
async fn subscribe(engine: Probed, events: Arc<Events>, mut stop: watch::Receiver<bool>) {
    loop {
        let opened = tokio::time::timeout(DIAL_TIMEOUT, async {
            let client = VllmEngineClient::connect(&engine.endpoint)
                .await
                .map_err(|error| format!("the dial: {error}"))?;
            client
                .subscribe_kv_events(0)
                .await
                .map_err(|status| format!("the subscription: {status}"))
        })
        .await
        .unwrap_or_else(|_| Err(format!("no stream within {DIAL_TIMEOUT:?}")));
        let refusal = match opened {
            Ok(mut stream) => {
                events.refused.lock().remove(&engine.engine);
                {
                    // The engine replays from zero: its rows start over,
                    // and the controller is told nothing.
                    let mut rows = events.rows.lock();
                    for id in &engine.ids {
                        rows.remove(id);
                    }
                }
                loop {
                    let batch = tokio::select! {
                        batch = stream.message() => batch,
                        _ = stop.wait_for(|stop| *stop) => return,
                    };
                    match batch {
                        Ok(Some(batch)) => keep(&engine, &events, &batch),
                        Ok(None) => break None,
                        Err(status) => break Some(format!("the stream: {status}")),
                    }
                }
            }
            Err(refusal) => Some(refusal),
        };
        if let Some(refusal) = refusal {
            events.refused.lock().insert(engine.engine, refusal);
        }
        tokio::select! {
            () = tokio::time::sleep(PROBE_INTERVAL) => {}
            _ = stop.wait_for(|stop| *stop) => return,
        }
    }
}

impl Harness {
    /// Starts the listener of one engine.
    pub(super) fn subscriber(&mut self, engine: Probed) {
        self.feeds.spawn(subscribe(
            engine,
            Arc::clone(&self.events),
            self.stop_feeds.subscribe(),
        ));
    }

    /// The first chain the first rank of the engine at `engine` stored, in
    /// the chain's order, each block by the hash of its tokens: what a
    /// request that brings the same prompt states of its head.
    pub fn head(&self, engine: usize) -> Vec<u64> {
        let streams = self.events.rows.lock();
        self.members
            .iter()
            .find(|(_, (of, rank))| *of == engine && *rank == 0)
            .and_then(|(id, _)| streams.get(id))
            .map(|stream| {
                stream
                    .head
                    .iter()
                    .map(|block| block.content_hash.0)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The KV events of the engine at `engine`, a rank a row: what each
    /// stream delivered beside what the controller's KV tables hold of it.
    pub fn heard(&self, engine: usize) -> Vec<Heard> {
        let streams = self.events.rows.lock();
        let mut rows: Vec<Heard> = self
            .members
            .iter()
            .filter(|(_, (of, _))| *of == engine)
            .map(|(id, (_, rank))| {
                let stream = streams.get(id);
                Heard {
                    rank: *rank,
                    last: stream.map_or(0, |stream| stream.last),
                    held: stream.map_or(0, |stream| stream.held.len()),
                    indexed: self.tables.indexed_blocks(*id, *rank),
                    cursor: self.tables.cursor(*id, *rank),
                }
            })
            .collect();
        rows.sort_unstable_by_key(|row| row.rank);
        rows
    }

    /// What refuses the event stream of the engine at `engine`, in the
    /// subscriber's words; `None` when nothing refuses it.
    pub fn refused(&self, engine: usize) -> Option<String> {
        self.events.refused.lock().get(&engine).cloned()
    }

    /// The gaps on any engine's stream: the resyncs the controller's KV
    /// tables count, under every reason.
    pub fn gaps(&self) -> u64 {
        RESYNCS
            .iter()
            .map(|reason| self.tables.resyncs(*reason))
            .sum()
    }

    /// The snapshot chunks any engine's stream carried.
    pub fn snapshot_chunks(&self) -> u64 {
        self.events.snapshots.load(Ordering::SeqCst)
    }
}
